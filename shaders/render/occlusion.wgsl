// render/occlusion.wgsl: screen-space ambient occlusion and contact shadows (todo 1.9), from the
// depth of a prepass, before the geometry pass shades, so each term goes where it belongs: the
// ambient occlusion on the environment's light, the contact shadow of a light on that light.
//
//   location 0 (r8unorm):    how much of the hemisphere above the surface is open within
//                            `frame.occlusion.x` metres (1 open, 0 closed).
//   location 1 (rgba8unorm): per shadow-map layer (x, y, z, w: layers 0-3), how much of that
//                            layer's light passes the first `frame.occlusion.y` metres towards it.
//
// Ambient occlusion: horizon-based (Bavoil and Sainz 2008). Along AO_SLICES screen directions and
// both ways, AO_STEPS depth samples out to the radius give the highest elevation of other geometry
// over the tangent plane; a side is closed by the square of that elevation's sine (the
// cosine-weighted share of the slice below the horizon), at full weight out to half the radius and
// fading linearly to nothing at the radius. Geometry below the
// tangent plane closes nothing, so flat and convex skin stays exactly open: only creases darken
// (nostril wing, lid fold, lip line, under the chin, behind the ear, between fingers). The normal
// is the depth buffer's own (the nearer neighbour on each axis), so an edge does not occlude the
// surface behind it and nothing glows around a silhouette.
//
// Contact shadows: CONTACT_STEPS steps of a ray towards each shadowed light; a step that lies
// behind the depth buffer by less than CONTACT_THICKNESS is blocked. The shadow fades out along the
// ray, where the shadow map takes over. This is what the map's bias cannot hold: the first
// fraction of a millimetre in a portrait, the first millimetres when the map covers a whole body.
//
// The slice directions and step offsets turn per pixel and per frame; the temporal accumulation
// averages them. Samples reach at most MAX_REACH_PIXELS: a tiled still needs that much overlap.
// Below the Still tier the targets have half the resolution (`frame.occlusion.w`, see fs_occlusion):
// the full-resolution pass cost 0.9 ms of a 2560 x 1440 frame, the half one a quarter of that.

#include "lib/math.wgsl"
#include "lib/sampling.wgsl"
#include "render/frame.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var depth_tex: texture_depth_2d;

const AO_SLICES: u32 = 2u;
const AO_STEPS: u32 = 6u;
const CONTACT_STEPS: u32 = 8u;
// The farthest a sample lies from its pixel.
const MAX_REACH_PIXELS: f32 = 96.0;
// Sine of the elevation below which geometry does not count (depth precision, tessellation).
const AO_BIAS: f32 = 0.08;
// How thick the depth buffer's surfaces are taken to be for a contact shadow, metres.
const CONTACT_THICKNESS: f32 = 0.012;

struct OcclusionOut {
    @location(0) ambient: vec4<f32>,
    @location(1) contact: vec4<f32>,
}

@vertex
fn vs_occlusion(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_position(index);
}

// The world position of a pixel centre at a reversed-Z depth.
fn world_at(position: vec2<f32>, depth: f32) -> vec3<f32> {
    let p = frame.inv_view_proj * vec4<f32>(pixel_to_ndc(position), depth, 1.0);
    return p.xyz / p.w;
}

// The depth at a pixel, 0 (nothing) outside the target.
fn depth_at(pixel: vec2<i32>, size: vec2<i32>) -> f32 {
    if (any(pixel < vec2<i32>(0)) || any(pixel >= size)) {
        return 0.0;
    }
    return textureLoad(depth_tex, pixel, 0);
}

// The step from `p` to the neighbour along `axis` that lies on the same surface: the nearer in
// depth of the two (the other one may be across an edge), signed so it points along +axis.
fn surface_step(position: vec2<f32>, pixel: vec2<i32>, size: vec2<i32>, axis: vec2<i32>, depth: f32, p: vec3<f32>) -> vec3<f32> {
    let before = depth_at(pixel - axis, size);
    let after = depth_at(pixel + axis, size);
    let use_after = after > 0.0 && (before <= 0.0 || abs(after - depth) <= abs(before - depth));
    if (use_after) {
        return world_at(position + vec2<f32>(axis), after) - p;
    }
    if (before > 0.0) {
        return p - world_at(position - vec2<f32>(axis), before);
    }
    // No neighbour at all: the step across the view at this depth.
    return world_at(position + vec2<f32>(axis), depth) - p;
}

@fragment
fn fs_occlusion(@builtin(position) position: vec4<f32>) -> OcclusionOut {
    var out: OcclusionOut;
    out.ambient = vec4<f32>(1.0);
    out.contact = vec4<f32>(1.0);
    let size = vec2<i32>(textureDimensions(depth_tex));
    // Below the Still tier the targets have half the resolution (`frame.occlusion.w` 2): a texel
    // stands for a 2 x 2 block of pixels and is computed at one of them, a different one each frame
    // (the temporal accumulation averages the four).
    let scale = max(i32(frame.occlusion.w + 0.5), 1);
    let phase = i32(frame.counters.x % 4u);
    let offset = select(vec2<i32>(0), vec2<i32>(phase & 1, phase >> 1u), scale > 1);
    let pixel = min(vec2<i32>(position.xy) * scale + offset, size - 1);
    let centre = vec2<f32>(pixel) + 0.5;
    let depth = textureLoad(depth_tex, pixel, 0);
    if (depth <= 0.0 || frame.occlusion.w < 0.5) {
        return out;
    }
    let p = world_at(centre, depth);
    let eye = frame.eye_exposure.xyz;
    let dx = surface_step(centre, pixel, size, vec2<i32>(1, 0), depth, p);
    let dy = surface_step(centre, pixel, size, vec2<i32>(0, 1), depth, p);
    var n = safe_normalize(cross(dx, dy), normalize(eye - p));
    if (dot(n, eye - p) < 0.0) {
        n = -n;
    }
    // What one pixel covers across the view at this depth, metres.
    let pixel_world = max(length(world_at(centre + vec2<f32>(1.0, 0.0), depth) - p), 1e-7);
    let seed = global_pixel(centre);
    let turn = interleaved_gradient_noise(seed + 5.588238 * f32(frame.counters.x % 64u));
    let shift = interleaved_gradient_noise(seed.yx + 11.71 + 3.3391 * f32(frame.counters.x % 64u));

    // Ambient occlusion.
    let radius = max(frame.occlusion.x, 1e-4);
    let reach = min(radius / pixel_world, MAX_REACH_PIXELS);
    if (reach >= 1.5) {
        // The radius the samples really cover (less than `radius` in a close-up).
        let covered = reach * pixel_world;
        var closed = 0.0;
        for (var slice = 0u; slice < AO_SLICES; slice++) {
            let angle = PI * (turn + f32(slice)) / f32(AO_SLICES);
            let direction = vec2<f32>(cos(angle), sin(angle));
            for (var side = 0u; side < 2u; side++) {
                let along = direction * select(-1.0, 1.0, side == 0u);
                var horizon = 0.0;
                for (var s = 0u; s < AO_STEPS; s++) {
                    // Denser near the pixel: a crease is narrow.
                    let t = (f32(s) + shift) / f32(AO_STEPS);
                    let at = centre + along * max(reach * t * t, 1.0);
                    let q = vec2<i32>(floor(at));
                    let d = depth_at(q, size);
                    if (d <= 0.0) {
                        continue;
                    }
                    let v = world_at(vec2<f32>(q) + 0.5, d) - p;
                    let distance = max(length(v), 1e-7);
                    let elevation = saturate(dot(v, n) / distance - AO_BIAS) / (1.0 - AO_BIAS);
                    let fade = saturate(2.0 - 2.0 * distance / covered);
                    horizon = max(horizon, elevation * elevation * fade);
                }
                closed += horizon;
            }
        }
        out.ambient = vec4<f32>(saturate(1.0 - closed / f32(2u * AO_SLICES)));
    }

    // Contact shadows, one per shadow-map layer.
    let near = max((frame.unjittered_view_proj * vec4<f32>(world_at(centre, 1.0), 1.0)).w, 1e-6);
    let length_world = min(frame.occlusion.y, MAX_REACH_PIXELS * pixel_world);
    let count = min(u32(frame.environment.w), MAX_LIGHTS);
    if (length_world > 0.0) {
        for (var i = 0u; i < count; i++) {
            let light = frame.lights[i];
            if (light.shadow.x < 0.0) {
                continue;
            }
            var to_light = -light.direction_cos_outer.xyz;
            if (u32(light.color_kind.w) != LIGHT_DIRECTIONAL) {
                to_light = safe_normalize(light.position_range.xyz - p, to_light);
            }
            // A surface that faces away from the light is dark already, and one edge-on to it
            // would catch the facets of its own mesh: the contact shadow fades in from a cosine of
            // 0.1 to 0.35.
            let facing = smoothstep(0.1, 0.35, dot(n, to_light));
            if (facing <= 0.0) {
                continue;
            }
            var open = 1.0;
            // The ray's clip position is linear in its length: the start and the step, projected once.
            let clip_start = frame.unjittered_view_proj * vec4<f32>(p + n * (0.5 * pixel_world), 1.0);
            let clip_ray = frame.unjittered_view_proj * vec4<f32>(to_light * length_world, 0.0);
            for (var s = 0u; s < CONTACT_STEPS; s++) {
                let t = (f32(s) + shift) / f32(CONTACT_STEPS);
                let clip = clip_start + clip_ray * t;
                if (clip.w <= near) {
                    break;
                }
                let ndc = clip.xy / clip.w;
                let q = vec2<i32>(floor(vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * frame.viewport.xy));
                if (all(q == pixel)) {
                    continue;
                }
                let d = depth_at(q, size);
                if (d <= 0.0) {
                    continue;
                }
                // How far the ray is behind the surface the depth buffer holds there, metres.
                let behind = clip.w - near / d;
                if (behind > 0.0003 + 2.0 * pixel_world && behind < CONTACT_THICKNESS) {
                    open = mix(1.0, smoothstep(0.5, 1.0, t), facing);
                    break;
                }
            }
            let layer = min(u32(light.shadow.x), MAX_SHADOWED - 1u);
            out.contact = select(out.contact, vec4<f32>(open), vec4<u32>(layer) == vec4<u32>(0u, 1u, 2u, 3u));
        }
    }
    return out;
}
