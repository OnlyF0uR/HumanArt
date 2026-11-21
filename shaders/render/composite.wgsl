// render/composite.wgsl: the lighting composite [RND-03] item 9: diffuse lighting times albedo plus
// specular where there is geometry; the backdrop elsewhere [LIT-04]: a flat grey, already divided
// by the exposure, the environment, or seamless studio paper (todo 1.9): a wall behind the subject
// that curves into the floor under it, shaded as a matte surface by the rig's lights and the
// environment, with the subject's shadow and a soft falloff away from the spot behind the subject.

#include "lib/brdf.wgsl"
#include "lib/color.wgsl"
#include "lib/packing.wgsl"
#include "render/frame.wgsl"
#include "render/lighting.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(2) @binding(0) var env_specular: texture_cube<f32>;
@group(2) @binding(1) var env_sampler: sampler;
@group(2) @binding(2) var dfg_lut: texture_2d<f32>;
@group(2) @binding(3) var<storage, read> irradiance_sh: array<vec4<f32>, 9>;
@group(2) @binding(4) var shadow_atlas: texture_depth_2d_array;
@group(2) @binding(5) var shadow_sampler: sampler_comparison;

@group(1) @binding(0) var diffuse_tex: texture_2d<f32>;
@group(1) @binding(1) var specular_tex: texture_2d<f32>;
@group(1) @binding(2) var surface_tex: texture_2d<u32>;
@group(1) @binding(3) var depth_tex: texture_depth_2d;

@vertex
fn vs_composite(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_position(index);
}

const BACKGROUND_PAPER: f32 = 1.5;
// Shadow taps per light on the paper.
const PAPER_SHADOW_TAPS: u32 = 12u;

// The radiance of the studio paper along the view ray through a pixel.
fn paper(position: vec2<f32>) -> vec3<f32> {
    let eye = frame.eye_exposure.xyz;
    let dir = pixel_direction(position);
    let wall = frame.backdrop_wall.xyz;
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let floor_y = frame.backdrop_floor.x;
    let cove = max(frame.backdrop_floor.y, 1e-3);
    // The nearer of the wall and the floor along the ray; a ray that meets neither (a camera
    // turned away from the wall, looking up) gets the wall's shading far away.
    var t = 1e4;
    let towards_wall = dot(dir, wall);
    if (towards_wall < -1e-5) {
        let hit = (frame.backdrop_wall.w - dot(eye, wall)) / towards_wall;
        if (hit > 0.0) {
            t = hit;
        }
    }
    if (dir.y < -1e-5) {
        let hit = (floor_y - eye.y) / dir.y;
        if (hit > 0.0) {
            t = min(t, hit);
        }
    }
    let p = eye + dir * t;
    // The cove: the normal turns from the floor's to the wall's over its radius on both sides of
    // the line where they meet, so the paper has no edge.
    let height = p.y - floor_y;
    let ahead = dot(p, wall) - frame.backdrop_wall.w;
    let turn = 0.5 + 0.5 * saturate(height / cove) - 0.5 * saturate(ahead / cove);
    let n = normalize(mix(up, wall, smoothstep(0.0, 1.0, turn)));
    // The paper is matte: the environment's irradiance and every light's, over pi.
    var radiance = sh_irradiance(n) * (frame.environment.x * INV_PI);
    let count = min(u32(frame.environment.w), MAX_LIGHTS);
    let pixel = global_pixel(position);
    for (var i = 0u; i < count; i++) {
        let light = frame.lights[i];
        // The subject's shadow: faded out towards the edge of the map (beyond it nothing is
        // known), and by how much softer than the filter can draw it the shadow should be.
        let shadow = shadow_sample(light, p, n, pixel, PAPER_SHADOW_TAPS);
        let known = smoothstep(0.0, 0.1, shadow.edge) / shadow.spread;
        let visibility = mix(1.0, shadow.visibility, known);
        radiance += shade_light(light, p, n, n, 1.0, vec3<f32>(0.0), visibility).diffuse;
    }
    let away = distance(p, frame.backdrop_spot.xyz) / max(frame.backdrop_floor.w, 1e-3);
    let falloff = 1.0 - frame.backdrop_floor.z * smoothstep(0.0, 1.0, away);
    return frame.background.rgb * radiance * falloff;
}

// The backdrop radiance behind a pixel.
fn background(position: vec2<f32>) -> vec3<f32> {
    if (frame.background.w < 0.5) {
        return frame.background.rgb;
    }
    if (frame.background.w > BACKGROUND_PAPER) {
        return paper(position);
    }
    let d = rotate_y(pixel_direction(position), -frame.environment.y);
    return textureSampleLevel(env_specular, env_sampler, d, 0.0).rgb * frame.environment.x;
}

@fragment
fn fs_composite(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(position.xy);
    let depth = textureLoad(depth_tex, p, 0);
    if (depth <= 0.0) {
        return vec4<f32>(background(position.xy), 1.0);
    }
    let surface = textureLoad(surface_tex, p, 0);
    let albedo = srgb_to_linear_rgb(unpack4x8unorm(surface.y).rgb);
    let diffuse = textureLoad(diffuse_tex, p, 0).rgb;
    let specular = textureLoad(specular_tex, p, 0).rgb;
    return vec4<f32>(diffuse * albedo + specular, 1.0);
}
