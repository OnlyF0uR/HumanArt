// hair/resolve.wgsl: deferred shading of the hair visibility buffer and its composite over the lit
// scene [HR-01] [HR-02] [HR-03]. One full-screen pass into the HDR target with premultiplied
// blending (src + dst (1 - src.a)):
// - where a strand is the nearest fibre of the pixel, its record is shaded: every light through
//   hair/bsdf.wgsl's dual scattering, with the expected number of fibres towards the light from
//   the light's deep opacity map (or, for lights without one, a march through the fibre density
//   grid) and the opaque scene's shadow map; the environment's L2 irradiance scattered back and
//   forward by the fibre, attenuated by the density grid (ambient occlusion of the inner layers,
//   lighter for light hair, which scatters forward); the pixel's alpha is
//   1 - (1 - c0) exp(-(coverage sum - c0)) with c0 the nearest fibre's own coverage (exact for
//   the front fibre, the expected opacity of independent fibres behind it);
// - behind the hair, the opaque scene (skin) is darkened where the hair shadows it: each light's
//   share of the pixel's irradiance (Lambert on the G-buffer normal, shadow-mapped) is scaled by
//   the deep opacity map's transmittance and the ambient share by the density grid's occlusion
//   (scalp occlusion). A screen-space approximation: specular and subsurface light are scaled
//   by the same factor as the diffuse.

#include "lib/math.wgsl"
#include "lib/packing.wgsl"
#include "render/frame.wgsl"
#include "hair/common.wgsl"
#include "hair/bsdf.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> view: HairView;

@group(1) @binding(0) var<storage, read> hair_depth: array<u32>;
@group(1) @binding(1) var<storage, read> hair_coverage: array<u32>;
@group(1) @binding(2) var<storage, read> hair_records: array<vec4<u32>>;
@group(1) @binding(3) var<storage, read> hair_voxels: array<u32>;
@group(1) @binding(4) var<storage, read> irradiance_sh: array<vec4<f32>, 9>;
@group(1) @binding(5) var hair_azimuthal: texture_3d<f32>;
@group(1) @binding(6) var hair_dual: texture_2d_array<f32>;
@group(1) @binding(7) var hair_lut_sampler: sampler;
@group(1) @binding(8) var dom_depth: texture_depth_2d_array;
@group(1) @binding(9) var dom_opacity: texture_2d_array<f32>;
@group(1) @binding(10) var shadow_atlas: texture_depth_2d_array;
@group(1) @binding(11) var shadow_sampler: sampler_comparison;
@group(1) @binding(12) var scene_depth: texture_depth_2d;
@group(1) @binding(13) var scene_surface: texture_2d<u32>;

const LUMINANCE: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

@vertex
fn vs_resolve(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_position(index);
}

// The L2 SH irradiance (per unit environment scale) on a surface facing `n`.
fn sh_irradiance(n_world: vec3<f32>) -> vec3<f32> {
    let n = rotate_y(n_world, -frame.environment.y);
    var e = irradiance_sh[0].rgb * 0.282095;
    e += irradiance_sh[1].rgb * (0.488603 * n.y);
    e += irradiance_sh[2].rgb * (0.488603 * n.z);
    e += irradiance_sh[3].rgb * (0.488603 * n.x);
    e += irradiance_sh[4].rgb * (1.092548 * n.x * n.y);
    e += irradiance_sh[5].rgb * (1.092548 * n.y * n.z);
    e += irradiance_sh[6].rgb * (0.315392 * (3.0 * n.z * n.z - 1.0));
    e += irradiance_sh[7].rgb * (1.092548 * n.x * n.z);
    e += irradiance_sh[8].rgb * (0.546274 * (n.x * n.x - n.y * n.y));
    return max(e, vec3<f32>(0.0));
}

// The world point of a pixel at reversed-Z depth `z`.
fn world_at(position: vec2<f32>, z: f32) -> vec3<f32> {
    let h = frame.inv_view_proj * vec4<f32>(pixel_to_ndc(position), z, 1.0);
    return h.xyz / h.w;
}

struct HairIncident {
    l: vec3<f32>,
    irradiance: vec3<f32>,
    // The variance (rad^2) of the source's directions about `l` along one axis: a disc of angular
    // radius a has a^2 / 4.
    spread: f32,
}

// Direction towards, irradiance from and angular spread of a light at `p` (rect lights as their
// centre with the panel's solid angle; the spread of a disc of the same solid angle).
fn hair_incident(light: Light, p: vec3<f32>) -> HairIncident {
    var out: HairIncident;
    out.l = -light.direction_cos_outer.xyz;
    out.irradiance = light.color_kind.rgb;
    // Directional: up_size.w is the angular diameter.
    out.spread = 0.0625 * sq(light.up_size.w);
    let kind = u32(light.color_kind.w);
    if (kind == LIGHT_SPOT) {
        let to_light = light.position_range.xyz - p;
        let d2 = max(dot(to_light, to_light), 1e-8);
        out.l = to_light * inverseSqrt(d2);
        let cos_axis = dot(-out.l, light.direction_cos_outer.xyz);
        let cone = saturate((cos_axis - light.direction_cos_outer.w) / max(light.right_cos_inner.w - light.direction_cos_outer.w, 1e-4));
        let range = max(light.position_range.w, 1e-4);
        let window = sq(saturate(1.0 - sq(d2 / (range * range))));
        out.irradiance = light.color_kind.rgb * (cone * cone * window / d2);
        out.spread = 0.25 * sq(light.up_size.w) / d2;
    } else if (kind == LIGHT_RECT) {
        let to_light = light.position_range.xyz - p;
        let d2 = max(dot(to_light, to_light), 1e-8);
        out.l = to_light * inverseSqrt(d2);
        let area = 4.0 * length(light.right_cos_inner.xyz) * length(light.up_size.xyz);
        let facing = max(dot(-out.l, light.direction_cos_outer.xyz), 0.0);
        out.irradiance = light.color_kind.rgb * (area * facing / d2);
        // Solid angle = pi a^2.
        out.spread = 0.25 * min(area * facing / d2, PI) * INV_PI;
    }
    return out;
}

// The opaque scene's shadow map at `p` for `light` (4-tap PCF, offset 2 texels towards the light).
fn scene_visibility(light: Light, p: vec3<f32>, l: vec3<f32>) -> f32 {
    let layer_f = light.shadow.x;
    if (layer_f < 0.0) {
        return 1.0;
    }
    let layer = min(u32(layer_f), MAX_SHADOWED - 1u);
    let params = frame.shadow_params[layer];
    let size = f32(max(frame.counters.z, 1u));
    var clip = frame.shadow_view_proj[layer] * vec4<f32>(p, 1.0);
    var texel = 2.0 * params.z / size;
    if (frame.shadow_kind[layer] > 0.5) {
        texel = 2.0 * clip.w * params.z / size;
    }
    clip = frame.shadow_view_proj[layer] * vec4<f32>(p + l * (2.0 * texel), 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
        return 1.0;
    }
    let depth = clamp(ndc.z, 0.0, 1.0);
    let o = 0.5 / size;
    var lit = 0.0;
    lit += textureSampleCompareLevel(shadow_atlas, shadow_sampler, uv + vec2<f32>(-o, -o), layer, depth);
    lit += textureSampleCompareLevel(shadow_atlas, shadow_sampler, uv + vec2<f32>(o, -o), layer, depth);
    lit += textureSampleCompareLevel(shadow_atlas, shadow_sampler, uv + vec2<f32>(-o, o), layer, depth);
    lit += textureSampleCompareLevel(shadow_atlas, shadow_sampler, uv + vec2<f32>(o, o), layer, depth);
    return 0.25 * lit;
}

// Steps of the screen-space march of scene_contact.
const HAIR_CONTACT_STEPS: u32 = 10u;
// How deep behind the depth buffer's surface a march point still counts as inside the occluder,
// metres (a head is about 0.2 m deep; anything further behind is taken to be open space).
const HAIR_CONTACT_THICKNESS: f32 = 0.25;
// Distance behind the surface over which a march point goes from open to blocked, metres (keeps
// the depth buffer's own steps and the jitter from showing as a hard line).
const HAIR_CONTACT_RAMP: f32 = 0.003;

// Light passing the opaque scene from `p` towards unit `l`, measured in screen space: a light
// without a shadow map would otherwise light hair through the head (an unshadowed rim light from
// behind lit the lashes in front of the eyes through the skull, forward-scattered straight at the
// camera: the pale fan of the eye macros). Points march from 2 mm to the hair grid's extent with
// growing steps; a point behind the depth buffer's surface (by up to HAIR_CONTACT_THICKNESS) is
// inside an occluder. Points off screen or over the backdrop pass.
fn scene_contact(p: vec3<f32>, l: vec3<f32>) -> f32 {
    let reach = clamp(f32(view.counts.x) * view.voxel.w, 0.02, 1.0);
    let eye = frame.eye_exposure.xyz;
    let size = vec2<i32>(textureDimensions(scene_depth));
    var through = 1.0;
    for (var k = 1u; k <= HAIR_CONTACT_STEPS; k++) {
        let f = f32(k) / f32(HAIR_CONTACT_STEPS);
        let q = p + l * (0.002 + reach * f * f);
        let clip = frame.unjittered_view_proj * vec4<f32>(q, 1.0);
        if (clip.w <= 1e-5) {
            break;
        }
        let ndc = clip.xy / clip.w;
        let pixel = vec2<i32>(floor(vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * frame.viewport.xy));
        if (any(pixel < vec2<i32>(0)) || any(pixel >= size)) {
            break;
        }
        let z = textureLoad(scene_depth, pixel, 0);
        if (z <= 0.0) {
            continue;
        }
        let surface = world_at(vec2<f32>(pixel) + vec2<f32>(0.5), z);
        let behind = distance(q, eye) - distance(surface, eye);
        if (behind > 0.0 && behind < HAIR_CONTACT_THICKNESS) {
            through = min(through, 1.0 - saturate(behind / HAIR_CONTACT_RAMP));
            if (through <= 0.0) {
                break;
            }
        }
    }
    return through;
}

// The fibre side area (voxel units) of grid cell `c`, 0 outside the grid.
fn voxel_cell(c: vec3<i32>, n: i32) -> f32 {
    if (any(c < vec3<i32>(0)) || any(c >= vec3<i32>(n))) {
        return 0.0;
    }
    return f32(hair_voxels[u32((c.z * n + c.y) * n + c.x)]);
}

// Optical depth per metre of the fibre density grid at `p`, trilinear between cell centres (a
// nearest-cell lookup shows the grid as blocks in the occlusion).
fn voxel_density(p: vec3<f32>) -> f32 {
    let n = i32(view.counts.x);
    let x = (p - view.voxel.xyz) / view.voxel.w - vec3<f32>(0.5);
    if (n == 0 || any(x < vec3<f32>(-1.0)) || any(x >= vec3<f32>(f32(n)))) {
        return 0.0;
    }
    let base = floor(x);
    let f = x - base;
    let c = vec3<i32>(base);
    let x00 = mix(voxel_cell(c, n), voxel_cell(c + vec3<i32>(1, 0, 0), n), f.x);
    let x10 = mix(voxel_cell(c + vec3<i32>(0, 1, 0), n), voxel_cell(c + vec3<i32>(1, 1, 0), n), f.x);
    let x01 = mix(voxel_cell(c + vec3<i32>(0, 0, 1), n), voxel_cell(c + vec3<i32>(1, 0, 1), n), f.x);
    let x11 = mix(voxel_cell(c + vec3<i32>(0, 1, 1), n), voxel_cell(c + vec3<i32>(1, 1, 1), n), f.x);
    let units = mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
    return units * HAIR_VOXEL_UNIT * view.shifts.w / (view.voxel.w * view.voxel.w * view.voxel.w);
}

// Optical depth from `p` along unit `d` through the density grid: four steps of 1, 1, 2 and 4
// cells starting half a cell out.
fn voxel_tau(p: vec3<f32>, d: vec3<f32>) -> f32 {
    let s = view.voxel.w;
    var tau = 0.0;
    tau += voxel_density(p + d * (1.0 * s)) * s;
    tau += voxel_density(p + d * (2.0 * s)) * s;
    tau += voxel_density(p + d * (3.5 * s)) * (2.0 * s);
    tau += voxel_density(p + d * (6.5 * s)) * (4.0 * s);
    return tau;
}

// The expected fibres crossed from `p` towards light slot `slot`: its deep opacity map, or the
// density grid along `l`.
fn hair_tau(slot: u32, p: vec3<f32>, l: vec3<f32>) -> f32 {
    for (var m = 0u; m < min(view.counts.y, HAIR_MAX_DOMS); m++) {
        let params = view.dom_params[m];
        if (u32(params.y) != slot) {
            continue;
        }
        let local = hair_dom_local(p, view.dom_right[m], view.dom_up[m], view.dom_forward[m], params.x);
        let uv = vec2<f32>(local.x * 0.5 + 0.5, 0.5 - local.y * 0.5);
        if (any(uv <= vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0))) {
            return 0.0;
        }
        // Bilinear over the four nearest texels of the optical depth itself (a nearest lookup of
        // the first-fibre depth shows the map's texels as steps where the hair's edge crosses).
        let dims = vec2<i32>(textureDimensions(dom_depth).xy);
        let x = uv * vec2<f32>(dims) - vec2<f32>(0.5);
        let base = vec2<i32>(floor(x));
        let f = x - floor(x);
        var taus = vec4<f32>(0.0);
        for (var k = 0; k < 4; k++) {
            let texel = clamp(base + vec2<i32>(k & 1, k >> 1u), vec2<i32>(0), dims - vec2<i32>(1));
            let first = textureLoad(dom_depth, texel, i32(m), 0);
            if (first > 0.0) {
                let behind = (first - local.z) * 2.0 * params.x - params.w;
                let layers = textureLoad(dom_opacity, texel, i32(m), 0);
                taus[k] = hair_dom_tau(layers, view.dom_layers, behind);
            }
        }
        return mix(mix(taus.x, taus.y, f.x), mix(taus.z, taus.w, f.x), f.y);
    }
    return voxel_tau(p, l);
}

// The six axis directions of the ambient occlusion estimate.
fn axis_dir(k: u32) -> vec3<f32> {
    var d = vec3<f32>(0.0);
    d[k % 3u] = select(1.0, -1.0, k >= 3u);
    return d;
}

// Fraction of the background's light the hair takes away at a scene pixel (1 = none).
fn skin_factor(position: vec2<f32>, z: f32) -> f32 {
    let p = world_at(position, z);
    let n = unpack_oct16(textureLoad(scene_surface, vec2<i32>(position), 0).x);
    let env = frame.environment.x;
    let ambient = dot(sh_irradiance(n), LUMINANCE) * env;
    var occluded_ambient = 0.0;
    if ((view.counts.w & HAIR_FLAG_OCCLUSION) != 0u) {
        // Occlusion of the upper hemisphere around the normal by the fibre density.
        var tau = voxel_tau(p, n);
        for (var k = 0u; k < 6u; k++) {
            let d = axis_dir(k);
            if (dot(d, n) > 0.3) {
                tau = max(tau, voxel_tau(p, normalize(d + n)));
            }
        }
        occluded_ambient = ambient * (1.0 - exp(-tau));
    }
    var total = ambient;
    var shadowed = occluded_ambient;
    let count = min(u32(frame.environment.w), MAX_LIGHTS);
    for (var i = 0u; i < count; i++) {
        let light = frame.lights[i];
        let inc = hair_incident(light, p);
        let e = dot(inc.irradiance, LUMINANCE) * max(dot(n, inc.l), 0.0);
        if (e <= 0.0) {
            continue;
        }
        let lit = e * scene_visibility(light, p, inc.l);
        total += lit;
        shadowed += lit * (1.0 - exp(-hair_tau(i, p, inc.l)));
    }
    if (total <= 0.0) {
        return 1.0;
    }
    return saturate(1.0 - shadowed / total);
}

@fragment
fn fs_resolve(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<u32>(position.xy);
    let index = pixel.y * u32(view.screen.x) + pixel.x;
    let bits = hair_depth[index];
    let scene_z = textureLoad(scene_depth, vec2<i32>(pixel), 0);
    var keep = 1.0;
    if (scene_z > 0.0 && (view.counts.w & HAIR_FLAG_SKIN_SHADOW) != 0u) {
        keep = skin_factor(position.xy, scene_z);
    }
    if (bits == 0u) {
        if (keep >= 0.9999) {
            discard;
        }
        return vec4<f32>(0.0, 0.0, 0.0, 1.0 - keep);
    }
    // Opacity: the nearest fibre's own coverage exactly, the rest as independent fibres
    // (1 - exp(-coverage)). The exponential alone left a fibre that covers the whole pixel 63 %
    // opaque, so wide fibres up close (lashes and brows in a macro, scalp hair in a close-up)
    // showed the skin through them: pale, see-through hair.
    let record = hair_records[index];
    let alpha = hair_alpha(hair_coverage[index], record);
    let p = world_at(position.xy, bitcast<f32>(bits));
    let t = unpack_oct16(record.x);
    let groom = min(hair_record_groom(record), HAIR_MAX_GROOMS - 1u);
    let sigma = hair_sigma_a(view.grooms[groom], hair_record_color(record), hair_record_grey(record));
    let wo = safe_normalize(frame.eye_exposure.xyz - p, vec3<f32>(0.0, 0.0, 1.0));
    let fibre = HairFibre(view.variances, vec4<f32>(view.shifts.xyz, 0.0));
    let dual = (view.counts.w & HAIR_FLAG_DUAL) != 0u;
    var radiance = vec3<f32>(0.0);
    let count = min(u32(frame.environment.w), MAX_LIGHTS);
    for (var i = 0u; i < count; i++) {
        let light = frame.lights[i];
        let inc = hair_incident(light, p);
        if (max3(inc.irradiance) <= 0.0) {
            continue;
        }
        // The shadow map where the light has one, else the screen-space march.
        var visibility = 1.0;
        if (light.shadow.x < 0.0) {
            visibility = scene_contact(p, inc.l);
        } else {
            visibility = scene_visibility(light, p, inc.l);
        }
        if (visibility <= 0.0) {
            continue;
        }
        let tau = hair_tau(i, p, inc.l);
        var f: vec3<f32>;
        if (dual) {
            f = hair_dual_scattering(sigma, t, wo, inc.l, tau, fibre, inc.spread);
        } else {
            let so = dot(wo, t);
            let si = dot(inc.l, t);
            let po = wo - t * so;
            let pi = inc.l - t * si;
            let phi = acos(clamp(dot(po, pi) * inverseSqrt(max(dot(po, po) * dot(pi, pi), 1e-12)), -1.0, 1.0));
            let o = vec2<f32>(so, sqrt(max(1.0 - so * so, 0.0)));
            let li = vec2<f32>(si, sqrt(max(1.0 - si * si, 0.0)));
            f = hair_single(sigma, o, li, phi, fibre, inc.spread) * exp(-tau);
        }
        radiance += inc.irradiance * visibility * f;
    }
    // Environment: back-scattered from the viewer's side, forward-scattered from behind.
    let side = safe_normalize(wo - t * dot(wo, t), wo);
    let albedo = hair_albedo(0.25 * PI, sigma);
    var occlusion = vec3<f32>(1.0);
    if ((view.counts.w & HAIR_FLAG_OCCLUSION) != 0u) {
        // The mean transmittance over the six axis directions (the outer side of the hair sees
        // the sky, the inner side the dense layers), each lightened by the forward albedo.
        occlusion = vec3<f32>(0.0);
        for (var k = 0u; k < 6u; k++) {
            occlusion += exp(-voxel_tau(p, axis_dir(k)) * (vec3<f32>(1.0) - albedo[0]));
        }
        occlusion /= 6.0;
    }
    let env = frame.environment.x * INV_PI;
    // The forward-scattered light comes from behind the fibre: through the head for hair lying on
    // the scalp and for lashes in front of the face, so the scene blocks it.
    let behind = scene_contact(p, -side);
    radiance += (albedo[1] * sh_irradiance(side) + albedo[0] * behind * sh_irradiance(-side)) * env * occlusion;
    return vec4<f32>(radiance * alpha, 1.0 - (1.0 - alpha) * keep);
}
