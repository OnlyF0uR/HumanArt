// render/subsurface.wgsl: what the geometry pass adds for skin (material class 1) [SHD-01]..[SHD-05]:
// the transmittance of the diffusion profile through thin geometry, the thickness the light
// crosses from its shadow map [SHD-03], and the pre-integrated diffuse falloff of the Interactive
// tier [SHD-04]. The profile is d'Eon & Luebke's six-Gaussian fit to the three-layer skin model;
// the derivations and the CPU reference are in crates/render/src/sss.rs (`transmittance`,
// `preintegrated_table`), see also docs/shader-notes.md.
//
// Needs lib/brdf.wgsl, render/frame.wgsl and render/lighting.wgsl included before it; the
// including shader declares `preintegrated_lut` (group 2, binding 6, the table of
// `sss::preintegrated_table`) next to the lighting bindings.

const SKIN_CLASS: u32 = 1u;
// Mirrors sss::PREINTEGRATED_SIZE and sss::PREINTEGRATED_MAX_INV_RADIUS.
const PREINTEGRATED_SIZE: f32 = 32.0;
const PREINTEGRATED_MAX_INV_RADIUS: f32 = 0.5;
// How far the shadow-map lookup of the thickness moves into the surface, metres (Jimenez et al.
// 2010 shrink the position into the surface). Kept well below the thickness of an ear rim
// (2 to 5 mm): a lookup point pushed through the back surface would see no material at all.
const THICKNESS_SHRINK: f32 = 0.0005;
// THICKNESS_SHRINK in millimetres.
const THICKNESS_SHRINK_MM: f32 = 0.5;
// The wrap of the transmitted light around the back of the surface (Jimenez et al. 2010).
const TRANSLUCENCY_WRAP: f32 = 0.3;

// Gaussian i of the profile at squared distance `d2` (mm²) after the profile's integral over the
// back plane: exp(-d² / 2v).
fn profile_plane(d2: f32, v: f32) -> f32 {
    return exp(-d2 / (2.0 * v));
}

// The transmittance of the profile through `d_mm` of skin for a profile `scale`, per channel
// (sss::transmittance): sum_i w_i exp(-d² / 2 v_i scale²).
fn skin_transmittance(d_mm: f32, scale: f32) -> vec3<f32> {
    let d2 = d_mm * d_mm / max(scale * scale, 1e-6);
    return vec3<f32>(0.233, 0.455, 0.649) * profile_plane(d2, 0.0064)
        + vec3<f32>(0.100, 0.336, 0.344) * profile_plane(d2, 0.0484)
        + vec3<f32>(0.118, 0.198, 0.0) * profile_plane(d2, 0.187)
        + vec3<f32>(0.113, 0.007, 0.007) * profile_plane(d2, 0.567)
        + vec3<f32>(0.358, 0.004, 0.0) * profile_plane(d2, 1.99)
        + vec3<f32>(0.078, 0.0, 0.0) * profile_plane(d2, 7.41);
}

// The slab of tissue between the surface a light enters and `world` (moved THICKNESS_SHRINK into
// the surface along `normal`), from the light's shadow map: x the distance in metres the light
// travels inside the body, negative when the light has no shadow map or the point is outside it;
// y the cosine between the light and the surface it entered through (the irradiance that entered
// is the light's times this).
//
// The occluder's depth is interpolated between the four texels around the lookup: a single texel
// on a surface that is grazing to the light is off by a texel's width times the slope, which read
// as millimetres of tissue on skin the light reaches directly (dark-red blotches around the lids
// under a side light, todo 1.4 / 1.9). The slope of the same four texels gives the cosine: at the
// foot of a fold or a bump the light enters at a grazing angle and little of it gets in, where the
// slab formula alone gave the full light (a deep-red band along every shadow edge).
fn shadow_slab(light: Light, world: vec3<f32>, normal: vec3<f32>) -> vec2<f32> {
    let layer_f = light.shadow.x;
    if (layer_f < 0.0) {
        return vec2<f32>(-1.0, 0.0);
    }
    let layer = min(u32(layer_f), MAX_SHADOWED - 1u);
    let params = frame.shadow_params[layer];
    let clip = frame.shadow_view_proj[layer] * vec4<f32>(world - normal * THICKNESS_SHRINK, 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
        return vec2<f32>(-1.0, 0.0);
    }
    let size = f32(max(frame.counters.z, 1u));
    let st = clamp(uv * size - 0.5, vec2<f32>(0.0), vec2<f32>(size - 1.0));
    let base = vec2<i32>(st);
    let f = fract(st);
    let last = i32(size) - 1;
    let x1 = min(base.x + 1, last);
    let y1 = min(base.y + 1, last);
    let perspective = frame.shadow_kind[layer] > 0.5;
    // The four occluder depths as distances along the light, metres.
    var d00 = textureLoad(shadow_atlas, base, i32(layer), 0);
    var d10 = textureLoad(shadow_atlas, vec2<i32>(x1, base.y), i32(layer), 0);
    var d01 = textureLoad(shadow_atlas, vec2<i32>(base.x, y1), i32(layer), 0);
    var d11 = textureLoad(shadow_atlas, vec2<i32>(x1, y1), i32(layer), 0);
    var here = clamp(ndc.z, 0.0, 1.0);
    var texel = 2.0 * params.z / size;
    if (perspective) {
        d00 = shadow_linear_depth(d00, params.x, params.y);
        d10 = shadow_linear_depth(d10, params.x, params.y);
        d01 = shadow_linear_depth(d01, params.x, params.y);
        d11 = shadow_linear_depth(d11, params.x, params.y);
        here = shadow_linear_depth(here, params.x, params.y);
        texel = 2.0 * d00 * params.z / size;
    } else {
        // Reversed-Z: the distance from the near plane grows as the depth falls.
        let range = params.y - params.x;
        d00 = (1.0 - d00) * range;
        d10 = (1.0 - d10) * range;
        d01 = (1.0 - d01) * range;
        d11 = (1.0 - d11) * range;
        here = (1.0 - here) * range;
    }
    let occluder = mix(mix(d00, d10, f.x), mix(d01, d11, f.x), f.y);
    let slope = vec2<f32>(mix(d10 - d00, d11 - d01, f.y), mix(d01 - d00, d11 - d10, f.x)) / max(texel, 1e-9);
    return vec2<f32>(max(here - occluder, 0.0), inverseSqrt(1.0 + dot(slope, slope)));
}

// The distance in metres the light travels inside the body before it reaches `world` (the x of
// `shadow_slab`). Negative when the light has no shadow map or the point is outside it.
fn shadow_thickness(light: Light, world: vec3<f32>, normal: vec3<f32>) -> f32 {
    return shadow_slab(light, world, normal).x;
}

// The pre-integrated diffuse falloff per channel for the cosine between the normal and the light
// and `inv_radius` = profile scale / curvature radius (1/mm), from the table.
fn preintegrated_falloff(cos_theta: f32, inv_radius: f32) -> vec3<f32> {
    let uv = vec2<f32>(saturate(cos_theta * 0.5 + 0.5), saturate(inv_radius / PREINTEGRATED_MAX_INV_RADIUS));
    // Texel centres span [0.5, size - 0.5].
    let st = (uv * (PREINTEGRATED_SIZE - 1.0) + 0.5) / PREINTEGRATED_SIZE;
    return textureSampleLevel(preintegrated_lut, env_sampler, st, 0.0).rgb;
}
