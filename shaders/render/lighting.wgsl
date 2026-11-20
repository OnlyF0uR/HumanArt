// render/lighting.wgsl: the light resources at group 2 and the shading of one surface point
// [LIT-01] [RND-03]: image-based light (L2 spherical-harmonic irradiance, the prefiltered GGX
// specular cube and the split-sum DFG table), directional, spot and rect-area lights, and shadow
// maps with PCSS-style soft shadows.
//
// Rect lights: the diffuse term is Lambert's exact irradiance from a polygon (edge integral,
// clamped at the horizon); the specular term uses the representative point (Karis 2013) with the
// lobe widened by the source size. Not LTC (see crates/render/src/light.rs).
// Needs lib/brdf.wgsl, lib/sampling.wgsl and render/frame.wgsl included before it; the including
// shader declares the bindings (group 2, bindings 0-5: env_specular, env_sampler, dfg_lut,
// irradiance_sh, shadow_atlas, shadow_sampler; see render/geometry.wgsl).

// The irradiance (lux per unit environment scale) arriving at a surface with normal `n`, from the
// L2 SH coefficients the IBL pass convolved with the clamped cosine.
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

// Prefiltered environment radiance around the reflection direction for a perceptual roughness.
fn env_prefiltered(r_world: vec3<f32>, roughness: f32) -> vec3<f32> {
    let r = rotate_y(r_world, -frame.environment.y);
    let lod = roughness * max(frame.environment.z - 1.0, 0.0);
    return textureSampleLevel(env_specular, env_sampler, r, lod).rgb;
}

// The split-sum (A, B) for a view cosine and perceptual roughness.
fn dfg_lookup(n_dot_v: f32, roughness: f32) -> vec2<f32> {
    return textureSampleLevel(dfg_lut, env_sampler, vec2<f32>(n_dot_v, roughness), 0.0).rg;
}

// The view distance of a perspective reversed-Z shadow depth.
fn shadow_linear_depth(depth: f32, near: f32, far: f32) -> f32 {
    let a = near / (far - near);
    return a * far / (depth + a);
}

// A Vogel-disk point `i` of `n`, rotated by `angle`, radius in [0, 1].
fn vogel_disk(i: u32, n: u32, angle: f32) -> vec2<f32> {
    let golden = 2.39996323;
    let r = sqrt((f32(i) + 0.5) / f32(n));
    let theta = f32(i) * golden + angle;
    return vec2<f32>(cos(theta), sin(theta)) * r;
}

// How much of a light reaches `world` through its shadow map (1 lit, 0 shadowed). `pixel` and the
// frame index seed the per-pixel rotation of the filter, which TAA resolves. `taps` is the quality
// variant's sample count.
fn shadow_visibility(light: Light, world: vec3<f32>, normal: vec3<f32>, pixel: vec2<f32>, taps: u32) -> f32 {
    let layer_f = light.shadow.x;
    if (layer_f < 0.0) {
        return 1.0;
    }
    let layer = min(u32(layer_f), MAX_SHADOWED - 1u);
    let params = frame.shadow_params[layer];
    let perspective = frame.shadow_kind[layer] > 0.5;
    let size = f32(max(frame.counters.z, 1u));
    // Normal offset of `bias` texels at the receiver.
    var clip = frame.shadow_view_proj[layer] * vec4<f32>(world, 1.0);
    var texel_world = 2.0 * params.z / size;
    if (perspective) {
        texel_world = 2.0 * clip.w * params.z / size;
    }
    clip = frame.shadow_view_proj[layer] * vec4<f32>(world + normal * (light.shadow.z * texel_world), 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    // Outside the map there are no casters. A receiver beyond the far plane is behind every
    // caster and one before the near plane in front of them all: clamping its depth keeps both
    // comparisons right (reversed-Z: larger is nearer the light).
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
        return 1.0;
    }
    let depth = clamp(ndc.z, 0.0, 1.0);
    let angle = TAU * interleaved_gradient_noise(pixel + 5.588238 * f32(frame.counters.x % 64u));
    var radius_uv = 1.5 / size;
    if (light.shadow.y > 0.5) {
        // Blocker search over the largest penumbra the source can make, in texels of the map.
        let search_uv = clamp(16.0 / size, 2.0 / size, 0.05);
        var blocker_sum = 0.0;
        var blockers = 0.0;
        let dims = vec2<f32>(size);
        for (var i = 0u; i < 16u; i++) {
            let o = vogel_disk(i, 16u, angle) * search_uv;
            let texel = vec2<i32>(clamp((uv + o) * dims, vec2<f32>(0.0), dims - 1.0));
            let d = textureLoad(shadow_atlas, texel, i32(layer), 0);
            // Reversed-Z: a blocker is nearer the light, so its depth is larger.
            if (d > depth) {
                blocker_sum += d;
                blockers += 1.0;
            }
        }
        if (blockers == 0.0) {
            return 1.0;
        }
        let blocker = blocker_sum / blockers;
        var penumbra_uv = 0.0;
        if (perspective) {
            let d_r = shadow_linear_depth(depth, params.x, params.y);
            let d_b = shadow_linear_depth(blocker, params.x, params.y);
            let width = params.w * max(d_r - d_b, 0.0) / max(d_b, 1e-4);
            penumbra_uv = width / max(2.0 * d_r * params.z, 1e-6);
        } else {
            let distance = (blocker - depth) * (params.y - params.x);
            penumbra_uv = distance * params.w / max(2.0 * params.z, 1e-6);
        }
        radius_uv = clamp(0.5 * penumbra_uv, 1.0 / size, 48.0 / size);
    }
    var lit = 0.0;
    let n = max(taps, 1u);
    for (var i = 0u; i < n; i++) {
        let o = vogel_disk(i, n, angle) * radius_uv;
        lit += textureSampleCompareLevel(shadow_atlas, shadow_sampler, uv + o, layer, depth);
    }
    return lit / f32(n);
}

// Lambert's irradiance factor of a polygon: the sum over edges of theta_i (n . gamma_i) / 2 for the
// unit vectors from the receiver to the four corners; times the luminance it is the irradiance.
fn polygon_form_factor(p: vec3<f32>, n: vec3<f32>, c0: vec3<f32>, c1: vec3<f32>, c2: vec3<f32>, c3: vec3<f32>) -> f32 {
    let v0 = normalize(c0 - p);
    let v1 = normalize(c1 - p);
    let v2 = normalize(c2 - p);
    let v3 = normalize(c3 - p);
    var sum = 0.0;
    sum += acos(clamp(dot(v0, v1), -1.0, 1.0)) * dot(safe_normalize(cross(v0, v1), vec3<f32>(0.0)), n);
    sum += acos(clamp(dot(v1, v2), -1.0, 1.0)) * dot(safe_normalize(cross(v1, v2), vec3<f32>(0.0)), n);
    sum += acos(clamp(dot(v2, v3), -1.0, 1.0)) * dot(safe_normalize(cross(v2, v3), vec3<f32>(0.0)), n);
    sum += acos(clamp(dot(v3, v0), -1.0, 1.0)) * dot(safe_normalize(cross(v3, v0), vec3<f32>(0.0)), n);
    return max(0.5 * sum, 0.0);
}

// The light arriving at a point from a directional or spot light: the unit direction towards the
// light and the irradiance on a surface facing it (no cosine, no shadow). Rect lights give zero
// (their diffuse is the polygon's form factor, see shade_light).
struct Incident {
    l: vec3<f32>,
    irradiance: vec3<f32>,
}

fn incident_light(light: Light, p: vec3<f32>) -> Incident {
    var out: Incident;
    out.l = -light.direction_cos_outer.xyz;
    out.irradiance = light.color_kind.rgb;
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
    } else if (kind == LIGHT_RECT) {
        out.irradiance = vec3<f32>(0.0);
    }
    return out;
}

// Diffuse (for unit albedo, already divided by pi) and specular radiance of one surface point.
struct Shading {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
}

// The light `light` at a point: position `p`, unit normal `n` and view `v`, GGX width `alpha`,
// Fresnel `f0`. `visibility` is the shadow term.
fn shade_light(light: Light, p: vec3<f32>, n: vec3<f32>, v: vec3<f32>, alpha: f32, f0: vec3<f32>, visibility: f32) -> Shading {
    var out: Shading;
    out.diffuse = vec3<f32>(0.0);
    out.specular = vec3<f32>(0.0);
    let kind = u32(light.color_kind.w);
    let radiance = light.color_kind.rgb;
    let n_dot_v = max(dot(n, v), 1e-4);
    var l = -light.direction_cos_outer.xyz;
    var illuminance = 1.0;
    var spec_alpha = alpha;
    var spec_norm = 1.0;
    if (kind == LIGHT_SPOT) {
        let to_light = light.position_range.xyz - p;
        let d2 = max(dot(to_light, to_light), 1e-8);
        l = to_light * inverseSqrt(d2);
        let cos_axis = dot(-l, light.direction_cos_outer.xyz);
        let cone = saturate((cos_axis - light.direction_cos_outer.w) / max(light.right_cos_inner.w - light.direction_cos_outer.w, 1e-4));
        let range = max(light.position_range.w, 1e-4);
        let window = sq(saturate(1.0 - sq(d2 / (range * range))));
        illuminance = cone * cone * window / d2;
        // Sphere-light widening of the lobe (Karis 2013), normalised.
        spec_alpha = saturate(alpha + light.up_size.w / (2.0 * sqrt(d2)));
        spec_norm = sq(alpha / spec_alpha);
    } else if (kind == LIGHT_RECT) {
        let center = light.position_range.xyz;
        let dir = light.direction_cos_outer.xyz;
        if (dot(p - center, dir) <= 0.0) {
            return out;
        }
        let r = light.right_cos_inner.xyz;
        let u = light.up_size.xyz;
        let factor = polygon_form_factor(p, n, center - r - u, center + r - u, center + r + u, center - r + u);
        out.diffuse = radiance * factor * INV_PI * visibility;
        // Representative point: where the reflected ray meets the panel, clamped to it.
        let refl = reflect(-v, n);
        let denom = dot(refl, dir);
        var q = center;
        if (denom < -1e-4) {
            let t = dot(center - p, dir) / denom;
            let hit = p + refl * max(t, 0.0);
            let local = hit - center;
            let half_w = max(length(r), 1e-6);
            let half_h = max(length(u), 1e-6);
            let x = clamp(dot(local, r) / half_w, -half_w, half_w);
            let y = clamp(dot(local, u) / half_h, -half_h, half_h);
            q = center + r / half_w * x + u / half_h * y;
        }
        let to_q = q - p;
        let dq = max(length(to_q), 1e-4);
        l = to_q / dq;
        let n_dot_l = dot(n, l);
        if (n_dot_l <= 0.0) {
            return out;
        }
        let extent = length(r) + length(u);
        spec_alpha = saturate(alpha + extent / (2.0 * dq));
        spec_norm = sq(alpha / spec_alpha);
        let h = normalize(l + v);
        let spec = specular_ggx(n_dot_v, n_dot_l, max(dot(n, h), 0.0), max(dot(v, h), 0.0), spec_alpha, f0);
        // The form factor already holds the cosine and the solid angle.
        out.specular = radiance * spec * spec_norm * factor * visibility;
        return out;
    }
    let n_dot_l = dot(n, l);
    if (n_dot_l <= 0.0) {
        return out;
    }
    let h = normalize(l + v);
    let e = radiance * illuminance * n_dot_l * visibility;
    out.diffuse = e * INV_PI;
    let spec = specular_ggx(n_dot_v, n_dot_l, max(dot(n, h), 0.0), max(dot(v, h), 0.0), spec_alpha, f0);
    out.specular = e * spec * spec_norm;
    return out;
}
