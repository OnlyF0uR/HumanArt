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

// The cosine and sine of the golden angle (pi (3 - sqrt 5)): the turn between two Vogel points.
const GOLDEN_TURN: vec2<f32> = vec2<f32>(-0.7373688780783197, 0.6754902942615238);

// The unit direction `d` turned by the golden angle. The shadow loops walk a Vogel disk of `n`
// points with it: point `i` is the direction (cos angle, sin angle) turned `i` times, at radius
// sqrt((i + 0.5) / n), without a sine and cosine per tap (todo 1.9, the viewport's budget).
fn golden_turn(d: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(d.x * GOLDEN_TURN.x - d.y * GOLDEN_TURN.y, d.x * GOLDEN_TURN.y + d.y * GOLDEN_TURN.x);
}

// Shadow maps (todo 1.9). The map of a light covers what the camera sees of the character
// (crates/render/src/focus.rs), so a texel on a face is a fraction of a millimetre.
//
// The widest filter and blocker search, texels of the map per PCF tap (the taps spread over it;
// their rotation per pixel and per frame is resolved by the temporal accumulation):
// Interactive's 8 taps reach 24 texels, Review's 16 reach 48, Still's 32 reach 96. A wide filter
// costs its cache misses, so only the still tier affords the widest penumbrae. The taps go out four
// at a time (independent fetches): on the 2560 x 1440 Review frame of tests/timings.rs that took the
// shadows' share of the geometry pass from 1.35 to about 0.7 ms.
const SHADOW_RADIUS_TEXELS_PER_TAP: f32 = 3.0;
// Depth fetches of the blocker search: half the PCF taps, at least 8 (an estimate of the mean
// blocker depth, which the temporal accumulation averages further), a multiple of 4.
fn shadow_search_taps(taps: u32) -> u32 {
    return (max(taps / 2u, 8u) + 3u) / 4u * 4u;
}
// Comparisons of the filter: the quality's taps rounded up to a multiple of 4 (8, 12, 16 and 32
// already are), at least 4.
fn shadow_filter_taps(taps: u32) -> u32 {
    return max((taps + 3u) / 4u, 1u) * 4u;
}
// The narrowest filter, texels: the width of the hardest shadow edge.
const SHADOW_MIN_RADIUS_TEXELS: f32 = 1.25;
// The narrowest filter of a soft light, metres: with texels of a quarter millimetre the
// one-texel shadow of a crease read as a scratch on the skin.
const SHADOW_MIN_PENUMBRA: f32 = 0.0008;
// The least normal offset, metres per unit of the light's bias (texels).
const SHADOW_MIN_OFFSET: f32 = 0.0003;
// The normal offset of a surface edge-on to the light, metres: near the terminator the flat
// triangles of the mesh shadow each other where the smooth surface would not (short dark strokes
// along concave edges once the texels are finer than the triangles' sagitta). It falls with the
// fourth power of the sine, so a surface that faces the light keeps its shadows attached.
const SHADOW_TERMINATOR_OFFSET: f32 = 0.0015;
// The steepest receiver the receiver-plane bias follows: metres of depth per metre across the map.
const SHADOW_MAX_SLOPE: f32 = 6.0;
// How much a receiver may curve toward the light away from its tangent plane, per metre squared of
// tap distance (half the curvature of a hollow of 15 mm radius): the receiver-plane comparison
// counted the rim of a concave hollow as a blocker of its own floor, which put dark specks in the
// jugular notch and the hollows over the clavicles under a frontal light (todo 1.1). A real blocker
// stands off the surface by far more than this over the few millimetres the taps reach.
const SHADOW_CURVATURE: f32 = 33.0;

// What a shadow lookup gives: how much of the light arrives (1 lit, 0 shadowed); how many times
// wider than the filter could reach the penumbra should have been (>= 1: the backdrop fades a
// shadow that ought to be that much softer); and the distance from the lookup to the edge of the
// map in map widths (negative outside, where nothing is known and the point counts as lit).
struct ShadowSample {
    visibility: f32,
    spread: f32,
    edge: f32,
}

// Shadow clip coordinates of a world point: uv (xy) and reversed-Z depth (z).
fn shadow_coords(layer: u32, world: vec3<f32>) -> vec3<f32> {
    let clip = frame.shadow_view_proj[layer] * vec4<f32>(world, 1.0);
    let ndc = clip.xyz / clip.w;
    return vec3<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
}

// The shadow lookup behind `shadow_visibility`.
//
// Bias: the receiver moves along its normal by the light's bias in texels, scaled by the sine of
// the angle between the normal and the light (a surface facing the light needs none, so a nostril
// rim or a lid keeps its shadow attached), and every tap compares against the receiver's own
// plane (the depth the receiver has under that tap), so a wide filter on a slanted cheek does not
// shadow itself. Penumbra: the average blocker depth over the region the light's size can reach
// gives the filter radius, which grows with the distance between blocker and receiver.
fn shadow_sample(light: Light, world: vec3<f32>, normal: vec3<f32>, pixel: vec2<f32>, taps: u32) -> ShadowSample {
    var out: ShadowSample;
    out.visibility = 1.0;
    out.spread = 1.0;
    out.edge = -1.0;
    let layer_f = light.shadow.x;
    if (layer_f < 0.0) {
        return out;
    }
    let layer = min(u32(layer_f), MAX_SHADOWED - 1u);
    let params = frame.shadow_params[layer];
    let perspective = frame.shadow_kind[layer] > 0.5;
    let size = f32(max(frame.counters.z, 1u));
    // The direction to the light and the size of a texel at the receiver.
    var to_light = -light.direction_cos_outer.xyz;
    var texel_world = 2.0 * params.z / size;
    var receiver_distance = 1.0;
    if (perspective) {
        let delta = light.position_range.xyz - world;
        receiver_distance = max(length(delta), 1e-4);
        to_light = delta / receiver_distance;
        texel_world = 2.0 * receiver_distance * params.z / size;
    }
    let n_dot_l = dot(normal, to_light);
    let sine = sqrt(saturate(1.0 - n_dot_l * n_dot_l));
    let grazing = sine * sine * sine * sine;
    let origin = world + normal * (light.shadow.z * max(texel_world, SHADOW_MIN_OFFSET) * (0.2 + 0.8 * sine) + SHADOW_TERMINATOR_OFFSET * grazing);
    let c = shadow_coords(layer, origin);
    let uv = c.xy;
    // Outside the map there are no casters. A receiver beyond the far plane is behind every
    // caster and one before the near plane in front of them all: clamping its depth keeps both
    // comparisons right (reversed-Z: larger is nearer the light).
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
        return out;
    }
    out.edge = min(min(uv.x, uv.y), min(1.0 - uv.x, 1.0 - uv.y));
    let depth = clamp(c.z, 0.0, 1.0);
    // Depth units per metre along the light at the receiver.
    var depth_per_metre = 1.0 / max(params.y - params.x, 1e-6);
    if (perspective) {
        let a = params.x / max(params.y - params.x, 1e-6);
        let view_depth = max(shadow_linear_depth(depth, params.x, params.y), 1e-4);
        depth_per_metre = a * params.y / (view_depth * view_depth);
    }
    // The receiver's plane in the map: the depth change per uv, from two steps along the surface.
    let basis = orthonormal_basis(normal);
    let c1 = shadow_coords(layer, origin + basis[0] * texel_world);
    let c2 = shadow_coords(layer, origin + basis[1] * texel_world);
    let d1 = c1 - c;
    let d2 = c2 - c;
    let det = d1.x * d2.y - d1.y * d2.x;
    var plane = vec2<f32>(0.0);
    if (abs(det) > 1e-14) {
        plane = vec2<f32>(d1.z * d2.y - d2.z * d1.y, d2.z * d1.x - d1.z * d2.x) / det;
    }
    // No steeper than SHADOW_MAX_SLOPE: a surface edge-on to the light has no usable plane.
    let uv_per_metre = 1.0 / max(texel_world * size, 1e-9);
    let steepest = SHADOW_MAX_SLOPE * depth_per_metre / uv_per_metre;
    let steepness = length(plane);
    if (steepness > steepest) {
        plane *= steepest / steepness;
    }
    // Depth allowance per uv squared of tap offset for a receiver curving toward the light.
    let curve = SHADOW_CURVATURE * depth_per_metre / (uv_per_metre * uv_per_metre);
    // The disc turns per pixel and per frame. With the Interactive tier's 8 taps it turns per 4 x 4
    // block of pixels instead, so neighbouring pixels read neighbouring texels of the map (a turn per
    // pixel scatters them over the cache); the price is noise in blocks, which the accumulated
    // frames of a still would keep (a grainy penumbra), so the other tiers do not pay it.
    let cell = select(pixel, floor(pixel * 0.25), taps <= 8u);
    let angle = TAU * interleaved_gradient_noise(cell + 5.588238 * f32(frame.counters.x % 64u));
    let turn = vec2<f32>(cos(angle), sin(angle));
    let min_radius = SHADOW_MIN_RADIUS_TEXELS / size;
    let max_radius = SHADOW_RADIUS_TEXELS_PER_TAP * f32(max(taps, 1u)) / size;
    var radius_uv = 1.5 / size;
    if (light.shadow.y > 0.5) {
        // Penumbra width per metre of gap between blocker and receiver.
        var rate = params.w;
        if (perspective) {
            rate = params.w / receiver_distance;
        }
        // The blocker search covers the penumbra of the largest gap there can be: a blocker at
        // the map's near plane.
        var largest_gap = (1.0 - depth) * (params.y - params.x);
        if (perspective) {
            largest_gap = max(shadow_linear_depth(depth, params.x, params.y) - params.x, 0.0);
        }
        let search_uv = clamp(0.5 * rate * largest_gap * uv_per_metre, 2.0 / size, max_radius);
        var blocker_sum = 0.0;
        var blockers = 0.0;
        let dims = vec2<f32>(size);
        let slack = 0.5 * texel_world * depth_per_metre;
        let search_taps = shadow_search_taps(taps);
        let search_step = search_uv * inverseSqrt(f32(search_taps));
        var d_search = turn;
        // Four probes at a time: their fetches are independent and go out together.
        for (var i = 0u; i < search_taps; i += 4u) {
            let o0 = d_search * (sqrt(f32(i) + 0.5) * search_step);
            d_search = golden_turn(d_search);
            let o1 = d_search * (sqrt(f32(i) + 1.5) * search_step);
            d_search = golden_turn(d_search);
            let o2 = d_search * (sqrt(f32(i) + 2.5) * search_step);
            d_search = golden_turn(d_search);
            let o3 = d_search * (sqrt(f32(i) + 3.5) * search_step);
            d_search = golden_turn(d_search);
            let z = vec4<f32>(
                textureLoad(shadow_atlas, vec2<i32>(clamp((uv + o0) * dims, vec2<f32>(0.0), dims - 1.0)), i32(layer), 0),
                textureLoad(shadow_atlas, vec2<i32>(clamp((uv + o1) * dims, vec2<f32>(0.0), dims - 1.0)), i32(layer), 0),
                textureLoad(shadow_atlas, vec2<i32>(clamp((uv + o2) * dims, vec2<f32>(0.0), dims - 1.0)), i32(layer), 0),
                textureLoad(shadow_atlas, vec2<i32>(clamp((uv + o3) * dims, vec2<f32>(0.0), dims - 1.0)), i32(layer), 0),
            );
            // Reversed-Z: a blocker is nearer the light than the receiver's plane, so larger.
            let reference = vec4<f32>(depth + slack)
                + vec4<f32>(dot(plane, o0), dot(plane, o1), dot(plane, o2), dot(plane, o3))
                + curve * vec4<f32>(dot(o0, o0), dot(o1, o1), dot(o2, o2), dot(o3, o3));
            let found = select(vec4<f32>(0.0), vec4<f32>(1.0), z > reference);
            blocker_sum += dot(z, found);
            blockers += dot(found, vec4<f32>(1.0));
        }
        if (blockers == 0.0) {
            return out;
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
        let wanted = max(0.5 * penumbra_uv, max(min_radius, SHADOW_MIN_PENUMBRA * uv_per_metre));
        radius_uv = min(wanted, max_radius);
        out.spread = wanted / radius_uv;
        // The umbra: every probe of the search found a blocker and the filter lies inside the
        // searched disc, so its taps would find nothing else (the side of a face turned away from
        // the light is most of the pixels a shadow map serves).
        if (blockers == f32(search_taps) && radius_uv <= search_uv) {
            out.visibility = 0.0;
            return out;
        }
    }
    var lit = 0.0;
    let n = shadow_filter_taps(taps);
    let step = radius_uv * inverseSqrt(f32(n));
    var d_filter = turn;
    // Four taps at a time, as in the search.
    for (var i = 0u; i < n; i += 4u) {
        let o0 = d_filter * (sqrt(f32(i) + 0.5) * step);
        d_filter = golden_turn(d_filter);
        let o1 = d_filter * (sqrt(f32(i) + 1.5) * step);
        d_filter = golden_turn(d_filter);
        let o2 = d_filter * (sqrt(f32(i) + 2.5) * step);
        d_filter = golden_turn(d_filter);
        let o3 = d_filter * (sqrt(f32(i) + 3.5) * step);
        d_filter = golden_turn(d_filter);
        lit += textureSampleCompareLevel(shadow_atlas, shadow_sampler, uv + o0, layer, clamp(depth + dot(plane, o0) + curve * dot(o0, o0), 0.0, 1.0))
            + textureSampleCompareLevel(shadow_atlas, shadow_sampler, uv + o1, layer, clamp(depth + dot(plane, o1) + curve * dot(o1, o1), 0.0, 1.0))
            + textureSampleCompareLevel(shadow_atlas, shadow_sampler, uv + o2, layer, clamp(depth + dot(plane, o2) + curve * dot(o2, o2), 0.0, 1.0))
            + textureSampleCompareLevel(shadow_atlas, shadow_sampler, uv + o3, layer, clamp(depth + dot(plane, o3) + curve * dot(o3, o3), 0.0, 1.0));
    }
    out.visibility = lit / f32(n);
    return out;
}

// How much of a light reaches `world` through its shadow map (1 lit, 0 shadowed). `pixel` and the
// frame index seed the per-pixel rotation of the filter, which TAA resolves. `taps` is the quality
// variant's sample count.
fn shadow_visibility(light: Light, world: vec3<f32>, normal: vec3<f32>, pixel: vec2<f32>, taps: u32) -> f32 {
    return shadow_sample(light, world, normal, pixel, taps).visibility;
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
