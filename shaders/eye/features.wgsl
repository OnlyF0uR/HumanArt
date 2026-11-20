// eye/features.wgsl: the eye and mouth material table of the geometry pass [EYE-01]..[EYE-07]
// [MTH-01]..[MTH-04], and the lighting of their dielectric surfaces. Mirrors `FeatureTable` in
// crates/render/src/eye/mod.rs.
//
// The includer declares the binding (group 2, binding 7: `features`) and includes lib/math.wgsl,
// lib/brdf.wgsl, render/frame.wgsl and render/lighting.wgsl (feature_lit uses the lights, the
// shadow maps and the image-based light). `slots` maps an object
// index to its entry (0 none, else entry + 1); an entry is 16 vec4s whose meaning depends on its
// kind (entry vec4 15, w): FEATURE_EYE (crates/render/src/eye/mod.rs, `EyeGpu`) or FEATURE_MOUTH
// (crates/render/src/mouth/mod.rs, `MouthGpu`).

const FEATURE_VEC4S: u32 = 16u;
const MAX_FEATURE_ENTRIES: u32 = 32u;
const MAX_FEATURE_OBJECTS: u32 = 4096u;

const FEATURE_EYE: u32 = 1u;
const FEATURE_MOUTH: u32 = 2u;

// Material classes of the eye and mouth parts (`Material::class`; 1 is skin, 3 hair).
// crates/render/src/eye/mod.rs holds the same numbers.
const EYE_CLASS: u32 = 2u;
const TEETH_CLASS: u32 = 4u;
const TEAR_CLASS: u32 = 5u;
const GUM_CLASS: u32 = 6u;
const TONGUE_CLASS: u32 = 7u;
const EYE_OCCLUSION_CLASS: u32 = 8u;
const MOUTH_BAG_CLASS: u32 = 9u;

struct FeatureTable {
    slots: array<u32, 4096>,
    entries: array<vec4<f32>, 512>,
}

// The entry of object `object` (0 none, else entry + 1).
fn feature_slot(object: u32) -> u32 {
    if (object >= MAX_FEATURE_OBJECTS) {
        return 0u;
    }
    let s = features.slots[object];
    return select(0u, s, s <= MAX_FEATURE_ENTRIES);
}

// Vec4 `k` of entry `slot` (1-based, as feature_slot returns it).
fn feature_vec(slot: u32, k: u32) -> vec4<f32> {
    return features.entries[(slot - 1u) * FEATURE_VEC4S + min(k, FEATURE_VEC4S - 1u)];
}

// Float `i` (0..63) of entry `slot`: the flat view the Rust packers use.
fn feature_float(slot: u32, i: u32) -> f32 {
    let v = feature_vec(slot, i / 4u);
    return v[i % 4u];
}

// The kind of entry `slot` (0 when there is none).
fn feature_kind(slot: u32) -> u32 {
    if (slot == 0u) {
        return 0u;
    }
    return u32(feature_vec(slot, 15u).w + 0.5);
}

// Result of an eye or mouth shading function, in the geometry pass's terms: albedo, diffuse
// lighting for unit albedo, specular radiance, the shading normal and roughness for the G-buffer.
struct FeatureShade {
    albedo: vec3<f32>,
    diffuse: vec3<f32>,
    specular: vec3<f32>,
    normal: vec3<f32>,
    roughness: f32,
}

// Integer hash (PCG-style) to [0, 1).
fn feature_hash(p: vec3<i32>) -> f32 {
    var h = u32(p.x) * 747796405u + u32(p.y) * 2891336453u + u32(p.z) * 277803737u;
    h = h ^ (h >> 16u);
    h = h * 2246822519u;
    h = h ^ (h >> 13u);
    h = h * 3266489917u;
    h = h ^ (h >> 16u);
    return f32(h) * (1.0 / 4294967296.0);
}

// Smooth value noise in [0, 1) on the integer lattice.
fn feature_noise(p: vec3<f32>) -> f32 {
    let i = vec3<i32>(floor(p));
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = feature_hash(i);
    let n100 = feature_hash(i + vec3<i32>(1, 0, 0));
    let n010 = feature_hash(i + vec3<i32>(0, 1, 0));
    let n110 = feature_hash(i + vec3<i32>(1, 1, 0));
    let n001 = feature_hash(i + vec3<i32>(0, 0, 1));
    let n101 = feature_hash(i + vec3<i32>(1, 0, 1));
    let n011 = feature_hash(i + vec3<i32>(0, 1, 1));
    let n111 = feature_hash(i + vec3<i32>(1, 1, 1));
    let x0 = mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y);
    let x1 = mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y);
    return mix(x0, x1, u.z);
}

// Three octaves of value noise, in [0, 1).
fn feature_fbm(p: vec3<f32>) -> f32 {
    return feature_noise(p) * 0.5714 + feature_noise(p * 2.03 + 17.0) * 0.2857 + feature_noise(p * 4.01 + 31.0) * 0.1429;
}

// Every light, the shadows and the image-based light at a dielectric point: diffuse for unit albedo
// with wrapped lighting (`wrap` 0 is Lambert; (n.l + w) / (1 + w)^2 keeps the energy) for the
// subsurface-soft tissues, and single-scatter GGX with multi-scatter compensation. `direct` and
// `ambient` scale the lights and the environment (occlusion); `shadow_n` is the normal the shadow
// lookup offsets along. The diffuse is weighted by what the specular does not reflect.
fn feature_lit(world: vec3<f32>, n: vec3<f32>, shadow_n: vec3<f32>, v: vec3<f32>, roughness: f32, f0: f32, wrap: f32, direct: f32, ambient: f32, pixel: vec2<f32>, taps: u32) -> FeatureShade {
    var out: FeatureShade;
    let r = clamp(roughness, 0.02, 1.0);
    let alpha = alpha_from_roughness(r);
    let f = vec3<f32>(f0);
    let n_dot_v = max(dot(n, v), 1e-4);
    let dfg = dfg_lookup(n_dot_v, r);
    let compensation = specular_energy_compensation(f, dfg);
    let count = min(u32(frame.environment.w), MAX_LIGHTS);
    var diffuse = vec3<f32>(0.0);
    var specular = vec3<f32>(0.0);
    let w = max(wrap, 0.0);
    for (var i = 0u; i < count; i++) {
        let light = frame.lights[i];
        let visibility = shadow_visibility(light, world, shadow_n, pixel, taps) * direct;
        let s = shade_light(light, world, n, v, alpha, f, visibility);
        specular += s.specular * compensation;
        if (u32(light.color_kind.w) == LIGHT_RECT || w <= 0.0) {
            diffuse += s.diffuse;
        } else {
            let incident = incident_light(light, world);
            let wrapped = max(dot(n, incident.l) + w, 0.0) / sq(1.0 + w);
            diffuse += incident.irradiance * (visibility * wrapped * INV_PI);
        }
    }
    let env = frame.environment.x;
    let reflectance = f0 * dfg.x + dfg.y;
    diffuse += sh_irradiance(n) * (env * INV_PI * ambient);
    specular += env_prefiltered(reflect(-v, n), r) * (env * ambient * reflectance) * compensation;
    out.albedo = vec3<f32>(1.0);
    out.diffuse = diffuse * saturate(1.0 - reflectance);
    out.specular = specular;
    out.normal = n;
    out.roughness = r;
    return out;
}
