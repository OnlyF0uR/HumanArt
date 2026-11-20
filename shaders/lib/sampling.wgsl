// lib/sampling.wgsl: low-discrepancy sequences, noise hooks and the warps from [0, 1)^2 to the
// directions the lighting integrals need, with their densities.
//
// Directions are in a local frame with +Z the normal (see orthonormal_basis in lib/math.wgsl).
// Derivations: docs/shader-notes.md, "Sampling".

#include "lib/math.wgsl"

// 2^-24: the top 24 bits of a 0.32 fixed-point fraction convert to f32 exactly, and the result
// stays below 1 (converting all 32 bits would round 0xFFFFFFFF up to 1.0).
const INV_2_POW_24: f32 = 5.9604644775390625e-8;

// The 0.32 fixed-point fraction `bits` as an f32 in [0, 1), truncated to 24 bits (exact).
fn fixed_to_unit(bits: u32) -> f32 {
    return f32(bits >> 8u) * INV_2_POW_24;
}

// The van der Corput radical inverse in base 2 of `bits`: the bits mirrored about the binary point.
fn radical_inverse_vdc(bits: u32) -> f32 {
    return fixed_to_unit(reverseBits(bits));
}

// Point `i` of the `n`-point Hammersley set: (i / n, radical inverse of i).
fn hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(max(n, 1u)), radical_inverse_vdc(i));
}

// Point `i` of Roberts' R2 sequence, fract(0.5 + i * (1/g, 1/g^2)) with g the plastic number,
// computed in 0.32 fixed point so it stays exact for every u32 index.
fn r2_sequence(i: u32) -> vec2<f32> {
    let fixed = vec2<u32>(i * 0xC13FA9A9u + 0x80000000u, i * 0x91E10DA6u + 0x80000000u);
    return vec2<f32>(fixed_to_unit(fixed.x), fixed_to_unit(fixed.y));
}

// Cranley-Patterson rotation: `u` shifted by `offset` on the unit torus.
fn cranley_patterson(u: vec2<f32>, offset: vec2<f32>) -> vec2<f32> {
    return fract(u + offset);
}

// Jimenez's interleaved gradient noise (2014) at a pixel centre: the fallback when no blue-noise
// texture is bound.
fn interleaved_gradient_noise(pixel: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(pixel, vec2<f32>(0.06711056, 0.00583715))));
}

// The blue-noise hook: a texel the caller read from its tiled blue-noise texture, decorrelated over
// time by an R2 rotation per frame, so the per-pixel pattern stays blue in space and in time.
fn blue_noise_sample(texel: vec2<f32>, frame: u32) -> vec2<f32> {
    return cranley_patterson(texel, r2_sequence(frame));
}

// Shirley and Chiu's concentric map from [0, 1)^2 to the unit disk (area-preserving).
fn sample_disk_concentric(u: vec2<f32>) -> vec2<f32> {
    let o = 2.0 * u - vec2<f32>(1.0);
    if (o.x == 0.0 && o.y == 0.0) {
        return vec2<f32>(0.0);
    }
    var r: f32;
    var theta: f32;
    if (abs(o.x) > abs(o.y)) {
        r = o.x;
        theta = 0.25 * PI * (o.y / o.x);
    } else {
        r = o.y;
        theta = HALF_PI - 0.25 * PI * (o.x / o.y);
    }
    return r * vec2<f32>(cos(theta), sin(theta));
}

// A uniformly distributed direction on the unit sphere; density 1 / (4 pi).
fn sample_uniform_sphere(u: vec2<f32>) -> vec3<f32> {
    let z = 1.0 - 2.0 * u.x;
    let r = sqrt(max(0.0, 1.0 - z * z));
    let phi = TAU * u.y;
    return vec3<f32>(r * cos(phi), r * sin(phi), z);
}

// A uniformly distributed direction on the +Z hemisphere; density 1 / (2 pi).
fn sample_uniform_hemisphere(u: vec2<f32>) -> vec3<f32> {
    let z = u.x;
    let r = sqrt(max(0.0, 1.0 - z * z));
    let phi = TAU * u.y;
    return vec3<f32>(r * cos(phi), r * sin(phi), z);
}

// A cosine-distributed direction on the +Z hemisphere (Malley's method); density cos(theta) / pi.
fn sample_cosine_hemisphere(u: vec2<f32>) -> vec3<f32> {
    let d = sample_disk_concentric(u);
    return vec3<f32>(d, sqrt(max(0.0, 1.0 - dot(d, d))));
}

// The density of sample_cosine_hemisphere for a direction with cosine `cos_theta` to the normal.
fn pdf_cosine_hemisphere(cos_theta: f32) -> f32 {
    return max(cos_theta, 0.0) * INV_PI;
}

// A half vector distributed like the GGX normal distribution projected on the normal (Walter et al.
// 2007): density D(h) * cos(theta_h). `alpha` is the GGX width (perceptual roughness squared).
fn sample_ggx_ndf(u: vec2<f32>, alpha: f32) -> vec3<f32> {
    let a2 = alpha * alpha;
    let cos2 = (1.0 - u.x) / (1.0 + (a2 - 1.0) * u.x);
    let cos_theta = sqrt(max(cos2, 0.0));
    let sin_theta = sqrt(max(1.0 - cos2, 0.0));
    let phi = TAU * u.y;
    return vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

// A half vector distributed like the GGX normals visible from `v` (unit, local, v.z > 0), Heitz,
// "Sampling the GGX Distribution of Visible Normals" (JCGT 2018): density
// G1(v) * max(v.h, 0) * D(h) / v.z.
fn sample_ggx_vndf(u: vec2<f32>, v: vec3<f32>, alpha: f32) -> vec3<f32> {
    let vh = normalize(vec3<f32>(alpha * v.x, alpha * v.y, v.z));
    let lensq = vh.x * vh.x + vh.y * vh.y;
    var t1 = vec3<f32>(1.0, 0.0, 0.0);
    if (lensq > 0.0) {
        t1 = vec3<f32>(-vh.y, vh.x, 0.0) * inverseSqrt(lensq);
    }
    let t2 = cross(vh, t1);
    let r = sqrt(u.x);
    let phi = TAU * u.y;
    let p1 = r * cos(phi);
    let s = 0.5 * (1.0 + vh.z);
    let p2 = (1.0 - s) * sqrt(max(0.0, 1.0 - p1 * p1)) + s * r * sin(phi);
    let nh = p1 * t1 + p2 * t2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * vh;
    return normalize(vec3<f32>(alpha * nh.x, alpha * nh.y, max(0.0, nh.z)));
}
