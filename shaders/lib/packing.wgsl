// lib/packing.wgsl: normalised integers, 10-10-10-2 words, half pairs, octahedral unit vectors and
// packed tangent frames [EVAL-05]. Every encoder follows the rules of its CPU twin in
// humanart_math::pack (same rounding, same "precise" octahedral search), so data packed on the CPU
// and in a compute pass are interchangeable. The GPU tests (crates/shaders/tests/library.rs) find
// the 10-10-10-2 and half encoders bit-exact; the octahedral search may pick the neighbouring code
// when two candidates are equally close within f32 rounding (GPU length and division are not
// correctly rounded), never a worse one.
//
// The WGSL built-ins pack4x8snorm, pack4x8unorm, pack2x16snorm, pack2x16unorm and pack2x16float
// (and their unpack twins) are used directly. The snorm/unorm built-ins are the driver's: a value
// at (or within f32 rounding of) a half-way point may round one code the other way than on the CPU.
// Derivations: docs/shader-notes.md, "Packing".

#include "lib/math.wgsl"

// True for NaN, by the bits (immune to fast-math folding of `v != v`).
fn is_nan_f32(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u;
}

// floor(0.5 + scale * clamp(v, -1, 1)); NaN gives 0.
fn quantize_snorm(v: f32, scale: f32) -> i32 {
    if (is_nan_f32(v)) {
        return 0;
    }
    return i32(floor(0.5 + scale * clamp(v, -1.0, 1.0)));
}

// floor(0.5 + scale * clamp(v, 0, 1)); NaN gives 0.
fn quantize_unorm(v: f32, scale: f32) -> u32 {
    if (is_nan_f32(v)) {
        return 0u;
    }
    return u32(floor(0.5 + scale * clamp(v, 0.0, 1.0)));
}

// max(i / scale, -1).
fn dequantize_snorm(i: i32, scale: f32) -> f32 {
    return max(f32(i) / scale, -1.0);
}

// x, y, z unorm with 1023 levels in bits 0..10, 10..20, 20..30 and w unorm with 3 levels in bits
// 30..32: the Rgb10a2Unorm texel layout.
fn pack_unorm_10_10_10_2(v: vec4<f32>) -> u32 {
    return quantize_unorm(v.x, 1023.0) | (quantize_unorm(v.y, 1023.0) << 10u)
        | (quantize_unorm(v.z, 1023.0) << 20u) | (quantize_unorm(v.w, 3.0) << 30u);
}

// The inverse of pack_unorm_10_10_10_2.
fn unpack_unorm_10_10_10_2(p: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(p & 0x3ffu) / 1023.0,
        f32((p >> 10u) & 0x3ffu) / 1023.0,
        f32((p >> 20u) & 0x3ffu) / 1023.0,
        f32(p >> 30u) / 3.0,
    );
}

// x, y, z snorm with 511 levels each side (two's complement in bits 0..10, 10..20, 20..30) and w
// snorm with one level each side in bits 30..32.
fn pack_snorm_10_10_10_2(v: vec4<f32>) -> u32 {
    let x = bitcast<u32>(quantize_snorm(v.x, 511.0)) & 0x3ffu;
    let y = bitcast<u32>(quantize_snorm(v.y, 511.0)) & 0x3ffu;
    let z = bitcast<u32>(quantize_snorm(v.z, 511.0)) & 0x3ffu;
    let w = bitcast<u32>(quantize_snorm(v.w, 1.0)) & 0x3u;
    return x | (y << 10u) | (z << 20u) | (w << 30u);
}

// The inverse of pack_snorm_10_10_10_2 (each field sign-extended; the minimums decode to -1).
fn unpack_snorm_10_10_10_2(p: u32) -> vec4<f32> {
    let s = bitcast<i32>(p);
    return vec4<f32>(
        dequantize_snorm(extractBits(s, 0u, 10u), 511.0),
        dequantize_snorm(extractBits(s, 10u, 10u), 511.0),
        dequantize_snorm(extractBits(s, 20u, 10u), 511.0),
        dequantize_snorm(extractBits(s, 30u, 2u), 1.0),
    );
}

// Four halves in two words, (x, y) then (z, w): the storage layout of a vec4<f16>.
fn pack4x16float(v: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(v.xy), pack2x16float(v.zw));
}

// The inverse of pack4x16float.
fn unpack4x16float(p: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(p.x), unpack2x16float(p.y));
}

// -1 for a negative sign bit (including -0), +1 otherwise: the octahedral fold never sees zero.
fn sign_not_zero(v: f32) -> f32 {
    return select(1.0, -1.0, (bitcast<u32>(v) & 0x80000000u) != 0u);
}

// The octahedral coordinates in [-1, 1]^2 of the direction `n` (any length; Cigolle et al. 2014).
// A zero or non-finite vector maps to (0, 0), which decodes to +Z.
fn oct_encode(n: vec3<f32>) -> vec2<f32> {
    let l1 = abs(n.x) + abs(n.y) + abs(n.z);
    if (!(l1 > 0.0 && l1 <= F32_MAX)) {
        return vec2<f32>(0.0);
    }
    let x = n.x / l1;
    let y = n.y / l1;
    if (n.z < 0.0) {
        return vec2<f32>((1.0 - abs(y)) * sign_not_zero(x), (1.0 - abs(x)) * sign_not_zero(y));
    }
    return vec2<f32>(x, y);
}

// The unit direction of the octahedral coordinates `e` (clamped to [-1, 1]^2).
fn oct_decode(e: vec2<f32>) -> vec3<f32> {
    let x = clamp(e.x, -1.0, 1.0);
    let y = clamp(e.y, -1.0, 1.0);
    let z = 1.0 - abs(x) - abs(y);
    var v = vec3<f32>(x, y, z);
    if (z < 0.0) {
        v = vec3<f32>((1.0 - abs(y)) * sign_not_zero(x), (1.0 - abs(x)) * sign_not_zero(y), z);
    }
    return v / length(v);
}

// The snorm pair (with `scale` levels each side per axis) whose octahedral decode is closest to
// `n`: the best of the four grid points around the exact encoding, compared by chord length.
fn oct_quantize(n: vec3<f32>, scale: vec2<f32>) -> vec2<i32> {
    let len = length(n);
    if (!(len > 0.0 && len <= F32_MAX)) {
        return vec2<i32>(0);
    }
    let unit = n / len;
    let f = floor(oct_encode(unit) * scale);
    var best = vec2<f32>(0.0);
    var best_chord = F32_MAX;
    for (var k = 0u; k < 4u; k++) {
        let q = clamp(f + vec2<f32>(f32(k & 1u), f32(k >> 1u)), -scale, scale);
        let d = oct_decode(q / scale) - unit;
        let chord = dot(d, d);
        if (chord < best_chord) {
            best_chord = chord;
            best = q;
        }
    }
    return vec2<i32>(best);
}

// A unit vector in 32 bits: octahedral coordinates as pack2x16snorm.
fn pack_oct16(n: vec3<f32>) -> u32 {
    let q = bitcast<vec2<u32>>(oct_quantize(n, vec2<f32>(32767.0)));
    return (q.x & 0xffffu) | (q.y << 16u);
}

// The unit vector of pack_oct16.
fn unpack_oct16(p: u32) -> vec3<f32> {
    return oct_decode(unpack2x16snorm(p));
}

// A unit vector in 16 bits (the low half of the result): octahedral coordinates as two snorm
// bytes, x in the low byte.
fn pack_oct8(n: vec3<f32>) -> u32 {
    let q = bitcast<vec2<u32>>(oct_quantize(n, vec2<f32>(127.0)));
    return (q.x & 0xffu) | ((q.y & 0xffu) << 8u);
}

// The unit vector of pack_oct8 (the low 16 bits of `p`).
fn unpack_oct8(p: u32) -> vec3<f32> {
    return oct_decode(unpack4x8snorm(p & 0xffffu).xy);
}

// A tangent and the bitangent sign in 32 bits: octahedral x as snorm16 in bits 0..16, octahedral
// y as a 15-bit snorm (16383 levels each side) in bits 16..31, bit 31 set when the sign is negative.
fn pack_tangent(tangent: vec3<f32>, bitangent_sign: f32) -> u32 {
    let q = bitcast<vec2<u32>>(oct_quantize(tangent, vec2<f32>(32767.0, 16383.0)));
    let sign_bit = select(0u, 0x80000000u, bitangent_sign < 0.0);
    return (q.x & 0xffffu) | ((q.y & 0x7fffu) << 16u) | sign_bit;
}

// The unit tangent (xyz) and the bitangent sign (w, 1 or -1) of pack_tangent.
fn unpack_tangent(p: u32) -> vec4<f32> {
    let x = unpack2x16snorm(p).x;
    let y = dequantize_snorm(extractBits(bitcast<i32>(p), 16u, 15u), 16383.0);
    return vec4<f32>(oct_decode(vec2<f32>(x, y)), select(1.0, -1.0, (p >> 31u) == 1u));
}
