//! Vertex and texel packings: normalised integers, the WGSL `pack*`/`unpack*` built-ins,
//! 10-10-10-2, octahedral unit vectors and packed tangent frames (`[EVAL-05]`).
//!
//! Every CPU encoder here produces exactly the bits the matching WGSL built-in produces, and every
//! decoder returns exactly what the shader decodes, so data packed on the CPU and data packed by a
//! compute pass are interchangeable. The normalised-integer rules are the WGSL ones:
//!
//! - snorm with `n` levels each side: `floor(0.5 + n * clamp(v, -1, 1))`, decoded as
//!   `max(i / n, -1)` (both `-n` and `-n - 1` decode to `-1`);
//! - unorm with `n` levels: `floor(0.5 + n * clamp(v, 0, 1))`, decoded as `i / n`.
//!
//! NaN inputs encode as zero. Component 0 always lives in the least significant bits.
//!
//! Octahedral encoding (Cigolle et al., "A Survey of Efficient Representations for Independent
//! Unit Vectors", JCGT 2014) maps the unit sphere onto the square `[-1, 1]²`: the upper hemisphere
//! (`z >= 0`) to the inner diamond, the lower one folded over the corners. The encoders here search
//! the four neighbouring grid points for the one that decodes closest to the input ("precise"
//! variant), which is never worse than rounding each coordinate and lowers the worst case (measured
//! in the tests).

use super::half::{f16_bits_to_f32, f32_to_f16_bits};
use super::vec::{Vec2, Vec3, Vec4};

/// `floor(0.5 + scale * clamp(v, -1, 1))`; NaN gives zero.
#[inline]
fn snorm(v: f32, scale: f32) -> i32 {
    if v.is_nan() { 0 } else { (0.5 + scale * v.clamp(-1.0, 1.0)).floor() as i32 }
}

/// `floor(0.5 + scale * clamp(v, 0, 1))`; NaN gives zero.
#[inline]
fn unorm(v: f32, scale: f32) -> u32 {
    if v.is_nan() { 0 } else { (0.5 + scale * v.clamp(0.0, 1.0)).floor() as u32 }
}

/// `max(i / scale, -1)`.
#[inline]
fn from_snorm(i: i32, scale: f32) -> f32 {
    (i as f32 / scale).max(-1.0)
}

/// Sign-extends the low `bits` bits of `v`.
#[inline]
const fn sign_extend(v: u32, bits: u32) -> i32 {
    ((v << (32 - bits)) as i32) >> (32 - bits)
}

/// `v` in `[-1, 1]` as a signed byte, `[-127, 127]`.
#[inline]
pub fn quantize_snorm8(v: f32) -> i8 {
    snorm(v, 127.0) as i8
}

/// The value of a signed-normalised byte; `-128` and `-127` both give `-1`.
#[inline]
pub fn dequantize_snorm8(i: i8) -> f32 {
    from_snorm(i32::from(i), 127.0)
}

/// `v` in `[0, 1]` as an unsigned byte, `[0, 255]`.
#[inline]
pub fn quantize_unorm8(v: f32) -> u8 {
    unorm(v, 255.0) as u8
}

/// The value of an unsigned-normalised byte.
#[inline]
pub fn dequantize_unorm8(u: u8) -> f32 {
    f32::from(u) / 255.0
}

/// `v` in `[-1, 1]` as a signed 16-bit integer, `[-32 767, 32 767]`.
#[inline]
pub fn quantize_snorm16(v: f32) -> i16 {
    snorm(v, 32767.0) as i16
}

/// The value of a signed-normalised 16-bit integer; `-32 768` and `-32 767` both give `-1`.
#[inline]
pub fn dequantize_snorm16(i: i16) -> f32 {
    from_snorm(i32::from(i), 32767.0)
}

/// `v` in `[0, 1]` as an unsigned 16-bit integer, `[0, 65 535]`.
#[inline]
pub fn quantize_unorm16(v: f32) -> u16 {
    unorm(v, 65535.0) as u16
}

/// The value of an unsigned-normalised 16-bit integer.
#[inline]
pub fn dequantize_unorm16(u: u16) -> f32 {
    f32::from(u) / 65535.0
}

/// WGSL `pack4x8snorm`: four snorm bytes, `x` in bits 0..8.
#[inline]
pub fn pack4x8snorm(v: Vec4) -> u32 {
    u32::from_le_bytes(
        [quantize_snorm8(v.x), quantize_snorm8(v.y), quantize_snorm8(v.z), quantize_snorm8(v.w)].map(|b| b as u8),
    )
}

/// WGSL `unpack4x8snorm`.
#[inline]
pub fn unpack4x8snorm(p: u32) -> Vec4 {
    let [x, y, z, w] = p.to_le_bytes().map(|b| dequantize_snorm8(b as i8));
    Vec4::new(x, y, z, w)
}

/// WGSL `pack4x8unorm`: four unorm bytes, `x` in bits 0..8.
#[inline]
pub fn pack4x8unorm(v: Vec4) -> u32 {
    u32::from_le_bytes([quantize_unorm8(v.x), quantize_unorm8(v.y), quantize_unorm8(v.z), quantize_unorm8(v.w)])
}

/// WGSL `unpack4x8unorm`.
#[inline]
pub fn unpack4x8unorm(p: u32) -> Vec4 {
    let [x, y, z, w] = p.to_le_bytes().map(dequantize_unorm8);
    Vec4::new(x, y, z, w)
}

/// WGSL `pack2x16snorm`: two snorm 16-bit integers, `x` in bits 0..16.
#[inline]
pub fn pack2x16snorm(v: Vec2) -> u32 {
    u32::from(quantize_snorm16(v.x) as u16) | (u32::from(quantize_snorm16(v.y) as u16) << 16)
}

/// WGSL `unpack2x16snorm`.
#[inline]
pub fn unpack2x16snorm(p: u32) -> Vec2 {
    Vec2::new(dequantize_snorm16(p as u16 as i16), dequantize_snorm16((p >> 16) as u16 as i16))
}

/// WGSL `pack2x16unorm`: two unorm 16-bit integers, `x` in bits 0..16.
#[inline]
pub fn pack2x16unorm(v: Vec2) -> u32 {
    u32::from(quantize_unorm16(v.x)) | (u32::from(quantize_unorm16(v.y)) << 16)
}

/// WGSL `unpack2x16unorm`.
#[inline]
pub fn unpack2x16unorm(p: u32) -> Vec2 {
    Vec2::new(dequantize_unorm16(p as u16), dequantize_unorm16((p >> 16) as u16))
}

/// WGSL `pack2x16float`: two halves (round to nearest even), `x` in bits 0..16. The layout of a
/// `vec2<f16>` in a storage buffer.
#[inline]
pub fn pack2x16float(v: Vec2) -> u32 {
    u32::from(f32_to_f16_bits(v.x)) | (u32::from(f32_to_f16_bits(v.y)) << 16)
}

/// WGSL `unpack2x16float` (exact).
#[inline]
pub fn unpack2x16float(p: u32) -> Vec2 {
    Vec2::new(f16_bits_to_f32(p as u16), f16_bits_to_f32((p >> 16) as u16))
}

/// Four halves in two words: `(x, y)` then `(z, w)`, the layout of a `vec4<f16>` in a storage
/// buffer.
#[inline]
pub fn pack4x16float(v: Vec4) -> [u32; 2] {
    [pack2x16float(Vec2::new(v.x, v.y)), pack2x16float(Vec2::new(v.z, v.w))]
}

/// The inverse of [`pack4x16float`] (exact).
#[inline]
pub fn unpack4x16float(p: [u32; 2]) -> Vec4 {
    let (a, b) = (unpack2x16float(p[0]), unpack2x16float(p[1]));
    Vec4::new(a.x, a.y, b.x, b.y)
}

/// The wgpu `Rgb10a2Unorm` texel / `Unorm10_10_10_2` vertex layout: `x`, `y`, `z` unorm with
/// 1023 levels in bits 0..10, 10..20, 20..30 and `w` unorm with 3 levels in bits 30..32.
#[inline]
pub fn pack_unorm_10_10_10_2(v: Vec4) -> u32 {
    unorm(v.x, 1023.0) | (unorm(v.y, 1023.0) << 10) | (unorm(v.z, 1023.0) << 20) | (unorm(v.w, 3.0) << 30)
}

/// The inverse of [`pack_unorm_10_10_10_2`].
#[inline]
pub fn unpack_unorm_10_10_10_2(p: u32) -> Vec4 {
    Vec4::new(
        (p & 0x3ff) as f32 / 1023.0,
        ((p >> 10) & 0x3ff) as f32 / 1023.0,
        ((p >> 20) & 0x3ff) as f32 / 1023.0,
        (p >> 30) as f32 / 3.0,
    )
}

/// `x`, `y`, `z` snorm with 511 levels each side (two's complement in bits 0..10, 10..20, 20..30)
/// and `w` snorm with one level each side (`-1`, `0`, `1` in bits 30..32): the classic packed
/// normal + bitangent-sign layout. Decoded in a shader by sign-extending each field.
#[inline]
pub fn pack_snorm_10_10_10_2(v: Vec4) -> u32 {
    let f = |c: f32, scale: f32, mask: u32| (snorm(c, scale) as u32) & mask;
    f(v.x, 511.0, 0x3ff) | (f(v.y, 511.0, 0x3ff) << 10) | (f(v.z, 511.0, 0x3ff) << 20) | (f(v.w, 1.0, 0x3) << 30)
}

/// The inverse of [`pack_snorm_10_10_10_2`]; the field minimums (`-512`, `-2`) decode to `-1`.
#[inline]
pub fn unpack_snorm_10_10_10_2(p: u32) -> Vec4 {
    Vec4::new(
        from_snorm(sign_extend(p & 0x3ff, 10), 511.0),
        from_snorm(sign_extend((p >> 10) & 0x3ff, 10), 511.0),
        from_snorm(sign_extend((p >> 20) & 0x3ff, 10), 511.0),
        from_snorm(sign_extend(p >> 30, 2), 1.0),
    )
}

/// `-1` for negative numbers (and `-0`), `+1` otherwise: the octahedral fold must never see zero.
#[inline]
fn sign_not_zero(v: f32) -> f32 {
    if v.is_sign_negative() { -1.0 } else { 1.0 }
}

/// The octahedral coordinates in `[-1, 1]²` of the direction `n` (any length). A zero or non-finite
/// vector maps to `(0, 0)`, which decodes to +Z.
#[inline]
pub fn oct_encode(n: Vec3) -> Vec2 {
    let l1 = n.x.abs() + n.y.abs() + n.z.abs();
    if !(l1 > 0.0 && l1.is_finite()) {
        return Vec2::ZERO;
    }
    let (x, y) = (n.x / l1, n.y / l1);
    if n.z < 0.0 {
        Vec2::new((1.0 - y.abs()) * sign_not_zero(x), (1.0 - x.abs()) * sign_not_zero(y))
    } else {
        Vec2::new(x, y)
    }
}

/// The unit direction of the octahedral coordinates `e` (clamped to `[-1, 1]²`).
#[inline]
pub fn oct_decode(e: Vec2) -> Vec3 {
    let (x, y) = (e.x.clamp(-1.0, 1.0), e.y.clamp(-1.0, 1.0));
    let z = 1.0 - x.abs() - y.abs();
    let v = if z < 0.0 {
        Vec3::new((1.0 - y.abs()) * sign_not_zero(x), (1.0 - x.abs()) * sign_not_zero(y), z)
    } else {
        Vec3::new(x, y, z)
    };
    // |v|₁ = 1, so |v|₂ >= 1/√3: the division is always safe.
    v / v.length()
}

/// The pair of snorm integers (with `scale_x` and `scale_y` levels each side) whose octahedral
/// decode is closest to `n`: the best of the four grid points around the exact encoding.
fn oct_quantize(n: Vec3, scale_x: f32, scale_y: f32) -> (i32, i32) {
    let Some(unit) = n.try_normalize() else {
        return (0, 0);
    };
    let e = oct_encode(unit);
    let (fx, fy) = ((e.x * scale_x).floor(), (e.y * scale_y).floor());
    let mut best = (0, 0);
    // Compared by chord length, which (unlike the dot product, all but 1 in f32 at these angles)
    // resolves the tiny differences between neighbouring candidates.
    let mut best_chord = f32::INFINITY;
    for (dx, dy) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
        let (qx, qy) = ((fx + dx).clamp(-scale_x, scale_x), (fy + dy).clamp(-scale_y, scale_y));
        let chord = oct_decode(Vec2::new(qx / scale_x, qy / scale_y)).distance_squared(unit);
        if chord < best_chord {
            best_chord = chord;
            best = (qx as i32, qy as i32);
        }
    }
    best
}

/// A unit vector in 32 bits: octahedral coordinates as WGSL `pack2x16snorm`, decoded in a shader
/// by `oct_decode(unpack2x16snorm(p))`. Worst-case angular error 0.0025° (measured over 200 000
/// directions in the tests). Zero or non-finite input encodes +Z.
#[inline]
pub fn pack_oct16(n: Vec3) -> u32 {
    let (x, y) = oct_quantize(n, 32767.0, 32767.0);
    u32::from(x as i16 as u16) | (u32::from(y as i16 as u16) << 16)
}

/// The unit vector of [`pack_oct16`].
#[inline]
pub fn unpack_oct16(p: u32) -> Vec3 {
    oct_decode(unpack2x16snorm(p))
}

/// A unit vector in 16 bits: octahedral coordinates as two snorm bytes (`x` in the low byte),
/// decoded in a shader by `oct_decode(unpack4x8snorm(p).xy)`. Worst-case angular error about
/// 0.63° (measured in the tests). Zero or non-finite input encodes +Z.
#[inline]
pub fn pack_oct8(n: Vec3) -> u16 {
    let (x, y) = oct_quantize(n, 127.0, 127.0);
    u16::from_le_bytes([x as i8 as u8, y as i8 as u8])
}

/// The unit vector of [`pack_oct8`].
#[inline]
pub fn unpack_oct8(p: u16) -> Vec3 {
    let [x, y] = p.to_le_bytes();
    oct_decode(Vec2::new(dequantize_snorm8(x as i8), dequantize_snorm8(y as i8)))
}

/// A tangent direction and the bitangent sign in 32 bits: octahedral `x` as snorm16 in bits 0..16,
/// octahedral `y` as a 15-bit snorm (16 383 levels each side) in bits 16..31, and bit 31 set when
/// `bitangent_sign` is negative (the bitangent is `sign * cross(normal, tangent)`). Worst-case
/// angular error 0.0043° (measured in the tests).
#[inline]
pub fn pack_tangent(tangent: Vec3, bitangent_sign: f32) -> u32 {
    let (x, y) = oct_quantize(tangent, 32767.0, 16383.0);
    let sign = if bitangent_sign < 0.0 { 1 << 31 } else { 0 };
    u32::from(x as i16 as u16) | (((y as u32) & 0x7fff) << 16) | sign
}

/// The unit tangent and the bitangent sign (`1` or `-1`) of [`pack_tangent`].
#[inline]
pub fn unpack_tangent(p: u32) -> (Vec3, f32) {
    let x = dequantize_snorm16(p as u16 as i16);
    let y = from_snorm(sign_extend((p >> 16) & 0x7fff, 15), 16383.0);
    (oct_decode(Vec2::new(x, y)), if p >> 31 == 1 { -1.0 } else { 1.0 })
}

/// Packs a normal channel with [`pack_oct16`]. Converts `min(src.len(), dst.len())` vectors and
/// returns that count; the rest of `dst` is untouched.
pub fn pack_oct16_slice(src: &[Vec3], dst: &mut [u32]) -> usize {
    let n = src.len().min(dst.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = pack_oct16(s);
    }
    n
}

/// Packs a normal channel stored as three structure-of-arrays component slices with
/// [`pack_oct16`]. Converts as many vectors as the shortest of the four slices holds and returns
/// that count; the rest of `dst` is untouched.
pub fn pack_oct16_soa(x: &[f32], y: &[f32], z: &[f32], dst: &mut [u32]) -> usize {
    let n = x.len().min(y.len()).min(z.len()).min(dst.len());
    for (((d, &x), &y), &z) in dst.iter_mut().zip(x).zip(y).zip(z) {
        *d = pack_oct16(Vec3::new(x, y, z));
    }
    n
}

/// Unpacks a channel packed with [`pack_oct16`]. Converts `min(src.len(), dst.len())` vectors and
/// returns that count; the rest of `dst` is untouched.
pub fn unpack_oct16_slice(src: &[u32], dst: &mut [Vec3]) -> usize {
    let n = src.len().min(dst.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = unpack_oct16(s);
    }
    n
}

/// Packs a tangent channel and its bitangent signs with [`pack_tangent`]. Converts as many as the
/// shortest of the three slices holds and returns that count; the rest of `dst` is untouched.
pub fn pack_tangent_slice(tangents: &[Vec3], signs: &[f32], dst: &mut [u32]) -> usize {
    let n = tangents.len().min(signs.len()).min(dst.len());
    for ((d, &t), &s) in dst.iter_mut().zip(tangents).zip(signs) {
        *d = pack_tangent(t, s);
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    fn rand_unit(rng: &mut Rng) -> Vec3 {
        loop {
            let v = Vec3::new(rng.normal_f32(), rng.normal_f32(), rng.normal_f32());
            if let Some(u) = v.try_normalize() {
                return u;
            }
        }
    }

    /// The angle between two unit vectors, accurate near zero.
    fn angle(a: Vec3, b: Vec3) -> f32 {
        a.cross(b).length().atan2(a.dot(b))
    }

    #[test]
    fn normalised_integers_follow_the_wgsl_rules() {
        // Every integer survives decode then encode, exhaustively.
        for i in i8::MIN..=i8::MAX {
            let back = quantize_snorm8(dequantize_snorm8(i));
            assert_eq!(back, i.max(-127), "{i}");
        }
        for u in 0..=u8::MAX {
            assert_eq!(quantize_unorm8(dequantize_unorm8(u)), u);
        }
        for i in i16::MIN..=i16::MAX {
            assert_eq!(quantize_snorm16(dequantize_snorm16(i)), i.max(-32767), "{i}");
        }
        for u in 0..=u16::MAX {
            assert_eq!(quantize_unorm16(dequantize_unorm16(u)), u);
        }
        // floor(0.5 + x), not round-half-away: -0.5 levels go up.
        assert_eq!(quantize_snorm8(-0.5 / 127.0), 0);
        assert_eq!(quantize_snorm8(0.5 / 127.0), 1);
        assert_eq!(quantize_unorm8(0.5), 128);
        // Clamping and NaN.
        assert_eq!((quantize_snorm8(-7.0), quantize_snorm8(7.0), quantize_snorm8(f32::NAN)), (-127, 127, 0));
        assert_eq!((quantize_unorm16(-1.0), quantize_unorm16(2.0), quantize_unorm16(f32::NAN)), (0, 65535, 0));
        assert_eq!((dequantize_snorm8(-128), dequantize_snorm16(-32768)), (-1.0, -1.0));
        assert_eq!((dequantize_unorm8(255), dequantize_unorm16(65535)), (1.0, 1.0));
        // The error is at most half a level.
        let mut rng = Rng::new(0x9a);
        for _ in 0..10_000 {
            let v = rng.range_f32(-1.0, 1.0);
            assert!((dequantize_snorm8(quantize_snorm8(v)) - v).abs() <= 0.5 / 127.0 + 1e-6);
            assert!((dequantize_snorm16(quantize_snorm16(v)) - v).abs() <= 0.5 / 32767.0 + 1e-7);
            let u = v.abs();
            assert!((dequantize_unorm8(quantize_unorm8(u)) - u).abs() <= 0.5 / 255.0 + 1e-6);
            assert!((dequantize_unorm16(quantize_unorm16(u)) - u).abs() <= 0.5 / 65535.0 + 1e-7);
        }
    }

    #[test]
    fn wgsl_pack_builtins_lay_out_component_zero_lowest() {
        assert_eq!(pack4x8unorm(Vec4::new(1.0, 0.0, 0.5, 1.0)), 0xff80_00ff);
        assert_eq!(pack4x8snorm(Vec4::new(-1.0, 1.0, 0.0, -0.5)), 0xc100_7f81);
        assert_eq!(pack2x16unorm(Vec2::new(1.0, 0.0)), 0x0000_ffff);
        assert_eq!(pack2x16snorm(Vec2::new(-1.0, 1.0)), 0x7fff_8001);
        assert_eq!(pack2x16float(Vec2::new(1.0, -2.0)), 0xc000_3c00);
        assert_eq!(unpack4x8unorm(0xff80_00ff), Vec4::new(1.0, 0.0, 128.0 / 255.0, 1.0));
        assert_eq!(unpack4x8snorm(0xc100_7f80), Vec4::new(-1.0, 1.0, 0.0, -63.0 / 127.0));
        assert_eq!(unpack2x16unorm(0x0000_ffff), Vec2::new(1.0, 0.0));
        assert_eq!(unpack2x16snorm(0x7fff_8000), Vec2::new(-1.0, 1.0));
        assert_eq!(unpack2x16float(0xc000_3c00), Vec2::new(1.0, -2.0));
        let v = Vec4::new(0.25, -3.5, 1024.0, 65504.0);
        assert_eq!(pack4x16float(v), [0xc300_3400, 0x7bff_6400]);
        assert_eq!(unpack4x16float(pack4x16float(v)), v);
        // Random round trips stay within half a level.
        let mut rng = Rng::new(0x9b);
        for _ in 0..10_000 {
            let s = Vec4::new(rng.range_f32(-1.0, 1.0), rng.range_f32(-1.0, 1.0), rng.range_f32(-1.0, 1.0), 0.3);
            assert!(unpack4x8snorm(pack4x8snorm(s)).abs_diff_eq(s, 0.5 / 127.0 + 1e-6));
            let u = s.abs();
            assert!(unpack4x8unorm(pack4x8unorm(u)).abs_diff_eq(u, 0.5 / 255.0 + 1e-6));
            let (s2, u2) = (Vec2::new(s.x, s.y), Vec2::new(u.x, u.y));
            assert!(unpack2x16snorm(pack2x16snorm(s2)).abs_diff_eq(s2, 0.5 / 32767.0 + 1e-7));
            assert!(unpack2x16unorm(pack2x16unorm(u2)).abs_diff_eq(u2, 0.5 / 65535.0 + 1e-7));
            assert!(unpack2x16float(pack2x16float(s2)).abs_diff_eq(s2, 2f32.powi(-11)));
        }
    }

    #[test]
    fn ten_ten_ten_two_round_trips_every_level() {
        for i in 0..1024_u32 {
            let p = i | (i << 10) | (i << 20) | ((i & 3) << 30);
            assert_eq!(pack_unorm_10_10_10_2(unpack_unorm_10_10_10_2(p)), p, "{i}");
            let s = sign_extend(i, 10);
            let w = sign_extend(i & 3, 2);
            let decoded = unpack_snorm_10_10_10_2(p);
            assert_eq!(decoded.x, (s as f32 / 511.0).max(-1.0));
            assert_eq!(decoded.w, (w as f32).max(-1.0));
            // -512 and -2 are the extra minimum codes: they re-encode as -511 and -1.
            let want = ((s.max(-511) as u32 & 0x3ff) * 0x0010_0401) | ((w.max(-1) as u32 & 3) << 30);
            assert_eq!(pack_snorm_10_10_10_2(decoded), want, "{i}");
        }
        assert_eq!(pack_unorm_10_10_10_2(Vec4::new(1.0, 0.0, 0.0, 1.0)), 0xc000_03ff);
        assert_eq!(pack_unorm_10_10_10_2(Vec4::new(0.0, 0.0, 1.0, 0.0)), 0x3ff0_0000);
        assert_eq!(pack_snorm_10_10_10_2(Vec4::new(-1.0, 0.0, 0.0, -1.0)), 0xc000_0201);
        assert_eq!(unpack_snorm_10_10_10_2(0xc000_0201), Vec4::new(-1.0, 0.0, 0.0, -1.0));
        assert_eq!(pack_snorm_10_10_10_2(Vec4::splat(f32::NAN)), 0);
        let mut rng = Rng::new(0x9c);
        for _ in 0..10_000 {
            let n = rand_unit(&mut rng);
            let v = Vec4::new(n.x, n.y, n.z, if rng.chance(0.5) { 1.0 } else { -1.0 });
            assert!(unpack_snorm_10_10_10_2(pack_snorm_10_10_10_2(v)).abs_diff_eq(v, 0.5 / 511.0 + 1e-6));
            let u = v.abs();
            assert!(unpack_unorm_10_10_10_2(pack_unorm_10_10_10_2(u)).abs_diff_eq(u, 0.5 / 1023.0 + 1e-6));
        }
    }

    #[test]
    fn eval_05_octahedral_encoding_is_accurate_everywhere() {
        // The exact (unquantised) mapping is a bijection of the sphere and the square.
        for v in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z] {
            assert!(oct_decode(oct_encode(v)).abs_diff_eq(v, 1e-7), "{v:?}");
            assert!(unpack_oct16(pack_oct16(v)).abs_diff_eq(v, 1e-7), "{v:?}");
            assert!(unpack_oct8(pack_oct8(v)).abs_diff_eq(v, 1e-7), "{v:?}");
            assert!(unpack_tangent(pack_tangent(v, 1.0)).0.abs_diff_eq(v, 1e-7), "{v:?}");
        }
        let mut rng = Rng::new(0x9d);
        let (mut worst16, mut worst8, mut worst_t, mut worst_rounded8) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
        for _ in 0..200_000 {
            let n = rand_unit(&mut rng);
            let e = oct_encode(n);
            assert!(e.x.abs() <= 1.0 && e.y.abs() <= 1.0);
            assert!(angle(oct_decode(e), n) < 2e-6);
            let p = pack_oct16(n * rng.range_f32(0.1, 10.0));
            assert_eq!(oct_decode(unpack2x16snorm(p)), unpack_oct16(p), "the shader decode");
            worst16 = worst16.max(angle(unpack_oct16(p), n));
            worst8 = worst8.max(angle(unpack_oct8(pack_oct8(n)), n));
            let sign = if rng.chance(0.5) { 1.0 } else { -1.0 };
            let (t, s) = unpack_tangent(pack_tangent(n, sign));
            assert_eq!(s, sign);
            worst_t = worst_t.max(angle(t, n));
            // The precise search is never worse than rounding each coordinate.
            let rounded = Vec2::new(dequantize_snorm8(quantize_snorm8(e.x)), dequantize_snorm8(quantize_snorm8(e.y)));
            let rounded_error = angle(oct_decode(rounded), n);
            assert!(angle(unpack_oct8(pack_oct8(n)), n) <= rounded_error + 1e-6);
            worst_rounded8 = worst_rounded8.max(rounded_error);
        }
        assert!(worst8 < worst_rounded8 * 0.9, "precise {worst8} vs rounded {worst_rounded8}");
        assert!(worst16.to_degrees() < 0.0025, "oct16 worst {}°", worst16.to_degrees());
        assert!(worst8.to_degrees() < 0.63, "oct8 worst {}°", worst8.to_degrees());
        assert!(worst_t.to_degrees() < 0.0043, "tangent worst {}°", worst_t.to_degrees());
        // Degenerate input encodes +Z and never panics.
        for bad in [Vec3::ZERO, Vec3::NAN, Vec3::INFINITY] {
            assert_eq!(unpack_oct16(pack_oct16(bad)), Vec3::Z);
            assert_eq!(unpack_oct8(pack_oct8(bad)), Vec3::Z);
            assert_eq!(oct_decode(oct_encode(bad)), Vec3::Z);
        }
        assert!(oct_decode(Vec2::new(5.0, -9.0)).is_normalized());
        assert_eq!(unpack_tangent(pack_tangent(Vec3::X, -0.0)).1, 1.0, "only negative numbers flip");
    }

    #[test]
    fn slice_packers_are_bounded_by_the_shortest_slice() {
        let normals = [Vec3::X, -Vec3::Y, Vec3::new(0.0, 0.6, -0.8)];
        let mut packed = [0_u32; 4];
        assert_eq!(pack_oct16_slice(&normals, &mut packed), 3);
        assert_eq!(packed[3], 0);
        let mut soa = [7_u32; 3];
        let (x, y, z) = (normals.map(|n| n.x), normals.map(|n| n.y), normals.map(|n| n.z));
        assert_eq!(pack_oct16_soa(&x, &y, &z[..2], &mut soa), 2);
        assert_eq!(soa, [packed[0], packed[1], 7]);
        let mut back = [Vec3::ONE; 2];
        assert_eq!(unpack_oct16_slice(&packed, &mut back), 2);
        assert!(back[0].abs_diff_eq(Vec3::X, 1e-6) && back[1].abs_diff_eq(-Vec3::Y, 1e-6));
        let mut tangents = [0_u32; 3];
        assert_eq!(pack_tangent_slice(&normals, &[1.0, -1.0], &mut tangents), 2);
        assert_eq!(tangents, [pack_tangent(Vec3::X, 1.0), pack_tangent(-Vec3::Y, -1.0), 0]);
        assert_eq!(pack_oct16_slice(&[], &mut packed), 0);
    }
}
