//! IEEE 754 binary16 ("half", `f16`) and its exact conversions to and from `f32`.
//!
//! `f16 -> f32` is exact for every one of the 65 536 bit patterns. `f32 -> f16` rounds to nearest,
//! ties to even, with gradual underflow into the subnormals and overflow to infinity, exactly as
//! the IEEE 754 `convertFormat` operation and the GPU's `pack2x16float` do. NaN payloads keep their
//! ten most significant bits, so every half value survives a trip through `f32` bit for bit
//! (`[AC-WP02-1]`, tested over the full 16-bit space).
//!
//! Binary formats store a half as two little-endian bytes; the slice helpers convert whole
//! structure-of-arrays channels without allocating.

use core::cmp::Ordering;
use core::fmt;

/// The sign bit of a half.
const SIGN: u16 = 0x8000;
/// The exponent field of a half: all ones for infinities and NaNs.
const EXP: u16 = 0x7c00;
/// The mantissa field of a half.
const MANT: u16 = 0x03ff;
/// The quiet bit of a half NaN (the top mantissa bit).
const QUIET: u16 = 0x0200;

/// Converts an `f32` to the bits of the nearest half, ties to even. Values of magnitude 65 520 and
/// above become infinities, values of magnitude 2⁻²⁵ and below become signed zeros, NaNs stay
/// NaNs (the ten most significant payload bits are kept; a payload that lives only in the low bits
/// becomes the quiet NaN payload).
pub const fn f32_to_f16_bits(value: f32) -> u16 {
    let x = value.to_bits();
    let sign = ((x >> 16) as u16) & SIGN;
    let abs = x & 0x7fff_ffff;
    if abs >= 0x7f80_0000 {
        if abs == 0x7f80_0000 {
            return sign | EXP;
        }
        let payload = ((abs >> 13) as u16) & MANT;
        return sign | EXP | if payload == 0 { QUIET } else { payload };
    }
    if abs >= 0x4780_0000 {
        // 65 536 and above: beyond even the rounding range of the largest finite half.
        return sign | EXP;
    }
    if abs >= 0x3880_0000 {
        // A normal half: rebias the exponent from 127 to 15 and drop 13 mantissa bits. A carry out
        // of the mantissa correctly bumps the exponent, up to infinity for [65 520, 65 536).
        let mut h = (abs - 0x3800_0000) >> 13;
        let rem = abs & 0x1fff;
        if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
            h += 1;
        }
        return sign | h as u16;
    }
    if abs <= 0x3300_0000 {
        // At most 2⁻²⁵, half the smallest subnormal: the tie goes to the even zero.
        return sign;
    }
    // A subnormal half m · 2⁻²⁴ with the f32 value mant · 2^(e - 150): m = mant >> (126 - e).
    let e = abs >> 23;
    let mant = (abs & 0x007f_ffff) | 0x0080_0000;
    let shift = 126 - e;
    let mut h = mant >> shift;
    let rem = mant & ((1 << shift) - 1);
    let half_ulp = 1 << (shift - 1);
    if rem > half_ulp || (rem == half_ulp && h & 1 == 1) {
        // Rounding up from the largest subnormal gives exactly the smallest normal's bits.
        h += 1;
    }
    sign | h as u16
}

/// Converts the bits of a half to the `f32` with the same value (exact; NaN payloads are kept).
pub const fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & SIGN) as u32) << 16;
    let exp = (bits & EXP) >> 10;
    let mant = (bits & MANT) as u32;
    let out = if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else if exp != 0 {
        sign | ((exp as u32 + 112) << 23) | (mant << 13)
    } else if mant == 0 {
        sign
    } else {
        // Subnormal: shift the leading one up to the implicit bit and lower the exponent to match.
        let shift = mant.leading_zeros() - 21;
        sign | ((113 - shift) << 23) | (((mant << shift) & 0x3ff) << 13)
    };
    f32::from_bits(out)
}

/// An IEEE 754 binary16 floating-point number, stored as its bits. Two bytes, the layout of WGSL's
/// `f16` and of half channels in EXR and vertex buffers.
///
/// Comparisons follow IEEE semantics (`NaN != NaN`, `0 == -0`), like `f32`; use
/// [`F16::to_bits`] for bitwise equality.
#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct F16(u16);

// The layout is a contract with `humanart-gpu` (`Pod`) and the file formats: one `u16`.
const _: () = assert!(size_of::<F16>() == 2 && align_of::<F16>() == 2);

impl F16 {
    /// Positive zero.
    pub const ZERO: Self = Self(0);
    /// Negative zero.
    pub const NEG_ZERO: Self = Self(SIGN);
    /// One.
    pub const ONE: Self = Self(0x3c00);
    /// Minus one.
    pub const NEG_ONE: Self = Self(0xbc00);
    /// The largest finite value, 65 504.
    pub const MAX: Self = Self(0x7bff);
    /// The most negative finite value, -65 504.
    pub const MIN: Self = Self(0xfbff);
    /// The smallest positive normal value, 2⁻¹⁴.
    pub const MIN_POSITIVE: Self = Self(0x0400);
    /// The smallest positive subnormal value, 2⁻²⁴.
    pub const MIN_POSITIVE_SUBNORMAL: Self = Self(0x0001);
    /// The difference between one and the next larger value, 2⁻¹⁰.
    pub const EPSILON: Self = Self(0x1400);
    /// Positive infinity.
    pub const INFINITY: Self = Self(EXP);
    /// Negative infinity.
    pub const NEG_INFINITY: Self = Self(SIGN | EXP);
    /// The canonical quiet NaN.
    pub const NAN: Self = Self(EXP | QUIET);

    /// The half nearest to `value` (see [`f32_to_f16_bits`]).
    #[inline]
    pub const fn from_f32(value: f32) -> Self {
        Self(f32_to_f16_bits(value))
    }

    /// The exact `f32` value of this half.
    #[inline]
    pub const fn to_f32(self) -> f32 {
        f16_bits_to_f32(self.0)
    }

    /// The exact `f64` value of this half.
    #[inline]
    pub const fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    /// The half with these bits.
    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// The bits of this half.
    #[inline]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// The half stored in two little-endian bytes.
    #[inline]
    pub const fn from_le_bytes(bytes: [u8; 2]) -> Self {
        Self(u16::from_le_bytes(bytes))
    }

    /// The two little-endian bytes of this half.
    #[inline]
    pub const fn to_le_bytes(self) -> [u8; 2] {
        self.0.to_le_bytes()
    }

    /// Whether this is a NaN.
    #[inline]
    pub const fn is_nan(self) -> bool {
        self.0 & EXP == EXP && self.0 & MANT != 0
    }

    /// Whether this is positive or negative infinity.
    #[inline]
    pub const fn is_infinite(self) -> bool {
        self.0 & !SIGN == EXP
    }

    /// Whether this is neither infinite nor NaN.
    #[inline]
    pub const fn is_finite(self) -> bool {
        self.0 & EXP != EXP
    }

    /// Whether this is a non-zero subnormal value.
    #[inline]
    pub const fn is_subnormal(self) -> bool {
        self.0 & EXP == 0 && self.0 & MANT != 0
    }

    /// Whether the sign bit is set (also for `-0` and negative NaNs).
    #[inline]
    pub const fn is_sign_negative(self) -> bool {
        self.0 & SIGN != 0
    }

    /// The absolute value (clears the sign bit).
    #[inline]
    pub const fn abs(self) -> Self {
        Self(self.0 & !SIGN)
    }
}

impl From<F16> for f32 {
    #[inline]
    fn from(h: F16) -> f32 {
        h.to_f32()
    }
}

impl From<F16> for f64 {
    #[inline]
    fn from(h: F16) -> f64 {
        h.to_f64()
    }
}

/// IEEE equality: NaN equals nothing, `0` equals `-0`.
impl PartialEq for F16 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.to_f32() == other.to_f32()
    }
}

/// IEEE ordering: NaN is unordered.
impl PartialOrd for F16 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

/// Shows the value, like `f32`.
impl fmt::Debug for F16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.to_f32(), f)
    }
}

/// Shows the value, like `f32`.
impl fmt::Display for F16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.to_f32(), f)
    }
}

/// Converts `src` into `dst` element by element, rounding to nearest even. Converts
/// `min(src.len(), dst.len())` values and returns that count; the rest of `dst` is untouched.
pub fn f32_to_f16_slice(src: &[f32], dst: &mut [F16]) -> usize {
    let n = src.len().min(dst.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = F16::from_f32(s);
    }
    n
}

/// Converts `src` into `dst` exactly. Converts `min(src.len(), dst.len())` values and returns that
/// count; the rest of `dst` is untouched.
pub fn f16_to_f32_slice(src: &[F16], dst: &mut [f32]) -> usize {
    let n = src.len().min(dst.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = s.to_f32();
    }
    n
}

/// Writes `src` as little-endian halves (two bytes each) into `dst`. Converts
/// `min(src.len(), dst.len() / 2)` values and returns that count; the rest of `dst` is untouched.
pub fn f32_to_f16_le_bytes(src: &[f32], dst: &mut [u8]) -> usize {
    let n = src.len().min(dst.len() / 2);
    for (d, &s) in dst.as_chunks_mut::<2>().0.iter_mut().zip(src) {
        *d = F16::from_f32(s).to_le_bytes();
    }
    n
}

/// Reads little-endian halves (two bytes each) from `src` into `dst` exactly. Converts
/// `min(src.len() / 2, dst.len())` values and returns that count; a trailing odd byte is ignored and
/// the rest of `dst` is untouched.
pub fn f16_le_bytes_to_f32(src: &[u8], dst: &mut [f32]) -> usize {
    let n = (src.len() / 2).min(dst.len());
    for (d, &s) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
        *d = F16::from_le_bytes(s).to_f32();
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    /// The value of a half from first principles, in `f64`.
    fn reference_value(bits: u16) -> f64 {
        let sign = if bits & SIGN != 0 { -1.0 } else { 1.0 };
        let exp = i32::from((bits >> 10) & 0x1f);
        let mant = f64::from(bits & MANT);
        match exp {
            0 => sign * mant * 2f64.powi(-24),
            0x1f if mant == 0.0 => sign * f64::INFINITY,
            0x1f => f64::NAN,
            _ => sign * (1.0 + mant / 1024.0) * 2f64.powi(exp - 15),
        }
    }

    #[test]
    fn ac_wp02_1_f16_to_f32_is_exact_on_the_full_16_bit_space() {
        for bits in 0..=u16::MAX {
            let h = F16::from_bits(bits);
            let f = h.to_f32();
            let want = reference_value(bits);
            if want.is_nan() {
                assert!(f.is_nan() && h.is_nan(), "{bits:#06x}");
                assert_eq!((f.to_bits() >> 13) & 0x3ff, u32::from(bits & MANT), "payload {bits:#06x}");
            } else {
                assert_eq!(f64::from(f), want, "{bits:#06x}");
                assert_eq!(f.is_sign_negative(), h.is_sign_negative(), "{bits:#06x}");
                assert_eq!(h.is_infinite(), want.is_infinite());
                assert_eq!(h.is_subnormal(), f64::from(f).abs() < 2f64.powi(-14) && want != 0.0);
            }
            assert_eq!(h.is_finite(), want.is_finite());
            // And back, bit for bit, NaNs included.
            assert_eq!(F16::from_f32(f).to_bits(), bits, "round trip {bits:#06x}");
            assert_eq!(h.to_f64().to_bits(), f64::from(f).to_bits());
        }
    }

    #[test]
    fn ac_wp02_1_f32_to_f16_rounds_to_nearest_even_at_every_midpoint() {
        // Between every pair of neighbouring finite halves (and between MAX and the overflow
        // threshold 2¹⁶), the exact midpoint is an f32: it must round to the even neighbour, and
        // the f32 values just beside it to the nearer neighbour.
        for bits in 0..0x7c00_u16 {
            let lo = f16_bits_to_f32(bits);
            let hi = if bits == 0x7bff { 65536.0 } else { f16_bits_to_f32(bits + 1) };
            let mid = f64::midpoint(f64::from(lo), f64::from(hi));
            let mid32 = mid as f32;
            assert_eq!(f64::from(mid32), mid, "midpoint is exact");
            let even = if bits & 1 == 0 { bits } else { bits + 1 };
            for sign in [0, SIGN] {
                let s = if sign == 0 { 1.0 } else { -1.0 };
                assert_eq!(f32_to_f16_bits(s * mid32), sign | even, "tie at {bits:#06x}");
                let below = f32::from_bits(mid32.to_bits() - 1);
                let above = f32::from_bits(mid32.to_bits() + 1);
                assert_eq!(f32_to_f16_bits(s * below), sign | bits, "below {bits:#06x}");
                assert_eq!(f32_to_f16_bits(s * above), sign | (bits + 1), "above {bits:#06x}");
                assert_eq!(f32_to_f16_bits(s * lo), sign | bits, "exact {bits:#06x}");
            }
        }
    }

    #[test]
    fn f32_to_f16_picks_the_nearest_half_for_random_inputs() {
        let mut rng = Rng::new(0x16);
        for _ in 0..200_000 {
            let f = f32::from_bits(rng.next_u32());
            let h = f32_to_f16_bits(f);
            if f.is_nan() {
                assert!(F16::from_bits(h).is_nan());
                continue;
            }
            assert_eq!(h & SIGN != 0, f.is_sign_negative(), "{f}");
            let got = f64::from(f16_bits_to_f32(h));
            let f = f64::from(f);
            if got.is_infinite() {
                assert!(f.abs() >= 65520.0, "{f}");
                continue;
            }
            let err = (got - f).abs();
            // No neighbour is closer.
            let mag = h & !SIGN;
            for n in [mag.wrapping_sub(1), mag + 1] {
                if n < 0x7c00 {
                    let other = f64::from(f16_bits_to_f32(n | (h & SIGN)));
                    assert!(err <= (other - f).abs(), "{f}: {got} vs {other}");
                }
            }
            assert!(f.abs() < 65520.0);
        }
    }

    #[test]
    fn special_values_and_constants() {
        assert_eq!(F16::from_f32(1.0), F16::ONE);
        assert_eq!(F16::from_f32(-1.0).to_bits(), F16::NEG_ONE.to_bits());
        assert_eq!(F16::MAX.to_f32(), 65504.0);
        assert_eq!(F16::MIN.to_f32(), -65504.0);
        assert_eq!(F16::MIN_POSITIVE.to_f32(), 2f32.powi(-14));
        assert_eq!(F16::MIN_POSITIVE_SUBNORMAL.to_f32(), 2f32.powi(-24));
        assert_eq!(F16::EPSILON.to_f32(), 2f32.powi(-10));
        assert_eq!(F16::from_f32(65519.996).to_bits(), F16::MAX.to_bits());
        assert_eq!(F16::from_f32(65520.0).to_bits(), F16::INFINITY.to_bits());
        assert_eq!(F16::from_f32(1e30).to_bits(), F16::INFINITY.to_bits());
        assert_eq!(F16::from_f32(f32::NEG_INFINITY).to_bits(), F16::NEG_INFINITY.to_bits());
        assert_eq!(F16::from_f32(-0.0).to_bits(), F16::NEG_ZERO.to_bits());
        assert_eq!(F16::from_f32(1e-30).to_bits(), 0);
        assert_eq!(F16::from_f32(-1e-30).to_bits(), SIGN);
        assert_eq!(F16::from_f32(f32::MIN_POSITIVE).to_bits(), 0);
        assert_eq!(F16::from_f32(2f32.powi(-25)).to_bits(), 0, "tie to the even zero");
        assert_eq!(F16::from_f32(f32::from_bits(0x3300_0001)).to_bits(), 1);
        assert_eq!(F16::from_f32(f32::NAN).to_bits() & EXP, EXP);
        assert!(F16::from_f32(f32::NAN).is_nan() && F16::NAN.is_nan() && !F16::INFINITY.is_nan());
        // A NaN whose payload is only in the low bits stays a NaN.
        assert!(F16::from_f32(f32::from_bits(0x7f80_0001)).is_nan());
        assert_eq!(F16::from_f32(f32::from_bits(0xff80_0001)).to_bits(), SIGN | EXP | QUIET);
        // IEEE comparisons.
        assert_eq!(F16::ZERO, F16::NEG_ZERO);
        assert_ne!(F16::NAN, F16::NAN);
        assert!(F16::ONE > F16::ZERO && F16::NEG_INFINITY < F16::MIN);
        assert_eq!(F16::NAN.partial_cmp(&F16::ONE), None);
        assert_eq!(F16::NEG_ONE.abs(), F16::ONE);
        assert_eq!(F16::default().to_bits(), 0);
        assert_eq!(format!("{} {:?}", F16::from_f32(0.5), F16::ONE), "0.5 1.0");
        assert_eq!(f32::from(F16::from_f32(0.333_333_34)), 0.333_251_95);
        assert_eq!(f64::from(F16::ONE), 1.0);
        // const evaluation works.
        assert_eq!(const { F16::from_f32(2.0) }.to_bits(), 0x4000);
    }

    #[test]
    fn slice_conversions_are_little_endian_and_bounded() {
        let src = [0.0_f32, 1.0, -2.0, 0.5, 65504.0, 1e9];
        let mut halves = [F16::ZERO; 4];
        assert_eq!(f32_to_f16_slice(&src, &mut halves), 4);
        assert_eq!(halves.map(F16::to_bits), [0x0000, 0x3c00, 0xc000, 0x3800]);
        let mut back = [9.0_f32; 6];
        assert_eq!(f16_to_f32_slice(&halves, &mut back), 4);
        assert_eq!(back, [0.0, 1.0, -2.0, 0.5, 9.0, 9.0]);

        let mut bytes = [0xaa_u8; 13];
        assert_eq!(f32_to_f16_le_bytes(&src, &mut bytes), 6);
        assert_eq!(&bytes[..6], &[0x00, 0x00, 0x00, 0x3c, 0x00, 0xc0]);
        assert_eq!(&bytes[8..13], &[0xff, 0x7b, 0x00, 0x7c, 0xaa]);
        let mut decoded = [0.0_f32; 8];
        assert_eq!(f16_le_bytes_to_f32(&bytes, &mut decoded), 6, "the odd byte is ignored");
        assert_eq!(&decoded[..6], &[0.0, 1.0, -2.0, 0.5, 65504.0, f32::INFINITY]);
        assert_eq!(f16_le_bytes_to_f32(&bytes[..3], &mut decoded), 1);
        assert_eq!(f16_le_bytes_to_f32(&[], &mut decoded), 0);
        assert_eq!(f32_to_f16_le_bytes(&src, &mut []), 0);
        assert_eq!(F16::from_le_bytes([0x00, 0x3c]), F16::ONE);
    }
}
