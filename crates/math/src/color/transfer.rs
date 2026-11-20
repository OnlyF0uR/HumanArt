//! Transfer functions: the sRGB / Display-P3 curve (IEC 61966-2-1), 8-bit tables and a generic gamma.
//!
//! "Decode" (the EOTF direction) turns an encoded display value into a linear one; "encode" (the
//! OETF direction) is its inverse. The piecewise curve is the exact one of the standard, with its
//! constants `0.04045`, `12.92`, `0.055`, `1.055`, `2.4` and `0.0031308`.
//!
//! Out-of-range input: the curve is extended as an odd function, `f(-v) = -f(v)` (the scRGB
//! convention), so negative out-of-gamut values survive a round trip, and values above 1 continue the
//! power segment. A result that would overflow saturates at the largest finite value, so finite input
//! always gives finite output. NaN input never panics; its result is unspecified.
//!
//! The 8-bit paths use tables built at compile time: [`SRGB8_TO_LINEAR`] for decoding and a threshold
//! table for [`linear_to_srgb8`], which finds the correctly rounded code with eight comparisons and no
//! `powf`.

/// Encoded values at or below this are on the linear segment of the sRGB curve.
pub const SRGB_DECODE_BREAK: f64 = 0.04045;
/// Linear values at or below this are on the linear segment of the sRGB curve.
pub const SRGB_ENCODE_BREAK: f64 = 0.003_130_8;

/// Decodes one sRGB-encoded channel to linear light (the sRGB EOTF).
#[inline]
pub fn srgb_to_linear(v: f64) -> f64 {
    let a = v.abs();
    let l = if a <= SRGB_DECODE_BREAK { a / 12.92 } else { ((a + 0.055) / 1.055).powf(2.4) };
    l.min(f64::MAX).copysign(v)
}

/// Encodes one linear channel with the sRGB curve (the inverse EOTF).
#[inline]
pub fn linear_to_srgb(v: f64) -> f64 {
    let a = v.abs();
    let e = if a <= SRGB_ENCODE_BREAK { a * 12.92 } else { 1.055 * a.powf(1.0 / 2.4) - 0.055 };
    e.min(f64::MAX).copysign(v)
}

/// [`srgb_to_linear`] in `f32`: the CPU reference for the shader-side decode.
#[inline]
pub fn srgb_to_linear_f32(v: f32) -> f32 {
    let a = v.abs();
    let l = if a <= 0.04045 { a / 12.92 } else { ((a + 0.055) / 1.055).powf(2.4) };
    l.min(f32::MAX).copysign(v)
}

/// [`linear_to_srgb`] in `f32`: the CPU reference for the shader-side encode.
#[inline]
pub fn linear_to_srgb_f32(v: f32) -> f32 {
    let a = v.abs();
    let e = if a <= 0.003_130_8 { a * 12.92 } else { 1.055 * a.powf(1.0 / 2.4) - 0.055 };
    e.min(f32::MAX).copysign(v)
}

/// Decodes an sRGB-encoded triple to linear, channel by channel.
#[inline]
pub fn srgb_to_linear_rgb(c: [f64; 3]) -> [f64; 3] {
    [srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2])]
}

/// Encodes a linear triple with the sRGB curve, channel by channel.
#[inline]
pub fn linear_to_srgb_rgb(c: [f64; 3]) -> [f64; 3] {
    [linear_to_srgb(c[0]), linear_to_srgb(c[1]), linear_to_srgb(c[2])]
}

/// [`srgb_to_linear_rgb`] in `f32`.
#[inline]
pub fn srgb_to_linear_rgb_f32(c: [f32; 3]) -> [f32; 3] {
    [srgb_to_linear_f32(c[0]), srgb_to_linear_f32(c[1]), srgb_to_linear_f32(c[2])]
}

/// [`linear_to_srgb_rgb`] in `f32`.
#[inline]
pub fn linear_to_srgb_rgb_f32(c: [f32; 3]) -> [f32; 3] {
    [linear_to_srgb_f32(c[0]), linear_to_srgb_f32(c[1]), linear_to_srgb_f32(c[2])]
}

/// Decodes one Display-P3-encoded channel. Display-P3 uses the sRGB curve; only its primaries
/// differ (see [`super::spaces`]).
#[inline]
pub fn display_p3_to_linear(v: f64) -> f64 {
    srgb_to_linear(v)
}

/// Encodes one linear Display-P3 channel (the sRGB curve).
#[inline]
pub fn linear_to_display_p3(v: f64) -> f64 {
    linear_to_srgb(v)
}

/// [`display_p3_to_linear`] in `f32`.
#[inline]
pub fn display_p3_to_linear_f32(v: f32) -> f32 {
    srgb_to_linear_f32(v)
}

/// [`linear_to_display_p3`] in `f32`.
#[inline]
pub fn linear_to_display_p3_f32(v: f32) -> f32 {
    linear_to_srgb_f32(v)
}

/// The exponent a pure-gamma function really uses: `gamma` itself, or 1 (the identity) when `gamma`
/// is not a finite positive number.
#[inline]
fn usable_gamma(gamma: f64) -> f64 {
    if gamma.is_finite() && gamma > 0.0 { gamma } else { 1.0 }
}

/// Decodes a pure power-law ("gamma") encoded channel: `sign(v) · |v|^gamma`.
///
/// A `gamma` that is not finite and positive is treated as 1 (the identity).
#[inline]
pub fn gamma_to_linear(v: f64, gamma: f64) -> f64 {
    v.abs().powf(usable_gamma(gamma)).min(f64::MAX).copysign(v)
}

/// Encodes a linear channel with a pure power law: `sign(v) · |v|^(1/gamma)`.
///
/// A `gamma` that is not finite and positive is treated as 1 (the identity).
#[inline]
pub fn linear_to_gamma(v: f64, gamma: f64) -> f64 {
    v.abs().powf(1.0 / usable_gamma(gamma)).min(f64::MAX).copysign(v)
}

/// The fifth root of `a` for `a` in `(0, 1]`, by Newton's method from above (which converges
/// monotonically for this convex function), usable in constant evaluation.
const fn fifth_root(a: f64) -> f64 {
    let mut y = 1.0;
    let mut i = 0;
    while i < 64 {
        let y4 = y * y * y * y;
        y -= (y4 * y - a) / (5.0 * y4);
        i += 1;
    }
    y
}

/// The sRGB decode for `v` in `[0, 1]` in constant evaluation: `x^2.4 = x² · (x^(1/5))²`.
const fn srgb_to_linear_const(v: f64) -> f64 {
    if v <= SRGB_DECODE_BREAK {
        v / 12.92
    } else {
        let x = (v + 0.055) / 1.055;
        let r = fifth_root(x);
        x * x * r * r
    }
}

const fn build_decode_table() -> [f32; 256] {
    let mut table = [0.0_f32; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = srgb_to_linear_const(i as f64 / 255.0) as f32;
        i += 1;
    }
    table
}

const fn build_encode_thresholds() -> [f32; 256] {
    let mut table = [0.0_f32; 256];
    let mut i = 1;
    while i < 256 {
        table[i] = srgb_to_linear_const((i as f64 - 0.5) / 255.0) as f32;
        i += 1;
    }
    table
}

/// The linear value of every 8-bit sRGB code: `SRGB8_TO_LINEAR[c] = srgb_to_linear(c / 255)`.
pub const SRGB8_TO_LINEAR: [f32; 256] = build_decode_table();

/// `ENCODE_THRESHOLDS[i]` (for `i ≥ 1`) is the linear value whose encoding is exactly `i − 0.5` codes:
/// the smallest linear value that rounds to code `i`. Entry 0 is unused.
const ENCODE_THRESHOLDS: [f32; 256] = build_encode_thresholds();

/// Decodes an 8-bit sRGB code to linear by table lookup.
#[inline]
pub const fn srgb8_to_linear(code: u8) -> f32 {
    SRGB8_TO_LINEAR[code as usize]
}

/// Encodes a linear value to the nearest 8-bit sRGB code: `round(255 · linear_to_srgb(v))`, clamped.
///
/// A binary search over the 255 rounding boundaries: eight comparisons, no `powf`, and monotonic by
/// construction. Values below 0 (and NaN) give 0; values above 1 give 255.
#[inline]
pub const fn linear_to_srgb8(v: f32) -> u8 {
    // The code is the number of boundaries at or below `v`; `lo + step` never exceeds 255.
    let mut lo = 0_usize;
    let mut step = 128_usize;
    while step > 0 {
        if v >= ENCODE_THRESHOLDS[lo + step] {
            lo += step;
        }
        step >>= 1;
    }
    lo as u8
}

/// Decodes an 8-bit sRGB triple to linear.
#[inline]
pub const fn srgb8_to_linear_rgb(c: [u8; 3]) -> [f32; 3] {
    [srgb8_to_linear(c[0]), srgb8_to_linear(c[1]), srgb8_to_linear(c[2])]
}

/// Encodes a linear triple to 8-bit sRGB (see [`linear_to_srgb8`] for the clamping).
#[inline]
pub const fn linear_to_srgb8_rgb(c: [f32; 3]) -> [u8; 3] {
    [linear_to_srgb8(c[0]), linear_to_srgb8(c[1]), linear_to_srgb8(c[2])]
}

/// An 8-bit normalised value (alpha, masks) as a float: `code / 255`.
#[inline]
pub const fn unorm8_to_f32(code: u8) -> f32 {
    code as f32 / 255.0
}

/// The nearest 8-bit normalised code for `v`, clamped to `[0, 1]`; NaN gives 0.
#[inline]
pub const fn f32_to_unorm8(v: f32) -> u8 {
    // A float-to-integer `as` cast saturates and maps NaN to 0.
    (v * 255.0 + 0.5) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    #[test]
    fn known_values_of_the_srgb_curve() {
        assert_eq!(srgb_to_linear(0.0), 0.0);
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-15);
        assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-15);
        // Reference values of the IEC 61966-2-1 formula.
        assert!((srgb_to_linear(0.5) - 0.214_041_140_482_232_55).abs() < 1e-14);
        assert!((linear_to_srgb(0.18) - 0.461_356_129_500_441_64).abs() < 1e-14);
        assert!((linear_to_srgb(0.5) - 0.735_356_983_052_449_5).abs() < 1e-14);
        // The linear segment.
        assert!((srgb_to_linear(0.02) - 0.02 / 12.92).abs() < 1e-18);
        assert!((linear_to_srgb(0.001) - 0.012_92).abs() < 1e-16);
        // f32 versions agree with f64.
        assert!((f64::from(srgb_to_linear_f32(0.5)) - 0.214_041_140_482_232_55).abs() < 1e-7);
        assert!((f64::from(linear_to_srgb_f32(0.18)) - 0.461_356_129_500_441_64).abs() < 1e-7);
    }

    #[test]
    fn the_two_segments_meet_at_the_break_points() {
        let below = srgb_to_linear(SRGB_DECODE_BREAK);
        let above = srgb_to_linear(SRGB_DECODE_BREAK + 1e-12);
        assert!((above - below).abs() < 1e-7);
        let below = linear_to_srgb(SRGB_ENCODE_BREAK);
        let above = linear_to_srgb(SRGB_ENCODE_BREAK + 1e-12);
        assert!((above - below).abs() < 1e-6);
        // The two break points correspond to each other.
        assert!((srgb_to_linear(SRGB_DECODE_BREAK) - SRGB_ENCODE_BREAK).abs() < 1e-7);
    }

    #[test]
    fn round_trips_and_monotonicity() {
        let mut rng = Rng::new(0x5B6B);
        for _ in 0..20_000 {
            let v = rng.range_f64(-2.0, 8.0);
            assert!((linear_to_srgb(srgb_to_linear(v)) - v).abs() < 1e-12, "{v}");
            assert!((srgb_to_linear(linear_to_srgb(v)) - v).abs() < 1e-12 * v.abs().max(1.0), "{v}");
            let f = v as f32;
            assert!((linear_to_srgb_f32(srgb_to_linear_f32(f)) - f).abs() < 4e-6 * f.abs().max(1.0), "{f}");
        }
        let mut last = srgb_to_linear(-1.0);
        for i in 1..=4000 {
            let now = srgb_to_linear(-1.0 + f64::from(i) * 0.001);
            assert!(now > last);
            last = now;
        }
    }

    #[test]
    fn the_curve_is_odd_and_saturates_instead_of_overflowing() {
        assert_eq!(srgb_to_linear(-0.5), -srgb_to_linear(0.5));
        assert_eq!(linear_to_srgb(-0.25), -linear_to_srgb(0.25));
        assert_eq!(srgb_to_linear_f32(-0.5), -srgb_to_linear_f32(0.5));
        assert_eq!(srgb_to_linear(f64::MAX), f64::MAX);
        assert_eq!(srgb_to_linear(-f64::MAX), -f64::MAX);
        assert_eq!(srgb_to_linear_f32(f32::MAX), f32::MAX);
        assert!(linear_to_srgb(f64::MAX).is_finite());
        assert!(linear_to_srgb_f32(f32::MAX).is_finite());
        assert!(gamma_to_linear(f64::MAX, 3.0).is_finite());
        assert!(linear_to_gamma(-f64::MAX, 0.25).is_finite());
    }

    #[test]
    fn decode_table_matches_the_runtime_curve() {
        for code in 0..=255_u8 {
            let exact = srgb_to_linear(f64::from(code) / 255.0);
            assert_eq!(srgb8_to_linear(code), exact as f32, "code {code}");
        }
        assert_eq!(srgb8_to_linear(0), 0.0);
        assert_eq!(srgb8_to_linear(255), 1.0);
        assert_eq!(srgb8_to_linear_rgb([0, 255, 128]), [0.0, 1.0, SRGB8_TO_LINEAR[128]]);
    }

    #[test]
    fn all_256_codes_round_trip_exactly() {
        for code in 0..=255_u8 {
            // Table decode, threshold encode.
            assert_eq!(linear_to_srgb8(srgb8_to_linear(code)), code);
            // The f64 and f32 analytic paths.
            let back = (linear_to_srgb(srgb_to_linear(f64::from(code) / 255.0)) * 255.0).round();
            assert_eq!(back, f64::from(code));
            let back = (linear_to_srgb_f32(srgb_to_linear_f32(f32::from(code) / 255.0)) * 255.0).round();
            assert_eq!(back, f32::from(code));
            assert_eq!(f32_to_unorm8(unorm8_to_f32(code)), code);
        }
        assert_eq!(linear_to_srgb8_rgb(srgb8_to_linear_rgb([12, 127, 250])), [12, 127, 250]);
    }

    #[test]
    fn fast_encode_matches_the_rounded_analytic_encode() {
        let mut rng = Rng::new(0xE2C0DE);
        let mut checked = 0;
        for i in 0..300_000 {
            // Half the samples are uniform in encoded space so the dark codes are exercised too.
            let v = if i % 2 == 0 { rng.f32() } else { srgb_to_linear_f32(rng.f32()) };
            let scaled = linear_to_srgb(f64::from(v)) * 255.0;
            let fraction = scaled - scaled.floor();
            if (fraction - 0.5).abs() < 1e-4 {
                continue; // on a rounding boundary to within the f32 resolution of the table
            }
            assert_eq!(f64::from(linear_to_srgb8(v)), scaled.round(), "linear {v}");
            checked += 1;
        }
        assert!(checked > 290_000);
    }

    #[test]
    fn fast_encode_clamps_and_is_monotonic() {
        assert_eq!(linear_to_srgb8(-1.0), 0);
        assert_eq!(linear_to_srgb8(f32::NAN), 0);
        assert_eq!(linear_to_srgb8(f32::NEG_INFINITY), 0);
        assert_eq!(linear_to_srgb8(0.0), 0);
        assert_eq!(linear_to_srgb8(1.0), 255);
        assert_eq!(linear_to_srgb8(7.5), 255);
        assert_eq!(linear_to_srgb8(f32::INFINITY), 255);
        let mut last = 0;
        for i in 0..=100_000 {
            let code = linear_to_srgb8(i as f32 / 100_000.0);
            assert!(code >= last);
            last = code;
        }
        assert_eq!(last, 255);
        assert_eq!(f32_to_unorm8(-3.0), 0);
        assert_eq!(f32_to_unorm8(f32::NAN), 0);
        assert_eq!(f32_to_unorm8(9.0), 255);
        assert_eq!(f32_to_unorm8(0.5), 128);
    }

    #[test]
    fn display_p3_shares_the_srgb_curve() {
        for v in [-0.3, 0.0, 0.01, 0.4, 1.0, 1.7] {
            assert_eq!(display_p3_to_linear(v), srgb_to_linear(v));
            assert_eq!(linear_to_display_p3(v), linear_to_srgb(v));
            assert_eq!(display_p3_to_linear_f32(v as f32), srgb_to_linear_f32(v as f32));
            assert_eq!(linear_to_display_p3_f32(v as f32), linear_to_srgb_f32(v as f32));
        }
    }

    #[test]
    fn generic_gamma_round_trips_and_rejects_bad_exponents() {
        assert!((gamma_to_linear(0.5, 2.2) - 0.5_f64.powf(2.2)).abs() < 1e-15);
        assert!((linear_to_gamma(0.5, 2.2) - 0.5_f64.powf(1.0 / 2.2)).abs() < 1e-15);
        let mut rng = Rng::new(0x6A33A);
        for _ in 0..5000 {
            let v = rng.range_f64(-3.0, 3.0);
            let gamma = rng.range_f64(0.4, 3.0);
            assert!((linear_to_gamma(gamma_to_linear(v, gamma), gamma) - v).abs() < 1e-12);
        }
        assert_eq!(gamma_to_linear(-0.25, 2.0), -0.0625);
        for bad in [0.0, -2.2, f64::NAN, f64::INFINITY] {
            assert_eq!(gamma_to_linear(0.3, bad), 0.3);
            assert_eq!(linear_to_gamma(0.3, bad), 0.3);
        }
    }

    #[test]
    fn triples_apply_the_curve_per_channel() {
        let c = [0.1, 0.5, 0.9];
        assert_eq!(srgb_to_linear_rgb(c), [srgb_to_linear(0.1), srgb_to_linear(0.5), srgb_to_linear(0.9)]);
        assert_eq!(linear_to_srgb_rgb(c), [linear_to_srgb(0.1), linear_to_srgb(0.5), linear_to_srgb(0.9)]);
        let f = [0.1_f32, 0.5, 0.9];
        assert_eq!(srgb_to_linear_rgb_f32(f)[1], srgb_to_linear_f32(0.5));
        assert_eq!(linear_to_srgb_rgb_f32(f)[2], linear_to_srgb_f32(0.9));
    }
}
