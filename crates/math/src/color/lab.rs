//! CIELAB, its cylindrical form LCh, and Oklab / Oklch `[COL-03]`.
//!
//! CIELAB follows CIE 15 with the exact rational constants `ε = 216/24389` and `κ = 24389/27`, so the
//! cube-root and linear segments meet without a step. `L*` is 0 for black and 100 for the white
//! point; the white is a parameter, and the `rgb_*` helpers use the D65 of the working space
//! ([`super::spaces::D65_XYZ`]) so that a grey has zero chroma. Triples are `[L, a, b]` and
//! `[L, C, h]` with the hue `h` in degrees in `[0, 360)`.
//!
//! Oklab (Björn Ottosson, "A perceptual color space for image processing", 2020) uses the matrices of
//! the publication for linear sRGB input; the inverse matrices are computed from them at compile
//! time, so a round trip is exact to rounding.
//!
//! Out-of-range input: negative and above-white tristimulus values are legal and convert both ways
//! (the functions extend smoothly). A white point component that is not finite and positive is
//! replaced by the D65 component. The hue of a colour with zero chroma is 0.

use super::matrix::{self, ColorMatrix};
use super::spaces::{D65_XYZ, built_in, rec709_to_xyz, xyz_to_rec709};
use super::transfer;

/// The CIELAB break point `ε = (6/29)³ = 216/24389`: below it the lightness function is linear.
pub const LAB_EPSILON: f64 = 216.0 / 24389.0;
/// The CIELAB slope constant `κ = (29/3)³ = 24389/27`: `L* = κ · Y/Yn` on the linear segment.
pub const LAB_KAPPA: f64 = 24389.0 / 27.0;

#[inline]
fn lab_f(t: f64) -> f64 {
    if t > LAB_EPSILON { t.cbrt() } else { (LAB_KAPPA * t + 16.0) / 116.0 }
}

#[inline]
fn lab_f_inverse(f: f64) -> f64 {
    let cube = f * f * f;
    if cube > LAB_EPSILON { cube } else { (116.0 * f - 16.0) / LAB_KAPPA }
}

/// The white point really used: each component that is not finite and positive falls back to D65.
#[inline]
fn usable_white(white: [f64; 3]) -> [f64; 3] {
    let mut out = white;
    for (w, fallback) in out.iter_mut().zip(D65_XYZ) {
        if !(w.is_finite() && *w > 0.0) {
            *w = fallback;
        }
    }
    out
}

/// XYZ → CIELAB `[L*, a*, b*]` relative to the white point `white` (XYZ, same scale as `xyz`).
pub fn xyz_to_lab(xyz: [f64; 3], white: [f64; 3]) -> [f64; 3] {
    let w = usable_white(white);
    let fx = lab_f(xyz[0] / w[0]);
    let fy = lab_f(xyz[1] / w[1]);
    let fz = lab_f(xyz[2] / w[2]);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// CIELAB `[L*, a*, b*]` → XYZ relative to the white point `white`.
pub fn lab_to_xyz(lab: [f64; 3], white: [f64; 3]) -> [f64; 3] {
    let w = usable_white(white);
    let fy = (lab[0] + 16.0) / 116.0;
    let fx = fy + lab[1] / 500.0;
    let fz = fy - lab[2] / 200.0;
    [lab_f_inverse(fx) * w[0], lab_f_inverse(fy) * w[1], lab_f_inverse(fz) * w[2]]
}

/// Linear Rec.709 RGB → CIELAB under D65.
#[inline]
pub fn rgb_to_lab(rgb: [f64; 3]) -> [f64; 3] {
    xyz_to_lab(rec709_to_xyz(rgb), D65_XYZ)
}

/// CIELAB under D65 → linear Rec.709 RGB (not clamped: colours outside the gamut go negative or
/// above one).
#[inline]
pub fn lab_to_rgb(lab: [f64; 3]) -> [f64; 3] {
    xyz_to_rec709(lab_to_xyz(lab, D65_XYZ))
}

/// An 8-bit sRGB colour → CIELAB under D65.
#[inline]
pub fn srgb8_to_lab(srgb: [u8; 3]) -> [f64; 3] {
    rgb_to_lab(matrix::widen(transfer::srgb8_to_linear_rgb(srgb)))
}

/// CIELAB under D65 → the nearest 8-bit sRGB colour, each channel clamped to the gamut.
#[inline]
pub fn lab_to_srgb8(lab: [f64; 3]) -> [u8; 3] {
    transfer::linear_to_srgb8_rgb(matrix::narrow(lab_to_rgb(lab)))
}

/// The hue angle of the opponent pair `(a, b)` in degrees in `[0, 360)`: 0 along `+a`, 90 along
/// `+b`. The hue of `(0, 0)` is 0.
#[inline]
pub fn hue_degrees(a: f64, b: f64) -> f64 {
    if a == 0.0 && b == 0.0 {
        return 0.0;
    }
    let h = b.atan2(a).to_degrees();
    let h = if h < 0.0 { h + 360.0 } else { h };
    // `-tiny + 360` rounds to 360 itself.
    if h >= 360.0 { 0.0 } else { h }
}

/// The chroma of the opponent pair `(a, b)`: its distance from the neutral axis.
#[inline]
pub fn chroma(a: f64, b: f64) -> f64 {
    a.hypot(b)
}

/// `[L, a, b]` → `[L, C, h]` (works for CIELAB and for Oklab alike).
#[inline]
pub fn lab_to_lch(lab: [f64; 3]) -> [f64; 3] {
    [lab[0], chroma(lab[1], lab[2]), hue_degrees(lab[1], lab[2])]
}

/// `[L, C, h]` → `[L, a, b]`. Any finite hue is accepted (it wraps); a negative chroma points the
/// opposite way.
#[inline]
pub fn lch_to_lab(lch: [f64; 3]) -> [f64; 3] {
    let (sin, cos) = lch[2].to_radians().sin_cos();
    [lch[0], lch[1] * cos, lch[1] * sin]
}

/// Linear Rec.709 RGB → CIE LCh(ab) under D65.
#[inline]
pub fn rgb_to_lch(rgb: [f64; 3]) -> [f64; 3] {
    lab_to_lch(rgb_to_lab(rgb))
}

/// CIE LCh(ab) under D65 → linear Rec.709 RGB (not clamped).
#[inline]
pub fn lch_to_rgb(lch: [f64; 3]) -> [f64; 3] {
    lab_to_rgb(lch_to_lab(lch))
}

/// Oklab's `M1` for linear sRGB input: linear Rec.709 RGB → cone-like LMS.
pub const OKLAB_RGB_TO_LMS: ColorMatrix = [
    [0.412_221_470_8, 0.536_332_536_3, 0.051_445_992_9],
    [0.211_903_498_2, 0.680_699_545_1, 0.107_396_956_6],
    [0.088_302_461_9, 0.281_718_837_6, 0.629_978_700_5],
];
/// Oklab's `M2`: cube-rooted LMS → `[L, a, b]`.
pub const OKLAB_LMS_TO_LAB: ColorMatrix = [
    [0.210_454_255_3, 0.793_617_785_0, -0.004_072_046_8],
    [1.977_998_495_1, -2.428_592_205_0, 0.450_593_709_9],
    [0.025_904_037_1, 0.782_771_766_2, -0.808_675_766_0],
];
const OKLAB_LMS_TO_RGB: ColorMatrix = built_in(matrix::invert(&OKLAB_RGB_TO_LMS));
const OKLAB_LAB_TO_LMS: ColorMatrix = built_in(matrix::invert(&OKLAB_LMS_TO_LAB));

/// Linear Rec.709 RGB → Oklab `[L, a, b]` (`L` is 0 for black and 1 for white).
///
/// Negative LMS values keep their sign through the cube root, so out-of-gamut input round-trips.
/// With the published eight-to-ten digit matrices a grey has a chroma of about `1e-8`, not exactly 0.
pub fn rgb_to_oklab(rgb: [f64; 3]) -> [f64; 3] {
    let lms = matrix::mul_vec(&OKLAB_RGB_TO_LMS, rgb);
    matrix::mul_vec(&OKLAB_LMS_TO_LAB, [lms[0].cbrt(), lms[1].cbrt(), lms[2].cbrt()])
}

/// Oklab `[L, a, b]` → linear Rec.709 RGB (not clamped).
pub fn oklab_to_rgb(lab: [f64; 3]) -> [f64; 3] {
    let root = matrix::mul_vec(&OKLAB_LAB_TO_LMS, lab);
    let lms = [root[0] * root[0] * root[0], root[1] * root[1] * root[1], root[2] * root[2] * root[2]];
    matrix::mul_vec(&OKLAB_LMS_TO_RGB, lms)
}

/// Oklab → Oklch `[L, C, h]`.
#[inline]
pub fn oklab_to_oklch(lab: [f64; 3]) -> [f64; 3] {
    lab_to_lch(lab)
}

/// Oklch `[L, C, h]` → Oklab.
#[inline]
pub fn oklch_to_oklab(lch: [f64; 3]) -> [f64; 3] {
    lch_to_lab(lch)
}

/// Linear Rec.709 RGB → Oklch `[L, C, h]`.
#[inline]
pub fn rgb_to_oklch(rgb: [f64; 3]) -> [f64; 3] {
    lab_to_lch(rgb_to_oklab(rgb))
}

/// Oklch `[L, C, h]` → linear Rec.709 RGB (not clamped).
#[inline]
pub fn oklch_to_rgb(lch: [f64; 3]) -> [f64; 3] {
    oklab_to_rgb(lch_to_lab(lch))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::spaces::D50_XYZ;
    use humanart_core::rng::Rng;

    fn assert_near(got: [f64; 3], want: [f64; 3], tolerance: f64) {
        for i in 0..3 {
            assert!((got[i] - want[i]).abs() <= tolerance, "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn white_black_and_greys() {
        assert_near(rgb_to_lab([1.0, 1.0, 1.0]), [100.0, 0.0, 0.0], 1e-12);
        assert_near(rgb_to_lab([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0], 1e-12);
        assert_near(xyz_to_lab(D50_XYZ, D50_XYZ), [100.0, 0.0, 0.0], 1e-12);
        // 18 % grey: L* = 116 · 0.18^(1/3) − 16.
        let grey = rgb_to_lab([0.18, 0.18, 0.18]);
        assert_near(grey, [116.0 * 0.18_f64.cbrt() - 16.0, 0.0, 0.0], 1e-12);
        assert!((grey[0] - 49.496).abs() < 1e-3);
        // The linear segment: L* = κ · Y.
        assert_near(rgb_to_lab([0.001, 0.001, 0.001]), [LAB_KAPPA * 0.001, 0.0, 0.0], 1e-12);
        assert!((LAB_KAPPA * LAB_EPSILON - 8.0).abs() < 1e-13);
        // The two segments meet in value at ε.
        assert!((lab_f(LAB_EPSILON) - lab_f(LAB_EPSILON + 1e-15)).abs() < 1e-12);
    }

    #[test]
    fn srgb_primaries_match_published_lab_values() {
        // The commonly published D65 CIELAB values of the sRGB primaries (they were computed with the
        // ASTM D65 white and the rounded matrix, hence the tolerance of 0.05).
        assert_near(rgb_to_lab([1.0, 0.0, 0.0]), [53.2408, 80.0925, 67.2032], 0.05);
        assert_near(rgb_to_lab([0.0, 1.0, 0.0]), [87.7347, -86.1827, 83.1793], 0.05);
        assert_near(rgb_to_lab([0.0, 0.0, 1.0]), [32.2970, 79.1875, -107.8602], 0.05);
        assert_near(srgb8_to_lab([255, 0, 0]), rgb_to_lab([1.0, 0.0, 0.0]), 1e-12);
    }

    #[test]
    fn lab_round_trips() {
        let mut rng = Rng::new(0x1AB);
        for _ in 0..20_000 {
            // Includes out-of-gamut and negative values, which cross the linear segment.
            let rgb = [rng.range_f64(-0.5, 2.0), rng.range_f64(-0.5, 2.0), rng.range_f64(-0.5, 2.0)];
            assert_near(lab_to_rgb(rgb_to_lab(rgb)), rgb, 1e-12);
            let xyz = [rng.range_f64(0.0, 1.2), rng.range_f64(0.0, 1.2), rng.range_f64(0.0, 1.2)];
            assert_near(lab_to_xyz(xyz_to_lab(xyz, D50_XYZ), D50_XYZ), xyz, 1e-13);
            let lab = [rng.range_f64(0.0, 100.0), rng.range_f64(-90.0, 90.0), rng.range_f64(-90.0, 90.0)];
            assert_near(xyz_to_lab(lab_to_xyz(lab, D65_XYZ), D65_XYZ), lab, 1e-10);
            assert_near(lch_to_lab(lab_to_lch(lab)), lab, 1e-11);
            assert_near(lch_to_rgb(rgb_to_lch(rgb)), rgb, 1e-11);
        }
    }

    #[test]
    fn eight_bit_round_trip_through_lab() {
        let mut rng = Rng::new(0x8B17);
        for _ in 0..20_000 {
            let c = [rng.below(256) as u8, rng.below(256) as u8, rng.below(256) as u8];
            assert_eq!(lab_to_srgb8(srgb8_to_lab(c)), c);
        }
        // Out-of-gamut Lab clamps per channel.
        assert_eq!(lab_to_srgb8([120.0, 0.0, 0.0]), [255, 255, 255]);
        assert_eq!(lab_to_srgb8([-20.0, 0.0, 0.0]), [0, 0, 0]);
    }

    #[test]
    fn an_unusable_white_point_falls_back_to_d65() {
        let xyz = [0.3, 0.4, 0.5];
        let reference = xyz_to_lab(xyz, D65_XYZ);
        for bad in [[0.0, 0.0, 0.0], [-1.0, f64::NAN, f64::INFINITY]] {
            assert_eq!(xyz_to_lab(xyz, bad), reference);
            assert_eq!(lab_to_xyz(reference, bad), lab_to_xyz(reference, D65_XYZ));
        }
        // Only the bad component is replaced.
        let partly = xyz_to_lab(xyz, [0.9, 0.0, 0.8]);
        assert_eq!(partly, xyz_to_lab(xyz, [0.9, D65_XYZ[1], 0.8]));
    }

    #[test]
    fn hue_and_chroma() {
        assert_eq!(hue_degrees(0.0, 0.0), 0.0);
        assert_eq!(hue_degrees(-0.0, -0.0), 0.0);
        assert_eq!(hue_degrees(1.0, 0.0), 0.0);
        assert!((hue_degrees(0.0, 2.0) - 90.0).abs() < 1e-12);
        assert!((hue_degrees(-3.0, 0.0) - 180.0).abs() < 1e-12);
        assert!((hue_degrees(0.0, -1.0) - 270.0).abs() < 1e-12);
        assert!((hue_degrees(1.0, -1.0) - 315.0).abs() < 1e-12);
        // Just below the +a axis must not round to 360.
        let h = hue_degrees(1.0, -1e-300);
        assert!((0.0..360.0).contains(&h));
        assert!((chroma(3.0, 4.0) - 5.0).abs() < 1e-15);
        assert_eq!(lab_to_lch([50.0, 0.0, 0.0]), [50.0, 0.0, 0.0]);
        // Hue wraps; negative chroma is the opposite hue.
        assert_near(lch_to_lab([50.0, 10.0, 450.0]), [50.0, 0.0, 10.0], 1e-12);
        assert_near(lch_to_lab([50.0, -10.0, 0.0]), [50.0, -10.0, 0.0], 1e-12);
        let mut rng = Rng::new(0x40E);
        for _ in 0..5000 {
            let h = hue_degrees(rng.range_f64(-1.0, 1.0), rng.range_f64(-1.0, 1.0));
            assert!((0.0..360.0).contains(&h));
        }
    }

    #[test]
    fn oklab_matches_published_values() {
        // The table in Ottosson's article (XYZ in, three decimals out).
        for (xyz, want) in [
            ([0.950, 1.000, 1.089], [1.000, 0.000, 0.000]),
            ([1.000, 0.000, 0.000], [0.450, 1.236, -0.019]),
            ([0.000, 1.000, 0.000], [0.922, -0.671, 0.263]),
            ([0.000, 0.000, 1.000], [0.153, -1.415, -0.449]),
        ] {
            assert_near(rgb_to_oklab(xyz_to_rec709(xyz)), want, 1.5e-3);
        }
        // The sRGB primaries as published with CSS Color 4 (six decimals).
        assert_near(rgb_to_oklab([1.0, 0.0, 0.0]), [0.627_955, 0.224_863, 0.125_846], 2e-6);
        assert_near(rgb_to_oklab([0.0, 1.0, 0.0]), [0.866_440, -0.233_888, 0.179_498], 2e-6);
        assert_near(rgb_to_oklab([0.0, 0.0, 1.0]), [0.452_014, -0.032_457, -0.311_528], 2e-6);
        assert_near(rgb_to_oklab([1.0, 1.0, 1.0]), [1.0, 0.0, 0.0], 1e-7);
        assert_near(rgb_to_oklab([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0], 0.0);
        let lch = rgb_to_oklch([1.0, 0.0, 0.0]);
        assert!((lch[1] - 0.257_683).abs() < 2e-6 && (lch[2] - 29.2339).abs() < 1e-3);
    }

    #[test]
    fn oklab_round_trips() {
        let mut rng = Rng::new(0x0C1AB);
        for _ in 0..20_000 {
            let rgb = [rng.range_f64(-0.5, 2.0), rng.range_f64(-0.5, 2.0), rng.range_f64(-0.5, 2.0)];
            assert_near(oklab_to_rgb(rgb_to_oklab(rgb)), rgb, 1e-12);
            assert_near(oklch_to_rgb(rgb_to_oklch(rgb)), rgb, 1e-12);
            let lab = rgb_to_oklab(rgb);
            assert_near(oklch_to_oklab(oklab_to_oklch(lab)), lab, 1e-13);
        }
    }

    #[test]
    fn extreme_finite_input_stays_finite() {
        for v in [1e100, -1e100, 1e-300, -1e-300] {
            for f in
                [rgb_to_lab, lab_to_rgb, rgb_to_oklab, oklab_to_rgb, rgb_to_lch, lch_to_rgb, lab_to_lch, lch_to_lab]
            {
                assert!(f([v, v, v]).iter().all(|c| c.is_finite()), "{v}");
                assert!(f([v, -v, 0.0]).iter().all(|c| c.is_finite()), "{v}");
            }
        }
    }
}
