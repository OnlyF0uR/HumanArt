//! Colour temperature: the Planckian and CIE daylight loci, temperature → linear RGB, and the
//! correlated colour temperature of a chromaticity `[COL-03]` `[LIT-01]`.
//!
//! A light in the studio rig has a colour temperature; these functions turn it into the linear
//! Rec.709 colour that multiplies the light's photometric intensity. The colours are normalised to a
//! luminance of 1, so changing a light's temperature does not change its illuminance.
//!
//! The loci are the published closed-form approximations, with their published constants:
//!
//! - **Planckian locus, 1667 K – 25 000 K**: the cubic-spline fit in CIE xy of Kang, Moon, Hong, Lee,
//!   Cho and Kim, "Design of Advanced Color Temperature Control System for HDTV Applications",
//!   Journal of the Korean Physical Society 41(6), 2002 ([`planckian_xy_kang`]).
//! - **Planckian locus, 1000 K – 15 000 K**: the rational fit in CIE 1960 uv of M. Krystek, "An
//!   algorithm to calculate correlated colour temperature", Color Research & Application 10(1), 1985
//!   ([`planckian_xy_krystek`]), stated by its author to be within `8·10⁻⁵` in u and `9·10⁻⁵` in v.
//! - **Daylight locus, 4000 K – 25 000 K**: the CIE formula for the chromaticity of a D illuminant
//!   (CIE 15, Colorimetry; Judd, MacAdam & Wyszecki 1964) ([`daylight_xy`]).
//! - **Correlated colour temperature**: C. S. McCamy, "Correlated color temperature as an explicit
//!   function of chromaticity coordinates", Color Research & Application 17(2), 1992
//!   ([`cct_mccamy`]).
//!
//! [`planckian_xy`] joins the two Planckian fits: Krystek's below 1667 K, Kang's from 2000 K, and a
//! linear cross-fade in between, where both are valid and (as the tests check) agree to better than
//! `10⁻³`. Every function clamps its temperature to the range it documents; a temperature that is not
//! finite is treated as 6500 K.

use super::matrix::narrow;
use super::spaces::{luminance, xy_to_xyz, xyz_to_rec709};

/// The lowest temperature of [`planckian_xy`] and [`blackbody_rgb`], in kelvin.
pub const MIN_KELVIN: f64 = 1000.0;
/// The highest temperature of every locus in this module, in kelvin.
pub const MAX_KELVIN: f64 = 25_000.0;
/// The lowest temperature of the CIE daylight locus, in kelvin.
pub const MIN_DAYLIGHT_KELVIN: f64 = 4000.0;
/// The lowest temperature of Kang et al.'s Planckian fit, in kelvin.
pub const MIN_KANG_KELVIN: f64 = 1667.0;
/// The highest temperature of Krystek's Planckian fit, in kelvin.
pub const MAX_KRYSTEK_KELVIN: f64 = 15_000.0;

/// The upper end of the cross-fade from Krystek's fit to Kang's in [`planckian_xy`], in kelvin.
const BLEND_END_KELVIN: f64 = 2000.0;

/// `kelvin` clamped to `[lo, hi]`; 6500 K when it is not finite.
#[inline]
fn usable_kelvin(kelvin: f64, lo: f64, hi: f64) -> f64 {
    if kelvin.is_finite() { kelvin.clamp(lo, hi) } else { 6500.0_f64.clamp(lo, hi) }
}

/// The CIE xy chromaticity of a black body by the fit of Kang et al. (2002); the temperature is
/// clamped to 1667 K – 25 000 K.
///
/// With `t = 1/T`:
///
/// - `x = −0.2661239·10⁹t³ − 0.2343589·10⁶t² + 0.8776956·10³t + 0.179910` for `T ≤ 4000 K`,
/// - `x = −3.0258469·10⁹t³ + 2.1070379·10⁶t² + 0.2226347·10³t + 0.240390` above;
/// - `y = −1.1063814x³ − 1.34811020x² + 2.18555832x − 0.20219683` for `T ≤ 2222 K`,
/// - `y = −0.9549476x³ − 1.37418593x² + 2.09137015x − 0.16748867` for `2222 K < T ≤ 4000 K`,
/// - `y = 3.0817580x³ − 5.87338670x² + 3.75112997x − 0.37001483` above.
pub fn planckian_xy_kang(kelvin: f64) -> [f64; 2] {
    let k = usable_kelvin(kelvin, MIN_KANG_KELVIN, MAX_KELVIN);
    let t = 1.0 / k;
    let x = if k <= 4000.0 {
        ((-0.266_123_9e9 * t - 0.234_358_9e6) * t + 0.877_695_6e3) * t + 0.179_910
    } else {
        ((-3.025_846_9e9 * t + 2.107_037_9e6) * t + 0.222_634_7e3) * t + 0.240_390
    };
    let y = if k <= 2222.0 {
        ((-1.106_381_4 * x - 1.348_110_20) * x + 2.185_558_32) * x - 0.202_196_83
    } else if k <= 4000.0 {
        ((-0.954_947_6 * x - 1.374_185_93) * x + 2.091_370_15) * x - 0.167_488_67
    } else {
        ((3.081_758_0 * x - 5.873_386_70) * x + 3.751_129_97) * x - 0.370_014_83
    };
    [x, y]
}

/// The CIE xy chromaticity of a black body by the fit of Krystek (1985); the temperature is clamped
/// to 1000 K – 15 000 K.
///
/// In the CIE 1960 UCS:
///
/// - `u = (0.860117757 + 1.54118254·10⁻⁴T + 1.28641212·10⁻⁷T²) / (1 + 8.42420235·10⁻⁴T + 7.08145163·10⁻⁷T²)`
/// - `v = (0.317398726 + 4.22806245·10⁻⁵T + 4.20481691·10⁻⁸T²) / (1 − 2.89741816·10⁻⁵T + 1.61456053·10⁻⁷T²)`
///
/// and then `x = 3u / (2u − 8v + 4)`, `y = 2v / (2u − 8v + 4)`.
pub fn planckian_xy_krystek(kelvin: f64) -> [f64; 2] {
    let t = usable_kelvin(kelvin, MIN_KELVIN, MAX_KRYSTEK_KELVIN);
    let u = (0.860_117_757 + (1.541_182_54e-4 + 1.286_412_12e-7 * t) * t)
        / (1.0 + (8.424_202_35e-4 + 7.081_451_63e-7 * t) * t);
    let v = (0.317_398_726 + (4.228_062_45e-5 + 4.204_816_91e-8 * t) * t)
        / (1.0 + (-2.897_418_16e-5 + 1.614_560_53e-7 * t) * t);
    // Positive over the whole clamped range (it is 1.5 or more).
    let d = 2.0 * u - 8.0 * v + 4.0;
    [3.0 * u / d, 2.0 * v / d]
}

/// The CIE xy chromaticity of a black body at `kelvin`, clamped to 1000 K – 25 000 K
/// ([`MIN_KELVIN`], [`MAX_KELVIN`]): Krystek's fit up to 1667 K, Kang's from 2000 K, a linear
/// cross-fade between them in between. The result is continuous in the temperature except for the
/// steps of a few `10⁻⁵` at the 2222 K and 4000 K joints of Kang's published pieces.
pub fn planckian_xy(kelvin: f64) -> [f64; 2] {
    let k = usable_kelvin(kelvin, MIN_KELVIN, MAX_KELVIN);
    if k <= MIN_KANG_KELVIN {
        return planckian_xy_krystek(k);
    }
    let kang = planckian_xy_kang(k);
    if k >= BLEND_END_KELVIN {
        return kang;
    }
    let krystek = planckian_xy_krystek(k);
    let w = (k - MIN_KANG_KELVIN) / (BLEND_END_KELVIN - MIN_KANG_KELVIN);
    [krystek[0] + (kang[0] - krystek[0]) * w, krystek[1] + (kang[1] - krystek[1]) * w]
}

/// The CIE xy chromaticity of the CIE daylight illuminant with the correlated colour temperature
/// `kelvin`, clamped to 4000 K – 25 000 K.
///
/// With `t = 1/T`:
///
/// - `x = −4.6070·10⁹t³ + 2.9678·10⁶t² + 0.09911·10³t + 0.244063` for `T ≤ 7000 K`,
/// - `x = −2.0064·10⁹t³ + 1.9018·10⁶t² + 0.24748·10³t + 0.237040` above,
/// - `y = −3.000x² + 2.870x − 0.275`.
///
/// D65 is the daylight at 6504 K and D50 the one at 5003 K (the nominal temperatures times
/// `1.4388/1.4380`, from the revision of the radiation constant `c₂`).
pub fn daylight_xy(kelvin: f64) -> [f64; 2] {
    let k = usable_kelvin(kelvin, MIN_DAYLIGHT_KELVIN, MAX_KELVIN);
    let t = 1.0 / k;
    let x = if k <= 7000.0 {
        ((-4.6070e9 * t + 2.9678e6) * t + 0.099_11e3) * t + 0.244_063
    } else {
        ((-2.0064e9 * t + 1.9018e6) * t + 0.247_48e3) * t + 0.237_040
    };
    [x, (-3.000 * x + 2.870) * x - 0.275]
}

/// The linear Rec.709 colour of the chromaticity `xy` at a luminance of 1.
///
/// A chromaticity outside the Rec.709 gamut has negative components there: they are clamped to 0 and
/// the luminance is restored, which moves the colour to the gamut boundary along a line of constant
/// hue of the RGB cube. A chromaticity that is not a colour (`y ≤ 0`, not finite) gives white.
pub fn chromaticity_to_rgb(xy: [f64; 2]) -> [f64; 3] {
    let rgb = xyz_to_rec709(xy_to_xyz(xy));
    let clamped = [rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)];
    let y = luminance(clamped);
    let out = [clamped[0] / y, clamped[1] / y, clamped[2] / y];
    if y > 0.0 && out.iter().all(|v| v.is_finite()) { out } else { [1.0, 1.0, 1.0] }
}

/// The linear Rec.709 colour, at a luminance of 1, of a black body at `kelvin` (clamped to
/// 1000 K – 25 000 K): the colour of a light with that colour temperature `[LIT-01]`.
///
/// 6504 K is close to, but not exactly, white `(1, 1, 1)`: the working space's white is D65, a
/// daylight, which lies slightly to the green side of the Planckian locus. Below about 1900 K the
/// locus leaves the Rec.709 gamut and the blue component is 0 (see [`chromaticity_to_rgb`]).
#[inline]
pub fn blackbody_rgb(kelvin: f64) -> [f64; 3] {
    chromaticity_to_rgb(planckian_xy(kelvin))
}

/// [`blackbody_rgb`] in `f32`, for the light uniforms.
#[inline]
pub fn blackbody_rgb_f32(kelvin: f32) -> [f32; 3] {
    narrow(blackbody_rgb(f64::from(kelvin)))
}

/// The linear Rec.709 colour, at a luminance of 1, of CIE daylight at `kelvin` (clamped to
/// 4000 K – 25 000 K). 6504 K gives white to within `2·10⁻³`.
#[inline]
pub fn daylight_rgb(kelvin: f64) -> [f64; 3] {
    chromaticity_to_rgb(daylight_xy(kelvin))
}

/// The correlated colour temperature of the chromaticity `xy` by McCamy's formula:
/// `CCT = 449n³ + 3525n² + 6823.3n + 5520.33` with `n = (x − 0.3320) / (0.1858 − y)`.
///
/// The formula is meant for chromaticities near the Planckian or daylight locus between about
/// 2000 K and 12 500 K; on the loci of this module it is within 1 % of the temperature from 2200 K
/// to 8000 K (tested). The formula assigns a temperature to every chromaticity, however saturated: it
/// does not say how far from the locus the colour is. `None` when the result is not finite or lies
/// outside 1000 K – 25 000 K, where it means nothing.
pub fn cct_mccamy(xy: [f64; 2]) -> Option<f64> {
    let n = (xy[0] - 0.3320) / (0.1858 - xy[1]);
    let cct = ((449.0 * n + 3525.0) * n + 6823.3) * n + 5520.33;
    (cct.is_finite() && (MIN_KELVIN..=MAX_KELVIN).contains(&cct)).then_some(cct)
}

/// The correlated colour temperature of a linear Rec.709 colour ([`cct_mccamy`] of its
/// chromaticity). `None` for black (no positive luminance) and where [`cct_mccamy`] gives none.
pub fn cct_of_rgb(rgb: [f64; 3]) -> Option<f64> {
    if luminance(rgb) > 0.0 { cct_mccamy(super::spaces::rec709_to_xy(rgb)) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::spaces::{D50_XY, D65_XY};

    fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
        (a[0] - b[0]).hypot(a[1] - b[1])
    }

    #[test]
    fn planckian_locus_matches_cie_illuminant_a() {
        // CIE standard illuminant A is the black body at 2856 K (on the current temperature scale);
        // its chromaticity for the 2° observer is (0.44757, 0.40745) (CIE 15).
        let a = [0.447_57, 0.407_45];
        assert!(distance(planckian_xy_kang(2856.0), a) < 6e-4);
        assert!(distance(planckian_xy_krystek(2856.0), a) < 4e-4);
        assert!(distance(planckian_xy(2856.0), a) < 6e-4);
        // Lies on the red side of D65 at 6500 K, and slightly magenta of the daylight locus.
        let p = planckian_xy(6500.0);
        assert!((p[0] - 0.3135).abs() < 3e-4 && (p[1] - 0.3237).abs() < 3e-4, "{p:?}");
    }

    #[test]
    fn the_two_published_planckian_fits_agree_where_both_are_valid() {
        let mut worst: f64 = 0.0;
        let mut k = MIN_KANG_KELVIN;
        while k <= MAX_KRYSTEK_KELVIN {
            worst = worst.max(distance(planckian_xy_kang(k), planckian_xy_krystek(k)));
            k += 1.0;
        }
        // Measured: 7.7e-4, at about 2600 K.
        assert!(worst < 1e-3, "{worst}");
    }

    #[test]
    fn planckian_locus_is_continuous_and_monotonic() {
        let mut last = planckian_xy(MIN_KELVIN);
        let mut k = MIN_KELVIN + 1.0;
        while k <= MAX_KELVIN {
            let now = planckian_xy(k);
            // x falls steadily with temperature; no step beyond the 7e-5 of Kang's 4000 K joint.
            assert!(now[0] < last[0] + 8e-5, "{k}");
            assert!(distance(now, last) < 6e-4, "{k}: {}", distance(now, last));
            last = now;
            k += 1.0;
        }
        assert!(planckian_xy(1000.0)[0] > 0.64 && planckian_xy(25_000.0)[0] < 0.26);
        // The pieces of the joined locus.
        assert_eq!(planckian_xy(1200.0), planckian_xy_krystek(1200.0));
        assert_eq!(planckian_xy(1667.0), planckian_xy_krystek(1667.0));
        assert_eq!(planckian_xy(2000.0), planckian_xy_kang(2000.0));
        assert_eq!(planckian_xy(9000.0), planckian_xy_kang(9000.0));
        let mid = planckian_xy(1833.5);
        let (a, b) = (planckian_xy_krystek(1833.5), planckian_xy_kang(1833.5));
        assert!((mid[0] - f64::midpoint(a[0], b[0])).abs() < 1e-12);
        assert!((mid[1] - f64::midpoint(a[1], b[1])).abs() < 1e-12);
    }

    #[test]
    fn daylight_locus_matches_the_d_illuminants() {
        // CIE 15: D50 (0.34567, 0.35850), D55 (0.33242, 0.34743), D65 (0.31271, 0.32902),
        // D75 (0.29902, 0.31485). The formula reproduces the tabulated values to about 1e-4.
        for (kelvin, xy) in [
            (5003.0, [0.345_67, 0.358_50]),
            (5503.0, [0.332_42, 0.347_43]),
            (6504.0, [0.312_71, 0.329_02]),
            (7504.0, [0.299_02, 0.314_85]),
        ] {
            assert!(distance(daylight_xy(kelvin), xy) < 2e-4, "{kelvin}: {:?}", daylight_xy(kelvin));
        }
        assert!(distance(daylight_xy(6504.0), D65_XY) < 2e-4);
        assert!(distance(daylight_xy(5003.0), D50_XY) < 2e-4);
        // The two pieces meet at 7000 K.
        assert!(distance(daylight_xy(7000.0), daylight_xy(7000.0 + 1e-6)) < 1e-5);
        // Daylight is on the green side of the black body of the same temperature.
        for kelvin in [4000.0, 5000.0, 6500.0, 10_000.0, 20_000.0] {
            assert!(daylight_xy(kelvin)[1] > planckian_xy(kelvin)[1]);
        }
    }

    #[test]
    fn temperatures_clamp_and_non_finite_ones_fall_back() {
        assert_eq!(planckian_xy(10.0), planckian_xy(MIN_KELVIN));
        assert_eq!(planckian_xy(-5000.0), planckian_xy(MIN_KELVIN));
        assert_eq!(planckian_xy(1e9), planckian_xy(MAX_KELVIN));
        assert_eq!(planckian_xy(f64::NAN), planckian_xy(6500.0));
        assert_eq!(planckian_xy(f64::INFINITY), planckian_xy(6500.0));
        assert_eq!(planckian_xy_kang(100.0), planckian_xy_kang(MIN_KANG_KELVIN));
        assert_eq!(planckian_xy_krystek(1e9), planckian_xy_krystek(MAX_KRYSTEK_KELVIN));
        assert_eq!(daylight_xy(1000.0), daylight_xy(MIN_DAYLIGHT_KELVIN));
        assert_eq!(daylight_xy(f64::NAN), daylight_xy(6500.0));
        assert_eq!(blackbody_rgb(f64::NEG_INFINITY), blackbody_rgb(6500.0));
    }

    #[test]
    fn lit_01_light_colours_have_unit_luminance_and_the_expected_tint() {
        let mut k = MIN_KELVIN;
        while k <= MAX_KELVIN {
            for rgb in [blackbody_rgb(k), daylight_rgb(k)] {
                assert!((luminance(rgb) - 1.0).abs() < 1e-12, "{k}");
                assert!(rgb.iter().all(|v| v.is_finite() && *v >= 0.0), "{k}: {rgb:?}");
            }
            k += 50.0;
        }
        // Warm lights are red-heavy, cool lights blue-heavy.
        let warm = blackbody_rgb(2700.0);
        assert!(warm[0] > warm[1] && warm[1] > warm[2], "{warm:?}");
        let cool = blackbody_rgb(12_000.0);
        assert!(cool[2] > cool[1] && cool[1] > cool[0], "{cool:?}");
        // Red rises and blue falls monotonically as the light gets warmer.
        let mut last = blackbody_rgb(MAX_KELVIN);
        let mut k = MAX_KELVIN - 100.0;
        while k >= MIN_KELVIN {
            let now = blackbody_rgb(k);
            assert!(now[0] >= last[0] - 1e-3 && now[2] <= last[2] + 1e-3, "{k}");
            last = now;
            k -= 100.0;
        }
        // Out of gamut below about 1900 K: blue is clamped to 0.
        assert_eq!(blackbody_rgb(1500.0)[2], 0.0);
        assert!(blackbody_rgb(2500.0)[2] > 0.0);
        // D65 daylight is the working white.
        let white = daylight_rgb(6504.0);
        assert!(white.iter().all(|v| (v - 1.0).abs() < 2e-3), "{white:?}");
        assert_eq!(chromaticity_to_rgb(D65_XY).map(|v| (v * 1e12).round() / 1e12), [1.0, 1.0, 1.0]);
        // The f32 version is the f64 one, narrowed.
        assert_eq!(blackbody_rgb_f32(3200.0), narrow(blackbody_rgb(3200.0)));
        // Not-a-colour chromaticities give white.
        for bad in [[0.3, 0.0], [0.3, -1.0], [f64::NAN, 0.3], [0.3, f64::INFINITY]] {
            assert_eq!(chromaticity_to_rgb(bad), [1.0, 1.0, 1.0]);
        }
    }

    #[test]
    fn mccamy_recovers_the_temperature_of_both_loci() {
        let mut worst: f64 = 0.0;
        let mut k = 2200.0;
        while k <= 8000.0 {
            let cct = cct_mccamy(planckian_xy(k)).unwrap();
            worst = worst.max((cct - k).abs() / k);
            if k >= MIN_DAYLIGHT_KELVIN {
                let cct = cct_mccamy(daylight_xy(k)).unwrap();
                worst = worst.max((cct - k).abs() / k);
            }
            k += 10.0;
        }
        assert!(worst < 0.01, "{worst}");
        assert!((cct_mccamy(D65_XY).unwrap() - 6504.0).abs() < 5.0);
        assert!((cct_of_rgb([1.0, 1.0, 1.0]).unwrap() - 6504.0).abs() < 5.0);
        assert!((cct_of_rgb(blackbody_rgb(3200.0)).unwrap() - 3200.0).abs() < 32.0);
        // No answer where the formula means nothing.
        assert_eq!(cct_mccamy([0.3320, 0.1858]), None);
        assert_eq!(cct_mccamy([0.2, 0.2]), None);
        assert_eq!(cct_mccamy([f64::NAN, 0.3]), None);
        assert_eq!(cct_of_rgb([0.0, 0.0, 0.0]), None);
        assert_eq!(cct_of_rgb([-1.0, -1.0, -1.0]), None);
        // Any chromaticity lies on one of the formula's isotherms: a saturated colour gets a number
        // too, and only the caller knows whether the colour is near enough to white for it to matter.
        assert!(cct_of_rgb([1.0, 0.5, 0.0]).is_some());
    }
}
