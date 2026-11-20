//! Display tone mapping and photographic exposure `[COL-02]` `[LIT-03]`.
//!
//! The display transform takes a scene-referred linear Rec.709 colour, multiplies it by the camera
//! exposure and maps it into the display range `[0, 1]` (still linear; the sRGB or Display-P3
//! encoding of [`super::transfer`] follows). These CPU versions are the references the WGSL tone
//! mappers are tested against.
//!
//! - **Khronos PBR Neutral** ([`pbr_neutral`]), the default `[COL-02]`: the tone mapper of the Khronos
//!   3D Formats Working Group, "Khronos PBR Neutral Tone Mapper" (2024), written here from its
//!   published specification. It leaves colours below 0.76 untouched apart from a small black
//!   offset, so base colours render as authored, and compresses highlights towards white.
//! - **ACES fitted** ([`aces_fitted`]), the optional filmic mapper: S. Hill's fit of the ACES 1.0
//!   reference rendering transform and sRGB output transform (RRT + ODT), as published in his
//!   `BakingLab` sample (`ACES.hlsl`, MIT licence), with its input and output matrices for linear
//!   sRGB primaries. Its constants are copied as published.
//!
//! Exposure follows the photometric camera model of S. Lagarde and C. de Rousiers, "Moving Frostbite
//! to Physically Based Rendering 3.0" (SIGGRAPH 2014 course notes), section 4.5: `EV100` from
//! aperture, shutter time and ISO, and the exposure scale `1 / (1.2 · 2^EV100)` of the
//! saturation-based sensitivity (ISO 12232, `78 / (S · q)` with `q = 0.65`).
//!
//! Out-of-range input: every mapper accepts any colour, negative components included, and returns
//! components in `[0, 1]`; the exposure helpers clamp their inputs to the ranges they document.

use super::matrix::{ColorMatrix, mul_vec, narrow, widen};

/// The tone mapper of the display transform, persisted by name in documents and commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ToneMapper {
    /// Khronos PBR Neutral, the default `[COL-02]`.
    #[default]
    PbrNeutral,
    /// Hill's ACES fitted filmic curve.
    AcesFitted,
    /// No tone curve: each component clamped to `[0, 1]`, for data and debug views.
    Clamp,
}

impl ToneMapper {
    /// Every tone mapper, in menu order.
    pub const ALL: [Self; 3] = [Self::PbrNeutral, Self::AcesFitted, Self::Clamp];

    /// The stable name used in documents and commands. Never renamed.
    pub const fn name(self) -> &'static str {
        match self {
            Self::PbrNeutral => "pbr-neutral",
            Self::AcesFitted => "aces-fitted",
            Self::Clamp => "clamp",
        }
    }

    /// The tone mapper with the stable name `name`, if there is one.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|mapper| mapper.name() == name)
    }

    /// Applies this tone mapper to `rgb · exposure`.
    pub fn apply(self, rgb: [f64; 3], exposure: f64) -> [f64; 3] {
        let exposed = [rgb[0] * exposure, rgb[1] * exposure, rgb[2] * exposure];
        match self {
            Self::PbrNeutral => pbr_neutral(exposed),
            Self::AcesFitted => aces_fitted(exposed),
            Self::Clamp => clamp_unit(exposed),
        }
    }

    /// [`Self::apply`] in `f32`.
    pub fn apply_f32(self, rgb: [f32; 3], exposure: f32) -> [f32; 3] {
        narrow(self.apply(widen(rgb), f64::from(exposure)))
    }
}

/// Each component clamped to `[0, 1]`; NaN becomes 0.
#[inline]
fn clamp_unit(c: [f64; 3]) -> [f64; 3] {
    c.map(|v| if v > 0.0 { v.min(1.0) } else { 0.0 })
}

/// Where PBR Neutral starts compressing: `0.8 − 0.04`.
const NEUTRAL_START_COMPRESSION: f64 = 0.8 - 0.04;
/// How much PBR Neutral desaturates compressed highlights.
const NEUTRAL_DESATURATION: f64 = 0.15;

/// The Khronos PBR Neutral tone mapper on a linear Rec.709 colour.
///
/// 1. `x = min(r, g, b)`; the colour is lowered by `x − 6.25x²` when `x < 0.08`, else by 0.04.
/// 2. With `p` the largest component after that: when `p < 0.76` the colour is the result.
/// 3. Otherwise, with `d = 0.24`, the new peak is `p′ = 1 − d² / (p + d − 0.76)`, the colour is
///    scaled by `p′ / p` and mixed towards the grey `p′` by `g = 1 − 1 / (0.15 (p − p′) + 1)`.
///
/// The result is clamped to `[0, 1]` (it only leaves that range for negative or non-finite input).
pub fn pbr_neutral(rgb: [f64; 3]) -> [f64; 3] {
    let x = rgb[0].min(rgb[1]).min(rgb[2]);
    let offset = if x < 0.08 { x - 6.25 * x * x } else { 0.04 };
    let color = rgb.map(|v| v - offset);
    let peak = color[0].max(color[1]).max(color[2]);
    if peak < NEUTRAL_START_COMPRESSION || !peak.is_finite() {
        return clamp_unit(color);
    }
    let d = 1.0 - NEUTRAL_START_COMPRESSION;
    let new_peak = 1.0 - d * d / (peak + d - NEUTRAL_START_COMPRESSION);
    let scale = new_peak / peak;
    let g = 1.0 - 1.0 / (NEUTRAL_DESATURATION * (peak - new_peak) + 1.0);
    clamp_unit(color.map(|v| {
        let scaled = v * scale;
        scaled + (new_peak - scaled) * g
    }))
}

/// [`pbr_neutral`] in `f32`.
#[inline]
pub fn pbr_neutral_f32(rgb: [f32; 3]) -> [f32; 3] {
    narrow(pbr_neutral(widen(rgb)))
}

/// Hill's ACES input matrix: linear sRGB → XYZ → D65-to-D60 → AP1, with the RRT saturation step.
pub const ACES_FITTED_INPUT: ColorMatrix =
    [[0.59719, 0.35458, 0.04823], [0.07600, 0.90834, 0.01566], [0.02840, 0.13383, 0.83777]];

/// Hill's ACES output matrix: the ODT saturation step, AP1 → XYZ → D60-to-D65 → linear sRGB.
pub const ACES_FITTED_OUTPUT: ColorMatrix =
    [[1.60475, -0.53108, -0.07367], [-0.10208, 1.10813, -0.00605], [-0.00327, -0.07276, 1.07602]];

/// Hill's rational fit of the RRT and ODT curve, per component:
/// `(v (v + 0.0245786) − 0.000090537) / (v (0.983729 v + 0.4329510) + 0.238081)`.
///
/// The denominator has no real root, so the fit is defined for every finite `v`.
#[inline]
pub fn aces_rrt_odt_fit(v: f64) -> f64 {
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    a / b
}

/// Hill's ACES fitted filmic tone mapper on a linear Rec.709 colour: the input matrix, the curve
/// [`aces_rrt_odt_fit`] per component, the output matrix, and a clamp to `[0, 1]`.
pub fn aces_fitted(rgb: [f64; 3]) -> [f64; 3] {
    let ap1 = mul_vec(&ACES_FITTED_INPUT, rgb);
    let curved = ap1.map(aces_rrt_odt_fit);
    clamp_unit(mul_vec(&ACES_FITTED_OUTPUT, curved))
}

/// [`aces_fitted`] in `f32`.
#[inline]
pub fn aces_fitted_f32(rgb: [f32; 3]) -> [f32; 3] {
    narrow(aces_fitted(widen(rgb)))
}

/// The smallest aperture f-number a camera accepts.
pub const MIN_APERTURE: f64 = 0.5;
/// The largest aperture f-number a camera accepts.
pub const MAX_APERTURE: f64 = 128.0;
/// The shortest shutter time a camera accepts, in seconds.
pub const MIN_SHUTTER_S: f64 = 1e-6;
/// The longest shutter time a camera accepts, in seconds.
pub const MAX_SHUTTER_S: f64 = 3600.0;
/// The lowest ISO sensitivity a camera accepts.
pub const MIN_ISO: f64 = 1.0;
/// The highest ISO sensitivity a camera accepts.
pub const MAX_ISO: f64 = 1e7;
/// The range [`exposure_from_ev100`] clamps `EV100` to.
pub const EV100_RANGE: [f64; 2] = [-32.0, 64.0];
/// The reflected-light meter calibration constant `K` (ISO 2720) used by
/// [`ev100_from_average_luminance`].
pub const METER_CALIBRATION_K: f64 = 12.5;

/// `value` clamped to `[lo, hi]`, or `fallback` when it is not finite.
#[inline]
fn usable(value: f64, lo: f64, hi: f64, fallback: f64) -> f64 {
    if value.is_finite() { value.clamp(lo, hi) } else { fallback }
}

/// The settings of a physically-based camera `[LIT-03]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicalCamera {
    /// The aperture f-number `N`.
    pub aperture: f64,
    /// The shutter time `t` in seconds.
    pub shutter_s: f64,
    /// The ISO sensitivity `S`.
    pub iso: f64,
}

impl PhysicalCamera {
    /// The "sunny 16" camera: f/16, 1/125 s, ISO 100 (`EV100 ≈ 14.97`).
    pub const SUNNY_16: Self = Self { aperture: 16.0, shutter_s: 1.0 / 125.0, iso: 100.0 };

    /// `EV100 = log2(N² / t · 100 / S)`.
    ///
    /// The settings are clamped to [`MIN_APERTURE`]..[`MAX_APERTURE`],
    /// [`MIN_SHUTTER_S`]..[`MAX_SHUTTER_S`] and [`MIN_ISO`]..[`MAX_ISO`]; a setting that is not
    /// finite takes its value from [`Self::SUNNY_16`].
    pub fn ev100(&self) -> f64 {
        let n = usable(self.aperture, MIN_APERTURE, MAX_APERTURE, Self::SUNNY_16.aperture);
        let t = usable(self.shutter_s, MIN_SHUTTER_S, MAX_SHUTTER_S, Self::SUNNY_16.shutter_s);
        let s = usable(self.iso, MIN_ISO, MAX_ISO, Self::SUNNY_16.iso);
        (n * n / t * 100.0 / s).log2()
    }

    /// The exposure scale of this camera, [`exposure_from_ev100`] of [`Self::ev100`].
    pub fn exposure(&self) -> f64 {
        exposure_from_ev100(self.ev100())
    }
}

impl Default for PhysicalCamera {
    fn default() -> Self {
        Self::SUNNY_16
    }
}

/// The scale from scene luminance (cd/m²) to the display range for an `EV100`:
/// `1 / L_max` with `L_max = 78 / (S · q) · N² / t = 1.2 · 2^EV100`.
///
/// `ev100` is clamped to [`EV100_RANGE`]; a value that is not finite is treated as 0.
pub fn exposure_from_ev100(ev100: f64) -> f64 {
    let ev = usable(ev100, EV100_RANGE[0], EV100_RANGE[1], 0.0);
    1.0 / (1.2 * ev.exp2())
}

/// The `EV100` a reflected-light meter reads for an average scene luminance `L` (cd/m²):
/// `log2(L · 100 / K)` with `K =` [`METER_CALIBRATION_K`].
///
/// The luminance is clamped to `[2⁻⁴⁰, 2⁶⁰]`; a value that is not finite is treated as the lower
/// bound.
pub fn ev100_from_average_luminance(luminance: f64) -> f64 {
    let lo = (-40.0_f64).exp2();
    let l = usable(luminance, lo, 60.0_f64.exp2(), lo);
    (l * 100.0 / METER_CALIBRATION_K).log2()
}

/// The exposure scale for an `EV100` with an exposure compensation in stops (positive brightens):
/// [`exposure_from_ev100`] of `ev100 − compensation`. A compensation that is not finite is 0.
pub fn exposure_with_compensation(ev100: f64, compensation_stops: f64) -> f64 {
    let compensation = if compensation_stops.is_finite() { compensation_stops } else { 0.0 };
    exposure_from_ev100(ev100 - compensation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    fn assert_near(got: [f64; 3], want: [f64; 3], tolerance: f64) {
        for i in 0..3 {
            assert!((got[i] - want[i]).abs() <= tolerance, "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn col_02_pbr_neutral_worked_values() {
        // Black stays black.
        assert_near(pbr_neutral([0.0; 3]), [0.0; 3], 0.0);
        // Below the compression start with min ≥ 0.08: only the 0.04 offset.
        assert_near(pbr_neutral([0.3, 0.2, 0.1]), [0.26, 0.16, 0.06], 1e-15);
        // Small minimum: the offset is x − 6.25x². x = 0.04 → offset 0.03.
        assert_near(pbr_neutral([0.5, 0.3, 0.04]), [0.47, 0.27, 0.01], 1e-15);
        // White: p = 0.96, p′ = 1 − 0.0576/0.44.
        let white = 1.0 - 0.0576 / 0.44;
        assert_near(pbr_neutral([1.0; 3]), [white; 3], 1e-12);
        // Very bright greys approach 1.
        let bright = pbr_neutral([1e6; 3]);
        assert!(bright[0] > 0.9999 && bright[0] <= 1.0);
    }

    #[test]
    fn col_02_pbr_neutral_is_continuous_and_monotonic_on_greys() {
        let mut previous = -1.0;
        for i in 0..=20_000 {
            let v = f64::from(i) * 1e-4;
            let out = pbr_neutral([v; 3]);
            assert_eq!(out[0], out[1]);
            assert_eq!(out[1], out[2]);
            assert!(out[0] >= previous - 1e-12, "{v}");
            // No jump: the step between neighbours stays below the input step.
            if previous >= 0.0 {
                assert!(out[0] - previous <= 1e-4 + 1e-12, "{v}");
            }
            previous = out[0];
        }
    }

    #[test]
    fn col_02_pbr_neutral_desaturates_highlights_but_keeps_hue_order() {
        let input = [4.0, 2.0, 1.0];
        let out = pbr_neutral(input);
        assert!(out[0] > out[1] && out[1] > out[2]);
        // Ratios move towards 1 (towards white).
        assert!(out[2] / out[0] > input[2] / input[0]);
    }

    #[test]
    fn col_02_aces_fitted_worked_values() {
        // The matrices keep greys grey (every row sums to 1 within the published rounding).
        for row in ACES_FITTED_INPUT {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 2e-5);
        }
        for row in ACES_FITTED_OUTPUT {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 2e-5);
        }
        // Middle grey 0.18: a = 0.18 · 0.2045786 − 0.000090537, b = 0.18 · 0.6100222 + 0.238081.
        let a = 0.18 * 0.2045786 - 0.000090537;
        let b = 0.18 * (0.983729 * 0.18 + 0.4329510) + 0.238081;
        let grey = pbr_like_grey(aces_fitted([0.18; 3]));
        assert!((grey - a / b).abs() < 1e-4, "{grey}");
        assert!((a / b - 0.10559).abs() < 1e-4);
        // Black maps to (nearly) black, bright greys saturate to 1.
        assert!(aces_fitted([0.0; 3]).iter().all(|&v| v < 1e-3));
        assert!(aces_fitted([100.0; 3]).iter().all(|&v| v > 0.99));
    }

    fn pbr_like_grey(c: [f64; 3]) -> f64 {
        assert!((c[0] - c[1]).abs() < 1e-4 && (c[1] - c[2]).abs() < 1e-4, "{c:?}");
        c[1]
    }

    #[test]
    fn col_02_aces_fitted_is_monotonic_on_greys() {
        let mut previous = -1.0;
        for i in 0..=10_000 {
            let v = f64::from(i) * 2e-3;
            let out = aces_fitted([v; 3])[1];
            assert!(out >= previous - 1e-12, "{v}");
            previous = out;
        }
    }

    #[test]
    fn tone_mappers_apply_exposure_and_stay_in_range() {
        let mut rng = Rng::new(0x70E);
        for _ in 0..20_000 {
            let rgb = [rng.range_f64(-2.0, 50.0), rng.range_f64(-2.0, 50.0), rng.range_f64(-2.0, 50.0)];
            let exposure = rng.range_f64(0.0, 4.0);
            for mapper in ToneMapper::ALL {
                let out = mapper.apply(rgb, exposure);
                assert!(out.iter().all(|v| (0.0..=1.0).contains(v)), "{mapper:?} {rgb:?} {out:?}");
            }
        }
        // Exposure multiplies before the curve.
        let rgb = [0.2, 0.4, 0.1];
        assert_eq!(ToneMapper::PbrNeutral.apply(rgb, 2.0), pbr_neutral([0.4, 0.8, 0.2]));
        assert_eq!(ToneMapper::AcesFitted.apply(rgb, 0.5), aces_fitted([0.1, 0.2, 0.05]));
        assert_eq!(ToneMapper::Clamp.apply(rgb, 4.0), [0.8, 1.0, 0.4]);
        assert_eq!(ToneMapper::default(), ToneMapper::PbrNeutral);
        // f32 versions follow the f64 ones.
        let f = ToneMapper::PbrNeutral.apply_f32([0.2, 0.4, 0.1], 2.0);
        assert_eq!(f, narrow(pbr_neutral([0.4, 0.8, 0.2])));
        assert_eq!(pbr_neutral_f32([1.0; 3]), narrow(pbr_neutral([1.0; 3])));
        assert_eq!(aces_fitted_f32([1.0; 3]), narrow(aces_fitted([1.0; 3])));
    }

    #[test]
    fn tone_mapper_names_are_stable_and_round_trip() {
        assert_eq!(ToneMapper::PbrNeutral.name(), "pbr-neutral");
        assert_eq!(ToneMapper::AcesFitted.name(), "aces-fitted");
        assert_eq!(ToneMapper::Clamp.name(), "clamp");
        for mapper in ToneMapper::ALL {
            assert_eq!(ToneMapper::from_name(mapper.name()), Some(mapper));
        }
        assert_eq!(ToneMapper::from_name("agx"), None);
        assert_eq!(ToneMapper::from_name(""), None);
    }

    #[test]
    fn lit_03_ev100_and_exposure() {
        // f/1, 1 s, ISO 100 is EV100 0.
        let unit = PhysicalCamera { aperture: 1.0, shutter_s: 1.0, iso: 100.0 };
        assert!(unit.ev100().abs() < 1e-12);
        assert!((unit.exposure() - 1.0 / 1.2).abs() < 1e-12);
        // Sunny 16: log2(256 · 125) = 14.9658.
        assert!((PhysicalCamera::SUNNY_16.ev100() - (256.0_f64 * 125.0).log2()).abs() < 1e-12);
        // Doubling ISO lowers EV100 by one stop and doubles the exposure.
        let iso200 = PhysicalCamera { iso: 200.0, ..PhysicalCamera::SUNNY_16 };
        assert!((PhysicalCamera::SUNNY_16.ev100() - iso200.ev100() - 1.0).abs() < 1e-12);
        assert!((iso200.exposure() / PhysicalCamera::SUNNY_16.exposure() - 2.0).abs() < 1e-12);
        // Meter: L = K / 100 reads EV100 0; the exposure scale is then 1/1.2.
        assert!(ev100_from_average_luminance(0.125).abs() < 1e-12);
        assert!((ev100_from_average_luminance(0.25) - 1.0).abs() < 1e-12);
        // Compensation: +1 stop doubles the exposure.
        assert!((exposure_with_compensation(10.0, 1.0) / exposure_from_ev100(10.0) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn exposure_inputs_clamp_and_non_finite_input_falls_back() {
        let bad = PhysicalCamera { aperture: f64::NAN, shutter_s: f64::INFINITY, iso: f64::NEG_INFINITY };
        assert_eq!(bad.ev100(), PhysicalCamera::SUNNY_16.ev100());
        let tiny = PhysicalCamera { aperture: 0.0, shutter_s: 0.0, iso: 0.0 };
        assert!(tiny.ev100().is_finite() && tiny.exposure().is_finite());
        assert_eq!(exposure_from_ev100(f64::NAN), exposure_from_ev100(0.0));
        assert_eq!(exposure_from_ev100(1e9), exposure_from_ev100(EV100_RANGE[1]));
        assert!(exposure_from_ev100(-1e9).is_finite());
        assert!(ev100_from_average_luminance(0.0).is_finite());
        assert!(ev100_from_average_luminance(f64::NAN).is_finite());
        assert!(ev100_from_average_luminance(1e300).is_finite());
        assert_eq!(exposure_with_compensation(3.0, f64::NAN), exposure_from_ev100(3.0));
    }

    #[test]
    fn extreme_and_non_finite_colours_never_panic() {
        for v in [1e100, -1e100, 1e-300, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for mapper in ToneMapper::ALL {
                let out = mapper.apply([v, 0.5, -v], 1.0);
                if v.is_finite() {
                    assert!(out.iter().all(|c| (0.0..=1.0).contains(c)), "{mapper:?} {v} {out:?}");
                }
            }
        }
        for v in [1e30_f32, -1e30, f32::NAN] {
            let _ = pbr_neutral_f32([v; 3]);
            let _ = aces_fitted_f32([v; 3]);
        }
    }
}
