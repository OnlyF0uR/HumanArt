//! HSV and HSL, the cylindrical RGB models of the colour picker `[COL-03]`.
//!
//! Both are plain rearrangements of an RGB triple (A. R. Smith, "Color Gamut Transform Pairs", 1978;
//! Joblove & Greenberg 1978), so they work on whatever RGB they are given: the picker feeds them
//! display-encoded values for its perceptual sliders and linear values for its linear ones. A triple
//! is `[h, s, v]` or `[h, s, l]` with the hue in degrees in `[0, 360)` (red 0, green 120, blue 240)
//! and saturation, value and lightness in `[0, 1]`.
//!
//! Out-of-range input:
//!
//! - RGB → HSV clamps negative components to 0 and keeps values above 1, so an HDR colour keeps its
//!   hue and saturation and comes back with `v > 1`.
//! - HSV → RGB wraps any finite hue, clamps the saturation to `[0, 1]` and the value to `≥ 0`.
//! - HSL is defined on the unit cube only: RGB → HSL clamps every component to `[0, 1]`, HSL → RGB
//!   clamps saturation and lightness to `[0, 1]`.
//! - The hue of a grey, and a hue that is not finite, is 0. The saturation of black is 0, and so is
//!   the HSL saturation of white.

use super::matrix::{narrow, widen};

/// The largest and smallest of three components.
#[inline]
fn extent(r: f64, g: f64, b: f64) -> (f64, f64) {
    (r.max(g).max(b), r.min(g).min(b))
}

/// The hue in degrees of a colour whose largest component is `max` and whose spread `delta` is
/// positive.
#[inline]
fn hue_of(r: f64, g: f64, b: f64, max: f64, delta: f64) -> f64 {
    let sector = if max == r {
        ((g - b) / delta).rem_euclid(6.0)
    } else if max == g {
        (b - r) / delta + 2.0
    } else {
        (r - g) / delta + 4.0
    };
    let h = sector * 60.0;
    // A tiny negative sector wraps to exactly 6, and non-finite input makes NaN.
    if (0.0..360.0).contains(&h) { h } else { 0.0 }
}

/// A finite hue wrapped to `[0, 6)` sextants; 0 for a hue that is not finite.
#[inline]
fn sextant(hue_degrees: f64) -> f64 {
    if !hue_degrees.is_finite() {
        return 0.0;
    }
    let s = hue_degrees.rem_euclid(360.0) / 60.0;
    if s < 6.0 { s } else { 0.0 }
}

/// The RGB triple of the hue sextant `s` in `[0, 6)` with chroma `c`, before the lightness offset.
#[inline]
fn hue_to_rgb(s: f64, c: f64) -> [f64; 3] {
    let x = c * (1.0 - ((s % 2.0) - 1.0).abs());
    if s < 1.0 {
        [c, x, 0.0]
    } else if s < 2.0 {
        [x, c, 0.0]
    } else if s < 3.0 {
        [0.0, c, x]
    } else if s < 4.0 {
        [0.0, x, c]
    } else if s < 5.0 {
        [x, 0.0, c]
    } else {
        [c, 0.0, x]
    }
}

/// RGB → HSV: `v = max`, `s = (max − min) / max`, hue by the sextant of the largest component.
pub fn rgb_to_hsv(rgb: [f64; 3]) -> [f64; 3] {
    let (r, g, b) = (rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0));
    let (max, min) = extent(r, g, b);
    let delta = max - min;
    if !(max > 0.0 && delta > 0.0) {
        return [0.0, 0.0, max];
    }
    let s = (delta / max).clamp(0.0, 1.0);
    [hue_of(r, g, b, max, delta), s, max]
}

/// HSV → RGB.
pub fn hsv_to_rgb(hsv: [f64; 3]) -> [f64; 3] {
    let s = hsv[1].clamp(0.0, 1.0);
    let v = hsv[2].max(0.0);
    let c = v * s;
    let m = v - c;
    let [r, g, b] = hue_to_rgb(sextant(hsv[0]), c);
    [r + m, g + m, b + m]
}

/// RGB → HSL: `l = (max + min) / 2`, `s = (max − min) / (1 − |2l − 1|)`.
pub fn rgb_to_hsl(rgb: [f64; 3]) -> [f64; 3] {
    let (r, g, b) = (rgb[0].clamp(0.0, 1.0), rgb[1].clamp(0.0, 1.0), rgb[2].clamp(0.0, 1.0));
    let (max, min) = extent(r, g, b);
    let delta = max - min;
    let l = f64::midpoint(max, min);
    let range = 1.0 - (2.0 * l - 1.0).abs();
    if !(delta > 0.0 && range > 0.0) {
        return [0.0, 0.0, l];
    }
    [hue_of(r, g, b, max, delta), (delta / range).clamp(0.0, 1.0), l]
}

/// HSL → RGB.
pub fn hsl_to_rgb(hsl: [f64; 3]) -> [f64; 3] {
    let s = hsl[1].clamp(0.0, 1.0);
    let l = hsl[2].clamp(0.0, 1.0);
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let m = l - 0.5 * c;
    let [r, g, b] = hue_to_rgb(sextant(hsl[0]), c);
    [r + m, g + m, b + m]
}

/// HSV → HSL (same hue; for values in the unit cube).
pub fn hsv_to_hsl(hsv: [f64; 3]) -> [f64; 3] {
    rgb_to_hsl(hsv_to_rgb(hsv))
}

/// HSL → HSV (same hue).
pub fn hsl_to_hsv(hsl: [f64; 3]) -> [f64; 3] {
    rgb_to_hsv(hsl_to_rgb(hsl))
}

/// [`rgb_to_hsv`] in `f32`, for the UI.
#[inline]
pub fn rgb_to_hsv_f32(rgb: [f32; 3]) -> [f32; 3] {
    narrow_hue_first(rgb_to_hsv(widen(rgb)))
}

/// [`hsv_to_rgb`] in `f32`, for the UI.
#[inline]
pub fn hsv_to_rgb_f32(hsv: [f32; 3]) -> [f32; 3] {
    narrow(hsv_to_rgb(widen(hsv)))
}

/// [`rgb_to_hsl`] in `f32`, for the UI.
#[inline]
pub fn rgb_to_hsl_f32(rgb: [f32; 3]) -> [f32; 3] {
    narrow_hue_first(rgb_to_hsl(widen(rgb)))
}

/// [`hsl_to_rgb`] in `f32`, for the UI.
#[inline]
pub fn hsl_to_rgb_f32(hsl: [f32; 3]) -> [f32; 3] {
    narrow(hsl_to_rgb(widen(hsl)))
}

/// Narrows `[h, s, x]` to `f32`; a hue that rounds up to 360 in `f32` becomes 0.
#[inline]
fn narrow_hue_first(v: [f64; 3]) -> [f32; 3] {
    let out = narrow(v);
    if out[0] < 360.0 { out } else { [0.0, out[1], out[2]] }
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
    fn primaries_secondaries_and_greys() {
        let cases = [
            ([1.0, 0.0, 0.0], 0.0),
            ([1.0, 1.0, 0.0], 60.0),
            ([0.0, 1.0, 0.0], 120.0),
            ([0.0, 1.0, 1.0], 180.0),
            ([0.0, 0.0, 1.0], 240.0),
            ([1.0, 0.0, 1.0], 300.0),
        ];
        for (rgb, hue) in cases {
            assert_near(rgb_to_hsv(rgb), [hue, 1.0, 1.0], 1e-12);
            assert_near(rgb_to_hsl(rgb), [hue, 1.0, 0.5], 1e-12);
            assert_near(hsv_to_rgb([hue, 1.0, 1.0]), rgb, 1e-12);
            assert_near(hsl_to_rgb([hue, 1.0, 0.5]), rgb, 1e-12);
        }
        assert_eq!(rgb_to_hsv([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
        assert_eq!(rgb_to_hsv([0.4, 0.4, 0.4]), [0.0, 0.0, 0.4]);
        assert_eq!(rgb_to_hsl([0.4, 0.4, 0.4]), [0.0, 0.0, 0.4]);
        assert_eq!(rgb_to_hsl([1.0, 1.0, 1.0]), [0.0, 0.0, 1.0]);
        assert_eq!(hsv_to_rgb([123.0, 0.0, 0.7]), [0.7, 0.7, 0.7]);
        assert_eq!(hsl_to_rgb([123.0, 0.0, 0.7]), [0.7, 0.7, 0.7]);
    }

    #[test]
    fn worked_examples() {
        // RGB (0.75, 0.25, 0.5): max 0.75, min 0.25, delta 0.5, hue 60·((0.25 − 0.5)/0.5 mod 6) = 330.
        assert_near(rgb_to_hsv([0.75, 0.25, 0.5]), [330.0, 2.0 / 3.0, 0.75], 1e-12);
        assert_near(rgb_to_hsl([0.75, 0.25, 0.5]), [330.0, 0.5, 0.5], 1e-12);
        // RGB (0.2, 0.6, 0.4): hue 60·((0.4 − 0.2)/0.4 + 2) = 150.
        assert_near(rgb_to_hsv([0.2, 0.6, 0.4]), [150.0, 2.0 / 3.0, 0.6], 1e-12);
        assert_near(rgb_to_hsl([0.2, 0.6, 0.4]), [150.0, 0.5, 0.4], 1e-12);
        assert_near(hsv_to_rgb([210.0, 0.5, 0.8]), [0.4, 0.6, 0.8], 1e-12);
        assert_near(hsl_to_rgb([210.0, 0.5, 0.6]), [0.4, 0.6, 0.8], 1e-12);
    }

    #[test]
    fn round_trips() {
        let mut rng = Rng::new(0x45B);
        for _ in 0..20_000 {
            let rgb = [rng.f64(), rng.f64(), rng.f64()];
            assert_near(hsv_to_rgb(rgb_to_hsv(rgb)), rgb, 1e-12);
            assert_near(hsl_to_rgb(rgb_to_hsl(rgb)), rgb, 1e-12);
            let hsv = rgb_to_hsv(rgb);
            assert!((0.0..360.0).contains(&hsv[0]) && (0.0..=1.0).contains(&hsv[1]));
            let hsl = rgb_to_hsl(rgb);
            assert!((0.0..360.0).contains(&hsl[0]) && (0.0..=1.0).contains(&hsl[1]));
            assert_near(hsv_to_rgb(hsl_to_hsv(hsv_to_hsl(hsv))), rgb, 1e-12);
            // HDR values keep hue and saturation through HSV.
            let hdr = [rgb[0] * 16.0, rgb[1] * 16.0, rgb[2] * 16.0];
            assert_near(hsv_to_rgb(rgb_to_hsv(hdr)), hdr, 1e-11);
            let f = narrow(rgb);
            let back = hsv_to_rgb_f32(rgb_to_hsv_f32(f));
            assert_near(widen(back), widen(f), 1e-5);
            let back = hsl_to_rgb_f32(rgb_to_hsl_f32(f));
            assert_near(widen(back), widen(f), 1e-5);
        }
    }

    #[test]
    fn hue_wraps_and_inputs_clamp() {
        assert_near(hsv_to_rgb([480.0, 1.0, 1.0]), hsv_to_rgb([120.0, 1.0, 1.0]), 1e-12);
        assert_near(hsv_to_rgb([-120.0, 1.0, 1.0]), hsv_to_rgb([240.0, 1.0, 1.0]), 1e-12);
        assert_near(hsv_to_rgb([360.0, 1.0, 1.0]), [1.0, 0.0, 0.0], 1e-12);
        assert_near(hsl_to_rgb([-1e-300, 1.0, 0.5]), [1.0, 0.0, 0.0], 1e-12);
        // Saturation above 1 and a negative value clamp.
        assert_eq!(hsv_to_rgb([0.0, 7.0, 1.0]), [1.0, 0.0, 0.0]);
        assert_eq!(hsv_to_rgb([0.0, -1.0, 0.5]), [0.5, 0.5, 0.5]);
        assert_eq!(hsv_to_rgb([0.0, 1.0, -2.0]), [0.0, 0.0, 0.0]);
        assert_eq!(hsl_to_rgb([0.0, 1.0, 9.0]), [1.0, 1.0, 1.0]);
        // Negative RGB clamps to 0 before the conversion.
        assert_eq!(rgb_to_hsv([-0.5, -0.1, 0.5]), [240.0, 1.0, 0.5]);
        assert_eq!(rgb_to_hsl([2.0, -1.0, -1.0]), [0.0, 1.0, 0.5]);
        // A hue just below red must not come out as 360.
        let h = rgb_to_hsv([1.0, 0.5, 0.5 + 1e-17])[0];
        assert!((0.0..360.0).contains(&h));
        assert!((0.0..360.0).contains(&rgb_to_hsv([1.0, 0.0, 1e-300])[0]));
        // 359.999999999 is below 360 in f64 and rounds to 360 in f32.
        assert!(rgb_to_hsv([1.0, 0.0, 1e-11])[0] > 359.9);
        assert_eq!(rgb_to_hsv_f32([1.0, 0.0, 1e-11])[0], 0.0);
        assert_eq!(rgb_to_hsl_f32([1.0, 0.0, 1e-11])[0], 0.0);
    }

    #[test]
    fn non_finite_and_extreme_input_never_panics() {
        for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for f in [rgb_to_hsv, hsv_to_rgb, rgb_to_hsl, hsl_to_rgb, hsv_to_hsl, hsl_to_hsv] {
                let _ = f([v, 0.5, 0.5]);
                let _ = f([0.5, v, 0.5]);
                let _ = f([0.5, 0.5, v]);
            }
        }
        assert_eq!(hsv_to_rgb([f64::NAN, 1.0, 1.0]), [1.0, 0.0, 0.0]);
        for v in [1e100, -1e100, 1e-300] {
            for f in [rgb_to_hsv, hsv_to_rgb, rgb_to_hsl, hsl_to_rgb] {
                assert!(f([v, v, v]).iter().all(|c| c.is_finite()));
                assert!(f([v, 0.25, -v]).iter().all(|c| c.is_finite()));
            }
        }
        for v in [1e30_f32, -1e30] {
            for f in [rgb_to_hsv_f32, hsv_to_rgb_f32, rgb_to_hsl_f32, hsl_to_rgb_f32] {
                assert!(f([v, 0.25, -v]).iter().all(|c| c.is_finite()));
            }
        }
    }
}
