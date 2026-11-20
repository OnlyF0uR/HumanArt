//! The colour picker's readouts `[SKN-13]`: CIELAB and LCh text, and the Munsell Value with the
//! neutral Munsell notation.
//!
//! **Munsell Value** is exact to its standard: ASTM D1535 defines the relation between the
//! luminance factor `Y` (0–100, perfect diffuser = 100) and the Munsell Value `V` (0–10) by the
//! fifth-order polynomial
//! `Y = 1.1914 V − 0.22533 V² + 0.23352 V³ − 0.020484 V⁴ + 0.00081939 V⁵`
//! ([`luminance_from_munsell_value`]); [`munsell_value_from_luminance`] inverts it by bisection
//! (the polynomial is strictly increasing on `[0, 10]`).
//!
//! **Munsell Hue and Chroma are not provided.** They are defined only through the 1943 renotation
//! (S. M. Newhall, D. Nickerson and D. B. Judd, "Final report of the O.S.A. subcommittee on the
//! spacing of the Munsell colors", JOSA 33(7), 1943): a table of about 2 700 xyY points under CIE
//! Illuminant C (the "real" set; 4 995 with the extrapolated "all" set), distributed by the Munsell
//! Color Science Laboratory at RIT. A conversion needs that table (with its provenance and licence
//! recorded), a chromatic adaptation from D65 to Illuminant C and an interpolation over it; no closed
//! form exists, and an approximation from the CIELAB hue angle would be an invented mapping.
//! Until the table is supplied, the picker shows the Value (and the full notation for neutrals,
//! `N V/`) next to the CIELAB and LCh readouts.

use std::fmt;

use super::lab::{lab_to_lch, rgb_to_lab};
use super::spaces::luminance;

/// ASTM D1535: the luminance factor `Y` (0–100) of a Munsell Value `V`. `V` is clamped to
/// `[0, 10]`; a value that is not finite is treated as 0.
pub fn luminance_from_munsell_value(value: f64) -> f64 {
    let v = if value.is_finite() { value.clamp(0.0, 10.0) } else { 0.0 };
    v * (1.1914 + v * (-0.22533 + v * (0.23352 + v * (-0.020484 + v * 0.00081939))))
}

/// The Munsell Value `V` (0–10) of a luminance factor `Y` (0–100, perfect diffuser = 100), the
/// inverse of ASTM D1535 to within `1e-12`. `Y` is clamped to `[0, 100]`; a value that is not finite
/// is treated as 0.
pub fn munsell_value_from_luminance(y: f64) -> f64 {
    let y = if y.is_finite() { y.clamp(0.0, 100.0) } else { 0.0 };
    let (mut lo, mut hi) = (0.0_f64, 10.0_f64);
    // 50 halvings of the interval [0, 10] leave less than 1e-14.
    for _ in 0..50 {
        let mid = f64::midpoint(lo, hi);
        if luminance_from_munsell_value(mid) < y {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    f64::midpoint(lo, hi)
}

/// The Munsell Value of a linear Rec.709 colour, from its luminance (`Y = 1` is white).
pub fn munsell_value_of_rgb(rgb: [f64; 3]) -> f64 {
    munsell_value_from_luminance(100.0 * luminance(rgb))
}

/// A CIELAB readout, `L* 52.3  a* 12.1  b* 18.0`, written without allocating.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabReadout(pub [f64; 3]);

impl LabReadout {
    /// The readout of a linear Rec.709 colour (CIELAB under D65).
    pub fn from_rgb(rgb: [f64; 3]) -> Self {
        Self(rgb_to_lab(rgb))
    }
}

/// A component rounded to one decimal, with `-0.0` shown as `0.0`.
fn one_decimal(v: f64) -> f64 {
    let r = (v * 10.0).round() / 10.0;
    if r == 0.0 { 0.0 } else { r }
}

impl fmt::Display for LabReadout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [l, a, b] = self.0.map(one_decimal);
        write!(f, "L* {l:.1}  a* {a:.1}  b* {b:.1}")
    }
}

/// An LCh readout, `L* 52.3  C* 21.8  h 56.1°`, written without allocating.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LchReadout(pub [f64; 3]);

impl LchReadout {
    /// The readout of a CIELAB colour.
    pub fn from_lab(lab: [f64; 3]) -> Self {
        Self(lab_to_lch(lab))
    }
}

impl fmt::Display for LchReadout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [l, c, h] = self.0.map(one_decimal);
        // A hue that rounds to 360.0 is shown as 0.0.
        let h = if h >= 360.0 { 0.0 } else { h };
        write!(f, "L* {l:.1}  C* {c:.1}  h {h:.1}°")
    }
}

/// The Munsell Value readout of a colour: `V 5.2`, or the neutral notation `N 5.2/` when the colour
/// is achromatic (CIELAB chroma below the given threshold).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MunsellValueReadout {
    /// The Munsell Value, 0–10.
    pub value: f64,
    /// Whether the colour counts as neutral.
    pub neutral: bool,
}

impl MunsellValueReadout {
    /// The readout of a linear Rec.709 colour; it is neutral when its CIELAB chroma (D65) is below
    /// `neutral_chroma` (non-finite or negative thresholds count as 0, so nothing is neutral).
    pub fn from_rgb(rgb: [f64; 3], neutral_chroma: f64) -> Self {
        let lch = lab_to_lch(rgb_to_lab(rgb));
        let threshold = if neutral_chroma.is_finite() { neutral_chroma.max(0.0) } else { 0.0 };
        Self { value: munsell_value_of_rgb(rgb), neutral: lch[1] < threshold }
    }
}

impl fmt::Display for MunsellValueReadout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = one_decimal(self.value);
        if self.neutral { write!(f, "N {v:.1}/") } else { write!(f, "V {v:.1}") }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::checker::{NEUTRAL_PATCHES, REFERENCE_LAB_D50};
    use crate::color::lab::lab_to_xyz;
    use crate::color::spaces::D50_XYZ;

    #[test]
    fn skn_13_astm_d1535_polynomial() {
        assert_eq!(luminance_from_munsell_value(0.0), 0.0);
        // V = 10 is the perfect diffuser: 11.914 − 22.533 + 233.52 − 204.84 + 81.939 = 100.
        assert!((luminance_from_munsell_value(10.0) - 100.0).abs() < 1e-9);
        // V = 5: 5.957 − 5.63325 + 29.19 − 12.8025 + 2.56059375.
        assert!((luminance_from_munsell_value(5.0) - 19.27184375).abs() < 1e-9);
        let mut previous = -1.0;
        for i in 0..=1000 {
            let v = f64::from(i) * 0.01;
            let y = luminance_from_munsell_value(v);
            assert!(y > previous);
            previous = y;
            assert!((munsell_value_from_luminance(y) - v).abs() < 1e-12, "{v}");
        }
    }

    #[test]
    fn skn_13_checker_neutrals_have_their_nominal_munsell_values() {
        // The ColorChecker's neutral patches are named by their nominal Munsell Value; X-Rite's
        // published L* gives back those Values through ASTM D1535 within 0.1.
        let nominal = [9.5, 8.0, 6.5, 5.0, 3.5, 2.0];
        for (&patch, want) in NEUTRAL_PATCHES.iter().zip(nominal) {
            let y = lab_to_xyz(REFERENCE_LAB_D50[patch], D50_XYZ)[1] * 100.0;
            let v = munsell_value_from_luminance(y);
            assert!((v - want).abs() < 0.1, "patch {patch}: {v}");
        }
    }

    #[test]
    fn value_inputs_clamp() {
        assert_eq!(munsell_value_from_luminance(-5.0), munsell_value_from_luminance(0.0));
        assert!(munsell_value_from_luminance(0.0) < 1e-12);
        assert!((munsell_value_from_luminance(250.0) - 10.0).abs() < 1e-12);
        assert!(munsell_value_from_luminance(f64::NAN) < 1e-12);
        assert_eq!(luminance_from_munsell_value(12.0), luminance_from_munsell_value(10.0));
        assert_eq!(luminance_from_munsell_value(f64::INFINITY), 0.0);
        assert!((munsell_value_of_rgb([1.0; 3]) - 10.0).abs() < 1e-9);
        assert!(munsell_value_of_rgb([0.0; 3]) < 1e-12);
    }

    #[test]
    fn readouts_format_without_surprises() {
        assert_eq!(LabReadout([52.34, 12.06, -18.0]).to_string(), "L* 52.3  a* 12.1  b* -18.0");
        assert_eq!(LabReadout([50.0, -0.04, 0.0]).to_string(), "L* 50.0  a* 0.0  b* 0.0");
        assert_eq!(LabReadout::from_rgb([1.0; 3]).to_string(), "L* 100.0  a* 0.0  b* 0.0");
        assert_eq!(LchReadout([50.0, 10.0, 359.97]).to_string(), "L* 50.0  C* 10.0  h 0.0°");
        assert_eq!(LchReadout::from_lab([50.0, 0.0, 10.0]).to_string(), "L* 50.0  C* 10.0  h 90.0°");
        let grey = MunsellValueReadout::from_rgb([0.18; 3], 1.0);
        assert!(grey.neutral);
        assert!(grey.to_string().starts_with("N 4.") || grey.to_string().starts_with("N 5."));
        assert!(grey.to_string().ends_with('/'));
        let red = MunsellValueReadout::from_rgb([0.5, 0.05, 0.05], 1.0);
        assert!(!red.neutral && red.to_string().starts_with("V "));
        assert!(!MunsellValueReadout::from_rgb([0.18; 3], f64::NAN).neutral);
    }
}
