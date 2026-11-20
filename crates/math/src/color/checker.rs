//! The 24-patch ColorChecker Classic: layout, reference colours and the calibration fits of the
//! reference-photo comparison `[CMP-02]` `[COL-03]`.
//!
//! **Reference data.** [`REFERENCE_LAB_D50`] is X-Rite's published CIELAB specification of the
//! ColorChecker Classic made after November 2014 (D50 illuminant, CIE 1931 2° observer), from X-Rite,
//! "New color specifications for ColorChecker SG and Classic Charts" (2016), as reproduced to two
//! decimals in the `colour-science` Python package (BSD-3-Clause,
//! `colour/characterisation/datasets/colour_checkers/chromaticity_coordinates.py`,
//! `DATA_COLORCHECKER24_AFTER_NOV2014_CIE_LAB`). The values are copied unchanged. Linear Rec.709
//! versions under D65 ([`reference_rgb`]) are derived from them with the CIE D50 white
//! ([`super::spaces::D50_XYZ`]) and Bradford adaptation to D65.
//!
//! **Layout.** Four rows of six patches, row-major, seen with the neutral row at the bottom: patch 0
//! (dark skin) is top-left, patch 23 (black) bottom-right.
//!
//! **Calibration.** The user marks patches on a reference photo; the tool reads their mean linear RGB
//! and fits either a per-channel gain ([`fit_gain`], white balance and exposure, best from the
//! neutral patches) or a full 3×3 matrix ([`fit_matrix`]) that maps the measured colours onto the
//! references. Both are ordinary least-squares fits without allocation.

use std::fmt;

use super::delta_e::delta_e_2000;
use super::lab::{lab_to_xyz, rgb_to_lab};
use super::matrix::{ColorMatrix, invert, mul_vec};
use super::spaces::{D50_TO_D65, D50_XYZ, XYZ_TO_REC709};

/// The number of patches.
pub const PATCH_COUNT: usize = 24;
/// The number of patch rows.
pub const ROWS: usize = 4;
/// The number of patch columns.
pub const COLUMNS: usize = 6;

/// The indices of the six neutral patches (white 9.5 to black 2), brightest first.
pub const NEUTRAL_PATCHES: [usize; 6] = [18, 19, 20, 21, 22, 23];

/// The patch names as printed on the chart. Stable ids: never renamed.
pub const PATCH_NAMES: [&str; PATCH_COUNT] = [
    "dark skin",
    "light skin",
    "blue sky",
    "foliage",
    "blue flower",
    "bluish green",
    "orange",
    "purplish blue",
    "moderate red",
    "purple",
    "yellow green",
    "orange yellow",
    "blue",
    "green",
    "red",
    "yellow",
    "magenta",
    "cyan",
    "white 9.5",
    "neutral 8",
    "neutral 6.5",
    "neutral 5",
    "neutral 3.5",
    "black 2",
];

/// X-Rite's CIELAB `[L*, a*, b*]` of every patch, D50 / 2°, chart editions after November 2014
/// (source in the module documentation).
pub const REFERENCE_LAB_D50: [[f64; 3]; PATCH_COUNT] = [
    [37.54, 14.37, 14.92],
    [64.66, 19.27, 17.5],
    [49.32, -3.82, -22.54],
    [43.46, -12.74, 22.72],
    [54.94, 9.61, -24.79],
    [70.48, -32.26, -0.37],
    [62.73, 35.83, 56.5],
    [39.43, 10.75, -45.17],
    [50.57, 48.64, 16.67],
    [30.1, 22.54, -20.87],
    [71.77, -24.13, 58.19],
    [71.51, 18.24, 67.37],
    [28.37, 15.42, -49.8],
    [54.38, -39.72, 32.27],
    [42.43, 51.05, 28.62],
    [81.8, 2.67, 80.41],
    [50.63, 51.28, -14.12],
    [49.57, -29.71, -28.32],
    [95.19, -1.03, 2.93],
    [81.29, -0.57, 0.44],
    [66.89, -0.75, -0.06],
    [50.76, -0.13, 0.14],
    [35.63, -0.46, -0.48],
    [20.64, 0.07, -0.46],
];

/// The patch index at `row` (0 = top) and `column` (0 = left), or `None` outside the chart.
pub const fn patch_index(row: usize, column: usize) -> Option<usize> {
    if row < ROWS && column < COLUMNS { Some(row * COLUMNS + column) } else { None }
}

/// The `(row, column)` of patch `index`, or `None` when there is no such patch.
pub const fn patch_position(index: usize) -> Option<(usize, usize)> {
    if index < PATCH_COUNT { Some((index / COLUMNS, index % COLUMNS)) } else { None }
}

/// The reference colour of patch `index` as linear Rec.709 under D65 (Bradford-adapted from D50), or
/// `None` when there is no such patch. Every patch lies inside the Rec.709 gamut except 17 (cyan),
/// whose red component is slightly negative (about −0.044).
pub fn reference_rgb(index: usize) -> Option<[f64; 3]> {
    let lab = REFERENCE_LAB_D50.get(index)?;
    Some(lab_d50_to_rgb(*lab))
}

/// The reference colours of all patches as linear Rec.709 under D65, see [`reference_rgb`].
pub fn reference_rgb_all() -> [[f64; 3]; PATCH_COUNT] {
    REFERENCE_LAB_D50.map(lab_d50_to_rgb)
}

/// CIELAB under D50 → linear Rec.709 under D65.
fn lab_d50_to_rgb(lab: [f64; 3]) -> [f64; 3] {
    mul_vec(&XYZ_TO_REC709, mul_vec(&D50_TO_D65, lab_to_xyz(lab, D50_XYZ)))
}

/// Why a calibration fit failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationError {
    /// The measured and the reference lists have different lengths.
    LengthMismatch {
        /// The number of measured colours.
        measured: usize,
        /// The number of reference colours.
        reference: usize,
    },
    /// Fewer samples than the fit needs.
    TooFewSamples {
        /// The number of samples given.
        got: usize,
        /// The number the fit needs.
        need: usize,
    },
    /// A colour component is NaN or infinite.
    NonFinite {
        /// The index of the offending sample.
        index: usize,
    },
    /// The measured colours do not determine the fit: a channel is zero in every sample (gain), or
    /// the samples lie in a plane through black (matrix).
    Degenerate,
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::LengthMismatch { measured, reference } => {
                write!(f, "{measured} measured colours but {reference} reference colours")
            }
            Self::TooFewSamples { got, need } => {
                write!(f, "the calibration needs at least {need} marked patches, got {got}")
            }
            Self::NonFinite { index } => write!(f, "sample {index} has a non-finite colour component"),
            Self::Degenerate => {
                f.write_str("the marked patches do not determine a correction; mark patches of different colours")
            }
        }
    }
}

impl std::error::Error for CalibrationError {}

/// Checks the sample lists: equal length, at least `need` samples, every component finite.
fn check_samples(measured: &[[f64; 3]], reference: &[[f64; 3]], need: usize) -> Result<(), CalibrationError> {
    if measured.len() != reference.len() {
        return Err(CalibrationError::LengthMismatch { measured: measured.len(), reference: reference.len() });
    }
    if measured.len() < need {
        return Err(CalibrationError::TooFewSamples { got: measured.len(), need });
    }
    for (index, (m, r)) in measured.iter().zip(reference).enumerate() {
        if !m.iter().chain(r).all(|v| v.is_finite()) {
            return Err(CalibrationError::NonFinite { index });
        }
    }
    Ok(())
}

/// The per-channel gain `g` minimising `Σ |g ⊙ measured − reference|²`:
/// `g_c = Σ m_c r_c / Σ m_c²` for each channel `c`.
///
/// Fit on the neutral patches it is a white balance plus exposure correction. Needs at least one
/// sample; fails with [`CalibrationError::Degenerate`] when a channel is zero in every sample.
pub fn fit_gain(measured: &[[f64; 3]], reference: &[[f64; 3]]) -> Result<[f64; 3], CalibrationError> {
    check_samples(measured, reference, 1)?;
    let mut num = [0.0; 3];
    let mut den = [0.0; 3];
    for (m, r) in measured.iter().zip(reference) {
        for c in 0..3 {
            num[c] += m[c] * r[c];
            den[c] += m[c] * m[c];
        }
    }
    let mut gain = [0.0; 3];
    for c in 0..3 {
        let g = num[c] / den[c];
        if !(den[c] > 0.0 && g.is_finite()) {
            return Err(CalibrationError::Degenerate);
        }
        gain[c] = g;
    }
    Ok(gain)
}

/// The 3×3 matrix `M` minimising `Σ |M · measured − reference|²`: `M = (Σ r mᵀ)(Σ m mᵀ)⁻¹`.
///
/// Needs at least three samples that do not lie in one plane through black; fails with
/// [`CalibrationError::Degenerate`] otherwise (the test is relative to the samples' scale, so a dark
/// photo is not rejected for being dark).
pub fn fit_matrix(measured: &[[f64; 3]], reference: &[[f64; 3]]) -> Result<ColorMatrix, CalibrationError> {
    check_samples(measured, reference, 3)?;
    let mut mm = [[0.0; 3]; 3];
    let mut rm = [[0.0; 3]; 3];
    for (m, r) in measured.iter().zip(reference) {
        for i in 0..3 {
            for j in 0..3 {
                mm[i][j] += m[i] * m[j];
                rm[i][j] += r[i] * m[j];
            }
        }
    }
    // Relative conditioning: the determinant of the Gram matrix against the cube of its trace.
    let trace = mm[0][0] + mm[1][1] + mm[2][2];
    let det = super::matrix::determinant(&mm);
    if !(trace > 0.0 && trace.is_finite() && det.is_finite() && det > 1e-12 * trace * trace * trace) {
        return Err(CalibrationError::Degenerate);
    }
    let inverse = invert(&mm).ok_or(CalibrationError::Degenerate)?;
    let fit = super::matrix::mul(&rm, &inverse);
    if fit.iter().flatten().all(|v| v.is_finite()) { Ok(fit) } else { Err(CalibrationError::Degenerate) }
}

/// A colour multiplied by a per-channel gain.
#[inline]
pub fn apply_gain(rgb: [f64; 3], gain: [f64; 3]) -> [f64; 3] {
    [rgb[0] * gain[0], rgb[1] * gain[1], rgb[2] * gain[2]]
}

/// The mean and the maximum CIEDE2000 between paired linear Rec.709 colours (through CIELAB under
/// D65), for reporting a calibration's residual; `None` when the lists are empty or of different
/// lengths.
pub fn delta_e_2000_summary(a: &[[f64; 3]], b: &[[f64; 3]]) -> Option<(f64, f64)> {
    if a.is_empty() || a.len() != b.len() {
        return None;
    }
    let mut sum = 0.0;
    let mut max = 0.0_f64;
    for (x, y) in a.iter().zip(b) {
        let d = delta_e_2000(rgb_to_lab(*x), rgb_to_lab(*y));
        sum += d;
        max = max.max(d);
    }
    Some((sum / a.len() as f64, max))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::lab::xyz_to_lab;
    use crate::color::spaces::{D65_TO_D50, REC709_TO_XYZ, luminance};
    use humanart_core::rng::Rng;

    #[test]
    fn layout_is_four_rows_of_six() {
        assert_eq!(patch_index(0, 0), Some(0));
        assert_eq!(patch_index(3, 5), Some(23));
        assert_eq!(patch_index(1, 0), Some(6));
        assert_eq!(patch_index(4, 0), None);
        assert_eq!(patch_index(0, 6), None);
        for i in 0..PATCH_COUNT {
            let (row, column) = patch_position(i).unwrap();
            assert_eq!(patch_index(row, column), Some(i));
        }
        assert_eq!(patch_position(24), None);
        assert_eq!(PATCH_NAMES[0], "dark skin");
        assert_eq!(PATCH_NAMES[NEUTRAL_PATCHES[0]], "white 9.5");
        assert_eq!(PATCH_NAMES[23], "black 2");
        assert_eq!(reference_rgb(24), None);
    }

    #[test]
    fn cmp_02_reference_rgb_round_trips_to_the_published_lab() {
        for (i, lab) in REFERENCE_LAB_D50.iter().enumerate() {
            let rgb = reference_rgb(i).unwrap();
            let xyz_d50 = mul_vec(&D65_TO_D50, mul_vec(&REC709_TO_XYZ, rgb));
            let back = xyz_to_lab(xyz_d50, D50_XYZ);
            for c in 0..3 {
                assert!((back[c] - lab[c]).abs() < 1e-9, "patch {i}: {back:?} vs {lab:?}");
            }
        }
        assert_eq!(reference_rgb_all()[5], reference_rgb(5).unwrap());
    }

    #[test]
    fn cmp_02_reference_rgb_is_plausible() {
        let all = reference_rgb_all();
        // Neutrals are nearly grey (white 9.5 is slightly warm, b* = 2.93: 6 % less blue), get darker
        // down the row, and white 9.5 has Y ≈ 0.88.
        let mut previous = f64::INFINITY;
        for &i in &NEUTRAL_PATCHES {
            let [r, g, b] = all[i];
            let y = luminance(all[i]);
            assert!(y < previous, "patch {i}");
            previous = y;
            assert!(
                (r - g).abs() < 0.07 * g.max(0.02) + 0.005 && (b - g).abs() < 0.07 * g.max(0.02) + 0.005,
                "{i}: {:?}",
                all[i]
            );
        }
        assert!((luminance(all[18]) - 0.8807).abs() < 0.01);
        // Gamut: only cyan (17) leaves Rec.709, with a red component of about -0.044 (cross-checked
        // with an independent Python computation).
        for (i, rgb) in all.iter().enumerate() {
            let inside = rgb.iter().all(|v| (0.0..=1.0).contains(v));
            assert_eq!(inside, i != 17, "patch {i}: {rgb:?}");
        }
        // The chromatic patches have the hue their names say.
        let [r, g, b] = all[14];
        assert!(r > g && r > b, "red");
        let [r, g, b] = all[13];
        assert!(g > r && g > b, "green");
        let [r, g, b] = all[12];
        assert!(b > r && b > g, "blue");
    }

    #[test]
    fn cmp_02_gain_fit_recovers_a_white_balance() {
        let reference = reference_rgb_all();
        let true_gain = [1.7, 1.0, 0.6];
        let neutrals: Vec<[f64; 3]> = NEUTRAL_PATCHES.iter().map(|&i| reference[i]).collect();
        let measured: Vec<[f64; 3]> =
            neutrals.iter().map(|c| [c[0] / true_gain[0], c[1] / true_gain[1], c[2] / true_gain[2]]).collect();
        let gain = fit_gain(&measured, &neutrals).unwrap();
        for c in 0..3 {
            assert!((gain[c] - true_gain[c]).abs() < 1e-12);
        }
        for (m, r) in measured.iter().zip(&neutrals) {
            let corrected = apply_gain(*m, gain);
            assert!((0..3).all(|c| (corrected[c] - r[c]).abs() < 1e-12));
        }
    }

    #[test]
    fn cmp_02_matrix_fit_recovers_a_camera_matrix_under_noise() {
        let reference = reference_rgb_all();
        let camera: ColorMatrix = [[0.8, 0.15, 0.05], [0.1, 0.7, 0.2], [0.02, 0.1, 0.5]];
        let mut rng = Rng::new(0xC4EC);
        let exact: Vec<[f64; 3]> = reference.iter().map(|r| mul_vec(&camera, *r)).collect();
        let fit = fit_matrix(&exact, &reference).unwrap();
        let expected = invert(&camera).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!((fit[i][j] - expected[i][j]).abs() < 1e-9, "{fit:?}");
            }
        }
        // With 0.2 % noise the corrected chart is within a small ΔE of the reference.
        let noisy: Vec<[f64; 3]> = exact.iter().map(|c| c.map(|v| v * (1.0 + rng.range_f64(-0.002, 0.002)))).collect();
        let fit = fit_matrix(&noisy, &reference).unwrap();
        let corrected: Vec<[f64; 3]> = noisy.iter().map(|c| mul_vec(&fit, *c)).collect();
        let (mean, max) = delta_e_2000_summary(&corrected, &reference).unwrap();
        assert!(mean < 0.3 && max < 1.0, "mean {mean}, max {max}");
        // Uncorrected, the error is large.
        let (mean, _) = delta_e_2000_summary(&noisy, &reference).unwrap();
        assert!(mean > 5.0);
        // A dark photo (everything scaled by 1e-4) is not degenerate.
        let dark: Vec<[f64; 3]> = exact.iter().map(|c| c.map(|v| v * 1e-4)).collect();
        assert!(fit_matrix(&dark, &reference).is_ok());
    }

    #[test]
    fn calibration_rejects_bad_input_with_a_reason() {
        let r = reference_rgb_all();
        assert_eq!(fit_gain(&r[..2], &r[..3]), Err(CalibrationError::LengthMismatch { measured: 2, reference: 3 }));
        assert_eq!(fit_gain(&[], &[]), Err(CalibrationError::TooFewSamples { got: 0, need: 1 }));
        assert_eq!(fit_matrix(&r[..2], &r[..2]), Err(CalibrationError::TooFewSamples { got: 2, need: 3 }));
        let mut bad = r;
        bad[4][1] = f64::NAN;
        assert_eq!(fit_matrix(&bad, &r), Err(CalibrationError::NonFinite { index: 4 }));
        assert_eq!(fit_gain(&r, &bad), Err(CalibrationError::NonFinite { index: 4 }));
        // Neutral patches only lie on a line: no matrix, but a gain.
        let neutrals: Vec<[f64; 3]> = NEUTRAL_PATCHES.iter().map(|&i| [r[i][1]; 3]).collect();
        assert_eq!(fit_matrix(&neutrals, &neutrals), Err(CalibrationError::Degenerate));
        assert!(fit_gain(&neutrals, &neutrals).is_ok());
        // A dead channel.
        let dead: Vec<[f64; 3]> = r.iter().map(|c| [c[0], 0.0, c[2]]).collect();
        assert_eq!(fit_gain(&dead, &r), Err(CalibrationError::Degenerate));
        assert_eq!(fit_matrix(&dead, &r), Err(CalibrationError::Degenerate));
        // Every error has a message.
        for e in [
            CalibrationError::LengthMismatch { measured: 1, reference: 2 },
            CalibrationError::TooFewSamples { got: 0, need: 3 },
            CalibrationError::NonFinite { index: 0 },
            CalibrationError::Degenerate,
        ] {
            assert!(!e.to_string().is_empty());
        }
        assert_eq!(delta_e_2000_summary(&[], &[]), None);
        assert_eq!(delta_e_2000_summary(&r[..1], &r[..2]), None);
        assert_eq!(delta_e_2000_summary(&r, &r), Some((0.0, 0.0)));
    }

    #[test]
    fn huge_input_never_panics() {
        let r = reference_rgb_all();
        let huge: Vec<[f64; 3]> = r.iter().map(|c| c.map(|v| v * 1e200)).collect();
        let _ = fit_matrix(&huge, &r);
        let _ = fit_gain(&huge, &r);
    }
}
