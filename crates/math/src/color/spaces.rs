//! RGB colour spaces and CIE XYZ: matrices derived from chromaticities, chromatic adaptation, xyY and
//! luminance `[COL-01]`.
//!
//! Nothing here is a typed-in matrix. Each RGB↔XYZ matrix is derived at compile time from the CIE xy
//! chromaticities of the primaries and of the white point that the standard defines (the method of
//! SMPTE RP 177), and the tests compare the result with the matrices the standards publish. All RGB
//! values in this module are linear; the transfer curves are in [`super::transfer`].
//!
//! XYZ is scaled so that the white point has `Y = 1`. The D65 white used throughout the colour module
//! is [`D65_XYZ`], computed from the chromaticity `(0.3127, 0.3290)` that Rec.709, sRGB, Display-P3
//! and Rec.2020 specify, so that RGB `(1, 1, 1)` is exactly the white of [`super::lab`] and a grey has
//! exactly zero chroma. (The tabulated ASTM E308 D65, `(0.95047, 1, 1.08883)`, differs from it in the
//! fourth decimal.)
//!
//! Conversions do not clamp: a colour outside the destination gamut comes back with negative or
//! above-one components, which is what a caller doing further linear math wants. Clamp at the point of
//! display.

use super::matrix::{self, ColorMatrix};

/// The CIE xy chromaticity of the D65 white point as Rec.709, sRGB, Display-P3 and Rec.2020 give it.
pub const D65_XY: [f64; 2] = [0.3127, 0.3290];
/// The CIE xy chromaticity of the D50 white point.
pub const D50_XY: [f64; 2] = [0.3457, 0.3585];

/// The D65 white in XYZ (`Y = 1`), from [`D65_XY`]: the white of every RGB space in this module.
pub const D65_XYZ: [f64; 3] = xy_to_xyz(D65_XY);
/// The D50 white in XYZ (`Y = 1`) for the CIE 1931 2° observer as tabulated in ASTM E308: the white
/// under which colour-chart Lab reference values are published.
pub const D50_XYZ: [f64; 3] = [0.964_22, 1.0, 0.825_21];

/// The chromaticities that define an RGB colour space: its three primaries and its white point, as
/// CIE 1931 xy pairs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RgbPrimaries {
    /// The red primary.
    pub red: [f64; 2],
    /// The green primary.
    pub green: [f64; 2],
    /// The blue primary.
    pub blue: [f64; 2],
    /// The white point (what RGB `(1, 1, 1)` looks like).
    pub white: [f64; 2],
}

/// ITU-R BT.709 / sRGB primaries with a D65 white: the working space `[COL-01]`.
pub const REC709: RgbPrimaries =
    RgbPrimaries { red: [0.64, 0.33], green: [0.30, 0.60], blue: [0.15, 0.06], white: D65_XY };
/// Display-P3: the DCI-P3 primaries with a D65 white (SMPTE EG 432-1).
pub const DISPLAY_P3: RgbPrimaries =
    RgbPrimaries { red: [0.680, 0.320], green: [0.265, 0.690], blue: [0.150, 0.060], white: D65_XY };
/// ITU-R BT.2020 primaries with a D65 white.
pub const REC2020: RgbPrimaries =
    RgbPrimaries { red: [0.708, 0.292], green: [0.170, 0.797], blue: [0.131, 0.046], white: D65_XY };

/// The XYZ tristimulus values, scaled to `Y = 1`, of the chromaticity `xy`.
///
/// A chromaticity whose `y` is not positive, or that is not finite, has no such colour: the result
/// is then black, `[0, 0, 0]`.
pub const fn xy_to_xyz(xy: [f64; 2]) -> [f64; 3] {
    let [x, y] = xy;
    if y > 0.0 && x.is_finite() && y.is_finite() {
        let out = [x / y, 1.0, (1.0 - x - y) / y];
        if out[0].is_finite() && out[2].is_finite() {
            return out;
        }
    }
    [0.0, 0.0, 0.0]
}

/// The matrix taking linear RGB with the given primaries to XYZ (white at `Y = 1`), or `None` when
/// the primaries do not span a colour space (collinear chromaticities, a `y` that is not positive, or
/// a white point they cannot reach).
///
/// Derivation: with `P` the matrix whose columns are the XYZ of the three primaries at unit `Y`, the
/// per-primary scales `S = P⁻¹ · W` make RGB `(1, 1, 1)` land on the white `W`; the result is
/// `P · diag(S)`.
pub const fn rgb_to_xyz_matrix(primaries: &RgbPrimaries) -> Option<ColorMatrix> {
    let r = xy_to_xyz(primaries.red);
    let g = xy_to_xyz(primaries.green);
    let b = xy_to_xyz(primaries.blue);
    let w = xy_to_xyz(primaries.white);
    if r[1] == 0.0 || g[1] == 0.0 || b[1] == 0.0 || w[1] == 0.0 {
        return None;
    }
    let p = [[r[0], g[0], b[0]], [r[1], g[1], b[1]], [r[2], g[2], b[2]]];
    let Some(p_inverse) = matrix::invert(&p) else {
        return None;
    };
    let s = matrix::mul_vec(&p_inverse, w);
    // A primary that the white needs none of (to rounding) leaves the space two-dimensional.
    if s[0].abs() <= 1e-12 || s[1].abs() <= 1e-12 || s[2].abs() <= 1e-12 {
        return None;
    }
    Some(matrix::mul(&p, &matrix::diagonal(s)))
}

/// The matrix taking XYZ to linear RGB with the given primaries: the inverse of
/// [`rgb_to_xyz_matrix`], `None` under the same conditions.
pub const fn xyz_to_rgb_matrix(primaries: &RgbPrimaries) -> Option<ColorMatrix> {
    match rgb_to_xyz_matrix(primaries) {
        Some(m) => matrix::invert(&m),
        None => None,
    }
}

/// The matrix converting linear RGB in the `from` space to linear RGB in the `to` space (both must
/// share a white point for the result to be colorimetric; the built-in spaces are all D65).
pub const fn rgb_to_rgb_matrix(from: &RgbPrimaries, to: &RgbPrimaries) -> Option<ColorMatrix> {
    match (rgb_to_xyz_matrix(from), xyz_to_rgb_matrix(to)) {
        (Some(to_xyz), Some(from_xyz)) => Some(matrix::mul(&from_xyz, &to_xyz)),
        _ => None,
    }
}

/// Unwraps a matrix derived from the built-in constants. Only evaluated at compile time: a failure
/// here is a build error, never a runtime panic.
pub(super) const fn built_in(m: Option<ColorMatrix>) -> ColorMatrix {
    match m {
        Some(m) => m,
        None => panic!("a built-in colour space is degenerate"),
    }
}

/// Linear Rec.709/sRGB → XYZ.
pub const REC709_TO_XYZ: ColorMatrix = built_in(rgb_to_xyz_matrix(&REC709));
/// XYZ → linear Rec.709/sRGB.
pub const XYZ_TO_REC709: ColorMatrix = built_in(xyz_to_rgb_matrix(&REC709));
/// Linear Display-P3 → XYZ.
pub const DISPLAY_P3_TO_XYZ: ColorMatrix = built_in(rgb_to_xyz_matrix(&DISPLAY_P3));
/// XYZ → linear Display-P3.
pub const XYZ_TO_DISPLAY_P3: ColorMatrix = built_in(xyz_to_rgb_matrix(&DISPLAY_P3));
/// Linear Rec.2020 → XYZ.
pub const REC2020_TO_XYZ: ColorMatrix = built_in(rgb_to_xyz_matrix(&REC2020));
/// XYZ → linear Rec.2020.
pub const XYZ_TO_REC2020: ColorMatrix = built_in(xyz_to_rgb_matrix(&REC2020));
/// Linear Rec.709 → linear Display-P3 (the display transform's gamut step for a P3 surface
/// `[COL-02]`). Every in-gamut Rec.709 colour is in gamut in P3.
pub const REC709_TO_DISPLAY_P3: ColorMatrix = built_in(rgb_to_rgb_matrix(&REC709, &DISPLAY_P3));
/// Linear Display-P3 → linear Rec.709. Saturated P3 colours come out with negative components.
pub const DISPLAY_P3_TO_REC709: ColorMatrix = built_in(rgb_to_rgb_matrix(&DISPLAY_P3, &REC709));
/// Linear Rec.709 → linear Rec.2020.
pub const REC709_TO_REC2020: ColorMatrix = built_in(rgb_to_rgb_matrix(&REC709, &REC2020));
/// Linear Rec.2020 → linear Rec.709. Saturated Rec.2020 colours come out with negative components.
pub const REC2020_TO_REC709: ColorMatrix = built_in(rgb_to_rgb_matrix(&REC2020, &REC709));

/// The luminance weights of linear Rec.709 RGB: the `Y` row of [`REC709_TO_XYZ`]
/// (0.2126, 0.7152, 0.0722 to four decimals).
pub const LUMINANCE_REC709: [f64; 3] = REC709_TO_XYZ[1];
/// [`LUMINANCE_REC709`] in `f32`, for the code that mirrors a shader.
pub const LUMINANCE_REC709_F32: [f32; 3] = matrix::narrow(LUMINANCE_REC709);

/// A 3×3 matrix in `f32`, row-major, for upload to a shader (which wants it transposed to
/// column-major if it is declared as a WGSL `mat3x3`).
pub type ColorMatrixF32 = [[f32; 3]; 3];

/// Narrows a matrix to `f32`.
pub const fn matrix_to_f32(m: &ColorMatrix) -> ColorMatrixF32 {
    [matrix::narrow(m[0]), matrix::narrow(m[1]), matrix::narrow(m[2])]
}

/// [`REC709_TO_DISPLAY_P3`] in `f32`.
pub const REC709_TO_DISPLAY_P3_F32: ColorMatrixF32 = matrix_to_f32(&REC709_TO_DISPLAY_P3);
/// [`DISPLAY_P3_TO_REC709`] in `f32`.
pub const DISPLAY_P3_TO_REC709_F32: ColorMatrixF32 = matrix_to_f32(&DISPLAY_P3_TO_REC709);

/// Linear Rec.709 RGB → XYZ.
#[inline]
pub const fn rec709_to_xyz(rgb: [f64; 3]) -> [f64; 3] {
    matrix::mul_vec(&REC709_TO_XYZ, rgb)
}

/// XYZ → linear Rec.709 RGB (not clamped).
#[inline]
pub const fn xyz_to_rec709(xyz: [f64; 3]) -> [f64; 3] {
    matrix::mul_vec(&XYZ_TO_REC709, xyz)
}

/// Linear Rec.709 RGB → linear Display-P3 RGB.
#[inline]
pub const fn rec709_to_display_p3(rgb: [f64; 3]) -> [f64; 3] {
    matrix::mul_vec(&REC709_TO_DISPLAY_P3, rgb)
}

/// Linear Display-P3 RGB → linear Rec.709 RGB (not clamped: out-of-gamut colours go negative).
#[inline]
pub const fn display_p3_to_rec709(rgb: [f64; 3]) -> [f64; 3] {
    matrix::mul_vec(&DISPLAY_P3_TO_REC709, rgb)
}

/// Linear Rec.709 RGB → linear Rec.2020 RGB.
#[inline]
pub const fn rec709_to_rec2020(rgb: [f64; 3]) -> [f64; 3] {
    matrix::mul_vec(&REC709_TO_REC2020, rgb)
}

/// Linear Rec.2020 RGB → linear Rec.709 RGB (not clamped: out-of-gamut colours go negative).
#[inline]
pub const fn rec2020_to_rec709(rgb: [f64; 3]) -> [f64; 3] {
    matrix::mul_vec(&REC2020_TO_REC709, rgb)
}

/// The product of an `f32` matrix and a column vector.
#[inline]
pub const fn mul_vec_f32(m: &ColorMatrixF32, v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// [`rec709_to_display_p3`] in `f32`: the CPU reference of the shader's P3 output step.
#[inline]
pub const fn rec709_to_display_p3_f32(rgb: [f32; 3]) -> [f32; 3] {
    mul_vec_f32(&REC709_TO_DISPLAY_P3_F32, rgb)
}

/// [`display_p3_to_rec709`] in `f32`.
#[inline]
pub const fn display_p3_to_rec709_f32(rgb: [f32; 3]) -> [f32; 3] {
    mul_vec_f32(&DISPLAY_P3_TO_REC709_F32, rgb)
}

/// The relative luminance `Y` of a linear Rec.709 colour.
#[inline]
pub const fn luminance(rgb: [f64; 3]) -> f64 {
    LUMINANCE_REC709[0] * rgb[0] + LUMINANCE_REC709[1] * rgb[1] + LUMINANCE_REC709[2] * rgb[2]
}

/// [`luminance`] in `f32`.
#[inline]
pub const fn luminance_f32(rgb: [f32; 3]) -> f32 {
    LUMINANCE_REC709_F32[0] * rgb[0] + LUMINANCE_REC709_F32[1] * rgb[1] + LUMINANCE_REC709_F32[2] * rgb[2]
}

/// The Bradford cone-response matrix (Lam 1985, as used by ICC profiles): XYZ → sharpened LMS.
pub const BRADFORD: ColorMatrix = [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]];

/// The linear Bradford chromatic adaptation matrix that maps XYZ colours seen under the white
/// `source_white` to the corresponding colours under `destination_white`:
/// `B⁻¹ · diag(B·dst / B·src) · B`.
///
/// `None` when a white point is not finite or has a cone response that is not positive (no
/// physical white does).
pub const fn bradford_adaptation(source_white: [f64; 3], destination_white: [f64; 3]) -> Option<ColorMatrix> {
    let src = matrix::mul_vec(&BRADFORD, source_white);
    let dst = matrix::mul_vec(&BRADFORD, destination_white);
    let mut gains = [0.0; 3];
    let mut i = 0;
    while i < 3 {
        if !(src[i] > 0.0 && dst[i] > 0.0 && src[i].is_finite() && dst[i].is_finite()) {
            return None;
        }
        gains[i] = dst[i] / src[i];
        if !gains[i].is_finite() {
            return None;
        }
        i += 1;
    }
    let Some(inverse) = matrix::invert(&BRADFORD) else {
        return None;
    };
    Some(matrix::mul(&inverse, &matrix::mul(&matrix::diagonal(gains), &BRADFORD)))
}

/// Bradford adaptation from [`D65_XYZ`] to [`D50_XYZ`].
pub const D65_TO_D50: ColorMatrix = built_in(bradford_adaptation(D65_XYZ, D50_XYZ));
/// Bradford adaptation from [`D50_XYZ`] to [`D65_XYZ`].
pub const D50_TO_D65: ColorMatrix = built_in(bradford_adaptation(D50_XYZ, D65_XYZ));

/// XYZ → CIE xyY (`[x, y, Y]`).
///
/// Black has no chromaticity: when `X + Y + Z` is not positive (or the division is not finite) the
/// chromaticity of D65 is returned, with the input's `Y`.
pub fn xyz_to_xyy(xyz: [f64; 3]) -> [f64; 3] {
    let sum = xyz[0] + xyz[1] + xyz[2];
    if sum > 0.0 {
        let (x, y) = (xyz[0] / sum, xyz[1] / sum);
        if x.is_finite() && y.is_finite() {
            return [x, y, xyz[1]];
        }
    }
    [D65_XY[0], D65_XY[1], xyz[1]]
}

/// CIE xyY (`[x, y, Y]`) → XYZ. A `y` that is not positive gives black.
pub fn xyy_to_xyz(xyy: [f64; 3]) -> [f64; 3] {
    let unit = xy_to_xyz([xyy[0], xyy[1]]);
    let out = [unit[0] * xyy[2], unit[1] * xyy[2], unit[2] * xyy[2]];
    if out.iter().all(|v| v.is_finite()) { out } else { [0.0, 0.0, 0.0] }
}

/// The CIE xy chromaticity of a linear Rec.709 colour (D65 for black; see [`xyz_to_xyy`]).
pub fn rec709_to_xy(rgb: [f64; 3]) -> [f64; 2] {
    let xyy = xyz_to_xyy(rec709_to_xyz(rgb));
    [xyy[0], xyy[1]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    fn assert_matrix_near(got: &ColorMatrix, want: &ColorMatrix, tolerance: f64) {
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (got[r][c] - want[r][c]).abs() <= tolerance,
                    "[{r}][{c}]: got {}, published {}",
                    got[r][c],
                    want[r][c]
                );
            }
        }
    }

    fn assert_near(got: [f64; 3], want: [f64; 3], tolerance: f64) {
        for i in 0..3 {
            assert!((got[i] - want[i]).abs() <= tolerance, "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn rec709_matrix_matches_iec_61966_2_1() {
        // The matrix printed in IEC 61966-2-1 (sRGB) and ITU-R BT.709, four decimals.
        let published = [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]];
        assert_matrix_near(&REC709_TO_XYZ, &published, 5e-5);
        // The inverse printed in the standard was computed from the rounded forward matrix, so it only
        // agrees to about three decimals with the exact inverse.
        let published_inverse = [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]];
        assert_matrix_near(&XYZ_TO_REC709, &published_inverse, 5e-4);
        assert_matrix_near(&matrix::mul(&XYZ_TO_REC709, &REC709_TO_XYZ), &matrix::IDENTITY, 1e-14);
        assert_near(LUMINANCE_REC709, [0.2126, 0.7152, 0.0722], 5e-5);
    }

    #[test]
    fn display_p3_matrix_matches_the_published_one() {
        // SMPTE EG 432-1 / the CSS Color 4 sample code, seven decimals.
        let published = [
            [0.486_570_9, 0.265_667_7, 0.198_217_3],
            [0.228_974_6, 0.691_738_5, 0.079_286_9],
            [0.0, 0.045_113_4, 1.043_944_4],
        ];
        assert_matrix_near(&DISPLAY_P3_TO_XYZ, &published, 1e-7);
        assert_matrix_near(&matrix::mul(&XYZ_TO_DISPLAY_P3, &DISPLAY_P3_TO_XYZ), &matrix::IDENTITY, 1e-14);
    }

    #[test]
    fn rec2020_matrix_matches_bt2020() {
        // ITU-R BT.2020 / BT.2087, four decimals; the luminance row is the standard's luma weights.
        let published = [[0.6370, 0.1446, 0.1689], [0.2627, 0.6780, 0.0593], [0.0, 0.0281, 1.0610]];
        assert_matrix_near(&REC2020_TO_XYZ, &published, 5e-5);
        assert_matrix_near(&matrix::mul(&XYZ_TO_REC2020, &REC2020_TO_XYZ), &matrix::IDENTITY, 1e-14);
    }

    #[test]
    fn gamut_conversion_matrices_match_the_published_ones() {
        // Rec.709 → Display-P3 as commonly published (seven decimals).
        let to_p3 =
            [[0.822_462_0, 0.177_538_0, 0.0], [0.033_194_2, 0.966_805_8, 0.0], [0.017_082_6, 0.072_397_4, 0.910_519_9]];
        assert_matrix_near(&REC709_TO_DISPLAY_P3, &to_p3, 1e-7);
        // Rec.709 → Rec.2020 from ITU-R BT.2087, four decimals.
        let to_2020 = [[0.6274, 0.3293, 0.0433], [0.0691, 0.9195, 0.0114], [0.0164, 0.0880, 0.8956]];
        assert_matrix_near(&REC709_TO_REC2020, &to_2020, 5e-5);
        assert_matrix_near(&matrix::mul(&DISPLAY_P3_TO_REC709, &REC709_TO_DISPLAY_P3), &matrix::IDENTITY, 1e-14);
        assert_matrix_near(&matrix::mul(&REC2020_TO_REC709, &REC709_TO_REC2020), &matrix::IDENTITY, 1e-14);
    }

    #[test]
    fn white_is_white_in_every_space() {
        for m in [&REC709_TO_XYZ, &DISPLAY_P3_TO_XYZ, &REC2020_TO_XYZ] {
            assert_near(matrix::mul_vec(m, [1.0, 1.0, 1.0]), D65_XYZ, 1e-14);
        }
        for m in [&REC709_TO_DISPLAY_P3, &DISPLAY_P3_TO_REC709, &REC709_TO_REC2020, &REC2020_TO_REC709] {
            assert_near(matrix::mul_vec(m, [1.0, 1.0, 1.0]), [1.0, 1.0, 1.0], 1e-14);
        }
        assert_near(D65_XYZ, [0.950_456, 1.0, 1.089_058], 1e-6);
        assert!((luminance([1.0, 1.0, 1.0]) - 1.0).abs() < 1e-15);
        assert!((luminance_f32([1.0, 1.0, 1.0]) - 1.0).abs() < 1e-6);
        assert!((luminance([0.2, 0.5, 0.9]) - rec709_to_xyz([0.2, 0.5, 0.9])[1]).abs() < 1e-15);
        assert_near(xyz_to_rec709(D65_XYZ), [1.0, 1.0, 1.0], 1e-14);
        let xy = rec709_to_xy([0.7, 0.7, 0.7]);
        assert!((xy[0] - D65_XY[0]).abs() < 1e-14 && (xy[1] - D65_XY[1]).abs() < 1e-14);
    }

    #[test]
    fn primaries_map_to_their_chromaticities() {
        for (space, to_xyz) in [(REC709, REC709_TO_XYZ), (DISPLAY_P3, DISPLAY_P3_TO_XYZ), (REC2020, REC2020_TO_XYZ)] {
            for (axis, xy) in
                [([1.0, 0.0, 0.0], space.red), ([0.0, 1.0, 0.0], space.green), ([0.0, 0.0, 1.0], space.blue)]
            {
                let xyy = xyz_to_xyy(matrix::mul_vec(&to_xyz, axis));
                assert!((xyy[0] - xy[0]).abs() < 1e-13 && (xyy[1] - xy[1]).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn gamut_conversions_round_trip_and_show_the_gamut_relation() {
        let mut rng = Rng::new(0x9A307);
        for _ in 0..5000 {
            let c = [rng.f64(), rng.f64(), rng.f64()];
            let p3 = rec709_to_display_p3(c);
            // Rec.709 is inside P3 (up to rounding of the two zero entries).
            assert!(p3.iter().all(|v| (-1e-12..=1.0 + 1e-12).contains(v)), "{p3:?}");
            assert_near(display_p3_to_rec709(p3), c, 1e-13);
            assert_near(rec2020_to_rec709(rec709_to_rec2020(c)), c, 1e-13);
            let f = matrix::narrow(c);
            let back = display_p3_to_rec709_f32(rec709_to_display_p3_f32(f));
            assert_near(matrix::widen(back), matrix::widen(f), 2e-6);
        }
        // The P3 primaries are outside Rec.709.
        assert!(display_p3_to_rec709([1.0, 0.0, 0.0])[1] < -0.01);
        assert!(display_p3_to_rec709([0.0, 1.0, 0.0])[0] < -0.1);
        assert!(rec2020_to_rec709([0.0, 1.0, 0.0])[0] < -0.3);
    }

    #[test]
    fn bradford_matches_the_published_d65_to_d50_matrix() {
        // Lindbloom's Bradford matrices are for the ASTM E308 whites.
        let d65 = [0.950_47, 1.0, 1.088_83];
        let d50 = [0.964_22, 1.0, 0.825_21];
        let published = [
            [1.047_811_2, 0.022_886_6, -0.050_127_0],
            [0.029_542_4, 0.990_484_4, -0.017_049_1],
            [-0.009_234_5, 0.015_043_6, 0.752_131_6],
        ];
        assert_matrix_near(&bradford_adaptation(d65, d50).unwrap(), &published, 1e-7);
        let published_back = [
            [0.955_576_6, -0.023_039_3, 0.063_163_6],
            [-0.028_289_5, 1.009_941_6, 0.021_007_7],
            [0.012_298_2, -0.020_483_0, 1.329_909_8],
        ];
        assert_matrix_near(&bradford_adaptation(d50, d65).unwrap(), &published_back, 1e-7);
    }

    #[test]
    fn adaptation_maps_white_to_white_and_inverts() {
        assert_near(matrix::mul_vec(&D65_TO_D50, D65_XYZ), D50_XYZ, 1e-14);
        assert_near(matrix::mul_vec(&D50_TO_D65, D50_XYZ), D65_XYZ, 1e-14);
        assert_matrix_near(&matrix::mul(&D50_TO_D65, &D65_TO_D50), &matrix::IDENTITY, 1e-14);
        assert_matrix_near(&bradford_adaptation(D65_XYZ, D65_XYZ).unwrap(), &matrix::IDENTITY, 1e-14);
        assert!(bradford_adaptation([0.0, 0.0, 0.0], D65_XYZ).is_none());
        assert!(bradford_adaptation(D65_XYZ, [f64::NAN, 1.0, 1.0]).is_none());
        assert!(bradford_adaptation(D65_XYZ, [-1.0, -1.0, -1.0]).is_none());
        assert!(bradford_adaptation([f64::INFINITY, 1.0, 1.0], D65_XYZ).is_none());
    }

    #[test]
    fn xyy_round_trips_and_handles_black() {
        let mut rng = Rng::new(0x0C1E);
        for _ in 0..2000 {
            let xyz = [rng.range_f64(0.01, 1.2), rng.range_f64(0.01, 1.2), rng.range_f64(0.01, 1.2)];
            assert_near(xyy_to_xyz(xyz_to_xyy(xyz)), xyz, 1e-13);
        }
        assert_eq!(xyz_to_xyy([0.0, 0.0, 0.0]), [D65_XY[0], D65_XY[1], 0.0]);
        assert_eq!(xyz_to_xyy([-1.0, 0.5, 0.2]), [D65_XY[0], D65_XY[1], 0.5]);
        assert_eq!(xyy_to_xyz([0.3, 0.0, 1.0]), [0.0, 0.0, 0.0]);
        assert_eq!(xyy_to_xyz([0.3, -0.2, 1.0]), [0.0, 0.0, 0.0]);
        assert_eq!(xyy_to_xyz([0.3, 1e-320, f64::MAX]), [0.0, 0.0, 0.0]);
        assert_eq!(xy_to_xyz([f64::NAN, 0.3]), [0.0, 0.0, 0.0]);
        assert_eq!(xy_to_xyz([0.3, f64::INFINITY]), [0.0, 0.0, 0.0]);
        assert_near(xy_to_xyz(D50_XY), D50_XYZ, 2e-4);
    }

    #[test]
    fn degenerate_primaries_have_no_matrix() {
        let collinear = RgbPrimaries { red: [0.6, 0.3], green: [0.4, 0.3], blue: [0.2, 0.3], white: D65_XY };
        assert!(rgb_to_xyz_matrix(&collinear).is_none());
        assert!(xyz_to_rgb_matrix(&collinear).is_none());
        assert!(rgb_to_rgb_matrix(&collinear, &REC709).is_none());
        assert!(rgb_to_rgb_matrix(&REC709, &collinear).is_none());
        let zero_y = RgbPrimaries { red: [0.64, 0.0], ..REC709 };
        assert!(rgb_to_xyz_matrix(&zero_y).is_none());
        let no_white = RgbPrimaries { white: [0.3, 0.0], ..REC709 };
        assert!(rgb_to_xyz_matrix(&no_white).is_none());
        // A white point on the line through two primaries needs none of the third.
        let white_on_edge = RgbPrimaries { white: [0.47, 0.465], ..REC709 };
        assert!(rgb_to_xyz_matrix(&white_on_edge).is_none());
        let nan = RgbPrimaries { green: [f64::NAN, 0.6], ..REC709 };
        assert!(rgb_to_xyz_matrix(&nan).is_none());
        assert_eq!(rgb_to_xyz_matrix(&REC709), Some(REC709_TO_XYZ));
    }
}
