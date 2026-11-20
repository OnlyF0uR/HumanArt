//! 3×3 matrix helpers for colour transforms.
//!
//! The colour module stands alone (it does not use the crate's linear-algebra types), so the few
//! matrix operations it needs live here. Everything is a `const fn`, which lets the RGB↔XYZ and
//! chromatic-adaptation matrices be derived from chromaticities at compile time.

/// A 3×3 matrix, row-major: `m[row][column]`. A colour transforms as the column vector `M · c`.
pub type ColorMatrix = [[f64; 3]; 3];

/// The identity matrix.
pub const IDENTITY: ColorMatrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// The matrix product `a · b` (apply `b` first, then `a`).
pub const fn mul(a: &ColorMatrix, b: &ColorMatrix) -> ColorMatrix {
    let mut out = [[0.0; 3]; 3];
    let mut r = 0;
    while r < 3 {
        let mut c = 0;
        while c < 3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
            c += 1;
        }
        r += 1;
    }
    out
}

/// The product `m · v` of a matrix and a column vector.
#[inline]
pub const fn mul_vec(m: &ColorMatrix, v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// The transpose of `m`.
pub const fn transpose(m: &ColorMatrix) -> ColorMatrix {
    [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]]
}

/// The diagonal matrix with `d` on its diagonal.
pub const fn diagonal(d: [f64; 3]) -> ColorMatrix {
    [[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]]
}

/// The determinant of `m`.
pub const fn determinant(m: &ColorMatrix) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// The inverse of `m` by the adjugate, or `None` when `m` is singular, numerically singular
/// (determinant not above `1e-13` times the cube of the largest entry) or not finite.
pub const fn invert(m: &ColorMatrix) -> Option<ColorMatrix> {
    let mut scale: f64 = 0.0;
    let mut r = 0;
    while r < 3 {
        let mut c = 0;
        while c < 3 {
            let a = m[r][c].abs();
            if !a.is_finite() {
                return None;
            }
            if a > scale {
                scale = a;
            }
            c += 1;
        }
        r += 1;
    }
    let det = determinant(m);
    let threshold = scale * scale * scale * 1e-13;
    if !det.is_finite() || det.abs() <= threshold {
        return None;
    }
    let inv = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv,
        ],
    ])
}

/// Widens an `f32` triple to `f64`.
#[inline]
pub const fn widen(v: [f32; 3]) -> [f64; 3] {
    [v[0] as f64, v[1] as f64, v[2] as f64]
}

/// Narrows an `f64` triple to `f32` (round to nearest).
#[inline]
pub const fn narrow(v: [f64; 3]) -> [f32; 3] {
    [v[0] as f32, v[1] as f32, v[2] as f32]
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    fn max_abs_diff(a: &ColorMatrix, b: &ColorMatrix) -> f64 {
        let mut worst: f64 = 0.0;
        for (ra, rb) in a.iter().zip(b) {
            for (x, y) in ra.iter().zip(rb) {
                worst = worst.max((x - y).abs());
            }
        }
        worst
    }

    #[test]
    fn identity_is_neutral() {
        let m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]];
        assert_eq!(mul(&m, &IDENTITY), m);
        assert_eq!(mul(&IDENTITY, &m), m);
        assert_eq!(mul_vec(&IDENTITY, [0.25, -3.0, 9.0]), [0.25, -3.0, 9.0]);
        assert_eq!(determinant(&IDENTITY), 1.0);
    }

    #[test]
    fn product_and_transpose_follow_the_row_major_convention() {
        let a = [[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        // Row 0 of `a` mixes the second component into the first.
        assert_eq!(mul_vec(&a, [10.0, 1.0, 5.0]), [12.0, 1.0, 5.0]);
        assert_eq!(transpose(&a), [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let b = diagonal([2.0, 3.0, 4.0]);
        // (a · b) · v == a · (b · v)
        let v = [1.0, 1.0, 1.0];
        assert_eq!(mul_vec(&mul(&a, &b), v), mul_vec(&a, mul_vec(&b, v)));
    }

    #[test]
    fn inverse_of_random_matrices_multiplies_to_identity() {
        let mut rng = Rng::new(0xC010);
        let mut tested = 0;
        for _ in 0..2000 {
            let mut m = [[0.0; 3]; 3];
            for row in &mut m {
                for v in row.iter_mut() {
                    *v = rng.range_f64(-2.0, 2.0);
                }
            }
            if determinant(&m).abs() < 0.05 {
                continue;
            }
            let inv = invert(&m).unwrap();
            assert!(max_abs_diff(&mul(&m, &inv), &IDENTITY) < 1e-10);
            assert!(max_abs_diff(&mul(&inv, &m), &IDENTITY) < 1e-10);
            tested += 1;
        }
        assert!(tested > 1000);
    }

    #[test]
    fn singular_and_non_finite_matrices_have_no_inverse() {
        assert!(invert(&[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 1.0, 5.0]]).is_none());
        assert!(invert(&[[0.0; 3]; 3]).is_none());
        assert!(invert(&[[f64::NAN, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).is_none());
        assert!(invert(&[[f64::INFINITY, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).is_none());
        // A tiny but well-conditioned matrix is still invertible: the threshold is relative.
        let tiny = diagonal([1e-30, 1e-30, 1e-30]);
        let inv = invert(&tiny).unwrap();
        assert!((inv[0][0] - 1e30).abs() < 1e16);
    }

    #[test]
    fn widen_and_narrow_round_trip() {
        let v = [0.25_f32, 1.5, -3.0];
        assert_eq!(narrow(widen(v)), v);
    }
}
