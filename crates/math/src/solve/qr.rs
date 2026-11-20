//! Householder QR with column pivoting, `A·P = Q·R`: least squares and numerical rank.
//!
//! Least squares through QR works on `A` itself, so it keeps full accuracy where the normal equations
//! square the condition number; it is the reference the ridge solver is tested against and the method
//! for regressions with nearly dependent features `[STAT-04]`. Column pivoting makes the diagonal of
//! `R` non-increasing in magnitude, which gives a reliable rank estimate: the rank is the number of
//! diagonal elements above `tolerance · |r₀₀|`.
//!
//! The factor is stored column-major internally (each column of `A` contiguous) so that forming and
//! applying a reflector are a dot product and an `axpy` on contiguous memory `[PERF-R2]`.

use super::dense::DenseMatrix;
use super::error::{SolveError, check_dim, check_finite};
use super::vector::{axpy, dot, norm_sq};

/// A column-pivoted Householder QR factorisation of an `m × n` matrix (any shape).
#[derive(Clone, Debug, PartialEq)]
pub struct Qr {
    m: usize,
    n: usize,
    /// Column-major `m × n`: column `j` is `factors[j*m..(j+1)*m]`. On and above the diagonal: `R`. Below
    /// the diagonal of column `k < min(m, n)`: the tail of the Householder vector `vₖ` (`vₖ[k] = 1`).
    factors: Vec<f64>,
    /// The reflector scales: `Hₖ = I − tau[k]·vₖ·vₖᵀ`.
    tau: Vec<f64>,
    /// Column `j` of `A·P` is column `perm[j]` of `A`.
    perm: Vec<usize>,
}

impl Qr {
    /// Factors `a`. Fails only on non-finite input.
    pub fn new(a: &DenseMatrix) -> Result<Self, SolveError> {
        check_finite(a.as_slice())?;
        let (m, n) = (a.rows(), a.cols());
        let mut qr = a.transpose().into_vec();
        let steps = m.min(n);
        let mut tau = vec![0.0; steps];
        let mut perm: Vec<usize> = (0..n).collect();
        for k in 0..steps {
            // Pivot: the remaining column with the largest norm below row k. The norms are recomputed
            // rather than downdated, which costs the same order as the factorisation and cannot drift.
            let mut best = k;
            let mut best_norm = -1.0;
            for j in k..n {
                let v = norm_sq(&qr[j * m + k..(j + 1) * m]);
                if v > best_norm {
                    best_norm = v;
                    best = j;
                }
            }
            if best != k {
                let (lo, hi) = qr.split_at_mut(best * m);
                lo[k * m..(k + 1) * m].swap_with_slice(&mut hi[..m]);
                perm.swap(k, best);
            }
            if best_norm <= 0.0 {
                // Every remaining column is zero below row k: R is complete, the reflectors are identity.
                break;
            }
            let (done, rest) = qr.split_at_mut((k + 1) * m);
            let column = &mut done[k * m + k..];
            let alpha = column[0];
            let length = best_norm.sqrt();
            let beta = if alpha >= 0.0 { -length } else { length };
            // v = x − beta·e₀, scaled so that v[0] = 1; no cancellation because beta has the sign
            // opposite to alpha.
            let inv = 1.0 / (alpha - beta);
            for v in &mut column[1..] {
                *v *= inv;
            }
            tau[k] = (beta - alpha) / beta;
            column[0] = beta;
            let v_tail = &column[1..];
            for other in rest.chunks_exact_mut(m) {
                let target = &mut other[k..];
                let w = tau[k] * (target[0] + dot(v_tail, &target[1..]));
                target[0] -= w;
                axpy(-w, v_tail, &mut target[1..]);
            }
        }
        Ok(Self { m, n, factors: qr, tau, perm })
    }

    /// The number of rows of the factored matrix.
    pub fn rows(&self) -> usize {
        self.m
    }

    /// The number of columns of the factored matrix.
    pub fn cols(&self) -> usize {
        self.n
    }

    /// The column permutation: column `j` of `A·P` is column `permutation()[j]` of `A`.
    pub fn permutation(&self) -> &[usize] {
        &self.perm
    }

    /// Element `(i, j)` of `R` for `i <= j`.
    #[inline]
    fn r_at(&self, i: usize, j: usize) -> f64 {
        self.factors[j * self.m + i]
    }

    /// The numerical rank with the default tolerance `max(m, n) · ε`.
    pub fn rank(&self) -> usize {
        self.rank_with_tolerance(self.m.max(self.n) as f64 * f64::EPSILON)
    }

    /// The numerical rank: the number of diagonal elements of `R` with `|rᵢᵢ| > tolerance · |r₀₀|`.
    pub fn rank_with_tolerance(&self, tolerance: f64) -> usize {
        let steps = self.m.min(self.n);
        if steps == 0 {
            return 0;
        }
        let largest = self.r_at(0, 0).abs();
        if largest == 0.0 {
            return 0;
        }
        (0..steps).take_while(|&i| self.r_at(i, i).abs() > tolerance * largest).count()
    }

    /// The upper trapezoidal factor `R`, `min(m, n) × n`.
    pub fn r(&self) -> DenseMatrix {
        let steps = self.m.min(self.n);
        DenseMatrix::from_fn(steps, self.n, |i, j| if i <= j { self.r_at(i, j) } else { 0.0 })
    }

    /// The thin orthonormal factor `Q`, `m × min(m, n)`.
    pub fn q(&self) -> DenseMatrix {
        let steps = self.m.min(self.n);
        let mut q = DenseMatrix::zeros(self.m, steps);
        let mut column = vec![0.0; self.m];
        for j in 0..steps {
            column.fill(0.0);
            column[j] = 1.0;
            // Q·eⱼ = H₀·H₁·…·e_j: apply the reflectors in reverse order.
            for k in (0..steps).rev() {
                self.apply_reflector(k, &mut column);
            }
            for (i, &v) in column.iter().enumerate() {
                q[(i, j)] = v;
            }
        }
        q
    }

    /// `x ← Hₖ·x`.
    fn apply_reflector(&self, k: usize, x: &mut [f64]) {
        let tau = self.tau[k];
        if tau == 0.0 {
            return;
        }
        let v_tail = &self.factors[k * self.m + k + 1..(k + 1) * self.m];
        let target = &mut x[k..];
        let w = tau * (target[0] + dot(v_tail, &target[1..]));
        target[0] -= w;
        axpy(-w, v_tail, &mut target[1..]);
    }

    /// `b ← Qᵀ·b` (with the full `m × m` orthogonal `Q`). No allocation.
    pub fn apply_qt_in_place(&self, b: &mut [f64]) -> Result<(), SolveError> {
        check_dim("qr: vector length", self.m, b.len())?;
        for k in 0..self.tau.len() {
            self.apply_reflector(k, b);
        }
        Ok(())
    }

    /// Solves `min ‖A·x − b‖₂` into `x`, using `scratch` (length `m`) for `Qᵀb`. No allocation.
    ///
    /// For a matrix of full column rank this is the unique least-squares solution. For a rank
    /// deficient one (rank `r`), it is the *basic* solution: the `n − r` components belonging to the
    /// columns the pivoting put last are zero, and the others minimise the residual. It is a
    /// minimiser, but not the one of smallest norm; use [`super::svd::Svd::solve_least_squares`] for
    /// that.
    ///
    /// Returns the numerical rank that was used.
    pub fn solve_least_squares_into(&self, b: &[f64], x: &mut [f64], scratch: &mut [f64]) -> Result<usize, SolveError> {
        check_dim("least squares: rhs length", self.m, b.len())?;
        check_dim("least squares: solution length", self.n, x.len())?;
        check_dim("least squares: scratch length", self.m, scratch.len())?;
        check_finite(b)?;
        scratch.copy_from_slice(b);
        self.apply_qt_in_place(scratch)?;
        let rank = self.rank();
        // Back substitution on the leading rank × rank block of R, column-oriented (columns of R are
        // contiguous).
        let y = &mut scratch[..rank];
        for i in (0..rank).rev() {
            let column = &self.factors[i * self.m..i * self.m + i];
            let yi = y[i] / self.r_at(i, i);
            y[i] = yi;
            axpy(-yi, column, &mut y[..i]);
        }
        x.fill(0.0);
        for (i, &yi) in y.iter().enumerate() {
            x[self.perm[i]] = yi;
        }
        Ok(rank)
    }

    /// Solves `min ‖A·x − b‖₂`; see [`Qr::solve_least_squares_into`] for the rank-deficient case.
    pub fn solve_least_squares(&self, b: &[f64]) -> Result<Vec<f64>, SolveError> {
        let mut x = vec![0.0; self.n];
        let mut scratch = vec![0.0; self.m];
        self.solve_least_squares_into(b, &mut x, &mut scratch)?;
        Ok(x)
    }
}

/// Solves `min ‖a·x − b‖₂` by column-pivoted QR (basic solution when `a` is rank deficient).
pub fn least_squares(a: &DenseMatrix, b: &[f64]) -> Result<Vec<f64>, SolveError> {
    Qr::new(a)?.solve_least_squares(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solve::testutil::{max_abs_diff, orthonormality_error, random_matrix, random_vec};
    use crate::solve::vector::norm;
    use humanart_core::rng::Rng;

    /// `A·P` rebuilt from the factors.
    fn rebuild(qr: &Qr) -> DenseMatrix {
        let qr_product = qr.q().mul(&qr.r()).unwrap();
        // Undo the permutation: column perm[j] of A is column j of Q·R.
        let mut a = DenseMatrix::zeros(qr.rows(), qr.cols());
        for (j, &p) in qr.permutation().iter().enumerate() {
            for i in 0..qr.rows() {
                a[(i, p)] = qr_product[(i, j)];
            }
        }
        a
    }

    #[test]
    fn ac_wp02_1_factors_reproduce_the_matrix_for_every_shape() {
        let mut rng = Rng::new(0x20);
        for (m, n) in [(1, 1), (4, 4), (7, 3), (3, 7), (20, 11), (5, 1), (1, 5)] {
            let a = random_matrix(&mut rng, m, n);
            let qr = Qr::new(&a).unwrap();
            assert!(orthonormality_error(&qr.q()) < 1e-14, "{m}x{n}");
            assert!(max_abs_diff(&rebuild(&qr), &a) < 1e-14, "{m}x{n}");
            assert_eq!(qr.rank(), m.min(n), "{m}x{n}");
            // R is upper trapezoidal with a non-increasing diagonal.
            let r = qr.r();
            for i in 1..r.rows() {
                assert!(r[(i, i)].abs() <= r[(i - 1, i - 1)].abs() + 1e-14);
            }
        }
    }

    #[test]
    fn ac_wp02_1_exact_line_fit() {
        // y = 2 + 3t sampled without noise: the fit recovers the coefficients.
        let t = [0.0, 1.0, 2.0, 3.0, 4.0];
        let a = DenseMatrix::from_fn(5, 2, |i, j| if j == 0 { 1.0 } else { t[i] });
        let b: Vec<f64> = t.iter().map(|t| 2.0 + 3.0 * t).collect();
        let x = least_squares(&a, &b).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-13 && (x[1] - 3.0).abs() < 1e-13);
    }

    #[test]
    fn ac_wp02_1_least_squares_residual_is_orthogonal_to_the_columns() {
        let mut rng = Rng::new(0x21);
        for (m, n) in [(12, 4), (40, 15), (9, 9)] {
            let a = random_matrix(&mut rng, m, n);
            let b = random_vec(&mut rng, m);
            let x = least_squares(&a, &b).unwrap();
            let ax = a.mul_vec(&x).unwrap();
            let r: Vec<f64> = ax.iter().zip(&b).map(|(p, q)| p - q).collect();
            // The normal equations Aᵀ(Ax − b) = 0 characterise the minimiser.
            let g = a.tr_mul_vec(&r).unwrap();
            assert!(norm(&g) < 1e-12 * norm(&b) * a.frobenius_norm(), "{m}x{n}: gradient {}", norm(&g));
        }
    }

    #[test]
    fn ac_wp02_1_rank_deficiency_is_detected_and_solved() {
        let mut rng = Rng::new(0x22);
        // A 10 × 6 matrix of rank 3: the product of 10 × 3 and 3 × 6 factors.
        let left = random_matrix(&mut rng, 10, 3);
        let right = random_matrix(&mut rng, 3, 6);
        let a = left.mul(&right).unwrap();
        let qr = Qr::new(&a).unwrap();
        assert_eq!(qr.rank(), 3);
        // A consistent right-hand side is reproduced exactly by the basic solution, which has at
        // most `rank` non-zero components.
        let b = a.mul_vec(&random_vec(&mut rng, 6)).unwrap();
        let x = qr.solve_least_squares(&b).unwrap();
        let ax = a.mul_vec(&x).unwrap();
        let err: Vec<f64> = ax.iter().zip(&b).map(|(p, q)| p - q).collect();
        assert!(norm(&err) < 1e-12 * norm(&b));
        assert_eq!(x.iter().filter(|v| **v != 0.0).count(), 3);
        // The zero matrix has rank 0 and the zero solution.
        let zero = Qr::new(&DenseMatrix::zeros(4, 3)).unwrap();
        assert_eq!(zero.rank(), 0);
        assert_eq!(zero.solve_least_squares(&[1.0, 2.0, 3.0, 4.0]).unwrap(), vec![0.0; 3]);
        // Two equal columns.
        let a = DenseMatrix::from_rows(&[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]).unwrap();
        assert_eq!(Qr::new(&a).unwrap().rank(), 1);
    }

    #[test]
    fn underdetermined_systems_get_a_solution_that_fits() {
        let mut rng = Rng::new(0x23);
        let a = random_matrix(&mut rng, 3, 8);
        let b = random_vec(&mut rng, 3);
        let x = least_squares(&a, &b).unwrap();
        let ax = a.mul_vec(&x).unwrap();
        for i in 0..3 {
            assert!((ax[i] - b[i]).abs() < 1e-13);
        }
    }

    #[test]
    fn bad_input_is_an_error() {
        let mut a = DenseMatrix::zeros(2, 2);
        a[(1, 1)] = f64::NAN;
        assert_eq!(Qr::new(&a).unwrap_err(), SolveError::NonFinite);
        let qr = Qr::new(&DenseMatrix::identity(3)).unwrap();
        assert!(matches!(qr.solve_least_squares(&[1.0]), Err(SolveError::DimensionMismatch { .. })));
        assert_eq!(qr.solve_least_squares(&[1.0, f64::NAN, 0.0]), Err(SolveError::NonFinite));
        assert!(qr.apply_qt_in_place(&mut [0.0; 2]).is_err());
        let mut x = [0.0; 3];
        assert!(qr.solve_least_squares_into(&[0.0; 3], &mut x, &mut [0.0; 2]).is_err());
        let empty = Qr::new(&DenseMatrix::zeros(0, 0)).unwrap();
        assert_eq!(empty.rank(), 0);
        assert!(empty.solve_least_squares(&[]).unwrap().is_empty());
    }
}
