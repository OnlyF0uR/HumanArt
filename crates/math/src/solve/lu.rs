//! LU factorisation with partial (row) pivoting, `P·A = L·U`, for general square systems.
//!
//! Used where a matrix is square but not symmetric positive definite: the augmented thin-plate RBF
//! system `[SKIN-02]`, camera and calibration fits, determinants and inverses of small matrices.
//! Elimination works on whole rows, so the inner loop is a contiguous `axpy` `[PERF-R2]`.
//!
//! Singularity detection: a pivot whose magnitude is at most `n · ε · max|aᵢⱼ|` marks the factor
//! singular. A singular factor still reports its [`Lu::determinant`] (the product of the pivots, a
//! value at rounding level or exactly zero); [`Lu::solve`] and [`Lu::inverse`] return
//! [`SolveError::Singular`].

use super::dense::{DenseMatrix, MatRef, swap_rows_in};
use super::error::{SolveError, check_dim, check_finite};
use super::vector::{axpy, dot};

/// An LU factorisation with partial pivoting.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Lu {
    n: usize,
    /// Row-major `n × n`: strict lower triangle = `L` (unit diagonal implied), upper triangle = `U`.
    factors: Vec<f64>,
    /// `perm[i]` is the row of `A` that ended up as row `i` of `P·A`.
    perm: Vec<usize>,
    /// +1 or −1: the sign of the permutation.
    sign: f64,
    singular: bool,
}

impl Lu {
    /// Factors the square matrix `a`.
    ///
    /// Fails only on a non-square or non-finite matrix; a singular matrix gives a factor for which
    /// [`Lu::is_singular`] is `true`.
    pub fn new(a: &DenseMatrix) -> Result<Self, SolveError> {
        let mut lu = Self::default();
        lu.factor(a.view())?;
        Ok(lu)
    }

    /// Replaces the factorisation by that of `a`, reusing the buffers (no allocation once warm).
    ///
    /// On failure the factor is left empty (order 0).
    pub fn factor(&mut self, a: MatRef<'_>) -> Result<(), SolveError> {
        self.n = 0;
        self.singular = false;
        self.sign = 1.0;
        let n = a.rows();
        check_dim("lu: matrix columns", n, a.cols())?;
        check_finite(a.as_slice())?;
        self.factors.clear();
        self.factors.extend_from_slice(a.as_slice());
        self.perm.clear();
        self.perm.extend(0..n);
        let tiny = a.max_abs() * f64::EPSILON * n as f64;
        for k in 0..n {
            // Partial pivoting: the largest magnitude in column k at or below the diagonal.
            let mut pivot_row = k;
            let mut pivot_abs = self.factors[k * n + k].abs();
            for i in k + 1..n {
                let v = self.factors[i * n + k].abs();
                if v > pivot_abs {
                    pivot_abs = v;
                    pivot_row = i;
                }
            }
            if pivot_row != k {
                swap_rows_in(&mut self.factors, n, k, pivot_row);
                self.perm.swap(k, pivot_row);
                self.sign = -self.sign;
            }
            if pivot_abs <= tiny {
                // The column is (numerically) dependent on the earlier ones: nothing to eliminate.
                self.singular = true;
                continue;
            }
            let (head, tail) = self.factors.split_at_mut((k + 1) * n);
            let pivot_row_data = &head[k * n..];
            let inv = 1.0 / pivot_row_data[k];
            for row in tail.chunks_exact_mut(n) {
                let factor = row[k] * inv;
                row[k] = factor;
                if factor != 0.0 {
                    axpy(-factor, &pivot_row_data[k + 1..], &mut row[k + 1..]);
                }
            }
        }
        self.n = n;
        Ok(())
    }

    /// The order `n` of the factored matrix.
    pub fn order(&self) -> usize {
        self.n
    }

    /// `true` when a pivot was zero to working precision: the matrix is singular or so close to it
    /// that a solve would be meaningless.
    pub fn is_singular(&self) -> bool {
        self.singular
    }

    /// The row permutation: row `i` of `P·A` is row `permutation()[i]` of `A`.
    pub fn permutation(&self) -> &[usize] {
        &self.perm
    }

    /// The determinant of `A`: the signed product of the pivots (1 for the empty matrix). Defined for
    /// singular factors too, where it is zero or at rounding level.
    pub fn determinant(&self) -> f64 {
        (0..self.n).fold(self.sign, |p, i| p * self.factors[i * self.n + i])
    }

    /// Solves `A·x = b` into `x`. No allocation.
    pub fn solve_into(&self, b: &[f64], x: &mut [f64]) -> Result<(), SolveError> {
        let n = self.n;
        check_dim("lu solve: rhs length", n, b.len())?;
        check_dim("lu solve: solution length", n, x.len())?;
        check_finite(b)?;
        if self.singular {
            return Err(SolveError::Singular);
        }
        for (xi, &p) in x.iter_mut().zip(&self.perm) {
            *xi = b[p];
        }
        // L·y = P·b (unit diagonal).
        for i in 0..n {
            x[i] -= dot(&self.factors[i * n..i * n + i], &x[..i]);
        }
        // U·x = y.
        for i in (0..n).rev() {
            let row = &self.factors[i * n..(i + 1) * n];
            x[i] = (x[i] - dot(&row[i + 1..], &x[i + 1..])) / row[i];
        }
        Ok(())
    }

    /// Solves `A·x = b` and returns `x`.
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>, SolveError> {
        let mut x = vec![0.0; b.len()];
        self.solve_into(b, &mut x)?;
        Ok(x)
    }

    /// Solves `A·X = B`; the columns of `b` are the right-hand sides.
    pub fn solve_many(&self, b: &DenseMatrix) -> Result<DenseMatrix, SolveError> {
        let n = self.n;
        check_dim("lu solve: rhs rows", n, b.rows())?;
        check_finite(b.as_slice())?;
        if self.singular {
            return Err(SolveError::Singular);
        }
        let k = b.cols();
        let mut x = DenseMatrix::zeros(n, k);
        if k == 0 {
            return Ok(x);
        }
        for (i, &p) in self.perm.iter().enumerate() {
            x.row_mut(i).copy_from_slice(b.row(p));
        }
        let data = x.as_mut_slice();
        for i in 0..n {
            let (above, rest) = data.split_at_mut(i * k);
            let row_i = &mut rest[..k];
            for (j, &lij) in self.factors[i * n..i * n + i].iter().enumerate() {
                if lij != 0.0 {
                    axpy(-lij, &above[j * k..(j + 1) * k], row_i);
                }
            }
        }
        for i in (0..n).rev() {
            let (upto, below) = data.split_at_mut((i + 1) * k);
            let row_i = &mut upto[i * k..];
            let u_row = &self.factors[i * n..(i + 1) * n];
            for (offset, &uij) in u_row[i + 1..].iter().enumerate() {
                if uij != 0.0 {
                    axpy(-uij, &below[offset * k..(offset + 1) * k], row_i);
                }
            }
            let inv = 1.0 / u_row[i];
            for v in row_i.iter_mut() {
                *v *= inv;
            }
        }
        Ok(x)
    }

    /// The inverse `A⁻¹`, or [`SolveError::Singular`].
    pub fn inverse(&self) -> Result<DenseMatrix, SolveError> {
        self.solve_many(&DenseMatrix::identity(self.n))
    }
}

/// Solves the square system `a·x = b` by LU with partial pivoting.
pub fn lu_solve(a: &DenseMatrix, b: &[f64]) -> Result<Vec<f64>, SolveError> {
    Lu::new(a)?.solve(b)
}

/// The determinant of the square matrix `a` (zero or rounding level when singular).
pub fn determinant(a: &DenseMatrix) -> Result<f64, SolveError> {
    Ok(Lu::new(a)?.determinant())
}

/// The inverse of the square matrix `a`, or [`SolveError::Singular`].
pub fn inverse(a: &DenseMatrix) -> Result<DenseMatrix, SolveError> {
    Lu::new(a)?.inverse()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solve::testutil::{max_abs_diff, random_matrix, random_vec, relative_residual};
    use humanart_core::rng::Rng;

    #[test]
    fn ac_wp02_1_known_system_determinant_and_inverse() {
        let a = DenseMatrix::from_rows(&[[2.0, 1.0, 1.0], [4.0, -6.0, 0.0], [-2.0, 7.0, 2.0]]).unwrap();
        let lu = Lu::new(&a).unwrap();
        assert!(!lu.is_singular());
        assert!((lu.determinant() - (-16.0)).abs() < 1e-12);
        let x = lu.solve(&[5.0, -2.0, 9.0]).unwrap();
        for (got, want) in x.iter().zip([1.0, 1.0, 2.0]) {
            assert!((got - want).abs() < 1e-13);
        }
        let inv = lu.inverse().unwrap();
        assert!(max_abs_diff(&a.mul(&inv).unwrap(), &DenseMatrix::identity(3)) < 1e-14);
        assert!((determinant(&a).unwrap() + 16.0).abs() < 1e-12);
        assert_eq!(max_abs_diff(&inverse(&a).unwrap(), &inv), 0.0);
        assert_eq!(lu_solve(&a, &[5.0, -2.0, 9.0]).unwrap(), x);
    }

    #[test]
    fn ac_wp02_1_identity_diagonal_permutation_and_empty() {
        let lu = Lu::new(&DenseMatrix::identity(4)).unwrap();
        assert_eq!(lu.determinant(), 1.0);
        assert_eq!(lu.solve(&[1.0, 2.0, 3.0, 4.0]).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        let lu = Lu::new(&DenseMatrix::from_diagonal(&[2.0, -3.0, 4.0])).unwrap();
        assert_eq!(lu.determinant(), -24.0);
        // A pure row swap needs pivoting and has determinant -1.
        let p = DenseMatrix::from_rows(&[[0.0, 1.0], [1.0, 0.0]]).unwrap();
        let lu = Lu::new(&p).unwrap();
        assert_eq!(lu.determinant(), -1.0);
        assert_eq!(lu.permutation(), &[1, 0]);
        assert_eq!(lu.solve(&[3.0, 5.0]).unwrap(), vec![5.0, 3.0]);
        let empty = Lu::new(&DenseMatrix::zeros(0, 0)).unwrap();
        assert_eq!(empty.determinant(), 1.0);
        assert_eq!(empty.order(), 0);
        assert!(empty.solve(&[]).unwrap().is_empty());
    }

    #[test]
    fn ac_wp02_1_random_systems_have_small_residuals() {
        let mut rng = Rng::new(0x10);
        for n in [1, 2, 5, 16, 50, 120] {
            let mut a = random_matrix(&mut rng, n, n);
            // Diagonal dominance keeps the condition number moderate.
            a.add_to_diagonal(2.0);
            let b = random_vec(&mut rng, n);
            let lu = Lu::new(&a).unwrap();
            let x = lu.solve(&b).unwrap();
            assert!(relative_residual(&a, &x, &b) < 1e-12, "n = {n}");
            let rhs = random_matrix(&mut rng, n, 3);
            let xs = lu.solve_many(&rhs).unwrap();
            assert!(max_abs_diff(&a.mul(&xs).unwrap(), &rhs) < 1e-11, "n = {n}");
            let inv = lu.inverse().unwrap();
            assert!(max_abs_diff(&inv.mul(&a).unwrap(), &DenseMatrix::identity(n)) < 1e-11, "n = {n}");
        }
    }

    #[test]
    fn determinant_is_multiplicative() {
        let mut rng = Rng::new(0x11);
        let a = random_matrix(&mut rng, 6, 6);
        let b = random_matrix(&mut rng, 6, 6);
        let ab = a.mul(&b).unwrap();
        let (da, db, dab) = (determinant(&a).unwrap(), determinant(&b).unwrap(), determinant(&ab).unwrap());
        assert!((da * db - dab).abs() < 1e-12 * (1.0 + dab.abs()));
    }

    #[test]
    fn singular_matrices_are_detected() {
        // Rank 2: the third row is the sum of the first two.
        let a = DenseMatrix::from_rows(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [5.0, 7.0, 9.0]]).unwrap();
        let lu = Lu::new(&a).unwrap();
        assert!(lu.is_singular());
        assert!(lu.determinant().abs() < 1e-12);
        assert_eq!(lu.solve(&[1.0, 2.0, 3.0]), Err(SolveError::Singular));
        assert_eq!(lu.inverse(), Err(SolveError::Singular));
        assert_eq!(lu.solve_many(&DenseMatrix::zeros(3, 1)), Err(SolveError::Singular));
        let zero = Lu::new(&DenseMatrix::zeros(3, 3)).unwrap();
        assert!(zero.is_singular());
        assert_eq!(zero.determinant(), 0.0);
        assert_eq!(inverse(&DenseMatrix::zeros(2, 2)), Err(SolveError::Singular));
    }

    #[test]
    fn bad_input_is_an_error() {
        assert!(matches!(Lu::new(&DenseMatrix::zeros(2, 3)), Err(SolveError::DimensionMismatch { .. })));
        let mut a = DenseMatrix::identity(2);
        a[(0, 1)] = f64::INFINITY;
        assert_eq!(Lu::new(&a), Err(SolveError::NonFinite));
        let lu = Lu::new(&DenseMatrix::identity(2)).unwrap();
        assert!(matches!(lu.solve(&[1.0]), Err(SolveError::DimensionMismatch { .. })));
        assert_eq!(lu.solve(&[1.0, f64::NAN]), Err(SolveError::NonFinite));
        assert!(lu.solve_many(&DenseMatrix::zeros(3, 1)).is_err());
        let mut x = [0.0; 3];
        assert!(lu.solve_into(&[1.0, 2.0], &mut x).is_err());
    }

    #[test]
    fn refactoring_reuses_the_buffers() {
        // [PERF-R2] `factor` and `solve_into` do not allocate once the factor is warm: the buffer
        // pointers are unchanged across refactorisations.
        let mut rng = Rng::new(0x12);
        let mut lu = Lu::default();
        let mut a = random_matrix(&mut rng, 10, 10);
        a.add_to_diagonal(3.0);
        lu.factor(a.view()).unwrap();
        let (p0, p1) = (lu.factors.as_ptr(), lu.perm.as_ptr());
        let mut x = vec![0.0; 10];
        for _ in 0..3 {
            let mut a = random_matrix(&mut rng, 10, 10);
            a.add_to_diagonal(3.0);
            let b = random_vec(&mut rng, 10);
            lu.factor(a.view()).unwrap();
            lu.solve_into(&b, &mut x).unwrap();
            assert!(relative_residual(&a, &x, &b) < 1e-12);
            assert_eq!((lu.factors.as_ptr(), lu.perm.as_ptr()), (p0, p1));
        }
    }
}
