//! Dense Cholesky factorisation `A = L·Lᵀ` of symmetric positive definite matrices, and the
//! pivot-free `A = L·D·Lᵀ` for symmetric matrices with non-zero leading minors.
//!
//! This is the solver of the landmark drag `[LM-01]`, of the small systems of registration `[REG-C2]`
//! and of every normal-equation solve in this module (ridge regression, Levenberg-Marquardt, RBF).
//! The factorisation is the row-oriented (Cholesky-Banachiewicz) form: with row-major storage its
//! inner loop is a dot product of two contiguous row prefixes `[PERF-R2]`.
//!
//! Only the lower triangle of the input is read. A pivot that is not strictly positive is reported as
//! [`SolveError::NotPositiveDefinite`]; NaN or infinite input as [`SolveError::NonFinite`].
//!
//! Allocation: [`Cholesky::factor`] reuses the factor's buffer and [`Cholesky::solve_into`] /
//! [`Cholesky::solve_in_place`] use none, so a [`Cholesky`] kept between drags makes the per-drag solve
//! allocation-free `[LM-03]`. [`cholesky_solve_into`] is the same with a caller-provided scratch slice.

use super::dense::{DenseMatrix, MatRef};
use super::error::{SolveError, check_dim, check_finite};
use super::vector::{axpy, dot};

/// Factors the symmetric positive definite `n × n` matrix stored row-major in `a`, in place.
///
/// On success the lower triangle of `a` (diagonal included) holds `L`; the strict upper triangle is
/// neither read nor written. On failure the content of `a` is unspecified.
pub fn cholesky_in_place(a: &mut [f64], n: usize) -> Result<(), SolveError> {
    check_dim("cholesky: matrix data length", n.saturating_mul(n), a.len())?;
    for i in 0..n {
        let (head, tail) = a.split_at_mut(i * n);
        let row_i = &mut tail[..n];
        for j in 0..i {
            let row_j = &head[j * n..=j * n + j];
            row_i[j] = (row_i[j] - dot(&row_i[..j], &row_j[..j])) / row_j[j];
        }
        let d = row_i[i] - dot(&row_i[..i], &row_i[..i]);
        if !d.is_finite() {
            return Err(SolveError::NonFinite);
        }
        if d <= 0.0 {
            return Err(SolveError::NotPositiveDefinite { pivot: i });
        }
        row_i[i] = d.sqrt();
    }
    Ok(())
}

/// Solves `L·y = b` in place (`b` becomes `y`), `l` holding `L` as written by [`cholesky_in_place`].
fn forward_substitute(l: &[f64], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let row = &l[i * n..=i * n + i];
        b[i] = (b[i] - dot(&row[..i], &b[..i])) / row[i];
    }
}

/// Solves `Lᵀ·x = y` in place (`y` becomes `x`). Column-oriented, so that the inner loop is an `axpy`
/// with a contiguous row of `L`.
fn backward_substitute(l: &[f64], n: usize, y: &mut [f64]) {
    for i in (0..n).rev() {
        let row = &l[i * n..=i * n + i];
        let xi = y[i] / row[i];
        y[i] = xi;
        axpy(-xi, &row[..i], &mut y[..i]);
    }
}

/// Solves `A·x = b` with a factor written by [`cholesky_in_place`]; `b` is overwritten with `x`.
pub fn cholesky_solve_in_place(l: &[f64], n: usize, b: &mut [f64]) -> Result<(), SolveError> {
    check_dim("cholesky solve: factor data length", n.saturating_mul(n), l.len())?;
    check_dim("cholesky solve: rhs length", n, b.len())?;
    forward_substitute(l, n, b);
    backward_substitute(l, n, b);
    Ok(())
}

/// Factors `a` and solves `a·x = b` without allocating: the factor is built in `scratch`, which must
/// hold at least `n²` values (only its first `n²` are used).
///
/// `a` is symmetric positive definite; only its lower triangle is read. This is the form for a caller
/// that owns all of its buffers; a kept [`Cholesky`] does the same with less ceremony.
pub fn cholesky_solve_into(a: MatRef<'_>, b: &[f64], x: &mut [f64], scratch: &mut [f64]) -> Result<(), SolveError> {
    let n = a.rows();
    check_dim("cholesky: matrix columns", n, a.cols())?;
    check_dim("cholesky solve: rhs length", n, b.len())?;
    check_dim("cholesky solve: solution length", n, x.len())?;
    if scratch.len() < n * n {
        return Err(SolveError::DimensionMismatch {
            what: "cholesky solve: scratch length",
            expected: n * n,
            found: scratch.len(),
        });
    }
    check_finite(b)?;
    let l = &mut scratch[..n * n];
    l.copy_from_slice(a.as_slice());
    cholesky_in_place(l, n)?;
    x.copy_from_slice(b);
    forward_substitute(l, n, x);
    backward_substitute(l, n, x);
    Ok(())
}

/// A Cholesky factor `L` (lower triangular, positive diagonal) with `A = L·Lᵀ`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Cholesky {
    n: usize,
    /// Row-major `n × n`; the strict upper triangle is zero.
    l: Vec<f64>,
}

impl Cholesky {
    /// Factors the symmetric positive definite matrix `a` (only its lower triangle is read).
    pub fn new(a: &DenseMatrix) -> Result<Self, SolveError> {
        let mut factor = Self::default();
        factor.factor(a.view())?;
        Ok(factor)
    }

    /// An empty factor (order 0) whose buffer can already hold a matrix of order `n`, so that the
    /// first [`Cholesky::factor`] of that size does not allocate.
    pub fn with_capacity(n: usize) -> Self {
        Self { n: 0, l: Vec::with_capacity(n.saturating_mul(n)) }
    }

    /// Replaces the factor by that of `a`, reusing the buffer: no allocation when the buffer has held
    /// a matrix of this order before (or was made by [`Cholesky::with_capacity`]).
    ///
    /// On failure the factor is left empty (order 0).
    pub fn factor(&mut self, a: MatRef<'_>) -> Result<(), SolveError> {
        self.n = 0;
        let n = a.rows();
        check_dim("cholesky: matrix columns", n, a.cols())?;
        self.l.clear();
        self.l.extend_from_slice(a.as_slice());
        // Only the lower triangle is input: the upper one is cleared before the finiteness check.
        for i in 0..n {
            self.l[i * n + i + 1..(i + 1) * n].fill(0.0);
        }
        check_finite(&self.l)?;
        cholesky_in_place(&mut self.l, n)?;
        self.n = n;
        Ok(())
    }

    /// The order `n` of the factored matrix (0 for an empty factor).
    pub fn order(&self) -> usize {
        self.n
    }

    /// The factor `L` as an `n × n` view (its strict upper triangle is zero).
    pub fn l(&self) -> MatRef<'_> {
        MatRef::from_parts(self.n, self.n, &self.l[..self.n * self.n])
    }

    /// Solves `A·x = b`; `b` is overwritten with `x`. No allocation.
    pub fn solve_in_place(&self, b: &mut [f64]) -> Result<(), SolveError> {
        check_dim("cholesky solve: rhs length", self.n, b.len())?;
        check_finite(b)?;
        forward_substitute(&self.l, self.n, b);
        backward_substitute(&self.l, self.n, b);
        Ok(())
    }

    /// Solves `A·x = b` into `x`. No allocation.
    pub fn solve_into(&self, b: &[f64], x: &mut [f64]) -> Result<(), SolveError> {
        check_dim("cholesky solve: rhs length", self.n, b.len())?;
        check_dim("cholesky solve: solution length", self.n, x.len())?;
        x.copy_from_slice(b);
        self.solve_in_place(x)
    }

    /// Solves `A·x = b` and returns `x`.
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>, SolveError> {
        let mut x = b.to_vec();
        self.solve_in_place(&mut x)?;
        Ok(x)
    }

    /// Solves `A·X = B` for many right-hand sides at once: the columns of `b` are the right-hand
    /// sides and are overwritten with the solutions. No allocation.
    pub fn solve_many_in_place(&self, b: &mut DenseMatrix) -> Result<(), SolveError> {
        let n = self.n;
        check_dim("cholesky solve: rhs rows", n, b.rows())?;
        check_finite(b.as_slice())?;
        let k = b.cols();
        if k == 0 {
            return Ok(());
        }
        let data = b.as_mut_slice();
        // L·Y = B, row by row: every operation is on whole (contiguous) rows of B.
        for i in 0..n {
            let (above, rest) = data.split_at_mut(i * k);
            let row_i = &mut rest[..k];
            let l_row = &self.l[i * n..=i * n + i];
            for (j, &lij) in l_row[..i].iter().enumerate() {
                if lij != 0.0 {
                    axpy(-lij, &above[j * k..(j + 1) * k], row_i);
                }
            }
            let inv = 1.0 / l_row[i];
            for v in row_i.iter_mut() {
                *v *= inv;
            }
        }
        // Lᵀ·X = Y, column-oriented in L.
        for i in (0..n).rev() {
            let (above, rest) = data.split_at_mut(i * k);
            let row_i = &mut rest[..k];
            let l_row = &self.l[i * n..=i * n + i];
            let inv = 1.0 / l_row[i];
            for v in row_i.iter_mut() {
                *v *= inv;
            }
            for (j, &lij) in l_row[..i].iter().enumerate() {
                if lij != 0.0 {
                    axpy(-lij, row_i, &mut above[j * k..(j + 1) * k]);
                }
            }
        }
        Ok(())
    }

    /// Solves `A·X = B`; the columns of `b` are the right-hand sides.
    pub fn solve_many(&self, b: &DenseMatrix) -> Result<DenseMatrix, SolveError> {
        let mut x = b.clone();
        self.solve_many_in_place(&mut x)?;
        Ok(x)
    }

    /// Solves `L·y = b` only (`b` is overwritten with `y`): `‖y‖²` is then `bᵀA⁻¹b`, the squared
    /// Mahalanobis length of `b` under the covariance `A`. No allocation.
    pub fn forward_in_place(&self, b: &mut [f64]) -> Result<(), SolveError> {
        check_dim("cholesky forward: rhs length", self.n, b.len())?;
        forward_substitute(&self.l, self.n, b);
        Ok(())
    }

    /// The inverse `A⁻¹` (symmetric).
    pub fn inverse(&self) -> DenseMatrix {
        let mut inv = DenseMatrix::identity(self.n);
        // The identity has the right shape and is finite, so the solve cannot fail; if it ever did
        // the caller would see the identity, which no test of the inverse can mistake for one.
        if self.solve_many_in_place(&mut inv).is_ok() {
            // Symmetrise the rounding noise.
            let n = self.n;
            let data = inv.as_mut_slice();
            for i in 0..n {
                for j in 0..i {
                    let mean = f64::midpoint(data[i * n + j], data[j * n + i]);
                    data[i * n + j] = mean;
                    data[j * n + i] = mean;
                }
            }
        }
        inv
    }

    /// The determinant of `A`, `Π lᵢᵢ²`. Overflows or underflows for large orders: use
    /// [`Cholesky::log_determinant`] there.
    pub fn determinant(&self) -> f64 {
        (0..self.n).map(|i| self.l[i * self.n + i]).fold(1.0, |p, d| p * d * d)
    }

    /// The natural logarithm of the determinant of `A`, `2·Σ ln lᵢᵢ`.
    pub fn log_determinant(&self) -> f64 {
        2.0 * (0..self.n).map(|i| self.l[i * self.n + i].ln()).sum::<f64>()
    }
}

/// A pivot-free factorisation `A = L·D·Lᵀ` with unit lower triangular `L` and diagonal `D`.
///
/// It needs no square roots and accepts symmetric matrices that are not positive definite, provided
/// every leading principal minor is non-zero (negative definite matrices, symmetric quasi-definite
/// saddle-point systems). Because there is no pivoting it is **not** stable for general indefinite
/// matrices: a pivot that is zero to working precision is reported as [`SolveError::Singular`], and
/// for anything else that is merely indefinite use [`super::lu::Lu`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ldlt {
    n: usize,
    /// Row-major `n × n`: strict lower triangle = `L`, diagonal = `D`, strict upper triangle zero.
    ld: Vec<f64>,
}

impl Ldlt {
    /// Factors the symmetric matrix `a` (only its lower triangle is read).
    pub fn new(a: &DenseMatrix) -> Result<Self, SolveError> {
        let n = a.rows();
        check_dim("ldlt: matrix columns", n, a.cols())?;
        let mut ld = a.as_slice().to_vec();
        for i in 0..n {
            ld[i * n + i + 1..(i + 1) * n].fill(0.0);
        }
        check_finite(&ld)?;
        let scale = ld.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let tiny = scale * f64::EPSILON * n as f64;
        // Row-oriented: row i of L needs rows j < i and the pivots d_j.
        for i in 0..n {
            let (head, tail) = ld.split_at_mut(i * n);
            let row_i = &mut tail[..n];
            for j in 0..i {
                let row_j = &head[j * n..=j * n + j];
                // row_i[..j] still holds L_ik·d_k (the division is deferred to the end of the row).
                row_i[j] -= dot(&row_i[..j], &row_j[..j]);
            }
            let mut d = row_i[i];
            for j in 0..i {
                let dj = head[j * n + j];
                let scaled = row_i[j];
                let lij = scaled / dj;
                d -= scaled * lij;
                row_i[j] = lij;
            }
            if !d.is_finite() {
                return Err(SolveError::NonFinite);
            }
            if d.abs() <= tiny {
                return Err(SolveError::Singular);
            }
            row_i[i] = d;
        }
        Ok(Self { n, ld })
    }

    /// The order `n` of the factored matrix.
    pub fn order(&self) -> usize {
        self.n
    }

    /// The diagonal `D` copied into `out`.
    pub fn diagonal_into(&self, out: &mut [f64]) -> Result<(), SolveError> {
        check_dim("ldlt: diagonal length", self.n, out.len())?;
        for (i, o) in out.iter_mut().enumerate() {
            *o = self.ld[i * self.n + i];
        }
        Ok(())
    }

    /// The number of negative pivots, which by Sylvester's law of inertia is the number of negative
    /// eigenvalues of `A`.
    pub fn negative_pivots(&self) -> usize {
        (0..self.n).filter(|&i| self.ld[i * self.n + i] < 0.0).count()
    }

    /// Solves `A·x = b`; `b` is overwritten with `x`. No allocation.
    pub fn solve_in_place(&self, b: &mut [f64]) -> Result<(), SolveError> {
        let n = self.n;
        check_dim("ldlt solve: rhs length", n, b.len())?;
        check_finite(b)?;
        for i in 0..n {
            let row = &self.ld[i * n..i * n + i];
            b[i] -= dot(row, &b[..i]);
        }
        for (i, bi) in b.iter_mut().enumerate() {
            *bi /= self.ld[i * n + i];
        }
        for i in (0..n).rev() {
            let row = &self.ld[i * n..i * n + i];
            let xi = b[i];
            axpy(-xi, row, &mut b[..i]);
        }
        Ok(())
    }

    /// Solves `A·x = b` and returns `x`.
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>, SolveError> {
        let mut x = b.to_vec();
        self.solve_in_place(&mut x)?;
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solve::testutil::{random_spd, relative_residual};
    use crate::solve::vector::{norm, sub_into};
    use humanart_core::rng::Rng;

    fn hilbert(n: usize) -> DenseMatrix {
        DenseMatrix::from_fn(n, n, |i, j| 1.0 / (i + j + 1) as f64)
    }

    #[test]
    fn ac_wp02_1_factor_of_a_known_matrix() {
        // The classic example: A = [[4,12,-16],[12,37,-43],[-16,-43,98]] has L = [[2],[6,1],[-8,5,3]].
        let a = DenseMatrix::from_rows(&[[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]]).unwrap();
        let c = Cholesky::new(&a).unwrap();
        assert_eq!(c.l().as_slice(), &[2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0]);
        assert_eq!(c.order(), 3);
        assert!((c.determinant() - 36.0).abs() < 1e-12);
        assert!((c.log_determinant() - 36.0_f64.ln()).abs() < 1e-12);
        let x = c.solve(&[1.0, 2.0, 3.0]).unwrap();
        assert!(relative_residual(&a, &x, &[1.0, 2.0, 3.0]) < 1e-12);
    }

    #[test]
    fn ac_wp02_1_identity_and_diagonal() {
        let c = Cholesky::new(&DenseMatrix::identity(5)).unwrap();
        assert_eq!(c.solve(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let d = DenseMatrix::from_diagonal(&[4.0, 9.0, 16.0]);
        let c = Cholesky::new(&d).unwrap();
        assert_eq!(c.solve(&[4.0, 9.0, 16.0]).unwrap(), vec![1.0, 1.0, 1.0]);
        assert_eq!(c.inverse().as_slice(), &[0.25, 0.0, 0.0, 0.0, 1.0 / 9.0, 0.0, 0.0, 0.0, 0.0625]);
        let empty = Cholesky::new(&DenseMatrix::zeros(0, 0)).unwrap();
        assert_eq!(empty.solve(&[]).unwrap(), Vec::<f64>::new());
        assert_eq!(empty.determinant(), 1.0);
    }

    #[test]
    fn ac_wp02_1_hilbert_matrices_solve_to_the_accuracy_their_condition_allows() {
        // Hilbert matrices are SPD with condition numbers 5e5 (n = 5) and 5e8 (n = 7). The residual
        // stays at rounding level regardless (Cholesky is backward stable); the solution error grows
        // with the condition number.
        for (n, error_bound) in [(3, 1e-13), (5, 1e-10), (7, 1e-7)] {
            let a = hilbert(n);
            let expect = vec![1.0; n];
            let b = a.mul_vec(&expect).unwrap();
            let x = Cholesky::new(&a).unwrap().solve(&b).unwrap();
            assert!(relative_residual(&a, &x, &b) < 1e-14, "n = {n}");
            let mut diff = vec![0.0; n];
            sub_into(&x, &expect, &mut diff);
            assert!(norm(&diff) / norm(&expect) < error_bound, "n = {n}: error {}", norm(&diff));
        }
    }

    #[test]
    fn ac_wp02_1_random_spd_systems_have_small_residuals() {
        let mut rng = Rng::new(0xC0DE);
        for n in [1, 2, 3, 8, 17, 64, 150] {
            let a = random_spd(&mut rng, n);
            let b: Vec<f64> = (0..n).map(|_| rng.range_f64(-1.0, 1.0)).collect();
            let c = Cholesky::new(&a).unwrap();
            let x = c.solve(&b).unwrap();
            assert!(relative_residual(&a, &x, &b) < 1e-12, "n = {n}");
            // L·Lᵀ reproduces A.
            let l = c.l().to_owned();
            let llt = l.mul(&l.transpose()).unwrap();
            let mut diff = llt;
            diff.add_scaled(-1.0, &a).unwrap();
            assert!(diff.max_abs() < 1e-12 * a.max_abs(), "n = {n}");
        }
    }

    #[test]
    fn many_right_hand_sides_and_inverse() {
        let mut rng = Rng::new(77);
        let n = 12;
        let a = random_spd(&mut rng, n);
        let c = Cholesky::new(&a).unwrap();
        let b = DenseMatrix::from_fn(n, 5, |_, _| rng.range_f64(-1.0, 1.0));
        let x = c.solve_many(&b).unwrap();
        let mut diff = a.mul(&x).unwrap();
        diff.add_scaled(-1.0, &b).unwrap();
        assert!(diff.max_abs() < 1e-12);
        // Each column equals the single right-hand-side solve.
        for col in 0..5 {
            let single = c.solve(&b.col(col).unwrap()).unwrap();
            let many = x.col(col).unwrap();
            for i in 0..n {
                assert!((single[i] - many[i]).abs() < 1e-13);
            }
        }
        let inv = c.inverse();
        assert!(inv.is_symmetric(0.0));
        let mut eye = a.mul(&inv).unwrap();
        eye.add_scaled(-1.0, &DenseMatrix::identity(n)).unwrap();
        assert!(eye.max_abs() < 1e-12);
        assert!(c.solve_many(&DenseMatrix::zeros(n + 1, 2)).is_err());
        assert_eq!(c.solve_many(&DenseMatrix::zeros(n, 0)).unwrap().cols(), 0);
    }

    #[test]
    fn not_positive_definite_is_reported_with_its_pivot() {
        // Indefinite: the second leading minor is negative.
        let a = DenseMatrix::from_rows(&[[1.0, 2.0], [2.0, 1.0]]).unwrap();
        assert_eq!(Cholesky::new(&a), Err(SolveError::NotPositiveDefinite { pivot: 1 }));
        // Semi-definite (rank 1).
        let a = DenseMatrix::from_rows(&[[1.0, 1.0], [1.0, 1.0]]).unwrap();
        assert_eq!(Cholesky::new(&a), Err(SolveError::NotPositiveDefinite { pivot: 1 }));
        let a = DenseMatrix::from_rows(&[[-1.0]]).unwrap();
        assert_eq!(Cholesky::new(&a), Err(SolveError::NotPositiveDefinite { pivot: 0 }));
        let a = DenseMatrix::from_rows(&[[0.0]]).unwrap();
        assert_eq!(Cholesky::new(&a), Err(SolveError::NotPositiveDefinite { pivot: 0 }));
    }

    #[test]
    fn bad_input_is_an_error() {
        assert!(matches!(Cholesky::new(&DenseMatrix::zeros(2, 3)), Err(SolveError::DimensionMismatch { .. })));
        let mut a = DenseMatrix::identity(3);
        a[(2, 0)] = f64::NAN;
        assert_eq!(Cholesky::new(&a), Err(SolveError::NonFinite));
        // The strict upper triangle is not input: a NaN there is ignored.
        let mut a = DenseMatrix::identity(3);
        a[(0, 2)] = f64::NAN;
        assert!(Cholesky::new(&a).is_ok());
        let c = Cholesky::new(&DenseMatrix::identity(3)).unwrap();
        assert!(matches!(c.solve(&[1.0, 2.0]), Err(SolveError::DimensionMismatch { .. })));
        assert_eq!(c.solve(&[1.0, f64::INFINITY, 0.0]), Err(SolveError::NonFinite));
        let mut x = [0.0; 2];
        assert!(c.solve_into(&[1.0, 2.0, 3.0], &mut x).is_err());
        let mut raw = [1.0, 0.0, 0.0];
        assert!(cholesky_in_place(&mut raw, 2).is_err());
        assert!(cholesky_solve_in_place(&[1.0], 1, &mut [1.0, 2.0]).is_err());
        let mut inf = [f64::INFINITY];
        assert_eq!(cholesky_in_place(&mut inf, 1), Err(SolveError::NonFinite));
    }

    #[test]
    fn lm_03_refactor_and_solve_into_do_not_reallocate() {
        // [LM-03][PERF-R2] One kept factor serves every drag event: `factor` reuses its buffer and
        // `solve_into` writes into the caller's vector. The buffer pointer staying put across calls is
        // the evidence that nothing was reallocated.
        let mut rng = Rng::new(5);
        let n = 40;
        let mut c = Cholesky::with_capacity(n);
        let before = c.l.as_ptr();
        let mut x = vec![0.0; n];
        let mut scratch = vec![0.0; n * n];
        for _ in 0..4 {
            let a = random_spd(&mut rng, n);
            let b: Vec<f64> = (0..n).map(|_| rng.range_f64(-1.0, 1.0)).collect();
            c.factor(a.view()).unwrap();
            c.solve_into(&b, &mut x).unwrap();
            assert_eq!(c.l.as_ptr(), before);
            assert!(relative_residual(&a, &x, &b) < 1e-12);
            // The scratch-slice form gives the same answer.
            let mut y = vec![0.0; n];
            cholesky_solve_into(a.view(), &b, &mut y, &mut scratch).unwrap();
            assert_eq!(x, y);
        }
        let a = random_spd(&mut rng, n);
        assert!(cholesky_solve_into(a.view(), &x, &mut vec![0.0; n], &mut scratch[..n]).is_err());
        // A failed factorisation leaves an empty factor, not a stale one.
        assert!(c.factor(DenseMatrix::from_rows(&[[1.0, 2.0], [2.0, 1.0]]).unwrap().view()).is_err());
        assert_eq!(c.order(), 0);
        assert_eq!(c.l().rows(), 0);
    }

    #[test]
    fn forward_solve_gives_the_mahalanobis_length() {
        let mut rng = Rng::new(9);
        let a = random_spd(&mut rng, 6);
        let c = Cholesky::new(&a).unwrap();
        let b: Vec<f64> = (0..6).map(|_| rng.range_f64(-1.0, 1.0)).collect();
        let x = c.solve(&b).unwrap();
        let expect = dot(&b, &x);
        let mut y = b.clone();
        c.forward_in_place(&mut y).unwrap();
        assert!((dot(&y, &y) - expect).abs() < 1e-13);
    }

    #[test]
    fn ldlt_solves_definite_and_quasi_definite_systems() {
        let mut rng = Rng::new(31);
        // Positive definite: agrees with Cholesky, D is positive.
        let a = random_spd(&mut rng, 9);
        let b: Vec<f64> = (0..9).map(|_| rng.range_f64(-1.0, 1.0)).collect();
        let f = Ldlt::new(&a).unwrap();
        assert_eq!(f.order(), 9);
        assert_eq!(f.negative_pivots(), 0);
        assert!(relative_residual(&a, &f.solve(&b).unwrap(), &b) < 1e-12);
        // Symmetric quasi-definite saddle point [[H, Bᵀ], [B, -C]] with H, C positive definite.
        let h = random_spd(&mut rng, 5);
        let c = random_spd(&mut rng, 3);
        let bm = DenseMatrix::from_fn(3, 5, |_, _| rng.range_f64(-1.0, 1.0));
        let k = DenseMatrix::from_fn(8, 8, |i, j| match (i < 5, j < 5) {
            (true, true) => h[(i, j)],
            (false, true) => bm[(i - 5, j)],
            (true, false) => bm[(j - 5, i)],
            (false, false) => -c[(i - 5, j - 5)],
        });
        let rhs: Vec<f64> = (0..8).map(|_| rng.range_f64(-1.0, 1.0)).collect();
        let f = Ldlt::new(&k).unwrap();
        assert_eq!(f.negative_pivots(), 3);
        assert!(relative_residual(&k, &f.solve(&rhs).unwrap(), &rhs) < 1e-11);
        let mut d = [0.0; 8];
        f.diagonal_into(&mut d).unwrap();
        assert!(d[..5].iter().all(|&v| v > 0.0) && d[5..].iter().all(|&v| v < 0.0));
    }

    #[test]
    fn ldlt_reports_zero_pivots_and_bad_input() {
        let a = DenseMatrix::from_rows(&[[0.0, 1.0], [1.0, 0.0]]).unwrap();
        assert_eq!(Ldlt::new(&a), Err(SolveError::Singular));
        let a = DenseMatrix::from_rows(&[[1.0, 1.0], [1.0, 1.0]]).unwrap();
        assert_eq!(Ldlt::new(&a), Err(SolveError::Singular));
        assert!(Ldlt::new(&DenseMatrix::zeros(2, 1)).is_err());
        let mut a = DenseMatrix::identity(2);
        a[(1, 0)] = f64::NAN;
        assert_eq!(Ldlt::new(&a), Err(SolveError::NonFinite));
        let f = Ldlt::new(&DenseMatrix::identity(2)).unwrap();
        assert!(f.solve(&[1.0]).is_err());
        assert!(f.diagonal_into(&mut [0.0; 3]).is_err());
    }
}
