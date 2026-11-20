//! Preconditioned conjugate gradients for large sparse symmetric positive definite systems, with the
//! Jacobi (diagonal) and zero-fill incomplete Cholesky (IC(0)) preconditioners.
//!
//! [`pcg`] is generic over the operator ([`LinearOperator`]: [`CsrMatrix`], [`DenseMatrix`] or a
//! matrix-free operator) and over the [`Preconditioner`], so neither call is a trait object in the
//! inner loop `[PERF-R4]`. All vectors of the iteration live in a [`PcgWorkspace`] the caller keeps:
//! once it has been used for a system of the same size, a solve allocates nothing `[PERF-R2]`.
//!
//! Convergence is measured by the relative residual `‖b − A·x‖₂ / ‖b‖₂`. The returned
//! [`PcgReport`] holds the residual recomputed from the final `x` (not the recurrence value, which can
//! drift from it), so it is evidence and not an estimate. A curvature `pᵀ·A·p ≤ 0` or a
//! preconditioned residual product `rᵀ·z ≤ 0` means the operator or the preconditioner is not
//! positive definite; the solve stops with [`SolveError::Breakdown`] instead of producing garbage.

use super::dense::DenseMatrix;
use super::error::{SolveError, check_dim, check_finite};
use super::sparse::{CsrMatrix, LinearOperator};
use super::vector::{axpy, dot, norm};

/// An approximation `M ≈ A` whose inverse is cheap to apply: `z = M⁻¹·r`.
pub trait Preconditioner {
    /// The order of the systems it applies to, or `None` when it applies to any size.
    fn order(&self) -> Option<usize>;
    /// Writes `M⁻¹·r` into `z`. The caller guarantees both have the order's length; the
    /// implementation must not allocate.
    fn apply(&self, r: &[f64], z: &mut [f64]);
}

/// No preconditioning: `M = I`, plain conjugate gradients.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IdentityPreconditioner;

impl Preconditioner for IdentityPreconditioner {
    fn order(&self) -> Option<usize> {
        None
    }

    fn apply(&self, r: &[f64], z: &mut [f64]) {
        z.copy_from_slice(r);
    }
}

/// The Jacobi preconditioner `M = diag(A)`: cheap, and it removes bad row/column scaling.
#[derive(Clone, Debug, PartialEq)]
pub struct JacobiPreconditioner {
    inverse_diagonal: Vec<f64>,
}

impl JacobiPreconditioner {
    /// From the diagonal of `a`. Fails with [`SolveError::NotPositiveDefinite`] at the first diagonal
    /// element that is not strictly positive (the matrix then cannot be SPD).
    pub fn new(a: &CsrMatrix) -> Result<Self, SolveError> {
        if a.rows() != a.cols() {
            return Err(SolveError::DimensionMismatch {
                what: "jacobi: matrix must be square",
                expected: a.rows(),
                found: a.cols(),
            });
        }
        let mut diagonal = vec![0.0; a.rows()];
        a.diagonal_into(&mut diagonal)?;
        Self::from_diagonal(&diagonal)
    }

    /// From a given diagonal, which must be finite and strictly positive.
    pub fn from_diagonal(diagonal: &[f64]) -> Result<Self, SolveError> {
        check_finite(diagonal)?;
        let mut inverse_diagonal = Vec::with_capacity(diagonal.len());
        for (pivot, &d) in diagonal.iter().enumerate() {
            if d <= 0.0 {
                return Err(SolveError::NotPositiveDefinite { pivot });
            }
            inverse_diagonal.push(1.0 / d);
        }
        Ok(Self { inverse_diagonal })
    }
}

impl Preconditioner for JacobiPreconditioner {
    fn order(&self) -> Option<usize> {
        Some(self.inverse_diagonal.len())
    }

    fn apply(&self, r: &[f64], z: &mut [f64]) {
        for ((zi, ri), d) in z.iter_mut().zip(r).zip(&self.inverse_diagonal) {
            *zi = ri * d;
        }
    }
}

/// The zero-fill incomplete Cholesky factorisation `M = L·Lᵀ ≈ A`: `L` has exactly the pattern of the
/// lower triangle of `A`. For a tridiagonal (or any no-fill) matrix it is the exact Cholesky factor.
#[derive(Clone, Debug, PartialEq)]
pub struct IncompleteCholesky {
    /// The factor `L`, row-major CSR; the last entry of each row is its diagonal.
    l: CsrMatrix,
}

impl IncompleteCholesky {
    /// Factors the symmetric matrix `a`, reading only its lower triangle (the diagonal must be stored).
    ///
    /// IC(0) can break down even for an SPD matrix; then this fails with
    /// [`SolveError::NotPositiveDefinite`] and [`IncompleteCholesky::with_shift`] is the remedy.
    pub fn new(a: &CsrMatrix) -> Result<Self, SolveError> {
        Self::with_shift(a, 0.0)
    }

    /// Factors `A + shift·diag(A)` (Manteuffel's shift): a positive `shift` makes the factorisation
    /// succeed for more matrices at the cost of a weaker preconditioner. `shift` must be finite and
    /// non-negative.
    pub fn with_shift(a: &CsrMatrix, shift: f64) -> Result<Self, SolveError> {
        if !(shift.is_finite() && shift >= 0.0) {
            return Err(SolveError::InvalidParameter { what: "incomplete cholesky: shift must be finite and >= 0" });
        }
        let n = a.rows();
        if n != a.cols() {
            return Err(SolveError::DimensionMismatch {
                what: "incomplete cholesky: matrix must be square",
                expected: n,
                found: a.cols(),
            });
        }
        // The lower triangle; a row without a stored diagonal cannot be factored.
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);
        for i in 0..n {
            let (cols, vals) = a.row(i);
            for (&c, &v) in cols.iter().zip(vals) {
                if c < i {
                    col_idx.push(c);
                    values.push(v);
                } else if c == i {
                    col_idx.push(c);
                    values.push(v * (1.0 + shift));
                }
            }
            if col_idx.last() != Some(&i) {
                return Err(SolveError::NotPositiveDefinite { pivot: i });
            }
            row_ptr.push(col_idx.len());
        }
        factor_lower_in_place(&row_ptr, &col_idx, &mut values)?;
        let l = CsrMatrix::from_parts(n, n, row_ptr, col_idx, values)?;
        Ok(Self { l })
    }

    /// The factor `L` (lower triangular, CSR).
    pub fn factor(&self) -> &CsrMatrix {
        &self.l
    }
}

/// `x > 0`, false for NaN: the positivity test of pivots and curvatures.
#[inline]
fn is_positive(x: f64) -> bool {
    x > 0.0
}

/// IC(0) on the lower-triangular CSR arrays (each row ends with its diagonal), in place.
fn factor_lower_in_place(row_ptr: &[usize], col_idx: &[usize], values: &mut [f64]) -> Result<(), SolveError> {
    for (i, window) in row_ptr.windows(2).enumerate() {
        let (start, diagonal) = (window[0], window[1] - 1);
        let (done, rest) = values.split_at_mut(start);
        for idx in start..diagonal {
            let k = col_idx[idx];
            // L[i][k] = (A[i][k] − Σ_{j<k} L[i][j]·L[k][j]) / L[k][k]: a merge of two sorted rows.
            let (k_start, k_diagonal) = (row_ptr[k], row_ptr[k + 1] - 1);
            let mut sum = 0.0;
            let (mut a, mut b) = (start, k_start);
            while a < idx && b < k_diagonal {
                match col_idx[a].cmp(&col_idx[b]) {
                    std::cmp::Ordering::Less => a += 1,
                    std::cmp::Ordering::Greater => b += 1,
                    std::cmp::Ordering::Equal => {
                        sum += rest[a - start] * done[b];
                        a += 1;
                        b += 1;
                    }
                }
            }
            rest[idx - start] = (rest[idx - start] - sum) / done[k_diagonal];
        }
        let row = &rest[..=diagonal - start];
        let (off, d) = row.split_at(row.len() - 1);
        let pivot = d[0] - dot(off, off);
        // Also catches NaN.
        if !is_positive(pivot) {
            return Err(SolveError::NotPositiveDefinite { pivot: i });
        }
        rest[diagonal - start] = pivot.sqrt();
    }
    Ok(())
}

impl Preconditioner for IncompleteCholesky {
    fn order(&self) -> Option<usize> {
        Some(self.l.rows())
    }

    fn apply(&self, r: &[f64], z: &mut [f64]) {
        let (row_ptr, col_idx, values) = (self.l.row_ptr(), self.l.col_indices(), self.l.values());
        // Forward: L·y = r (row-oriented).
        for (i, window) in row_ptr.windows(2).enumerate() {
            let diagonal = window[1] - 1;
            let mut sum = r[i];
            for (&c, &v) in col_idx[window[0]..diagonal].iter().zip(&values[window[0]..diagonal]) {
                sum -= v * z[c];
            }
            z[i] = sum / values[diagonal];
        }
        // Backward: Lᵀ·z = y, column-oriented over the rows of L, in place.
        for (i, window) in row_ptr.windows(2).enumerate().rev() {
            let diagonal = window[1] - 1;
            let zi = z[i] / values[diagonal];
            z[i] = zi;
            for (&c, &v) in col_idx[window[0]..diagonal].iter().zip(&values[window[0]..diagonal]) {
                z[c] -= v * zi;
            }
        }
    }
}

/// When [`pcg`] stops.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PcgSettings {
    /// Converged when `‖b − A·x‖₂ ≤ tolerance · ‖b‖₂`. Finite and non-negative.
    pub tolerance: f64,
    /// The iteration limit; reaching it is [`SolveError::NotConverged`].
    pub max_iterations: usize,
}

impl Default for PcgSettings {
    /// A relative residual of `1e-10` within 10 000 iterations.
    fn default() -> Self {
        Self { tolerance: 1e-10, max_iterations: 10_000 }
    }
}

/// The outcome of a converged [`pcg`] solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PcgReport {
    /// The iterations used (0 when the initial guess already met the tolerance).
    pub iterations: usize,
    /// `‖b − A·x‖₂ / ‖b‖₂` recomputed from the returned `x` (0 for `b = 0`).
    pub relative_residual: f64,
}

/// The four vectors of the conjugate-gradient iteration, kept between solves.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PcgWorkspace {
    r: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
    q: Vec<f64>,
}

impl PcgWorkspace {
    /// A workspace for systems of order `n`.
    pub fn new(n: usize) -> Self {
        Self { r: vec![0.0; n], z: vec![0.0; n], p: vec![0.0; n], q: vec![0.0; n] }
    }

    /// Sizes every vector to `n`; allocates only when `n` exceeds what it held before.
    fn resize(&mut self, n: usize) {
        for v in [&mut self.r, &mut self.z, &mut self.p, &mut self.q] {
            v.resize(n, 0.0);
        }
    }
}

/// Solves the symmetric positive definite system `A·x = b` by preconditioned conjugate gradients.
///
/// `x` holds the initial guess on entry (zeros are a fine default) and the solution on success. The
/// workspace is resized to the order of `A` and reused.
///
/// Errors: [`SolveError::DimensionMismatch`] (non-square `A`, wrong vector lengths, a preconditioner of
/// another order), [`SolveError::InvalidParameter`] (bad tolerance), [`SolveError::NonFinite`],
/// [`SolveError::Breakdown`] (operator or preconditioner not positive definite) and
/// [`SolveError::NotConverged`] (iteration limit; `x` then holds the last iterate).
pub fn pcg<A: LinearOperator, M: Preconditioner>(
    a: &A,
    b: &[f64],
    x: &mut [f64],
    preconditioner: &M,
    settings: &PcgSettings,
    workspace: &mut PcgWorkspace,
) -> Result<PcgReport, SolveError> {
    let n = a.rows();
    check_dim("pcg: operator must be square (columns)", n, a.cols())?;
    check_dim("pcg: rhs length", n, b.len())?;
    check_dim("pcg: solution length", n, x.len())?;
    if let Some(order) = preconditioner.order() {
        check_dim("pcg: preconditioner order", n, order)?;
    }
    if !(settings.tolerance.is_finite() && settings.tolerance >= 0.0) {
        return Err(SolveError::InvalidParameter { what: "pcg: tolerance must be finite and >= 0" });
    }
    check_finite(b)?;
    check_finite(x)?;
    let b_norm = norm(b);
    if b_norm == 0.0 {
        x.fill(0.0);
        return Ok(PcgReport { iterations: 0, relative_residual: 0.0 });
    }
    workspace.resize(n);
    let PcgWorkspace { r, z, p, q } = workspace;
    let threshold = settings.tolerance * b_norm;
    // r = b − A·x
    a.apply(x, q);
    for ((ri, bi), qi) in r.iter_mut().zip(b).zip(q.iter()) {
        *ri = bi - qi;
    }
    let mut r_norm = norm(r);
    if !r_norm.is_finite() {
        return Err(SolveError::NonFinite);
    }
    let mut iterations = 0;
    if r_norm > threshold {
        preconditioner.apply(r, z);
        let mut rz = dot(r, z);
        if !is_positive(rz) {
            return Err(SolveError::Breakdown { iteration: 0 });
        }
        p.copy_from_slice(z);
        loop {
            if iterations == settings.max_iterations {
                return Err(SolveError::NotConverged { iterations, residual: r_norm / b_norm });
            }
            iterations += 1;
            a.apply(p, q);
            let curvature = dot(p, q);
            if !is_positive(curvature) {
                return Err(SolveError::Breakdown { iteration: iterations });
            }
            let alpha = rz / curvature;
            axpy(alpha, p, x);
            axpy(-alpha, q, r);
            r_norm = norm(r);
            if !r_norm.is_finite() {
                return Err(SolveError::NonFinite);
            }
            if r_norm <= threshold {
                break;
            }
            preconditioner.apply(r, z);
            let rz_next = dot(r, z);
            if !is_positive(rz_next) {
                return Err(SolveError::Breakdown { iteration: iterations });
            }
            let beta = rz_next / rz;
            rz = rz_next;
            for (pi, zi) in p.iter_mut().zip(z.iter()) {
                *pi = zi + beta * *pi;
            }
        }
    }
    // The true residual of the returned x.
    a.apply(x, q);
    for ((ri, bi), qi) in r.iter_mut().zip(b).zip(q.iter()) {
        *ri = bi - qi;
    }
    Ok(PcgReport { iterations, relative_residual: norm(r) / b_norm })
}

/// Convenience: solves `A·x = b` from `x = 0` with the default settings, a fresh workspace and the
/// Jacobi preconditioner built from `a`.
pub fn pcg_solve(a: &CsrMatrix, b: &[f64]) -> Result<Vec<f64>, SolveError> {
    let preconditioner = JacobiPreconditioner::new(a)?;
    let mut x = vec![0.0; a.rows()];
    let mut workspace = PcgWorkspace::new(a.rows());
    pcg(a, b, &mut x, &preconditioner, &PcgSettings::default(), &mut workspace)?;
    Ok(x)
}

/// The dense counterpart for tests and small systems: `IdentityPreconditioner` CG on a dense matrix.
pub fn cg_dense(a: &DenseMatrix, b: &[f64], settings: &PcgSettings) -> Result<(Vec<f64>, PcgReport), SolveError> {
    let mut x = vec![0.0; a.rows()];
    let mut workspace = PcgWorkspace::new(a.rows());
    let report = pcg(a, b, &mut x, &IdentityPreconditioner, settings, &mut workspace)?;
    Ok((x, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solve::cholesky::Cholesky;
    use crate::solve::testutil::{random_spd, random_vec};
    use humanart_core::rng::Rng;

    /// The 1D Laplacian `tridiag(−1, 2, −1)` of order `n`.
    fn laplacian_1d(n: usize) -> CsrMatrix {
        let mut t = Vec::new();
        for i in 0..n {
            t.push((i, i, 2.0));
            if i + 1 < n {
                t.push((i, i + 1, -1.0));
                t.push((i + 1, i, -1.0));
            }
        }
        CsrMatrix::from_triplets(n, n, &t).unwrap()
    }

    /// A 2D 5-point diffusion operator on a `side × side` grid with random positive edge
    /// conductivities and a small mass term, then scaled symmetrically by `D·A·D` with `D` spanning
    /// two decades: SPD and badly scaled, the case the preconditioners are for.
    fn diffusion_2d(rng: &mut Rng, side: usize) -> CsrMatrix {
        let n = side * side;
        let scaling: Vec<f64> = (0..n).map(|_| 10f64.powf(rng.range_f64(-1.0, 1.0))).collect();
        let mut t = Vec::new();
        for y in 0..side {
            for x in 0..side {
                let i = y * side + x;
                t.push((i, i, 1e-3 * scaling[i] * scaling[i]));
                let mut edge = |j: usize, rng: &mut Rng| {
                    let k = rng.range_f64(0.1, 10.0);
                    t.push((i, i, k * scaling[i] * scaling[i]));
                    t.push((j, j, k * scaling[j] * scaling[j]));
                    t.push((i, j, -k * scaling[i] * scaling[j]));
                    t.push((j, i, -k * scaling[i] * scaling[j]));
                };
                if x + 1 < side {
                    edge(i + 1, rng);
                }
                if y + 1 < side {
                    edge(i + side, rng);
                }
            }
        }
        CsrMatrix::from_triplets(n, n, &t).unwrap()
    }

    fn relative_residual(a: &CsrMatrix, x: &[f64], b: &[f64]) -> f64 {
        let ax = a.mul_vec(x).unwrap();
        ax.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum::<f64>().sqrt() / norm(b)
    }

    #[test]
    fn ac_wp02_1_known_solution_of_the_1d_laplacian() {
        let n = 50;
        let a = laplacian_1d(n);
        let expect: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let b = a.mul_vec(&expect).unwrap();
        let settings = PcgSettings { tolerance: 1e-12, max_iterations: 200 };
        let mut workspace = PcgWorkspace::default();
        let mut x = vec![0.0; n];
        let report = pcg(&a, &b, &mut x, &IdentityPreconditioner, &settings, &mut workspace).unwrap();
        // Exact arithmetic finishes within n steps; rounding may add a few.
        assert!(report.iterations <= n + 5, "{report:?}");
        assert!(report.relative_residual < 1e-12);
        assert!((relative_residual(&a, &x, &b) - report.relative_residual).abs() < 1e-15);
        for (p, q) in x.iter().zip(&expect) {
            assert!((p - q).abs() < 1e-9);
        }
        // IC(0) of a tridiagonal matrix has no dropped fill: it is the exact factor, one step suffices.
        let ic = IncompleteCholesky::new(&a).unwrap();
        let mut x = vec![0.0; n];
        let report = pcg(&a, &b, &mut x, &ic, &settings, &mut workspace).unwrap();
        assert_eq!(report.iterations, 1);
        assert!(report.relative_residual < 1e-13);
        // ... and its factor equals the dense Cholesky factor.
        let dense = Cholesky::new(&a.to_dense()).unwrap();
        for i in 0..n {
            for j in 0..=i {
                assert!((ic.factor().get(i, j) - dense.l().at(i, j)).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn ac_wp02_1_preconditioners_reduce_iterations_on_a_badly_scaled_problem() {
        let mut rng = Rng::new(0x80);
        let a = diffusion_2d(&mut rng, 32);
        assert!(a.is_symmetric(0.0));
        let b = random_vec(&mut rng, a.rows());
        let settings = PcgSettings { tolerance: 1e-10, max_iterations: 20_000 };
        let mut workspace = PcgWorkspace::new(a.rows());
        let mut run = |m: &dyn Fn(&mut [f64], &mut PcgWorkspace) -> PcgReport| {
            let mut x = vec![0.0; a.rows()];
            let report = m(&mut x, &mut workspace);
            assert!(relative_residual(&a, &x, &b) < 1e-10, "{report:?}");
            report.iterations
        };
        let plain = run(&|x, w| pcg(&a, &b, x, &IdentityPreconditioner, &settings, w).unwrap());
        let jacobi = JacobiPreconditioner::new(&a).unwrap();
        let with_jacobi = run(&|x, w| pcg(&a, &b, x, &jacobi, &settings, w).unwrap());
        let ic = IncompleteCholesky::new(&a).unwrap();
        let with_ic = run(&|x, w| pcg(&a, &b, x, &ic, &settings, w).unwrap());
        assert!(with_ic < with_jacobi && with_jacobi < plain, "{plain} / {with_jacobi} / {with_ic}");
        // pcg_solve uses Jacobi and reaches the same accuracy.
        assert!(relative_residual(&a, &pcg_solve(&a, &b).unwrap(), &b) < 1e-10);
    }

    #[test]
    fn ac_wp02_1_matches_dense_cholesky_on_random_spd_systems() {
        let mut rng = Rng::new(0x81);
        for n in [1, 5, 40] {
            let dense = random_spd(&mut rng, n);
            let b = random_vec(&mut rng, n);
            let expect = Cholesky::new(&dense).unwrap().solve(&b).unwrap();
            let settings = PcgSettings { tolerance: 1e-13, max_iterations: 500 };
            let (x, report) = cg_dense(&dense, &b, &settings).unwrap();
            assert!(report.relative_residual < 1e-13);
            let sparse = CsrMatrix::from_dense(&dense, 0.0).unwrap();
            let ic = IncompleteCholesky::new(&sparse).unwrap();
            let mut y = vec![0.0; n];
            let report_ic = pcg(&sparse, &b, &mut y, &ic, &settings, &mut PcgWorkspace::default()).unwrap();
            // A dense pattern has no fill to drop: IC(0) is exact Cholesky.
            assert!(report_ic.iterations <= 2, "{report_ic:?}");
            for ((p, q), r) in x.iter().zip(&expect).zip(&y) {
                assert!((p - q).abs() < 1e-11 && (r - q).abs() < 1e-11);
            }
        }
    }

    #[test]
    fn kershaw_matrix_needs_a_shift() {
        // Kershaw's 4×4 SPD matrix, on which IC(0) meets a negative pivot.
        let rows = [[3.0, -2.0, 0.0, 2.0], [-2.0, 3.0, -2.0, 0.0], [0.0, -2.0, 3.0, -2.0], [2.0, 0.0, -2.0, 3.0]];
        let a = CsrMatrix::from_dense(&DenseMatrix::from_rows(&rows).unwrap(), 0.0).unwrap();
        assert!(Cholesky::new(&a.to_dense()).is_ok());
        assert!(matches!(IncompleteCholesky::new(&a), Err(SolveError::NotPositiveDefinite { .. })));
        let ic = IncompleteCholesky::with_shift(&a, 1.0).unwrap();
        let b = [1.0, 2.0, 3.0, 4.0];
        let mut x = [0.0; 4];
        let report = pcg(&a, &b, &mut x, &ic, &PcgSettings::default(), &mut PcgWorkspace::default()).unwrap();
        assert!(report.relative_residual < 1e-10 && report.iterations <= 6);
        assert!(IncompleteCholesky::with_shift(&a, -1.0).is_err());
        assert!(IncompleteCholesky::with_shift(&a, f64::NAN).is_err());
    }

    #[test]
    fn workspace_is_reused_without_allocation() {
        let a = laplacian_1d(64);
        let b = vec![1.0; 64];
        let mut workspace = PcgWorkspace::new(64);
        let pointers = |w: &PcgWorkspace| [w.r.as_ptr(), w.z.as_ptr(), w.p.as_ptr(), w.q.as_ptr()];
        let before = pointers(&workspace);
        for _ in 0..3 {
            let mut x = vec![0.0; 64];
            pcg(&a, &b, &mut x, &IdentityPreconditioner, &PcgSettings::default(), &mut workspace).unwrap();
            assert_eq!(pointers(&workspace), before);
        }
        // A smaller system reuses the same buffers.
        let small = laplacian_1d(10);
        let mut x = vec![0.0; 10];
        pcg(&small, &[1.0; 10], &mut x, &IdentityPreconditioner, &PcgSettings::default(), &mut workspace).unwrap();
        assert_eq!(pointers(&workspace), before);
    }

    #[test]
    fn trivial_cases_and_errors() {
        let a = laplacian_1d(4);
        let settings = PcgSettings::default();
        let mut w = PcgWorkspace::default();
        // b = 0 gives x = 0; an exact initial guess takes no iteration.
        let mut x = [5.0; 4];
        assert_eq!(pcg(&a, &[0.0; 4], &mut x, &IdentityPreconditioner, &settings, &mut w).unwrap().iterations, 0);
        assert_eq!(x, [0.0; 4]);
        let expect = [1.0, -1.0, 2.0, 0.5];
        let b = a.mul_vec(&expect).unwrap();
        let mut x = expect;
        assert_eq!(pcg(&a, &b, &mut x, &IdentityPreconditioner, &settings, &mut w).unwrap().iterations, 0);
        // Iteration limit.
        let mut x = [0.0; 4];
        let limited = PcgSettings { tolerance: 1e-12, max_iterations: 1 };
        assert!(matches!(
            pcg(&a, &b, &mut x, &IdentityPreconditioner, &limited, &mut w),
            Err(SolveError::NotConverged { iterations: 1, .. })
        ));
        // Indefinite operator: zero curvature along the first direction.
        let indefinite = CsrMatrix::from_triplets(2, 2, &[(0, 0, 1.0), (1, 1, -1.0)]).unwrap();
        let mut x = [0.0; 2];
        assert_eq!(
            pcg(&indefinite, &[1.0, 1.0], &mut x, &IdentityPreconditioner, &settings, &mut w),
            Err(SolveError::Breakdown { iteration: 1 })
        );
        assert_eq!(JacobiPreconditioner::new(&indefinite), Err(SolveError::NotPositiveDefinite { pivot: 1 }));
        assert!(matches!(IncompleteCholesky::new(&indefinite), Err(SolveError::NotPositiveDefinite { pivot: 1 })));
        // A row without a stored diagonal.
        let no_diagonal = CsrMatrix::from_triplets(2, 2, &[(0, 0, 1.0), (1, 0, 0.5)]).unwrap();
        assert_eq!(IncompleteCholesky::new(&no_diagonal), Err(SolveError::NotPositiveDefinite { pivot: 1 }));
        // Dimensions, tolerance, non-finite input.
        let mut x = [0.0; 4];
        assert!(pcg(&a, &[1.0; 3], &mut x, &IdentityPreconditioner, &settings, &mut w).is_err());
        assert!(pcg(&a, &[1.0; 4], &mut [0.0; 3], &IdentityPreconditioner, &settings, &mut w).is_err());
        let jacobi = JacobiPreconditioner::from_diagonal(&[1.0; 3]).unwrap();
        assert!(matches!(pcg(&a, &b, &mut x, &jacobi, &settings, &mut w), Err(SolveError::DimensionMismatch { .. })));
        let bad = PcgSettings { tolerance: f64::NAN, max_iterations: 10 };
        assert!(matches!(
            pcg(&a, &b, &mut x, &IdentityPreconditioner, &bad, &mut w),
            Err(SolveError::InvalidParameter { .. })
        ));
        assert_eq!(
            pcg(&a, &[f64::NAN; 4], &mut x, &IdentityPreconditioner, &settings, &mut w),
            Err(SolveError::NonFinite)
        );
        let rect = CsrMatrix::zeros(2, 3);
        assert!(pcg(&rect, &[1.0; 2], &mut [0.0; 2], &IdentityPreconditioner, &settings, &mut w).is_err());
        assert!(JacobiPreconditioner::new(&rect).is_err());
        assert!(IncompleteCholesky::new(&rect).is_err());
        assert!(JacobiPreconditioner::from_diagonal(&[1.0, f64::INFINITY]).is_err());
    }
}
