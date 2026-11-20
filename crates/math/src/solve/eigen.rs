//! Eigen-decomposition of real symmetric matrices, `A = V·Λ·Vᵀ`.
//!
//! [`SymmetricEigen`] is the cyclic Jacobi method: plane rotations annihilate the off-diagonal
//! elements pair by pair until the matrix is diagonal to rounding. It is slower than tridiagonal QR
//! for large orders but simple, unconditionally stable and accurate for the small eigenvalues too,
//! which is what PCA of a few hundred samples and the covariance fits of this crate need. Rotations
//! act on contiguous rows `[PERF-R2]`.
//!
//! [`symmetric_eigen3`] is the 3×3 special case without heap use: the closed trigonometric solution
//! of the characteristic cubic with eigenvectors from cross products (the scheme of Eberly's "A
//! Robust Eigensolver for 3×3 Symmetric Matrices"), followed by Jacobi rotations in that basis, which
//! repair the loss of accuracy the closed form has when two eigenvalues nearly coincide.
//!
//! Conventions: only the lower triangle of the input is read; eigenvalues are sorted in descending
//! order; eigenvectors are orthonormal and their sign is fixed by making the component of largest
//! magnitude positive, so results are reproducible.

use super::dense::{DenseMatrix, MatRef};
use super::error::{SolveError, check_dim, check_finite};
use super::vector::{norm_sq, rotate};

/// The sweep cap of the Jacobi iterations. Convergence is quadratic; well under 20 sweeps are needed
/// in practice, whatever the order.
pub(crate) const MAX_SWEEPS: usize = 64;

/// Off-diagonal mass (as a fraction of the squared Frobenius norm) below which a matrix counts as
/// diagonal: `(0.1·ε)²`.
const OFF_DIAGONAL_TOLERANCE_SQ: f64 = 0.01 * f64::EPSILON * f64::EPSILON;

/// The rotation `(c, s, t)` that annihilates the off-diagonal element `apq` of the symmetric 2×2
/// block `[[app, apq], [apq, aqq]]`. `apq` must be non-zero.
#[inline]
pub(crate) fn jacobi_rotation(app: f64, aqq: f64, apq: f64) -> (f64, f64, f64) {
    let theta = (aqq - app) / (2.0 * apq);
    // The smaller root of t² + 2θt − 1 = 0; an infinite θ (a negligible apq) gives t = 0.
    let t = theta.signum() / (theta.abs() + theta.hypot(1.0));
    let c = 1.0 / t.hypot(1.0);
    (c, t * c, t)
}

/// Makes the component of largest magnitude positive (the first such component on a tie).
fn fix_sign(v: &mut [f64]) {
    let mut largest = 0.0_f64;
    let mut negative = false;
    for &x in v.iter() {
        if x.abs() > largest {
            largest = x.abs();
            negative = x < 0.0;
        }
    }
    if negative {
        for x in v {
            *x = -*x;
        }
    }
}

/// The eigen-decomposition of a real symmetric matrix by the cyclic Jacobi method.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SymmetricEigen {
    n: usize,
    /// The eigenvalues, descending.
    values: Vec<f64>,
    /// Row `j` is the eigenvector of `values[j]` (this is `Vᵀ`).
    vectors_t: DenseMatrix,
    /// The matrix being diagonalised.
    work: Vec<f64>,
}

impl SymmetricEigen {
    /// Decomposes the symmetric matrix `a` (only its lower triangle is read).
    pub fn new(a: &DenseMatrix) -> Result<Self, SolveError> {
        let mut eigen = Self::default();
        eigen.factor(a.view())?;
        Ok(eigen)
    }

    /// Replaces the decomposition by that of `a`, reusing the buffers (no allocation once warm).
    ///
    /// Fails on a non-square or non-finite matrix, and with [`SolveError::NotConverged`] if the
    /// rotations do not converge (not observed for finite input). On failure the decomposition is
    /// left empty (order 0).
    pub fn factor(&mut self, a: MatRef<'_>) -> Result<(), SolveError> {
        self.n = 0;
        let n = a.rows();
        check_dim("eigen: matrix columns", n, a.cols())?;
        self.work.clear();
        self.work.extend_from_slice(a.as_slice());
        for i in 0..n {
            for j in 0..i {
                self.work[j * n + i] = self.work[i * n + j];
            }
        }
        check_finite(&self.work)?;
        self.vectors_t.resize(n, n);
        self.vectors_t.add_to_diagonal(1.0);
        self.values.clear();
        self.values.resize(n, 0.0);
        // Scaled to unit magnitude so that the squared norms below can neither overflow nor vanish.
        let scale = self.work.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if scale > 0.0 {
            for v in &mut self.work {
                *v /= scale;
            }
            let threshold = OFF_DIAGONAL_TOLERANCE_SQ * norm_sq(&self.work);
            let mut sweeps = 0;
            loop {
                let off_sq = off_diagonal_sq(&self.work, n);
                if off_sq <= threshold {
                    break;
                }
                if sweeps == MAX_SWEEPS {
                    return Err(SolveError::NotConverged { iterations: sweeps, residual: off_sq.sqrt() * scale });
                }
                jacobi_sweep(&mut self.work, self.vectors_t.as_mut_slice(), n);
                sweeps += 1;
            }
            for (i, value) in self.values.iter_mut().enumerate() {
                *value = self.work[i * n + i] * scale;
            }
            check_finite(&self.values)?;
        }
        // Selection sort, descending: n row swaps at most.
        for i in 0..n {
            let mut best = i;
            for j in i + 1..n {
                if self.values[j] > self.values[best] {
                    best = j;
                }
            }
            if best != i {
                self.values.swap(i, best);
                self.vectors_t.swap_rows(i, best)?;
            }
            fix_sign(self.vectors_t.row_mut(i));
        }
        self.n = n;
        Ok(())
    }

    /// The order `n` of the decomposed matrix.
    pub fn order(&self) -> usize {
        self.n
    }

    /// The eigenvalues in descending order.
    pub fn values(&self) -> &[f64] {
        &self.values[..self.n]
    }

    /// The eigenvector of `values()[j]` (unit length). Panics when `j >= order()`, like slice indexing.
    pub fn vector(&self, j: usize) -> &[f64] {
        self.vectors_t.row(j)
    }

    /// `Vᵀ`: row `j` is the eigenvector of `values()[j]`. No allocation.
    pub fn vectors_transposed(&self) -> MatRef<'_> {
        self.vectors_t.view()
    }

    /// `V`: column `j` is the eigenvector of `values()[j]`.
    pub fn vectors(&self) -> DenseMatrix {
        self.vectors_t.transpose()
    }

    /// `V·f(Λ)·Vᵀ`: the matrix function of `A` for a scalar function of its eigenvalues (the identity
    /// function rebuilds `A`, `1/λ` gives the inverse, `√λ` the symmetric square root).
    pub fn map_values(&self, mut f: impl FnMut(f64) -> f64) -> DenseMatrix {
        let n = self.n;
        let mut out = DenseMatrix::zeros(n, n);
        for (j, &value) in self.values().iter().enumerate() {
            let weight = f(value);
            let v = self.vectors_t.row(j);
            for (r, &vr) in v.iter().enumerate() {
                let w = weight * vr;
                if w != 0.0 {
                    super::vector::axpy(w, v, out.row_mut(r));
                }
            }
        }
        out
    }
}

/// `Σ_{i<j} aᵢⱼ²` of a row-major `n × n` matrix.
fn off_diagonal_sq(a: &[f64], n: usize) -> f64 {
    (0..n).map(|i| norm_sq(&a[i * n + i + 1..(i + 1) * n])).sum()
}

/// One cyclic sweep over all pairs `p < q` of the symmetric matrix `a`, accumulating the rotations in
/// the rows of `vt`.
fn jacobi_sweep(a: &mut [f64], vt: &mut [f64], n: usize) {
    for p in 0..n {
        for q in p + 1..n {
            let apq = a[p * n + q];
            if apq == 0.0 {
                continue;
            }
            let (app, aqq) = (a[p * n + p], a[q * n + q]);
            let (c, s, t) = jacobi_rotation(app, aqq, apq);
            // Rows p and q of Jᵀ·A; by symmetry they are also columns p and q of Jᵀ·A·J outside the
            // 2×2 block, which is set from the closed formulas.
            let (head, tail) = a.split_at_mut(q * n);
            rotate(&mut head[p * n..(p + 1) * n], &mut tail[..n], c, s);
            for k in 0..n {
                a[k * n + p] = a[p * n + k];
                a[k * n + q] = a[q * n + k];
            }
            a[p * n + p] = app - t * apq;
            a[q * n + q] = aqq + t * apq;
            a[p * n + q] = 0.0;
            a[q * n + p] = 0.0;
            let (head, tail) = vt.split_at_mut(q * n);
            rotate(&mut head[p * n..(p + 1) * n], &mut tail[..n], c, s);
        }
    }
}

/// The eigen-decomposition of a symmetric 3×3 matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eigen3 {
    /// The eigenvalues, descending.
    pub values: [f64; 3],
    /// The matrix `V` (row-major, `vectors[row][column]`): column `j` is the unit eigenvector of
    /// `values[j]`. The columns are orthonormal; the determinant may be +1 or −1.
    pub vectors: [[f64; 3]; 3],
}

impl Eigen3 {
    /// The eigenvectors as three vectors: `axes()[j]` is the eigenvector of `values[j]`.
    pub fn axes(&self) -> [[f64; 3]; 3] {
        transpose3(&self.vectors)
    }
}

/// The transpose of a 3×3 matrix.
pub(crate) fn transpose3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]]
}

/// The cross product `a × b`.
#[inline]
pub(crate) fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// The dot product of two 3-vectors.
#[inline]
pub(crate) fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The product `m · v` of a row-major 3×3 matrix and a vector.
#[inline]
pub(crate) fn mul3_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [dot3(m[0], v), dot3(m[1], v), dot3(m[2], v)]
}

/// Two unit vectors `(u, v)` that complete the unit vector `w` to a right-handed orthonormal basis
/// `u × v = w`.
pub(crate) fn orthonormal_complement3(w: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let u = if w[0].abs() > w[1].abs() {
        let inv = 1.0 / w[0].hypot(w[2]);
        [-w[2] * inv, 0.0, w[0] * inv]
    } else {
        let inv = 1.0 / w[1].hypot(w[2]);
        [0.0, w[2] * inv, -w[1] * inv]
    };
    (u, cross3(w, u))
}

/// A unit vector in the null space of `m − value·I` for a (numerically) simple eigenvalue: the
/// largest cross product of two rows.
fn eigenvector_of_simple3(m: &[[f64; 3]; 3], value: f64) -> [f64; 3] {
    let r0 = [m[0][0] - value, m[0][1], m[0][2]];
    let r1 = [m[1][0], m[1][1] - value, m[1][2]];
    let r2 = [m[2][0], m[2][1], m[2][2] - value];
    let candidates = [cross3(r0, r1), cross3(r0, r2), cross3(r1, r2)];
    let mut best = [1.0, 0.0, 0.0];
    let mut best_sq = 0.0;
    for candidate in candidates {
        let length_sq = dot3(candidate, candidate);
        if length_sq > best_sq {
            best_sq = length_sq;
            best = candidate;
        }
    }
    if best_sq > 0.0 {
        let inv = 1.0 / best_sq.sqrt();
        best = [best[0] * inv, best[1] * inv, best[2] * inv];
    }
    best
}

/// A unit eigenvector for `value` orthogonal to the unit eigenvector `first`: the null vector of the
/// 2×2 matrix that `m − value·I` induces on the plane orthogonal to `first`.
fn eigenvector_in_complement3(m: &[[f64; 3]; 3], first: [f64; 3], value: f64) -> [f64; 3] {
    let (u, v) = orthonormal_complement3(first);
    let mu = mul3_vec(m, u);
    let mv = mul3_vec(m, v);
    let m00 = dot3(u, mu) - value;
    let m01 = dot3(u, mv);
    let m11 = dot3(v, mv) - value;
    // The null vector is orthogonal to the larger row of [[m00, m01], [m01, m11]].
    let (x, y) = if m00.abs() >= m11.abs() { (m01, -m00) } else { (m11, -m01) };
    let length = x.hypot(y);
    if length > 0.0 {
        let (x, y) = (x / length, y / length);
        [x * u[0] + y * v[0], x * u[1] + y * v[1], x * u[2] + y * v[2]]
    } else {
        u
    }
}

/// The closed-form starting basis of [`symmetric_eigen3`]: three orthonormal vectors (as an array of
/// vectors) that diagonalise the symmetric `m` up to the accuracy of the trigonometric formula.
fn closed_form_basis3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let off_sq = m[1][0] * m[1][0] + m[2][0] * m[2][0] + m[2][1] * m[2][1];
    if off_sq == 0.0 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let q = (m[0][0] + m[1][1] + m[2][2]) / 3.0;
    let (b00, b11, b22) = (m[0][0] - q, m[1][1] - q, m[2][2] - q);
    let p = ((b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * off_sq) / 6.0).sqrt();
    let (c00, c11, c22) = (b00 / p, b11 / p, b22 / p);
    let (c01, c02, c12) = (m[1][0] / p, m[2][0] / p, m[2][1] / p);
    let det = c00 * (c11 * c22 - c12 * c12) - c01 * (c01 * c22 - c12 * c02) + c02 * (c01 * c12 - c11 * c02);
    let half_det = (det * 0.5).clamp(-1.0, 1.0);
    let angle = half_det.acos() / 3.0;
    let beta_max = 2.0 * angle.cos();
    let beta_min = 2.0 * (angle + 2.0 * std::f64::consts::FRAC_PI_3).cos();
    let beta_mid = -(beta_min + beta_max);
    let (value_max, value_mid, value_min) = (q + p * beta_max, q + p * beta_mid, q + p * beta_min);
    // Start from the eigenvalue that is best separated from the other two.
    if half_det >= 0.0 {
        let e_max = eigenvector_of_simple3(m, value_max);
        let e_mid = eigenvector_in_complement3(m, e_max, value_mid);
        [e_max, e_mid, cross3(e_max, e_mid)]
    } else {
        let e_min = eigenvector_of_simple3(m, value_min);
        let e_mid = eigenvector_in_complement3(m, e_min, value_mid);
        [cross3(e_mid, e_min), e_mid, e_min]
    }
}

/// Diagonalises the symmetric 3×3 matrix `b` in place by cyclic Jacobi rotations, accumulating them
/// in the rows of `vt`. Returns `false` if the sweep cap was reached.
fn jacobi3(b: &mut [[f64; 3]; 3], vt: &mut [[f64; 3]; 3]) -> bool {
    let total: f64 = b.iter().flatten().map(|v| v * v).sum();
    let threshold = OFF_DIAGONAL_TOLERANCE_SQ * total;
    for _ in 0..MAX_SWEEPS {
        let off_sq = b[0][1] * b[0][1] + b[0][2] * b[0][2] + b[1][2] * b[1][2];
        if off_sq <= threshold {
            return true;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let apq = b[p][q];
            if apq == 0.0 {
                continue;
            }
            let (app, aqq) = (b[p][p], b[q][q]);
            let (c, s, t) = jacobi_rotation(app, aqq, apq);
            let r = 3 - p - q;
            let (brp, brq) = (b[r][p], b[r][q]);
            let (new_rp, new_rq) = (c * brp - s * brq, s * brp + c * brq);
            b[r][p] = new_rp;
            b[p][r] = new_rp;
            b[r][q] = new_rq;
            b[q][r] = new_rq;
            b[p][p] = app - t * apq;
            b[q][q] = aqq + t * apq;
            b[p][q] = 0.0;
            b[q][p] = 0.0;
            let (vp, vq) = (vt[p], vt[q]);
            for k in 0..3 {
                vt[p][k] = c * vp[k] - s * vq[k];
                vt[q][k] = s * vp[k] + c * vq[k];
            }
        }
    }
    b[0][1] * b[0][1] + b[0][2] * b[0][2] + b[1][2] * b[1][2] <= threshold
}

/// The eigen-decomposition of the symmetric 3×3 matrix `a` (row-major; only its lower triangle is
/// read). No heap use.
///
/// The closed-form solution provides the starting basis; Jacobi rotations in that basis then bring
/// every eigenpair to rounding accuracy, including nearly repeated eigenvalues. Fails with
/// [`SolveError::NonFinite`] on NaN or infinite input.
pub fn symmetric_eigen3(a: &[[f64; 3]; 3]) -> Result<Eigen3, SolveError> {
    let lower = [a[0][0], a[1][0], a[1][1], a[2][0], a[2][1], a[2][2]];
    check_finite(&lower)?;
    let scale = lower.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if scale == 0.0 {
        return Ok(Eigen3 { values: [0.0; 3], vectors: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] });
    }
    let [m00, m10, m11, m20, m21, m22] = lower.map(|v| v / scale);
    let m = [[m00, m10, m20], [m10, m11, m21], [m20, m21, m22]];
    let mut vt = closed_form_basis3(&m);
    // B = Vᵀ·M·V, symmetric by construction.
    let mv = [mul3_vec(&m, vt[0]), mul3_vec(&m, vt[1]), mul3_vec(&m, vt[2])];
    let mut b = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            let value = dot3(vt[i], mv[j]);
            b[i][j] = value;
            b[j][i] = value;
        }
    }
    if !jacobi3(&mut b, &mut vt) {
        let off_sq = b[0][1] * b[0][1] + b[0][2] * b[0][2] + b[1][2] * b[1][2];
        return Err(SolveError::NotConverged { iterations: MAX_SWEEPS, residual: off_sq.sqrt() * scale });
    }
    let mut values = [b[0][0] * scale, b[1][1] * scale, b[2][2] * scale];
    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        if values[j] > values[i] {
            values.swap(i, j);
            vt.swap(i, j);
        }
    }
    for axis in &mut vt {
        fix_sign(axis);
    }
    Ok(Eigen3 { values, vectors: transpose3(&vt) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solve::testutil::{max_abs_diff, naive_mul, orthonormality_error, random_symmetric};
    use humanart_core::rng::Rng;

    /// `max |A·V − V·Λ|`.
    fn eigen_residual(a: &DenseMatrix, e: &SymmetricEigen) -> f64 {
        let v = e.vectors();
        let av = naive_mul(a, &v);
        let vl = DenseMatrix::from_fn(a.rows(), a.rows(), |i, j| v[(i, j)] * e.values()[j]);
        max_abs_diff(&av, &vl)
    }

    fn residual3(a: &[[f64; 3]; 3], e: &Eigen3) -> f64 {
        let mut worst = 0.0_f64;
        for (j, axis) in e.axes().into_iter().enumerate() {
            let av = mul3_vec(a, axis);
            for k in 0..3 {
                worst = worst.max((av[k] - e.values[j] * axis[k]).abs());
            }
        }
        worst
    }

    fn orthonormality3(e: &Eigen3) -> f64 {
        let axes = e.axes();
        let mut worst = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                let expect = if i == j { 1.0 } else { 0.0 };
                worst = worst.max((dot3(axes[i], axes[j]) - expect).abs());
            }
        }
        worst
    }

    #[test]
    fn ac_wp02_1_hand_solved_two_by_two() {
        let a = DenseMatrix::from_rows(&[[2.0, 1.0], [1.0, 2.0]]).unwrap();
        let e = SymmetricEigen::new(&a).unwrap();
        assert!((e.values()[0] - 3.0).abs() < 1e-15 && (e.values()[1] - 1.0).abs() < 1e-15);
        let h = std::f64::consts::FRAC_1_SQRT_2;
        assert!((e.vector(0)[0] - h).abs() < 1e-15 && (e.vector(0)[1] - h).abs() < 1e-15);
        assert!((e.vector(1)[0] - h).abs() < 1e-15 && (e.vector(1)[1] + h).abs() < 1e-15);
        assert_eq!(e.order(), 2);
    }

    #[test]
    fn ac_wp02_1_diagonal_identity_zero_and_empty() {
        let e = SymmetricEigen::new(&DenseMatrix::from_diagonal(&[1.0, 5.0, -2.0, 3.0])).unwrap();
        assert_eq!(e.values(), &[5.0, 3.0, 1.0, -2.0]);
        assert_eq!(e.vector(0), &[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(e.vector(3), &[0.0, 0.0, 1.0, 0.0]);
        let e = SymmetricEigen::new(&DenseMatrix::identity(3)).unwrap();
        assert_eq!(e.values(), &[1.0, 1.0, 1.0]);
        assert_eq!(e.vectors(), DenseMatrix::identity(3));
        let e = SymmetricEigen::new(&DenseMatrix::zeros(3, 3)).unwrap();
        assert_eq!(e.values(), &[0.0, 0.0, 0.0]);
        assert_eq!(orthonormality_error(&e.vectors()), 0.0);
        let e = SymmetricEigen::new(&DenseMatrix::zeros(0, 0)).unwrap();
        assert!(e.values().is_empty());
        let e = SymmetricEigen::new(&DenseMatrix::from_rows(&[[-7.0]]).unwrap()).unwrap();
        assert_eq!(e.values(), &[-7.0]);
        assert_eq!(e.vector(0), &[1.0]);
    }

    #[test]
    fn ac_wp02_1_laplacian_has_its_analytic_spectrum() {
        // The 1-D Dirichlet Laplacian tridiag(-1, 2, -1) of order n has the eigenvalues
        // 2 − 2·cos(kπ/(n+1)), k = 1..n.
        let n = 24;
        let a = DenseMatrix::from_fn(n, n, |i, j| match i.abs_diff(j) {
            0 => 2.0,
            1 => -1.0,
            _ => 0.0,
        });
        let e = SymmetricEigen::new(&a).unwrap();
        for (j, &value) in e.values().iter().enumerate() {
            let k = (n - j) as f64;
            let expect = 2.0 - 2.0 * (k * std::f64::consts::PI / (n as f64 + 1.0)).cos();
            assert!((value - expect).abs() < 1e-13, "eigenvalue {j}: {value} vs {expect}");
        }
        assert!(eigen_residual(&a, &e) < 1e-13);
        assert!(orthonormality_error(&e.vectors()) < 1e-13);
    }

    #[test]
    fn ac_wp02_1_hilbert_matrix_spectrum() {
        // The 5×5 Hilbert matrix: condition number 4.766e5. Jacobi resolves the smallest eigenvalue
        // (3.29e-6) to full absolute accuracy; the published values are quoted to 13 digits.
        let a = DenseMatrix::from_fn(5, 5, |i, j| 1.0 / (i + j + 1) as f64);
        let e = SymmetricEigen::new(&a).unwrap();
        let published = [
            1.567_050_691_098_2,
            2.085_342_186_110_1e-1,
            1.140_749_162_342_0e-2,
            3.058_980_401_511_9e-4,
            3.287_928_772_171_9e-6,
        ];
        for (got, want) in e.values().iter().zip(published) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
        let condition = e.values()[0] / e.values()[4];
        assert!((condition - 4.766e5).abs() < 1e2, "condition {condition}");
        let trace: f64 = (0..5).map(|i| a[(i, i)]).sum();
        assert!((e.values().iter().sum::<f64>() - trace).abs() < 1e-14);
    }

    #[test]
    fn ac_wp02_1_random_symmetric_matrices() {
        let mut rng = Rng::new(0xE1);
        for n in [2, 3, 7, 20, 61] {
            let a = random_symmetric(&mut rng, n);
            let e = SymmetricEigen::new(&a).unwrap();
            assert!(eigen_residual(&a, &e) < 1e-13 * n as f64, "n = {n}");
            assert!(orthonormality_error(&e.vectors()) < 1e-13, "n = {n}");
            assert!(e.values().windows(2).all(|w| w[0] >= w[1]), "n = {n}");
            assert!(max_abs_diff(&e.map_values(|v| v), &a) < 1e-13 * n as f64, "n = {n}");
            // The sign convention: the largest component of every eigenvector is positive.
            for j in 0..n {
                let v = e.vector(j);
                let largest = v.iter().fold(0.0_f64, |m, x| if x.abs() > m.abs() { *x } else { m });
                assert!(largest > 0.0);
            }
        }
    }

    #[test]
    fn matrix_functions_through_the_spectrum() {
        let mut rng = Rng::new(0xE2);
        let a = crate::solve::testutil::random_spd(&mut rng, 6);
        let e = SymmetricEigen::new(&a).unwrap();
        let inverse = e.map_values(|v| 1.0 / v);
        assert!(max_abs_diff(&naive_mul(&a, &inverse), &DenseMatrix::identity(6)) < 1e-13);
        let root = e.map_values(f64::sqrt);
        assert!(max_abs_diff(&naive_mul(&root, &root), &a) < 1e-13);
    }

    #[test]
    fn only_the_lower_triangle_is_read_and_scale_does_not_matter() {
        let mut a = DenseMatrix::from_rows(&[[2.0, 99.0], [1.0, 2.0]]).unwrap();
        let e = SymmetricEigen::new(&a).unwrap();
        assert!((e.values()[0] - 3.0).abs() < 1e-15);
        a[(0, 1)] = f64::NAN;
        assert!(SymmetricEigen::new(&a).is_ok());
        for scale in [1e-200, 1e200] {
            let a = DenseMatrix::from_rows(&[[2.0 * scale, scale], [scale, 2.0 * scale]]).unwrap();
            let e = SymmetricEigen::new(&a).unwrap();
            assert!((e.values()[0] / scale - 3.0).abs() < 1e-14 && (e.values()[1] / scale - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn bad_input_is_an_error_and_buffers_are_reused() {
        assert!(matches!(SymmetricEigen::new(&DenseMatrix::zeros(2, 3)), Err(SolveError::DimensionMismatch { .. })));
        let mut a = DenseMatrix::identity(3);
        a[(2, 1)] = f64::INFINITY;
        assert_eq!(SymmetricEigen::new(&a), Err(SolveError::NonFinite));
        let mut rng = Rng::new(0xE3);
        let mut e = SymmetricEigen::new(&random_symmetric(&mut rng, 8)).unwrap();
        let pointers = (e.values.as_ptr(), e.work.as_ptr(), e.vectors_t.as_slice().as_ptr());
        let a = random_symmetric(&mut rng, 8);
        e.factor(a.view()).unwrap();
        assert!(eigen_residual(&a, &e) < 1e-13);
        assert_eq!((e.values.as_ptr(), e.work.as_ptr(), e.vectors_t.as_slice().as_ptr()), pointers);
        assert!(e.factor(DenseMatrix::zeros(1, 2).view()).is_err());
        assert_eq!(e.order(), 0);
        assert!(e.values().is_empty());
    }

    #[test]
    fn ac_wp02_1_three_by_three_matches_the_general_solver() {
        let mut rng = Rng::new(0xE4);
        for _ in 0..200 {
            let dense = random_symmetric(&mut rng, 3);
            let a = [
                [dense[(0, 0)], dense[(0, 1)], dense[(0, 2)]],
                [dense[(1, 0)], dense[(1, 1)], dense[(1, 2)]],
                [dense[(2, 0)], dense[(2, 1)], dense[(2, 2)]],
            ];
            let e3 = symmetric_eigen3(&a).unwrap();
            let e = SymmetricEigen::new(&dense).unwrap();
            for j in 0..3 {
                assert!((e3.values[j] - e.values()[j]).abs() < 1e-14);
            }
            assert!(residual3(&a, &e3) < 1e-14);
            assert!(orthonormality3(&e3) < 1e-14);
            assert!(e3.values[0] >= e3.values[1] && e3.values[1] >= e3.values[2]);
        }
    }

    #[test]
    fn ac_wp02_1_three_by_three_degenerate_and_extreme_cases() {
        // Diagonal, repeated and zero spectra.
        let e = symmetric_eigen3(&[[1.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 2.0]]).unwrap();
        assert_eq!(e.values, [3.0, 2.0, 1.0]);
        assert_eq!(e.axes(), [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]);
        let e = symmetric_eigen3(&[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]).unwrap();
        assert_eq!(e.values, [2.0, 2.0, 2.0]);
        assert_eq!(orthonormality3(&e), 0.0);
        let e = symmetric_eigen3(&[[0.0; 3]; 3]).unwrap();
        assert_eq!(e.values, [0.0; 3]);
        // Rank one: n·nᵀ with n = (1, 2, 2)/3 has the spectrum {1, 0, 0}.
        let n = [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0];
        let mut a = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] = n[i] * n[j];
            }
        }
        let e = symmetric_eigen3(&a).unwrap();
        assert!((e.values[0] - 1.0).abs() < 1e-15 && e.values[1].abs() < 1e-15 && e.values[2].abs() < 1e-15);
        for (row, expect) in e.vectors.iter().zip(n) {
            assert!((row[0] - expect).abs() < 1e-15);
        }
        assert!(orthonormality3(&e) < 1e-15);
        // Two eigenvalues that differ by 1e-9: the closed form alone loses half the digits here.
        let (c, s) = (0.6, 0.8);
        let rotation = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let spectrum = [1.0 + 1e-9, 1.0, 0.25];
        let mut a = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] = (0..3).map(|k| rotation[i][k] * spectrum[k] * rotation[j][k]).sum();
            }
        }
        let e = symmetric_eigen3(&a).unwrap();
        for (got, want) in e.values.iter().zip(spectrum) {
            assert!((got - want).abs() < 2e-15, "{:?}", e.values);
        }
        assert!(residual3(&a, &e) < 1e-15);
        assert!(orthonormality3(&e) < 1e-15);
        // Extreme magnitudes neither overflow nor vanish.
        for scale in [1e-300, 1e300] {
            let a = [[2.0 * scale, scale, 0.0], [scale, 2.0 * scale, 0.0], [0.0, 0.0, 0.5 * scale]];
            let e = symmetric_eigen3(&a).unwrap();
            assert!((e.values[0] / scale - 3.0).abs() < 1e-14);
            assert!((e.values[1] / scale - 1.0).abs() < 1e-14);
            assert!((e.values[2] / scale - 0.5).abs() < 1e-14);
        }
        // Only the lower triangle is read; non-finite input is an error.
        let e = symmetric_eigen3(&[[2.0, f64::NAN, f64::NAN], [1.0, 2.0, f64::NAN], [0.0, 0.0, 0.0]]).unwrap();
        assert!((e.values[0] - 3.0).abs() < 1e-15);
        assert_eq!(symmetric_eigen3(&[[f64::NAN, 0.0, 0.0], [0.0; 3], [0.0; 3]]), Err(SolveError::NonFinite));
        assert_eq!(
            symmetric_eigen3(&[[1.0, 0.0, 0.0], [0.0; 3], [0.0, f64::INFINITY, 0.0]]),
            Err(SolveError::NonFinite)
        );
    }
}
