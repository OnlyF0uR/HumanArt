//! Singular value decomposition `A = U·Σ·Vᵀ` by one-sided Jacobi rotations, and what is built on it:
//! pseudo-inverse, numerical rank, condition number, minimum-norm least squares, the 3×3 SVD, the
//! polar decomposition and the rigid (Kabsch) and similarity (Procrustes) fits of point sets.
//!
//! The one-sided (Hestenes) Jacobi method rotates pairs of columns of `A` until all are mutually
//! orthogonal; their lengths are then the singular values. It never forms `AᵀA`, so small singular
//! values keep their relative accuracy, and it needs nothing but dot products and plane rotations of
//! contiguous vectors `[PERF-R2]`: the working matrix is stored so that the vectors being rotated
//! are its rows. Matrices with more columns than rows are handled by decomposing the transpose.
//!
//! The 3×3 routines ([`svd3`], [`polar3`], [`nearest_rotation3`], [`kabsch`], [`procrustes`]) use no
//! heap memory, so the rigid alignment inside registration loops `[REG-C2]` allocates nothing.
//! 3×3 matrices are row-major arrays, `m[row][column]`.

use super::dense::{DenseMatrix, MatRef};
use super::eigen::{MAX_SWEEPS, cross3, dot3, jacobi_rotation, mul3_vec, orthonormal_complement3, transpose3};
use super::error::{SolveError, check_dim, check_finite};
use super::vector::{axpy, dot, norm, norm_sq, rotate, scale};

/// Makes the `k` rows (each of length `l`) of the row-major `w` mutually orthogonal by one-sided
/// Jacobi rotations, applying the same rotations to the rows of the `k × k` matrix `r`.
fn orthogonalize_rows(w: &mut [f64], r: &mut [f64], k: usize, l: usize) -> Result<(), SolveError> {
    // Two rows count as orthogonal when their cosine is below this: the rounding level of a dot
    // product of length l.
    let tolerance = f64::EPSILON * (l as f64).sqrt();
    let mut worst = 0.0_f64;
    for _ in 0..MAX_SWEEPS {
        worst = 0.0;
        for p in 0..k {
            for q in p + 1..k {
                let (head, tail) = w.split_at_mut(q * l);
                let (wp, wq) = (&mut head[p * l..(p + 1) * l], &mut tail[..l]);
                let (alpha, beta, gamma) = (norm_sq(wp), norm_sq(wq), dot(wp, wq));
                let lengths = alpha.sqrt() * beta.sqrt();
                if gamma.abs() <= tolerance * lengths {
                    continue;
                }
                worst = worst.max(gamma.abs() / lengths);
                let (c, s, _) = jacobi_rotation(alpha, beta, gamma);
                rotate(wp, wq, c, s);
                let (head, tail) = r.split_at_mut(q * k);
                rotate(&mut head[p * k..(p + 1) * k], &mut tail[..k], c, s);
            }
        }
        if worst == 0.0 {
            return Ok(());
        }
    }
    Err(SolveError::NotConverged { iterations: MAX_SWEEPS, residual: worst })
}

/// Replaces row `j` of `w` (rows of length `l`) by a unit vector orthogonal to the rows before it,
/// which must be orthonormal. Requires `j < l`.
fn complete_row(w: &mut [f64], j: usize, l: usize) {
    let (done, rest) = w.split_at_mut(j * l);
    let row = &mut rest[..l];
    // The coordinate axis that the finished rows cover least: its residual has squared length
    // 1 − Σ rows[·][axis]², which is at least (l − j)/l for the best axis.
    let mut best_axis = 0;
    let mut best_residual = -1.0;
    for axis in 0..l {
        let covered: f64 = done.chunks_exact(l).map(|finished| finished[axis] * finished[axis]).sum();
        if 1.0 - covered > best_residual {
            best_residual = 1.0 - covered;
            best_axis = axis;
        }
    }
    row.fill(0.0);
    row[best_axis] = 1.0;
    // Gram-Schmidt, twice for full orthogonality.
    for _ in 0..2 {
        for finished in done.chunks_exact(l) {
            let projection = dot(finished, row);
            axpy(-projection, finished, row);
        }
    }
    let length = norm(row);
    if length > 0.0 {
        scale(1.0 / length, row);
    }
}

/// The thin singular value decomposition of an `m × n` matrix: `A = U·diag(σ)·Vᵀ` with
/// `k = min(m, n)` singular values in descending order, `U` of size `m × k` and `V` of size `n × k`,
/// both with orthonormal columns (also when `A` is rank deficient).
#[derive(Clone, Debug, PartialEq)]
pub struct Svd {
    m: usize,
    n: usize,
    /// `k × m`: row `j` is the left singular vector `uⱼ`.
    ut: DenseMatrix,
    /// The singular values, descending, non-negative.
    sigma: Vec<f64>,
    /// `k × n`: row `j` is the right singular vector `vⱼ`.
    vt: DenseMatrix,
}

impl Svd {
    /// Decomposes `a` (any shape). Fails on non-finite input, and with
    /// [`SolveError::NotConverged`] if the rotations do not converge (not observed for finite input).
    pub fn new(a: &DenseMatrix) -> Result<Self, SolveError> {
        check_finite(a.as_slice())?;
        let (m, n) = (a.rows(), a.cols());
        let tall = m >= n;
        // The k vectors being orthogonalised (the columns of A when tall, its rows otherwise) are
        // the rows of w, each of length l.
        let (k, l) = if tall { (n, m) } else { (m, n) };
        let mut w = if tall { a.transpose() } else { a.clone() };
        let mut r = DenseMatrix::identity(k);
        let mut sigma = vec![0.0; k];
        let magnitude = a.max_abs();
        if magnitude > 0.0 {
            // Scaled to unit magnitude so that squared lengths can neither overflow nor vanish.
            for v in w.as_mut_slice() {
                *v /= magnitude;
            }
            orthogonalize_rows(w.as_mut_slice(), r.as_mut_slice(), k, l)?;
            for (j, value) in sigma.iter_mut().enumerate() {
                *value = norm(w.row(j));
            }
        }
        for i in 0..k {
            let mut best = i;
            for j in i + 1..k {
                if sigma[j] > sigma[best] {
                    best = j;
                }
            }
            if best != i {
                sigma.swap(i, best);
                w.swap_rows(i, best)?;
                r.swap_rows(i, best)?;
            }
        }
        for (j, value) in sigma.iter_mut().enumerate() {
            if *value > f64::MIN_POSITIVE {
                scale(1.0 / *value, w.row_mut(j));
            } else {
                // A zero singular value leaves its vector undetermined: any unit vector orthogonal
                // to the others completes the basis.
                *value = 0.0;
                complete_row(w.as_mut_slice(), j, l);
            }
            *value *= magnitude;
        }
        check_finite(&sigma)?;
        let (ut, vt) = if tall { (w, r) } else { (r, w) };
        Ok(Self { m, n, ut, sigma, vt })
    }

    /// The number of rows of the decomposed matrix.
    pub fn rows(&self) -> usize {
        self.m
    }

    /// The number of columns of the decomposed matrix.
    pub fn cols(&self) -> usize {
        self.n
    }

    /// The `min(m, n)` singular values, descending and non-negative.
    pub fn singular_values(&self) -> &[f64] {
        &self.sigma
    }

    /// `Uᵀ` (`k × m`): row `j` is the left singular vector of `singular_values()[j]`. No allocation.
    pub fn u_transposed(&self) -> MatRef<'_> {
        self.ut.view()
    }

    /// `Vᵀ` (`k × n`): row `j` is the right singular vector of `singular_values()[j]`. No allocation.
    pub fn v_transposed(&self) -> MatRef<'_> {
        self.vt.view()
    }

    /// `U` (`m × k`): column `j` is the left singular vector of `singular_values()[j]`.
    pub fn u(&self) -> DenseMatrix {
        self.ut.transpose()
    }

    /// `V` (`n × k`): column `j` is the right singular vector of `singular_values()[j]`.
    pub fn v(&self) -> DenseMatrix {
        self.vt.transpose()
    }

    /// The spectral norm `‖A‖₂`: the largest singular value (0 for an empty matrix).
    pub fn norm2(&self) -> f64 {
        self.sigma.first().copied().unwrap_or(0.0)
    }

    /// The numerical rank with the default tolerance `max(m, n) · ε`.
    pub fn rank(&self) -> usize {
        self.rank_with_tolerance(self.default_tolerance())
    }

    /// The numerical rank: the number of singular values above `tolerance · σ_max`.
    pub fn rank_with_tolerance(&self, tolerance: f64) -> usize {
        let largest = self.norm2();
        if largest > 0.0 { self.sigma.iter().take_while(|&&s| s > tolerance * largest).count() } else { 0 }
    }

    /// The relative tolerance [`Svd::rank`] and [`Svd::pseudo_inverse`] use: `max(m, n) · ε`.
    pub fn default_tolerance(&self) -> f64 {
        self.m.max(self.n) as f64 * f64::EPSILON
    }

    /// The 2-norm condition number `σ_max / σ_min` over the `min(m, n)` singular values: infinite
    /// when the smallest is zero (a rank deficient matrix), 1 for an empty matrix.
    ///
    /// Solving a system with this matrix can lose about `log₁₀` of this many decimal digits.
    pub fn condition_number(&self) -> f64 {
        match (self.sigma.first(), self.sigma.last()) {
            (Some(&largest), Some(&smallest)) if smallest > 0.0 => largest / smallest,
            (Some(_), Some(_)) => f64::INFINITY,
            _ => 1.0,
        }
    }

    /// The Moore-Penrose pseudo-inverse `A⁺` (`n × m`) with the default tolerance.
    pub fn pseudo_inverse(&self) -> DenseMatrix {
        self.pseudo_inverse_with_tolerance(self.default_tolerance())
    }

    /// The pseudo-inverse `A⁺ = Σⱼ vⱼ·uⱼᵀ / σⱼ` over the singular values above `tolerance · σ_max`;
    /// the others are treated as zero.
    pub fn pseudo_inverse_with_tolerance(&self, tolerance: f64) -> DenseMatrix {
        let mut out = DenseMatrix::zeros(self.n, self.m);
        for j in 0..self.rank_with_tolerance(tolerance) {
            let u = self.ut.row(j);
            let inverse = 1.0 / self.sigma[j];
            for (i, &v) in self.vt.row(j).iter().enumerate() {
                if v != 0.0 {
                    axpy(v * inverse, u, out.row_mut(i));
                }
            }
        }
        out
    }

    /// Solves `min ‖A·x − b‖₂` into `x` and returns the numerical rank used. No allocation.
    ///
    /// When `A` is rank deficient or has more columns than rows, the minimiser is not unique: this
    /// is the one of smallest Euclidean norm, `x = A⁺·b`.
    pub fn solve_least_squares_into(&self, b: &[f64], x: &mut [f64]) -> Result<usize, SolveError> {
        check_dim("svd least squares: rhs length", self.m, b.len())?;
        check_dim("svd least squares: solution length", self.n, x.len())?;
        check_finite(b)?;
        let rank = self.rank();
        x.fill(0.0);
        for j in 0..rank {
            axpy(dot(self.ut.row(j), b) / self.sigma[j], self.vt.row(j), x);
        }
        Ok(rank)
    }

    /// Solves `min ‖A·x − b‖₂` (the minimum-norm solution); see [`Svd::solve_least_squares_into`].
    pub fn solve_least_squares(&self, b: &[f64]) -> Result<Vec<f64>, SolveError> {
        let mut x = vec![0.0; self.n];
        self.solve_least_squares_into(b, &mut x)?;
        Ok(x)
    }

    /// `U·diag(σ)·Vᵀ`: the decomposed matrix rebuilt from its factors.
    pub fn reconstruct(&self) -> DenseMatrix {
        let mut out = DenseMatrix::zeros(self.m, self.n);
        for (j, &sigma) in self.sigma.iter().enumerate() {
            let v = self.vt.row(j);
            for (i, &u) in self.ut.row(j).iter().enumerate() {
                if u != 0.0 {
                    axpy(u * sigma, v, out.row_mut(i));
                }
            }
        }
        out
    }
}

/// The pseudo-inverse of `a` with the default tolerance (see [`Svd::pseudo_inverse`]).
pub fn pseudo_inverse(a: &DenseMatrix) -> Result<DenseMatrix, SolveError> {
    Ok(Svd::new(a)?.pseudo_inverse())
}

/// The singular value decomposition of a 3×3 matrix, `A = U·diag(s)·Vᵀ`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Svd3 {
    /// The left singular vectors as the columns of an orthogonal matrix (determinant ±1).
    pub u: [[f64; 3]; 3],
    /// The singular values, descending and non-negative.
    pub s: [f64; 3],
    /// The right singular vectors as the columns of an orthogonal matrix (determinant ±1).
    pub v: [[f64; 3]; 3],
}

/// The determinant of a 3×3 matrix.
fn det3(m: &[[f64; 3]; 3]) -> f64 {
    dot3(m[0], cross3(m[1], m[2]))
}

/// `a · diag(d) · bᵀ` for row-major 3×3 matrices.
fn mul3_diag_transposed(a: &[[f64; 3]; 3], d: [f64; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (out_row, a_row) in out.iter_mut().zip(a) {
        for (value, b_row) in out_row.iter_mut().zip(b) {
            *value = a_row[0] * d[0] * b_row[0] + a_row[1] * d[1] * b_row[1] + a_row[2] * d[2] * b_row[2];
        }
    }
    out
}

/// The singular value decomposition of the 3×3 matrix `a` (row-major). No heap use.
///
/// `U` and `V` are orthogonal also for rank deficient input (the undetermined vectors are completed
/// to a basis); their determinants can be +1 or −1. Fails on non-finite input.
pub fn svd3(a: &[[f64; 3]; 3]) -> Result<Svd3, SolveError> {
    check_finite(a.as_flattened())?;
    let magnitude = a.as_flattened().iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    if magnitude == 0.0 {
        return Ok(Svd3 { u: identity, s: [0.0; 3], v: identity });
    }
    // Rows of w: the columns of A, scaled to unit magnitude. Rows of r: the right singular vectors.
    let mut w = transpose3(a);
    for value in w.as_flattened_mut() {
        *value /= magnitude;
    }
    let mut r = identity;
    orthogonalize_rows(w.as_flattened_mut(), r.as_flattened_mut(), 3, 3)?;
    let mut s = [dot3(w[0], w[0]).sqrt(), dot3(w[1], w[1]).sqrt(), dot3(w[2], w[2]).sqrt()];
    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        if s[j] > s[i] {
            s.swap(i, j);
            w.swap(i, j);
            r.swap(i, j);
        }
    }
    let mut rank = 0;
    for (value, row) in s.iter_mut().zip(&mut w) {
        if *value > f64::MIN_POSITIVE {
            let inverse = 1.0 / *value;
            *row = [row[0] * inverse, row[1] * inverse, row[2] * inverse];
            rank += 1;
        } else {
            *value = 0.0;
        }
    }
    // The magnitude is non-zero, so the rank is at least one.
    if rank == 1 {
        let (second, third) = orthonormal_complement3(w[0]);
        w[1] = second;
        w[2] = third;
    } else if rank == 2 {
        w[2] = cross3(w[0], w[1]);
    }
    let s = s.map(|value| value * magnitude);
    check_finite(&s)?;
    Ok(Svd3 { u: transpose3(&w), s, v: transpose3(&r) })
}

/// The polar decomposition of a 3×3 matrix, `A = Q·S`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Polar3 {
    /// The orthogonal factor `Q = U·Vᵀ`: the orthogonal matrix nearest to `A` in the Frobenius norm.
    /// Its determinant has the sign of `det A` (so it is a reflection when `A` inverts orientation;
    /// see [`nearest_rotation3`] for the factor constrained to rotations).
    pub orthogonal: [[f64; 3]; 3],
    /// The symmetric positive semi-definite stretch `S = V·diag(s)·Vᵀ`.
    pub stretch: [[f64; 3]; 3],
}

/// The polar decomposition `a = Q·S` of a 3×3 matrix (row-major). No heap use.
///
/// For a singular `a` the orthogonal factor is not unique; the one returned is still orthogonal.
pub fn polar3(a: &[[f64; 3]; 3]) -> Result<Polar3, SolveError> {
    let svd = svd3(a)?;
    let mut stretch = mul3_diag_transposed(&svd.v, svd.s, &svd.v);
    for (i, j) in [(1, 0), (2, 0), (2, 1)] {
        let mean = f64::midpoint(stretch[i][j], stretch[j][i]);
        stretch[i][j] = mean;
        stretch[j][i] = mean;
    }
    Ok(Polar3 { orthogonal: mul3_diag_transposed(&svd.u, [1.0; 3], &svd.v), stretch })
}

/// The proper rotation `R` (determinant +1) that maximises `trace(Rᵀ·A)`, from the factors of `a`.
fn rotation_from_svd(svd: &Svd3) -> ([[f64; 3]; 3], f64) {
    let sign = if det3(&svd.u) * det3(&svd.v) < 0.0 { -1.0 } else { 1.0 };
    (mul3_diag_transposed(&svd.u, [1.0, 1.0, sign], &svd.v), sign)
}

/// The rotation matrix (orthogonal, determinant +1) nearest to the 3×3 matrix `a` in the Frobenius
/// norm. No heap use.
///
/// Unlike the orthogonal factor of [`polar3`] the result is never a reflection: when `det a < 0` the
/// direction of the smallest singular value is flipped. This re-orthonormalises drifting rotation
/// matrices and extracts the rotation of a deformation gradient.
pub fn nearest_rotation3(a: &[[f64; 3]; 3]) -> Result<[[f64; 3]; 3], SolveError> {
    Ok(rotation_from_svd(&svd3(a)?).0)
}

/// A similarity transform `p ↦ scale · rotation · p + translation` fitted to point pairs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidFit {
    /// The rotation (row-major, orthogonal, determinant +1: never a reflection).
    pub rotation: [[f64; 3]; 3],
    /// The translation, applied after rotation and scale.
    pub translation: [f64; 3],
    /// The uniform scale: exactly 1 from [`kabsch`], fitted by [`procrustes`].
    pub scale: f64,
    /// The weighted root-mean-square distance between the transformed source points and their
    /// targets, `sqrt(Σ wᵢ·‖T(sourceᵢ) − targetᵢ‖² / Σ wᵢ)`.
    pub rms_error: f64,
}

impl RigidFit {
    /// Transforms a point: `scale · rotation · p + translation`.
    #[inline]
    pub fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        let rotated = mul3_vec(&self.rotation, p);
        [
            self.scale * rotated[0] + self.translation[0],
            self.scale * rotated[1] + self.translation[1],
            self.scale * rotated[2] + self.translation[2],
        ]
    }
}

/// The rigid transform (rotation and translation) that best maps `source` onto `target` in the
/// weighted least-squares sense: the Kabsch algorithm. No heap use.
///
/// `source[i]` corresponds to `target[i]`; `weights`, when given, holds one non-negative weight per
/// pair (a zero weight removes the pair). The rotation is always proper, also when the point sets
/// are mirror images of each other, planar or collinear (in the last case the rotation about the
/// line is not determined by the data and one valid choice is returned).
///
/// Errors: differing lengths, no points, a negative weight, weights summing to zero, non-finite
/// input.
pub fn kabsch(source: &[[f64; 3]], target: &[[f64; 3]], weights: Option<&[f64]>) -> Result<RigidFit, SolveError> {
    fit_points(source, target, weights, false)
}

/// The similarity transform (rotation, translation and uniform scale) that best maps `source` onto
/// `target` in the weighted least-squares sense: Procrustes alignment with Umeyama's reflection-safe
/// scale. No heap use.
///
/// Same conventions and errors as [`kabsch`]; additionally [`SolveError::Singular`] when all source
/// points coincide, which leaves the scale undetermined.
pub fn procrustes(source: &[[f64; 3]], target: &[[f64; 3]], weights: Option<&[f64]>) -> Result<RigidFit, SolveError> {
    fit_points(source, target, weights, true)
}

/// The weighted centroid of `points` (`total` is the sum of the weights).
fn centroid(points: &[[f64; 3]], weights: Option<&[f64]>, total: f64) -> [f64; 3] {
    let mut sum = [0.0; 3];
    for (i, p) in points.iter().enumerate() {
        let w = weights.and_then(|w| w.get(i)).copied().unwrap_or(1.0);
        sum = [sum[0] + w * p[0], sum[1] + w * p[1], sum[2] + w * p[2]];
    }
    [sum[0] / total, sum[1] / total, sum[2] / total]
}

/// The shared implementation of [`kabsch`] and [`procrustes`].
fn fit_points(
    source: &[[f64; 3]],
    target: &[[f64; 3]],
    weights: Option<&[f64]>,
    with_scale: bool,
) -> Result<RigidFit, SolveError> {
    check_dim("point fit: target point count", source.len(), target.len())?;
    if source.is_empty() {
        return Err(SolveError::InvalidParameter { what: "point fit: no points" });
    }
    check_finite(source.as_flattened())?;
    check_finite(target.as_flattened())?;
    let total = if let Some(weights) = weights {
        check_dim("point fit: weight count", source.len(), weights.len())?;
        check_finite(weights)?;
        if weights.iter().any(|&w| w < 0.0) {
            return Err(SolveError::InvalidParameter { what: "point fit: negative weight" });
        }
        weights.iter().sum()
    } else {
        source.len() as f64
    };
    if total <= 0.0 {
        return Err(SolveError::InvalidParameter { what: "point fit: weights sum to zero" });
    }
    let source_centre = centroid(source, weights, total);
    let target_centre = centroid(target, weights, total);
    // The cross-covariance H = Σ w·(y − ȳ)·(x − x̄)ᵀ and the spread of the source Σ w·‖x − x̄‖².
    let mut covariance = [[0.0; 3]; 3];
    let mut source_spread = 0.0;
    for (i, (p, q)) in source.iter().zip(target).enumerate() {
        let w = weights.and_then(|w| w.get(i)).copied().unwrap_or(1.0);
        let x = [p[0] - source_centre[0], p[1] - source_centre[1], p[2] - source_centre[2]];
        let y = [q[0] - target_centre[0], q[1] - target_centre[1], q[2] - target_centre[2]];
        source_spread += w * dot3(x, x);
        for (row, yi) in covariance.iter_mut().zip(y) {
            let wy = w * yi;
            *row = [row[0] + wy * x[0], row[1] + wy * x[1], row[2] + wy * x[2]];
        }
    }
    let svd = svd3(&covariance)?;
    let (rotation, sign) = rotation_from_svd(&svd);
    let scale = if with_scale {
        if source_spread <= 0.0 {
            return Err(SolveError::Singular);
        }
        (svd.s[0] + svd.s[1] + sign * svd.s[2]) / source_spread
    } else {
        1.0
    };
    let rotated_centre = mul3_vec(&rotation, source_centre);
    let translation = [
        target_centre[0] - scale * rotated_centre[0],
        target_centre[1] - scale * rotated_centre[1],
        target_centre[2] - scale * rotated_centre[2],
    ];
    let mut fit = RigidFit { rotation, translation, scale, rms_error: 0.0 };
    let mut error = 0.0;
    for (i, (p, q)) in source.iter().zip(target).enumerate() {
        let w = weights.and_then(|w| w.get(i)).copied().unwrap_or(1.0);
        let mapped = fit.apply(*p);
        let d = [mapped[0] - q[0], mapped[1] - q[1], mapped[2] - q[2]];
        error += w * dot3(d, d);
    }
    fit.rms_error = (error / total).sqrt();
    if !(fit.rms_error.is_finite() && fit.scale.is_finite()) {
        return Err(SolveError::NonFinite);
    }
    Ok(fit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solve::eigen::SymmetricEigen;
    use crate::solve::qr::Qr;
    use crate::solve::testutil::{max_abs_diff, naive_mul, orthonormality_error, random_matrix, random_vec};
    use humanart_core::rng::Rng;

    const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    /// The product of two row-major 3×3 matrices.
    fn mul3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
        mul3_diag_transposed(a, [1.0; 3], &transpose3(b))
    }

    fn hilbert(n: usize) -> DenseMatrix {
        DenseMatrix::from_fn(n, n, |i, j| 1.0 / (i + j + 1) as f64)
    }

    /// The largest deviation of the factors from the SVD properties: reconstruction, orthonormal
    /// singular vectors, sorted non-negative singular values.
    fn check_factors(a: &DenseMatrix, svd: &Svd) -> f64 {
        let k = a.rows().min(a.cols());
        assert_eq!(svd.singular_values().len(), k);
        assert!(svd.singular_values().windows(2).all(|w| w[0] >= w[1]));
        assert!(svd.singular_values().iter().all(|&s| s >= 0.0));
        assert_eq!((svd.u().rows(), svd.u().cols()), (a.rows(), k));
        assert_eq!((svd.v().rows(), svd.v().cols()), (a.cols(), k));
        max_abs_diff(&svd.reconstruct(), a).max(orthonormality_error(&svd.u())).max(orthonormality_error(&svd.v()))
    }

    fn random3(rng: &mut Rng) -> [[f64; 3]; 3] {
        let mut m = [[0.0; 3]; 3];
        for value in m.as_flattened_mut() {
            *value = rng.range_f64(-1.0, 1.0);
        }
        m
    }

    /// A rotation by `angle` about the (normalised) `axis`: Rodrigues' formula.
    fn rotation(axis: [f64; 3], angle: f64) -> [[f64; 3]; 3] {
        let length = dot3(axis, axis).sqrt();
        let [x, y, z] = axis.map(|v| v / length);
        let (s, c) = angle.sin_cos();
        let t = 1.0 - c;
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    }

    fn max_diff3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> f64 {
        a.as_flattened().iter().zip(b.as_flattened()).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
    }

    fn orthogonality3(m: &[[f64; 3]; 3]) -> f64 {
        max_diff3(&mul3(&transpose3(m), m), &IDENTITY)
    }

    fn check_svd3(a: &[[f64; 3]; 3], svd: &Svd3) -> f64 {
        assert!(svd.s[0] >= svd.s[1] && svd.s[1] >= svd.s[2] && svd.s[2] >= 0.0);
        let rebuilt = mul3_diag_transposed(&svd.u, svd.s, &svd.v);
        max_diff3(&rebuilt, a).max(orthogonality3(&svd.u)).max(orthogonality3(&svd.v))
    }

    fn random_points(rng: &mut Rng, count: usize) -> Vec<[f64; 3]> {
        (0..count).map(|_| [rng.range_f64(-1.0, 1.0), rng.range_f64(-1.0, 1.0), rng.range_f64(-1.0, 1.0)]).collect()
    }

    #[test]
    fn ac_wp02_1_hand_solved_singular_values() {
        // AᵀA = [[25, -15], [-15, 25]] has the eigenvalues 40 and 10.
        let a = DenseMatrix::from_rows(&[[4.0, 0.0], [3.0, -5.0]]).unwrap();
        let svd = Svd::new(&a).unwrap();
        assert!((svd.singular_values()[0] - 40.0_f64.sqrt()).abs() < 1e-14);
        assert!((svd.singular_values()[1] - 10.0_f64.sqrt()).abs() < 1e-14);
        assert!(check_factors(&a, &svd) < 1e-14);
        assert!((svd.condition_number() - 2.0).abs() < 1e-14);
        assert!((svd.norm2() - 40.0_f64.sqrt()).abs() < 1e-14);
        // A diagonal matrix with a negative entry: singular values are the magnitudes, sorted.
        let d = DenseMatrix::from_diagonal(&[1.0, -3.0, 2.0]);
        let svd = Svd::new(&d).unwrap();
        assert_eq!(svd.singular_values(), &[3.0, 2.0, 1.0]);
        assert!(check_factors(&d, &svd) < 1e-15);
    }

    #[test]
    fn ac_wp02_1_factors_reproduce_matrices_of_every_shape() {
        let mut rng = Rng::new(0x5D);
        for (m, n) in [(1, 1), (4, 4), (7, 3), (3, 7), (20, 11), (11, 20), (5, 1), (1, 5), (40, 40)] {
            let a = random_matrix(&mut rng, m, n);
            let svd = Svd::new(&a).unwrap();
            assert!(check_factors(&a, &svd) < 1e-13, "{m}x{n}");
            assert_eq!(svd.rank(), m.min(n), "{m}x{n}");
            assert_eq!((svd.rows(), svd.cols()), (m, n));
            // The squared singular values are the eigenvalues of the smaller Gram matrix.
            let gram = if m >= n { a.gram() } else { a.outer_gram() };
            let eigen = SymmetricEigen::new(&gram).unwrap();
            for (s, lambda) in svd.singular_values().iter().zip(eigen.values()) {
                assert!((s * s - lambda).abs() < 1e-12, "{m}x{n}");
            }
        }
    }

    #[test]
    fn ac_wp02_1_hilbert_condition_numbers() {
        // Published 2-norm condition numbers of Hilbert matrices. They grow like e^{3.5 n}: at n = 8
        // a solve already loses ten of sixteen digits, which is why the bound is relative.
        for (n, published) in [(3, 5.240_567_775_86e2), (5, 4.766_072_502_4e5), (8, 1.525_757_55e10)] {
            let condition = Svd::new(&hilbert(n)).unwrap().condition_number();
            assert!((condition / published - 1.0).abs() < 1e-6, "n = {n}: {condition}");
        }
    }

    #[test]
    fn ac_wp02_1_rank_deficiency_pseudo_inverse_and_minimum_norm() {
        let mut rng = Rng::new(0x5E);
        for (m, n) in [(10, 6), (6, 10)] {
            // Rank 3: the product of m × 3 and 3 × n factors.
            let a = naive_mul(&random_matrix(&mut rng, m, 3), &random_matrix(&mut rng, 3, n));
            let svd = Svd::new(&a).unwrap();
            assert_eq!(svd.rank(), 3);
            assert!(svd.singular_values()[3] < 1e-14);
            assert!(svd.condition_number() > 1e14);
            assert!(check_factors(&a, &svd) < 1e-13);
            // The four Moore-Penrose conditions.
            let pinv = svd.pseudo_inverse();
            assert_eq!((pinv.rows(), pinv.cols()), (n, m));
            let a_pinv = naive_mul(&a, &pinv);
            let pinv_a = naive_mul(&pinv, &a);
            assert!(max_abs_diff(&naive_mul(&a_pinv, &a), &a) < 1e-13);
            assert!(max_abs_diff(&naive_mul(&pinv_a, &pinv), &pinv) < 1e-12);
            assert!(a_pinv.is_symmetric(1e-13));
            assert!(pinv_a.is_symmetric(1e-13));
            assert_eq!(max_abs_diff(&pseudo_inverse(&a).unwrap(), &pinv), 0.0);
            // Least squares: the same residual as the QR basic solution, with a norm no larger.
            let b = random_vec(&mut rng, m);
            let x = svd.solve_least_squares(&b).unwrap();
            let x_basic = Qr::new(&a).unwrap().solve_least_squares(&b).unwrap();
            let residual = |x: &[f64]| {
                let ax = a.mul_vec(x).unwrap();
                ax.iter().zip(&b).map(|(p, q)| (p - q) * (p - q)).sum::<f64>().sqrt()
            };
            assert!((residual(&x) - residual(&x_basic)).abs() < 1e-12);
            assert!(norm(&x) <= norm(&x_basic) + 1e-12);
            let from_pinv = pinv.mul_vec(&b).unwrap();
            for (p, q) in x.iter().zip(&from_pinv) {
                assert!((p - q).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn ac_wp02_1_full_rank_systems_are_solved() {
        let mut rng = Rng::new(0x5F);
        let mut a = random_matrix(&mut rng, 9, 9);
        a.add_to_diagonal(3.0);
        let expect = random_vec(&mut rng, 9);
        let b = a.mul_vec(&expect).unwrap();
        let svd = Svd::new(&a).unwrap();
        let x = svd.solve_least_squares(&b).unwrap();
        for (p, q) in x.iter().zip(&expect) {
            assert!((p - q).abs() < 1e-13);
        }
        assert!(max_abs_diff(&naive_mul(&a, &svd.pseudo_inverse()), &DenseMatrix::identity(9)) < 1e-13);
        // A wide system: the minimum-norm solution lies in the row space, i.e. x = Aᵀ·y.
        let a = random_matrix(&mut rng, 3, 8);
        let b = random_vec(&mut rng, 3);
        let svd = Svd::new(&a).unwrap();
        let x = svd.solve_least_squares(&b).unwrap();
        let ax = a.mul_vec(&x).unwrap();
        for (p, q) in ax.iter().zip(&b) {
            assert!((p - q).abs() < 1e-13);
        }
        let y = crate::solve::lu::lu_solve(&a.outer_gram(), &b).unwrap();
        let expect = a.tr_mul_vec(&y).unwrap();
        for (p, q) in x.iter().zip(&expect) {
            assert!((p - q).abs() < 1e-12);
        }
    }

    #[test]
    fn zero_and_degenerate_matrices_keep_orthonormal_factors() {
        for (m, n) in [(4, 3), (3, 4), (3, 3)] {
            let zero = DenseMatrix::zeros(m, n);
            let svd = Svd::new(&zero).unwrap();
            assert_eq!(svd.rank(), 0);
            assert!(svd.singular_values().iter().all(|&s| s == 0.0));
            assert!(svd.condition_number().is_infinite());
            assert!(check_factors(&zero, &svd) < 1e-15);
            assert_eq!(svd.pseudo_inverse().max_abs(), 0.0);
            assert_eq!(svd.solve_least_squares(&vec![1.0; m]).unwrap(), vec![0.0; n]);
        }
        // One non-zero element: rank one, the other vectors are completed.
        let mut a = DenseMatrix::zeros(5, 3);
        a[(2, 1)] = -4.0;
        let svd = Svd::new(&a).unwrap();
        assert_eq!(svd.singular_values(), &[4.0, 0.0, 0.0]);
        assert_eq!(svd.rank(), 1);
        assert!(check_factors(&a, &svd) < 1e-15);
        let pinv = svd.pseudo_inverse();
        assert!((pinv[(1, 2)] + 0.25).abs() < 1e-15);
        // Extreme magnitudes.
        for magnitude in [1e-300, 1e300] {
            let a = DenseMatrix::from_rows(&[[4.0 * magnitude, 0.0], [3.0 * magnitude, -5.0 * magnitude]]).unwrap();
            let svd = Svd::new(&a).unwrap();
            assert!((svd.singular_values()[0] / magnitude - 40.0_f64.sqrt()).abs() < 1e-13);
            assert!((svd.singular_values()[1] / magnitude - 10.0_f64.sqrt()).abs() < 1e-13);
        }
        let empty = Svd::new(&DenseMatrix::zeros(0, 4)).unwrap();
        assert!(empty.singular_values().is_empty());
        assert_eq!(empty.rank(), 0);
        assert_eq!(empty.condition_number(), 1.0);
        assert_eq!(empty.norm2(), 0.0);
        assert_eq!(empty.solve_least_squares(&[]).unwrap(), vec![0.0; 4]);
    }

    #[test]
    fn bad_input_is_an_error() {
        let mut a = DenseMatrix::zeros(2, 2);
        a[(0, 1)] = f64::NAN;
        assert_eq!(Svd::new(&a).unwrap_err(), SolveError::NonFinite);
        assert_eq!(pseudo_inverse(&a).unwrap_err(), SolveError::NonFinite);
        let svd = Svd::new(&DenseMatrix::identity(3)).unwrap();
        assert!(matches!(svd.solve_least_squares(&[1.0]), Err(SolveError::DimensionMismatch { .. })));
        assert_eq!(svd.solve_least_squares(&[1.0, f64::INFINITY, 0.0]), Err(SolveError::NonFinite));
        let mut x = [0.0; 2];
        assert!(svd.solve_least_squares_into(&[0.0; 3], &mut x).is_err());
        assert_eq!(svd3(&[[0.0, f64::NAN, 0.0], [0.0; 3], [0.0; 3]]).unwrap_err(), SolveError::NonFinite);
        assert_eq!(polar3(&[[f64::INFINITY, 0.0, 0.0], [0.0; 3], [0.0; 3]]).unwrap_err(), SolveError::NonFinite);
        assert!(nearest_rotation3(&[[f64::NAN; 3]; 3]).is_err());
    }

    #[test]
    fn ac_wp02_1_three_by_three_svd() {
        let mut rng = Rng::new(0x60);
        for _ in 0..200 {
            let a = random3(&mut rng);
            let svd = svd3(&a).unwrap();
            assert!(check_svd3(&a, &svd) < 1e-14);
            let dense = DenseMatrix::from_rows(&a).unwrap();
            let general = Svd::new(&dense).unwrap();
            for (s, expect) in svd.s.iter().zip(general.singular_values()) {
                assert!((s - expect).abs() < 1e-14);
            }
        }
        // Rank two (third column = sum of the first two), rank one (outer product) and rank zero.
        let rank_two = [[1.0, 2.0, 3.0], [0.0, 1.0, 1.0], [2.0, -1.0, 1.0]];
        let svd = svd3(&rank_two).unwrap();
        assert!(svd.s[2] < 1e-15 && svd.s[1] > 0.5);
        assert!(check_svd3(&rank_two, &svd) < 1e-14);
        let rank_one = [[2.0, 4.0, -2.0], [1.0, 2.0, -1.0], [3.0, 6.0, -3.0]];
        let svd = svd3(&rank_one).unwrap();
        assert!((svd.s[0] - (14.0_f64 * 6.0).sqrt()).abs() < 1e-14 && svd.s[1] < 1e-15 && svd.s[2] < 1e-15);
        assert!(check_svd3(&rank_one, &svd) < 1e-14);
        let svd = svd3(&[[0.0; 3]; 3]).unwrap();
        assert_eq!((svd.u, svd.s, svd.v), (IDENTITY, [0.0; 3], IDENTITY));
        let tiny = [[3e-300, 0.0, 0.0], [0.0, 1e-300, 0.0], [0.0, 0.0, 2e-300]];
        let svd = svd3(&tiny).unwrap();
        assert!((svd.s[0] / 3e-300 - 1.0).abs() < 1e-14 && (svd.s[2] / 1e-300 - 1.0).abs() < 1e-14);
    }

    #[test]
    fn ac_wp02_1_polar_decomposition_and_nearest_rotation() {
        let mut rng = Rng::new(0x61);
        for _ in 0..100 {
            let a = random3(&mut rng);
            let polar = polar3(&a).unwrap();
            assert!(orthogonality3(&polar.orthogonal) < 1e-14);
            assert_eq!(max_diff3(&polar.stretch, &transpose3(&polar.stretch)), 0.0);
            assert!(max_diff3(&mul3(&polar.orthogonal, &polar.stretch), &a) < 1e-14);
            // The stretch is positive semi-definite and the orthogonal factor carries the sign of det A.
            let stretch_eigen = crate::solve::eigen::symmetric_eigen3(&polar.stretch).unwrap();
            assert!(stretch_eigen.values[2] > -1e-14);
            assert!(det3(&polar.orthogonal) * det3(&a) > 0.0);
            let nearest = nearest_rotation3(&a).unwrap();
            assert!(orthogonality3(&nearest) < 1e-14);
            assert!((det3(&nearest) - 1.0).abs() < 1e-14);
        }
        // A rotation followed by a known symmetric stretch is recovered exactly.
        let q = rotation([1.0, 2.0, -0.5], 0.9);
        let stretch = [[2.0, 0.3, 0.0], [0.3, 1.5, -0.2], [0.0, -0.2, 1.0]];
        let polar = polar3(&mul3(&q, &stretch)).unwrap();
        assert!(max_diff3(&polar.orthogonal, &q) < 1e-14);
        assert!(max_diff3(&polar.stretch, &stretch) < 1e-14);
        assert!(max_diff3(&nearest_rotation3(&q).unwrap(), &q) < 1e-15);
        // A mirror: the polar factor is the mirror itself, the nearest rotation is a proper one.
        let mirror = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]];
        assert!(max_diff3(&polar3(&mirror).unwrap().orthogonal, &mirror) < 1e-15);
        assert!((det3(&nearest_rotation3(&mirror).unwrap()) - 1.0).abs() < 1e-15);
        // Singular input still gives orthogonal factors.
        let polar = polar3(&[[0.0; 3]; 3]).unwrap();
        assert_eq!(polar.orthogonal, IDENTITY);
        assert_eq!(polar.stretch, [[0.0; 3]; 3]);
    }

    #[test]
    fn ac_wp02_1_kabsch_recovers_a_known_rigid_motion() {
        let mut rng = Rng::new(0x62);
        for count in [3, 4, 10, 200] {
            let source = random_points(&mut rng, count);
            let r = rotation([rng.normal(), rng.normal(), rng.normal() + 0.1], rng.range_f64(-3.0, 3.0));
            let t = [0.3, -1.2, 2.5];
            let target: Vec<[f64; 3]> = source
                .iter()
                .map(|&p| {
                    let q = mul3_vec(&r, p);
                    [q[0] + t[0], q[1] + t[1], q[2] + t[2]]
                })
                .collect();
            let fit = kabsch(&source, &target, None).unwrap();
            assert!(max_diff3(&fit.rotation, &r) < 1e-13, "count = {count}");
            for (got, want) in fit.translation.iter().zip(t) {
                assert!((got - want).abs() < 1e-13);
            }
            assert_eq!(fit.scale, 1.0);
            assert!(fit.rms_error < 1e-14);
            // With weights the exact motion is still the optimum.
            let weights: Vec<f64> = (0..count).map(|_| rng.range_f64(0.0, 2.0)).collect();
            let weighted = kabsch(&source, &target, Some(&weights)).unwrap();
            assert!(max_diff3(&weighted.rotation, &r) < 1e-13);
            assert!(weighted.rms_error < 1e-14);
        }
    }

    #[test]
    fn ac_wp02_1_procrustes_recovers_scale() {
        let mut rng = Rng::new(0x63);
        let source = random_points(&mut rng, 30);
        let r = rotation([0.2, -1.0, 0.4], 2.1);
        let target: Vec<[f64; 3]> = source
            .iter()
            .map(|&p| {
                let q = mul3_vec(&r, p);
                [2.5 * q[0] + 1.0, 2.5 * q[1] - 2.0, 2.5 * q[2] + 0.5]
            })
            .collect();
        let fit = procrustes(&source, &target, None).unwrap();
        assert!((fit.scale - 2.5).abs() < 1e-13);
        assert!(max_diff3(&fit.rotation, &r) < 1e-13);
        assert!(fit.rms_error < 1e-13);
        let mapped = fit.apply(source[7]);
        for k in 0..3 {
            assert!((mapped[k] - target[7][k]).abs() < 1e-13);
        }
        // The rigid fit of the same data cannot absorb the scale.
        assert!(kabsch(&source, &target, None).unwrap().rms_error > 0.1);
        // Coincident source points leave the scale undetermined.
        assert_eq!(procrustes(&[[1.0; 3]; 4], &target[..4], None), Err(SolveError::Singular));
        // The rigid fit of coincident points is a pure translation.
        let fit = kabsch(&[[1.0; 3]; 4], &[[3.0, 1.0, 0.0]; 4], None).unwrap();
        assert_eq!(fit.rotation, IDENTITY);
        assert_eq!(fit.translation, [2.0, 0.0, -1.0]);
    }

    #[test]
    fn ac_wp02_1_kabsch_is_optimal_and_reflection_safe() {
        let mut rng = Rng::new(0x64);
        let source = random_points(&mut rng, 40);
        // A mirror image cannot be reached by a rotation: the fit stays proper and reports the gap.
        let mirrored: Vec<[f64; 3]> = source.iter().map(|p| [p[0], p[1], -p[2]]).collect();
        let fit = kabsch(&source, &mirrored, None).unwrap();
        assert!((det3(&fit.rotation) - 1.0).abs() < 1e-14);
        assert!(orthogonality3(&fit.rotation) < 1e-14);
        assert!(fit.rms_error > 0.1);
        let fit = procrustes(&source, &mirrored, None).unwrap();
        assert!((det3(&fit.rotation) - 1.0).abs() < 1e-14 && fit.scale > 0.0 && fit.scale < 1.0);
        // Noisy targets: no nearby rigid motion does better than the fit.
        let r = rotation([1.0, 1.0, 0.2], 0.7);
        let noisy: Vec<[f64; 3]> = source
            .iter()
            .map(|&p| {
                let q = mul3_vec(&r, p);
                [q[0] + 0.05 * rng.normal(), q[1] + 0.05 * rng.normal(), q[2] + 0.05 * rng.normal()]
            })
            .collect();
        let weights: Vec<f64> = (0..40).map(|_| rng.range_f64(0.1, 1.0)).collect();
        let fit = kabsch(&source, &noisy, Some(&weights)).unwrap();
        let cost = |candidate: &RigidFit| {
            let total: f64 = weights.iter().sum();
            let sum: f64 = source
                .iter()
                .zip(&noisy)
                .zip(&weights)
                .map(|((p, q), w)| {
                    let m = candidate.apply(*p);
                    w * ((m[0] - q[0]).powi(2) + (m[1] - q[1]).powi(2) + (m[2] - q[2]).powi(2))
                })
                .sum();
            (sum / total).sqrt()
        };
        assert!((cost(&fit) - fit.rms_error).abs() < 1e-15);
        for _ in 0..50 {
            let nudge = rotation([rng.normal(), rng.normal(), rng.normal() + 0.01], 0.02 * rng.normal());
            let mut candidate = fit;
            candidate.rotation = mul3(&nudge, &fit.rotation);
            candidate.translation[rng.index(3)] += 0.01 * rng.normal();
            assert!(cost(&candidate) >= fit.rms_error - 1e-12);
        }
        // A zero weight removes a pair: the outlier does not move the fit.
        let mut with_outlier = source.clone();
        let mut exact: Vec<[f64; 3]> = source.iter().map(|&p| mul3_vec(&r, p)).collect();
        with_outlier.push([5.0, 5.0, 5.0]);
        exact.push([-40.0, 3.0, 9.0]);
        let mut weights = vec![1.0; 41];
        weights[40] = 0.0;
        let fit = kabsch(&with_outlier, &exact, Some(&weights)).unwrap();
        assert!(max_diff3(&fit.rotation, &r) < 1e-13 && fit.rms_error < 1e-14);
        // Planar and collinear point sets give proper rotations that still map the points.
        let planar: Vec<[f64; 3]> = source.iter().map(|p| [p[0], p[1], 0.0]).collect();
        let planar_target: Vec<[f64; 3]> = planar.iter().map(|&p| mul3_vec(&r, p)).collect();
        let fit = kabsch(&planar, &planar_target, None).unwrap();
        assert!(max_diff3(&fit.rotation, &r) < 1e-13 && fit.rms_error < 1e-14);
        let line: Vec<[f64; 3]> = source.iter().map(|p| [p[0], 2.0 * p[0], -p[0]]).collect();
        let line_target: Vec<[f64; 3]> = line.iter().map(|&p| mul3_vec(&r, p)).collect();
        let fit = kabsch(&line, &line_target, None).unwrap();
        assert!((det3(&fit.rotation) - 1.0).abs() < 1e-14 && fit.rms_error < 1e-14);
    }

    #[test]
    fn point_fit_rejects_bad_input() {
        let points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(matches!(kabsch(&points, &points[..2], None), Err(SolveError::DimensionMismatch { .. })));
        assert!(matches!(kabsch(&points, &points, Some(&[1.0, 1.0])), Err(SolveError::DimensionMismatch { .. })));
        assert!(matches!(kabsch(&[], &[], None), Err(SolveError::InvalidParameter { .. })));
        assert!(matches!(kabsch(&points, &points, Some(&[1.0, -1.0, 1.0])), Err(SolveError::InvalidParameter { .. })));
        assert!(matches!(kabsch(&points, &points, Some(&[0.0; 3])), Err(SolveError::InvalidParameter { .. })));
        assert_eq!(kabsch(&points, &points, Some(&[1.0, f64::NAN, 1.0])), Err(SolveError::NonFinite));
        let mut bad = points;
        bad[1][2] = f64::INFINITY;
        assert_eq!(kabsch(&bad, &points, None), Err(SolveError::NonFinite));
        assert_eq!(procrustes(&points, &bad, None), Err(SolveError::NonFinite));
        let fit = kabsch(&points, &points, None).unwrap();
        assert!(max_diff3(&fit.rotation, &IDENTITY) < 1e-15 && fit.rms_error < 1e-15);
    }
}
