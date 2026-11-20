//! Regularised and constrained least squares: Tikhonov (ridge) regression, non-negative least
//! squares (NNLS) and the Levenberg-Marquardt method for non-linear least squares.
//!
//! - [`tikhonov`] minimises `‖A·x − b‖² + λ·‖x − x₀‖²`. It solves the normal equations of whichever
//!   side of `A` is smaller: `(AᵀA + λI)` for tall matrices and the dual `(AAᵀ + λI)` for wide ones, so
//!   fitting many coefficients to few observations (the statistical shape model's case) costs the
//!   small system only.
//! - [`nnls`] is the Lawson-Hanson active-set method on the Gram matrix: the solver of blend weights
//!   and other quantities that cannot be negative.
//! - [`levenberg_marquardt`] minimises `½‖r(p)‖²` for a [`LeastSquaresProblem`] with an analytic
//!   Jacobian, with Marquardt's diagonal scaling and Nielsen's damping update.
//!
//! Each has an `_into` form that takes a workspace the caller keeps; once warm for a problem size, a
//! solve allocates nothing `[PERF-R2]`.

use super::cholesky::Cholesky;
use super::dense::DenseMatrix;
use super::error::{SolveError, check_dim, check_finite};
use super::vector::{axpy, dot, norm, norm_inf, norm_sq};

/// Maps the "not positive definite" of a normal-equation factorisation to what it means for least
/// squares: the columns (or rows) are linearly dependent.
fn rank_deficient(error: SolveError) -> SolveError {
    match error {
        SolveError::NotPositiveDefinite { .. } => SolveError::Singular,
        other => other,
    }
}

/// Reusable storage of [`tikhonov_into`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TikhonovWorkspace {
    normal: DenseMatrix,
    cholesky: Cholesky,
    rhs: Vec<f64>,
    misfit: Vec<f64>,
}

/// Minimises `‖A·x − b‖₂² + λ·‖x − x₀‖₂²` into `x`, with `x₀` = `prior` (zero when `None`).
///
/// `lambda` must be finite and non-negative. With `lambda = 0` this is ordinary least squares and
/// needs `A` of full column rank ([`SolveError::Singular`] otherwise; [`super::svd::Svd`] gives the
/// minimum-norm solution instead). Squaring the condition number in the normal equations is
/// harmless for the regularised systems this is meant for; use [`super::qr::Qr`] for ill-conditioned
/// unregularised ones.
pub fn tikhonov_into(
    a: &DenseMatrix,
    b: &[f64],
    lambda: f64,
    prior: Option<&[f64]>,
    x: &mut [f64],
    workspace: &mut TikhonovWorkspace,
) -> Result<(), SolveError> {
    let (m, n) = (a.rows(), a.cols());
    check_dim("tikhonov: rhs length", m, b.len())?;
    check_dim("tikhonov: solution length", n, x.len())?;
    if !(lambda.is_finite() && lambda >= 0.0) {
        return Err(SolveError::InvalidParameter { what: "tikhonov: lambda must be finite and >= 0" });
    }
    check_finite(a.as_slice())?;
    check_finite(b)?;
    if let Some(prior) = prior {
        check_dim("tikhonov: prior length", n, prior.len())?;
        check_finite(prior)?;
    }
    let TikhonovWorkspace { normal, cholesky, rhs, misfit } = workspace;
    // d = b − A·x₀; then x = x₀ + δ with δ minimising ‖A·δ − d‖² + λ‖δ‖².
    misfit.clear();
    misfit.extend_from_slice(b);
    if let Some(prior) = prior {
        for (i, d) in misfit.iter_mut().enumerate() {
            *d -= dot(a.row(i), prior);
        }
    }
    if m >= n || lambda == 0.0 {
        // Primal: (AᵀA + λI)·δ = Aᵀd.
        a.view().gram_into(normal);
        normal.add_to_diagonal(lambda);
        cholesky.factor(normal.view()).map_err(rank_deficient)?;
        a.tr_mul_vec_into(misfit, x)?;
        cholesky.solve_in_place(x)?;
    } else {
        // Dual: δ = Aᵀ·y with (AAᵀ + λI)·y = d; an m × m system.
        a.view().outer_gram_into(normal);
        normal.add_to_diagonal(lambda);
        cholesky.factor(normal.view()).map_err(rank_deficient)?;
        rhs.clear();
        rhs.extend_from_slice(misfit);
        cholesky.solve_in_place(rhs)?;
        a.tr_mul_vec_into(rhs, x)?;
    }
    if let Some(prior) = prior {
        axpy(1.0, prior, x);
    }
    check_finite(x)
}

/// Minimises `‖A·x − b‖₂² + λ·‖x − x₀‖₂²`; see [`tikhonov_into`].
pub fn tikhonov(a: &DenseMatrix, b: &[f64], lambda: f64, prior: Option<&[f64]>) -> Result<Vec<f64>, SolveError> {
    let mut x = vec![0.0; a.cols()];
    tikhonov_into(a, b, lambda, prior, &mut x, &mut TikhonovWorkspace::default())?;
    Ok(x)
}

/// Reusable storage of [`nnls_into`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NnlsWorkspace {
    gram: DenseMatrix,
    atb: Vec<f64>,
    gradient: Vec<f64>,
    passive: Vec<bool>,
    indices: Vec<usize>,
    sub: DenseMatrix,
    cholesky: Cholesky,
    trial: Vec<f64>,
    residual: Vec<f64>,
}

/// The outcome of [`nnls_into`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NnlsReport {
    /// The sub-problem solves used.
    pub iterations: usize,
    /// `‖A·x − b‖₂` at the solution.
    pub residual_norm: f64,
    /// The number of strictly positive coefficients.
    pub active: usize,
}

/// Solves the unconstrained least-squares problem on the passive set: `trial[P] = (AᵀA)[P,P]⁻¹·(Aᵀb)[P]`,
/// zero elsewhere. A relative ridge of the order of rounding keeps the factorisation defined when
/// passive columns are nearly dependent.
fn solve_passive(
    gram: &DenseMatrix,
    atb: &[f64],
    indices: &[usize],
    sub: &mut DenseMatrix,
    cholesky: &mut Cholesky,
    trial: &mut [f64],
) -> Result<(), SolveError> {
    let p = indices.len();
    sub.resize(p, p);
    let mut largest = 0.0_f64;
    for (r, &i) in indices.iter().enumerate() {
        let row = gram.row(i);
        for (c, &j) in indices.iter().enumerate() {
            sub[(r, c)] = row[j];
        }
        largest = largest.max(row[i]);
    }
    sub.add_to_diagonal(f64::EPSILON * largest * p as f64);
    cholesky.factor(sub.view()).map_err(rank_deficient)?;
    trial.fill(0.0);
    // The rhs is gathered into the first p slots of `trial`, solved there and scattered afterwards.
    for (slot, &i) in indices.iter().enumerate() {
        trial[slot] = atb[i];
    }
    cholesky.solve_in_place(&mut trial[..p])?;
    // Scatter back from the front, highest slot first: indices are increasing and index ≥ slot, so
    // a slot is never overwritten before it is read.
    for (slot, &i) in indices.iter().enumerate().rev() {
        let value = trial[slot];
        trial[slot] = 0.0;
        trial[i] = value;
    }
    Ok(())
}

/// Solves `min ‖A·x − b‖₂` subject to `x ≥ 0` (element-wise) into `x`: the Lawson-Hanson active-set
/// method on the Gram matrix `AᵀA`.
///
/// At the solution the Karush-Kuhn-Tucker conditions hold to rounding: `w = Aᵀ(b − A·x)` is zero
/// where `x > 0` and non-positive where `x = 0`. Errors: dimensions, non-finite input, and
/// [`SolveError::NotConverged`] if the active set cycles (a safeguard; not observed).
pub fn nnls_into(
    a: &DenseMatrix,
    b: &[f64],
    x: &mut [f64],
    workspace: &mut NnlsWorkspace,
) -> Result<NnlsReport, SolveError> {
    let (m, n) = (a.rows(), a.cols());
    check_dim("nnls: rhs length", m, b.len())?;
    check_dim("nnls: solution length", n, x.len())?;
    check_finite(a.as_slice())?;
    check_finite(b)?;
    let NnlsWorkspace { gram, atb, gradient, passive, indices, sub, cholesky, trial, residual } = workspace;
    a.view().gram_into(gram);
    atb.resize(n, 0.0);
    a.tr_mul_vec_into(b, atb)?;
    gradient.clear();
    gradient.extend_from_slice(atb);
    passive.clear();
    passive.resize(n, false);
    trial.resize(n, 0.0);
    indices.clear();
    indices.reserve(n);
    x.fill(0.0);
    // MATLAB's lsqnonneg tolerance: 10·ε·‖A‖₁·max(m, n), with ‖A‖₁ the largest column sum.
    let mut column_sums = 0.0_f64;
    for j in 0..n {
        column_sums = column_sums.max((0..m).map(|i| a.row(i)[j].abs()).sum());
    }
    let tolerance = 10.0 * f64::EPSILON * column_sums * m.max(n) as f64;
    let limit = 5 * n + 30;
    let mut iterations = 0;
    loop {
        // The most promising inactive coefficient.
        let mut best: Option<(usize, f64)> = None;
        for (j, (&w, &is_passive)) in gradient.iter().zip(passive.iter()).enumerate() {
            if !is_passive && w > tolerance && best.is_none_or(|(_, top)| w > top) {
                best = Some((j, w));
            }
        }
        let Some((entering, _)) = best else { break };
        passive[entering] = true;
        let mut first = true;
        loop {
            if iterations == limit {
                return Err(SolveError::NotConverged { iterations, residual: norm_inf(gradient) });
            }
            iterations += 1;
            indices.clear();
            indices.extend(passive.iter().enumerate().filter(|&(_, &p)| p).map(|(j, _)| j));
            solve_passive(gram, atb, indices, sub, cholesky, trial)?;
            if indices.iter().all(|&j| trial[j] > 0.0) {
                x.copy_from_slice(trial);
                break;
            }
            if first && trial[entering] <= 0.0 {
                // The entering gradient was positive by rounding only: no descent is possible, the
                // current x already satisfies the optimality conditions to working precision.
                passive[entering] = false;
                return finish(a, b, x, residual, iterations);
            }
            first = false;
            // Step from x towards the trial point until the first passive coefficient reaches zero.
            let mut alpha = 1.0_f64;
            for &j in indices.iter() {
                if trial[j] <= 0.0 {
                    alpha = alpha.min(x[j] / (x[j] - trial[j]));
                }
            }
            for &j in indices.iter() {
                x[j] += alpha * (trial[j] - x[j]);
                if x[j] <= tolerance {
                    x[j] = 0.0;
                    passive[j] = false;
                }
            }
        }
        // w = Aᵀb − AᵀA·x
        for (j, w) in gradient.iter_mut().enumerate() {
            *w = atb[j] - dot(gram.row(j), x);
        }
    }
    finish(a, b, x, residual, iterations)
}

/// The report of a finished NNLS solve.
fn finish(
    a: &DenseMatrix,
    b: &[f64],
    x: &[f64],
    residual: &mut Vec<f64>,
    iterations: usize,
) -> Result<NnlsReport, SolveError> {
    residual.resize(b.len(), 0.0);
    a.mul_vec_into(x, residual)?;
    for (r, bi) in residual.iter_mut().zip(b) {
        *r -= bi;
    }
    Ok(NnlsReport { iterations, residual_norm: norm(residual), active: x.iter().filter(|&&v| v > 0.0).count() })
}

/// Solves `min ‖A·x − b‖₂` subject to `x ≥ 0`; see [`nnls_into`].
pub fn nnls(a: &DenseMatrix, b: &[f64]) -> Result<Vec<f64>, SolveError> {
    let mut x = vec![0.0; a.cols()];
    nnls_into(a, b, &mut x, &mut NnlsWorkspace::default())?;
    Ok(x)
}

/// A non-linear least-squares problem `min ½·‖r(p)‖₂²` for [`levenberg_marquardt`].
pub trait LeastSquaresProblem {
    /// The number of residuals `m`.
    fn residual_count(&self) -> usize;
    /// The number of parameters `n`.
    fn parameter_count(&self) -> usize;
    /// Writes the `m` residuals at `params` into `out`. Non-finite residuals make the solver reject
    /// the point (a way to signal leaving the problem's domain); an error stops the solve.
    fn residuals(&self, params: &[f64], out: &mut [f64]) -> Result<(), SolveError>;
    /// Writes the `m × n` Jacobian `∂rᵢ/∂pⱼ` at `params` into `out`, which arrives with that shape and
    /// zero-filled. Should not allocate.
    fn jacobian(&self, params: &[f64], out: &mut DenseMatrix) -> Result<(), SolveError>;
}

/// Stopping rules of [`levenberg_marquardt`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LmSettings {
    /// The iteration limit (accepted and rejected steps both count).
    pub max_iterations: usize,
    /// Converged when the gradient `‖Jᵀ·r‖_∞` is at most this.
    pub gradient_tolerance: f64,
    /// Converged when a step is at most `step_tolerance · (‖p‖₂ + step_tolerance)` long.
    pub step_tolerance: f64,
    /// The initial damping relative to the scaled diagonal of `JᵀJ` (`10⁻³` is the usual choice;
    /// larger starts more like gradient descent).
    pub initial_damping: f64,
}

impl Default for LmSettings {
    /// 200 iterations, gradient `1e-12`, step `1e-12`, damping `1e-3`.
    fn default() -> Self {
        Self { max_iterations: 200, gradient_tolerance: 1e-12, step_tolerance: 1e-12, initial_damping: 1e-3 }
    }
}

/// Why [`levenberg_marquardt`] stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LmStop {
    /// The gradient fell below its tolerance (a stationary point).
    Gradient,
    /// The step fell below its tolerance (no further progress possible at working precision).
    Step,
}

/// The outcome of a converged [`levenberg_marquardt`] solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LmReport {
    /// The iterations used.
    pub iterations: usize,
    /// `½‖r(p)‖₂²` at the solution.
    pub cost: f64,
    /// `‖Jᵀ·r‖_∞` at the solution.
    pub gradient_norm: f64,
    /// The stopping rule that fired.
    pub stop: LmStop,
}

/// Reusable storage of [`levenberg_marquardt`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LmWorkspace {
    residual: Vec<f64>,
    trial_residual: Vec<f64>,
    trial: Vec<f64>,
    jacobian: DenseMatrix,
    normal: DenseMatrix,
    damped: DenseMatrix,
    gradient: Vec<f64>,
    step: Vec<f64>,
    scaling: Vec<f64>,
    cholesky: Cholesky,
}

/// Minimises `½‖r(p)‖₂²` by Levenberg-Marquardt, starting from and updating `params`.
///
/// Each iteration solves `(JᵀJ + μ·D)·h = −Jᵀr` with `D` the running maximum of `diag(JᵀJ)`
/// (Marquardt's scaling, which makes the method invariant to parameter units), accepts the step when it
/// lowers the cost, and adapts `μ` by Nielsen's rule from the ratio of actual to predicted decrease.
///
/// On success `params` holds the solution. On [`SolveError::NotConverged`] it holds the best point
/// found. Other errors: dimension mismatches, non-finite start or Jacobian, invalid settings, and any
/// error the problem returns.
pub fn levenberg_marquardt<P: LeastSquaresProblem>(
    problem: &P,
    params: &mut [f64],
    settings: &LmSettings,
    workspace: &mut LmWorkspace,
) -> Result<LmReport, SolveError> {
    let (m, n) = (problem.residual_count(), problem.parameter_count());
    check_dim("levenberg-marquardt: parameter count", n, params.len())?;
    check_finite(params)?;
    let valid = |v: f64| v.is_finite() && v >= 0.0;
    if !(valid(settings.gradient_tolerance) && valid(settings.step_tolerance) && valid(settings.initial_damping))
        || settings.initial_damping == 0.0
    {
        return Err(SolveError::InvalidParameter { what: "levenberg-marquardt: tolerances must be >= 0, damping > 0" });
    }
    let LmWorkspace { residual, trial_residual, trial, jacobian, normal, damped, gradient, step, scaling, cholesky } =
        workspace;
    residual.resize(m, 0.0);
    trial_residual.resize(m, 0.0);
    trial.resize(n, 0.0);
    gradient.resize(n, 0.0);
    step.resize(n, 0.0);
    scaling.clear();
    scaling.resize(n, 0.0);
    problem.residuals(params, residual)?;
    check_finite(residual)?;
    let mut cost = 0.5 * norm_sq(residual);
    let mut damping = settings.initial_damping;
    let mut growth = 2.0;
    let mut fresh = true;
    let mut gradient_norm = f64::INFINITY;
    for iteration in 0..settings.max_iterations {
        if fresh {
            jacobian.resize(m, n);
            problem.jacobian(params, jacobian)?;
            check_finite(jacobian.as_slice())?;
            jacobian.view().gram_into(normal);
            jacobian.tr_mul_vec_into(residual, gradient)?;
            for (j, s) in scaling.iter_mut().enumerate() {
                *s = s.max(normal[(j, j)]);
            }
            gradient_norm = norm_inf(gradient);
            fresh = false;
        }
        if gradient_norm <= settings.gradient_tolerance {
            return Ok(LmReport { iterations: iteration, cost, gradient_norm, stop: LmStop::Gradient });
        }
        damped.copy_from(normal.view());
        for (j, &s) in scaling.iter().enumerate() {
            // A parameter the residuals have not yet depended on is damped in absolute units.
            damped[(j, j)] += damping * if s > 0.0 { s } else { 1.0 };
        }
        if cholesky.factor(damped.view()).is_err() {
            damping *= growth;
            growth *= 2.0;
            continue;
        }
        for (h, g) in step.iter_mut().zip(gradient.iter()) {
            *h = -g;
        }
        cholesky.solve_in_place(step)?;
        let step_length = norm(step);
        if step_length <= settings.step_tolerance * (norm(params) + settings.step_tolerance) {
            return Ok(LmReport { iterations: iteration, cost, gradient_norm, stop: LmStop::Step });
        }
        for ((t, p), h) in trial.iter_mut().zip(params.iter()).zip(step.iter()) {
            *t = p + h;
        }
        problem.residuals(trial, trial_residual)?;
        let trial_cost = 0.5 * norm_sq(trial_residual);
        // The decrease the linear model predicts: ½·hᵀ·(μ·D·h − g), positive for a descent step.
        let mut predicted = 0.0;
        for (j, (&h, &g)) in step.iter().zip(gradient.iter()).enumerate() {
            let s = if scaling[j] > 0.0 { scaling[j] } else { 1.0 };
            predicted += 0.5 * h * (damping * s * h - g);
        }
        let ratio = (cost - trial_cost) / predicted;
        if trial_cost.is_finite() && ratio > 0.0 {
            params.copy_from_slice(trial);
            std::mem::swap(residual, trial_residual);
            cost = trial_cost;
            let t = 2.0 * ratio - 1.0;
            damping *= (1.0 - t * t * t).max(1.0 / 3.0);
            growth = 2.0;
            fresh = true;
        } else {
            damping *= growth;
            growth *= 2.0;
            if !damping.is_finite() {
                return Ok(LmReport { iterations: iteration + 1, cost, gradient_norm, stop: LmStop::Step });
            }
        }
    }
    Err(SolveError::NotConverged { iterations: settings.max_iterations, residual: gradient_norm })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solve::qr::Qr;
    use crate::solve::testutil::{random_matrix, random_vec};
    use humanart_core::rng::Rng;

    fn close(x: &[f64], y: &[f64], tolerance: f64) -> bool {
        x.len() == y.len() && x.iter().zip(y).all(|(p, q)| (p - q).abs() <= tolerance)
    }

    /// `‖Aᵀ(A·x − b) + λ(x − x₀)‖_∞`: the gradient of the Tikhonov objective (halved).
    fn tikhonov_gradient(a: &DenseMatrix, b: &[f64], lambda: f64, prior: &[f64], x: &[f64]) -> f64 {
        let r: Vec<f64> = a.mul_vec(x).unwrap().iter().zip(b).map(|(p, q)| p - q).collect();
        let g = a.tr_mul_vec(&r).unwrap();
        g.iter().zip(x).zip(prior).fold(0.0, |acc, ((g, x), p)| acc.max((g + lambda * (x - p)).abs()))
    }

    #[test]
    fn ac_wp02_1_tikhonov_closed_forms_and_optimality() {
        // A = I: x = (b + λ·x₀) / (1 + λ).
        let a = DenseMatrix::identity(3);
        let x = tikhonov(&a, &[2.0, -4.0, 6.0], 1.0, Some(&[0.0, 2.0, 0.0])).unwrap();
        assert!(close(&x, &[1.0, -1.0, 3.0], 1e-15), "{x:?}");
        let mut rng = Rng::new(0x90);
        for (m, n) in [(12, 5), (5, 12), (7, 7), (1, 30)] {
            let a = random_matrix(&mut rng, m, n);
            let b = random_vec(&mut rng, m);
            let prior = random_vec(&mut rng, n);
            for lambda in [1e-6, 0.1, 10.0] {
                let x = tikhonov(&a, &b, lambda, Some(&prior)).unwrap();
                assert!(tikhonov_gradient(&a, &b, lambda, &prior, &x) < 1e-11, "{m}x{n} λ={lambda}");
            }
            // Huge λ pins x to the prior.
            let x = tikhonov(&a, &b, 1e12, Some(&prior)).unwrap();
            assert!(x.iter().zip(&prior).all(|(p, q)| (p - q).abs() < 1e-10));
        }
        // λ = 0 on a tall full-rank matrix is ordinary least squares.
        let a = random_matrix(&mut rng, 15, 4);
        let b = random_vec(&mut rng, 15);
        let x = tikhonov(&a, &b, 0.0, None).unwrap();
        let expect = Qr::new(&a).unwrap().solve_least_squares(&b).unwrap();
        assert!(x.iter().zip(&expect).all(|(p, q)| (p - q).abs() < 1e-12));
    }

    #[test]
    fn tikhonov_errors_and_workspace_reuse() {
        let a = DenseMatrix::from_rows(&[[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]]).unwrap();
        assert_eq!(tikhonov(&a, &[1.0; 3], 0.0, None), Err(SolveError::Singular));
        assert!(tikhonov(&a, &[1.0; 3], 1e-3, None).is_ok());
        assert!(matches!(tikhonov(&a, &[1.0; 3], -1.0, None), Err(SolveError::InvalidParameter { .. })));
        assert!(matches!(tikhonov(&a, &[1.0; 2], 1.0, None), Err(SolveError::DimensionMismatch { .. })));
        assert!(matches!(tikhonov(&a, &[1.0; 3], 1.0, Some(&[0.0])), Err(SolveError::DimensionMismatch { .. })));
        assert_eq!(tikhonov(&a, &[1.0, f64::NAN, 1.0], 1.0, None), Err(SolveError::NonFinite));
        // No observations: the prior is the answer.
        assert_eq!(tikhonov(&DenseMatrix::zeros(0, 2), &[], 2.0, Some(&[3.0, 4.0])).unwrap(), vec![3.0, 4.0]);
        let mut workspace = TikhonovWorkspace::default();
        let mut x = [0.0; 2];
        tikhonov_into(&a, &[1.0, 2.0, 3.0], 0.5, None, &mut x, &mut workspace).unwrap();
        let pointer = workspace.normal.as_slice().as_ptr();
        tikhonov_into(&a, &[3.0, 2.0, 1.0], 0.7, None, &mut x, &mut workspace).unwrap();
        assert_eq!(workspace.normal.as_slice().as_ptr(), pointer);
    }

    /// The NNLS solution by exhaustive search over all active sets (small n only).
    fn brute_force_nnls(a: &DenseMatrix, b: &[f64]) -> (Vec<f64>, f64) {
        let n = a.cols();
        let mut best = (vec![0.0; n], b.iter().map(|v| v * v).sum::<f64>());
        for mask in 1_usize..(1 << n) {
            let columns: Vec<usize> = (0..n).filter(|j| mask & (1 << j) != 0).collect();
            let sub = DenseMatrix::from_fn(a.rows(), columns.len(), |i, k| a[(i, columns[k])]);
            let Ok(s) = Qr::new(&sub).and_then(|qr| qr.solve_least_squares(b)) else { continue };
            if s.iter().any(|&v| v < 0.0) {
                continue;
            }
            let mut x = vec![0.0; n];
            for (k, &j) in columns.iter().enumerate() {
                x[j] = s[k];
            }
            let r: f64 = a.mul_vec(&x).unwrap().iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum();
            if r < best.1 {
                best = (x, r);
            }
        }
        best
    }

    #[test]
    fn ac_wp02_1_nnls_known_answers_and_exhaustive_search() {
        let x = nnls(&DenseMatrix::identity(3), &[1.0, -2.0, 3.0]).unwrap();
        assert!(close(&x, &[1.0, 0.0, 3.0], 1e-15) && x[1] == 0.0, "{x:?}");
        // Lawson-Hanson's classic: the unconstrained solution has a negative entry.
        let a = DenseMatrix::from_rows(&[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]).unwrap();
        let x = nnls(&a, &[2.0, 1.0, -1.0]).unwrap();
        assert!(close(&x, &[1.5, 0.0], 1e-15) && x[1] == 0.0, "{x:?}");
        let mut rng = Rng::new(0x91);
        let mut workspace = NnlsWorkspace::default();
        for (m, n) in [(10, 6), (6, 6), (4, 7), (20, 3)] {
            for _ in 0..20 {
                let a = random_matrix(&mut rng, m, n);
                let b = random_vec(&mut rng, m);
                let mut x = vec![0.0; n];
                let report = nnls_into(&a, &b, &mut x, &mut workspace).unwrap();
                assert!(x.iter().all(|&v| v >= 0.0));
                let (expect, best) = brute_force_nnls(&a, &b);
                assert!((report.residual_norm * report.residual_norm - best).abs() < 1e-11, "{m}x{n}");
                if m >= n {
                    // Full column rank: the minimiser is unique.
                    assert!(x.iter().zip(&expect).all(|(p, q)| (p - q).abs() < 1e-10), "{x:?} {expect:?}");
                }
                // Karush-Kuhn-Tucker conditions.
                let r: Vec<f64> = b.iter().zip(a.mul_vec(&x).unwrap()).map(|(p, q)| p - q).collect();
                let w = a.tr_mul_vec(&r).unwrap();
                for (&xj, &wj) in x.iter().zip(&w) {
                    assert!(if xj > 0.0 { wj.abs() < 1e-11 } else { wj < 1e-11 }, "x {xj} w {wj}");
                }
                assert_eq!(report.active, x.iter().filter(|&&v| v > 0.0).count());
            }
        }
    }

    #[test]
    fn nnls_edge_cases() {
        // All-negative correlation: x = 0.
        let a = DenseMatrix::from_rows(&[[1.0, 2.0], [3.0, 4.0]]).unwrap();
        let x = nnls(&a, &[-1.0, -1.0]).unwrap();
        assert_eq!(x, vec![0.0, 0.0]);
        // A zero column never becomes active; duplicate columns share the work.
        let a = DenseMatrix::from_rows(&[[1.0, 0.0, 1.0], [2.0, 0.0, 2.0], [0.5, 0.0, 0.5]]).unwrap();
        let b = [2.0, 4.0, 1.0];
        let mut x = vec![0.0; 3];
        let report = nnls_into(&a, &b, &mut x, &mut NnlsWorkspace::default()).unwrap();
        assert!(report.residual_norm < 1e-12 && x[1] == 0.0 && (x[0] + x[2] - 2.0).abs() < 1e-12);
        assert!(nnls(&DenseMatrix::zeros(3, 0), &[1.0; 3]).unwrap().is_empty());
        assert!(matches!(nnls(&a, &[1.0; 2]), Err(SolveError::DimensionMismatch { .. })));
        assert_eq!(nnls(&a, &[1.0, f64::NAN, 1.0]), Err(SolveError::NonFinite));
    }

    /// Rosenbrock's function as a least-squares problem: r = (10(p₁ − p₀²), 1 − p₀).
    struct Rosenbrock;

    impl LeastSquaresProblem for Rosenbrock {
        fn residual_count(&self) -> usize {
            2
        }
        fn parameter_count(&self) -> usize {
            2
        }
        fn residuals(&self, p: &[f64], out: &mut [f64]) -> Result<(), SolveError> {
            out[0] = 10.0 * (p[1] - p[0] * p[0]);
            out[1] = 1.0 - p[0];
            Ok(())
        }
        fn jacobian(&self, p: &[f64], out: &mut DenseMatrix) -> Result<(), SolveError> {
            out[(0, 0)] = -20.0 * p[0];
            out[(0, 1)] = 10.0;
            out[(1, 0)] = -1.0;
            Ok(())
        }
    }

    /// Fitting `y = a·exp(b·t)` to samples.
    struct ExponentialFit {
        t: Vec<f64>,
        y: Vec<f64>,
    }

    impl LeastSquaresProblem for ExponentialFit {
        fn residual_count(&self) -> usize {
            self.t.len()
        }
        fn parameter_count(&self) -> usize {
            2
        }
        fn residuals(&self, p: &[f64], out: &mut [f64]) -> Result<(), SolveError> {
            for ((r, t), y) in out.iter_mut().zip(&self.t).zip(&self.y) {
                *r = p[0] * (p[1] * t).exp() - y;
            }
            Ok(())
        }
        fn jacobian(&self, p: &[f64], out: &mut DenseMatrix) -> Result<(), SolveError> {
            for (i, t) in self.t.iter().enumerate() {
                let e = (p[1] * t).exp();
                out[(i, 0)] = e;
                out[(i, 1)] = p[0] * t * e;
            }
            Ok(())
        }
    }

    /// A linear problem r = A·p − b.
    struct Linear {
        a: DenseMatrix,
        b: Vec<f64>,
    }

    impl LeastSquaresProblem for Linear {
        fn residual_count(&self) -> usize {
            self.a.rows()
        }
        fn parameter_count(&self) -> usize {
            self.a.cols()
        }
        fn residuals(&self, p: &[f64], out: &mut [f64]) -> Result<(), SolveError> {
            self.a.mul_vec_into(p, out)?;
            out.iter_mut().zip(&self.b).for_each(|(r, b)| *r -= b);
            Ok(())
        }
        fn jacobian(&self, _: &[f64], out: &mut DenseMatrix) -> Result<(), SolveError> {
            out.copy_from(self.a.view());
            Ok(())
        }
    }

    #[test]
    fn ac_wp02_1_levenberg_marquardt_minimises_rosenbrock() {
        let mut p = [-1.2, 1.0];
        let mut workspace = LmWorkspace::default();
        let report = levenberg_marquardt(&Rosenbrock, &mut p, &LmSettings::default(), &mut workspace).unwrap();
        assert!((p[0] - 1.0).abs() < 1e-10 && (p[1] - 1.0).abs() < 1e-10, "{p:?} {report:?}");
        assert!(report.cost < 1e-20 && report.iterations < 100, "{report:?}");
        // Too few iterations: NotConverged, the parameters hold the best point so far.
        let mut p = [-1.2, 1.0];
        let limited = LmSettings { max_iterations: 2, ..LmSettings::default() };
        assert!(matches!(
            levenberg_marquardt(&Rosenbrock, &mut p, &limited, &mut workspace),
            Err(SolveError::NotConverged { iterations: 2, .. })
        ));
        let mut r = [0.0; 2];
        Rosenbrock.residuals(&p, &mut r).unwrap();
        // The starting cost is ½(4.4² + 2.2²) = 12.1; accepted steps only lower it.
        assert!(0.5 * norm_sq(&r) < 12.1);
    }

    #[test]
    fn ac_wp02_1_levenberg_marquardt_fits_exponentials() {
        let t: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();
        let exact = ExponentialFit { y: t.iter().map(|t| 2.5 * (-1.3 * t).exp()).collect(), t: t.clone() };
        let mut p = [1.0, 0.0];
        let report = levenberg_marquardt(&exact, &mut p, &LmSettings::default(), &mut LmWorkspace::default()).unwrap();
        assert!((p[0] - 2.5).abs() < 1e-10 && (p[1] + 1.3).abs() < 1e-10, "{p:?} {report:?}");
        // Noisy data: a non-zero-residual minimum where the gradient vanishes.
        let mut rng = Rng::new(0x92);
        let noisy = ExponentialFit { y: exact.y.iter().map(|y| y + 0.01 * rng.normal()).collect(), t };
        let mut p = [1.0, 0.0];
        let report = levenberg_marquardt(&noisy, &mut p, &LmSettings::default(), &mut LmWorkspace::default()).unwrap();
        assert!(report.cost > 1e-5 && (p[0] - 2.5).abs() < 0.05 && (p[1] + 1.3).abs() < 0.05, "{p:?} {report:?}");
        let mut j = DenseMatrix::zeros(30, 2);
        let mut r = vec![0.0; 30];
        noisy.jacobian(&p, &mut j).unwrap();
        noisy.residuals(&p, &mut r).unwrap();
        assert!(norm_inf(&j.tr_mul_vec(&r).unwrap()) < 1e-9, "{report:?}");
    }

    #[test]
    fn levenberg_marquardt_on_a_linear_problem_is_least_squares() {
        let mut rng = Rng::new(0x93);
        let problem = Linear { a: random_matrix(&mut rng, 12, 4), b: random_vec(&mut rng, 12) };
        let expect = Qr::new(&problem.a).unwrap().solve_least_squares(&problem.b).unwrap();
        let mut p = [0.0; 4];
        let mut workspace = LmWorkspace::default();
        levenberg_marquardt(&problem, &mut p, &LmSettings::default(), &mut workspace).unwrap();
        assert!(p.iter().zip(&expect).all(|(x, y)| (x - y).abs() < 1e-10), "{p:?} {expect:?}");
        // A second solve of the same size reuses the buffers.
        let pointer = workspace.jacobian.as_slice().as_ptr();
        let mut p = [1.0; 4];
        levenberg_marquardt(&problem, &mut p, &LmSettings::default(), &mut workspace).unwrap();
        assert_eq!(workspace.jacobian.as_slice().as_ptr(), pointer);
        // Errors.
        let mut w = LmWorkspace::default();
        assert!(matches!(
            levenberg_marquardt(&problem, &mut [0.0; 3], &LmSettings::default(), &mut w),
            Err(SolveError::DimensionMismatch { .. })
        ));
        assert_eq!(
            levenberg_marquardt(&problem, &mut [f64::NAN; 4], &LmSettings::default(), &mut w),
            Err(SolveError::NonFinite)
        );
        let bad = LmSettings { initial_damping: 0.0, ..LmSettings::default() };
        assert!(matches!(
            levenberg_marquardt(&problem, &mut [0.0; 4], &bad, &mut w),
            Err(SolveError::InvalidParameter { .. })
        ));
    }
}
