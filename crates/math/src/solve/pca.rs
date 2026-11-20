//! Principal component analysis for the statistical shape and appearance models `[STAT-02]`.
//!
//! The data are `N` samples of dimension `D` (rows of a [`DenseMatrix`]; for a shape model a row is
//! one registered mesh, its `3V` vertex coordinates flattened). With `X` the centred data:
//!
//! - when `N ≤ D` (few subjects, many vertices: the usual case) the `N × N` Gram matrix `X·Xᵀ` is
//!   eigen-decomposed by the in-house Jacobi solver and each component is recovered as
//!   `vⱼ = Xᵀ·uⱼ / √λⱼ` (the "snapshot" method), so the cost is `O(N²·D + N³)` and nothing of size
//!   `D × D` is formed;
//! - otherwise the `D × D` covariance `XᵀX` is decomposed directly.
//!
//! Both give the same components (up to rounding; [`PcaMethod`] forces one for comparison). The
//! variance of a component is its eigenvalue over `N − 1` (the unbiased sample variance). Components
//! whose variance is at the rounding level of the largest are not kept, so a model of `N` samples has
//! at most `N − 1` components: centring removes one degree of freedom.
//!
//! The component count is chosen by explained variance ([`Pca::components_for_variance`]) or by
//! held-out reconstruction error ([`leave_one_out`]).
//!
//! For large `N` (and large `D`), when only the leading `k` components are wanted,
//! [`Pca::fit_randomized`] is the randomised range finder of Halko, Martinsson and Tropp (2011,
//! "Finding structure with randomness", algorithms 4.4 and 5.1): `l = k + p` Gaussian test vectors
//! (`p` the oversampling, seeded [`humanart_core::rng::Rng`], so the result is deterministic) span
//! `Y = X·Ω`; `q` power iterations `Y ← X·(Xᵀ·Y)`, each followed by re-orthonormalisation, sharpen
//! the spectrum; with `Q` an orthonormal basis of `Y`, the `l × D` matrix `B = Qᵀ·X` is small, and
//! the eigen-decomposition of `B·Bᵀ` (`l × l`, Jacobi) gives the components as with the snapshot
//! method. The cost is `O((2q + 2)·N·D·l + l³)`: no `N × N` or `D × D` matrix is formed. On data
//! whose spectrum decays it matches the exact components to rounding (tested against the exact PCA
//! on random low-rank data). The variances are those of the kept components; the total variance is
//! exact (the Frobenius norm of the centred data), so explained-variance ratios stay meaningful.
//!
//! [`Pca::project_into`] and [`Pca::reconstruct_into`] write into caller storage and do not allocate,
//! so evaluating a model per frame is allocation-free `[PERF-R2]`.

use super::dense::DenseMatrix;
use super::eigen::SymmetricEigen;
use super::error::{SolveError, check_dim, check_finite};
use super::vector::{axpy, dot, norm, scale};

/// Which eigen-problem [`Pca::fit_with`] solves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcaMethod {
    /// The Gram matrix when `N ≤ D`, the covariance otherwise.
    Auto,
    /// The `N × N` Gram matrix `X·Xᵀ` (snapshot method).
    Gram,
    /// The `D × D` covariance `XᵀX`.
    Covariance,
}

/// The settings of [`Pca::fit_randomized`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RandomizedPca {
    /// The number `k` of leading components wanted (at least 1, at most `min(N, D)`).
    pub components: usize,
    /// Extra test vectors `p` beyond `k` (Halko et al. recommend 5 to 10).
    pub oversampling: usize,
    /// Power iterations `q`: more for slowly decaying spectra (each costs two passes over the data).
    pub power_iterations: usize,
    /// Seed of the Gaussian test vectors.
    pub seed: u64,
}

impl RandomizedPca {
    /// `components` leading components with oversampling 10, 4 power iterations and a fixed seed.
    pub const fn new(components: usize) -> Self {
        Self { components, oversampling: 10, power_iterations: 4, seed: 0x5EED_0000_0000_FCA0 }
    }
}

/// A fitted principal component model: mean, orthonormal components in order of decreasing variance,
/// and their variances.
#[derive(Clone, Debug, PartialEq)]
pub struct Pca {
    mean: Vec<f64>,
    /// `k × D`: row `j` is the unit component `vⱼ`.
    components: DenseMatrix,
    /// The variance along each component, descending.
    variances: Vec<f64>,
    /// `vⱼ · mean` per component, so that projection needs no scratch storage.
    mean_projection: Vec<f64>,
    /// The total variance of the data (the trace of the sample covariance), kept components or not.
    total_variance: f64,
    samples: usize,
}

impl Pca {
    /// Fits a model to the rows of `data` (`N` samples × `D` values) with [`PcaMethod::Auto`].
    ///
    /// Errors: [`SolveError::InvalidParameter`] for no samples, [`SolveError::NonFinite`], and
    /// [`SolveError::NotConverged`] from the eigen-solver (not observed for finite input).
    pub fn fit(data: &DenseMatrix) -> Result<Self, SolveError> {
        Self::fit_with(data, PcaMethod::Auto)
    }

    /// Fits a model with the given eigen-problem; see [`Pca::fit`].
    pub fn fit_with(data: &DenseMatrix, method: PcaMethod) -> Result<Self, SolveError> {
        let (n, d) = (data.rows(), data.cols());
        let (mean, centred) = centre(data)?;
        let degrees = (n - 1).max(1) as f64;
        let total_variance = centred.frobenius_norm().powi(2) / degrees;
        let use_gram = match method {
            PcaMethod::Auto => n <= d,
            PcaMethod::Gram => true,
            PcaMethod::Covariance => false,
        };
        let eigen = SymmetricEigen::new(&if use_gram { centred.outer_gram() } else { centred.gram() })?;
        let largest = eigen.values().first().copied().unwrap_or(0.0);
        // Eigenvalues within rounding of the largest carry no direction.
        let floor = (n.max(d) as f64) * f64::EPSILON * largest;
        let kept = eigen.values().iter().take_while(|&&lambda| lambda > floor && lambda > 0.0).count();
        let mut components = DenseMatrix::zeros(kept, d);
        let mut variances = Vec::with_capacity(kept);
        for j in 0..kept {
            let lambda = eigen.values()[j];
            let row = components.row_mut(j);
            if use_gram {
                centred.tr_mul_vec_into(eigen.vector(j), row)?;
                scale(1.0 / lambda.sqrt(), row);
            } else {
                row.copy_from_slice(eigen.vector(j));
            }
            variances.push(lambda / degrees);
        }
        let kept = orthonormalize(&mut components, kept);
        components.truncate_rows(kept);
        variances.truncate(kept);
        for j in 0..kept {
            fix_sign(components.row_mut(j));
        }
        let mean_projection = (0..kept).map(|j| dot(components.row(j), &mean)).collect();
        Ok(Self { mean, components, variances, mean_projection, total_variance, samples: n })
    }

    /// Fits the leading `settings.components` components by randomised PCA (see the module
    /// documentation), for data too large for the exact Gram or covariance eigen-problem
    /// `[STAT-02]`. Fewer components are kept when the data have lower rank.
    ///
    /// Errors: [`SolveError::InvalidParameter`] for no samples or a component count of 0 or above
    /// `min(N, D)`, [`SolveError::NonFinite`], and [`SolveError::NotConverged`] from the eigen-solver.
    pub fn fit_randomized(data: &DenseMatrix, settings: &RandomizedPca) -> Result<Self, SolveError> {
        let (n, d) = (data.rows(), data.cols());
        let k = settings.components;
        if k == 0 || k > n.min(d) {
            return Err(SolveError::InvalidParameter { what: "pca: randomized component count" });
        }
        let (mean, centred) = centre(data)?;
        let degrees = (n - 1).max(1) as f64;
        let total_variance = centred.frobenius_norm().powi(2) / degrees;
        let l = k.saturating_add(settings.oversampling).min(n.min(d));

        // Ωᵀ: l Gaussian test vectors of dimension D (rows).
        let mut rng = humanart_core::rng::Rng::new(settings.seed);
        let omega = DenseMatrix::from_fn(l, d, |_, _| rng.normal());
        // Qᵀ (l × N): an orthonormal basis of the range of X·Ω, as rows.
        let mut basis = DenseMatrix::zeros(l, n);
        for j in 0..l {
            centred.mul_vec_into(omega.row(j), basis.row_mut(j))?;
        }
        let mut rank = orthonormal_range(&mut basis, l);
        let mut wide = DenseMatrix::zeros(l, d);
        for _ in 0..settings.power_iterations {
            for j in 0..rank {
                centred.tr_mul_vec_into(basis.row(j), wide.row_mut(j))?;
            }
            rank = orthonormal_range(&mut wide, rank);
            for j in 0..rank {
                centred.mul_vec_into(wide.row(j), basis.row_mut(j))?;
            }
            rank = orthonormal_range(&mut basis, rank);
        }
        // B = Qᵀ·X (rank × D), and the eigen-problem of B·Bᵀ.
        let mut small = DenseMatrix::zeros(rank, d);
        for j in 0..rank {
            centred.tr_mul_vec_into(basis.row(j), small.row_mut(j))?;
        }
        let eigen = SymmetricEigen::new(&small.outer_gram())?;
        let largest = eigen.values().first().copied().unwrap_or(0.0);
        let floor = (n.max(d) as f64) * f64::EPSILON * largest;
        let kept = eigen.values().iter().take(k).take_while(|&&lambda| lambda > floor && lambda > 0.0).count();
        let mut components = DenseMatrix::zeros(kept, d);
        let mut variances = Vec::with_capacity(kept);
        for j in 0..kept {
            let lambda = eigen.values()[j];
            let row = components.row_mut(j);
            small.tr_mul_vec_into(eigen.vector(j), row)?;
            scale(1.0 / lambda.sqrt(), row);
            variances.push(lambda / degrees);
        }
        let kept = orthonormalize(&mut components, kept);
        components.truncate_rows(kept);
        variances.truncate(kept);
        for j in 0..kept {
            fix_sign(components.row_mut(j));
        }
        let mean_projection = (0..kept).map(|j| dot(components.row(j), &mean)).collect();
        Ok(Self { mean, components, variances, mean_projection, total_variance, samples: n })
    }

    /// The dimension `D` of a sample.
    pub fn dimension(&self) -> usize {
        self.mean.len()
    }

    /// The number of samples the model was fitted to.
    pub fn sample_count(&self) -> usize {
        self.samples
    }

    /// The mean sample.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// The number of components `k` kept.
    pub fn component_count(&self) -> usize {
        self.variances.len()
    }

    /// Component `j` (unit length; its largest-magnitude element is positive), or `None` past the end.
    pub fn component(&self, j: usize) -> Option<&[f64]> {
        (j < self.component_count()).then(|| self.components.row(j))
    }

    /// All components as the rows of a `k × D` matrix.
    pub fn components(&self) -> &DenseMatrix {
        &self.components
    }

    /// The variance along each component, descending.
    pub fn variances(&self) -> &[f64] {
        &self.variances
    }

    /// The total variance of the data (the trace of the sample covariance), including the part no kept
    /// component explains.
    pub fn total_variance(&self) -> f64 {
        self.total_variance
    }

    /// The fraction of the total variance each component explains.
    pub fn explained_variance_ratio(&self, j: usize) -> f64 {
        match self.variances.get(j) {
            Some(&v) if self.total_variance > 0.0 => v / self.total_variance,
            _ => 0.0,
        }
    }

    /// The fraction of the total variance the first `count` components explain together (1 for data
    /// without variance).
    pub fn cumulative_explained_variance(&self, count: usize) -> f64 {
        if self.total_variance > 0.0 {
            (self.variances.iter().take(count).sum::<f64>() / self.total_variance).min(1.0)
        } else {
            1.0
        }
    }

    /// The smallest number of components that explains at least `fraction` (in `[0, 1]`) of the total
    /// variance, capped at the number kept.
    pub fn components_for_variance(&self, fraction: f64) -> usize {
        let target = fraction.clamp(0.0, 1.0);
        (0..=self.component_count())
            .find(|&k| self.cumulative_explained_variance(k) >= target)
            .unwrap_or(self.component_count())
    }

    /// Keeps only the first `count` components (no effect when `count` is not smaller).
    pub fn truncate(&mut self, count: usize) {
        let count = count.min(self.component_count());
        self.components.truncate_rows(count);
        self.variances.truncate(count);
        self.mean_projection.truncate(count);
    }

    /// The coefficients of `sample`: `cⱼ = vⱼ·(x − mean)` for the first `coefficients.len()`
    /// components. No allocation.
    pub fn project_into(&self, sample: &[f64], coefficients: &mut [f64]) -> Result<(), SolveError> {
        check_dim("pca project: sample length", self.dimension(), sample.len())?;
        if coefficients.len() > self.component_count() {
            return Err(SolveError::DimensionMismatch {
                what: "pca project: at most one coefficient per component",
                expected: self.component_count(),
                found: coefficients.len(),
            });
        }
        check_finite(sample)?;
        for (j, c) in coefficients.iter_mut().enumerate() {
            *c = dot(self.components.row(j), sample) - self.mean_projection[j];
        }
        Ok(())
    }

    /// The coefficients of `sample` on every component.
    pub fn project(&self, sample: &[f64]) -> Result<Vec<f64>, SolveError> {
        let mut coefficients = vec![0.0; self.component_count()];
        self.project_into(sample, &mut coefficients)?;
        Ok(coefficients)
    }

    /// `mean + Σⱼ cⱼ·vⱼ` into `out`, over as many components as there are coefficients (fewer than
    /// `k` gives the truncated reconstruction). No allocation.
    pub fn reconstruct_into(&self, coefficients: &[f64], out: &mut [f64]) -> Result<(), SolveError> {
        check_dim("pca reconstruct: output length", self.dimension(), out.len())?;
        if coefficients.len() > self.component_count() {
            return Err(SolveError::DimensionMismatch {
                what: "pca reconstruct: at most one coefficient per component",
                expected: self.component_count(),
                found: coefficients.len(),
            });
        }
        check_finite(coefficients)?;
        out.copy_from_slice(&self.mean);
        for (j, &c) in coefficients.iter().enumerate() {
            axpy(c, self.components.row(j), out);
        }
        Ok(())
    }

    /// `mean + Σⱼ cⱼ·vⱼ`; see [`Pca::reconstruct_into`].
    pub fn reconstruct(&self, coefficients: &[f64]) -> Result<Vec<f64>, SolveError> {
        let mut out = vec![0.0; self.dimension()];
        self.reconstruct_into(coefficients, &mut out)?;
        Ok(out)
    }

    /// `‖x − x̂‖₂`: the distance of `sample` from its reconstruction with the first `count` components
    /// (capped at `k`).
    pub fn reconstruction_error(&self, sample: &[f64], count: usize) -> Result<f64, SolveError> {
        let mut coefficients = vec![0.0; count.min(self.component_count())];
        self.project_into(sample, &mut coefficients)?;
        let mut reconstruction = self.reconstruct(&coefficients)?;
        for (r, x) in reconstruction.iter_mut().zip(sample) {
            *r -= x;
        }
        Ok(norm(&reconstruction))
    }

    /// The Mahalanobis length `√(Σⱼ cⱼ² / σⱼ²)` of coefficients (one per leading component): how
    /// many standard deviations from the mean a coefficient vector lies, the basis of the plausibility
    /// prior `[STAT-08]`.
    pub fn mahalanobis(&self, coefficients: &[f64]) -> Result<f64, SolveError> {
        if coefficients.len() > self.component_count() {
            return Err(SolveError::DimensionMismatch {
                what: "pca mahalanobis: at most one coefficient per component",
                expected: self.component_count(),
                found: coefficients.len(),
            });
        }
        check_finite(coefficients)?;
        Ok(coefficients.iter().zip(&self.variances).map(|(c, v)| c * c / v).sum::<f64>().sqrt())
    }
}

/// The mean of the rows of `data` and the centred copy; checks for samples and finite values.
fn centre(data: &DenseMatrix) -> Result<(Vec<f64>, DenseMatrix), SolveError> {
    let (n, d) = (data.rows(), data.cols());
    if n == 0 {
        return Err(SolveError::InvalidParameter { what: "pca: no samples" });
    }
    check_finite(data.as_slice())?;
    let mut mean = vec![0.0; d];
    for i in 0..n {
        axpy(1.0, data.row(i), &mut mean);
    }
    scale(1.0 / n as f64, &mut mean);
    let mut centred = data.clone();
    for i in 0..n {
        axpy(-1.0, &mean, centred.row_mut(i));
    }
    Ok((mean, centred))
}

/// Below this fraction of the longest row, a row of a range basis is rounding noise.
const RANGE_TOLERANCE: f64 = 1e-10;

/// Orthonormalises the first `count` rows of `m` for the randomised range finder: two passes of
/// modified Gram-Schmidt, a row whose remainder is below [`RANGE_TOLERANCE`] times the longest row
/// is dropped (the rows after it move up), and the number of rows kept is returned. Unlike
/// [`orthonormalize`], a row that loses most of its own length is kept: the rows of `X·Ω` all lean
/// towards the leading component, and their small remainders are the directions sought.
fn orthonormal_range(m: &mut DenseMatrix, count: usize) -> usize {
    let d = m.cols();
    let longest = (0..count).map(|j| norm(m.row(j))).fold(0.0, f64::max);
    let tolerance = RANGE_TOLERANCE * longest;
    let mut kept = 0;
    for j in 0..count {
        if j != kept {
            m.as_mut_slice().copy_within(j * d..(j + 1) * d, kept * d);
        }
        let (done, rest) = m.as_mut_slice().split_at_mut(kept * d);
        let row = &mut rest[..d];
        for _ in 0..2 {
            for finished in done.chunks_exact(d) {
                let projection = dot(finished, row);
                axpy(-projection, finished, row);
            }
        }
        let length = norm(row);
        if length > tolerance && length > 0.0 {
            scale(1.0 / length, row);
            kept += 1;
        }
    }
    kept
}

/// Two passes of modified Gram-Schmidt over the first `count` rows of `m`; returns how many leading
/// rows stay independent (a row that loses most of its length is where the kept set ends).
fn orthonormalize(m: &mut DenseMatrix, count: usize) -> usize {
    let d = m.cols();
    for j in 0..count {
        let (done, rest) = m.as_mut_slice().split_at_mut(j * d);
        let row = &mut rest[..d];
        let before = norm(row);
        for _ in 0..2 {
            for finished in done.chunks_exact(d) {
                let projection = dot(finished, row);
                axpy(-projection, finished, row);
            }
        }
        let after = norm(row);
        if !(after > 0.5 * before && after > 0.0) {
            return j;
        }
        scale(1.0 / after, row);
    }
    count
}

/// Makes the largest-magnitude element of `v` positive: a deterministic sign for each component.
fn fix_sign(v: &mut [f64]) {
    let mut largest = 0.0_f64;
    for &x in v.iter() {
        if x.abs() > largest.abs() {
            largest = x;
        }
    }
    if largest < 0.0 {
        scale(-1.0, v);
    }
}

/// Held-out reconstruction errors for choosing the component count `[STAT-02]`.
#[derive(Clone, Debug, PartialEq)]
pub struct LeaveOneOut {
    /// Entry `k`: the mean over samples of `‖xᵢ − x̂ᵢ‖²`, where `x̂ᵢ` is the reconstruction of sample
    /// `i` with `k` components by a model fitted without it.
    pub mean_squared_error: Vec<f64>,
}

impl LeaveOneOut {
    /// The smallest component count whose held-out error is at most `target`, if any.
    pub fn components_for_error(&self, target: f64) -> Option<usize> {
        self.mean_squared_error.iter().position(|&e| e <= target)
    }

    /// The smallest component count whose held-out error is within `relative` (for example `0.01`) of
    /// the best: beyond it, extra components buy almost no generalisation.
    pub fn components_within(&self, relative: f64) -> usize {
        let best = self.mean_squared_error.iter().copied().fold(f64::INFINITY, f64::min);
        let span = self.mean_squared_error.first().copied().unwrap_or(0.0) - best;
        self.mean_squared_error.iter().position(|&e| e - best <= relative * span).unwrap_or(0)
    }
}

/// Leave-one-out cross-validation of PCA reconstruction for `0..=max_components` components: fits
/// `N` models of `N − 1` samples each. Needs at least 2 samples.
pub fn leave_one_out(data: &DenseMatrix, max_components: usize) -> Result<LeaveOneOut, SolveError> {
    let (n, d) = (data.rows(), data.cols());
    if n < 2 {
        return Err(SolveError::InvalidParameter { what: "pca leave-one-out: needs at least 2 samples" });
    }
    let mut sums = vec![0.0; max_components + 1];
    let mut rest = DenseMatrix::zeros(n - 1, d);
    let mut coefficients = Vec::with_capacity(max_components);
    let mut reconstruction = vec![0.0; d];
    for held in 0..n {
        for (target, source) in (0..n).filter(|&i| i != held).enumerate() {
            rest.row_mut(target).copy_from_slice(data.row(source));
        }
        let model = Pca::fit(&rest)?;
        let sample = data.row(held);
        coefficients.clear();
        coefficients.resize(model.component_count().min(max_components), 0.0);
        model.project_into(sample, &mut coefficients)?;
        for (k, sum) in sums.iter_mut().enumerate() {
            model.reconstruct_into(&coefficients[..k.min(coefficients.len())], &mut reconstruction)?;
            *sum += reconstruction.iter().zip(sample).map(|(r, x)| (r - x) * (r - x)).sum::<f64>();
        }
    }
    let mean_squared_error = sums.iter().map(|s| s / n as f64).collect();
    Ok(LeaveOneOut { mean_squared_error })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solve::svd::Svd;
    use crate::solve::testutil::{orthonormality_error, random_matrix};
    use humanart_core::rng::Rng;

    /// `N` samples of a known linear model `mean + Σ zⱼ·σⱼ·bⱼ + noise`, `bⱼ` orthonormal, `zⱼ` standard
    /// normal. Returns the data and the basis (rows).
    fn linear_model(rng: &mut Rng, n: usize, d: usize, sigmas: &[f64], noise: f64) -> (DenseMatrix, DenseMatrix) {
        let mut basis = random_matrix(rng, sigmas.len(), d);
        assert_eq!(orthonormal_range(&mut basis, sigmas.len()), sigmas.len());
        let mean: Vec<f64> = (0..d).map(|_| rng.range_f64(-5.0, 5.0)).collect();
        let mut data = DenseMatrix::zeros(n, d);
        for i in 0..n {
            let row = data.row_mut(i);
            row.copy_from_slice(&mean);
            for (j, &s) in sigmas.iter().enumerate() {
                axpy(s * rng.normal(), basis.row(j), row);
            }
            for v in row.iter_mut() {
                *v += noise * rng.normal();
            }
        }
        (data, basis)
    }

    /// The largest principal-angle sine between the row spaces of two matrices with orthonormal rows:
    /// `‖(I − PₐPₐᵀ)·b‖` over the rows of `b`.
    fn subspace_gap(a: &DenseMatrix, b: &DenseMatrix) -> f64 {
        let mut worst = 0.0_f64;
        for i in 0..b.rows() {
            let mut r = b.row(i).to_vec();
            for j in 0..a.rows() {
                axpy(-dot(a.row(j), &r), a.row(j), &mut r);
            }
            worst = worst.max(norm(&r));
        }
        worst
    }

    #[test]
    fn ac_wp02_1_points_on_a_line() {
        // x = (3, −1) + t·(1, 2), t ∈ {−2, …, 2}: sample variance of t is 2.5, so the component (1, 2)/√5
        // has variance 2.5 · 5 = 12.5 and nothing else varies.
        let data = DenseMatrix::from_fn(5, 2, |i, c| {
            let t = i as f64 - 2.0;
            if c == 0 { 3.0 + t } else { -1.0 + 2.0 * t }
        });
        for method in [PcaMethod::Gram, PcaMethod::Covariance, PcaMethod::Auto] {
            let pca = Pca::fit_with(&data, method).unwrap();
            assert_eq!(pca.component_count(), 1, "{method:?}");
            assert!((pca.variances()[0] - 12.5).abs() < 1e-13);
            assert!((pca.total_variance() - 12.5).abs() < 1e-13);
            let v = pca.component(0).unwrap();
            assert!((v[0] - 1.0 / 5f64.sqrt()).abs() < 1e-14 && (v[1] - 2.0 / 5f64.sqrt()).abs() < 1e-14);
            assert_eq!(pca.mean(), &[3.0, -1.0]);
            assert!((pca.project(&[5.0, 3.0]).unwrap()[0] - 2.0 * 5f64.sqrt()).abs() < 1e-13);
            assert!((pca.cumulative_explained_variance(1) - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn ac_wp27_1_recovers_a_known_subspace_with_fewer_samples_than_dimensions() {
        let mut rng = Rng::new(0xA0);
        let sigmas = [5.0, 3.0, 2.0, 1.0, 0.5];
        // N = 40 samples of dimension D = 300 (the shape-model regime).
        let (data, basis) = linear_model(&mut rng, 40, 300, &sigmas, 0.0);
        let pca = Pca::fit(&data).unwrap();
        assert_eq!(pca.component_count(), 5);
        assert!(orthonormality_error(&pca.components().transpose()) < 1e-13);
        assert!(subspace_gap(pca.components(), &basis) < 1e-12);
        assert!((pca.cumulative_explained_variance(5) - 1.0).abs() < 1e-12);
        // Any sample of the model, seen or not, is reconstructed exactly.
        let (fresh, _) = linear_model(&mut Rng::new(0xA0), 3, 300, &sigmas, 0.0);
        for i in 0..3 {
            assert!(pca.reconstruction_error(fresh.row(i), 5).unwrap() < 1e-11);
            assert!(pca.reconstruction_error(data.row(i), 5).unwrap() < 1e-11);
        }
        // With noise the subspace is still found, to within the noise level.
        let (noisy, basis) = linear_model(&mut rng, 40, 300, &sigmas, 0.01);
        let mut pca = Pca::fit(&noisy).unwrap();
        assert_eq!(pca.component_count(), 39);
        pca.truncate(5);
        assert!(subspace_gap(pca.components(), &basis) < 0.1, "{}", subspace_gap(pca.components(), &basis));
        assert!(pca.components_for_variance(0.99) <= 5);
    }

    #[test]
    fn ac_wp02_1_gram_and_covariance_paths_agree_with_the_svd() {
        let mut rng = Rng::new(0xA1);
        for (n, d) in [(8, 20), (20, 8), (10, 10), (2, 5)] {
            let data = random_matrix(&mut rng, n, d);
            let gram = Pca::fit_with(&data, PcaMethod::Gram).unwrap();
            let covariance = Pca::fit_with(&data, PcaMethod::Covariance).unwrap();
            assert_eq!(gram.component_count(), (n - 1).min(d), "{n}x{d}");
            assert_eq!(covariance.component_count(), gram.component_count());
            // Variances are the squared singular values of the centred data over N − 1.
            let mut centred = data.clone();
            for i in 0..n {
                axpy(-1.0, gram.mean(), centred.row_mut(i));
            }
            let svd = Svd::new(&centred).unwrap();
            for ((g, c), s) in gram.variances().iter().zip(covariance.variances()).zip(svd.singular_values()) {
                let expect = s * s / (n - 1) as f64;
                assert!((g - expect).abs() < 1e-12 && (c - expect).abs() < 1e-12);
            }
            for j in 0..gram.component_count() {
                let (a, b) = (gram.component(j).unwrap(), covariance.component(j).unwrap());
                assert!(a.iter().zip(b).all(|(p, q)| (p - q).abs() < 1e-10), "{n}x{d} component {j}");
            }
            let total: f64 = gram.variances().iter().sum();
            assert!((total - gram.total_variance()).abs() < 1e-12);
            assert!(gram.explained_variance_ratio(0) >= gram.explained_variance_ratio(1));
        }
    }

    #[test]
    fn projection_reconstruction_and_mahalanobis() {
        let mut rng = Rng::new(0xA2);
        let (data, _) = linear_model(&mut rng, 30, 12, &[4.0, 2.0, 1.0], 0.0);
        let pca = Pca::fit(&data).unwrap();
        assert_eq!(pca.component_count(), 3);
        let sample = data.row(4);
        let mut c = [0.0; 3];
        pca.project_into(sample, &mut c).unwrap();
        let mut out = vec![0.0; 12];
        pca.reconstruct_into(&c, &mut out).unwrap();
        assert!(out.iter().zip(sample).all(|(p, q)| (p - q).abs() < 1e-12));
        // Coefficients are uncorrelated with the component variances as their sample variances.
        let all: Vec<Vec<f64>> = (0..30).map(|i| pca.project(data.row(i)).unwrap()).collect();
        for j in 0..3 {
            let variance: f64 = all.iter().map(|c| c[j] * c[j]).sum::<f64>() / 29.0;
            assert!((variance - pca.variances()[j]).abs() < 1e-10);
            let mean: f64 = all.iter().map(|c| c[j]).sum::<f64>() / 30.0;
            assert!(mean.abs() < 1e-12);
        }
        // One standard deviation along the second component.
        let one_sigma = [0.0, pca.variances()[1].sqrt()];
        assert!((pca.mahalanobis(&one_sigma).unwrap() - 1.0).abs() < 1e-14);
        // The truncated reconstruction misses exactly the dropped coefficients.
        let error = pca.reconstruction_error(sample, 1).unwrap();
        assert!((error - (c[1] * c[1] + c[2] * c[2]).sqrt()).abs() < 1e-12);
        assert_eq!(pca.reconstruct(&[]).unwrap(), pca.mean());
    }

    #[test]
    fn ac_wp02_1_leave_one_out_selects_the_true_rank() {
        let mut rng = Rng::new(0xA3);
        let (data, _) = linear_model(&mut rng, 25, 60, &[3.0, 2.0, 1.0], 1e-4);
        let loo = leave_one_out(&data, 6).unwrap();
        assert_eq!(loo.mean_squared_error.len(), 7);
        assert!(loo.mean_squared_error.windows(2).take(3).all(|w| w[1] < 0.5 * w[0]), "{loo:?}");
        // Beyond the true rank only noise is left: 60 values of variance 1e-8.
        assert!(loo.mean_squared_error[3] < 1e-5, "{loo:?}");
        assert_eq!(loo.components_for_error(1e-5), Some(3));
        assert_eq!(loo.components_within(0.001), 3);
        let pca = Pca::fit(&data).unwrap();
        assert_eq!(pca.components_for_variance(0.9999), 3);
    }

    #[test]
    fn degenerate_data_and_errors() {
        // One sample: the mean, no components.
        let one = DenseMatrix::from_rows(&[[1.0, 2.0, 3.0]]).unwrap();
        let pca = Pca::fit(&one).unwrap();
        assert_eq!((pca.component_count(), pca.total_variance()), (0, 0.0));
        assert_eq!(pca.cumulative_explained_variance(0), 1.0);
        assert_eq!(pca.components_for_variance(0.9), 0);
        assert_eq!(pca.reconstruction_error(&[1.0, 2.0, 4.0], 3).unwrap(), 1.0);
        // Identical samples: no variance.
        let same = DenseMatrix::from_rows(&[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]).unwrap();
        assert_eq!(Pca::fit(&same).unwrap().component_count(), 0);
        assert!(matches!(Pca::fit(&DenseMatrix::zeros(0, 3)), Err(SolveError::InvalidParameter { .. })));
        let mut bad = DenseMatrix::zeros(3, 2);
        bad[(1, 1)] = f64::NAN;
        assert_eq!(Pca::fit(&bad), Err(SolveError::NonFinite));
        let pca = Pca::fit(&random_matrix(&mut Rng::new(0xA4), 6, 4)).unwrap();
        assert!(pca.project(&[0.0; 3]).is_err());
        assert!(pca.project_into(&[0.0; 4], &mut [0.0; 9]).is_err());
        assert!(pca.reconstruct(&[0.0; 9]).is_err());
        assert!(pca.reconstruct_into(&[0.0; 2], &mut [0.0; 3]).is_err());
        assert!(pca.mahalanobis(&[0.0; 9]).is_err());
        assert_eq!(pca.project(&[0.0, f64::NAN, 0.0, 0.0]), Err(SolveError::NonFinite));
        assert!(pca.component(4).is_none());
        assert_eq!(pca.explained_variance_ratio(10), 0.0);
        assert!(leave_one_out(&one, 2).is_err());
        let mut truncated = pca.clone();
        truncated.truncate(2);
        assert_eq!(truncated.component_count(), 2);
        assert_eq!(truncated.components().rows(), 2);
        truncated.truncate(10);
        assert_eq!(truncated.component_count(), 2);
    }

    /// Largest |a − b| / max(|b|, tiny) over two variance lists of the same length.
    fn relative_gap(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y).abs() / y.abs().max(1e-300)).fold(0.0, f64::max)
    }

    #[test]
    fn stat_02_randomized_pca_matches_the_exact_pca_on_low_rank_data() {
        // Large N (more samples than dimensions and than a Gram matrix would like), exact rank 6
        // plus a little noise: the randomised fit of 6 components equals the exact one.
        let mut rng = Rng::new(0x57A7_0002);
        let sigmas = [9.0, 6.0, 4.0, 2.5, 1.5, 1.0];
        let (data, basis) = linear_model(&mut rng, 1500, 120, &sigmas, 1e-3);
        let exact = Pca::fit(&data).unwrap();
        let fast = Pca::fit_randomized(&data, &RandomizedPca::new(6)).unwrap();
        assert_eq!(fast.component_count(), 6);
        let orthonormality = orthonormality_error(&fast.components().transpose());
        assert!(orthonormality < 1e-12, "{orthonormality:e}");
        let gap = relative_gap(fast.variances(), &exact.variances()[..6]);
        assert!(gap < 1e-9, "variances differ by {gap:e}");
        println!("[STAT-02] randomized vs exact, N 1500, D 120, k 6: variance gap {gap:.2e}");
        for j in 0..6 {
            let agreement = dot(fast.component(j).unwrap(), exact.component(j).unwrap());
            assert!((agreement - 1.0).abs() < 1e-9, "component {j}: dot {agreement}");
        }
        assert!(subspace_gap(fast.components(), &basis) < 1e-3);
        assert_eq!(fast.total_variance(), exact.total_variance());
        assert_eq!(fast.mean(), exact.mean());
        // Projection and reconstruction behave as for the exact model.
        let sample = data.row(17);
        let (a, b) = (fast.project(sample).unwrap(), exact.project(sample).unwrap());
        for (x, y) in a.iter().zip(&b[..6]) {
            assert!((x - y).abs() < 1e-8, "{x} vs {y}");
        }
        assert!(fast.reconstruction_error(sample, 6).unwrap() < 0.05);
        // Deterministic for a seed.
        assert_eq!(Pca::fit_randomized(&data, &RandomizedPca::new(6)).unwrap(), fast);
    }

    #[test]
    fn stat_02_randomized_pca_on_exactly_low_rank_and_wide_data() {
        // Exact rank 4, N < D, asking for more components than the rank: only 4 are kept, and
        // they span the true subspace. Without power iterations the subspace is still exact,
        // since the range of X has dimension 4 < k + p.
        let mut rng = Rng::new(0x57A7_0003);
        let (data, basis) = linear_model(&mut rng, 300, 500, &[5.0, 3.0, 2.0, 1.0], 0.0);
        let exact = Pca::fit(&data).unwrap();
        for power_iterations in [0, 2] {
            let settings = RandomizedPca { components: 8, oversampling: 6, power_iterations, seed: 9 };
            let fast = Pca::fit_randomized(&data, &settings).unwrap();
            assert_eq!(fast.component_count(), 4, "q = {power_iterations}");
            assert!(subspace_gap(fast.components(), &basis) < 1e-9);
            assert!(relative_gap(fast.variances(), &exact.variances()[..4]) < 1e-9);
        }
        // A slowly decaying spectrum: the leading variances still agree closely with q = 4.
        let sigmas: Vec<f64> = (0..40).map(|i| 10.0 / (1.0 + i as f64)).collect();
        let (data, _) = linear_model(&mut rng, 800, 200, &sigmas, 0.05);
        let exact = Pca::fit(&data).unwrap();
        let fast = Pca::fit_randomized(&data, &RandomizedPca::new(10)).unwrap();
        let gap = relative_gap(fast.variances(), &exact.variances()[..10]);
        assert!(gap < 1e-3, "slow spectrum: variances differ by {gap:e}");
        println!("[STAT-02] randomized vs exact, slow spectrum (1/i, 40 terms), k 10, q 4: variance gap {gap:.2e}");
    }

    #[test]
    fn stat_02_randomized_pca_refuses_bad_settings() {
        let data = random_matrix(&mut Rng::new(0xA5), 10, 4);
        for k in [0, 5] {
            assert!(matches!(
                Pca::fit_randomized(&data, &RandomizedPca::new(k)),
                Err(SolveError::InvalidParameter { .. })
            ));
        }
        assert!(Pca::fit_randomized(&DenseMatrix::zeros(0, 4), &RandomizedPca::new(1)).is_err());
        let mut bad = data.clone();
        bad[(2, 1)] = f64::INFINITY;
        assert_eq!(Pca::fit_randomized(&bad, &RandomizedPca::new(2)), Err(SolveError::NonFinite));
        let same = DenseMatrix::from_rows(&[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]).unwrap();
        assert_eq!(Pca::fit_randomized(&same, &RandomizedPca::new(2)).unwrap().component_count(), 0);
    }
}
