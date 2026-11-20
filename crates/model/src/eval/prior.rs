//! The identity prior: the plausibility score `[ID-04]` and sampling of new identities `[ID-06]`.
//!
//! **Model.** The prior is a Gaussian over the shape coefficients, block-diagonal by *groups*: each
//! group (the global identity space, one region's local components, the asymmetry components, ...)
//! has a mean `μ_g` and a covariance `Σ_g` over its own components, and groups are independent. A
//! fitted model whose groups correlate must put them into one group. The statistics come from the
//! retained coefficient covariance of the model build `[STAT-08]`; for PCA coefficients `Σ_g` is the
//! diagonal of the retained eigenvalues ([`Covariance::Diagonal`]), a full block is also accepted.
//! Components in no group (authored semantic axes, expressions, correctives) carry no statistics:
//! they do not enter the score and sampling leaves them alone.
//!
//! **Score** `[ID-04]`. A group's squared Mahalanobis distance is `d²_g = (c−μ)ᵀ Σ_g⁻¹ (c−μ)`, computed
//! by forward substitution with the covariance's Cholesky factor. Under the prior `d²_g` follows a
//! chi-square distribution with `k_g` (the group's dimension) degrees of freedom, so the score
//!
//! `score = P(χ²_k > d²) = Q(k/2, d²/2)` (the regularised upper incomplete gamma function)
//!
//! is the fraction of identities drawn from the prior that lie further from the mean than this one:
//! 1 at the mean, uniform on `[0, 1]` for identities drawn from the prior (tested), towards 0 far
//! outside. The overall score uses `d² = Σ_g d²_g` with `k = Σ_g k_g`. A score below
//! [`WARNING_THRESHOLD`] (0.2, section 17) is a warning for the command result `[CMD-06]`; nothing
//! is ever blocked. A prior with no groups (no-data operation `[ID-07]`) scores everything 1 with
//! dimension 0, which callers report as "no prior".
//!
//! **Soft constraint** `[ID-04]`. [`IdentityPrior::regularisation_weights`] gives `ω_c = √(Σ⁻¹)_cc`,
//! the per-component weights of the landmark solver's regularisation `[LM-01]`, so a drag prefers
//! changes along the directions the data varies in; [`IdentityPrior::gradient`] gives `Σ⁻¹ (c−μ)`,
//! the gradient of `½ d²`, for optimisers that add the prior as a penalty.
//!
//! **Sampling** `[ID-06]`. A sample is `c = μ + T·L·z` per group (`L` the covariance factor, `T`
//! the temperature, `z` standard normal deviates from [`humanart_core::rng::Rng`], one stream per
//! group so that constraining one group does not change the others' draws). Fixed components and
//! the components of locked regions keep their current values and the free ones are drawn from the
//! conditional Gaussian: with the precision `Λ = Σ⁻¹`, `c_U ~ N(μ_U − Λ_UU⁻¹ Λ_UF (c_F − μ_F),
//! T² Λ_UU⁻¹)`. `T = 0` gives the (conditional) mean. A seed decides the sample completely.
//!
//! **JSON form** (`identity-prior.json` of a fitted model):
//!
//! ```json
//! { "groups": [
//!     { "name": "global", "components": ["id.0", "id.1"], "mean": [0, 0], "variance": [1.0, 0.4] },
//!     { "name": "nose", "components": ["nose.0", "nose.1"],
//!       "covariance": [[0.5, 0.1], [0.1, 0.2]] } ] }
//! ```
//!
//! `mean` is optional (zeros); exactly one of `variance` (the diagonal) and `covariance` (a full
//! symmetric positive definite matrix, row by row) is given. Components are delta component names
//! of the template and must be shape components (identity global, regional, semantic, asymmetry).
//! Every count is checked against [`MAX_GROUPS`], [`MAX_GROUP_DIMENSION`] and
//! [`MAX_MATRIX_ENTRIES`] before anything is allocated `[ROB-01]`.

use humanart_core::rng::Rng;
use humanart_io::json::{Node, Value};
use humanart_math::solve::{Cholesky, DenseMatrix};

use super::error::{EvalError, EvalResult};
use super::evaluator::is_shape_kind;
use crate::delta::DeltaBases;

/// The most groups of a prior.
pub const MAX_GROUPS: usize = 1024;
/// The most components of one group.
pub const MAX_GROUP_DIMENSION: usize = 1024;
/// The most stored covariance entries over all groups (`Σ k_g²`, 32 MiB of `f64`).
pub const MAX_MATRIX_ENTRIES: usize = 1 << 22;
/// The highest sampling temperature.
pub const MAX_TEMPERATURE: f32 = 10.0;
/// An overall score below this warns (section 17: "overall score < 0.2 warns; never blocks").
pub const WARNING_THRESHOLD: f32 = 0.2;
/// The relative asymmetry a full covariance may have before it is refused.
const SYMMETRY_TOLERANCE: f64 = 1e-9;
/// "In no group".
const NO_GROUP: u32 = u32::MAX;

/// The covariance of one group.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Covariance<'a> {
    /// Per-component variances (independent components, PCA coefficients); each finite and positive.
    Diagonal(&'a [f64]),
    /// A full `k × k` symmetric positive definite matrix, row-major.
    Full(&'a [f64]),
}

/// The description of one group for [`IdentityPrior::add_group`].
#[derive(Clone, Copy, Debug)]
pub struct GroupSpec<'a> {
    /// The group's name (unique in the prior).
    pub name: &'a str,
    /// Its delta components (shape kinds only, each in at most one group).
    pub components: &'a [u32],
    /// The mean per component (empty: zeros).
    pub mean: &'a [f64],
    /// The covariance over the components.
    pub covariance: Covariance<'a>,
}

/// One group of the prior.
#[derive(Clone, Debug)]
struct Group {
    name: String,
    region: Option<u16>,
    components: Vec<u32>,
    mean: Vec<f64>,
    /// The Cholesky factor of the covariance.
    factor: Cholesky,
    /// `√(Σ⁻¹)_cc` per component.
    weights: Vec<f64>,
}

/// The plausibility of one group.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupScore {
    /// The group index.
    pub group: u32,
    /// The squared Mahalanobis distance `d²_g` (its contribution to the overall distance, the
    /// quantity of the plausibility heat map).
    pub distance_sq: f64,
    /// The group's dimension `k_g`.
    pub dimension: usize,
    /// `P(χ²_k > d²)` in `[0, 1]`.
    pub score: f32,
}

/// The plausibility of a set of coefficients `[ID-04]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Plausibility {
    /// The overall score in `[0, 1]` (1 when the prior is empty).
    pub overall: f32,
    /// The overall squared Mahalanobis distance.
    pub distance_sq: f64,
    /// The overall dimension (0: no prior, the score says nothing).
    pub dimension: usize,
    /// Per group, in group order.
    pub groups: Vec<GroupScore>,
}

impl Plausibility {
    /// Whether the overall score is below [`WARNING_THRESHOLD`] (a warning, never a refusal).
    #[must_use]
    pub fn warns(&self) -> bool {
        self.dimension > 0 && self.overall < WARNING_THRESHOLD
    }

    /// The groups whose own score is below [`WARNING_THRESHOLD`] (out of distribution).
    pub fn out_of_distribution(&self) -> impl Iterator<Item = &GroupScore> + '_ {
        self.groups.iter().filter(|g| g.score < WARNING_THRESHOLD)
    }
}

/// One sampling request `[ID-06]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SampleRequest<'a> {
    /// The seed: equal requests with equal seeds give bit-identical samples.
    pub seed: u64,
    /// The spread relative to the prior, in `[0, MAX_TEMPERATURE]` (0: the mean, 1: the prior).
    pub temperature: f32,
    /// Per delta component: `true` = keep the current value (empty: none fixed).
    pub fixed: &'a [bool],
    /// Per region: `true` = keep every component whose bounding region this is (empty: none).
    pub locked_regions: &'a [bool],
}

/// The identity prior of one template's delta components.
#[derive(Clone, Debug)]
pub struct IdentityPrior {
    shape: Vec<bool>,
    regions: Vec<u16>,
    owner: Vec<u32>,
    groups: Vec<Group>,
    matrix_entries: usize,
}

impl IdentityPrior {
    /// An empty prior (no groups) for the components of `deltas`: the no-data prior `[ID-07]`.
    #[must_use]
    pub fn new(deltas: &DeltaBases) -> Self {
        Self {
            shape: deltas.components().iter().map(|c| is_shape_kind(c.kind)).collect(),
            regions: deltas.components().iter().map(|c| c.region).collect(),
            owner: vec![NO_GROUP; deltas.len()],
            groups: Vec::new(),
            matrix_entries: 0,
        }
    }

    /// Adds a group and returns its index. The covariance is factored here; a matrix that is not
    /// symmetric positive definite is refused.
    pub fn add_group(&mut self, spec: &GroupSpec<'_>) -> EvalResult<u32> {
        let k = spec.components.len();
        if self.groups.len() >= MAX_GROUPS {
            return Err(EvalError::limit("prior group count", MAX_GROUPS, self.groups.len() + 1));
        }
        if k == 0 {
            return Err(EvalError::Range { what: "prior group", index: self.groups.len(), rule: "has no components" });
        }
        if k > MAX_GROUP_DIMENSION {
            return Err(EvalError::limit("prior group dimension", MAX_GROUP_DIMENSION, k));
        }
        if self.matrix_entries + k * k > MAX_MATRIX_ENTRIES {
            return Err(EvalError::limit("prior covariance entries", MAX_MATRIX_ENTRIES, self.matrix_entries + k * k));
        }
        if self.groups.iter().any(|g| g.name == spec.name) {
            return Err(EvalError::Range { what: "prior group", index: self.groups.len(), rule: "name used twice" });
        }
        for (i, &c) in spec.components.iter().enumerate() {
            let Some(&shape) = self.shape.get(c as usize) else {
                return Err(EvalError::Missing { what: "delta component", index: i });
            };
            if !shape {
                return Err(EvalError::Range { what: "prior component", index: i, rule: "not a shape component" });
            }
            if self.owner[c as usize] != NO_GROUP || spec.components[..i].contains(&c) {
                return Err(EvalError::Range { what: "prior component", index: i, rule: "in more than one group" });
            }
        }
        let mean = match spec.mean.len() {
            0 => vec![0.0; k],
            n if n == k => spec.mean.to_vec(),
            n => return Err(EvalError::Length { what: "prior mean", limit: k, got: n }),
        };
        if let Some(i) = mean.iter().position(|m| !m.is_finite()) {
            return Err(EvalError::NonFinite { what: "prior mean", index: i });
        }
        let matrix = match spec.covariance {
            Covariance::Diagonal(v) => {
                if v.len() != k {
                    return Err(EvalError::Length { what: "prior variances", limit: k, got: v.len() });
                }
                if let Some(i) = v.iter().position(|x| !(x.is_finite() && *x > 0.0)) {
                    return Err(EvalError::Range {
                        what: "prior variance",
                        index: i,
                        rule: "must be finite and positive",
                    });
                }
                DenseMatrix::from_diagonal(v)
            }
            Covariance::Full(m) => {
                if m.len() != k * k {
                    return Err(EvalError::Length { what: "prior covariance", limit: k * k, got: m.len() });
                }
                if let Some(i) = m.iter().position(|x| !x.is_finite()) {
                    return Err(EvalError::NonFinite { what: "prior covariance", index: i });
                }
                for r in 0..k {
                    for s in 0..r {
                        let (a, b) = (m[r * k + s], m[s * k + r]);
                        if (a - b).abs() > SYMMETRY_TOLERANCE * (a.abs() + b.abs()).max(f64::MIN_POSITIVE) {
                            return Err(EvalError::Range {
                                what: "prior covariance",
                                index: r * k + s,
                                rule: "not symmetric",
                            });
                        }
                    }
                }
                DenseMatrix::from_vec(k, k, m.to_vec())?
            }
        };
        let factor = Cholesky::new(&matrix)?;
        let precision = factor.inverse();
        let weights: Vec<f64> = (0..k).map(|i| precision[(i, i)].max(0.0).sqrt()).collect();
        let first = self.regions[spec.components[0] as usize];
        let region = spec.components.iter().all(|&c| self.regions[c as usize] == first).then_some(first);
        let index = self.groups.len() as u32;
        for &c in spec.components {
            self.owner[c as usize] = index;
        }
        self.matrix_entries += k * k;
        self.groups.push(Group {
            name: spec.name.to_owned(),
            region,
            components: spec.components.to_vec(),
            mean,
            factor,
            weights,
        });
        Ok(index)
    }

    /// Reads a prior from its JSON form (see the module comment), resolving component names against
    /// `deltas`.
    pub fn from_json(value: &Value, deltas: &DeltaBases) -> EvalResult<Self> {
        let mut prior = Self::new(deltas);
        let root = value.node();
        root.deny_unknown(&["groups"])?;
        let groups = root.field("groups")?;
        if groups.array_len()? > MAX_GROUPS {
            return Err(EvalError::limit("prior group count", MAX_GROUPS, groups.array_len()?));
        }
        let mut components: Vec<u32> = Vec::new();
        let mut mean: Vec<f64> = Vec::new();
        let mut matrix: Vec<f64> = Vec::new();
        for group in groups.elements()? {
            group.deny_unknown(&["name", "components", "mean", "variance", "covariance"])?;
            let list = group.field("components")?;
            let k = list.array_len()?;
            if k > MAX_GROUP_DIMENSION {
                return Err(EvalError::limit("prior group dimension", MAX_GROUP_DIMENSION, k));
            }
            if prior.matrix_entries + k * k > MAX_MATRIX_ENTRIES {
                return Err(EvalError::limit(
                    "prior covariance entries",
                    MAX_MATRIX_ENTRIES,
                    prior.matrix_entries + k * k,
                ));
            }
            components.clear();
            for name in list.elements()? {
                let c = deltas.find(name.str()?).ok_or_else(|| name.invalid("no delta component of this name"))?;
                components.push(c);
            }
            mean.clear();
            if let Some(m) = group.opt_field("mean")? {
                read_numbers(&m, k, &mut mean)?;
            }
            matrix.clear();
            let covariance = match (group.opt_field("variance")?, group.opt_field("covariance")?) {
                (Some(v), None) => {
                    read_numbers(&v, k, &mut matrix)?;
                    Covariance::Diagonal(&matrix)
                }
                (None, Some(rows)) => {
                    if rows.array_len()? != k {
                        return Err(rows.invalid("the covariance needs one row per component").into());
                    }
                    for row in rows.elements()? {
                        read_numbers(&row, k, &mut matrix)?;
                    }
                    Covariance::Full(&matrix)
                }
                _ => return Err(group.invalid("give exactly one of variance and covariance").into()),
            };
            let name = group.field("name")?.str()?;
            prior.add_group(&GroupSpec { name, components: &components, mean: &mean, covariance })?;
        }
        Ok(prior)
    }

    /// The number of groups (0: no-data prior).
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// The total dimension (components with statistics).
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.groups.iter().map(|g| g.components.len()).sum()
    }

    /// The name of group `g`.
    #[must_use]
    pub fn group_name(&self, g: usize) -> Option<&str> {
        self.groups.get(g).map(|x| x.name.as_str())
    }

    /// The region of group `g` when all its components share one bounding region (the region of
    /// its plausibility contribution in the heat map), `None` for groups spanning regions.
    #[must_use]
    pub fn group_region(&self, g: usize) -> Option<u16> {
        self.groups.get(g).and_then(|x| x.region)
    }

    /// The delta components of group `g` (empty when out of range).
    #[must_use]
    pub fn group_components(&self, g: usize) -> &[u32] {
        self.groups.get(g).map_or(&[], |x| &x.components)
    }

    /// The group of delta component `c`, if it has statistics.
    #[must_use]
    pub fn group_of(&self, c: usize) -> Option<u32> {
        self.owner.get(c).copied().filter(|&g| g != NO_GROUP)
    }

    /// Scores `coefficients` (shape weights per delta component; missing trailing entries are 0).
    pub fn score(&self, coefficients: &[f32]) -> EvalResult<Plausibility> {
        self.check_coefficients(coefficients)?;
        let mut scratch = vec![0.0; self.groups.iter().map(|g| g.components.len()).max().unwrap_or(0)];
        let mut groups = Vec::with_capacity(self.groups.len());
        let (mut total, mut dimension) = (0.0, 0);
        for (index, g) in self.groups.iter().enumerate() {
            let k = g.components.len();
            let z = &mut scratch[..k];
            centred(g, coefficients, z);
            g.factor.forward_in_place(z)?;
            let d2: f64 = z.iter().map(|x| x * x).sum();
            total += d2;
            dimension += k;
            groups.push(GroupScore {
                group: index as u32,
                distance_sq: d2,
                dimension: k,
                score: chi_square_survival(d2, k) as f32,
            });
        }
        Ok(Plausibility {
            overall: chi_square_survival(total, dimension) as f32,
            distance_sq: total,
            dimension,
            groups,
        })
    }

    /// Writes `ω_c = √(Σ⁻¹)_cc` into `out` (one entry per delta component) for every component with
    /// statistics, for the landmark solver's regularisation `[LM-01]`; other entries are left as they
    /// are (fill them with 1 first for the solver's default).
    pub fn regularisation_weights(&self, out: &mut [f32]) -> EvalResult<()> {
        self.check_output(out)?;
        for g in &self.groups {
            for (&c, &w) in g.components.iter().zip(&g.weights) {
                out[c as usize] = w as f32;
            }
        }
        Ok(())
    }

    /// Writes the gradient of `½ d²`, `Σ⁻¹ (c − μ)`, into `out` (one entry per delta component; 0
    /// for components without statistics).
    pub fn gradient(&self, coefficients: &[f32], out: &mut [f32]) -> EvalResult<()> {
        self.check_coefficients(coefficients)?;
        self.check_output(out)?;
        out.fill(0.0);
        let mut scratch = vec![0.0; self.groups.iter().map(|g| g.components.len()).max().unwrap_or(0)];
        for g in &self.groups {
            let x = &mut scratch[..g.components.len()];
            centred(g, coefficients, x);
            g.factor.solve_in_place(x)?;
            for (&c, &v) in g.components.iter().zip(x.iter()) {
                out[c as usize] = v as f32;
            }
        }
        Ok(())
    }

    /// Samples an identity `[ID-06]`: `out` (one entry per delta component) becomes `current` with
    /// every free component that has statistics drawn from the prior, conditioned on the fixed ones.
    /// On error `out` is left unchanged.
    pub fn sample(&self, current: &[f32], request: &SampleRequest<'_>, out: &mut [f32]) -> EvalResult<()> {
        self.check_coefficients(current)?;
        self.check_output(out)?;
        let t = request.temperature;
        if !(t.is_finite() && (0.0..=MAX_TEMPERATURE).contains(&t)) {
            return Err(EvalError::Range { what: "sampling temperature", index: 0, rule: "must be in [0, 10]" });
        }
        if request.fixed.len() > self.owner.len() {
            return Err(EvalError::Length { what: "fixed flags", limit: self.owner.len(), got: request.fixed.len() });
        }
        let t = f64::from(t);
        let mut result: Vec<f32> = (0..out.len()).map(|c| current.get(c).copied().unwrap_or(0.0)).collect();
        let (mut z, mut free, mut fixed) = (Vec::new(), Vec::new(), Vec::new());
        for (index, g) in self.groups.iter().enumerate() {
            let k = g.components.len();
            // One stream per group: constraining one group leaves the others' draws as they were.
            let mut rng = Rng::with_stream(request.seed, index as u64);
            z.clear();
            z.extend((0..k).map(|_| rng.normal()));
            free.clear();
            fixed.clear();
            for (slot, &c) in g.components.iter().enumerate() {
                let c = c as usize;
                let held = request.fixed.get(c).copied().unwrap_or(false)
                    || request.locked_regions.get(self.regions[c] as usize).copied().unwrap_or(false);
                if held {
                    fixed.push(slot);
                } else {
                    free.push(slot);
                }
            }
            if free.is_empty() {
                continue;
            }
            if fixed.is_empty() {
                // c = μ + T·L·z.
                let l = g.factor.l();
                for i in 0..k {
                    let v = g.mean[i] + t * (0..=i).map(|j| l.at(i, j) * z[j]).sum::<f64>();
                    result[g.components[i] as usize] = v as f32;
                }
            } else {
                let drawn = conditional_draw(g, current, &free, &fixed, &z, t)?;
                for (&slot, &v) in free.iter().zip(&drawn) {
                    result[g.components[slot] as usize] = v as f32;
                }
            }
        }
        if let Some(i) = result.iter().position(|v| !v.is_finite()) {
            return Err(EvalError::NonFinite { what: "sampled coefficient", index: i });
        }
        out.copy_from_slice(&result);
        Ok(())
    }

    fn check_coefficients(&self, coefficients: &[f32]) -> EvalResult<()> {
        if coefficients.len() > self.owner.len() {
            return Err(EvalError::Length { what: "shape weights", limit: self.owner.len(), got: coefficients.len() });
        }
        if let Some(i) = coefficients.iter().position(|v| !v.is_finite()) {
            return Err(EvalError::NonFinite { what: "shape weight", index: i });
        }
        Ok(())
    }

    fn check_output(&self, out: &[f32]) -> EvalResult<()> {
        if out.len() == self.owner.len() {
            Ok(())
        } else {
            Err(EvalError::Length { what: "output weights", limit: self.owner.len(), got: out.len() })
        }
    }
}

/// `c − μ` of group `g` into `out` (missing coefficients are 0).
fn centred(g: &Group, coefficients: &[f32], out: &mut [f64]) {
    for ((o, &c), &m) in out.iter_mut().zip(&g.components).zip(&g.mean) {
        *o = f64::from(coefficients.get(c as usize).copied().unwrap_or(0.0)) - m;
    }
}

/// Draws the free slots of `g` from the Gaussian conditioned on the fixed slots' current values.
fn conditional_draw(
    g: &Group,
    current: &[f32],
    free: &[usize],
    fixed: &[usize],
    z: &[f64],
    t: f64,
) -> EvalResult<Vec<f64>> {
    let precision = g.factor.inverse();
    let u = free.len();
    let puu = DenseMatrix::from_fn(u, u, |r, s| precision[(free[r], free[s])]);
    // The conditional covariance Λ_UU⁻¹ and its factor.
    let cov = Cholesky::new(&puu)?.inverse();
    let factor = Cholesky::new(&cov)?;
    let deviation: Vec<f64> = fixed
        .iter()
        .map(|&j| f64::from(current.get(g.components[j] as usize).copied().unwrap_or(0.0)) - g.mean[j])
        .collect();
    let shift: Vec<f64> =
        free.iter().map(|&i| fixed.iter().zip(&deviation).map(|(&j, d)| precision[(i, j)] * d).sum()).collect();
    let l = factor.l();
    Ok((0..u)
        .map(|r| {
            let mean = g.mean[free[r]] - (0..u).map(|s| cov[(r, s)] * shift[s]).sum::<f64>();
            mean + t * (0..=r).map(|j| l.at(r, j) * z[free[j]]).sum::<f64>()
        })
        .collect())
}

/// Appends the `k` numbers of the JSON array `node` to `out`.
fn read_numbers(node: &Node<'_, '_>, k: usize, out: &mut Vec<f64>) -> EvalResult<()> {
    if node.array_len()? != k {
        return Err(node.invalid("needs one number per component").into());
    }
    for x in node.elements()? {
        out.push(x.f64()?);
    }
    Ok(())
}

/// The chi-square survival function `P(χ²_k > x) = Q(k/2, x/2)`: the plausibility score of a squared
/// Mahalanobis distance `x` in `k` dimensions `[ID-04]`. 1 for `x ≤ 0` or `k = 0`, 0 for NaN or
/// infinite `x`.
#[must_use]
pub fn chi_square_survival(x: f64, k: usize) -> f64 {
    if k == 0 || x <= 0.0 {
        return 1.0;
    }
    if !x.is_finite() {
        return 0.0;
    }
    regularized_gamma_q(k as f64 / 2.0, x / 2.0)
}

/// The most terms of the series and continued fraction (convergence needs about `√a` terms).
const GAMMA_MAX_ITER: usize = 100_000;

/// The regularised upper incomplete gamma function `Q(a, x)` for `a > 0`, `x > 0` finite: the
/// series for `P` below `a + 1`, the continued fraction (modified Lentz) for `Q` above.
fn regularized_gamma_q(a: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let ln_prefix = a * x.ln() - x - ln_gamma(a);
    if x < a + 1.0 {
        let (mut term, mut ap) = (1.0 / a, a);
        let mut sum = term;
        for _ in 0..GAMMA_MAX_ITER {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() < sum.abs() * f64::EPSILON {
                break;
            }
        }
        (1.0 - sum * ln_prefix.exp()).clamp(0.0, 1.0)
    } else {
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..GAMMA_MAX_ITER {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < TINY {
                d = TINY;
            }
            c = b + an / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            let step = d * c;
            h *= step;
            if (step - 1.0).abs() < f64::EPSILON {
                break;
            }
        }
        (ln_prefix.exp() * h).clamp(0.0, 1.0)
    }
}

/// `ln Γ(x)` for `x ≥ 0.5` by the Lanczos approximation (`g = 7`, nine coefficients; relative error
/// about 1e-15).
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let mut sum = COEFFICIENTS[0];
    for (i, &c) in COEFFICIENTS.iter().enumerate().skip(1) {
        sum += c / (x + i as f64);
    }
    let t = x + G + 0.5;
    0.5 * std::f64::consts::TAU.ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta::DeltaKind;
    use crate::eval::landmarks::{LandmarkGoal, LandmarkSolver, SolveRequest, SolveSettings};
    use crate::eval::tests::face_patch;
    use crate::template::Template;
    use humanart_math::Vec3;

    fn ids(t: &Template, names: &[&str]) -> Vec<u32> {
        names.iter().map(|n| t.deltas().find(n).unwrap()).collect()
    }

    /// A prior on the face patch: a diagonal global group, a full-covariance group and a regional
    /// group (region 2).
    fn patch_prior(t: &Template) -> IdentityPrior {
        let mut p = IdentityPrior::new(t.deltas());
        let global = ids(t, &["id.0", "id.1", "id.2"]);
        p.add_group(&GroupSpec {
            name: "global",
            components: &global,
            mean: &[0.1, 0.0, -0.2],
            covariance: Covariance::Diagonal(&[1.0, 0.25, 4.0]),
        })
        .unwrap();
        let pair = ids(t, &["id.3", "id.4"]);
        p.add_group(&GroupSpec {
            name: "pair",
            components: &pair,
            mean: &[],
            covariance: Covariance::Full(&[2.0, 0.6, 0.6, 0.5]),
        })
        .unwrap();
        let cheek = ids(t, &["cheek-R.width"]);
        p.add_group(&GroupSpec {
            name: "cheek-R",
            components: &cheek,
            mean: &[0.0],
            covariance: Covariance::Diagonal(&[0.09]),
        })
        .unwrap();
        p
    }

    /// Values from the chi-square tables (`scipy.stats.chi2`) and closed forms.
    #[test]
    fn id_04_chi_square_survival_matches_tables() {
        for x in [0.1, 1.0, 3.7, 20.0] {
            assert!((chi_square_survival(x, 2) - (-x / 2.0).exp()).abs() < 1e-14, "k = 2, x = {x}");
        }
        // erfc(1/√2).
        assert!((chi_square_survival(1.0, 1) - 0.317_310_507_862_914_15).abs() < 1e-13);
        // 95th percentiles for 3, 10 and 100 degrees of freedom.
        for (x, k) in [(7.814_727_903_251_178, 3), (18.307_038_053_275_146, 10), (124.342_113_404_004_07, 100)] {
            assert!((chi_square_survival(x, k) - 0.05).abs() < 1e-11, "k = {k}");
        }
        assert_eq!(chi_square_survival(0.0, 5), 1.0);
        assert_eq!(chi_square_survival(f64::INFINITY, 5), 0.0);
        assert_eq!(chi_square_survival(f64::NAN, 5), 0.0);
        assert_eq!(chi_square_survival(3.0, 0), 1.0);
        // Large dimensions: the median of χ²_k is close to k − 2/3; monotone in x.
        for k in [1_000_usize, 60_000] {
            let at_median = chi_square_survival(k as f64 - 2.0 / 3.0, k);
            assert!((at_median - 0.5).abs() < 1e-3, "k = {k}: {at_median}");
            let xs = [0.5, 0.9, 1.0, 1.1, 2.0].map(|f| f * k as f64);
            let qs = xs.map(|x| chi_square_survival(x, k));
            assert!(qs.windows(2).all(|w| w[0] >= w[1]), "{qs:?}");
        }
    }

    #[test]
    fn id_04_score_is_the_mahalanobis_tail_probability() {
        let t = face_patch(8, 6);
        let p = patch_prior(&t);
        assert_eq!((p.group_count(), p.dimension()), (3, 6));
        assert_eq!(p.group_region(2), Some(2));
        assert_eq!(p.group_region(0), Some(0));
        assert_eq!(p.group_name(1), Some("pair"));
        assert_eq!(p.group_of(t.deltas().find("id.3").unwrap() as usize), Some(1));
        assert_eq!(p.group_of(t.deltas().find("nose.height").unwrap() as usize), None);
        // At the mean everything scores 1.
        let mut c = vec![0.0_f32; t.deltas().len()];
        for (&k, m) in p.group_components(0).iter().zip([0.1, 0.0, -0.2]) {
            c[k as usize] = m;
        }
        let s = p.score(&c).unwrap();
        assert!(s.overall > 0.999_999, "{s:?}");
        assert!(!s.warns());
        // A known offset: d² by hand.
        let at = |name: &str| t.deltas().find(name).unwrap() as usize;
        c[at("id.1")] = 1.5; // (1.5 / 0.5)² = 9
        c[at("id.3")] = 1.0; // [1, 0] Σ⁻¹ [1, 0]ᵀ = 0.5 / (2·0.5 − 0.36) = 0.78125
        c[at("cheek-R.width")] = -0.9; // (0.9 / 0.3)² = 9
        c[at("nose.height")] = 50.0; // no statistics: ignored
        let s = p.score(&c).unwrap();
        let want = [9.0, 0.78125, 9.0];
        for (g, w) in s.groups.iter().zip(want) {
            assert!((g.distance_sq - w).abs() < 1e-6, "{g:?}");
            assert!((f64::from(g.score) - chi_square_survival(w, g.dimension)).abs() < 1e-6);
        }
        assert!((s.distance_sq - want.iter().sum::<f64>()).abs() < 1e-6);
        assert_eq!(s.dimension, 6);
        let ood: Vec<u32> = s.out_of_distribution().map(|g| g.group).collect();
        assert_eq!(ood, [0, 2]);
        // Far outside: warns, never refuses.
        c[at("id.2")] = 40.0;
        let s = p.score(&c).unwrap();
        assert!(s.warns() && s.overall < 1e-6, "{s:?}");
        // No-data prior [ID-07]: dimension 0, score 1, no warning.
        let empty = IdentityPrior::new(t.deltas()).score(&c).unwrap();
        assert_eq!((empty.overall, empty.dimension, empty.groups.len()), (1.0, 0, 0));
        assert!(!empty.warns());
    }

    /// Identities drawn from the prior score uniformly on `[0, 1]`: the mapping is calibrated, about
    /// 20 % of the prior's own samples warn.
    #[test]
    fn id_04_scores_of_prior_samples_are_uniform() {
        let t = face_patch(4, 6);
        let p = patch_prior(&t);
        let current = vec![0.0; t.deltas().len()];
        let mut out = vec![0.0; t.deltas().len()];
        let n = 4000_u32;
        let mut bins = [0_u32; 5];
        for seed in 0..n {
            let req = SampleRequest { seed: u64::from(seed), temperature: 1.0, ..SampleRequest::default() };
            p.sample(&current, &req, &mut out).unwrap();
            let s = p.score(&out).unwrap().overall;
            bins[((s * 5.0) as usize).min(4)] += 1;
        }
        for b in bins {
            let f = f64::from(b) / f64::from(n);
            assert!((f - 0.2).abs() < 0.03, "{bins:?}");
        }
    }

    #[test]
    fn id_04_soft_constraint_weights_and_gradient() {
        let t = face_patch(16, 6);
        let p = patch_prior(&t);
        let len = t.deltas().len();
        let mut w = vec![1.0; len];
        p.regularisation_weights(&mut w).unwrap();
        let at = |name: &str| t.deltas().find(name).unwrap() as usize;
        assert!((w[at("id.1")] - 2.0).abs() < 1e-6);
        assert!((w[at("cheek-R.width")] - 1.0 / 0.3).abs() < 1e-5);
        // Full block: √(Σ⁻¹)_00 = √(0.5 / 0.64).
        assert!((w[at("id.3")] - (0.5_f32 / 0.64).sqrt()).abs() < 1e-6);
        assert_eq!(w[at("nose.height")], 1.0);
        // The gradient is that of ½ d² (central differences).
        let mut c: Vec<f32> = (0..len).map(|i| 0.3 * ((i as f32) * 1.7).sin()).collect();
        let mut g = vec![0.0; len];
        p.gradient(&c, &mut g).unwrap();
        for k in 0..len {
            let h = 1e-2;
            let base = c[k];
            c[k] = base + h;
            let up = p.score(&c).unwrap().distance_sq;
            c[k] = base - h;
            let down = p.score(&c).unwrap().distance_sq;
            c[k] = base;
            let fd = (up - down) / (4.0 * f64::from(h));
            assert!((fd - f64::from(g[k])).abs() < 1e-3 * (1.0 + fd.abs()), "{k}: {fd} vs {}", g[k]);
        }
        // In the landmark solver: a component the prior says varies little changes less.
        let mut solver = LandmarkSolver::new(&t, SolveSettings::default());
        let rest = t.mesh().positions().to_vec();
        let target = solver.landmark_position(12, &rest).unwrap() + Vec3::Z * 0.01;
        let targets = [LandmarkGoal { landmark: 12, position: target }];
        solver.solve(&rest, &SolveRequest { targets: &targets, ..SolveRequest::default() }).unwrap();
        let plain = solver.delta()[at("id.1")].abs();
        let mut tight = IdentityPrior::new(t.deltas());
        tight
            .add_group(&GroupSpec {
                name: "id.1",
                components: &[at("id.1") as u32],
                mean: &[],
                covariance: Covariance::Diagonal(&[1e-4]),
            })
            .unwrap();
        let mut weights = vec![1.0; len];
        tight.regularisation_weights(&mut weights).unwrap();
        let req = SolveRequest { targets: &targets, regularisation: &weights, ..SolveRequest::default() };
        let report = solver.solve(&rest, &req).unwrap();
        assert!(report.max_residual < 1e-4);
        assert!(solver.delta()[at("id.1")].abs() < plain * 0.05, "{plain} vs {}", solver.delta()[at("id.1")]);
    }

    #[test]
    fn id_06_sampling_matches_the_prior_and_its_conditionals() {
        let t = face_patch(4, 3);
        let mut p = IdentityPrior::new(t.deltas());
        let comps = ids(&t, &["id.0", "id.1", "id.2"]);
        let mean = [0.5, -1.0, 0.25];
        let cov = [4.0, 1.2, 0.4, 1.2, 2.0, 0.3, 0.4, 0.3, 1.0];
        p.add_group(&GroupSpec { name: "g", components: &comps, mean: &mean, covariance: Covariance::Full(&cov) })
            .unwrap();
        let len = t.deltas().len();
        let current = vec![0.0; len];
        let mut out = vec![0.0; len];
        let moments = |temperature: f32, fixed: &[bool], current: &[f32]| {
            let n = 20_000_u32;
            let mut sum = [0.0_f64; 3];
            let mut sq = [[0.0_f64; 3]; 3];
            let mut out = vec![0.0; len];
            for seed in 0..n {
                let req = SampleRequest { seed: u64::from(seed), temperature, fixed, ..SampleRequest::default() };
                p.sample(current, &req, &mut out).unwrap();
                let x: Vec<f64> = comps.iter().map(|&c| f64::from(out[c as usize])).collect();
                for r in 0..3 {
                    sum[r] += x[r];
                    for s in 0..3 {
                        sq[r][s] += x[r] * x[s];
                    }
                }
            }
            let n = f64::from(n);
            let m = sum.map(|s| s / n);
            let c: Vec<f64> = (0..9).map(|i| sq[i / 3][i % 3] / n - m[i / 3] * m[i % 3]).collect();
            (m, c)
        };
        let (m, c) = moments(1.0, &[], &current);
        for i in 0..3 {
            assert!((m[i] - mean[i]).abs() < 0.05, "mean {m:?}");
        }
        for i in 0..9 {
            assert!((c[i] - cov[i]).abs() < 0.06 * cov[0], "cov {c:?}");
        }
        // Temperature scales the spread; 0 is the mean exactly.
        let (_, c) = moments(0.5, &[], &current);
        assert!((c[0] - 1.0).abs() < 0.05 && (c[4] - 0.5).abs() < 0.03, "{c:?}");
        p.sample(&current, &SampleRequest { seed: 3, ..SampleRequest::default() }, &mut out).unwrap();
        for (i, &k) in comps.iter().enumerate() {
            assert!((f64::from(out[k as usize]) - mean[i]).abs() < 1e-6);
        }
        // Fixing id.0 at 2.5: the others follow the conditional Gaussian (closed form).
        let mut fixed = vec![false; len];
        fixed[comps[0] as usize] = true;
        let mut at = vec![0.0; len];
        at[comps[0] as usize] = 2.5;
        let d = 2.5 - mean[0];
        let cond_mean = [mean[1] + 1.2 / 4.0 * d, mean[2] + 0.4 / 4.0 * d];
        let cond_cov = [2.0 - 1.2 * 1.2 / 4.0, 0.3 - 1.2 * 0.4 / 4.0, 1.0 - 0.4 * 0.4 / 4.0];
        p.sample(&at, &SampleRequest { seed: 9, fixed: &fixed, ..SampleRequest::default() }, &mut out).unwrap();
        assert_eq!(out[comps[0] as usize], 2.5);
        assert!((f64::from(out[comps[1] as usize]) - cond_mean[0]).abs() < 1e-6);
        assert!((f64::from(out[comps[2] as usize]) - cond_mean[1]).abs() < 1e-6);
        let (m, c) = moments(1.0, &fixed, &at);
        assert!((m[0] - 2.5).abs() < 1e-12 && c[0].abs() < 1e-9, "{m:?} {c:?}");
        assert!((m[1] - cond_mean[0]).abs() < 0.04 && (m[2] - cond_mean[1]).abs() < 0.03, "{m:?}");
        assert!((c[4] - cond_cov[0]).abs() < 0.06 && (c[5] - cond_cov[1]).abs() < 0.04, "{c:?}");
        assert!((c[8] - cond_cov[2]).abs() < 0.04, "{c:?}");
    }

    /// `[AC-WP09-3]`: sampling is deterministic per seed; constraints hold.
    #[test]
    fn ac_wp09_3_sampling_is_deterministic_per_seed() {
        let t = face_patch(6, 6);
        let p = patch_prior(&t);
        let len = t.deltas().len();
        let current: Vec<f32> = (0..len).map(|i| i as f32 * 0.01).collect();
        let draw = |req: &SampleRequest<'_>| {
            let mut out = vec![0.0; len];
            p.sample(&current, req, &mut out).unwrap();
            out
        };
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let req = SampleRequest { seed: 0xA11C_E5ED, temperature: 0.8, ..SampleRequest::default() };
        let a = draw(&req);
        assert_eq!(bits(&a), bits(&draw(&req)));
        assert_ne!(a, draw(&SampleRequest { seed: 1, ..req }));
        // Components without statistics keep their values.
        for c in 0..len {
            if p.group_of(c).is_none() {
                assert_eq!(a[c], current[c]);
            }
        }
        // Locking region 2 keeps the cheek group; fixing a component of the pair group leaves the
        // global group's draw unchanged (one stream per group).
        let at = |name: &str| t.deltas().find(name).unwrap() as usize;
        let locked = [false, false, true];
        let l = draw(&SampleRequest { locked_regions: &locked, ..req });
        assert_eq!(l[at("cheek-R.width")], current[at("cheek-R.width")]);
        let mut fixed = vec![false; len];
        fixed[at("id.3")] = true;
        let f = draw(&SampleRequest { fixed: &fixed, ..req });
        assert_eq!(f[at("id.3")], current[at("id.3")]);
        assert_ne!(f[at("id.4")], a[at("id.4")]);
        for &c in p.group_components(0) {
            assert_eq!(f[c as usize].to_bits(), a[c as usize].to_bits());
        }
        // Everything fixed: the sample is the current state.
        let all = vec![true; len];
        assert_eq!(draw(&SampleRequest { fixed: &all, ..req }), current);
    }

    #[test]
    fn bad_priors_and_requests_are_refused() {
        fn spec<'a>(components: &'a [u32], covariance: Covariance<'a>) -> GroupSpec<'a> {
            GroupSpec { name: "g", components, mean: &[], covariance }
        }
        let t = face_patch(4, 4);
        let len = t.deltas().len();
        let expression = t.deltas().components().iter().position(|c| c.kind == DeltaKind::Expression).unwrap() as u32;
        let id = ids(&t, &["id.0", "id.1"]);
        let twice = [id[0], id[0]];
        let diag = Covariance::Diagonal(&[1.0, 1.0]);
        let cases = [
            spec(&[], Covariance::Diagonal(&[])),
            spec(&[9999], Covariance::Diagonal(&[1.0])),
            spec(std::slice::from_ref(&expression), Covariance::Diagonal(&[1.0])),
            spec(&twice, diag),
            spec(&id, Covariance::Diagonal(&[1.0, 0.0])),
            spec(&id, Covariance::Diagonal(&[1.0])),
            spec(&id, Covariance::Full(&[1.0, 0.5, 0.4, 1.0])),
            spec(&id, Covariance::Full(&[1.0, 2.0, 2.0, 1.0])),
            spec(&id, Covariance::Full(&[1.0, f64::NAN, f64::NAN, 1.0])),
            GroupSpec { mean: &[0.0], ..spec(&id, diag) },
            GroupSpec { mean: &[0.0, f64::INFINITY], ..spec(&id, diag) },
        ];
        for (i, s) in cases.iter().enumerate() {
            assert!(IdentityPrior::new(t.deltas()).add_group(s).is_err(), "case {i}");
        }
        let mut p = IdentityPrior::new(t.deltas());
        p.add_group(&spec(&id, diag)).unwrap();
        let second = [id[1]];
        assert!(p.add_group(&GroupSpec { name: "h", ..spec(&second, Covariance::Diagonal(&[1.0])) }).is_err());
        let third = ids(&t, &["id.2"]);
        assert!(p.add_group(&spec(&third, Covariance::Diagonal(&[1.0]))).is_err());
        let ok = vec![0.0; len];
        let mut out = vec![0.0; len];
        for temperature in [-0.1, f32::NAN, 11.0] {
            let req = SampleRequest { temperature, ..SampleRequest::default() };
            assert!(p.sample(&ok, &req, &mut out).is_err());
        }
        let long = vec![false; len + 1];
        assert!(p.sample(&ok, &SampleRequest { fixed: &long, ..SampleRequest::default() }, &mut out).is_err());
        assert!(p.sample(&ok, &SampleRequest::default(), &mut out[1..]).is_err());
        assert!(p.score(&[f32::NAN]).is_err());
        assert!(p.score(&vec![0.0; len + 1]).is_err());
        assert!(p.gradient(&ok, &mut [0.0]).is_err());
        assert!(p.regularisation_weights(&mut [0.0]).is_err());
        assert!(p.group_name(5).is_none() && p.group_components(5).is_empty());
    }

    const PRIOR: &str = r#"{"groups": [
        {"name": "global", "components": ["id.0", "id.1"], "mean": [0.1, 0], "variance": [1.0, 0.4]},
        {"name": "pair", "components": ["id.2", "cheek-R.width"], "covariance": [[0.5, 0.1], [0.1, 0.2]]}
    ]}"#;

    #[test]
    fn prior_from_json() {
        let t = face_patch(4, 3);
        let p = IdentityPrior::from_json(&humanart_io::json::parse_str(PRIOR).unwrap(), t.deltas()).unwrap();
        assert_eq!((p.group_count(), p.dimension()), (2, 4));
        assert_eq!(p.group_region(1), None);
        let mut w = vec![1.0; t.deltas().len()];
        p.regularisation_weights(&mut w).unwrap();
        assert!((w[t.deltas().find("id.1").unwrap() as usize] - 0.4_f32.sqrt().recip()).abs() < 1e-6);
        for bad in [
            r#"{"groups": [{"name": "g", "components": ["id.0"], "variance": [1], "covariance": [[1]]}]}"#,
            r#"{"groups": [{"name": "g", "components": ["id.0"]}]}"#,
            r#"{"groups": [{"name": "g", "components": ["nope"], "variance": [1]}]}"#,
            r#"{"groups": [{"name": "g", "components": ["id.0"], "variance": [1, 2]}]}"#,
            r#"{"groups": [{"name": "g", "components": ["id.0"], "covariance": [[1], [1]]}]}"#,
            r#"{"groups": [{"name": "g", "components": ["id.0"], "variance": [1e999]}]}"#,
            r#"{"groups": [], "extra": 1}"#,
        ] {
            let v = humanart_io::json::parse_str(bad);
            assert!(v.is_err() || IdentityPrior::from_json(&v.unwrap(), t.deltas()).is_err(), "{bad}");
        }
    }

    /// `[ROB-01]`: mutated prior files never panic or hang; accepted ones score and sample finitely.
    #[test]
    fn prior_json_mutation_fuzz() {
        let deltas = face_patch(4, 3).deltas().clone();
        let corpus = vec![PRIOR.as_bytes().to_vec(), br#"{"groups": []}"#.to_vec()];
        humanart_core::fuzz::run(
            "identity-prior",
            &corpus,
            3000,
            0x1D04_F022,
            std::time::Duration::from_secs(60),
            move |bytes| {
                let Ok(v) = humanart_io::json::parse(bytes) else { return };
                if let Ok(p) = IdentityPrior::from_json(&v, &deltas) {
                    let c = vec![0.3; deltas.len()];
                    let s = p.score(&c).unwrap();
                    assert!((0.0..=1.0).contains(&s.overall));
                    let mut out = vec![0.0; deltas.len()];
                    let req = SampleRequest { seed: 1, temperature: 1.0, ..SampleRequest::default() };
                    if p.sample(&c, &req, &mut out).is_ok() {
                        assert!(out.iter().all(|x| x.is_finite()));
                    }
                }
            },
        );
    }
}
