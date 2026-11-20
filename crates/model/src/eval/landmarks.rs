//! The landmark solver: direct manipulation of the identity through landmarks `[LM-01]`..`[LM-03]`.
//!
//! **Problem.** Dragging landmarks to targets solves for the smallest change `Δc` of the shape
//! coefficients (identity global, identity regional, semantic and asymmetry components: the shape
//! components of the evaluator) that moves them there, while pinned landmarks stay put
//! `[LOCK-03]` and locked components do not change:
//!
//! `min Σ_targets ‖J_t Δc − (x_t − p_t)‖² + ρ Σ_pins ‖J_p Δc − (x_p − p_p)‖² + λ Σ_c ω_c² Δc_c²`
//!
//! over the unlocked components (locked ones are removed: `Δc_c = 0`). `p` are the landmarks' current
//! rest positions, `x` the targets (for a pin, the position it is pinned at), `ρ` the pin weight
//! (pins are heavily weighted rows), `λ` the regularisation and `ω_c` per-component regularisation
//! weights (1 by default; the identity prior's precision fits here, `[ID-04]` as a soft constraint).
//! `J` is the landmark Jacobian, computed once from the sparse delta bases at the three corner
//! vertices of each landmark: `J_l,c = Σ_k b_k · Δ_c[v_k]`. The rest shape is linear in the
//! coefficients, so one solve is exact (no iteration).
//!
//! **Solve.** With `m = 3·(targets + pins)` rows far fewer than the components, the solver uses the
//! dual (kernel) form of the normal equations, an `m × m` symmetric positive definite system solved
//! with the in-house dense Cholesky: `(J W⁻¹ Jᵀ + λ R⁻¹) y = b`, `Δc = W⁻¹ Jᵀ y` (`W = diag(ω²)`,
//! `R` = row weights). It is algebraically the same minimiser as the `n × n` primal system (tested).
//! All scratch is allocated by [`LandmarkSolver::new`]; a solve allocates nothing.
//!
//! **Result** `[LM-02]`: a parameter change ([`LandmarkSolver::delta`]), not a sculpt; the report
//! gives the largest target residual. When it exceeds the model's span (residual above a
//! threshold) [`LandmarkSolver::residual_layer`] turns the remainder into a sparse sculpt layer.

use humanart_math::Vec3;
use humanart_math::solve::cholesky::{cholesky_in_place, cholesky_solve_in_place};

use super::error::{EvalError, EvalResult};
use super::evaluator::is_shape_kind;
use super::layer::SculptLayer;
use crate::template::Template;

/// The most targets one solve may have.
pub const MAX_TARGETS: usize = 32;
/// The most pins one solve may have.
pub const MAX_PINS: usize = 256;
/// The most rows of one solve.
pub const MAX_ROWS: usize = 3 * (MAX_TARGETS + MAX_PINS);

/// Solver settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveSettings {
    /// The regularisation relative to the system's scale, positive: the `λ` of the problem is this
    /// times the mean diagonal of `J W⁻¹ Jᵀ` over the rows, so the setting does not depend on units
    /// or on how many components there are.
    pub lambda: f64,
    /// The weight `ρ` of pin rows relative to target rows, positive.
    pub pin_weight: f64,
    /// A target residual above this many metres means the drag exceeds the model's span `[LM-02]`.
    pub span_threshold: f32,
}

impl Default for SolveSettings {
    fn default() -> Self {
        Self { lambda: 1e-8, pin_weight: 1e4, span_threshold: 5e-4 }
    }
}

/// A landmark and a position: a drag target or a pin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LandmarkGoal {
    /// The landmark index in the template.
    pub landmark: u32,
    /// Where it should be (rest shape, metres).
    pub position: Vec3,
}

/// One solve's input.
#[derive(Clone, Copy, Debug, Default)]
pub struct SolveRequest<'a> {
    /// Landmarks to move, at most [`MAX_TARGETS`].
    pub targets: &'a [LandmarkGoal],
    /// Landmarks to keep at their positions, at most [`MAX_PINS`].
    pub pins: &'a [LandmarkGoal],
    /// Per delta component: `true` = locked (empty: nothing locked). Non-shape components are always
    /// left out.
    pub locked: &'a [bool],
    /// Per delta component: the regularisation weight `ω_c > 0` (empty: all 1).
    pub regularisation: &'a [f32],
}

/// What a solve achieved.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SolveReport {
    /// The largest distance between a target and where the solved change puts its landmark.
    pub max_residual: f32,
    /// The largest distance a pinned landmark moves from its pinned position.
    pub max_pin_drift: f32,
    /// The number of rows of the system (3 per target and pin).
    pub rows: usize,
    /// Whether `max_residual` is above the span threshold `[LM-02]`.
    pub exceeds_span: bool,
    /// The absolute `λ` the solve used (m² per squared coefficient).
    pub lambda: f64,
}

/// The landmark solver of one template.
#[derive(Clone, Debug)]
pub struct LandmarkSolver {
    components: usize,
    shape: Vec<bool>,
    corners: Vec<[u32; 3]>,
    weights: Vec<[f32; 3]>,
    jac_offsets: Vec<u32>,
    jac_components: Vec<u32>,
    jac_values: Vec<Vec3>,
    settings: SolveSettings,
    rows: Vec<u32>,
    row_rhs: Vec<Vec3>,
    row_weight: Vec<f64>,
    system: Vec<f64>,
    rhs: Vec<f64>,
    inv_w: Vec<f64>,
    delta: Vec<f32>,
    residuals: Vec<Vec3>,
}

impl LandmarkSolver {
    /// Builds the landmark Jacobian of `template` (sparse: per landmark, the components that move
    /// it) and all scratch.
    #[must_use]
    pub fn new(template: &Template, settings: SolveSettings) -> Self {
        let deltas = template.deltas();
        let shape: Vec<bool> = deltas.components().iter().map(|c| is_shape_kind(c.kind)).collect();
        let landmarks = template.landmarks();
        let mut jac_offsets = vec![0_u32];
        let mut jac_components = Vec::new();
        let mut jac_values = Vec::new();
        for lm in landmarks {
            for (c, _) in shape.iter().enumerate().filter(|(_, s)| **s) {
                let indices = deltas.indices(c);
                let values = deltas.deltas(c);
                let mut j = Vec3::ZERO;
                for (&v, &b) in lm.vertices.iter().zip(&lm.weights) {
                    if let Ok(i) = indices.binary_search(&v)
                        && let Some(&d) = values.get(i)
                    {
                        j += crate::delta::DeltaBases::delta_f32(d) * b;
                    }
                }
                if j != Vec3::ZERO {
                    jac_components.push(c as u32);
                    jac_values.push(j);
                }
            }
            jac_offsets.push(jac_components.len() as u32);
        }
        Self {
            components: deltas.len(),
            corners: landmarks.iter().map(|l| l.vertices).collect(),
            weights: landmarks.iter().map(|l| l.weights).collect(),
            jac_offsets,
            jac_components,
            jac_values,
            settings,
            rows: Vec::with_capacity(MAX_TARGETS + MAX_PINS),
            row_rhs: Vec::with_capacity(MAX_TARGETS + MAX_PINS),
            row_weight: Vec::with_capacity(MAX_TARGETS + MAX_PINS),
            system: vec![0.0; MAX_ROWS * MAX_ROWS],
            rhs: vec![0.0; MAX_ROWS],
            inv_w: vec![0.0; deltas.len()],
            delta: vec![0.0; deltas.len()],
            residuals: Vec::with_capacity(MAX_TARGETS + MAX_PINS),
            shape,
        }
    }

    /// The settings in force.
    #[must_use]
    pub const fn settings(&self) -> &SolveSettings {
        &self.settings
    }

    /// The Jacobian entries of landmark `l`: `(component, ∂position/∂weight)` (empty when out of
    /// range).
    pub fn jacobian(&self, l: usize) -> impl Iterator<Item = (u32, Vec3)> + '_ {
        let range = row_range(&self.jac_offsets, l as u32);
        self.jac_components[range.clone()].iter().copied().zip(self.jac_values[range].iter().copied())
    }

    /// The position of landmark `l` in `positions` (`None` when out of range).
    #[must_use]
    pub fn landmark_position(&self, l: usize, positions: &[Vec3]) -> Option<Vec3> {
        let (c, w) = (self.corners.get(l)?, self.weights.get(l)?);
        let mut p = Vec3::ZERO;
        for (&v, &b) in c.iter().zip(w) {
            p += *positions.get(v as usize)? * b;
        }
        Some(p)
    }

    /// Solves `request` against the current rest shape `rest` (the evaluator's
    /// [`rest_positions`](super::Evaluator::rest_positions)); the change is then in
    /// [`delta`](Self::delta). On error the previous delta is kept. Allocation-free.
    pub fn solve(&mut self, rest: &[Vec3], request: &SolveRequest<'_>) -> EvalResult<SolveReport> {
        self.check(rest, request)?;
        // Rows: targets first, then pins.
        self.rows.clear();
        self.row_rhs.clear();
        self.row_weight.clear();
        for (goals, weight) in [(request.targets, 1.0), (request.pins, self.settings.pin_weight)] {
            for g in goals {
                let p = self.landmark_position(g.landmark as usize, rest).unwrap_or(g.position);
                self.rows.push(g.landmark);
                self.row_rhs.push(g.position - p);
                self.row_weight.push(weight);
            }
        }
        for c in 0..self.components {
            let locked = request.locked.get(c).copied().unwrap_or(false);
            let w = f64::from(request.regularisation.get(c).copied().unwrap_or(1.0));
            self.inv_w[c] = if self.shape[c] && !locked { 1.0 / (w * w) } else { 0.0 };
        }
        let n = self.rows.len();
        let m = 3 * n;
        let (offsets, comps, values, inv_w) = (&self.jac_offsets, &self.jac_components, &self.jac_values, &self.inv_w);
        // A = J W⁻¹ Jᵀ + λ R⁻¹ (3x3 blocks per landmark pair), b = target offsets.
        let a = &mut self.system[..m * m];
        for i in 0..n {
            for j in 0..=i {
                let block = block_product(
                    row_range(offsets, self.rows[i]),
                    row_range(offsets, self.rows[j]),
                    comps,
                    values,
                    inv_w,
                );
                for (r, row) in block.iter().enumerate() {
                    for (s, &v) in row.iter().enumerate() {
                        a[(3 * i + r) * m + 3 * j + s] = v;
                        a[(3 * j + s) * m + 3 * i + r] = v;
                    }
                }
            }
            let b = self.row_rhs[i];
            self.rhs[3 * i..3 * i + 3].copy_from_slice(&[f64::from(b.x), f64::from(b.y), f64::from(b.z)]);
        }
        let trace: f64 = (0..m).map(|k| a[k * m + k]).sum();
        let scale = if m > 0 && trace > 0.0 && trace.is_finite() { trace / m as f64 } else { 1.0 };
        let lambda = self.settings.lambda * scale;
        for i in 0..n {
            let reg = lambda / self.row_weight[i];
            for r in 0..3 {
                a[(3 * i + r) * m + 3 * i + r] += reg;
            }
        }
        cholesky_in_place(a, m)?;
        cholesky_solve_in_place(a, m, &mut self.rhs[..m])?;
        // Δc = W⁻¹ Jᵀ y.
        self.delta.fill(0.0);
        for i in 0..n {
            let y = [self.rhs[3 * i], self.rhs[3 * i + 1], self.rhs[3 * i + 2]];
            for k in row_range(offsets, self.rows[i]) {
                let c = comps[k] as usize;
                let j = values[k];
                let d = (f64::from(j.x) * y[0] + f64::from(j.y) * y[1] + f64::from(j.z) * y[2]) * inv_w[c];
                self.delta[c] += d as f32;
            }
        }
        if let Some(c) = self.delta.iter().position(|d| !d.is_finite()) {
            self.delta.fill(0.0);
            return Err(EvalError::NonFinite { what: "solved coefficient change", index: c });
        }
        // Residuals: where the change puts every row's landmark against its goal.
        self.residuals.clear();
        let mut report = SolveReport { rows: m, lambda, ..SolveReport::default() };
        for i in 0..n {
            let moved: Vec3 = self.jacobian(self.rows[i] as usize).map(|(c, j)| j * self.delta[c as usize]).sum();
            let r = self.row_rhs[i] - moved;
            self.residuals.push(r);
            if i < request.targets.len() {
                report.max_residual = report.max_residual.max(r.length());
            } else {
                report.max_pin_drift = report.max_pin_drift.max(r.length());
            }
        }
        report.exceeds_span = report.max_residual > self.settings.span_threshold;
        Ok(report)
    }

    fn check(&self, rest: &[Vec3], request: &SolveRequest<'_>) -> EvalResult<()> {
        if request.targets.len() > MAX_TARGETS {
            return Err(EvalError::limit("landmark targets", MAX_TARGETS, request.targets.len()));
        }
        if request.pins.len() > MAX_PINS {
            return Err(EvalError::limit("landmark pins", MAX_PINS, request.pins.len()));
        }
        if request.locked.len() > self.components {
            return Err(EvalError::Length { what: "locked flags", limit: self.components, got: request.locked.len() });
        }
        if request.regularisation.len() > self.components {
            return Err(EvalError::Length {
                what: "regularisation weights",
                limit: self.components,
                got: request.regularisation.len(),
            });
        }
        if let Some(i) = request.regularisation.iter().position(|w| !(w.is_finite() && *w > 0.0)) {
            return Err(EvalError::Range {
                what: "regularisation weight",
                index: i,
                rule: "must be finite and positive",
            });
        }
        let positive = |x: f64| x > 0.0 && x.is_finite();
        if !positive(self.settings.lambda) || !positive(self.settings.pin_weight) {
            return Err(EvalError::Range {
                what: "solver setting",
                index: 0,
                rule: "lambda and pin weight must be positive",
            });
        }
        for (i, g) in request.targets.iter().chain(request.pins).enumerate() {
            if !g.position.is_finite() {
                return Err(EvalError::NonFinite { what: "landmark goal", index: i });
            }
            if self.landmark_position(g.landmark as usize, rest).is_none() {
                return Err(EvalError::Missing { what: "landmark", index: i });
            }
        }
        for (i, t) in request.targets.iter().enumerate() {
            if request.targets[..i].iter().chain(request.pins).any(|o| o.landmark == t.landmark) {
                return Err(EvalError::Range { what: "landmark goal", index: i, rule: "landmark constrained twice" });
            }
        }
        Ok(())
    }

    /// The coefficient change of the last successful solve, one entry per delta component (0 for
    /// locked, non-shape and unaffected components).
    #[must_use]
    pub fn delta(&self) -> &[f32] {
        &self.delta
    }

    /// Adds the last solve's change to `shape` (shape weights per delta component; it must have one
    /// entry per component). Allocation-free.
    pub fn apply(&self, shape: &mut [f32]) -> EvalResult<()> {
        if shape.len() != self.components {
            return Err(EvalError::Length { what: "shape weights", limit: self.components, got: shape.len() });
        }
        for (s, d) in shape.iter_mut().zip(&self.delta) {
            *s += d;
        }
        Ok(())
    }

    /// `[LM-02]`: the part of the last solve's targets the model could not reach, as a sculpt layer
    /// on `rest` (the rest shape the solve ran on, with the solved change applied or not: only
    /// positions near the landmarks are read). Each target's residual is spread over the vertices
    /// within `radius` of its landmark with the falloff `(1 − (d/radius)²)²`, scaled so the landmark
    /// itself moves by exactly its residual; the falloff is multiplied by `1 − falloff` of every
    /// pinned landmark so that pins stay put. Vertices at one position get one displacement (seam
    /// safe). A target whose residual is negligible or that sits on a pin contributes nothing.
    pub fn residual_layer(&self, rest: &[Vec3], request: &SolveRequest<'_>, radius: f32) -> EvalResult<SculptLayer> {
        if !(radius.is_finite() && radius > 0.0) {
            return Err(EvalError::Range { what: "residual radius", index: 0, rule: "must be finite and positive" });
        }
        if self.residuals.len() != request.targets.len() + request.pins.len() {
            return Err(EvalError::Range {
                what: "residual layer",
                index: 0,
                rule: "request differs from the last solve",
            });
        }
        let pins: Vec<Vec3> =
            request.pins.iter().filter_map(|g| self.landmark_position(g.landmark as usize, rest)).collect();
        let falloff = |p: Vec3, centre: Vec3| {
            let t = p.distance(centre) / radius;
            if t >= 1.0 { 0.0 } else { (1.0 - t * t) * (1.0 - t * t) }
        };
        let mask = |p: Vec3| pins.iter().fold(1.0, |m, &q| m * (1.0 - falloff(p, q)));
        let mut sum = vec![Vec3::ZERO; rest.len()];
        for (t, goal) in request.targets.iter().enumerate() {
            let r = self.residuals[t];
            let (Some(centre), Some(corners), Some(weights)) = (
                self.landmark_position(goal.landmark as usize, rest),
                self.corners.get(goal.landmark as usize),
                self.weights.get(goal.landmark as usize),
            ) else {
                continue;
            };
            let at_landmark: f32 = corners
                .iter()
                .zip(weights)
                .map(|(&v, &b)| rest.get(v as usize).map_or(0.0, |&p| b * falloff(p, centre) * mask(p)))
                .sum();
            if r.length() < 1e-9 || at_landmark < 1e-6 {
                continue;
            }
            let scale = at_landmark.recip();
            for (out, &p) in sum.iter_mut().zip(rest) {
                let f = falloff(p, centre);
                if f > 0.0 {
                    *out += r * (f * mask(p) * scale);
                }
            }
        }
        let pairs: Vec<(u32, Vec3)> =
            sum.iter().enumerate().filter(|(_, d)| **d != Vec3::ZERO).map(|(v, &d)| (v as u32, d)).collect();
        SculptLayer::from_pairs(&pairs, rest.len())
    }
}

/// The entries of landmark `l` in the CSR Jacobian (empty when out of range).
fn row_range(offsets: &[u32], l: u32) -> std::ops::Range<usize> {
    match (offsets.get(l as usize), offsets.get(l as usize + 1)) {
        (Some(&a), Some(&b)) if a <= b => a as usize..b as usize,
        _ => 0..0,
    }
}

/// `Σ_c J_p,c ⊗ J_q,c / ω_c²` over the components both landmarks share (both lists sorted).
fn block_product(
    p: std::ops::Range<usize>,
    q: std::ops::Range<usize>,
    components: &[u32],
    values: &[Vec3],
    inv_w: &[f64],
) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    let (mut i, mut j) = (p.start, q.start);
    while i < p.end && j < q.end {
        let (ci, cj) = (components[i], components[j]);
        match ci.cmp(&cj) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                let w = inv_w[ci as usize];
                if w != 0.0 {
                    let (a, b) = (values[i].to_array().map(f64::from), values[j].to_array().map(f64::from));
                    for r in 0..3 {
                        for s in 0..3 {
                            out[r][s] += a[r] * b[s] * w;
                        }
                    }
                }
                i += 1;
                j += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::eval::evaluator::{EvalInput, Evaluator};
    use crate::eval::tests::face_patch;
    use humanart_math::solve::{Cholesky, DenseMatrix};

    fn goal(landmark: u32, position: Vec3) -> LandmarkGoal {
        LandmarkGoal { landmark, position }
    }

    /// `[AC-WP09-1]`: a reachable drag lands within tolerance, pins stay, locks hold.
    #[test]
    fn ac_wp09_1_targets_reached_pins_and_locks_honoured() {
        let t = Arc::new(face_patch(24, 18));
        let mut e = Evaluator::new(t.clone());
        let mut solver = LandmarkSolver::new(&t, SolveSettings::default());
        let rest = e.rest_positions().to_vec();
        // Targets made reachable: positions of a known coefficient change.
        let mut truth = vec![0.0; t.deltas().len()];
        for k in [1, 4, 7] {
            truth[t.deltas().find(&format!("id.{k}")).unwrap() as usize] = 0.6;
        }
        let truth_shape = t.deformed_positions(&truth);
        let targets: Vec<LandmarkGoal> = [3_u32, 10, 17, 24]
            .iter()
            .map(|&l| goal(l, solver.landmark_position(l as usize, &truth_shape).unwrap()))
            .collect();
        let report = solver.solve(&rest, &SolveRequest { targets: &targets, ..SolveRequest::default() }).unwrap();
        assert!(report.max_residual < 1e-4, "{report:?}");
        assert!(!report.exceeds_span);
        // Apply and evaluate: the landmarks really are there.
        let mut shape = vec![0.0; t.deltas().len()];
        solver.apply(&mut shape).unwrap();
        e.evaluate(&EvalInput { shape: &shape, ..EvalInput::default() }).unwrap();
        for g in &targets {
            let p = solver.landmark_position(g.landmark as usize, e.rest_positions()).unwrap();
            assert!(p.distance(g.position) < 1e-4);
        }
        // With pins: two other landmarks keep their places while two targets move. The targets come
        // from a change in the null space of the pins' Jacobian, so an exact answer exists.
        let pin_ids = [0_u32, 30];
        let pins: Vec<LandmarkGoal> =
            pin_ids.iter().map(|&l| goal(l, solver.landmark_position(l as usize, &rest).unwrap())).collect();
        let c = t.deltas().len();
        let mut jp = DenseMatrix::zeros(6, c);
        for (i, &l) in pin_ids.iter().enumerate() {
            for (k, j) in solver.jacobian(l as usize) {
                for r in 0..3 {
                    jp[(3 * i + r, k as usize)] = f64::from(j[r]);
                }
            }
        }
        let wanted: Vec<f64> = (0..c).map(|k| f64::from(truth[k]) + if k % 2 == 0 { 0.3 } else { -0.2 }).collect();
        let mut gram = DenseMatrix::zeros(6, 6);
        for r in 0..6 {
            for s in 0..6 {
                gram[(r, s)] = (0..c).map(|k| jp[(r, k)] * jp[(s, k)]).sum();
            }
        }
        let jw: Vec<f64> = (0..6).map(|r| (0..c).map(|k| jp[(r, k)] * wanted[k]).sum()).collect();
        let y = Cholesky::new(&gram).unwrap().solve(&jw).unwrap();
        let projected: Vec<f32> =
            (0..c).map(|k| (wanted[k] - (0..6).map(|r| jp[(r, k)] * y[r]).sum::<f64>()) as f32).collect();
        let shifted = t.deformed_positions(&projected);
        for g in &pins {
            assert!(solver.landmark_position(g.landmark as usize, &shifted).unwrap().distance(g.position) < 1e-6);
        }
        let moved: Vec<LandmarkGoal> =
            [10_u32, 24].iter().map(|&l| goal(l, solver.landmark_position(l as usize, &shifted).unwrap())).collect();
        let report =
            solver.solve(&rest, &SolveRequest { targets: &moved, pins: &pins, ..SolveRequest::default() }).unwrap();
        assert!(report.max_pin_drift < 1e-5, "{report:?}");
        assert!(report.max_residual < 1e-4, "{report:?}");
        // Locks: locked components do not change, and nothing outside the shape kinds ever does.
        let mut locked = vec![false; t.deltas().len()];
        for k in 0..9 {
            locked[t.deltas().find(&format!("id.{k}")).unwrap() as usize] = true;
        }
        solver.solve(&rest, &SolveRequest { targets: &targets, locked: &locked, ..SolveRequest::default() }).unwrap();
        for (c, comp) in t.deltas().components().iter().enumerate() {
            if locked[c] || !is_shape_kind(comp.kind) {
                assert_eq!(solver.delta()[c], 0.0, "{}", comp.name);
            }
        }
        assert!(solver.delta().iter().any(|&d| d != 0.0));
    }

    /// The dual form equals the primal normal equations `(JᵀRJ + λW) Δc = JᵀR b`.
    #[test]
    fn lm_01_dual_form_matches_primal_normal_equations() {
        let t = face_patch(12, 6);
        let settings = SolveSettings { lambda: 1e-3, pin_weight: 50.0, ..SolveSettings::default() };
        let mut solver = LandmarkSolver::new(&t, settings);
        let rest = t.mesh().positions().to_vec();
        let targets = [goal(2, Vec3::new(0.3, 0.2, 0.05)), goal(9, Vec3::new(0.6, 0.5, -0.02))];
        let pins = [goal(5, solver.landmark_position(5, &rest).unwrap())];
        let reg: Vec<f32> = (0..t.deltas().len()).map(|c| 1.0 + c as f32 * 0.1).collect();
        let req = SolveRequest { targets: &targets, pins: &pins, regularisation: &reg, ..SolveRequest::default() };
        let lambda = solver.solve(&rest, &req).unwrap().lambda;
        // Primal, dense, over all components (non-shape ones get a huge regularisation instead of removal).
        let c = t.deltas().len();
        let mut a = DenseMatrix::zeros(c, c);
        let mut b = vec![0.0; c];
        for (g, rho) in targets.iter().map(|g| (g, 1.0)).chain(pins.iter().map(|g| (g, settings.pin_weight))) {
            let p = solver.landmark_position(g.landmark as usize, &rest).unwrap();
            let rhs = (g.position - p).to_array().map(f64::from);
            let row: Vec<(u32, Vec3)> = solver.jacobian(g.landmark as usize).collect();
            for &(ci, ji) in &row {
                let ji = ji.to_array().map(f64::from);
                b[ci as usize] += rho * (0..3).map(|r| ji[r] * rhs[r]).sum::<f64>();
                for &(cj, jj) in &row {
                    let jj = jj.to_array().map(f64::from);
                    a[(ci as usize, cj as usize)] += rho * (0..3).map(|r| ji[r] * jj[r]).sum::<f64>();
                }
            }
        }
        for k in 0..c {
            let w = f64::from(reg[k]);
            a[(k, k)] += lambda * w * w * if is_shape_kind(t.deltas().components()[k].kind) { 1.0 } else { 1e12 };
        }
        let x = Cholesky::new(&a).unwrap().solve(&b).unwrap();
        for (k, (&want, &got)) in x.iter().zip(solver.delta()).enumerate() {
            assert!((want as f32 - got).abs() < 1e-4 * (1.0 + want.abs() as f32), "{k}: {want} vs {got}");
        }
    }

    #[test]
    fn lm_02_out_of_span_drag_reports_and_converts_to_a_layer() {
        let t = face_patch(16, 3);
        let mut solver = LandmarkSolver::new(&t, SolveSettings::default());
        let rest = t.mesh().positions().to_vec();
        // The identity components move mostly in z; a large x drag is outside their span.
        let p = solver.landmark_position(12, &rest).unwrap();
        let targets = [goal(12, p + Vec3::X * 0.05)];
        let pins = [goal(0, solver.landmark_position(0, &rest).unwrap())];
        let req = SolveRequest { targets: &targets, pins: &pins, ..SolveRequest::default() };
        let report = solver.solve(&rest, &req).unwrap();
        assert!(report.exceeds_span, "{report:?}");
        let mut shape = vec![0.0; t.deltas().len()];
        solver.apply(&mut shape).unwrap();
        let mut solved = t.deformed_positions(&shape);
        let layer = solver.residual_layer(&rest, &req, 0.2).unwrap();
        assert!(layer.seam_consistent(t.mesh()));
        layer.accumulate(1.0, &mut solved);
        let now = solver.landmark_position(12, &solved).unwrap();
        assert!(now.distance(targets[0].position) < 1e-3, "{now:?} vs {:?}", targets[0].position);
        let pinned = solver.landmark_position(0, &solved).unwrap();
        assert!(pinned.distance(pins[0].position) < 1e-3);
        assert!(solver.residual_layer(&rest, &SolveRequest::default(), 0.2).is_err());
        assert!(solver.residual_layer(&rest, &req, 0.0).is_err());
    }

    #[test]
    fn bad_requests_are_refused() {
        let t = face_patch(8, 2);
        let mut solver = LandmarkSolver::new(&t, SolveSettings::default());
        let rest = t.mesh().positions().to_vec();
        let ok = [goal(1, Vec3::ZERO)];
        let many = vec![goal(1, Vec3::ZERO); MAX_TARGETS + 1];
        let c = t.deltas().len();
        let cases: [SolveRequest<'_>; 6] = [
            SolveRequest { targets: &[goal(999, Vec3::ZERO)], ..SolveRequest::default() },
            SolveRequest { targets: &[goal(1, Vec3::splat(f32::NAN))], ..SolveRequest::default() },
            SolveRequest { targets: &many, ..SolveRequest::default() },
            SolveRequest { targets: &ok, pins: &ok, ..SolveRequest::default() },
            SolveRequest { targets: &ok, regularisation: &[0.0], ..SolveRequest::default() },
            SolveRequest { targets: &ok, locked: &vec![false; c + 1], ..SolveRequest::default() },
        ];
        for (i, r) in cases.iter().enumerate() {
            assert!(solver.solve(&rest, r).is_err(), "case {i}");
        }
        assert!(solver.apply(&mut [0.0]).is_err());
        let mut bad = LandmarkSolver::new(&t, SolveSettings { lambda: 0.0, ..SolveSettings::default() });
        assert!(bad.solve(&rest, &SolveRequest { targets: &ok, ..SolveRequest::default() }).is_err());
        assert_eq!(solver.settings().pin_weight, 1e4);
    }

    /// `[LM-03]` / `[AC-WP09-1]`: solve + apply latency for 8 simultaneous constraints (apply = the
    /// parameter change of `[LM-02]`), and the solve itself allocates nothing. The CPU
    /// re-accumulation of the rest mesh that follows is measured and reported separately (the
    /// editor does it on the GPU, WP-10).
    #[test]
    fn lm_03_latency_with_eight_constraints() {
        let t = Arc::new(face_patch(100, 100));
        let mut e = Evaluator::new(t.clone());
        let mut solver = LandmarkSolver::new(&t, SolveSettings::default());
        let rest = e.rest_positions().to_vec();
        let mut shape = vec![0.0_f32; t.deltas().len()];
        let landmarks = t.landmarks().len() as u32;
        let (mut solve_ms, mut apply_ms, mut rest_ms) = (Vec::new(), Vec::new(), Vec::new());
        for run in 0..21_u32 {
            let targets: Vec<LandmarkGoal> = (0..8_u32)
                .map(|k| {
                    let l = (k * 5 + run) % landmarks;
                    let p = solver.landmark_position(l as usize, e.rest_positions()).unwrap();
                    goal(l, p + Vec3::new(0.0, 0.0, 0.002 * (k as f32 - 3.5)))
                })
                .collect();
            let req = SolveRequest { targets: &targets, ..SolveRequest::default() };
            let start = std::time::Instant::now();
            let stats = humanart_core::alloc::count_in(|| {
                solver.solve(e.rest_positions(), &req).unwrap();
            });
            let solved = start.elapsed();
            solver.apply(&mut shape).unwrap();
            let applied = start.elapsed();
            e.update_rest(&shape, &[]).unwrap();
            let updated = start.elapsed();
            assert_eq!(stats.allocations, 0);
            if run > 0 {
                solve_ms.push(solved.as_secs_f64() * 1e3);
                apply_ms.push(applied.as_secs_f64() * 1e3);
                rest_ms.push(updated.saturating_sub(applied).as_secs_f64() * 1e3);
            }
        }
        let median = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let (s, a, r) = (median(&mut solve_ms), median(&mut apply_ms), median(&mut rest_ms));
        println!(
            "[LM-03] timing: {} vertices, {} dense components, 8 targets: solve {s:.3} ms, solve + apply {a:.3} ms; \
             CPU rest re-accumulation {r:.3} ms (medians of 20)",
            rest.len(),
            t.deltas().len()
        );
        assert!(a < 2.0, "solve + apply {a} ms");
    }
}
