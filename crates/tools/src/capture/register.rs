//! Non-rigid registration of the template surface onto a scan `[WP-25]`: landmark-guided
//! non-rigid ICP with a smoothness term, coarse to fine, with masks so only skin pulls.
//!
//! **Unknowns.** The displacement `u` of every node of a [`FitMesh`] (the template faces being
//! fitted, welded so UV seams cannot tear) from its start position `x₀` (the template after
//! [`super::align`]).
//!
//! **Energy** per level, minimised by one sparse SPD solve over all `3n` coordinates (IC(0)
//! preconditioned conjugate gradients, warm-started) per ICP iteration:
//!
//! `E(u) = Σᵢ wᵢ ‖x₀ᵢ + uᵢ − cᵢ‖²_P + β Σₗ ‖Bₗ(x₀ + u) − sₗ‖² + α ‖L u‖² + γ Σ₍ᵢⱼ₎ ‖uᵢ − uⱼ‖² + δ ‖u‖²`
//!
//! - data: `cᵢ` the closest point of the scan to the node's current position, searched only on
//!   triangles whose mask weight is positive and whose normal lies within
//!   [`Schedule::max_normal_angle`] of the node's, no farther than [`Schedule::max_distance`];
//!   `wᵢ` = node weight × triangle mask weight × a Cauchy robust factor `1 / (1 + (d/σ)²)`;
//!   `‖v‖²_P = vᵀ(n nᵀ + λ I)v` with `n` the scan triangle's normal: mostly point-to-plane, so
//!   the surface may slide along the scan (only the landmarks and the smoothness say where), with
//!   a small point-to-point share [`Schedule::point_weight`] that keeps it from drifting;
//! - landmarks: barycentric points `Bₗ` of the template pulled to annotated scan points `sₗ`;
//! - smoothness: the bi-Laplacian of the displacement (`L` the uniform umbrella operator: smooth
//!   and locally affine fields are cheap), plus a weak membrane term (`γ`) so regions without data
//!   (masked hair, clothing, holes) settle to the displacement around them instead of
//!   extrapolating it;
//! - `δ` a tiny Tikhonov term that keeps the system positive definite for nodes no term reaches.
//!
//! **Coarse to fine.** Each level states a smoothing length `ℓ` in metres; with `h` the mean
//! edge length, `α = 16 (ℓ / 2πh)⁴` and `γ = μ (ℓ / 2πh)²` make the fit a low-pass filter whose
//! cut-off wavelength is about `ℓ` independent of the mesh resolution (a regular grid's umbrella
//! operator is `−h²/4 ∇²`).
//!
//! **Landmarks fade** by [`Schedule::landmark_decay`] per level: they place the coarse levels, then
//! leave the fine levels to the data, so a pick a millimetre off the true point cannot pin the
//! surface into a crease.
//!
//! **Fold guard.** The bi-Laplacian leaves locally affine fields free, so a strong pull can
//! compress a region past zero area and flip triangles (the data term never sees orientation).
//! After the iterations of each level, every triangle whose orientation flipped against the start
//! raises the *stiffness* `sᵢ` of its nodes and their [`Schedule::fold_ring`]-ring neighbourhood by
//! [`Schedule::fold_stiffening`], and the level runs one more iteration, up to
//! [`Schedule::fold_passes`] times. A stiff node weighs its bi-Laplacian row by `sᵢ` and its edges
//! carry an extra membrane `κ (max(sᵢ, sⱼ) − 1)` ([`Schedule::fold_membrane`], in units of one fully
//! weighted data node), which limits the displacement gradient there. Stiffness persists into the
//! finer levels. The data weights are never touched: the surface there follows the scan at a
//! longer smoothing length instead of tearing. Remaining flips are reported, not hidden.
//!
//! **Result** ([`Fit`]): node positions, per-node fit error (distance to the nearest masked scan
//! point), the correspondences of the last iteration, a [`FitReport`] and [`FitFlags`] that mark a
//! bad fit against [`Thresholds`].

use humanart_math::Vec3;
use humanart_math::solve::SolveError;
use humanart_math::solve::pcg::{IncompleteCholesky, JacobiPreconditioner, PcgSettings, PcgWorkspace, pcg};
use humanart_math::solve::sparse::CsrMatrix;

use super::nearest::TriangleTree;
use super::scan::ScanMesh;
use super::{CaptureError, Result};

/// Most nodes of one fit.
pub const MAX_FIT_NODES: usize = 1 << 21;
/// Most landmarks of one fit.
pub const MAX_FIT_LANDMARKS: usize = 4096;

/// The surface being fitted: welded node positions and triangles.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FitMesh {
    /// Node positions (start of the fit).
    pub positions: Vec<Vec3>,
    /// Three node indices per triangle, counter-clockwise from outside.
    pub triangles: Vec<u32>,
}

impl FitMesh {
    /// Checks lengths, index ranges, finite positions and the limits.
    ///
    /// # Errors
    /// [`CaptureError::Invalid`] or [`CaptureError::Limit`] naming the problem.
    pub fn validate(&self) -> Result<()> {
        let n = self.positions.len();
        if n == 0 || n > MAX_FIT_NODES {
            return Err(CaptureError::limit("fit node count", MAX_FIT_NODES, n));
        }
        if !self.triangles.len().is_multiple_of(3) || self.triangles.iter().any(|&i| i as usize >= n) {
            return Err(CaptureError::invalid("fit triangles are malformed"));
        }
        if !self.positions.iter().all(|p| p.is_finite()) {
            return Err(CaptureError::invalid("a fit node position is not finite"));
        }
        Ok(())
    }

    /// Area-weighted unit normals of the nodes at `positions` (+Y where undefined).
    pub fn normals(&self, positions: &[Vec3]) -> Vec<Vec3> {
        let mut sum = vec![Vec3::ZERO; positions.len()];
        for t in self.triangles.as_chunks::<3>().0 {
            let [a, b, c] = t.map(|i| i as usize);
            let (Some(&pa), Some(&pb), Some(&pc)) = (positions.get(a), positions.get(b), positions.get(c)) else {
                continue;
            };
            let face = (pb - pa).cross(pc - pa);
            for i in [a, b, c] {
                if let Some(s) = sum.get_mut(i) {
                    *s += face;
                }
            }
        }
        sum.into_iter().map(|v| v.normalize_or(Vec3::Y)).collect()
    }

    /// The sorted, unique neighbours of every node (the edges of the triangles).
    pub fn neighbours(&self) -> Vec<Vec<u32>> {
        let mut adj = vec![Vec::new(); self.positions.len()];
        for t in self.triangles.as_chunks::<3>().0 {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                if a == b {
                    continue;
                }
                if let Some(l) = adj.get_mut(a as usize) {
                    l.push(b);
                }
                if let Some(l) = adj.get_mut(b as usize) {
                    l.push(a);
                }
            }
        }
        for l in &mut adj {
            l.sort_unstable();
            l.dedup();
        }
        adj
    }
}

/// A barycentric point of the fit mesh pulled to a scan point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitLandmark {
    /// Three nodes.
    pub nodes: [u32; 3],
    /// Their barycentric weights.
    pub weights: [f32; 3],
    /// The scan point.
    pub target: Vec3,
}

impl FitLandmark {
    /// The landmark's position for node positions `x`.
    pub fn position(&self, x: &[Vec3]) -> Vec3 {
        self.nodes
            .iter()
            .zip(self.weights)
            .map(|(&n, w)| x.get(n as usize).copied().unwrap_or(Vec3::ZERO) * w)
            .fold(Vec3::ZERO, |a, b| a + b)
    }
}

/// The scan as a target: the closest-point tree, triangle normals and the mask.
#[derive(Clone, Debug)]
pub struct TargetSurface {
    tree: TriangleTree,
    normals: Vec<Vec3>,
    weights: Vec<f32>,
    rim: Vec<bool>,
}

impl TargetSurface {
    /// The target of `scan` with one mask weight per triangle (0: never pulls, e.g. hair,
    /// clothing; 1: skin). The triangles on the rim of a hole ([`ScanMesh::boundary_triangles`])
    /// are found but never accepted as a correspondence: a node over a hole would otherwise be
    /// pulled onto the rim.
    ///
    /// # Errors
    /// [`CaptureError::Invalid`] when the weights do not match the triangle count or are not in
    /// `0..=1`; the errors of [`TriangleTree::build`].
    pub fn new(scan: &ScanMesh, weights: Vec<f32>) -> Result<Self> {
        if weights.len() != scan.triangle_count() || !weights.iter().all(|w| (0.0..=1.0).contains(w)) {
            return Err(CaptureError::invalid("the mask needs one weight in 0..=1 per scan triangle"));
        }
        let tree = TriangleTree::build(&scan.positions, &scan.indices)?;
        let normals = scan
            .indices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| scan.positions.get(i as usize).copied().unwrap_or(Vec3::ZERO));
                (b - a).cross(c - a).normalize_or(Vec3::ZERO)
            })
            .collect();
        let rim = scan.boundary_triangles();
        Ok(Self { tree, normals, weights, rim })
    }

    /// The unit normal of triangle `t` (zero when out of range or degenerate).
    pub fn normal(&self, t: u32) -> Vec3 {
        self.normals.get(t as usize).copied().unwrap_or(Vec3::ZERO)
    }

    /// The mask weight of triangle `t` (0 when out of range).
    pub fn weight(&self, t: u32) -> f32 {
        self.weights.get(t as usize).copied().unwrap_or(0.0)
    }

    /// The correspondence of a node at `p` with normal `normal`: the closest masked point within
    /// `max_distance` whose triangle faces within `min_cos` of `normal`, unless that triangle lies
    /// on the rim of a hole. Point, distance and triangle.
    pub fn closest(&self, p: Vec3, normal: Vec3, min_cos: f32, max_distance: f32) -> Option<(Vec3, f32, u32)> {
        let hit = self.tree.nearest(p, max_distance, |t| {
            self.weight(t) > 0.0 && self.normals.get(t as usize).is_some_and(|n| n.dot(normal) >= min_cos)
        })?;
        if self.rim.get(hit.triangle as usize).copied().unwrap_or(true) {
            return None;
        }
        Some((hit.point, hit.distance, hit.triangle))
    }

    /// The distance from `p` to the nearest masked point, any orientation, rims included, within
    /// `max_distance`.
    pub fn distance(&self, p: Vec3, max_distance: f32) -> Option<f32> {
        self.tree.nearest(p, max_distance, |t| self.weight(t) > 0.0).map(|h| h.distance)
    }
}

/// The coarse-to-fine schedule and the weights of the energy (module documentation).
#[derive(Clone, Debug, PartialEq)]
pub struct Schedule {
    /// The smoothing length of each level, metres, coarse to fine.
    pub smoothing_lengths: Vec<f32>,
    /// ICP iterations per level.
    pub iterations: u32,
    /// The farthest correspondence, metres.
    pub max_distance: f32,
    /// The largest angle between a node's normal and its scan triangle's, radians.
    pub max_normal_angle: f32,
    /// Landmark weight `β` (per landmark, in units of one fully weighted data node).
    pub landmark_weight: f32,
    /// The point-to-point share `λ` of the data term, `0..=1`: the rest pulls only along the scan
    /// normal (point-to-plane), so the surface can slide tangentially where the smoothness and
    /// the landmarks move it.
    pub point_weight: f32,
    /// The Cauchy scale `σ` of the robust data weight, metres.
    pub robust_scale: f32,
    /// The membrane factor `μ`.
    pub membrane: f32,
    /// Relative residual of each linear solve.
    pub solve_tolerance: f64,
    /// Iteration limit of each linear solve (reaching it is reported, not an error).
    pub solve_iterations: usize,
    /// Factor applied to the landmark weight at each level after the first (1: constant).
    pub landmark_decay: f32,
    /// Most fold-guard passes per level (0: no guard).
    pub fold_passes: u32,
    /// Factor by which a fold raises the stiffness of the nodes around it.
    pub fold_stiffening: f32,
    /// Rings of neighbours around a flipped triangle that are stiffened with it.
    pub fold_ring: u32,
    /// The membrane weight `κ` a stiffened edge gains per unit of stiffness above 1.
    pub fold_membrane: f32,
}

impl Default for Schedule {
    /// Five levels from 8 cm down to 5 mm, 4 iterations each, correspondences within 3 cm and
    /// 60°, landmarks worth 50 nodes at every level, a data term 10 % point-to-point and 90 %
    /// point-to-plane, σ = 5 mm, μ = 0.05; fold guard of up to 6 passes per level, stiffness ×4
    /// over 2 rings, κ = 0.25.
    fn default() -> Self {
        Self {
            smoothing_lengths: vec![0.08, 0.04, 0.02, 0.01, 0.005],
            iterations: 4,
            max_distance: 0.03,
            max_normal_angle: 60_f32.to_radians(),
            landmark_weight: 50.0,
            point_weight: 0.1,
            robust_scale: 0.005,
            membrane: 0.05,
            solve_tolerance: 1e-6,
            solve_iterations: 2000,
            landmark_decay: 1.0,
            fold_passes: 6,
            fold_stiffening: 4.0,
            fold_ring: 2,
            fold_membrane: 0.25,
        }
    }
}

/// Limits beyond which a fit is flagged as bad.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Thresholds {
    /// Largest RMS fit error of the matched weighted nodes, metres.
    pub rms_error: f32,
    /// Smallest fraction of weighted nodes with a correspondence.
    pub matched_fraction: f32,
    /// Largest RMS landmark residual, and largest distance of a landmark target from the masked
    /// scan, metres.
    pub landmark_rms: f32,
    /// Largest fraction of triangles whose orientation flipped.
    pub folded_fraction: f32,
    /// Largest edge stretch factor (and its inverse, the smallest).
    pub stretch: f32,
}

impl Default for Thresholds {
    /// 2 mm RMS, 60 % matched, 4 mm landmark RMS, 0.5 % folded triangles, stretch within 4×.
    fn default() -> Self {
        Self { rms_error: 0.002, matched_fraction: 0.6, landmark_rms: 0.004, folded_fraction: 0.005, stretch: 4.0 }
    }
}

/// Failure flags of a fit (a bit set).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FitFlags(pub u32);

impl FitFlags {
    /// The RMS fit error exceeds [`Thresholds::rms_error`].
    pub const HIGH_ERROR: Self = Self(1);
    /// Too few nodes found a correspondence.
    pub const FEW_CORRESPONDENCES: Self = Self(2);
    /// The landmarks are not met.
    pub const LANDMARK_MISMATCH: Self = Self(4);
    /// Triangles flipped.
    pub const FOLDED: Self = Self(8);
    /// Edges stretched or shrank beyond the limit.
    pub const DISTORTED: Self = Self(16);
    /// A linear solve hit its iteration limit.
    pub const SOLVER_LIMIT: Self = Self(32);

    /// Whether every flag of `other` is set.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether no flag is set.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The names of the set flags.
    pub fn names(self) -> Vec<&'static str> {
        [
            (Self::HIGH_ERROR, "high-error"),
            (Self::FEW_CORRESPONDENCES, "few-correspondences"),
            (Self::LANDMARK_MISMATCH, "landmark-mismatch"),
            (Self::FOLDED, "folded"),
            (Self::DISTORTED, "distorted"),
            (Self::SOLVER_LIMIT, "solver-limit"),
        ]
        .into_iter()
        .filter(|(f, _)| self.contains(*f))
        .map(|(_, n)| n)
        .collect()
    }
}

/// Measured quality of a fit.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FitReport {
    /// RMS fit error of the weighted nodes matched in the last iteration, metres (nodes over a
    /// hole or a masked region have no match and do not count; see `matched_fraction`).
    pub rms_error: f32,
    /// Largest fit error of a matched weighted node, metres.
    pub max_error: f32,
    /// Fraction of weighted nodes with a correspondence in the last iteration.
    pub matched_fraction: f32,
    /// RMS landmark residual, metres (0 without landmarks).
    pub landmark_rms: f32,
    /// The largest distance of a landmark target from the masked scan surface, metres: an
    /// annotation that is not on the scan.
    pub landmark_off_surface: f32,
    /// Fraction of triangles whose orientation flipped against the start.
    pub folded_fraction: f32,
    /// Largest edge length ratio fitted / start.
    pub max_stretch: f32,
    /// Smallest edge length ratio fitted / start.
    pub min_stretch: f32,
    /// ICP iterations run.
    pub iterations: u32,
    /// Linear solves that hit their iteration limit.
    pub solver_limits: u32,
    /// Fold-guard passes run (each one extra iteration).
    pub fold_passes: u32,
    /// Nodes whose stiffness the fold guard raised.
    pub stiffened_nodes: u32,
    /// The largest node stiffness at the end (1: no fold was guarded).
    pub max_stiffness: f32,
}

/// The result of [`register`].
#[derive(Clone, Debug, PartialEq)]
pub struct Fit {
    /// Fitted node positions.
    pub positions: Vec<Vec3>,
    /// Per node: the distance to the nearest masked scan point, any orientation, rims included,
    /// within 10 ×
    /// [`Schedule::max_distance`] (`f32::INFINITY` when none).
    pub error: Vec<f32>,
    /// Per node: whether it had a correspondence in the last iteration.
    pub matched: Vec<bool>,
    /// The quality numbers.
    pub report: FitReport,
    /// What is wrong with the fit (empty: nothing).
    pub flags: FitFlags,
}

/// The smoothness part of the system before weighting: the bi-Laplacian `LᵀL` as triplets tagged
/// with the row of `L` (node) they come from, and the edges of the membrane `K`.
struct Smoothness {
    bilaplacian: Vec<(u32, u32, u32, f64)>,
    edges: Vec<(u32, u32)>,
    mean_edge: f64,
}

fn smoothness(mesh: &FitMesh, adj: &[Vec<u32>]) -> Smoothness {
    let mut bilaplacian = Vec::new();
    let mut edges = Vec::new();
    let mut edge_sum = 0.0_f64;
    let mut row: Vec<(u32, f64)> = Vec::new();
    for (i, nb) in adj.iter().enumerate() {
        if nb.is_empty() {
            continue;
        }
        let node = i as u32;
        let inv = 1.0 / nb.len() as f64;
        row.clear();
        row.push((node, 1.0));
        row.extend(nb.iter().map(|&j| (j, -inv)));
        for &(a, va) in &row {
            for &(b, vb) in &row {
                bilaplacian.push((node, a, b, va * vb));
            }
        }
        for &j in nb {
            if j > node {
                let (Some(p), Some(q)) = (mesh.positions.get(i), mesh.positions.get(j as usize)) else { continue };
                edge_sum += f64::from(p.distance(*q));
                edges.push((node, j));
            }
        }
    }
    let mean_edge = if edges.is_empty() { 1.0 } else { edge_sum / edges.len() as f64 };
    Smoothness { bilaplacian, edges, mean_edge }
}

/// Fits `mesh` to `target`: see the module documentation. `node_weights` (one per node, `0..=1`)
/// says how much each node is pulled by the data (0 for eyes, mouth interior and other parts the
/// scan does not show).
///
/// # Errors
/// [`CaptureError::Invalid`] / [`CaptureError::Limit`] for malformed input (weights, landmark
/// nodes, an empty schedule); [`CaptureError::Solve`] when a linear system breaks down.
pub fn register(
    mesh: &FitMesh,
    node_weights: &[f32],
    landmarks: &[FitLandmark],
    target: &TargetSurface,
    schedule: &Schedule,
    thresholds: &Thresholds,
) -> Result<Fit> {
    mesh.validate()?;
    let n = mesh.positions.len();
    if node_weights.len() != n || !node_weights.iter().all(|w| (0.0..=1.0).contains(w)) {
        return Err(CaptureError::invalid("the node weights need one value in 0..=1 per node"));
    }
    if landmarks.len() > MAX_FIT_LANDMARKS {
        return Err(CaptureError::limit("fit landmark count", MAX_FIT_LANDMARKS, landmarks.len()));
    }
    if landmarks.iter().any(|l| l.nodes.iter().any(|&v| v as usize >= n) || !l.target.is_finite()) {
        return Err(CaptureError::invalid("a fit landmark names a missing node or a non-finite target"));
    }
    if schedule.smoothing_lengths.is_empty()
        || schedule.smoothing_lengths.iter().any(|l| !(l.is_finite() && *l > 0.0))
        || !(schedule.max_distance.is_finite() && schedule.max_distance > 0.0)
        || !(schedule.robust_scale.is_finite() && schedule.robust_scale > 0.0)
    {
        return Err(CaptureError::invalid("the schedule needs positive smoothing lengths and distances"));
    }
    let adj = mesh.neighbours();
    let smooth = smoothness(mesh, &adj);
    let mut state = State {
        u: vec![0.0_f64; 3 * n],
        x: mesh.positions.clone(),
        matched: vec![false; n],
        workspace: PcgWorkspace::new(3 * n),
        weights: vec![0.0_f64; n],
        closest: vec![Vec3::ZERO; n],
        planes: vec![Vec3::ZERO; n],
        rhs: vec![0.0_f64; 3 * n],
        triplets: Vec::new(),
        stiffness: vec![1.0_f64; n],
        report: FitReport::default(),
    };
    let mut beta = f64::from(schedule.landmark_weight.max(0.0));
    let decay = f64::from(schedule.landmark_decay.clamp(0.0, 1.0));
    let mut marked = vec![false; n];
    for (level, &length) in schedule.smoothing_lengths.iter().enumerate() {
        if level > 0 {
            beta *= decay;
        }
        let ratio = f64::from(length) / (std::f64::consts::TAU * smooth.mean_edge);
        let weights = Weights {
            alpha: 16.0 * ratio.powi(4),
            gamma: f64::from(schedule.membrane.max(0.0)) * ratio * ratio,
            kappa: f64::from(schedule.fold_membrane.max(0.0)),
            beta,
        };
        for _ in 0..schedule.iterations {
            state.iterate(mesh, node_weights, landmarks, target, schedule, &smooth, &weights)?;
        }
        for _ in 0..schedule.fold_passes {
            let flipped = flipped_triangles(mesh, &state.x);
            if flipped.is_empty() {
                break;
            }
            mark_rings(&mesh.triangles, &flipped, &adj, schedule.fold_ring, &mut marked);
            let factor = f64::from(schedule.fold_stiffening.max(1.0));
            for (s, m) in state.stiffness.iter_mut().zip(&marked) {
                if *m {
                    *s *= factor;
                }
            }
            state.report.fold_passes += 1;
            state.iterate(mesh, node_weights, landmarks, target, schedule, &smooth, &weights)?;
        }
    }
    state.report.stiffened_nodes = state.stiffness.iter().filter(|&&s| s > 1.0).count() as u32;
    state.report.max_stiffness = state.stiffness.iter().copied().fold(1.0, f64::max) as f32;
    let State { x, matched, report, .. } = state;
    Ok(finish(mesh, node_weights, landmarks, target, schedule, thresholds, x, matched, report))
}

/// The weights of one level.
struct Weights {
    alpha: f64,
    gamma: f64,
    kappa: f64,
    beta: f64,
}

/// The working state of [`register`].
struct State {
    /// Unknowns interleaved per node: `u[3i + k]` is coordinate `k` of node `i`.
    u: Vec<f64>,
    x: Vec<Vec3>,
    matched: Vec<bool>,
    workspace: PcgWorkspace,
    weights: Vec<f64>,
    closest: Vec<Vec3>,
    planes: Vec<Vec3>,
    rhs: Vec<f64>,
    triplets: Vec<(usize, usize, f64)>,
    stiffness: Vec<f64>,
    report: FitReport,
}

impl State {
    /// One ICP iteration: correspondences at the current positions, one solve.
    #[allow(clippy::too_many_arguments)]
    fn iterate(
        &mut self,
        mesh: &FitMesh,
        node_weights: &[f32],
        landmarks: &[FitLandmark],
        target: &TargetSurface,
        schedule: &Schedule,
        smooth: &Smoothness,
        w: &Weights,
    ) -> Result<()> {
        let n = mesh.positions.len();
        let x0 = &mesh.positions;
        let lambda = f64::from(schedule.point_weight.clamp(0.0, 1.0));
        let min_cos = schedule.max_normal_angle.cos();
        self.report.iterations += 1;
        // Correspondences at the current positions.
        let normals = mesh.normals(&self.x);
        for i in 0..n {
            self.weights[i] = 0.0;
            self.matched[i] = false;
            let w_node = f64::from(node_weights[i]);
            if w_node <= 0.0 {
                continue;
            }
            if let Some((c, d, t)) = target.closest(self.x[i], normals[i], min_cos, schedule.max_distance) {
                let r = f64::from(d / schedule.robust_scale);
                self.weights[i] = w_node * f64::from(target.weight(t)) / (1.0 + r * r);
                self.closest[i] = c;
                self.planes[i] = target.normal(t);
                self.matched[i] = true;
            }
        }
        // The system matrix: smoothness and landmarks per coordinate, the data term as a 3x3
        // block per node, w (n n^T + lambda I).
        let stiffness = &self.stiffness;
        let stiff = |i: u32| stiffness.get(i as usize).copied().unwrap_or(1.0);
        self.triplets.clear();
        for k in 0..3 {
            let at = |i: u32| 3 * i as usize + k;
            for &(row, a, b, v) in &smooth.bilaplacian {
                self.triplets.push((at(a), at(b), w.alpha * stiff(row) * v));
            }
            for &(i, j) in &smooth.edges {
                let m = stiff(i).max(stiff(j));
                let e = w.gamma * m + w.kappa * (m - 1.0);
                self.triplets.extend([(at(i), at(i), e), (at(j), at(j), e), (at(i), at(j), -e), (at(j), at(i), -e)]);
            }
            for l in landmarks {
                for (&a, &wa) in l.nodes.iter().zip(&l.weights) {
                    for (&b, &wb) in l.nodes.iter().zip(&l.weights) {
                        self.triplets.push((at(a), at(b), w.beta * f64::from(wa) * f64::from(wb)));
                    }
                }
            }
        }
        let delta = 1e-8 * (1.0 + w.alpha);
        self.rhs.fill(0.0);
        for (i, &p0) in x0.iter().enumerate() {
            let wi = self.weights[i];
            let nn = self.planes[i].to_array().map(f64::from);
            let gap = (self.closest[i] - p0).to_array().map(f64::from);
            for r in 0..3 {
                for c in 0..3 {
                    let block = wi * (nn[r] * nn[c] + if r == c { lambda } else { 0.0 });
                    let diagonal = if r == c { delta } else { 0.0 };
                    self.triplets.push((3 * i + r, 3 * i + c, block + diagonal));
                    self.rhs[3 * i + r] += block * gap[c];
                }
            }
        }
        for l in landmarks {
            let residual = (l.target - l.position(x0)).to_array().map(f64::from);
            for (&v, &wv) in l.nodes.iter().zip(&l.weights) {
                for (k, r) in residual.iter().enumerate() {
                    self.rhs[3 * v as usize + k] += w.beta * f64::from(wv) * r;
                }
            }
        }
        let a = CsrMatrix::from_triplets(3 * n, 3 * n, &self.triplets)?;
        let settings = PcgSettings { tolerance: schedule.solve_tolerance, max_iterations: schedule.solve_iterations };
        let solved = if let Ok(m) = IncompleteCholesky::new(&a).or_else(|_| IncompleteCholesky::with_shift(&a, 1e-3)) {
            pcg(&a, &self.rhs, &mut self.u, &m, &settings, &mut self.workspace)
        } else {
            let m = JacobiPreconditioner::new(&a)?;
            pcg(&a, &self.rhs, &mut self.u, &m, &settings, &mut self.workspace)
        };
        match solved {
            Ok(_) => {}
            Err(SolveError::NotConverged { .. }) => self.report.solver_limits += 1,
            Err(e) => return Err(e.into()),
        }
        let u = &self.u;
        for (i, x) in self.x.iter_mut().enumerate() {
            *x = x0[i] + Vec3::new(u[3 * i] as f32, u[3 * i + 1] as f32, u[3 * i + 2] as f32);
        }
        Ok(())
    }
}

/// The triangles of `mesh` whose orientation at `x` flipped against the start (degenerate start
/// triangles excluded).
pub fn flipped_triangles(mesh: &FitMesh, x: &[Vec3]) -> Vec<u32> {
    let mut out = Vec::new();
    for (t, tri) in mesh.triangles.as_chunks::<3>().0.iter().enumerate() {
        let [a, b, c] = tri.map(|i| i as usize);
        let (Some(&p0), Some(&p1), Some(&p2)) = (mesh.positions.get(a), mesh.positions.get(b), mesh.positions.get(c))
        else {
            continue;
        };
        let (Some(&q0), Some(&q1), Some(&q2)) = (x.get(a), x.get(b), x.get(c)) else { continue };
        let before = (p1 - p0).cross(p2 - p0);
        if before.length_squared() > 0.0 && before.dot((q1 - q0).cross(q2 - q0)) <= 0.0 {
            out.push(t as u32);
        }
    }
    out
}

/// Sets `marked` to the nodes of the `flipped` triangles grown by `rings` rings of neighbours.
fn mark_rings(triangles: &[u32], flipped: &[u32], adj: &[Vec<u32>], rings: u32, marked: &mut [bool]) {
    marked.fill(false);
    let mut front: Vec<u32> = Vec::new();
    for &t in flipped {
        for k in 0..3 {
            if let Some(&v) = triangles.get(3 * t as usize + k)
                && let Some(m) = marked.get_mut(v as usize)
                && !*m
            {
                *m = true;
                front.push(v);
            }
        }
    }
    let mut next = Vec::new();
    for _ in 0..rings {
        next.clear();
        for &v in &front {
            for &j in adj.get(v as usize).map_or(&[][..], Vec::as_slice) {
                if let Some(m) = marked.get_mut(j as usize)
                    && !*m
                {
                    *m = true;
                    next.push(j);
                }
            }
        }
        std::mem::swap(&mut front, &mut next);
    }
}

/// Measures the fit and sets the flags.
fn finish(
    mesh: &FitMesh,
    node_weights: &[f32],
    landmarks: &[FitLandmark],
    target: &TargetSurface,
    schedule: &Schedule,
    thresholds: &Thresholds,
    x: Vec<Vec3>,
    matched: Vec<bool>,
    mut report: FitReport,
) -> Fit {
    let reach = schedule.max_distance * 10.0;
    let error: Vec<f32> = x.iter().map(|&p| target.distance(p, reach).unwrap_or(f32::INFINITY)).collect();
    let (mut sum2, mut max, mut count, mut weighted, mut hits) = (0.0_f64, 0.0_f32, 0_usize, 0_usize, 0_usize);
    for i in 0..x.len() {
        if node_weights[i] <= 0.0 {
            continue;
        }
        weighted += 1;
        if matched[i] {
            hits += 1;
            if error[i].is_finite() {
                sum2 += f64::from(error[i]) * f64::from(error[i]);
                max = max.max(error[i]);
                count += 1;
            }
        }
    }
    report.rms_error = if count > 0 { (sum2 / count as f64).sqrt() as f32 } else { f32::INFINITY };
    report.max_error = max;
    report.matched_fraction = if weighted > 0 { hits as f32 / weighted as f32 } else { 0.0 };
    if !landmarks.is_empty() {
        let s: f32 = landmarks.iter().map(|l| l.position(&x).distance_squared(l.target)).sum();
        report.landmark_rms = (s / landmarks.len() as f32).sqrt();
        report.landmark_off_surface =
            landmarks.iter().map(|l| target.distance(l.target, reach).unwrap_or(f32::INFINITY)).fold(0.0, f32::max);
    }
    let (mut folded, mut tris) = (0_usize, 0_usize);
    let (mut lo, mut hi) = (f32::MAX, 0.0_f32);
    for t in mesh.triangles.as_chunks::<3>().0 {
        let [a, b, c] = t.map(|i| i as usize);
        let (Some(&p0), Some(&p1), Some(&p2)) = (mesh.positions.get(a), mesh.positions.get(b), mesh.positions.get(c))
        else {
            continue;
        };
        let (Some(&q0), Some(&q1), Some(&q2)) = (x.get(a), x.get(b), x.get(c)) else { continue };
        let before = (p1 - p0).cross(p2 - p0);
        let after = (q1 - q0).cross(q2 - q0);
        if before.length_squared() == 0.0 {
            continue;
        }
        tris += 1;
        if before.dot(after) <= 0.0 {
            folded += 1;
        }
        for (e0, e1) in
            [(p0.distance(p1), q0.distance(q1)), (p1.distance(p2), q1.distance(q2)), (p2.distance(p0), q2.distance(q0))]
        {
            if e0 > 0.0 {
                let r = e1 / e0;
                lo = lo.min(r);
                hi = hi.max(r);
            }
        }
    }
    report.folded_fraction = if tris > 0 { folded as f32 / tris as f32 } else { 0.0 };
    report.max_stretch = hi;
    report.min_stretch = if lo == f32::MAX { 1.0 } else { lo };
    let mut flags = 0;
    // NaN or infinite errors count as failures.
    if report.rms_error.partial_cmp(&thresholds.rms_error) != Some(std::cmp::Ordering::Less) {
        flags |= FitFlags::HIGH_ERROR.0;
    }
    if report.matched_fraction < thresholds.matched_fraction {
        flags |= FitFlags::FEW_CORRESPONDENCES.0;
    }
    let off = report.landmark_off_surface.partial_cmp(&thresholds.landmark_rms);
    if report.landmark_rms > thresholds.landmark_rms
        || !matches!(off, Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal))
    {
        flags |= FitFlags::LANDMARK_MISMATCH.0;
    }
    if report.folded_fraction > thresholds.folded_fraction {
        flags |= FitFlags::FOLDED.0;
    }
    if report.max_stretch > thresholds.stretch || report.min_stretch < 1.0 / thresholds.stretch {
        flags |= FitFlags::DISTORTED.0;
    }
    if report.solver_limits > 0 {
        flags |= FitFlags::SOLVER_LIMIT.0;
    }
    Fit { positions: x, error, matched, report, flags: FitFlags(flags) }
}

#[cfg(test)]
mod tests;
