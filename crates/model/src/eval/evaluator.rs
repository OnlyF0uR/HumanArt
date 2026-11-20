//! The CPU reference evaluator: steps 1–5 of section 6.2 `[EVAL-01]`, plus the strain of step 6.
//!
//! [`Evaluator::new`] does all set-up work once (triangulation, adjacency, the dual-quaternion vertex
//! map, every output and scratch buffer); [`Evaluator::evaluate`] then runs without heap allocation
//! `[PERF-R1]` and writes SoA buffers:
//!
//! 1. **Rest shape** `P_rest = P_template + Σ w_c·Δ_c + Σ a_l·ΔLAYER_l` over the *shape* components
//!    (kinds identity-global, identity-regional, semantic, asymmetry: [`is_shape_kind`]) and the
//!    sculpt layers. Evaluated incrementally `[EVAL-03]`: only components whose weight changed are
//!    subtracted and re-added with the difference; after [`FULL_REFRESH_INTERVAL`] such updates the
//!    buffer is re-accumulated from scratch to bound floating-point drift.
//! 2. **Expression** `P_expr = P_rest + Σ e_j·ΔEXP_j + Σ_k act_k·ΔCOMB_k`, the combination
//!    correctives activated by the product or minimum of their input weights `[EXP-02]` (their deltas
//!    are limited to their bounding region by the template's validation).
//! 3. **Pose correctives** `+ Σ_m act_m·ΔPOSE_m`, `act_m` from the driving joint's rotation
//!    relative to rest `[SKIN-02]`.
//! 4. **Skinning** (see [`super::skinning`]): joint bind positions from the joint regressor for the
//!    current shape weights `[MOD-07]`, posed frames, then LBS or DQS per vertex `[SKIN-01]`;
//!    then, when one is set ([`Evaluator::set_guard`]), the eyelid and lip penetration guard
//!    (see [`super::penetration`]) `[EVAL-08]`.
//! 5. **Normals and tangents** (see [`super::normals`]) `[EVAL-04]`.
//! 6. Edge strain against the rest shape, reduced per region (see [`super::strain`]) `[EVAL-06]`.
//!
//! Inputs ([`EvalInput`]) are plain arrays: shape weights indexed by delta component (entries of
//! non-shape components are ignored: expressions and correctives are driven by expression weights
//! and the pose), expression weights indexed by [`crate::expression::ExpressionSet::expressions`],
//! layer weights indexed by the layers added with [`Evaluator::add_layer`], and joint rotations
//! relative to rest. Shorter arrays mean zero weights and rest rotations; longer arrays, non-finite
//! values, weights beyond ±[`MAX_WEIGHT`] and zero-length rotations are refused before anything
//! changes.
//!
//! The GPU evaluation (WP-10) must reproduce [`Evaluator::positions`] within 1e-4 of the head height
//! and [`Evaluator::normals`] within 0.5° `[EVAL-01]`; [`super::parity`] measures that.

use std::sync::Arc;

use humanart_math::{F16, Quat, Vec3, Vec4};

use super::error::{EvalError, EvalResult};
use super::layer::SculptLayer;
use super::normals::{TriangleFrames, normal_corner_gather, own_corner_gather};
use super::penetration::{PenetrationGuard, PenetrationReport};
use super::skinning::{JointFrames, SkinningMode};
use super::strain::StrainField;
use crate::delta::{DeltaBases, DeltaKind};
use crate::template::Template;
use crate::topology::Csr;

/// The largest magnitude any weight may have (shape, expression or layer). It keeps every result
/// finite for any finite template.
pub const MAX_WEIGHT: f32 = 1.0e4;

/// How many incremental component updates the rest shape takes before it is re-accumulated from
/// scratch `[EVAL-03]`.
pub const FULL_REFRESH_INTERVAL: u32 = 256;

/// The most sculpt layers one evaluator holds.
pub const MAX_LAYERS: usize = 1024;

/// Whether components of `kind` are part of the rest shape (step 1): identity, semantic and
/// asymmetry components. Expressions and correctives are not.
#[must_use]
pub const fn is_shape_kind(kind: DeltaKind) -> bool {
    matches!(kind, DeltaKind::IdentityGlobal | DeltaKind::IdentityRegional | DeltaKind::Semantic | DeltaKind::Asymmetry)
}

/// The inputs of one evaluation.
#[derive(Clone, Copy, Debug, Default)]
pub struct EvalInput<'a> {
    /// Weight per delta component (only shape components are read).
    pub shape: &'a [f32],
    /// Weight per expression.
    pub expressions: &'a [f32],
    /// Weight per sculpt layer.
    pub layers: &'a [f32],
    /// Rotation of each joint relative to its rest orientation, in the joint's frame.
    pub pose: &'a [Quat],
    /// Which skinning to use.
    pub skinning: SkinningMode,
}

/// The CPU reference evaluator of one template.
#[derive(Clone, Debug)]
pub struct Evaluator {
    template: Arc<Template>,
    is_shape: Vec<bool>,
    /// Per delta component: none of its halves is subnormal (a cheaper decode applies).
    plain: Vec<bool>,
    triangles: Vec<u32>,
    normal_gather: Csr,
    own_corners: Csr,
    dq_vertex: Vec<bool>,
    parents: Vec<Option<u16>>,
    bind_rotations: Vec<Quat>,
    rest_world_positions: Vec<Vec3>,
    layers: Vec<SculptLayer>,
    applied_shape: Vec<f32>,
    applied_layers: Vec<f32>,
    incremental_updates: u32,
    rest_dirty: bool,
    rest: Vec<Vec3>,
    pre_skin: Vec<Vec3>,
    positions: Vec<Vec3>,
    pose: Vec<Quat>,
    frames: TriangleFrames,
    normals: Vec<Vec3>,
    tangents: Vec<Vec4>,
    combination_activations: Vec<f32>,
    pose_activations: Vec<f32>,
    joints: JointFrames,
    strain: StrainField,
    guard: Option<PenetrationGuard>,
    guard_report: PenetrationReport,
}

impl Evaluator {
    /// Sets up an evaluator for `template` and evaluates the neutral state (all weights 0, rest
    /// pose), so every output is valid from the start.
    #[must_use]
    pub fn new(template: Arc<Template>) -> Self {
        let t = &*template;
        let mesh = t.mesh();
        let v = mesh.vertex_count();
        let tri = mesh.triangulate();
        let topology = mesh.topology();
        let triangle_count = tri.len();
        let normal_gather = normal_corner_gather(mesh, &tri.indices);
        let own_corners = own_corner_gather(v, &tri.indices);
        let regions = t.regions();
        let dq_regions = &t.skin().dual_quaternion_regions;
        let dq_vertex =
            regions.vertex_regions.iter().map(|&r| dq_regions.iter().any(|&d| regions.is_within(r, d))).collect();
        let skeleton = t.skeleton();
        let rest_world = skeleton.rest_world();
        let joint_count = skeleton.joints.len();
        let strain = StrainField::new(mesh, &topology, regions);
        let components = t.deltas().len();
        let mut out = Self {
            is_shape: t.deltas().components().iter().map(|c| is_shape_kind(c.kind)).collect(),
            plain: (0..t.deltas().len()).map(|c| plain_halves(t.deltas().deltas(c).as_flattened())).collect(),
            triangles: tri.indices,
            normal_gather,
            own_corners,
            dq_vertex,
            parents: skeleton.joints.iter().map(|j| j.parent).collect(),
            bind_rotations: rest_world.iter().map(|w| w.rotation).collect(),
            rest_world_positions: rest_world.iter().map(|w| w.translation).collect(),
            layers: Vec::new(),
            applied_shape: vec![0.0; components],
            applied_layers: Vec::new(),
            incremental_updates: 0,
            rest_dirty: false,
            rest: mesh.positions().to_vec(),
            pre_skin: mesh.positions().to_vec(),
            positions: mesh.positions().to_vec(),
            pose: vec![Quat::IDENTITY; joint_count],
            frames: TriangleFrames::new(triangle_count),
            normals: vec![Vec3::Z; v],
            tangents: vec![Vec4::new(1.0, 0.0, 0.0, 1.0); v],
            combination_activations: vec![0.0; t.expressions().combinations.len()],
            pose_activations: vec![0.0; t.expressions().pose_correctives.len()],
            joints: JointFrames::new(joint_count),
            strain,
            guard: None,
            guard_report: PenetrationReport::default(),
            template,
        };
        // The neutral state cannot be refused: every input is empty.
        if out.evaluate(&EvalInput::default()).is_err() {
            out.positions.clone_from(&out.rest);
        }
        out
    }

    /// The template.
    #[must_use]
    pub fn template(&self) -> &Arc<Template> {
        &self.template
    }

    /// Adds a sculpt layer (weight 0 until an input gives it one) and returns its index.
    pub fn add_layer(&mut self, layer: SculptLayer) -> EvalResult<usize> {
        if self.layers.len() >= MAX_LAYERS {
            return Err(EvalError::limit("sculpt layer count", MAX_LAYERS, self.layers.len() + 1));
        }
        self.check_layer(&layer)?;
        self.layers.push(layer);
        self.applied_layers.push(0.0);
        Ok(self.layers.len() - 1)
    }

    /// Replaces layer `index`; its current weight carries over (the rest shape is updated by the
    /// difference).
    pub fn replace_layer(&mut self, index: usize, layer: SculptLayer) -> EvalResult<()> {
        self.check_layer(&layer)?;
        let (Some(slot), Some(&w)) = (self.layers.get_mut(index), self.applied_layers.get(index)) else {
            return Err(EvalError::Missing { what: "sculpt layer", index });
        };
        if w != 0.0 {
            slot.accumulate(-w, &mut self.rest);
            layer.accumulate(w, &mut self.rest);
            self.rest_dirty = true;
            self.incremental_updates += 1;
        }
        *slot = layer;
        Ok(())
    }

    /// The sculpt layers.
    #[must_use]
    pub fn layers(&self) -> &[SculptLayer] {
        &self.layers
    }

    fn check_layer(&self, layer: &SculptLayer) -> EvalResult<()> {
        match layer.indices().last() {
            Some(&v) if v as usize >= self.rest.len() => Err(EvalError::Missing { what: "vertex", index: v as usize }),
            _ => Ok(()),
        }
    }

    /// Checks `input` against the template without changing anything.
    pub fn check_input(&self, input: &EvalInput<'_>) -> EvalResult<()> {
        check_weights("shape weights", input.shape, self.applied_shape.len())?;
        check_weights("expression weights", input.expressions, self.template.expressions().expressions.len())?;
        check_weights("layer weights", input.layers, self.layers.len())?;
        if input.pose.len() > self.pose.len() {
            return Err(EvalError::Length { what: "joint rotations", limit: self.pose.len(), got: input.pose.len() });
        }
        for (i, q) in input.pose.iter().enumerate() {
            if !q.is_finite() || q.length() < 1e-6 {
                return Err(EvalError::NonFinite { what: "joint rotation", index: i });
            }
        }
        Ok(())
    }

    /// Runs steps 1–5 (and the strain of step 6) for `input`. On error nothing changes.
    /// Allocation-free.
    pub fn evaluate(&mut self, input: &EvalInput<'_>) -> EvalResult<()> {
        self.check_input(input)?;
        self.apply_rest(input.shape, input.layers)?;
        self.apply_expressions(input)?;
        for (slot, q) in self.pose.iter_mut().enumerate() {
            *q = input.pose.get(slot).map_or(Quat::IDENTITY, |q| q.normalize());
        }
        self.apply_pose_correctives()?;
        self.skin(input.skinning);
        if let Some(guard) = &self.guard {
            self.guard_report = guard.solve(&mut self.positions);
        }
        self.recompute_normals();
        if self.rest_dirty {
            self.strain.set_rest(&self.rest);
            self.rest_dirty = false;
        }
        self.strain.compute(&self.positions);
        Ok(())
    }

    /// Step 1 only: brings the rest shape to `shape` and `layers` weights (checked as in
    /// [`evaluate`](Self::evaluate)) by difference accumulation. The other outputs keep their last
    /// values. This is the "apply" half of a landmark drag `[LM-03]`. Allocation-free.
    pub fn update_rest(&mut self, shape: &[f32], layers: &[f32]) -> EvalResult<()> {
        check_weights("shape weights", shape, self.applied_shape.len())?;
        check_weights("layer weights", layers, self.layers.len())?;
        self.apply_rest(shape, layers)
    }

    fn apply_rest(&mut self, shape: &[f32], layers: &[f32]) -> EvalResult<()> {
        let deltas = self.template.deltas();
        let (is_shape, applied) = (&self.is_shape, &self.applied_shape);
        let target = |c: usize| if is_shape[c] { shape.get(c).copied().unwrap_or(0.0) } else { 0.0 };
        // Every changed component at once (one pass over the vertices for runs of contiguous ones).
        accumulate_all(deltas, applied.len(), |c| target(c) - applied[c], &self.plain, &mut self.rest)?;
        for c in 0..self.applied_shape.len() {
            let w = if self.is_shape[c] { shape.get(c).copied().unwrap_or(0.0) } else { 0.0 };
            if w != self.applied_shape[c] {
                self.applied_shape[c] = w;
                self.incremental_updates += 1;
                self.rest_dirty = true;
            }
        }
        for (l, layer) in self.layers.iter().enumerate() {
            let w = layers.get(l).copied().unwrap_or(0.0);
            let old = self.applied_layers[l];
            if w != old {
                layer.accumulate(w - old, &mut self.rest);
                self.applied_layers[l] = w;
                self.incremental_updates += 1;
                self.rest_dirty = true;
            }
        }
        if self.incremental_updates >= FULL_REFRESH_INTERVAL {
            self.refresh_rest()?;
        }
        Ok(())
    }

    /// Re-accumulates the rest shape from the template and the applied weights (the periodic full
    /// pass of `[EVAL-03]`). Allocation-free.
    pub fn refresh_rest(&mut self) -> EvalResult<()> {
        let deltas = self.template.deltas();
        let applied = &self.applied_shape;
        // Checked before the copy, so a failure leaves the buffer as it was.
        check_components(deltas, applied.len(), |c| applied[c], self.rest.len())?;
        self.rest.copy_from_slice(self.template.mesh().positions());
        accumulate_all(deltas, applied.len(), |c| applied[c], &self.plain, &mut self.rest)?;
        for (layer, &w) in self.layers.iter().zip(&self.applied_layers) {
            if w != 0.0 {
                layer.accumulate(w, &mut self.rest);
            }
        }
        self.incremental_updates = 0;
        self.rest_dirty = true;
        Ok(())
    }

    fn apply_expressions(&mut self, input: &EvalInput<'_>) -> EvalResult<()> {
        let t = &*self.template;
        let deltas = t.deltas();
        let set = t.expressions();
        self.pre_skin.copy_from_slice(&self.rest);
        for (e, &w) in set.expressions.iter().zip(input.expressions) {
            if w != 0.0 {
                accumulate(deltas, e.component as usize, w, &mut self.pre_skin)?;
            }
        }
        for (c, act) in set.combinations.iter().zip(&mut self.combination_activations) {
            *act = c.activation(input.expressions);
            if *act != 0.0 {
                accumulate(deltas, c.component as usize, *act, &mut self.pre_skin)?;
            }
        }
        Ok(())
    }

    fn apply_pose_correctives(&mut self) -> EvalResult<()> {
        let t = &*self.template;
        let deltas = t.deltas();
        for (pc, act) in t.expressions().pose_correctives.iter().zip(&mut self.pose_activations) {
            let q = self.pose.get(pc.joint as usize).copied().unwrap_or(Quat::IDENTITY);
            *act = pc.activation(q);
            if *act != 0.0 {
                accumulate(deltas, pc.component as usize, *act, &mut self.pre_skin)?;
            }
        }
        Ok(())
    }

    fn skin(&mut self, mode: SkinningMode) {
        let t = &*self.template;
        let skin = t.skin();
        if self.joints.is_empty() || skin.influences == 0 {
            self.positions.copy_from_slice(&self.pre_skin);
            return;
        }
        // Bind positions: rest world positions plus the regressor's offsets for the shape weights.
        self.joints.bind_positions.copy_from_slice(&self.rest_world_positions);
        for (c, &w) in self.applied_shape.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            let (joints, offsets) = t.skeleton().regressor.component(c);
            for (&j, &d) in joints.iter().zip(offsets) {
                if let Some(p) = self.joints.bind_positions.get_mut(j as usize) {
                    *p += d * w;
                }
            }
        }
        self.joints.pose(&self.parents, &self.bind_rotations, &self.pose);
        for (v, (out, &p)) in self.positions.iter_mut().zip(&self.pre_skin).enumerate() {
            let (joints, weights) = skin.vertex(v);
            let dq = match mode {
                SkinningMode::ByRegion => self.dq_vertex.get(v).copied().unwrap_or(false),
                SkinningMode::Linear => false,
                SkinningMode::DualQuaternion => true,
            };
            *out = if dq {
                self.joints.skin_dual_quat(p, joints, weights)
            } else {
                self.joints.skin_linear(p, joints, weights)
            };
        }
    }

    fn recompute_normals(&mut self) {
        self.frames.compute(&self.positions, self.template.mesh().uvs(), &self.triangles);
        self.frames.gather_normals(&self.normal_gather, &mut self.normals);
        self.frames.gather_tangents(&self.own_corners, &self.normals, &mut self.tangents);
    }

    /// The rest shape (step 1).
    #[must_use]
    pub fn rest_positions(&self) -> &[Vec3] {
        &self.rest
    }

    /// The shape before skinning (steps 1–3).
    #[must_use]
    pub fn unposed_positions(&self) -> &[Vec3] {
        &self.pre_skin
    }

    /// The evaluated positions (steps 1–4).
    #[must_use]
    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    /// The unit vertex normals of [`positions`](Self::positions) (step 5).
    #[must_use]
    pub fn normals(&self) -> &[Vec3] {
        &self.normals
    }

    /// The unit vertex tangents with handedness in `w` (step 5).
    #[must_use]
    pub fn tangents(&self) -> &[Vec4] {
        &self.tangents
    }

    /// The triangle list the normals are computed on (three vertex indices per triangle).
    #[must_use]
    pub fn triangles(&self) -> &[u32] {
        &self.triangles
    }

    /// The triangle pass of the last evaluation (unit normals, tangents, corner angles).
    #[must_use]
    pub fn triangle_frames(&self) -> &TriangleFrames {
        &self.frames
    }

    /// Runs `guard` after skinning in every evaluation from now on, before the normals
    /// `[EVAL-08]`, and corrects the current output with it at once (`None` turns it off; the
    /// output keeps its last correction until the next evaluation).
    pub fn set_guard(&mut self, guard: Option<PenetrationGuard>) {
        self.guard = guard;
        if let Some(guard) = &self.guard {
            self.guard_report = guard.solve(&mut self.positions);
            self.recompute_normals();
        }
    }

    /// The penetration guard run after skinning, if one is set.
    #[must_use]
    pub fn guard(&self) -> Option<&PenetrationGuard> {
        self.guard.as_ref()
    }

    /// What the guard found (and corrected) in the last evaluation; empty without a guard.
    #[must_use]
    pub fn guard_report(&self) -> PenetrationReport {
        self.guard_report
    }

    /// The posed joint frames and skinning transforms.
    #[must_use]
    pub fn joints(&self) -> &JointFrames {
        &self.joints
    }

    /// The activation of every combination corrective in the last evaluation.
    #[must_use]
    pub fn combination_activations(&self) -> &[f32] {
        &self.combination_activations
    }

    /// The activation of every pose corrective in the last evaluation.
    #[must_use]
    pub fn pose_activations(&self) -> &[f32] {
        &self.pose_activations
    }

    /// The shape weights currently in the rest shape.
    #[must_use]
    pub fn shape_weights(&self) -> &[f32] {
        &self.applied_shape
    }

    /// The edge strain of the last evaluation and its per-region reduction.
    #[must_use]
    pub fn strain(&self) -> &StrainField {
        &self.strain
    }

    /// Whether vertex `v` is skinned with dual quaternions under [`SkinningMode::ByRegion`].
    #[must_use]
    pub fn uses_dual_quaternions(&self, v: usize) -> bool {
        self.dq_vertex.get(v).copied().unwrap_or(false)
    }
}

/// The magnitude of a finite half float, exactly, without branches and without touching a
/// subnormal `f32` (an operation on one costs a slow microcode assist on x86, about 15x the whole
/// loop): a normal half's exponent and mantissa are moved into an `f32` by integer arithmetic
/// (rebiasing the exponent by 112), a subnormal half is its mantissa times `2^-24`; the two are
/// computed side by side and one is selected, so the loop vectorises.
#[inline(always)]
fn f16_magnitude(bits: u16) -> f32 {
    let b = u32::from(bits);
    let normal = ((b & 0x7fff) << 13) + (112 << 23);
    // The mantissa is below 2^10, so the conversion through `i32` is exact.
    #[allow(clippy::cast_possible_wrap)]
    let subnormal = ((b & 0x03ff) as i32 as f32 * f32::from_bits(0x3380_0000)).to_bits(); // 2^-24
    // All ones for a subnormal (zero exponent), as a bit mask rather than a branch.
    let mask = 0_u32.wrapping_sub(u32::from(b & 0x7c00 == 0));
    f32::from_bits((subnormal & mask) | (normal & !mask))
}

/// Decodes a finite half float without branches ([`f16_magnitude`], then the sign). Infinities
/// and NaN do not occur in delta bases ([`DeltaBases::from_parts`] refuses them). Much faster than
/// a general decode, which makes it the inner loop of every delta accumulation here.
#[inline]
#[must_use]
pub fn decode_finite_f16(h: F16) -> f32 {
    let bits = h.to_bits();
    f32::from_bits(f16_magnitude(bits).to_bits() | (u32::from(bits & 0x8000) << 16))
}

/// Adds `weight` times delta component `c` to `positions` (the CPU form of one `[EVAL-02]`
/// accumulation step), with [`decode_finite_f16`]. Fails without changing anything when `c` does
/// not exist or one of its vertices is outside `positions`. Allocation-free.
pub fn accumulate(deltas: &DeltaBases, c: usize, weight: f32, positions: &mut [Vec3]) -> EvalResult<()> {
    if c >= deltas.len() {
        return Err(EvalError::Missing { what: "delta component", index: c });
    }
    let (indices, values) = (deltas.indices(c), deltas.deltas(c));
    // Indices increase within a component, so the last one is the largest.
    if indices.last().is_some_and(|&v| v as usize >= positions.len()) {
        return Err(EvalError::Missing { what: "vertex", index: c });
    }
    // The exact magnitude times the weight, then the half's sign applied by flipping the product's
    // sign bit: the same bits as `decode_finite_f16(h) * weight`.
    let term = |h: F16| {
        let bits = h.to_bits();
        let m = f16_magnitude(bits) * weight;
        f32::from_bits(m.to_bits() ^ (u32::from(bits & 0x8000) << 16))
    };
    // A component whose vertices form one contiguous range (every dense component, and regional
    // ones over a contiguous vertex block): strictly increasing indices whose span equals their
    // count are exactly `first..first + len`, so the slices are walked side by side without the
    // per-entry index, a loop the compiler vectorises.
    if let (Some(&first), Some(&last)) = (indices.first(), indices.last())
        && (last - first) as usize + 1 == indices.len()
        && let Some(range) = positions.get_mut(first as usize..=last as usize)
    {
        for (p, d) in range.iter_mut().zip(values) {
            p.x += term(d[0]);
            p.y += term(d[1]);
            p.z += term(d[2]);
        }
        return Ok(());
    }
    for (&v, d) in indices.iter().zip(values) {
        if let Some(p) = positions.get_mut(v as usize) {
            *p += Vec3::new(term(d[0]), term(d[1]), term(d[2]));
        }
    }
    Ok(())
}

/// The vertex range `first..=last` of component `c` when its vertices are exactly that range
/// (strictly increasing indices whose span equals their count) inside `len` positions.
fn contiguous(deltas: &DeltaBases, c: usize, len: usize) -> Option<(usize, usize)> {
    let indices = deltas.indices(c);
    let (&first, &last) = (indices.first()?, indices.last()?);
    ((last - first) as usize + 1 == indices.len() && (last as usize) < len).then_some((first as usize, last as usize))
}

/// Fails, as [`accumulate`] would, when a component with a nonzero weight among the first `count`
/// does not exist or moves a vertex beyond `len`.
fn check_components(deltas: &DeltaBases, count: usize, weight: impl Fn(usize) -> f32, len: usize) -> EvalResult<()> {
    for c in (0..count).filter(|&c| weight(c) != 0.0) {
        if c >= deltas.len() {
            return Err(EvalError::Missing { what: "delta component", index: c });
        }
        if deltas.indices(c).last().is_some_and(|&v| v as usize >= len) {
            return Err(EvalError::Missing { what: "vertex", index: c });
        }
    }
    Ok(())
}

/// Vertices per block of [`accumulate_all`]: the block's positions, as flat floats, stay in L1.
const BLOCK: usize = 512;

/// Adds `weight(c)` times component `c` for every `c` below `count` whose weight is not zero, with
/// exactly the result of calling [`accumulate`] for each in order. Runs of consecutive components
/// whose vertices form contiguous ranges are added block by block: a block of positions is copied
/// into a flat float buffer once, every component of the run is added to it in component order
/// (flat loops the compiler vectorises), and it is copied back, so the positions are read and
/// written once per run instead of once per component. Every component is checked first: on
/// error nothing changed. Allocation-free.
pub fn accumulate_all(
    deltas: &DeltaBases,
    count: usize,
    weight: impl Fn(usize) -> f32,
    plain: &[bool],
    positions: &mut [Vec3],
) -> EvalResult<()> {
    check_components(deltas, count, &weight, positions.len())?;
    let mut c = 0;
    while c < count {
        let start = c;
        let (mut lo, mut hi) = (usize::MAX, 0);
        while c < count {
            if weight(c) != 0.0 {
                let Some((first, last)) = contiguous(deltas, c, positions.len()) else { break };
                lo = lo.min(first);
                hi = hi.max(last + 1);
            }
            c += 1;
        }
        if lo < hi {
            accumulate_run(deltas, start..c, &weight, plain, lo..hi, positions);
        }
        if c < count {
            // A changed component with scattered vertices: the indexed loop.
            accumulate(deltas, c, weight(c), positions)?;
            c += 1;
        }
    }
    Ok(())
}

/// `dst[i] += w * src[i]` with the halves decoded as [`decode_finite_f16`] does (the same bits as
/// [`accumulate`]'s term). Kept out of line: compiled on its own the loop vectorises, inlined into
/// its caller (rustc 1.99) it does not.
#[inline(never)]
fn add_halves(dst: &mut [f32], src: &[F16], w: f32) {
    for (b, h) in dst.iter_mut().zip(src) {
        let bits = h.to_bits();
        let m = f16_magnitude(bits) * w;
        *b += f32::from_bits(m.to_bits() ^ (u32::from(bits & 0x8000) << 16));
    }
}

/// Whether no half of `values` is subnormal (zero exponent with a nonzero mantissa).
fn plain_halves(values: &[F16]) -> bool {
    // A nonzero exponent, or a zero mantissa (the low 10 bits).
    values.iter().all(|h| h.to_bits() & 0x7c00 != 0 || h.to_bits().trailing_zeros() >= 10)
}

/// [`add_halves`] for halves that are normal or zero ([`plain_halves`]): the magnitude is the
/// rebiased bits, masked to zero for a zero half. The same bits as [`add_halves`] on such input.
#[inline(never)]
fn add_plain_halves(dst: &mut [f32], src: &[F16], w: f32) {
    for (b, h) in dst.iter_mut().zip(src) {
        let bits = u32::from(h.to_bits());
        let magnitude = ((bits & 0x7fff) << 13) + (112 << 23);
        let nonzero = 0_u32.wrapping_sub(u32::from(bits & 0x7fff != 0));
        let m = f32::from_bits((magnitude & nonzero) | ((bits & 0x8000) << 16)) * w;
        *b += m;
    }
}

/// Adds the components `run` (each contiguous, or with a zero weight) over the vertices `span`.
fn accumulate_run(
    deltas: &DeltaBases,
    run: std::ops::Range<usize>,
    weight: &impl Fn(usize) -> f32,
    plain: &[bool],
    span: std::ops::Range<usize>,
    positions: &mut [Vec3],
) {
    let mut flat = [0.0_f32; 3 * BLOCK];
    let mut block = span.start;
    while block < span.end {
        let end = (block + BLOCK).min(span.end);
        let Some(target) = positions.get_mut(block..end) else { return };
        let buf = &mut flat[..3 * target.len()];
        for (b, p) in buf.as_chunks_mut::<3>().0.iter_mut().zip(target.iter()) {
            *b = p.to_array();
        }
        for c in run.clone() {
            let w = weight(c);
            if w == 0.0 {
                continue;
            }
            let Some((first, last)) = contiguous(deltas, c, usize::MAX) else { continue };
            // This component's part of the block.
            let (from, to) = (first.max(block), (last + 1).min(end));
            if from >= to {
                continue;
            }
            let values = deltas.deltas(c).as_flattened();
            let (Some(src), Some(dst)) =
                (values.get(3 * (from - first)..3 * (to - first)), buf.get_mut(3 * (from - block)..3 * (to - block)))
            else {
                continue;
            };
            if plain.get(c).copied().unwrap_or(false) {
                add_plain_halves(dst, src, w);
            } else {
                add_halves(dst, src, w);
            }
        }
        for (p, b) in target.iter_mut().zip(buf.as_chunks::<3>().0) {
            *p = Vec3::from_array(*b);
        }
        block = end;
    }
}

fn check_weights(what: &'static str, weights: &[f32], limit: usize) -> EvalResult<()> {
    if weights.len() > limit {
        return Err(EvalError::Length { what, limit, got: weights.len() });
    }
    for (i, w) in weights.iter().enumerate() {
        if !w.is_finite() {
            return Err(EvalError::NonFinite { what, index: i });
        }
        if w.abs() > MAX_WEIGHT {
            return Err(EvalError::Range { what, index: i, rule: "magnitude above MAX_WEIGHT" });
        }
    }
    Ok(())
}
