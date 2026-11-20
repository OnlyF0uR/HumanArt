//! The evaluated character: the template, the CPU reference evaluator and the bindings from
//! registry parameters to evaluation inputs.
//!
//! [`Model::new`] binds every registry parameter whose evaluation names a template delta component
//! (`[REG-01]` `component`) or an expression (`expression`) to the evaluator's input slot; the rest
//! (materials, grooms, editor values, measurements) do not move vertices and stay unbound
//! ([`Model::coverage`] reports both). [`Model::sync`] reads the document's values into the input
//! vectors, compares them with what was last evaluated and re-evaluates only when they differ,
//! without allocating `[PERF-R1]`.
//!
//! [`Model::pick`] is the CPU fallback picking of `[VP-01]`: a ray cast against the evaluated
//! triangles, reporting triangle, barycentrics, point and region. [`Model::region_outline`] gives
//! the boundary edges of a region (and its sub-regions) for viewport highlights (`[APP-03]`).

use std::collections::HashMap;
use std::sync::Arc;

use humanart_doc::document::Document;
use humanart_doc::expression_drive::ExpressionDrive;
use humanart_doc::registry::{Evaluation, ParamValue, Registry};
use humanart_math::{Quat, Vec3};
use humanart_model::Template;
use humanart_model::eval::{EvalInput, Evaluator, SkinningMode};

use super::error::EngineError;

/// Where a registry parameter goes in the evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binding {
    /// It does not move vertices (or the template has no target for it).
    None,
    /// The weight of a shape delta component.
    Shape(u32),
    /// The weight of an expression.
    Expression(u32),
}

/// How much of the registry drives the template.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Coverage {
    /// Parameters bound to a shape component.
    pub shape: usize,
    /// Parameters bound to an expression.
    pub expression: usize,
    /// Parameters naming a component or expression the template does not have.
    pub missing: usize,
    /// Parameters that do not move vertices by design (materials, grooms, editor values).
    pub other: usize,
}

/// A ray hit on the evaluated surface `[VP-01]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceHit {
    /// The triangle (index into the triangle list).
    pub triangle: u32,
    /// Barycentric weights of its three corners.
    pub barycentric: [f32; 3],
    /// The world point.
    pub point: Vec3,
    /// Distance along the ray.
    pub distance: f32,
    /// The region of the triangle's first corner.
    pub region: u16,
}

/// The evaluated character (see the module documentation).
#[derive(Debug)]
pub struct Model {
    template: Arc<Template>,
    eval: Evaluator,
    bindings: Vec<Binding>,
    coverage: Coverage,
    shape: Vec<f32>,
    expressions: Vec<f32>,
    pose: Vec<Quat>,
    applied_shape: Vec<f32>,
    applied_expressions: Vec<f32>,
    applied_pose: Vec<Quat>,
    revision: u64,
    outlines: Vec<Option<Vec<[u32; 2]>>>,
    /// The expression control rig driven from the document (`todo.md` 1.8); `None` when the
    /// template has no rig (test fixtures), then expression parameters bind straight to targets.
    drive: Option<ExpressionDrive>,
}

fn scalar(value: &ParamValue) -> Option<f32> {
    match value {
        #[allow(clippy::cast_possible_truncation)] // Registry values are small coefficients.
        ParamValue::Scalar(v) => Some(*v as f32),
        ParamValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

impl Model {
    /// Binds `registry` to `template` and evaluates the neutral state.
    pub fn new(template: Arc<Template>, registry: &Registry) -> Self {
        let mut eval = Evaluator::new(Arc::clone(&template));
        // The eyelid and lip penetration guard [EVAL-08] (super::features).
        eval.set_guard(super::features::penetration_guard(&template));
        let t = &*template;
        let mut coverage = Coverage::default();
        let bindings = registry
            .params()
            .iter()
            .map(|p| match &p.evaluation {
                Evaluation::Component(name) => {
                    if let Some(c) = t.deltas().find(name) {
                        coverage.shape += 1;
                        Binding::Shape(c)
                    } else {
                        coverage.missing += 1;
                        Binding::None
                    }
                }
                Evaluation::Expression(name) => {
                    if let Some(e) = t.expressions().expressions.iter().position(|e| e.name == *name) {
                        coverage.expression += 1;
                        Binding::Expression(u32::try_from(e).unwrap_or(u32::MAX))
                    } else {
                        coverage.missing += 1;
                        Binding::None
                    }
                }
                _ => {
                    coverage.other += 1;
                    Binding::None
                }
            })
            .collect();
        // Expression parameters go through the template's control rig (jaw and tongue rotate).
        let drive = ExpressionDrive::new(t, registry).ok();
        let components = t.deltas().len();
        let expressions = t.expressions().expressions.len();
        let joints = t.skeleton().joints.len();
        let regions = t.regions().regions.len();
        Self {
            eval,
            bindings,
            coverage,
            shape: vec![0.0; components],
            expressions: vec![0.0; expressions],
            pose: vec![Quat::IDENTITY; joints],
            applied_shape: vec![0.0; components],
            applied_expressions: vec![0.0; expressions],
            applied_pose: vec![Quat::IDENTITY; joints],
            revision: 1,
            outlines: vec![None; regions],
            drive,
            template,
        }
    }

    /// The template.
    pub fn template(&self) -> &Arc<Template> {
        &self.template
    }

    /// The binding of each registry parameter, by registry index.
    pub fn bindings(&self) -> &[Binding] {
        &self.bindings
    }

    /// How much of the registry drives the template.
    pub fn coverage(&self) -> Coverage {
        self.coverage
    }

    /// Counts up every time the evaluated surface changes.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// The evaluated positions (posed), one per template vertex.
    pub fn positions(&self) -> &[Vec3] {
        self.eval.positions()
    }

    /// The evaluated normals.
    pub fn normals(&self) -> &[Vec3] {
        self.eval.normals()
    }

    /// Three vertex indices per triangle.
    pub fn triangles(&self) -> &[u32] {
        self.eval.triangles()
    }

    /// The evaluation input of the current surface (what the last evaluation used), for a GPU
    /// deformer that mirrors this model.
    pub fn input(&self) -> EvalInput<'_> {
        EvalInput {
            shape: &self.applied_shape,
            expressions: &self.applied_expressions,
            layers: &[],
            pose: &self.applied_pose,
            skinning: SkinningMode::ByRegion,
        }
    }

    /// The evaluator (for metrics and tools).
    pub fn evaluator(&self) -> &Evaluator {
        &self.eval
    }

    /// Reads the document's values into the evaluation inputs and re-evaluates when they changed.
    /// Returns whether the surface changed. Allocation-free.
    pub fn sync(&mut self, doc: &Document, registry: &Registry) -> Result<bool, EngineError> {
        if registry.len() != self.bindings.len() {
            return Err(EngineError::Invalid { what: "the registry differs from the one the model was bound to" });
        }
        self.shape.fill(0.0);
        self.expressions.fill(0.0);
        for (i, (param, binding)) in registry.params().iter().zip(&self.bindings).enumerate() {
            let slot = match *binding {
                Binding::None => continue,
                Binding::Expression(_) if self.drive.as_ref().is_some_and(|d| d.drives_param(i)) => continue,
                Binding::Shape(c) => self.shape.get_mut(c as usize),
                Binding::Expression(e) => self.expressions.get_mut(e as usize),
            };
            let value = doc.params.get(&param.id).unwrap_or(&param.default);
            if let (Some(slot), Some(v)) = (slot, scalar(value)) {
                *slot = v;
            }
        }
        self.pose.fill(Quat::IDENTITY);
        let joints = &self.template.skeleton().joints;
        for (name, q) in &doc.pose {
            if let Some(j) = joints.iter().position(|j| j.name == *name)
                && let Some(slot) = self.pose.get_mut(j)
            {
                #[allow(clippy::cast_possible_truncation)] // Unit quaternion components.
                let quat = Quat::from_xyzw(q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32);
                *slot = quat;
            }
        }
        if let Some(drive) = self.drive.as_mut() {
            drive
                .evaluate(doc, registry)
                .map_err(|_| EngineError::Invalid { what: "the expression rig could not be evaluated" })?;
            drive.write(&mut self.expressions, &mut self.pose);
        }
        if self.shape == self.applied_shape
            && self.expressions == self.applied_expressions
            && self.pose == self.applied_pose
        {
            return Ok(false);
        }
        let input = EvalInput {
            shape: &self.shape,
            expressions: &self.expressions,
            layers: &[],
            pose: &self.pose,
            skinning: SkinningMode::ByRegion,
        };
        self.eval.evaluate(&input)?;
        self.applied_shape.copy_from_slice(&self.shape);
        self.applied_expressions.copy_from_slice(&self.expressions);
        self.applied_pose.copy_from_slice(&self.pose);
        self.revision += 1;
        Ok(true)
    }

    /// The bounding sphere of the evaluated surface (centre, radius).
    pub fn bounds(&self) -> (Vec3, f32) {
        bounds_of(self.positions().iter().copied())
    }

    /// The axis-aligned extent of the evaluated surface (lowest and highest corner).
    pub fn extent(&self) -> (Vec3, Vec3) {
        extent(self.positions().iter().copied())
    }

    /// A sphere around the head: the part of the surface above the neck (the top 13 % of the
    /// height, about one head height of an adult).
    pub fn head_bounds(&self) -> (Vec3, f32) {
        let (lo, hi) = extent(self.positions().iter().copied());
        let cut = hi.y - (hi.y - lo.y) * 0.13;
        bounds_of(self.positions().iter().copied().filter(|p| p.y >= cut))
    }

    /// The region of a triangle (of its first corner).
    pub fn triangle_region(&self, triangle: u32) -> Option<u16> {
        let first = *self.triangles().get(triangle as usize * 3)?;
        self.template.regions().vertex_regions.get(first as usize).copied()
    }

    /// The region id of a region name.
    pub fn region_id(&self, name: &str) -> Option<u16> {
        self.template.regions().find(name)
    }

    /// The closest hit of the ray from `origin` along `dir` (need not be unit) on the evaluated
    /// triangles `[VP-01]` (CPU fallback picking; Möller-Trumbore, both faces).
    pub fn pick(&self, origin: Vec3, dir: Vec3) -> Option<SurfaceHit> {
        let dir = dir.normalize_or(Vec3::ZERO);
        if dir == Vec3::ZERO || !origin.is_finite() {
            return None;
        }
        let positions = self.positions();
        let mut best: Option<SurfaceHit> = None;
        for (t, tri) in self.triangles().as_chunks::<3>().0.iter().enumerate() {
            let (Some(&a), Some(&b), Some(&c)) =
                (positions.get(tri[0] as usize), positions.get(tri[1] as usize), positions.get(tri[2] as usize))
            else {
                continue;
            };
            let (e1, e2) = (b - a, c - a);
            let p = dir.cross(e2);
            let det = e1.dot(p);
            if det.abs() < 1e-12 {
                continue;
            }
            let inv = 1.0 / det;
            let s = origin - a;
            let u = s.dot(p) * inv;
            if !(0.0..=1.0).contains(&u) {
                continue;
            }
            let q = s.cross(e1);
            let v = dir.dot(q) * inv;
            if v < 0.0 || u + v > 1.0 {
                continue;
            }
            let distance = e2.dot(q) * inv;
            if distance > 1e-6 && best.is_none_or(|h| distance < h.distance) {
                let triangle = u32::try_from(t).unwrap_or(u32::MAX);
                best = Some(SurfaceHit {
                    triangle,
                    barycentric: [1.0 - u - v, u, v],
                    point: origin + dir * distance,
                    distance,
                    region: 0,
                });
            }
        }
        best.map(|mut h| {
            h.region = self.triangle_region(h.triangle).unwrap_or(0);
            h
        })
    }

    /// The boundary edges (vertex pairs) of `region` and its sub-regions: edges of the region's
    /// triangles that have no second triangle inside the region. Computed on the first request
    /// for a region and kept.
    pub fn region_outline(&mut self, region: u16) -> &[[u32; 2]] {
        let regions = self.template.regions();
        if usize::from(region) >= regions.regions.len() {
            return &[];
        }
        let slot = usize::from(region);
        if self.outlines.get(slot).is_some_and(Option::is_none) {
            let inside = |v: u32| regions.vertex_regions.get(v as usize).is_some_and(|&r| regions.is_within(r, region));
            let mut count: HashMap<(u32, u32), u32> = HashMap::new();
            for tri in self.eval.triangles().as_chunks::<3>().0 {
                if !inside(tri[0]) {
                    continue;
                }
                for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                    *count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
                }
            }
            let mut edges: Vec<[u32; 2]> =
                count.into_iter().filter(|&(_, n)| n == 1).map(|((a, b), _)| [a, b]).collect();
            edges.sort_unstable();
            if let Some(entry) = self.outlines.get_mut(slot) {
                *entry = Some(edges);
            }
        }
        self.outlines.get(slot).and_then(Option::as_deref).unwrap_or(&[])
    }
}

fn extent(points: impl Iterator<Item = Vec3>) -> (Vec3, Vec3) {
    points.fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), p| (lo.min(p), hi.max(p)))
}

fn bounds_of(points: impl Iterator<Item = Vec3> + Clone) -> (Vec3, f32) {
    let (lo, hi) = extent(points);
    if lo.x > hi.x {
        return (Vec3::ZERO, 1.0);
    }
    let center = (lo + hi) * 0.5;
    (center, ((hi - lo) * 0.5).length().max(1e-3))
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_doc::document::{Author, AuthorKind};
    use humanart_doc::registry::{Channel, Param, ParamKind, Range, Symmetry};
    use humanart_math::F16;
    use humanart_model::delta::{DeltaBases, DeltaComponent, DeltaKind};

    fn param(id: &str, evaluation: Evaluation) -> Param {
        Param {
            id: id.to_owned(),
            display_name: id.to_owned(),
            description: format!("The test parameter {id}."),
            group: "identity".to_owned(),
            region: None,
            kind: ParamKind::Scalar,
            dimension: 1,
            choices: Vec::new(),
            unit: String::new(),
            range: Some(Range { min: -3.0, max: 3.0, soft_min: -2.0, soft_max: 2.0, step: 0.01 }),
            default: ParamValue::Scalar(0.0),
            symmetry: Symmetry::None,
            affects: vec![Channel::Shape],
            evaluation,
            read_only: false,
            locked_by_default: false,
            plausibility_weight: 1.0,
            since_version: 1,
            tags: Vec::new(),
            provisional: None,
        }
    }

    /// The generated template (without LODs) with one more shape component, `semantic.lift`,
    /// that lifts one vertex by 1 m per unit weight. It is appended after the template's own
    /// components (identity, expressions' correctives ...) in the region of the last one, so the
    /// order by region and every existing component index stay as they are; the joint regressor
    /// gets an empty entry for it. Returns the template and the lifted vertex.
    fn lifted() -> (Arc<Template>, usize) {
        let mut parts = super::super::tests::template().parts().clone();
        let (mut components, mut offsets, mut indices, mut deltas) = parts.deltas.clone().into_parts();
        let region = components.last().map_or(0, |c| c.region);
        let map = &parts.regions;
        let vertex = map.vertex_regions.iter().position(|&r| map.is_within(r, region)).unwrap();
        components.push(DeltaComponent { name: "semantic.lift".into(), kind: DeltaKind::Semantic, region });
        indices.push(vertex as u32);
        deltas.push([F16::from_f32(0.0), F16::from_f32(1.0), F16::from_f32(0.0)]);
        offsets.push(indices.len() as u32);
        parts.deltas = DeltaBases::from_parts(components, offsets, indices, deltas).unwrap();
        let reg = &mut parts.skeleton.regressor;
        let end = reg.offsets.last().copied().unwrap_or(0);
        reg.offsets.push(end);
        (Arc::new(Template::new(parts).unwrap()), vertex)
    }

    fn registry() -> Registry {
        Registry::new(vec![
            param("identity.lift", Evaluation::Component("semantic.lift".into())),
            param("identity.absent", Evaluation::Component("semantic.absent".into())),
            param("skin.tone", Evaluation::Material("tone".into())),
        ])
        .unwrap()
    }

    fn doc() -> Document {
        let me = Author { id: "human.test".into(), kind: AuthorKind::Human, name: "Test".into() };
        Document::new(humanart_core::id::Id128::from_u128(7), "test", "generated", 1, 0, me)
    }

    #[test]
    fn reg_01_bindings_and_sync_re_evaluate_only_on_change() {
        let reg = registry();
        let (template, v) = lifted();
        let mut model = Model::new(template, &reg);
        assert_eq!(model.coverage(), Coverage { shape: 1, expression: 0, missing: 1, other: 1 });
        let mut d = doc();
        let rest = model.positions()[v];
        assert!(!model.sync(&d, &reg).unwrap(), "defaults: nothing changes");
        d.set_param(&reg, "identity.lift", ParamValue::Scalar(0.5), "human.test").unwrap();
        let r0 = model.revision();
        assert!(model.sync(&d, &reg).unwrap());
        assert!(model.revision() > r0);
        assert!((model.positions()[v].y - (rest.y + 0.5)).abs() < 1e-3, "{:?} vs {rest:?}", model.positions()[v]);
        assert!(!model.sync(&d, &reg).unwrap(), "same values: no evaluation");
        let counts = humanart_core::alloc::count_in(|| {
            let _ = model.sync(&d, &reg);
        });
        assert_eq!(counts.allocations, 0, "[PERF-R1] a sync without change does not allocate");
    }

    #[test]
    fn vp_01_cpu_pick_finds_the_nearest_triangle_and_its_region() {
        let reg = registry();
        let mut model = Model::new(lifted().0, &reg);
        let (center, radius) = model.bounds();
        let origin = center + Vec3::new(0.0, 0.0, radius * 4.0);
        let hit = model.pick(origin, center - origin).expect("the body is hit");
        assert!(hit.distance > 0.0 && hit.distance < radius * 4.0);
        assert!((hit.barycentric.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(model.pick(origin, origin - center).is_none(), "looking away misses");
        assert!(model.pick(origin, Vec3::ZERO).is_none());
        let outline = model.region_outline(hit.region).to_vec();
        assert!(outline.iter().all(|e| e[0] < e[1]));
        assert!(model.region_outline(u16::MAX).is_empty());
    }
}
