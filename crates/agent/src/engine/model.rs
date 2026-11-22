//! The evaluated character: the template, the CPU reference evaluator and the bindings from
//! registry parameters to evaluation inputs.
//!
//! [`Model::new`] binds every registry parameter whose evaluation names a template delta component
//! (`[REG-01]` `component`) or an expression (`expression`) to the evaluator's input slot; the rest
//! (materials, grooms, editor values, measurements) do not move vertices and stay unbound
//! ([`Model::coverage`] reports both). [`Model::sync`] reads the document's values into the input
//! vectors, compares them with what was last evaluated and re-evaluates only when they differ,
//! without allocating `[PERF-R1]`. It also reads the document's `skin.*` material values
//! ([`Model::skin`], todo 1.4), which the views hand to the skin compositor, the eye, mouth and
//! hair-colour materials ([`Model::materials`], todo 2.3), which the views hand to the eye and
//! mouth feature table and the hair renderer, the editor's gaze preview (`eyes.gaze.*`), which
//! turns the eye joints, and the document's procedural skin layers with the global
//! `skin.features.*` values ([`Model::feature_layers`]), which the views generate into the skin.
//!
//! [`Model::pick`] is the CPU fallback picking of `[VP-01]`: a ray cast against the evaluated
//! triangles, reporting triangle, barycentrics, point and region. [`Model::region_outline`] gives
//! the boundary edges of a region (and its sub-regions) for viewport highlights (`[APP-03]`).

use std::collections::HashMap;
use std::sync::Arc;

use humanart_doc::document::{Document, SkinLayer, SkinSource};
use humanart_doc::expression_drive::ExpressionDrive;
use humanart_doc::registry::{Evaluation, ParamValue, Registry};
use humanart_math::{Quat, Vec3};
use humanart_model::Template;
use humanart_model::eval::{EvalInput, Evaluator, SkinningMode};
use humanart_model::eye::{EyeParams, Gaze};
use humanart_model::hair::settings::{GroomSettings, GroomTarget};
use humanart_model::skin_appearance::feature_layers::{FeatureGlobals, FeatureLayer, generator_of};
use humanart_model::skin_appearance::features::{GeneratorKind, GeneratorParams};
use humanart_model::skin_appearance::materials::{MaterialTarget, MaterialValues};
use humanart_model::skin_appearance::values::{SkinTarget, SkinValues};

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
    /// Registry index and destination of every `skin.*` material parameter the skin reads.
    skin_bindings: Vec<(usize, SkinTarget)>,
    /// The document's skin values (registry defaults until a document is read).
    skin: SkinValues,
    /// Counts up every time [`Model::skin`] changes.
    skin_revision: u64,
    /// Registry index and destination of every eye, mouth and hair-colour material parameter.
    material_bindings: Vec<(usize, MaterialTarget)>,
    /// The document's eye, mouth and hair-colour values (registry defaults until a document is
    /// read).
    materials: MaterialValues,
    /// Counts up every time [`Model::materials`] changes.
    materials_revision: u64,
    /// Whether the document sets a hair colour value itself (not only the registry default).
    hair_color_set: bool,
    /// Registry index and destination of every groom setting.
    groom_bindings: Vec<(usize, GroomTarget)>,
    /// The document's groom settings (registry defaults until a document is read).
    grooms: GroomSettings,
    /// Whether the document sets the body hair density itself.
    body_hair_set: bool,
    /// Counts up every time [`Model::grooms`] or [`Model::body_hair_set`] changes.
    grooms_revision: u64,
    /// Registry index and generator slot of every `skin.features.*` parameter.
    feature_slots: Vec<(usize, (usize, usize))>,
    /// The document's global generator settings.
    features: FeatureGlobals,
    /// The document's procedural skin layers, resolved.
    feature_layers: Vec<FeatureLayer>,
    /// Counts up every time [`Model::feature_layers`] changes.
    features_revision: u64,
    /// Registry indices of the gaze preview's yaw and pitch (`eyes.gaze.*`, degrees).
    gaze_params: [Option<usize>; 2],
    /// The eye joints and their rest world rotations.
    eye_joints: Vec<(usize, Quat)>,
}

fn scalar(value: &ParamValue) -> Option<f32> {
    match value {
        #[allow(clippy::cast_possible_truncation)] // Registry values are small coefficients.
        ParamValue::Scalar(v) => Some(*v as f32),
        ParamValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// Writes one document value into the skin values (scalars and the lip colour).
pub fn set_skin(values: &mut SkinValues, target: SkinTarget, value: &ParamValue) {
    match value {
        #[allow(clippy::cast_possible_truncation)] // Colour channels are 0..1.
        ParamValue::Color(c) => {
            values.set_color(target, [c[0] as f32, c[1] as f32, c[2] as f32]);
        }
        other => {
            if let Some(v) = scalar(other) {
                values.set(target, v);
            }
        }
    }
}

/// Writes one document value into the materials (scalars, colours and choices).
pub fn set_material(values: &mut MaterialValues, target: MaterialTarget, value: &ParamValue) {
    match value {
        #[allow(clippy::cast_possible_truncation)] // Colour channels are 0..1.
        ParamValue::Color(c) => {
            values.set_color(target, [c[0] as f32, c[1] as f32, c[2] as f32]);
        }
        ParamValue::Enum(choice) => {
            values.set_choice(target, choice);
        }
        other => {
            if let Some(v) = scalar(other) {
                values.set_scalar(target, v);
            }
        }
    }
}

/// A document skin layer a view generates: visible, procedural, of a known generator that needs
/// no image (the tattoo stamps the user's picture, which a procedural layer does not carry).
fn procedural(layer: &SkinLayer) -> Option<(GeneratorKind, &SkinSource, u64)> {
    let SkinSource::Procedural { generator, seed, .. } = &layer.source else { return None };
    let kind = generator_of(generator).filter(|k| *k != GeneratorKind::Tattoo)?;
    layer.visible.then_some((kind, &layer.source, *seed))
}

/// The resolved settings of a procedural `source` of generator `kind` under `globals`.
fn layer_params(globals: &FeatureGlobals, kind: GeneratorKind, source: &SkinSource) -> GeneratorParams {
    match source {
        SkinSource::Procedural { params, .. } => {
            globals.layer_params(kind, params.iter().map(|(k, v)| (k.as_str(), *v)))
        }
        _ => *globals.of(kind),
    }
}

/// The joints the gaze preview turns.
pub const EYE_JOINTS: [&str; 2] = ["L.eye", "R.eye"];

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
        let skin_bindings: Vec<(usize, SkinTarget)> = registry
            .params()
            .iter()
            .enumerate()
            .filter_map(|(i, p)| match &p.evaluation {
                Evaluation::Material(target) => SkinTarget::from_target(target).map(|t| (i, t)),
                _ => None,
            })
            .collect();
        let mut skin = SkinValues::default();
        for &(i, target) in &skin_bindings {
            if let Some(p) = registry.params().get(i) {
                set_skin(&mut skin, target, &p.default);
            }
        }
        let material_bindings: Vec<(usize, MaterialTarget)> = registry
            .params()
            .iter()
            .enumerate()
            .filter_map(|(i, p)| match &p.evaluation {
                Evaluation::Material(target) => MaterialTarget::from_target(target).map(|t| (i, t)),
                _ => None,
            })
            .collect();
        let mut materials = MaterialValues::default();
        for &(i, target) in &material_bindings {
            if let Some(p) = registry.params().get(i) {
                set_material(&mut materials, target, &p.default);
            }
        }
        let groom_bindings: Vec<(usize, GroomTarget)> = registry
            .params()
            .iter()
            .enumerate()
            .filter_map(|(i, p)| match &p.evaluation {
                Evaluation::Groom(target) => GroomTarget::from_target(target).map(|t| (i, t)),
                _ => None,
            })
            .collect();
        let mut grooms = GroomSettings::default();
        for &(i, target) in &groom_bindings {
            if let Some(v) = registry.params().get(i).and_then(|p| scalar(&p.default)) {
                grooms.set(target, v);
            }
        }
        let feature_slots: Vec<(usize, (usize, usize))> = registry
            .params()
            .iter()
            .enumerate()
            .filter_map(|(i, p)| FeatureGlobals::slot_of(&p.id).map(|s| (i, s)))
            .collect();
        let mut features = FeatureGlobals::default();
        for &(i, slot) in &feature_slots {
            if let Some(v) = registry.params().get(i).and_then(|p| scalar(&p.default)) {
                features.set(slot, v);
            }
        }
        let editor = |name: &str| {
            registry.params().iter().position(|p| matches!(&p.evaluation, Evaluation::Editor(t) if t == name))
        };
        let gaze_params = [editor("gaze.yaw"), editor("gaze.pitch")];
        let rest_world = t.skeleton().rest_world();
        let eye_joints = EYE_JOINTS
            .iter()
            .filter_map(|name| t.skeleton().find(name))
            .filter_map(|j| rest_world.get(usize::from(j)).map(|r| (usize::from(j), r.rotation)))
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
            skin_bindings,
            skin,
            skin_revision: 1,
            material_bindings,
            materials,
            materials_revision: 1,
            hair_color_set: false,
            groom_bindings,
            grooms,
            body_hair_set: false,
            grooms_revision: 1,
            feature_slots,
            features,
            feature_layers: Vec::new(),
            features_revision: 1,
            gaze_params,
            eye_joints,
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

    /// The document's skin values (`skin.*` material parameters; registry defaults before the
    /// first [`Model::sync`]).
    pub fn skin(&self) -> &SkinValues {
        &self.skin
    }

    /// Counts up every time [`Model::skin`] changes.
    pub fn skin_revision(&self) -> u64 {
        self.skin_revision
    }

    /// The document's eye, mouth and hair-colour materials (registry defaults before the first
    /// [`Model::sync`]).
    pub fn materials(&self) -> &MaterialValues {
        &self.materials
    }

    /// Counts up every time [`Model::materials`] or [`Model::hair_color_set`] changes.
    pub fn materials_revision(&self) -> u64 {
        self.materials_revision
    }

    /// Whether the document sets a `hair.color.*` value itself; when it does not, a view keeps
    /// the hair colour of its own options (`HairOptions::shade`).
    pub fn hair_color_set(&self) -> bool {
        self.hair_color_set
    }

    /// The document's brow, lash, facial and body hair settings (`groom` parameters; registry
    /// defaults before the first [`Model::sync`]).
    pub fn grooms(&self) -> &GroomSettings {
        &self.grooms
    }

    /// Whether the document sets `hair.body.density` itself; a view then grows body hair even
    /// when its options do not ask for it.
    pub fn body_hair_set(&self) -> bool {
        self.body_hair_set
    }

    /// Counts up every time [`Model::grooms`] or [`Model::body_hair_set`] changes.
    pub fn grooms_revision(&self) -> u64 {
        self.grooms_revision
    }

    /// The document's global generator settings (`skin.features.*`).
    pub fn feature_globals(&self) -> &FeatureGlobals {
        &self.features
    }

    /// The document's visible procedural skin layers with their resolved settings (each
    /// parameter the layer does not set takes the global value), in document order.
    pub fn feature_layers(&self) -> &[FeatureLayer] {
        &self.feature_layers
    }

    /// Counts up every time [`Model::feature_layers`] changes.
    pub fn features_revision(&self) -> u64 {
        self.features_revision
    }

    /// Reads the global generator settings and the procedural skin layers of `doc`; rebuilds the
    /// layer list (allocating) only when something changed.
    fn sync_features(&mut self, doc: &Document, registry: &Registry) {
        let mut globals = self.features;
        for &(i, slot) in &self.feature_slots {
            let Some(param) = registry.params().get(i) else { continue };
            if let Some(v) = scalar(doc.params.get(&param.id).unwrap_or(&param.default)) {
                globals.set(slot, v);
            }
        }
        let mut same = globals == self.features;
        let mut n = 0;
        for layer in &doc.skin_layers {
            let Some((kind, source, seed)) = procedural(layer) else { continue };
            #[allow(clippy::cast_possible_truncation)] // Opacity is 0..1.
            let opacity = layer.opacity as f32;
            let params = layer_params(&globals, kind, source);
            same &= self.feature_layers.get(n).is_some_and(|f| {
                f.id == layer.id
                    && f.params == params
                    && f.seed == seed
                    && f.region.as_deref() == layer.region.as_deref()
                    && f.opacity.to_bits() == opacity.to_bits()
            });
            n += 1;
        }
        if same && n == self.feature_layers.len() {
            return;
        }
        self.features = globals;
        self.feature_layers = doc
            .skin_layers
            .iter()
            .filter_map(|layer| {
                let (kind, source, seed) = procedural(layer)?;
                #[allow(clippy::cast_possible_truncation)] // Opacity is 0..1.
                let opacity = layer.opacity as f32;
                Some(FeatureLayer {
                    id: layer.id,
                    params: layer_params(&globals, kind, source),
                    seed,
                    region: layer.region.clone(),
                    opacity,
                })
            })
            .collect();
        self.features_revision += 1;
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
        let mut skin = self.skin;
        for &(i, target) in &self.skin_bindings {
            let Some(param) = registry.params().get(i) else { continue };
            set_skin(&mut skin, target, doc.params.get(&param.id).unwrap_or(&param.default));
        }
        if skin != self.skin {
            self.skin = skin;
            self.skin_revision += 1;
        }
        let mut materials = self.materials;
        let mut hair_color_set = false;
        for &(i, target) in &self.material_bindings {
            let Some(param) = registry.params().get(i) else { continue };
            let own = doc.params.get(&param.id);
            hair_color_set |= own.is_some() && matches!(target, MaterialTarget::Hair(_));
            set_material(&mut materials, target, own.unwrap_or(&param.default));
        }
        if materials != self.materials || hair_color_set != self.hair_color_set {
            self.materials = materials;
            self.hair_color_set = hair_color_set;
            self.materials_revision += 1;
        }
        self.sync_features(doc, registry);
        let mut grooms = self.grooms;
        let mut body_hair_set = false;
        for &(i, target) in &self.groom_bindings {
            let Some(param) = registry.params().get(i) else { continue };
            let own = doc.params.get(&param.id);
            body_hair_set |= own.is_some() && target == GroomTarget::BodyDensity;
            if let Some(v) = scalar(own.unwrap_or(&param.default)) {
                grooms.set(target, v);
            }
        }
        if grooms != self.grooms || body_hair_set != self.body_hair_set {
            self.grooms = grooms;
            self.body_hair_set = body_hair_set;
            self.grooms_revision += 1;
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
        self.apply_gaze(doc, registry);
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

    /// Turns the eye joints to the document's gaze preview (`eyes.gaze.yaw` / `pitch`, degrees,
    /// clamped to the eye's limits) after the pose and the rig's rotations.
    fn apply_gaze(&mut self, doc: &Document, registry: &Registry) {
        let angle = |slot: Option<usize>| {
            let p = registry.params().get(slot?)?;
            scalar(doc.params.get(&p.id).unwrap_or(&p.default))
        };
        let (yaw, pitch) = (angle(self.gaze_params[0]).unwrap_or(0.0), angle(self.gaze_params[1]).unwrap_or(0.0));
        if yaw == 0.0 && pitch == 0.0 {
            return;
        }
        let gaze = Gaze { yaw: yaw.to_radians(), pitch: pitch.to_radians() }.clamped(&EyeParams::default());
        for &(j, rest) in &self.eye_joints {
            if let Some(slot) = self.pose.get_mut(j) {
                *slot = gaze.joint_rotation(rest) * *slot;
            }
        }
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

    /// todo 2.3: the document's eye, mouth and hair-colour materials, groom settings and gaze
    /// preview reach the model; an unchanged document is synced without allocating.
    #[test]
    fn todo_2_3_materials_grooms_and_gaze_follow_the_document() {
        use humanart_model::eval::penetration::part_vertices;
        use humanart_model::skin_appearance::materials::Side;
        let reg = super::super::tests::registry();
        let template = Arc::new(super::super::tests::template().clone());
        let mut model = Model::new(Arc::clone(&template), reg);
        let mut d = doc();
        model.sync(&d, reg).unwrap();
        assert!(!model.sync(&d, reg).unwrap(), "an unchanged document changes nothing");
        let (m0, g0) = (model.materials_revision(), model.grooms_revision());
        assert!(!model.hair_color_set() && !model.body_hair_set());
        let right = model.materials().eye_params(Side::Right);
        d.params.insert("eyes.iris.color.l".into(), ParamValue::Color([0.05, 0.2, 0.5]));
        d.params.insert("mouth.teeth.stain".into(), ParamValue::Scalar(0.9));
        d.params.insert("brows_lashes.brows.density".into(), ParamValue::Scalar(0.9));
        d.params.insert("hair.color.eumelanin".into(), ParamValue::Scalar(0.2));
        model.sync(&d, reg).unwrap();
        assert!(model.materials_revision() > m0 && model.grooms_revision() > g0);
        assert_eq!(model.materials().eye_params(Side::Left).iris.outer_color, [0.05, 0.2, 0.5]);
        assert_eq!(model.materials().eye_params(Side::Right), right, "the right eye keeps its colour");
        assert_eq!(model.materials().mouth_params().teeth.stain, 0.9);
        assert!(model.hair_color_set());
        assert!((model.grooms().get(GroomTarget::BrowDensity) - 0.9).abs() < 1e-6);
        let before = model.positions().to_vec();
        d.params.insert("eyes.gaze.yaw".into(), ParamValue::Scalar(20.0));
        assert!(model.sync(&d, reg).unwrap(), "the gaze turns the eyes");
        let moved = |v: u32| (model.positions()[v as usize] - before[v as usize]).length();
        let eye = part_vertices(&template, "eyeball-L").iter().map(|&v| moved(v)).fold(0.0_f32, f32::max);
        let (far, at) =
            (0..before.len() as u32).map(|v| (moved(v), v)).fold((0.0_f32, 0), |a, b| if b.0 > a.0 { b } else { a });
        let region = template
            .regions()
            .vertex_regions
            .get(at as usize)
            .and_then(|&r| template.regions().regions.get(usize::from(r)))
            .map(|r| r.name.as_str());
        println!(
            "[todo 2.3] gaze yaw 20 deg: eyeball moves {:.2} mm, farthest vertex {:.2} mm ({region:?})",
            eye * 1e3,
            far * 1e3
        );
        assert!(eye > 1e-3, "the eyeball turns: {eye}");
        let rev = (model.revision(), model.materials_revision(), model.grooms_revision());
        let counts = humanart_core::alloc::count_in(|| {
            let _ = model.sync(&d, reg);
        });
        assert_eq!(counts.allocations, 0, "[PERF-R1] an unchanged document syncs without allocating");
        assert_eq!(rev, (model.revision(), model.materials_revision(), model.grooms_revision()));
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
