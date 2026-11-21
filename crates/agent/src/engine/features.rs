//! The eyes and the mouth in a view `[EYE-01]`..`[EYE-07]` `[MTH-01]`..`[MTH-04]` `[EVAL-08]`.
//!
//! The template's eye and mouth parts are drawn as their own objects, each in the material class
//! the geometry pass shades analytically (`humanart_render::eye`, `humanart_render::mouth`):
//! both eyeballs as one object, both tear lines as one, both eye-occlusion shells as one (no
//! shadow casting: they are a film and a liquid strip), the teeth, the gums, the tongue, and the
//! body's mouth-interior faces (the mouth bag) taken out of the skin object so the cavity's
//! occlusion darkens them too. [`add_parts`] puts them in the scene; [`Features`] keeps their
//! entries of the renderer's feature table current: when the model's revision moves,
//! [`Features::sync`] fits each eyeball (centre, radius, gaze axis), reads the lid aperture from
//! the lid-margin landmarks, the lip opening from the lip landmarks and the incisal edges from the
//! teeth, all on the CPU result, writes the entries in place and hands the table to the renderer
//! (`Renderer::set_features`). Nothing is allocated after creation `[PERF-R1]`.
//!
//! The penetration guard `[EVAL-08]`: the engine's CPU evaluator runs it ([`penetration_guard`],
//! `Evaluator::set_guard`); the GPU deformation the view draws from does not, so the corrected
//! positions and normals of the guarded vertices are mirrored over the deformer's outputs after
//! every deformation pass ([`Features::encode`], `humanart_render::eye::guard::GuardMirror`).
//!
//! The lash shadow is a hook: [`Features::set_lashes`] (0 = no lashes, the default until the
//! groom draws lashes) darkens the band of the eyeball under the upper lid.

use humanart_gpu::resources::Resources;
use humanart_math::{Quat, Vec3};
use humanart_model::Template;
use humanart_model::eval::penetration::{PenetrationGuard, part_vertices};
use humanart_model::eye::{EyeParams, EyeVertices, LidLandmarks};
use humanart_model::mouth::{LipLandmarks, MouthParams};
use humanart_render::camera::{Camera, ReviewCamera};
use humanart_render::deform::GpuDeformer;
use humanart_render::eye::guard::GuardMirror;
use humanart_render::eye::{
    EYE_CLASS, EYE_OCCLUSION_CLASS, EyeShading, FeatureTable, GUM_CLASS, MOUTH_BAG_CLASS, TEAR_CLASS, TEETH_CLASS,
    TONGUE_CLASS,
};
use humanart_render::mouth::MouthShading;
use humanart_render::renderer::Renderer;
use humanart_render::scene::{Material, Mesh, Scene};

use super::error::EngineError;
use super::model::Model;
use super::skin::SKIN_PART;
use super::view::Framing;

/// The region whose body faces are the mouth bag.
pub const MOUTH_BAG_REGION: &str = "mouth-interior";
/// Shortest crown height the teeth gradient runs over, metres.
const MIN_CROWN: f32 = 0.006;
/// Longest crown height, metres.
const MAX_CROWN: f32 = 0.014;

/// A material of class `class` (its colour shows only where the table has no entry).
const fn part_material(base_color: [f32; 3], roughness: f32, class: u8) -> Material {
    Material { base_color, roughness, metallic: 0.0, f0: 0.02, thickness: 0.5, class }
}

/// What each group of parts is drawn as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Eyeballs,
    TearLines,
    Occlusion,
    Teeth,
    Gums,
    Tongue,
    MouthBag,
}

/// The parts of each role and their material; the part names are the template's stable names.
const GROUPS: [(Role, &[&str], Material); 6] = [
    (Role::Eyeballs, &["eyeball-L", "eyeball-R"], part_material([0.8, 0.78, 0.74], 0.1, EYE_CLASS)),
    (Role::TearLines, &["tear-line-L", "tear-line-R"], part_material([0.6, 0.4, 0.38], 0.06, TEAR_CLASS)),
    (
        Role::Occlusion,
        &["eye-occlusion-L", "eye-occlusion-R"],
        part_material([0.0, 0.0, 0.0], 1.0, EYE_OCCLUSION_CLASS),
    ),
    (Role::Teeth, &["teeth-upper", "teeth-lower"], part_material([0.8, 0.76, 0.66], 0.25, TEETH_CLASS)),
    (Role::Gums, &["gums-upper", "gums-lower"], part_material([0.62, 0.25, 0.25], 0.35, GUM_CLASS)),
    (Role::Tongue, &["tongue"], part_material([0.6, 0.27, 0.27], 0.45, TONGUE_CLASS)),
];
/// The mouth bag's material.
const MOUTH_BAG: Material = part_material([0.5, 0.2, 0.2], 0.3, MOUTH_BAG_CLASS);

/// The scene objects [`add_parts`] made.
#[derive(Clone, Debug, Default)]
pub struct FeatureParts {
    objects: Vec<(usize, Role)>,
    meshes: Vec<usize>,
    eyes: bool,
}

impl FeatureParts {
    /// The meshes to hand to the deformer (one per object).
    pub fn meshes(&self) -> &[usize] {
        &self.meshes
    }
}

/// Whether the template's eyes can be shaded: both eyeballs fit, both lids have their landmarks
/// and there is a head joint.
fn eyes_usable(template: &Template) -> bool {
    let positions = template.mesh().positions();
    let fits = ["eyeball-L", "eyeball-R"]
        .iter()
        .all(|p| EyeVertices::find(template, p).ok().and_then(|v| v.frame(positions)).is_some());
    let lids = ["L", "R"].iter().all(|s| LidLandmarks::find(template.landmarks(), s).is_ok());
    fits && lids && template.skeleton().find("head").is_some()
}

/// Moves the triangles of `skin` whose corners all lie in [`MOUTH_BAG_REGION`] into a mesh of
/// their own (same vertex streams); `None` when there are none.
fn split_mouth_bag(template: &Template, skin: &mut Mesh) -> Option<Mesh> {
    let regions = template.regions();
    let bag_region = regions.find(MOUTH_BAG_REGION)?;
    let in_bag = |v: u32| regions.vertex_regions.get(v as usize).is_some_and(|&r| regions.is_within(r, bag_region));
    let mut kept = Vec::with_capacity(skin.indices.len());
    let mut bag = Vec::new();
    for tri in skin.indices.as_chunks::<3>().0 {
        if tri.iter().all(|&v| in_bag(v)) {
            bag.extend_from_slice(tri);
        } else {
            kept.extend_from_slice(tri);
        }
    }
    if bag.is_empty() {
        return None;
    }
    skin.indices = kept;
    Some(Mesh { indices: bag, ..skin.clone() })
}

/// Adds the eye and mouth objects of `template` to `scene` and takes the mouth bag out of the
/// skin mesh `skin` (the template's [`SKIN_PART`], in the template's vertex order). Parts that
/// are neither skin nor eye or mouth parts are added as one object in `other` (as before). When
/// the eyes cannot be shaded (see the module documentation) the eyeballs are drawn plain and the
/// tear lines and occlusion shells are left out (they would be opaque).
///
/// # Errors
/// [`EngineError::Render`] when a part mesh exceeds the render limits.
pub fn add_parts(
    scene: &mut Scene,
    template: &Template,
    skin: &mut Mesh,
    other: Material,
) -> Result<FeatureParts, EngineError> {
    let tm = template.mesh();
    let eyes = eyes_usable(template);
    let mut out = FeatureParts { eyes, ..FeatureParts::default() };
    let mut push = |scene: &mut Scene, mesh: Mesh, material: Material, role: Option<Role>| {
        let object = scene.add(mesh, material);
        if let Some(o) = scene.objects.get_mut(object) {
            o.casts_shadow = !matches!(role, Some(Role::TearLines | Role::Occlusion));
            out.meshes.push(o.mesh);
        }
        if let Some(role) = role {
            out.objects.push((object, role));
        }
    };
    for (role, names, material) in GROUPS {
        if !eyes && matches!(role, Role::TearLines | Role::Occlusion) {
            continue;
        }
        let mut merged: Option<Mesh> = None;
        for &name in names {
            if !tm.parts().iter().any(|p| p.name == name) {
                continue;
            }
            let piece = Mesh::from_template(tm, Some(name))?;
            match merged.as_mut() {
                Some(m) => m.indices.extend_from_slice(&piece.indices),
                None => merged = Some(piece),
            }
        }
        if let Some(mesh) = merged.filter(|m| !m.indices.is_empty()) {
            let featured = role != Role::Eyeballs || eyes;
            push(scene, mesh, material, featured.then_some(role));
        }
    }
    if let Some(bag) = split_mouth_bag(template, skin) {
        push(scene, bag, MOUTH_BAG, Some(Role::MouthBag));
    }
    let mut rest: Option<Mesh> = None;
    let known = |name: &str| name == SKIN_PART || GROUPS.iter().any(|(_, names, _)| names.contains(&name));
    for part in tm.parts().iter().filter(|p| !known(&p.name)) {
        let piece = Mesh::from_template(tm, Some(&part.name))?;
        match rest.as_mut() {
            Some(r) => r.indices.extend_from_slice(&piece.indices),
            None => rest = Some(piece),
        }
    }
    if let Some(rest) = rest.filter(|r| !r.indices.is_empty()) {
        push(scene, rest, other, None);
    }
    Ok(out)
}

/// The penetration guard of `template` with the default eye, `None` when the template lacks the
/// parts or regions it needs (`PenetrationGuard::new`).
#[must_use]
pub fn penetration_guard(template: &Template) -> Option<PenetrationGuard> {
    PenetrationGuard::new(template, &EyeParams::default()).ok()
}

/// One eye's inputs, found once.
#[derive(Clone, Debug)]
struct EyeSlot {
    vertices: EyeVertices,
    lids: LidLandmarks,
    entry: usize,
}

/// The mouth's inputs, found once.
#[derive(Clone, Debug)]
struct MouthSlot {
    lips: LipLandmarks,
    upper: Vec<u32>,
    lower: Vec<u32>,
    crown: f32,
    entry: usize,
}

/// The feature table of one view and what keeps it current (see the module documentation).
pub struct Features {
    table: FeatureTable,
    eyes: Vec<EyeSlot>,
    shading: Vec<EyeShading>,
    mouth: Option<MouthSlot>,
    head: Option<usize>,
    eye: EyeParams,
    mouth_params: MouthParams,
    lashes: f32,
    mirror: Option<GuardMirror>,
    synced: u64,
}

impl std::fmt::Debug for Features {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Features")
            .field("entries", &self.table.len())
            .field("eyes", &self.eyes.len())
            .field("mouth", &self.mouth.is_some())
            .field("guard_mirror", &self.mirror.is_some())
            .finish_non_exhaustive()
    }
}

impl Features {
    /// The table for the objects of `parts`, filled from `model`'s current result on the first
    /// [`sync`](Self::sync). The guard mirror is made when the model's evaluator runs a guard.
    ///
    /// # Errors
    /// [`EngineError::Render`] for a table or mirror that cannot be made; [`EngineError::Model`]
    /// when the mouth landmarks are malformed.
    pub fn new(
        resources: &mut Resources,
        model: &Model,
        parts: &FeatureParts,
        deformer: &GpuDeformer,
    ) -> Result<Self, EngineError> {
        let template = model.template();
        let landmarks = template.landmarks();
        let positions = model.positions();
        let mut table = FeatureTable::new();
        let mut eyes = Vec::new();
        let mut shading = Vec::new();
        let head = template.skeleton().find("head").map(usize::from);
        if parts.eyes {
            for (part, side) in [("eyeball-L", "L"), ("eyeball-R", "R")] {
                let vertices = EyeVertices::find(template, part)?;
                let lids = LidLandmarks::find(landmarks, side)?;
                let frame = vertices
                    .frame(positions)
                    .ok_or(EngineError::Invalid { what: "an eyeball of the template does not fit a sphere" })?;
                let s = EyeShading {
                    frame,
                    head: Quat::IDENTITY,
                    params: EyeParams::default(),
                    aperture: lids.aperture(landmarks, positions, frame.centre, Quat::IDENTITY)?,
                    lashes: 0.0,
                    partner: None,
                };
                let entry = table.add(&s.entry()?)?;
                eyes.push(EyeSlot { vertices, lids, entry });
                shading.push(s);
            }
            if let [a, b] = eyes.as_slice() {
                let (ea, eb) = (a.entry, b.entry);
                shading[0].partner = Some(eb);
                shading[1].partner = Some(ea);
            }
        }
        let mut mouth = None;
        let has_mouth_object = parts.objects.iter().any(|(_, r)| matches!(r, Role::Teeth | Role::Gums | Role::Tongue));
        if has_mouth_object && let Ok(lips) = LipLandmarks::find(landmarks) {
            let upper = part_vertices(template, "teeth-upper");
            let lower = part_vertices(template, "teeth-lower");
            let rest = template.mesh().positions();
            let (lo, hi) = upper
                .iter()
                .filter_map(|&v| rest.get(v as usize))
                .fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p.y), hi.max(p.y)));
            let crown = if hi > lo { (hi - lo).clamp(MIN_CROWN, MAX_CROWN) } else { 0.0105 };
            let mut slot = MouthSlot { lips, upper, lower, crown, entry: 0 };
            let first = mouth_shading(&slot, template, positions, Quat::IDENTITY, &MouthParams::default())?;
            slot.entry = table.add(&first.entry()?)?;
            mouth = Some(slot);
        }
        for &(object, role) in &parts.objects {
            let entry = match role {
                Role::Eyeballs | Role::TearLines | Role::Occlusion => eyes.first().map(|e| e.entry),
                Role::Teeth | Role::Gums | Role::Tongue | Role::MouthBag => mouth.as_ref().map(|m| m.entry),
            };
            table.assign(object, entry)?;
        }
        let mirror = match model.evaluator().guard() {
            Some(guard) => Some(GuardMirror::new(resources, deformer, guard.guarded())?),
            None => None,
        };
        Ok(Self {
            table,
            eyes,
            shading,
            mouth,
            head,
            eye: EyeParams::default(),
            mouth_params: MouthParams::default(),
            lashes: 0.0,
            mirror,
            synced: 0,
        })
    }

    /// The table (entries and object slots).
    pub fn table(&self) -> &FeatureTable {
        &self.table
    }

    /// The lash density whose shadow falls on the eyeballs (0 to 1): the groom's hook.
    pub fn set_lashes(&mut self, density: f32) {
        let d = if density.is_finite() { density.clamp(0.0, 1.0) } else { 0.0 };
        if d != self.lashes {
            self.lashes = d;
            self.synced = 0;
        }
    }

    /// The eye parameters (shape, iris, sclera) of both eyes.
    pub fn set_eye_params(&mut self, params: EyeParams) {
        self.eye = params;
        self.synced = 0;
    }

    /// The teeth, gums and tongue parameters.
    pub fn set_mouth_params(&mut self, params: MouthParams) {
        self.mouth_params = params;
        self.synced = 0;
    }

    /// Brings the entries and the guard mirror to `model`'s current result when its revision
    /// moved (or a parameter changed) and hands the table to `renderer`. An eye or mouth whose
    /// inputs are degenerate in this result keeps its last entry. No allocation.
    ///
    /// # Errors
    /// [`EngineError::Render`] when the table or the mirror cannot be written.
    pub fn sync(
        &mut self,
        resources: &mut Resources,
        renderer: &mut Renderer,
        model: &Model,
    ) -> Result<(), EngineError> {
        if self.synced == model.revision() {
            return Ok(());
        }
        let template = model.template();
        let landmarks = template.landmarks();
        let positions = model.positions();
        let head =
            self.head.and_then(|h| model.evaluator().joints().skin_rotations.get(h).copied()).unwrap_or(Quat::IDENTITY);
        for (slot, shading) in self.eyes.iter().zip(self.shading.iter_mut()) {
            let Some(frame) = slot.vertices.frame(positions) else { continue };
            let Ok(aperture) = slot.lids.aperture(landmarks, positions, frame.centre, head) else { continue };
            let next = EyeShading { frame, head, params: self.eye, aperture, lashes: self.lashes, ..*shading };
            if let Ok(entry) = next.entry() {
                self.table.set(slot.entry, &entry)?;
                *shading = next;
            }
        }
        if let Some(slot) = &self.mouth
            && let Ok(m) = mouth_shading(slot, template, positions, head, &self.mouth_params)
            && let Ok(entry) = m.entry()
        {
            self.table.set(slot.entry, &entry)?;
        }
        renderer.set_features(resources, &self.table)?;
        if let Some(mirror) = &mut self.mirror {
            mirror.update(resources, positions, model.normals())?;
        }
        self.synced = model.revision();
        Ok(())
    }

    /// Records the guard mirror's dispatch; call it right after a recorded deformation pass.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(mirror) = &self.mirror {
            mirror.encode(encoder);
        }
    }

    /// Frees the mirror's buffers.
    pub fn destroy(self, resources: &mut Resources) {
        if let Some(mirror) = self.mirror {
            mirror.destroy(resources);
        }
    }
}

/// A close-up for the macro stills of the eyes and the mouth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CloseUp {
    /// The character's left eye with its lids and canthi.
    Eye,
    /// The lips and the mouth's interior.
    Mouth,
}

/// Half the extent a close-up frames, metres.
const EYE_CLOSE_UP: f32 = 0.017;
const MOUTH_CLOSE_UP: f32 = 0.032;

/// The camera of `close_up` on `model`'s current result, from the direction of `framing`
/// (front for [`Framing::Body`] and [`Framing::Head`]), at aspect ratio `aspect`. `None` when the
/// template has no fitted left eyeball (eye) or no lip landmarks (mouth).
#[must_use]
pub fn close_up_camera(model: &Model, close_up: CloseUp, framing: Framing, aspect: f32) -> Option<Camera> {
    let template = model.template();
    let positions = model.positions();
    let (centre, radius) = match close_up {
        CloseUp::Eye => {
            let frame = EyeVertices::find(template, "eyeball-L").ok()?.frame(positions)?;
            (frame.centre + frame.axis * (0.6 * frame.radius), EYE_CLOSE_UP)
        }
        CloseUp::Mouth => {
            let lips = LipLandmarks::find(template.landmarks()).ok()?;
            let opening = lips.opening(template.landmarks(), positions, Quat::IDENTITY).ok()?;
            (opening.centre, MOUTH_CLOSE_UP)
        }
    };
    let kind = match framing {
        Framing::Body | Framing::Head => ReviewCamera::Front,
        Framing::ThreeQuarter => ReviewCamera::ThreeQuarterLeft,
        Framing::Profile => ReviewCamera::ProfileLeft,
    };
    Some(Camera::review(kind, centre, radius, aspect))
}

/// The mouth's shading inputs on `positions` with the head rotated by `head`.
fn mouth_shading(
    slot: &MouthSlot,
    template: &Template,
    positions: &[Vec3],
    head: Quat,
    params: &MouthParams,
) -> Result<MouthShading, EngineError> {
    let opening = slot.lips.opening(template.landmarks(), positions, head)?;
    let up = head.mul_vec3(Vec3::Y).normalize_or(Vec3::Y);
    let extreme = |vertices: &[u32], lowest: bool| {
        vertices.iter().filter_map(|&v| positions.get(v as usize).copied()).reduce(|a, b| {
            let lower = b.dot(up) < a.dot(up);
            if lower == lowest { b } else { a }
        })
    };
    let upper_edge = extreme(&slot.upper, true).unwrap_or(opening.centre + up * 0.002);
    let lower_edge = extreme(&slot.lower, false).unwrap_or(opening.centre - up * 0.002);
    Ok(MouthShading { opening, up, upper_edge, lower_edge, crown_height: slot.crown, params: *params })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use std::sync::Arc;

    use humanart_model::generate::build::{Options, template};

    use super::*;

    #[test]
    fn parts_become_objects_and_the_mouth_bag_leaves_the_skin() {
        let t = Arc::new(template(&Options { lods: false, ..Options::default() }).unwrap().template);
        assert!(eyes_usable(&t), "the generated template has fitted eyeballs, lid landmarks and a head");
        let mut skin = Mesh::from_template(t.mesh(), Some(SKIN_PART)).unwrap();
        let before = skin.indices.len();
        let mut scene = Scene::default();
        let parts = add_parts(&mut scene, &t, &mut skin, part_material([0.5; 3], 0.5, 0)).unwrap();
        let roles: Vec<Role> = parts.objects.iter().map(|o| o.1).collect();
        for r in
            [Role::Eyeballs, Role::TearLines, Role::Occlusion, Role::Teeth, Role::Gums, Role::Tongue, Role::MouthBag]
        {
            assert!(roles.contains(&r), "{r:?} missing from {roles:?}");
        }
        let bag = parts.objects.iter().find(|o| o.1 == Role::MouthBag).unwrap().0;
        let bag_tris = scene.meshes[scene.objects[bag].mesh].indices.len();
        assert!(bag_tris > 0 && skin.indices.len() + bag_tris == before, "every skin triangle is kept once");
        for (object, role) in &parts.objects {
            let shadow = scene.objects[*object].casts_shadow;
            assert_eq!(shadow, !matches!(role, Role::TearLines | Role::Occlusion), "{role:?}");
        }
        assert_eq!(parts.meshes().len(), scene.objects.len(), "every object's mesh is handed to the deformer");
    }

    #[test]
    fn eye_05_the_lid_aperture_of_the_template_frames_the_visible_eyeball() {
        let t = Arc::new(template(&Options { lods: false, ..Options::default() }).unwrap().template);
        let pos = t.mesh().positions();
        let frame = EyeVertices::find(&t, "eyeball-L").unwrap().frame(pos).unwrap();
        let lids = LidLandmarks::find(t.landmarks(), "L").unwrap();
        let a = lids.aperture(t.landmarks(), pos, frame.centre, Quat::IDENTITY).unwrap();
        println!("frame {frame:?}\naperture {a:?}");
        for l in lids.upper.iter().chain(&lids.lower) {
            let p = t.landmarks()[*l].position(pos).unwrap();
            println!(
                "{} {:?}",
                t.landmarks()[*l].name,
                humanart_model::eye::aperture::angles(frame.centre, Quat::IDENTITY, p)
            );
        }
        let apex = humanart_model::eye::aperture::angles(frame.centre, Quat::IDENTITY, frame.centre + frame.axis);
        println!("axis angles {apex:?}, openness there {}", a.openness(apex.0, apex.1));
        assert!(a.openness(apex.0, apex.1) > 0.05);
    }

    /// The closest point to `p` on triangle `(a, b, c)` (Ericson, Real-Time Collision Detection 5.1.5).
    fn closest_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
        let (ab, ac, ap) = (b - a, c - a, p - a);
        let (d1, d2) = (ab.dot(ap), ac.dot(ap));
        if d1 <= 0.0 && d2 <= 0.0 {
            return a;
        }
        let bp = p - b;
        let (d3, d4) = (ab.dot(bp), ac.dot(bp));
        if d3 >= 0.0 && d4 <= d3 {
            return b;
        }
        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            return a + ab * (d1 / (d1 - d3));
        }
        let cp = p - c;
        let (d5, d6) = (ab.dot(cp), ac.dot(cp));
        if d6 >= 0.0 && d5 <= d6 {
            return c;
        }
        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            return a + ac * (d2 / (d2 - d6));
        }
        let va = d3 * d6 - d5 * d4;
        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
            return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
        }
        let denom = 1.0 / (va + vb + vc);
        a + ab * (vb * denom) + ac * (vc * denom)
    }

    /// The engine's model with `params` set (registry ids and values), synced.
    fn posed(params: &[(&str, f64)]) -> Model {
        use humanart_doc::document::{Author, AuthorKind, Document};
        use humanart_doc::registry::ParamValue;
        let reg = super::super::tests::registry();
        let mut model = Model::new(Arc::new(super::super::tests::template().clone()), reg);
        let me = Author { id: "human.test".into(), kind: AuthorKind::Human, name: "Test".into() };
        let mut doc = Document::new(humanart_core::id::Id128::from_u128(3), "t", "generated", 1, 0, me);
        for (id, v) in params {
            doc.params.insert((*id).to_owned(), ParamValue::Scalar(*v));
        }
        model.sync(&doc, reg).unwrap();
        model
    }

    /// todo 1.5: the viewer that the editor viewport (`crates/app` studio, `Viewer::new`) and
    /// `humanart-cli render` (`Viewer::with_options`, surfaced skin) both build hands the eye and
    /// mouth table to the renderer, and the engine's model runs the penetration guard in its
    /// evaluator `[EVAL-08]`. The proof on the GPU: the eye close-up's centre pixel is the traced
    /// pupil (near black); without the table the eyeball would be its plain light grey material.
    #[test]
    fn todo_1_5_the_viewer_shades_the_traced_eye_and_the_model_runs_the_guard() {
        use humanart_gpu::resources::Budget;

        use super::super::view::Viewer;
        let mut model = posed(&[]);
        assert!(model.evaluator().guard().is_some(), "Model::new sets the penetration guard on its Evaluator");
        let gpu = humanart_gpu::shared_headless().unwrap();
        let mut res = Resources::new(gpu, Budget::UNLIMITED);
        let (w, h) = (96, 64);
        let mut viewer = Viewer::new(gpu, &mut res, &model, w, h).unwrap();
        assert!(viewer.features_error().is_none(), "{:?}", viewer.features_error());
        assert_eq!(viewer.features_mut().unwrap().table().len(), 3, "two eyes and the mouth");
        #[allow(clippy::cast_precision_loss)]
        let camera = close_up_camera(&model, CloseUp::Eye, Framing::Head, w as f32 / h as f32).unwrap();
        viewer.set_camera(camera);
        for _ in 0..4 {
            let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("t") });
            viewer.record(&mut res, &mut enc, &mut model, &[]).unwrap();
            gpu.queue.submit([enc.finish()]);
            viewer.after_submit();
        }
        let (tw, th, rgba) =
            humanart_render::capture::read_texture(&mut res, viewer.display_texture().unwrap(), 0).unwrap();
        let at = |x: u32, y: u32| {
            let i = ((y * tw + x) * 4) as usize;
            (u32::from(rgba[i]) + u32::from(rgba[i + 1]) + u32::from(rgba[i + 2])) / 3
        };
        let (pupil, sclera) = (at(tw / 2, th / 2), at(tw / 2 + tw / 4, th / 2));
        println!("[todo 1.5] eye close-up: pupil {pupil}/255, sclera side {sclera}/255");
        viewer.destroy(&mut res);
        assert!(pupil < 40, "the traced pupil is dark (plain eyeball material would be light): {pupil}");
        assert!(sclera > pupil + 40, "the sclera beside the iris is light: {sclera} vs {pupil}");
    }

    /// The incisor show and the molar clearance with the jaw open `[MTH-04]` (todo 1.6): the
    /// upper incisal edge against the upper lip's inner margin at the midline, and the lower
    /// molars against the skin (cheeks and mouth bag), whose outward side is the cavity.
    #[test]
    fn mth_04_incisor_show_and_molar_cheek_clearance_with_the_jaw_open() {
        let t = super::super::tests::template();
        let skin = Mesh::from_template(t.mesh(), Some(SKIN_PART)).unwrap();
        let lips = LipLandmarks::find(t.landmarks()).unwrap();
        let upper_teeth = part_vertices(t, "teeth-upper");
        let lower_teeth = part_vertices(t, "teeth-lower");
        let rest = t.mesh().positions();
        let molars: Vec<u32> = lower_teeth.iter().copied().filter(|&v| rest[v as usize].x.abs() > 0.018).collect();
        assert!(!molars.is_empty());
        let mut worst = f32::MAX;
        for jaw in [0.0, 0.5, 0.9, 1.0] {
            let model = posed(&[("expression.au26_jaw_drop", jaw)]);
            let p = model.positions();
            let mid = |list: &[usize]| {
                let at = |l: usize| t.landmarks()[l].position(p).unwrap();
                at(*list.iter().min_by(|a, b| at(**a).x.abs().total_cmp(&at(**b).x.abs())).unwrap())
            };
            let (lip_up, lip_low) = (mid(&lips.upper), mid(&lips.lower));
            let near_mid = |v: &&u32| p[**v as usize].x.abs() < 0.004;
            let incisal_up = upper_teeth.iter().filter(near_mid).map(|&v| p[v as usize].y).fold(f32::MAX, f32::min);
            let incisal_low = lower_teeth.iter().filter(near_mid).map(|&v| p[v as usize].y).fold(f32::MIN, f32::max);
            // Lower molars against every skin triangle within 2 cm: the signed distance along the
            // nearest triangle's outward (cavity-side) normal.
            let (mut closest, mut signed) = (f32::MAX, f32::MAX);
            for &v in &molars {
                let q = p[v as usize];
                for tri in skin.indices.as_chunks::<3>().0 {
                    let [a, b, c] = tri.map(|i| p[i as usize]);
                    if (a - q).length() > 0.02 {
                        continue;
                    }
                    let on = closest_on_triangle(q, a, b, c);
                    let d = (q - on).length();
                    if d < closest {
                        closest = d;
                        signed = (q - on).dot((b - a).cross(c - a).normalize_or_zero());
                    }
                }
            }
            println!(
                "[MTH-04] jaw drop {jaw}: upper lip inner margin y {:.1} mm, upper incisal edge {:.1} mm, show {:.1} mm; \
                 lower lip margin {:.1} mm, lower incisal edge {:.1} mm; lip gap {:.1} mm; lower molars: nearest skin \
                 {:.2} mm (signed {:.2} mm, negative = inside the tissue)",
                lip_up.y * 1e3,
                incisal_up * 1e3,
                (lip_up.y - incisal_up) * 1e3,
                lip_low.y * 1e3,
                incisal_low * 1e3,
                (lip_up.y - lip_low.y) * 1e3,
                closest * 1e3,
                signed * 1e3
            );
            worst = worst.min(signed);
        }
        assert!(worst > 0.0, "a lower molar is inside the cheek ({:.2} mm)", worst * 1e3);
    }
}
