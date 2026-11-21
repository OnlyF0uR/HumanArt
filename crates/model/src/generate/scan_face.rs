//! The scan-derived face `[WP-25]` (todo 1.1): the mean face of several HSRD-100 scans, applied to
//! the generated head.
//!
//! `data/template/face/face-model.hafm` ([`crate::scanfit::face_model`], built by
//! `humanart-cli face-model` from `fit-scan` results) holds, per node of the procedural head and
//! neck, the mean offset of the fitted subjects from this generator's own head, already blended (1
//! in the face, fading over the lower neck, 0 under the scalp, where the scans show hair or caps).
//! [`apply`] runs after the landmarks are found (they are barycentric, so they move with the
//! surface) and before the rig, the identity components and the expressions are built on the mesh:
//!
//! - every `body` vertex within [`MAX_DISTANCE`] of the model's rest surface moves by the
//!   interpolated mean offset;
//! - the rigid parts move as wholes so they keep their shape: each eyeball, tear line and eye
//!   occlusion part by the mean offset of its eye's lid and canthus landmarks, the teeth, gums and
//!   tongue by the mean offset of the lip landmarks;
//! - the skin around each eye (within [`EYE_RIGID_MARGIN`] of the eyeball surface) moves with its
//!   eye, fading to the field over [`EYE_BLEND`]: the lids, the lid helper weights, the lashes and
//!   the eyelid guard were designed around the procedural eyeball, and a field offset that differs
//!   across the lid by a millimetre pushes the lid margin into the eye (lashes, blink);
//! - vertices of any other part within [`MAX_PART_DISTANCE`] follow the field like the body;
//! - the offsets are made vertex-exactly mirror-symmetric: the field is symmetric only up to its
//!   interpolation, and mirror editing, the symmetric identity components and the symmetric tests
//!   need a template whose every vertex has an exact mirror image. Natural asymmetry belongs to the
//!   `asymmetry.*` identity components, not to the template.
//!
//! The caller moves the eye joints and lid helpers with the eyes: [`ScanFaceReport::eyes`] is the
//! left eye's translation ([`Anatomy::eye_center`] is the left eye), the right one its mirror.

use std::collections::HashMap;

use humanart_math::Vec3;

use super::anatomy::{Anatomy, mirror, sided};
use super::assemble::Assembled;
use super::landmarks::Described;
use crate::error::{ModelError, Result};
use crate::mesh::Mesh;
use crate::scanfit::face_model::{FaceField, FaceModel, FaceSample};
use crate::template::ProvenanceEntry;

/// The face model file, embedded.
pub const FACE_MODEL_BYTES: &[u8] = include_bytes!("../../../../data/template/face/face-model.hafm");
/// The farthest a body vertex may lie from the model's rest surface and still move, metres.
pub const MAX_DISTANCE: f32 = 0.002;
/// The same for the vertices of the non-rigid parts, metres.
pub const MAX_PART_DISTANCE: f32 = 0.005;
/// The radius over which the field offsets of a non-rigid part are averaged, metres.
pub const PART_SMOOTHING: f32 = 0.003;
/// Skin within this distance of the eyeball surface moves rigidly with its eye, metres: the
/// footprint of the lid helper weights (`rig::skin`, full weight up to 1.7 mm and none past 5 mm
/// outside the eyeball) plus the lid's thickness.
pub const EYE_RIGID_MARGIN: f32 = 0.006;
/// Width of the fade from the eye's translation to the field outside [`EYE_RIGID_MARGIN`], metres.
pub const EYE_BLEND: f32 = 0.012;
/// The face mask's fade in front of the ears: head-frame depth (millimetres in front of the
/// tragion midpoint) where it starts and where it is full ([`face_mask`]).
pub const FACE_DEPTH: (f32, f32) = (10.0, 45.0);
/// The face mask's fade under the chin: head-frame height (millimetres) where it starts and where
/// it is full (the chin's lowest point is 81 mm below the tragions).
pub const FACE_BOTTOM: (f32, f32) = (-120.0, -85.0);
/// The face mask's fade at the hairline: head-frame height where it starts and where it is gone.
pub const FACE_TOP: (f32, f32) = (60.0, 95.0);
/// The component mask's fade in front of the ears (head-frame depth, mm; [`component_mask`]).
pub const COMPONENT_DEPTH: (f32, f32) = (10.0, 70.0);
/// The component mask's fade under the chin (head-frame height, mm).
pub const COMPONENT_BOTTOM: (f32, f32) = (-120.0, -80.0);
/// The component mask's fade toward the hairline (head-frame height, mm).
pub const COMPONENT_TOP: (f32, f32) = (35.0, 95.0);
/// The lip closure ([`close_lips`]): full within `.0` metres of the contact line, fading out over the
/// next `.1` metres.
pub const LIP_CLOSURE: (f32, f32) = (0.002, 0.006);
/// How deep behind the contact line the lip closure reaches, metres (the inner lip lies 5 to 8 mm
/// behind it).
pub const LIP_DEPTH: f32 = 0.010;
/// Width of the Gaussian with which the mouth interior, teeth, gums and tongue follow the skin
/// around them, metres ([`interior_offset`]).
pub const INTERIOR_KERNEL: f32 = 0.008;
/// Two vertices are mirror images when one lies this close to the other's mirrored position,
/// metres (the generated surface is mirror-exact to 0.05 mm, `mod_01`).
pub const MIRROR_TOLERANCE: f32 = 1.0e-4;

/// What [`apply`] did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScanFaceReport {
    /// Body vertices moved.
    pub moved: usize,
    /// Largest body vertex offset, metres.
    pub max_offset: f32,
    /// Translation of the left and right eye parts, metres (mirror images of each other).
    pub eyes: [Vec3; 2],
    /// Translation of the upper teeth and gums, and of the lower teeth, gums and tongue, metres
    /// (no sideways part).
    pub mouth: [Vec3; 2],
    /// Vertices of the mouth interior that follow the skin around them.
    pub interior: usize,
    /// Vertices whose offset was averaged with their mirror image's.
    pub mirrored: usize,
    /// Vertices without a mirror image (their offset is used as it is).
    pub unpaired: usize,
    /// Largest left-right difference of the field offsets before symmetrising, metres.
    pub max_asymmetry: f32,
    /// Largest difference of the field offsets of the upper and lower lip contact lines that
    /// [`close_lips`] took out, metres.
    pub lip_gap: f32,
    /// The subjects the model averages.
    pub subjects: u32,
}

/// The rigid group of a part: 0 left eye, 1 right eye, 2 upper jaw (teeth, gums), 3 lower jaw
/// (teeth, gums, tongue).
fn rigid_group(name: &str) -> Option<usize> {
    let eye = name.starts_with("eyeball") || name.starts_with("tear-line") || name.starts_with("eye-occlusion");
    if eye && name.ends_with("-L") {
        Some(0)
    } else if eye && name.ends_with("-R") {
        Some(1)
    } else if name == "teeth-upper" || name == "gums-upper" {
        Some(2)
    } else if name.starts_with("teeth") || name.starts_with("gums") || name == "tongue" {
        Some(3)
    } else {
        None
    }
}

/// The landmarks whose mean offset moves each rigid group.
const GROUP_LANDMARKS: [&[&str]; 2] = [
    &["L.eye.canthus.lateral", "L.eye.canthus.medial", "L.eye.upper.04", "L.eye.lower.04"],
    &["R.eye.canthus.lateral", "R.eye.canthus.medial", "R.eye.upper.04", "R.eye.lower.04"],
];

fn invalid(what: &'static str) -> ModelError {
    ModelError::invalid("scan face", 0, what)
}

/// Where the scans show skin: `1` on the face, fading to `0` toward the ears (head-frame depth
/// [`FACE_DEPTH`]), the neck ([`FACE_BOTTOM`]) and the hairline ([`FACE_TOP`]). Behind the face
/// the subjects' hair, caps and head pose dominate the mean (it pushed the nape out by 19 mm and
/// the sides of the head by 6 to 9 mm, a step where the model's surface ends).
fn face_mask(p: Vec3, anatomy: &Anatomy) -> f32 {
    let h = (p - anatomy.head_origin) * 1000.0;
    let ramp = |x: f32, (a, b): (f32, f32)| smoothstep((x - a) / (b - a));
    ramp(h.z, FACE_DEPTH) * ramp(h.y, FACE_BOTTOM) * (1.0 - ramp(h.y, FACE_TOP))
}

/// The mask of the shape components: inside [`face_mask`] with wider fades
/// ([`COMPONENT_DEPTH`], [`COMPONENT_BOTTOM`], [`COMPONENT_TOP`]). The components' largest
/// differences between the subjects lie at the face's edges (hair, ears, pose), so the face mask's
/// fades showed as creases at the ends of a slider's range.
fn component_mask(p: Vec3, anatomy: &Anatomy) -> f32 {
    let h = (p - anatomy.head_origin) * 1000.0;
    let ramp = |x: f32, (a, b): (f32, f32)| smoothstep((x - a) / (b - a));
    ramp(h.z, COMPONENT_DEPTH) * ramp(h.y, COMPONENT_BOTTOM) * (1.0 - ramp(h.y, COMPONENT_TOP))
}

/// `0` at `x <= 0`, `1` at `x >= 1`, smooth between.
fn smoothstep(x: f32) -> f32 {
    let t = x.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The template's provenance entry for the face shape `[LIC-02]`: the embedded model's own
/// description (subjects, dataset, licence and attribution).
///
/// # Errors
/// [`ModelError`] when the embedded model is malformed.
pub fn provenance() -> Result<ProvenanceEntry> {
    let model = FaceModel::parse(FACE_MODEL_BYTES).map_err(|_| invalid("the embedded face model is malformed"))?;
    Ok(ProvenanceEntry {
        item: "face shape".to_owned(),
        source: model.description,
        licence: "CC BY 4.0 (HSRD-100, Digital Reality Lab 2025): attribution required".to_owned(),
    })
}

/// One shape component of the face model carried to the template ([`apply`]): per template vertex
/// the displacement of `+1` standard deviation, and the left eye's translation (the eye joints and
/// lid helpers move by it; the right eye's is its mirror image).
#[derive(Clone, Debug, PartialEq)]
pub struct ScanComponent {
    /// Its index in the face model (0 explains the most variance).
    pub index: usize,
    /// The share of the subjects' shape variance it explains, `0..=1`.
    pub variance: f32,
    /// Per template vertex, metres.
    pub offsets: Vec<Vec3>,
    /// The left eye's translation, metres.
    pub eye: Vec3,
}

/// Applies the embedded face model to the assembled template (module documentation) and returns
/// its shape components carried to the template the same way (sampled before the move, where the
/// model's nodes are).
///
/// # Errors
/// [`ModelError`] when the embedded model is malformed or was built for another template, or a
/// rigid group's landmarks are missing.
pub fn apply(
    asm: &mut Assembled,
    described: &[Described],
    anatomy: &Anatomy,
) -> Result<(ScanFaceReport, Vec<ScanComponent>)> {
    let model = FaceModel::parse(FACE_MODEL_BYTES).map_err(|_| invalid("the embedded face model is malformed"))?;
    apply_model(asm, described, anatomy, &model)
}

/// [`apply`] with a given model.
///
/// # Errors
/// As [`apply`].
pub fn apply_model(
    asm: &mut Assembled,
    described: &[Described],
    anatomy: &Anatomy,
    model: &FaceModel,
) -> Result<(ScanFaceReport, Vec<ScanComponent>)> {
    if model.template_name != super::build::TEMPLATE_NAME {
        return Err(invalid("the face model was built for another template"));
    }
    let field = model.field().map_err(|_| invalid("the face model's surface is malformed"))?;
    let transfer = Transfer::new(asm, anatomy, &field);
    let mean = transfer.run(asm, described, &|s| field.mean(s), false)?;
    let mut components = Vec::with_capacity(model.components.len());
    for (index, c) in model.components.iter().enumerate() {
        let t = transfer.run(asm, described, &|s| field.component(s, index), true)?;
        components.push(ScanComponent { index, variance: c.variance, offsets: t.offsets, eye: t.groups[0] });
    }
    let mut report = ScanFaceReport {
        moved: transfer.body.iter().filter(|b| b.is_some()).count(),
        eyes: [mean.groups[0], mean.groups[1]],
        mouth: [mean.groups[2], mean.groups[3]],
        interior: transfer.interior.iter().filter(|&&i| i).count(),
        mirrored: transfer.pairs.iter().filter(|&&m| m != u32::MAX).count(),
        unpaired: transfer.pairs.iter().filter(|&&m| m == u32::MAX).count(),
        max_asymmetry: mean.max_asymmetry,
        lip_gap: mean.lip_gap,
        subjects: model.subjects,
        ..ScanFaceReport::default()
    };
    let mut data = asm.mesh.clone().into_data();
    for (v, (p, o)) in data.positions.iter_mut().zip(&mean.offsets).enumerate() {
        *p += *o;
        if asm.source.get(v).is_some_and(|s| s.0 == 0) {
            report.max_offset = report.max_offset.max(o.length());
        }
    }
    asm.mesh = Mesh::new(data)?;
    Ok((report, components))
}

/// A table of the face model carried to the template: per vertex offsets and the rigid groups'
/// translations.
struct Transferred {
    offsets: Vec<Vec3>,
    /// Left eye, right eye, upper jaw, lower jaw.
    groups: [Vec3; 4],
    lip_gap: f32,
    max_asymmetry: f32,
}

/// A non-body part in a [`Transfer`]: its rigid group, its vertices and, for a non-rigid part, per
/// vertex the closest model points of the part's vertices within [`PART_SMOOTHING`].
type PartTransfer = (Option<usize>, Vec<u32>, Vec<Vec<FaceSample>>);

/// What carrying a table of the face model to the template needs, found once (module
/// documentation): the closest model points, masks and weights, the mouth interior, the parts and
/// the mirror pairs.
struct Transfer {
    /// Per vertex: the closest model point of a body vertex on the face, its [`face_mask`] and its
    /// [`component_mask`].
    body: Vec<Option<(FaceSample, f32, f32)>>,
    /// Per vertex: the weight with which a body vertex moves with its eye, and the eye's group.
    eye: Vec<(f32, usize)>,
    /// Per vertex: whether it is a body vertex of the mouth interior.
    interior: Vec<bool>,
    /// The skin vertices around the mouth interior.
    sources: Vec<u32>,
    /// The centroids of the upper and lower jaw parts.
    jaws: [Vec3; 2],
    /// The non-body parts.
    parts: Vec<PartTransfer>,
    /// Per vertex, its mirror image (`u32::MAX` for none).
    pairs: Vec<u32>,
}

impl Transfer {
    fn new(asm: &Assembled, anatomy: &Anatomy, field: &FaceField<'_>) -> Self {
        let mesh = &asm.mesh;
        let positions = mesh.positions();
        let body_vertex = |v: usize| asm.source.get(v).is_some_and(|s| s.0 == 0);
        let body = positions
            .iter()
            .enumerate()
            .map(|(v, &p)| {
                let mask = face_mask(p, anatomy);
                if !body_vertex(v) || mask <= 0.0 {
                    return None;
                }
                field.sample(p, MAX_DISTANCE).map(|s| (s, mask, component_mask(p, anatomy)))
            })
            .collect();
        let rigid = anatomy.eye_radius + EYE_RIGID_MARGIN;
        let eye = positions
            .iter()
            .enumerate()
            .map(|(v, &p)| {
                let (side, group) = if p.x >= 0.0 { (1.0, 0) } else { (-1.0, 1) };
                let d = p.distance(sided(anatomy.eye_center, side));
                let w = if body_vertex(v) { smoothstep((rigid + EYE_BLEND - d) / EYE_BLEND) } else { 0.0 };
                (w, group)
            })
            .collect();
        let interior = mouth_interior(asm);
        let sources = skin_sources(asm, &interior);
        let parts = mesh.parts();
        let mut jaws = [Vec3::ZERO; 2];
        for (g, jaw) in jaws.iter_mut().enumerate() {
            let (sum, n) = parts
                .iter()
                .filter(|p| rigid_group(&p.name) == Some(g + 2))
                .flat_map(|p| p.faces().flat_map(|f| mesh.face(f).iter().copied()))
                .filter_map(|v| positions.get(v as usize))
                .fold((Vec3::ZERO, 0_u32), |(s, n), &p| (s + p, n + 1));
            *jaw = sum / n.max(1) as f32;
        }
        let mut part_of = vec![u16::MAX; positions.len()];
        let mut part_list = Vec::new();
        for (index, part) in parts.iter().enumerate() {
            let mut vertices: Vec<u32> = part.faces().flat_map(|f| mesh.face(f).iter().copied()).collect();
            vertices.sort_unstable();
            vertices.dedup();
            for &v in &vertices {
                if let Some(slot) = part_of.get_mut(v as usize) {
                    *slot = index as u16;
                }
            }
            if part.name == "body" {
                continue;
            }
            let group = rigid_group(&part.name);
            let mut near = Vec::new();
            if group.is_none() {
                // A thin part (the lip-line helper is a 0.4 mm tube) would be flattened by a field
                // that differs across it: each vertex takes the mean of the part within
                // [`PART_SMOOTHING`].
                let local: Vec<(Vec3, Option<FaceSample>)> = vertices
                    .iter()
                    .filter_map(|&v| positions.get(v as usize))
                    .map(|&p| (p, field.sample(p, MAX_PART_DISTANCE)))
                    .collect();
                near = local
                    .iter()
                    .map(|&(p, _)| {
                        local.iter().filter(|(q, _)| q.distance(p) <= PART_SMOOTHING).filter_map(|&(_, s)| s).collect()
                    })
                    .collect();
            }
            part_list.push((group, vertices, near));
        }
        let mirror_part: Vec<u16> = parts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let swapped = p
                    .name
                    .strip_suffix("-L")
                    .map(|b| format!("{b}-R"))
                    .or_else(|| p.name.strip_suffix("-R").map(|b| format!("{b}-L")))
                    .unwrap_or_else(|| p.name.clone());
                parts.iter().position(|q| q.name == swapped).unwrap_or(i) as u16
            })
            .collect();
        let pairs = mirror_pairs(positions, &part_of, &mirror_part);
        Self { body, eye, interior, sources, jaws, parts: part_list, pairs }
    }

    /// Carries the table that `at` samples (the mean or a component) to the template.
    /// `component` picks the [`component_mask`] instead of the [`face_mask`].
    fn run(
        &self,
        asm: &Assembled,
        described: &[Described],
        at: &dyn Fn(&FaceSample) -> Vec3,
        component: bool,
    ) -> Result<Transferred> {
        let positions = asm.mesh.positions();
        // Body vertices on the face: the field.
        let mut offsets: Vec<Vec3> = self
            .body
            .iter()
            .map(|b| b.as_ref().map_or(Vec3::ZERO, |(s, m, c)| at(s) * if component { *c } else { *m }))
            .collect();
        let lip_gap = close_lips(&mut offsets, positions, described);
        // The eyes' translations from their landmarks, made mirror images of each other.
        let landmark_offset = |name: &str| -> Option<Vec3> {
            let l = &described.iter().find(|d| d.landmark.name == name)?.landmark;
            Some(
                l.vertices
                    .iter()
                    .zip(l.weights)
                    .map(|(&v, w)| offsets.get(v as usize).copied().unwrap_or(Vec3::ZERO) * w)
                    .fold(Vec3::ZERO, |a, b| a + b),
            )
        };
        let mut groups = [Vec3::ZERO; 4];
        for (g, names) in groups.iter_mut().zip(GROUP_LANDMARKS) {
            let found: Vec<Vec3> = names.iter().filter_map(|n| landmark_offset(n)).collect();
            if found.len() != names.len() {
                return Err(invalid("a landmark that places the eyes is missing"));
            }
            *g = found.iter().fold(Vec3::ZERO, |a, &b| a + b) / found.len() as f32;
        }
        let left_eye = (groups[0] + mirror(groups[1])) * 0.5;
        groups[0] = left_eye;
        groups[1] = mirror(left_eye);
        // The skin around each eye moves with it.
        for (slot, &(w, g)) in offsets.iter_mut().zip(&self.eye) {
            if w > 0.0 {
                *slot = *slot * (1.0 - w) + groups.get(g).copied().unwrap_or(Vec3::ZERO) * w;
            }
        }
        // The mouth interior follows the skin around it; the teeth, gums and tongue move with it.
        let sources: Vec<(Vec3, Vec3)> = self
            .sources
            .iter()
            .filter_map(|&v| Some((*positions.get(v as usize)?, *offsets.get(v as usize)?)))
            .collect();
        for ((slot, &inside), &p) in offsets.iter_mut().zip(&self.interior).zip(positions) {
            if inside {
                *slot = interior_offset(p, &sources);
            }
        }
        for (slot, &c) in groups.iter_mut().skip(2).zip(&self.jaws) {
            *slot = interior_offset(c, &sources);
            slot.x = 0.0;
        }
        // The other parts.
        for (group, vertices, near) in &self.parts {
            if let Some(g) = group {
                let t = groups.get(*g).copied().unwrap_or(Vec3::ZERO);
                for &v in vertices {
                    if let Some(slot) = offsets.get_mut(v as usize) {
                        *slot = t;
                    }
                }
                continue;
            }
            for (&v, samples) in vertices.iter().zip(near) {
                let sum = samples.iter().fold(Vec3::ZERO, |a, s| a + at(s));
                if let Some(slot) = offsets.get_mut(v as usize) {
                    *slot = sum / samples.len().max(1) as f32;
                }
            }
        }
        // Vertex-exact mirror symmetry.
        let mut max_asymmetry = 0.0_f32;
        let symmetric = offsets
            .iter()
            .zip(&self.pairs)
            .map(|(&o, &m)| match offsets.get(m as usize) {
                Some(&om) => {
                    max_asymmetry = max_asymmetry.max(o.distance(mirror(om)));
                    (o + mirror(om)) * 0.5
                }
                None => o,
            })
            .collect();
        Ok(Transferred { offsets: symmetric, groups, lip_gap, max_asymmetry })
    }
}

/// The lips keep their procedural contact: the mean of subjects with lips a little apart moves the
/// upper and lower contact lines by different amounts and opens a slit that shows the teeth (a
/// light line between the lips). Near the contact line (full within [`LIP_CLOSURE`]`.0` of it,
/// fading out over [`LIP_CLOSURE`]`.1`) both lips take the mean of the two contact lines'
/// offsets, interpolated across the mouth from the `lip.{upper,lower}.inner.NN` landmark pairs and
/// fading out toward the corners. Returns the largest upper-lower difference taken out, metres.
fn close_lips(offsets: &mut [Vec3], positions: &[Vec3], described: &[Described]) -> f32 {
    let at = |name: &str| -> Option<(Vec3, Vec3)> {
        let l = &described.iter().find(|d| d.landmark.name == name)?.landmark;
        let mut p = Vec3::ZERO;
        let mut o = Vec3::ZERO;
        for (&v, w) in l.vertices.iter().zip(l.weights) {
            p += *positions.get(v as usize)? * w;
            o += *offsets.get(v as usize)? * w;
        }
        Some((p, o))
    };
    // Per landmark pair: x, contact height, contact depth, and the upper lip's correction.
    let mut pairs: Vec<(f32, f32, f32, Vec3)> = (1..=11)
        .filter_map(|k| {
            let (pu, ou) = at(&format!("lip.upper.inner.{k:02}"))?;
            let (pl, ol) = at(&format!("lip.lower.inner.{k:02}"))?;
            Some((pu.x.midpoint(pl.x), pu.y.midpoint(pl.y), pu.z.midpoint(pl.z), (ol - ou) * 0.5))
        })
        .collect();
    let corner =
        ["L.lip.corner", "R.lip.corner"].iter().filter_map(|n| at(n)).map(|(p, _)| p.x.abs()).fold(0.0, f32::max);
    if pairs.len() < 2 || corner <= 0.0 {
        return 0.0;
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let gap = pairs.iter().map(|q| 2.0 * q.3.length()).fold(0.0, f32::max);
    let (first, last) = (pairs[0], pairs[pairs.len() - 1]);
    // The pair data at `x`: linear between pairs, fading to nothing at the corners.
    let sample = |x: f32| -> (f32, f32, Vec3) {
        if x <= first.0 || x >= last.0 {
            let q = if x <= first.0 { first } else { last };
            let fade = 1.0 - smoothstep((x.abs() - q.0.abs()) / (corner - q.0.abs()).max(1e-4));
            return (q.1, q.2, q.3 * fade);
        }
        let i = pairs.iter().position(|q| q.0 >= x).unwrap_or(1).max(1);
        let (a, b) = (pairs[i - 1], pairs[i]);
        let t = (x - a.0) / (b.0 - a.0).max(1e-6);
        (a.1 + (b.1 - a.1) * t, a.2 + (b.2 - a.2) * t, a.3 + (b.3 - a.3) * t)
    };
    for (o, &p) in offsets.iter_mut().zip(positions) {
        if p.x.abs() > corner + LIP_CLOSURE.1 {
            continue;
        }
        let (y, z, upper_fix) = sample(p.x);
        let d = (p.y - y).abs();
        // Fading out in front of the lips (a hard cut there left a step beside the mouth corners
        // at the ends of a shape component's range); behind, the mouth bag ends it.
        let depth = 1.0 - smoothstep((p.z - (z + LIP_CLOSURE.1)) / LIP_CLOSURE.1);
        if d > LIP_CLOSURE.0 + LIP_CLOSURE.1 || p.z < z - LIP_DEPTH || depth <= 0.0 {
            continue;
        }
        let c = (1.0 - smoothstep((d - LIP_CLOSURE.0) / LIP_CLOSURE.1)) * depth;
        *o += if p.y >= y { upper_fix } else { -upper_fix } * c;
    }
    gap
}

/// Per vertex: whether it is a body vertex of the mouth interior (the closed bag behind the lips).
fn mouth_interior(asm: &Assembled) -> Vec<bool> {
    let regions = &asm.regions;
    let Some(bag) = regions.find("mouth-interior") else { return vec![false; asm.source.len()] };
    asm.source
        .iter()
        .enumerate()
        .map(|(v, s)| s.0 == 0 && regions.vertex_regions.get(v).is_some_and(|&r| regions.is_within(r, bag)))
        .collect()
}

/// The skin (body vertices outside the mouth interior) near the mouth interior.
fn skin_sources(asm: &Assembled, interior: &[bool]) -> Vec<u32> {
    let positions = asm.mesh.positions();
    let reach = INTERIOR_KERNEL * 3.0;
    let (lo, hi) = positions
        .iter()
        .zip(interior)
        .filter(|(_, inside)| **inside)
        .fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(l, h), (&p, _)| (l.min(p), h.max(p)));
    let (lo, hi) = (lo - Vec3::splat(reach), hi + Vec3::splat(reach));
    positions
        .iter()
        .zip(interior)
        .zip(&asm.source)
        .enumerate()
        .filter(|(_, ((p, inside), s))| {
            s.0 == 0
                && !**inside
                && p.x >= lo.x
                && p.y >= lo.y
                && p.z >= lo.z
                && p.x <= hi.x
                && p.y <= hi.y
                && p.z <= hi.z
        })
        .map(|(v, _)| v as u32)
        .collect()
}

/// The skin offset at `p` inside the head: the Gaussian-weighted mean of `sources` (width
/// [`INTERIOR_KERNEL`], cut at three widths), zero with no skin in reach.
fn interior_offset(p: Vec3, sources: &[(Vec3, Vec3)]) -> Vec3 {
    let (cut, inv) = ((3.0 * INTERIOR_KERNEL).powi(2), 1.0 / (INTERIOR_KERNEL * INTERIOR_KERNEL));
    let (mut sum, mut weight) = (Vec3::ZERO, 0.0_f32);
    for &(q, o) in sources {
        let d = q - p;
        let d2 = d.dot(d);
        if d2 < cut {
            let w = (-d2 * inv).exp();
            sum += o * w;
            weight += w;
        }
    }
    if weight > 1.0e-12 { sum / weight } else { Vec3::ZERO }
}

/// Per vertex, the vertex of the mirrored part nearest its mirrored position within
/// [`MIRROR_TOLERANCE`] (itself on the midline), `u32::MAX` where there is none.
fn mirror_pairs(positions: &[Vec3], part_of: &[u16], mirror_part: &[u16]) -> Vec<u32> {
    const CELL: f32 = 1.0e-3;
    let key = |p: Vec3| ((p.x / CELL).floor() as i32, (p.y / CELL).floor() as i32, (p.z / CELL).floor() as i32);
    let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
    for (v, &p) in positions.iter().enumerate() {
        grid.entry(key(p)).or_default().push(v as u32);
    }
    positions
        .iter()
        .enumerate()
        .map(|(v, &p)| {
            let part = part_of.get(v).copied().unwrap_or(u16::MAX);
            let want = mirror_part.get(part as usize).copied().unwrap_or(u16::MAX);
            let target = mirror(p);
            let (cx, cy, cz) = key(target);
            let mut best: Option<(u32, f32)> = None;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let Some(list) = grid.get(&(cx + dx, cy + dy, cz + dz)) else { continue };
                        for &m in list {
                            if part_of.get(m as usize).copied() != Some(want) {
                                continue;
                            }
                            let Some(&q) = positions.get(m as usize) else { continue };
                            let d = q.distance(target);
                            if d <= MIRROR_TOLERANCE && best.is_none_or(|(_, bd)| d < bd) {
                                best = Some((m, d));
                            }
                        }
                    }
                }
            }
            best.map_or(u32::MAX, |(m, _)| m)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wp25_embedded_face_model_reads_and_matches_this_template() {
        let model = FaceModel::parse(FACE_MODEL_BYTES).unwrap();
        assert_eq!(model.template_name, super::super::build::TEMPLATE_NAME);
        assert!(model.subjects >= 2, "a mean of one subject is one person's face");
        assert!(!model.rest.is_empty() && model.description.contains("CC BY 4.0"), "{}", model.description);
        assert_eq!(rigid_group("eyeball-L"), Some(0));
        assert_eq!(rigid_group("eye-occlusion-R"), Some(1));
        assert_eq!(rigid_group("teeth-upper"), Some(2));
        assert_eq!(rigid_group("teeth-lower"), Some(3));
        assert_eq!(rigid_group("tongue"), Some(3));
        assert_eq!(rigid_group("eyelashes-L"), None);
    }

    #[test]
    fn wp25_mirror_pairs_match_mirrored_parts_only() {
        let pos = [
            Vec3::new(0.01, 0.0, 0.0),
            Vec3::new(-0.01, 0.0, 0.0),
            Vec3::new(0.0, 0.02, 0.0),
            Vec3::new(0.03, 0.0, 0.0),
            Vec3::new(-0.03, 0.0, 0.0),
            Vec3::new(0.05, 0.0, 0.0),
        ];
        // Parts: 0 body, 1 eye-L, 2 eye-R.
        let part_of = [0, 0, 0, 1, 0, 2];
        let mirror_part = [0, 2, 1];
        let pairs = mirror_pairs(&pos, &part_of, &mirror_part);
        assert_eq!(pairs, vec![1, 0, 2, u32::MAX, u32::MAX, u32::MAX]);
    }

    /// The assembled procedural template, its landmarks and the anatomy, before the scan face.
    fn assembled() -> (Assembled, Vec<Described>, Anatomy) {
        use super::super::{assemble::assemble, body_surface, cage::Cage, landmarks, parts, uv};
        let surface = body_surface().unwrap();
        let parts = parts::build(&surface.anatomy, &surface.body, &surface.targets).unwrap();
        let pieces: Vec<&Cage> = std::iter::once(surface.mesh()).chain(parts.iter().map(|p| &p.mesh)).collect();
        let layout = uv::layout(&pieces).unwrap();
        let asm = assemble(&surface, &parts, &layout).unwrap();
        let described = landmarks::build(&surface, &asm).unwrap();
        (asm, described, surface.anatomy)
    }

    /// `[WP-25]` `[AC-WP20-3]` The scan face keeps the body mirror-exact, moves the eyes as mirror
    /// images, the mouth parts straight, and leaves the back of the head, the ears and the neck
    /// below the face where the procedural generator put them (the mean of the scans there is
    /// hair and head pose). It keeps the lips' contact gap.
    #[test]
    fn wp25_scan_face_is_mirror_exact_and_stays_on_the_face() {
        let (mut asm, described, anatomy) = assembled();
        let before = asm.mesh.positions().to_vec();
        let lip = |pos: &[Vec3], k: u32| {
            let at = |name: String| {
                described.iter().find(|d| d.landmark.name == name).unwrap().landmark.position(pos).unwrap()
            };
            at(format!("lip.upper.inner.{k:02}")).distance(at(format!("lip.lower.inner.{k:02}")))
        };
        let gaps: Vec<f32> = (1..=11).map(|k| lip(&before, k)).collect();
        let (report, components) = apply(&mut asm, &described, &anatomy).unwrap();
        assert_eq!(components.len(), 7);
        println!("[WP-25] scan face: {report:?}");
        let after = asm.mesh.positions();
        assert_eq!(report.subjects, 8);
        assert!(report.moved > 5000 && report.max_offset > 0.003 && report.max_offset < 0.03, "{report:?}");
        assert_eq!(report.eyes[1], mirror(report.eyes[0]));
        assert!(report.mouth.iter().all(|m| m.x == 0.0) && report.interior > 500, "{report:?}");
        // Mirror-exact body.
        let body: Vec<Vec3> = asm.source.iter().zip(after).filter(|(s, _)| s.0 == 0).map(|(_, &p)| p).collect();
        let pairs = mirror_pairs(&body, &vec![0; body.len()], &[0]);
        let unpaired = pairs.iter().filter(|&&m| m == u32::MAX).count();
        assert_eq!(unpaired, 0, "body vertices without a mirror image after the scan face");
        let worst = body.iter().zip(&pairs).map(|(&p, &m)| body[m as usize].distance(mirror(p))).fold(0.0, f32::max);
        assert!(worst < 5.0e-5, "mirror error {worst} m");
        // Off the face nothing moves.
        let eye = anatomy.eye_radius + EYE_RIGID_MARGIN + EYE_BLEND;
        let mut still = 0;
        for ((&p, &q), s) in before.iter().zip(after).zip(&asm.source) {
            let near_eye = p.distance(sided(anatomy.eye_center, p.x.signum())) < eye;
            if s.0 == 0 && face_mask(p, &anatomy) == 0.0 && !near_eye {
                assert_eq!(p, q, "a vertex off the face moved");
                still += 1;
            }
        }
        assert!(still > 10_000, "{still}");
        // The lips keep their contact gap (within 0.1 mm).
        for (k, g) in (1..=11).zip(&gaps) {
            let now = lip(after, k);
            assert!((now - g).abs() < 1.0e-4, "lip pair {k}: gap {g} -> {now}");
        }
    }

    /// `[WP-25]` The fitting base (`BASE_VERSION`, the head before the scan face, with the base
    /// auricle) is the surface the face model was measured on: every face vertex lies on the model's
    /// rest surface (within 0.01 mm). If the procedural face changes (or anything the cage fit
    /// couples to it, such as the ears), this fails: refit the scans, rebuild the face model and
    /// raise `BASE_VERSION`.
    #[test]
    fn wp25_the_fitting_base_is_the_face_models_rest_surface() {
        let (asm, _, anatomy) = assembled();
        let model = FaceModel::parse(FACE_MODEL_BYTES).unwrap();
        let field = model.field().unwrap();
        assert_eq!(model.template_version, super::super::build::BASE_VERSION);
        let (mut face, mut edge, mut checked) = (0.0_f32, 0.0_f32, 0);
        for (&p, s) in asm.mesh.positions().iter().zip(&asm.source) {
            let mask = face_mask(p, &anatomy);
            if s.0 == 0 && mask > 0.0 {
                let d = field.sample(p, 0.01).map_or(f32::INFINITY, |s| s.distance);
                if mask > 0.05 {
                    face = face.max(d);
                } else {
                    edge = edge.max(d);
                }
                checked += 1;
            }
        }
        println!("[WP-25] base vs model: {checked} face vertices; farthest {face} m (mask > 5 %), {edge} m (edge)");
        assert!(checked > 5000 && face < 1.0e-5 && edge < 1.0e-5, "{face} / {edge} m off the model");
    }

    #[test]
    fn wp25_face_mask_is_the_face() {
        let a = Anatomy::neutral();
        let at = |x: f32, y: f32, z: f32| face_mask(a.head_origin + Vec3::new(x, y, z) * 0.001, &a);
        assert_eq!(at(0.0, -20.0, 120.0), 1.0, "nose");
        assert_eq!(at(30.0, 10.0, 80.0), 1.0, "eye");
        assert_eq!(at(70.0, 0.0, 0.0), 0.0, "ear");
        assert_eq!(at(0.0, -100.0, -70.0), 0.0, "nape");
        assert_eq!(at(0.0, -130.0, 40.0), 0.0, "neck");
        assert_eq!(at(0.0, 110.0, 50.0), 0.0, "scalp");
        assert!((0.0..1.0).contains(&at(40.0, -100.0, 60.0)), "under the jaw: fading");
    }
}
