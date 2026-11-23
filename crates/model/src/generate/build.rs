//! The whole generated template (WP-06, ADR-006): surface, parts, UV layout, regions, landmarks,
//! skeleton, skin weights and LODs put together and checked by [`Template::new`].
//!
//! The identity components (`crate::identity`) and the expression targets and correctives
//! (`crate::expression::build`) are added to the LOD0 template before the LODs are generated.

use super::anatomy::Anatomy;
use super::assemble::{Assembled, assemble};
use super::genitals::{self, Variant};
use super::landmarks::{self, Described};
use super::regions::TILES;
use super::{Surface, parts, rig, uv};
use crate::delta::DeltaBases;
use crate::error::{ModelError, Result};
use crate::expression::ExpressionSet;
use crate::expression::build::{CorrectiveSpecs, with_expressions};
use crate::expression::error::ExprError;
use crate::identity::scan::ScanField;
use crate::landmark::Landmark;
use crate::simplify::{LodErrorReport, LodInput, LodSettings, ProtectedLoop, generate_lods};
use crate::template::{ProvenanceEntry, Template, TemplateInfo, TemplateParts, UvTile};

/// Stable name of the generated template.
pub const TEMPLATE_NAME: &str = "humanart-neutral";
/// Content version of the generated template; raised whenever the generator's output changes.
pub const TEMPLATE_VERSION: u32 = 2;
/// Version of the fitting base: the geometry built with [`Options::scan_face`] off, which scans are
/// fitted to (`fit-scan`) and the face model's offsets are measured from (`face-model`). Raised only
/// when that geometry changes; then every fit and the face model must be rebuilt. Fits and face
/// models record this number, not [`TEMPLATE_VERSION`] (which also rises for changes the scan face
/// and the finishing steps make on top of the base).
pub const BASE_VERSION: u32 = 1;
/// The template's licence when the scanned face is applied ([`Options::scan_face`]).
pub const SCAN_FACE_LICENCE: &str = "procedural (LIC-02 c) except the face shape: the mean of HSRD-100 scans,      CC BY 4.0 (attribution in the provenance entry 'face shape')";
/// The generator string stored in the template.
pub const GENERATOR: &str = "humanart-model generate (procedural in-house template, ADR-006)";

/// What to build.
#[derive(Clone, Debug)]
pub struct Options {
    /// Whether to generate LOD1..LOD4 (the slowest step).
    pub lods: bool,
    /// The LOD settings.
    pub lod_settings: LodSettings,
    /// The external genitalia in the body (ADR-039). With [`Variant::Female`] (the default) the
    /// male piece ships as well, stowed inside the pelvis ([`genitals::male_piece`],
    /// [`genitals::stow`]); the `sex.male` component brings it out. With [`Variant::Male`] only the
    /// male variant is built (review packages, tests).
    pub genitals: Variant,
    /// Whether the head is finished: the scanned mean face ([`super::scan_face`]) and the steps
    /// after it (the cranium). Off only for the fitting base ([`BASE_VERSION`]) that `fit-scan` and
    /// `face-model` work on: the face model's offsets are measured from the head without them.
    pub scan_face: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self { lods: true, lod_settings: LodSettings::default(), genitals: Variant::default(), scan_face: true }
    }
}

/// The generated template with what the template's tables cannot hold.
#[derive(Clone, Debug)]
pub struct Generated {
    /// The checked template.
    pub template: Template,
    /// The landmarks with their descriptions (`data/template/landmarks.json`).
    pub landmarks: Vec<Described>,
    /// Per LOD level (LOD1 first), the landmarks re-expressed on that level's mesh: the template's
    /// LOD record has no landmark table yet.
    pub lod_landmarks: Vec<Vec<Landmark>>,
    /// Per LOD level, the measured errors.
    pub lod_reports: Vec<LodErrorReport>,
    /// The eyelid and lip loops the LOD generator preserved.
    pub loops: Vec<ProtectedLoop>,
}

/// The eyelid margins (lash line) and the lip lines (vermilion border, contact line) as loops of
/// LOD0 vertices: the cage rings refined through the subdivision levels.
pub fn protected_loops(surface: &Surface, asm: &Assembled) -> Result<Vec<ProtectedLoop>> {
    let first = first_copies(asm);
    let refine = |ring: &[u32]| -> Result<Vec<u32>> {
        let mut line = ring.to_vec();
        for level in &surface.levels {
            line = level.refine(&line, true);
        }
        line.iter()
            .map(|&v| first.get(v as usize).copied().filter(|&m| m != u32::MAX))
            .collect::<Option<Vec<u32>>>()
            .ok_or(ModelError::invalid("protected loop", 0, "loop vertex is not in the mesh"))
    };
    let loops = &surface.body.loops;
    let mut out = Vec::new();
    for (eye, side) in loops.eyes.iter().zip(["L", "R"]) {
        let ring = eye.rings.get(2).ok_or(ModelError::invalid("protected loop", 2, "eye has no lid margin ring"))?;
        out.push(ProtectedLoop {
            name: format!("{side}.eyelid-margin"),
            vertices: refine(ring)?,
            lock_through: 2,
            keep_through: 4,
        });
    }
    for (index, name) in [(2_usize, "lip.vermilion-border"), (4, "lip.contact")] {
        let ring =
            loops.mouth.rings.get(index).ok_or(ModelError::invalid("protected loop", index, "mouth ring missing"))?;
        out.push(ProtectedLoop { name: name.to_owned(), vertices: refine(ring)?, lock_through: 2, keep_through: 4 });
    }
    Ok(out)
}

/// The mesh vertex of each body (piece 0) vertex: its first copy, `u32::MAX` where there is none.
fn first_copies(asm: &Assembled) -> Vec<u32> {
    let count = asm.source.iter().filter(|s| s.0 == 0).map(|s| s.1 as usize + 1).max().unwrap_or(0);
    let mut first = vec![u32::MAX; count];
    for (m, &(piece, v)) in asm.source.iter().enumerate() {
        if piece == 0
            && let Some(slot) = first.get_mut(v as usize).filter(|s| **s == u32::MAX)
        {
            *slot = m as u32;
        }
    }
    first
}

/// The vertices of `cage` inside the genital attachment loop: on genital faces only.
fn vulva_vertices(cage: &super::cage::Cage) -> Vec<bool> {
    let mut genital = vec![false; cage.pos.len()];
    let mut other = vec![false; cage.pos.len()];
    for f in 0..cage.face_count() {
        let is = super::regions::R::from_id(cage.tags[f].region).is_some_and(super::regions::R::is_genital);
        for &v in cage.corners_of(f) {
            if let Some(slot) = if is { genital.get_mut(v as usize) } else { other.get_mut(v as usize) } {
                *slot = true;
            }
        }
    }
    genital.iter().zip(&other).map(|(&g, &o)| g && !o).collect()
}

/// The assembled mesh with both genital variants where they belong: the vulva (body cage vertices
/// flagged in `vulva`) as `identity.sex` −1 shows it and the male piece (body cage vertices from
/// `body_vertices` on) as +1 shows it, so their skin weights are those of the real groin.
fn real_state(asm: &Assembled, body_vertices: usize, vulva: &[bool]) -> Result<crate::mesh::Mesh> {
    let mut data = asm.mesh.data().clone();
    for (p, &(piece, v)) in data.positions.iter_mut().zip(&asm.source) {
        if piece != 0 {
            continue;
        }
        if v as usize >= body_vertices {
            *p = genitals::male_real(*p);
        } else if vulva.get(v as usize).copied().unwrap_or(false) {
            *p = genitals::vulva_states(*p).0;
        }
    }
    crate::mesh::Mesh::new(data)
}

/// The template's info: name, version, generator, licence and provenance `[LIC-02]` of the procedural
/// template; [`template`] adds the scanned face's licence and entry when it applies the face.
pub fn info() -> TemplateInfo {
    let procedural = |item: &str, source: &str| ProvenanceEntry {
        item: item.to_owned(),
        source: source.to_owned(),
        licence: "procedural (LIC-02 c): generated by HumanArt code, no third-party data".to_owned(),
    };
    TemplateInfo {
        name: TEMPLATE_NAME.to_owned(),
        version: TEMPLATE_VERSION,
        generator: GENERATOR.to_owned(),
        licence: "procedural (LIC-02 c): generated by HumanArt code, no third-party data".to_owned(),
        provenance: vec![
            procedural(
                "proportions",
                "humanart-model generate::anatomy: a few dozen published summary means (ANSUR II 2012, Farkas 1994), sexes averaged; no dataset embedded",
            ),
            procedural("topology, parts and UV layout", "humanart-model generate::{body, face, limbs, parts, uv}"),
            procedural("regions and landmarks", "humanart-model generate::{regions, assemble, landmarks}"),
            procedural("skeleton and skin weights", "humanart-model generate::rig"),
            procedural("LODs", "humanart-model simplify (attribute-preserving edge collapse)"),
        ],
        ..TemplateInfo::default()
    }
}

/// An expression build error as a model error: model errors pass through, the others keep their
/// index and a short phrase (call `expression::build` directly for the detail).
fn expression_error(e: ExprError) -> ModelError {
    match e {
        ExprError::Model(m) => m,
        ExprError::Missing { .. } => ModelError::invalid(
            "expression recipe",
            0,
            "a recipe names a landmark, joint, region or part the template lacks",
        ),
        ExprError::Invalid { index, .. } => {
            ModelError::invalid("expression recipe", index, "an expression target or corrective is invalid")
        }
        ExprError::Eval(_) => ModelError::invalid("expression recipe", 0, "a corrective could not be evaluated"),
        ExprError::Json(_) | ExprError::Parse(_) => {
            ModelError::invalid("expression recipe", 0, "the built-in corrective data is malformed")
        }
    }
}

/// Builds the whole template.
pub fn template(options: &Options) -> Result<Generated> {
    let auricle = if options.scan_face { super::anatomy::Auricle::Finished } else { super::anatomy::Auricle::Base };
    let mut surface = super::body_surface_for(options.genitals, auricle)?;
    // ADR-039, todo 1.3: with the female variant in the body, the male piece ships too. Both sit
    // halfway between their two states (`genitals::male_rest`, `genitals::vulva_rest`): the `sex`
    // component (`identity.sex`, −1 female, +1 male) shows one and hides the other, no rebuild.
    let body_vertices = surface.mesh().pos.len();
    let mut vulva = vec![false; body_vertices];
    if options.genitals == Variant::Female {
        let male = super::body_surface_for(Variant::Male, auricle)?;
        let mut piece = genitals::male_piece(male.mesh())?;
        for p in &mut piece.pos {
            *p = genitals::male_rest(*p);
        }
        let last = surface.levels.last_mut().ok_or(ModelError::invalid("body surface", 0, "no subdivision level"))?;
        vulva = vulva_vertices(&last.cage);
        for (p, _) in last.cage.pos.iter_mut().zip(&vulva).filter(|(_, v)| **v) {
            *p = genitals::vulva_rest(*p);
        }
        last.cage.append(&piece);
    }
    let parts = parts::build(&surface.anatomy, &surface.body, &surface.targets)?;
    let pieces: Vec<&super::cage::Cage> =
        std::iter::once(surface.mesh()).chain(parts.iter().map(|p| &p.mesh)).collect();
    let layout = uv::layout(&pieces)?;
    let mut asm = assemble(&surface, &parts, &layout)?;
    let mut described = landmarks::build(&surface, &asm)?;
    // The scanned face moves the eyes; the eye joints and lid helpers move with them.
    let mut anatomy: Anatomy = surface.anatomy.clone();
    let mut scan_components = Vec::new();
    if options.scan_face {
        let (report, components) = super::scan_face::apply(&mut asm, &described, &surface.anatomy)?;
        anatomy.eye_center += report.eyes[0];
        scan_components = components;
        super::finish::cranium(&mut asm, &surface.anatomy)?;
        super::finish::fair_nose_root(&mut asm, &surface.anatomy)?;
        // The midline profile points are found again on the finished face (todo 1.3).
        landmarks::refit_profile(&surface, &asm, &mut described)?;
    }
    let mut info = info();
    if options.scan_face {
        SCAN_FACE_LICENCE.clone_into(&mut info.licence);
        info.provenance.push(super::scan_face::provenance()?);
    }
    let anatomy = &anatomy;
    let rig = rig::skeleton(anatomy, 0)?;
    let skin = rig::skin(&rig, anatomy, &real_state(&asm, body_vertices, &vulva)?, &asm.regions)?;
    let tiles = TILES
        .iter()
        .map(|&(_, name, resolution, target)| UvTile {
            name: name.to_owned(),
            resolution,
            target_texels_per_mm: target,
        })
        .collect();
    let template = Template::new(TemplateParts {
        info,
        mesh: asm.mesh.clone(),
        tiles,
        regions: asm.regions.clone(),
        landmarks: described.iter().map(|d| d.landmark.clone()).collect(),
        skeleton: rig.skeleton,
        skin,
        deltas: DeltaBases::empty(),
        expressions: ExpressionSet::default(),
        lods: Vec::new(),
    })?;
    // Both genital variants: half the one-sided sex features (breasts, larynx) go into the rest
    // shape, so the switch (`identity.sex`) gives the female all of them and the male none. Not on
    // the fitting base (`scan_face` off), which the face model is measured against.
    let template = if options.genitals == Variant::Female && options.scan_face {
        let shift = crate::identity::sex::rest_shift(&template)?;
        let mut parts = template.into_parts();
        let mut data = parts.mesh.into_data();
        for (v, d) in shift {
            if let Some(p) = data.positions.get_mut(v as usize) {
                *p += d;
            }
        }
        parts.mesh = crate::mesh::Mesh::new(data)?;
        Template::new(parts)?
    } else {
        template
    };
    // The identity components (shape axes and joint-regressor rows) and the expression targets
    // and correctives are authored on the finished LOD0 template.
    let (template, _) = crate::identity::with_identity(&template)?;
    // The face-shape components of the scans (todo 1.2), bound as `face.scan.shape_NN`.
    let fields: Vec<ScanField<'_>> =
        scan_components.iter().map(|c| ScanField { offsets: &c.offsets, eye: c.eye }).collect();
    let template = crate::identity::add_components(&template, &crate::identity::scan::components(&template, &fields)?)?;
    let specs = CorrectiveSpecs::builtin().map_err(expression_error)?;
    let (template, _) = with_expressions(&template, &specs).map_err(expression_error)?;
    // The body's pose correctives (todo 1.3), derived on the finished rest shape and skin weights.
    let (template, _) = super::correctives::with_pose_correctives(&template)?;
    let loops = protected_loops(&surface, &asm)?;
    if !options.lods {
        return Ok(Generated {
            template,
            landmarks: described,
            lod_landmarks: Vec::new(),
            lod_reports: Vec::new(),
            loops,
        });
    }
    let generated = generate_lods(&LodInput::from_template(&template, &loops), &options.lod_settings)?;
    let mut parts = template.parts().clone();
    let mut lod_landmarks = Vec::with_capacity(generated.len());
    let mut lod_reports = Vec::with_capacity(generated.len());
    for g in generated {
        parts.lods.push(g.level);
        lod_landmarks.push(g.landmarks);
        lod_reports.push(g.report);
    }
    let template = Template::new(parts)?;
    Ok(Generated { template, landmarks: described, lod_landmarks, lod_reports, loops })
}
