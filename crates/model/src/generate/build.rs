//! The whole generated template (WP-06, ADR-006): surface, parts, UV layout, regions, landmarks,
//! skeleton, skin weights and LODs put together and checked by [`Template::new`].
//!
//! The identity components (`crate::identity`) and the expression targets and correctives
//! (`crate::expression::build`) are added to the LOD0 template before the LODs are generated.

use super::anatomy::Anatomy;
use super::assemble::{Assembled, assemble};
use super::genitals::Variant;
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
    /// The external genitalia (ADR-039).
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
    let surface = super::body_surface_for(options.genitals, auricle)?;
    let parts = parts::build(&surface.anatomy, &surface.body, &surface.targets)?;
    let pieces: Vec<&super::cage::Cage> =
        std::iter::once(surface.mesh()).chain(parts.iter().map(|p| &p.mesh)).collect();
    let layout = uv::layout(&pieces)?;
    let mut asm = assemble(&surface, &parts, &layout)?;
    let described = landmarks::build(&surface, &asm)?;
    // The scanned face moves the eyes; the eye joints and lid helpers move with them.
    let mut anatomy: Anatomy = surface.anatomy.clone();
    let mut scan_components = Vec::new();
    if options.scan_face {
        let (report, components) = super::scan_face::apply(&mut asm, &described, &surface.anatomy)?;
        anatomy.eye_center += report.eyes[0];
        scan_components = components;
        super::finish::cranium(&mut asm, &surface.anatomy)?;
        super::finish::fair_nose_root(&mut asm, &surface.anatomy)?;
    }
    let mut info = info();
    if options.scan_face {
        SCAN_FACE_LICENCE.clone_into(&mut info.licence);
        info.provenance.push(super::scan_face::provenance()?);
    }
    let anatomy = &anatomy;
    let rig = rig::skeleton(anatomy, 0)?;
    let skin = rig::skin(&rig, anatomy, &asm.mesh, &asm.regions)?;
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
    // The identity components (shape axes and joint-regressor rows) and the expression targets
    // and correctives are authored on the finished LOD0 template.
    let (template, _) = crate::identity::with_identity(&template)?;
    // The face-shape components of the scans (todo 1.2), bound as `face.scan.shape_NN`.
    let fields: Vec<ScanField<'_>> =
        scan_components.iter().map(|c| ScanField { offsets: &c.offsets, eye: c.eye }).collect();
    let template = crate::identity::add_components(&template, &crate::identity::scan::components(&template, &fields)?)?;
    let specs = CorrectiveSpecs::builtin().map_err(expression_error)?;
    let (template, _) = with_expressions(&template, &specs).map_err(expression_error)?;
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
