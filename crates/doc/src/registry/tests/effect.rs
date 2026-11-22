//! Every registry parameter moves the character (`todo.md` 2.3).
//!
//! [`every_parameter_moves_the_character`] walks every parameter of `data/template/params/`, sets it
//! to the two ends of its plausible span through the command path the editor and the agent use
//! ([`Session::execute`] with [`Action::SetParams`]), evaluates the character on the CPU the way the
//! viewer's model does (`crates/agent/src/engine/model.rs`: shape components, the expression rig
//! and the gaze preview into the reference [`Evaluator`], `skin.*` materials into [`SkinValues`])
//! and what the viewer hands the renderer for the rest (`crates/agent/src/engine`: the eye and mouth
//! feature-table parameters [`MaterialValues::eye_params`] / [`MaterialValues::mouth_params`], the
//! hair groom set of `hair::spec_for`, the generator settings of the document's procedural skin
//! layers, `Model::feature_layers`), and compares the two results. The test document holds one
//! procedural skin layer of every generator that overrides nothing, so the global
//! `skin.features.*` values reach them. A parameter whose two ends
//! give the same evaluation does nothing; the list of those, with the reason and where the hook is
//! missing, is [`KNOWN_DEAD`]. The test fails when a parameter that moved goes dead and when a
//! listed one starts to move (then the list is out of date). A parameter that only acts with
//! another one set ([`PREREQUISITES`]) is measured with that one set.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::Arc;

use humanart_core::id::Id128;
use humanart_core::platform::{ManualClock, MemoryFs};
use humanart_math::{Quat, Vec3};
use humanart_model::Template;
use humanart_model::eval::{EvalInput, Evaluator, SkinningMode};
use humanart_model::eye::{EyeParams, Gaze};
use humanart_model::generate::build::{Options, template};
use humanart_model::hair::GroomSetSpec;
use humanart_model::hair::settings::{GroomSettings, GroomTarget};
use humanart_model::skin_appearance::feature_layers::{FeatureGlobals, generator_of};
use humanart_model::skin_appearance::features::{GeneratorKind, GeneratorParams};
use humanart_model::skin_appearance::materials::{MaterialTarget, MaterialValues, Side};
use humanart_model::skin_appearance::values::{SkinTarget, SkinValues};

use super::real_registry;
use crate::command::{Action, Command, RangePolicy};
use crate::document::{Author, AuthorKind, Document, SkinBlend, SkinLayer, SkinSource};
use crate::expression_drive::ExpressionDrive;
use crate::registry::{Evaluation, Param, ParamKind, ParamValue, Registry};
use crate::session::{Session, SessionConfig};

const HUMAN: &str = "human.test";

/// Displacement below which a vertex counts as unmoved, in metres (0.01 mm: float noise of the
/// evaluator is far below, the smallest real effect far above).
const MOVED_M: f32 = 1e-5;

/// Eye optics with no field to carry them: `EyeParams` has no refractive indices or tear film and
/// the eye shader packs constants (`humanart_render::eye::EyeShading::entry`).
const EYE_OPTICS: &str = "no field: EyeParams (model/eye) has no IOR or tear film, render/eye packs constants";
/// Captured iris textures: nothing reads the blob.
const IRIS_TEXTURE: &str = "asset-ref iris texture: nothing reads the blob (render/eye)";
/// The tattoo stamps the user's picture; a procedural skin layer carries no image, so the engine
/// generates no tattoo layer (`engine/model.rs` `procedural`).
const TATTOO: &str = "tattoo: the generator needs the user's image, a procedural layer has none (doc SkinSource)";
/// Per-region maps: no consumer reads the blob.
const MAP: &str = "asset-ref map: nothing reads the blob (render/skin)";

/// The parameters that do nothing in the viewer's evaluation, with the reason. Each line names
/// where the hook is missing.
const KNOWN_DEAD: &[(&str, &str)] = &[
    ("eyes.aqueous.ior", EYE_OPTICS),
    ("eyes.cornea.ior", EYE_OPTICS),
    ("eyes.iris.texture.l", IRIS_TEXTURE),
    ("eyes.iris.texture.r", IRIS_TEXTURE),
    ("eyes.tear.film", EYE_OPTICS),
    ("skin.features.tattoo.center_u", TATTOO),
    ("skin.features.tattoo.center_v", TATTOO),
    ("skin.features.tattoo.opacity", TATTOO),
    ("skin.features.tattoo.rotation", TATTOO),
    ("skin.features.tattoo.size", TATTOO),
    ("skin.fine_lines.depth_map", MAP),
    ("skin.pores.density_map", MAP),
];

/// Parameters that act only with another one set, and that setting: the central iris colour
/// shows with central heterochromia.
const PREREQUISITES: &[(&str, &str, &str)] = &[
    ("eyes.iris.central_color.l", "eyes.iris.heterochromia", "central"),
    ("eyes.iris.central_color.r", "eyes.iris.heterochromia", "central"),
];

/// What a parameter does to the evaluation (what moved between the two ends of its span).
#[derive(Clone, Debug, PartialEq)]
enum Effect {
    /// The largest vertex displacement, in metres.
    Mesh(f32),
    /// The skin values changed.
    Skin,
    /// The eye or mouth feature-table parameters changed.
    Material,
    /// The hair's groom set changed (its colour or its strands).
    Groom,
    /// The generator settings of the document's procedural skin layers changed.
    Layers,
    /// Nothing changed.
    Nothing,
    /// The parameter cannot be set through a command (read-only measurement).
    ReadOnly,
}

/// The viewer's evaluation of a document, mirrored from `crates/agent/src/engine/model.rs`.
struct Character {
    eval: Evaluator,
    template: Arc<Template>,
    drive: ExpressionDrive,
    shape: Vec<(usize, usize)>,
    skin: Vec<(usize, SkinTarget)>,
    materials: Vec<(usize, MaterialTarget)>,
    grooms: Vec<(usize, GroomTarget)>,
    features: Vec<(usize, (usize, usize))>,
    gaze: [Option<usize>; 2],
    eyes: Vec<(usize, Quat)>,
}

/// What the viewer hands the renderer besides the mesh: the skin values, both eyes' and the
/// mouth's feature parameters and the hair's groom set.
#[derive(Clone, Debug, PartialEq)]
struct Appearance {
    skin: SkinValues,
    eyes: [EyeParams; 2],
    mouth: humanart_model::mouth::MouthParams,
    hair: GroomSetSpec,
    layers: Vec<GeneratorParams>,
}

fn scalar(value: &ParamValue) -> Option<f32> {
    match value {
        #[allow(clippy::cast_possible_truncation)]
        ParamValue::Scalar(v) => Some(*v as f32),
        ParamValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

impl Character {
    fn new(template: Arc<Template>, registry: &Registry) -> Self {
        let mut shape = Vec::new();
        let mut skin = Vec::new();
        let mut materials = Vec::new();
        let mut grooms = Vec::new();
        let mut gaze = [None; 2];
        let features = registry
            .params()
            .iter()
            .enumerate()
            .filter_map(|(i, p)| FeatureGlobals::slot_of(&p.id).map(|s| (i, s)))
            .collect();
        for (i, p) in registry.params().iter().enumerate() {
            match &p.evaluation {
                Evaluation::Component(name) => {
                    if let Some(c) = template.deltas().find(name) {
                        shape.push((i, c as usize));
                    }
                }
                Evaluation::Material(target) => {
                    if let Some(t) = SkinTarget::from_target(target) {
                        skin.push((i, t));
                    }
                    if let Some(t) = MaterialTarget::from_target(target) {
                        materials.push((i, t));
                    }
                }
                Evaluation::Groom(target) => {
                    if let Some(t) = GroomTarget::from_target(target) {
                        grooms.push((i, t));
                    }
                }
                Evaluation::Editor(target) if target == "gaze.yaw" => gaze[0] = Some(i),
                Evaluation::Editor(target) if target == "gaze.pitch" => gaze[1] = Some(i),
                _ => {}
            }
        }
        let rest = template.skeleton().rest_world();
        let eyes = ["L.eye", "R.eye"]
            .iter()
            .filter_map(|n| template.skeleton().find(n))
            .map(|j| (usize::from(j), rest[usize::from(j)].rotation))
            .collect();
        let drive = ExpressionDrive::new(&template, registry).unwrap();
        Self {
            eval: Evaluator::new(Arc::clone(&template)),
            template,
            drive,
            shape,
            skin,
            materials,
            grooms,
            features,
            gaze,
            eyes,
        }
    }

    /// Evaluates `doc`: the posed positions and the appearance.
    fn evaluate(&mut self, doc: &Document, registry: &Registry) -> (Vec<Vec3>, Appearance) {
        let t = &self.template;
        let mut shape = vec![0.0; t.deltas().len()];
        let mut expressions = vec![0.0; t.expressions().expressions.len()];
        let mut pose = vec![Quat::IDENTITY; t.skeleton().joints.len()];
        let mut skin = SkinValues::default();
        let params = registry.params();
        for &(i, c) in &self.shape {
            let p = &params[i];
            shape[c] = scalar(doc.params.get(&p.id).unwrap_or(&p.default)).unwrap();
        }
        for &(i, target) in &self.skin {
            let p = &params[i];
            match doc.params.get(&p.id).unwrap_or(&p.default) {
                #[allow(clippy::cast_possible_truncation)]
                ParamValue::Color(c) => {
                    skin.set_color(target, [c[0] as f32, c[1] as f32, c[2] as f32]);
                }
                other => {
                    if let Some(v) = scalar(other) {
                        skin.set(target, v);
                    }
                }
            }
        }
        // engine/model.rs `sync_features`: global values, then each procedural layer's overrides.
        let mut globals = FeatureGlobals::default();
        for &(i, slot) in &self.features {
            let p = &params[i];
            if let Some(v) = scalar(doc.params.get(&p.id).unwrap_or(&p.default)) {
                globals.set(slot, v);
            }
        }
        let layers = doc
            .skin_layers
            .iter()
            .filter_map(|l| match &l.source {
                SkinSource::Procedural { generator, params, .. } if l.visible => {
                    let kind = generator_of(generator).filter(|k| *k != GeneratorKind::Tattoo)?;
                    Some(globals.layer_params(kind, params.iter().map(|(k, v)| (k.as_str(), *v))))
                }
                _ => None,
            })
            .collect();
        // engine/model.rs `set_material`: scalars, colours and choices.
        let mut materials = MaterialValues::default();
        let mut hair_color_set = false;
        for &(i, target) in &self.materials {
            let p = &params[i];
            let own = doc.params.get(&p.id);
            hair_color_set |= own.is_some() && matches!(target, MaterialTarget::Hair(_));
            match own.unwrap_or(&p.default) {
                #[allow(clippy::cast_possible_truncation)]
                ParamValue::Color(c) => {
                    materials.set_color(target, [c[0] as f32, c[1] as f32, c[2] as f32]);
                }
                ParamValue::Enum(choice) => {
                    materials.set_choice(target, choice);
                }
                other => {
                    if let Some(v) = scalar(other) {
                        materials.set_scalar(target, v);
                    }
                }
            }
        }
        let mut grooms = GroomSettings::default();
        let mut body_hair_set = false;
        for &(i, target) in &self.grooms {
            let p = &params[i];
            let own = doc.params.get(&p.id);
            body_hair_set |= own.is_some() && target == GroomTarget::BodyDensity;
            if let Some(v) = scalar(own.unwrap_or(&p.default)) {
                grooms.set(target, v);
            }
        }
        // engine/hair.rs `spec_for` with the default HairOptions (dark brown, no body hair).
        let mut hair = GroomSetSpec { eumelanin: 2.5, pheomelanin: 0.1, peach_fuzz: false, ..GroomSetSpec::default() };
        grooms.apply(&mut hair);
        if hair_color_set {
            (hair.eumelanin, hair.pheomelanin) = materials.hair_melanin();
        }
        hair.grey_fraction = materials.hair_grey();
        hair.body_hair = body_hair_set;
        self.drive.evaluate(doc, registry).unwrap();
        self.drive.write(&mut expressions, &mut pose);
        // engine/model.rs `apply_gaze`.
        let angle = |slot: Option<usize>| {
            let p = &params[slot?];
            scalar(doc.params.get(&p.id).unwrap_or(&p.default))
        };
        let (yaw, pitch) = (angle(self.gaze[0]).unwrap_or(0.0), angle(self.gaze[1]).unwrap_or(0.0));
        if yaw != 0.0 || pitch != 0.0 {
            let gaze = Gaze { yaw: yaw.to_radians(), pitch: pitch.to_radians() }.clamped(&EyeParams::default());
            for &(j, rest) in &self.eyes {
                pose[j] = gaze.joint_rotation(rest) * pose[j];
            }
        }
        let input = EvalInput {
            shape: &shape,
            expressions: &expressions,
            layers: &[],
            pose: &pose,
            skinning: SkinningMode::ByRegion,
        };
        self.eval.evaluate(&input).unwrap();
        let appearance = Appearance {
            skin,
            eyes: Side::BOTH.map(|s| materials.eye_params(s)),
            mouth: materials.mouth_params(),
            hair,
            layers,
        };
        (self.eval.positions().to_vec(), appearance)
    }
}

/// The two ends of the parameter's plausible span (or the two extreme choices), or `None` when the
/// parameter has no second value to set.
fn ends(p: &Param) -> Option<[ParamValue; 2]> {
    let r = p.range;
    Some(match p.kind {
        ParamKind::Scalar => {
            let r = r?;
            [ParamValue::Scalar(r.soft_min), ParamValue::Scalar(r.soft_max)]
        }
        ParamKind::Vector => {
            let r = r?;
            let n = usize::from(p.dimension);
            [ParamValue::Vector(vec![r.soft_min; n]), ParamValue::Vector(vec![r.soft_max; n])]
        }
        ParamKind::Color => {
            let r = r?;
            [ParamValue::Color([r.soft_min; 3]), ParamValue::Color([r.soft_max; 3])]
        }
        ParamKind::Enum => [ParamValue::Enum(p.choices.first()?.clone()), ParamValue::Enum(p.choices.last()?.clone())],
        ParamKind::Bool => [ParamValue::Bool(false), ParamValue::Bool(true)],
        ParamKind::Curve => {
            let r = r?;
            [
                ParamValue::Curve(vec![[0.0, r.soft_min], [1.0, r.soft_min]]),
                ParamValue::Curve(vec![[0.0, r.soft_max], [1.0, r.soft_max]]),
            ]
        }
        ParamKind::AssetRef => [
            ParamValue::AssetRef(humanart_core::hash::Hash128::ZERO),
            ParamValue::AssetRef(humanart_core::hash::hash128(p.id.as_bytes())),
        ],
    })
}

fn set(session: &mut Session<'_>, id: &str, value: ParamValue) {
    let action =
        Action::SetParams { values: vec![(id.to_owned(), value)], relative: false, policy: RangePolicy::Reject };
    session.execute(Command::new(HUMAN, action)).unwrap();
}

fn effect(character: &mut Character, session: &mut Session<'_>, registry: &Registry, p: &Param) -> Effect {
    if p.read_only {
        return Effect::ReadOnly;
    }
    let Some([lo, hi]) = ends(p) else { return Effect::Nothing };
    let prerequisite = PREREQUISITES.iter().find(|(id, _, _)| *id == p.id);
    if let Some((_, other, choice)) = prerequisite {
        set(session, other, ParamValue::Enum((*choice).to_owned()));
    }
    set(session, &p.id, lo);
    let (a, look_a) = character.evaluate(session.document(), registry);
    set(session, &p.id, hi);
    let (b, look_b) = character.evaluate(session.document(), registry);
    let mut ids = vec![p.id.clone()];
    ids.extend(prerequisite.map(|(_, other, _)| (*other).to_owned()));
    session.execute(Command::new(HUMAN, Action::ResetParams { ids })).unwrap();
    let moved = a.iter().zip(&b).map(|(x, y)| (*x - *y).length()).fold(0.0_f32, f32::max);
    if moved > MOVED_M {
        Effect::Mesh(moved)
    } else if look_a.skin != look_b.skin {
        Effect::Skin
    } else if look_a.eyes != look_b.eyes || look_a.mouth != look_b.mouth {
        Effect::Material
    } else if look_a.hair != look_b.hair {
        Effect::Groom
    } else if look_a.layers != look_b.layers {
        Effect::Layers
    } else {
        Effect::Nothing
    }
}

/// `todo.md` 2.3: every registry parameter moves the character, except the listed ones.
#[test]
fn every_parameter_moves_the_character() {
    let registry = real_registry();
    assert!(registry.len() >= 224, "{} parameters", registry.len());
    let generated = template(&Options { lods: false, ..Options::default() }).unwrap().template;
    let mut character = Character::new(Arc::new(generated), &registry);
    let fs = MemoryFs::new();
    let clock = ManualClock::new(1_763_800_000_000);
    let author = Author { id: HUMAN.into(), kind: AuthorKind::Human, name: "Test".into() };
    let mut doc = Document::new(Id128::from_u128(0x2_3), "sliders", "humanart-template", 1, 0, author);
    // One procedural layer of every generator, overriding nothing: the global values reach them.
    for (k, kind) in GeneratorKind::ALL.into_iter().enumerate() {
        doc.skin_layers.push(SkinLayer {
            id: Id128::from_u128(0x5_0000 + k as u128),
            name: kind.name().to_owned(),
            channels: Vec::new(),
            blend: SkinBlend::Mix,
            opacity: 1.0,
            region: None,
            source: SkinSource::Procedural {
                generator: kind.name().to_owned(),
                params: BTreeMap::new(),
                seed: k as u64,
            },
            locked: false,
            visible: true,
            author: HUMAN.to_owned(),
        });
    }
    let mut session = Session::new(doc, &registry, &fs, &clock, SessionConfig::DEFAULT, Id128::from_u128(23));

    // group -> (parameters, moving)
    let mut groups: Vec<(String, usize, usize)> = Vec::new();
    let mut dead: Vec<String> = Vec::new();
    let mut report = String::new();
    for p in registry.params() {
        let e = effect(&mut character, &mut session, &registry, p);
        let group = p.group.split('/').next().unwrap_or("").to_owned();
        let slot = groups.iter().position(|g| g.0 == group).unwrap_or_else(|| {
            groups.push((group, 0, 0));
            groups.len() - 1
        });
        groups[slot].1 += 1;
        let line = match &e {
            Effect::Mesh(m) => format!("{:<48} mesh {:.3} mm", p.id, m * 1000.0),
            Effect::Skin => format!("{:<48} skin", p.id),
            Effect::Material => format!("{:<48} eye/mouth material", p.id),
            Effect::Groom => format!("{:<48} hair groom set", p.id),
            Effect::Layers => format!("{:<48} skin feature layers", p.id),
            Effect::ReadOnly => format!("{:<48} read-only measurement", p.id),
            Effect::Nothing => {
                dead.push(p.id.clone());
                format!("{:<48} NOTHING ({} {})", p.id, p.evaluation.kind_name(), p.evaluation.target().unwrap_or(""))
            }
        };
        if matches!(e, Effect::Mesh(_) | Effect::Skin | Effect::Material | Effect::Groom | Effect::Layers) {
            groups[slot].2 += 1;
        }
        writeln!(report, "{line}").unwrap();
    }
    for (g, n, moving) in &groups {
        writeln!(report, "group {g:<16} {moving:>3} of {n:>3} move the character").unwrap();
    }
    println!("{report}");
    let known: Vec<&str> = KNOWN_DEAD.iter().map(|(id, _)| *id).collect();
    let new_dead: Vec<&String> = dead.iter().filter(|d| !known.contains(&d.as_str())).collect();
    let revived: Vec<&&str> = known.iter().filter(|k| !dead.iter().any(|d| d == **k)).collect();
    assert!(new_dead.is_empty(), "parameters that do nothing (not in KNOWN_DEAD): {new_dead:?}");
    assert!(revived.is_empty(), "listed as dead but they move now (update KNOWN_DEAD): {revived:?}");
}
