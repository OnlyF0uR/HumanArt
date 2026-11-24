//! `humanart-cli render` `[CLI-01]`: one still of the character, rendered offscreen by the same
//! engine viewer the editor's viewport uses (`humanart_agent::engine::Viewer`), so a headless
//! still shows what the viewport shows: the template's skin part as surfaced skin (compositor,
//! zone masks, micro-detail) unless `--skin plain` asks for the old single-material look.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use humanart_agent::engine::features::{CloseUp, close_up_camera};
use humanart_agent::engine::{EngineError, Framing, HairOptions, HairShade, Model, Viewer, ViewerOptions};
use humanart_core::id::Id128;
use humanart_doc::document::{Author, AuthorKind, Document};
use humanart_doc::expression_drive::ExpressionDrive;
use humanart_doc::registry::{ParamValue, Registry, RegistryLimits};
use humanart_gpu::resources::{Budget, Resources};
use humanart_gpu::{DeviceRequest, Gpu};
use humanart_math::Vec3;
use humanart_model::expression::presets::Presets;
use humanart_model::generate::build::{Options, template};
use humanart_model::hair::ScalpStyle;
use humanart_model::{PackageLimits, Template, read_package};
use humanart_render::camera::{Camera, Lens, PortraitPreset, PortraitSubject};
use humanart_render::capture::{encode_png_rgba8, read_texture};
use humanart_render::light::{BackdropChoice, BuiltinRig};
use humanart_render::renderer::Quality;

use crate::args::{CommandSpec, OptSpec, Parsed};
use crate::error::CliError;

/// The views `--view` takes, in [`Framing`] order.
pub const VIEWS: [&str; 4] = ["body", "head", "three-quarter", "profile"];
const FRAMINGS: [Framing; 4] = [Framing::Body, Framing::Head, Framing::ThreeQuarter, Framing::Profile];
/// The tiers `--quality` takes.
pub const QUALITIES: [&str; 3] = ["interactive", "review", "still"];
pub(crate) const TIERS: [Quality; 3] = [Quality::Interactive, Quality::Review, Quality::Still];
/// The rigs `--rig` takes.
pub const RIGS: [&str; 13] = [
    "neutral-studio",
    "flat-beauty",
    "harsh-sidelight",
    "backlight",
    "overcast-outdoor",
    "golden-hour",
    "reference",
    "three-point",
    "rembrandt",
    "loop",
    "clamshell",
    "rim-only",
    "sunset",
];
/// The rigs of [`RIGS`], in its order ([`BuiltinRig::ALL`]'s).
pub(crate) const BUILTIN_RIGS: [BuiltinRig; 13] = BuiltinRig::ALL;
/// The cameras `--camera` takes: `view` (the `--view` framing), then the portrait presets.
pub const CAMERAS: [&str; 6] = ["view", "head-and-shoulders", "tight-face", "full-body", "three-quarter", "profile"];
/// The backdrops `--background` takes, in [`BACKDROPS`] order.
pub const BACKGROUNDS: [&str; 4] = ["studio", "rig", "flat", "environment"];
const BACKDROPS: [BackdropChoice; 4] =
    [BackdropChoice::Studio, BackdropChoice::Rig, BackdropChoice::Flat, BackdropChoice::Environment];
/// Largest focus distance `--focus` takes, metres.
pub const MAX_FOCUS_M: f32 = 100.0;
/// The hair `--hair` takes: `none` (no hair at all), then the scalp styles (`bald` keeps brows,
/// lashes and peach fuzz).
pub const HAIR_STYLES: [&str; 8] = ["none", "bald", "short", "medium-straight", "long-wavy", "curly", "coily", "afro"];
/// The colours `--hair-color` takes.
pub const HAIR_SHADES: [&str; 5] = ["black", "dark-brown", "brown", "blonde", "auburn"];
/// Most scalp strands `--hair-strands` takes.
pub const MAX_HAIR_STRANDS: u32 = 400_000;
/// Largest image side.
pub const MAX_SIDE: u32 = 8192;
/// Most frames accumulated for one still.
pub const MAX_FRAMES: u32 = 1024;

/// The close-ups `--close-up` takes (eyes and mouth macro stills).
pub const CLOSE_UPS: [&str; 3] = ["none", "eye", "mouth"];
/// Most `--set` values on one command line.
pub const MAX_SET_VALUES: usize = 64;
/// Most `--pose` joints on one command line.
pub const MAX_POSE_JOINTS: usize = 64;
/// Most axis steps of one `--pose` rotation.
pub const MAX_POSE_STEPS: usize = 8;

const OPTIONS: [OptSpec; 23] = [
    OptSpec {
        name: "camera",
        value: Some("NAME"),
        default: Some("view"),
        help: "Portrait camera: view (use --view), head-and-shoulders (85 mm), tight-face (105 mm), full-body (50 mm), three-quarter, profile.",
    },
    OptSpec {
        name: "aperture",
        value: Some("F"),
        default: Some("off"),
        help: "Depth of field: the lens's f-number (0.7 to 64, e.g. 2.8), or off for everything sharp.",
    },
    OptSpec {
        name: "focus",
        value: Some("M"),
        default: Some("eye"),
        help: "Focus distance in metres along the view, or eye: the eye nearer the camera.",
    },
    OptSpec {
        name: "background",
        value: Some("NAME"),
        default: Some("studio"),
        help: "studio (seamless paper lit by the rig; outdoor rigs keep their sky), rig (the rig's own), flat (grey) or environment.",
    },
    OptSpec {
        name: "occlusion",
        value: Some("ON"),
        default: Some("on"),
        help: "Screen-space ambient occlusion and contact shadows: on or off.",
    },
    OptSpec {
        name: "close-up",
        value: Some("NAME"),
        default: Some("none"),
        help: "none, eye (the left eye) or mouth: a macro still seen from the --view direction.",
    },
    OptSpec {
        name: "set",
        value: Some("ID=V,..."),
        default: None,
        help: "Scalar parameter values before the render (e.g. expression.au26_jaw_drop=0.6).",
    },
    OptSpec {
        name: "preset",
        value: Some("NAME"),
        default: Some("none"),
        help: "Expression preset before --set: none, neutral, {happiness,sadness,surprise,fear,anger,disgust,contempt}.{low,medium,high} or viseme.{sil,PP,FF,TH,DD,kk,CH,SS,nn,RR,aa,E,ih,oh,ou}.",
    },
    OptSpec {
        name: "pose",
        value: Some("JOINT=ROT,..."),
        default: None,
        help: "Joint rotations relative to rest, in the joint's frame (Y along the bone): each ROT is axis letters with degrees applied in order, e.g. L.forearm=z120,L.upper-arm=x-70y20.",
    },
    OptSpec {
        name: "view",
        value: Some("NAME"),
        default: Some("body"),
        help: "Framing: body, head, three-quarter or profile.",
    },
    OptSpec { name: "out", value: Some("FILE"), default: Some("render.png"), help: "PNG file to write." },
    OptSpec { name: "width", value: Some("PX"), default: Some("1080"), help: "Image width." },
    OptSpec { name: "height", value: Some("PX"), default: Some("1350"), help: "Image height." },
    OptSpec {
        name: "frames",
        value: Some("N"),
        default: Some("32"),
        help: "Frames rendered before the capture (temporal accumulation settles).",
    },
    OptSpec {
        name: "skin",
        value: Some("MODE"),
        default: Some("surface"),
        help: "surface: surfaced skin as in the viewport; plain: one material.",
    },
    OptSpec { name: "quality", value: Some("TIER"), default: Some("review"), help: "interactive, review or still." },
    OptSpec {
        name: "rig",
        value: Some("NAME"),
        default: Some("neutral-studio"),
        help: "Light rig: neutral-studio, flat-beauty, harsh-sidelight, backlight, overcast-outdoor, golden-hour, reference, three-point, rembrandt, loop, clamshell, rim-only, sunset.",
    },
    OptSpec {
        name: "hair",
        value: Some("STYLE"),
        default: Some("short"),
        help: "Hair: none, bald (brows, lashes, fuzz only), short, medium-straight, long-wavy, curly, coily, afro.",
    },
    OptSpec {
        name: "hair-color",
        value: Some("SHADE"),
        default: Some("dark-brown"),
        help: "Hair colour: black, dark-brown, brown, blonde, auburn.",
    },
    OptSpec { name: "hair-strands", value: Some("N"), default: Some("100000"), help: "Scalp hair strands." },
    OptSpec { name: "body-hair", value: Some("ON"), default: Some("off"), help: "Sparse arm and leg hair: on or off." },
    OptSpec {
        name: "data",
        value: Some("DIR"),
        default: Some("data"),
        help: "Data folder (template/params registry, template/template.antmpl).",
    },
    OptSpec { name: "json-errors", value: None, default: None, help: "Errors as one JSON line on stderr." },
];

/// The `render` command.
pub static RENDER: CommandSpec =
    CommandSpec { name: "render", summary: "Renders one still of the character offscreen.", options: &OPTIONS };

/// What `render` was asked to do.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderJob {
    /// The framing.
    pub framing: Framing,
    /// The file to write.
    pub out: PathBuf,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Frames before the capture.
    pub frames: u32,
    /// Surfaced skin or one material.
    pub skin_surface: bool,
    /// The quality tier.
    pub quality: Quality,
    /// The light rig.
    pub rig: BuiltinRig,
    /// The data folder.
    pub data: PathBuf,
    /// The hair (`None`: no hair at all).
    pub hair: Option<HairOptions>,
    /// A macro still of the eye or the mouth instead of the framing's own camera.
    pub close_up: Option<CloseUp>,
    /// Scalar parameter values set before the render: registry id and the value's text (checked
    /// to be a finite number).
    pub params: Vec<(String, String)>,
    /// A built-in expression preset (emotion or viseme, `presets.json`) applied before
    /// [`RenderJob::params`].
    pub preset: Option<String>,
    /// A portrait camera instead of the framing's own.
    pub camera: Option<PortraitPreset>,
    /// Depth of field: the f-number in hundredths (280 is f/2.8); `None`: everything sharp.
    pub aperture: Option<u32>,
    /// The focus distance, millimetres; `None`: the eye nearer the camera.
    pub focus_mm: Option<u32>,
    /// The backdrop.
    pub backdrop: BackdropChoice,
    /// Screen-space ambient occlusion and contact shadows.
    pub occlusion: bool,
    /// Joint rotations relative to rest (joint name, unit quaternion `x, y, z, w`), set on the
    /// document before the render.
    pub pose: Vec<(String, [f64; 4])>,
}

impl RenderJob {
    /// The job of a parsed `render` command line.
    ///
    /// # Errors
    /// [`CliError::Usage`] for values out of range or not among the choices.
    pub fn from_args(p: &Parsed) -> Result<Self, CliError> {
        let pick = |name: &str, choices: &[&str]| p.choice(name, choices);
        Ok(Self {
            framing: FRAMINGS[pick("view", &VIEWS)?],
            out: PathBuf::from(p.get("out").unwrap_or("render.png")),
            width: p.number("width", 1, MAX_SIDE)?,
            height: p.number("height", 1, MAX_SIDE)?,
            frames: p.number("frames", 1, MAX_FRAMES)?,
            skin_surface: pick("skin", &["surface", "plain"])? == 0,
            quality: TIERS[pick("quality", &QUALITIES)?],
            rig: BUILTIN_RIGS[pick("rig", &RIGS)?],
            data: PathBuf::from(p.get("data").unwrap_or("data")),
            hair: hair_options(p)?,
            close_up: [None, Some(CloseUp::Eye), Some(CloseUp::Mouth)][pick("close-up", &CLOSE_UPS)?],
            params: set_values(p.get("set").unwrap_or(""))?,
            preset: preset_name(p.get("preset").unwrap_or("none"))?,
            camera: match pick("camera", &CAMERAS)? {
                0 => None,
                i => PortraitPreset::ALL.get(i - 1).copied(),
            },
            aperture: aperture(p.get("aperture").unwrap_or("off"))?,
            focus_mm: focus(p.get("focus").unwrap_or("eye"))?,
            backdrop: BACKDROPS[pick("background", &BACKGROUNDS)?],
            occlusion: pick("occlusion", &["on", "off"])? == 0,
            pose: pose_values(p.get("pose").unwrap_or(""))?,
        })
    }
}

/// `--aperture`: `off`, or an f-number inside [`Lens::F_NUMBER_RANGE`], kept in hundredths.
fn aperture(text: &str) -> Result<Option<u32>, CliError> {
    if text == "off" {
        return Ok(None);
    }
    let [low, high] = Lens::F_NUMBER_RANGE;
    match text.trim().parse::<f32>() {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Checked to be 0.7..=64.
        Ok(f) if (low..=high).contains(&f) => Ok(Some((f * 100.0).round() as u32)),
        _ => Err(CliError::Usage(crate::args::UsageError(format!(
            "--aperture: `{text}` is not off or an f-number from {low} to {high}"
        )))),
    }
}

/// `--focus`: `eye`, or a distance from 1 cm to [`MAX_FOCUS_M`], kept in millimetres.
fn focus(text: &str) -> Result<Option<u32>, CliError> {
    if text == "eye" {
        return Ok(None);
    }
    match text.trim().parse::<f32>() {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Checked to be 0.01..=100.
        Ok(m) if (0.01..=MAX_FOCUS_M).contains(&m) => Ok(Some((m * 1000.0).round() as u32)),
        _ => Err(CliError::Usage(crate::args::UsageError(format!(
            "--focus: `{text}` is not eye or a distance from 0.01 to {MAX_FOCUS_M} metres"
        )))),
    }
}

/// What the portrait cameras frame on `model`: its head and its standing height.
fn portrait_subject(model: &Model) -> PortraitSubject {
    let (head_center, head_radius) = model.head_bounds();
    let (lo, hi) = model.extent();
    let (feet, top) = if lo.y <= hi.y { (lo.y, hi.y) } else { (head_center.y - 13.0 * head_radius, head_center.y) };
    // The model's head sphere reaches the neck and the ears; a head is 13 % of the stature high.
    PortraitSubject { head_center, head_radius: (0.065 * (top - feet)).min(head_radius), feet, top }
}

/// The distance along `camera`'s view axis to the eye nearer the camera (the front of the left
/// eyeball, or its mirror image across the head's middle), or to the camera's target when the
/// template has no fitted eyeball.
fn eye_focus(model: &Model, camera: &Camera, aspect: f32) -> f32 {
    let along = |p: Vec3| (p - camera.position).dot(camera.forward());
    let Some(left) = close_up_camera(model, CloseUp::Eye, Framing::Head, aspect).map(|c| c.target) else {
        return along(camera.target).max(0.01);
    };
    let middle = model.head_bounds().0.x;
    let right = Vec3::new(2.0 * middle - left.x, left.y, left.z);
    let nearer = if left.distance(camera.position) <= right.distance(camera.position) { left } else { right };
    along(nearer).max(0.01)
}

/// `--preset`: `none`, or the name of a built-in expression preset.
fn preset_name(text: &str) -> Result<Option<String>, CliError> {
    if text == "none" {
        return Ok(None);
    }
    if builtin_presets()?.find(text).is_none() {
        return Err(CliError::Usage(crate::args::UsageError(format!(
            "--preset: `{text}` is not none or an expression preset (neutral, happiness.high, viseme.PP, ...)"
        ))));
    }
    Ok(Some(text.to_owned()))
}

fn builtin_presets() -> Result<Presets, CliError> {
    Presets::builtin().map_err(|e| CliError::Data(format!("expression presets: {e}")))
}

/// The document values of expression preset `name`: for every control of the preset, the registry
/// parameter that drives it and the control's value. The viseme controls are driven by the
/// `expression.viseme.*` parameters (todo 1.8).
///
/// # Errors
/// [`CliError::Data`] when there is no such preset or a control has no parameter.
pub fn preset_values(registry: &Registry, name: &str) -> Result<Vec<(String, f64)>, CliError> {
    let presets = builtin_presets()?;
    let preset = presets.find(name).ok_or_else(|| CliError::Data(format!("no expression preset `{name}`")))?;
    preset
        .controls
        .iter()
        .map(|(control, value)| {
            ExpressionDrive::param_of_control(registry, control)
                .map(|p| (p.id.clone(), f64::from(*value)))
                .ok_or_else(|| CliError::Data(format!("preset `{name}`: no parameter drives the control `{control}`")))
        })
        .collect()
}

/// The `--set` list: `id=value` pairs separated by commas.
pub(crate) fn set_values(list: &str) -> Result<Vec<(String, String)>, CliError> {
    let mut out = Vec::new();
    for item in list.split(',').map(str::trim).filter(|i| !i.is_empty()) {
        let bad = || CliError::Usage(crate::args::UsageError(format!("--set: `{item}` is not ID=NUMBER")));
        let (id, value) = item.split_once('=').ok_or_else(bad)?;
        let value: f64 = value.trim().parse().map_err(|_| bad())?;
        if id.trim().is_empty() || !value.is_finite() || out.len() >= MAX_SET_VALUES {
            return Err(bad());
        }
        out.push((id.trim().to_owned(), value.to_string()));
    }
    Ok(out)
}

/// The `--pose` list: `JOINT=ROT` pairs separated by commas, each `ROT` a sequence of an axis
/// letter (`x`, `y`, `z`) and degrees, applied in order (`z120x-10` turns 120 degrees about Z, then
/// 10 degrees back about the turned X). Joint names are checked against the template later.
fn pose_values(list: &str) -> Result<Vec<(String, [f64; 4])>, CliError> {
    let mut out = Vec::new();
    for item in list.split(',').map(str::trim).filter(|i| !i.is_empty()) {
        let bad = || {
            CliError::Usage(crate::args::UsageError(format!(
                "--pose: `{item}` is not JOINT=ROT with ROT like z120 or x-30y15 (at most {MAX_POSE_STEPS} steps)"
            )))
        };
        let (joint, rot) = item.split_once('=').ok_or_else(bad)?;
        let (joint, rot) = (joint.trim(), rot.trim());
        if joint.is_empty() || rot.is_empty() || out.len() >= MAX_POSE_JOINTS {
            return Err(bad());
        }
        let mut q = humanart_math::DQuat::IDENTITY;
        let mut steps = 0;
        let mut rest = rot;
        while let Some(axis) = rest.chars().next() {
            let axis = match axis {
                'x' | 'X' => humanart_math::DVec3::X,
                'y' | 'Y' => humanart_math::DVec3::Y,
                'z' | 'Z' => humanart_math::DVec3::Z,
                _ => return Err(bad()),
            };
            let tail = &rest[1..];
            let end = tail.find(|c: char| c.is_ascii_alphabetic()).unwrap_or(tail.len());
            let degrees: f64 = tail[..end].parse().map_err(|_| bad())?;
            if !degrees.is_finite() || degrees.abs() > 360.0 {
                return Err(bad());
            }
            q *= humanart_math::DQuat::from_axis_angle(axis, degrees.to_radians());
            rest = &tail[end..];
            steps += 1;
            if steps > MAX_POSE_STEPS {
                return Err(bad());
            }
        }
        out.push((joint.to_owned(), [q.x, q.y, q.z, q.w]));
    }
    Ok(out)
}

/// The hair options of a parsed `render` command line.
pub(crate) fn hair_options(p: &Parsed) -> Result<Option<HairOptions>, CliError> {
    let style = p.choice("hair", &HAIR_STYLES)?;
    let shade =
        HairShade::from_name(HAIR_SHADES[p.choice("hair-color", &HAIR_SHADES)?]).unwrap_or(HairShade::DarkBrown);
    let strands = p.number("hair-strands", 0, MAX_HAIR_STRANDS)?;
    let body_hair = p.choice("body-hair", &["off", "on"])? == 1;
    Ok(HAIR_STYLES.get(style).and_then(|name| ScalpStyle::from_name(name)).map(|style| HairOptions {
        style,
        shade,
        strands,
        body_hair,
        ..HairOptions::default()
    }))
}

/// What a finished render reports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderSummary {
    /// The file written.
    pub out: PathBuf,
    /// Where the template came from.
    pub template: &'static str,
    /// The GPU adapter.
    pub adapter: String,
    /// Whether the skin surface is on.
    pub skin_surface: bool,
    /// Zone names the skin bake did not find (empty when complete).
    pub missing_names: Vec<&'static str>,
    /// Hair strands drawn (0 without hair).
    pub hair_strands: u64,
}

/// An empty document: every parameter at its registry default when a model syncs to it.
pub(crate) fn default_document() -> Document {
    Document::new(
        Id128::from_u128(1),
        "render",
        "template",
        1,
        0,
        Author { id: "human.cli".into(), kind: AuthorKind::Human, name: "humanart-cli".into() },
    )
}

/// Loads every `*.json` registry file under `<data>/template/params`.
pub(crate) fn load_registry(data: &Path) -> Result<Registry, CliError> {
    let dir = data.join("template").join("params");
    let entries = std::fs::read_dir(&dir).map_err(|e| CliError::io(&dir, e))?;
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    for entry in entries {
        let path = entry.map_err(|e| CliError::io(&dir, e))?.path();
        if path.extension().is_some_and(|x| x == "json") {
            let bytes = std::fs::read(&path).map_err(|e| CliError::io(&path, e))?;
            files.push((path.display().to_string(), bytes));
        }
        if files.len() as u64 > RegistryLimits::DEFAULT.max_files {
            return Err(CliError::Data(format!("more than {} registry files", RegistryLimits::DEFAULT.max_files)));
        }
    }
    files.sort();
    let list: Vec<(&str, &[u8])> = files.iter().map(|(n, b)| (n.as_str(), b.as_slice())).collect();
    Registry::from_files(&list, &RegistryLimits::DEFAULT).map_err(|e| CliError::Data(format!("registry: {e}")))
}

/// The template package `<data>/template/template.antmpl`, or the generated template when there is
/// no package (as the editor does).
pub(crate) fn load_template(data: &Path) -> Result<(Template, &'static str), CliError> {
    let path = data.join("template").join("template.antmpl");
    if path.exists() {
        let bytes = std::fs::read(&path).map_err(|e| CliError::io(&path, e))?;
        let t =
            read_package(&bytes, &PackageLimits::default()).map_err(|e| CliError::Data(format!("template: {e}")))?;
        return Ok((t, "package"));
    }
    let generated = template(&Options { lods: false, ..Options::default() })
        .map_err(|e| CliError::Data(format!("template: {e}")))?;
    Ok((generated.template, "generated"))
}

/// Runs `job`: loads the registry and template, renders `job.frames` frames on a headless GPU and
/// writes the display image as PNG.
///
/// # Errors
/// [`CliError::Data`] for the registry or template, [`CliError::Gpu`] when there is no adapter,
/// [`CliError::Render`] when the engine fails, [`CliError::Io`] for the output file.
pub fn run(job: &RenderJob) -> Result<RenderSummary, CliError> {
    let registry = load_registry(&job.data)?;
    let (template, source) = load_template(&job.data)?;
    let mut model = Model::new(Arc::new(template), &registry);
    // Always through a document, so the registry defaults apply (`identity.sex` -1: the template's
    // rest shape is no character, todo 1.3).
    {
        let mut doc = default_document();
        if let Some(name) = &job.preset {
            for (id, value) in preset_values(&registry, name)? {
                doc.params.insert(id, ParamValue::Scalar(value));
            }
        }
        for (joint, q) in &job.pose {
            if model.template().skeleton().find(joint).is_none() {
                return Err(CliError::Data(format!("--pose: the template has no joint `{joint}`")));
            }
            doc.pose.insert(joint.clone(), *q);
        }
        for (id, value) in &job.params {
            if !registry.params().iter().any(|p| p.id == *id) {
                return Err(CliError::Data(format!("--set: no parameter `{id}` in the registry")));
            }
            let value: f64 = value.parse().map_err(|_| CliError::Data(format!("--set: `{value}` is not a number")))?;
            doc.params.insert(id.clone(), ParamValue::Scalar(value));
        }
        model.sync(&doc, &registry)?;
    }
    let gpu = Gpu::headless_blocking(&DeviceRequest::default()).map_err(|e| CliError::Gpu(e.to_string()))?;
    let mut resources = Resources::new(&gpu, Budget::UNLIMITED);
    let options = ViewerOptions { skin_surface: job.skin_surface, hair: job.hair };
    let mut viewer = Viewer::with_options(&gpu, &mut resources, &model, job.width, job.height, options)?;
    let result = render_with(&gpu, &mut resources, &mut viewer, &mut model, job);
    let summary = RenderSummary {
        out: job.out.clone(),
        template: source,
        adapter: gpu.adapter_info().name.clone(),
        skin_surface: viewer.skin().is_some(),
        missing_names: viewer
            .skin()
            .map(|s| {
                let r = s.report();
                r.missing_regions.iter().chain(&r.missing_joints).chain(&r.missing_landmarks).copied().collect()
            })
            .unwrap_or_default(),
        hair_strands: viewer.hair().map_or(0, humanart_agent::engine::hair::CharacterHair::strand_count),
    };
    if job.skin_surface
        && let Some(e) = viewer.skin_error()
    {
        let message = format!("the skin surface could not be built: {e}");
        viewer.destroy(&mut resources);
        return Err(CliError::Data(message));
    }
    if job.hair.is_some()
        && let Some(e) = viewer.hair_error()
    {
        let message = format!("the hair could not be built: {e}");
        viewer.destroy(&mut resources);
        return Err(CliError::Data(message));
    }
    viewer.destroy(&mut resources);
    result.map(|()| summary)
}

fn render_with(
    gpu: &Gpu,
    resources: &mut Resources,
    viewer: &mut Viewer,
    model: &mut Model,
    job: &RenderJob,
) -> Result<(), CliError> {
    {
        let s = viewer.settings_mut();
        s.quality = job.quality;
        s.rig = job.rig;
        s.backdrop = job.backdrop;
        s.occlusion = job.occlusion;
    }
    viewer.frame_model(model, job.framing);
    let (w, h) = viewer.size();
    #[allow(clippy::cast_precision_loss)] // Image sides are at most MAX_SIDE.
    let aspect = w.max(1) as f32 / h.max(1) as f32;
    if let Some(preset) = job.camera {
        viewer.set_camera(Camera::portrait(preset, &portrait_subject(model), aspect));
    }
    if let Some(close_up) = job.close_up {
        let camera = close_up_camera(model, close_up, job.framing, aspect)
            .ok_or(CliError::Data("the template has no eye or lip landmarks to frame a close-up on".to_owned()))?;
        viewer.set_camera(camera);
    }
    if let Some(hundredths) = job.aperture {
        let camera = viewer.settings().camera;
        #[allow(clippy::cast_precision_loss)] // At most 6400 and 100 000.
        let (f_number, focus) = (
            hundredths as f32 / 100.0,
            job.focus_mm.map_or_else(|| eye_focus(model, &camera, aspect), |mm| mm as f32 / 1000.0),
        );
        viewer.settings_mut().lens = Some(Lens::new(f_number, focus));
    }
    for _ in 0..job.frames {
        let mut encoder =
            gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cli render") });
        viewer.record(resources, &mut encoder, model, &[])?;
        gpu.queue.submit([encoder.finish()]);
        viewer.after_submit();
    }
    let texture = viewer.display_texture().ok_or(EngineError::Invalid { what: "the viewer has no display image" })?;
    let (w, h, rgba) = read_texture(resources, texture, 0).map_err(EngineError::from)?;
    let png = encode_png_rgba8(w, h, &rgba).map_err(EngineError::from)?;
    if let Some(dir) = job.out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| CliError::io(dir, e))?;
    }
    std::fs::write(&job.out, png).map_err(|e| CliError::io(&job.out, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(list: &[&str]) -> Result<RenderJob, CliError> {
        let args: Vec<String> = list.iter().map(|s| (*s).to_owned()).collect();
        RenderJob::from_args(&Parsed::parse(&RENDER, &args).map_err(CliError::Usage)?)
    }

    #[test]
    fn cli_01_render_arguments_map_to_the_job() {
        let j = job(&["--view", "head", "--skin", "plain", "--quality", "still", "--rig", "golden-hour"]).unwrap();
        assert_eq!(j.framing, Framing::Head);
        assert!(!j.skin_surface);
        assert_eq!(j.quality, Quality::Still);
        assert_eq!(j.rig, BuiltinRig::GoldenHour);
        let h = job(&["--hair", "long-wavy", "--hair-color", "auburn", "--hair-strands", "5000", "--body-hair", "on"])
            .unwrap()
            .hair
            .unwrap();
        assert_eq!((h.style, h.shade, h.strands, h.body_hair), (ScalpStyle::LongWavy, HairShade::Auburn, 5000, true));
        assert_eq!(job(&["--hair", "none"]).unwrap().hair, None);
        assert_eq!(job(&["--hair", "bald"]).unwrap().hair.map(|h| h.style), Some(ScalpStyle::Bald));
        let d = job(&[]).unwrap();
        assert_eq!((d.framing, d.width, d.height, d.frames), (Framing::Body, 1080, 1350, 32));
        assert!(d.skin_surface);
        // todo 1.9: review renders stand in front of studio paper with occlusion, everything sharp.
        assert_eq!(
            (d.camera, d.aperture, d.focus_mm, d.backdrop, d.occlusion),
            (None, None, None, BackdropChoice::Studio, true)
        );
        for bad in [
            &["--view", "side"][..],
            &["--width", "0"],
            &["--frames", "5000"],
            &["--skin", "shiny"],
            &["--hair", "mohawk"],
            &["--hair-strands", "500000"],
            &["--body-hair", "yes"],
            &["--rig", "disco"],
            &["--camera", "selfie"],
            &["--aperture", "0.1"],
            &["--aperture", "wide"],
            &["--aperture", "NaN"],
            &["--focus", "0"],
            &["--focus", "far"],
            &["--focus", "1000"],
            &["--background", "beach"],
            &["--occlusion", "maybe"],
            &["--preset", "smirk"],
            &["--preset", "viseme.pp"],
        ] {
            assert!(matches!(job(bad), Err(CliError::Usage(_))), "{bad:?}");
        }
    }

    /// todo 1.3: `--pose` turns joints for posed stills (pose correctives).
    #[test]
    fn todo_1_3_pose_from_the_command_line() {
        assert!(job(&[]).unwrap().pose.is_empty());
        let p = job(&["--pose", "L.forearm=z90, R.thigh=x-30y10"]).unwrap().pose;
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].0, "L.forearm");
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let q = p[0].1;
        assert!((q[2] - s).abs() < 1e-12 && (q[3] - s).abs() < 1e-12 && q[0].abs() < 1e-12 && q[1].abs() < 1e-12);
        let q = p[1].1;
        assert!(((q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt() - 1.0).abs() < 1e-12);
        for bad in [
            "L.forearm",
            "=z90",
            "L.forearm=",
            "L.forearm=w90",
            "L.forearm=z",
            "L.forearm=z9000",
            "a=x1x1x1x1x1x1x1x1x1",
        ] {
            assert!(matches!(job(&["--pose", bad]), Err(CliError::Usage(_))), "{bad}");
        }
    }

    /// todo 1.8: expression presets, visemes included, are set from the command line; every
    /// control of every built-in preset reaches a registry parameter of the repository's data.
    #[test]
    fn todo_1_8_presets_and_visemes_from_the_command_line() {
        assert_eq!(job(&[]).unwrap().preset, None);
        assert_eq!(job(&["--preset", "viseme.PP"]).unwrap().preset.as_deref(), Some("viseme.PP"));
        assert_eq!(job(&["--preset", "contempt.high"]).unwrap().preset.as_deref(), Some("contempt.high"));
        let data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data");
        let registry = load_registry(&data).unwrap();
        let presets = Presets::builtin().unwrap();
        for p in &presets.presets {
            let values = preset_values(&registry, &p.name).unwrap();
            assert_eq!(values.len(), p.controls.len(), "{}", p.name);
            for (id, v) in &values {
                let param = registry.get(id).unwrap();
                assert!(param.check_value(&ParamValue::Scalar(*v)).is_ok(), "{}: {id} = {v}", p.name);
            }
        }
        let aa = preset_values(&registry, "viseme.aa").unwrap();
        assert!(aa.iter().any(|(id, v)| id == "expression.viseme.jaw_open_wide" && *v > 0.0), "{aa:?}");
        assert!(matches!(preset_values(&registry, "viseme.zz"), Err(CliError::Data(_))));
    }

    /// todo 1.9: the light and camera options of the command line.
    #[test]
    fn todo_1_9_light_and_camera_arguments_map_to_the_job() {
        for (name, rig) in RIGS.iter().zip(BuiltinRig::ALL) {
            assert_eq!(job(&["--rig", name]).unwrap().rig, rig);
            assert_eq!(name.replace('-', "_").replace("reference", "reference_matched"), rig.name());
        }
        for (name, preset) in CAMERAS.iter().skip(1).zip(PortraitPreset::ALL) {
            assert_eq!(job(&["--camera", name]).unwrap().camera, Some(preset));
            assert_eq!(name.replace('-', "_"), preset.name());
        }
        let j =
            job(&["--camera", "tight-face", "--aperture", "2.8", "--focus", "1.25", "--background", "flat"]).unwrap();
        assert_eq!((j.camera, j.aperture, j.focus_mm), (Some(PortraitPreset::TightFace), Some(280), Some(1250)));
        assert_eq!(j.backdrop, BackdropChoice::Flat);
        let j = job(&["--aperture", "1.4", "--background", "environment", "--occlusion", "off"]).unwrap();
        assert_eq!(
            (j.aperture, j.focus_mm, j.backdrop, j.occlusion),
            (Some(140), None, BackdropChoice::Environment, false)
        );
        assert_eq!(job(&["--background", "rig"]).unwrap().backdrop, BackdropChoice::Rig);
        assert_eq!(job(&["--aperture", "off", "--focus", "2"]).unwrap().aperture, None);
    }

    /// todo 1.9: a portrait preset with a lens renders, focused on the nearer eye.
    #[test]
    fn todo_1_9_render_with_a_portrait_camera_and_a_lens() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let out = std::env::temp_dir().join(format!("humanart-cli-lens-{}.png", std::process::id()));
        let j = RenderJob {
            out: out.clone(),
            data: root.join("data"),
            width: 200,
            height: 250,
            frames: 3,
            ..job(&["--camera", "three-quarter", "--aperture", "2", "--rig", "rembrandt", "--hair", "none"]).unwrap()
        };
        run(&j).unwrap();
        let bytes = std::fs::read(&out).unwrap();
        assert_eq!(&bytes[1..4], b"PNG");
        std::fs::remove_file(&out).unwrap();
        // The focus is on the eye nearer the camera: closer than the head's centre in every preset.
        let registry = load_registry(&j.data).unwrap();
        let (template, _) = load_template(&j.data).unwrap();
        let model = Model::new(Arc::new(template), &registry);
        let subject = portrait_subject(&model);
        for preset in PortraitPreset::ALL {
            let camera = Camera::portrait(preset, &subject, 0.8);
            camera.validate().unwrap();
            let focus = eye_focus(&model, &camera, 0.8);
            let centre = (subject.head_center - camera.position).dot(camera.forward());
            assert!(focus > 0.2 && focus < centre, "{}: focus {focus} m, head centre {centre} m", preset.name());
        }
    }

    #[test]
    fn cli_01_render_writes_a_surfaced_still_of_the_template() {
        // [CLI-01] [SKN-01] The headless render of the repository's data writes a PNG through the
        // engine viewer with the skin surface bound (every zone name found on the template).
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let out = std::env::temp_dir().join(format!("humanart-cli-test-{}.png", std::process::id()));
        let j = RenderJob {
            out: out.clone(),
            data: root.join("data"),
            width: 240,
            height: 300,
            frames: 4,
            framing: Framing::Head,
            ..job(&[]).unwrap()
        };
        let summary = run(&j).unwrap();
        assert!(summary.skin_surface, "{summary:?}");
        assert!(summary.missing_names.is_empty(), "{summary:?}");
        let bytes = std::fs::read(&out).unwrap();
        assert_eq!(&bytes[1..4], b"PNG");
        std::fs::remove_file(&out).unwrap();
        let missing = RenderJob { data: root.join("no-such-data-folder"), ..j };
        assert!(matches!(run(&missing), Err(CliError::Io { .. })));
    }
}
