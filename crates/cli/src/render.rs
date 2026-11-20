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
use humanart_doc::registry::{ParamValue, Registry, RegistryLimits};
use humanart_gpu::resources::{Budget, Resources};
use humanart_gpu::{DeviceRequest, Gpu};
use humanart_model::generate::build::{Options, template};
use humanart_model::hair::ScalpStyle;
use humanart_model::{PackageLimits, Template, read_package};
use humanart_render::capture::{encode_png_rgba8, read_texture};
use humanart_render::light::BuiltinRig;
use humanart_render::renderer::Quality;

use crate::args::{CommandSpec, OptSpec, Parsed};
use crate::error::CliError;

/// The views `--view` takes, in [`Framing`] order.
pub const VIEWS: [&str; 4] = ["body", "head", "three-quarter", "profile"];
const FRAMINGS: [Framing; 4] = [Framing::Body, Framing::Head, Framing::ThreeQuarter, Framing::Profile];
/// The tiers `--quality` takes.
pub const QUALITIES: [&str; 3] = ["interactive", "review", "still"];
const TIERS: [Quality; 3] = [Quality::Interactive, Quality::Review, Quality::Still];
/// The rigs `--rig` takes.
pub const RIGS: [&str; 7] =
    ["neutral-studio", "flat-beauty", "harsh-sidelight", "backlight", "overcast-outdoor", "golden-hour", "reference"];
const BUILTIN_RIGS: [BuiltinRig; 7] = [
    BuiltinRig::NeutralStudio,
    BuiltinRig::FlatBeauty,
    BuiltinRig::HarshSidelight,
    BuiltinRig::Backlight,
    BuiltinRig::OvercastOutdoor,
    BuiltinRig::GoldenHour,
    BuiltinRig::ReferenceMatched,
];
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

const OPTIONS: [OptSpec; 16] = [
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
        help: "Light rig: neutral-studio, flat-beauty, harsh-sidelight, backlight, overcast-outdoor, golden-hour, reference.",
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
#[derive(Clone, Debug, PartialEq, Eq)]
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
        })
    }
}

/// The `--set` list: `id=value` pairs separated by commas.
fn set_values(list: &str) -> Result<Vec<(String, String)>, CliError> {
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

/// The hair options of a parsed `render` command line.
fn hair_options(p: &Parsed) -> Result<Option<HairOptions>, CliError> {
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

/// Loads every `*.json` registry file under `<data>/template/params`.
fn load_registry(data: &Path) -> Result<Registry, CliError> {
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
fn load_template(data: &Path) -> Result<(Template, &'static str), CliError> {
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
    if !job.params.is_empty() {
        let mut doc = Document::new(
            Id128::from_u128(1),
            "render",
            "template",
            1,
            0,
            Author { id: "human.cli".into(), kind: AuthorKind::Human, name: "humanart-cli".into() },
        );
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
    }
    viewer.frame_model(model, job.framing);
    if let Some(close_up) = job.close_up {
        let (w, h) = viewer.size();
        #[allow(clippy::cast_precision_loss)] // Image sides are at most MAX_SIDE.
        let aspect = w.max(1) as f32 / h.max(1) as f32;
        let camera = close_up_camera(model, close_up, job.framing, aspect)
            .ok_or(CliError::Data("the template has no eye or lip landmarks to frame a close-up on".to_owned()))?;
        viewer.set_camera(camera);
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
        for bad in [
            &["--view", "side"][..],
            &["--width", "0"],
            &["--frames", "5000"],
            &["--skin", "shiny"],
            &["--hair", "mohawk"],
            &["--hair-strands", "500000"],
            &["--body-hair", "yes"],
        ] {
            assert!(matches!(job(bad), Err(CliError::Usage(_))), "{bad:?}");
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
