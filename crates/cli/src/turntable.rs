//! `humanart-cli turntable`: `--views` stills of the character, the camera turned around it in
//! equal steps, rendered offscreen through the agent server's render service (the editor's
//! viewer), written as `<out>/turn_000.png`, `turn_001.png`, ... With `--project` the saved
//! document of the project is shown (read only: the project is not opened for editing and its
//! journal is neither replayed nor touched).

use std::path::PathBuf;
use std::sync::Arc;

use humanart_agent::engine::{HairOptions, Model, ViewerOptions};
use humanart_agent::mcp::render::{CAMERA_PRESETS, CameraSpec, RenderService, ViewRequest};
use humanart_platform::StdFileSystem;
use humanart_render::capture::encode_png_rgba8;

use crate::args::{CommandSpec, OptSpec, Parsed, UsageError};
use crate::error::CliError;
use crate::host;
use crate::render::{BUILTIN_RIGS, MAX_FRAMES, MAX_SIDE, QUALITIES, RIGS, TIERS};

/// Most views of one turntable.
pub const MAX_VIEWS: u32 = 360;

const OPTIONS: [OptSpec; 17] = [
    OptSpec { name: "views", value: Some("N"), default: Some("12"), help: "Stills around the character." },
    OptSpec {
        name: "camera",
        value: Some("NAME"),
        default: Some("body"),
        help: "Start camera: body, head, three_quarter, profile or a review camera (front, left_45, ...).",
    },
    OptSpec { name: "out", value: Some("DIR"), default: Some("turntable"), help: "Folder for the PNG files." },
    OptSpec { name: "width", value: Some("PX"), default: Some("512"), help: "Image width." },
    OptSpec { name: "height", value: Some("PX"), default: Some("640"), help: "Image height." },
    OptSpec {
        name: "frames",
        value: Some("N"),
        default: Some("16"),
        help: "Frames rendered per still before the capture (temporal accumulation settles).",
    },
    OptSpec { name: "quality", value: Some("TIER"), default: Some("review"), help: "interactive, review or still." },
    OptSpec { name: "rig", value: Some("NAME"), default: Some("neutral-studio"), help: "Light rig (as for render)." },
    OptSpec {
        name: "skin",
        value: Some("MODE"),
        default: Some("surface"),
        help: "surface: surfaced skin as in the viewport; plain: one material.",
    },
    OptSpec { name: "hair", value: Some("STYLE"), default: Some("short"), help: "Hair (as for render)." },
    OptSpec {
        name: "hair-color",
        value: Some("SHADE"),
        default: Some("dark-brown"),
        help: "Hair colour (as for render).",
    },
    OptSpec { name: "hair-strands", value: Some("N"), default: Some("100000"), help: "Scalp hair strands." },
    OptSpec { name: "body-hair", value: Some("ON"), default: Some("off"), help: "Sparse arm and leg hair: on or off." },
    host::PROJECT,
    host::DATA,
    OptSpec { name: "quiet", value: None, default: None, help: "No progress lines." },
    host::JSON_ERRORS,
];

/// The `turntable` command.
pub static TURNTABLE: CommandSpec = CommandSpec {
    name: "turntable",
    summary: "Renders stills around the character (a turntable) offscreen.",
    options: &OPTIONS,
};

/// What `turntable` was asked to do.
#[derive(Clone, Debug, PartialEq)]
pub struct TurntableJob {
    /// Stills.
    pub views: u32,
    /// The first still's request (its `turn_degrees` is replaced per still).
    pub request: ViewRequest,
    /// The output folder.
    pub out: PathBuf,
    /// How the character is drawn.
    pub options: ViewerOptions,
    /// The project whose saved document is shown.
    pub project: Option<PathBuf>,
    /// The data folder.
    pub data: PathBuf,
    /// No progress lines.
    pub quiet: bool,
}

impl TurntableJob {
    /// The job of a parsed `turntable` command line.
    ///
    /// # Errors
    /// [`CliError::Usage`] for values out of range or not among the choices.
    pub fn from_args(p: &Parsed) -> Result<Self, CliError> {
        let camera_name = p.get("camera").unwrap_or("body");
        let camera = CameraSpec::preset(camera_name).ok_or_else(|| {
            CliError::Usage(UsageError(format!(
                "--camera: '{camera_name}' is not one of {}",
                CAMERA_PRESETS.join(", ")
            )))
        })?;
        let rig = BUILTIN_RIGS[p.choice("rig", &RIGS)?];
        let quality = TIERS[p.choice("quality", &QUALITIES)?];
        let hair: Option<HairOptions> = crate::render::hair_options(p)?;
        Ok(Self {
            views: p.number("views", 1, MAX_VIEWS)?,
            request: ViewRequest {
                camera,
                rig,
                quality,
                width: p.number("width", 1, MAX_SIDE)?,
                height: p.number("height", 1, MAX_SIDE)?,
                frames: p.number("frames", 1, MAX_FRAMES)?,
                ..ViewRequest::default()
            },
            out: PathBuf::from(p.get("out").unwrap_or("turntable")),
            options: ViewerOptions { skin_surface: p.choice("skin", &["surface", "plain"])? == 0, hair },
            project: p.get("project").map(PathBuf::from),
            data: PathBuf::from(p.get("data").unwrap_or("data")),
            quiet: p.get("quiet").is_some(),
        })
    }

    /// The camera turn of still `i`, degrees.
    #[allow(clippy::cast_precision_loss)] // At most MAX_VIEWS stills.
    pub fn degrees(&self, i: u32) -> f32 {
        360.0 * i as f32 / self.views.max(1) as f32
    }
}

/// What a finished turntable reports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TurntableSummary {
    /// The files written, in turn order.
    pub files: Vec<PathBuf>,
    /// The GPU adapter.
    pub adapter: String,
}

/// Runs `job`.
///
/// # Errors
/// [`CliError::Data`] for the data folder, [`CliError::Project`] for a project that does not
/// read, [`CliError::Render`] when the renderer fails, [`CliError::Io`] for the output files.
pub fn run(job: &TurntableJob) -> Result<TurntableSummary, CliError> {
    let registry = crate::render::load_registry(&job.data)?;
    let (template, _) = crate::render::load_template(&job.data)?;
    let mut model = Model::new(Arc::new(template), &registry);
    if let Some(dir) = &job.project {
        let (parent, name) = host::split_project(dir)?;
        let fs = StdFileSystem::rooted(&parent).map_err(|e| CliError::Project(format!("{}: {e}", parent.display())))?;
        let (doc, _) = host::read_document(&fs, &name, &registry)
            .map_err(|e| CliError::Project(format!("{}: {e}", dir.display())))?;
        model.sync(&doc, &registry)?;
    }
    std::fs::create_dir_all(&job.out).map_err(|e| CliError::io(&job.out, e))?;
    let mut service = RenderService::new(job.options);
    let mut files = Vec::with_capacity(job.views as usize);
    for i in 0..job.views {
        let request = ViewRequest { turn_degrees: job.degrees(i), ..job.request };
        let image = service.render(&mut model, &request).map_err(|e| match e {
            humanart_agent::engine::EngineError::Gpu(g) => CliError::Gpu(g.to_string()),
            other => CliError::Render(other),
        })?;
        let png = encode_png_rgba8(image.width, image.height, &image.rgba)
            .map_err(|e| CliError::Render(humanart_agent::engine::EngineError::from(e)))?;
        let path = job.out.join(format!("turn_{i:03}.png"));
        std::fs::write(&path, png).map_err(|e| CliError::io(&path, e))?;
        if !job.quiet {
            eprintln!("{} turntable: {} ({:.1} degrees)", crate::PROGRAM, path.display(), request.turn_degrees);
        }
        files.push(path);
    }
    Ok(TurntableSummary { files, adapter: service.adapter().to_owned() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_render::light::BuiltinRig;
    use humanart_render::renderer::Quality;

    fn job(list: &[&str]) -> Result<TurntableJob, CliError> {
        let args: Vec<String> = list.iter().map(|s| (*s).to_owned()).collect();
        TurntableJob::from_args(&Parsed::parse(&TURNTABLE, &args)?)
    }

    #[test]
    fn turntable_arguments_map_to_the_job() {
        let d = job(&[]).unwrap();
        assert_eq!((d.views, d.request.width, d.request.height, d.request.frames), (12, 512, 640, 16));
        assert!((d.degrees(3) - 90.0).abs() < 1e-4);
        let j =
            job(&["--views", "4", "--camera", "front", "--rig", "golden-hour", "--quality", "still", "--hair", "none"])
                .unwrap();
        assert_eq!(j.request.rig, BuiltinRig::GoldenHour);
        assert_eq!(j.request.quality, Quality::Still);
        assert!(j.options.hair.is_none());
        assert!((j.degrees(1) - 90.0).abs() < 1e-4);
        for rig in RIGS {
            assert!(job(&["--rig", rig]).is_ok(), "{rig}");
        }
        for bad in
            [&["--views", "0"][..], &["--views", "361"], &["--camera", "behind"], &["--width", "0"], &["--skin", "x"]]
        {
            assert!(matches!(job(bad), Err(CliError::Usage(_))), "{bad:?}");
        }
    }

    #[test]
    fn turntable_writes_one_still_per_view() {
        // A small GPU run: two views of the generated template, one frame each.
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let out = std::env::temp_dir().join(format!("humanart-cli-turntable-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let mut j =
            job(&["--views", "2", "--width", "48", "--height", "64", "--frames", "1", "--hair", "none", "--quiet"])
                .unwrap();
        j.out.clone_from(&out);
        j.data = root.join("data");
        let summary = run(&j).unwrap();
        assert_eq!(summary.files.len(), 2);
        let first = std::fs::read(&summary.files[0]).unwrap();
        let second = std::fs::read(&summary.files[1]).unwrap();
        assert_eq!(&first[1..4], b"PNG");
        assert_ne!(first, second, "the camera turned between the stills");
        std::fs::remove_dir_all(&out).unwrap();
    }
}
