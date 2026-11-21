//! `humanart-cli render-mesh` (WP-25, the data check of `todo.md` item 1.1): a still of an
//! external mesh (a scan, an anatomy atlas part) through our renderer, with the same light rigs,
//! review cameras, quality tiers and tone mapping as `render`, so it can be put next to the
//! template.
//!
//! The mesh is read with `humanart_io::obj` into a [`ScanMesh`], normalised to our conventions
//! (metres, Y up, facing +Z, standing on `y = 0`; units and forward axis guessed unless given) and
//! framed exactly as the engine viewer frames the template: the body by its extent, the head by
//! the top 13 % of the height, which also places the light rig.
//!
//! Without `--albedo` the mesh is drawn in the viewer's plain skin material (as `render --skin
//! plain` draws the template). With `--albedo` it is drawn as surfaced skin: the scan's texture,
//! converted to linear and box-filtered to `--albedo-size`, is written as a full-coverage colour
//! override layer `[SKN-12]` into a one-tile skin compositor, so the albedo replaces the pigment
//! model while scattering, specular and the subsurface passes stay the template's. No
//! micro-detail is bound: what shows is the scan's own detail.
//!
//! `--frame-obj FILE` loads a second mesh in the same file frame that is normalised and framed
//! with the parts but not drawn: a fitted head is then framed exactly like the scan it was fitted to.
//!
//! `--obj-list FILE` draws several parts that share one frame (an atlas's bones and muscles): one
//! line per part, `path` or `path #rrggbb` (sRGB colour of a plain material).

use std::path::{Path, PathBuf};

use humanart_agent::engine::EngineError;
use humanart_agent::engine::skin::SURFACED_SKIN_MATERIAL;
use humanart_agent::engine::view::SKIN_MATERIAL;
use humanart_gpu::resources::{Budget, Resources};
use humanart_gpu::{DeviceRequest, Gpu};
use humanart_io::Limits;
use humanart_math::color::transfer::linear_to_srgb_rgb_f32;
use humanart_math::color::transfer::srgb_to_linear_f32;
use humanart_math::{Vec2, Vec3};
use humanart_model::generate::build::{BASE_VERSION, Options as GenOptions, template};
use humanart_model::skin_appearance::channels::{Attribute, BaseValues};
use humanart_model::skin_appearance::dirty::Rect;
use humanart_model::skin_appearance::layers::{
    BlendMode, Layer, LayerContent, LayerKind, LayerStack, MAX_CONTENT_SIDE,
};
use humanart_model::skin_appearance::pigment::Mapping;
use humanart_model::skin_appearance::zones::ZoneTable;
use humanart_model::template::Template;
use humanart_render::camera::{Camera, Exposure, ReviewCamera};
use humanart_render::capture::{encode_png_rgba8, read_texture};
use humanart_render::error::RenderError;
use humanart_render::ibl::EnvironmentSource;
use humanart_render::light::{BuiltinRig, LightRig};
use humanart_render::renderer::{FrameInput, Quality, RenderConfig, Renderer};
use humanart_render::scene::{GpuScene, Material, Mesh, Scene};
use humanart_render::skin::{SkinCompositor, SkinDesc};
use humanart_render::sss::SkinShading;
use humanart_render::surface::SurfaceSettings;
use humanart_tools::capture::albedo::Albedo;
use humanart_tools::capture::align::fit_similarity;
use humanart_tools::capture::fitted::{FittedNode, FittedSurface};
use humanart_tools::capture::landmarks::LandmarkSet;
use humanart_tools::capture::register::{FitLandmark, FitMesh, Schedule, TargetSurface, Thresholds, register};
use humanart_tools::capture::scan::{Axis, Normalisation, Orientation, ScanMesh, Units};
use humanart_tools::capture::template_fit::{RegionWeights, TemplateFit};
use humanart_tools::capture::transfer::{
    FittedLayout, ScanSource, TransferSettings, state as transfer_state, transfer_and_clean,
};

use crate::args::{CommandSpec, OptSpec, Parsed};
use crate::error::CliError;

/// The views `--view` takes.
pub const VIEWS: [&str; 4] = ["body", "head", "three-quarter", "profile"];
const CAMERAS: [ReviewCamera; 4] =
    [ReviewCamera::FullBody, ReviewCamera::Front, ReviewCamera::ThreeQuarterLeft, ReviewCamera::ProfileLeft];
/// The tiers `--quality` takes.
pub const QUALITIES: [&str; 3] = ["interactive", "review", "still"];
const TIERS: [Quality; 3] = [Quality::Interactive, Quality::Review, Quality::Still];
/// The rigs `--rig` takes (the names of `render`).
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
/// Largest image side.
pub const MAX_SIDE: u32 = 8192;
/// Most frames accumulated for one still.
pub const MAX_FRAMES: u32 = 1024;
/// Most parts of an `--obj-list`.
pub const MAX_PARTS: usize = 2048;
/// Largest mesh or list file read, in bytes.
pub const MAX_FILE_BYTES: u64 = 2 << 30;
/// Largest albedo PNG read, in bytes.
pub const MAX_ALBEDO_BYTES: u64 = 1 << 30;
/// The fraction of the height that frames the head (as the engine viewer frames the template).
const HEAD_FRACTION: f32 = 0.13;

const OPTIONS: [OptSpec; 16] = [
    OptSpec { name: "obj", value: Some("FILE"), default: None, help: "Wavefront OBJ mesh to draw." },
    OptSpec {
        name: "frame-obj",
        value: Some("FILE"),
        default: None,
        help: "OBJ in the same file frame, normalised and framed with the parts but not drawn (a scan).",
    },
    OptSpec {
        name: "obj-list",
        value: Some("FILE"),
        default: None,
        help: "Text file of OBJ parts sharing one frame: 'path' or 'path #rrggbb' per line.",
    },
    OptSpec {
        name: "albedo",
        value: Some("PNG"),
        default: None,
        help: "Colour texture of --obj; drawn as surfaced skin with this albedo.",
    },
    OptSpec {
        name: "albedo-size",
        value: Some("PX"),
        default: Some("4096"),
        help: "Working size of the albedo (box-filtered), at most 4096.",
    },
    OptSpec { name: "units", value: Some("UNIT"), default: Some("auto"), help: "File units: auto, m, cm or mm." },
    OptSpec {
        name: "up",
        value: Some("AXIS"),
        default: Some("+y"),
        help: "The file's up axis: +x, -x, +y, -y, +z, -z.",
    },
    OptSpec {
        name: "forward",
        value: Some("AXIS"),
        default: Some("auto"),
        help: "The axis the subject faces in the file, or auto (from the feet).",
    },
    OptSpec {
        name: "view",
        value: Some("NAME"),
        default: Some("body"),
        help: "Framing: body, head, three-quarter or profile.",
    },
    OptSpec { name: "out", value: Some("FILE"), default: Some("render-mesh.png"), help: "PNG file to write." },
    OptSpec { name: "width", value: Some("PX"), default: Some("1080"), help: "Image width." },
    OptSpec { name: "height", value: Some("PX"), default: Some("1350"), help: "Image height." },
    OptSpec {
        name: "frames",
        value: Some("N"),
        default: Some("32"),
        help: "Frames rendered before the capture (temporal accumulation settles).",
    },
    OptSpec { name: "quality", value: Some("TIER"), default: Some("review"), help: "interactive, review or still." },
    OptSpec { name: "rig", value: Some("NAME"), default: Some("neutral-studio"), help: "Light rig, as for render." },
    OptSpec { name: "json-errors", value: None, default: None, help: "Errors as one JSON line on stderr." },
];

/// The `render-mesh` command.
pub static RENDER_MESH: CommandSpec = CommandSpec {
    name: "render-mesh",
    summary: "Renders an external mesh (scan, atlas) with the template's lights and cameras.",
    options: &OPTIONS,
};

/// One part of a mesh to draw: its OBJ file and an optional plain colour (linear).
pub type Part = (PathBuf, Option<[f32; 3]>);

/// What `render-mesh` was asked to do.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshJob {
    /// The parts: path and an optional plain colour (linear).
    pub parts: Vec<Part>,
    /// The albedo texture (single `--obj` only).
    pub albedo: Option<PathBuf>,
    /// Albedo working size.
    pub albedo_size: u32,
    /// File units (`None`: guessed).
    pub units: Option<Units>,
    /// The file's up axis.
    pub up: Axis,
    /// The file's forward axis (`None`: guessed).
    pub forward: Option<Axis>,
    /// The review camera.
    pub camera: ReviewCamera,
    /// The file to write.
    pub out: PathBuf,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Frames before the capture.
    pub frames: u32,
    /// The quality tier.
    pub quality: Quality,
    /// The light rig.
    pub rig: BuiltinRig,
    /// A mesh in the same file frame that is normalised and framed with the parts but not drawn
    /// (so a fitted head is framed exactly as the whole scan it was fitted to).
    pub frame: Option<PathBuf>,
}

fn usage(text: String) -> CliError {
    CliError::Usage(crate::args::UsageError(text))
}

impl MeshJob {
    /// The job of a parsed `render-mesh` command line (reads the `--obj-list` file).
    ///
    /// # Errors
    /// [`CliError::Usage`] for values out of range or missing; [`CliError::Io`] or
    /// [`CliError::Data`] for an unreadable or malformed list.
    pub fn from_args(p: &Parsed) -> Result<Self, CliError> {
        let pick = |name: &str, choices: &[&str]| p.choice(name, choices);
        let parts = match (p.get("obj"), p.get("obj-list")) {
            (Some(obj), None) => vec![(PathBuf::from(obj), None)],
            (None, Some(list)) => read_list(Path::new(list))?,
            _ => return Err(usage("render-mesh: give exactly one of --obj and --obj-list".into())),
        };
        let albedo = p.get("albedo").map(PathBuf::from);
        if albedo.is_some() && p.get("obj").is_none() {
            return Err(usage("--albedo needs --obj".into()));
        }
        let units = match p.get("units").unwrap_or("auto") {
            "auto" => None,
            name => Some(
                Units::from_name(name).ok_or_else(|| usage(format!("--units: '{name}' is not auto, m, cm or mm")))?,
            ),
        };
        let axis = |name: &str, text: &str| {
            Axis::from_name(text)
                .ok_or_else(|| usage(format!("--{name}: '{text}' is not one of {}", Axis::NAMES.join(", "))))
        };
        let up = axis("up", p.get("up").unwrap_or("+y"))?;
        let forward = match p.get("forward").unwrap_or("auto") {
            "auto" => None,
            text => Some(axis("forward", text)?),
        };
        Ok(Self {
            parts,
            albedo,
            albedo_size: p.number("albedo-size", 1, MAX_CONTENT_SIDE)?,
            units,
            up,
            forward,
            camera: CAMERAS[pick("view", &VIEWS)?],
            out: PathBuf::from(p.get("out").unwrap_or("render-mesh.png")),
            width: p.number("width", 1, MAX_SIDE)?,
            height: p.number("height", 1, MAX_SIDE)?,
            frames: p.number("frames", 1, MAX_FRAMES)?,
            quality: TIERS[pick("quality", &QUALITIES)?],
            rig: BUILTIN_RIGS[pick("rig", &RIGS)?],
            frame: p.get("frame-obj").map(PathBuf::from),
        })
    }
}

/// Reads a file of at most `limit` bytes.
fn read_file(path: &Path, limit: u64) -> Result<Vec<u8>, CliError> {
    let size = std::fs::metadata(path).map_err(|e| CliError::io(path, e))?.len();
    if size > limit {
        return Err(CliError::Data(format!("{}: {size} bytes, the limit is {limit}", path.display())));
    }
    std::fs::read(path).map_err(|e| CliError::io(path, e))
}

/// The parts of an `--obj-list` file: one `path` or `path #rrggbb` per non-empty line; relative
/// paths are relative to the list's folder.
fn read_list(list: &Path) -> Result<Vec<Part>, CliError> {
    let bytes = read_file(list, 1 << 20)?;
    let text = std::str::from_utf8(&bytes).map_err(|_| CliError::Data(format!("{}: not UTF-8", list.display())))?;
    parse_list(text, list.parent().unwrap_or(Path::new("")))
}

fn parse_list(text: &str, base: &Path) -> Result<Vec<Part>, CliError> {
    let mut parts = Vec::new();
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with("//")) {
        if parts.len() >= MAX_PARTS {
            return Err(CliError::Data(format!("the part list has more than {MAX_PARTS} lines")));
        }
        let (path, color) = match line.rsplit_once(" #") {
            Some((path, hex)) => (path.trim(), Some(parse_hex(hex)?)),
            None => (line, None),
        };
        parts.push((base.join(path), color));
    }
    if parts.is_empty() {
        return Err(CliError::Data("the part list is empty".into()));
    }
    Ok(parts)
}

/// `rrggbb` (sRGB) as linear RGB.
fn parse_hex(hex: &str) -> Result<[f32; 3], CliError> {
    let bad = || CliError::Data(format!("'#{hex}' is not a #rrggbb colour"));
    if hex.len() != 6 || !hex.is_ascii() {
        return Err(bad());
    }
    let mut out = [0.0; 3];
    for (k, o) in out.iter_mut().enumerate() {
        let byte = u8::from_str_radix(&hex[k * 2..k * 2 + 2], 16).map_err(|_| bad())?;
        *o = srgb_to_linear_f32(f32::from(byte) / 255.0);
    }
    Ok(out)
}

/// What a finished render reports.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshSummary {
    /// The file written.
    pub out: PathBuf,
    /// Vertices and triangles drawn.
    pub vertices: usize,
    /// Triangles drawn.
    pub triangles: usize,
    /// The units used.
    pub units: Units,
    /// The forward axis used.
    pub forward: Axis,
    /// Height of the normalised mesh, metres.
    pub height: f32,
    /// Albedo working size (0 without an albedo).
    pub albedo_size: u32,
    /// The GPU adapter.
    pub adapter: String,
}

/// The parts of a job loaded as one normalised scan.
struct Loaded {
    scan: ScanMesh,
    /// Each part's triangle range in `scan`.
    ranges: Vec<std::ops::Range<usize>>,
    units: Units,
    forward: Axis,
    /// What the normalisation applied (file units to ours).
    normalisation: Normalisation,
}

/// Loads, normalises and returns the parts as one scan with each part's triangle range.
fn load(job: &MeshJob) -> Result<Loaded, CliError> {
    let mut scan = ScanMesh::default();
    let mut ranges = Vec::with_capacity(job.parts.len());
    for (path, _) in &job.parts {
        let bytes = read_file(path, MAX_FILE_BYTES)?;
        let part = ScanMesh::parse_obj(&bytes, &Limits::DEFAULT)
            .map_err(|e| CliError::Data(format!("{}: {e}", path.display())))?;
        let first = scan.triangle_count();
        scan.append(&part).map_err(|e| CliError::Data(e.to_string()))?;
        ranges.push(first..scan.triangle_count());
    }
    if let Some(path) = &job.frame {
        let bytes = read_file(path, MAX_FILE_BYTES)?;
        let frame = ScanMesh::parse_obj(&bytes, &Limits::DEFAULT)
            .map_err(|e| CliError::Data(format!("{}: {e}", path.display())))?;
        scan.append(&frame).map_err(|e| CliError::Data(e.to_string()))?;
    }
    let units = if let Some(u) = job.units {
        u
    } else {
        let (lo, hi) = scan.extent();
        Units::guess((hi - lo).max_element())
            .ok_or_else(|| CliError::Data("cannot guess the units; give --units".into()))?
    };
    let forward = match job.forward {
        Some(f) => f,
        None => Orientation::guess_forward(&scan.positions, job.up)
            .ok_or_else(|| CliError::Data("cannot guess the forward axis; give --forward".into()))?,
    };
    let normalisation = scan
        .normalise(units, Orientation { up: job.up, forward })
        .map_err(|e| CliError::Data(format!("{e} (--up {} --forward {})", axis_name(job.up), axis_name(forward))))?;
    Ok(Loaded { scan, ranges, units, forward, normalisation })
}

fn axis_name(a: Axis) -> &'static str {
    Axis::ALL.iter().position(|x| *x == a).and_then(|i| Axis::NAMES.get(i)).copied().unwrap_or("?")
}

/// The render mesh of the triangles `range` of `scan` (vertices compacted), u clamped into tile 0.
fn render_mesh(scan: &ScanMesh, range: std::ops::Range<usize>) -> Result<Mesh, CliError> {
    let mut map = vec![u32::MAX; scan.positions.len()];
    let mut mesh = Mesh::default();
    for t in range {
        let tri = scan.triangle(t).ok_or_else(|| CliError::Data("triangle out of range".into()))?;
        for v in tri {
            let slot = map.get_mut(v as usize).ok_or_else(|| CliError::Data("index out of range".into()))?;
            if *slot == u32::MAX {
                *slot = mesh.positions.len() as u32;
                let i = v as usize;
                mesh.positions.push(scan.positions.get(i).copied().unwrap_or(Vec3::ZERO));
                mesh.normals.push(scan.normals.get(i).copied().unwrap_or(Vec3::Y));
                let uv = scan.uvs.get(i).copied().unwrap_or(Vec2::ZERO);
                mesh.uvs.push(Vec2::new(uv.x.clamp(0.0, 0.999_99), uv.y.clamp(0.0, 1.0)));
            }
            mesh.indices.push(*slot);
        }
    }
    Ok(mesh)
}

/// The extent of the positions.
fn extent(points: &[Vec3]) -> (Vec3, Vec3) {
    points.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), &p| (lo.min(p), hi.max(p)))
}

/// The centre and radius of the head: the bounding sphere of the top [`HEAD_FRACTION`] of the
/// height, as `humanart_agent::engine::Model::head_bounds` computes it for the template.
fn head_bounds(points: &[Vec3]) -> (Vec3, f32) {
    let (lo, hi) = extent(points);
    let cut = hi.y - (hi.y - lo.y) * HEAD_FRACTION;
    let (l, h) = extent(&points.iter().copied().filter(|p| p.y >= cut).collect::<Vec<_>>());
    if l.x > h.x {
        return (Vec3::ZERO, 1.0);
    }
    ((l + h) * 0.5, ((h - l) * 0.5).length().max(1e-3))
}

/// The camera of `kind` for `points`, as `humanart_agent::engine::view::frame_camera` frames the
/// template.
fn frame(points: &[Vec3], kind: ReviewCamera, aspect: f32) -> Camera {
    if kind == ReviewCamera::FullBody {
        let (lo, hi) = extent(points);
        let (half_w, half_h) = (0.5 * (hi.x - lo.x), 0.5 * (hi.y - lo.y));
        let a = aspect.max(1e-3);
        let radius = if a < 1.0 { (half_h * a).max(half_w) } else { half_h.max(half_w / a) };
        return Camera::review(ReviewCamera::FullBody, (lo + hi) * 0.5, radius, aspect);
    }
    let (center, radius) = head_bounds(points);
    Camera::review(kind, center, radius, aspect)
}

/// A one-tile skin compositor whose resolved albedo is `albedo` (a full-coverage colour override
/// layer over the default base values), resolved once on `gpu`'s queue.
fn albedo_compositor(gpu: &Gpu, albedo: &Albedo, tile: u32) -> Result<SkinCompositor, CliError> {
    let skin = |e| CliError::Render(EngineError::Render(RenderError::Skin(e)));
    let model = |e| CliError::Data(format!("skin layers: {e}"));
    let mut compositor =
        SkinCompositor::new(gpu, SkinDesc { tiles: 1, tile_size: tile, atlas_size: tile }).map_err(skin)?;
    compositor.set_globals(gpu, &BaseValues::default(), &ZoneTable::default(), &Mapping::DEFAULT);
    let texels = (tile as usize) * (tile as usize);
    let baked: Vec<u8> = [255, 255, 0, 0].repeat(texels);
    let zones = vec![0_u8; texels * 4];
    compositor.upload_inputs(gpu, 0, &baked, [&zones, &zones, &zones]).map_err(skin)?;
    let mut content = LayerContent::new(0, Rect::new(0, 0, tile, tile)).map_err(model)?;
    let inv = 1.0 / tile as f32;
    for y in 0..tile {
        for x in 0..tile {
            let i = (y * tile + x) as usize;
            let c = albedo.sample(Vec2::new((x as f32 + 0.5) * inv, (y as f32 + 0.5) * inv));
            if let (Some(v), Some(k)) = (content.values.get_mut(i), content.coverage.get_mut(i)) {
                *v = [c[0].clamp(0.0, 1.0), c[1].clamp(0.0, 1.0), c[2].clamp(0.0, 1.0), 1.0];
                *k = 1.0;
            }
        }
    }
    let mut stack = LayerStack::new(1, tile).map_err(model)?;
    let id = stack.allocate_id();
    let mut layer = Layer::content(
        id,
        "scan albedo",
        BlendMode::Mix,
        [
            Some(Attribute::OverrideRed),
            Some(Attribute::OverrideGreen),
            Some(Attribute::OverrideBlue),
            Some(Attribute::OverrideCoverage),
        ],
    );
    if let LayerKind::Content { tiles, .. } = &mut layer.kind {
        tiles.push(content);
    }
    stack.push(layer).map_err(model)?;
    stack.dirty_mut().mark_all();
    compositor.sync_layers(gpu, &stack).map_err(skin)?;
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("scan albedo") });
    compositor.resolve(gpu, &mut encoder, &mut stack, None).map_err(skin)?;
    gpu.queue.submit([encoder.finish()]);
    Ok(compositor)
}

/// Runs `job`: loads and normalises the mesh, renders `job.frames` frames on a headless GPU and
/// writes the display image as PNG.
///
/// # Errors
/// [`CliError::Io`] for the files, [`CliError::Data`] for malformed meshes or textures,
/// [`CliError::Gpu`] without an adapter, [`CliError::Render`] when the renderer fails.
pub fn run(job: &MeshJob) -> Result<MeshSummary, CliError> {
    let Loaded { scan, ranges, units, forward, .. } = load(job)?;
    let albedo = match &job.albedo {
        Some(path) => {
            let bytes = read_file(path, MAX_ALBEDO_BYTES)?;
            let image = humanart_io::png::decode(&bytes, &Limits::DEFAULT)
                .map_err(|e| CliError::Data(format!("{}: {e}", path.display())))?;
            drop(bytes);
            Some(Albedo::from_image(&image, job.albedo_size).map_err(|e| CliError::Data(e.to_string()))?)
        }
        None => None,
    };
    let gpu = Gpu::headless_blocking(&DeviceRequest::default()).map_err(|e| CliError::Gpu(e.to_string()))?;
    let mut resources = Resources::new(&gpu, Budget::UNLIMITED);
    let config = RenderConfig { overlays: false, quality: job.quality, ..RenderConfig::new(job.width, job.height) };
    let mut renderer = Renderer::new(&gpu, &mut resources, config).map_err(EngineError::from)?;
    let mut scene = Scene::default();
    let mut compositor = None;
    for ((_, color), range) in job.parts.iter().zip(ranges) {
        let mesh = render_mesh(&scan, range)?;
        if mesh.indices.is_empty() {
            continue;
        }
        if let Some(albedo) = &albedo {
            let tile = job.albedo_size;
            let built = albedo_compositor(&gpu, albedo, tile)?;
            let mut settings = SurfaceSettings::default();
            settings.mm_per_uv[0] = scan.mm_per_uv().unwrap_or(0.0);
            let surface = renderer.skin_surface_mut();
            surface.set_settings(&gpu, settings).map_err(EngineError::from)?;
            surface.bind_compositor(&gpu, &built).map_err(EngineError::from)?;
            compositor = Some(built);
            let object = scene.add(mesh, SURFACED_SKIN_MATERIAL);
            if let Some(o) = scene.objects.get_mut(object) {
                o.skin = SkinShading { surface: true, ..SkinShading::default() };
            }
        } else {
            let material = color.map_or(SKIN_MATERIAL, |c| Material { base_color: c, ..SKIN_MATERIAL });
            scene.add(mesh, material);
        }
    }
    let mut gpu_scene = GpuScene::new(&mut resources, &scene).map_err(EngineError::from)?;
    let aspect = job.width as f32 / job.height.max(1) as f32;
    let (center, radius) = head_bounds(&scan.positions);
    let rig = LightRig::builtin(job.rig, center, radius);
    renderer
        .set_environment(&mut resources, EnvironmentSource::Procedural(rig.environment))
        .map_err(EngineError::from)?;
    let camera = Camera { exposure: Exposure::ev100(rig.ev100), ..frame(&scan.positions, job.camera, aspect) };
    let result = render_frames(&gpu, &mut resources, &mut renderer, &mut gpu_scene, &camera, &rig, job);
    gpu_scene.destroy(&mut resources);
    renderer.destroy(&mut resources);
    drop(compositor);
    result?;
    let (lo, hi) = scan.extent();
    Ok(MeshSummary {
        out: job.out.clone(),
        vertices: scan.positions.len(),
        triangles: scan.triangle_count(),
        units,
        forward,
        height: hi.y - lo.y,
        albedo_size: albedo.as_ref().map_or(0, |_| job.albedo_size),
        adapter: gpu.adapter_info().name.clone(),
    })
}

fn render_frames(
    gpu: &Gpu,
    resources: &mut Resources,
    renderer: &mut Renderer,
    scene: &mut GpuScene,
    camera: &Camera,
    rig: &LightRig,
    job: &MeshJob,
) -> Result<(), CliError> {
    for frame in 0..job.frames {
        let mut encoder =
            gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cli render-mesh") });
        let input = FrameInput {
            camera,
            rig,
            scene: &mut *scene,
            overlay: &[],
            wireframe: false,
            still: None,
            pick: None,
            reset_history: frame == 0,
        };
        renderer.render(resources, &mut encoder, input).map_err(EngineError::from)?;
        gpu.queue.submit([encoder.finish()]);
        renderer.after_submit();
    }
    let texture =
        renderer.display_texture().ok_or(EngineError::Invalid { what: "the renderer has no display image" })?;
    let (w, h, rgba) = read_texture(resources, texture, 0).map_err(EngineError::from)?;
    let png = encode_png_rgba8(w, h, &rgba).map_err(EngineError::from)?;
    if let Some(dir) = job.out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| CliError::io(dir, e))?;
    }
    std::fs::write(&job.out, png).map_err(|e| CliError::io(&job.out, e))
}

/// Runs the `render-mesh` command line `args` (after the command name) and prints the summary.
///
/// # Errors
/// As [`MeshJob::from_args`] and [`run`].
pub fn command(program: &str, args: &[String]) -> Result<(), CliError> {
    let parsed = Parsed::parse(&RENDER_MESH, args)?;
    if parsed.help() {
        print!("{}", crate::args::help(program, &RENDER_MESH));
        return Ok(());
    }
    let job = MeshJob::from_args(&parsed)?;
    let s = run(&job)?;
    println!(
        "wrote {} ({} vertices, {} triangles, units {}, forward {}, height {:.4} m, albedo {}, adapter: {})",
        s.out.display(),
        s.vertices,
        s.triangles,
        Units::ALL.iter().position(|u| *u == s.units).and_then(|i| Units::NAMES.get(i)).copied().unwrap_or("?"),
        axis_name(s.forward),
        s.height,
        if s.albedo_size == 0 { "none".to_owned() } else { format!("{0}x{0}", s.albedo_size) },
        s.adapter
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// pick-landmarks: scan landmarks from pixel positions on the review renders.
// ---------------------------------------------------------------------------------------------

/// Most picks in one file.
pub const MAX_PICKS: usize = 4096;

const PICK_OPTIONS: [OptSpec; 10] = [
    OptSpec { name: "obj", value: Some("FILE"), default: None, help: "The scan OBJ the renders were made of." },
    OptSpec {
        name: "picks",
        value: Some("FILE"),
        default: None,
        help: "Text file: one 'view name x y' per line (pixel centre coordinates in the render).",
    },
    OptSpec { name: "units", value: Some("UNIT"), default: Some("auto"), help: "File units: auto, m, cm or mm." },
    OptSpec { name: "up", value: Some("AXIS"), default: Some("+y"), help: "The file's up axis." },
    OptSpec { name: "forward", value: Some("AXIS"), default: Some("auto"), help: "The axis the subject faces." },
    OptSpec { name: "width", value: Some("PX"), default: Some("1080"), help: "Width of the renders picked on." },
    OptSpec { name: "height", value: Some("PX"), default: Some("1350"), help: "Height of the renders picked on." },
    OptSpec { name: "scan-id", value: Some("TEXT"), default: Some("scan"), help: "The scan's name in the file." },
    OptSpec {
        name: "out",
        value: Some("FILE"),
        default: Some("landmarks.json"),
        help: "Landmark annotation (humanart-scan-landmarks JSON) to write.",
    },
    OptSpec { name: "json-errors", value: None, default: None, help: "Errors as one JSON line on stderr." },
];

/// The `pick-landmarks` command.
pub static PICK_LANDMARKS: CommandSpec = CommandSpec {
    name: "pick-landmarks",
    summary: "Turns pixel picks on render-mesh stills into 3D scan landmarks (ray cast).",
    options: &PICK_OPTIONS,
};

/// One pick: the view, the landmark name and the pixel.
#[derive(Clone, Debug, PartialEq)]
pub struct Pick {
    /// The review camera of the render.
    pub camera: ReviewCamera,
    /// The template landmark name.
    pub name: String,
    /// Pixel x (0 = left edge, 0.5 = centre of the first column).
    pub x: f32,
    /// Pixel y (0 = top edge).
    pub y: f32,
}

/// Parses a picks file: `view name x y` per line, `#` comments, views as for `--view`.
///
/// # Errors
/// [`CliError::Data`] naming the line.
pub fn parse_picks(text: &str) -> Result<Vec<Pick>, CliError> {
    let mut picks = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if picks.len() >= MAX_PICKS {
            return Err(CliError::Data(format!("more than {MAX_PICKS} picks")));
        }
        let bad = || CliError::Data(format!("picks line {}: expected 'view name x y'", n + 1));
        let f: Vec<&str> = line.split_whitespace().collect();
        let [view, name, x, y] = f.as_slice() else { return Err(bad()) };
        let v = VIEWS.iter().position(|v| v == view).ok_or_else(bad)?;
        let (x, y): (f32, f32) = (x.parse().map_err(|_| bad())?, y.parse().map_err(|_| bad())?);
        if !(x.is_finite() && y.is_finite()) {
            return Err(bad());
        }
        picks.push(Pick { camera: CAMERAS[v], name: (*name).to_owned(), x, y });
    }
    Ok(picks)
}

/// The world ray through pixel `(x, y)` of a `width × height` image seen by `camera` (the
/// renderer's pinhole: `look_at_rh` and a symmetric vertical field of view).
fn pixel_ray(camera: &Camera, x: f32, y: f32, width: u32, height: u32) -> (Vec3, Vec3) {
    let f = camera.forward();
    let r = f.cross(camera.up).normalize_or(Vec3::X);
    let u = r.cross(f);
    let aspect = width as f32 / height.max(1) as f32;
    let tan = (camera.fov_y() * 0.5).tan();
    let ndc_x = 2.0 * x / width as f32 - 1.0;
    let ndc_y = 1.0 - 2.0 * y / height as f32;
    let dir = (f + r * (ndc_x * tan * aspect) + u * (ndc_y * tan)).normalize_or(f);
    (camera.position, dir)
}

/// Runs the `pick-landmarks` command line `args`.
///
/// # Errors
/// [`CliError::Usage`], [`CliError::Io`], [`CliError::Data`] (a pick that misses the scan).
pub fn pick_command(program: &str, args: &[String]) -> Result<(), CliError> {
    let p = Parsed::parse(&PICK_LANDMARKS, args)?;
    if p.help() {
        print!("{}", crate::args::help(program, &PICK_LANDMARKS));
        return Ok(());
    }
    let obj = p.get("obj").ok_or_else(|| usage("--obj is required".into()))?;
    let picks_path = p.get("picks").ok_or_else(|| usage("--picks is required".into()))?;
    let (width, height) = (p.number("width", 1, MAX_SIDE)?, p.number("height", 1, MAX_SIDE)?);
    let (units, up, forward) = frame_options(&p)?;
    let job = mesh_job(PathBuf::from(obj), units, up, forward);
    let picks_bytes = read_file(Path::new(picks_path), 1 << 20)?;
    let text = std::str::from_utf8(&picks_bytes).map_err(|_| CliError::Data("the picks file is not UTF-8".into()))?;
    let picks = parse_picks(text)?;
    let Loaded { scan, .. } = load(&job)?;
    let bvh = humanart_tools::bake::bvh::Bvh::build(&scan.positions, &scan.indices)
        .map_err(|e| CliError::Data(e.to_string()))?;
    let aspect = width as f32 / height as f32;
    let mut set = LandmarkSet { scan: p.get("scan-id").unwrap_or("scan").to_owned(), landmarks: Vec::new() };
    for pick in &picks {
        let camera = frame(&scan.positions, pick.camera, aspect);
        let (o, d) = pixel_ray(&camera, pick.x, pick.y, width, height);
        let hit = bvh.intersect(o, d, 0.0, 100.0).ok_or_else(|| {
            CliError::Data(format!("pick '{}' at ({}, {}) misses the scan", pick.name, pick.x, pick.y))
        })?;
        let point = o + d * hit.t;
        set.push(&pick.name, point).map_err(|e| CliError::Data(e.to_string()))?;
        println!("{:<28} ({:8.4}, {:8.4}, {:8.4})", pick.name, point.x, point.y, point.z);
    }
    let out = PathBuf::from(p.get("out").unwrap_or("landmarks.json"));
    write_bytes(&out, set.to_json().as_bytes())?;
    println!("wrote {} ({} landmarks)", out.display(), set.landmarks.len());
    Ok(())
}

/// `--units`, `--up` and `--forward` of a parsed command line.
fn frame_options(p: &Parsed) -> Result<(Option<Units>, Axis, Option<Axis>), CliError> {
    let units = match p.get("units").unwrap_or("auto") {
        "auto" => None,
        name => {
            Some(Units::from_name(name).ok_or_else(|| usage(format!("--units: '{name}' is not auto, m, cm or mm")))?)
        }
    };
    let axis = |name: &str, text: &str| {
        Axis::from_name(text)
            .ok_or_else(|| usage(format!("--{name}: '{text}' is not one of {}", Axis::NAMES.join(", "))))
    };
    let up = axis("up", p.get("up").unwrap_or("+y"))?;
    let forward = match p.get("forward").unwrap_or("auto") {
        "auto" => None,
        text => Some(axis("forward", text)?),
    };
    Ok((units, up, forward))
}

/// A render-less mesh job: only the loading fields matter.
fn mesh_job(obj: PathBuf, units: Option<Units>, up: Axis, forward: Option<Axis>) -> MeshJob {
    MeshJob {
        parts: vec![(obj, None)],
        albedo: None,
        albedo_size: 1,
        units,
        up,
        forward,
        camera: ReviewCamera::Front,
        out: PathBuf::new(),
        width: 1,
        height: 1,
        frames: 1,
        quality: Quality::Review,
        rig: BuiltinRig::NeutralStudio,
        frame: None,
    }
}

fn write_bytes(path: &Path, bytes: &[u8]) -> Result<(), CliError> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| CliError::io(dir, e))?;
    }
    std::fs::write(path, bytes).map_err(|e| CliError::io(path, e))
}

// ---------------------------------------------------------------------------------------------
// fit-scan: the template's head fitted onto a scan, the scan albedo transferred.
// ---------------------------------------------------------------------------------------------

const FIT_OPTIONS: [OptSpec; 21] = [
    OptSpec { name: "obj", value: Some("FILE"), default: None, help: "The scan OBJ." },
    OptSpec { name: "albedo", value: Some("PNG"), default: None, help: "The scan's colour texture (optional)." },
    OptSpec {
        name: "landmarks",
        value: Some("FILE"),
        default: None,
        help: "Scan landmarks (humanart-scan-landmarks JSON, template names, normalised frame).",
    },
    OptSpec { name: "units", value: Some("UNIT"), default: Some("auto"), help: "File units: auto, m, cm or mm." },
    OptSpec { name: "up", value: Some("AXIS"), default: Some("+y"), help: "The file's up axis." },
    OptSpec { name: "forward", value: Some("AXIS"), default: Some("auto"), help: "The axis the subject faces." },
    OptSpec {
        name: "crop",
        value: Some("CM"),
        default: Some("45"),
        help: "Scan kept: this many centimetres below its top (head and neck).",
    },
    OptSpec {
        name: "regions",
        value: Some("LIST"),
        default: Some("head,neck"),
        help: "Template regions fitted (comma-separated, each with its subtree).",
    },
    OptSpec {
        name: "albedo-size",
        value: Some("PX"),
        default: Some("4096"),
        help: "Working size of the scan albedo (box-filtered), at most 16384.",
    },
    OptSpec { name: "tile-size", value: Some("PX"), default: Some("2048"), help: "Side of each transferred tile." },
    OptSpec { name: "rays", value: Some("N"), default: Some("48"), help: "Occlusion rays per scan vertex." },
    OptSpec {
        name: "levels",
        value: Some("CM,.."),
        default: Some("8,4,2,1"),
        help: "Smoothing length of each registration level, centimetres, coarse to fine.",
    },
    OptSpec {
        name: "landmark-weight",
        value: Some("W"),
        default: Some("20"),
        help: "Weight of each landmark, in fully weighted nodes.",
    },
    OptSpec {
        name: "mask-regions",
        value: Some("LIST"),
        default: Some(""),
        help: "Template regions the scan does not show (hair over the ears, a collar): data weight 0.",
    },
    OptSpec {
        name: "landmark-decay",
        value: Some("F"),
        default: Some("0.5"),
        help: "Factor on the landmark weight at each level after the first.",
    },
    OptSpec {
        name: "fold-guard",
        value: Some("N,F,R,K"),
        default: Some("6,2,1,0.05"),
        help: "Fold guard: passes per level (0: off), stiffening factor, rings, extra membrane.",
    },
    OptSpec {
        name: "visibility",
        value: Some("LO,HI"),
        default: Some("0.25,0.6"),
        help: "Node weights scaled by template visibility from LO (0) to HI (1), or 'off'.",
    },
    OptSpec { name: "scan-id", value: Some("TEXT"), default: Some("scan"), help: "The scan's id in the fit file." },
    OptSpec {
        name: "attribution",
        value: Some("TEXT"),
        default: Some(""),
        help: "Attribution the scan's licence asks for (stored in the fit file).",
    },
    OptSpec { name: "out-dir", value: Some("DIR"), default: Some("fit"), help: "Folder for the results." },
    OptSpec { name: "json-errors", value: None, default: None, help: "Errors as one JSON line on stderr." },
];

/// The `fit-scan` command.
pub static FIT_SCAN: CommandSpec = CommandSpec {
    name: "fit-scan",
    summary: "Fits the template's head onto a scan (landmarks + non-rigid ICP) and transfers its albedo.",
    options: &FIT_OPTIONS,
};

/// Per-region error numbers of a fit.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RegionError {
    /// Region name.
    pub name: String,
    /// Weighted nodes in it.
    pub nodes: usize,
    /// Of those, matched in the last iteration.
    pub matched: usize,
    /// RMS distance to the scan of its weighted nodes, metres.
    pub rms: f32,
    /// Largest distance, metres.
    pub max: f32,
}

/// The regions whose errors the report lists: each node counts in the first listed ancestor.
const REPORT_REGIONS: [&str; 18] = [
    "nose", "eye-L", "eye-R", "brow-L", "brow-R", "mouth", "chin", "cheek-L", "cheek-R", "jaw-L", "jaw-R", "forehead",
    "ear-L", "ear-R", "face", "scalp", "hairline", "neck",
];

fn region_errors(template: &Template, fit: &TemplateFit, errors: &[f32], matched: &[bool]) -> Vec<RegionError> {
    let regions = template.regions();
    let ids: Vec<Option<u16>> = REPORT_REGIONS.iter().map(|n| regions.find(n)).collect();
    let mut acc: Vec<(usize, usize, f64, f32)> = vec![(0, 0, 0.0, 0.0); REPORT_REGIONS.len() + 1];
    for (n, &v) in fit.node_vertex.iter().enumerate() {
        if fit.node_weights.get(n).copied().unwrap_or(0.0) <= 0.0 {
            continue;
        }
        let r = regions.vertex_regions.get(v as usize).copied().unwrap_or(0);
        let slot = ids.iter().position(|a| a.is_some_and(|a| regions.is_within(r, a))).unwrap_or(REPORT_REGIONS.len());
        let e = errors.get(n).copied().unwrap_or(f32::INFINITY);
        if let Some(a) = acc.get_mut(slot) {
            a.0 += 1;
            if matched.get(n).copied().unwrap_or(false) {
                a.1 += 1;
            }
            if e.is_finite() {
                a.2 += f64::from(e * e);
                a.3 = a.3.max(e);
            }
        }
    }
    acc.iter()
        .enumerate()
        .filter(|(_, a)| a.0 > 0)
        .map(|(i, a)| RegionError {
            name: REPORT_REGIONS.get(i).copied().unwrap_or("other").to_owned(),
            nodes: a.0,
            matched: a.1,
            rms: (a.2 / a.0 as f64).sqrt() as f32,
            max: a.3,
        })
        .collect()
}

/// Folded triangles of one report region: `(total, folded)` where every node weight is at least
/// 0.5 (pulled) and elsewhere.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RegionFolds {
    /// Region name.
    pub name: String,
    /// Pulled triangles: total, folded.
    pub pulled: (usize, usize),
    /// The other triangles: total, folded.
    pub others: (usize, usize),
}

/// Orientation flips of the fit triangles from `start` to `fitted`, per report region (a
/// triangle counts in the region of its first node).
fn region_folds(template: &Template, fit: &TemplateFit, start: &[Vec3], fitted: &[Vec3]) -> Vec<RegionFolds> {
    let regions = template.regions();
    let ids: Vec<Option<u16>> = REPORT_REGIONS.iter().map(|n| regions.find(n)).collect();
    let mut acc = vec![RegionFolds::default(); REPORT_REGIONS.len() + 1];
    let get = |x: &[Vec3], i: u32| x.get(i as usize).copied().unwrap_or(Vec3::ZERO);
    for t in fit.mesh.triangles.as_chunks::<3>().0 {
        let normal = |x: &[Vec3]| (get(x, t[1]) - get(x, t[0])).cross(get(x, t[2]) - get(x, t[0]));
        let v = fit.node_vertex.get(t[0] as usize).copied().unwrap_or(0);
        let r = regions.vertex_regions.get(v as usize).copied().unwrap_or(0);
        let slot = ids.iter().position(|a| a.is_some_and(|a| regions.is_within(r, a))).unwrap_or(REPORT_REGIONS.len());
        let pulled = t.iter().all(|&n| fit.node_weights.get(n as usize).copied().unwrap_or(0.0) >= 0.5);
        let folded = usize::from(normal(start).dot(normal(fitted)) < 0.0);
        if let Some(a) = acc.get_mut(slot) {
            let bin = if pulled { &mut a.pulled } else { &mut a.others };
            bin.0 += 1;
            bin.1 += folded;
        }
    }
    acc.into_iter()
        .enumerate()
        .filter(|(_, a)| a.pulled.0 + a.others.0 > 0)
        .map(|(i, mut a)| {
            REPORT_REGIONS.get(i).copied().unwrap_or("other").clone_into(&mut a.name);
            a
        })
        .collect()
}

/// Writes an OBJ of `nodes` (positions mapped by `to_file`) with one texture coordinate per
/// triangle corner.
fn write_obj(
    path: &Path,
    nodes: &[Vec3],
    triangles: &[u32],
    corner_uvs: &[Vec2],
    to_file: impl Fn(Vec3) -> Vec3,
) -> Result<(), CliError> {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(nodes.len() * 40 + triangles.len() * 30);
    s.push_str("# humanart-cli fit-scan\n");
    for &p in nodes {
        let q = to_file(p);
        let _ = writeln!(s, "v {:.6} {:.6} {:.6}", q.x, q.y, q.z);
    }
    for uv in corner_uvs {
        // OBJ texture coordinates have V up; ours are V down.
        let _ = writeln!(s, "vt {:.6} {:.6}", uv.x, 1.0 - uv.y);
    }
    for (t, tri) in triangles.as_chunks::<3>().0.iter().enumerate() {
        let k = 3 * t + 1;
        if corner_uvs.is_empty() {
            let _ = writeln!(s, "f {} {} {}", tri[0] + 1, tri[1] + 1, tri[2] + 1);
        } else {
            let _ = writeln!(s, "f {}/{} {}/{} {}/{}", tri[0] + 1, k, tri[1] + 1, k + 1, tri[2] + 1, k + 2);
        }
    }
    write_bytes(path, s.as_bytes())
}

/// An sRGB 8-bit PNG of linear RGB rows.
fn write_rgb_png(path: &Path, width: u32, height: u32, texels: &[[f32; 3]]) -> Result<(), CliError> {
    let mut rgba = Vec::with_capacity(texels.len() * 4);
    for c in texels {
        let s = linear_to_srgb_rgb_f32(*c);
        rgba.extend(s.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8));
        rgba.push(255);
    }
    let png = encode_png_rgba8(width, height, &rgba).map_err(EngineError::from)?;
    write_bytes(path, &png)
}

/// The colour of a fit distance (metres) in the error maps: blue 0, green 1 mm, yellow 2 mm,
/// red 4 mm and more; magenta where no scan point was found (linear RGB).
#[must_use]
pub fn error_colour(d: f32) -> [f32; 3] {
    if d < 0.0 || !d.is_finite() {
        return [0.8, 0.0, 0.8];
    }
    let mm = d * 1000.0;
    let stops: [(f32, [f32; 3]); 4] =
        [(0.0, [0.0, 0.05, 0.6]), (1.0, [0.0, 0.5, 0.05]), (2.0, [0.7, 0.6, 0.0]), (4.0, [0.7, 0.02, 0.0])];
    for w in stops.windows(2) {
        let ((a, ca), (b, cb)) = (w[0], w[1]);
        if mm <= b {
            let t = ((mm - a) / (b - a)).clamp(0.0, 1.0);
            return [0, 1, 2].map(|k| ca[k] + (cb[k] - ca[k]) * t);
        }
    }
    stops[3].1
}

/// Runs the `fit-scan` command line `args`.
///
/// # Errors
/// [`CliError::Usage`], [`CliError::Io`], [`CliError::Data`] for malformed input or a fit that
/// cannot be computed.
#[allow(clippy::too_many_lines)]
pub fn fit_command(program: &str, args: &[String]) -> Result<(), CliError> {
    let p = Parsed::parse(&FIT_SCAN, args)?;
    if p.help() {
        print!("{}", crate::args::help(program, &FIT_SCAN));
        return Ok(());
    }
    let data = |e: humanart_tools::capture::CaptureError| CliError::Data(e.to_string());
    let obj = PathBuf::from(p.get("obj").ok_or_else(|| usage("--obj is required".into()))?);
    let landmarks_path = p.get("landmarks").ok_or_else(|| usage("--landmarks is required".into()))?;
    let (units, up, forward) = frame_options(&p)?;
    let crop = p.number("crop", 5, 250)? as f32 * 0.01;
    let albedo_size = p.number("albedo-size", 1, humanart_tools::capture::albedo::MAX_ALBEDO_SIDE)?;
    let tile_size = p.number("tile-size", 16, 8192)?;
    let rays = p.number("rays", 1, 1024)?;
    let out_dir = PathBuf::from(p.get("out-dir").unwrap_or("fit"));
    let regions_text = p.get("regions").unwrap_or("head,neck").to_owned();
    let numbers = |name: &str, text: &str| -> Result<Vec<f32>, CliError> {
        text.split(',')
            .map(|t| t.trim().parse::<f32>().ok().filter(|v| v.is_finite() && *v > 0.0))
            .collect::<Option<Vec<f32>>>()
            .filter(|v| !v.is_empty() && v.len() <= 16)
            .ok_or_else(|| usage(format!("--{name}: '{text}' is not a list of positive numbers")))
    };
    let levels = numbers("levels", p.get("levels").unwrap_or("8,4,2,1"))?;
    let landmark_weight =
        *numbers("landmark-weight", p.get("landmark-weight").unwrap_or("20"))?.first().unwrap_or(&20.0);
    let landmark_decay = *numbers("landmark-decay", p.get("landmark-decay").unwrap_or("0.5"))?.first().unwrap_or(&0.5);
    let guard_text = p.get("fold-guard").unwrap_or("6,2,1,0.05");
    let guard: Vec<f32> = guard_text.split(',').filter_map(|t| t.trim().parse::<f32>().ok()).collect();
    let [passes, stiffening, ring, kappa] = guard.as_slice() else {
        return Err(usage(format!("--fold-guard: '{guard_text}' is not N,F,R,K")));
    };
    if !(0.0..=64.0).contains(passes)
        || !(1.0..=64.0).contains(stiffening)
        || !(0.0..=8.0).contains(ring)
        || !(0.0..=100.0).contains(kappa)
    {
        return Err(usage(format!("--fold-guard: '{guard_text}' is out of range")));
    }
    let (fold_passes, fold_ring) = (*passes as u32, *ring as u32);
    let visibility = match p.get("visibility").unwrap_or("0.25,0.6") {
        "off" => None,
        text => match numbers("visibility", text)?.as_slice() {
            [lo, hi] if lo < hi => Some((*lo, *hi)),
            _ => return Err(usage("--visibility: give 'LO,HI' with LO < HI, or 'off'".into())),
        },
    };
    let select: Vec<&str> = regions_text.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();

    // The scan, normalised, cropped to the head.
    let job = mesh_job(obj.clone(), units, up, forward);
    let Loaded { scan: full, normalisation, .. } = load(&job)?;
    let (_, top) = full.extent();
    let scan = full.crop(|q| q.y >= top.y - crop);
    drop(full);
    println!(
        "scan: {} vertices, {} triangles in the top {:.0} cm",
        scan.positions.len(),
        scan.triangle_count(),
        crop * 100.0
    );

    // The template and its head as a fit surface.
    // The fitting base: the procedural head without the scanned face (fits are measured from it).
    let generated = template(&GenOptions { lods: false, scan_face: false, ..GenOptions::default() })
        .map_err(|e| CliError::Data(format!("template: {e}")))?;
    let template = generated.template;
    let mut region_weights = RegionWeights::head_scan();
    let masked: Vec<String> = p
        .get("mask-regions")
        .unwrap_or("")
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect();
    for name in &masked {
        if template.regions().find(name).is_none() {
            return Err(usage(format!("--mask-regions: the template has no region '{name}'")));
        }
        region_weights.weights.retain(|(n, _)| n != name);
        region_weights.weights.push((name.clone(), 0.0));
    }
    let mut fit =
        TemplateFit::new(template.mesh(), template.regions(), "body", &select, &region_weights).map_err(data)?;
    if let Some((lo, hi)) = visibility {
        let vis = fit.weight_by_visibility(template.mesh(), 64, 0.03, lo, hi).map_err(data)?;
        let hidden = vis.iter().filter(|&&v| v <= lo).count();
        println!(
            "visibility: {hidden} of {} nodes hidden from the scan (weight 0), mean {:.2}",
            vis.len(),
            vis.iter().sum::<f32>() / vis.len().max(1) as f32
        );
    }
    println!(
        "template: {} fit nodes, {} triangles ({})",
        fit.mesh.positions.len(),
        fit.mesh.triangles.len() / 3,
        select.join(", ")
    );

    // Landmarks: alignment, then fit landmarks.
    let set = LandmarkSet::parse(&read_file(Path::new(landmarks_path), 1 << 20)?).map_err(data)?;
    let (pairs, unknown) = set.match_template(&template);
    if !unknown.is_empty() {
        println!("landmarks the template does not have (ignored): {}", unknown.join(", "));
    }
    let (mut src, mut dst, mut used) = (Vec::new(), Vec::new(), Vec::new());
    for &(i, target) in &pairs {
        let Some(l) = template.landmarks().get(i) else { continue };
        if let Some(rest) = fit.landmark_rest(l) {
            src.push(rest);
            dst.push(target);
            used.push(i);
        }
    }
    let alignment = fit_similarity(&src, &dst, None, true).map_err(data)?;
    let sim = alignment.transform;
    println!("alignment: {} landmarks, scale {:.4}, RMS {:.2} mm", src.len(), sim.scale, alignment.rms * 1000.0);
    let start = fit.mapped_positions(|q| sim.apply(q));
    let fit_mesh = FitMesh { positions: start.clone(), triangles: fit.mesh.triangles.clone() };
    let fit_landmarks: Vec<FitLandmark> = used
        .iter()
        .zip(&dst)
        .filter_map(|(&i, &t)| template.landmarks().get(i).and_then(|l| fit.landmark(l, t)))
        .collect();

    // Registration.
    let target = TargetSurface::new(&scan, vec![1.0; scan.triangle_count()]).map_err(data)?;
    let schedule = Schedule {
        smoothing_lengths: levels.iter().map(|cm| cm * 0.01).collect(),
        landmark_weight,
        landmark_decay: landmark_decay.min(1.0),
        fold_passes,
        fold_stiffening: *stiffening,
        fold_ring,
        fold_membrane: *kappa,
        ..Schedule::default()
    };
    let result = register(&fit_mesh, &fit.node_weights, &fit_landmarks, &target, &schedule, &Thresholds::default())
        .map_err(data)?;
    let r = &result.report;
    let flags = result.flags.names();
    println!(
        "fit: RMS {:.2} mm, max {:.2} mm, matched {:.1} %, landmark RMS {:.2} mm, folded {:.3} %, stretch {:.2}..{:.2}, flags [{}]",
        r.rms_error * 1000.0,
        r.max_error * 1000.0,
        r.matched_fraction * 100.0,
        r.landmark_rms * 1000.0,
        r.folded_fraction * 100.0,
        r.min_stretch,
        r.max_stretch,
        flags.join(", ")
    );

    // Scan to fitted surface: what the template does not reach (weighted area only).
    let fitted_tree = humanart_tools::capture::nearest::TriangleTree::build(&result.positions, &fit.mesh.triangles)
        .map_err(|e| data(e.into()))?;
    let full_weight: Vec<bool> = fit
        .mesh
        .triangles
        .as_chunks::<3>()
        .0
        .iter()
        .map(|t| t.iter().all(|&n| fit.node_weights.get(n as usize).copied().unwrap_or(0.0) >= 1.0))
        .collect();
    let mut back: Vec<f32> = Vec::new();
    for &q in &scan.positions {
        if let Some(h) = fitted_tree.nearest(q, 0.02, |_| true)
            && full_weight.get(h.triangle as usize).copied().unwrap_or(false)
            && q.y > top.y - crop + 0.03
        {
            back.push(h.distance);
        }
    }
    back.sort_by(f32::total_cmp);
    let pct =
        |f: f32| back.get(((back.len() as f32 * f) as usize).min(back.len().saturating_sub(1))).copied().unwrap_or(0.0);
    let back_rms = (back.iter().map(|d| f64::from(d * d)).sum::<f64>() / back.len().max(1) as f64).sqrt() as f32;
    println!(
        "scan to fit ({} scan vertices on fully weighted triangles): RMS {:.2} mm, median {:.2} mm, p95 {:.2} mm, max {:.2} mm",
        back.len(),
        back_rms * 1000.0,
        pct(0.5) * 1000.0,
        pct(0.95) * 1000.0,
        pct(1.0) * 1000.0
    );
    let per_region = region_errors(&template, &fit, &result.error, &result.matched);
    // Folded triangles (orientation flipped against the start) where the scan pulls (all node
    // weights at least 0.5) and elsewhere.
    let mut folds = [(0_usize, 0_usize); 2];
    for t in fit.mesh.triangles.as_chunks::<3>().0 {
        let get = |x: &[Vec3], i: u32| x.get(i as usize).copied().unwrap_or(Vec3::ZERO);
        let normal = |x: &[Vec3]| (get(x, t[1]) - get(x, t[0])).cross(get(x, t[2]) - get(x, t[0]));
        let pulled = t.iter().all(|&n| fit.node_weights.get(n as usize).copied().unwrap_or(0.0) >= 0.5);
        let slot = &mut folds[usize::from(!pulled)];
        slot.0 += 1;
        if normal(&start).dot(normal(&result.positions)) < 0.0 {
            slot.1 += 1;
        }
    }
    let fold_pct = |k: usize| 100.0 * folds[k].1 as f32 / folds[k].0.max(1) as f32;
    let region_folds = region_folds(&template, &fit, &start, &result.positions);
    println!(
        "folded: {:.2} % of {} pulled triangles, {:.2} % of {} others",
        fold_pct(0),
        folds[0].0,
        fold_pct(1),
        folds[1].0
    );

    // The fit file, in the template's frame.
    let inverse = sim.inverse();
    let fitted = FittedSurface {
        source: set.scan.clone(),
        attribution: p.get("attribution").unwrap_or("").to_owned(),
        template_name: template.info().name.clone(),
        template_version: BASE_VERSION,
        template_vertices: template.mesh().vertex_count() as u32,
        alignment: sim,
        summary: humanart_tools::capture::fitted::summary(&result.report, result.flags.0),
        nodes: fit
            .mesh
            .positions
            .iter()
            .enumerate()
            .map(|(n, &rest)| FittedNode {
                template_vertex: fit.node_vertex.get(n).copied().unwrap_or(0),
                rest,
                fitted: inverse.apply(result.positions.get(n).copied().unwrap_or(rest)),
                error: result.error.get(n).copied().filter(|e| e.is_finite()).unwrap_or(-1.0),
                weight: fit.node_weights.get(n).copied().unwrap_or(0.0),
                matched: result.matched.get(n).copied().unwrap_or(false),
            })
            .collect(),
        triangles: fit.mesh.triangles.clone(),
    };
    write_bytes(&out_dir.join("fit.hafit"), &fitted.to_bytes().map_err(|e| data(e.into()))?)?;

    // UV atlas of the tiles the fit touches: tile slots side by side.
    let tiles: Vec<u16> = fit
        .triangle_faces
        .iter()
        .map(|&f| template.mesh().face_tiles().get(f as usize).copied().unwrap_or(0))
        .collect();
    let mut used_tiles: Vec<u16> = tiles.clone();
    used_tiles.sort_unstable();
    used_tiles.dedup();
    let slots = used_tiles.len().max(1) as f32;
    let uvs = template.mesh().uvs();
    let corner_uvs: Vec<Vec2> =
        fit.corners.iter().map(|&v| uvs.get(v as usize).copied().unwrap_or(Vec2::ZERO)).collect();
    let atlas_uvs: Vec<Vec2> = corner_uvs
        .iter()
        .enumerate()
        .map(|(k, uv)| {
            let slot = tiles.get(k / 3).and_then(|t| used_tiles.iter().position(|u| u == t)).unwrap_or(0) as f32;
            Vec2::new((slot + uv.x.clamp(0.0, 1.0)) / slots, uv.y)
        })
        .collect();
    let to_file = |q: Vec3| normalisation.invert(q);
    write_obj(&out_dir.join("fitted-head.obj"), &result.positions, &fit.mesh.triangles, &atlas_uvs, to_file)?;
    write_obj(&out_dir.join("start-head.obj"), &start, &fit.mesh.triangles, &atlas_uvs, to_file)?;
    // The whole template with the fitted head, in the template frame (Y up, metres).
    let body = template.mesh().parts().iter().find(|part| part.name == "body").map(humanart_model::MeshPart::faces);
    let posed = fit
        .apply(template.mesh().positions(), &fitted.nodes.iter().map(|v| v.fitted).collect::<Vec<_>>())
        .map_err(data)?;
    let tri = template.mesh().triangulate();
    let body_tris: Vec<u32> = tri
        .triangle_faces
        .iter()
        .enumerate()
        .filter(|(_, f)| body.as_ref().is_some_and(|b| b.contains(&(**f as usize))))
        .flat_map(|(t, _)| tri.triangle(t).unwrap_or([0, 0, 0]))
        .collect();
    write_obj(&out_dir.join("fitted-template.obj"), &posed, &body_tris, &[], |q| q)?;

    // Albedo transfer and the error map, tile by tile.
    let side = tile_size as usize;
    let atlas_w = side * used_tiles.len().max(1);
    let mut albedo_atlas = vec![[0.0_f32; 3]; atlas_w * side];
    let mut error_atlas = vec![[0.0_f32; 3]; atlas_w * side];
    let albedo = match p.get("albedo") {
        Some(path) => {
            let bytes = read_file(Path::new(path), MAX_ALBEDO_BYTES)?;
            let image = humanart_io::png::decode(&bytes, &Limits::DEFAULT)
                .map_err(|e| CliError::Data(format!("{path}: {e}")))?;
            drop(bytes);
            Some(Albedo::from_image(&image, albedo_size).map_err(data)?)
        }
        None => None,
    };
    let occlusion = if albedo.is_some() {
        humanart_tools::capture::transfer::scan_occlusion(&scan, rays, 0.05).map_err(data)?
    } else {
        Vec::new()
    };
    let empty = Albedo { width: 1, height: 1, texels: vec![[0.5; 3]] };
    let layout = FittedLayout {
        nodes: &result.positions,
        triangles: &fit.mesh.triangles,
        corner_uvs: &corner_uvs,
        tiles: &tiles,
    };
    let source =
        ScanSource { scan: &scan, target: &target, albedo: albedo.as_ref().unwrap_or(&empty), occlusion: &occlusion };
    let settings = TransferSettings { size: tile_size, ..TransferSettings::default() };
    let mut census = [0_usize; 5];
    let mut speck_total = 0;
    for (slot, &tile) in used_tiles.iter().enumerate() {
        let (t, specks) = transfer_and_clean(&layout, &source, tile, &settings).map_err(data)?;
        speck_total += specks;
        for (k, c) in t.census().iter().enumerate() {
            if let Some(x) = census.get_mut(k) {
                *x += c;
            }
        }
        for y in 0..side {
            for x in 0..side {
                let i = y * side + x;
                let j = y * atlas_w + slot * side + x;
                if let (Some(a), Some(e)) = (albedo_atlas.get_mut(j), error_atlas.get_mut(j)) {
                    *a = t.texels.get(i).copied().unwrap_or([0.0; 3]);
                    *e = match t.states.get(i) {
                        Some(&s) if s == transfer_state::EMPTY || s == transfer_state::PADDING => [0.3; 3],
                        _ => error_colour(t.distance.get(i).copied().unwrap_or(-1.0)),
                    };
                }
            }
        }
    }
    let atlas_w32 = atlas_w as u32;
    if albedo.is_some() {
        write_rgb_png(&out_dir.join("albedo-atlas.png"), atlas_w32, tile_size, &albedo_atlas)?;
    }
    write_rgb_png(&out_dir.join("error-atlas.png"), atlas_w32, tile_size, &error_atlas)?;

    // The report.
    let mut report = String::new();
    {
        use std::fmt::Write as _;
        let _ = writeln!(report, "humanart-cli fit-scan report");
        let _ = writeln!(
            report,
            "scan: {} ({}), top {:.0} cm: {} vertices",
            set.scan,
            obj.display(),
            crop * 100.0,
            scan.positions.len()
        );
        let _ = writeln!(
            report,
            "template: {} base v{} ({} fit nodes, regions {})",
            template.info().name,
            BASE_VERSION,
            fit.mesh.positions.len(),
            select.join(", ")
        );
        let _ = writeln!(
            report,
            "alignment: {} landmarks, scale {:.4}, RMS {:.2} mm",
            src.len(),
            sim.scale,
            alignment.rms * 1000.0
        );
        let _ = writeln!(
            report,
            "fit (template to scan, weighted matched nodes): RMS {:.2} mm, max {:.2} mm, matched {:.1} %, landmark RMS {:.2} mm, folded {:.3} %, stretch {:.2}..{:.2}, flags [{}]",
            r.rms_error * 1000.0,
            r.max_error * 1000.0,
            r.matched_fraction * 100.0,
            r.landmark_rms * 1000.0,
            r.folded_fraction * 100.0,
            r.min_stretch,
            r.max_stretch,
            flags.join(", ")
        );
        let _ = writeln!(
            report,
            "schedule: levels {levels:?} cm, landmark weight {landmark_weight} (decay {landmark_decay} per level), fold guard {guard_text}, visibility {visibility:?}, masked regions [{}]",
            masked.join(", ")
        );
        let _ = writeln!(
            report,
            "folded: {:.2} % of {} pulled triangles (node weights >= 0.5), {:.2} % of {} others",
            fold_pct(0),
            folds[0].0,
            fold_pct(1),
            folds[1].0
        );
        let _ = writeln!(
            report,
            "scan to fit ({} scan vertices): RMS {:.2} mm, median {:.2} mm, p95 {:.2} mm, max {:.2} mm",
            back.len(),
            back_rms * 1000.0,
            pct(0.5) * 1000.0,
            pct(0.95) * 1000.0,
            pct(1.0) * 1000.0
        );
        let _ = writeln!(
            report,
            "fold guard: {} passes, {} nodes stiffened, max stiffness {:.0}; solver limits {}",
            r.fold_passes, r.stiffened_nodes, r.max_stiffness, r.solver_limits
        );
        let _ = writeln!(
            report,
            "folds per region (triangle in the region of its first node): pulled folded/total, others folded/total"
        );
        for f in &region_folds {
            let _ = writeln!(
                report,
                "  {:<10} {:5}/{:<6} {:5}/{:<6}",
                f.name, f.pulled.1, f.pulled.0, f.others.1, f.others.0
            );
        }
        let _ = writeln!(report, "per region (weighted nodes; distance to the scan): nodes, matched, RMS mm, max mm");
        for e in &per_region {
            let _ = writeln!(
                report,
                "  {:<10} {:6} {:6} {:7.2} {:7.2}",
                e.name,
                e.nodes,
                e.matched,
                e.rms * 1000.0,
                e.max * 1000.0
            );
        }
        let _ = writeln!(report, "landmark residuals (mm):");
        for (&i, l) in used.iter().zip(&fit_landmarks) {
            let name = template.landmarks().get(i).map_or("?", |l| l.name.as_str());
            let _ =
                writeln!(report, "  {:<28} {:6.2}", name, l.position(&result.positions).distance(l.target) * 1000.0);
        }
        let _ = writeln!(
            report,
            "albedo transfer: tiles {:?} at {}², texels scan {}, specks inpainted {}, no scan point {}, padding {}",
            used_tiles,
            tile_size,
            census[usize::from(transfer_state::SCAN)],
            speck_total,
            census[usize::from(transfer_state::NO_SCAN)],
            census[usize::from(transfer_state::PADDING)]
        );
    }
    write_bytes(&out_dir.join("fit-report.txt"), report.as_bytes())?;
    print!("{report}");
    println!("wrote {}", out_dir.display());
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// face-model: the mean face and its shape components from several fits.
// ---------------------------------------------------------------------------------------------

const FACE_MODEL_OPTIONS: [OptSpec; 6] = [
    OptSpec { name: "fits", value: Some("LIST"), default: None, help: "Comma-separated .hafit files (fit-scan)." },
    OptSpec {
        name: "out",
        value: Some("FILE"),
        default: Some("face-model.hafm"),
        help: "The face model file (humanart-face-model).",
    },
    OptSpec { name: "components", value: Some("N"), default: Some("9"), help: "Most shape components kept." },
    OptSpec {
        name: "mesh-dir",
        value: Some("DIR"),
        default: None,
        help: "Also write OBJs (template frame) of the mean head and each component at -2 / +2 SD.",
    },
    OptSpec {
        name: "description",
        value: Some("TEXT"),
        default: Some(""),
        help: "Sources and attribution stored in the model.",
    },
    OptSpec { name: "json-errors", value: None, default: None, help: "Errors as one JSON line on stderr." },
];

/// The `face-model` command.
pub static FACE_MODEL: CommandSpec = CommandSpec {
    name: "face-model",
    summary: "Averages several scan fits into the mean face offset and its shape components.",
    options: &FACE_MODEL_OPTIONS,
};

/// The blend mask and the alignment weights of the face model per fit node, from the template's
/// regions: the face (and everything not listed) blends fully and aligns fully; the ears blend
/// fully but align at 0.3 (hair hides them on most scans); the hairline blends at 0.5 and the scalp
/// not at all (hair, caps); the neck fades from 1 under the jaw to 0 over its lower half and aligns
/// at 0.2 (collars, pose). The blend is then smoothed over the surface (80 umbrella rounds, a
/// falloff of a few centimetres) so no region border leaves a step.
pub fn face_masks(template: &Template, fit: &FittedSurface) -> (Vec<f32>, Vec<f32>) {
    let regions = template.regions();
    let id = |name: &str| regions.find(name);
    let (scalp, hairline, neck) = (id("scalp"), id("hairline"), id("neck"));
    let ears = [id("ear-L"), id("ear-R")];
    let region_of = |v: u32| regions.vertex_regions.get(v as usize).copied().unwrap_or(0);
    let within = |r: u16, a: Option<u16>| a.is_some_and(|a| regions.is_within(r, a));
    let neck_y: Vec<f32> =
        fit.nodes.iter().filter(|v| within(region_of(v.template_vertex), neck)).map(|v| v.rest.y).collect();
    let bottom = fit.nodes.iter().map(|v| v.rest.y).fold(f32::MAX, f32::min);
    let top = neck_y.iter().copied().fold(bottom, f32::max);
    let mut blend = Vec::with_capacity(fit.nodes.len());
    let mut align = Vec::with_capacity(fit.nodes.len());
    for v in &fit.nodes {
        let r = region_of(v.template_vertex);
        let (b, a) = if within(r, scalp) {
            (0.0, 0.0)
        } else if within(r, hairline) {
            (0.5, 0.0)
        } else if ears.iter().any(|&e| within(r, e)) {
            (1.0, 0.3)
        } else if within(r, neck) {
            let t = ((v.rest.y - bottom) / (top - bottom).max(1.0e-6) - 0.4) / 0.5;
            let t = t.clamp(0.0, 1.0);
            (t * t * (3.0 - 2.0 * t), 0.2)
        } else {
            (1.0, 1.0)
        };
        blend.push(b);
        align.push(a);
    }
    // Smooth the blend over the surface.
    let n = fit.nodes.len();
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for t in fit.triangles.as_chunks::<3>().0 {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            if let Some(l) = adj.get_mut(a as usize) {
                l.push(b);
            }
            if let Some(l) = adj.get_mut(b as usize) {
                l.push(a);
            }
        }
    }
    let mut next = blend.clone();
    for _ in 0..80 {
        for (i, nb) in adj.iter().enumerate() {
            if nb.is_empty() {
                continue;
            }
            let mean = nb.iter().map(|&j| blend.get(j as usize).copied().unwrap_or(0.0)).sum::<f32>() / nb.len() as f32;
            next[i] = f32::midpoint(blend[i], mean);
        }
        blend.copy_from_slice(&next);
    }
    (blend, align)
}

/// Whether fits of `fit` can feed a face model of this build: same template name and the same
/// fitting base ([`BASE_VERSION`]); the template's own version also counts the scan face and the
/// finishing steps on top of the base, which the fits never saw.
///
/// # Errors
/// [`CliError::Data`] naming both versions when they differ.
pub fn base_matches(template_name: &str, fit: &FittedSurface) -> Result<(), CliError> {
    if template_name != fit.template_name || fit.template_version != BASE_VERSION {
        return Err(CliError::Data(format!(
            "the fits are of template {} base v{}, this build fits to {} base v{}: refit the scans",
            fit.template_name, fit.template_version, template_name, BASE_VERSION
        )));
    }
    Ok(())
}

/// Runs the `face-model` command line `args`.
///
/// # Errors
/// [`CliError::Usage`], [`CliError::Io`], [`CliError::Data`] for unreadable or mismatched fits.
#[allow(clippy::too_many_lines)]
pub fn face_model_command(program: &str, args: &[String]) -> Result<(), CliError> {
    use humanart_tools::capture::face_model::{BuildSettings, build, flipped};
    use std::fmt::Write as _;
    let p = Parsed::parse(&FACE_MODEL, args)?;
    if p.help() {
        print!("{}", crate::args::help(program, &FACE_MODEL));
        return Ok(());
    }
    let data = |e: humanart_tools::capture::CaptureError| CliError::Data(e.to_string());
    let list = p.get("fits").ok_or_else(|| usage("--fits is required".into()))?;
    let mut fits = Vec::new();
    for path in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let bytes = read_file(Path::new(path), 1 << 30)?;
        fits.push(FittedSurface::parse(&bytes).map_err(|e| CliError::Data(format!("{path}: {e}")))?);
    }
    let first = fits.first().ok_or_else(|| usage("--fits names no file".into()))?;
    // The fitting base: the procedural head without the scanned face (fits are measured from it).
    let generated = template(&GenOptions { lods: false, scan_face: false, ..GenOptions::default() })
        .map_err(|e| CliError::Data(format!("template: {e}")))?;
    let template = generated.template;
    base_matches(&template.info().name, first)?;
    let (blend, align) = face_masks(&template, first);
    let settings = BuildSettings { components: p.number("components", 0, 64)? as usize, ..BuildSettings::default() };
    let description = p.get("description").unwrap_or("").to_owned();
    let (model, report) = build(&fits, &blend, &align, description, &settings).map_err(data)?;
    let out = PathBuf::from(p.get("out").unwrap_or("face-model.hafm"));
    let bytes = model.to_bytes().map_err(|e| CliError::Data(e.to_string()))?;
    write_bytes(&out, &bytes)?;

    // Folds of the mean and of each component at -2 / +2 SD, per report region.
    let regions = template.regions();
    let ids: Vec<Option<u16>> = REPORT_REGIONS.iter().map(|n| regions.find(n)).collect();
    let region_of_triangle = |t: u32| {
        let node = model.triangles.get(3 * t as usize).copied().unwrap_or(0);
        let v = model.template_vertices.get(node as usize).copied().unwrap_or(0);
        let r = regions.vertex_regions.get(v as usize).copied().unwrap_or(0);
        ids.iter()
            .position(|a| a.is_some_and(|a| regions.is_within(r, a)))
            .and_then(|i| REPORT_REGIONS.get(i).copied())
            .unwrap_or("other")
    };
    let fold_text = |offsets: &[Vec3]| {
        let flips = flipped(&model.rest, &model.triangles, offsets);
        let mut names: Vec<&str> = flips.iter().map(|&t| region_of_triangle(t)).collect();
        names.sort_unstable();
        let mut text = format!("{} flipped", flips.len());
        for name in REPORT_REGIONS.iter().chain(std::iter::once(&"other")) {
            let count = names.iter().filter(|n| *n == name).count();
            if count > 0 {
                let _ = write!(text, ", {name} {count}");
            }
        }
        text
    };
    let mut text = String::new();
    let _ = writeln!(text, "humanart-cli face-model report");
    let _ = writeln!(
        text,
        "template {} v{}: {} nodes, {} triangles; {} subjects; {} bytes written to {}",
        model.template_name,
        model.template_version,
        model.rest.len(),
        model.triangles.len() / 3,
        model.subjects,
        bytes.len(),
        out.display()
    );
    let _ = writeln!(text, "subjects (dense alignment scale, RMS from the template, RMS from the mean, mm):");
    for s in &report.subjects {
        let _ = writeln!(
            text,
            "  {:<28} {:.4} {:6.2} {:6.2}",
            s.source,
            s.scale,
            s.rms_from_template * 1000.0,
            s.rms_from_mean * 1000.0
        );
    }
    let _ = writeln!(
        text,
        "mean (blended): RMS {:.2} mm, max {:.2} mm; asymmetry before symmetrising up to {:.2} mm",
        report.mean_rms * 1000.0,
        report.mean_max * 1000.0,
        report.asymmetry_max * 1000.0
    );
    let _ = writeln!(
        text,
        "mean folds: {} before untangling, {} after {} rounds; as stored: {}",
        report.mean_flips_before,
        report.mean_flips,
        report.untangle_rounds,
        fold_text(&model.mean)
    );
    for (k, c) in model.components.iter().enumerate() {
        let at = |sd: f32| -> Vec<Vec3> { model.mean.iter().zip(&c.offsets).map(|(&m, &o)| m + o * sd).collect() };
        let _ = writeln!(
            text,
            "component {}: {:.1} % of the variance, +1 SD RMS {:.2} mm; folds at -2 SD: {}; at +2 SD: {}",
            k + 1,
            c.variance * 100.0,
            report.component_rms.get(k).copied().unwrap_or(0.0) * 1000.0,
            fold_text(&at(-2.0)),
            fold_text(&at(2.0))
        );
    }
    if let Some(dir) = p.get("mesh-dir") {
        let dir = PathBuf::from(dir);
        let moved = |offsets: &[Vec3]| -> Vec<Vec3> { model.rest.iter().zip(offsets).map(|(&r, &o)| r + o).collect() };
        write_obj(&dir.join("rest-head.obj"), &model.rest, &model.triangles, &[], |q| q)?;
        write_obj(&dir.join("mean-head.obj"), &moved(&model.mean), &model.triangles, &[], |q| q)?;
        for (k, c) in model.components.iter().enumerate() {
            for sd in [-2.0_f32, 2.0] {
                let offsets: Vec<Vec3> = model.mean.iter().zip(&c.offsets).map(|(&m, &o)| m + o * sd).collect();
                let name = format!("pc{}-{}2sd.obj", k + 1, if sd < 0.0 { "minus" } else { "plus" });
                write_obj(&dir.join(name), &moved(&offsets), &model.triangles, &[], |q| q)?;
            }
        }
        let _ = writeln!(text, "meshes: {}", dir.display());
    }
    print!("{text}");
    let report_path = p
        .get("mesh-dir")
        .map_or_else(|| out.with_extension("report.txt"), |d| PathBuf::from(d).join("face-model-report.txt"));
    write_bytes(&report_path, text.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `[WP-25]` Fits feed a face model when they were fitted to this build's fitting base: the
    /// template's own version (raised by the scan face) does not matter, the base version does.
    #[test]
    fn wp25_face_model_takes_fits_of_the_same_base() {
        use humanart_model::scanfit::fitted::FittedSummary;
        use humanart_model::scanfit::similarity::Similarity;
        let fit = |name: &str, version: u32| FittedSurface {
            source: "s".into(),
            attribution: String::new(),
            template_name: name.into(),
            template_version: version,
            template_vertices: 0,
            alignment: Similarity::IDENTITY,
            summary: FittedSummary::default(),
            nodes: Vec::new(),
            triangles: Vec::new(),
        };
        let name = humanart_model::generate::build::TEMPLATE_NAME;
        assert!(base_matches(name, &fit(name, BASE_VERSION)).is_ok());
        assert!(base_matches(name, &fit(name, BASE_VERSION + 1)).is_err());
        assert!(base_matches(name, &fit("other", BASE_VERSION)).is_err());
    }

    fn job(list: &[&str]) -> Result<MeshJob, CliError> {
        let args: Vec<String> = list.iter().map(|s| (*s).to_owned()).collect();
        MeshJob::from_args(&Parsed::parse(&RENDER_MESH, &args).map_err(CliError::Usage)?)
    }

    #[test]
    fn wp25_render_mesh_arguments_map_to_the_job() {
        let j =
            job(&["--obj", "a.obj", "--units", "mm", "--up", "+z", "--forward", "-y", "--view", "profile"]).unwrap();
        assert_eq!(j.parts, vec![(PathBuf::from("a.obj"), None)]);
        assert_eq!((j.units, j.up, j.forward), (Some(Units::Millimetres), Axis::PosZ, Some(Axis::NegY)));
        assert_eq!(j.camera, ReviewCamera::ProfileLeft);
        let d = job(&["--obj", "a.obj"]).unwrap();
        assert_eq!((d.units, d.up, d.forward, d.albedo_size), (None, Axis::PosY, None, 4096));
        for bad in [
            &[][..],
            &["--obj", "a", "--units", "inch"],
            &["--obj", "a", "--up", "z"],
            &["--obj", "a", "--albedo-size", "8192"],
            &["--obj", "a", "--view", "side"],
        ] {
            assert!(matches!(job(bad), Err(CliError::Usage(_))), "{bad:?}");
        }
    }

    #[test]
    fn wp25_part_lists_parse_paths_and_colours() {
        let parts = parse_list("// bones\nFJ1.obj #ffffff\n\n  sub/FJ2.obj  \n", Path::new("base")).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].0, Path::new("base").join("FJ1.obj"));
        assert!((parts[0].1.unwrap()[0] - 1.0).abs() < 1e-6);
        assert_eq!(parts[1], (Path::new("base").join("sub/FJ2.obj"), None));
        assert!(parse_list("", Path::new("")).is_err());
        assert!(parse_list("a.obj #12345g\n", Path::new("")).is_err());
        assert!(parse_list("a.obj #ééé\n", Path::new("")).is_err());
    }

    #[test]
    fn wp25_picks_parse_and_rays_go_through_their_pixels() {
        let picks = parse_picks(
            "# front
head nose.tip 540 700.5

profile L.tragus 300 650 # ear
",
        )
        .unwrap();
        assert_eq!(picks.len(), 2);
        assert_eq!((picks[0].camera, picks[0].name.as_str(), picks[0].y), (ReviewCamera::Front, "nose.tip", 700.5));
        assert_eq!(picks[1].camera, ReviewCamera::ProfileLeft);
        for bad in ["head nose.tip 1", "side a 1 2", "head a x 2", "head a 1 NaN"] {
            assert!(parse_picks(bad).is_err(), "{bad}");
        }
        // The centre of the image looks along the camera's axis; a point projected by the
        // renderer's matrices comes back on its pixel's ray.
        let pts: Vec<Vec3> = (0..=180).map(|i| Vec3::new(0.0, i as f32 * 0.01, 0.0)).collect();
        let cam = frame(&pts, ReviewCamera::ThreeQuarterLeft, 0.8);
        let (o, d) = pixel_ray(&cam, 400.0, 500.0, 800, 1000);
        assert!(d.abs_diff_eq(cam.forward(), 1e-5) && o == cam.position);
        let p = Vec3::new(0.03, 1.72, 0.05);
        let ndc = (cam.projection(0.8) * cam.view()).project_point3(p);
        let (x, y) = ((ndc.x * 0.5 + 0.5) * 800.0, (0.5 - ndc.y * 0.5) * 1000.0);
        let (o, d) = pixel_ray(&cam, x, y, 800, 1000);
        let along = (p - o).dot(d);
        assert!((o + d * along).distance(p) < 1e-4, "{x} {y}");
    }

    #[test]
    fn wp25_error_colours_ramp_and_mark_misses() {
        assert_eq!(error_colour(-1.0), [0.8, 0.0, 0.8]);
        assert_eq!(error_colour(f32::INFINITY), [0.8, 0.0, 0.8]);
        let near = |a: [f32; 3], b: [f32; 3]| a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-5);
        assert!(near(error_colour(0.0), [0.0, 0.05, 0.6]));
        assert!(near(error_colour(0.001), [0.0, 0.5, 0.05]));
        assert!(near(error_colour(0.01), [0.7, 0.02, 0.0]));
        let mid = error_colour(0.003);
        assert!(mid[0] > 0.69 && mid[1] > 0.2 && mid[1] < 0.6);
    }

    #[test]
    fn wp25_framing_matches_the_viewer_rule() {
        // A 1.8 m column: the head sphere covers the top 13 % of the height.
        let pts: Vec<Vec3> = (0..=180).map(|i| Vec3::new(0.0, i as f32 * 0.01, 0.0)).collect();
        let (c, r) = head_bounds(&pts);
        assert!((c.y - (1.8 - 0.5 * 0.13 * 1.8)).abs() < 0.01, "{c:?}");
        assert!((r - 0.5 * 0.13 * 1.8).abs() < 0.01, "{r}");
        let cam = frame(&pts, ReviewCamera::Front, 0.8);
        assert!(cam.position.z > c.z && (cam.target - c).length() < 1e-6);
    }

    #[test]
    fn wp25_render_mesh_draws_a_textured_obj() {
        // [WP-25] A small textured quad (two triangles) renders through the surfaced-skin path
        // and the plain path, and writes PNGs.
        let dir = std::env::temp_dir().join(format!("humanart-render-mesh-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let obj = dir.join("quad.obj");
        std::fs::write(
            &obj,
            "v -0.2 0 0\nv 0.2 0 0\nv 0.2 1.7 0\nv -0.2 1.7 0\nv 0 0 0.1\nvt 0 1\nvt 1 1\nvt 1 0\nvt 0 0\n\
             f 1/1 2/2 3/3\nf 1/1 3/3 4/4\nf 1/1 5/1 2/2\n",
        )
        .unwrap();
        let png = dir.join("albedo.png");
        let mut pixels = Vec::new();
        for y in 0..8_u8 {
            for x in 0..8_u8 {
                pixels.extend_from_slice(&[x * 30, y * 30, 128]);
            }
        }
        let image = humanart_io::Image::new(
            humanart_io::Format::Png,
            8,
            8,
            3,
            humanart_io::ColorSpace::Srgb,
            humanart_io::Samples::U8(pixels),
        )
        .unwrap();
        std::fs::write(&png, humanart_io::png::encode(&image).unwrap()).unwrap();
        let out = dir.join("out.png");
        let base = MeshJob {
            out: out.clone(),
            width: 64,
            height: 80,
            frames: 2,
            forward: Some(Axis::PosZ),
            ..job(&["--obj", obj.to_str().unwrap(), "--albedo-size", "64"]).unwrap()
        };
        for albedo in [Some(png.clone()), None] {
            let j = MeshJob { albedo: albedo.clone(), ..base.clone() };
            let s = run(&j).unwrap();
            assert_eq!((s.vertices, s.triangles, s.units), (5, 3, Units::Metres));
            assert_eq!(s.albedo_size, if albedo.is_some() { 64 } else { 0 });
            assert!((s.height - 1.7).abs() < 1e-5);
            assert_eq!(&std::fs::read(&out).unwrap()[1..4], b"PNG");
        }
        let missing = MeshJob { parts: vec![(dir.join("none.obj"), None)], ..base };
        assert!(matches!(run(&missing), Err(CliError::Io { .. })));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
