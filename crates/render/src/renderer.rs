//! The renderer: the frame graph of the core passes and their execution `[RND-01]`..`[RND-04]`.
//!
//! Pass list (section 8.1 item numbers; the later work packages add theirs between them):
//!
//! | pass        | shader                     | item | reads                                   | writes |
//! |-------------|----------------------------|------|-----------------------------------------|--------|
//! | `shadows`   | `render/shadow.wgsl`       | 4    | scene                                   | shadow atlas (imported) |
//! | `geometry`  | `render/geometry.wgsl`     | 5    | scene, lights, IBL, shadow atlas        | diffuse, specular (rgba16f), surface (rgba32u), depth (d32f) |
//! | `sss_h`     | `render/sss.wgsl`          | 6    | diffuse, specular (profile scale)       | sss temp (rgba16f) |
//! | `sss_v`     | `render/sss.wgsl`          | 6    | sss temp, specular                      | diffuse sss (rgba16f) |
//! | `composite` | `render/composite.wgsl`    | 9    | diffuse (sss), specular, surface, depth | hdr (rgba16f) |
//! | `taa`       | `render/taa.wgsl`          | 10   | hdr, history in, surface, depth         | history out (imported) |
//! | `tonemap`   | `post/tonemap.wgsl`        | 10   | history out (or hdr without TAA)        | display (imported) |
//! | `debug`     | `render/debug.wgsl`        | 11   | diffuse, specular, surface, depth, hdr  | display (replaces `tonemap`) |
//! | `overlay`   | `render/overlay.wgsl`      | 11   | depth                                   | display |
//! | `pick`      | `render/pick.wgsl`         | -    | scene                                   | pick target (side effect: readback) |
//!
//! The graph is compiled when the configuration changes ([`Renderer::configure`]: size, quality,
//! debug view, overlays, picking, TAA); then the transient targets, their bind groups and the
//! pipelines of the configuration exist, and [`Renderer::render`] only walks the compiled passes,
//! pushes the frame constants into the uniform ring and records commands: no allocation in the
//! steady state `[PERF-R1]`, pipelines and bind groups created once `[PERF-R4]`, timings per pass
//! through the GPU profiler `[PERF-R6]`, picking read back asynchronously `[PERF-R3]`.
//!
//! The subsurface passes exist in the Review and Still tiers (the separable skin diffusion,
//! `[SHD-02]`); the Interactive tier shades skin with the pre-integrated falloff in the geometry
//! pass instead (`[SHD-04]`, `SSS_MODE` 1), and the composite reads the geometry's diffuse.
//!
//! The history of the TAA is two imported textures used in turn; the colour space of every target
//! is linear Rec.709, depth is reversed-Z 32-bit float `[RND-02]`.

use humanart_gpu::cache::PipelineCache;
use humanart_gpu::pod::{bytes_of, slice_as_bytes};
use humanart_gpu::profiler::{FrameReport, GpuProfiler};
use humanart_gpu::readback::ReadbackRing;
use humanart_gpu::resources::{BufferDesc, BufferId, Category, Resources, SamplerKey, Tag, TextureDesc, TextureId};
use humanart_gpu::uniform_ring::{UniformRing, UniformRingDesc};
use humanart_gpu::{Gpu, GpuError};
use humanart_math::color::tonemap::ToneMapper;
use humanart_math::{Vec2, taa_jitter_pixels};

use crate::camera::{Camera, TileRect, ViewMatrices};
use crate::error::RenderError;
use crate::graph::{CompiledGraph, GraphBuilder, ResourceId, TextureInfo};
use crate::ibl::{EnvironmentSource, Ibl, SPECULAR_MIPS};
use crate::light::{Background, GpuLight, LightRig, MAX_SHADOWED, ShadowFrustum};
use crate::scene::{GpuScene, SCENE_LAYOUT};
use crate::surface::SkinSurface;
use crate::uniforms::{FrameUniforms, RING_BINDING_SIZE, ShadowViewUniforms, TonemapUniforms};

/// Format of the HDR targets `[RND-02]`.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Format of the G-buffer-lite surface target.
pub const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;
/// Depth format: reversed-Z, 32-bit float `[RND-02]`.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Format of the picking target: pick id, depth bits.
pub const PICK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;
/// Most overlay primitives per frame.
pub const MAX_OVERLAY_PRIMITIVES: usize = 4096;
/// Largest render target side the renderer accepts; bigger images are tiled stills `[QUAL-04]`.
pub const MAX_TARGET_SIDE: u32 = 8192;
/// Frames of the TAA jitter sequence (Halton 2, 3).
pub const JITTER_PERIOD: u32 = 16;

/// The quality tiers `[QUAL-01]`: they choose shader variants (override constants).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Quality {
    /// Fewer shadow taps, for dragging.
    Interactive,
    /// The default.
    Review,
    /// Most taps, for stills.
    Still,
}

impl Quality {
    /// The skin diffusion (`SSS_MODE` of `render/geometry.wgsl`): 0 the separable subsurface
    /// passes, 1 the pre-integrated falloff of the Interactive tier `[SHD-04]`.
    pub const fn sss_mode(self) -> u32 {
        match self {
            Self::Interactive => 1,
            Self::Review | Self::Still => 0,
        }
    }

    /// PCF taps per shadowed light (`SHADOW_TAPS` of `render/geometry.wgsl`).
    pub const fn shadow_taps(self) -> u32 {
        match self {
            Self::Interactive => 8,
            Self::Review => 16,
            Self::Still => 32,
        }
    }
}

/// The debug and analysis views of section 8.5 that the core passes provide (numbering of
/// `render/debug.wgsl`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DebugView {
    /// The final image.
    Beauty,
    /// Base colour.
    Albedo,
    /// Diffuse lighting times albedo.
    Diffuse,
    /// Specular only.
    Specular,
    /// Diffuse lighting for unit albedo.
    Irradiance,
    /// World normals.
    Normals,
    /// Perceptual roughness.
    Roughness,
    /// Metalness.
    Metalness,
    /// Thickness.
    Thickness,
    /// Ambient occlusion.
    Occlusion,
    /// Material classes.
    MaterialClass,
    /// Depth.
    Depth,
    /// Motion vectors.
    MotionVectors,
    /// The first shadow-map layer.
    ShadowMap,
    /// Exposure in false colour.
    FalseColor,
}

impl DebugView {
    /// Every view.
    pub const ALL: [Self; 15] = [
        Self::Beauty,
        Self::Albedo,
        Self::Diffuse,
        Self::Specular,
        Self::Irradiance,
        Self::Normals,
        Self::Roughness,
        Self::Metalness,
        Self::Thickness,
        Self::Occlusion,
        Self::MaterialClass,
        Self::Depth,
        Self::MotionVectors,
        Self::ShadowMap,
        Self::FalseColor,
    ];

    /// The shader's view number.
    pub const fn code(self) -> u32 {
        self as u32
    }

    /// The stable name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Beauty => "beauty",
            Self::Albedo => "albedo",
            Self::Diffuse => "diffuse",
            Self::Specular => "specular",
            Self::Irradiance => "irradiance",
            Self::Normals => "normals",
            Self::Roughness => "roughness",
            Self::Metalness => "metalness",
            Self::Thickness => "thickness",
            Self::Occlusion => "occlusion",
            Self::MaterialClass => "material_class",
            Self::Depth => "depth",
            Self::MotionVectors => "motion_vectors",
            Self::ShadowMap => "shadow_map",
            Self::FalseColor => "false_color",
        }
    }
}

/// What the graph is built for. Changing it recompiles the graph and recreates the targets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderConfig {
    /// Target width, pixels.
    pub width: u32,
    /// Target height, pixels.
    pub height: u32,
    /// Quality tier.
    pub quality: Quality,
    /// The view shown.
    pub debug: DebugView,
    /// Whether the overlay pass runs.
    pub overlays: bool,
    /// Whether the picking pass exists.
    pub picking: bool,
    /// Temporal anti-aliasing (and accumulation).
    pub taa: bool,
    /// The display tone mapper `[COL-02]`.
    pub tone_mapper: ToneMapper,
    /// Side of each shadow-map layer.
    pub shadow_map_size: u32,
}

impl RenderConfig {
    /// A review configuration of `width` x `height`.
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            quality: Quality::Review,
            debug: DebugView::Beauty,
            overlays: false,
            picking: false,
            taa: true,
            tone_mapper: ToneMapper::PbrNeutral,
            shadow_map_size: 2048,
        }
    }

    fn validate(&self, limits: &wgpu::Limits) -> Result<(), RenderError> {
        let max = limits.max_texture_dimension_2d.min(MAX_TARGET_SIDE);
        if self.width == 0 || self.height == 0 {
            return Err(RenderError::InvalidParameter { what: "a render target of zero size" });
        }
        if self.width > max || self.height > max {
            return Err(RenderError::TooLarge {
                what: "render target side (use a tiled still)",
                value: u64::from(self.width.max(self.height)),
                limit: u64::from(max),
            });
        }
        if !(256..=max).contains(&self.shadow_map_size) {
            return Err(RenderError::InvalidParameter { what: "shadow map size outside 256..max texture side" });
        }
        Ok(())
    }
}

humanart_gpu::pod_struct! {
    /// One overlay primitive (`struct Primitive` of `render/overlay.wgsl`).
    #[derive(Debug, PartialEq)]
    pub struct OverlayPrimitive(size = 48, align = 4) {
        /// Start (world), width in pixels.
        pub a: [f32; 4],
        /// End (world), kind (0 line, 1 point) + 16 if drawn on top.
        pub b: [f32; 4],
        /// Display colour (sRGB-encoded), straight alpha.
        pub color: [f32; 4],
    }
}

impl OverlayPrimitive {
    /// A line from `a` to `b`, `width` pixels, depth-tested unless `on_top`.
    pub fn line(a: humanart_math::Vec3, b: humanart_math::Vec3, width: f32, color: [f32; 4], on_top: bool) -> Self {
        Self { a: [a.x, a.y, a.z, width], b: [b.x, b.y, b.z, if on_top { 16.0 } else { 0.0 }], color }
    }

    /// A round point at `p` of `diameter` pixels (a landmark, a brush centre).
    pub fn point(p: humanart_math::Vec3, diameter: f32, color: [f32; 4], on_top: bool) -> Self {
        Self { a: [p.x, p.y, p.z, diameter], b: [p.x, p.y, p.z, if on_top { 17.0 } else { 1.0 }], color }
    }
}

/// A tile of a still `[QUAL-04]`: which part of the full image this frame renders, the sub-pixel
/// offset of this sample, and how many samples were accumulated before it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StillFrame {
    /// Full image width.
    pub full_width: u32,
    /// Full image height.
    pub full_height: u32,
    /// The rectangle of the full image this frame covers (its size is the target's).
    pub tile: TileRect,
    /// Jitter of this sample, pixels.
    pub jitter: Vec2,
    /// Samples already accumulated (0 starts a new mean).
    pub sample: u32,
}

/// Everything one frame needs.
#[derive(Debug)]
pub struct FrameInput<'a> {
    /// The camera.
    pub camera: &'a Camera,
    /// The lights (the environment is set with [`Renderer::set_environment`]).
    pub rig: &'a LightRig,
    /// The scene.
    pub scene: &'a mut GpuScene,
    /// Overlay lines and points.
    pub overlay: &'a [OverlayPrimitive],
    /// Draw the wireframe overlay of every object.
    pub wireframe: bool,
    /// A still tile and sample, instead of an interactive frame.
    pub still: Option<StillFrame>,
    /// A picking request at this pixel.
    pub pick: Option<[u32; 2]>,
    /// Forget the TAA history (camera cut, scene switch).
    pub reset_history: bool,
}

/// What a picking readback found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PickHit {
    /// The pixel asked about.
    pub pixel: [u32; 2],
    /// The object's pick id; 0 when nothing was hit.
    pub pick_id: u32,
    /// Reversed-Z depth of the hit (0 when nothing).
    pub depth: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Key {
    Shader(&'static str),
    Geometry(u32, u32),
    Sss(bool),
    Shadow,
    Composite,
    Taa,
    Tonemap(u32),
    Debug,
    OverlayLines,
    OverlayWire,
    OverlaySelection,
    Pick,
}

/// The passes of the core graph (the payload of its compiled passes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassKind {
    /// Shadow maps.
    Shadows,
    /// Opaque geometry with lighting.
    Geometry,
    /// Subsurface scattering of skin along x.
    SubsurfaceH,
    /// Subsurface scattering of skin along y.
    SubsurfaceV,
    /// Lighting composite.
    Composite,
    /// Hair: deep opacity maps, visibility, resolve over the composite, motion (`crate::hair`).
    Hair,
    /// Temporal anti-aliasing.
    Taa,
    /// Display transform.
    Tonemap,
    /// Debug view.
    Debug,
    /// Overlays.
    Overlay,
    /// Picking.
    Pick,
}

/// The resources of the compiled graph.
#[derive(Debug, Clone, Copy)]
struct GraphResources {
    diffuse: ResourceId,
    sss_temp: Option<ResourceId>,
    diffuse_sss: Option<ResourceId>,
    specular: ResourceId,
    surface: ResourceId,
    depth: ResourceId,
    hdr: ResourceId,
}

/// Bind-group layouts and pipeline layouts, created once.
#[derive(Debug)]
struct Layouts {
    scene: wgpu::BindGroupLayout,
    lighting: wgpu::BindGroupLayout,
    composite: wgpu::BindGroupLayout,
    taa: wgpu::BindGroupLayout,
    tonemap: wgpu::BindGroupLayout,
    debug: wgpu::BindGroupLayout,
    overlay: wgpu::BindGroupLayout,
    sss: wgpu::BindGroupLayout,
    geometry_pipeline: wgpu::PipelineLayout,
    sss_pipeline: wgpu::PipelineLayout,
    shadow_pipeline: wgpu::PipelineLayout,
    composite_pipeline: wgpu::PipelineLayout,
    taa_pipeline: wgpu::PipelineLayout,
    tonemap_pipeline: wgpu::PipelineLayout,
    debug_pipeline: wgpu::PipelineLayout,
    overlay_pipeline: wgpu::PipelineLayout,
    pick_pipeline: wgpu::PipelineLayout,
}

/// The targets and bind groups of one configuration.
#[derive(Debug)]
struct Targets {
    graph: CompiledGraph<PassKind>,
    res: GraphResources,
    physical: Vec<TextureId>,
    views: Vec<wgpu::TextureView>,
    history: [TextureId; 2],
    history_views: [wgpu::TextureView; 2],
    display: TextureId,
    display_view: wgpu::TextureView,
    composite_bind: wgpu::BindGroup,
    /// The subsurface passes' inputs (horizontal, vertical) when they run.
    sss_bind: Option<[wgpu::BindGroup; 2]>,
    taa_bind: [wgpu::BindGroup; 2],
    tonemap_bind: [wgpu::BindGroup; 2],
    debug_bind: wgpu::BindGroup,
}

/// The group-1 bind groups of the bound scene: the shared streams, and one per deformed mesh
/// whose streams are its deformer's `[EVAL-05]`.
#[derive(Debug)]
struct SceneBinds {
    /// The scene buffers and stream generation the groups were made for.
    key: ([BufferId; 6], u64),
    shared: wgpu::BindGroup,
    per_mesh: Vec<Option<wgpu::BindGroup>>,
}

impl SceneBinds {
    /// The group a draw of mesh `mesh` binds.
    fn for_mesh(&self, mesh: u32) -> &wgpu::BindGroup {
        self.per_mesh.get(mesh as usize).and_then(Option::as_ref).unwrap_or(&self.shared)
    }
}

/// Binds the group-1 bind group of a draw of mesh `mesh` unless it is bound already.
fn bind_scene<'a>(
    rp: &mut wgpu::RenderPass<'_>,
    binds: &'a SceneBinds,
    bound: &mut Option<&'a wgpu::BindGroup>,
    mesh: u32,
) {
    let group = binds.for_mesh(mesh);
    if !bound.is_some_and(|b| std::ptr::eq(b, group)) {
        rp.set_bind_group(1, group, &[]);
        *bound = Some(group);
    }
}

/// The persistent per-frame state.
#[derive(Debug, Clone, Copy)]
struct FrameState {
    index: u32,
    parity: usize,
    accumulated: u32,
    prev_view_proj: Option<humanart_math::Mat4>,
    prev_jitter: Vec2,
}

/// The renderer. See the module documentation.
#[derive(Debug)]
pub struct Renderer {
    config: RenderConfig,
    cache: PipelineCache<Key>,
    layouts: Layouts,
    ring: UniformRing,
    profiler: GpuProfiler,
    ibl: Ibl,
    shadow_atlas: TextureId,
    shadow_layers: Vec<wgpu::TextureView>,
    lighting_bind: wgpu::BindGroup,
    preintegrated: TextureId,
    /// The eye and mouth table of the geometry pass (`crate::eye::FeatureTable`).
    features: BufferId,
    sss_kernel: BufferId,
    scene_binds: Option<SceneBinds>,
    overlay_buffer: BufferId,
    overlay_bind: wgpu::BindGroup,
    tonemap_buffer: BufferId,
    pick_target: TextureId,
    pick_depth: TextureId,
    pick_views: [wgpu::TextureView; 2],
    readback: ReadbackRing,
    linear_sampler: wgpu::Sampler,
    surface: SkinSurface,
    targets: Option<Targets>,
    state: FrameState,
    /// The hair passes, when hair is drawn (`crate::hair`, `[HR-01]`).
    hair: Option<crate::hair::HairRenderer>,
}

const fn texture_entry(
    binding: u32,
    sample_type: wgpu::TextureSampleType,
    dim: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture { sample_type, view_dimension: dim, multisampled: false },
        count: None,
    }
}

const FLOAT: wgpu::TextureSampleType = wgpu::TextureSampleType::Float { filterable: true };
const UINT: wgpu::TextureSampleType = wgpu::TextureSampleType::Uint;
const DEPTH: wgpu::TextureSampleType = wgpu::TextureSampleType::Depth;
const D2: wgpu::TextureViewDimension = wgpu::TextureViewDimension::D2;

fn sampler_entry(binding: u32, ty: wgpu::SamplerBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(ty),
        count: None,
    }
}

fn storage_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl Layouts {
    fn new(device: &wgpu::Device, ring: &wgpu::BindGroupLayout, surface: &wgpu::BindGroupLayout) -> Self {
        let bgl = |label, entries: &[wgpu::BindGroupLayoutEntry]| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries })
        };
        let scene = bgl("scene", &SCENE_LAYOUT);
        let lighting = bgl(
            "lighting",
            &[
                texture_entry(0, FLOAT, wgpu::TextureViewDimension::Cube),
                sampler_entry(1, wgpu::SamplerBindingType::Filtering),
                texture_entry(2, FLOAT, D2),
                storage_entry(3, wgpu::ShaderStages::FRAGMENT),
                texture_entry(4, DEPTH, wgpu::TextureViewDimension::D2Array),
                sampler_entry(5, wgpu::SamplerBindingType::Comparison),
                texture_entry(6, FLOAT, D2),
                // The eye and mouth table [EYE-01] [MTH-01] (crate::eye::FeatureTable).
                storage_entry(7, wgpu::ShaderStages::FRAGMENT),
            ],
        );
        let composite = bgl(
            "composite inputs",
            &[
                texture_entry(0, FLOAT, D2),
                texture_entry(1, FLOAT, D2),
                texture_entry(2, UINT, D2),
                texture_entry(3, DEPTH, D2),
            ],
        );
        let taa = bgl(
            "taa inputs",
            &[
                texture_entry(0, FLOAT, D2),
                texture_entry(1, FLOAT, D2),
                texture_entry(2, UINT, D2),
                texture_entry(3, DEPTH, D2),
                sampler_entry(4, wgpu::SamplerBindingType::Filtering),
            ],
        );
        let tonemap = bgl(
            "tonemap inputs",
            &[
                texture_entry(0, FLOAT, D2),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );
        let debug = bgl(
            "debug inputs",
            &[
                texture_entry(0, FLOAT, D2),
                texture_entry(1, FLOAT, D2),
                texture_entry(2, UINT, D2),
                texture_entry(3, DEPTH, D2),
                texture_entry(4, FLOAT, D2),
                texture_entry(5, FLOAT, D2),
            ],
        );
        let overlay = bgl("overlay primitives", &[storage_entry(0, wgpu::ShaderStages::VERTEX)]);
        let sss = bgl(
            "subsurface inputs",
            &[
                texture_entry(0, FLOAT, D2),
                texture_entry(1, FLOAT, D2),
                sampler_entry(2, wgpu::SamplerBindingType::Filtering),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );
        let pl = |label, groups: &[Option<&wgpu::BindGroupLayout>]| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: groups,
                immediate_size: 0,
            })
        };
        Self {
            geometry_pipeline: pl("geometry", &[Some(ring), Some(&scene), Some(&lighting), Some(surface)]),
            shadow_pipeline: pl("shadow", &[Some(ring), Some(&scene)]),
            composite_pipeline: pl("composite", &[Some(ring), Some(&composite), Some(&lighting)]),
            taa_pipeline: pl("taa", &[Some(ring), Some(&taa)]),
            tonemap_pipeline: pl("tonemap", &[Some(&tonemap)]),
            debug_pipeline: pl("debug", &[Some(ring), Some(&debug), Some(&lighting)]),
            overlay_pipeline: pl("overlay", &[Some(ring), Some(&scene), Some(&overlay)]),
            pick_pipeline: pl("pick", &[Some(ring), Some(&scene)]),
            sss_pipeline: pl("subsurface", &[Some(ring), Some(&sss)]),
            sss,
            scene,
            lighting,
            composite,
            taa,
            tonemap,
            debug,
            overlay,
        }
    }
}

fn raw_texture(resources: &Resources, id: TextureId) -> Result<&wgpu::Texture, RenderError> {
    Ok(resources.texture(id).ok_or(GpuError::Stale { what: "render texture" })?.raw())
}

fn raw_buffer(resources: &Resources, id: BufferId) -> Result<&wgpu::Buffer, RenderError> {
    Ok(resources.buffer(id).ok_or(GpuError::Stale { what: "render buffer" })?.raw())
}

/// One colour attachment.
fn color_attachment(view: &wgpu::TextureView, clear: Option<wgpu::Color>) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: clear.map_or(wgpu::LoadOp::Load, wgpu::LoadOp::Clear),
            store: wgpu::StoreOp::Store,
        },
    }
}

/// The tone mapper's index in `lib/color.wgsl`.
const fn tone_mapper_code(mapper: ToneMapper) -> u32 {
    match mapper {
        ToneMapper::PbrNeutral => 0,
        ToneMapper::AcesFitted => 1,
        ToneMapper::Clamp => 2,
    }
}

impl Renderer {
    /// Creates the renderer: layouts, pipelines of `config`, the IBL with a procedural studio
    /// environment, the shadow atlas, the uniform ring, the profiler and the readback ring.
    pub fn new(gpu: &Gpu, resources: &mut Resources, config: RenderConfig) -> Result<Self, RenderError> {
        config.validate(resources.limits())?;
        let device = resources.device().clone();
        let ring =
            UniformRing::new(resources, UniformRingDesc::new(64 * 1024, RING_BINDING_SIZE, "render frame constants"))?;
        let surface = SkinSurface::new(gpu)?;
        let layouts = Layouts::new(&device, ring.layout(), surface.layout());
        let profiler = GpuProfiler::new(gpu, resources, 2, 16)?;
        let mut ibl = Ibl::new(resources)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ibl build") });
        ibl.build(resources, &mut encoder, EnvironmentSource::Procedural(crate::light::ProceduralEnvironment::Studio))?;
        resources.queue().submit([encoder.finish()]);
        let size = config.shadow_map_size;
        let (shadow_atlas, _) = resources.create_texture(&TextureDesc::array(
            size,
            size,
            MAX_SHADOWED as u32,
            DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            Tag::shared(Category::RenderTargets, "shadow atlas"),
        ))?;
        let atlas = raw_texture(resources, shadow_atlas)?.clone();
        let shadow_layers = (0..MAX_SHADOWED as u32)
            .map(|layer| {
                atlas.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("shadow layer"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let atlas_view = atlas.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow atlas"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let compare = resources
            .sampler(&SamplerKey { compare: Some(wgpu::CompareFunction::GreaterEqual), ..SamplerKey::LINEAR_CLAMP })?
            .clone();
        let linear_sampler = resources.sampler(&SamplerKey::LINEAR_CLAMP)?.clone();
        let sh = raw_buffer(resources, ibl.sh_buffer())?.clone();
        // The skin tables `[SHD-02]` `[SHD-04]`, made once on the CPU (crate::sss).
        let lut_side = crate::sss::PREINTEGRATED_SIZE as u32;
        let (preintegrated, _) = resources.create_texture(&TextureDesc::d2(
            lut_side,
            lut_side,
            HDR_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            Tag::shared(Category::Textures, "pre-integrated skin table"),
        ))?;
        resources.write_texture_level(preintegrated, 0, 0, &crate::sss::preintegrated_bytes())?;
        let kernel = crate::sss::kernel_bytes();
        let (sss_kernel, _) = resources.create_buffer(&BufferDesc {
            size: kernel.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            tag: Tag::shared(Category::Uniforms, "subsurface kernel"),
        })?;
        resources.write_buffer(sss_kernel, 0, &kernel)?;
        let lut_view = raw_texture(resources, preintegrated)?.create_view(&wgpu::TextureViewDescriptor::default());
        let (features, _) = resources.create_buffer(&BufferDesc {
            size: crate::eye::FEATURE_TABLE_BYTES as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            tag: Tag::shared(Category::Other, "eye and mouth table"),
        })?;
        let features_buffer = raw_buffer(resources, features)?.clone();
        let lighting_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lighting"),
            layout: &layouts.lighting,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(ibl.specular_view()) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(ibl.sampler()) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(ibl.dfg_view()) },
                wgpu::BindGroupEntry { binding: 3, resource: sh.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&atlas_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&compare) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&lut_view) },
                wgpu::BindGroupEntry { binding: 7, resource: features_buffer.as_entire_binding() },
            ],
        });
        let (overlay_buffer, _) = resources.create_buffer(&BufferDesc {
            size: (MAX_OVERLAY_PRIMITIVES * std::mem::size_of::<OverlayPrimitive>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            tag: Tag::shared(Category::Other, "overlay primitives"),
        })?;
        let overlay_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay"),
            layout: &layouts.overlay,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: raw_buffer(resources, overlay_buffer)?.as_entire_binding(),
            }],
        });
        let (tonemap_buffer, _) = resources.create_buffer(&BufferDesc {
            size: std::mem::size_of::<TonemapUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            tag: Tag::shared(Category::Uniforms, "tonemap constants"),
        })?;
        let pick_usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC;
        let (pick_target, _) = resources.create_texture(&TextureDesc::d2(
            1,
            1,
            PICK_FORMAT,
            pick_usage,
            Tag::shared(Category::RenderTargets, "pick target"),
        ))?;
        let (pick_depth, _) = resources.create_texture(&TextureDesc::d2(
            1,
            1,
            DEPTH_FORMAT,
            pick_usage,
            Tag::shared(Category::RenderTargets, "pick depth"),
        ))?;
        let pick_views = [
            raw_texture(resources, pick_target)?.create_view(&wgpu::TextureViewDescriptor::default()),
            raw_texture(resources, pick_depth)?.create_view(&wgpu::TextureViewDescriptor::default()),
        ];
        let readback = ReadbackRing::new(resources, 8, 256, "pick readback")?;
        let mut renderer = Self {
            config,
            cache: PipelineCache::new(gpu),
            layouts,
            ring,
            profiler,
            ibl,
            shadow_atlas,
            shadow_layers,
            lighting_bind,
            preintegrated,
            features,
            sss_kernel,
            scene_binds: None,
            overlay_buffer,
            overlay_bind,
            tonemap_buffer,
            pick_target,
            pick_depth,
            pick_views,
            readback,
            linear_sampler,
            surface,
            targets: None,
            state: FrameState { index: 0, parity: 0, accumulated: 0, prev_view_proj: None, prev_jitter: Vec2::ZERO },
            hair: None,
        };
        renderer.configure(resources, config)?;
        renderer.cache.mark_warm();
        Ok(renderer)
    }

    /// The skin surface of the geometry pass (group 3, [`crate::surface`]).
    pub const fn skin_surface(&self) -> &SkinSurface {
        &self.surface
    }

    /// The skin surface, to bind a compositor, wrinkles or micro-detail, or change its settings.
    pub const fn skin_surface_mut(&mut self) -> &mut SkinSurface {
        &mut self.surface
    }

    /// The configuration in use.
    pub fn config(&self) -> &RenderConfig {
        &self.config
    }

    /// The compiled frame graph (passes, culled passes, aliasing).
    pub fn graph(&self) -> Option<&CompiledGraph<PassKind>> {
        self.targets.as_ref().map(|t| &t.graph)
    }

    /// Pipeline-cache counters: `late` creations after start-up must stay 0 in steady state.
    pub fn cache_stats(&self) -> &humanart_gpu::cache::CacheStats {
        self.cache.stats()
    }

    /// Uploads the eye and mouth table `[EYE-01]` `[MTH-01]` (`crate::eye`): the objects it maps
    /// shade as eyes, teeth, gums or tongue from the next frame on. A queue write of the table's
    /// fixed-size image, no allocation `[PERF-R1]`.
    pub fn set_features(
        &mut self,
        resources: &mut Resources,
        table: &crate::eye::FeatureTable,
    ) -> Result<(), RenderError> {
        resources.write_buffer(self.features, 0, table.as_bytes())?;
        self.state.accumulated = 0;
        Ok(())
    }

    /// Rebuilds the IBL from `source` (a rig change). One submission.
    pub fn set_environment(
        &mut self,
        resources: &mut Resources,
        source: EnvironmentSource<'_>,
    ) -> Result<(), RenderError> {
        let device = resources.device().clone();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ibl build") });
        self.ibl.build(resources, &mut encoder, source)?;
        resources.queue().submit([encoder.finish()]);
        self.state.accumulated = 0;
        Ok(())
    }

    /// The IBL (its textures, for tests and tools).
    pub fn ibl(&self) -> &Ibl {
        &self.ibl
    }

    /// Applies a new configuration: recompiles the graph and recreates targets and bind groups
    /// when anything changed, and creates the pipelines of the configuration (pre-warm).
    pub fn configure(&mut self, resources: &mut Resources, config: RenderConfig) -> Result<(), RenderError> {
        config.validate(resources.limits())?;
        if self.targets.is_some() && config == self.config {
            return Ok(());
        }
        if config.shadow_map_size != self.config.shadow_map_size && self.targets.is_some() {
            return Err(RenderError::InvalidParameter {
                what: "the shadow map size is fixed when the renderer is created",
            });
        }
        self.create_pipelines(resources, &config)?;
        if let Some(old) = self.targets.take() {
            for id in old.physical.iter().chain(&old.history).chain(std::iter::once(&old.display)) {
                let _ = resources.destroy_texture(*id);
            }
        }
        self.config = config;
        self.targets = Some(self.build_targets(resources)?);
        self.state.accumulated = 0;
        self.state.prev_view_proj = None;
        Ok(())
    }

    fn shader(&mut self, path: &'static str) -> Result<(), RenderError> {
        if self.cache.get_shader(&Key::Shader(path)).is_none() {
            let composed = humanart_shaders::compose(path, &[])?;
            self.cache.shader(Key::Shader(path), path, composed.source())?;
        }
        Ok(())
    }

    fn create_pipelines(&mut self, resources: &Resources, config: &RenderConfig) -> Result<(), RenderError> {
        for path in [
            "render/geometry.wgsl",
            "render/shadow.wgsl",
            "render/composite.wgsl",
            "render/sss.wgsl",
            "render/taa.wgsl",
            "post/tonemap.wgsl",
            "render/debug.wgsl",
            "render/overlay.wgsl",
            "render/pick.wgsl",
        ] {
            self.shader(path)?;
        }
        let _ = resources;
        let layouts = &self.layouts;
        let depth_state = |write: bool| wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(write),
            depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };
        let target = |format: wgpu::TextureFormat, blend: Option<wgpu::BlendState>| {
            Some(wgpu::ColorTargetState { format, blend, write_mask: wgpu::ColorWrites::ALL })
        };
        let primitive = wgpu::PrimitiveState { cull_mode: None, ..wgpu::PrimitiveState::default() };
        let make = |parts: &humanart_gpu::cache::Parts<'_, Key>,
                    label: &str,
                    path: &'static str,
                    layout: &wgpu::PipelineLayout,
                    vs: &str,
                    fs: Option<&str>,
                    targets: &[Option<wgpu::ColorTargetState>],
                    depth: Option<wgpu::DepthStencilState>,
                    constants: &[(&str, f64)]|
         -> Result<wgpu::RenderPipeline, GpuError> {
            let module = parts.shader(&Key::Shader(path))?;
            let options = wgpu::PipelineCompilationOptions { constants, zero_initialize_workgroup_memory: true };
            Ok(parts.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some(vs),
                    compilation_options: options.clone(),
                    buffers: &[],
                },
                primitive,
                depth_stencil: depth,
                multisample: wgpu::MultisampleState::default(),
                fragment: fs.map(|fs| wgpu::FragmentState {
                    module,
                    entry_point: Some(fs),
                    compilation_options: options,
                    targets,
                }),
                multiview_mask: None,
                cache: None,
            }))
        };
        let taps = config.quality.shadow_taps();
        let sss_mode = config.quality.sss_mode();
        self.cache.render_pipeline(Key::Geometry(taps, sss_mode), |p| {
            make(
                p,
                "geometry",
                "render/geometry.wgsl",
                &layouts.geometry_pipeline,
                "vs_geometry",
                Some("fs_geometry"),
                &[target(HDR_FORMAT, None), target(HDR_FORMAT, None), target(SURFACE_FORMAT, None)],
                Some(depth_state(true)),
                &[("SHADOW_TAPS", f64::from(taps)), ("SSS_MODE", f64::from(sss_mode))],
            )
        })?;
        for (horizontal, fs) in [(true, "fs_sss_h"), (false, "fs_sss_v")] {
            self.cache.render_pipeline(Key::Sss(horizontal), |p| {
                make(
                    p,
                    fs,
                    "render/sss.wgsl",
                    &layouts.sss_pipeline,
                    "vs_sss",
                    Some(fs),
                    &[target(HDR_FORMAT, None)],
                    None,
                    &[],
                )
            })?;
        }
        self.cache.render_pipeline(Key::Shadow, |p| {
            let mut depth = depth_state(true);
            // Reversed-Z: a negative bias moves stored depths away from the light.
            depth.bias = wgpu::DepthBiasState { constant: -4, slope_scale: -2.0, clamp: 0.0 };
            make(p, "shadow", "render/shadow.wgsl", &layouts.shadow_pipeline, "vs_shadow", None, &[], Some(depth), &[])
        })?;
        self.cache.render_pipeline(Key::Composite, |p| {
            make(
                p,
                "composite",
                "render/composite.wgsl",
                &layouts.composite_pipeline,
                "vs_composite",
                Some("fs_composite"),
                &[target(HDR_FORMAT, None)],
                None,
                &[],
            )
        })?;
        self.cache.render_pipeline(Key::Taa, |p| {
            make(
                p,
                "taa",
                "render/taa.wgsl",
                &layouts.taa_pipeline,
                "vs_taa",
                Some("fs_taa"),
                &[target(HDR_FORMAT, None)],
                None,
                &[],
            )
        })?;
        let mapper = tone_mapper_code(config.tone_mapper);
        self.cache.render_pipeline(Key::Tonemap(mapper), |p| {
            make(
                p,
                "tonemap",
                "post/tonemap.wgsl",
                &layouts.tonemap_pipeline,
                "vs_fullscreen",
                Some("fs_tonemap"),
                &[target(wgpu::TextureFormat::Rgba8Unorm, None)],
                None,
                &[("TONE_MAPPER", f64::from(mapper)), ("ENCODE_SRGB", 1.0)],
            )
        })?;
        self.cache.render_pipeline(Key::Debug, |p| {
            make(
                p,
                "debug",
                "render/debug.wgsl",
                &layouts.debug_pipeline,
                "vs_debug",
                Some("fs_debug"),
                &[target(wgpu::TextureFormat::Rgba8Unorm, None)],
                None,
                &[("ENCODE_SRGB", 1.0)],
            )
        })?;
        let blend = Some(wgpu::BlendState::ALPHA_BLENDING);
        for (key, vs) in
            [(Key::OverlayLines, "vs_lines"), (Key::OverlayWire, "vs_wire"), (Key::OverlaySelection, "vs_selection")]
        {
            self.cache.render_pipeline(key, |p| {
                make(
                    p,
                    vs,
                    "render/overlay.wgsl",
                    &layouts.overlay_pipeline,
                    vs,
                    Some("fs_overlay"),
                    &[target(wgpu::TextureFormat::Rgba8Unorm, blend)],
                    Some(depth_state(false)),
                    &[],
                )
            })?;
        }
        self.cache.render_pipeline(Key::Pick, |p| {
            make(
                p,
                "pick",
                "render/pick.wgsl",
                &layouts.pick_pipeline,
                "vs_pick",
                Some("fs_pick"),
                &[target(PICK_FORMAT, None)],
                Some(depth_state(true)),
                &[],
            )
        })?;
        Ok(())
    }

    fn build_targets(&self, resources: &mut Resources) -> Result<Targets, RenderError> {
        let c = &self.config;
        let (w, h) = (c.width, c.height);
        let rt = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC;
        let mut g: GraphBuilder<PassKind> = GraphBuilder::new();
        let shadow = g.import(
            "shadow atlas",
            TextureInfo {
                width: c.shadow_map_size,
                height: c.shadow_map_size,
                layers: MAX_SHADOWED as u32,
                format: DEPTH_FORMAT,
                usage: rt,
            },
        );
        let diffuse = g.texture("diffuse", TextureInfo::d2(w, h, HDR_FORMAT, rt));
        let specular = g.texture("specular", TextureInfo::d2(w, h, HDR_FORMAT, rt));
        let surface = g.texture("surface", TextureInfo::d2(w, h, SURFACE_FORMAT, rt));
        let depth = g.texture("depth", TextureInfo::d2(w, h, DEPTH_FORMAT, rt));
        let hdr = g.texture("hdr", TextureInfo::d2(w, h, HDR_FORMAT, rt));
        let history_in = g.import("history in", TextureInfo::d2(w, h, HDR_FORMAT, rt));
        let history_out = g.import("history out", TextureInfo::d2(w, h, HDR_FORMAT, rt));
        let display = g.import("display", TextureInfo::d2(w, h, wgpu::TextureFormat::Rgba8Unorm, rt));
        let pick = g.import("pick", TextureInfo::d2(1, 1, PICK_FORMAT, rt));
        g.pass("shadows", PassKind::Shadows).write(shadow);
        g.pass("geometry", PassKind::Geometry).read(shadow).write(diffuse).write(specular).write(surface).write(depth);
        let (sss_temp, diffuse_sss) = if c.quality.sss_mode() == 0 {
            let temp = g.texture("sss temp", TextureInfo::d2(w, h, HDR_FORMAT, rt));
            let out = g.texture("diffuse sss", TextureInfo::d2(w, h, HDR_FORMAT, rt));
            g.pass("sss_h", PassKind::SubsurfaceH).read(diffuse).read(specular).write(temp);
            g.pass("sss_v", PassKind::SubsurfaceV).read(temp).read(specular).write(out);
            (Some(temp), Some(out))
        } else {
            (None, None)
        };
        // The diffuse lighting after the skin diffusion.
        let lit = diffuse_sss.unwrap_or(diffuse);
        g.pass("composite", PassKind::Composite).read(lit).read(specular).read(surface).read(depth).write(hdr);
        if self.hair.is_some() {
            g.pass("hair", PassKind::Hair)
                .read(shadow)
                .read(depth)
                .read(surface)
                .write(hdr)
                .write(surface)
                .write(depth);
        }
        if c.taa {
            g.pass("taa", PassKind::Taa).read(hdr).read(history_in).read(surface).read(depth).write(history_out);
        }
        if c.debug == DebugView::Beauty {
            g.pass("tonemap", PassKind::Tonemap).read(if c.taa { history_out } else { hdr }).write(display);
            if c.taa {
                g.output(history_out);
            }
        } else {
            g.pass("debug", PassKind::Debug)
                .read(lit)
                .read(diffuse)
                .read(specular)
                .read(surface)
                .read(depth)
                .read(hdr)
                .read(shadow)
                .write(display);
        }
        if c.overlays {
            g.pass("overlay", PassKind::Overlay).read(depth).write(display);
        }
        if c.picking {
            g.pass("pick", PassKind::Pick).write(pick).side_effect();
        }
        g.output(display);
        let graph = g.compile()?;
        let mut physical = Vec::with_capacity(graph.physical_textures().len());
        let mut views = Vec::with_capacity(graph.physical_textures().len());
        for slot in graph.physical_textures() {
            let i = slot.info;
            let (id, _) = resources.create_texture(&TextureDesc::d2(
                i.width,
                i.height,
                i.format,
                i.usage,
                Tag::shared(Category::RenderTargets, "render graph target"),
            ))?;
            physical.push(id);
            views.push(raw_texture(resources, id)?.create_view(&wgpu::TextureViewDescriptor::default()));
        }
        let make =
            |resources: &mut Resources, format, purpose| -> Result<(TextureId, wgpu::TextureView), RenderError> {
                let (id, _) = resources.create_texture(&TextureDesc::d2(
                    w,
                    h,
                    format,
                    rt,
                    Tag::shared(Category::RenderTargets, purpose),
                ))?;
                let view = raw_texture(resources, id)?.create_view(&wgpu::TextureViewDescriptor::default());
                Ok((id, view))
            };
        let (h0, h0v) = make(resources, HDR_FORMAT, "taa history")?;
        let (h1, h1v) = make(resources, HDR_FORMAT, "taa history")?;
        let (display_id, display_view) = make(resources, wgpu::TextureFormat::Rgba8Unorm, "display")?;
        let res = GraphResources { diffuse, sss_temp, diffuse_sss, specular, surface, depth, hdr };
        // A view of a resource; an unused one (culled) gets the display view, never bound.
        let view_of = |r: ResourceId| graph.physical(r).and_then(|s| views.get(s)).unwrap_or(&display_view);
        let device = resources.device();
        let tv = wgpu::BindingResource::TextureView;
        let composite_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite inputs"),
            layout: &self.layouts.composite,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tv(view_of(lit)) },
                wgpu::BindGroupEntry { binding: 1, resource: tv(view_of(specular)) },
                wgpu::BindGroupEntry { binding: 2, resource: tv(view_of(surface)) },
                wgpu::BindGroupEntry { binding: 3, resource: tv(view_of(depth)) },
            ],
        });
        let sss_bind = match sss_temp {
            Some(temp) => {
                let kernel = raw_buffer(resources, self.sss_kernel)?;
                Some([diffuse, temp].map(|source| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("subsurface inputs"),
                        layout: &self.layouts.sss,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: tv(view_of(source)) },
                            wgpu::BindGroupEntry { binding: 1, resource: tv(view_of(specular)) },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                            },
                            wgpu::BindGroupEntry { binding: 3, resource: kernel.as_entire_binding() },
                        ],
                    })
                }))
            }
            None => None,
        };
        let history_views = [h0v, h1v];
        let taa_bind = [0, 1].map(|parity: usize| {
            // Parity p writes history[p] and reads history[1 - p].
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("taa inputs"),
                layout: &self.layouts.taa,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: tv(view_of(hdr)) },
                    wgpu::BindGroupEntry { binding: 1, resource: tv(&history_views[1 - parity]) },
                    wgpu::BindGroupEntry { binding: 2, resource: tv(view_of(surface)) },
                    wgpu::BindGroupEntry { binding: 3, resource: tv(view_of(depth)) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.linear_sampler) },
                ],
            })
        });
        let tonemap_buffer = raw_buffer(resources, self.tonemap_buffer)?;
        let tonemap_bind = [0, 1].map(|parity: usize| {
            let source = if c.taa { &history_views[parity] } else { view_of(hdr) };
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tonemap inputs"),
                layout: &self.layouts.tonemap,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: tv(source) },
                    wgpu::BindGroupEntry { binding: 1, resource: tonemap_buffer.as_entire_binding() },
                ],
            })
        });
        let debug_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("debug inputs"),
            layout: &self.layouts.debug,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tv(view_of(lit)) },
                wgpu::BindGroupEntry { binding: 1, resource: tv(view_of(specular)) },
                wgpu::BindGroupEntry { binding: 2, resource: tv(view_of(surface)) },
                wgpu::BindGroupEntry { binding: 3, resource: tv(view_of(depth)) },
                wgpu::BindGroupEntry { binding: 4, resource: tv(view_of(hdr)) },
                wgpu::BindGroupEntry { binding: 5, resource: tv(view_of(diffuse)) },
            ],
        });
        Ok(Targets {
            graph,
            res,
            physical,
            views,
            history: [h0, h1],
            history_views,
            display: display_id,
            display_view,
            composite_bind,
            sss_bind,
            taa_bind,
            tonemap_bind,
            debug_bind,
        })
    }

    /// The display texture (Rgba8Unorm, sRGB-encoded) of the last frame.
    pub fn display_texture(&self) -> Option<TextureId> {
        self.targets.as_ref().map(|t| t.display)
    }

    /// The resolved HDR image of the last frame (the TAA output; the composite without TAA),
    /// linear Rec.709 before exposure.
    pub fn resolved_texture(&self) -> Option<TextureId> {
        let t = self.targets.as_ref()?;
        if self.config.taa {
            t.history.get(1 - self.state.parity).copied()
        } else {
            t.graph.physical(t.res.hdr).and_then(|s| t.physical.get(s)).copied()
        }
    }

    /// The physical texture behind a transient resource of the last frame: `"diffuse"` (the
    /// diffuse lighting the composite uses, after the skin diffusion), `"specular"`, `"surface"`,
    /// `"depth"`, `"hdr"`, and `"irradiance"` (the geometry pass's diffuse before the skin
    /// diffusion; it outlives the frame only in a debug view, whose pass reads it, otherwise a
    /// later target may reuse its texture).
    pub fn target_texture(&self, name: &str) -> Option<TextureId> {
        let t = self.targets.as_ref()?;
        let r = match name {
            "diffuse" => t.res.diffuse_sss.unwrap_or(t.res.diffuse),
            "irradiance" => t.res.diffuse,
            "specular" => t.res.specular,
            "surface" => t.res.surface,
            "depth" => t.res.depth,
            "hdr" => t.res.hdr,
            _ => return None,
        };
        t.graph.physical(r).and_then(|s| t.physical.get(s)).copied()
    }

    /// The shadow atlas.
    pub fn shadow_atlas(&self) -> TextureId {
        self.shadow_atlas
    }

    /// The timings of the latest collected frame `[PERF-R6]`.
    pub fn timings(&self) -> &FrameReport {
        self.profiler.latest()
    }

    /// Call after the submission that holds the frame's commands.
    pub fn after_submit(&mut self) {
        self.profiler.after_submit();
        self.readback.after_submit();
    }

    /// Collects finished timestamp queries (after a device poll). Returns whether a new report
    /// arrived.
    pub fn collect_timings(&mut self) -> bool {
        self.profiler.collect()
    }

    /// Polls the picking readbacks without blocking; returns the newest finished hit.
    pub fn poll_pick(&mut self, device: &wgpu::Device) -> Result<Option<PickHit>, RenderError> {
        self.readback.poll(device)?;
        let mut hit = None;
        self.readback.drain_ready(|r| {
            let user = r.user();
            let pixel = [(user >> 32) as u32, user as u32];
            if let Some(row) = r.row(0)
                && let (Some(id), Some(depth)) = (row.get(0..4), row.get(4..8))
            {
                let id = u32::from_le_bytes([id[0], id[1], id[2], id[3]]);
                let depth = f32::from_le_bytes([depth[0], depth[1], depth[2], depth[3]]);
                hit = Some(PickHit { pixel, pick_id: id, depth });
            }
        });
        Ok(hit)
    }

    /// Records one frame into `encoder`. Call [`Self::after_submit`] after submitting it.
    pub fn render(
        &mut self,
        resources: &mut Resources,
        encoder: &mut wgpu::CommandEncoder,
        input: FrameInput<'_>,
    ) -> Result<(), RenderError> {
        let FrameInput { camera, rig, scene, overlay, wireframe, still, pick, reset_history } = input;
        camera.validate()?;
        rig.validate()?;
        if overlay.len() > MAX_OVERLAY_PRIMITIVES {
            return Err(RenderError::TooLarge {
                what: "overlay primitives",
                value: overlay.len() as u64,
                limit: MAX_OVERLAY_PRIMITIVES as u64,
            });
        }
        // The scene's bind groups, made when the scene's buffers or deformed meshes change.
        let key = (scene.bind_buffers(), scene.stream_generation());
        if self.scene_binds.as_ref().is_none_or(|b| b.key != key) {
            self.scene_binds = Some(self.make_scene_binds(resources, scene, key)?);
        }
        let config = self.config;
        let (w, h) = (config.width, config.height);
        let state = self.state;
        // Jitter and accumulation.
        let reset = reset_history || state.prev_view_proj.is_none();
        let (jitter, tile, accumulate, sample) = match still {
            Some(s) => {
                if s.tile.width != w || s.tile.height != h {
                    return Err(RenderError::InvalidParameter { what: "a still tile must have the target's size" });
                }
                (s.jitter, Some((s.full_width, s.full_height, s.tile)), true, s.sample)
            }
            None if config.taa => (
                taa_jitter_pixels(state.index % JITTER_PERIOD + 1, JITTER_PERIOD),
                None,
                false,
                if reset { 0 } else { state.accumulated },
            ),
            None => (Vec2::ZERO, None, false, 0),
        };
        let view = ViewMatrices::new(camera, w, h, jitter, tile);
        let prev_view_proj = if reset || still.is_some() {
            view.view_projection
        } else {
            state.prev_view_proj.unwrap_or(view.view_projection)
        };
        let exposure = camera.exposure.scale();
        let bounds = scene.bounds();
        // Frame constants (on the stack).
        let mut frame = FrameUniforms::ZERO;
        frame.view_proj = view.jittered.to_cols_array();
        frame.unjittered_view_proj = view.view_projection.to_cols_array();
        frame.prev_view_proj = prev_view_proj.to_cols_array();
        frame.inv_view_proj = view.inverse_view_projection.to_cols_array();
        let mut shadow_views = [ShadowViewUniforms { view_proj: [0.0; 16] }; MAX_SHADOWED];
        let mut shadow_count = 0;
        for (slot, light) in rig.lights.iter().enumerate() {
            let layer = if light.shadow.enabled && shadow_count < MAX_SHADOWED {
                let f = ShadowFrustum::fit(light, bounds.center, bounds.radius);
                frame.shadow_view_proj[shadow_count] = f.view_projection.to_cols_array();
                frame.shadow_params[shadow_count] = [f.near, f.far, f.half_extent, f.penumbra];
                frame.shadow_kind[shadow_count] = if f.perspective { 1.0 } else { 0.0 };
                shadow_views[shadow_count].view_proj = f.view_projection.to_cols_array();
                shadow_count += 1;
                Some(shadow_count - 1)
            } else {
                None
            };
            if let Some(gpu_light) = frame.lights.get_mut(slot) {
                *gpu_light = GpuLight::new(light, layer);
            }
        }
        frame.eye_exposure = [view.eye.x, view.eye.y, view.eye.z, exposure];
        frame.viewport = [w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32];
        frame.tile = match tile {
            Some((fw, fh, rect)) => [rect.x as f32, rect.y as f32, fw as f32, fh as f32],
            None => [0.0, 0.0, w as f32, h as f32],
        };
        frame.jitter = [jitter.x, jitter.y, state.prev_jitter.x, state.prev_jitter.y];
        frame.environment =
            [rig.environment_nits, rig.environment_rotation, SPECULAR_MIPS as f32, rig.lights.len() as f32];
        frame.background = match rig.background {
            Background::Gray(g) => {
                let v = g.max(0.0) / exposure.max(1e-12);
                [v, v, v, 0.0]
            }
            Background::Environment => [0.0, 0.0, 0.0, 1.0],
        };
        frame.taa = [
            if config.quality == Quality::Interactive { 0.2 } else { 0.1 },
            if accumulate { 1.0 } else { 0.0 },
            sample as f32,
            match still {
                Some(s) if s.sample == 0 => 1.0,
                None if reset => 1.0,
                Some(_) | None => 0.0,
            },
        ];
        // The noise seed: the frame index, or for stills the sample index, so every tile of a
        // still has the noise of the untiled render [QUAL-04].
        let seed = still.map_or(state.index, |s| s.sample);
        frame.counters = [seed, config.debug.code(), config.shadow_map_size, 0];
        self.ring.begin_frame();
        self.profiler.begin_frame();
        let frame_offset = self.ring.push(&frame)?;
        let mut shadow_offsets = [0_u32; MAX_SHADOWED];
        for (offset, view) in shadow_offsets.iter_mut().zip(&shadow_views).take(shadow_count) {
            *offset = self.ring.push(view)?;
        }
        // The picking frame: a 1x1 tile at the pixel, unjittered.
        let pick_offset = match pick {
            Some([px, py]) if config.picking && px < w && py < h => {
                let rect = TileRect { x: px as i32, y: py as i32, width: 1, height: 1 };
                let full = match tile {
                    Some((fw, fh, t)) => {
                        (fw, fh, TileRect { x: t.x + px as i32, y: t.y + py as i32, width: 1, height: 1 })
                    }
                    None => (w, h, rect),
                };
                let pick_view = ViewMatrices::new(camera, 1, 1, Vec2::ZERO, Some(full));
                let mut pick_frame = frame;
                pick_frame.view_proj = pick_view.view_projection.to_cols_array();
                pick_frame.unjittered_view_proj = pick_frame.view_proj;
                Some((self.ring.push(&pick_frame)?, [px, py]))
            }
            _ => None,
        };
        self.ring.finish_frame(resources)?;
        resources.write_buffer(self.tonemap_buffer, 0, bytes_of(&TonemapUniforms { exposure, pad: [0.0; 3] }))?;
        if !overlay.is_empty() {
            resources.write_buffer(self.overlay_buffer, 0, slice_as_bytes(overlay))?;
        }
        scene.begin_frame(resources, encoder)?;

        let Some(targets) = self.targets.as_ref() else {
            return Err(RenderError::InvalidParameter { what: "the renderer is not configured" });
        };
        let Some(binds) = self.scene_binds.as_ref() else {
            return Err(RenderError::InvalidParameter { what: "no scene bound" });
        };
        let parity = state.parity;
        if let Some(hair) = self.hair.as_mut().filter(|_| targets.graph.is_live("hair")) {
            let target = |r: ResourceId| targets.graph.physical(r).and_then(|s| targets.views.get(s));
            let stale = || RenderError::Gpu(GpuError::Stale { what: "hair input" });
            let ring = raw_buffer(resources, self.ring.buffer())?.clone();
            let sh = raw_buffer(resources, self.ibl.sh_buffer())?.clone();
            let atlas = resources.texture(self.shadow_atlas).ok_or_else(stale)?.view().clone();
            let frame = crate::hair::HairFrame {
                width: w,
                height: h,
                view: &view,
                lights: &rig.lights,
                ring: &ring,
                depth: target(targets.res.depth).ok_or_else(stale)?,
                surface: target(targets.res.surface).ok_or_else(stale)?,
                shadow_atlas: &atlas,
                irradiance_sh: &sh,
                reset_history: reset,
            };
            hair.prepare(resources, &frame)?;
        }
        let index_buffer = raw_buffer(resources, scene.index_buffer())?;
        let res = targets.res;
        let view_of = |r: ResourceId| targets.graph.physical(r).and_then(|s| targets.views.get(s));
        let missing = || RenderError::Gpu(GpuError::Stale { what: "render graph target" });
        let pipeline = |key: Key| {
            self.cache.get_render_pipeline(&key).ok_or(RenderError::Gpu(GpuError::Stale { what: "render pipeline" }))
        };
        let ring_bind = self.ring.bind_group();
        let draws = scene.draws();
        for pass in targets.graph.passes() {
            match pass.payload {
                PassKind::Shadows => {
                    let p = pipeline(Key::Shadow)?;
                    for (layer, view) in self.shadow_layers.iter().enumerate() {
                        let timestamps = if layer == 0 { self.profiler.render_pass_writes("shadows") } else { None };
                        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("shadows"),
                            color_attachments: &[],
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view,
                                depth_ops: Some(wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(0.0),
                                    store: wgpu::StoreOp::Store,
                                }),
                                stencil_ops: None,
                            }),
                            timestamp_writes: timestamps,
                            ..Default::default()
                        });
                        if layer >= shadow_count {
                            continue;
                        }
                        rp.set_pipeline(p);
                        rp.set_bind_group(0, ring_bind, &[shadow_offsets[layer]]);
                        rp.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        let mut bound = None;
                        for d in draws.iter().filter(|d| d.casts_shadow && d.index_count > 0) {
                            bind_scene(&mut rp, binds, &mut bound, d.mesh);
                            rp.draw_indexed(
                                d.first_index..d.first_index + d.index_count,
                                d.base_vertex,
                                d.object..d.object + 1,
                            );
                        }
                    }
                }
                PassKind::Geometry => {
                    let (Some(dv), Some(sv), Some(fv), Some(zv)) =
                        (view_of(res.diffuse), view_of(res.specular), view_of(res.surface), view_of(res.depth))
                    else {
                        return Err(missing());
                    };
                    let black = Some(wgpu::Color::TRANSPARENT);
                    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("geometry"),
                        color_attachments: &[
                            Some(color_attachment(dv, black)),
                            Some(color_attachment(sv, black)),
                            Some(color_attachment(fv, black)),
                        ],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: zv,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(0.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: self.profiler.render_pass_writes("geometry"),
                        ..Default::default()
                    });
                    rp.set_pipeline(pipeline(Key::Geometry(config.quality.shadow_taps(), config.quality.sss_mode()))?);
                    rp.set_bind_group(0, ring_bind, &[frame_offset]);
                    rp.set_bind_group(2, &self.lighting_bind, &[]);
                    rp.set_bind_group(3, self.surface.bind_group(), &[]);
                    rp.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    let mut bound = None;
                    for d in draws.iter().filter(|d| d.index_count > 0) {
                        bind_scene(&mut rp, binds, &mut bound, d.mesh);
                        rp.draw_indexed(
                            d.first_index..d.first_index + d.index_count,
                            d.base_vertex,
                            d.object..d.object + 1,
                        );
                    }
                }
                PassKind::SubsurfaceH | PassKind::SubsurfaceV => {
                    let horizontal = pass.payload == PassKind::SubsurfaceH;
                    let (out, label) = if horizontal { (res.sss_temp, "sss_h") } else { (res.diffuse_sss, "sss_v") };
                    let ov = out.and_then(view_of).ok_or_else(missing)?;
                    let bind = targets.sss_bind.as_ref().ok_or_else(missing)?;
                    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some(label),
                        color_attachments: &[Some(color_attachment(ov, None))],
                        timestamp_writes: self.profiler.render_pass_writes(label),
                        ..Default::default()
                    });
                    rp.set_pipeline(pipeline(Key::Sss(horizontal))?);
                    rp.set_bind_group(0, ring_bind, &[frame_offset]);
                    rp.set_bind_group(1, &bind[usize::from(!horizontal)], &[]);
                    rp.draw(0..3, 0..1);
                }
                PassKind::Composite => {
                    let hv = view_of(res.hdr).ok_or_else(missing)?;
                    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("composite"),
                        color_attachments: &[Some(color_attachment(hv, None))],
                        timestamp_writes: self.profiler.render_pass_writes("composite"),
                        ..Default::default()
                    });
                    rp.set_pipeline(pipeline(Key::Composite)?);
                    rp.set_bind_group(0, ring_bind, &[frame_offset]);
                    rp.set_bind_group(1, &targets.composite_bind, &[]);
                    rp.set_bind_group(2, &self.lighting_bind, &[]);
                    rp.draw(0..3, 0..1);
                }
                PassKind::Hair => {
                    if let Some(hair) = self.hair.as_mut() {
                        let hair_targets = crate::hair::HairTargets {
                            hdr: view_of(res.hdr).ok_or_else(missing)?,
                            depth: view_of(res.depth).ok_or_else(missing)?,
                            surface: view_of(res.surface).ok_or_else(missing)?,
                        };
                        hair.encode(encoder, Some(&mut self.profiler), frame_offset, &hair_targets)?;
                    }
                }
                PassKind::Taa => {
                    let out = targets.history_views.get(parity).ok_or_else(missing)?;
                    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("taa"),
                        color_attachments: &[Some(color_attachment(out, None))],
                        timestamp_writes: self.profiler.render_pass_writes("taa"),
                        ..Default::default()
                    });
                    rp.set_pipeline(pipeline(Key::Taa)?);
                    rp.set_bind_group(0, ring_bind, &[frame_offset]);
                    rp.set_bind_group(1, targets.taa_bind.get(parity).ok_or_else(missing)?, &[]);
                    rp.draw(0..3, 0..1);
                }
                PassKind::Tonemap => {
                    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("tonemap"),
                        color_attachments: &[Some(color_attachment(&targets.display_view, None))],
                        timestamp_writes: self.profiler.render_pass_writes("tonemap"),
                        ..Default::default()
                    });
                    rp.set_pipeline(pipeline(Key::Tonemap(tone_mapper_code(config.tone_mapper)))?);
                    rp.set_bind_group(0, targets.tonemap_bind.get(parity).ok_or_else(missing)?, &[]);
                    rp.draw(0..3, 0..1);
                }
                PassKind::Debug => {
                    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("debug"),
                        color_attachments: &[Some(color_attachment(&targets.display_view, None))],
                        timestamp_writes: self.profiler.render_pass_writes("debug"),
                        ..Default::default()
                    });
                    rp.set_pipeline(pipeline(Key::Debug)?);
                    rp.set_bind_group(0, ring_bind, &[frame_offset]);
                    rp.set_bind_group(1, &targets.debug_bind, &[]);
                    rp.set_bind_group(2, &self.lighting_bind, &[]);
                    rp.draw(0..3, 0..1);
                }
                PassKind::Overlay => {
                    let zv = view_of(res.depth).ok_or_else(missing)?;
                    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("overlay"),
                        color_attachments: &[Some(color_attachment(&targets.display_view, None))],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: zv,
                            depth_ops: None,
                            stencil_ops: None,
                        }),
                        timestamp_writes: self.profiler.render_pass_writes("overlay"),
                        ..Default::default()
                    });
                    rp.set_bind_group(0, ring_bind, &[frame_offset]);
                    rp.set_bind_group(1, &binds.shared, &[]);
                    rp.set_bind_group(2, &self.overlay_bind, &[]);
                    let mut bound = Some(&binds.shared);
                    if wireframe {
                        rp.set_pipeline(pipeline(Key::OverlayWire)?);
                        for d in draws.iter().filter(|d| d.index_count > 0) {
                            bind_scene(&mut rp, binds, &mut bound, d.mesh);
                            rp.draw(0..d.index_count, d.object..d.object + 1);
                        }
                    }
                    if draws.iter().any(|d| d.selected) {
                        rp.set_pipeline(pipeline(Key::OverlaySelection)?);
                        for d in draws.iter().filter(|d| d.selected && d.index_count > 0) {
                            bind_scene(&mut rp, binds, &mut bound, d.mesh);
                            rp.draw(0..d.index_count, d.object..d.object + 1);
                        }
                    }
                    if bound.is_some_and(|b| !std::ptr::eq(b, std::ptr::from_ref(&binds.shared))) {
                        rp.set_bind_group(1, &binds.shared, &[]);
                    }
                    if !overlay.is_empty() {
                        rp.set_pipeline(pipeline(Key::OverlayLines)?);
                        rp.draw(0..overlay.len() as u32 * 6, 0..1);
                    }
                }
                PassKind::Pick => {
                    let Some((offset, [px, py])) = pick_offset else { continue };
                    {
                        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("pick"),
                            color_attachments: &[Some(color_attachment(
                                &self.pick_views[0],
                                Some(wgpu::Color::TRANSPARENT),
                            ))],
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: &self.pick_views[1],
                                depth_ops: Some(wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(0.0),
                                    store: wgpu::StoreOp::Store,
                                }),
                                stencil_ops: None,
                            }),
                            timestamp_writes: self.profiler.render_pass_writes("pick"),
                            ..Default::default()
                        });
                        rp.set_pipeline(pipeline(Key::Pick)?);
                        rp.set_bind_group(0, ring_bind, &[offset]);
                        rp.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        let mut bound = None;
                        for d in draws.iter().filter(|d| d.index_count > 0) {
                            bind_scene(&mut rp, binds, &mut bound, d.mesh);
                            rp.draw_indexed(
                                d.first_index..d.first_index + d.index_count,
                                d.base_vertex,
                                d.object..d.object + 1,
                            );
                        }
                    }
                    let texture = raw_texture(resources, self.pick_target)?;
                    let user = (u64::from(px) << 32) | u64::from(py);
                    // A full ring drops the request: picking is retried on the next frame.
                    let _ = self.readback.copy_texture(encoder, texture, 0, [0, 0, 0], [1, 1, 1], user);
                }
            }
        }
        self.profiler.end_frame(encoder);
        self.state = FrameState {
            index: state.index.wrapping_add(1),
            parity: if config.taa { 1 - parity } else { parity },
            accumulated: if reset { 1 } else { state.accumulated.saturating_add(1) },
            prev_view_proj: Some(view.view_projection),
            prev_jitter: jitter,
        };
        Ok(())
    }

    /// The group-1 bind groups of `scene`: the shared streams and one per deformed mesh (its
    /// deformer's positions, previous positions and normals, its own uvs, the shared object table
    /// and indices). Made only when the scene's buffers or its deformed meshes change.
    fn make_scene_binds(
        &self,
        resources: &Resources,
        scene: &GpuScene,
        key: ([BufferId; 6], u64),
    ) -> Result<SceneBinds, RenderError> {
        let [positions, prev_positions, normals, uvs, objects, indices] = key.0;
        let raw = |id| raw_buffer(resources, id);
        let (objects, indices) = (raw(objects)?, raw(indices)?);
        let device = resources.device();
        let make = |label: &str, streams: [&wgpu::Buffer; 4]| {
            let [p, q, n, u] = streams;
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.layouts.scene,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: p.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: q.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: n.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: u.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: objects.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: indices.as_entire_binding() },
                ],
            })
        };
        let shared = make("scene", [raw(positions)?, raw(prev_positions)?, raw(normals)?, raw(uvs)?]);
        let mesh_count = scene.ranges().len();
        let mut per_mesh = Vec::with_capacity(mesh_count);
        for mesh in 0..mesh_count {
            per_mesh.push(match scene.external_streams(mesh) {
                Some(e) => Some(make("deformed mesh", [&e.positions, &e.prev_positions, &e.normals, raw(e.uvs)?])),
                None => None,
            });
        }
        Ok(SceneBinds { key, shared, per_mesh })
    }

    /// The hair passes, when set (`crate::hair`).
    pub fn hair(&self) -> Option<&crate::hair::HairRenderer> {
        self.hair.as_ref()
    }

    /// The hair passes, when set, to change their grooms and settings.
    pub fn hair_mut(&mut self) -> Option<&mut crate::hair::HairRenderer> {
        self.hair.as_mut()
    }

    /// Sets (or removes) the hair passes `[HR-01]`; the frame graph gains (or loses) the `hair`
    /// pass between the composite and the TAA. Returns the hair renderer it replaces.
    pub fn set_hair(
        &mut self,
        resources: &mut Resources,
        hair: Option<crate::hair::HairRenderer>,
    ) -> Result<Option<crate::hair::HairRenderer>, RenderError> {
        let old = std::mem::replace(&mut self.hair, hair);
        if old.is_some() != self.hair.is_some()
            && let Some(targets) = self.targets.take()
        {
            for id in targets.physical.iter().chain(&targets.history).chain(std::iter::once(&targets.display)) {
                let _ = resources.destroy_texture(*id);
            }
            self.targets = Some(self.build_targets(resources)?);
            self.state.accumulated = 0;
            self.state.prev_view_proj = None;
        }
        Ok(old)
    }

    /// Frees every texture and buffer the renderer made.
    pub fn destroy(mut self, resources: &mut Resources) {
        if let Some(hair) = self.hair.take() {
            hair.destroy(resources);
        }
        if let Some(t) = self.targets.take() {
            for id in t.physical.iter().chain(&t.history).chain(std::iter::once(&t.display)) {
                let _ = resources.destroy_texture(*id);
            }
        }
        for id in [self.shadow_atlas, self.pick_target, self.pick_depth, self.preintegrated] {
            let _ = resources.destroy_texture(id);
        }
        for id in [self.overlay_buffer, self.tonemap_buffer, self.sss_kernel, self.features] {
            let _ = resources.destroy_buffer(id);
        }
        self.ibl.destroy(resources);
        self.readback.destroy(resources);
        self.profiler.destroy(resources);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_view_codes_match_the_shader() {
        let text = humanart_shaders::manifest::source("render/debug.wgsl").unwrap();
        for view in DebugView::ALL.iter().skip(1) {
            let upper = match view {
                DebugView::MaterialClass => "MATERIAL".to_owned(),
                DebugView::MotionVectors => "MOTION".to_owned(),
                DebugView::FalseColor => "FALSE_COLOR".to_owned(),
                other => other.name().to_uppercase(),
            };
            let line = format!("const VIEW_{upper}: u32 = {}u;", view.code());
            assert!(text.contains(&line), "{line}");
        }
    }

    #[test]
    fn configs_are_checked() {
        let limits = wgpu::Limits::default();
        assert!(RenderConfig::new(0, 10).validate(&limits).is_err());
        assert!(RenderConfig::new(9000, 10).validate(&limits).is_err());
        assert!(RenderConfig { shadow_map_size: 16, ..RenderConfig::new(64, 64) }.validate(&limits).is_err());
        assert!(RenderConfig::new(1920, 1080).validate(&limits).is_ok());
    }

    #[test]
    fn quality_tiers_choose_listed_variants() {
        let geometry = humanart_shaders::manifest::shader("render/geometry.wgsl").unwrap();
        let (taps, sss) = (geometry.overrides[0].values, geometry.overrides[1].values);
        for q in [Quality::Interactive, Quality::Review, Quality::Still] {
            assert!(taps.contains(&f64::from(q.shadow_taps())), "{q:?} uses an unvalidated variant");
            assert!(sss.contains(&f64::from(q.sss_mode())), "{q:?} uses an unvalidated variant");
        }
    }
}
