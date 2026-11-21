//! [`HairRenderer`]: the hair passes of a frame `[HR-01]`..`[HR-05]`, run by the core renderer
//! between the lighting composite and the TAA (`PassKind::Hair`).
//!
//! Per frame, in one command encoder:
//!
//! 1. **Deep opacity maps** `[HR-03]` (`hair/dom.wgsl`): for the (at most two) lights that light
//!    the hair most, an orthographic frame around the grooms' bounds; the strands (subsampled to at
//!    most [`DOM_STRANDS`](super::view::DOM_STRANDS) with coverage compensation) are drawn as lines,
//!    first depth only (the first fibre per texel), then additively into four opacity layers.
//! 2. **Visibility** `[HR-01]`: a coarse fibre density grid (`voxelize`), then either the compute
//!    software rasteriser (`hair/raster.wgsl`: coverage sum and nearest depth, then the nearest
//!    fibre's record) or the hardware ribbon fallback (`hair/ribbon.wgsl`) into the same
//!    per-pixel visibility buffer. No sorting: the coverage sum is order-independent and the
//!    record is the nearest fibre's.
//! 3. **Resolve** `[HR-02]` `[HR-03]` (`hair/resolve.wgsl`): the visibility buffer is shaded
//!    (dual scattering, deep opacity, shadow maps, environment with density occlusion) and
//!    composited over the HDR target with alpha `1 - (1 - c0) exp(-(coverage - c0))` (`c0` the
//!    nearest fibre's own coverage, kept in its record); the skin behind and around
//!    the hair is darkened by the hair's shadows and the scalp occlusion.
//! 4. **Motion** `[HR-05]` (`hair/motion.wgsl`): where hair covers half a pixel, its depth and
//!    motion replace the G-buffer's, so the TAA reprojects hair with the strands' own motion.
//!
//! The level of detail `[HR-04]` is chosen per groom per frame on the CPU ([`GroomLod`]): far
//! hair is thinned (hash pruning) with coverage compensation and its segments joined.
//!
//! Pipelines and layouts are built once in [`HairRenderer::new`] `[PERF-R4]`; per-groom buffers
//! and bind groups when the grooms change ([`HairRenderer::set_grooms`]); the screen buffers and
//! the frame bind groups when the target size or the renderer's targets change. A steady frame
//! writes two small uniform blocks per groom and allocates nothing `[PERF-R1]`.

use humanart_gpu::cache::PipelineCache;
use humanart_gpu::pod::bytes_of;
use humanart_gpu::profiler::GpuProfiler;
use humanart_gpu::resources::{BufferDesc, BufferId, Category, Resources, SamplerKey, Tag, TextureDesc, TextureId};
use humanart_gpu::{Gpu, GpuError};
use humanart_math::Vec3;
use humanart_model::hair::{GroomParams, HairColor};

use super::HairError;
use super::bsdf::FibreParams;
use super::luts::HairLuts;
use super::view::{
    COVERAGE_ONE, COVERAGE_SATURATION, DOM_LAYERS, DomFrame, DrawBlock, FLAG_CULL, FLAG_DUAL, FLAG_OCCLUSION,
    FLAG_SKIN_SHADOW, Grid, GroomLod, LOD_MIN_KEEP, LodInput, MAX_DOMS, MAX_GROOMS, TILE, VOXEL_TAU_FACTOR, ViewBlock,
    dom_lights,
};
use crate::camera::ViewMatrices;
use crate::groom::GpuGroom;
use crate::light::Light;
use crate::uniforms::{FrameUniforms, RING_BINDING_SIZE};

/// Format of the HDR target the hair is composited into.
const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Format of the renderer's surface target (motion in `w`).
const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;
/// Reversed-Z depth.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Format of the deep opacity layers.
const OPACITY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Format of the ribbon passes' dummy colour target.
const RIBBON_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
/// Bytes of one visibility record.
const RECORD_BYTES: u64 = 16;

/// Which rasteriser fills the visibility buffer `[HR-01]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HairPath {
    /// The compute software rasteriser (the default on every tier, see the STATUS / ADR).
    Compute,
    /// Hardware ribbons (two triangles per segment): the fallback without the compute walk.
    Ribbon,
}

/// The hair renderer's settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HairSettings {
    /// The rasteriser.
    pub path: HairPath,
    /// Multiple scattering (dual scattering, `[HR-02]`); single scattering times the deep opacity
    /// transmittance without.
    pub dual_scattering: bool,
    /// Ambient occlusion from the fibre density grid (the hair's and the scalp's).
    pub occlusion: bool,
    /// Hair shadows on the skin (deep opacity maps and density grid, `[HR-03]`).
    pub skin_shadow: bool,
    /// Side of each deep opacity map (256..=4096).
    pub dom_size: u32,
    /// Cells per side of the fibre density grid (8..=128).
    pub voxels: u32,
    /// Fixed level of detail: draw every strand and segment whatever the distance, and put every
    /// strand in the deep opacity maps (for reference renders of the far field, `[HR-04]`, and
    /// order-invariance checks: the maps otherwise subsample the strands by index).
    pub full_detail: bool,
    /// Split the compute raster's long segment walks over several threads ([`GroomLod::split`]);
    /// off draws one thread per segment (the reference for the close-range timings, `[HR-04]`).
    /// The image is the same either way.
    pub split_walks: bool,
    /// Skip, in the compute raster, segments hidden behind the opaque scene (hair behind the
    /// head): a pass keeps the farthest scene depth of every 16-pixel tile, and a segment behind
    /// it in every tile it touches is not walked. The image is the same either way.
    pub cull_hidden: bool,
}

impl Default for HairSettings {
    fn default() -> Self {
        Self {
            path: HairPath::Compute,
            dual_scattering: true,
            occlusion: true,
            skin_shadow: true,
            dom_size: 1024,
            voxels: 64,
            full_detail: false,
            split_walks: true,
            cull_hidden: true,
        }
    }
}

impl HairSettings {
    fn check(&self) -> Result<(), HairError> {
        if !(256..=4096).contains(&self.dom_size) {
            return Err(HairError::Invalid { what: "deep opacity map size outside 256..=4096" });
        }
        if !(8..=128).contains(&self.voxels) {
            return Err(HairError::Invalid { what: "density grid size outside 8..=128" });
        }
        Ok(())
    }
}

/// One groom's strands on the GPU, as the hair renderer reads them.
#[derive(Debug, Clone)]
pub struct HairGroom {
    /// World position and diameter per point (vec4), strand after strand.
    pub points: wgpu::Buffer,
    /// Two `vec4<u32>` per strand; `[1].x` holds the colour id (low 16 bits) and grey (bits 16..24).
    pub attrs: wgpu::Buffer,
    /// Strands.
    pub strands: u32,
    /// Points per strand (2..=32).
    pub points_per_strand: u32,
    /// The groom's colour (melanin, variation, dye).
    pub color: HairColor,
    /// Centre of a sphere around every point, metres.
    pub centre: Vec3,
    /// Its radius.
    pub radius: f32,
    /// Mean strand diameter, metres.
    pub diameter: f32,
    /// Mean strand length, metres.
    pub length: f32,
    /// The least fraction of the strands the far-field thinning keeps (`LodInput::min_keep`):
    /// [`LOD_MIN_KEEP`] for dense scalp hair, 1 for sparse grooms (fuzz, brows, lashes).
    pub min_keep: f32,
}

impl HairGroom {
    /// The attached strands of `groom` (generated with `params`), inside the sphere (`centre`,
    /// `radius`), with strands of mean length `length`.
    pub fn from_gpu(groom: &GpuGroom, params: &GroomParams, centre: Vec3, radius: f32, length: f32) -> Self {
        let out = groom.outputs();
        Self {
            points: out.world.clone(),
            attrs: out.attrs.clone(),
            strands: groom.strand_count(),
            points_per_strand: groom.capacity().points,
            color: HairColor::from_params(params),
            centre,
            radius,
            diameter: params.thickness * (1.0 + params.tip_ratio) * 0.5,
            length,
            min_keep: LOD_MIN_KEEP,
        }
    }

    /// The same groom with the thinning floor `min_keep` (see [`HairGroom::min_keep`]).
    #[must_use]
    pub fn with_min_keep(self, min_keep: f32) -> Self {
        Self { min_keep, ..self }
    }

    fn check(&self) -> Result<(), HairError> {
        if !(2..=32).contains(&self.points_per_strand) {
            return Err(HairError::Invalid { what: "a groom needs 2..=32 points per strand" });
        }
        let points = u64::from(self.strands) * u64::from(self.points_per_strand) * 16;
        if self.points.size() < points {
            return Err(HairError::Invalid { what: "the groom's point buffer is smaller than its strands" });
        }
        if self.attrs.size() < u64::from(self.strands) * 32 {
            return Err(HairError::Invalid { what: "the groom's attribute buffer is smaller than its strands" });
        }
        let finite = self.centre.x.is_finite() && self.centre.y.is_finite() && self.centre.z.is_finite();
        if !finite || self.radius <= 0.0 || !self.radius.is_finite() {
            return Err(HairError::Invalid { what: "a groom's bounds must be a finite sphere" });
        }
        if !(self.diameter >= 0.0 && self.diameter.is_finite() && self.length >= 0.0 && self.length.is_finite()) {
            return Err(HairError::Invalid { what: "a groom's diameter and length must be finite" });
        }
        Ok(())
    }
}

/// What the hair passes read from the core renderer in one frame.
#[derive(Debug, Clone, Copy)]
pub struct HairFrame<'a> {
    /// Target width.
    pub width: u32,
    /// Target height.
    pub height: u32,
    /// The camera's matrices of this frame.
    pub view: &'a ViewMatrices,
    /// The lights (slots of the frame's light array).
    pub lights: &'a [Light],
    /// The uniform ring holding the frame constants (`Frame`, read at a dynamic offset).
    pub ring: &'a wgpu::Buffer,
    /// The scene depth target (reversed-Z).
    pub depth: &'a wgpu::TextureView,
    /// The surface target (normal, ..., motion).
    pub surface: &'a wgpu::TextureView,
    /// The shadow atlas, viewed as a 2D array.
    pub shadow_atlas: &'a wgpu::TextureView,
    /// The environment's L2 irradiance (nine vec4).
    pub irradiance_sh: &'a wgpu::Buffer,
    /// Forget the previous strand positions (camera cut, new grooms).
    pub reset_history: bool,
}

/// The attachments of the hair passes in one frame.
#[derive(Debug, Clone, Copy)]
pub struct HairTargets<'a> {
    /// The HDR target the hair is composited into.
    pub hdr: &'a wgpu::TextureView,
    /// The scene depth (written where hair covers the pixel).
    pub depth: &'a wgpu::TextureView,
    /// The surface target (its motion written where hair covers the pixel).
    pub surface: &'a wgpu::TextureView,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Key {
    Raster,
    Dom,
    Resolve,
    Motion,
    Ribbon,
    ViewLayout,
    DrawLayout,
    RasterLayout,
    RibbonLayout,
    DomLayout,
    ResolveLayout,
    MotionLayout,
    ComputePipelineLayout,
    DomDepthPipelineLayout,
    DomOpacityPipelineLayout,
    ResolvePipelineLayout,
    MotionPipelineLayout,
    RibbonPipelineLayout,
    RasterDepth,
    RasterRecords,
    Voxelize,
    DepthTiles,
    DomDepth,
    DomOpacity,
    ResolvePipeline,
    MotionPipeline,
    RibbonDepth,
    RibbonRecords,
}

/// The shaders and their cache keys.
const SHADERS: [(Key, &str); 5] = [
    (Key::Raster, "hair/raster.wgsl"),
    (Key::Dom, "hair/dom.wgsl"),
    (Key::Resolve, "hair/resolve.wgsl"),
    (Key::Motion, "hair/motion.wgsl"),
    (Key::Ribbon, "hair/ribbon.wgsl"),
];

/// A buffer the renderer made.
#[derive(Debug)]
struct Owned {
    id: BufferId,
    raw: wgpu::Buffer,
}

fn make_buffer(
    resources: &mut Resources,
    size: u64,
    usage: wgpu::BufferUsages,
    purpose: &'static str,
) -> Result<Owned, HairError> {
    let size = size.max(16).next_multiple_of(16);
    if usage.contains(wgpu::BufferUsages::STORAGE) {
        let limit = resources.limits().max_storage_buffer_binding_size;
        if size > limit {
            return Err(HairError::TooLarge { what: purpose, value: size, limit });
        }
    }
    let (id, _) = resources.create_buffer(&BufferDesc {
        size,
        usage: usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        tag: Tag::shared(Category::Hair, purpose),
    })?;
    let raw = resources.buffer(id).ok_or(GpuError::Stale { what: "hair buffer" })?.raw().clone();
    Ok(Owned { id, raw })
}

/// One registered groom: its strands, its draw block, the previous frame's points and its bind
/// group.
#[derive(Debug)]
struct Slot {
    groom: HairGroom,
    draw: Owned,
    prev: Owned,
    bind: wgpu::BindGroup,
    lod: GroomLod,
    grid: Grid,
    fresh: bool,
}

/// The per-pixel visibility buffer and the ribbon passes' dummy target, sized for the target.
#[derive(Debug)]
struct Screen {
    width: u32,
    height: u32,
    depth: Owned,
    coverage: Owned,
    records: Owned,
    /// The farthest scene depth per tile (`depth_tiles`).
    tiles: Owned,
    ribbon: TextureId,
    ribbon_view: wgpu::TextureView,
}

/// The renderer's inputs the frame bind groups were made for.
#[derive(Debug, Clone, PartialEq, Eq)]
struct BindKey {
    ring: wgpu::Buffer,
    depth: wgpu::TextureView,
    surface: wgpu::TextureView,
    shadow: wgpu::TextureView,
    sh: wgpu::Buffer,
    size: (u32, u32),
}

/// The bind groups of a frame that depend on the renderer's targets.
#[derive(Debug)]
struct FrameBinds {
    key: BindKey,
    view: wgpu::BindGroup,
    raster: wgpu::BindGroup,
    ribbon: wgpu::BindGroup,
    resolve: wgpu::BindGroup,
    motion: wgpu::BindGroup,
}

/// The deep opacity maps.
#[derive(Debug)]
struct Doms {
    size: u32,
    depth: TextureId,
    opacity: TextureId,
    depth_layers: Vec<wgpu::TextureView>,
    opacity_layers: Vec<wgpu::TextureView>,
    depth_array: wgpu::TextureView,
    opacity_array: wgpu::TextureView,
    bind: wgpu::BindGroup,
}

/// The hair passes. See the module documentation.
#[derive(Debug)]
pub struct HairRenderer {
    cache: PipelineCache<Key>,
    settings: HairSettings,
    luts: HairLuts,
    view_buffer: Owned,
    voxels: Owned,
    doms: Doms,
    shadow_sampler: wgpu::Sampler,
    slots: Vec<Slot>,
    screen: Option<Screen>,
    binds: Option<FrameBinds>,
    view: ViewBlock,
    dom_count: usize,
    prepared: bool,
}

const ALL_STAGES: wgpu::ShaderStages =
    wgpu::ShaderStages::COMPUTE.union(wgpu::ShaderStages::VERTEX).union(wgpu::ShaderStages::FRAGMENT);

fn buffer_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BufferBindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

fn storage(binding: u32, visibility: wgpu::ShaderStages, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    buffer_entry(binding, visibility, wgpu::BufferBindingType::Storage { read_only })
}

fn texture(
    binding: u32,
    visibility: wgpu::ShaderStages,
    sample_type: wgpu::TextureSampleType,
    view_dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture { sample_type, view_dimension, multisampled: false },
        count: None,
    }
}

fn sampler(binding: u32, ty: wgpu::SamplerBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(ty),
        count: None,
    }
}

const FLOAT: wgpu::TextureSampleType = wgpu::TextureSampleType::Float { filterable: true };
const DEPTH: wgpu::TextureSampleType = wgpu::TextureSampleType::Depth;
const D2: wgpu::TextureViewDimension = wgpu::TextureViewDimension::D2;
const D2_ARRAY: wgpu::TextureViewDimension = wgpu::TextureViewDimension::D2Array;
const FRAGMENT: wgpu::ShaderStages = wgpu::ShaderStages::FRAGMENT;
const COMPUTE: wgpu::ShaderStages = wgpu::ShaderStages::COMPUTE;

fn entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

fn layer_view(texture: &wgpu::Texture, layer: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("hair layer"),
        dimension: Some(D2),
        base_array_layer: layer,
        array_layer_count: Some(1),
        ..wgpu::TextureViewDescriptor::default()
    })
}

fn array_view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("hair layers"),
        dimension: Some(D2_ARRAY),
        ..wgpu::TextureViewDescriptor::default()
    })
}

impl HairRenderer {
    /// Builds the pipelines, the fibre tables for `fibre`, the deep opacity maps and the density
    /// grid.
    pub fn new(
        gpu: &Gpu,
        resources: &mut Resources,
        settings: HairSettings,
        fibre: &FibreParams,
    ) -> Result<Self, HairError> {
        settings.check()?;
        let mut cache = PipelineCache::new(gpu);
        for (key, path) in SHADERS {
            let source = humanart_shaders::compose(path, &[])?.into_source();
            cache.shader(key, path, &source)?;
        }
        Self::build_layouts(&mut cache)?;
        Self::build_pipelines(&mut cache)?;
        cache.mark_warm();
        let luts = HairLuts::new(resources, fibre)?;
        let view_buffer =
            make_buffer(resources, std::mem::size_of::<ViewBlock>() as u64, wgpu::BufferUsages::UNIFORM, "hair view")?;
        let voxels = make_buffer(
            resources,
            u64::from(settings.voxels).pow(3) * 4,
            wgpu::BufferUsages::STORAGE,
            "hair density grid",
        )?;
        let doms = Self::make_doms(&cache, resources, settings.dom_size)?;
        let shadow_sampler = resources
            .sampler(&SamplerKey { compare: Some(wgpu::CompareFunction::GreaterEqual), ..SamplerKey::LINEAR_CLAMP })?
            .clone();
        Ok(Self {
            cache,
            settings,
            luts,
            view_buffer,
            voxels,
            doms,
            shadow_sampler,
            slots: Vec::new(),
            screen: None,
            binds: None,
            view: ViewBlock::ZERO,
            dom_count: 0,
            prepared: false,
        })
    }

    fn build_layouts(cache: &mut PipelineCache<Key>) -> Result<(), HairError> {
        let frame_size = wgpu::BufferSize::new(std::mem::size_of::<FrameUniforms>() as u64);
        cache.bind_group_layout(
            Key::ViewLayout,
            "hair view",
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ALL_STAGES,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: frame_size,
                    },
                    count: None,
                },
                buffer_entry(1, ALL_STAGES, wgpu::BufferBindingType::Uniform),
            ],
        )?;
        cache.bind_group_layout(
            Key::DrawLayout,
            "hair draw",
            &[
                buffer_entry(0, ALL_STAGES, wgpu::BufferBindingType::Uniform),
                storage(1, ALL_STAGES, true),
                storage(2, ALL_STAGES, true),
                storage(3, ALL_STAGES, true),
            ],
        )?;
        cache.bind_group_layout(
            Key::RasterLayout,
            "hair raster",
            &[
                texture(0, COMPUTE, DEPTH, D2),
                storage(1, COMPUTE, false),
                storage(2, COMPUTE, false),
                storage(3, COMPUTE, false),
                storage(4, COMPUTE, false),
                storage(5, COMPUTE, false),
            ],
        )?;
        cache.bind_group_layout(
            Key::RibbonLayout,
            "hair ribbon",
            &[
                storage(0, FRAGMENT, false),
                storage(1, FRAGMENT, false),
                storage(2, FRAGMENT, false),
                texture(3, FRAGMENT, DEPTH, D2),
            ],
        )?;
        cache.bind_group_layout(Key::DomLayout, "hair dom", &[texture(0, FRAGMENT, DEPTH, D2_ARRAY)])?;
        cache.bind_group_layout(
            Key::ResolveLayout,
            "hair resolve",
            &[
                storage(0, FRAGMENT, true),
                storage(1, FRAGMENT, true),
                storage(2, FRAGMENT, true),
                storage(3, FRAGMENT, true),
                storage(4, FRAGMENT, true),
                texture(5, FRAGMENT, FLOAT, wgpu::TextureViewDimension::D3),
                texture(6, FRAGMENT, FLOAT, D2_ARRAY),
                sampler(7, wgpu::SamplerBindingType::Filtering),
                texture(8, FRAGMENT, DEPTH, D2_ARRAY),
                texture(9, FRAGMENT, FLOAT, D2_ARRAY),
                texture(10, FRAGMENT, DEPTH, D2_ARRAY),
                sampler(11, wgpu::SamplerBindingType::Comparison),
                texture(12, FRAGMENT, DEPTH, D2),
                texture(13, FRAGMENT, wgpu::TextureSampleType::Uint, D2),
            ],
        )?;
        cache.bind_group_layout(
            Key::MotionLayout,
            "hair motion",
            &[storage(0, FRAGMENT, true), storage(1, FRAGMENT, true), storage(2, FRAGMENT, true)],
        )?;
        let v = Some(&Key::ViewLayout);
        let d = Some(&Key::DrawLayout);
        cache.pipeline_layout(Key::ComputePipelineLayout, "hair raster", &[v, d, Some(&Key::RasterLayout)])?;
        cache.pipeline_layout(Key::DomDepthPipelineLayout, "hair dom depth", &[v, d])?;
        cache.pipeline_layout(Key::DomOpacityPipelineLayout, "hair dom opacity", &[v, d, Some(&Key::DomLayout)])?;
        cache.pipeline_layout(Key::ResolvePipelineLayout, "hair resolve", &[v, Some(&Key::ResolveLayout)])?;
        cache.pipeline_layout(Key::MotionPipelineLayout, "hair motion", &[v, Some(&Key::MotionLayout)])?;
        cache.pipeline_layout(Key::RibbonPipelineLayout, "hair ribbon", &[v, d, Some(&Key::RibbonLayout)])?;
        Ok(())
    }

    fn build_pipelines(cache: &mut PipelineCache<Key>) -> Result<(), HairError> {
        struct Desc {
            key: Key,
            label: &'static str,
            shader: Key,
            layout: Key,
            vs: &'static str,
            fs: Option<&'static str>,
            target: Option<wgpu::ColorTargetState>,
            depth: Option<wgpu::DepthStencilState>,
            topology: wgpu::PrimitiveTopology,
        }
        for (key, entry_point) in [
            (Key::RasterDepth, "raster_depth"),
            (Key::RasterRecords, "raster_records"),
            (Key::Voxelize, "voxelize"),
            (Key::DepthTiles, "depth_tiles"),
        ] {
            cache.compute_pipeline(key, |parts| {
                Ok(parts.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry_point),
                    layout: Some(parts.pipeline_layout(&Key::ComputePipelineLayout)?),
                    module: parts.shader(&Key::Raster)?,
                    entry_point: Some(entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                }))
            })?;
        }
        let depth = |write: bool, compare: wgpu::CompareFunction| {
            Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(write),
                depth_compare: Some(compare),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            })
        };
        let additive = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let over = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        let ribbon_target =
            Some(wgpu::ColorTargetState { format: RIBBON_FORMAT, blend: None, write_mask: wgpu::ColorWrites::empty() });
        let descs = [
            Desc {
                key: Key::DomDepth,
                label: "hair dom depth",
                shader: Key::Dom,
                layout: Key::DomDepthPipelineLayout,
                vs: "vs_dom",
                fs: None,
                target: None,
                depth: depth(true, wgpu::CompareFunction::Greater),
                topology: wgpu::PrimitiveTopology::LineList,
            },
            Desc {
                key: Key::DomOpacity,
                label: "hair dom opacity",
                shader: Key::Dom,
                layout: Key::DomOpacityPipelineLayout,
                vs: "vs_dom",
                fs: Some("fs_dom_opacity"),
                target: Some(wgpu::ColorTargetState {
                    format: OPACITY_FORMAT,
                    blend: Some(wgpu::BlendState { color: additive, alpha: additive }),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                depth: None,
                topology: wgpu::PrimitiveTopology::LineList,
            },
            Desc {
                key: Key::ResolvePipeline,
                label: "hair resolve",
                shader: Key::Resolve,
                layout: Key::ResolvePipelineLayout,
                vs: "vs_resolve",
                fs: Some("fs_resolve"),
                target: Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(wgpu::BlendState { color: over, alpha: over }),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                depth: None,
                topology: wgpu::PrimitiveTopology::TriangleList,
            },
            Desc {
                key: Key::MotionPipeline,
                label: "hair motion",
                shader: Key::Motion,
                layout: Key::MotionPipelineLayout,
                vs: "vs_motion",
                fs: Some("fs_motion"),
                target: Some(wgpu::ColorTargetState {
                    format: SURFACE_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALPHA,
                }),
                depth: depth(true, wgpu::CompareFunction::GreaterEqual),
                topology: wgpu::PrimitiveTopology::TriangleList,
            },
            Desc {
                key: Key::RibbonDepth,
                label: "hair ribbon depth",
                shader: Key::Ribbon,
                layout: Key::RibbonPipelineLayout,
                vs: "vs_ribbon",
                fs: Some("fs_ribbon_depth"),
                target: ribbon_target.clone(),
                // No depth attachment: the fragments write storage buffers, which a late depth
                // test would not stop, so they test the scene depth themselves.
                depth: None,
                topology: wgpu::PrimitiveTopology::TriangleList,
            },
            Desc {
                key: Key::RibbonRecords,
                label: "hair ribbon records",
                shader: Key::Ribbon,
                layout: Key::RibbonPipelineLayout,
                vs: "vs_ribbon",
                fs: Some("fs_ribbon_records"),
                target: ribbon_target,
                depth: None,
                topology: wgpu::PrimitiveTopology::TriangleList,
            },
        ];
        for d in descs {
            cache.render_pipeline(d.key, |parts| {
                let module = parts.shader(&d.shader)?;
                let targets = [d.target.clone()];
                Ok(parts.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(d.label),
                    layout: Some(parts.pipeline_layout(&d.layout)?),
                    vertex: wgpu::VertexState {
                        module,
                        entry_point: Some(d.vs),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: d.topology,
                        cull_mode: None,
                        ..wgpu::PrimitiveState::default()
                    },
                    depth_stencil: d.depth.clone(),
                    multisample: wgpu::MultisampleState::default(),
                    fragment: d.fs.map(|fs| wgpu::FragmentState {
                        module,
                        entry_point: Some(fs),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &targets,
                    }),
                    multiview_mask: None,
                    cache: None,
                }))
            })?;
        }
        Ok(())
    }

    fn make_doms(cache: &PipelineCache<Key>, resources: &mut Resources, size: u32) -> Result<Doms, HairError> {
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let layers = MAX_DOMS as u32;
        let (depth, _) = resources.create_texture(&TextureDesc::array(
            size,
            size,
            layers,
            DEPTH_FORMAT,
            usage,
            Tag::shared(Category::Hair, "hair deep opacity depth"),
        ))?;
        let opacity = match resources.create_texture(&TextureDesc::array(
            size,
            size,
            layers,
            OPACITY_FORMAT,
            usage,
            Tag::shared(Category::Hair, "hair deep opacity layers"),
        )) {
            Ok((id, _)) => id,
            Err(e) => {
                let _ = resources.destroy_texture(depth);
                return Err(e.into());
            }
        };
        let raw = |id| resources.texture(id).map(|t| t.raw().clone()).ok_or(GpuError::Stale { what: "hair texture" });
        let (depth_raw, opacity_raw) = (raw(depth)?, raw(opacity)?);
        let depth_layers = (0..layers).map(|m| layer_view(&depth_raw, m)).collect();
        let opacity_layers = (0..layers).map(|m| layer_view(&opacity_raw, m)).collect();
        let depth_array = array_view(&depth_raw);
        let opacity_array = array_view(&opacity_raw);
        let layout = cache.get_bind_group_layout(&Key::DomLayout).ok_or(GpuError::Stale { what: "hair dom layout" })?;
        let bind = resources.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hair dom"),
            layout,
            entries: &[entry(0, wgpu::BindingResource::TextureView(&depth_array))],
        });
        Ok(Doms { size, depth, opacity, depth_layers, opacity_layers, depth_array, opacity_array, bind })
    }

    /// The settings.
    pub fn settings(&self) -> &HairSettings {
        &self.settings
    }

    /// Changes the settings; the deep opacity maps and the density grid are remade when their
    /// sizes change.
    pub fn set_settings(&mut self, resources: &mut Resources, settings: HairSettings) -> Result<(), HairError> {
        settings.check()?;
        if settings.dom_size != self.doms.size {
            let doms = Self::make_doms(&self.cache, resources, settings.dom_size)?;
            let old = std::mem::replace(&mut self.doms, doms);
            let _ = resources.destroy_texture(old.depth);
            let _ = resources.destroy_texture(old.opacity);
            self.binds = None;
        }
        if settings.voxels != self.settings.voxels {
            let voxels = make_buffer(
                resources,
                u64::from(settings.voxels).pow(3) * 4,
                wgpu::BufferUsages::STORAGE,
                "hair density grid",
            )?;
            let old = std::mem::replace(&mut self.voxels, voxels);
            let _ = resources.destroy_buffer(old.id);
            self.binds = None;
        }
        self.settings = settings;
        Ok(())
    }

    /// The fibre tables (their roughness and cuticle tilt).
    pub fn luts(&self) -> &HairLuts {
        &self.luts
    }

    /// Changes the fibre's roughness and tilt (the tables are integrated again when they change).
    pub fn set_fibre(&mut self, resources: &mut Resources, fibre: &FibreParams) -> Result<(), HairError> {
        if self.luts.set_params(resources, fibre)? {
            self.binds = None;
        }
        Ok(())
    }

    /// Replaces the grooms drawn (at most [`MAX_GROOMS`]); their per-groom buffers and bind groups
    /// are made here.
    pub fn set_grooms(&mut self, resources: &mut Resources, grooms: &[HairGroom]) -> Result<(), HairError> {
        if grooms.len() > MAX_GROOMS {
            return Err(HairError::TooLarge {
                what: "grooms in one view",
                value: grooms.len() as u64,
                limit: MAX_GROOMS as u64,
            });
        }
        for g in grooms {
            g.check()?;
        }
        self.clear_grooms(resources);
        let layout =
            self.cache.get_bind_group_layout(&Key::DrawLayout).ok_or(GpuError::Stale { what: "hair draw layout" })?;
        for g in grooms {
            let draw = make_buffer(
                resources,
                std::mem::size_of::<DrawBlock>() as u64,
                wgpu::BufferUsages::UNIFORM,
                "hair draw",
            )?;
            let prev =
                match make_buffer(resources, g.points.size(), wgpu::BufferUsages::STORAGE, "hair previous points") {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = resources.destroy_buffer(draw.id);
                        return Err(e);
                    }
                };
            let bind = resources.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hair groom"),
                layout,
                entries: &[
                    entry(0, draw.raw.as_entire_binding()),
                    entry(1, g.points.as_entire_binding()),
                    entry(2, prev.raw.as_entire_binding()),
                    entry(3, g.attrs.as_entire_binding()),
                ],
            });
            let lod = GroomLod::select(&LodInput {
                strands: g.strands,
                points: g.points_per_strand,
                diameter: g.diameter,
                length: g.length,
                distance: 1.0,
                focal_px: 1.0,
                min_keep: g.min_keep,
            })?;
            self.slots.push(Slot {
                groom: g.clone(),
                draw,
                prev,
                bind,
                lod,
                grid: Grid { x: 0, y: 0, row: 0 },
                fresh: true,
            });
        }
        Ok(())
    }

    /// Moves groom `index`'s bounding sphere (the head moved).
    pub fn set_bounds(&mut self, index: usize, centre: Vec3, radius: f32) -> Result<(), HairError> {
        let slot = self.slots.get_mut(index).ok_or(HairError::Invalid { what: "no groom at this index" })?;
        let mut g = slot.groom.clone();
        g.centre = centre;
        g.radius = radius;
        g.check()?;
        slot.groom = g;
        Ok(())
    }

    /// Removes every groom.
    pub fn clear_grooms(&mut self, resources: &mut Resources) {
        for slot in self.slots.drain(..) {
            let _ = resources.destroy_buffer(slot.draw.id);
            let _ = resources.destroy_buffer(slot.prev.id);
        }
    }

    /// Grooms drawn.
    pub fn groom_count(&self) -> usize {
        self.slots.len()
    }

    /// The level of detail chosen for groom `index` in the last prepared frame.
    pub fn lod(&self, index: usize) -> Option<GroomLod> {
        self.slots.get(index).map(|s| s.lod)
    }

    /// Lights with a deep opacity map in the last prepared frame.
    pub fn dom_count(&self) -> usize {
        self.dom_count
    }

    /// The per-pixel visibility buffer of the last frame: nearest depth bits, coverage sum
    /// (fixed point, [`COVERAGE_ONE`] per unit) and records (for tests and debug views).
    pub fn visibility_buffers(&self) -> Option<[&wgpu::Buffer; 3]> {
        self.screen.as_ref().map(|s| [&s.depth.raw, &s.coverage.raw, &s.records.raw])
    }

    fn ensure_screen(&mut self, resources: &mut Resources, width: u32, height: u32) -> Result<(), HairError> {
        if self.screen.as_ref().is_some_and(|s| s.width == width && s.height == height) {
            return Ok(());
        }
        self.drop_screen(resources);
        let pixels = u64::from(width) * u64::from(height);
        let storage = wgpu::BufferUsages::STORAGE;
        let depth = make_buffer(resources, pixels * 4, storage, "hair visibility depth")?;
        let coverage = make_buffer(resources, pixels * 4, storage, "hair visibility coverage");
        let records = make_buffer(resources, pixels * RECORD_BYTES, storage, "hair visibility records");
        let ribbon = resources.create_texture(&TextureDesc::d2(
            width,
            height,
            RIBBON_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            Tag::shared(Category::Hair, "hair ribbon target"),
        ));
        let (coverage, records, ribbon) = match (coverage, records, ribbon) {
            (Ok(c), Ok(r), Ok((t, _))) => (c, r, t),
            (c, r, t) => {
                let _ = resources.destroy_buffer(depth.id);
                let mut error = None;
                if let Ok(c) = c {
                    let _ = resources.destroy_buffer(c.id);
                } else if let Err(e) = c {
                    error = Some(e);
                }
                match r {
                    Ok(r) => {
                        let _ = resources.destroy_buffer(r.id);
                    }
                    Err(e) => error = error.or(Some(e)),
                }
                match t {
                    Ok((t, _)) => {
                        let _ = resources.destroy_texture(t);
                    }
                    Err(e) => error = error.or(Some(e.into())),
                }
                return Err(error.unwrap_or(HairError::Invalid { what: "the hair visibility buffer" }));
            }
        };
        let tile_count = u64::from(width.div_ceil(TILE)) * u64::from(height.div_ceil(TILE));
        let tiles = match make_buffer(resources, tile_count * 4, storage, "hair scene depth tiles") {
            Ok(t) => t,
            Err(e) => {
                for id in [depth.id, coverage.id, records.id] {
                    let _ = resources.destroy_buffer(id);
                }
                let _ = resources.destroy_texture(ribbon);
                return Err(e);
            }
        };
        let ribbon_view = resources
            .texture(ribbon)
            .map(|t| t.raw().create_view(&wgpu::TextureViewDescriptor::default()))
            .ok_or(GpuError::Stale { what: "hair ribbon target" })?;
        self.screen = Some(Screen { width, height, depth, coverage, records, tiles, ribbon, ribbon_view });
        self.binds = None;
        Ok(())
    }

    fn drop_screen(&mut self, resources: &mut Resources) {
        if let Some(s) = self.screen.take() {
            for id in [s.depth.id, s.coverage.id, s.records.id, s.tiles.id] {
                let _ = resources.destroy_buffer(id);
            }
            let _ = resources.destroy_texture(s.ribbon);
        }
        self.binds = None;
    }

    fn make_binds(&self, resources: &Resources, frame: &HairFrame<'_>, key: BindKey) -> Result<FrameBinds, HairError> {
        let screen = self.screen.as_ref().ok_or(HairError::Invalid { what: "the hair visibility buffer" })?;
        let device = resources.device();
        let layout = |k: Key| self.cache.get_bind_group_layout(&k).ok_or(GpuError::Stale { what: "hair layout" });
        let view = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hair view"),
            layout: layout(Key::ViewLayout)?,
            entries: &[
                entry(
                    0,
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: frame.ring,
                        offset: 0,
                        size: wgpu::BufferSize::new(u64::from(RING_BINDING_SIZE)),
                    }),
                ),
                entry(1, self.view_buffer.raw.as_entire_binding()),
            ],
        });
        let raster = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hair raster"),
            layout: layout(Key::RasterLayout)?,
            entries: &[
                entry(0, wgpu::BindingResource::TextureView(frame.depth)),
                entry(1, screen.depth.raw.as_entire_binding()),
                entry(2, screen.coverage.raw.as_entire_binding()),
                entry(3, screen.records.raw.as_entire_binding()),
                entry(4, self.voxels.raw.as_entire_binding()),
                entry(5, screen.tiles.raw.as_entire_binding()),
            ],
        });
        let ribbon = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hair ribbon"),
            layout: layout(Key::RibbonLayout)?,
            entries: &[
                entry(0, screen.depth.raw.as_entire_binding()),
                entry(1, screen.coverage.raw.as_entire_binding()),
                entry(2, screen.records.raw.as_entire_binding()),
                entry(3, wgpu::BindingResource::TextureView(frame.depth)),
            ],
        });
        let resolve = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hair resolve"),
            layout: layout(Key::ResolveLayout)?,
            entries: &[
                entry(0, screen.depth.raw.as_entire_binding()),
                entry(1, screen.coverage.raw.as_entire_binding()),
                entry(2, screen.records.raw.as_entire_binding()),
                entry(3, self.voxels.raw.as_entire_binding()),
                entry(4, frame.irradiance_sh.as_entire_binding()),
                entry(5, wgpu::BindingResource::TextureView(self.luts.azimuthal_view())),
                entry(6, wgpu::BindingResource::TextureView(self.luts.dual_view())),
                entry(7, wgpu::BindingResource::Sampler(self.luts.sampler())),
                entry(8, wgpu::BindingResource::TextureView(&self.doms.depth_array)),
                entry(9, wgpu::BindingResource::TextureView(&self.doms.opacity_array)),
                entry(10, wgpu::BindingResource::TextureView(frame.shadow_atlas)),
                entry(11, wgpu::BindingResource::Sampler(&self.shadow_sampler)),
                entry(12, wgpu::BindingResource::TextureView(frame.depth)),
                entry(13, wgpu::BindingResource::TextureView(frame.surface)),
            ],
        });
        let motion = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hair motion"),
            layout: layout(Key::MotionLayout)?,
            entries: &[
                entry(0, screen.depth.raw.as_entire_binding()),
                entry(1, screen.coverage.raw.as_entire_binding()),
                entry(2, screen.records.raw.as_entire_binding()),
            ],
        });
        Ok(FrameBinds { key, view, raster, ribbon, resolve, motion })
    }

    /// Prepares a frame: sizes the visibility buffer, (re)makes the frame bind groups when the
    /// renderer's targets changed, chooses each groom's level of detail and the lights' deep
    /// opacity frames, and writes the constant blocks. Returns whether there is hair to draw.
    pub fn prepare(&mut self, resources: &mut Resources, frame: &HairFrame<'_>) -> Result<bool, HairError> {
        self.prepared = false;
        if self.slots.is_empty() {
            return Ok(false);
        }
        if frame.width == 0 || frame.height == 0 {
            return Err(HairError::Invalid { what: "a hair target of zero size" });
        }
        self.ensure_screen(resources, frame.width, frame.height)?;
        let key = BindKey {
            ring: frame.ring.clone(),
            depth: frame.depth.clone(),
            surface: frame.surface.clone(),
            shadow: frame.shadow_atlas.clone(),
            sh: frame.irradiance_sh.clone(),
            size: (frame.width, frame.height),
        };
        if self.binds.as_ref().is_none_or(|b| b.key != key) {
            self.binds = Some(self.make_binds(resources, frame, key)?);
        }
        let projection = frame.view.projection.cols[1].y;
        let focal_px = (projection * frame.height as f32 * 0.5).abs();
        let eye = frame.view.eye;
        // The union of the grooms' spheres: the deep opacity maps' and the density grid's extent.
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for s in &self.slots {
            lo = lo.min(s.groom.centre - Vec3::splat(s.groom.radius));
            hi = hi.max(s.groom.centre + Vec3::splat(s.groom.radius));
        }
        let centre = (lo + hi) * 0.5;
        let radius = (hi - lo).length() * 0.5 * 1.02;
        let mut view = ViewBlock::ZERO;
        view.screen = [frame.width as f32, frame.height as f32, focal_px, COVERAGE_SATURATION * COVERAGE_ONE];
        let (order, lights) = dom_lights(frame.lights, centre);
        let size = self.doms.size as f32;
        let mut dom_count = 0;
        for &slot in order.iter().take(lights) {
            let Some(light) = frame.lights.get(slot) else { continue };
            let Some(f) = DomFrame::fit(light, centre, radius) else { continue };
            let texel = 2.0 * radius / size;
            view.dom_right[dom_count] = [f.right.x, f.right.y, f.right.z, f.centre.x];
            view.dom_up[dom_count] = [f.up.x, f.up.y, f.up.z, f.centre.y];
            view.dom_forward[dom_count] = [f.forward.x, f.forward.y, f.forward.z, f.centre.z];
            view.dom_params[dom_count] = [f.radius, slot as f32, texel, 1.5 * texel];
            dom_count += 1;
        }
        self.dom_count = dom_count;
        view.dom_layers = DOM_LAYERS;
        let n = self.settings.voxels;
        let cell = 2.0 * radius / n as f32;
        let origin = centre - Vec3::splat(radius);
        view.voxel = [origin.x, origin.y, origin.z, cell];
        let fibre = self.luts.params();
        view.variances = fibre.variances();
        view.shifts = [fibre.shift(0), fibre.shift(1), fibre.shift(2), VOXEL_TAU_FACTOR];
        let mut flags = 0;
        if self.settings.dual_scattering {
            flags |= FLAG_DUAL;
        }
        if self.settings.occlusion {
            flags |= FLAG_OCCLUSION;
        }
        if self.settings.skin_shadow {
            flags |= FLAG_SKIN_SHADOW;
        }
        if self.settings.cull_hidden {
            flags |= FLAG_CULL;
        }
        view.counts = [n, dom_count as u32, self.slots.len() as u32, flags];
        let full = self.settings.full_detail;
        for (index, slot) in self.slots.iter_mut().enumerate() {
            let g = &slot.groom;
            let natural = g.color.natural_sigma_a();
            // The dye's own sigma_a: the colour with the dye at full strength.
            let dyed = HairColor { dye_amount: 1.0, ..g.color }.strand_sigma_a(0, 0);
            let dye = if g.color.dye_amount.is_finite() { g.color.dye_amount.clamp(0.0, 1.0) } else { 0.0 };
            let variation = if g.color.variation.is_finite() { g.color.variation.clamp(0.0, 1.0) } else { 0.0 };
            if let Some(out) = view.grooms.get_mut(index) {
                *out = [natural[0], natural[1], natural[2], variation, dyed[0], dyed[1], dyed[2], dye];
            }
            let distance = ((g.centre - eye).length() - g.radius).max(0.05);
            let lod = GroomLod::select(&LodInput {
                strands: g.strands,
                points: g.points_per_strand,
                diameter: g.diameter,
                length: g.length,
                distance: if full { 1e-3 } else { distance },
                focal_px,
                min_keep: g.min_keep,
            })?;
            let lod = if full {
                GroomLod { keep: 1.0, keep_threshold: u32::MAX, coverage_scale: 1.0, dom_step: 1, ..lod }
            } else {
                lod
            };
            let split =
                if self.settings.split_walks && self.settings.path == HairPath::Compute { lod.split } else { 1 };
            let lod = GroomLod { split, ..lod };
            let grid = Grid::new(g.strands, lod.segments.saturating_mul(split))?;
            let draw = DrawBlock {
                counts: [g.strands, g.points_per_strand, lod.stride, index as u32],
                dispatch: [lod.segments, grid.row, lod.keep_threshold, lod.dom_step],
                scale: [lod.coverage_scale, lod.dom_step as f32, 1.0, split as f32],
            };
            resources.write_buffer(slot.draw.id, 0, bytes_of(&draw))?;
            slot.lod = lod;
            slot.grid = grid;
            if frame.reset_history {
                slot.fresh = true;
            }
        }
        resources.write_buffer(self.view_buffer.id, 0, bytes_of(&view))?;
        self.view = view;
        self.prepared = true;
        Ok(true)
    }

    /// Records the hair passes of the prepared frame: deep opacity maps, visibility, resolve over
    /// `targets.hdr`, motion and depth into the TAA inputs. `frame_offset` is the frame constants'
    /// offset in the ring. Does nothing when [`HairRenderer::prepare`] found no hair.
    pub fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        mut profiler: Option<&mut GpuProfiler>,
        frame_offset: u32,
        targets: &HairTargets<'_>,
    ) -> Result<(), HairError> {
        if !self.prepared {
            return Ok(());
        }
        self.prepared = false;
        let (Some(screen), Some(binds)) = (self.screen.as_ref(), self.binds.as_ref()) else {
            return Err(HairError::Invalid { what: "the hair frame was not prepared" });
        };
        let stale = |what| HairError::Gpu(GpuError::Stale { what });
        let render = |k: Key| self.cache.get_render_pipeline(&k).ok_or(stale("hair render pipeline"));
        let compute = |k: Key| self.cache.get_compute_pipeline(&k).ok_or(stale("hair compute pipeline"));
        let offsets = [frame_offset];
        // First frame of a groom: its previous points are its current ones (no motion).
        for slot in &self.slots {
            if slot.fresh {
                let size = slot.groom.points.size().min(slot.prev.raw.size());
                encoder.copy_buffer_to_buffer(&slot.groom.points, 0, &slot.prev.raw, 0, size);
            }
        }
        encoder.clear_buffer(&screen.depth.raw, 0, None);
        encoder.clear_buffer(&screen.coverage.raw, 0, None);
        encoder.clear_buffer(&self.voxels.raw, 0, None);

        // 1. Deep opacity maps.
        for m in 0..self.dom_count {
            let (Some(depth_view), Some(opacity_view)) =
                (self.doms.depth_layers.get(m), self.doms.opacity_layers.get(m))
            else {
                continue;
            };
            let map = m as u32;
            {
                let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("hair dom depth"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: if m == 0 {
                        profiler.as_deref_mut().and_then(|p| p.render_pass_writes("hair dom"))
                    } else {
                        None
                    },
                    ..Default::default()
                });
                rp.set_pipeline(render(Key::DomDepth)?);
                rp.set_bind_group(0, &binds.view, &offsets);
                for slot in &self.slots {
                    rp.set_bind_group(1, &slot.bind, &[]);
                    rp.draw(
                        0..dom_vertices(slot.groom.strands, slot.groom.points_per_strand, slot.lod.dom_step),
                        map..map + 1,
                    );
                }
            }
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hair dom opacity"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: opacity_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            rp.set_pipeline(render(Key::DomOpacity)?);
            rp.set_bind_group(0, &binds.view, &offsets);
            rp.set_bind_group(2, &self.doms.bind, &[]);
            for slot in &self.slots {
                rp.set_bind_group(1, &slot.bind, &[]);
                rp.draw(
                    0..dom_vertices(slot.groom.strands, slot.groom.points_per_strand, slot.lod.dom_step),
                    map..map + 1,
                );
            }
        }

        // 2. Visibility. The farthest scene depth per tile first (the hidden-segment test of the
        //    compute raster).
        if self.settings.path == HairPath::Compute
            && self.settings.cull_hidden
            && let Some(first) = self.slots.iter().find(|s| s.grid.x > 0)
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hair depth tiles"),
                timestamp_writes: profiler.as_deref_mut().and_then(|p| p.compute_pass_writes("hair depth tiles")),
            });
            cp.set_bind_group(0, &binds.view, &offsets);
            // The draw group is unused by the kernel but part of the pipeline layout.
            cp.set_bind_group(1, &first.bind, &[]);
            cp.set_bind_group(2, &binds.raster, &[]);
            cp.set_pipeline(compute(Key::DepthTiles)?);
            cp.dispatch_workgroups(screen.width.div_ceil(TILE), screen.height.div_ceil(TILE), 1);
        }
        // One timed compute pass per kernel.
        let passes: &[(Key, &'static str)] = match self.settings.path {
            HairPath::Compute => &[
                (Key::Voxelize, "hair voxelize"),
                (Key::RasterDepth, "hair raster depth"),
                (Key::RasterRecords, "hair raster records"),
            ],
            HairPath::Ribbon => &[(Key::Voxelize, "hair voxelize")],
        };
        for &(key, scope) in passes {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(scope),
                timestamp_writes: profiler.as_deref_mut().and_then(|p| p.compute_pass_writes(scope)),
            });
            cp.set_bind_group(0, &binds.view, &offsets);
            cp.set_bind_group(2, &binds.raster, &[]);
            cp.set_pipeline(compute(key)?);
            for slot in &self.slots {
                if slot.grid.x == 0 {
                    continue;
                }
                cp.set_bind_group(1, &slot.bind, &[]);
                cp.dispatch_workgroups(slot.grid.x, slot.grid.y, 1);
            }
        }
        if self.settings.path == HairPath::Ribbon {
            for (key, scope) in [(Key::RibbonDepth, "hair ribbon depth"), (Key::RibbonRecords, "hair ribbon records")] {
                let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("hair ribbon"),
                    timestamp_writes: profiler.as_deref_mut().and_then(|p| p.render_pass_writes(scope)),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &screen.ribbon_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Discard,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                rp.set_pipeline(render(key)?);
                rp.set_bind_group(0, &binds.view, &offsets);
                rp.set_bind_group(2, &binds.ribbon, &[]);
                for slot in &self.slots {
                    let vertices = ribbon_vertices(slot.groom.strands, slot.lod.segments)?;
                    rp.set_bind_group(1, &slot.bind, &[]);
                    rp.draw(0..vertices, 0..1);
                }
            }
        }

        // 3. Resolve over the lit scene.
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hair resolve"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: targets.hdr,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                timestamp_writes: profiler.and_then(|p| p.render_pass_writes("hair resolve")),
                ..Default::default()
            });
            rp.set_pipeline(render(Key::ResolvePipeline)?);
            rp.set_bind_group(0, &binds.view, &offsets);
            rp.set_bind_group(1, &binds.resolve, &[]);
            rp.draw(0..3, 0..1);
        }

        // 4. Motion and depth for the TAA.
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hair motion"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: targets.surface,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: targets.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            rp.set_pipeline(render(Key::MotionPipeline)?);
            rp.set_bind_group(0, &binds.view, &offsets);
            rp.set_bind_group(1, &binds.motion, &[]);
            rp.draw(0..3, 0..1);
        }

        // The next frame's previous points.
        for slot in &mut self.slots {
            let size = slot.groom.points.size().min(slot.prev.raw.size());
            encoder.copy_buffer_to_buffer(&slot.groom.points, 0, &slot.prev.raw, 0, size);
            slot.fresh = false;
        }
        Ok(())
    }

    /// Frees every buffer and texture the hair renderer made.
    pub fn destroy(mut self, resources: &mut Resources) {
        self.clear_grooms(resources);
        self.drop_screen(resources);
        let _ = resources.destroy_buffer(self.view_buffer.id);
        let _ = resources.destroy_buffer(self.voxels.id);
        let _ = resources.destroy_texture(self.doms.depth);
        let _ = resources.destroy_texture(self.doms.opacity);
        self.luts.destroy(resources);
    }
}

/// Vertices of a groom's deep-opacity line list: two per segment of every `step`-th strand.
fn dom_vertices(strands: u32, points: u32, step: u32) -> u32 {
    let kept = u64::from(strands.div_ceil(step.max(1)));
    let v = kept * u64::from(points.saturating_sub(1)) * 2;
    u32::try_from(v).unwrap_or(u32::MAX)
}

/// Vertices of a groom's ribbons: six per drawn segment.
fn ribbon_vertices(strands: u32, segments: u32) -> Result<u32, HairError> {
    let v = u64::from(strands) * u64::from(segments) * 6;
    u32::try_from(v).map_err(|_| HairError::TooLarge {
        what: "hair ribbon vertices",
        value: v,
        limit: u64::from(u32::MAX),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hr_04_vertex_counts_follow_the_lod() {
        assert_eq!(dom_vertices(1000, 24, 4), 250 * 23 * 2);
        assert_eq!(dom_vertices(1001, 24, 4), 251 * 23 * 2);
        assert_eq!(dom_vertices(10, 2, 0), 20);
        assert_eq!(ribbon_vertices(100_000, 23).unwrap(), 13_800_000);
        assert!(ribbon_vertices(u32::MAX, 31).is_err());
    }

    #[test]
    fn settings_are_checked() {
        assert!(HairSettings::default().check().is_ok());
        assert!(HairSettings { dom_size: 100, ..HairSettings::default() }.check().is_err());
        assert!(HairSettings { voxels: 300, ..HairSettings::default() }.check().is_err());
    }
}
