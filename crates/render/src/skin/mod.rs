//! The skin appearance system on the GPU (WP-13): the layer compositor `[SKN-02]`, the micro-detail
//! generator `[SKN-20]`, the streaming integration of the resolved set `[SKN-03]` and wrinkle /
//! tension map blending `[SKN-30]` ([`wrinkle`]).
//!
//! [`SkinCompositor`] mirrors a [`LayerStack`] of `humanart_model::skin_appearance` on the GPU and
//! re-resolves only its dirty rectangles. Per dirty rectangle it records two compute dispatches
//! sized to the rectangle (so the cost follows the area, `[AC-WP13-3]`):
//!
//! 1. `skin/composite.wgsl`: base values, zone offsets `[SKN-11]` and every layer, into the five
//!    parameter planes (one `rgba16float` array, layer `tile * 5 + plane`);
//! 2. `skin/resolve.wgsl`: the spectral pigment model `[SKN-10]`, overrides `[SKN-12]`, flush
//!    `[SKN-31]` and wetness `[SKN-32]`, into the resolved set (one `rgba16float` array, layer
//!    `tile * 4 + k`: albedo, scatter, detail, flush) and the specular array (`rgba8unorm`, layer
//!    `tile`) `[SKN-01]`.
//!
//! Every pipeline, bind group and buffer is created in [`SkinCompositor::new`] (`[PERF-R4]`);
//! [`SkinCompositor::resolve`] reuses preallocated tables and allocates nothing (`[PERF-R1]`).
//! Layer content (strokes, generated features, decals) lives in a content atlas uploaded when a
//! layer changes ([`SkinCompositor::sync_layers`], [`SkinCompositor::upload_rect`]), which is
//! editor-time work.

pub mod atlas;
pub mod microdetail;
pub mod page_copy;
pub mod stream;
pub mod wrinkle;

use std::fmt;
use std::num::NonZeroU64;

use humanart_gpu::cache::PipelineCache;
use humanart_gpu::{Gpu, GpuError};
use humanart_model::skin_appearance::channels::{ATTRIBUTE_COUNT, BaseValues, PLANE_COUNT};
use humanart_model::skin_appearance::dirty::{MAX_RECTS_PER_TILE, MAX_TILES, Rect};
use humanart_model::skin_appearance::layers::{LayerId, LayerKind, LayerStack, MAX_LAYERS};
use humanart_model::skin_appearance::pigment::{GPU_TABLE_BYTES, Mapping, PigmentModel};
use humanart_model::skin_appearance::zones::{ZONE_SLOTS, ZoneTable};
use humanart_shaders::ShaderError;

use atlas::ShelfAtlas;

/// `u32` words per layer record (`Layer` in `skin/composite.wgsl`).
const LAYER_WORDS: usize = 32;
/// Bytes per dispatch parameter slot (the minimum uniform offset alignment).
const SLOT_BYTES: u64 = 256;
/// Bytes of `Params` in `skin/common.wgsl`.
const PARAMS_BYTES: u64 = 32;
/// Most rectangles resolved in one call; beyond this every tile is resolved as one rectangle.
pub const MAX_DISPATCH_RECTS: usize = 256;
/// Words of `Globals` in `skin/common.wgsl`.
const GLOBAL_WORDS: usize = 20 + ZONE_SLOTS * 24 + 12;
/// Workgroup edge.
const GROUP: u32 = 8;
/// "No zone" in a layer record.
const NO_ZONE: u32 = u32::MAX;
/// An unused content component.
const UNUSED: u32 = 0xff;

/// Why a skin GPU operation failed.
#[derive(Debug)]
pub enum SkinGpuError {
    /// The device refused a resource or pipeline.
    Gpu(GpuError),
    /// A shader could not be composed.
    Shader(ShaderError),
    /// The model refused the data.
    Model(humanart_model::skin_appearance::SkinError),
    /// The request is malformed.
    Invalid {
        /// What is wrong.
        what: &'static str,
    },
}

impl fmt::Display for SkinGpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gpu(e) => write!(f, "skin compositor: {e}"),
            Self::Shader(e) => write!(f, "skin compositor shader: {e}"),
            Self::Model(e) => write!(f, "skin compositor: {e}"),
            Self::Invalid { what } => write!(f, "skin compositor: {what}"),
        }
    }
}

impl std::error::Error for SkinGpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gpu(e) => Some(e),
            Self::Shader(e) => Some(e),
            Self::Model(e) => Some(e),
            Self::Invalid { .. } => None,
        }
    }
}

impl From<GpuError> for SkinGpuError {
    fn from(e: GpuError) -> Self {
        Self::Gpu(e)
    }
}

impl From<ShaderError> for SkinGpuError {
    fn from(e: ShaderError) -> Self {
        Self::Shader(e)
    }
}

impl From<humanart_model::skin_appearance::SkinError> for SkinGpuError {
    fn from(e: humanart_model::skin_appearance::SkinError) -> Self {
        Self::Model(e)
    }
}

/// The size of one character's skin texture set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SkinDesc {
    /// UV tiles (array layers per channel).
    pub tiles: u32,
    /// Texels per tile side.
    pub tile_size: u32,
    /// Content atlas side in texels.
    pub atlas_size: u32,
}

/// What one [`SkinCompositor::resolve`] recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ResolveReport {
    /// Rectangles resolved.
    pub rects: u32,
    /// Texels resolved (sum of rectangle areas).
    pub texels: u64,
    /// Whether the rectangles were coarsened to whole tiles (too many to dispatch one by one).
    pub coarsened: bool,
}

/// Cache keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Key {
    CompositeShader,
    ResolveShader,
    CompositeLayout,
    ResolveLayout,
    CompositePipelineLayout,
    ResolvePipelineLayout,
    Composite,
    Resolve,
}

/// Where a layer's content on one tile sits in the atlas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
struct Uploaded {
    id: u32,
    version: u64,
}

/// The GPU skin compositor of one texture set. See the module documentation.
pub struct SkinCompositor {
    desc: SkinDesc,
    cache: PipelineCache<Key>,
    planes: wgpu::Texture,
    resolved: wgpu::Texture,
    specular: wgpu::Texture,
    inputs: wgpu::Texture,
    atlas_values: wgpu::Texture,
    atlas_coverage: wgpu::Texture,
    layer_table: wgpu::Buffer,
    layer_tiles: wgpu::Buffer,
    globals: wgpu::Buffer,
    params: wgpu::Buffer,
    composite_group: wgpu::BindGroup,
    resolve_group: wgpu::BindGroup,
    atlas: ShelfAtlas,
    /// Mirror of `layer_table`.
    layer_words: Vec<u32>,
    /// Mirror of `layer_tiles`.
    tile_words: Vec<u32>,
    /// Per layer slot: which layer version is in the atlas.
    uploaded: Vec<Uploaded>,
    layer_count: u32,
    /// Mirror of the parameter slots written by `resolve`.
    param_bytes: Vec<u8>,
    /// The rectangles of one resolve.
    rects: Vec<(u32, Rect)>,
    /// Conversion scratch for uploads (editor-time).
    scratch: Vec<u8>,
}

const fn compute_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None }
}

const fn sampled(dimension: wgpu::TextureViewDimension) -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: false },
        view_dimension: dimension,
        multisampled: false,
    }
}

const fn storage_texture(format: wgpu::TextureFormat) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        access: wgpu::StorageTextureAccess::WriteOnly,
        format,
        view_dimension: wgpu::TextureViewDimension::D2Array,
    }
}

const STORAGE_READ: wgpu::BindingType = wgpu::BindingType::Buffer {
    ty: wgpu::BufferBindingType::Storage { read_only: true },
    has_dynamic_offset: false,
    min_binding_size: None,
};

const PARAMS_UNIFORM: wgpu::BindingType = wgpu::BindingType::Buffer {
    ty: wgpu::BufferBindingType::Uniform,
    has_dynamic_offset: true,
    min_binding_size: NonZeroU64::new(PARAMS_BYTES),
};

/// A view of every layer of an array texture.
fn array_view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("skin array"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}

fn texture(
    gpu: &Gpu,
    label: &str,
    size: (u32, u32, u32),
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: size.2 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn buffer(gpu: &Gpu, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
}

fn words_to_bytes(words: &[u32], out: &mut Vec<u8>) {
    out.clear();
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
}

impl SkinCompositor {
    /// Creates the textures, buffers, pipelines and bind groups for `desc`, with the default
    /// globals (registry defaults, artistic zone table, default pigment mapping).
    pub fn new(gpu: &Gpu, desc: SkinDesc) -> Result<Self, SkinGpuError> {
        let limits = gpu.limits();
        if desc.tiles == 0
            || desc.tiles > MAX_TILES
            || desc.tile_size == 0
            || desc.tile_size > limits.max_texture_dimension_2d
        {
            return Err(SkinGpuError::Invalid { what: "tile count or tile size outside the device's limits" });
        }
        if desc.tiles * PLANE_COUNT as u32 > limits.max_texture_array_layers {
            return Err(SkinGpuError::Invalid { what: "too many tiles for the device's array layers" });
        }
        if desc.atlas_size == 0 || desc.atlas_size > limits.max_texture_dimension_2d.min(1 << 16) {
            return Err(SkinGpuError::Invalid { what: "content atlas size outside the device's limits" });
        }
        let mut cache = PipelineCache::new(gpu);
        for (key, path) in [(Key::CompositeShader, "skin/composite.wgsl"), (Key::ResolveShader, "skin/resolve.wgsl")] {
            let source = humanart_shaders::compose(path, &[])?.into_source();
            cache.shader(key, path, &source)?;
        }
        let d2 = wgpu::TextureViewDimension::D2;
        let array = wgpu::TextureViewDimension::D2Array;
        cache.bind_group_layout(
            Key::CompositeLayout,
            "skin composite",
            &[
                compute_entry(0, storage_texture(wgpu::TextureFormat::Rgba16Float)),
                compute_entry(1, sampled(d2)),
                compute_entry(2, sampled(d2)),
                compute_entry(3, sampled(array)),
                compute_entry(4, STORAGE_READ),
                compute_entry(5, STORAGE_READ),
                compute_entry(6, STORAGE_READ),
                compute_entry(7, PARAMS_UNIFORM),
            ],
        )?;
        cache.bind_group_layout(
            Key::ResolveLayout,
            "skin resolve",
            &[
                compute_entry(0, sampled(array)),
                compute_entry(1, sampled(array)),
                compute_entry(2, storage_texture(wgpu::TextureFormat::Rgba16Float)),
                compute_entry(3, storage_texture(wgpu::TextureFormat::Rgba8Unorm)),
                compute_entry(4, STORAGE_READ),
                compute_entry(5, STORAGE_READ),
                compute_entry(6, PARAMS_UNIFORM),
            ],
        )?;
        cache.pipeline_layout(Key::CompositePipelineLayout, "skin composite", &[Some(&Key::CompositeLayout)])?;
        cache.pipeline_layout(Key::ResolvePipelineLayout, "skin resolve", &[Some(&Key::ResolveLayout)])?;
        for (key, shader, layout, entry) in [
            (Key::Composite, Key::CompositeShader, Key::CompositePipelineLayout, "composite"),
            (Key::Resolve, Key::ResolveShader, Key::ResolvePipelineLayout, "resolve"),
        ] {
            cache.compute_pipeline(key, |parts| {
                Ok(parts.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("skin compositor"),
                    layout: Some(parts.pipeline_layout(&layout)?),
                    module: parts.shader(&shader)?,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                }))
            })?;
        }
        cache.mark_warm();

        let (tiles, size) = (desc.tiles, desc.tile_size);
        let read_write =
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC;
        let upload = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let planes = texture(
            gpu,
            "skin planes",
            (size, size, tiles * PLANE_COUNT as u32),
            wgpu::TextureFormat::Rgba16Float,
            read_write,
        );
        let resolved =
            texture(gpu, "skin resolved", (size, size, tiles * 4), wgpu::TextureFormat::Rgba16Float, read_write);
        let specular = texture(gpu, "skin specular", (size, size, tiles), wgpu::TextureFormat::Rgba8Unorm, read_write);
        let inputs = texture(gpu, "skin inputs", (size, size, tiles * 4), wgpu::TextureFormat::Rgba8Unorm, upload);
        let a = desc.atlas_size;
        let atlas_values = texture(gpu, "skin content values", (a, a, 1), wgpu::TextureFormat::Rgba16Float, upload);
        let atlas_coverage = texture(gpu, "skin content coverage", (a, a, 1), wgpu::TextureFormat::R16Float, upload);

        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let layer_table = buffer(gpu, "skin layers", (MAX_LAYERS * LAYER_WORDS * 4) as u64, storage);
        let tile_count = MAX_LAYERS * MAX_TILES as usize * 4;
        let layer_tiles = buffer(gpu, "skin layer tiles", (tile_count * 4) as u64, storage);
        let globals = buffer(gpu, "skin globals", (GLOBAL_WORDS * 4) as u64, storage);
        let bands = buffer(gpu, "skin pigment bands", GPU_TABLE_BYTES as u64, storage);
        let params = buffer(
            gpu,
            "skin dispatch parameters",
            MAX_DISPATCH_RECTS as u64 * SLOT_BYTES,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let mut band_bytes = vec![0_u8; GPU_TABLE_BYTES];
        PigmentModel::default().write_gpu_table(&mut band_bytes);
        gpu.queue.write_buffer(&bands, 0, &band_bytes);

        let params_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &params,
            offset: 0,
            size: NonZeroU64::new(PARAMS_BYTES),
        });
        let layout = cache
            .get_bind_group_layout(&Key::CompositeLayout)
            .ok_or(GpuError::Stale { what: "skin composite layout" })?;
        let (planes_store, values_view, coverage_view, inputs_view) = (
            array_view(&planes),
            atlas_values.create_view(&wgpu::TextureViewDescriptor::default()),
            atlas_coverage.create_view(&wgpu::TextureViewDescriptor::default()),
            array_view(&inputs),
        );
        let composite_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skin composite"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&planes_store) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&values_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&coverage_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&inputs_view) },
                wgpu::BindGroupEntry { binding: 4, resource: layer_table.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: layer_tiles.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: globals.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: params_binding.clone() },
            ],
        });
        let layout =
            cache.get_bind_group_layout(&Key::ResolveLayout).ok_or(GpuError::Stale { what: "skin resolve layout" })?;
        let resolve_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skin resolve"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&array_view(&planes)) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&inputs_view) },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&array_view(&resolved)),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&array_view(&specular)),
                },
                wgpu::BindGroupEntry { binding: 4, resource: globals.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: bands.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: params_binding },
            ],
        });

        let mut compositor = Self {
            desc,
            cache,
            planes,
            resolved,
            specular,
            inputs,
            atlas_values,
            atlas_coverage,
            layer_table,
            layer_tiles,
            globals,
            params,
            composite_group,
            resolve_group,
            atlas: ShelfAtlas::new(a),
            layer_words: vec![0; MAX_LAYERS * LAYER_WORDS],
            tile_words: vec![0; tile_count],
            uploaded: vec![Uploaded::default(); MAX_LAYERS],
            layer_count: 0,
            param_bytes: vec![0; MAX_DISPATCH_RECTS * SLOT_BYTES as usize],
            rects: Vec::with_capacity(MAX_TILES as usize * MAX_RECTS_PER_TILE),
            scratch: Vec::new(),
        };
        compositor.set_globals(gpu, &BaseValues::default(), &ZoneTable::default(), &Mapping::DEFAULT);
        // Neutral inputs: half thickness, open cavity, no zone.
        let neutral: Vec<u8> = [128_u8, 255, 0, 0].repeat((size * size) as usize);
        let empty = vec![0_u8; (size * size * 4) as usize];
        for tile in 0..tiles {
            compositor.upload_inputs(gpu, tile, &neutral, [&empty, &empty, &empty])?;
        }
        Ok(compositor)
    }

    /// The set's size.
    pub const fn desc(&self) -> &SkinDesc {
        &self.desc
    }

    /// Pipelines created after [`SkinCompositor::new`] (always 0).
    pub fn late_pipelines(&self) -> u32 {
        self.cache.stats().late
    }

    /// The parameter planes (`rgba16float`, layer `tile * 5 + plane`).
    pub const fn planes(&self) -> &wgpu::Texture {
        &self.planes
    }

    /// The resolved set (`rgba16float`, layer `tile * 4 + k`: albedo, scatter, detail, flush).
    pub const fn resolved(&self) -> &wgpu::Texture {
        &self.resolved
    }

    /// The resolved specular (`rgba8unorm`, layer `tile`).
    pub const fn specular(&self) -> &wgpu::Texture {
        &self.specular
    }

    /// The baked inputs (`rgba8unorm`, layer `tile * 4 + k`: k = 0 thickness, cavity; k = 1..3 the
    /// zone mask slots 0..11), read by the shading pass for zone-dependent layers.
    pub const fn inputs(&self) -> &wgpu::Texture {
        &self.inputs
    }

    /// Writes the base values, the zone table and the pigment mapping. Every tile must be
    /// re-resolved afterwards (`LayerStack::dirty_mut().mark_all()`).
    pub fn set_globals(&mut self, gpu: &Gpu, base: &BaseValues, zones: &ZoneTable, mapping: &Mapping) {
        let mut words = [0.0_f32; GLOBAL_WORDS];
        words[..ATTRIBUTE_COUNT].copy_from_slice(&base.0);
        let zone_words = zones.gpu_words();
        words[ATTRIBUTE_COUNT..ATTRIBUTE_COUNT + zone_words.len()].copy_from_slice(&zone_words);
        let m = ATTRIBUTE_COUNT + zone_words.len();
        let mapping_words = [
            mapping.melanin_max,
            mapping.blood_min,
            mapping.blood_max,
            mapping.epidermis_min_mm,
            mapping.epidermis_max_mm,
            mapping.carotene_max_molar,
            f64::from(zones.variation_melanin),
            f64::from(zones.variation_haemoglobin),
        ];
        for (dst, v) in words[m..m + 8].iter_mut().zip(mapping_words) {
            *dst = v as f32;
        }
        let mut bytes = [0_u8; GLOBAL_WORDS * 4];
        for (chunk, w) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            *chunk = w.to_le_bytes();
        }
        gpu.queue.write_buffer(&self.globals, 0, &bytes);
    }

    /// Uploads the baked inputs of one tile: `baked` RGBA8 (thickness, cavity, -, -) and the three
    /// RGBA8 zone mask layers (zone slots 0..11), each `tile_size²` texels.
    pub fn upload_inputs(&mut self, gpu: &Gpu, tile: u32, baked: &[u8], zones: [&[u8]; 3]) -> Result<(), SkinGpuError> {
        let size = self.desc.tile_size;
        let bytes = (size * size * 4) as usize;
        if tile >= self.desc.tiles || baked.len() != bytes || zones.iter().any(|z| z.len() != bytes) {
            return Err(SkinGpuError::Invalid {
                what: "input layers must be tile_size² RGBA8 texels of an existing tile",
            });
        }
        for (k, data) in std::iter::once(baked).chain(zones).enumerate() {
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.inputs,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: tile * 4 + k as u32 },
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size * 4), rows_per_image: Some(size) },
                wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            );
        }
        Ok(())
    }

    /// Mirrors the layer stack: rebuilds the layer table and uploads the content of every layer
    /// whose version changed (repacking the atlas when it is full). Layers whose content was
    /// already sent through [`SkinCompositor::upload_rect`] are not re-sent.
    pub fn sync_layers(&mut self, gpu: &Gpu, stack: &LayerStack) -> Result<(), SkinGpuError> {
        let layers = stack.layers();
        if layers.len() > MAX_LAYERS {
            return Err(SkinGpuError::Invalid { what: "more layers than the GPU table holds" });
        }
        let mut repack = false;
        for (slot, layer) in layers.iter().enumerate() {
            let fresh = Uploaded { id: layer.id.0, version: layer.version };
            if self.uploaded[slot] != fresh && matches!(layer.kind, LayerKind::Content { .. }) {
                repack = true;
            }
        }
        if repack {
            self.atlas.reset();
            self.tile_words.fill(0);
        }
        for (slot, layer) in layers.iter().enumerate() {
            let mut words = [0_u32; LAYER_WORDS];
            let skipped = !layer.enabled;
            words[0] = layer.blend as u32;
            words[1] = layer.opacity.to_bits();
            words[2] = layer.zone.map_or(NO_ZONE, |z| z.slot() as u32);
            match &layer.kind {
                LayerKind::Fill { attributes, values } => {
                    words[3] = if skipped { 2 } else { 0 };
                    words[4..8].fill(UNUSED);
                    words[8] = attributes.0;
                    for (dst, v) in words[12..32].iter_mut().zip(values) {
                        *dst = v.to_bits();
                    }
                }
                LayerKind::Content { attributes, tiles } => {
                    words[3] = if skipped { 2 } else { 1 };
                    for (dst, a) in words[4..8].iter_mut().zip(attributes) {
                        *dst = a.map_or(UNUSED, |a| a.index() as u32);
                    }
                    if repack {
                        for content in tiles {
                            let placed =
                                self.atlas.allocate(content.rect.w, content.rect.h).ok_or(SkinGpuError::Invalid {
                                    what: "the content atlas is full; use a larger atlas_size",
                                })?;
                            let t = (slot * MAX_TILES as usize + content.tile as usize) * 4;
                            self.tile_words[t] = placed.x | placed.y << 16;
                            self.tile_words[t + 1] = content.rect.x | content.rect.y << 16;
                            self.tile_words[t + 2] = content.rect.w | content.rect.h << 16;
                            self.tile_words[t + 3] = 1;
                            self.write_content(
                                gpu,
                                (placed.x, placed.y),
                                content,
                                Rect::new(0, 0, content.rect.w, content.rect.h),
                            );
                        }
                    }
                }
            }
            self.layer_words[slot * LAYER_WORDS..(slot + 1) * LAYER_WORDS].copy_from_slice(&words);
            self.uploaded[slot] = Uploaded { id: layer.id.0, version: layer.version };
        }
        for stale in &mut self.uploaded[layers.len()..] {
            *stale = Uploaded::default();
        }
        self.layer_count = layers.len() as u32;
        let mut bytes = std::mem::take(&mut self.scratch);
        words_to_bytes(&self.layer_words, &mut bytes);
        gpu.queue.write_buffer(&self.layer_table, 0, &bytes);
        words_to_bytes(&self.tile_words, &mut bytes);
        gpu.queue.write_buffer(&self.layer_tiles, 0, &bytes);
        self.scratch = bytes;
        Ok(())
    }

    /// Uploads only `rect` (tile texels) of one layer's content on `tile` after a paint stroke
    /// `[SKN-02]`, and records the layer's version as current so [`SkinCompositor::sync_layers`]
    /// does not re-send it. The layer must have been synced before.
    pub fn upload_rect(
        &mut self,
        gpu: &Gpu,
        stack: &LayerStack,
        id: LayerId,
        tile: u32,
        rect: Rect,
    ) -> Result<(), SkinGpuError> {
        let slot = stack.position(id).ok_or(SkinGpuError::Invalid { what: "no layer with this id" })?;
        let layer = &stack.layers()[slot];
        if self.uploaded[slot].id != id.0 {
            return Err(SkinGpuError::Invalid { what: "the layer was not synced yet" });
        }
        let content =
            layer.content_of(tile).ok_or(SkinGpuError::Invalid { what: "the layer has no content on this tile" })?;
        let t = (slot * MAX_TILES as usize + tile as usize) * 4;
        if self.tile_words.get(t + 3) != Some(&1) {
            return Err(SkinGpuError::Invalid { what: "the content is not in the atlas" });
        }
        let atlas = (self.tile_words[t] & 0xffff, self.tile_words[t] >> 16);
        let clip = rect.intersect(content.rect);
        if !clip.is_empty() {
            let local = Rect::new(clip.x - content.rect.x, clip.y - content.rect.y, clip.w, clip.h);
            self.write_content(gpu, atlas, content, local);
        }
        self.uploaded[slot].version = layer.version;
        Ok(())
    }

    /// Writes `local` (content-relative texels) of `content` into the atlas at `atlas + local`.
    fn write_content(
        &mut self,
        gpu: &Gpu,
        atlas: (u32, u32),
        content: &humanart_model::skin_appearance::layers::LayerContent,
        local: Rect,
    ) {
        let mut bytes = std::mem::take(&mut self.scratch);
        bytes.clear();
        for y in local.y..local.bottom() {
            for x in local.x..local.right() {
                let i = (y * content.rect.w + x) as usize;
                let v = content.values.get(i).copied().unwrap_or([0.0; 4]);
                for c in v {
                    bytes.extend_from_slice(&humanart_math::f32_to_f16_bits(c).to_le_bytes());
                }
            }
        }
        let origin = wgpu::Origin3d { x: atlas.0 + local.x, y: atlas.1 + local.y, z: 0 };
        let extent = wgpu::Extent3d { width: local.w, height: local.h, depth_or_array_layers: 1 };
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.atlas_values,
                mip_level: 0,
                origin,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(local.w * 8), rows_per_image: Some(local.h) },
            extent,
        );
        bytes.clear();
        for y in local.y..local.bottom() {
            for x in local.x..local.right() {
                let i = (y * content.rect.w + x) as usize;
                let c = content.coverage.get(i).copied().unwrap_or(0.0);
                bytes.extend_from_slice(&humanart_math::f32_to_f16_bits(c).to_le_bytes());
            }
        }
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.atlas_coverage,
                mip_level: 0,
                origin,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(local.w * 2), rows_per_image: Some(local.h) },
            extent,
        );
        self.scratch = bytes;
    }

    /// The CPU half of a resolve `[PERF-R1]`: takes the dirty rectangles of `stack` (clearing
    /// them; coarsened to one rectangle per tile beyond [`MAX_DISPATCH_RECTS`]) and fills the
    /// dispatch parameter mirror. Allocates nothing. [`SkinCompositor::record`] then records them.
    pub fn prepare(&mut self, stack: &mut LayerStack) -> Result<ResolveReport, SkinGpuError> {
        if stack.tile_size() != self.desc.tile_size || stack.dirty().tiles() != self.desc.tiles {
            return Err(SkinGpuError::Invalid { what: "the layer stack does not match the texture set" });
        }
        self.rects.clear();
        let total: usize = (0..self.desc.tiles).map(|t| stack.dirty().rects(t).len()).sum();
        let coarsen = total > MAX_DISPATCH_RECTS;
        if coarsen {
            for tile in 0..self.desc.tiles {
                let union = stack.dirty().rects(tile).iter().fold(Rect::default(), |u, r| u.union(*r));
                if !union.is_empty() {
                    self.rects.push((tile, union));
                }
            }
            stack.dirty_mut().clear();
        } else {
            let rects = &mut self.rects;
            stack.dirty_mut().drain(|tile, rect| rects.push((tile, rect)));
        }
        let mut report = ResolveReport { rects: self.rects.len() as u32, texels: 0, coarsened: coarsen };
        for (i, &(tile, rect)) in self.rects.iter().enumerate() {
            let words = [tile, self.layer_count, rect.x, rect.y, rect.w, rect.h, 0, 0];
            let start = i * SLOT_BYTES as usize;
            let slot = &mut self.param_bytes[start..start + PARAMS_BYTES as usize];
            for (chunk, w) in slot.as_chunks_mut::<4>().0.iter_mut().zip(words) {
                *chunk = w.to_le_bytes();
            }
            report.texels += rect.area();
        }
        Ok(report)
    }

    /// The GPU half: uploads the prepared parameters and records the two dispatches per prepared
    /// rectangle in one compute pass (with `timestamp_writes` around it when given, `[PERF-R6]`).
    /// Records nothing when nothing was prepared.
    pub fn record(
        &self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) -> Result<(), SkinGpuError> {
        if self.rects.is_empty() {
            return Ok(());
        }
        let used = self.rects.len() * SLOT_BYTES as usize;
        gpu.queue.write_buffer(&self.params, 0, &self.param_bytes[..used]);
        let composite = self
            .cache
            .get_compute_pipeline(&Key::Composite)
            .ok_or(GpuError::Stale { what: "skin composite pipeline" })?;
        let resolve =
            self.cache.get_compute_pipeline(&Key::Resolve).ok_or(GpuError::Stale { what: "skin resolve pipeline" })?;
        let mut pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("skin resolve"), timestamp_writes });
        for (i, &(_, rect)) in self.rects.iter().enumerate() {
            let offset = (i as u64 * SLOT_BYTES) as u32;
            let (gx, gy) = (rect.w.div_ceil(GROUP), rect.h.div_ceil(GROUP));
            pass.set_pipeline(composite);
            pass.set_bind_group(0, &self.composite_group, &[offset]);
            pass.dispatch_workgroups(gx, gy, 1);
            pass.set_pipeline(resolve);
            pass.set_bind_group(0, &self.resolve_group, &[offset]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        Ok(())
    }

    /// [`SkinCompositor::prepare`] then [`SkinCompositor::record`]: re-resolves every dirty
    /// rectangle of `stack`. Call [`SkinCompositor::sync_layers`] first when layers changed.
    pub fn resolve(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        stack: &mut LayerStack,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) -> Result<ResolveReport, SkinGpuError> {
        let report = self.prepare(stack)?;
        self.record(gpu, encoder, timestamp_writes)?;
        Ok(report)
    }
}

#[cfg(test)]
mod tests;
