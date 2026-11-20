//! Editor-time texture processing on the GPU: mip generation `[IO-05]` and the BC7 / BC6H compute
//! encoders `[SKN-04]`.
//!
//! [`Compressor`] owns every pipeline of the three shaders under `shaders/texture/` (created once in
//! [`Compressor::new`], `[PERF-R4]`; variants are override constants, `[PERF-R5]`), the partition and
//! mode tables of `humanart_io::bc` in one storage buffer, and a small ring of uniform slots. Work is
//! recorded into the caller's command encoder:
//!
//! - [`Compressor::generate_mips`]: levels `1..n` of a texture from level 0, through an `rgba32float`
//!   working chain ([`create_work_texture`]), following `humanart_io::mip` (exact area filter, sRGB
//!   decoded to linear light, premultiplied alpha, normal maps renormalised with the roughness in
//!   alpha widened by the normal variance).
//! - [`Compressor::encode_bc7`] / [`Compressor::encode_bc6h`]: one level into a storage buffer of
//!   16-byte blocks, in the block order `humanart_io::bc::decode` reads.
//!
//! The blocking helpers in [`offline`] (native only, for the CLI, the exporter and tests) upload an
//! [`humanart_io::Image`], run the chain and read the blocks back.
//!
//! These are editor-time operations, not per-frame paths: bind groups reference the caller's
//! textures and are created per call. At most [`PARAM_SLOTS`] dispatches may be recorded between
//! [`Compressor::begin`] and the submission (each dispatch's parameters live in their own slot).

#[cfg(not(target_arch = "wasm32"))]
pub mod offline;

use std::fmt;
use std::num::NonZeroU64;

use humanart_gpu::cache::PipelineCache;
use humanart_gpu::profiler::GpuProfiler;
use humanart_gpu::{Gpu, GpuError};
use humanart_io::ColorSpace;
use humanart_io::bc::{bc6h, tables};
use humanart_shaders::ShaderError;

/// Uniform slots, one per dispatch recorded between [`Compressor::begin`] and the submission.
pub const PARAM_SLOTS: u32 = 64;
/// Bytes per uniform slot (the minimum uniform offset alignment wgpu guarantees).
const SLOT_BYTES: u64 = 256;
/// Workgroup edge of every texture shader (8 × 8 invocations).
const GROUP: u32 = 8;

/// Why a texture operation failed.
#[derive(Debug)]
pub enum CompressError {
    /// The device refused a resource or a pipeline.
    Gpu(GpuError),
    /// A shader could not be composed.
    Shader(ShaderError),
    /// A codec or container error (image conversion, KTX2).
    Io(humanart_io::Error),
    /// The request is malformed (sizes, formats, too many dispatches).
    Invalid {
        /// What is wrong.
        what: &'static str,
    },
}

impl fmt::Display for CompressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gpu(e) => write!(f, "texture compression: {e}"),
            Self::Shader(e) => write!(f, "texture compression shader: {e}"),
            Self::Io(e) => write!(f, "texture compression: {e}"),
            Self::Invalid { what } => write!(f, "texture compression: {what}"),
        }
    }
}

impl std::error::Error for CompressError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gpu(e) => Some(e),
            Self::Shader(e) => Some(e),
            Self::Io(e) => Some(e),
            Self::Invalid { .. } => None,
        }
    }
}

impl From<GpuError> for CompressError {
    fn from(e: GpuError) -> Self {
        Self::Gpu(e)
    }
}

impl From<ShaderError> for CompressError {
    fn from(e: ShaderError) -> Self {
        Self::Shader(e)
    }
}

impl From<humanart_io::Error> for CompressError {
    fn from(e: humanart_io::Error) -> Self {
        Self::Io(e)
    }
}

/// Encoder effort, the `QUALITY` override of the encoders.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Quality {
    /// BC7 mode 6 (and 5 with alpha); BC6H one-region modes.
    Fast,
    /// Adds the best two-subset partition (BC7 modes 1, 3, 7; BC6H modes 0 to 9).
    Normal,
    /// Adds a second partition, BC7 three-subset modes and every rotation, two refinement passes.
    High,
}

impl Quality {
    /// Every level.
    pub const ALL: [Self; 3] = [Self::Fast, Self::Normal, Self::High];

    /// The override value.
    const fn value(self) -> f64 {
        match self {
            Self::Fast => 0.0,
            Self::Normal => 1.0,
            Self::High => 2.0,
        }
    }
}

/// What a mip chain holds: the `COLOR_SPACE` override of `texture/mip.wgsl`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MipContent {
    /// Linear colour, premultiplied by alpha while filtering.
    Linear,
    /// sRGB-encoded colour: filtered in linear light, premultiplied by alpha.
    Srgb,
    /// Data: every channel as stored.
    Data,
    /// Normal map packed as RGB = `n · 0.5 + 0.5`, A = perceptual roughness (8-bit textures only).
    NormalRoughness,
}

impl MipContent {
    /// Every content kind.
    pub const ALL: [Self; 4] = [Self::Linear, Self::Srgb, Self::Data, Self::NormalRoughness];

    /// The content of an image with this colour space (`Normal` images must carry roughness in
    /// alpha).
    pub const fn from_color_space(space: ColorSpace) -> Self {
        match space {
            ColorSpace::Srgb => Self::Srgb,
            ColorSpace::Linear => Self::Linear,
            ColorSpace::Normal => Self::NormalRoughness,
            ColorSpace::Data => Self::Data,
        }
    }

    const fn value(self) -> f64 {
        match self {
            Self::Linear => 0.0,
            Self::Srgb => 1.0,
            Self::Data => 2.0,
            Self::NormalRoughness => 3.0,
        }
    }
}

/// BC7 encoder settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bc7Options {
    /// Effort.
    pub quality: Quality,
    /// Error weight per channel (RGBA). Use 0 for a channel the content does not use.
    pub weights: [f32; 4],
}

impl Default for Bc7Options {
    fn default() -> Self {
        Self { quality: Quality::Normal, weights: [1.0; 4] }
    }
}

impl Bc7Options {
    /// Settings for a normal map (RGB = `n · 0.5 + 0.5`, A = roughness) whose shader reconstructs
    /// `z = sqrt(1 − x² − y²)` from x and y: blue carries no error weight, so the endpoint and
    /// index precision goes to x, y and the roughness. The stored blue is whatever the fit gives
    /// and must not be read.
    pub const fn normal_xy(quality: Quality) -> Self {
        Self { quality, weights: [1.0, 1.0, 0.0, 1.0] }
    }
}

/// BC6H encoder settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bc6hOptions {
    /// Effort.
    pub quality: Quality,
    /// `BC6H_SF16` instead of `BC6H_UF16`.
    pub signed: bool,
}

impl Default for Bc6hOptions {
    fn default() -> Self {
        Self { quality: Quality::Normal, signed: false }
    }
}

/// One level to encode.
#[derive(Clone, Copy, Debug)]
pub struct BcJob<'a> {
    /// A single-level view of the source: `Rgba8Unorm` (stored values) for BC7, a float format
    /// for BC6H.
    pub source: &'a wgpu::TextureView,
    /// The level's width in texels.
    pub width: u32,
    /// The level's height in texels.
    pub height: u32,
    /// The output storage buffer (`STORAGE` usage), 16 bytes per block.
    pub out: &'a wgpu::Buffer,
    /// The first block written, in 16-byte units.
    pub out_block_offset: u32,
}

/// A mip chain to fill.
#[derive(Clone, Copy, Debug)]
pub struct MipJob<'a> {
    /// The texture: level 0 filled, `TEXTURE_BINDING | STORAGE_BINDING` usage, `Rgba8Unorm` or
    /// `Rgba16Float`, at least 2 levels.
    pub texture: &'a wgpu::Texture,
    /// The working chain from [`create_work_texture`] for the texture's size.
    pub work: &'a wgpu::Texture,
    /// What the texture holds.
    pub content: MipContent,
}

/// Cache keys of the module's GPU objects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Key {
    Bc7Shader,
    Bc6hShader,
    MipShader,
    BcLayout,
    MipUnormLayout,
    MipFloatLayout,
    BcPipelineLayout,
    MipUnormPipelineLayout,
    MipFloatPipelineLayout,
    Bc7(Quality),
    Bc6h(bool, Quality),
    Mip(MipContent, bool, bool),
}

/// One pipeline to pre-warm: its key, shader, pipeline layout, entry point, override constants and
/// how many of them are used.
type Variant = (Key, Key, Key, &'static str, [(&'static str, f64); 2], usize);

/// The profiler scope of a BC7 encoder pass.
pub const SCOPE_BC7: &str = "bc7 encode";
/// The profiler scope of a BC6H encoder pass.
pub const SCOPE_BC6H: &str = "bc6h encode";
/// The profiler scope of one mip level's pass.
pub const SCOPE_MIP: &str = "mip level";

/// The GPU texture processor: pipelines, tables and parameter slots, created once.
pub struct Compressor {
    cache: PipelineCache<Key>,
    tables: wgpu::Buffer,
    params: wgpu::Buffer,
    next_slot: u32,
}

/// Word offset of the BC7 single-colour table in [`table_bytes`] (`SINGLE_BASE` in
/// `texture/bc7_encode.wgsl`).
const SINGLE_COLOR_BASE: usize = tables::GPU_WORDS + 14 * bc6h::GPU_WORDS_PER_MODE;

/// The table words: the BC7 partition tables, then the BC6H mode table, then the BC7 optimal
/// single-colour table (at [`SINGLE_COLOR_BASE`]).
fn table_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity((SINGLE_COLOR_BASE + tables::SINGLE_COLOR_GPU_WORDS) * 4);
    for word in
        tables::gpu_words().iter().chain(bc6h::gpu_words().iter()).chain(tables::single_color_gpu_words().iter())
    {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// A storage-buffer or texture layout entry visible to compute.
const fn entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None }
}

const SOURCE_TEXTURE: wgpu::BindingType = wgpu::BindingType::Texture {
    sample_type: wgpu::TextureSampleType::Float { filterable: false },
    view_dimension: wgpu::TextureViewDimension::D2,
    multisampled: false,
};

const fn storage_texture(format: wgpu::TextureFormat) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        access: wgpu::StorageTextureAccess::WriteOnly,
        format,
        view_dimension: wgpu::TextureViewDimension::D2,
    }
}

const fn uniform(size: u64) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: true,
        min_binding_size: NonZeroU64::new(size),
    }
}

const fn storage(read_only: bool) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

/// Bytes of the encoder parameters (`Params` in the encoders).
const BC_PARAMS: u64 = 48;
/// Bytes of the mip parameters.
const MIP_PARAMS: u64 = 16;

/// Creates the working chain for mip generation of a `width × height` texture: `rgba32float`, the
/// size of level 1, one level per generated level.
pub fn create_work_texture(gpu: &Gpu, width: u32, height: u32) -> Result<wgpu::Texture, CompressError> {
    let levels = humanart_io::mip::level_count(width, height);
    if levels < 2 {
        return Err(CompressError::Invalid { what: "a 1 x 1 texture has no mips to generate" });
    }
    let (w, h) = humanart_io::mip::level_size(width, height, 1);
    Ok(gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mip working chain"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: levels - 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    }))
}

/// A single-level view of `texture`.
fn level_view(texture: &wgpu::Texture, level: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("texture level"),
        base_mip_level: level,
        mip_level_count: Some(1),
        ..Default::default()
    })
}

impl Compressor {
    /// Creates every pipeline (3 BC7, 6 BC6H, 16 mip variants), the table buffer and the uniform
    /// slots. Call once at start-up or before the first export; nothing is created later.
    pub fn new(gpu: &Gpu) -> Result<Self, CompressError> {
        let mut cache = PipelineCache::new(gpu);
        for (key, path) in [
            (Key::Bc7Shader, "texture/bc7_encode.wgsl"),
            (Key::Bc6hShader, "texture/bc6h_encode.wgsl"),
            (Key::MipShader, "texture/mip.wgsl"),
        ] {
            let source = humanart_shaders::compose(path, &[])?.into_source();
            cache.shader(key, path, &source)?;
        }
        cache.bind_group_layout(
            Key::BcLayout,
            "bc encoder",
            &[
                entry(0, SOURCE_TEXTURE),
                entry(1, storage(false)),
                entry(2, storage(true)),
                entry(3, uniform(BC_PARAMS)),
            ],
        )?;
        cache.bind_group_layout(
            Key::MipUnormLayout,
            "mip unorm",
            &[
                entry(0, SOURCE_TEXTURE),
                entry(1, storage_texture(wgpu::TextureFormat::Rgba32Float)),
                entry(2, storage_texture(wgpu::TextureFormat::Rgba8Unorm)),
                entry(4, uniform(MIP_PARAMS)),
            ],
        )?;
        cache.bind_group_layout(
            Key::MipFloatLayout,
            "mip float",
            &[
                entry(0, SOURCE_TEXTURE),
                entry(1, storage_texture(wgpu::TextureFormat::Rgba32Float)),
                entry(3, storage_texture(wgpu::TextureFormat::Rgba16Float)),
                entry(4, uniform(MIP_PARAMS)),
            ],
        )?;
        cache.pipeline_layout(Key::BcPipelineLayout, "bc encoder", &[Some(&Key::BcLayout)])?;
        cache.pipeline_layout(Key::MipUnormPipelineLayout, "mip unorm", &[Some(&Key::MipUnormLayout)])?;
        cache.pipeline_layout(Key::MipFloatPipelineLayout, "mip float", &[Some(&Key::MipFloatLayout)])?;

        let mut variants: Vec<Variant> = Vec::new();
        for q in Quality::ALL {
            variants.push((
                Key::Bc7(q),
                Key::Bc7Shader,
                Key::BcPipelineLayout,
                "encode",
                [("QUALITY", q.value()), ("", 0.0)],
                1,
            ));
            for signed in [false, true] {
                variants.push((
                    Key::Bc6h(signed, q),
                    Key::Bc6hShader,
                    Key::BcPipelineLayout,
                    "encode",
                    [("SIGNED", f64::from(u8::from(signed))), ("QUALITY", q.value())],
                    2,
                ));
            }
        }
        for content in MipContent::ALL {
            for from_source in [false, true] {
                for float in [false, true] {
                    let (layout, entry_point) = if float {
                        (Key::MipFloatPipelineLayout, "reduce_float")
                    } else {
                        (Key::MipUnormPipelineLayout, "reduce_unorm")
                    };
                    variants.push((
                        Key::Mip(content, from_source, float),
                        Key::MipShader,
                        layout,
                        entry_point,
                        [("COLOR_SPACE", content.value()), ("FROM_SOURCE", f64::from(u8::from(from_source)))],
                        2,
                    ));
                }
            }
        }
        for (key, shader, layout, entry_point, constants, count) in variants {
            cache.compute_pipeline(key, |parts| {
                Ok(parts.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("texture processing"),
                    layout: Some(parts.pipeline_layout(&layout)?),
                    module: parts.shader(&shader)?,
                    entry_point: Some(entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants[..count],
                        zero_initialize_workgroup_memory: false,
                    },
                    cache: None,
                }))
            })?;
        }
        cache.mark_warm();

        let table = table_bytes();
        let tables = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bc tables"),
            size: table.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&tables, 0, &table);
        let params = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("texture processing parameters"),
            size: u64::from(PARAM_SLOTS) * SLOT_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self { cache, tables, params, next_slot: 0 })
    }

    /// Pipelines created after [`Compressor::new`] (always 0: every variant is pre-warmed).
    pub fn late_pipelines(&self) -> u32 {
        self.cache.stats().late
    }

    /// Starts a batch of dispatches for one submission: frees every parameter slot.
    pub fn begin(&mut self) {
        self.next_slot = 0;
    }

    /// Writes `words` into the next parameter slot and returns its offset.
    fn push_params(&mut self, gpu: &Gpu, words: &[u32]) -> Result<u32, CompressError> {
        if self.next_slot >= PARAM_SLOTS {
            return Err(CompressError::Invalid { what: "more dispatches than parameter slots before the submission" });
        }
        let mut bytes = [0_u8; 64];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        let offset = u64::from(self.next_slot) * SLOT_BYTES;
        gpu.queue.write_buffer(&self.params, offset, &bytes[..words.len().min(16) * 4]);
        self.next_slot += 1;
        Ok(offset as u32)
    }

    /// Checks a BC job and returns the block grid.
    fn check_job(gpu: &Gpu, job: &BcJob<'_>) -> Result<(u32, u32), CompressError> {
        if job.width == 0 || job.height == 0 {
            return Err(CompressError::Invalid { what: "an empty level" });
        }
        let (bx, by) = humanart_io::bc::block_dims(job.width, job.height);
        let end = (u64::from(job.out_block_offset) + u64::from(bx) * u64::from(by)) * 16;
        if end > job.out.size() {
            return Err(CompressError::Invalid { what: "the output buffer is too small for the level" });
        }
        if job.out.size() > gpu.limits().max_storage_buffer_binding_size {
            return Err(CompressError::Invalid { what: "the output buffer exceeds the storage binding limit" });
        }
        let max = gpu.limits().max_compute_workgroups_per_dimension;
        if bx.div_ceil(GROUP) > max || by.div_ceil(GROUP) > max {
            return Err(CompressError::Invalid { what: "the level is too large for one dispatch" });
        }
        Ok((bx, by))
    }

    /// Records one encoder dispatch, timed under `scope` when a profiler is given.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_bc(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        job: &BcJob<'_>,
        pipeline: Key,
        weights: [f32; 4],
        profiler: Option<&mut GpuProfiler>,
        scope: &'static str,
    ) -> Result<(), CompressError> {
        let (bx, by) = Self::check_job(gpu, job)?;
        let w = weights.map(f32::to_bits);
        let offset = self.push_params(
            gpu,
            &[bx, by, job.width, job.height, job.out_block_offset, 0, 0, 0, w[0], w[1], w[2], w[3]],
        )?;
        let layout =
            self.cache.get_bind_group_layout(&Key::BcLayout).ok_or(GpuError::Stale { what: "bc bind-group layout" })?;
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bc encoder"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(job.source) },
                wgpu::BindGroupEntry { binding: 1, resource: job.out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.tables.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.params,
                        offset: 0,
                        size: NonZeroU64::new(BC_PARAMS),
                    }),
                },
            ],
        });
        let pipeline =
            self.cache.get_compute_pipeline(&pipeline).ok_or(GpuError::Stale { what: "bc encoder pipeline" })?;
        // The scope is taken last, after everything that can fail, so a timed scope always has its pass.
        let timestamp_writes = profiler.and_then(|p| p.compute_pass_writes(scope));
        let mut pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("bc encode"), timestamp_writes });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[offset]);
        pass.dispatch_workgroups(bx.div_ceil(GROUP), by.div_ceil(GROUP), 1);
        Ok(())
    }

    /// Records the BC7 encoding of one level.
    pub fn encode_bc7(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        job: &BcJob<'_>,
        options: &Bc7Options,
    ) -> Result<(), CompressError> {
        self.bc7(gpu, encoder, job, options, None)
    }

    /// [`Compressor::encode_bc7`] with the pass timed by `profiler` under the scope
    /// [`SCOPE_BC7`] `[PERF-R6]` (timestamp queries where the adapter has them; the times arrive
    /// through [`GpuProfiler::collect`] a few frames later, nothing waits for them).
    pub fn encode_bc7_profiled(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        job: &BcJob<'_>,
        options: &Bc7Options,
        profiler: &mut GpuProfiler,
    ) -> Result<(), CompressError> {
        self.bc7(gpu, encoder, job, options, Some(profiler))
    }

    fn bc7(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        job: &BcJob<'_>,
        options: &Bc7Options,
        profiler: Option<&mut GpuProfiler>,
    ) -> Result<(), CompressError> {
        if options.weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(CompressError::Invalid { what: "BC7 channel weights must be finite and non-negative" });
        }
        self.dispatch_bc(gpu, encoder, job, Key::Bc7(options.quality), options.weights, profiler, SCOPE_BC7)
    }

    /// Records the BC6H encoding of one level.
    pub fn encode_bc6h(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        job: &BcJob<'_>,
        options: &Bc6hOptions,
    ) -> Result<(), CompressError> {
        self.dispatch_bc(gpu, encoder, job, Key::Bc6h(options.signed, options.quality), [1.0; 4], None, SCOPE_BC6H)
    }

    /// [`Compressor::encode_bc6h`] with the pass timed by `profiler` under the scope
    /// [`SCOPE_BC6H`] `[PERF-R6]`.
    pub fn encode_bc6h_profiled(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        job: &BcJob<'_>,
        options: &Bc6hOptions,
        profiler: &mut GpuProfiler,
    ) -> Result<(), CompressError> {
        let key = Key::Bc6h(options.signed, options.quality);
        self.dispatch_bc(gpu, encoder, job, key, [1.0; 4], Some(profiler), SCOPE_BC6H)
    }

    /// Records the generation of levels `1..` of `job.texture` from its level 0.
    pub fn generate_mips(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        job: &MipJob<'_>,
    ) -> Result<(), CompressError> {
        self.mips(gpu, encoder, job, None)
    }

    /// [`Compressor::generate_mips`] with every level's pass timed by `profiler` under the scope
    /// [`SCOPE_MIP`] `[PERF-R6]` (one scope per level, so the profiler needs that many free).
    pub fn generate_mips_profiled(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        job: &MipJob<'_>,
        profiler: &mut GpuProfiler,
    ) -> Result<(), CompressError> {
        self.mips(gpu, encoder, job, Some(profiler))
    }

    fn mips(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        job: &MipJob<'_>,
        mut profiler: Option<&mut GpuProfiler>,
    ) -> Result<(), CompressError> {
        let texture = job.texture;
        let (width, height) = (texture.width(), texture.height());
        let float = match texture.format() {
            wgpu::TextureFormat::Rgba8Unorm => false,
            wgpu::TextureFormat::Rgba16Float => true,
            _ => return Err(CompressError::Invalid { what: "mips need an Rgba8Unorm or Rgba16Float texture" }),
        };
        if float && job.content == MipContent::NormalRoughness {
            return Err(CompressError::Invalid { what: "normal maps are generated in Rgba8Unorm textures" });
        }
        let levels = texture.mip_level_count();
        if levels < 2 || levels > humanart_io::mip::level_count(width, height) {
            return Err(CompressError::Invalid { what: "the texture's mip level count does not fit its size" });
        }
        let level1 = humanart_io::mip::level_size(width, height, 1);
        if job.work.format() != wgpu::TextureFormat::Rgba32Float
            || (job.work.width(), job.work.height()) != level1
            || job.work.mip_level_count() < levels - 1
        {
            return Err(CompressError::Invalid { what: "the working chain does not match the texture" });
        }
        if levels as usize - 1 > (PARAM_SLOTS - self.next_slot) as usize {
            return Err(CompressError::Invalid { what: "more dispatches than parameter slots before the submission" });
        }
        let layout_key = if float { Key::MipFloatLayout } else { Key::MipUnormLayout };
        for level in 1..levels {
            let (sw, sh) = humanart_io::mip::level_size(width, height, level - 1);
            let (dw, dh) = humanart_io::mip::level_size(width, height, level);
            let offset = self.push_params(gpu, &[sw, sh, dw, dh])?;
            let from_source = level == 1;
            let src = if from_source { level_view(texture, 0) } else { level_view(job.work, level - 2) };
            let work = level_view(job.work, level - 1);
            let out = level_view(texture, level);
            let layout = self
                .cache
                .get_bind_group_layout(&layout_key)
                .ok_or(GpuError::Stale { what: "mip bind-group layout" })?;
            let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mip level"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&work) },
                    wgpu::BindGroupEntry {
                        binding: if float { 3 } else { 2 },
                        resource: wgpu::BindingResource::TextureView(&out),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.params,
                            offset: 0,
                            size: NonZeroU64::new(MIP_PARAMS),
                        }),
                    },
                ],
            });
            let pipeline = self
                .cache
                .get_compute_pipeline(&Key::Mip(job.content, from_source, float))
                .ok_or(GpuError::Stale { what: "mip pipeline" })?;
            let timestamp_writes = profiler.as_deref_mut().and_then(|p| p.compute_pass_writes(SCOPE_MIP));
            let mut pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("mip level"), timestamp_writes });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[offset]);
            pass.dispatch_workgroups(dw.div_ceil(GROUP), dh.div_ceil(GROUP), 1);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
