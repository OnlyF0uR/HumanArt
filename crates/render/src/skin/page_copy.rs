//! GPU-to-pool copy of the editor's resolved skin set `[SKN-03]`: streaming pages come straight
//! from the compositor's textures, without a readback.
//!
//! [`ResolvedPages`] binds one source array of [`super::SkinCompositor`] (the resolved set,
//! `rgba16float`, four layers per UV tile, or the specular array, `rgba8unorm`, one layer per UV
//! tile) and one channel of it. [`super::stream::SkinStreaming::update_from_gpu`] lets the pool
//! pick slots for the queued pages as usual, then records, per loaded page, `skin/page_copy.wgsl`
//! (the page's source square box-averaged down to the page's mip) into a page-sized staging
//! texture and a texture copy from it into the pool slot.
//!
//! The pool's own upload path still writes each new slot once (with zeros) when it hands it out;
//! those queue writes run at the start of the next submission, before the command buffers in it,
//! so the copies recorded here overwrite them as long as the encoder goes out in the same
//! submission as (or any submission after) the update — submit it in the frame it was recorded.
//!
//! Everything is created in [`ResolvedPages::new`] (`[PERF-R4]`); per update the parameter block of
//! every copy is written from a preallocated buffer and the request list reuses its capacity
//! (`[PERF-R1]`).

use std::num::NonZeroU64;

use humanart_gpu::cache::PipelineCache;
use humanart_gpu::streaming::TileRequest;
use humanart_gpu::{Gpu, GpuError};

use super::SkinGpuError;

/// Most pages copied in one update.
pub const MAX_COPIES_PER_UPDATE: usize = 64;
/// Bytes of `Copy` in `skin/page_copy.wgsl`.
const COPY_BYTES: u64 = 32;
/// Bytes between two parameter blocks (the minimum uniform offset alignment).
const SLOT_BYTES: u64 = 256;
/// Workgroup edge.
const GROUP: u32 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Key {
    Shader,
    Layout,
    PipelineLayout,
    Pipeline,
}

/// The source of streaming pages: one channel of a compositor texture array.
pub struct ResolvedPages {
    cache: PipelineCache<Key>,
    params: wgpu::Buffer,
    params_bytes: Vec<u8>,
    staging: wgpu::Texture,
    group: wgpu::BindGroup,
    source_size: u32,
    layers_per_tile: u32,
    channel: u32,
    page_size: u32,
    format: wgpu::TextureFormat,
    pub(super) pending: Vec<TileRequest>,
}

const fn entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None }
}

const fn store(format: wgpu::TextureFormat) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        access: wgpu::StorageTextureAccess::WriteOnly,
        format,
        view_dimension: wgpu::TextureViewDimension::D2,
    }
}

fn page_texture(gpu: &Gpu, label: &str, size: u32, format: wgpu::TextureFormat) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

impl ResolvedPages {
    /// Pages of channel `channel` (of `layers_per_tile` layers per UV tile) of `source`, which
    /// must be the compositor's resolved array (`rgba16float`, 4 layers per tile) or specular array
    /// (`rgba8unorm`, 1 layer per tile), with `TEXTURE_BINDING` usage.
    pub fn new(
        gpu: &Gpu,
        source: &wgpu::Texture,
        layers_per_tile: u32,
        channel: u32,
        page_size: u32,
    ) -> Result<Self, SkinGpuError> {
        let format = source.format();
        if !matches!(format, wgpu::TextureFormat::Rgba16Float | wgpu::TextureFormat::Rgba8Unorm) {
            return Err(SkinGpuError::Invalid { what: "streamed skin pages are rgba16float or rgba8unorm" });
        }
        if layers_per_tile == 0
            || channel >= layers_per_tile
            || !source.depth_or_array_layers().is_multiple_of(layers_per_tile)
        {
            return Err(SkinGpuError::Invalid { what: "channel outside the source's layers per tile" });
        }
        if source.width() != source.height() || page_size == 0 || !source.width().is_multiple_of(page_size) {
            return Err(SkinGpuError::Invalid { what: "the source tiles must be a whole number of square pages" });
        }
        if !source.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING) {
            return Err(SkinGpuError::Invalid { what: "the source texture cannot be read by a shader" });
        }
        let mut cache = PipelineCache::new(gpu);
        let source_text = humanart_shaders::compose("skin/page_copy.wgsl", &[])?.into_source();
        cache.shader(Key::Shader, "skin/page_copy.wgsl", &source_text)?;
        let uniform = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: NonZeroU64::new(COPY_BYTES),
        };
        let texture = wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        };
        let (target_binding, entry_point) =
            if format == wgpu::TextureFormat::Rgba16Float { (2, "copy_f16") } else { (3, "copy_unorm8") };
        cache.bind_group_layout(
            Key::Layout,
            "skin page copy",
            &[entry(0, uniform), entry(1, texture), entry(target_binding, store(format))],
        )?;
        cache.pipeline_layout(Key::PipelineLayout, "skin page copy", &[Some(&Key::Layout)])?;
        cache.compute_pipeline(Key::Pipeline, |parts| {
            Ok(parts.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("skin page copy"),
                layout: Some(parts.pipeline_layout(&Key::PipelineLayout)?),
                module: parts.shader(&Key::Shader)?,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            }))
        })?;
        cache.mark_warm();
        let params_size = SLOT_BYTES * MAX_COPIES_PER_UPDATE as u64;
        let params = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skin page copy params"),
            size: params_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = page_texture(gpu, "skin page copy staging", page_size, format);
        let source_view = source.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..wgpu::TextureViewDescriptor::default()
        });
        let staging_view = staging.create_view(&wgpu::TextureViewDescriptor::default());
        let layout = cache.get_bind_group_layout(&Key::Layout).ok_or(GpuError::Stale { what: "page copy layout" })?;
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skin page copy"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params,
                        offset: 0,
                        size: NonZeroU64::new(COPY_BYTES),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&source_view) },
                wgpu::BindGroupEntry {
                    binding: target_binding,
                    resource: wgpu::BindingResource::TextureView(&staging_view),
                },
            ],
        });
        Ok(Self {
            cache,
            params,
            params_bytes: vec![0; params_size as usize],
            staging,
            group,
            source_size: source.width(),
            layers_per_tile,
            channel,
            page_size,
            format,
            pending: Vec::with_capacity(MAX_COPIES_PER_UPDATE),
        })
    }

    /// Pipelines created after [`ResolvedPages::new`] (always 0).
    pub fn late_pipelines(&self) -> u32 {
        self.cache.stats().late
    }

    /// The pool format the pages are in.
    pub const fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Edge of a UV tile of the source, in texels.
    pub const fn source_size(&self) -> u32 {
        self.source_size
    }

    /// Edge of a page, in texels.
    pub const fn page_size(&self) -> u32 {
        self.page_size
    }

    /// Records the copy of every `(page, UV tile, pool slot corner x, y and layer)` (at most
    /// [`MAX_COPIES_PER_UPDATE`]); returns how many were recorded.
    pub(super) fn record(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        pool: &wgpu::Texture,
        targets: impl Iterator<Item = (TileRequest, u32, [u32; 3])>,
    ) -> Result<u32, SkinGpuError> {
        let pipeline =
            self.cache.get_compute_pipeline(&Key::Pipeline).ok_or(GpuError::Stale { what: "page copy pipeline" })?;
        let mut jobs = [(0_u32, [0_u32; 3]); MAX_COPIES_PER_UPDATE];
        let mut count = 0;
        for (request, uv_tile, location) in targets.take(MAX_COPIES_PER_UPDATE) {
            let span = self.page_size << request.mip;
            let words = [
                u32::from(request.x) * span,
                u32::from(request.y) * span,
                uv_tile * self.layers_per_tile + self.channel,
                u32::from(request.mip),
                self.page_size,
                0,
                0,
                0,
            ];
            let at = count * SLOT_BYTES as usize;
            for (chunk, w) in
                self.params_bytes[at..at + COPY_BYTES as usize].as_chunks_mut::<4>().0.iter_mut().zip(words)
            {
                *chunk = w.to_le_bytes();
            }
            jobs[count] = (at as u32, location);
            count += 1;
        }
        if count == 0 {
            return Ok(0);
        }
        gpu.queue.write_buffer(&self.params, 0, &self.params_bytes[..count * SLOT_BYTES as usize]);
        let groups = self.page_size.div_ceil(GROUP);
        for &(offset, [x, y, layer]) in &jobs[..count] {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("skin page copy"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &self.group, &[offset]);
                pass.dispatch_workgroups(groups, groups, 1);
            }
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.staging,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: pool,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: layer },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d { width: self.page_size, height: self.page_size, depth_or_array_layers: 1 },
            );
        }
        Ok(count as u32)
    }
}
