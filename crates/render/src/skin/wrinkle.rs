//! Wrinkle and tension map blending on the GPU `[SKN-30]`: the baked wrinkle maps of
//! `humanart_model::skin_appearance::wrinkle`, the per-driver activations computed from the
//! deformer's region strain `[EVAL-06]` without a readback, and the blend over a rectangle.
//!
//! Per frame the caller writes the expression part of every driver's fold
//! ([`WrinkleGpu::set_terms`], allocation-free) and records [`WrinkleGpu::record_activate`] after the
//! deformer's strain passes: `skin/wrinkle_drivers.wgsl` `activate` reads the deformer's
//! `region_strain` buffer (bound once by [`WrinkleGpu::bind_strain`]) and writes one `(fold, stretch)`
//! per driver into [`WrinkleGpu::activations`]. The skin shading pass includes `skin/wrinkle.wgsl`,
//! binds [`WrinkleGpu::maps`] (three `rgba16float` layers per wrinkle tile) and the activations,
//! finds the tile's wrinkle slot in [`WrinkleGpu::tile_slots`] (`u32::MAX`: no maps), and calls
//! `wrinkle_fetch`, `wrinkle_driver` and `wrinkle_blend` per pixel: no re-resolve of the skin
//! texture set when the face moves. [`WrinkleGpu::blend_rect`] evaluates the same blend over a
//! rectangle into a buffer (inspection readback and the CPU twin test).
//!
//! Pipelines, layouts and the per-frame buffers are created in [`WrinkleGpu::new`] (`[PERF-R4]`).

use std::num::NonZeroU64;

use humanart_gpu::cache::PipelineCache;
use humanart_gpu::{Gpu, GpuError};
use humanart_math::f32_to_f16_bits;
use humanart_model::skin_appearance::dirty::{MAX_TILES, Rect};
use humanart_model::skin_appearance::wrinkle::{
    DriverTable, MAX_WRINKLE_DRIVERS, MAX_WRINKLE_SIZE, MAX_WRINKLE_TILES, WRINKLE_LAYERS, WrinkleMaps,
};

use super::SkinGpuError;

/// Bytes of `Params` in `skin/wrinkle_drivers.wgsl`.
const PARAMS_BYTES: u64 = 32;
/// Bytes of one driver record.
const DRIVER_BYTES: u64 = 16;
/// Bytes of one activation.
const ACTIVATION_BYTES: u64 = 8;
/// Bytes of the blended state of one texel (two vec4).
pub const BLEND_TEXEL_BYTES: u64 = 32;
/// Threads per workgroup of `activate`.
const ACTIVATE_GROUP: u32 = 64;
/// Workgroup edge of `blend`.
const BLEND_GROUP: u32 = 8;
/// "No wrinkle maps" in the tile-slot table.
pub const NO_WRINKLE_SLOT: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Key {
    Shader,
    Params,
    Activate,
    Blend,
    ActivateLayout,
    BlendLayout,
    ActivatePipeline,
    BlendPipeline,
}

/// The size of a character's wrinkle set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WrinkleDesc {
    /// Edge of a wrinkle tile in texels.
    pub size: u32,
    /// How many UV tiles may carry wrinkle maps.
    pub tiles: u32,
    /// How many drivers the buffers hold.
    pub drivers: u32,
}

/// The wrinkle maps, driver table and activations of one character on the GPU.
pub struct WrinkleGpu {
    desc: WrinkleDesc,
    cache: PipelineCache<Key>,
    params: wgpu::Buffer,
    drivers: wgpu::Buffer,
    terms: wgpu::Buffer,
    activations: wgpu::Buffer,
    tile_slots: wgpu::Buffer,
    maps: wgpu::Texture,
    maps_view: wgpu::TextureView,
    params_group: wgpu::BindGroup,
    strain_group: Option<wgpu::BindGroup>,
    driver_count: u32,
    region_count: u32,
    slots: [u32; MAX_TILES as usize],
    terms_bytes: Vec<u8>,
}

const fn entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None }
}

const fn storage(read_only: bool) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

fn buffer(gpu: &Gpu, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16).next_multiple_of(16),
        usage,
        mapped_at_creation: false,
    })
}

impl WrinkleGpu {
    /// Creates the pipelines, the maps texture and the buffers.
    pub fn new(gpu: &Gpu, desc: WrinkleDesc) -> Result<Self, SkinGpuError> {
        if desc.size == 0 || desc.size > MAX_WRINKLE_SIZE || desc.size > gpu.limits().max_texture_dimension_2d {
            return Err(SkinGpuError::Invalid { what: "wrinkle tile size outside the limits" });
        }
        if desc.tiles == 0 || desc.tiles as usize > MAX_WRINKLE_TILES {
            return Err(SkinGpuError::Invalid { what: "wrinkle tile count outside 1..=MAX_WRINKLE_TILES" });
        }
        if desc.drivers as usize > MAX_WRINKLE_DRIVERS {
            return Err(SkinGpuError::Invalid { what: "more wrinkle drivers than MAX_WRINKLE_DRIVERS" });
        }
        let mut cache = PipelineCache::new(gpu);
        let source = humanart_shaders::compose("skin/wrinkle_drivers.wgsl", &[])?.into_source();
        cache.shader(Key::Shader, "skin/wrinkle_drivers.wgsl", &source)?;
        let uniform = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(PARAMS_BYTES),
        };
        let texture = wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        };
        cache.bind_group_layout(Key::Params, "wrinkle params", &[entry(0, uniform)])?;
        cache.bind_group_layout(
            Key::Activate,
            "wrinkle activate",
            &[entry(0, storage(true)), entry(1, storage(true)), entry(2, storage(true)), entry(3, storage(false))],
        )?;
        cache.bind_group_layout(
            Key::Blend,
            "wrinkle blend",
            &[entry(4, texture), entry(5, storage(true)), entry(6, storage(false))],
        )?;
        cache.pipeline_layout(Key::ActivateLayout, "wrinkle activate", &[Some(&Key::Params), Some(&Key::Activate)])?;
        cache.pipeline_layout(Key::BlendLayout, "wrinkle blend", &[Some(&Key::Params), Some(&Key::Blend)])?;
        for (key, layout, name) in
            [(Key::ActivatePipeline, Key::ActivateLayout, "activate"), (Key::BlendPipeline, Key::BlendLayout, "blend")]
        {
            cache.compute_pipeline(key, |parts| {
                Ok(parts.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("skin wrinkle"),
                    layout: Some(parts.pipeline_layout(&layout)?),
                    module: parts.shader(&Key::Shader)?,
                    entry_point: Some(name),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                }))
            })?;
        }
        cache.mark_warm();
        let copy = wgpu::BufferUsages::COPY_DST;
        let params = buffer(gpu, "wrinkle params", PARAMS_BYTES, wgpu::BufferUsages::UNIFORM | copy);
        let drivers_capacity = u64::from(desc.drivers.max(1));
        let drivers =
            buffer(gpu, "wrinkle drivers", drivers_capacity * DRIVER_BYTES, wgpu::BufferUsages::STORAGE | copy);
        let terms = buffer(gpu, "wrinkle terms", drivers_capacity * 4, wgpu::BufferUsages::STORAGE | copy);
        let activations = buffer(
            gpu,
            "wrinkle activations",
            drivers_capacity * ACTIVATION_BYTES,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | copy,
        );
        gpu.queue.write_buffer(&activations, 0, &vec![0; (drivers_capacity * ACTIVATION_BYTES) as usize]);
        let tile_slots =
            buffer(gpu, "wrinkle tile slots", u64::from(MAX_TILES) * 4, wgpu::BufferUsages::STORAGE | copy);
        let slots = [NO_WRINKLE_SLOT; MAX_TILES as usize];
        let slot_bytes: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
        gpu.queue.write_buffer(&tile_slots, 0, &slot_bytes);
        let maps = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("wrinkle maps"),
            size: wgpu::Extent3d {
                width: desc.size,
                height: desc.size,
                depth_or_array_layers: desc.tiles * WRINKLE_LAYERS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let maps_view = maps.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..wgpu::TextureViewDescriptor::default()
        });
        let params_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wrinkle params"),
            layout: cache.get_bind_group_layout(&Key::Params).ok_or(GpuError::Stale { what: "wrinkle layout" })?,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() }],
        });
        let out = Self {
            desc,
            cache,
            params,
            drivers,
            terms,
            activations,
            tile_slots,
            maps,
            maps_view,
            params_group,
            strain_group: None,
            driver_count: 0,
            region_count: 0,
            slots,
            terms_bytes: vec![0; desc.drivers as usize * 4],
        };
        out.write_params(gpu, 0, Rect::new(0, 0, 0, 0));
        Ok(out)
    }

    /// Pipelines created after [`WrinkleGpu::new`] (always 0).
    pub fn late_pipelines(&self) -> u32 {
        self.cache.stats().late
    }

    /// The baked maps: `rgba16float`, array layer `3k + j` for wrinkle slot `k`.
    pub const fn maps(&self) -> &wgpu::Texture {
        &self.maps
    }

    /// A `D2Array` view of [`WrinkleGpu::maps`].
    pub const fn maps_view(&self) -> &wgpu::TextureView {
        &self.maps_view
    }

    /// One `vec2<f32>` (fold, stretch) per driver.
    pub const fn activations(&self) -> &wgpu::Buffer {
        &self.activations
    }

    /// One `u32` per UV tile: its wrinkle slot, or [`NO_WRINKLE_SLOT`].
    pub const fn tile_slots(&self) -> &wgpu::Buffer {
        &self.tile_slots
    }

    /// The wrinkle slot of UV tile `tile`.
    pub fn tile_slot(&self, tile: u32) -> Option<u32> {
        self.slots.get(tile as usize).copied().filter(|&s| s != NO_WRINKLE_SLOT)
    }

    /// How many drivers are set.
    pub const fn driver_count(&self) -> u32 {
        self.driver_count
    }

    fn write_params(&self, gpu: &Gpu, slot: u32, rect: Rect) {
        let words = [self.driver_count, self.region_count, slot, 0, rect.x, rect.y, rect.w, rect.h];
        let mut bytes = [0_u8; PARAMS_BYTES as usize];
        for (chunk, w) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            *chunk = w.to_le_bytes();
        }
        gpu.queue.write_buffer(&self.params, 0, &bytes);
    }

    /// Uploads the driver table (editor-time: when the drivers change).
    pub fn set_drivers(&mut self, gpu: &Gpu, table: &DriverTable) -> Result<(), SkinGpuError> {
        if table.len() > self.desc.drivers as usize {
            return Err(SkinGpuError::Invalid { what: "more drivers than the wrinkle buffers hold" });
        }
        let mut records = vec![[0_u32; 4]; table.len()];
        table.gpu_words(&mut records)?;
        let bytes: Vec<u8> = records.iter().flatten().flat_map(|w| w.to_le_bytes()).collect();
        if !bytes.is_empty() {
            gpu.queue.write_buffer(&self.drivers, 0, &bytes);
        }
        self.driver_count = table.len() as u32;
        self.write_params(gpu, 0, Rect::new(0, 0, 0, 0));
        Ok(())
    }

    /// Binds the deformer's per-region strain buffer (`vec2<f32>` (stretch, compression) per
    /// region, `region_count` of them). Call once when that buffer is created.
    pub fn bind_strain(
        &mut self,
        gpu: &Gpu,
        region_strain: &wgpu::Buffer,
        region_count: u32,
    ) -> Result<(), SkinGpuError> {
        if u64::from(region_count) * 8 > region_strain.size() {
            return Err(SkinGpuError::Invalid { what: "the region strain buffer is smaller than its region count" });
        }
        let layout =
            self.cache.get_bind_group_layout(&Key::Activate).ok_or(GpuError::Stale { what: "wrinkle layout" })?;
        self.strain_group = Some(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wrinkle activate"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.drivers.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.terms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: region_strain.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.activations.as_entire_binding() },
            ],
        }));
        self.region_count = region_count;
        self.write_params(gpu, 0, Rect::new(0, 0, 0, 0));
        Ok(())
    }

    /// Writes this frame's expression terms (one per driver, `DriverTable::expression_terms`).
    /// Allocation-free (`[PERF-R1]`).
    pub fn set_terms(&mut self, gpu: &Gpu, terms: &[f32]) -> Result<(), SkinGpuError> {
        let n = self.driver_count as usize;
        if terms.len() < n {
            return Err(SkinGpuError::Invalid { what: "one expression term per driver" });
        }
        if n == 0 {
            return Ok(());
        }
        for (chunk, t) in self.terms_bytes.as_chunks_mut::<4>().0.iter_mut().zip(&terms[..n]) {
            *chunk = t.to_le_bytes();
        }
        gpu.queue.write_buffer(&self.terms, 0, &self.terms_bytes[..n * 4]);
        Ok(())
    }

    /// Records the activation pass (after the deformer's strain passes, before shading).
    pub fn record_activate(&self, encoder: &mut wgpu::CommandEncoder) -> Result<(), SkinGpuError> {
        let Some(group) = &self.strain_group else {
            return Err(SkinGpuError::Invalid { what: "bind the region strain buffer before activating" });
        };
        if self.driver_count == 0 {
            return Ok(());
        }
        let pipeline = self
            .cache
            .get_compute_pipeline(&Key::ActivatePipeline)
            .ok_or(GpuError::Stale { what: "wrinkle activate pipeline" })?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wrinkle activate"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.params_group, &[]);
        pass.set_bind_group(1, group, &[]);
        pass.dispatch_workgroups(self.driver_count.div_ceil(ACTIVATE_GROUP), 1, 1);
        Ok(())
    }

    /// Uploads baked maps (editor-time); the maps' tiles become wrinkle slots `0..`.
    pub fn upload_maps(&mut self, gpu: &Gpu, maps: &WrinkleMaps) -> Result<(), SkinGpuError> {
        if maps.size() != self.desc.size || maps.tiles().len() > self.desc.tiles as usize {
            return Err(SkinGpuError::Invalid { what: "the wrinkle maps do not fit the wrinkle set" });
        }
        let size = self.desc.size;
        let mut values = vec![[0.0_f32; 4]; size as usize * size as usize];
        let mut bytes = vec![0_u8; values.len() * 8];
        for k in 0..maps.tiles().len() {
            for layer in 0..WRINKLE_LAYERS {
                maps.gpu_layer(k, layer, &mut values)?;
                for (chunk, v) in bytes.as_chunks_mut::<8>().0.iter_mut().zip(&values) {
                    for (c, x) in v.iter().enumerate() {
                        chunk[2 * c..2 * c + 2].copy_from_slice(&f32_to_f16_bits(*x).to_le_bytes());
                    }
                }
                gpu.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.maps,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: (k * WRINKLE_LAYERS + layer) as u32 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &bytes,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(size * 8),
                        rows_per_image: Some(size),
                    },
                    wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
                );
            }
        }
        self.slots = [NO_WRINKLE_SLOT; MAX_TILES as usize];
        for (k, &tile) in maps.tiles().iter().enumerate() {
            if let Some(s) = self.slots.get_mut(tile as usize) {
                *s = k as u32;
            }
        }
        let slot_bytes: Vec<u8> = self.slots.iter().flat_map(|s| s.to_le_bytes()).collect();
        gpu.queue.write_buffer(&self.tile_slots, 0, &slot_bytes);
        Ok(())
    }

    /// Records the blend of `rect` of wrinkle slot `slot` into `out` (`BLEND_TEXEL_BYTES` per
    /// texel, row-major in `rect`; usage `STORAGE`). Editor-time (creates a bind group); writes the
    /// shared parameter block, so record at most one blend per submission and submit before the
    /// next [`WrinkleGpu::set_drivers`] or [`WrinkleGpu::bind_strain`].
    pub fn blend_rect(
        &self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        slot: u32,
        rect: Rect,
        out: &wgpu::Buffer,
    ) -> Result<(), SkinGpuError> {
        let size = self.desc.size;
        if slot >= self.desc.tiles
            || rect.is_empty()
            || u64::from(rect.x) + u64::from(rect.w) > u64::from(size)
            || u64::from(rect.y) + u64::from(rect.h) > u64::from(size)
        {
            return Err(SkinGpuError::Invalid { what: "blend rectangle outside the wrinkle tile" });
        }
        if out.size() < u64::from(rect.w) * u64::from(rect.h) * BLEND_TEXEL_BYTES {
            return Err(SkinGpuError::Invalid { what: "blend output buffer too small" });
        }
        self.write_params(gpu, slot, rect);
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wrinkle blend"),
            layout: self.cache.get_bind_group_layout(&Key::Blend).ok_or(GpuError::Stale { what: "wrinkle layout" })?,
            entries: &[
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.maps_view) },
                wgpu::BindGroupEntry { binding: 5, resource: self.activations.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: out.as_entire_binding() },
            ],
        });
        let pipeline = self
            .cache
            .get_compute_pipeline(&Key::BlendPipeline)
            .ok_or(GpuError::Stale { what: "wrinkle blend pipeline" })?;
        let mut pass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("wrinkle blend"), timestamp_writes: None });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.params_group, &[]);
        pass.set_bind_group(1, &group, &[]);
        pass.dispatch_workgroups(rect.w.div_ceil(BLEND_GROUP), rect.h.div_ceil(BLEND_GROUP), 1);
        Ok(())
    }
}
