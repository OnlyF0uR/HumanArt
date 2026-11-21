//! The GPU bakers: UV-space G-buffer, thickness `[BAKE-01]`, occlusion, cavity and bent normals
//! `[BAKE-02]`, seam-padding dilation `[BAKE-03]`, each the twin of its CPU reference.
//!
//! - [`GpuBaker`] owns every pipeline of `shaders/bake/` (G-buffer, thickness, occlusion,
//!   curvature interpolation, dilation), all created once in [`GpuBaker::new`] (`[PERF-R4]`; the
//!   dilation mode is an override constant, `[PERF-R5]`), and a ring of uniform slots. Work is recorded into the caller's command encoder.
//! - [`GpuMesh`] is a [`BakeMesh`] on the device: the vertex streams (structure of arrays), the
//!   triangle list, the triangle tiles and the bounding-volume hierarchy, uploaded once per mesh.
//! - [`GpuTarget`] holds the textures of one tile at one size: the G-buffer (`r32uint` triangle
//!   plus one, `rg32float` barycentrics), two `rgba32float` outputs with their `r32uint` coverage masks
//!   (0 empty, non-zero filled), and one scratch pair for the dilation ping-pong.
//! - [`PendingMap`] and [`PendingSurface`] are asynchronous readbacks `[PERF-R3]`: record the
//!   copies, submit, call `request`, then `poll` until the map arrives (the native-only `wait`
//!   blocks, for tools and tests).
//!
//! The ray-traced kernels run in bands of rows ([`rows_per_dispatch`]) so that no single dispatch
//! casts more than [`RAYS_PER_DISPATCH`] rays and trips a driver watchdog on big maps. Bakes are
//! editor-time operations, not per-frame paths: bind groups reference the caller's mesh and
//! target and are made per call, and at most [`PARAM_SLOTS`] dispatches may be recorded between
//! [`GpuBaker::begin`] and the submission.
//!
//! Memory: a target of `w × h` texels takes `w · h · 68` bytes (1.1 GB at 4096²).

use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use humanart_gpu::cache::PipelineCache;
use humanart_gpu::{Gpu, GpuError};

use super::dilate::DilateMode;
use super::occlusion::OcclusionSettings;
use super::raster::{NO_TRIANGLE, SurfaceMap};
use super::thickness::ThicknessSettings;
use super::{BakeError, BakeMap, BakeMesh, BakeTarget, Coverage, Result};

/// Uniform slots, one per dispatch recorded between [`GpuBaker::begin`] and the submission.
pub const PARAM_SLOTS: u32 = 1024;
/// Bytes per uniform slot (the minimum uniform offset alignment wgpu guarantees).
const SLOT_BYTES: u64 = 256;
/// Workgroup edge of the compute kernels (8 × 8 invocations).
const GROUP: u32 = 8;
/// The most rays one dispatch of a ray-traced kernel casts.
pub const RAYS_PER_DISPATCH: u64 = 1 << 24;
/// Bytes of the ray kernels' parameters.
const RAY_PARAMS: u64 = 48;
/// Bytes of the G-buffer and dilation parameters.
const SMALL_PARAMS: u64 = 16;

/// Cache keys of the module's GPU objects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Key {
    GbufferShader,
    ThicknessShader,
    OcclusionShader,
    DilateShader,
    CurvatureShader,
    GbufferLayout,
    ThicknessLayout,
    OcclusionLayout,
    DilateLayout,
    CurvatureLayout,
    GbufferPipelineLayout,
    ThicknessPipelineLayout,
    OcclusionPipelineLayout,
    DilatePipelineLayout,
    CurvaturePipelineLayout,
    Gbuffer,
    Thickness,
    Occlusion,
    Curvature,
    Dilate(bool),
}

/// Which output texture of a [`GpuTarget`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Output {
    /// Output 0: thickness `(thickness, translucency)` or occlusion `(ao, cavity)`.
    First,
    /// Output 1: the bent normals of the occlusion kernel.
    Second,
}

impl Output {
    const fn index(self) -> usize {
        match self {
            Self::First => 0,
            Self::Second => 1,
        }
    }
}

/// The GPU baker: pipelines and parameter slots, created once.
pub struct GpuBaker {
    cache: PipelineCache<Key>,
    params: wgpu::Buffer,
    next_slot: u32,
}

/// A layout entry.
const fn entry(binding: u32, visibility: wgpu::ShaderStages, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility, ty, count: None }
}

const fn uniform(size: u64) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: true,
        min_binding_size: NonZeroU64::new(size),
    }
}

const STORAGE_READ: wgpu::BindingType = wgpu::BindingType::Buffer {
    ty: wgpu::BufferBindingType::Storage { read_only: true },
    has_dynamic_offset: false,
    min_binding_size: None,
};

const fn sampled(sample_type: wgpu::TextureSampleType) -> wgpu::BindingType {
    wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }
}

const UINT_TEXTURE: wgpu::BindingType = sampled(wgpu::TextureSampleType::Uint);
const FLOAT_TEXTURE: wgpu::BindingType = sampled(wgpu::TextureSampleType::Float { filterable: false });

const fn storage_texture(format: wgpu::TextureFormat) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        access: wgpu::StorageTextureAccess::WriteOnly,
        format,
        view_dimension: wgpu::TextureViewDimension::D2,
    }
}

/// How many rows of a `width`-texel map one dispatch of `rays` rays per texel covers (a multiple
/// of the workgroup edge, at least one workgroup row).
pub fn rows_per_dispatch(width: u32, rays: u32) -> u32 {
    let per_row = u64::from(width.max(1)) * u64::from(rays.max(1));
    let rows = (RAYS_PER_DISPATCH / per_row).max(1) as u32;
    (rows / GROUP).max(1) * GROUP
}

/// The bake mesh on the device.
pub struct GpuMesh {
    positions: wgpu::Buffer,
    normals: wgpu::Buffer,
    tangents: wgpu::Buffer,
    indices: wgpu::Buffer,
    uvs: wgpu::Buffer,
    tiles: wgpu::Buffer,
    nodes: wgpu::Buffer,
    tris: wgpu::Buffer,
    triangle_count: u32,
}

/// A storage buffer holding `words` (at least one word).
fn storage_buffer(gpu: &Gpu, label: &str, words: &[u32]) -> Result<wgpu::Buffer> {
    let size = (words.len().max(1) * 4) as u64;
    if size > gpu.limits().max_storage_buffer_binding_size {
        return Err(BakeError::invalid("a mesh stream exceeds the storage binding limit of the device"));
    }
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut bytes = Vec::with_capacity(size as usize);
    for w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes.resize(size as usize, 0);
    gpu.queue.write_buffer(&buffer, 0, &bytes);
    Ok(buffer)
}

impl GpuMesh {
    /// Uploads `mesh` and its hierarchy.
    pub fn upload(gpu: &Gpu, mesh: &BakeMesh) -> Result<Self> {
        let bvh = mesh.bvh()?;
        let mut positions = Vec::with_capacity(mesh.positions().len() * 4);
        for (p, &r) in mesh.positions().iter().zip(mesh.vertex_regions()) {
            positions.extend([p.x.to_bits(), p.y.to_bits(), p.z.to_bits(), u32::from(r)]);
        }
        let mut normals = Vec::with_capacity(mesh.normals().len() * 4);
        for n in mesh.normals() {
            normals.extend([n.x.to_bits(), n.y.to_bits(), n.z.to_bits(), 0]);
        }
        let tangents: Vec<u32> = mesh.tangents().iter().flat_map(|t| t.to_array().map(f32::to_bits)).collect();
        let uvs: Vec<u32> = mesh.uvs().iter().flat_map(|uv| [uv.x.to_bits(), uv.y.to_bits()]).collect();
        let tiles: Vec<u32> = mesh.triangle_tiles().iter().map(|&t| u32::from(t)).collect();
        Ok(Self {
            positions: storage_buffer(gpu, "bake positions", &positions)?,
            normals: storage_buffer(gpu, "bake normals", &normals)?,
            tangents: storage_buffer(gpu, "bake tangents", &tangents)?,
            indices: storage_buffer(gpu, "bake indices", mesh.indices())?,
            uvs: storage_buffer(gpu, "bake uvs", &uvs)?,
            tiles: storage_buffer(gpu, "bake tiles", &tiles)?,
            nodes: storage_buffer(gpu, "bake bvh nodes", &bvh.gpu_nodes())?,
            tris: storage_buffer(gpu, "bake bvh triangles", &bvh.gpu_triangles())?,
            triangle_count: mesh.triangle_count() as u32,
        })
    }
}

/// The textures of one bake target (see the module documentation).
pub struct GpuTarget {
    target: BakeTarget,
    ids: wgpu::Texture,
    bary: wgpu::Texture,
    values: [wgpu::Texture; 2],
    masks: [wgpu::Texture; 2],
    scratch: wgpu::Texture,
    scratch_mask: wgpu::Texture,
}

fn texture(
    gpu: &Gpu,
    label: &str,
    target: &BakeTarget,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: target.width, height: target.height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn view(t: &wgpu::Texture) -> wgpu::TextureView {
    t.create_view(&wgpu::TextureViewDescriptor::default())
}

impl GpuTarget {
    /// Creates the textures of `target`.
    pub fn new(gpu: &Gpu, target: &BakeTarget) -> Result<Self> {
        use wgpu::TextureUsages as U;
        target.check()?;
        let max = gpu.limits().max_texture_dimension_2d;
        if target.width > max || target.height > max {
            return Err(BakeError::invalid("the map is larger than the device's largest texture"));
        }
        let data = U::STORAGE_BINDING | U::TEXTURE_BINDING | U::COPY_SRC | U::COPY_DST;
        let gbuffer = U::RENDER_ATTACHMENT | U::TEXTURE_BINDING | U::COPY_SRC;
        let (f32x4, u32x1) = (wgpu::TextureFormat::Rgba32Float, wgpu::TextureFormat::R32Uint);
        Ok(Self {
            target: *target,
            ids: texture(gpu, "bake ids", target, u32x1, gbuffer),
            bary: texture(gpu, "bake barycentrics", target, wgpu::TextureFormat::Rg32Float, gbuffer),
            values: [
                texture(gpu, "bake output 0", target, f32x4, data),
                texture(gpu, "bake output 1", target, f32x4, data),
            ],
            masks: [texture(gpu, "bake mask 0", target, u32x1, data), texture(gpu, "bake mask 1", target, u32x1, data)],
            scratch: texture(gpu, "bake scratch", target, f32x4, data),
            scratch_mask: texture(gpu, "bake scratch mask", target, u32x1, data),
        })
    }

    /// The tile and size.
    pub const fn target(&self) -> &BakeTarget {
        &self.target
    }

    /// Writes a CPU map (at most 4 channels, missing ones 0) into output `output`, with its
    /// coverage as the mask (to dilate a CPU-baked map on the GPU).
    pub fn upload_map(&self, gpu: &Gpu, output: Output, map: &BakeMap) -> Result<()> {
        if (map.width, map.height) != (self.target.width, self.target.height) {
            return Err(BakeError::invalid("the map size differs from the target"));
        }
        let c = map.channels as usize;
        let mut bytes = Vec::with_capacity(map.texel_count() * 16);
        let mut mask = Vec::with_capacity(map.texel_count() * 4);
        for i in 0..map.texel_count() {
            let t = map.texel(i).ok_or(BakeError::invalid("the map's arrays do not match its size"))?;
            for k in 0..4 {
                bytes.extend_from_slice(&t.get(k).filter(|_| k < c).copied().unwrap_or(0.0).to_le_bytes());
            }
            let m: u32 = match map.coverage.get(i) {
                Some(Coverage::Surface) => 1,
                Some(Coverage::Padding) => 2,
                _ => 0,
            };
            mask.extend_from_slice(&m.to_le_bytes());
        }
        let write = |t: &wgpu::Texture, data: &[u8], texel: u32| {
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: t,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(map.width * texel),
                    rows_per_image: Some(map.height),
                },
                wgpu::Extent3d { width: map.width, height: map.height, depth_or_array_layers: 1 },
            );
        };
        write(&self.values[output.index()], &bytes, 16);
        write(&self.masks[output.index()], &mask, 4);
        Ok(())
    }
}

impl GpuBaker {
    /// Creates every pipeline (G-buffer, thickness, occlusion, two dilation modes) and the uniform
    /// slots. Nothing is created later.
    pub fn new(gpu: &Gpu) -> Result<Self> {
        use wgpu::ShaderStages as S;
        let mut cache = PipelineCache::new(gpu);
        for (key, path) in [
            (Key::GbufferShader, "bake/gbuffer.wgsl"),
            (Key::ThicknessShader, "bake/thickness.wgsl"),
            (Key::OcclusionShader, "bake/cavity_ao.wgsl"),
            (Key::DilateShader, "bake/dilate.wgsl"),
            (Key::CurvatureShader, "bake/curvature.wgsl"),
        ] {
            let source = humanart_shaders::compose(path, &[])?.into_source();
            cache.shader(key, path, &source)?;
        }
        let c = S::COMPUTE;
        cache.bind_group_layout(
            Key::GbufferLayout,
            "bake gbuffer",
            &[
                entry(0, S::VERTEX, uniform(SMALL_PARAMS)),
                entry(1, S::VERTEX, STORAGE_READ),
                entry(2, S::VERTEX, STORAGE_READ),
                entry(3, S::VERTEX, STORAGE_READ),
            ],
        )?;
        let mut ray = vec![entry(0, c, uniform(RAY_PARAMS))];
        ray.extend((1..=6).map(|b| entry(b, c, STORAGE_READ)));
        ray.push(entry(8, c, UINT_TEXTURE));
        ray.push(entry(9, c, FLOAT_TEXTURE));
        ray.push(entry(10, c, storage_texture(wgpu::TextureFormat::Rgba32Float)));
        let mut thickness = ray.clone();
        thickness.push(entry(7, c, STORAGE_READ));
        cache.bind_group_layout(Key::ThicknessLayout, "bake thickness", &thickness)?;
        let mut occlusion = ray;
        occlusion.push(entry(11, c, storage_texture(wgpu::TextureFormat::Rgba32Float)));
        cache.bind_group_layout(Key::OcclusionLayout, "bake occlusion", &occlusion)?;
        cache.bind_group_layout(
            Key::DilateLayout,
            "bake dilate",
            &[
                entry(0, c, uniform(SMALL_PARAMS)),
                entry(1, c, FLOAT_TEXTURE),
                entry(2, c, UINT_TEXTURE),
                entry(3, c, storage_texture(wgpu::TextureFormat::Rgba32Float)),
                entry(4, c, storage_texture(wgpu::TextureFormat::R32Uint)),
            ],
        )?;
        cache.bind_group_layout(
            Key::CurvatureLayout,
            "bake curvature",
            &[
                entry(0, c, uniform(SMALL_PARAMS)),
                entry(4, c, STORAGE_READ),
                entry(7, c, STORAGE_READ),
                entry(8, c, UINT_TEXTURE),
                entry(9, c, FLOAT_TEXTURE),
                entry(10, c, storage_texture(wgpu::TextureFormat::Rgba32Float)),
            ],
        )?;
        for (pl, layout, label) in [
            (Key::GbufferPipelineLayout, Key::GbufferLayout, "bake gbuffer"),
            (Key::ThicknessPipelineLayout, Key::ThicknessLayout, "bake thickness"),
            (Key::OcclusionPipelineLayout, Key::OcclusionLayout, "bake occlusion"),
            (Key::DilatePipelineLayout, Key::DilateLayout, "bake dilate"),
            (Key::CurvaturePipelineLayout, Key::CurvatureLayout, "bake curvature"),
        ] {
            cache.pipeline_layout(pl, label, &[Some(&layout)])?;
        }
        cache.render_pipeline(Key::Gbuffer, |parts| {
            let module = parts.shader(&Key::GbufferShader)?;
            let target =
                |format| Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL });
            Ok(parts.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("bake gbuffer"),
                layout: Some(parts.pipeline_layout(&Key::GbufferPipelineLayout)?),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_gbuffer"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState { cull_mode: None, ..wgpu::PrimitiveState::default() },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs_gbuffer"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[target(wgpu::TextureFormat::R32Uint), target(wgpu::TextureFormat::Rg32Float)],
                }),
                multiview_mask: None,
                cache: None,
            }))
        })?;
        let computes: [(Key, Key, Key, &str, f64); 5] = [
            (Key::Curvature, Key::CurvatureShader, Key::CurvaturePipelineLayout, "curvature", -1.0),
            (Key::Thickness, Key::ThicknessShader, Key::ThicknessPipelineLayout, "thickness", -1.0),
            (Key::Occlusion, Key::OcclusionShader, Key::OcclusionPipelineLayout, "cavity_ao", -1.0),
            (Key::Dilate(false), Key::DilateShader, Key::DilatePipelineLayout, "dilate", 0.0),
            (Key::Dilate(true), Key::DilateShader, Key::DilatePipelineLayout, "dilate", 1.0),
        ];
        for (key, shader, layout, entry_point, mode) in computes {
            let constants = [("MODE", mode)];
            cache.compute_pipeline(key, |parts| {
                Ok(parts.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("bake"),
                    layout: Some(parts.pipeline_layout(&layout)?),
                    module: parts.shader(&shader)?,
                    entry_point: Some(entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: if mode < 0.0 { &[] } else { &constants },
                        zero_initialize_workgroup_memory: false,
                    },
                    cache: None,
                }))
            })?;
        }
        cache.mark_warm();
        let params = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bake parameters"),
            size: u64::from(PARAM_SLOTS) * SLOT_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self { cache, params, next_slot: 0 })
    }

    /// Pipelines created after [`GpuBaker::new`] (always 0: every pipeline is pre-warmed).
    pub fn late_pipelines(&self) -> u32 {
        self.cache.stats().late
    }

    /// Starts a batch of dispatches for one submission: frees every parameter slot.
    pub fn begin(&mut self) {
        self.next_slot = 0;
    }

    /// Writes `words` into the next parameter slot and returns its offset.
    fn push_params(&mut self, gpu: &Gpu, words: &[u32]) -> Result<u32> {
        if self.next_slot >= PARAM_SLOTS {
            return Err(BakeError::invalid("more dispatches than parameter slots before the submission"));
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

    fn params_binding(&self, binding: u32, size: u64) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &self.params,
                offset: 0,
                size: NonZeroU64::new(size),
            }),
        }
    }

    fn layout(&self, key: Key) -> Result<&wgpu::BindGroupLayout> {
        Ok(self.cache.get_bind_group_layout(&key).ok_or(GpuError::Stale { what: "bake bind-group layout" })?)
    }

    /// Records the G-buffer of `target`'s tile, and resets both output masks to its coverage.
    pub fn rasterize(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &GpuMesh,
        target: &GpuTarget,
    ) -> Result<()> {
        let offset = self.push_params(gpu, &[u32::from(target.target.tile), 0, 0, 0])?;
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bake gbuffer"),
            layout: self.layout(Key::GbufferLayout)?,
            entries: &[
                self.params_binding(0, SMALL_PARAMS),
                wgpu::BindGroupEntry { binding: 1, resource: mesh.uvs.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: mesh.indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: mesh.tiles.as_entire_binding() },
            ],
        });
        let pipeline =
            self.cache.get_render_pipeline(&Key::Gbuffer).ok_or(GpuError::Stale { what: "bake gbuffer pipeline" })?;
        let (ids, bary) = (view(&target.ids), view(&target.bary));
        let attachment = |v| {
            Some(wgpu::RenderPassColorAttachment {
                view: v,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })
        };
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bake gbuffer"),
                color_attachments: &[attachment(&ids), attachment(&bary)],
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[offset]);
            pass.draw(0..mesh.triangle_count.saturating_mul(3), 0..1);
        }
        let extent =
            wgpu::Extent3d { width: target.target.width, height: target.target.height, depth_or_array_layers: 1 };
        for mask in &target.masks {
            encoder.copy_texture_to_texture(target.ids.as_image_copy(), mask.as_image_copy(), extent);
        }
        Ok(())
    }

    /// The mesh bindings 1 to 6 of the ray kernels.
    fn mesh_entries(mesh: &GpuMesh) -> [wgpu::BindGroupEntry<'_>; 6] {
        [
            wgpu::BindGroupEntry { binding: 1, resource: mesh.positions.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: mesh.normals.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: mesh.tangents.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: mesh.indices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: mesh.nodes.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: mesh.tris.as_entire_binding() },
        ]
    }

    /// Checks a band of rows.
    fn check_rows(target: &GpuTarget, rows: (u32, u32)) -> Result<u32> {
        let (row0, count) = rows;
        if count == 0 || u64::from(row0) + u64::from(count) > u64::from(target.target.height) {
            return Err(BakeError::invalid("the row band lies outside the map"));
        }
        Ok(count)
    }

    fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        key: Key,
        group: &wgpu::BindGroup,
        offset: u32,
        width: u32,
        rows: u32,
    ) -> Result<()> {
        let pipeline = self.cache.get_compute_pipeline(&key).ok_or(GpuError::Stale { what: "bake pipeline" })?;
        let mut pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("bake"), timestamp_writes: None });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[offset]);
        pass.dispatch_workgroups(width.div_ceil(GROUP), rows.div_ceil(GROUP), 1);
        Ok(())
    }

    /// Records the thickness kernel over the rows `rows = (first, count)` into output 0.
    pub fn thickness(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &GpuMesh,
        target: &GpuTarget,
        settings: &ThicknessSettings,
        offset_m: f32,
        rows: (u32, u32),
    ) -> Result<()> {
        settings.check()?;
        let count = Self::check_rows(target, rows)?;
        let t = &target.target;
        let words = [
            t.width,
            t.height,
            rows.0,
            settings.seed,
            settings.rays,
            settings.region_scale.len() as u32,
            settings.cone_angle.cos().to_bits(),
            settings.max_distance.to_bits(),
            settings.falloff.to_bits(),
            offset_m.to_bits(),
        ];
        let offset = self.push_params(gpu, &words)?;
        let scales: Vec<u32> = settings.region_scale.iter().map(|s| s.to_bits()).collect();
        let scales = storage_buffer(gpu, "bake region scales", &scales)?;
        let (ids, bary, out) = (view(&target.ids), view(&target.bary), view(&target.values[0]));
        let mut entries = vec![self.params_binding(0, RAY_PARAMS)];
        entries.extend(Self::mesh_entries(mesh));
        entries.extend([
            wgpu::BindGroupEntry { binding: 7, resource: scales.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&ids) },
            wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&bary) },
            wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&out) },
        ]);
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bake thickness"),
            layout: self.layout(Key::ThicknessLayout)?,
            entries: &entries,
        });
        self.dispatch(encoder, Key::Thickness, &group, offset, t.width, count)
    }

    /// Records the occlusion kernel over the rows `rows = (first, count)`: `(ao, cavity)` into
    /// output 0, the tangent-space bent normal into output 1.
    pub fn occlusion(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &GpuMesh,
        target: &GpuTarget,
        settings: &OcclusionSettings,
        offset_m: f32,
        rows: (u32, u32),
    ) -> Result<()> {
        settings.check()?;
        let count = Self::check_rows(target, rows)?;
        let t = &target.target;
        let words = [
            t.width,
            t.height,
            rows.0,
            settings.seed,
            settings.rays,
            0,
            settings.max_distance.to_bits(),
            settings.cavity_distance.to_bits(),
            offset_m.to_bits(),
        ];
        let offset = self.push_params(gpu, &words)?;
        let (ids, bary) = (view(&target.ids), view(&target.bary));
        let (out0, out1) = (view(&target.values[0]), view(&target.values[1]));
        let mut entries = vec![self.params_binding(0, RAY_PARAMS)];
        entries.extend(Self::mesh_entries(mesh));
        entries.extend([
            wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&ids) },
            wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&bary) },
            wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&out0) },
            wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&out1) },
        ]);
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bake occlusion"),
            layout: self.layout(Key::OcclusionLayout)?,
            entries: &entries,
        });
        self.dispatch(encoder, Key::Occlusion, &group, offset, t.width, count)
    }

    /// Records the curvature kernel over the rows `rows = (first, count)` into output 0: the
    /// per-vertex values `vertex_values` (one per vertex of the mesh, normally
    /// [`super::surface::vertex_mean_curvature`]) interpolated into the covered texels.
    pub fn curvature(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &GpuMesh,
        target: &GpuTarget,
        vertex_values: &[f32],
        rows: (u32, u32),
    ) -> Result<()> {
        let count = Self::check_rows(target, rows)?;
        if vertex_values.is_empty() || vertex_values.iter().any(|v| !v.is_finite()) {
            return Err(BakeError::invalid("curvature: the vertex values are empty or not finite"));
        }
        let t = &target.target;
        let offset = self.push_params(gpu, &[t.width, t.height, rows.0, 0])?;
        let words: Vec<u32> = vertex_values.iter().map(|v| v.to_bits()).collect();
        let values = storage_buffer(gpu, "bake vertex values", &words)?;
        let (ids, bary, out) = (view(&target.ids), view(&target.bary), view(&target.values[0]));
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bake curvature"),
            layout: self.layout(Key::CurvatureLayout)?,
            entries: &[
                self.params_binding(0, SMALL_PARAMS),
                wgpu::BindGroupEntry { binding: 4, resource: mesh.indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: values.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&ids) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&bary) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&out) },
            ],
        });
        self.dispatch(encoder, Key::Curvature, &group, offset, t.width, count)
    }

    /// Records `passes` dilation passes of output `output` (the result ends in the output and its
    /// mask).
    pub fn dilate(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        target: &GpuTarget,
        output: Output,
        passes: u32,
        mode: DilateMode,
    ) -> Result<()> {
        if passes > super::dilate::MAX_PASSES {
            return Err(BakeError::invalid("more than 256 dilation passes"));
        }
        let t = &target.target;
        let k = output.index();
        let pairs = [(&target.values[k], &target.masks[k]), (&target.scratch, &target.scratch_mask)];
        let key = Key::Dilate(mode == DilateMode::Nearest);
        for pass in 0..passes {
            let (src, src_mask) = pairs[(pass % 2) as usize];
            let (dst, dst_mask) = pairs[((pass + 1) % 2) as usize];
            let offset = self.push_params(gpu, &[t.width, t.height, 0, 0])?;
            let (sv, smv, dv, dmv) = (view(src), view(src_mask), view(dst), view(dst_mask));
            let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bake dilate"),
                layout: self.layout(Key::DilateLayout)?,
                entries: &[
                    self.params_binding(0, SMALL_PARAMS),
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&sv) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&smv) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&dv) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&dmv) },
                ],
            });
            self.dispatch(encoder, key, &group, offset, t.width, t.height)?;
        }
        if passes % 2 == 1 {
            let extent = wgpu::Extent3d { width: t.width, height: t.height, depth_or_array_layers: 1 };
            encoder.copy_texture_to_texture(target.scratch.as_image_copy(), target.values[k].as_image_copy(), extent);
            encoder.copy_texture_to_texture(
                target.scratch_mask.as_image_copy(),
                target.masks[k].as_image_copy(),
                extent,
            );
        }
        Ok(())
    }

    /// Records the readback of output `output` as a map of `channels` channels (coverage: surface
    /// where the G-buffer has a triangle, padding where only the mask is filled).
    pub fn read_map(
        &self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        target: &GpuTarget,
        output: Output,
        channels: u8,
    ) -> Result<PendingMap> {
        if channels == 0 || channels > super::MAX_CHANNELS {
            return Err(BakeError::invalid("a map has 1 to 4 channels"));
        }
        let k = output.index();
        Ok(PendingMap {
            target: target.target,
            channels,
            values: PendingTexture::record(gpu, encoder, &target.values[k], 16),
            ids: PendingTexture::record(gpu, encoder, &target.ids, 4),
            mask: PendingTexture::record(gpu, encoder, &target.masks[k], 4),
        })
    }

    /// Records the readback of the G-buffer as a [`SurfaceMap`].
    pub fn read_surface(
        &self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        target: &GpuTarget,
    ) -> Result<PendingSurface> {
        Ok(PendingSurface {
            target: target.target,
            ids: PendingTexture::record(gpu, encoder, &target.ids, 4),
            bary: PendingTexture::record(gpu, encoder, &target.bary, 8),
        })
    }
}

/// The readback of one texture, in bands of rows that each fit one staging buffer.
struct PendingTexture {
    bands: Vec<(u32, u32, wgpu::Buffer)>,
    width: u32,
    texel: u32,
    padded: u32,
    mapped: Arc<AtomicU32>,
    failed: Arc<AtomicBool>,
}

impl PendingTexture {
    fn record(gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture, texel: u32) -> Self {
        let (width, height) = (texture.width(), texture.height());
        let row = width * texel;
        let padded = row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let max_rows = (gpu.limits().max_buffer_size / u64::from(padded)).clamp(1, u64::from(height)) as u32;
        let mut bands = Vec::new();
        let mut row0 = 0;
        while row0 < height {
            let rows = max_rows.min(height - row0);
            let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bake readback"),
                size: u64::from(padded) * u64::from(rows),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: row0, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded),
                        rows_per_image: Some(rows),
                    },
                },
                wgpu::Extent3d { width, height: rows, depth_or_array_layers: 1 },
            );
            bands.push((row0, rows, buffer));
            row0 += rows;
        }
        Self {
            bands,
            width,
            texel,
            padded,
            mapped: Arc::new(AtomicU32::new(0)),
            failed: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Asks for every band to be mapped (after the copies were submitted).
    fn request(&self) {
        for (_, _, buffer) in &self.bands {
            let (mapped, failed) = (Arc::clone(&self.mapped), Arc::clone(&self.failed));
            buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                if result.is_err() {
                    failed.store(true, Ordering::Release);
                }
                mapped.fetch_add(1, Ordering::AcqRel);
            });
        }
    }

    fn ready(&self) -> Result<bool> {
        if self.failed.load(Ordering::Acquire) {
            return Err(GpuError::Map { message: String::from("a bake readback could not be mapped") }.into());
        }
        Ok(self.mapped.load(Ordering::Acquire) as usize >= self.bands.len())
    }

    /// The texels, rows tightly packed.
    fn bytes(&self) -> Result<Vec<u8>> {
        let row = (self.width * self.texel) as usize;
        let mut out = Vec::new();
        for (_, rows, buffer) in &self.bands {
            {
                let data = buffer.slice(..).get_mapped_range().map_err(|e| GpuError::Map { message: e.to_string() })?;
                for r in 0..*rows as usize {
                    let start = r * self.padded as usize;
                    out.extend_from_slice(data.get(start..start + row).ok_or(BakeError::invalid("short readback"))?);
                }
            }
            buffer.unmap();
        }
        Ok(out)
    }
}

/// Reads little-endian words.
fn words(bytes: &[u8]) -> impl Iterator<Item = u32> + '_ {
    bytes.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w))
}

/// An output map on its way back from the device.
pub struct PendingMap {
    target: BakeTarget,
    channels: u8,
    values: PendingTexture,
    ids: PendingTexture,
    mask: PendingTexture,
}

impl PendingMap {
    /// Asks for the staging buffers to be mapped; call once, after submitting the encoder the
    /// copies were recorded into.
    pub fn request(&self) {
        self.values.request();
        self.ids.request();
        self.mask.request();
    }

    /// Polls the device without blocking; the map once every buffer is mapped.
    pub fn poll(&self, gpu: &Gpu) -> Result<Option<BakeMap>> {
        gpu.device.poll(wgpu::PollType::Poll).map_err(GpuError::from)?;
        if self.values.ready()? && self.ids.ready()? && self.mask.ready()? { self.finish().map(Some) } else { Ok(None) }
    }

    /// Blocks until the map arrives (native only: tools and tests).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait(&self, gpu: &Gpu) -> Result<BakeMap> {
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).map_err(GpuError::from)?;
        match self.poll(gpu)? {
            Some(map) => Ok(map),
            None => Err(GpuError::Map { message: String::from("the bake readback never completed") }.into()),
        }
    }

    fn finish(&self) -> Result<BakeMap> {
        let mut map = BakeMap::new(&self.target, self.channels, 0.0)?;
        let values = self.values.bytes()?;
        let ids: Vec<u32> = words(&self.ids.bytes()?).collect();
        let mask: Vec<u32> = words(&self.mask.bytes()?).collect();
        let floats: Vec<f32> = words(&values).map(f32::from_bits).collect();
        let c = self.channels as usize;
        for i in 0..map.texel_count() {
            let (Some(&id), Some(&m), Some(v)) = (ids.get(i), mask.get(i), floats.get(i * 4..i * 4 + 4)) else {
                return Err(BakeError::invalid("short readback"));
            };
            let coverage = if id != 0 {
                Coverage::Surface
            } else if m != 0 {
                Coverage::Padding
            } else {
                Coverage::Empty
            };
            map.data[i * c..i * c + c].copy_from_slice(&v[..c]);
            map.coverage[i] = coverage;
        }
        Ok(map)
    }
}

/// The G-buffer on its way back from the device.
pub struct PendingSurface {
    target: BakeTarget,
    ids: PendingTexture,
    bary: PendingTexture,
}

impl PendingSurface {
    /// Asks for the staging buffers to be mapped (after the submission).
    pub fn request(&self) {
        self.ids.request();
        self.bary.request();
    }

    /// Polls the device without blocking; the surface once both buffers are mapped.
    pub fn poll(&self, gpu: &Gpu) -> Result<Option<SurfaceMap>> {
        gpu.device.poll(wgpu::PollType::Poll).map_err(GpuError::from)?;
        if !(self.ids.ready()? && self.bary.ready()?) {
            return Ok(None);
        }
        let ids: Vec<u32> = words(&self.ids.bytes()?).collect();
        let bary: Vec<f32> = words(&self.bary.bytes()?).map(f32::from_bits).collect();
        let n = self.target.texel_count();
        if ids.len() != n || bary.len() != 2 * n {
            return Err(BakeError::invalid("short readback"));
        }
        Ok(Some(SurfaceMap {
            target: self.target,
            triangles: ids.iter().map(|&id| if id == 0 { NO_TRIANGLE } else { id - 1 }).collect(),
            barycentrics: bary.as_chunks::<2>().0.to_vec(),
        }))
    }

    /// Blocks until the surface arrives (native only).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait(&self, gpu: &Gpu) -> Result<SurfaceMap> {
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).map_err(GpuError::from)?;
        match self.poll(gpu)? {
            Some(s) => Ok(s),
            None => Err(GpuError::Map { message: String::from("the bake readback never completed") }.into()),
        }
    }
}

/// Blocking whole-map bakes for tools and tests (native only: they wait for the device).
#[cfg(not(target_arch = "wasm32"))]
impl GpuBaker {
    /// Submits `encoder` after recording into it with `record`.
    fn submit(
        &mut self,
        gpu: &Gpu,
        record: impl FnOnce(&mut Self, &mut wgpu::CommandEncoder) -> Result<()>,
    ) -> Result<()> {
        self.begin();
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bake") });
        record(self, &mut encoder)?;
        gpu.queue.submit([encoder.finish()]);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).map_err(GpuError::from)?;
        Ok(())
    }

    /// Rasterises, runs the thickness kernel band by band, pads by `padding` passes and reads the
    /// 2-channel map back.
    pub fn bake_thickness_blocking(
        &mut self,
        gpu: &Gpu,
        mesh: &BakeMesh,
        gpu_mesh: &GpuMesh,
        target: &GpuTarget,
        settings: &ThicknessSettings,
        padding: u32,
    ) -> Result<BakeMap> {
        self.submit(gpu, |b, e| b.rasterize(gpu, e, gpu_mesh, target))?;
        let (w, h) = (target.target.width, target.target.height);
        let band = rows_per_dispatch(w, settings.rays);
        let mut row0 = 0;
        while row0 < h {
            let rows = band.min(h - row0);
            self.submit(gpu, |b, e| b.thickness(gpu, e, gpu_mesh, target, settings, mesh.ray_offset(), (row0, rows)))?;
            row0 += rows;
        }
        self.read_blocking(gpu, target, Output::First, 2, padding, DilateMode::Average)
    }

    /// Rasterises, runs the occlusion kernel band by band, pads and reads back
    /// `([ao, cavity], bent normals)`.
    pub fn bake_occlusion_blocking(
        &mut self,
        gpu: &Gpu,
        mesh: &BakeMesh,
        gpu_mesh: &GpuMesh,
        target: &GpuTarget,
        settings: &OcclusionSettings,
        padding: u32,
    ) -> Result<(BakeMap, BakeMap)> {
        self.submit(gpu, |b, e| b.rasterize(gpu, e, gpu_mesh, target))?;
        let (w, h) = (target.target.width, target.target.height);
        let band = rows_per_dispatch(w, settings.rays);
        let mut row0 = 0;
        while row0 < h {
            let rows = band.min(h - row0);
            self.submit(gpu, |b, e| b.occlusion(gpu, e, gpu_mesh, target, settings, mesh.ray_offset(), (row0, rows)))?;
            row0 += rows;
        }
        let ao = self.read_blocking(gpu, target, Output::First, 2, padding, DilateMode::Average)?;
        let bent = self.read_blocking(gpu, target, Output::Second, 3, padding, DilateMode::Average)?;
        Ok((ao, bent))
    }

    /// Rasterises, computes the per-vertex mean curvature on the CPU (an O(V) pass), interpolates
    /// it on the GPU, pads and reads the 1-channel map back.
    pub fn bake_curvature_blocking(
        &mut self,
        gpu: &Gpu,
        mesh: &BakeMesh,
        gpu_mesh: &GpuMesh,
        target: &GpuTarget,
        padding: u32,
    ) -> Result<BakeMap> {
        let h = super::surface::vertex_mean_curvature(mesh);
        let rows = target.target.height;
        self.submit(gpu, |b, e| {
            b.rasterize(gpu, e, gpu_mesh, target)?;
            b.curvature(gpu, e, gpu_mesh, target, &h, (0, rows))
        })?;
        self.read_blocking(gpu, target, Output::First, 1, padding, DilateMode::Average)
    }

    /// Pads output `output` by `padding` passes and reads it back.
    pub fn read_blocking(
        &mut self,
        gpu: &Gpu,
        target: &GpuTarget,
        output: Output,
        channels: u8,
        padding: u32,
        mode: DilateMode,
    ) -> Result<BakeMap> {
        let mut pending = None;
        self.submit(gpu, |b, e| {
            b.dilate(gpu, e, target, output, padding, mode)?;
            pending = Some(b.read_map(gpu, e, target, output, channels)?);
            Ok(())
        })?;
        let pending = pending.ok_or(BakeError::invalid("the readback was not recorded"))?;
        pending.request();
        pending.wait(gpu)
    }

    /// The G-buffer of `target` as a [`SurfaceMap`] (rasterises first).
    pub fn surface_blocking(&mut self, gpu: &Gpu, gpu_mesh: &GpuMesh, target: &GpuTarget) -> Result<SurfaceMap> {
        let mut pending = None;
        self.submit(gpu, |b, e| {
            b.rasterize(gpu, e, gpu_mesh, target)?;
            pending = Some(b.read_surface(gpu, e, target)?);
            Ok(())
        })?;
        let pending = pending.ok_or(BakeError::invalid("the readback was not recorded"))?;
        pending.request();
        pending.wait(gpu)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::{Mutex, MutexGuard, OnceLock};

    use humanart_math::Vec3;

    use super::*;
    use crate::bake::dilate::{dilate, padding_for_levels};
    use crate::bake::occlusion::bake_occlusion;
    use crate::bake::raster::rasterize;
    use crate::bake::tests::{box_mesh, ellipsoid_mesh, sphere_mesh};
    use crate::bake::thickness::bake_thickness;

    pub(crate) fn gpu() -> &'static Gpu {
        // The process's shared device (ADR-040).
        humanart_gpu::shared_headless().expect("this machine has a GPU")
    }

    pub(crate) fn baker() -> MutexGuard<'static, GpuBaker> {
        static BAKER: OnceLock<Mutex<GpuBaker>> = OnceLock::new();
        BAKER
            .get_or_init(|| Mutex::new(GpuBaker::new(gpu()).expect("every bake pipeline builds")))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn flap() -> BakeMesh {
        let m = ellipsoid_mesh(Vec3::new(0.015, 0.03, 0.0025), 48, 32);
        BakeMesh::new(&m, m.positions(), None).unwrap()
    }

    #[test]
    fn bake_08_gpu_gbuffer_matches_the_cpu_rasteriser() {
        let gpu = gpu();
        println!("adapter: {}", gpu.adapter_info().name);
        for (mesh, target) in
            [(box_mesh(0.1, 0.06, 0.04, 7).0, BakeTarget::new(0, 150, 100)), (flap(), BakeTarget::new(0, 128, 128))]
        {
            let gm = GpuMesh::upload(gpu, &mesh).unwrap();
            let gt = GpuTarget::new(gpu, &target).unwrap();
            let gs = baker().surface_blocking(gpu, &gm, &gt).unwrap();
            let cs = rasterize(&mesh, &target).unwrap();
            // Texel centres exactly on a shared edge (common on these regular grids) may go to
            // either triangle: the hardware snaps vertices to 1/256 texel first. What must agree is
            // whether a texel is covered and the surface point it maps to.
            let (mut coverage_differs, mut other_triangle, mut max_offset) = (0, 0, 0.0_f32);
            for i in 0..target.texel_count() {
                let (g, c) = (gs.triangles[i], cs.triangles[i]);
                if (g == NO_TRIANGLE) != (c == NO_TRIANGLE) {
                    coverage_differs += 1;
                } else if c != NO_TRIANGLE {
                    other_triangle += usize::from(g != c);
                    let (bg, bc) = (gs.barycentrics[i], cs.barycentrics[i]);
                    let pg = mesh.point(g as usize, bg[0], bg[1]).unwrap().position;
                    let pc = mesh.point(c as usize, bc[0], bc[1]).unwrap().position;
                    max_offset = max_offset.max(pg.distance(pc));
                }
            }
            let texel_size = mesh.extent() / target.width as f32;
            println!(
                "gbuffer: coverage differs on {coverage_differs} of {} covered texels, {other_triangle} on the other triangle of an edge, largest surface offset {:.4} texel",
                cs.covered_count(),
                max_offset / texel_size
            );
            assert!(coverage_differs * 500 <= cs.covered_count(), "{coverage_differs}");
            assert!(max_offset < 0.01 * texel_size, "{max_offset}");
        }
        assert_eq!(baker().late_pipelines(), 0);
    }

    /// Compares a GPU map with the CPU map on the texels both put on the surface; returns the
    /// fraction within `tolerance[k]` on every channel `k` and the mean absolute difference of
    /// channel 0.
    pub(crate) fn agreement(gpu_map: &BakeMap, cpu_map: &BakeMap, tolerance: &[f32]) -> (f32, f32) {
        let c = tolerance.len().min(cpu_map.channels as usize);
        let (mut n, mut ok, mut diff) = (0, 0, 0.0_f32);
        for i in 0..cpu_map.texel_count() {
            if gpu_map.coverage[i] != Coverage::Surface || cpu_map.coverage[i] != Coverage::Surface {
                continue;
            }
            let (a, b) = (gpu_map.texel(i).unwrap(), cpu_map.texel(i).unwrap());
            n += 1;
            ok += usize::from((0..c).all(|k| (a[k] - b[k]).abs() <= tolerance[k]));
            diff += (a[0] - b[0]).abs();
        }
        assert!(n > 100, "{n}");
        (ok as f32 / n as f32, diff / n as f32)
    }

    #[test]
    fn bake_01_gpu_thickness_matches_the_cpu_reference() {
        let gpu = gpu();
        let settings = ThicknessSettings { rays: 32, ..ThicknessSettings::default() };
        for (mesh, target) in
            [(flap(), BakeTarget::new(0, 128, 128)), (sphere_mesh(0.02, 32, 16), BakeTarget::new(0, 96, 48))]
        {
            let gm = GpuMesh::upload(gpu, &mesh).unwrap();
            let gt = GpuTarget::new(gpu, &target).unwrap();
            let g = baker().bake_thickness_blocking(gpu, &mesh, &gm, &gt, &settings, 0).unwrap();
            let c = bake_thickness(&mesh, &rasterize(&mesh, &target).unwrap(), &settings).unwrap();
            // Thickness within 20 um, translucency within 0.5 %.
            let (fraction, mean) = agreement(&g, &c, &[2e-5, 5e-3]);
            println!("thickness: {fraction:.4} of texels within 20 um, mean difference {mean:e} m");
            assert!(fraction > 0.99 && mean < 2e-6, "{fraction} {mean}");
        }
    }

    #[test]
    fn bake_01_gpu_region_scale() {
        let gpu = gpu();
        let (mesh, _) = box_mesh(0.1, 0.1, 0.01, 4);
        let target = BakeTarget::new(0, 48, 32);
        let gm = GpuMesh::upload(gpu, &mesh).unwrap();
        let gt = GpuTarget::new(gpu, &target).unwrap();
        let base = ThicknessSettings { rays: 16, ..ThicknessSettings::default() };
        let scaled = ThicknessSettings { region_scale: vec![2.0], ..base.clone() };
        let a = baker().bake_thickness_blocking(gpu, &mesh, &gm, &gt, &base, 0).unwrap();
        let b = baker().bake_thickness_blocking(gpu, &mesh, &gm, &gt, &scaled, 0).unwrap();
        let mut n = 0;
        for i in 0..target.texel_count() {
            if a.coverage[i] == Coverage::Surface {
                assert_eq!(b.texel(i).unwrap()[0], 2.0 * a.texel(i).unwrap()[0]);
                n += 1;
            }
        }
        assert!(n > 100);
    }

    #[test]
    fn bake_03_gpu_curvature_matches_the_cpu_reference() {
        let gpu = gpu();
        let mesh = flap();
        let target = BakeTarget::new(0, 128, 128);
        let gm = GpuMesh::upload(gpu, &mesh).unwrap();
        let gt = GpuTarget::new(gpu, &target).unwrap();
        let g = baker().bake_curvature_blocking(gpu, &mesh, &gm, &gt, 0).unwrap();
        let c = crate::bake::surface::bake_curvature(&mesh, &rasterize(&mesh, &target).unwrap()).unwrap();
        // H spans about 10 to 900 1/m on the flap and changes by up to ~100 1/m across a
        // triangle, so the G-buffer's barycentric differences (< 1e-3) allow ~0.1 1/m: within 1.
        let (fraction, mean) = agreement(&g, &c, &[1.0]);
        println!("curvature: {fraction:.4} within 1 1/m, mean difference {mean:e}");
        assert!(fraction > 0.99, "{fraction}");
    }

    #[test]
    fn bake_02_gpu_occlusion_matches_the_cpu_reference() {
        let gpu = gpu();
        let (mesh, _) = box_mesh(0.05, 0.05, 0.05, 6);
        let settings = OcclusionSettings { rays: 32, max_distance: 0.05, cavity_distance: 0.01, seed: 1 };
        let target = BakeTarget::new(0, 96, 64);
        let gm = GpuMesh::upload(gpu, &mesh).unwrap();
        let gt = GpuTarget::new(gpu, &target).unwrap();
        let (ao, bent) = baker().bake_occlusion_blocking(gpu, &mesh, &gm, &gt, &settings, 0).unwrap();
        let cpu = bake_occlusion(&mesh, &rasterize(&mesh, &target).unwrap(), &settings).unwrap();
        let one_ray = 1.0 / 32.0 + 1e-6;
        let (fraction, _) = agreement(&ao, &cpu.occlusion, &[one_ray, one_ray]);
        let (bent_fraction, _) = agreement(&bent, &cpu.bent_normals, &[1e-3; 3]);
        println!("occlusion: {fraction:.4} within one ray, bent normals {bent_fraction:.4} within 1e-3");
        assert!(fraction > 0.99 && bent_fraction > 0.98, "{fraction} {bent_fraction}");
    }

    #[test]
    fn bake_03_gpu_dilation_is_bit_identical_to_the_cpu() {
        let gpu = gpu();
        let mesh = flap();
        let target = BakeTarget::new(0, 96, 80);
        let surface = rasterize(&mesh, &target).unwrap();
        let settings = ThicknessSettings { rays: 8, ..ThicknessSettings::default() };
        let map = bake_thickness(&mesh, &surface, &settings).unwrap();
        let gt = GpuTarget::new(gpu, &target).unwrap();
        for mode in [DilateMode::Average, DilateMode::Nearest] {
            for passes in [1, padding_for_levels(4)] {
                gt.upload_map(gpu, Output::First, &map).unwrap();
                let g = baker().read_blocking(gpu, &gt, Output::First, 2, passes, mode).unwrap();
                let mut c = map.clone();
                dilate(&mut c, passes, mode).unwrap();
                // This target's G-buffer was never drawn, so the GPU reports every filled texel as
                // padding: compare filled versus empty, and the values bit for bit.
                for i in 0..target.texel_count() {
                    assert_eq!(g.coverage[i] == Coverage::Empty, c.coverage[i] == Coverage::Empty, "texel {i}");
                    let (a, b) = (g.texel(i).unwrap(), c.texel(i).unwrap());
                    assert!(
                        a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                        "{mode:?} {passes} texel {i}: {a:?} vs {b:?}"
                    );
                }
            }
        }
    }
}
