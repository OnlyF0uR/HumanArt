//! The GPU micro-detail generator `[SKN-20]` `[SKN-21]`: `skin/microdetail.wgsl` driven from a
//! [`ExemplarPlan`] of `humanart_model::skin_appearance::microdetail`.
//!
//! [`MicroDetailGpu::generate`] evaluates the periodic detail exemplar (pores, fine lines, relief,
//! age) on the GPU into an `r32float` height texture and derives the detail texture (`rgba16float`:
//! tangent normal x, y, height, cavity). [`MicroDetailGpu::synthesize`] applies histogram-preserving
//! tiling and blending of a Gaussianised exemplar over an output texture. Generation and
//! synthesis are editor-time (when a micro-detail parameter changes, or for a bake); the pipelines
//! are created once in [`MicroDetailGpu::new`].

use std::num::NonZeroU64;

use humanart_gpu::cache::PipelineCache;
use humanart_gpu::{Gpu, GpuError};
use humanart_model::skin_appearance::antitile::{GaussianExemplar, LUT_SIZE};
use humanart_model::skin_appearance::microdetail::ExemplarPlan;

use super::SkinGpuError;

/// Bytes of `Plan` in `skin/microdetail.wgsl`.
const PLAN_BYTES: u64 = 80;
/// Workgroup edge.
const GROUP: u32 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Key {
    Shader,
    PlanFull,
    PlanOnly,
    Derive,
    Synth,
    GenerateLayout,
    DeriveLayout,
    SynthLayout,
    Generate,
    DerivePipeline,
    SynthPipeline,
}

/// One generated exemplar on the GPU.
pub struct GpuExemplar {
    /// Side in texels.
    pub size: u32,
    /// Height in mm (`r32float`).
    pub height: wgpu::Texture,
    /// Normal x, y, height, cavity (`rgba16float`).
    pub detail: wgpu::Texture,
    plan: wgpu::Buffer,
}

/// A Gaussianised exemplar uploaded for synthesis.
pub struct GpuGaussian {
    /// Gaussianised values (`r32float`).
    pub gauss: wgpu::Texture,
    /// Inverse table, `LUT_SIZE × 1` (`r32float`).
    pub lut: wgpu::Texture,
}

/// The generator's pipelines.
pub struct MicroDetailGpu {
    cache: PipelineCache<Key>,
}

const fn entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None }
}

const UNIFORM: wgpu::BindingType = wgpu::BindingType::Buffer {
    ty: wgpu::BufferBindingType::Uniform,
    has_dynamic_offset: false,
    min_binding_size: NonZeroU64::new(PLAN_BYTES),
};

const STORAGE: wgpu::BindingType = wgpu::BindingType::Buffer {
    ty: wgpu::BufferBindingType::Storage { read_only: true },
    has_dynamic_offset: false,
    min_binding_size: None,
};

const TEXTURE: wgpu::BindingType = wgpu::BindingType::Texture {
    sample_type: wgpu::TextureSampleType::Float { filterable: false },
    view_dimension: wgpu::TextureViewDimension::D2,
    multisampled: false,
};

const fn store(format: wgpu::TextureFormat) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        access: wgpu::StorageTextureAccess::WriteOnly,
        format,
        view_dimension: wgpu::TextureViewDimension::D2,
    }
}

fn texture_2d(
    gpu: &Gpu,
    label: &str,
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

/// The storage buffer of `bytes`, at least 16 bytes long (empty tables are legal).
fn storage_buffer(gpu: &Gpu, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    let size = (bytes.len() as u64).max(16).next_multiple_of(4);
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !bytes.is_empty() {
        gpu.queue.write_buffer(&buffer, 0, bytes);
    }
    buffer
}

/// The `Plan` uniform of an exemplar plan (synthesis fields zero).
fn plan_bytes(plan: &ExemplarPlan, tiles: f32, seed: u32) -> [u8; PLAN_BYTES as usize] {
    let p = &plan.params;
    let [l0, l1] = plan.lines;
    let words: [u32; 20] = [
        p.size,
        plan.grid,
        p.seed,
        plan.pores.len() as u32,
        l0[0] as u32,
        l0[1] as u32,
        l0[2] as u32,
        l1[0] as u32,
        l1[1] as u32,
        l1[2] as u32,
        0,
        0,
        plan.age_terms[0].to_bits(),
        plan.age_terms[1].to_bits(),
        plan.age_terms[2].to_bits(),
        plan.age_terms[3].to_bits(),
        p.texel_mm.to_bits(),
        tiles.to_bits(),
        seed,
        0,
    ];
    let mut out = [0_u8; PLAN_BYTES as usize];
    for (chunk, w) in out.as_chunks_mut::<4>().0.iter_mut().zip(words) {
        *chunk = w.to_le_bytes();
    }
    out
}

impl MicroDetailGpu {
    /// Creates the three pipelines.
    pub fn new(gpu: &Gpu) -> Result<Self, SkinGpuError> {
        let mut cache = PipelineCache::new(gpu);
        let source = humanart_shaders::compose("skin/microdetail.wgsl", &[])?.into_source();
        cache.shader(Key::Shader, "skin/microdetail.wgsl", &source)?;
        cache.bind_group_layout(
            Key::PlanFull,
            "micro-detail plan",
            &[entry(0, UNIFORM), entry(1, STORAGE), entry(2, STORAGE), entry(3, store(wgpu::TextureFormat::R32Float))],
        )?;
        cache.bind_group_layout(Key::PlanOnly, "micro-detail plan uniform", &[entry(0, UNIFORM)])?;
        cache.bind_group_layout(
            Key::Derive,
            "micro-detail derive",
            &[entry(0, TEXTURE), entry(1, store(wgpu::TextureFormat::Rgba16Float))],
        )?;
        cache.bind_group_layout(
            Key::Synth,
            "micro-detail synthesis",
            &[entry(2, TEXTURE), entry(3, TEXTURE), entry(4, store(wgpu::TextureFormat::R32Float))],
        )?;
        cache.pipeline_layout(Key::GenerateLayout, "micro-detail generate", &[Some(&Key::PlanFull)])?;
        cache.pipeline_layout(Key::DeriveLayout, "micro-detail derive", &[Some(&Key::PlanOnly), Some(&Key::Derive)])?;
        cache.pipeline_layout(
            Key::SynthLayout,
            "micro-detail synthesis",
            &[Some(&Key::PlanOnly), Some(&Key::Synth)],
        )?;
        for (key, layout, name) in [
            (Key::Generate, Key::GenerateLayout, "generate"),
            (Key::DerivePipeline, Key::DeriveLayout, "derive"),
            (Key::SynthPipeline, Key::SynthLayout, "synthesize"),
        ] {
            cache.compute_pipeline(key, |parts| {
                Ok(parts.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("skin micro-detail"),
                    layout: Some(parts.pipeline_layout(&layout)?),
                    module: parts.shader(&Key::Shader)?,
                    entry_point: Some(name),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                }))
            })?;
        }
        cache.mark_warm();
        Ok(Self { cache })
    }

    /// Pipelines created after [`MicroDetailGpu::new`] (always 0).
    pub fn late_pipelines(&self) -> u32 {
        self.cache.stats().late
    }

    fn layout(&self, key: Key) -> Result<&wgpu::BindGroupLayout, GpuError> {
        self.cache.get_bind_group_layout(&key).ok_or(GpuError::Stale { what: "micro-detail layout" })
    }

    fn pipeline(&self, key: Key) -> Result<&wgpu::ComputePipeline, GpuError> {
        self.cache.get_compute_pipeline(&key).ok_or(GpuError::Stale { what: "micro-detail pipeline" })
    }

    /// Records the generation of `plan`'s exemplar and its detail texture.
    pub fn generate(
        &self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        plan: &ExemplarPlan,
    ) -> Result<GpuExemplar, SkinGpuError> {
        let size = plan.params.size;
        if size > gpu.limits().max_texture_dimension_2d {
            return Err(SkinGpuError::Invalid { what: "exemplar larger than the device's textures" });
        }
        let usage =
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC;
        let height = texture_2d(gpu, "micro-detail height", size, size, wgpu::TextureFormat::R32Float, usage);
        let detail = texture_2d(gpu, "micro-detail detail", size, size, wgpu::TextureFormat::Rgba16Float, usage);
        let plan_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("micro-detail plan"),
            size: PLAN_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&plan_buffer, 0, &plan_bytes(plan, 0.0, 0));
        let pores = storage_buffer(gpu, "micro-detail pores", &plan.gpu_pore_bytes());
        let cells = storage_buffer(gpu, "micro-detail cells", &plan.gpu_cell_bytes());
        let height_view = height.create_view(&wgpu::TextureViewDescriptor::default());
        let detail_view = detail.create_view(&wgpu::TextureViewDescriptor::default());
        let full = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("micro-detail generate"),
            layout: self.layout(Key::PlanFull)?,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: plan_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pores.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cells.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&height_view) },
            ],
        });
        let plan_only = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("micro-detail plan"),
            layout: self.layout(Key::PlanOnly)?,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: plan_buffer.as_entire_binding() }],
        });
        let derive = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("micro-detail derive"),
            layout: self.layout(Key::Derive)?,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&height_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&detail_view) },
            ],
        });
        let groups = size.div_ceil(GROUP);
        let mut pass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("micro-detail"), timestamp_writes: None });
        pass.set_pipeline(self.pipeline(Key::Generate)?);
        pass.set_bind_group(0, &full, &[]);
        pass.dispatch_workgroups(groups, groups, 1);
        pass.set_pipeline(self.pipeline(Key::DerivePipeline)?);
        pass.set_bind_group(0, &plan_only, &[]);
        pass.set_bind_group(1, &derive, &[]);
        pass.dispatch_workgroups(groups, groups, 1);
        drop(pass);
        Ok(GpuExemplar { size, height, detail, plan: plan_buffer })
    }

    /// Uploads a Gaussianised exemplar (prepared on the CPU: the rank transform needs a sort).
    pub fn upload_gaussian(gpu: &Gpu, exemplar: &GaussianExemplar) -> Result<GpuGaussian, SkinGpuError> {
        let size = exemplar.size as u32;
        if size == 0 || size > gpu.limits().max_texture_dimension_2d {
            return Err(SkinGpuError::Invalid { what: "exemplar size outside the device's limits" });
        }
        let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let gauss = texture_2d(gpu, "micro-detail gaussian", size, size, wgpu::TextureFormat::R32Float, usage);
        let lut = texture_2d(gpu, "micro-detail lut", LUT_SIZE as u32, 1, wgpu::TextureFormat::R32Float, usage);
        for (texture, data, w, h) in [(&gauss, &exemplar.gauss, size, size), (&lut, &exemplar.lut, LUT_SIZE as u32, 1)]
        {
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &bytes,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        }
        Ok(GpuGaussian { gauss, lut })
    }

    /// Records the histogram-preserving synthesis of `source` over `out` (`r32float`, storage
    /// usage), `tiles` exemplar periods across, with the vertex-offset `seed`. `plan` supplies the
    /// uniform block (its exemplar's plan buffer is rewritten with the synthesis fields).
    pub fn synthesize(
        &self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        exemplar: &GpuExemplar,
        plan: &ExemplarPlan,
        source: &GpuGaussian,
        out: &wgpu::Texture,
        tiles: f32,
        seed: u32,
    ) -> Result<(), SkinGpuError> {
        if out.format() != wgpu::TextureFormat::R32Float || !tiles.is_finite() || tiles <= 0.0 {
            return Err(SkinGpuError::Invalid { what: "synthesis needs an r32float output and a positive tile count" });
        }
        gpu.queue.write_buffer(&exemplar.plan, 0, &plan_bytes(plan, tiles, seed));
        let plan_only = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("micro-detail plan"),
            layout: self.layout(Key::PlanOnly)?,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: exemplar.plan.as_entire_binding() }],
        });
        let (gauss, lut, out_view) = (
            source.gauss.create_view(&wgpu::TextureViewDescriptor::default()),
            source.lut.create_view(&wgpu::TextureViewDescriptor::default()),
            out.create_view(&wgpu::TextureViewDescriptor::default()),
        );
        let synth = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("micro-detail synthesis"),
            layout: self.layout(Key::Synth)?,
            entries: &[
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&gauss) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&lut) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&out_view) },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("micro-detail synthesis"),
            timestamp_writes: None,
        });
        pass.set_pipeline(self.pipeline(Key::SynthPipeline)?);
        pass.set_bind_group(0, &plan_only, &[]);
        pass.set_bind_group(1, &synth, &[]);
        pass.dispatch_workgroups(out.width().div_ceil(GROUP), out.height().div_ceil(GROUP), 1);
        Ok(())
    }
}
