//! Image-based light preprocessing on the GPU `[LIT-01]`: the environment cube (procedural, or an
//! imported latitude-longitude HDR image), its mip chain, the GGX-prefiltered specular cube, the
//! L2 SH irradiance and the split-sum DFG table. Everything runs in `shaders/render/ibl.wgsl` when
//! the environment changes; frames only sample the results.
//!
//! There are no downloaded HDRIs in the project: the built-in environments are generated in code
//! ([`ProceduralEnvironment`]) and none of them is a measurement of a real place.

use humanart_gpu::pod::{bytes_of, slice_as_bytes};
use humanart_gpu::resources::{BufferDesc, BufferId, Category, Resources, Tag, TextureDesc, TextureId};
use humanart_math::f32_to_f16_bits;

use crate::error::RenderError;
use crate::light::ProceduralEnvironment;
use crate::uniforms::IblUniforms;

/// Face size of the environment cube.
pub const ENV_SIZE: u32 = 256;
/// Mip levels of the environment cube (256 down to 1).
pub const ENV_MIPS: u32 = 9;
/// Face size of the prefiltered specular cube.
pub const SPECULAR_SIZE: u32 = 128;
/// Mip levels of the specular cube: roughness 0, 0.2, ..., 1.
pub const SPECULAR_MIPS: u32 = 6;
/// Side of the DFG table.
pub const DFG_SIZE: u32 = 64;
/// GGX samples per texel of the prefilter.
pub const PREFILTER_SAMPLES: u32 = 256;
/// Samples of the SH projection (256 threads x 64).
pub const SH_SAMPLES: u32 = 16_384;
/// Samples per texel of the DFG table.
pub const DFG_SAMPLES: u32 = 1024;
/// Largest imported equirectangular image side (the device's 2D limit applies too).
pub const MAX_EQUIRECT_SIDE: u32 = 8192;

/// Bytes between two dispatch parameter blocks (the uniform offset alignment).
const PARAM_STRIDE: u64 = 256;
/// Parameter blocks: 1 source + 8 downsamples + 6 prefilter levels + SH + DFG, rounded up.
const PARAM_SLOTS: u64 = 32;

/// Where the environment comes from.
#[derive(Debug, Clone, Copy)]
pub enum EnvironmentSource<'a> {
    /// Generated in code.
    Procedural(ProceduralEnvironment),
    /// An imported latitude-longitude image (Radiance HDR or EXR through `humanart-io`), linear
    /// Rec.709, 3 or 4 channels: longitude 0 at -Z increasing towards +X, latitude from +Y.
    Equirect(&'a humanart_io::Image),
}

/// The entry points of `ibl.wgsl`, in pipeline order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Entry {
    Generate,
    EquirectToCube,
    Downsample,
    Prefilter,
    IrradianceSh,
    Dfg,
}

const ENTRIES: [Entry; 6] =
    [Entry::Generate, Entry::EquirectToCube, Entry::Downsample, Entry::Prefilter, Entry::IrradianceSh, Entry::Dfg];

impl Entry {
    const fn name(self) -> &'static str {
        match self {
            Self::Generate => "generate",
            Self::EquirectToCube => "equirect_to_cube",
            Self::Downsample => "downsample",
            Self::Prefilter => "prefilter",
            Self::IrradianceSh => "irradiance_sh",
            Self::Dfg => "dfg",
        }
    }

    /// The bindings of the entry point (see the shader).
    fn layout(self) -> Vec<wgpu::BindGroupLayoutEntry> {
        let cs = wgpu::ShaderStages::COMPUTE;
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: cs, ty, count: None };
        let params = entry(
            0,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<IblUniforms>() as u64),
            },
        );
        let storage_texture = |binding, view_dimension| {
            entry(
                binding,
                wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension,
                },
            )
        };
        let texture = |binding, view_dimension| {
            entry(
                binding,
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension,
                    multisampled: false,
                },
            )
        };
        let sampler = entry(3, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering));
        let cube_out = storage_texture(1, wgpu::TextureViewDimension::D2Array);
        match self {
            Self::Generate => vec![params, cube_out],
            Self::EquirectToCube => vec![params, cube_out, texture(2, wgpu::TextureViewDimension::D2), sampler],
            Self::Downsample => vec![params, cube_out, texture(4, wgpu::TextureViewDimension::D2Array)],
            Self::Prefilter => vec![params, cube_out, sampler, texture(5, wgpu::TextureViewDimension::Cube)],
            Self::IrradianceSh => vec![
                params,
                sampler,
                texture(5, wgpu::TextureViewDimension::Cube),
                entry(
                    6,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
            Self::Dfg => vec![params, storage_texture(7, wgpu::TextureViewDimension::D2)],
        }
    }
}

/// The IBL resources and their pipelines. See the module documentation.
#[derive(Debug)]
pub struct Ibl {
    layouts: Vec<wgpu::BindGroupLayout>,
    pipelines: Vec<wgpu::ComputePipeline>,
    env: TextureId,
    specular: TextureId,
    dfg: TextureId,
    sh: BufferId,
    params: BufferId,
    equirect: Option<TextureId>,
    env_cube: wgpu::TextureView,
    specular_cube: wgpu::TextureView,
    dfg_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
}

/// A view of one mip level of a cube texture as a 6-layer 2D array.
fn level_view(texture: &wgpu::Texture, mip: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("ibl cube level"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        base_mip_level: mip,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(6),
        ..Default::default()
    })
}

/// A cube view of every mip level.
fn cube_view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("ibl cube"),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    })
}

fn raw_texture(resources: &Resources, id: TextureId) -> Result<wgpu::Texture, RenderError> {
    Ok(resources.texture(id).ok_or(humanart_gpu::GpuError::Stale { what: "ibl texture" })?.raw().clone())
}

/// Runs pipeline setup inside error scopes where their answer can be read at once (native), so a
/// bad shader or pipeline is a `Result`. In a browser a scope only answers asynchronously: the
/// work runs unguarded and its errors arrive through the device's event queue, as for every other
/// setup call of `humanart-gpu`.
fn guarded<T>(device: &wgpu::Device, what: &'static str, work: impl FnOnce() -> T) -> Result<T, RenderError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Ok(humanart_gpu::error::capture(device, what, work)?)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (device, what);
        Ok(work())
    }
}

fn raw_buffer(resources: &Resources, id: BufferId) -> Result<wgpu::Buffer, RenderError> {
    Ok(resources.buffer(id).ok_or(humanart_gpu::GpuError::Stale { what: "ibl buffer" })?.raw().clone())
}

impl Ibl {
    /// Creates the pipelines and textures and computes the DFG table (one submission).
    pub fn new(resources: &mut Resources) -> Result<Self, RenderError> {
        let device = resources.device().clone();
        let composed = humanart_shaders::compose("render/ibl.wgsl", &[])?;
        let module = guarded(&device, "ibl shader", || {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("render/ibl.wgsl"),
                source: wgpu::ShaderSource::Wgsl(composed.source().into()),
            })
        })?;
        let mut layouts = Vec::with_capacity(ENTRIES.len());
        let mut pipelines = Vec::with_capacity(ENTRIES.len());
        for entry in ENTRIES {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(entry.name()),
                entries: &entry.layout(),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry.name()),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipeline = guarded(&device, "ibl pipeline", || {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry.name()),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry.name()),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
            })?;
            layouts.push(layout);
            pipelines.push(pipeline);
        }
        let cube_usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING;
        let format = wgpu::TextureFormat::Rgba16Float;
        let tag = |purpose| Tag::shared(Category::Textures, purpose);
        let (env, _) = resources.create_texture(
            &TextureDesc::array(ENV_SIZE, ENV_SIZE, 6, format, cube_usage, tag("ibl environment")).mips(ENV_MIPS),
        )?;
        let (specular, _) = resources.create_texture(
            &TextureDesc::array(SPECULAR_SIZE, SPECULAR_SIZE, 6, format, cube_usage, tag("ibl specular"))
                .mips(SPECULAR_MIPS),
        )?;
        let (dfg, _) =
            resources.create_texture(&TextureDesc::d2(DFG_SIZE, DFG_SIZE, format, cube_usage, tag("ibl dfg")))?;
        let (sh, _) = resources.create_buffer(&BufferDesc {
            size: 9 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            tag: Tag::shared(Category::Textures, "ibl sh irradiance"),
        })?;
        let (params, _) = resources.create_buffer(&BufferDesc {
            size: PARAM_STRIDE * PARAM_SLOTS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            tag: Tag::shared(Category::Uniforms, "ibl parameters"),
        })?;
        let sampler = resources
            .sampler(&humanart_gpu::resources::SamplerKey {
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..humanart_gpu::resources::SamplerKey::LINEAR_CLAMP
            })?
            .clone();
        let env_texture = raw_texture(resources, env)?;
        let specular_texture = raw_texture(resources, specular)?;
        let dfg_texture = raw_texture(resources, dfg)?;
        let ibl = Self {
            env_cube: cube_view(&env_texture),
            specular_cube: cube_view(&specular_texture),
            dfg_view: dfg_texture.create_view(&wgpu::TextureViewDescriptor::default()),
            layouts,
            pipelines,
            env,
            specular,
            dfg,
            sh,
            params,
            equirect: None,
            sampler,
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ibl dfg") });
        let slot = ibl.write_params(resources, 0, [DFG_SIZE, 0, DFG_SAMPLES, 0], [0.0; 4], [0.0; 4])?;
        {
            let bind =
                ibl.bind_group(resources, Entry::Dfg, slot, &[(7, wgpu::BindingResource::TextureView(&ibl.dfg_view))])?;
            let mut pass = encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ibl dfg"), timestamp_writes: None });
            ibl.dispatch(&mut pass, Entry::Dfg, &bind, DFG_SIZE, 1);
        }
        resources.queue().submit([encoder.finish()]);
        Ok(ibl)
    }

    fn write_params(
        &self,
        resources: &mut Resources,
        slot: u64,
        ints: [u32; 4],
        floats: [f32; 4],
        sun: [f32; 4],
    ) -> Result<u64, RenderError> {
        let block = IblUniforms { ints, floats, sun };
        resources.write_buffer(self.params, slot * PARAM_STRIDE, bytes_of(&block))?;
        Ok(slot)
    }

    fn bind_group(
        &self,
        resources: &Resources,
        entry: Entry,
        slot: u64,
        extra: &[(u32, wgpu::BindingResource<'_>)],
    ) -> Result<wgpu::BindGroup, RenderError> {
        let index = ENTRIES.iter().position(|&e| e == entry).unwrap_or(0);
        let layout = self.layouts.get(index).ok_or(RenderError::InvalidParameter { what: "ibl entry" })?;
        let params = raw_buffer(resources, self.params)?;
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &params,
                offset: slot * PARAM_STRIDE,
                size: wgpu::BufferSize::new(std::mem::size_of::<IblUniforms>() as u64),
            }),
        }];
        entries.extend(
            extra
                .iter()
                .map(|(binding, resource)| wgpu::BindGroupEntry { binding: *binding, resource: resource.clone() }),
        );
        Ok(resources.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(entry.name()),
            layout,
            entries: &entries,
        }))
    }

    fn dispatch(&self, pass: &mut wgpu::ComputePass<'_>, entry: Entry, bind: &wgpu::BindGroup, size: u32, layers: u32) {
        let index = ENTRIES.iter().position(|&e| e == entry).unwrap_or(0);
        let Some(pipeline) = self.pipelines.get(index) else { return };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind, &[]);
        if entry == Entry::IrradianceSh {
            pass.dispatch_workgroups(1, 1, 1);
        } else {
            pass.dispatch_workgroups(size.div_ceil(8), size.div_ceil(8), layers);
        }
    }

    /// Records the whole preprocessing of `source` into `encoder`: environment cube, its mips, the
    /// specular cube and the SH irradiance. The parameter blocks are written immediately, so submit
    /// the encoder before the next `build`.
    pub fn build(
        &mut self,
        resources: &mut Resources,
        encoder: &mut wgpu::CommandEncoder,
        source: EnvironmentSource<'_>,
    ) -> Result<(), RenderError> {
        let env_texture = raw_texture(resources, self.env)?;
        let specular_texture = raw_texture(resources, self.specular)?;
        let level0 = level_view(&env_texture, 0);
        let mut bind_groups: Vec<(Entry, wgpu::BindGroup, u32)> = Vec::new();
        // 1. The source into level 0.
        match source {
            EnvironmentSource::Procedural(kind) => {
                let sun = match kind {
                    ProceduralEnvironment::ClearSky { sun_direction }
                    | ProceduralEnvironment::Sunset { sun_direction } => {
                        let s = sun_direction.normalize_or(humanart_math::Vec3::Y);
                        [s.x, s.y, s.z, 0.0]
                    }
                    _ => [0.0, 1.0, 0.0, 0.0],
                };
                let slot = self.write_params(resources, 1, [ENV_SIZE, 0, 0, kind.code()], [0.0; 4], sun)?;
                let bind = self.bind_group(
                    resources,
                    Entry::Generate,
                    slot,
                    &[(1, wgpu::BindingResource::TextureView(&level0))],
                )?;
                bind_groups.push((Entry::Generate, bind, ENV_SIZE));
            }
            EnvironmentSource::Equirect(image) => {
                let texture = self.upload_equirect(resources, image)?;
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                let slot = self.write_params(resources, 1, [ENV_SIZE, 0, 0, 0], [0.0; 4], [0.0; 4])?;
                let bind = self.bind_group(
                    resources,
                    Entry::EquirectToCube,
                    slot,
                    &[
                        (1, wgpu::BindingResource::TextureView(&level0)),
                        (2, wgpu::BindingResource::TextureView(&view)),
                        (3, wgpu::BindingResource::Sampler(&self.sampler)),
                    ],
                )?;
                bind_groups.push((Entry::EquirectToCube, bind, ENV_SIZE));
            }
        }
        // 2. The mip chain.
        for mip in 1..ENV_MIPS {
            let size = (ENV_SIZE >> mip).max(1);
            let slot = self.write_params(resources, 1 + u64::from(mip), [size, mip, 0, 0], [0.0; 4], [0.0; 4])?;
            let source = level_view(&env_texture, mip - 1);
            let target = level_view(&env_texture, mip);
            let bind = self.bind_group(
                resources,
                Entry::Downsample,
                slot,
                &[(1, wgpu::BindingResource::TextureView(&target)), (4, wgpu::BindingResource::TextureView(&source))],
            )?;
            bind_groups.push((Entry::Downsample, bind, size));
        }
        // 3. The specular cube, one level per roughness.
        for mip in 0..SPECULAR_MIPS {
            let size = (SPECULAR_SIZE >> mip).max(1);
            let roughness = mip as f32 / (SPECULAR_MIPS - 1) as f32;
            let slot = self.write_params(
                resources,
                10 + u64::from(mip),
                [size, mip, PREFILTER_SAMPLES, 0],
                [roughness, ENV_SIZE as f32, ENV_MIPS as f32, 0.0],
                [0.0; 4],
            )?;
            let target = level_view(&specular_texture, mip);
            let bind = self.bind_group(
                resources,
                Entry::Prefilter,
                slot,
                &[
                    (1, wgpu::BindingResource::TextureView(&target)),
                    (3, wgpu::BindingResource::Sampler(&self.sampler)),
                    (5, wgpu::BindingResource::TextureView(&self.env_cube)),
                ],
            )?;
            bind_groups.push((Entry::Prefilter, bind, size));
        }
        // 4. SH irradiance.
        let sh = raw_buffer(resources, self.sh)?;
        let slot = self.write_params(
            resources,
            20,
            [0, 0, SH_SAMPLES, 0],
            [0.0, ENV_SIZE as f32, ENV_MIPS as f32, 0.0],
            [0.0; 4],
        )?;
        let bind = self.bind_group(
            resources,
            Entry::IrradianceSh,
            slot,
            &[
                (3, wgpu::BindingResource::Sampler(&self.sampler)),
                (5, wgpu::BindingResource::TextureView(&self.env_cube)),
                (6, sh.as_entire_binding()),
            ],
        )?;
        bind_groups.push((Entry::IrradianceSh, bind, 1));
        // One pass per dispatch: each reads what the previous one wrote.
        for (entry, bind, size) in &bind_groups {
            let mut pass = encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(entry.name()), timestamp_writes: None });
            self.dispatch(&mut pass, *entry, bind, *size, 6);
        }
        Ok(())
    }

    /// Uploads an equirectangular image as an f16 texture (kept for the next build).
    fn upload_equirect(
        &mut self,
        resources: &mut Resources,
        image: &humanart_io::Image,
    ) -> Result<wgpu::Texture, RenderError> {
        let limit = MAX_EQUIRECT_SIDE.min(resources.limits().max_texture_dimension_2d);
        if image.width > limit || image.height > limit {
            return Err(RenderError::TooLarge {
                what: "environment image side",
                value: u64::from(image.width.max(image.height)),
                limit: u64::from(limit),
            });
        }
        if image.channels < 3 {
            return Err(RenderError::InvalidParameter { what: "an environment image needs 3 or 4 channels" });
        }
        let channels = usize::from(image.channels);
        let value = |i: usize| -> f32 {
            match &image.samples {
                humanart_io::Samples::U8(v) => v.get(i).map_or(0.0, |&x| f32::from(x) / 255.0),
                humanart_io::Samples::U16(v) => v.get(i).map_or(0.0, |&x| f32::from(x) / 65535.0),
                humanart_io::Samples::F16(v) => v.get(i).map_or(0.0, |&x| humanart_math::f16_bits_to_f32(x)),
                humanart_io::Samples::F32(v) => v.get(i).copied().unwrap_or(0.0),
            }
        };
        let pixels = image.pixel_count();
        let mut halves: Vec<u16> = Vec::with_capacity(pixels * 4);
        for p in 0..pixels {
            for c in 0..3 {
                let v = value(p * channels + c);
                // Finite and non-negative radiance; f16 saturates at 65504.
                let v = if v.is_finite() { v.clamp(0.0, 65504.0) } else { 0.0 };
                halves.push(f32_to_f16_bits(v));
            }
            halves.push(f32_to_f16_bits(1.0));
        }
        if let Some(old) = self.equirect.take() {
            let _ = resources.destroy_texture(old);
        }
        let (id, _) = resources.create_texture(&TextureDesc::d2(
            image.width,
            image.height,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            Tag::shared(Category::Textures, "ibl equirect source"),
        ))?;
        self.equirect = Some(id);
        resources.write_texture_level(id, 0, 0, slice_as_bytes(&halves))?;
        raw_texture(resources, id)
    }

    /// The prefiltered specular cube (all levels).
    pub fn specular_view(&self) -> &wgpu::TextureView {
        &self.specular_cube
    }

    /// The environment cube (all levels).
    pub fn environment_view(&self) -> &wgpu::TextureView {
        &self.env_cube
    }

    /// The DFG table.
    pub fn dfg_view(&self) -> &wgpu::TextureView {
        &self.dfg_view
    }

    /// The nine SH irradiance coefficients (vec4 each).
    pub fn sh_buffer(&self) -> BufferId {
        self.sh
    }

    /// The trilinear clamp sampler the IBL is sampled with.
    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    /// Frees the textures and buffers.
    pub fn destroy(&mut self, resources: &mut Resources) {
        for id in [self.env, self.specular, self.dfg] {
            let _ = resources.destroy_texture(id);
        }
        if let Some(id) = self.equirect.take() {
            let _ = resources.destroy_texture(id);
        }
        let _ = resources.destroy_buffer(self.sh);
        let _ = resources.destroy_buffer(self.params);
    }
}
