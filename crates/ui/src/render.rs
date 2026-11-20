//! The GPU side of the painter: one pipeline, one storage buffer of instances, one draw call per
//! batch `[UI-02]`.
//!
//! [`UiRenderer`] owns everything the UI pass needs: the pipeline of `shaders/ui/quad.wgsl` for the
//! target format, the instance storage buffer, the glyph atlas (a 2D array texture, one layer per
//! atlas page) and the image textures registered with [`UiRenderer::add_image`]. Pipelines, layouts
//! and bind groups are created once `[PERF-R4]`; the instance buffer and its bind group are only
//! re-created when a frame needs more instances than ever before (capacity doubles). A frame is
//! [`UiRenderer::prepare`] (two queue writes) then [`UiRenderer::draw`] into a render pass, or
//! [`UiRenderer::record`], which opens the pass itself.
//!
//! The renderer keeps CPU copies of the atlas and the images, so it can rebuild itself on a new
//! device after a loss ([`humanart_gpu::recovery::Rebuild`]).

use humanart_gpu::recovery::Rebuild;
use humanart_gpu::resources::Resources;
use humanart_gpu::{Gpu, GpuError, pod_struct};

use crate::color::Color;
use crate::error::UiError;
use crate::paint::{DrawList, TextureId};

/// The shader of the UI pass.
pub const SHADER: &str = "ui/quad.wgsl";

/// The fewest instances the buffer holds.
const MIN_INSTANCES: u64 = 1024;

/// Bytes per instance.
const INSTANCE_BYTES: u64 = 112;

/// The largest image side accepted by [`UiRenderer::add_image`].
pub const MAX_IMAGE_SIDE: u32 = 8192;

pod_struct! {
    /// Mirrors `struct Globals` in `shaders/ui/quad.wgsl`.
    #[derive(Debug, PartialEq)]
    struct Globals(size = 16, align = 4) {
        viewport: [f32; 2],
        pad: [f32; 2],
    }
}

/// The glyph atlas as the renderer uploads it: `layers` pages of `width × height` RGBA8 texels
/// (multi-channel distances, not colours), page after page.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtlasImage {
    /// Width of a page in texels.
    pub width: u32,
    /// Height of a page in texels.
    pub height: u32,
    /// The number of pages.
    pub layers: u32,
    /// `width × height × layers × 4` bytes.
    pub rgba: Vec<u8>,
}

impl AtlasImage {
    /// A one-texel atlas, for a UI without text.
    pub fn empty() -> Self {
        Self { width: 1, height: 1, layers: 1, rgba: vec![0; 4] }
    }

    /// Whether the data length matches the dimensions.
    pub fn is_consistent(&self) -> bool {
        let expect = u64::from(self.width) * u64::from(self.height) * u64::from(self.layers) * 4;
        self.width > 0 && self.height > 0 && self.layers > 0 && self.rgba.len() as u64 == expect
    }
}

/// A registered image texture with the CPU copy it is rebuilt from.
#[derive(Debug)]
struct ImageSlot {
    width: u32,
    height: u32,
    /// The texels, or empty for an image written on the GPU ([`UiRenderer::add_gpu_image`]).
    rgba: Vec<u8>,
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
}

/// Counters of the last prepared frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderStats {
    /// Instances uploaded.
    pub instances: u32,
    /// Draw calls issued by the last [`UiRenderer::draw`].
    pub draw_calls: u32,
    /// Times the instance buffer grew since the renderer was made.
    pub buffer_growths: u32,
}

/// The UI pass on the GPU.
#[derive(Debug)]
pub struct UiRenderer {
    format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    layout0: wgpu::BindGroupLayout,
    layout1: wgpu::BindGroupLayout,
    globals: wgpu::Buffer,
    globals_value: Globals,
    instances: wgpu::Buffer,
    capacity: u64,
    sampler: wgpu::Sampler,
    atlas: AtlasImage,
    atlas_texture: wgpu::Texture,
    group0: wgpu::BindGroup,
    /// Index = [`TextureId`]; slot 0 is the white texel, removed slots are `None`.
    images: Vec<Option<ImageSlot>>,
    stats: RenderStats,
}

/// Runs GPU setup work and turns validation and memory errors into `Err` (native; on the web the
/// errors arrive asynchronously through the device's error handler).
fn guarded<T>(device: &wgpu::Device, what: &'static str, work: impl FnOnce() -> T) -> Result<T, UiError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        humanart_gpu::error::capture(device, what, work).map_err(UiError::Gpu)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (device, what);
        Ok(work())
    }
}

impl UiRenderer {
    /// The renderer for render targets of `format`, with `atlas` as the glyph atlas.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        atlas: AtlasImage,
    ) -> Result<Self, UiError> {
        check_atlas(device, &atlas)?;
        let composed = humanart_shaders::compose(SHADER, &[])?;
        let module = guarded(device, "UI shader module", || {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(SHADER),
                source: wgpu::ShaderSource::Wgsl(composed.source().into()),
            })
        })?;
        let layout0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ui globals + instances + atlas"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layout1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ui image"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui"),
            bind_group_layouts: &[Some(&layout0), Some(&layout1)],
            immediate_size: 0,
        });
        let constants: &[(&str, f64)] = &[("ENCODE_SRGB", if format.is_srgb() { 0.0 } else { 1.0 })];
        let options = || wgpu::PipelineCompilationOptions { constants, zero_initialize_workgroup_memory: true };
        let pipeline = guarded(device, "UI pipeline", || {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ui"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: options(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    compilation_options: options(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        })?;
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui globals"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instances = instance_buffer(device, MIN_INSTANCES);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ui linear"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let atlas_texture = upload_atlas(device, queue, &atlas);
        let group0 = make_group0(device, &layout0, &globals, &instances, &atlas_texture, &sampler);
        let mut renderer = Self {
            format,
            pipeline,
            layout0,
            layout1,
            globals,
            globals_value: Globals { viewport: [0.0, 0.0], pad: [0.0, 0.0] },
            instances,
            capacity: MIN_INSTANCES,
            sampler,
            atlas,
            atlas_texture,
            group0,
            images: Vec::new(),
            stats: RenderStats::default(),
        };
        renderer.images.push(Some(renderer.make_image(device, Some(queue), 1, 1, vec![255; 4])));
        queue.write_buffer(&renderer.globals, 0, humanart_gpu::pod::bytes_of(&renderer.globals_value));
        Ok(renderer)
    }

    /// The render-target format the pipeline was built for.
    #[inline]
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// The counters of the last frame.
    #[inline]
    pub fn stats(&self) -> RenderStats {
        self.stats
    }

    /// Replaces the glyph atlas (fonts loaded or changed).
    pub fn set_atlas(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, atlas: AtlasImage) -> Result<(), UiError> {
        check_atlas(device, &atlas)?;
        self.atlas_texture = upload_atlas(device, queue, &atlas);
        self.atlas = atlas;
        self.group0 =
            make_group0(device, &self.layout0, &self.globals, &self.instances, &self.atlas_texture, &self.sampler);
        Ok(())
    }

    /// Registers an image: `width × height` texels of 8-bit sRGB RGBA with straight alpha.
    pub fn add_image(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
    ) -> Result<TextureId, UiError> {
        check_image(width, height, &rgba)?;
        let slot = self.make_image(device, Some(queue), width, height, rgba);
        self.insert_image(slot)
    }

    /// Registers an image of `width × height` texels whose contents are written on the GPU through
    /// [`UiRenderer::image_texture`] (a copy or a render pass into the `Rgba8UnormSrgb` texture,
    /// which can be the copy destination of an `Rgba8Unorm` texture holding sRGB-encoded bytes).
    /// It starts transparent black. No CPU copy is kept, so after [`UiRenderer::rebuild`] it is
    /// transparent again until the next write. This is how a GPU-rendered view (a 3D viewport)
    /// becomes part of the UI pass and gets the right z-order under popups and tooltips.
    pub fn add_gpu_image(&mut self, device: &wgpu::Device, width: u32, height: u32) -> Result<TextureId, UiError> {
        check_side(width, height)?;
        let slot = self.make_image(device, None, width, height, Vec::new());
        self.insert_image(slot)
    }

    /// Gives the image `id` a new size, keeping its id: its texture is re-created transparent
    /// black and it becomes an image written on the GPU (as from [`UiRenderer::add_gpu_image`]).
    /// Does nothing when it already is a GPU image of `width × height`.
    pub fn resize_gpu_image(
        &mut self,
        device: &wgpu::Device,
        id: TextureId,
        width: u32,
        height: u32,
    ) -> Result<(), UiError> {
        check_side(width, height)?;
        let count = self.images.len() as u64;
        let current = match self.images.get(id.0 as usize) {
            Some(Some(slot)) if id != TextureId::NONE => (slot.width, slot.height, slot.rgba.is_empty()),
            _ => return Err(UiError::Size { what: "image id", got: u64::from(id.0), limit: count }),
        };
        if current == (width, height, true) {
            return Ok(());
        }
        let slot = self.make_image(device, None, width, height, Vec::new());
        if let Some(entry) = self.images.get_mut(id.0 as usize) {
            *entry = Some(slot);
        }
        Ok(())
    }

    /// The texture of the registered image `id` (`Rgba8UnormSrgb`, usage `TEXTURE_BINDING |
    /// COPY_DST`, plus `RENDER_ATTACHMENT` for images from [`UiRenderer::add_gpu_image`]), or
    /// `None` for [`TextureId::NONE`], an unknown or a removed id. A GPU write into it shows in the
    /// next UI pass; it does not update the CPU copy of an image from [`UiRenderer::add_image`].
    pub fn image_texture(&self, id: TextureId) -> Option<&wgpu::Texture> {
        if id == TextureId::NONE {
            return None;
        }
        self.images.get(id.0 as usize).and_then(Option::as_ref).map(|slot| &slot.texture)
    }

    /// Puts `slot` at the lowest free id (never 0, the white texel).
    fn insert_image(&mut self, slot: ImageSlot) -> Result<TextureId, UiError> {
        let index = if let Some(free) = self.images.iter().skip(1).position(Option::is_none) {
            free + 1
        } else {
            self.images.push(None);
            self.images.len() - 1
        };
        let id = u32::try_from(index).map_err(|_| UiError::Size {
            what: "image count",
            got: index as u64,
            limit: u64::from(u32::MAX),
        })?;
        if let Some(entry) = self.images.get_mut(index) {
            *entry = Some(slot);
        }
        Ok(TextureId(id))
    }

    /// Replaces the texels of a registered image of the same size.
    pub fn update_image(&mut self, queue: &wgpu::Queue, id: TextureId, rgba: &[u8]) -> Result<(), UiError> {
        let Some(Some(slot)) = self.images.get_mut(id.0 as usize).filter(|_| id != TextureId::NONE) else {
            return Err(UiError::Size { what: "image id", got: u64::from(id.0), limit: self.images.len() as u64 });
        };
        check_image(slot.width, slot.height, rgba)?;
        slot.rgba.clear();
        slot.rgba.extend_from_slice(rgba);
        write_image(queue, &slot.texture, slot.width, slot.height, &slot.rgba);
        Ok(())
    }

    /// Forgets a registered image; its id may be handed out again.
    pub fn remove_image(&mut self, id: TextureId) {
        if id != TextureId::NONE
            && let Some(entry) = self.images.get_mut(id.0 as usize)
        {
            *entry = None;
        }
    }

    /// An image slot with the texels `rgba` written through `queue`, or, with an empty `rgba`, an
    /// image written on the GPU, left as wgpu creates it (zeroed: transparent black).
    fn make_image(
        &self,
        device: &wgpu::Device,
        queue: Option<&wgpu::Queue>,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
    ) -> ImageSlot {
        let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        if rgba.is_empty() {
            usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ui image"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage,
            view_formats: &[],
        });
        if let Some(queue) = queue.filter(|_| !rgba.is_empty()) {
            write_image(queue, &texture, width, height, &rgba);
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui image"),
            layout: &self.layout1,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) }],
        });
        ImageSlot { width, height, rgba, texture, bind_group }
    }

    /// Uploads a finished draw list ([`DrawList::finish`]) for a target of `width × height` physical
    /// pixels. Grows the instance buffer when the frame is the largest so far; otherwise two queue
    /// writes and no allocation in this crate.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        list: &DrawList,
        width: u32,
        height: u32,
    ) -> Result<(), UiError> {
        #[allow(clippy::cast_precision_loss)] // INVARIANT: target sizes are at most 16384.
        let viewport = [width.max(1) as f32, height.max(1) as f32];
        if self.globals_value.viewport != viewport {
            self.globals_value.viewport = viewport;
            queue.write_buffer(&self.globals, 0, humanart_gpu::pod::bytes_of(&self.globals_value));
        }
        let instances = list.instances();
        let count = instances.len() as u64;
        if count > self.capacity {
            let limit = device.limits().max_storage_buffer_binding_size / INSTANCE_BYTES;
            if count > limit {
                return Err(UiError::Size { what: "UI instances in a frame", got: count, limit });
            }
            let capacity = count.next_power_of_two().clamp(MIN_INSTANCES, limit.max(MIN_INSTANCES));
            self.instances = instance_buffer(device, capacity);
            self.capacity = capacity;
            self.group0 =
                make_group0(device, &self.layout0, &self.globals, &self.instances, &self.atlas_texture, &self.sampler);
            self.stats.buffer_growths += 1;
        }
        if !instances.is_empty() {
            queue.write_buffer(&self.instances, 0, humanart_gpu::pod::slice_as_bytes(instances));
        }
        self.stats.instances = u32::try_from(count).unwrap_or(u32::MAX);
        Ok(())
    }

    /// Draws the prepared list into an open render pass whose target has this renderer's format:
    /// one draw call per batch. Batches naming a removed image are skipped. Returns the number of
    /// draw calls.
    pub fn draw(&mut self, pass: &mut wgpu::RenderPass<'_>, list: &DrawList) -> u32 {
        let mut calls = 0;
        let uploaded = self.stats.instances;
        if list.instances().is_empty() {
            self.stats.draw_calls = 0;
            return 0;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group0, &[]);
        let mut bound = None;
        for batch in list.batches() {
            let end = batch.end.min(uploaded);
            if batch.start >= end {
                continue;
            }
            let Some(Some(image)) = self.images.get(batch.texture.0 as usize) else { continue };
            if bound != Some(batch.texture) {
                pass.set_bind_group(1, &image.bind_group, &[]);
                bound = Some(batch.texture);
            }
            pass.draw(0..4, batch.start..end);
            calls += 1;
        }
        self.stats.draw_calls = calls;
        calls
    }

    /// Opens a render pass on `view` (cleared to `clear` when given, else loaded), draws the
    /// prepared list and closes the pass. Returns the number of draw calls.
    pub fn record(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        clear: Option<Color>,
        list: &DrawList,
    ) -> u32 {
        self.record_timed(encoder, view, clear, list, None)
    }

    /// [`UiRenderer::record`] with timestamp writes on the pass (from
    /// `humanart_gpu::profiler::GpuProfiler::render_pass_writes`): the GPU time of the UI pass
    /// `[UI-10]`.
    pub fn record_timed(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        clear: Option<Color>,
        list: &DrawList,
        timestamps: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) -> u32 {
        let load = match clear {
            Some(c) => {
                let c = if self.format.is_srgb() {
                    c
                } else {
                    let [r, g, b, a] = c.to_srgb8();
                    Color::linear(
                        f32::from(r) / 255.0,
                        f32::from(g) / 255.0,
                        f32::from(b) / 255.0,
                        f32::from(a) / 255.0,
                    )
                };
                wgpu::LoadOp::Clear(wgpu::Color {
                    r: f64::from(c.r * c.a),
                    g: f64::from(c.g * c.a),
                    b: f64::from(c.b * c.a),
                    a: f64::from(c.a),
                })
            }
            None => wgpu::LoadOp::Load,
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ui"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
            })],
            timestamp_writes: timestamps,
            ..Default::default()
        });
        self.draw(&mut pass, list)
    }

    /// Redraws only `region` (physical pixels, from [`crate::ui::Dirty::Region`]) of `view`, keeping
    /// the rest of its content: the dirty-region path of `[UI-01]`. The target must hold the
    /// previous frame of the same UI (a persistent texture, or a surface whose previous image is
    /// preserved), and the list must start with an opaque background
    /// ([`crate::ui::Ui::set_background`]) so the region is painted over completely. Returns the
    /// number of draw calls (0 when the region is empty).
    pub fn record_region(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        list: &DrawList,
        region: crate::geom::Rect,
    ) -> u32 {
        let [tw, th] = self.globals_value.viewport;
        let x0 = region.x.max(0.0).min(tw);
        let y0 = region.y.max(0.0).min(th);
        let x1 = region.right().max(x0).min(tw);
        let y1 = region.bottom().max(y0).min(th);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        // The values are clamped to the target, at most 16384.
        let (x, y, w, h) =
            (x0.floor() as u32, y0.floor() as u32, (x1.ceil() - x0.floor()) as u32, (y1.ceil() - y0.floor()) as u32);
        if w == 0 || h == 0 {
            self.stats.draw_calls = 0;
            return 0;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ui region"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_scissor_rect(x, y, w, h);
        self.draw(&mut pass, list)
    }

    /// Re-creates every GPU object on `device` from the CPU copies (after a device loss).
    pub fn rebuild(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), UiError> {
        let mut fresh = Self::new(device, queue, self.format, std::mem::replace(&mut self.atlas, AtlasImage::empty()))?;
        for slot in self.images.drain(..).skip(1) {
            let image = slot.map(|old| fresh.make_image(device, Some(queue), old.width, old.height, old.rgba));
            fresh.images.push(image);
        }
        fresh.stats.buffer_growths = self.stats.buffer_growths;
        *self = fresh;
        Ok(())
    }
}

impl Rebuild for UiRenderer {
    fn rebuild_after_loss(&mut self, gpu: &Gpu, _resources: &mut Resources) -> Result<(), GpuError> {
        self.rebuild(&gpu.device, &gpu.queue).map_err(|e| match e {
            UiError::Gpu(g) => g,
            other => GpuError::Validation { scope: "UI renderer rebuild", message: other.to_string() },
        })
    }
}

/// Checks the atlas against its own data and the device's texture limits.
fn check_atlas(device: &wgpu::Device, atlas: &AtlasImage) -> Result<(), UiError> {
    if !atlas.is_consistent() {
        return Err(UiError::Size { what: "atlas data length", got: atlas.rgba.len() as u64, limit: 0 });
    }
    let limits = device.limits();
    let side = atlas.width.max(atlas.height);
    if side > limits.max_texture_dimension_2d {
        return Err(UiError::Size {
            what: "atlas page side",
            got: u64::from(side),
            limit: u64::from(limits.max_texture_dimension_2d),
        });
    }
    if atlas.layers > limits.max_texture_array_layers {
        return Err(UiError::Size {
            what: "atlas pages",
            got: u64::from(atlas.layers),
            limit: u64::from(limits.max_texture_array_layers),
        });
    }
    Ok(())
}

/// Checks an image's size and data length.
fn check_image(width: u32, height: u32, rgba: &[u8]) -> Result<(), UiError> {
    check_side(width, height)?;
    let expect = u64::from(width) * u64::from(height) * 4;
    if rgba.len() as u64 != expect {
        return Err(UiError::Size { what: "image data length", got: rgba.len() as u64, limit: expect });
    }
    Ok(())
}

/// Checks an image's sides against `1..=MAX_IMAGE_SIDE`.
fn check_side(width: u32, height: u32) -> Result<(), UiError> {
    if width == 0 || height == 0 || width > MAX_IMAGE_SIDE || height > MAX_IMAGE_SIDE {
        return Err(UiError::Size {
            what: "image side",
            got: u64::from(width.max(height)),
            limit: u64::from(MAX_IMAGE_SIDE),
        });
    }
    Ok(())
}

fn instance_buffer(device: &wgpu::Device, capacity: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ui instances"),
        size: capacity * INSTANCE_BYTES,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn upload_atlas(device: &wgpu::Device, queue: &wgpu::Queue, atlas: &AtlasImage) -> wgpu::Texture {
    let size = wgpu::Extent3d { width: atlas.width, height: atlas.height, depth_or_array_layers: atlas.layers };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ui glyph atlas"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        texture.as_image_copy(),
        &atlas.rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(atlas.width * 4),
            rows_per_image: Some(atlas.height),
        },
        size,
    );
    texture
}

fn write_image(queue: &wgpu::Queue, texture: &wgpu::Texture, width: u32, height: u32, rgba: &[u8]) {
    queue.write_texture(
        texture.as_image_copy(),
        rgba,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(width * 4), rows_per_image: None },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
}

fn make_group0(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    globals: &wgpu::Buffer,
    instances: &wgpu::Buffer,
    atlas: &wgpu::Texture,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    let view = atlas.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ui globals + instances + atlas"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: globals.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: instances.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{Point, Rect};
    use crate::paint::{Axis, Corners, Layer};
    use crate::tests::{assert_golden, gpu, gpu_lock, pixel, render, renderer};
    use crate::theme::Theme;

    /// A frame using every primitive, in logical pixels, themed like the editor.
    fn showcase(list: &mut DrawList, theme: &Theme, scale: f32, image: TextureId) {
        let p = theme.palette;
        let m = theme.metrics;
        let (w, h) = ((240.0 * scale).round() as u32, (160.0 * scale).round() as u32);
        list.begin(w, h, scale);
        list.rect(Rect::new(0.0, 0.0, 240.0, 160.0), p.panel, 0.0);
        list.rect(Rect::new(8.0, 8.0, 224.0, 144.0), p.bg, m.radius);
        // A card with a shadow and a border.
        let card = Rect::new(20.0, 20.0, 110.0, 60.0);
        let s = theme.shadows.menu;
        list.shadow(card.translate(Point::new(0.0, s.offset_y)), p.shadow, s.radius, s.blur);
        list.bordered(card, p.surface, m.radius, m.border, p.border_strong);
        // Buttons: primary, secondary, focused.
        list.rect(Rect::new(30.0, 30.0, 40.0, m.control_height), p.accent, m.radius_sm);
        let secondary = Rect::new(76.0, 30.0, 44.0, m.control_height);
        list.bordered(secondary, p.hover, m.radius_sm, m.border, p.border_strong);
        list.focus_ring(secondary, m.radius_sm, m.focus_ring_width, m.focus_ring_offset, p.focus_ring);
        // A slider: track, fill, thumb.
        list.rect(Rect::new(140.0, 30.0, 80.0, m.slider_track), p.active, m.slider_track * 0.5);
        list.rect(Rect::new(140.0, 30.0, 50.0, m.slider_track), p.accent, m.slider_track * 0.5);
        list.circle(Point::new(190.0, 32.0), m.slider_thumb * 0.5, p.text);
        // Gradients, a tab and lines.
        let (red, blue) = (Color::hex(0xFF0000), Color::hex(0x0000FF));
        list.gradient(Rect::new(140.0, 44.0, 80.0, 12.0), red, blue, 3.0, Axis::Horizontal);
        list.gradient(
            Rect::new(140.0, 60.0, 80.0, 20.0),
            Color::WHITE,
            Color::BLACK,
            Corners::top(6.0),
            Axis::Vertical,
        );
        list.line(Point::new(20.0, 95.0), Point::new(130.0, 120.0), 1.5, p.danger);
        list.line(Point::new(20.0, 130.0), Point::new(130.0, 130.0), 1.0, p.border_strong);
        // Clipped content: only the part inside the clip shows.
        list.push_clip(Rect::new(140.0, 90.0, 40.0, 40.0));
        list.rect(Rect::new(130.0, 80.0, 60.0, 60.0), p.positive, 30.0);
        list.pop_clip();
        // An image with rounded corners.
        list.image(Rect::new(190.0, 90.0, 32.0, 32.0), image, [0.0, 0.0, 1.0, 1.0], Color::WHITE, 6.0);
        // A tooltip recorded early but drawn on top.
        list.set_layer(Layer::Tooltip);
        let tip = Rect::new(150.0, 128.0, 60.0, 20.0);
        list.shadow(tip.translate(Point::new(0.0, 4.0)), p.shadow, 8.0, 10.0);
        list.bordered(tip, p.surface, m.radius, m.border, p.border_strong);
        list.set_layer(Layer::Base);
        list.rect(Rect::new(150.0, 135.0, 70.0, 6.0), p.warning, 0.0);
        list.finish();
    }

    /// A 4 x 4 checker image, sRGB RGBA8.
    fn checker() -> Vec<u8> {
        let mut out = Vec::new();
        for y in 0..4 {
            for x in 0..4 {
                out.extend_from_slice(if (x + y) % 2 == 0 { &[230, 120, 40, 255] } else { &[40, 90, 200, 255] });
            }
        }
        out
    }

    fn close(a: [u8; 4], b: [u8; 4]) -> bool {
        a.iter().zip(&b).all(|(x, y)| x.abs_diff(*y) <= 1)
    }

    #[test]
    fn ui_02_showcase_golden_at_100_150_200_percent() {
        // [UI-02] [UI-09] Every primitive at three UI scales, one golden each; few draw calls (the
        // image breaks the shape batch once).
        let _lock = gpu_lock();
        let g = gpu();
        let mut r = renderer(AtlasImage::empty());
        let image = r.add_image(&g.device, &g.queue, 4, 4, checker()).unwrap();
        let theme = Theme::dark();
        for (scale, name) in
            [(1.0, "painter_showcase_100"), (1.5, "painter_showcase_150"), (2.0, "painter_showcase_200")]
        {
            let mut list = DrawList::new();
            showcase(&mut list, &theme, scale, image);
            let (w, h) = ((240.0 * scale) as u32, (160.0 * scale) as u32);
            let rgba = render(&mut r, &list, w, h, theme.palette.panel);
            assert!(r.stats().draw_calls <= 3, "{:?}", r.stats());
            // Analytic checks: the primary button's centre is exactly the accent colour, the card's
            // top border is border_strong, the clip keeps the green disc out of x < 140, and the
            // tooltip covers the warning bar recorded after it.
            let px = |x: f32, y: f32| pixel(&rgba, w, (x * scale) as u32, (y * scale) as u32);
            assert!(close(px(50.0, 44.0), theme.palette.accent.to_srgb8()), "{:?}", px(50.0, 44.0));
            assert!(close(px(75.0, 20.2), theme.palette.border_strong.to_srgb8()), "{:?}", px(75.0, 20.2));
            assert!(close(px(139.0, 110.0), theme.palette.bg.to_srgb8()), "{:?}", px(139.0, 110.0));
            assert!(close(px(160.0, 110.0), theme.palette.positive.to_srgb8()));
            assert!(close(px(180.0, 138.0), theme.palette.surface.to_srgb8()), "{:?}", px(180.0, 138.0));
            assert_golden(name, w, h, &rgba, 2, (w * h / 1000) as usize);
        }
    }

    #[test]
    fn ui_08_themes_render_their_palettes() {
        // [UI-08] The same frame in the light, high-contrast and colour-neutral themes.
        let _lock = gpu_lock();
        let g = gpu();
        let mut r = renderer(AtlasImage::empty());
        let image = r.add_image(&g.device, &g.queue, 4, 4, checker()).unwrap();
        for (theme, name) in [
            (Theme::light(), "painter_light"),
            (Theme::high_contrast(), "painter_high_contrast"),
            (Theme::dark().colour_neutral(), "painter_colour_neutral"),
        ] {
            let mut list = DrawList::new();
            showcase(&mut list, &theme, 1.0, image);
            let rgba = render(&mut r, &list, 240, 160, theme.palette.panel);
            assert_golden(name, 240, 160, &rgba, 2, 40);
        }
    }

    #[test]
    fn ui_02_many_shapes_under_many_clips_are_one_draw_call() {
        // [UI-02] 6000 primitives under 200 different clips: one batch, one draw call; the buffer
        // grows once to fit and the next frame reuses it.
        let _lock = gpu_lock();
        let mut r = renderer(AtlasImage::empty());
        let mut list = DrawList::new();
        let frame = |list: &mut DrawList| {
            list.begin(640, 480, 1.0);
            for i in 0..2000 {
                let (x, y) = ((i % 40) as f32 * 16.0, (i / 40) as f32 * 9.0);
                if i % 10 == 0 {
                    list.pop_clip();
                    list.push_clip(Rect::new(x, 0.0, 400.0, 480.0));
                }
                list.shadow(Rect::new(x, y, 14.0, 7.0), Color::BLACK.fade(0.3), 2.0, 2.0);
                list.bordered(Rect::new(x, y, 14.0, 7.0), Color::hex(0x303030), 2.0, 1.0, Color::hex(0x505050));
                list.line(Point::new(x, y), Point::new(x + 10.0, y + 5.0), 1.0, Color::WHITE);
            }
            list.finish();
        };
        frame(&mut list);
        let _ = render(&mut r, &list, 640, 480, Color::BLACK);
        assert_eq!(r.stats().draw_calls, 1);
        assert_eq!(r.stats().buffer_growths, 1);
        frame(&mut list);
        let _ = render(&mut r, &list, 640, 480, Color::BLACK);
        assert_eq!(r.stats().buffer_growths, 1);
        assert!(r.stats().instances > 1024);
    }

    #[test]
    fn images_are_validated_and_ids_reused() {
        let _lock = gpu_lock();
        let g = gpu();
        let mut r = renderer(AtlasImage::empty());
        assert!(r.add_image(&g.device, &g.queue, 4, 4, vec![0; 10]).is_err());
        assert!(r.add_image(&g.device, &g.queue, 0, 4, Vec::new()).is_err());
        let a = r.add_image(&g.device, &g.queue, 4, 4, checker()).unwrap();
        let b = r.add_image(&g.device, &g.queue, 4, 4, checker()).unwrap();
        assert_ne!(a, b);
        r.remove_image(a);
        assert!(r.update_image(&g.queue, a, &checker()).is_err());
        assert!(r.update_image(&g.queue, b, &checker()).is_ok());
        assert!(r.update_image(&g.queue, TextureId::NONE, &[255; 4]).is_err());
        let c = r.add_image(&g.device, &g.queue, 4, 4, checker()).unwrap();
        assert_eq!(a, c, "a freed id is handed out again");
        // A removed image's batch is skipped, not drawn with a stale texture.
        r.remove_image(c);
        let mut list = DrawList::new();
        list.begin(16, 16, 1.0);
        list.image(Rect::new(0.0, 0.0, 16.0, 16.0), c, [0.0, 0.0, 1.0, 1.0], Color::WHITE, 0.0);
        list.finish();
        let rgba = render(&mut r, &list, 16, 16, Color::BLACK);
        assert_eq!(r.stats().draw_calls, 0);
        assert_eq!(pixel(&rgba, 16, 8, 8), [0, 0, 0, 255]);
        let bad = AtlasImage { width: 2, height: 2, layers: 1, rgba: vec![0; 3] };
        assert!(r.set_atlas(&g.device, &g.queue, bad).is_err());
    }

    #[test]
    fn unorm_target_gets_exact_srgb_and_srgb_target_records() {
        // ENCODE_SRGB: a UNORM target receives the exact sRGB code of a token; an sRGB target uses
        // the other specialisation and records the same draw.
        let _lock = gpu_lock();
        let g = gpu();
        let mut unorm = renderer(AtlasImage::empty());
        let srgb_format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let mut srgb = UiRenderer::new(&g.device, &g.queue, srgb_format, AtlasImage::empty()).unwrap();
        assert_eq!(srgb.format(), srgb_format);
        let mut list = DrawList::new();
        list.begin(8, 8, 1.0);
        list.rect(Rect::new(0.0, 0.0, 8.0, 8.0), Color::hex(0x7DB3FF), 0.0);
        list.finish();
        let a = render(&mut unorm, &list, 8, 8, Color::BLACK);
        assert_eq!(pixel(&a, 8, 4, 4), [0x7D, 0xB3, 0xFF, 255]);
        srgb.prepare(&g.device, &g.queue, &list, 8, 8).unwrap();
        let target = g.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: srgb_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = g.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        assert_eq!(srgb.record(&mut encoder, &view, Some(Color::BLACK), &list), 1);
        g.queue.submit([encoder.finish()]);
    }

    #[test]
    fn rebuild_restores_images() {
        let _lock = gpu_lock();
        let g = gpu();
        let mut r = renderer(AtlasImage::empty());
        let id = r.add_image(&g.device, &g.queue, 4, 4, checker()).unwrap();
        r.rebuild(&g.device, &g.queue).unwrap();
        let mut list = DrawList::new();
        list.begin(8, 8, 1.0);
        list.image(Rect::new(0.0, 0.0, 8.0, 8.0), id, [0.0, 0.0, 0.125, 0.125], Color::WHITE, 0.0);
        list.finish();
        let rgba = render(&mut r, &list, 8, 8, Color::BLACK);
        assert_eq!(pixel(&rgba, 8, 4, 4), [230, 120, 40, 255]);
    }

    #[test]
    fn gpu_images_are_drawn_from_what_the_gpu_wrote_and_keep_their_id_when_resized() {
        let _lock = gpu_lock();
        let g = gpu();
        let mut r = renderer(AtlasImage::empty());
        assert!(r.image_texture(TextureId::NONE).is_none());
        assert!(r.add_gpu_image(&g.device, 0, 4).is_err());
        assert!(r.add_gpu_image(&g.device, MAX_IMAGE_SIDE + 1, 4).is_err());
        let id = r.add_gpu_image(&g.device, 4, 4).unwrap();
        r.resize_gpu_image(&g.device, id, 8, 8).unwrap();
        assert!(r.resize_gpu_image(&g.device, TextureId::NONE, 8, 8).is_err());
        assert!(r.resize_gpu_image(&g.device, TextureId(99), 8, 8).is_err());
        let texture = r.image_texture(id).unwrap();
        assert_eq!((texture.width(), texture.height()), (8, 8));
        assert_eq!(texture.format(), wgpu::TextureFormat::Rgba8UnormSrgb);
        // A source in the frame's UNORM format holding sRGB-encoded bytes, copied on the GPU.
        let source = g.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let texels: Vec<u8> = [30u8, 140, 220, 255].repeat(64);
        write_image(&g.queue, &source, 8, 8, &texels);
        let mut encoder = g.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_texture(
            source.as_image_copy(),
            texture.as_image_copy(),
            wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 1 },
        );
        g.queue.submit([encoder.finish()]);
        // Drawn 1:1 through the UI pass, under a popup-like rectangle drawn after it.
        let mut list = DrawList::new();
        list.begin(16, 16, 1.0);
        list.image(Rect::new(4.0, 4.0, 8.0, 8.0), id, [0.0, 0.0, 1.0, 1.0], Color::WHITE, 0.0);
        list.rect(Rect::new(9.0, 9.0, 7.0, 7.0), Color::hex(0xFF0000), 0.0);
        list.finish();
        let rgba = render(&mut r, &list, 16, 16, Color::BLACK);
        let got = pixel(&rgba, 16, 6, 6);
        for (g, w) in got.iter().zip([30u8, 140, 220, 255]) {
            assert!(g.abs_diff(w) <= 1, "image texel {got:?}");
        }
        assert_eq!(pixel(&rgba, 16, 10, 10), [255, 0, 0, 255], "the later rectangle covers the image");
        assert_eq!(pixel(&rgba, 16, 1, 1), [0, 0, 0, 255]);
        // Same size: kept. Removed: no texture.
        r.resize_gpu_image(&g.device, id, 8, 8).unwrap();
        r.remove_image(id);
        assert!(r.image_texture(id).is_none());
        // A rebuild keeps GPU images (transparent until written again).
        let again = r.add_gpu_image(&g.device, 2, 2).unwrap();
        r.rebuild(&g.device, &g.queue).unwrap();
        assert!(r.image_texture(again).is_some());
    }
}
