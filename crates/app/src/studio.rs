//! The GPU side of the editor: the UI renderer, one [`Viewer`] per docked viewport, and the
//! composition of a frame into a target texture.
//!
//! A frame is recorded into one encoder: every viewport renders its own camera, view mode, rig and
//! quality tier (`[VP-04]`) into its viewer's display texture, which is copied on the GPU into the
//! UI image of its slot ([`viewport_image`], an image written on the GPU); then the UI pass draws
//! the panels into the frame target, the viewport images among them as ordinary image quads. So
//! a viewport sits in the UI's z-order: popups, menus, tooltips and modal overlays drawn after it
//! cover it. The same path draws the window and the offscreen shots.

use humanart_agent::engine::{Highlight, Model, ViewSettings, Viewer};
use humanart_core::platform::FileSystem;
use humanart_gpu::Gpu;
use humanart_gpu::resources::{Budget, Resources};
use humanart_ui::Color;
use humanart_ui::paint::TextureId;
use humanart_ui::render::UiRenderer;
use humanart_ui::text::Fonts;
use humanart_ui::theme::FontFace;
use humanart_ui::ui::Ui;

use crate::error::AppError;

/// The format of the frame target: the renderer's display format (8-bit UNORM holding
/// sRGB-encoded values; the UI pass encodes into it).
pub const FRAME_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// The most viewports `[APP-01]`.
pub const MAX_VIEWPORTS: usize = 4;

/// The UI image that shows viewport `slot` (`0..MAX_VIEWPORTS`): the shell paints it into the
/// viewport's panel and [`Studio::record`] fills it with the viewer's frame. [`Studio::new`]
/// registers these ids first, so they are fixed.
pub const fn viewport_image(slot: usize) -> TextureId {
    #[allow(clippy::cast_possible_truncation)] // INVARIANT: slots are below MAX_VIEWPORTS.
    TextureId(slot as u32 + 1)
}

/// Loads the five baked UI fonts (`<dir>/<face>.anfont`).
pub fn load_fonts(fs: &dyn FileSystem, dir: &str) -> Result<Fonts, AppError> {
    let mut files: [Vec<u8>; 5] = Default::default();
    for (slot, face) in files.iter_mut().zip(FontFace::ALL) {
        let path = humanart_core::platform::join_path(dir, &format!("{}.anfont", face.name()));
        *slot = fs.read(&path).map_err(|source| AppError::Platform { path, source })?;
    }
    let [a, b, c, d, e] = &files;
    Ok(Fonts::from_bytes([a, b, c, d, e])?)
}

/// One viewport of a frame: where its image goes and what it shows.
#[derive(Clone, Copy, Debug)]
pub struct ViewportFrame<'h> {
    /// The viewport slot, `0..MAX_VIEWPORTS`.
    pub slot: usize,
    /// The image rectangle in physical pixels of the target: x, y, width, height.
    pub rect: [u32; 4],
    /// What it shows.
    pub settings: ViewSettings,
    /// Regions outlined.
    pub highlights: &'h [Highlight],
}

/// The GPU side (see the module documentation).
pub struct Studio {
    resources: Resources,
    ui: UiRenderer,
    viewers: Vec<Option<Viewer>>,
}

impl std::fmt::Debug for Studio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Studio").field("viewers", &self.viewers.iter().flatten().count()).finish_non_exhaustive()
    }
}

impl Studio {
    /// The GPU side on `gpu` with a memory budget of `budget_mb` MiB.
    pub fn new(gpu: &Gpu, fonts: &Fonts, budget_mb: u32) -> Result<Self, AppError> {
        let hard = u64::from(budget_mb) << 20;
        let budget = Budget { soft: hard / 10 * 8, hard, ..Budget::UNLIMITED };
        let resources = Resources::new(gpu, budget);
        let mut ui = UiRenderer::new(&gpu.device, &gpu.queue, FRAME_FORMAT, fonts.atlas())?;
        for slot in 0..MAX_VIEWPORTS {
            if ui.add_gpu_image(&gpu.device, 1, 1)? != viewport_image(slot) {
                return Err(AppError::Invalid { what: "the viewport images did not get their fixed ids" });
            }
        }
        Ok(Self { resources, ui, viewers: (0..MAX_VIEWPORTS).map(|_| None).collect() })
    }

    /// The GPU resources.
    pub fn resources_mut(&mut self) -> &mut Resources {
        &mut self.resources
    }

    /// The viewer of viewport `slot`, if it was drawn.
    pub fn viewer(&self, slot: usize) -> Option<&Viewer> {
        self.viewers.get(slot).and_then(Option::as_ref)
    }

    /// Frees the viewers of the slots not in `keep` (a viewport was closed).
    pub fn retain(&mut self, keep: &[bool; MAX_VIEWPORTS]) {
        for (slot, viewer) in self.viewers.iter_mut().enumerate() {
            if !keep.get(slot).copied().unwrap_or(false)
                && let Some(v) = viewer.take()
            {
                v.destroy(&mut self.resources);
            }
        }
    }

    /// Records a whole frame into `encoder`: the viewports, their copies into the UI images of
    /// their slots, then the UI of `ui` (after its `end_frame`, with the viewport images painted)
    /// into `target` (format [`FRAME_FORMAT`], usage `RENDER_ATTACHMENT`). Submit the encoder,
    /// then call [`Studio::after_submit`]. No allocation once the viewers and images have their
    /// sizes.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::Texture,
        view: &wgpu::TextureView,
        ui: &Ui,
        clear: Color,
        model: &mut Model,
        viewports: &[ViewportFrame<'_>],
    ) -> Result<(), AppError> {
        let (width, height) = (target.width(), target.height());
        for v in viewports {
            let [_, _, w, h] = v.rect;
            if w == 0 || h == 0 {
                continue;
            }
            let Some(slot) = self.viewers.get_mut(v.slot) else { continue };
            if slot.is_none() {
                *slot = Some(Viewer::new(gpu, &mut self.resources, model, w, h)?);
            }
            if let Some(viewer) = slot.as_mut() {
                viewer.resize(&mut self.resources, w, h)?;
                *viewer.settings_mut() = v.settings;
                // The document's skin.* values reach the compositor (todo 1.4), its groom
                // settings and hair colour the hair (todo 2.3).
                viewer.sync_skin(gpu, model)?;
                viewer.sync_hair(gpu, &mut self.resources, model)?;
                viewer.record(&mut self.resources, encoder, model, v.highlights)?;
            }
            let Some(viewer) = self.viewers.get(v.slot).and_then(Option::as_ref) else { continue };
            let Some(display) = viewer.display_texture().and_then(|t| self.resources.texture(t)) else { continue };
            let display = display.raw();
            let (dw, dh) = (display.width(), display.height());
            let image = viewport_image(v.slot);
            // Same id at the display's size: a no-op unless the viewport was resized.
            self.ui.resize_gpu_image(&gpu.device, image, dw, dh)?;
            let Some(texture) = self.ui.image_texture(image) else { continue };
            // Rgba8Unorm holding sRGB-encoded bytes into Rgba8UnormSrgb: a plain copy, decoded to
            // linear when the UI pass samples it and encoded again into the frame.
            encoder.copy_texture_to_texture(
                display.as_image_copy(),
                texture.as_image_copy(),
                wgpu::Extent3d { width: dw, height: dh, depth_or_array_layers: 1 },
            );
        }
        self.ui.prepare(&gpu.device, &gpu.queue, ui.draw_list(), width, height)?;
        self.ui.record(encoder, view, Some(clear), ui.draw_list());
        Ok(())
    }

    /// Copies the pass timings of the first viewer into `profile` (the profiler panel).
    pub fn fill_profile(&mut self, device: &wgpu::Device, profile: &mut crate::shell::Profile) {
        let _ = device;
        profile.count = 0;
        let Some(v) = self.viewers.iter_mut().flatten().next() else { return };
        let report = v.renderer().timings();
        for (slot, t) in profile.passes.iter_mut().zip(&report.scopes) {
            #[allow(clippy::cast_precision_loss)] // Nanoseconds of one frame.
            let ms = t.gpu_nanos.map(|n| n as f32 / 1e6);
            *slot = (t.name, ms);
            profile.count += 1;
        }
    }

    /// Tells the viewers the frame was submitted.
    pub fn after_submit(&mut self) {
        for v in self.viewers.iter_mut().flatten() {
            v.after_submit();
        }
    }

    /// Frees every GPU object.
    pub fn destroy(mut self) {
        for v in self.viewers.drain(..).flatten() {
            v.destroy(&mut self.resources);
        }
    }
}

/// An offscreen frame target of `width` x `height` (for shots and tests).
pub fn offscreen_target(gpu: &Gpu, width: u32, height: u32) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen frame"),
        size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FRAME_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}
