//! Offscreen renders for the agent (`render_view`) `[AG-01]`, `[AC-WP23-3]`.
//!
//! The [`RenderService`] draws through the editor's own [`Viewer`] (same deformation, passes,
//! rig, quality tiers and skin surface as a viewport), on a headless device it creates on the
//! first render. One viewer is kept and resized; every render points its camera with
//! [`Viewer::set_camera`] or [`Viewer::frame_model`], which restart the temporal history, and
//! accumulates `frames` frames before reading the display image back. So a render of a document
//! state is the image a viewport with the same camera, rig, tier and view mode settles on.

use humanart_gpu::resources::{Budget, Resources};
use humanart_gpu::{DeviceRequest, Gpu};
use humanart_math::Vec3;
use humanart_render::camera::{Camera, ReviewCamera};
use humanart_render::capture::read_texture;
use humanart_render::light::BuiltinRig;
use humanart_render::renderer::{DebugView, Quality};

use crate::engine::view::frame_camera;
use crate::engine::{EngineError, Framing, Model, Viewer, ViewerOptions};

/// Limits of one render request `[ROB-02]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderLimits {
    /// Largest image side.
    pub max_side: u32,
    /// Most views in one batch.
    pub max_views: usize,
    /// Most frames accumulated per view.
    pub max_frames: u32,
    /// Renders remembered (as resources and for region statistics).
    pub keep: usize,
}

impl RenderLimits {
    /// 4096 px sides, 24 views, 256 frames, 32 renders kept.
    pub const DEFAULT: Self = Self { max_side: 4096, max_views: 24, max_frames: 256, keep: 32 };
}

impl Default for RenderLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// The camera presets `render_view` takes by name: the editor's four framings and the eight
/// standard review cameras.
pub const CAMERA_PRESETS: [&str; 12] = [
    "body",
    "head",
    "three_quarter",
    "profile",
    "front",
    "three_quarter_left",
    "three_quarter_right",
    "profile_left",
    "profile_right",
    "hair_back",
    "macro",
    "full_body",
];

/// The light rigs by name, in [`BuiltinRig::ALL`] order.
pub const RIG_NAMES: [&str; 7] =
    ["neutral_studio", "flat_beauty", "harsh_sidelight", "backlight", "overcast_outdoor", "golden_hour", "reference"];

/// The rig named `name`.
pub fn rig_by_name(name: &str) -> Option<BuiltinRig> {
    RIG_NAMES.iter().position(|n| *n == name).and_then(|i| BuiltinRig::ALL.get(i).copied())
}

/// The name of `rig`.
pub fn rig_name(rig: BuiltinRig) -> &'static str {
    BuiltinRig::ALL.iter().position(|r| *r == rig).and_then(|i| RIG_NAMES.get(i).copied()).unwrap_or("neutral_studio")
}

/// The quality tiers by name.
pub const QUALITY_NAMES: [&str; 3] = ["interactive", "review", "still"];

/// The tier named `name`.
pub fn quality_by_name(name: &str) -> Option<Quality> {
    [Quality::Interactive, Quality::Review, Quality::Still].get(QUALITY_NAMES.iter().position(|n| *n == name)?).copied()
}

/// The name of a tier.
pub fn quality_name(q: Quality) -> &'static str {
    match q {
        Quality::Interactive => "interactive",
        Quality::Review => "review",
        Quality::Still => "still",
    }
}

/// The view mode named `name` ([`DebugView::name`]).
pub fn view_by_name(name: &str) -> Option<DebugView> {
    DebugView::ALL.iter().find(|v| v.name() == name).copied()
}

/// Where the camera of a view comes from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CameraSpec {
    /// One of the editor's framings.
    Framing(Framing),
    /// A standard review camera around the head (the full-body one around the body).
    Review(ReviewCamera),
    /// An explicit camera.
    Explicit(Camera),
}

impl CameraSpec {
    /// The preset named `name` (see [`CAMERA_PRESETS`]).
    pub fn preset(name: &str) -> Option<Self> {
        Some(match name {
            "body" => Self::Framing(Framing::Body),
            "head" => Self::Framing(Framing::Head),
            "three_quarter" => Self::Framing(Framing::ThreeQuarter),
            "profile" => Self::Framing(Framing::Profile),
            other => Self::Review(*ReviewCamera::ALL.iter().find(|r| r.name() == other)?),
        })
    }
}

/// One view to render.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewRequest {
    /// The camera.
    pub camera: CameraSpec,
    /// A turntable rotation of the camera around its target, degrees.
    pub turn_degrees: f32,
    /// The light rig.
    pub rig: BuiltinRig,
    /// The quality tier.
    pub quality: Quality,
    /// The view mode.
    pub view: DebugView,
    /// Draw the wireframe.
    pub wireframe: bool,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Frames accumulated before the capture.
    pub frames: u32,
}

impl Default for ViewRequest {
    fn default() -> Self {
        Self {
            camera: CameraSpec::Framing(Framing::Head),
            turn_degrees: 0.0,
            rig: BuiltinRig::NeutralStudio,
            quality: Quality::Review,
            view: DebugView::Beauty,
            wireframe: false,
            width: 768,
            height: 768,
            frames: 16,
        }
    }
}

/// A rendered image (8-bit RGBA, sRGB-encoded, rows top first).
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// The pixels.
    pub rgba: Vec<u8>,
    /// The camera it was taken with.
    pub camera: Camera,
}

impl Image {
    /// The `[x0, y0, x1, y1)` rectangle (clamped to the image) as a new image.
    pub fn crop(&self, rect: [u32; 4]) -> Self {
        let x0 = rect[0].min(self.width);
        let y0 = rect[1].min(self.height);
        let x1 = rect[2].clamp(x0, self.width);
        let y1 = rect[3].clamp(y0, self.height);
        let (w, h) = (x1 - x0, y1 - y0);
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in y0..y1 {
            let start = ((y * self.width + x0) * 4) as usize;
            let end = start + (w * 4) as usize;
            rgba.extend_from_slice(self.rgba.get(start..end).unwrap_or(&[]));
        }
        Self { width: w, height: h, rgba, camera: self.camera }
    }

    /// The pixel `(x, y)` (RGB), when inside.
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 3]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let i = ((y * self.width + x) * 4) as usize;
        Some([*self.rgba.get(i)?, *self.rgba.get(i + 1)?, *self.rgba.get(i + 2)?])
    }
}

/// Projects a world point into the pixel grid of `camera` at `width` x `height` (the inverse of
/// [`crate::engine::view::pixel_ray`]); `None` behind the camera.
pub fn project(camera: &Camera, p: Vec3, width: u32, height: u32) -> Option<[f32; 2]> {
    let (w, h) = (width.max(1) as f32, height.max(1) as f32);
    let forward = camera.forward();
    let right = forward.cross(camera.up).normalize_or(Vec3::X);
    let up = right.cross(forward);
    let t = (0.5 * camera.fov_y()).tan();
    let d = p - camera.position;
    let z = d.dot(forward);
    if z <= camera.near {
        return None;
    }
    let nx = d.dot(right) / (z * t * (w / h));
    let ny = d.dot(up) / (z * t);
    Some([f32::midpoint(nx, 1.0) * w, f32::midpoint(1.0, -ny) * h])
}

/// Renders views of the character offscreen (see the module documentation).
pub struct RenderService {
    gpu: Option<Gpu>,
    resources: Option<Resources>,
    viewer: Option<Viewer>,
    options: ViewerOptions,
    /// Whether the hair follows the model as the editor's viewport does (`Viewer::new`: the hair
    /// of a starter character of the document's sex) instead of `options.hair`.
    model_hair: bool,
    adapter: String,
}

impl std::fmt::Debug for RenderService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderService")
            .field("adapter", &self.adapter)
            .field("ready", &self.gpu.is_some())
            .finish_non_exhaustive()
    }
}

impl Default for RenderService {
    /// The editor viewport's look: the default options with the hair `Viewer::new` gives the
    /// model (`[AC-WP23-3]`: a headless render matches the viewport).
    fn default() -> Self {
        let mut service = Self::new(ViewerOptions::default());
        service.model_hair = true;
        service
    }
}

impl RenderService {
    /// A service that creates its headless device on the first render.
    pub fn new(options: ViewerOptions) -> Self {
        Self { gpu: None, resources: None, viewer: None, options, model_hair: false, adapter: String::new() }
    }

    /// The adapter's name (empty before the first render).
    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    /// Renders `request` of `model`.
    ///
    /// # Errors
    /// [`EngineError`] when there is no GPU, or the renderer fails.
    pub fn render(&mut self, model: &mut Model, request: &ViewRequest) -> Result<Image, EngineError> {
        if self.gpu.is_none() {
            let gpu = Gpu::headless_blocking(&DeviceRequest::default())?;
            self.adapter.clone_from(&gpu.adapter_info().name);
            self.resources = Some(Resources::new(&gpu, Budget::UNLIMITED));
            self.gpu = Some(gpu);
        }
        let (Some(gpu), Some(resources)) = (self.gpu.as_ref(), self.resources.as_mut()) else {
            return Err(EngineError::Invalid { what: "no GPU device" });
        };
        let (w, h) = (request.width.max(1), request.height.max(1));
        let viewer = match &mut self.viewer {
            Some(v) => {
                if v.size() != (w, h) {
                    v.resize(resources, w, h)?;
                }
                v
            }
            slot @ None => {
                let mut options = self.options;
                if self.model_hair {
                    options.hair = Some(crate::engine::hair::HairOptions::for_model(model));
                }
                slot.insert(Viewer::with_options(gpu, resources, model, w, h, options)?)
            }
        };
        {
            let s = viewer.settings_mut();
            s.quality = request.quality;
            s.rig = request.rig;
            s.debug = request.view;
            s.wireframe = request.wireframe;
        }
        let aspect = w as f32 / h as f32;
        let mut camera = match request.camera {
            CameraSpec::Framing(f) => frame_camera(model, f, aspect),
            CameraSpec::Review(ReviewCamera::FullBody) => frame_camera(model, Framing::Body, aspect),
            CameraSpec::Review(kind) => {
                let (center, radius) = model.head_bounds();
                Camera::review(kind, center, radius, aspect)
            }
            CameraSpec::Explicit(c) => c,
        };
        if request.turn_degrees != 0.0 {
            camera.turntable(request.turn_degrees.to_radians());
        }
        viewer.set_camera(camera);
        // The document's skin.* values (todo 1.4).
        viewer.sync_skin(gpu, model)?;
        for _ in 0..request.frames.max(1) {
            let mut encoder =
                gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("agent render") });
            viewer.record(resources, &mut encoder, model, &[])?;
            gpu.queue.submit([encoder.finish()]);
            viewer.after_submit();
        }
        let texture =
            viewer.display_texture().ok_or(EngineError::Invalid { what: "the viewer has no display image" })?;
        let (width, height, rgba) = read_texture(resources, texture, 0)?;
        Ok(Image { width, height, rgba, camera: viewer.settings().camera })
    }
}

impl Drop for RenderService {
    fn drop(&mut self) {
        if let (Some(v), Some(r)) = (self.viewer.take(), self.resources.as_mut()) {
            v.destroy(r);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_map_to_presets_rigs_tiers_and_views() {
        for name in CAMERA_PRESETS {
            assert!(CameraSpec::preset(name).is_some(), "{name}");
        }
        assert!(CameraSpec::preset("nadir").is_none());
        for (i, name) in RIG_NAMES.iter().enumerate() {
            assert_eq!(rig_by_name(name), Some(BuiltinRig::ALL[i]));
            assert_eq!(rig_name(BuiltinRig::ALL[i]), *name);
        }
        for name in QUALITY_NAMES {
            assert_eq!(quality_name(quality_by_name(name).unwrap()), name);
        }
        assert_eq!(view_by_name("normals"), Some(DebugView::Normals));
    }

    #[test]
    fn projection_inverts_the_pixel_ray() {
        let camera = Camera::default();
        let (w, h) = (320, 200);
        for (x, y) in [(10.5_f32, 20.5_f32), (160.0, 100.0), (300.25, 190.75)] {
            let (o, d) = crate::engine::view::pixel_ray(&camera, x, y, w, h);
            let p = project(&camera, o + d * 2.0, w, h).unwrap();
            assert!((p[0] - x).abs() < 1e-2 && (p[1] - y).abs() < 1e-2, "{p:?} vs {x},{y}");
        }
        assert!(project(&camera, camera.position - camera.forward(), w, h).is_none());
        let img = Image { width: 4, height: 2, rgba: (0..32).collect(), camera };
        let c = img.crop([1, 1, 3, 9]);
        assert_eq!((c.width, c.height), (2, 1));
        assert_eq!(c.rgba, vec![20, 21, 22, 23, 24, 25, 26, 27]);
        assert_eq!(img.pixel(3, 1), Some([28, 29, 30]));
        assert_eq!(img.pixel(4, 0), None);
    }
}
