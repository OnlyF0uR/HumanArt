//! A view of the character: one renderer, the character uploaded once and deformed on the GPU from
//! the [`Model`]'s evaluation input, its own camera, view mode, light rig and quality tier
//! (`[VP-04]`), and region outlines drawn in the overlay pass (`[APP-03]` viewport highlight).
//!
//! The character mesh is the template's own vertex order, handed to a [`GpuDeformer`] so every pass
//! pulls deformed positions and normals from the deformer's buffers `[EVAL-05]`. The template's skin
//! part is drawn as surfaced skin ([`super::skin`]: compositor, zone masks, micro-detail, bound once
//! at creation), the eye and mouth parts as objects of their own shaded analytically
//! ([`super::features`]) and any other part as one more object, all fed by the same deformer;
//! [`ViewerOptions::skin_surface`] off draws the whole character in [`SKIN_MATERIAL`]. A frame only
//! uploads a new evaluation input when the model's revision moved. Overlay primitives are written
//! into a vector reserved at creation; a frame does not allocate unless the configuration, the rig
//! or an outline (computed once per region) changes `[PERF-R1]`.

use std::sync::Arc;

use humanart_gpu::Gpu;
use humanart_gpu::resources::{Resources, TextureId};
use humanart_math::Vec3;
use humanart_render::camera::{Camera, Exposure, Lens, ReviewCamera};
use humanart_render::deform::{DeformSettings, GpuDeformer};
use humanart_render::ibl::EnvironmentSource;
use humanart_render::light::{BackdropChoice, BuiltinRig, LightRig};
use humanart_render::renderer::{
    DebugView, FrameInput, MAX_OVERLAY_PRIMITIVES, OverlayPrimitive, PickHit, Quality, RenderConfig, Renderer,
};
use humanart_render::scene::{GpuScene, Material, Mesh, Scene};
use humanart_render::sss::SkinShading;
use humanart_render::surface::SurfaceSettings;

use super::error::EngineError;
use super::features::{FeatureParts, Features, add_parts};
use super::hair::{CharacterHair, HairOptions};
use super::model::Model;
use super::skin::{SKIN_TILE_SIZE, SURFACED_SKIN_MATERIAL, SkinLook, SkinWrinkles, character_meshes};
use humanart_model::hair::settings::GroomTarget;

/// The material of the character's non-skin parts, and of the whole character when the skin
/// surface is off (a medium skin albedo, linear Rec.709).
pub const SKIN_MATERIAL: Material =
    Material { base_color: [0.42, 0.28, 0.22], roughness: 0.55, metallic: 0.0, f0: 0.028, thickness: 0.5, class: 0 };

/// What one viewport shows `[VP-04]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewSettings {
    /// The camera.
    pub camera: Camera,
    /// The view mode (beauty or a debug view).
    pub debug: DebugView,
    /// The quality tier.
    pub quality: Quality,
    /// The light rig.
    pub rig: BuiltinRig,
    /// Draw the wireframe over the surface.
    pub wireframe: bool,
    /// The backdrop (todo 1.9): the rig's own, studio paper, flat grey or the environment.
    pub backdrop: BackdropChoice,
    /// The lens of the depth of field (todo 1.9); `None`: everything is sharp.
    pub lens: Option<Lens>,
    /// Screen-space ambient occlusion and contact shadows (todo 1.9).
    pub occlusion: bool,
}

impl Default for ViewSettings {
    fn default() -> Self {
        Self {
            camera: Camera::default(),
            debug: DebugView::Beauty,
            quality: Quality::Interactive,
            rig: BuiltinRig::NeutralStudio,
            wireframe: false,
            backdrop: BackdropChoice::Rig,
            lens: None,
            occlusion: true,
        }
    }
}

/// A region drawn outlined in the viewport.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Highlight {
    /// The region id.
    pub region: u16,
    /// Display colour (sRGB-encoded), straight alpha; the alpha fades a flash.
    pub color: [f32; 4],
}

/// Which framing [`Viewer::frame_camera`] computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Framing {
    /// The whole body from the front.
    Body,
    /// The head from the front.
    Head,
    /// The head from 45 degrees to the character's left.
    ThreeQuarter,
    /// The head in profile.
    Profile,
}

/// How a [`Viewer`] draws the character.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewerOptions {
    /// Draw the template's skin part as surfaced skin ([`super::skin`]); off, the whole character
    /// is one object in [`SKIN_MATERIAL`].
    pub skin_surface: bool,
    /// The hair to draw ([`super::hair`]); `None` draws none.
    pub hair: Option<HairOptions>,
}

impl Default for ViewerOptions {
    fn default() -> Self {
        Self { skin_surface: true, hair: Some(HairOptions::default()) }
    }
}

/// One viewport's renderer and GPU character (see the module documentation).
pub struct Viewer {
    renderer: Renderer,
    scene: GpuScene,
    deformer: GpuDeformer,
    meshes: Vec<usize>,
    skin: Option<SkinLook>,
    skin_error: Option<EngineError>,
    wrinkles: Option<SkinWrinkles>,
    wrinkles_error: Option<EngineError>,
    /// The scene object of the surfaced skin.
    skin_object: Option<usize>,
    hair: Option<CharacterHair>,
    hair_error: Option<EngineError>,
    features: Option<Features>,
    features_error: Option<EngineError>,
    /// The hair the viewer was asked to draw.
    hair_options: Option<HairOptions>,
    /// The model's material revision the feature table shows.
    materials_seen: u64,
    /// The model's material and groom revisions the hair shows.
    hair_seen: (u64, u64),
    attached: bool,
    uploaded: u64,
    settings: ViewSettings,
    rig: LightRig,
    rig_kind: Option<(BuiltinRig, BackdropChoice)>,
    overlay: Vec<OverlayPrimitive>,
    reset: bool,
}

impl std::fmt::Debug for Viewer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Viewer")
            .field("settings", &self.settings)
            .field("uploaded", &self.uploaded)
            .field("skin", &self.skin)
            .finish_non_exhaustive()
    }
}

impl Viewer {
    /// A viewer of `model` rendering `width` x `height` pixels, framing the body, with the default
    /// [`ViewerOptions`] (surfaced skin).
    pub fn new(
        gpu: &Gpu,
        resources: &mut Resources,
        model: &Model,
        width: u32,
        height: u32,
    ) -> Result<Self, EngineError> {
        // The hair of a starter character of the document's sex (todo 1.3).
        let options =
            ViewerOptions { hair: Some(super::hair::HairOptions::for_model(model)), ..ViewerOptions::default() };
        Self::with_options(gpu, resources, model, width, height, options)
    }

    /// [`Viewer::new`] with `options`. When the skin surface cannot be built for the template (no
    /// skin part, no UV tiles), the viewer draws the plain character and keeps the reason
    /// ([`Viewer::skin_error`]).
    pub fn with_options(
        gpu: &Gpu,
        resources: &mut Resources,
        model: &Model,
        width: u32,
        height: u32,
        options: ViewerOptions,
    ) -> Result<Self, EngineError> {
        let template = model.template();
        let mut scene = Scene::default();
        let mut meshes = Vec::with_capacity(2);
        let mut skin = None;
        let mut skin_error = None;
        let surfaced = if options.skin_surface {
            let built = character_meshes(template)
                .and_then(|m| SkinLook::new(gpu, template, model.skin(), SKIN_TILE_SIZE).map(|look| (m, look)));
            built.map_err(|e| skin_error = Some(e)).ok()
        } else {
            None
        };
        let mut parts = FeatureParts::default();
        let mut skin_object = None;
        if let Some(((mut skin_mesh, _rest), look)) = surfaced {
            // The eye and mouth parts as their own objects ([`super::features`]).
            parts = add_parts(&mut scene, template, &mut skin_mesh, SKIN_MATERIAL)?;
            meshes.extend_from_slice(parts.meshes());
            let object = scene.add(skin_mesh, SURFACED_SKIN_MATERIAL);
            if let Some(o) = scene.objects.get_mut(object) {
                o.skin = SkinShading { surface: true, ..SkinShading::default() };
                meshes.push(o.mesh);
                skin_object = Some(object);
            }
            skin = Some(look);
        } else {
            let tm = template.mesh();
            let mut mesh = Mesh::from_polygons(tm.positions(), tm.faces())?;
            mesh.uvs = tm.uvs().to_vec();
            mesh.normals = model.normals().to_vec();
            if mesh.normals.len() != mesh.positions.len() {
                mesh.normals = vec![Vec3::Y; mesh.positions.len()];
            }
            let object = scene.add(mesh, SKIN_MATERIAL);
            meshes.extend(scene.objects.get(object).map(|o| o.mesh));
        }
        let config = RenderConfig { overlays: true, quality: Quality::Interactive, ..RenderConfig::new(width, height) };
        let mut renderer = Renderer::new(gpu, resources, config)?;
        if let Some(look) = &mut skin {
            look.bind(gpu, &mut renderer)?;
            // The document's feature layers (todo 2.3): generated before the first frame; a
            // failure (layers beyond the content atlas) leaves the skin without them.
            if let Err(e) = look.sync(gpu, &mut renderer, model) {
                skin_error = Some(e);
            }
        }
        let scene = GpuScene::new(resources, &scene)?;
        let deformer = GpuDeformer::new(
            gpu,
            resources,
            Arc::clone(template),
            DeformSettings { lods: false, ..DeformSettings::default() },
        )?;
        // The expression wrinkles read the deformer's region strain (a failure leaves them off).
        let (wrinkles, wrinkles_error) = match &skin {
            Some(_) => match SkinWrinkles::new(gpu, template, &deformer) {
                Ok(w) => (w, None),
                Err(e) => (None, Some(e)),
            },
            None => (None, None),
        };
        if let Some(w) = &wrinkles {
            w.bind(gpu, &mut renderer);
        }
        let (hair, hair_error) =
            super::hair::build_for_view(gpu, resources, model, options.hair.as_ref(), &mut renderer, &deformer);
        let (mut features, features_error) = match Features::new(resources, model, &parts, &deformer) {
            Ok(f) => (Some(f), None),
            Err(e) => (None, Some(e)),
        };
        // The drawn lashes shade the eyeballs under the upper lid.
        if let Some(f) = &mut features {
            f.set_lashes(if hair.is_some() && options.hair.as_ref().is_some_and(|h| h.lashes) { 1.0 } else { 0.0 });
        }
        let aspect = aspect(width, height);
        let settings = ViewSettings { camera: frame_camera(model, Framing::Body, aspect), ..ViewSettings::default() };
        Ok(Self {
            renderer,
            scene,
            deformer,
            meshes,
            skin,
            skin_error,
            wrinkles,
            skin_object,
            hair,
            hair_error,
            wrinkles_error,
            features,
            features_error,
            hair_options: options.hair,
            materials_seen: 0,
            hair_seen: (0, 0),
            attached: false,
            uploaded: 0,
            settings,
            rig: LightRig::builtin(BuiltinRig::NeutralStudio, Vec3::ZERO, 1.0),
            rig_kind: None,
            overlay: Vec::with_capacity(MAX_OVERLAY_PRIMITIVES),
            reset: true,
        })
    }

    /// Brings the surfaced skin up to date with `model`'s `skin.*` values (todo 1.4): re-resolves
    /// the compositor, regenerates the micro-detail or rewrites the flush when they changed since
    /// the last call ([`SkinLook::sync`]). Call it before [`Viewer::record`] when the document may
    /// have changed; it does nothing (and allocates nothing) otherwise. Returns whether the skin
    /// changed.
    ///
    /// # Errors
    /// The compositor's and the surface's errors.
    pub fn sync_skin(&mut self, gpu: &Gpu, model: &Model) -> Result<bool, EngineError> {
        let Some(look) = self.skin.as_mut() else { return Ok(false) };
        let changed = look.sync(gpu, &mut self.renderer, model)?;
        if changed {
            self.reset = true;
        }
        Ok(changed)
    }

    /// Replaces the surfaced skin's layer settings (for tools and calibration; the defaults come
    /// from [`SkinLook`]). Does nothing without the surfaced skin.
    ///
    /// # Errors
    /// [`EngineError::Render`] for a setting out of its range.
    pub fn set_skin_settings(&mut self, gpu: &Gpu, settings: SurfaceSettings) -> Result<(), EngineError> {
        if self.skin.is_some() {
            self.renderer.skin_surface_mut().set_settings(gpu, settings)?;
            self.reset = true;
        }
        Ok(())
    }

    /// Replaces the surfaced skin object's shading (lobes, profile scale, translucency); does
    /// nothing without the surfaced skin.
    ///
    /// # Errors
    /// [`EngineError::Render`] for a value out of its range.
    pub fn set_skin_shading(&mut self, shading: SkinShading) -> Result<(), EngineError> {
        if let Some(object) = self.skin_object {
            self.scene.set_skin(object, SkinShading { surface: true, ..shading })?;
            self.reset = true;
        }
        Ok(())
    }

    /// The surfaced skin's layer settings.
    pub fn skin_settings(&self) -> &SurfaceSettings {
        self.renderer.skin_surface().settings()
    }

    /// The skin the viewer binds (`None` when the surface is off or could not be built).
    pub fn skin(&self) -> Option<&SkinLook> {
        self.skin.as_ref()
    }

    /// Why the skin surface asked for in [`ViewerOptions`] could not be built, if it could not.
    pub fn skin_error(&self) -> Option<&EngineError> {
        self.skin_error.as_ref()
    }

    /// The expression wrinkles the viewer folds (`None` without the surfaced skin or when the
    /// template has none).
    pub fn wrinkles(&self) -> Option<&SkinWrinkles> {
        self.wrinkles.as_ref()
    }

    /// Brings the hair up to the document (todo 2.3, `hair::sync_view`): the brow, lash, facial
    /// and body hair settings and the hair colour of `model`. Does nothing (and allocates
    /// nothing) while the model's material and groom revisions stay the same; a colour change
    /// recolours the grown grooms, a change of a setting that shapes the strands grows the set
    /// anew (editor-time work). Returns whether the hair changed. Call it before
    /// [`Viewer::record`], like [`Viewer::sync_skin`].
    ///
    /// # Errors
    /// [`EngineError::Render`] when the recolour or the renderer's hair pass fail; a set that
    /// cannot be grown leaves the viewer without hair and the reason in [`Viewer::hair_error`].
    pub fn sync_hair(&mut self, gpu: &Gpu, resources: &mut Resources, model: &Model) -> Result<bool, EngineError> {
        let seen = (model.materials_revision(), model.grooms_revision());
        let Some(options) = self.hair_options else { return Ok(false) };
        if self.hair_seen == seen {
            return Ok(false);
        }
        self.hair_seen = seen;
        let had = self.hair.is_some();
        let (changed, error) = super::hair::sync_view(
            gpu,
            resources,
            model,
            &options,
            &mut self.hair,
            &mut self.renderer,
            &self.deformer,
        )?;
        if changed && (had || self.hair.is_some()) {
            self.hair_error = error;
            let lashes = self.hair.is_some() && options.lashes && model.grooms().get(GroomTarget::LashDensity) > 0.0;
            if let Some(f) = &mut self.features {
                f.set_lashes(if lashes { 1.0 } else { 0.0 });
            }
            self.reset = true;
        }
        Ok(changed)
    }

    /// The hair the viewer draws (`None` when off or when it could not be built).
    pub fn hair(&self) -> Option<&CharacterHair> {
        self.hair.as_ref()
    }

    /// The eyes and mouth the viewer shades ([`super::features`]; `None` without the surfaced
    /// skin or when they could not be set up), to set the lash density or the eye and mouth
    /// parameters.
    pub fn features_mut(&mut self) -> Option<&mut Features> {
        self.features.as_mut()
    }

    /// Why the eyes and mouth could not be set up, if they could not.
    pub fn features_error(&self) -> Option<&EngineError> {
        self.features_error.as_ref()
    }

    /// Why the hair asked for in [`ViewerOptions`] could not be built, if it could not.
    pub fn hair_error(&self) -> Option<&EngineError> {
        self.hair_error.as_ref()
    }

    /// Why the expression wrinkles could not be set up (the viewer then draws without them), if
    /// they could not.
    #[must_use]
    pub fn wrinkles_error(&self) -> Option<&EngineError> {
        self.wrinkles_error.as_ref()
    }

    /// What the viewer shows.
    pub fn settings(&self) -> &ViewSettings {
        &self.settings
    }

    /// What the viewer shows, to change (camera moves, view mode, rig, quality).
    pub fn settings_mut(&mut self) -> &mut ViewSettings {
        &mut self.settings
    }

    /// The render target size.
    pub fn size(&self) -> (u32, u32) {
        let c = self.renderer.config();
        (c.width, c.height)
    }

    /// Resizes the render target (recompiles the graph when the size changed).
    pub fn resize(&mut self, resources: &mut Resources, width: u32, height: u32) -> Result<(), EngineError> {
        let config = RenderConfig { width: width.max(1), height: height.max(1), ..*self.renderer.config() };
        if config != *self.renderer.config() {
            self.renderer.configure(resources, config)?;
            self.reset = true;
        }
        Ok(())
    }

    /// Frames `model` in this viewer's camera.
    pub fn frame_model(&mut self, model: &Model, framing: Framing) {
        let (w, h) = self.size();
        let exposure = self.settings.camera.exposure;
        self.settings.camera = Camera { exposure, ..frame_camera(model, framing, aspect(w, h)) };
        self.reset = true;
    }

    /// Points the camera (position, target, lens) without changing the exposure the light rig set,
    /// and restarts the temporal history so no frame of the previous view leaks in. The agent
    /// server's explicit and turntable cameras come through here.
    pub fn set_camera(&mut self, camera: Camera) {
        let exposure = self.settings.camera.exposure;
        self.settings.camera = Camera { exposure, ..camera };
        self.reset = true;
    }

    /// The world ray through pixel `(x, y)` of the target (origin, unit direction).
    pub fn ray(&self, x: f32, y: f32) -> (Vec3, Vec3) {
        let (w, h) = self.size();
        pixel_ray(&self.settings.camera, x, y, w, h)
    }

    /// Records one frame into `encoder`: the deformation when the model changed, then the passes
    /// with the outlines of `highlights`. Submit the encoder, then call [`Viewer::after_submit`].
    pub fn record(
        &mut self,
        resources: &mut Resources,
        encoder: &mut wgpu::CommandEncoder,
        model: &mut Model,
        highlights: &[Highlight],
    ) -> Result<(), EngineError> {
        self.apply_settings(resources, model)?;
        if !self.attached {
            for &mesh in &self.meshes {
                self.scene.attach_deformer(resources, encoder, mesh, &self.deformer)?;
            }
            self.attached = true;
        }
        if self.uploaded != model.revision() {
            self.deformer.update(resources, &model.input())?;
            self.uploaded = model.revision();
            super::hair::sync_bounds(self.hair.as_ref(), model, &mut self.renderer)?;
        }
        if let Some(features) = &mut self.features {
            // The document's eye and mouth materials (todo 2.3); allocation-free.
            if self.materials_seen != model.materials_revision() {
                features.set_materials(model.materials());
                self.materials_seen = model.materials_revision();
            }
            features.sync(resources, &mut self.renderer, model)?;
        }
        if let Some(w) = &mut self.wrinkles {
            w.update(resources.queue(), model)?;
        }
        if self.deformer.encode(encoder)
            && let Some(features) = &self.features
        {
            features.encode(encoder);
        }
        if let Some(w) = &self.wrinkles {
            w.record(encoder)?;
        }
        if let Some(hair) = &mut self.hair {
            hair.encode(encoder);
        }
        self.overlay.clear();
        for h in highlights {
            outline(&mut self.overlay, model, h);
        }
        let input = FrameInput {
            camera: &self.settings.camera,
            rig: &self.rig,
            scene: &mut self.scene,
            overlay: &self.overlay,
            wireframe: self.settings.wireframe,
            still: None,
            pick: None,
            reset_history: self.reset,
        };
        self.renderer.render(resources, encoder, input)?;
        self.reset = false;
        Ok(())
    }

    /// Tells the renderer the frame was submitted (timings, readbacks).
    pub fn after_submit(&mut self) {
        self.renderer.after_submit();
    }

    /// A finished GPU picking readback (object id and depth), when one arrived.
    pub fn poll_pick(&mut self, device: &wgpu::Device) -> Result<Option<PickHit>, EngineError> {
        Ok(self.renderer.poll_pick(device)?)
    }

    /// The display texture (8-bit RGBA, sRGB-encoded values): what the viewport shows.
    pub fn display_texture(&self) -> Option<TextureId> {
        self.renderer.display_texture()
    }

    /// The renderer (timings, configuration).
    pub fn renderer(&self) -> &Renderer {
        &self.renderer
    }

    /// Frees every GPU object of the viewer.
    pub fn destroy(mut self, resources: &mut Resources) {
        for &mesh in &self.meshes {
            self.scene.detach_deformer(resources, mesh);
        }
        self.scene.destroy(resources);
        self.deformer.destroy(resources);
        if let Some(hair) = self.hair {
            hair.destroy(resources);
        }
        if let Some(features) = self.features {
            features.destroy(resources);
        }
        self.renderer.destroy(resources);
    }

    fn apply_settings(&mut self, resources: &mut Resources, model: &Model) -> Result<(), EngineError> {
        let s = self.settings;
        let current = *self.renderer.config();
        let config = RenderConfig { quality: s.quality, debug: s.debug, occlusion: s.occlusion, ..current };
        if config != current {
            self.renderer.configure(resources, config)?;
            self.reset = true;
        }
        if self.renderer.lens().copied() != s.lens {
            self.renderer.set_lens(resources, s.lens)?;
            self.reset = true;
        }
        if self.rig_kind != Some((s.rig, s.backdrop)) {
            let (center, radius) = model.head_bounds();
            self.rig = LightRig::builtin(s.rig, center, radius).with_backdrop(s.backdrop);
            self.renderer.set_environment(resources, EnvironmentSource::Procedural(self.rig.environment))?;
            self.settings.camera.exposure = Exposure::ev100(self.rig.ev100);
            self.rig_kind = Some((s.rig, s.backdrop));
            self.reset = true;
        }
        Ok(())
    }
}

fn aspect(width: u32, height: u32) -> f32 {
    #[allow(clippy::cast_precision_loss)] // Pixel sizes are far below 2^24.
    let a = width.max(1) as f32 / height.max(1) as f32;
    a
}

/// The camera of `framing` for `model` at aspect ratio `aspect`.
pub fn frame_camera(model: &Model, framing: Framing, aspect: f32) -> Camera {
    match framing {
        Framing::Body => {
            // Fit the extent's box rather than its bounding sphere (the arm span would leave the
            // body small): the radius that fits both half sizes in the narrower field of view.
            let (lo, hi) = model.extent();
            if lo.x > hi.x {
                let (center, radius) = model.bounds();
                return Camera::review(ReviewCamera::FullBody, center, radius, aspect);
            }
            let (half_w, half_h) = (0.5 * (hi.x - lo.x), 0.5 * (hi.y - lo.y));
            let a = aspect.max(1e-3);
            let radius = if a < 1.0 { (half_h * a).max(half_w) } else { half_h.max(half_w / a) };
            Camera::review(ReviewCamera::FullBody, (lo + hi) * 0.5, radius, aspect)
        }
        Framing::Head => {
            let (center, radius) = model.head_bounds();
            Camera::review(ReviewCamera::Front, center, radius, aspect)
        }
        Framing::ThreeQuarter => {
            let (center, radius) = model.head_bounds();
            Camera::review(ReviewCamera::ThreeQuarterLeft, center, radius, aspect)
        }
        Framing::Profile => {
            let (center, radius) = model.head_bounds();
            Camera::review(ReviewCamera::ProfileLeft, center, radius, aspect)
        }
    }
}

/// The world ray of `camera` through pixel `(x, y)` (top-left origin, pixel centres at `.5`) of a
/// `width` x `height` target: origin and unit direction.
pub fn pixel_ray(camera: &Camera, x: f32, y: f32, width: u32, height: u32) -> (Vec3, Vec3) {
    #[allow(clippy::cast_precision_loss)] // Pixel sizes are far below 2^24.
    let (w, h) = (width.max(1) as f32, height.max(1) as f32);
    let forward = camera.forward();
    let right = forward.cross(camera.up).normalize_or(Vec3::X);
    let up = right.cross(forward);
    let t = (0.5 * camera.fov_y()).tan();
    let nx = 2.0 * x / w - 1.0;
    let ny = 1.0 - 2.0 * y / h;
    let dir = (forward + right * (nx * t * (w / h)) + up * (ny * t)).normalize_or(forward);
    (camera.position, dir)
}

/// Appends the outline of `h.region` as on-top overlay lines, up to the overlay limit.
fn outline(out: &mut Vec<OverlayPrimitive>, model: &mut Model, h: &Highlight) {
    let edges = model.region_outline(h.region).len();
    for i in 0..edges {
        if out.len() >= MAX_OVERLAY_PRIMITIVES {
            return;
        }
        let Some(&[a, b]) = model.region_outline(h.region).get(i) else {
            return;
        };
        let p = model.positions();
        if let (Some(&pa), Some(&pb)) = (p.get(a as usize), p.get(b as usize)) {
            out.push(OverlayPrimitive::line(pa, pb, 2.0, h.color, true));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_ray_through_the_centre_is_the_forward_axis() {
        let camera = Camera::default();
        let (o, d) = pixel_ray(&camera, 50.0, 50.0, 100, 100);
        assert_eq!(o, camera.position);
        assert!((d - camera.forward()).length() < 1e-5);
        let (_, left) = pixel_ray(&camera, 0.0, 50.0, 100, 100);
        assert!(left.x < d.x, "the left edge looks to -X for a camera looking down -Z");
        let (_, top) = pixel_ray(&camera, 50.0, 0.0, 100, 100);
        assert!(top.y > d.y, "the top edge looks up");
    }
}
