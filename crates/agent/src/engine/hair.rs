//! The character's hair in a view (todo 1.7, `[HAIR-03]` `[HAIR-04]` `[HR-01]`): the groom set
//! grown on the evaluated template (`humanart_model::hair::template`: scalp preset, brows, lashes,
//! peach fuzz, optional body hair), generated on the GPU once ([`GpuGroom`]), attached every frame
//! to the deformation's output buffers so the hair follows shape and pose edits, and drawn by the
//! renderer's hair pass ([`HairRenderer`], `Renderer::set_hair`).
//!
//! [`CharacterHair::new`] does the CPU work (placement, about 0.3 s for 100 000 strands) and the
//! uploads; [`CharacterHair::bind`] binds the deformer's positions and normals once;
//! [`CharacterHair::encode`] records the generation on the first frame and the attach step on every
//! frame (no allocation, `[PERF-R1]`); [`CharacterHair::update_bounds`] moves the grooms' bounding
//! spheres when the evaluated surface changed.

use humanart_gpu::Gpu;
use humanart_gpu::resources::Resources;
use humanart_math::Vec3;
use humanart_model::hair::{
    CharacterGroom, GroomKind, GroomSetSpec, HairError as GroomError, ScalpStyle, TemplateHair, character_grooms,
};
use humanart_render::deform::GpuDeformer;
use humanart_render::error::RenderError;
use humanart_render::groom::{GpuGroom, GroomCapacity, GroomGpuError};
use humanart_render::hair::bsdf::FibreParams;
use humanart_render::hair::{HairError, HairGroom, HairRenderer, HairSettings};

use super::error::EngineError;
use super::model::Model;

/// A natural hair colour (melanin concentrations on the pbrt scale, `humanart_model::hair::color`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairShade {
    /// Black (eumelanin 8).
    Black,
    /// Dark brown (eumelanin 2.5, a little pheomelanin).
    DarkBrown,
    /// Medium brown (eumelanin 1.3).
    Brown,
    /// Blonde (eumelanin 0.3, pheomelanin 0.15).
    Blonde,
    /// Auburn (eumelanin 0.8, pheomelanin 1.2).
    Auburn,
}

impl HairShade {
    /// Every shade, in menu order.
    pub const ALL: [Self; 5] = [Self::Black, Self::DarkBrown, Self::Brown, Self::Blonde, Self::Auburn];

    /// The stable id.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Black => "black",
            Self::DarkBrown => "dark-brown",
            Self::Brown => "brown",
            Self::Blonde => "blonde",
            Self::Auburn => "auburn",
        }
    }

    /// The shade with id `name`.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|s| s.name() == name)
    }

    /// Eumelanin and pheomelanin.
    pub const fn melanin(self) -> (f32, f32) {
        match self {
            Self::Black => (8.0, 0.0),
            Self::DarkBrown => (2.5, 0.1),
            Self::Brown => (1.3, 0.2),
            Self::Blonde => (0.3, 0.15),
            Self::Auburn => (0.8, 1.2),
        }
    }
}

/// Which hair a view draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HairOptions {
    /// The scalp hair style.
    pub style: ScalpStyle,
    /// The colour of the scalp, brow, lash and body hair.
    pub shade: HairShade,
    /// Scalp strands (at most `MAX_SCALP_STRANDS`).
    pub strands: u32,
    /// Eyebrows.
    pub brows: bool,
    /// Upper and lower lashes.
    pub lashes: bool,
    /// Vellus hair on the face.
    pub peach_fuzz: bool,
    /// Sparse hair on the arms and legs.
    pub body_hair: bool,
    /// The groom seed.
    pub seed: u32,
}

impl Default for HairOptions {
    fn default() -> Self {
        Self {
            style: ScalpStyle::Short,
            shade: HairShade::DarkBrown,
            strands: 100_000,
            brows: true,
            lashes: true,
            // Off by default: at portrait distance the strands still read as specks (see the STATUS).
            peach_fuzz: false,
            body_hair: false,
            seed: 1,
        }
    }
}

impl HairOptions {
    /// The groom set these options ask for.
    pub fn spec(&self) -> GroomSetSpec {
        let (eumelanin, pheomelanin) = self.shade.melanin();
        GroomSetSpec {
            scalp: self.style,
            scalp_strands: self.strands,
            brows: self.brows,
            lashes: self.lashes,
            peach_fuzz: self.peach_fuzz,
            body_hair: self.body_hair,
            eumelanin,
            pheomelanin,
            seed: u64::from(self.seed),
            ..GroomSetSpec::default()
        }
    }
}

/// One groom on the GPU and what the renderer needs to know of it.
struct Groom {
    gpu: GpuGroom,
    kind: GroomKind,
    length: f32,
}

/// The hair of one view (see the module documentation).
pub struct CharacterHair {
    grooms: Vec<Groom>,
    generated: bool,
}

impl std::fmt::Debug for CharacterHair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CharacterHair")
            .field("grooms", &self.grooms.iter().map(|g| (g.kind, g.gpu.strand_count())).collect::<Vec<_>>())
            .field("generated", &self.generated)
            .finish()
    }
}

fn groom_error(e: GroomError) -> EngineError {
    gpu_groom_error(GroomGpuError::Hair(e))
}

fn gpu_groom_error(e: GroomGpuError) -> EngineError {
    hair_error(HairError::Groom(e))
}

fn hair_error(e: HairError) -> EngineError {
    EngineError::Render(RenderError::Hair(e))
}

impl CharacterHair {
    /// Grows the groom set of `options` on `model`'s evaluated surface, uploads every groom and
    /// builds the hair renderer that draws them (hand it to `Renderer::set_hair`). With no groom
    /// to draw (bald, every feature off) the result holds no groom.
    ///
    /// # Errors
    /// [`EngineError::Render`] when the template has no hair areas, a groom cannot be grown or the
    /// GPU objects cannot be made.
    pub fn new(
        gpu: &Gpu,
        resources: &mut Resources,
        model: &Model,
        options: &HairOptions,
    ) -> Result<(Self, HairRenderer), EngineError> {
        let template = model.template();
        let hair = TemplateHair::new(template, model.positions(), model.normals()).map_err(groom_error)?;
        let set = character_grooms(&hair, &options.spec()).map_err(groom_error)?;
        let mut grooms: Vec<Groom> = Vec::with_capacity(set.len());
        let built = (|| -> Result<(), EngineError> {
            for g in &set {
                grooms.push(upload(gpu, resources, &hair, g)?);
            }
            Ok(())
        })();
        let renderer = built.and_then(|()| {
            let mut renderer = HairRenderer::new(gpu, resources, HairSettings::default(), &FibreParams::default())
                .map_err(hair_error)?;
            let list: Vec<HairGroom> = grooms
                .iter()
                .zip(&set)
                .map(|(g, c)| {
                    let (centre, radius) = bounds_for(model, g.kind, g.length);
                    let drawn = HairGroom::from_gpu(&g.gpu, &c.params, centre, radius, g.length);
                    // Sparse grooms are never thinned (their survivors would show as specks).
                    if g.kind == GroomKind::Scalp { drawn } else { drawn.with_min_keep(1.0) }
                })
                .collect();
            if let Err(e) = renderer.set_grooms(resources, &list) {
                renderer.destroy(resources);
                return Err(hair_error(e));
            }
            Ok(renderer)
        });
        let this = Self { grooms, generated: false };
        match renderer {
            Ok(r) => Ok((this, r)),
            Err(e) => {
                this.destroy(resources);
                Err(e)
            }
        }
    }

    /// Grooms held.
    pub fn groom_count(&self) -> usize {
        self.grooms.len()
    }

    /// Strands of every groom.
    pub fn strand_count(&self) -> u64 {
        self.grooms.iter().map(|g| u64::from(g.gpu.strand_count())).sum()
    }

    /// The kind and strand count of each groom, in renderer order.
    pub fn grooms(&self) -> impl Iterator<Item = (GroomKind, u32)> + '_ {
        self.grooms.iter().map(|g| (g.kind, g.gpu.strand_count()))
    }

    /// Binds the deformation's positions and normals as the skin every groom attaches to. Call once
    /// after the deformer exists (its buffers do not change).
    ///
    /// # Errors
    /// [`EngineError::Render`] when the deformer's vertex count differs from the grooms'.
    pub fn bind(&mut self, device: &wgpu::Device, deformer: &GpuDeformer) -> Result<(), EngineError> {
        let out = deformer.outputs();
        for g in &mut self.grooms {
            g.gpu.bind_skin(device, out.positions, out.normals).map_err(gpu_groom_error)?;
        }
        Ok(())
    }

    /// Records the generation (first call only) and the attach step of every groom. Record it
    /// after the deformation and before the renderer's passes.
    pub fn encode(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if !self.generated {
            for g in &self.grooms {
                g.gpu.encode_generate(encoder);
            }
            self.generated = true;
        }
        for g in &self.grooms {
            g.gpu.encode_attach(encoder);
        }
    }

    /// Moves every groom's bounding sphere to `model`'s evaluated surface (call when its revision
    /// moved).
    ///
    /// # Errors
    /// [`EngineError::Render`] when `renderer` holds other grooms.
    pub fn update_bounds(&self, model: &Model, renderer: &mut HairRenderer) -> Result<(), EngineError> {
        for (i, g) in self.grooms.iter().enumerate() {
            let (centre, radius) = bounds_for(model, g.kind, g.length);
            renderer.set_bounds(i, centre, radius).map_err(hair_error)?;
        }
        Ok(())
    }

    /// Frees the grooms' GPU objects.
    pub fn destroy(self, resources: &mut Resources) {
        for g in self.grooms {
            g.gpu.destroy(resources);
        }
    }
}

/// Builds the hair of `options` for a viewer and hands its renderer to `renderer`. Returns the
/// hair, or the reason there is none (the viewer then draws the character without hair, as it
/// does without the skin surface).
pub fn build_for_view(
    gpu: &Gpu,
    resources: &mut Resources,
    model: &Model,
    options: Option<&HairOptions>,
    renderer: &mut humanart_render::renderer::Renderer,
    deformer: &GpuDeformer,
) -> (Option<CharacterHair>, Option<EngineError>) {
    let Some(options) = options else { return (None, None) };
    let built = CharacterHair::new(gpu, resources, model, options).and_then(|(mut hair, drawn)| {
        if let Err(e) = hair.bind(&gpu.device, deformer) {
            drawn.destroy(resources);
            hair.destroy(resources);
            return Err(e);
        }
        match renderer.set_hair(resources, Some(drawn)) {
            Ok(_) => Ok(hair),
            Err(e) => {
                hair.destroy(resources);
                Err(EngineError::Render(e))
            }
        }
    });
    match built {
        Ok(hair) => (Some(hair), None),
        Err(e) => (None, Some(e)),
    }
}

/// Moves the bounds of `hair`'s grooms in `renderer`'s hair pass to `model` (call when the
/// evaluated surface changed).
///
/// # Errors
/// As [`CharacterHair::update_bounds`].
pub fn sync_bounds(
    hair: Option<&CharacterHair>,
    model: &Model,
    renderer: &mut humanart_render::renderer::Renderer,
) -> Result<(), EngineError> {
    match (hair, renderer.hair_mut()) {
        (Some(h), Some(r)) => h.update_bounds(model, r),
        _ => Ok(()),
    }
}

/// Uploads one groom: buffers, parameters, guides and layout.
fn upload(gpu: &Gpu, resources: &mut Resources, hair: &TemplateHair, g: &CharacterGroom) -> Result<Groom, EngineError> {
    let points = g.params.point_count();
    let capacity = GroomCapacity {
        strands: u32::try_from(g.layout.len().max(1)).unwrap_or(u32::MAX),
        guides: u32::try_from(g.guides.len().max(1)).unwrap_or(u32::MAX),
        points,
    };
    // Only the scalp hair is long enough to need the collision field.
    let sdf = (g.kind == GroomKind::Scalp).then(|| hair.sdf());
    let mut groom = GpuGroom::new(gpu, resources, hair.surface(), sdf, capacity).map_err(gpu_groom_error)?;
    let filled = (|| -> Result<(), GroomGpuError> {
        groom.set_params(resources, &g.params)?;
        // Blended in world directions: root frames flip across the scalp's UV seam.
        let curves = g.guides.resampled_world(points, hair.surface()).map_err(GroomGpuError::Hair)?;
        groom.set_world_guides(resources, &curves)?;
        groom.set_layout(resources, hair.surface(), &g.layout)
    })();
    match filled {
        Ok(()) => Ok(Groom { gpu: groom, kind: g.kind, length: g.length }),
        Err(e) => {
            groom.destroy(resources);
            Err(gpu_groom_error(e))
        }
    }
}

/// A sphere around a groom on `model`: the head's sphere grown by the hair length for head hair,
/// the whole body's for body hair.
fn bounds_for(model: &Model, kind: GroomKind, length: f32) -> (Vec3, f32) {
    let (centre, radius) = if kind == GroomKind::Body { model.bounds() } else { model.head_bounds() };
    (centre, (radius + length.max(0.0) * 1.1).max(1e-3))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use humanart_gpu::Gpu;
    use humanart_gpu::resources::Budget;

    use super::*;
    use crate::engine::tests::{registry, template};
    use crate::engine::{Framing, Viewer, ViewerOptions};

    /// The review folder of the hair stills.
    fn review_dir() -> std::path::PathBuf {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/review/hair");
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Renders `frames` frames of `framing` and returns the display image (RGBA8).
    fn shoot(
        gpu: &Gpu,
        res: &mut Resources,
        model: &mut Model,
        options: ViewerOptions,
        framing: Framing,
        frames: u32,
        name: &str,
    ) -> (Vec<u8>, u64) {
        let mut viewer = Viewer::with_options(gpu, res, model, 480, 600, options).unwrap();
        assert!(viewer.hair_error().is_none(), "{:?}", viewer.hair_error());
        viewer.frame_model(model, framing);
        for _ in 0..frames {
            let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("t") });
            viewer.record(res, &mut enc, model, &[]).unwrap();
            gpu.queue.submit([enc.finish()]);
            viewer.after_submit();
        }
        let (w, h, rgba) = humanart_render::capture::read_texture(res, viewer.display_texture().unwrap(), 0).unwrap();
        let png = humanart_render::capture::encode_png_rgba8(w, h, &rgba).unwrap();
        std::fs::write(review_dir().join(name), png).unwrap();
        let strands = viewer.hair().map_or(0, CharacterHair::strand_count);
        viewer.destroy(res);
        (rgba, strands)
    }

    /// `[HAIR-04]` `[HR-01]` The viewer grows the default groom set on the template (scalp,
    /// brows, lashes, peach fuzz), attaches it to the deformed skin and draws it: the head still
    /// changes where hair grows, and the picture goes to target/review/hair/viewer_head.png.
    #[test]
    fn hair_viewer_draws_the_groom_set_on_the_template() {
        let gpu = humanart_gpu::shared_headless().expect("this machine has a GPU");
        let mut res = Resources::new(gpu, Budget::UNLIMITED);
        let mut model = Model::new(Arc::new(template().clone()), registry());
        let start = std::time::Instant::now();
        let (bare, none) = shoot(
            gpu,
            &mut res,
            &mut model,
            ViewerOptions { hair: None, ..ViewerOptions::default() },
            Framing::Head,
            8,
            "viewer_head_bare.png",
        );
        assert_eq!(none, 0);
        let bare_time = start.elapsed();
        let start = std::time::Instant::now();
        let (hair, strands) =
            shoot(gpu, &mut res, &mut model, ViewerOptions::default(), Framing::Head, 8, "viewer_head.png");
        println!("viewer with hair: {strands} strands, {:?} (without: {bare_time:?})", start.elapsed());
        assert!(strands > 100_000, "{strands}");
        let changed = bare.as_chunks::<4>().0.iter().zip(hair.as_chunks::<4>().0).filter(|(a, b)| a != b).count();
        println!("pixels changed by the hair: {changed} of {}", bare.len() / 4);
        assert!(changed > 5000, "{changed}");
        // The top of the head (scalp) is darker with dark-brown hair than bare skin.
        let row = |img: &[u8], y: usize| -> f32 {
            (180..300).map(|x| f32::from(img[(y * 480 + x) * 4 + 1])).sum::<f32>() / 120.0
        };
        let (top_bare, top_hair) = (row(&bare, 175), row(&hair, 175));
        println!("green at the crown row: bare {top_bare:.1}, with hair {top_hair:.1}");
        assert!(top_hair < 0.8 * top_bare, "dark hair covers the crown");
        for (style, fuzz, framing, name) in [
            (ScalpStyle::LongWavy, false, Framing::ThreeQuarter, "viewer_long_wavy.png"),
            (ScalpStyle::Coily, false, Framing::ThreeQuarter, "viewer_coily.png"),
            (ScalpStyle::Afro, false, Framing::ThreeQuarter, "viewer_afro.png"),
            (ScalpStyle::MediumStraight, true, Framing::Profile, "viewer_peach_fuzz.png"),
        ] {
            let options = ViewerOptions {
                hair: Some(HairOptions { style, shade: HairShade::Black, peach_fuzz: fuzz, ..HairOptions::default() }),
                ..ViewerOptions::default()
            };
            shoot(gpu, &mut res, &mut model, options, framing, 8, name);
        }
    }

    #[test]
    fn hair_options_map_to_the_groom_set() {
        let o =
            HairOptions { style: ScalpStyle::LongWavy, shade: HairShade::Blonde, seed: 7, ..HairOptions::default() };
        let s = o.spec();
        assert_eq!(s.scalp, ScalpStyle::LongWavy);
        assert_eq!((s.eumelanin, s.pheomelanin), HairShade::Blonde.melanin());
        assert_eq!(s.seed, 7);
        for shade in HairShade::ALL {
            assert_eq!(HairShade::from_name(shade.name()), Some(shade));
        }
        assert_eq!(HairShade::from_name("green"), None);
    }
}
