//! The character's skin in a view `[SKN-01]` `[SKN-11]` `[SKN-20]`: the template's skin part
//! drawn as surfaced skin (`humanart_render::surface`), reading the skin compositor's resolved set
//! (pigment model, zone masks baked from the template by region and landmark names) and the
//! micro-detail exemplar.
//!
//! Everything here is built once per view, when the view is created: the zone masks are baked on
//! the CPU, uploaded, the compositor resolves every tile once and the renderer's skin surface binds
//! the result. A frame reads the bound textures and allocates nothing for the skin `[PERF-R1]`.
//!
//! What is not bound: wrinkle maps (`[SKN-30]`). The template has no authored wrinkle maps or
//! drivers with maps yet, so there is nothing real to bind; the renderer's wrinkle slot keeps its
//! neutral default. The compositor's base values are the registry defaults
//! ([`BaseValues::default`]): the document's `skin.*` values do not reach the compositor yet.

use humanart_gpu::Gpu;
use humanart_model::Template;
use humanart_model::skin_appearance::channels::BaseValues;
use humanart_model::skin_appearance::layers::LayerStack;
use humanart_model::skin_appearance::microdetail::{ExemplarPlan, MicroDetailParams};
use humanart_model::skin_appearance::pigment::Mapping;
use humanart_model::skin_appearance::zone_bake::{BakeReport, TEMPLATE_ZONES, bake_tile, vertex_weights};
use humanart_model::skin_appearance::zones::ZoneTable;
use humanart_render::error::RenderError;
use humanart_render::renderer::Renderer;
use humanart_render::scene::{Material, Mesh};
use humanart_render::skin::{SkinCompositor, SkinDesc, SkinGpuError};
use humanart_render::sss::SKIN_CLASS;
use humanart_render::surface::{MAX_UV_TILES, SurfaceSettings};

use super::error::EngineError;

/// The template part drawn as skin.
pub const SKIN_PART: &str = "body";
/// Texels per side of one compositor tile (eight template tiles: 8 x 512² texels per layer).
pub const SKIN_TILE_SIZE: u32 = 512;
/// Content atlas side of the compositor (paint layers; none yet).
pub const SKIN_ATLAS_SIZE: u32 = 256;
/// Texels the zone masks are dilated past the UV islands.
pub const ZONE_DILATE: u32 = 8;

/// The material of surfaced skin: the albedo comes from the compositor, so `base_color` only shows
/// where the surface has no compositor bound; roughness and `f0` are the skin specular defaults of
/// the surface tests.
pub const SURFACED_SKIN_MATERIAL: Material = Material {
    base_color: [0.42, 0.28, 0.22],
    roughness: 0.45,
    metallic: 0.0,
    f0: 0.028,
    thickness: 0.5,
    class: SKIN_CLASS,
};

/// The render meshes of the character in the template's own vertex order (so one deformer feeds
/// both): the skin part with its UV tile in the integer part of u ([`Mesh::from_template`]), and
/// the other parts (eyes, teeth, gums, tongue, ...) as one mesh, `None` when there are none.
///
/// # Errors
/// [`EngineError::Render`] when the template has no [`SKIN_PART`] or exceeds the render limits.
pub fn character_meshes(template: &Template) -> Result<(Mesh, Option<Mesh>), EngineError> {
    let tm = template.mesh();
    let skin = Mesh::from_template(tm, Some(SKIN_PART))?;
    let mut rest: Option<Mesh> = None;
    for part in tm.parts().iter().filter(|p| p.name != SKIN_PART) {
        let piece = Mesh::from_template(tm, Some(&part.name))?;
        match rest.as_mut() {
            Some(r) => r.indices.extend_from_slice(&piece.indices),
            None => rest = Some(piece),
        }
    }
    Ok((skin, rest.filter(|r| !r.indices.is_empty())))
}

/// The skin data a view binds (see the module documentation). Keep it alive as long as the
/// renderer it was bound to draws: the renderer reads the compositor's textures.
pub struct SkinLook {
    compositor: SkinCompositor,
    plan: ExemplarPlan,
    settings: SurfaceSettings,
    report: BakeReport,
}

impl std::fmt::Debug for SkinLook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkinLook").field("report", &self.report).finish_non_exhaustive()
    }
}

fn skin_gpu(e: SkinGpuError) -> EngineError {
    EngineError::Render(RenderError::Skin(e))
}

impl SkinLook {
    /// Bakes the template's zone masks, uploads them with `base` to a new compositor over every UV
    /// tile at `tile_size`², resolves it once (submitted on `gpu`'s queue) and plans the default
    /// micro-detail exemplar.
    ///
    /// # Errors
    /// [`EngineError::Invalid`] for a template without UV tiles or with more than the surface
    /// holds; [`EngineError::Skin`] when the zones or the exemplar cannot be built;
    /// [`EngineError::Render`] when the compositor fails.
    pub fn new(gpu: &Gpu, template: &Template, base: &BaseValues, tile_size: u32) -> Result<Self, EngineError> {
        let tiles = template.tiles();
        if tiles.is_empty() || tiles.len() > MAX_UV_TILES {
            return Err(EngineError::Invalid { what: "the template's UV tile count does not fit the skin surface" });
        }
        let count = u32::try_from(tiles.len())
            .map_err(|_| EngineError::Invalid { what: "the template's UV tile count does not fit the skin surface" })?;
        let desc = SkinDesc { tiles: count, tile_size, atlas_size: SKIN_ATLAS_SIZE };
        let mut compositor = SkinCompositor::new(gpu, desc).map_err(skin_gpu)?;
        compositor.set_globals(gpu, base, &ZoneTable::default(), &Mapping::DEFAULT);
        let mesh = template.mesh();
        let triangles = mesh.triangulate();
        let (weights, report) = vertex_weights(template, &TEMPLATE_ZONES)?;
        // No baked thickness (byte 255) and full cavity occlusion term (byte 255 = 1).
        let texels = usize::try_from(u64::from(tile_size) * u64::from(tile_size))
            .map_err(|_| EngineError::Invalid { what: "skin tile size" })?;
        let baked: Vec<u8> = [255, 255, 0, 0].repeat(texels);
        for tile in 0..count {
            let index = u16::try_from(tile).map_err(|_| EngineError::Invalid { what: "skin tile index" })?;
            let masks = bake_tile(mesh, &triangles, &weights, index, tile_size, ZONE_DILATE)?;
            compositor.upload_inputs(gpu, tile, &baked, masks.layers()).map_err(skin_gpu)?;
        }
        let mut stack = LayerStack::new(count, tile_size)?;
        stack.dirty_mut().mark_all();
        compositor.sync_layers(gpu, &stack).map_err(skin_gpu)?;
        let mut encoder =
            gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("skin look") });
        compositor.resolve(gpu, &mut encoder, &mut stack, None).map_err(skin_gpu)?;
        gpu.queue.submit([encoder.finish()]);
        let plan = ExemplarPlan::new(&MicroDetailParams::default())?;
        Ok(Self { compositor, plan, settings: SurfaceSettings::for_tiles(tiles), report })
    }

    /// Binds the resolved set, the micro-detail and the per-tile scale to `renderer`'s skin
    /// surface. Objects with `SkinShading::surface` set then draw with it.
    ///
    /// # Errors
    /// [`EngineError::Render`] when the surface refuses the data.
    pub fn bind(&self, gpu: &Gpu, renderer: &mut Renderer) -> Result<(), EngineError> {
        let surface = renderer.skin_surface_mut();
        surface.set_settings(gpu, self.settings)?;
        surface.bind_compositor(gpu, &self.compositor)?;
        surface.set_micro_detail(gpu, &self.plan)?;
        Ok(())
    }

    /// The names the zone bake did not find on the template (empty when complete).
    pub fn report(&self) -> &BakeReport {
        &self.report
    }
}
