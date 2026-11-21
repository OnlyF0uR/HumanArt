//! The character's skin in a view `[SKN-01]` `[SKN-11]` `[SKN-20]`: the template's skin part
//! drawn as surfaced skin (`humanart_render::surface`), reading the skin compositor's resolved set
//! (pigment model, zone masks baked from the template by region and landmark names, the measured
//! regional surface of `skin_appearance::regional`, the baked thickness and occlusion) and the
//! micro-detail exemplar.
//!
//! [`SkinLook::new`] builds everything once per view: the zone masks are baked on the CPU, the
//! thickness `[BAKE-01]` and ambient occlusion `[BAKE-02]` on the GPU ([`bake_inputs`], converted
//! from metres / 0..1 floats to the compositor's bytes by `skin_appearance::baked`), uploaded, the
//! compositor resolves every tile once and the renderer's skin surface binds the result. The
//! document's `skin.*` values ([`Model::skin`], todo 1.4) are the compositor's base values;
//! [`SkinLook::sync`] re-resolves when they change (editor-time work, only on a change). A frame
//! reads the bound textures and allocates nothing for the skin `[PERF-R1]`.
//!
//! The expression wrinkles `[SKN-30]` ([`SkinWrinkles`]): the template's procedurally authored
//! maps for the brow, forehead, crow's feet and nasolabial folds, folded every frame by the wrinkle
//! drivers from the evaluated expressions, the deformer's region strain and the document's resting
//! (age) wrinkle depths.

use std::sync::{Arc, Mutex, PoisonError, Weak};

use humanart_gpu::Gpu;
use humanart_model::Template;
use humanart_model::expression::controls::wrinkle_drivers;
use humanart_model::skin_appearance::baked::baked_inputs;
use humanart_model::skin_appearance::dirty::Rect;
use humanart_model::skin_appearance::features::{GeneratorKind, GeneratorParams, Target, generate};
use humanart_model::skin_appearance::layers::{LayerId, LayerStack};
use humanart_model::skin_appearance::microdetail::ExemplarPlan;
use humanart_model::skin_appearance::pigment::Mapping;
use humanart_model::skin_appearance::values::{RestingWrinkle, SkinValues};
use humanart_model::skin_appearance::variation::{bake_tile_values, vertex_variation};
use humanart_model::skin_appearance::wrinkle::{DriverTable, WrinkleMaps};
use humanart_model::skin_appearance::wrinkle_author::{AuthorReport, author_template_wrinkles};
use humanart_model::skin_appearance::zone_bake::{BakeReport, TEMPLATE_ZONES, bake_tile, vertex_weights};
use humanart_model::skin_appearance::zones::{Zone, ZoneTable};
use humanart_render::deform::GpuDeformer;
use humanart_render::error::RenderError;
use humanart_render::renderer::Renderer;
use humanart_render::scene::{Material, Mesh};
use humanart_render::skin::wrinkle::{WrinkleDesc, WrinkleGpu};
use humanart_render::skin::{SkinCompositor, SkinDesc, SkinGpuError};
use humanart_render::sss::SKIN_CLASS;
use humanart_render::surface::{MAX_UV_TILES, SurfaceSettings};
use humanart_tools::bake::gpu::{GpuBaker, GpuMesh, GpuTarget};
use humanart_tools::bake::occlusion::OcclusionSettings;
use humanart_tools::bake::thickness::ThicknessSettings;
use humanart_tools::bake::{BakeMap, BakeMesh, BakeTarget, Coverage};

use super::error::EngineError;
use super::model::Model;

/// The template part drawn as skin.
pub const SKIN_PART: &str = "body";
/// Texels per side of one compositor tile (eight template tiles: 8 x 512² texels per layer).
pub const SKIN_TILE_SIZE: u32 = 512;
/// Content atlas side of the compositor (generated layers: the age spots over the sun-exposed
/// face zones of one 512² tile).
pub const SKIN_ATLAS_SIZE: u32 = 512;
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

/// Texels of padding the baked thickness and occlusion get past the UV islands.
pub const BAKE_PADDING: u32 = 8;

/// The thickness and occlusion bake of the skin (rays per texel: the bakers' defaults, 64).
fn bake_settings() -> (ThicknessSettings, OcclusionSettings) {
    (ThicknessSettings::default(), OcclusionSettings::default())
}

/// One value per texel of channel `channel` of `map`, `empty` where nothing was baked or padded.
fn channel_of(map: &BakeMap, channel: usize, empty: f32, out: &mut Vec<f32>) {
    let c = usize::from(map.channels).max(1);
    out.clear();
    out.extend(map.coverage.iter().enumerate().map(|(i, cov)| match cov {
        Coverage::Empty => empty,
        _ => map.data.get(i * c + channel).copied().unwrap_or(empty),
    }));
}

/// Seed of the low-frequency tone variation.
pub const VARIATION_SEED: u32 = 0x70e5_0001;
/// Melanin added at full tone variation (`ZoneTable::variation_melanin`; artistic, a look value
/// checked on the stills: a slow ±0.02 unevenness on the 0..1 registry scale).
pub const VARIATION_MELANIN: f32 = 0.02;
/// Haemoglobin added at full tone variation (artistic: ±0.06, slow ruddier and paler patches).
pub const VARIATION_HAEMOGLOBIN: f32 = 0.06;

/// Bakes the template's thickness `[BAKE-01]` and ambient occlusion `[BAKE-02]` (rest pose, every
/// part, so the eyeballs shade the eye corners and the teeth the inner lips) on the GPU, rasterises
/// the low-frequency tone variation (`skin_appearance::variation`) and converts them to the
/// compositor's first input layer, one `tile_size²` RGBA8 layer per UV tile
/// (`skin_appearance::baked`). Blocking, editor-time work; [`baked_inputs_cached`] reuses the
/// result for the same template.
///
/// # Errors
/// [`EngineError::Bake`] when the bake fails, [`EngineError::Skin`] when a map does not fit.
pub fn bake_inputs(gpu: &Gpu, template: &Template, tile_size: u32) -> Result<Vec<Vec<u8>>, EngineError> {
    let mesh = template.mesh();
    let bake_mesh = BakeMesh::new(mesh, mesh.positions(), None)?;
    let gpu_mesh = GpuMesh::upload(gpu, &bake_mesh)?;
    let mut baker = GpuBaker::new(gpu)?;
    let (thickness, occlusion) = bake_settings();
    let texels = usize::try_from(u64::from(tile_size) * u64::from(tile_size))
        .map_err(|_| EngineError::Invalid { what: "skin tile size" })?;
    let mut layers = Vec::with_capacity(template.tiles().len());
    let (mut metres, mut open) = (Vec::with_capacity(texels), Vec::with_capacity(texels));
    let triangles = mesh.triangulate();
    let tone = vertex_variation(mesh.positions(), VARIATION_SEED);
    for tile in 0..template.tiles().len() {
        let index = u16::try_from(tile).map_err(|_| EngineError::Invalid { what: "skin tile index" })?;
        let target = GpuTarget::new(gpu, &BakeTarget::new(index, tile_size, tile_size))?;
        let t = baker.bake_thickness_blocking(gpu, &bake_mesh, &gpu_mesh, &target, &thickness, BAKE_PADDING)?;
        let (ao, _bent) =
            baker.bake_occlusion_blocking(gpu, &bake_mesh, &gpu_mesh, &target, &occlusion, BAKE_PADDING)?;
        channel_of(&t, 0, f32::NAN, &mut metres);
        channel_of(&ao, 0, 1.0, &mut open);
        let variation = bake_tile_values(mesh, &triangles, &tone, index, tile_size, BAKE_PADDING)?;
        let mut layer = Vec::new();
        baked_inputs(texels, Some(&metres), Some(&open), Some(&variation), &mut layer)?;
        layers.push(layer);
    }
    Ok(layers)
}

/// The baked input layers of one template at one tile size (see [`baked_inputs_cached`]).
type BakedLayers = Arc<Vec<Vec<u8>>>;

/// Templates whose bakes are kept.
pub const BAKE_CACHE_ENTRIES: usize = 4;

/// The bakes of the last [`BAKE_CACHE_ENTRIES`] templates: the template (weakly, so a dropped
/// template is never matched by a new one at the same address), the tile size and the layers.
static BAKE_CACHE: Mutex<Vec<(Weak<Template>, u32, BakedLayers)>> = Mutex::new(Vec::new());

/// [`bake_inputs`] once per template and tile size in the process: a second viewer of the same
/// template (another viewport, a re-created view) reuses the layers instead of baking again
/// (7.3 s in a debug build). Keyed by the template's identity (the `Arc`), not its content: a new
/// template (another generation, a loaded package) is baked anew; entries of dropped templates are
/// evicted first. Editor-time work, not a per-frame path.
///
/// # Errors
/// [`bake_inputs`]'s errors (nothing is cached then).
pub fn baked_inputs_cached(gpu: &Gpu, template: &Arc<Template>, tile_size: u32) -> Result<BakedLayers, EngineError> {
    let find = |cache: &[(Weak<Template>, u32, BakedLayers)]| {
        cache
            .iter()
            .find(|(t, size, _)| *size == tile_size && t.upgrade().is_some_and(|t| Arc::ptr_eq(&t, template)))
            .map(|(_, _, layers)| Arc::clone(layers))
    };
    if let Some(layers) = find(&BAKE_CACHE.lock().unwrap_or_else(PoisonError::into_inner)) {
        return Ok(layers);
    }
    let layers = Arc::new(bake_inputs(gpu, template, tile_size)?);
    let mut cache = BAKE_CACHE.lock().unwrap_or_else(PoisonError::into_inner);
    cache.retain(|(t, _, _)| t.strong_count() > 0);
    if cache.len() >= BAKE_CACHE_ENTRIES {
        cache.remove(0);
    }
    cache.push((Arc::downgrade(template), tile_size, Arc::clone(&layers)));
    Ok(layers)
}

/// Edge of the wrinkle-map tile in texels (about 0.4 mm per texel on the face tile).
pub const WRINKLE_SIZE: u32 = 1024;

/// The character's expression wrinkles `[SKN-30]` (todo 1.4): the template's authored wrinkle
/// maps (`skin_appearance::wrinkle_author`), its wrinkle drivers
/// (`data/template/expressions/wrinkles.json`) and their GPU blending state. Each frame the
/// expression part of every driver's fold plus the document's resting wrinkle depth
/// ([`SkinValues::resting`], the age lines) is written ([`SkinWrinkles::update`]) and the
/// activation pass reads the deformer's region strain ([`SkinWrinkles::record`]).
pub struct SkinWrinkles {
    wrinkles: WrinkleGpu,
    table: DriverTable,
    /// Per driver: the resting wrinkle it shows.
    resting: Vec<Option<RestingWrinkle>>,
    terms: Vec<f32>,
    report: AuthorReport,
}

impl std::fmt::Debug for SkinWrinkles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkinWrinkles")
            .field("drivers", &self.terms.len())
            .field("report", &self.report)
            .finish_non_exhaustive()
    }
}

impl SkinWrinkles {
    /// Authors and uploads the template's wrinkle maps and drivers and binds `deformer`'s region
    /// strain. `None` when the template has no drivers or no map could be authored.
    ///
    /// # Errors
    /// [`EngineError::Invalid`] when the drivers cannot be read; the skin and GPU errors of the
    /// maps and the wrinkle set.
    pub fn new(gpu: &Gpu, template: &Template, deformer: &GpuDeformer) -> Result<Option<Self>, EngineError> {
        let drivers = wrinkle_drivers(template)
            .map_err(|_| EngineError::Invalid { what: "the template's wrinkle drivers cannot be read" })?;
        if drivers.is_empty() {
            return Ok(None);
        }
        let list = drivers.to_drivers();
        let names: Vec<&str> = drivers.names().iter().map(String::as_str).collect();
        let (maps, report) = author_template_wrinkles(template, &names, WRINKLE_SIZE)?;
        if maps.is_empty() {
            return Ok(None);
        }
        let mut tiles: Vec<u32> = maps.iter().map(|m| m.tile).collect();
        tiles.sort_unstable();
        tiles.dedup();
        let mut texel_mm = Vec::with_capacity(tiles.len());
        for &t in &tiles {
            let tile =
                template.tiles().get(t as usize).ok_or(EngineError::Invalid { what: "a wrinkle map's UV tile" })?;
            texel_mm.push(tile.resolution as f32 / tile.target_texels_per_mm / WRINKLE_SIZE as f32);
        }
        let baked = WrinkleMaps::bake(WRINKLE_SIZE, &tiles, &texel_mm, &maps)?;
        let count = |n: usize| u32::try_from(n).map_err(|_| EngineError::Invalid { what: "wrinkle set size" });
        let desc = WrinkleDesc { size: WRINKLE_SIZE, tiles: count(tiles.len())?, drivers: count(list.len())? };
        let mut wrinkles = WrinkleGpu::new(gpu, desc).map_err(skin_gpu)?;
        let table = DriverTable::new(&list)?;
        wrinkles.set_drivers(gpu, &table).map_err(skin_gpu)?;
        wrinkles.upload_maps(gpu, &baked).map_err(skin_gpu)?;
        let regions = count(template.regions().regions.len())?;
        wrinkles.bind_strain(gpu, deformer.outputs().region_strain, regions).map_err(skin_gpu)?;
        let resting = names.iter().map(|n| RestingWrinkle::of_driver(n)).collect();
        Ok(Some(Self { wrinkles, table, resting, terms: vec![0.0; list.len()], report }))
    }

    /// Writes this frame's fold terms: the expression part of `model`'s evaluated expressions plus
    /// the document's resting depth of each driver's wrinkle. Allocation-free.
    ///
    /// # Errors
    /// The table's and the wrinkle set's errors.
    pub fn update(&mut self, queue: &wgpu::Queue, model: &Model) -> Result<(), EngineError> {
        self.table.expression_terms(model.input().expressions, &mut self.terms)?;
        let resting = &model.skin().resting;
        for (term, r) in self.terms.iter_mut().zip(&self.resting) {
            if let Some(r) = r {
                *term += resting.get(*r as usize).copied().unwrap_or(0.0);
            }
        }
        self.wrinkles.write_terms(queue, &self.terms).map_err(skin_gpu)?;
        Ok(())
    }

    /// Records the activation pass (after the deformer's strain passes, before the frame).
    ///
    /// # Errors
    /// The wrinkle set's errors.
    pub fn record(&self, encoder: &mut wgpu::CommandEncoder) -> Result<(), EngineError> {
        self.wrinkles.record_activate(encoder).map_err(skin_gpu)
    }

    /// Binds the maps and activations to `renderer`'s skin surface.
    pub fn bind(&self, gpu: &Gpu, renderer: &mut Renderer) {
        renderer.skin_surface_mut().set_wrinkles(gpu, &self.wrinkles);
    }

    /// What the authoring did not find.
    pub fn report(&self) -> &AuthorReport {
        &self.report
    }
}

/// The skin data a view binds (see the module documentation). Keep it alive as long as the
/// renderer it was bound to draws: the renderer reads the compositor's textures.
pub struct SkinLook {
    compositor: SkinCompositor,
    stack: LayerStack,
    plan: ExemplarPlan,
    settings: SurfaceSettings,
    report: BakeReport,
    values: SkinValues,
    /// The [`Model::skin_revision`] the look shows (0: built from values, not from a model).
    revision: u64,
    /// Where age spots go: the face tile, its sun-exposed zone weight per texel and its texel
    /// size in mm (`None` without a face tile).
    spot_area: Option<SpotArea>,
    /// The age-spot layer, when there is one.
    spots: Option<LayerId>,
}

/// The sun-exposed part of the face tile, where solar lentigines appear.
struct SpotArea {
    tile: u32,
    /// The rectangle of the tile the sun-exposed zones cover.
    rect: Rect,
    /// Their weight per texel of `rect`, row-major.
    mask: Vec<f32>,
    texel_mm: f32,
}

/// The zones that take age spots (sun-exposed face skin).
const SUN_ZONES: [Zone; 5] = [Zone::Cheeks, Zone::Nose, Zone::Forehead, Zone::Temples, Zone::Chin];
/// Most age spots on the face tile at amount 1 (artistic: the generator's count range is 0..80).
pub const AGE_SPOTS_MAX: f32 = 40.0;
/// Seed of the age-spot placement.
const AGE_SPOT_SEED: u64 = 0x5907_a6e1;

impl std::fmt::Debug for SkinLook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkinLook").field("report", &self.report).field("values", &self.values).finish_non_exhaustive()
    }
}

fn skin_gpu(e: SkinGpuError) -> EngineError {
    EngineError::Render(RenderError::Skin(e))
}

/// The layer settings of the template's surfaced skin: [`SurfaceSettings::for_tiles`] with the
/// document's flush; the lips' roughness comes from the measured zone table
/// (`skin_appearance::regional`), so the shader's extra lip smoothing is off.
fn surface_settings(template: &Template, values: &SkinValues) -> SurfaceSettings {
    SurfaceSettings { flush: values.flush, lip_roughness_scale: 1.0, ..SurfaceSettings::for_tiles(template.tiles()) }
}

impl SkinLook {
    /// Bakes the template's zone masks, thickness, occlusion and tone variation (the last three once
    /// per template, [`baked_inputs_cached`]), uploads them with `values` to a
    /// new compositor over every UV tile at `tile_size`², resolves it once (submitted on `gpu`'s
    /// queue) and plans the micro-detail exemplar of `values`.
    ///
    /// # Errors
    /// [`EngineError::Invalid`] for a template without UV tiles or with more than the surface
    /// holds; [`EngineError::Skin`] when the zones or the exemplar cannot be built;
    /// [`EngineError::Bake`] when the thickness or occlusion bake fails;
    /// [`EngineError::Render`] when the compositor fails.
    pub fn new(gpu: &Gpu, template: &Arc<Template>, values: &SkinValues, tile_size: u32) -> Result<Self, EngineError> {
        let tiles = template.tiles();
        if tiles.is_empty() || tiles.len() > MAX_UV_TILES {
            return Err(EngineError::Invalid { what: "the template's UV tile count does not fit the skin surface" });
        }
        let count = u32::try_from(tiles.len())
            .map_err(|_| EngineError::Invalid { what: "the template's UV tile count does not fit the skin surface" })?;
        let desc = SkinDesc { tiles: count, tile_size, atlas_size: SKIN_ATLAS_SIZE };
        let mut compositor = SkinCompositor::new(gpu, desc).map_err(skin_gpu)?;
        let mesh = template.mesh();
        let triangles = mesh.triangulate();
        let mut spot_area = None;
        let (weights, report) = vertex_weights(template, &TEMPLATE_ZONES)?;
        let baked = baked_inputs_cached(gpu, template, tile_size)?;
        for (tile, layer) in (0..count).zip(baked.iter()) {
            let index = u16::try_from(tile).map_err(|_| EngineError::Invalid { what: "skin tile index" })?;
            let masks = bake_tile(mesh, &triangles, &weights, index, tile_size, ZONE_DILATE)?;
            compositor.upload_inputs(gpu, tile, layer, masks.layers()).map_err(skin_gpu)?;
            if spot_area.is_none()
                && let Some(t) = tiles.get(tile as usize).filter(|t| t.name == "face")
            {
                let sun = |x: u32, y: u32| SUN_ZONES.iter().map(|z| masks.weight(z.slot(), x, y)).fold(0.0, f32::max);
                let (mut lo, mut hi) = ((tile_size, tile_size), (0, 0));
                for y in 0..tile_size {
                    for x in 0..tile_size {
                        if sun(x, y) > 0.0 {
                            lo = (lo.0.min(x), lo.1.min(y));
                            hi = (hi.0.max(x), hi.1.max(y));
                        }
                    }
                }
                if hi.0 >= lo.0 && hi.1 >= lo.1 {
                    let rect = Rect::new(lo.0, lo.1, hi.0 - lo.0 + 1, hi.1 - lo.1 + 1);
                    let mask = (rect.y..rect.bottom())
                        .flat_map(|y| (rect.x..rect.right()).map(move |x| (x, y)))
                        .map(|(x, y)| sun(x, y))
                        .collect();
                    let texel_mm = t.resolution as f32 / t.target_texels_per_mm / tile_size as f32;
                    spot_area = Some(SpotArea { tile, rect, mask, texel_mm });
                }
            }
        }
        let stack = LayerStack::new(count, tile_size)?;
        let plan = ExemplarPlan::new(&values.micro_detail())?;
        let mut look = Self {
            compositor,
            stack,
            plan,
            settings: surface_settings(template, values),
            report,
            values: *values,
            revision: 0,
            spot_area,
            spots: None,
        };
        look.set_age_spots(values.age_spots)?;
        look.compositor.sync_layers(gpu, &look.stack).map_err(skin_gpu)?;
        Self::resolve(gpu, &mut look.compositor, &mut look.stack, values)?;
        Ok(look)
    }

    /// Puts the age-spot layer for `amount` (0..1) into the layer stack, or takes it out at 0:
    /// solar lentigines from the `SunSpots` generator on the sun-exposed face zones, up to
    /// [`AGE_SPOTS_MAX`] spots, larger (6 to 20 mm) and darker with the amount (artistic values). Sync and resolve afterwards.
    fn set_age_spots(&mut self, amount: f32) -> Result<(), EngineError> {
        let Some(area) = &self.spot_area else { return Ok(()) };
        if amount <= 0.0 || !amount.is_finite() {
            if let Some(id) = self.spots.take() {
                self.stack.remove(id);
            }
            return Ok(());
        }
        let mut params = GeneratorParams::new(GeneratorKind::SunSpots);
        params.set("skin.features.sun_spots.count", (AGE_SPOTS_MAX * amount).round());
        params.set("skin.features.sun_spots.amount", 0.1 + 0.25 * amount);
        params.set("skin.features.sun_spots.size", 6.0 + 14.0 * amount);
        let target = Target { tile: area.tile, rect: area.rect, texel_mm: area.texel_mm, mask: Some(&area.mask) };
        let id = self.spots.unwrap_or_else(|| self.stack.allocate_id());
        let layer = generate(&params, &target, id, AGE_SPOT_SEED, None)?;
        if self.spots.is_some() {
            self.stack.replace(layer)?;
        } else {
            self.stack.push(layer)?;
            self.spots = Some(id);
        }
        Ok(())
    }

    /// Writes the globals of `values` and re-resolves every tile (submitted on `gpu`'s queue).
    fn resolve(
        gpu: &Gpu,
        compositor: &mut SkinCompositor,
        stack: &mut LayerStack,
        values: &SkinValues,
    ) -> Result<(), EngineError> {
        let table = ZoneTable {
            variation_melanin: VARIATION_MELANIN,
            variation_haemoglobin: VARIATION_HAEMOGLOBIN,
            ..ZoneTable::default()
        };
        let (base, zones) = values.resolved_globals(&table);
        compositor.set_globals(gpu, &base, &zones, &Mapping::DEFAULT);
        stack.dirty_mut().mark_all();
        let mut encoder =
            gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("skin look") });
        compositor.resolve(gpu, &mut encoder, stack, None).map_err(skin_gpu)?;
        gpu.queue.submit([encoder.finish()]);
        Ok(())
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

    /// Brings the look up to date with `model`'s skin values (todo 1.4): a change of the base
    /// values re-resolves the compositor, a change of the pore and fine-line inputs regenerates
    /// the micro-detail, a change of the flush rewrites the surface settings. Returns whether
    /// anything changed; nothing is done (and nothing allocated) while [`Model::skin_revision`]
    /// stays the same.
    ///
    /// # Errors
    /// The compositor's, the micro-detail generator's and the surface's errors.
    pub fn sync(&mut self, gpu: &Gpu, renderer: &mut Renderer, model: &Model) -> Result<bool, EngineError> {
        if self.revision == model.skin_revision() {
            return Ok(false);
        }
        self.revision = model.skin_revision();
        let next = *model.skin();
        if next == self.values {
            return Ok(false);
        }
        let spots = next.age_spots.to_bits() != self.values.age_spots.to_bits();
        if spots {
            self.set_age_spots(next.age_spots)?;
            self.compositor.sync_layers(gpu, &self.stack).map_err(skin_gpu)?;
        }
        if spots || next.base_differs(&self.values) {
            Self::resolve(gpu, &mut self.compositor, &mut self.stack, &next)?;
        }
        if next.detail_differs(&self.values) {
            self.plan = ExemplarPlan::new(&next.micro_detail())?;
            renderer.skin_surface_mut().set_micro_detail(gpu, &self.plan)?;
        }
        if next.flush.to_bits() != self.values.flush.to_bits() {
            self.settings.flush = next.flush;
            renderer.skin_surface_mut().set_settings(gpu, self.settings)?;
        }
        self.values = next;
        Ok(true)
    }

    /// The generated layers in the compositor's stack (the age spots, when there are any).
    pub fn layer_count(&self) -> usize {
        self.stack.layers().len()
    }

    /// The skin values the look shows.
    pub fn values(&self) -> &SkinValues {
        &self.values
    }

    /// The compositor (its resolved set, for tests and tools).
    pub fn compositor(&self) -> &SkinCompositor {
        &self.compositor
    }

    /// The names the zone bake did not find on the template (empty when complete).
    pub fn report(&self) -> &BakeReport {
        &self.report
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use humanart_doc::document::{Author, AuthorKind, Document};
    use humanart_doc::registry::ParamValue;
    use humanart_gpu::Gpu;
    use humanart_gpu::resources::{Budget, Resources};
    use humanart_model::skin_appearance::channels::Attribute;

    use super::super::features::{CloseUp, close_up_camera};
    use super::super::view::{Framing, Viewer};
    use super::*;

    fn doc(params: &[(&str, f64)]) -> Document {
        let me = Author { id: "human.test".into(), kind: AuthorKind::Human, name: "Test".into() };
        let mut doc = Document::new(humanart_core::id::Id128::from_u128(5), "t", "generated", 1, 0, me);
        for (id, v) in params {
            doc.params.insert((*id).to_owned(), ParamValue::Scalar(*v));
        }
        doc
    }

    /// Mean 8-bit channel values (r, g, b) of the display image's centre half.
    fn centre_mean(res: &mut Resources, viewer: &Viewer) -> [f64; 3] {
        let (w, h, rgba) = humanart_render::capture::read_texture(res, viewer.display_texture().unwrap(), 0).unwrap();
        let mut sum = [0.0; 3];
        let mut n = 0.0;
        for y in h / 4..h * 3 / 4 {
            for x in w / 4..w * 3 / 4 {
                let i = ((y * w + x) * 4) as usize;
                for (s, c) in sum.iter_mut().zip(&rgba[i..i + 3]) {
                    *s += f64::from(*c);
                }
                n += 1.0;
            }
        }
        sum.map(|s| s / n)
    }

    fn frames(gpu: &Gpu, res: &mut Resources, viewer: &mut Viewer, model: &mut Model) {
        for _ in 0..4 {
            let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("t") });
            viewer.record(res, &mut enc, model, &[]).unwrap();
            gpu.queue.submit([enc.finish()]);
            viewer.after_submit();
        }
    }

    /// todo 1.4: the document's `skin.*` values reach the compositor. The model reads them from
    /// the document, the viewer's skin re-resolves when they change, and the composited image
    /// changes: more melanin darkens the skin, more blood reddens it; an unchanged document does
    /// no work. The baked thickness and occlusion are in the compositor's inputs.
    #[test]
    fn todo_1_4_document_skin_values_change_the_composited_image() {
        let reg = super::super::tests::registry();
        let template = super::super::tests::template();
        let mut model = Model::new(Arc::new(template.clone()), reg);
        let defaults = *model.skin();
        assert!((defaults.base.get(Attribute::Melanin) - 0.3).abs() < 1e-6, "registry default");
        let gpu = humanart_gpu::shared_headless().unwrap();
        let mut res = Resources::new(gpu, Budget::UNLIMITED);
        let (w, h) = (96, 96);
        let started = std::time::Instant::now();
        let mut viewer = Viewer::new(gpu, &mut res, &model, w, h).unwrap();
        let first = started.elapsed().as_secs_f64();
        assert!(viewer.skin_error().is_none(), "{:?}", viewer.skin_error());
        // The bake is cached per template: a second viewer of the same template does not bake.
        let started = std::time::Instant::now();
        let second = Viewer::new(gpu, &mut res, &model, w, h).unwrap();
        let again = started.elapsed().as_secs_f64();
        second.destroy(&mut res);
        let a = baked_inputs_cached(gpu, model.template(), SKIN_TILE_SIZE).unwrap();
        let b = baked_inputs_cached(gpu, model.template(), SKIN_TILE_SIZE).unwrap();
        assert!(Arc::ptr_eq(&a, &b), "one bake per template");
        let other = Arc::new(template.clone());
        assert!(!baked_inputs_cached(gpu, &other, 16).is_ok_and(|c| Arc::ptr_eq(&c, &a)), "another key bakes anew");
        println!("[todo 1.4] viewer with baked skin built in {first:.2} s, the second viewer in {again:.2} s");
        assert!(again < first, "the second viewer reuses the bake: {again} s vs {first} s");
        #[allow(clippy::cast_precision_loss)]
        let camera = close_up_camera(&model, CloseUp::Mouth, Framing::Head, w as f32 / h as f32).unwrap();
        viewer.set_camera(camera);
        let shot = |model: &mut Model, viewer: &mut Viewer, res: &mut Resources| {
            frames(gpu, res, viewer, model);
            centre_mean(res, viewer)
        };
        let neutral = shot(&mut model, &mut viewer, &mut res);
        // Same document: nothing to do.
        model.sync(&doc(&[]), reg).unwrap();
        assert!(!viewer.sync_skin(gpu, &model).unwrap(), "unchanged values do not re-resolve");
        model.sync(&doc(&[("skin.pigment.melanin", 0.9)]), reg).unwrap();
        assert!((model.skin().base.get(Attribute::Melanin) - 0.9).abs() < 1e-6);
        assert!(viewer.sync_skin(gpu, &model).unwrap());
        assert!(!viewer.sync_skin(gpu, &model).unwrap(), "applied once");
        let dark = shot(&mut model, &mut viewer, &mut res);
        model.sync(&doc(&[("skin.pigment.haemoglobin", 1.0)]), reg).unwrap();
        assert!(viewer.sync_skin(gpu, &model).unwrap());
        let red = shot(&mut model, &mut viewer, &mut res);
        model.sync(&doc(&[("skin.pores.depth", 1.0), ("skin.flush.amount", 1.0)]), reg).unwrap();
        assert!(viewer.sync_skin(gpu, &model).unwrap(), "pores and flush are skin values too");
        assert_eq!(viewer.skin().unwrap().values().flush, 1.0);
        let luma = |c: [f64; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        // Age spots, seen in a head still (they lie on the sun-exposed face zones).
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/agents/skin-surface/shots");
        std::fs::create_dir_all(&dir).unwrap();
        viewer.resize(&mut res, 480, 600).unwrap();
        viewer.frame_model(&model, Framing::Head);
        let head = |model: &mut Model, viewer: &mut Viewer, res: &mut Resources, name: &str| {
            frames(gpu, res, viewer, model);
            let (tw, th, rgba) =
                humanart_render::capture::read_texture(res, viewer.display_texture().unwrap(), 0).unwrap();
            std::fs::write(dir.join(name), humanart_render::capture::encode_png_rgba8(tw, th, &rgba).unwrap()).unwrap();
            rgba
        };
        let clear = head(&mut model, &mut viewer, &mut res, "test-age-spots-0.png");
        let aged = [("skin.pores.depth", 1.0), ("skin.flush.amount", 1.0), ("skin.age_spots.amount", 1.0)];
        model.sync(&doc(&aged), reg).unwrap();
        assert!(viewer.sync_skin(gpu, &model).unwrap(), "age spots are a skin value");
        assert_eq!(viewer.skin().unwrap().values().age_spots, 1.0);
        assert_eq!(viewer.skin().unwrap().layer_count(), 1, "the age-spot layer is in the stack");
        let spotted = head(&mut model, &mut viewer, &mut res, "test-age-spots-1.png");
        // Pixels clearly darker after (spots) against the frame-to-frame noise of a still scene.
        let darker = clear
            .as_chunks::<4>()
            .0
            .iter()
            .zip(spotted.as_chunks::<4>().0)
            .filter(|(a, b)| i32::from(a[1]) - i32::from(b[1]) > 12)
            .count();
        println!("[todo 1.4] age spots 1: {darker} pixels of the head still clearly darker");
        assert!(darker > 50, "age spots show: {darker} darker pixels");
        model.sync(&doc(&[("skin.pores.depth", 1.0), ("skin.flush.amount", 1.0)]), reg).unwrap();
        assert!(viewer.sync_skin(gpu, &model).unwrap());
        assert_eq!(viewer.skin().unwrap().layer_count(), 0, "amount 0 removes the layer");
        println!(
            "[todo 1.4] mouth close-up mean rgb: neutral {neutral:?}, melanin 0.9 {dark:?}, haemoglobin 1 {red:?}"
        );
        viewer.destroy(&mut res);
        assert!(luma(dark) < luma(neutral) - 10.0, "melanin 0.9 darkens: {dark:?} vs {neutral:?}");
        assert!(red[0] / red[1].max(1.0) > neutral[0] / neutral[1].max(1.0) + 0.05, "blood reddens: {red:?}");
    }

    /// todo 1.4 `[SKN-30]`: the template's authored wrinkle maps are bound and folded. The
    /// document's resting (age) wrinkle depths change only the shading of the face (the geometry
    /// is the same): in an eye close-up (crow's feet, brow, upper cheek) the image changes well
    /// above the frame-to-frame noise of the temporal accumulation, and the fine detail grows.
    /// Stills: `target/agents/skin-surface/shots/test-wrinkles-{rest,aged}.png`.
    #[test]
    fn todo_1_4_resting_wrinkles_crease_the_face() {
        let reg = super::super::tests::registry();
        let template = super::super::tests::template();
        let mut model = Model::new(Arc::new(template.clone()), reg);
        // The registry defaults move the surface on the first sync: apply them before the stills.
        model.sync(&doc(&[]), reg).unwrap();
        let gpu = humanart_gpu::shared_headless().unwrap();
        let mut res = Resources::new(gpu, Budget::UNLIMITED);
        let (w, h) = (480, 320);
        let mut viewer = Viewer::new(gpu, &mut res, &model, w, h).unwrap();
        let wrinkles = viewer.wrinkles().expect("the template's wrinkles are authored and bound");
        assert!(wrinkles.report().missing.is_empty(), "{wrinkles:?}");
        #[allow(clippy::cast_precision_loss)]
        let camera = close_up_camera(&model, CloseUp::Eye, Framing::Head, w as f32 / h as f32).unwrap();
        viewer.set_camera(camera);
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/agents/skin-surface/shots");
        std::fs::create_dir_all(&dir).unwrap();
        let grab = |model: &mut Model, viewer: &mut Viewer, res: &mut Resources, name: &str| {
            frames(gpu, res, viewer, model);
            let (tw, th, rgba) =
                humanart_render::capture::read_texture(res, viewer.display_texture().unwrap(), 0).unwrap();
            let png = humanart_render::capture::encode_png_rgba8(tw, th, &rgba).unwrap();
            std::fs::write(dir.join(name), png).unwrap();
            rgba
        };
        let rest = grab(&mut model, &mut viewer, &mut res, "test-wrinkles-rest.png");
        let rest2 = grab(&mut model, &mut viewer, &mut res, "test-wrinkles-rest2.png");
        let aged = [
            ("skin.wrinkles.forehead", 1.0),
            ("skin.wrinkles.glabella", 1.0),
            ("skin.wrinkles.crows_feet.l", 1.0),
            ("skin.wrinkles.crows_feet.r", 1.0),
            ("skin.wrinkles.nasolabial.l", 1.0),
            ("skin.wrinkles.nasolabial.r", 1.0),
        ];
        model.sync(&doc(&aged), reg).unwrap();
        assert_eq!(model.skin().resting, [1.0; 6]);
        let old = grab(&mut model, &mut viewer, &mut res, "test-wrinkles-aged.png");
        viewer.destroy(&mut res);
        let luma = |img: &[u8], i: usize| {
            0.2126 * f64::from(img[4 * i]) + 0.7152 * f64::from(img[4 * i + 1]) + 0.0722 * f64::from(img[4 * i + 2])
        };
        // The crow's feet: the lateral quarter of the left eye's close-up (+X is image right).
        let x0 = w * 3 / 4;
        let n = ((w - 1 - x0) * (h - 2)) as usize;
        let diff = |a: &[u8], b: &[u8]| {
            let mut d = 0.0;
            for y in 1..h - 1 {
                for x in x0..w - 1 {
                    let i = (y * w + x) as usize;
                    d += (luma(a, i) - luma(b, i)).abs();
                }
            }
            d / n as f64
        };
        let hf = |img: &[u8]| {
            let mut e = 0.0;
            for y in 1..h - 1 {
                for x in x0..w - 1 {
                    let at = |dx: u32, dy: u32| luma(img, ((y + dy - 1) * w + x + dx - 1) as usize);
                    let m =
                        (0..3).flat_map(|dy| (0..3).map(move |dx| (dx, dy))).map(|(dx, dy)| at(dx, dy)).sum::<f64>();
                    e += (at(1, 1) - m / 9.0).abs();
                }
            }
            e / n as f64
        };
        let (noise, change) = (diff(&rest, &rest2), diff(&rest2, &old));
        let (hf_rest, hf_old) = (hf(&rest2), hf(&old));
        println!(
            "[todo 1.4] resting wrinkles, eye close-up, lateral quarter: mean |diff| {change:.3} (noise {noise:.3}); HF {hf_rest:.3} -> {hf_old:.3}"
        );
        assert!(change > 1.5 * noise.max(0.05), "the resting wrinkles change the face: {change} vs noise {noise}");
        assert!(hf_old > hf_rest * 1.05, "creases add fine detail: {hf_rest} -> {hf_old}");
    }

    /// Relative fine detail (mean |luma - 3x3 mean| / mean luma) of a rectangle of an RGBA8 image.
    fn relative_detail(rgba: &[u8], w: u32, rect: [u32; 4]) -> f64 {
        let luma = |x: u32, y: u32| {
            let i = ((y * w + x) * 4) as usize;
            0.2126 * f64::from(rgba[i]) + 0.7152 * f64::from(rgba[i + 1]) + 0.0722 * f64::from(rgba[i + 2])
        };
        let (mut hf, mut sum) = (0.0, 0.0);
        for y in rect[1]..rect[3] {
            for x in rect[0]..rect[2] {
                let mut m = 0.0;
                for dy in 0..3 {
                    for dx in 0..3 {
                        m += luma(x + dx - 1, y + dy - 1);
                    }
                }
                hf += (luma(x, y) - m / 9.0).abs();
                sum += luma(x, y);
            }
        }
        hf / sum.max(1e-9)
    }

    /// todo 1.4, step 3 (measurement, run by hand with `--ignored`): the relative fine detail of a
    /// cheek and a forehead patch in the portrait still (head framing, 1080 x 1350, the CLI's
    /// `render --view head`) with the micro-detail off and on, and with the tone variation off
    /// (both amplitudes 0 in the zone table is the compositor default; here the engine's look is
    /// measured as built). Stills in `target/agents/skin-surface/shots/detail-*.png`.
    #[test]
    #[ignore = "measurement run, writes stills"]
    fn todo_1_4_portrait_fine_detail() {
        let reg = super::super::tests::registry();
        let template = super::super::tests::template();
        let mut model = Model::new(Arc::new(template.clone()), reg);
        model.sync(&doc(&[]), reg).unwrap();
        let gpu = humanart_gpu::shared_headless().unwrap();
        let mut res = Resources::new(gpu, Budget::UNLIMITED);
        let (w, h) = (1080, 1350);
        let mut viewer = Viewer::new(gpu, &mut res, &model, w, h).unwrap();
        viewer.settings_mut().quality = humanart_render::renderer::Quality::Review;
        viewer.frame_model(&model, Framing::Head);
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/agents/skin-surface/shots");
        std::fs::create_dir_all(&dir).unwrap();
        let base = *viewer.skin_settings();
        for detail in [0.0, 1.0] {
            viewer.set_skin_settings(gpu, SurfaceSettings { detail_strength: detail, ..base }).unwrap();
            for _ in 0..8 {
                let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("t") });
                viewer.record(&mut res, &mut enc, &mut model, &[]).unwrap();
                gpu.queue.submit([enc.finish()]);
                viewer.after_submit();
            }
            let (tw, th, rgba) =
                humanart_render::capture::read_texture(&mut res, viewer.display_texture().unwrap(), 0).unwrap();
            let png = humanart_render::capture::encode_png_rgba8(tw, th, &rgba).unwrap();
            std::fs::write(dir.join(format!("detail-{detail}.png")), png).unwrap();
            let cheek = relative_detail(&rgba, tw, [620, 760, 700, 840]);
            let forehead = relative_detail(&rgba, tw, [480, 470, 600, 540]);
            println!("[todo 1.4] detail {detail}: cheek {:.3} %, forehead {:.3} %", 100.0 * cheek, 100.0 * forehead);
        }
        viewer.destroy(&mut res);
    }

    /// todo 1.4 round 2 (diagnosis, run by hand with `--ignored`): the harsh side light on the eye
    /// and the head with the skin's translucency on and off, stills in
    /// `target/agents/skin-surface/shots/diag-*.png`.
    #[test]
    #[ignore = "diagnosis run, writes stills"]
    fn todo_1_4_side_light_translucency_diagnosis() {
        let reg = super::super::tests::registry();
        let template = super::super::tests::template();
        let mut model = Model::new(Arc::new(template.clone()), reg);
        model.sync(&doc(&[]), reg).unwrap();
        let gpu = humanart_gpu::shared_headless().unwrap();
        let mut res = Resources::new(gpu, Budget::UNLIMITED);
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/agents/skin-surface/shots");
        std::fs::create_dir_all(&dir).unwrap();
        for (name, w, h) in [("eye", 600_u32, 400_u32), ("head", 540, 675)] {
            let mut viewer = Viewer::new(gpu, &mut res, &model, w, h).unwrap();
            viewer.settings_mut().quality = humanart_render::renderer::Quality::Review;
            viewer.settings_mut().rig = humanart_render::light::BuiltinRig::HarshSidelight;
            if name == "eye" {
                #[allow(clippy::cast_precision_loss)]
                let camera = close_up_camera(&model, CloseUp::Eye, Framing::Head, w as f32 / h as f32).unwrap();
                viewer.set_camera(camera);
            } else {
                viewer.frame_model(&model, Framing::Head);
            }
            for translucency in [1.0, 0.0] {
                let shading = humanart_render::sss::SkinShading { translucency, ..Default::default() };
                viewer.set_skin_shading(shading).unwrap();
                for _ in 0..8 {
                    let mut enc =
                        gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("t") });
                    viewer.record(&mut res, &mut enc, &mut model, &[]).unwrap();
                    gpu.queue.submit([enc.finish()]);
                    viewer.after_submit();
                }
                let (tw, th, rgba) =
                    humanart_render::capture::read_texture(&mut res, viewer.display_texture().unwrap(), 0).unwrap();
                let png = humanart_render::capture::encode_png_rgba8(tw, th, &rgba).unwrap();
                std::fs::write(dir.join(format!("diag-{name}-t{translucency}.png")), png).unwrap();
            }
            viewer.destroy(&mut res);
        }
    }
}
