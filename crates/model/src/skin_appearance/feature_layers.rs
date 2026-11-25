//! The document's skin feature layers (`todo.md` 2.3, `[DOC-04]`, `[OTH-02]`): a procedural
//! skin layer of the document names a generator ([`super::features`]) and may override some of
//! its parameters; every parameter it does not override takes the document's global value
//! (`skin.features.<generator>.<name>`, [`FeatureGlobals`]), which defaults to the registry's.
//! So a global slider changes every layer of its generator that does not set the value itself,
//! and a layer keeps what it sets. [`FeatureGlobals::layer_params`] resolves one layer.
//!
//! Where a layer goes: [`region_masks`] rasterises the layer's region (and its sub-regions; the
//! whole skin without one) into each UV tile it touches, with the zone bake's rasteriser and
//! dilation, as a placement mask over the covered rectangle; [`generate_layer`] runs the
//! generator once per tile into one layer. Counts (parameters in `count`) are per layer, so each tile gets its
//! share of the masked area; densities (`per-cm2`, `per-mm`) need no share. Editor-time work.
//!
//! The generators draw in a tile's texels, so what they draw lies as the UV layout turns it.
//! Lines that must run around a body part (the neck's resting lines) come from [`level_lines`]:
//! level lines of the rest height, whatever the layout.

use humanart_core::id::Id128;
use humanart_core::rng::Rng;
use humanart_math::{Vec2, Vec3};

use super::dirty::Rect;
use super::error::SkinError;
use super::features::{GeneratorKind, GeneratorParams, Target, generate, groove};
use super::layers::{Layer, LayerContent, LayerId, LayerKind};
use super::zone_bake::{ZoneWeights, bake_tile};
use crate::Template;
use crate::mesh::{Mesh, Triangulation};

/// Texels the region masks are dilated past the UV islands (as the zone masks).
pub const MASK_DILATE: u32 = 8;

/// The generator with a stable name; the registry's spelling with underscores
/// (`stretch_marks`) is taken too.
pub fn generator_of(name: &str) -> Option<GeneratorKind> {
    GeneratorKind::ALL.into_iter().find(|k| {
        let n = k.name();
        n.len() == name.len() && n.bytes().zip(name.bytes()).all(|(a, b)| a == b || (a == b'-' && b == b'_'))
    })
}

/// The document's global generator settings, one [`GeneratorParams`] per generator (in
/// [`GeneratorKind::ALL`] order). See the module documentation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeatureGlobals {
    /// Per generator.
    pub params: [GeneratorParams; GeneratorKind::ALL.len()],
}

impl Default for FeatureGlobals {
    /// The generators' defaults (equal to the registry's, a test of `features` keeps them so).
    fn default() -> Self {
        Self { params: GeneratorKind::ALL.map(GeneratorParams::new) }
    }
}

impl FeatureGlobals {
    /// The generator and parameter index of a registry parameter id
    /// (`skin.features.<generator>.<name>`), `None` for any other id.
    pub fn slot_of(id: &str) -> Option<(usize, usize)> {
        GeneratorKind::ALL
            .iter()
            .enumerate()
            .find_map(|(k, kind)| kind.params().iter().position(|s| s.id == id).map(|i| (k, i)))
    }

    /// Sets the value of a slot ([`FeatureGlobals::slot_of`]), clamped to its range; a non-finite
    /// value or a slot out of range is ignored.
    pub fn set(&mut self, slot: (usize, usize), value: f32) {
        if let Some(p) = self.params.get_mut(slot.0)
            && let Some(spec) = p.kind.params().get(slot.1)
            && value.is_finite()
            && let Some(v) = p.values.get_mut(slot.1)
        {
            *v = value.clamp(spec.min, spec.max);
        }
    }

    /// The global settings of `kind`.
    pub fn of(&self, kind: GeneratorKind) -> &GeneratorParams {
        let i = GeneratorKind::ALL.iter().position(|k| *k == kind).unwrap_or(0);
        &self.params[i.min(self.params.len() - 1)]
    }

    /// The settings of a layer of generator `kind` that overrides `overrides` (by full registry id
    /// or by the parameter's last name, `density`): each override replaces the global value
    /// (clamped); unknown names are ignored. No allocation.
    pub fn layer_params<'a>(
        &self,
        kind: GeneratorKind,
        overrides: impl IntoIterator<Item = (&'a str, f64)>,
    ) -> GeneratorParams {
        let mut p = *self.of(kind);
        for (name, value) in overrides {
            let found = kind.params().iter().find(|s| s.id == name || s.id.rsplit('.').next() == Some(name));
            if let Some(spec) = found {
                #[allow(clippy::cast_possible_truncation)] // Generator values are small.
                p.set(spec.id, value as f32);
            }
        }
        p
    }
}

/// Where a layer's region lies on one UV tile.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionMask {
    /// The UV tile.
    pub tile: u32,
    /// The covered rectangle of the tile.
    pub rect: Rect,
    /// The placement weight per texel of `rect`, row-major, 0..1.
    pub mask: Vec<f32>,
    /// Texel size on the tile, millimetres.
    pub texel_mm: f32,
    /// This tile's share of the region's masked area, 0..1.
    pub share: f32,
}

/// Rasterises `region` (with its sub-regions; the whole skin for `None`) of `template` into the
/// UV tiles at `tile_size`² texels (see the module documentation). Empty when the template has no
/// such region or it covers no texel.
///
/// # Errors
/// [`SkinError::Invalid`] when the template's tiles or region map are unusable.
pub fn region_masks(template: &Template, region: Option<&str>, tile_size: u32) -> Result<Vec<RegionMask>, SkinError> {
    let regions = template.regions();
    let id = match region {
        Some(name) => match regions.find(name) {
            Some(id) => Some(id),
            None => return Ok(Vec::new()),
        },
        None => None,
    };
    let mesh = template.mesh();
    let values: Vec<f32> = regions
        .vertex_regions
        .iter()
        .map(|&r| if id.is_none_or(|id| regions.is_within(r, id)) { 1.0 } else { 0.0 })
        .collect();
    let mut weights = ZoneWeights::new(mesh.vertex_count());
    if !weights.set_plain(0, &values) {
        return Err(SkinError::Invalid { what: "the region map does not match the mesh" });
    }
    let triangles = mesh.triangulate();
    let mut out = Vec::new();
    for (index, tile) in template.tiles().iter().enumerate() {
        let index16 = u16::try_from(index).map_err(|_| SkinError::Invalid { what: "UV tile index" })?;
        let masks = bake_tile(mesh, &triangles, &weights, index16, tile_size, MASK_DILATE)?;
        let (mut lo, mut hi) = ((tile_size, tile_size), (0, 0));
        for y in 0..tile_size {
            for x in 0..tile_size {
                if masks.weight(0, x, y) > 0.0 {
                    lo = (lo.0.min(x), lo.1.min(y));
                    hi = (hi.0.max(x), hi.1.max(y));
                }
            }
        }
        if hi.0 < lo.0 || hi.1 < lo.1 {
            continue;
        }
        let rect = Rect::new(lo.0, lo.1, hi.0 - lo.0 + 1, hi.1 - lo.1 + 1);
        let mask: Vec<f32> = (rect.y..rect.bottom())
            .flat_map(|y| (rect.x..rect.right()).map(move |x| (x, y)))
            .map(|(x, y)| masks.weight(0, x, y))
            .collect();
        #[allow(clippy::cast_precision_loss)] // Tile sizes are far below 2^24.
        let texel_mm = tile.resolution as f32 / tile.target_texels_per_mm / tile_size as f32;
        let tile = u32::try_from(index).map_err(|_| SkinError::Invalid { what: "UV tile index" })?;
        out.push(RegionMask { tile, rect, mask, texel_mm, share: 0.0 });
    }
    let area = |m: &RegionMask| m.mask.iter().sum::<f32>() * m.texel_mm * m.texel_mm;
    let total: f32 = out.iter().map(area).sum();
    for m in &mut out {
        m.share = if total > 0.0 { area(m) / total } else { 0.0 };
    }
    Ok(out)
}

/// Generates one feature layer: `params` over every tile of `masks` (one content rectangle per
/// tile, in one layer), counts scaled by the tile's share, with `opacity` (0..1), id `id` and a
/// seed per tile derived from `seed`. `None` when the masks cover nothing.
///
/// # Errors
/// The generator's errors.
pub fn generate_layer(
    params: &GeneratorParams,
    masks: &[RegionMask],
    seed: u64,
    opacity: f32,
    id: LayerId,
) -> Result<Option<Layer>, SkinError> {
    let mut out: Option<Layer> = None;
    for m in masks {
        if m.share <= 0.0 {
            continue;
        }
        let mut p = *params;
        for (v, spec) in p.values.iter_mut().zip(p.kind.params()) {
            if spec.unit == "count" {
                *v = (*v * m.share).round().clamp(spec.min, spec.max);
            }
        }
        // Another seed per tile that keeps `seed % 3` (the wrinkle generator's pattern).
        let tile_seed = seed.wrapping_add((u64::from(m.tile) << 32).wrapping_mul(3));
        let target = Target { tile: m.tile, rect: m.rect, texel_mm: m.texel_mm, mask: Some(&m.mask) };
        let layer = generate(&p, &target, id, tile_seed, None)?;
        match (&mut out, layer.kind) {
            (Some(Layer { kind: LayerKind::Content { tiles, .. }, .. }), LayerKind::Content { tiles: more, .. }) => {
                tiles.extend(more);
            }
            (None, kind) => out = Some(Layer { kind, ..layer }),
            _ => {}
        }
    }
    if let Some(layer) = &mut out {
        layer.opacity = if opacity.is_finite() { opacity.clamp(0.0, 1.0) } else { 1.0 };
    }
    Ok(out)
}

/// The landmarks spot features (moles) keep clear of, with the clearance in metres (todo 1.1): a
/// mole in the jugular notch, on a nipple, in the navel or at an eye or mouth corner reads as a
/// blemish of the model, not of the person. Moles are placed in texture space, so a UV repack
/// moved one onto the notch.
pub const SPOT_CLEARANCE: &[(&str, f32)] = &[
    ("torso.suprasternal", 0.025),
    ("L.chest.nipple", 0.02),
    ("R.chest.nipple", 0.02),
    ("torso.navel", 0.02),
    ("L.eye.canthus.medial", 0.008),
    ("L.eye.canthus.lateral", 0.008),
    ("R.eye.canthus.medial", 0.008),
    ("R.eye.canthus.lateral", 0.008),
    ("L.lip.corner", 0.008),
    ("R.lip.corner", 0.008),
];

/// `masks` (made from `template` at `tile_size`² texels) with the texels whose rest surface lies
/// within each landmark's clearance set to zero, fading to full over the next half clearance.
/// Landmarks the template lacks are skipped. The tiles' shares are kept (counts stay those of
/// the whole region).
#[must_use]
pub fn clear_of_landmarks(
    template: &Template,
    masks: &[RegionMask],
    tile_size: u32,
    clearance: &[(&str, f32)],
) -> Vec<RegionMask> {
    let mesh = template.mesh();
    let points: Vec<(Vec3, f32)> = clearance
        .iter()
        .filter_map(|&(name, r)| {
            let l = template.landmarks().iter().find(|l| l.name == name)?;
            Some((l.position(mesh.positions())?, r))
        })
        .collect();
    if points.is_empty() {
        return masks.to_vec();
    }
    let triangles = mesh.triangulate();
    masks
        .iter()
        .map(|m| {
            let mut out = m.clone();
            if m.mask.len() as u64 != m.rect.area() {
                return out;
            }
            let surface = surface_under(mesh, &triangles, m, tile_size);
            for ((w, texel), &covered) in out.mask.iter_mut().zip(&surface.texels).zip(&surface.covered) {
                if !covered {
                    continue;
                }
                let p = Vec3::new(texel[0], texel[1], texel[2]);
                for &(q, r) in &points {
                    let d = p.distance(q);
                    let t = ((d - r) / (0.5 * r)).clamp(0.0, 1.0);
                    *w *= t * t * (3.0 - 2.0 * t);
                }
            }
            out
        })
        .collect()
}

/// Most lines [`level_lines`] draws.
pub const MAX_LEVEL_LINES: usize = 32;
/// The least slope of the rest height along the surface that [`level_lines`] divides by: on a
/// nearly level patch (under the chin) a line would otherwise widen into a band.
const MIN_SLOPE: f32 = 0.25;
/// Largest wave of a level line about its height, millimetres.
const LEVEL_WAVE_MM: f32 = 1.5;

/// The rest surface under the texels of one region mask, row-major over its rectangle.
pub(crate) struct Surface {
    /// Rest position `(x, y, z)` in metres and the slope of the height along the surface (1 on a
    /// wall, 0 on a floor) per texel.
    pub(crate) texels: Vec<[f32; 4]>,
    /// Whether a triangle or the dilation reached the texel.
    pub(crate) covered: Vec<bool>,
}

/// Rasterises the rest positions of the triangles on `m`'s tile into the texels of its rectangle
/// (texel centres, as [`bake_tile`]) and grows them [`MASK_DILATE`] texels past the islands.
pub(crate) fn surface_under(mesh: &Mesh, triangles: &Triangulation, m: &RegionMask, tile_size: u32) -> Surface {
    surface_under_where(mesh, triangles, m, tile_size, |_| true)
}

/// [`surface_under`] with only the triangles of the faces `keep` accepts.
pub(crate) fn surface_under_where(
    mesh: &Mesh,
    triangles: &Triangulation,
    m: &RegionMask,
    tile_size: u32,
    keep: impl Fn(u32) -> bool,
) -> Surface {
    let (positions, uvs, face_tiles) = (mesh.positions(), mesh.uvs(), mesh.face_tiles());
    let r = m.rect;
    let (w, h) = (r.w as usize, r.h as usize);
    let mut texels = vec![[0.0_f32; 4]; w * h];
    let mut covered = vec![false; w * h];
    let scale = tile_size as f32;
    let edge = |a: Vec2, b: Vec2, c: Vec2| (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    for t in 0..triangles.len() {
        let (Some(corners), Some(&face)) = (triangles.triangle(t), triangles.triangle_faces.get(t)) else { continue };
        if face_tiles.get(face as usize).map(|&tile| u32::from(tile)) != Some(m.tile) || !keep(face) {
            continue;
        }
        let corner = |k: usize| {
            let v = *corners.get(k)? as usize;
            Some((*uvs.get(v)? * scale, *positions.get(v)?))
        };
        let (Some((a, pa)), Some((b, pb)), Some((c, pc))) = (corner(0), corner(1), corner(2)) else { continue };
        let area = edge(a, b, c);
        if !area.is_finite() || area.abs() < 1e-12 {
            continue;
        }
        // The texels of the rectangle whose centres the triangle's bounding box holds.
        let x0 = (a.x.min(b.x).min(c.x) - 0.5).floor().max(r.x as f32);
        let x1 = (a.x.max(b.x).max(c.x) - 0.5).ceil().min(r.right() as f32 - 1.0);
        let y0 = (a.y.min(b.y).min(c.y) - 0.5).floor().max(r.y as f32);
        let y1 = (a.y.max(b.y).max(c.y) - 0.5).ceil().min(r.bottom() as f32 - 1.0);
        if x0 > x1 || y0 > y1 {
            continue;
        }
        let (e1, e2) = ([pb.x - pa.x, pb.y - pa.y, pb.z - pa.z], [pc.x - pa.x, pc.y - pa.y, pc.z - pa.z]);
        let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        let slope = if length > 0.0 { n[0].hypot(n[2]) / length } else { 1.0 };
        let tol = -1e-4 * area.abs();
        for y in y0 as u32..=y1 as u32 {
            for x in x0 as u32..=x1 as u32 {
                let at = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let (l0, l1, l2) =
                    (edge(b, c, at) * area.signum(), edge(c, a, at) * area.signum(), edge(a, b, at) * area.signum());
                if l0 < tol || l1 < tol || l2 < tol {
                    continue;
                }
                let inv = 1.0 / area.abs();
                let mix = |p: f32, q: f32, s: f32| (l0 * p + l1 * q + l2 * s) * inv;
                let i = (y - r.y) as usize * w + (x - r.x) as usize;
                if let (Some(texel), Some(seen)) = (texels.get_mut(i), covered.get_mut(i)) {
                    *texel = [mix(pa.x, pb.x, pc.x), mix(pa.y, pb.y, pc.y), mix(pa.z, pb.z, pc.z), slope];
                    *seen = true;
                }
            }
        }
    }
    // One ring per pass: a new texel takes the mean of its covered 4-neighbours.
    let mut grown: Vec<(usize, [f32; 4])> = Vec::new();
    for _ in 0..MASK_DILATE {
        grown.clear();
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if covered[i] {
                    continue;
                }
                let around = [
                    (x > 0).then(|| i - 1),
                    (x + 1 < w).then(|| i + 1),
                    (y > 0).then(|| i - w),
                    (y + 1 < h).then(|| i + w),
                ];
                let (mut sum, mut count) = ([0.0_f32; 4], 0.0_f32);
                for j in around.into_iter().flatten().filter(|&j| covered[j]) {
                    count += 1.0;
                    for (acc, v) in sum.iter_mut().zip(texels[j]) {
                        *acc += v;
                    }
                }
                if count > 0.0 {
                    grown.push((i, sum.map(|s| s / count)));
                }
            }
        }
        if grown.is_empty() {
            break;
        }
        for &(i, v) in &grown {
            texels[i] = v;
            covered[i] = true;
        }
    }
    Surface { texels, covered }
}

/// The range that holds the middle 98 % of `values` (sorted in place), `None` without values: a
/// few stray texels of a region (a vertex tagged with it far from the rest) do not stretch it.
fn core_range(values: &mut [f32]) -> Option<(f32, f32)> {
    values.sort_unstable_by(f32::total_cmp);
    let skip = values.len() / 100;
    Some((*values.get(skip)?, *values.get(values.len().checked_sub(skip + 1)?)?))
}

/// One level line: where it lies and how it varies along its way.
#[derive(Clone, Copy)]
struct LevelLine {
    /// Rest height, metres.
    level: f32,
    /// Wave about the height: amplitude (metres), wave number (radians per metre of x), phase.
    wave: (f32, f32, f32),
    /// Depth along the line: wave number and phase of its slow swell.
    swell: (f32, f32),
    /// Depth of the whole line, 0..1.
    strength: f32,
}

/// Lines that run AROUND a body part, each at one rest height: the wrinkle generator's settings
/// (`params`: depth, count, width, age, as its rows) drawn as level lines of the rest height over
/// the region `masks` were made from on `template` at `tile_size`² texels. The lines follow the
/// body, not the UV layout: they cross seams and tiles and stay level on an island that is
/// curved or turned on its tile (the neck's is both, so rows drawn in the tile's texels wander
/// up and down the neck). Every line has its own slight wave, a slow swell of its depth and is
/// deepest in front (+Z); a line is at least a texel wide. The count is the number of lines
/// over the whole region (up to [`MAX_LEVEL_LINES`]), not per tile, spread over the heights that
/// hold the middle 98 % of the region's texels. One content rectangle per tile, in one layer;
/// `None` when the masks cover no surface.
///
/// # Errors
/// [`SkinError::Invalid`] when `params` are not the wrinkle generator's or a mask does not fit
/// its rectangle and the tile; [`SkinError::Limit`] for a content rectangle that is too large.
pub fn level_lines(
    template: &Template,
    masks: &[RegionMask],
    tile_size: u32,
    params: &GeneratorParams,
    seed: u64,
    opacity: f32,
    id: LayerId,
) -> Result<Option<Layer>, SkinError> {
    if params.kind != GeneratorKind::Wrinkles {
        return Err(SkinError::Invalid { what: "level lines take the wrinkle generator's settings" });
    }
    for m in masks {
        let fits = m.rect.right() <= tile_size && m.rect.bottom() <= tile_size && !m.rect.is_empty();
        if !fits || m.mask.len() as u64 != m.rect.area() || !m.texel_mm.is_finite() || m.texel_mm <= 0.0 {
            return Err(SkinError::Invalid { what: "a region mask does not fit its rectangle and the tile" });
        }
    }
    let mesh = template.mesh();
    let triangles = mesh.triangulate();
    let surfaces: Vec<Surface> = masks.iter().map(|m| surface_under(mesh, &triangles, m, tile_size)).collect();
    // The region's extent: where its mask is solid.
    let mut solid: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (m, s) in masks.iter().zip(&surfaces) {
        for ((&weight, texel), &covered) in m.mask.iter().zip(&s.texels).zip(&s.covered) {
            if covered && weight >= 0.5 {
                for (axis, &v) in solid.iter_mut().zip(texel) {
                    axis.push(v);
                }
            }
        }
    }
    let [Some(x), Some(y), Some(z)] = solid.each_mut().map(|axis| core_range(axis)) else { return Ok(None) };
    let (lo, hi) = ([x.0, y.0, z.0], [x.1, y.1, z.1]);
    let span = hi[1] - lo[1];
    if !span.is_finite() || span <= 1e-4 {
        return Ok(None);
    }
    let age = params.get(3);
    let depth = -params.get(0) * (0.3 + 0.7 * age);
    let radius_mm = params.get(2) * 0.5 * (1.0 + 0.5 * age);
    let count = (params.get(1).round().max(1.0) as usize).min(MAX_LEVEL_LINES);
    let spacing = span / (count as f32 + 1.0);
    let mut rng = Rng::new(seed);
    let tau = std::f32::consts::TAU;
    let lines: Vec<LevelLine> = (0..count)
        .map(|i| LevelLine {
            level: lo[1] + spacing * (i as f32 + 1.0 + rng.range_f32(-0.2, 0.2)),
            wave: (
                rng.range_f32(0.3, 1.0) * (LEVEL_WAVE_MM * 1e-3).min(0.15 * spacing),
                tau / rng.range_f32(0.03, 0.08),
                rng.range_f32(0.0, tau),
            ),
            swell: (tau / rng.range_f32(0.05, 0.12), rng.range_f32(0.0, tau)),
            strength: rng.range_f32(0.6, 1.0),
        })
        .collect();
    let values = [depth, -0.4 * depth.abs().min(1.0) * 2.0];
    let depth_z = (hi[2] - lo[2]).max(1e-4);
    let (blend, attributes) = params.kind.layout();
    let mut layer = Layer::content(id, params.kind.name(), blend, attributes);
    let mut tiles = Vec::with_capacity(masks.len());
    for (m, s) in masks.iter().zip(&surfaces) {
        let mut content = LayerContent::new(m.tile, m.rect)?;
        // A groove narrower than a texel would break into dots.
        let radius = radius_mm.max(m.texel_mm) * 1e-3;
        let mut any = false;
        let texels = m.mask.iter().zip(&s.texels).zip(&s.covered);
        for ((out, cover), ((&weight, texel), &covered)) in
            content.values.iter_mut().zip(content.coverage.iter_mut()).zip(texels)
        {
            *cover = 1.0;
            if !covered || weight <= 0.0 {
                continue;
            }
            let [x, y, z, slope] = *texel;
            let mut best = 0.0_f32;
            for line in &lines {
                let level = line.level + line.wave.0 * (line.wave.1 * x + line.wave.2).sin();
                let d = (y - level).abs() / slope.max(MIN_SLOPE) / radius;
                let swell = 0.75 + 0.25 * (line.swell.0 * x + line.swell.1).sin();
                best = best.max(groove(d) * line.strength * swell);
            }
            let front = ((z - lo[2]) / depth_z).clamp(0.0, 1.0);
            let amount = best * (0.45 + 0.55 * front * front * (3.0 - 2.0 * front)) * weight.clamp(0.0, 1.0);
            if amount > 0.0 {
                any = true;
                for (dst, v) in out.iter_mut().zip(values) {
                    *dst = v * amount;
                }
            }
        }
        if any {
            tiles.push(content);
        }
    }
    if tiles.is_empty() {
        return Ok(None);
    }
    layer.kind = LayerKind::Content { attributes, tiles };
    layer.opacity = if opacity.is_finite() { opacity.clamp(0.0, 1.0) } else { 1.0 };
    Ok(Some(layer))
}

/// One procedural feature layer as a view generates it: the document layer's id, generator,
/// resolved settings, seed, region and opacity.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureLayer {
    /// The document layer's id.
    pub id: Id128,
    /// The resolved settings (global values with the layer's overrides).
    pub params: GeneratorParams,
    /// The seed.
    pub seed: u64,
    /// The region, `None` for the whole skin.
    pub region: Option<String>,
    /// Opacity, 0..1.
    pub opacity: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn todo_2_3_globals_reach_layers_without_an_override() {
        assert_eq!(generator_of("stretch_marks"), Some(GeneratorKind::StretchMarks));
        assert_eq!(generator_of("stretch-marks"), Some(GeneratorKind::StretchMarks));
        assert_eq!(generator_of("stretch"), None);
        let mut g = FeatureGlobals::default();
        let slot = FeatureGlobals::slot_of("skin.features.freckles.density").unwrap();
        assert_eq!(FeatureGlobals::slot_of("skin.pores.density"), None);
        g.set(slot, 120.0);
        let free = g.layer_params(GeneratorKind::Freckles, []);
        assert_eq!(free.values[0], 120.0, "the global value reaches a layer that does not set it");
        let own = g.layer_params(GeneratorKind::Freckles, [("density", 10.0), ("skin.features.freckles.size", 2.0)]);
        assert_eq!((own.values[0], own.values[1]), (10.0, 2.0), "the layer's overrides win");
        let clamped = g.layer_params(GeneratorKind::Freckles, [("density", 1e9), ("nonsense", 1.0)]);
        assert_eq!(clamped.values[0], 200.0);
        g.set(slot, f32::NAN);
        assert_eq!(g.of(GeneratorKind::Freckles).values[0], 120.0);
        g.set((99, 0), 1.0);
        for kind in GeneratorKind::ALL {
            for spec in kind.params() {
                assert!(FeatureGlobals::slot_of(spec.id).is_some(), "{}", spec.id);
            }
        }
    }

    /// The region masks of the template cover the region on its tiles and share its area; the
    /// layers of a count generator split the count by the shares.
    #[test]
    fn todo_2_3_region_masks_and_layers_follow_the_region() {
        let t = crate::generate::build::template(&crate::generate::build::Options {
            lods: false,
            ..crate::generate::build::Options::default()
        })
        .unwrap()
        .template;
        assert!(region_masks(&t, Some("no-such-region"), 128).unwrap().is_empty());
        let cheek = region_masks(&t, Some("cheek-L"), 128).unwrap();
        assert_eq!(cheek.len(), 1, "one cheek lies on one tile");
        assert!((cheek[0].share - 1.0).abs() < 1e-6);
        assert!(cheek[0].mask.iter().any(|&m| m > 0.9));
        let all = region_masks(&t, None, 64).unwrap();
        println!(
            "[todo 2.3] whole skin on {} tiles, shares {:?}",
            all.len(),
            all.iter().map(|m| m.share).collect::<Vec<_>>()
        );
        assert!(all.len() > 1 && (all.iter().map(|m| m.share).sum::<f32>() - 1.0).abs() < 1e-4);
        let mut g = FeatureGlobals::default();
        g.set(FeatureGlobals::slot_of("skin.features.moles.count").unwrap(), 40.0);
        let params = g.layer_params(GeneratorKind::Moles, []);
        let layer = generate_layer(&params, &all, 7, 0.8, LayerId(3)).unwrap().unwrap();
        assert_eq!(layer.opacity, 0.8);
        let LayerKind::Content { tiles, .. } = &layer.kind else { panic!("a content layer") };
        assert!(tiles.len() > 1, "one rectangle per tile, one layer");
    }

    /// todo 1.1: the masks moles are placed with keep clear of the jugular notch, the nipples, the
    /// navel and the eye and mouth corners; elsewhere they are untouched.
    #[test]
    fn todo_1_1_moles_keep_clear_of_landmarks() {
        let t = crate::generate::build::template(&crate::generate::build::Options {
            lods: false,
            ..crate::generate::build::Options::default()
        })
        .unwrap()
        .template;
        let size = 256;
        let all = region_masks(&t, None, size).unwrap();
        let cleared = clear_of_landmarks(&t, &all, size, SPOT_CLEARANCE);
        assert_eq!(cleared.len(), all.len());
        let triangles = t.mesh().triangulate();
        let landmark =
            |n: &str| t.landmarks().iter().find(|l| l.name == n).unwrap().position(t.mesh().positions()).unwrap();
        let (mut near, mut kept, mut far) = (0, 0, 0);
        for (before, after) in all.iter().zip(&cleared) {
            assert_eq!((before.tile, before.rect, before.share), (after.tile, after.rect, after.share));
            let s = surface_under(t.mesh(), &triangles, before, size);
            for ((&b, &a), (texel, &covered)) in
                before.mask.iter().zip(&after.mask).zip(s.texels.iter().zip(&s.covered))
            {
                if !covered || b <= 0.0 {
                    continue;
                }
                let p = Vec3::new(texel[0], texel[1], texel[2]);
                let gap =
                    SPOT_CLEARANCE.iter().map(|&(n, r)| p.distance(landmark(n)) / r).fold(f32::INFINITY, f32::min);
                if gap < 1.0 {
                    near += 1;
                    assert_eq!(a, 0.0, "a texel inside a clearance keeps weight {a}");
                } else if gap > 1.5 {
                    far += 1;
                    kept += usize::from(a == b);
                }
            }
        }
        println!("[todo 1.1] {near} texels cleared, {kept} of {far} far texels kept");
        assert!(near > 20, "the clearances cover texels");
        assert_eq!(kept, far, "texels away from every landmark keep their weight");
    }

    /// The neck's lines are level lines of the rest height: every marked texel lies near one of
    /// the few heights (rows in the tile's texels wander over the whole neck, its island is an
    /// arc), and the lines go around the neck (they span its width, front and back).
    #[test]
    fn todo_2_3_level_lines_run_around_the_neck_at_their_heights() {
        let t = crate::generate::build::template(&crate::generate::build::Options {
            lods: false,
            ..crate::generate::build::Options::default()
        })
        .unwrap()
        .template;
        let size = 512;
        let masks = region_masks(&t, Some("neck"), size).unwrap();
        assert!(!masks.is_empty());
        let mut params = GeneratorParams::new(GeneratorKind::Wrinkles);
        params.set("skin.features.wrinkles.count", 4.0);
        params.set("skin.features.wrinkles.depth", 0.7);
        params.set("skin.features.wrinkles.width", 1.8);
        params.set("skin.features.wrinkles.age", 1.0);
        let layer = level_lines(&t, &masks, size, &params, 9, 0.5, LayerId(5)).unwrap().unwrap();
        assert_eq!(layer.opacity, 0.5);
        let again = level_lines(&t, &masks, size, &params, 9, 0.5, LayerId(5)).unwrap().unwrap();
        assert_eq!(layer, again, "deterministic");
        let LayerKind::Content { tiles, .. } = &layer.kind else { panic!("a content layer") };
        let triangles = t.mesh().triangulate();
        // Rest positions of the region's texels, of the texels a line touches, and the rest
        // heights (mm bins) of the texels in the deep part of a groove.
        let mut bins = std::collections::BTreeSet::new();
        let mut solid: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut touched: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut deepest = 0.0_f32;
        for m in &masks {
            let s = surface_under(t.mesh(), &triangles, m, size);
            let content = tiles.iter().find(|c| c.tile == m.tile);
            assert!(content.is_none_or(|c| c.rect == m.rect));
            for (i, (texel, &weight)) in s.texels.iter().zip(&m.mask).enumerate() {
                if weight >= 0.5 && s.covered[i] {
                    solid.iter_mut().zip(texel).for_each(|(axis, &v)| axis.push(v));
                }
                let Some(v) = content.map(|c| c.values[i]) else { continue };
                assert!(v[0] <= 0.0 && v[1] <= 0.0 && v[2] == 0.0 && v[3] == 0.0, "grooves only: {v:?}");
                deepest = deepest.min(v[0]);
                if v[0] < -0.02 {
                    touched.iter_mut().zip(texel).for_each(|(axis, &v)| axis.push(v));
                }
                if v[0] < -0.2 {
                    bins.insert((texel[1] * 1000.0).floor() as i32);
                }
            }
        }
        let count = solid[1].len();
        let [x, y, z] = solid.each_mut().map(|axis| core_range(axis).unwrap());
        let [tx, _, tz] = touched.each_mut().map(|axis| core_range(axis).unwrap());
        let span_mm = (y.1 - y.0) * 1000.0;
        let texel_mm = masks.iter().map(|m| m.texel_mm).fold(0.0, f32::max);
        println!(
            "[todo 2.3] neck lines: {} of {count} texels touched, the deep ones on {} mm of heights of a {span_mm:.0} mm \
             neck, deepest {deepest:.2} mm, largest texel {texel_mm:.2} mm; x {tx:.3?} of {x:.3?}, z {tz:.3?} of {z:.3?}",
            touched[1].len(),
            bins.len(),
        );
        assert!(touched[1].len() > 400 && (-0.7..-0.4).contains(&deepest));
        assert!((60.0..200.0).contains(&span_mm), "a neck, not the strays of its region map: {span_mm} mm");
        // Four lines, each within its wave and the groove's deep part of its height.
        let band_mm = 2.0 * (LEVEL_WAVE_MM + 1.35_f32.max(texel_mm)) + 1.0;
        assert!((bins.len() as f32) <= 4.0 * band_mm, "{} mm of heights", bins.len());
        assert!((bins.len() as f32) < 0.5 * span_mm, "the lines keep to their heights");
        // Around the neck: nearly its whole width, front and back.
        assert!(tx.1 - tx.0 > 0.8 * (x.1 - x.0));
        assert!(tz.1 - tz.0 > 0.8 * (z.1 - z.0));
        assert_eq!(core_range(&mut []), None);
        // Other settings and broken masks are refused, an empty region gives nothing.
        let freckles = GeneratorParams::new(GeneratorKind::Freckles);
        assert!(level_lines(&t, &masks, size, &freckles, 9, 1.0, LayerId(5)).is_err());
        assert!(level_lines(&t, &masks, 64, &params, 9, 1.0, LayerId(5)).is_err(), "masks of another tile size");
        assert_eq!(level_lines(&t, &[], size, &params, 9, 1.0, LayerId(5)).unwrap(), None);
    }
}
