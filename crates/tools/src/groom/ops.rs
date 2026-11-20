//! Guide-editing operations `[HAIR-03]` `[TOOL-05]`.
//!
//! Every operation edits groom data in place and returns an [`EditReport`] (the guides it changed
//! and the number of guide points, map texels or triangles it touched, the `elements` of the
//! command outcome). Brushes work in world space on the rest surface: a [`Brush`] has a centre, a
//! radius and a strength, and weighs a point at distance `d` by
//! `strength · (1 − (d / radius)²)²` inside the radius (smooth to zero at the rim). Map painting
//! works in UV space (the editor maps its brush to the UV under the cursor). The operations
//! allocate per call, not per frame; they run on a stroke sample, not in a render loop.
//!
//! - [`place_guide`]: a new guide grown at the closest skin point.
//! - [`comb`]: turns guide segments toward a stroke direction, more toward the tip, keeping
//!   segment lengths, and pushes points out of the head.
//! - [`cut`]: shortens guides where they enter the brush sphere.
//! - [`paint_map`]: paints a groom map (density, clump, length, curl, frizz, colour, direction)
//!   toward a value.
//! - [`edit_hairline`]: adds or removes scalp triangles under the brush (the hairline).
//! - [`set_parting`] and [`place_cowlick`]: parting and cowlick placement in the parameters.
//! - [`mirror`]: replaces the guides on one side of the `x = 0` plane by the mirror image of the
//!   other side.

use humanart_math::{Vec2, Vec3};
use humanart_model::hair::collide::{SdfGrid, TriangleGrid};
use humanart_model::hair::params::MAX_COWLICKS;
use humanart_model::hair::presets::{GrowSpec, grow_guide};
use humanart_model::hair::surface::normalize_or;
use humanart_model::hair::{Cowlick, GroomMap, GroomParams, Guides, HairSurface, Parting};

use super::{GroomToolError, Result};

/// A world-space brush.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Brush {
    /// The centre.
    pub centre: Vec3,
    /// The radius, metres (> 0).
    pub radius: f32,
    /// The strength, `0..=1`.
    pub strength: f32,
}

impl Brush {
    /// The weight of a point at `p` (0 outside).
    #[must_use]
    pub fn weight(&self, p: Vec3) -> f32 {
        let d2 = (p - self.centre).length_squared() / (self.radius * self.radius);
        if d2 >= 1.0 { 0.0 } else { self.strength.clamp(0.0, 1.0) * (1.0 - d2) * (1.0 - d2) }
    }

    fn check(&self) -> Result<()> {
        if self.centre.is_finite() && self.radius.is_finite() && self.radius > 0.0 && self.strength.is_finite() {
            Ok(())
        } else {
            Err(GroomToolError::Invalid("brush centre, radius or strength not usable"))
        }
    }
}

/// What an operation touched.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EditReport {
    /// The guides that changed (ascending).
    pub guides: Vec<u32>,
    /// Guide points, map texels or triangles touched.
    pub elements: u64,
}

/// Grows a new guide at the skin point closest to `at` and returns its index.
pub fn place_guide(
    guides: &mut Guides,
    surface: &HairSurface,
    grid: &TriangleGrid,
    at: Vec3,
    spec: &GrowSpec,
    sdf: Option<&SdfGrid>,
) -> Result<usize> {
    if !at.is_finite() {
        return Err(GroomToolError::Invalid("guide position not finite"));
    }
    let hit = grid.closest(at).ok_or(GroomToolError::Invalid("no skin to place the guide on"))?;
    let frame = surface
        .frame(hit.triangle, hit.bary)
        .ok_or(GroomToolError::Invalid("closest triangle not on the groom surface"))?;
    let world = grow_guide(&frame, spec, guides.points(), sdf);
    let local: Vec<Vec3> = world.iter().map(|&w| frame.to_local(w)).collect();
    Ok(guides.push(hit.triangle, hit.bary, &local)?)
}

/// Combs the guides under `brush` toward the world `direction`: each segment turns toward the
/// direction by the brush weight at its end times its position along the guide (the root stays,
/// the tip follows most); segment lengths are kept; with `sdf`, points are pushed out to
/// `offset` above the skin.
pub fn comb(
    guides: &mut Guides,
    surface: &HairSurface,
    brush: &Brush,
    direction: Vec3,
    sdf: Option<&SdfGrid>,
    offset: f32,
) -> Result<EditReport> {
    brush.check()?;
    let dir = direction.try_normalize().ok_or(GroomToolError::Invalid("comb direction is zero or not finite"))?;
    let mut report = EditReport::default();
    let p = guides.points() as usize;
    for i in 0..guides.len() {
        let Some(frame) = guides.frame(i, surface) else { continue };
        let world = guides.world_points(i, &frame);
        if world.iter().all(|&q| brush.weight(q) == 0.0) {
            continue;
        }
        let mut out = Vec::with_capacity(p);
        out.push(world[0]);
        let mut touched = 0_u64;
        for k in 1..p {
            let seg = world[k] - world[k - 1];
            let len = seg.length();
            let w = brush.weight(world[k]) * (k as f32 / (p - 1) as f32);
            touched += u64::from(w > 0.0);
            let d = normalize_or(seg, dir);
            let turned = normalize_or(d.lerp(dir, w), d);
            let mut q = out[k - 1] + turned * len;
            if let Some(grid) = sdf {
                let (dist, g) = grid.sample(q);
                if dist < offset {
                    q += normalize_or(g, frame.n) * (offset - dist);
                    q = out[k - 1] + normalize_or(q - out[k - 1], turned) * len;
                }
            }
            out.push(q);
        }
        if touched == 0 {
            continue;
        }
        for (o, w) in guides.offsets_mut(i).iter_mut().zip(&out) {
            *o = frame.to_local(*w);
        }
        report.guides.push(i as u32);
        report.elements += touched;
    }
    Ok(report)
}

/// Cuts the guides that enter the brush sphere where they first enter it (the root never counts),
/// keeping at least `min_length`; the remaining curve is resampled to the guide's point count.
pub fn cut(guides: &mut Guides, surface: &HairSurface, brush: &Brush, min_length: f32) -> Result<EditReport> {
    brush.check()?;
    let r2 = brush.radius * brush.radius;
    let mut report = EditReport::default();
    for i in 0..guides.len() {
        let Some(frame) = guides.frame(i, surface) else { continue };
        let world = guides.world_points(i, &frame);
        let inside = |q: Vec3| (q - brush.centre).length_squared() < r2;
        let Some(k) = (1..world.len()).find(|&k| inside(world[k])) else { continue };
        // The entry point on segment k-1..k, by bisection (the start is outside or the root).
        let (mut lo, mut hi) = (0.0_f32, 1.0_f32);
        for _ in 0..24 {
            let mid = lo.midpoint(hi);
            if inside(world[k - 1].lerp(world[k], mid)) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let mut kept: Vec<Vec3> = world[..k].to_vec();
        kept.push(world[k - 1].lerp(world[k], lo));
        let length: f32 = kept.windows(2).map(|w| (w[1] - w[0]).length()).sum();
        if length < min_length.max(0.0) {
            // Extend along the last direction to the minimum length.
            let last = kept.len() - 1;
            let d = normalize_or(kept[last] - kept[last - 1], frame.n);
            kept[last] += d * (min_length - length);
        }
        guides.set_world_points(i, &frame, &kept)?;
        report.guides.push(i as u32);
        report.elements += (world.len() - k) as u64;
    }
    Ok(report)
}

/// Paints `map` toward `value` (one entry per channel) inside the UV circle at `centre` with
/// `radius`, weighted like a brush with `strength`. Returns the texels changed.
pub fn paint_map(map: &mut GroomMap, centre: Vec2, radius: f32, strength: f32, value: &[f32]) -> Result<u64> {
    if value.len() != map.channels() as usize || !centre.is_finite() || !(radius > 0.0 && radius.is_finite()) {
        return Err(GroomToolError::Invalid("paint value, centre or radius not usable"));
    }
    if !strength.is_finite() || value.iter().any(|v| !v.is_finite()) {
        return Err(GroomToolError::Invalid("paint strength or value not finite"));
    }
    let (w, h) = (map.width(), map.height());
    let x0 = (((centre.x - radius) * w as f32).floor().max(0.0) as u32).min(w);
    let x1 = (((centre.x + radius) * w as f32).ceil().max(0.0) as u32).min(w);
    let y0 = (((centre.y - radius) * h as f32).floor().max(0.0) as u32).min(h);
    let y1 = (((centre.y + radius) * h as f32).ceil().max(0.0) as u32).min(h);
    let mut changed = 0;
    for y in y0..y1 {
        for x in x0..x1 {
            let uv = Vec2::new((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
            let d2 = (uv - centre).length_squared() / (radius * radius);
            if d2 >= 1.0 {
                continue;
            }
            let a = strength.clamp(0.0, 1.0) * (1.0 - d2) * (1.0 - d2);
            if let Some(texel) = map.texel_mut(x, y) {
                for (t, &v) in texel.iter_mut().zip(value) {
                    *t += (v - *t) * a;
                }
                changed += 1;
            }
        }
    }
    Ok(changed)
}

/// Adds (`add`) or removes the triangles of `surface` whose centroid is inside the brush to or from
/// the scalp mask `mask` (kept sorted, without duplicates). Returns the triangles that changed.
pub fn edit_hairline(mask: &mut Vec<u32>, surface: &HairSurface, brush: &Brush, add: bool) -> Result<u64> {
    brush.check()?;
    mask.sort_unstable();
    mask.dedup();
    let r2 = brush.radius * brush.radius;
    let under: Vec<u32> = triangles_where_centroid(surface, |c| (c - brush.centre).length_squared() < r2);
    let before = mask.len();
    if add {
        mask.extend_from_slice(&under);
        mask.sort_unstable();
        mask.dedup();
        Ok((mask.len() - before) as u64)
    } else {
        mask.retain(|t| under.binary_search(t).is_err());
        Ok((before - mask.len()) as u64)
    }
}

/// The triangles of `surface` whose centroid satisfies `keep`, ascending.
fn triangles_where_centroid(surface: &HairSurface, mut keep: impl FnMut(Vec3) -> bool) -> Vec<u32> {
    (0..surface.triangles().len() as u32)
        .filter(|&t| surface.corners(t).is_some_and(|[a, b, c]| keep((a + b + c) * (1.0 / 3.0))))
        .collect()
}

/// Places the parting along a stroke: through `point` on the skin with surface normal `normal`,
/// running along `along`; the parting plane contains the stroke and the normal. `width` and
/// `strength` go to the parameters unchanged (checked against their ranges).
pub fn set_parting(
    params: &mut GroomParams,
    point: Vec3,
    normal: Vec3,
    along: Vec3,
    width: f32,
    strength: f32,
) -> Result<()> {
    let plane =
        along.cross(normal).try_normalize().ok_or(GroomToolError::Invalid("parting stroke parallel to the normal"))?;
    let mut next = *params;
    next.set_parting(Parting { point, normal: plane, width, strength });
    next.check()?;
    *params = next;
    Ok(())
}

/// Sets cowlick `index` (0..4) at `centre` with `radius` and `strength` (checked against the
/// parameter ranges).
pub fn place_cowlick(params: &mut GroomParams, index: usize, centre: Vec3, radius: f32, strength: f32) -> Result<()> {
    if index >= MAX_COWLICKS {
        return Err(GroomToolError::Invalid("cowlick index out of range"));
    }
    let mut next = *params;
    next.set_cowlick(index, Cowlick { centre, radius, strength });
    next.check()?;
    *params = next;
    Ok(())
}

/// Mirrors across the `x = 0` plane: removes the guides whose root is on the destination side
/// (`x < 0` when `positive_source`, else `x > 0`) and adds the mirror image of every source-side
/// guide, rooted at the skin point closest to the mirrored root. Guides within `1e-4` m of the
/// plane are left as they are. Returns the new guides.
pub fn mirror(
    guides: &mut Guides,
    surface: &HairSurface,
    grid: &TriangleGrid,
    positive_source: bool,
) -> Result<EditReport> {
    const PLANE: f32 = 1e-4;
    let side = if positive_source { 1.0 } else { -1.0 };
    let mut sources = Vec::new();
    for i in 0..guides.len() {
        let Some(frame) = guides.frame(i, surface) else { continue };
        if frame.origin.x * side > PLANE {
            sources.push(guides.world_points(i, &frame));
        }
    }
    let root_x: Vec<Option<f32>> = (0..guides.len()).map(|i| guides.frame(i, surface).map(|f| f.origin.x)).collect();
    guides.retain(|i| root_x.get(i).copied().flatten().is_none_or(|x| x * side >= -PLANE));
    let mut report = EditReport::default();
    for world in sources {
        let flipped: Vec<Vec3> = world.iter().map(|p| Vec3::new(-p.x, p.y, p.z)).collect();
        let Some(hit) = grid.closest(flipped[0]) else { continue };
        let Some(frame) = surface.frame(hit.triangle, hit.bary) else { continue };
        let shift = frame.origin - flipped[0];
        let local: Vec<Vec3> = flipped.iter().map(|&w| frame.to_local(w + shift)).collect();
        let index = guides.push(hit.triangle, hit.bary, &local)?;
        report.guides.push(index as u32);
        report.elements += guides.points() as u64;
    }
    Ok(report)
}
