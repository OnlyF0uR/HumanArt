//! Procedural wrinkle maps of the template (todo 1.4, `[SKN-30]`): the expression wrinkles of the
//! brow (glabellar frown lines), the forehead (horizontal rows), the crow's feet and the
//! nasolabial folds, authored from the template's own regions and landmarks for the wrinkle
//! drivers of `data/template/expressions/wrinkles.json` (`forehead.raise`, `glabella.L/R`,
//! `crows-feet.L/R`, `nasolabial.L/R`), which fold them with the expressions.
//!
//! **Where.** Each map covers a set of regions on the skin tile they lie on: the forehead for the
//! rows and (one side each) the frown lines, the outer canthus, temple and malar regions for the
//! crow's feet, the cheek, malar, upper-lip and lip-corner regions for the nasolabial fold (the
//! drivers themselves read the strain of their own regions). The rasteriser interpolates the rest
//! position and the region membership of the vertices to every texel of the triangles with at least
//! two corners in the regions, so the creases are drawn in skin millimetres, independent of the UV
//! layout and of the mesh density.
//!
//! **Shape.** A crease profile is `−depth · exp(−d² / (2 w²))` at distance `d` from the crease line
//! (Gaussian of width `w`), with a cavity delta proportional to the depth. The line patterns follow
//! the anatomy of the folds: forehead rows run across the forehead (perpendicular to the frontalis
//! fibres, which run up), the frown lines run up from the glabella beside the midline
//! (perpendicular to the corrugator fibres), the crow's feet fan out from the lateral canthus
//! (perpendicular to the orbicularis oculi fibres) and the nasolabial fold runs from the alar
//! crease past the mouth corner.
//!
//! **Sizes.** Depth, line count and width are the defaults of the registry's wrinkle generator
//! parameters (`skin.features.wrinkles.depth` 0.2 mm, `.count` 4, `.width` 0.8 mm), the line
//! spacing a quarter of the region's extent across the lines, the nasolabial fold twice as deep and
//! wide as a line (it is a fold, not a line). These are **authored** values (a starting look, not
//! measured data); the maps are a template asset to be replaced by scan-derived maps.

use humanart_math::{Vec2, Vec3};

use super::dirty::Rect;
use super::error::SkinError;
use super::wrinkle::{MAX_WRINKLE_SIZE, WrinkleMap};
use crate::template::Template;

/// Crease depth at full fold, mm (registry default of `skin.features.wrinkles.depth`).
pub const LINE_DEPTH_MM: f32 = 0.2;
/// Crease half width (Gaussian σ), mm (registry default of `skin.features.wrinkles.width`).
pub const LINE_WIDTH_MM: f32 = 0.8;
/// Lines per pattern (registry default of `skin.features.wrinkles.count`).
pub const LINE_COUNT: usize = 4;
/// Cavity delta per mm of crease depth at full fold.
pub const CAVITY_PER_MM: f32 = 1.5;
/// Share of the micro-detail contrast a fully stretched region loses.
pub const FLATTEN: f32 = 0.5;

/// The line pattern of one driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pattern {
    /// Horizontal rows across the region (forehead).
    Rows,
    /// Near-vertical lines beside the midline (glabellar frown lines).
    Frown,
    /// A fan of lines radiating from a landmark (crow's feet at the lateral canthus).
    Fan,
    /// One deep fold along a curve between two landmarks (nasolabial fold).
    Fold,
    /// Oblique lines on the side of the upper nose below the nasal root, running down and out
    /// (the "bunny lines" of the nose wrinkler, AU9: todo 1.8, disgust showed none).
    Bunny,
}

/// One authored driver: its name in the driver table, its region, its pattern and the landmarks
/// the pattern is anchored to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AuthoredWrinkle {
    /// The wrinkle driver's stable name.
    pub driver: &'static str,
    /// The regions the map covers (the frown lines lie on the forehead above the driver's brow
    /// region, the crow's feet reach the temple, the nasolabial fold runs along the border of the
    /// cheek and the upper lip).
    pub regions: &'static [&'static str],
    /// Only the texels on this side of the anchor's midline: `1` +X (the character's left), `-1`
    /// −X, `0` both.
    pub side: i8,
    /// The line pattern.
    pub pattern: Pattern,
    /// Anchor landmarks: the fan's origin, the fold's start and end, the frown lines' midline
    /// point (empty for rows).
    pub anchors: &'static [&'static str],
}

/// The template's authored wrinkles.
pub const TEMPLATE_WRINKLES: [AuthoredWrinkle; 9] = [
    AuthoredWrinkle { driver: "forehead.raise", regions: &["forehead"], side: 0, pattern: Pattern::Rows, anchors: &[] },
    AuthoredWrinkle {
        driver: "glabella.L",
        regions: &["forehead"],
        side: 1,
        pattern: Pattern::Frown,
        anchors: &["forehead.glabella"],
    },
    AuthoredWrinkle {
        driver: "glabella.R",
        regions: &["forehead"],
        side: -1,
        pattern: Pattern::Frown,
        anchors: &["forehead.glabella"],
    },
    AuthoredWrinkle {
        driver: "crows-feet.L",
        side: 1,
        regions: &["outer-canthus-L", "temple-L", "malar-L"],
        pattern: Pattern::Fan,
        anchors: &["L.eye.canthus.lateral"],
    },
    AuthoredWrinkle {
        driver: "crows-feet.R",
        side: -1,
        regions: &["outer-canthus-R", "temple-R", "malar-R"],
        pattern: Pattern::Fan,
        anchors: &["R.eye.canthus.lateral"],
    },
    AuthoredWrinkle {
        driver: "nasolabial.L",
        side: 1,
        regions: &["cheek-L", "malar-L", "upper-lip", "lip-corner-L"],
        pattern: Pattern::Fold,
        anchors: &["L.nose.alar-crease", "L.lip.corner"],
    },
    AuthoredWrinkle {
        driver: "nasolabial.R",
        side: -1,
        regions: &["cheek-R", "malar-R", "upper-lip", "lip-corner-R"],
        pattern: Pattern::Fold,
        anchors: &["R.nose.alar-crease", "R.lip.corner"],
    },
    AuthoredWrinkle { driver: "nose.L", side: 1, regions: &["nose"], pattern: Pattern::Bunny, anchors: &["nose.root"] },
    AuthoredWrinkle {
        driver: "nose.R",
        side: -1,
        regions: &["nose"],
        pattern: Pattern::Bunny,
        anchors: &["nose.root"],
    },
];

/// What [`author_template_wrinkles`] could not find.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AuthorReport {
    /// Driver, region or landmark names that are not on the template or in the driver list.
    pub missing: Vec<&'static str>,
}

/// The texels of one region on its tile: rest position (metres) and membership per texel.
struct RegionRaster {
    tile: u16,
    rect: Rect,
    position: Vec<Vec3>,
    weight: Vec<f32>,
}

fn edge(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Tiles that hold no skin (the eye and mouth parts' own tiles).
const NOT_SKIN_TILES: [&str; 2] = ["eyes", "mouth"];

/// Rasterises the triangles with at least two corners in `regions` (with their sub-regions) onto
/// the skin tile most of them lie on, at `size²` texels per tile.
fn raster_region(template: &Template, regions_in: &[u16], size: u32) -> Option<RegionRaster> {
    let mesh = template.mesh();
    let regions = template.regions();
    let positions = mesh.positions();
    let uvs = mesh.uvs();
    let membership: Vec<f32> = (0..positions.len())
        .map(|v| match regions.vertex_regions.get(v) {
            Some(&r) if regions_in.iter().any(|&q| regions.is_within(r, q)) => 1.0,
            _ => 0.0,
        })
        .collect();
    let triangles = mesh.triangulate();
    let face_tiles = mesh.face_tiles();
    let mut tile_counts = std::collections::BTreeMap::<u16, usize>::new();
    let mut chosen = Vec::new();
    for t in 0..triangles.len() {
        let (Some(c), Some(&face)) = (triangles.triangle(t), triangles.triangle_faces.get(t)) else { continue };
        let inside = c.iter().filter(|&&v| membership.get(v as usize).copied().unwrap_or(0.0) > 0.0).count();
        let tile = face_tiles.get(face as usize).copied().unwrap_or(0);
        let skin = template.tiles().get(usize::from(tile)).is_some_and(|t| !NOT_SKIN_TILES.contains(&t.name.as_str()));
        if inside >= 2 && skin {
            *tile_counts.entry(tile).or_insert(0) += 1;
            chosen.push((t, tile));
        }
    }
    let tile = tile_counts.iter().max_by_key(|(_, n)| **n).map(|(t, _)| *t)?;
    let scale = size as f32;
    let mut lo = Vec2::splat(f32::MAX);
    let mut hi = Vec2::splat(f32::MIN);
    for &(t, tt) in &chosen {
        if tt != tile {
            continue;
        }
        for &v in &triangles.triangle(t)? {
            let uv = *uvs.get(v as usize)? * scale;
            lo = lo.min(uv);
            hi = hi.max(uv);
        }
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Clamped to the tile.
    let (x0, y0, x1, y1) = (
        (lo.x - 1.0).floor().clamp(0.0, scale - 1.0) as u32,
        (lo.y - 1.0).floor().clamp(0.0, scale - 1.0) as u32,
        (hi.x + 1.0).ceil().clamp(0.0, scale - 1.0) as u32,
        (hi.y + 1.0).ceil().clamp(0.0, scale - 1.0) as u32,
    );
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let rect = Rect::new(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    let n = rect.w as usize * rect.h as usize;
    let mut position = vec![Vec3::ZERO; n];
    let mut weight = vec![0.0_f32; n];
    for &(t, tt) in &chosen {
        if tt != tile {
            continue;
        }
        let c = triangles.triangle(t)?;
        let p = [uvs.get(c[0] as usize)?, uvs.get(c[1] as usize)?, uvs.get(c[2] as usize)?].map(|uv| *uv * scale);
        let area = edge(p[0], p[1], p[2]);
        if !area.is_finite() || area.abs() < 1e-12 {
            continue;
        }
        let bx0 = p[0].x.min(p[1].x).min(p[2].x).floor().max(rect.x as f32);
        let by0 = p[0].y.min(p[1].y).min(p[2].y).floor().max(rect.y as f32);
        let bx1 = p[0].x.max(p[1].x).max(p[2].x).ceil().min(rect.right() as f32 - 1.0);
        let by1 = p[0].y.max(p[1].y).max(p[2].y).ceil().min(rect.bottom() as f32 - 1.0);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Inside the rectangle.
        let (ix0, iy0, ix1, iy1) = (bx0 as u32, by0 as u32, bx1.max(bx0) as u32, by1.max(by0) as u32);
        let tol = -1e-4 * area.abs();
        let inv = 1.0 / area.abs();
        for y in iy0..=iy1 {
            for x in ix0..=ix1 {
                let q = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let b0 = edge(p[1], p[2], q) * area.signum();
                let b1 = edge(p[2], p[0], q) * area.signum();
                let b2 = edge(p[0], p[1], q) * area.signum();
                if b0 < tol || b1 < tol || b2 < tol {
                    continue;
                }
                let l = [b0 * inv, b1 * inv, b2 * inv];
                let i = (y - rect.y) as usize * rect.w as usize + (x - rect.x) as usize;
                let mut pos = Vec3::ZERO;
                let mut w = 0.0_f32;
                for (k, &v) in c.iter().enumerate() {
                    pos += *positions.get(v as usize)? * l[k];
                    w += membership.get(v as usize).copied().unwrap_or(0.0) * l[k];
                }
                position[i] = pos;
                weight[i] = w.clamp(0.0, 1.0);
            }
        }
    }
    Some(RegionRaster { tile, rect, position, weight })
}

/// The crease profile at distance `d_mm` from a line of depth `depth_mm` and width `width_mm`.
fn crease(d_mm: f32, depth_mm: f32, width_mm: f32) -> f32 {
    -depth_mm * (-(d_mm * d_mm) / (2.0 * width_mm * width_mm)).exp()
}

/// Distance from `p` to the nearest of `count` evenly spaced values over `[lo, hi]` (inner lines:
/// the first and last sit half a spacing in from the ends).
fn nearest_line(p: f32, lo: f32, hi: f32, count: usize) -> f32 {
    let spacing = (hi - lo) / count.max(1) as f32;
    (0..count).map(|k| lo + spacing * (k as f32 + 0.5)).map(|c| (p - c).abs()).fold(f32::MAX, f32::min)
}

/// Distance from `p` to the segment `a`..`b`, and the segment parameter of the closest point.
fn segment_distance(p: Vec3, a: Vec3, b: Vec3) -> (f32, f32) {
    let ab = b - a;
    let len2 = ab.dot(ab);
    let t = if len2 > 0.0 { ((p - a).dot(ab) / len2).clamp(0.0, 1.0) } else { 0.0 };
    ((p - (a + ab * t)).length(), t)
}

/// Authors one driver's height field (mm at full fold) over its raster.
fn author_heights(w: &AuthoredWrinkle, r: &RegionRaster, anchors: &[Vec3]) -> Vec<f32> {
    let covered: Vec<Vec3> = r.position.iter().zip(&r.weight).filter(|(_, w)| **w > 0.0).map(|(p, _)| *p).collect();
    let (lo, hi) =
        covered.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(l, h), p| (l.min(*p), h.max(*p)));
    let mm = 1000.0;
    r.position
        .iter()
        .zip(&r.weight)
        .map(|(&p, &m)| {
            let mid = anchors.first().map_or(0.0, |a| if w.pattern == Pattern::Frown { a.x } else { 0.0 });
            if m <= 0.0 || (w.side > 0 && p.x < mid) || (w.side < 0 && p.x > mid) {
                return 0.0;
            }
            let h = match w.pattern {
                Pattern::Rows => {
                    // Rows across the forehead; fading towards the temples (the lines shorten).
                    let d = nearest_line(p.y, lo.y, hi.y, LINE_COUNT) * mm;
                    let half = 0.5 * (hi.x - lo.x).max(1e-4);
                    let across = 1.0 - ((p.x - lo.x.midpoint(hi.x)).abs() / half).powi(4);
                    crease(d, LINE_DEPTH_MM, LINE_WIDTH_MM) * across.max(0.0)
                }
                Pattern::Frown => {
                    // Two vertical lines per side, 4 and 9 mm from the midline, rising 20 mm from
                    // the glabella.
                    let g = anchors.first().copied().unwrap_or(Vec3::ZERO);
                    let x = (p.x - g.x).abs() * mm;
                    let d = (x - 4.0).abs().min((x - 9.0).abs());
                    let rise = (p.y - g.y) * mm;
                    let along = if (-3.0..=20.0).contains(&rise) { 1.0 - (rise / 20.0).max(0.0).powi(2) } else { 0.0 };
                    crease(d, LINE_DEPTH_MM, LINE_WIDTH_MM) * along
                }
                Pattern::Fan => {
                    // Lines radiating laterally from the canthus between -40 and +40 degrees,
                    // starting 4 mm out, over the next 25 mm.
                    let o = anchors.first().copied().unwrap_or(Vec3::ZERO);
                    let lateral = if o.x >= 0.0 { 1.0 } else { -1.0 };
                    let dx = (p.x - o.x) * lateral;
                    let dy = p.y - o.y;
                    let r_mm = (dx * dx + dy * dy).sqrt() * mm;
                    if dx <= 0.0 || !(4.0..=30.0).contains(&r_mm) {
                        0.0
                    } else {
                        let angle = dy.atan2(dx);
                        let spread = 40_f32.to_radians();
                        let da = nearest_line(angle, -spread, spread, LINE_COUNT);
                        let d = da * r_mm;
                        let fade = ((r_mm - 4.0) / 3.0).min(1.0) * (1.0 - (r_mm - 4.0) / 26.0).max(0.0);
                        crease(d, LINE_DEPTH_MM, LINE_WIDTH_MM) * fade
                    }
                }
                Pattern::Bunny => {
                    // Three lines 2.5 mm apart, tilted 40 degrees down from the horizontal going
                    // out from the midline, from 2 to 13 mm out and 2 to 16 mm below the root.
                    let g = anchors.first().copied().unwrap_or(Vec3::ZERO);
                    let dx = (p.x - g.x).abs() * mm;
                    let dy = (p.y - g.y) * mm;
                    let (sin, cos) = 40_f32.to_radians().sin_cos();
                    // Across the lines (u) and along them (v), mm.
                    let u = dx * sin + dy * cos;
                    let v = dx * cos - dy * sin;
                    if (-16.0..=1.0).contains(&dy) {
                        let along = (1.0 - ((v - 7.5) / 5.5).powi(2)).max(0.0);
                        crease(nearest_line(u, -9.0, -1.5, 3), 3.0 * LINE_DEPTH_MM, 0.8 * LINE_WIDTH_MM) * along
                    } else {
                        0.0
                    }
                }
                Pattern::Fold => {
                    // From the alar crease to 4 mm lateral of and 8 mm below the mouth corner.
                    let a = anchors.first().copied().unwrap_or(Vec3::ZERO);
                    let c = anchors.get(1).copied().unwrap_or(a);
                    let lateral = if a.x >= 0.0 { 1.0 } else { -1.0 };
                    let b = c + Vec3::new(0.004 * lateral, -0.008, 0.0);
                    let (d, t) = segment_distance(p, a, b);
                    let ends = (t * 6.0).min(1.0) * ((1.0 - t) * 4.0).min(1.0);
                    crease(d * mm, 2.0 * LINE_DEPTH_MM, 2.0 * LINE_WIDTH_MM) * ends
                }
            };
            h * m
        })
        .collect()
}

/// Authors the maps of [`TEMPLATE_WRINKLES`] on `template` at `size²` texels per UV tile, for the
/// drivers named `drivers` (driver table order; a map's `driver` is the index of its name there).
/// Returns the maps and what was missing (a missing name skips its map).
///
/// # Errors
/// [`SkinError::Limit`] for a size above [`MAX_WRINKLE_SIZE`] or more drivers than a map indexes.
pub fn author_template_wrinkles(
    template: &Template,
    drivers: &[&str],
    size: u32,
) -> Result<(Vec<WrinkleMap>, AuthorReport), SkinError> {
    if size == 0 || size > MAX_WRINKLE_SIZE {
        return Err(SkinError::limit("wrinkle map size", u64::from(MAX_WRINKLE_SIZE), u64::from(size)));
    }
    let mut report = AuthorReport::default();
    let mut maps = Vec::new();
    let positions = template.mesh().positions();
    for w in &TEMPLATE_WRINKLES {
        let Some(index) = drivers.iter().position(|d| *d == w.driver) else {
            report.missing.push(w.driver);
            continue;
        };
        let driver =
            u16::try_from(index).map_err(|_| SkinError::limit("wrinkle drivers", u64::from(u16::MAX), index as u64))?;
        let mut regions = Vec::with_capacity(w.regions.len());
        for &name in w.regions {
            match template.regions().find(name) {
                Some(r) => regions.push(r),
                None => report.missing.push(name),
            }
        }
        if regions.len() != w.regions.len() {
            continue;
        }
        let mut anchors = Vec::with_capacity(w.anchors.len());
        for &name in w.anchors {
            match template.landmarks().iter().find(|l| l.name == name).and_then(|l| l.position(positions)) {
                Some(p) => anchors.push(p),
                None => report.missing.push(name),
            }
        }
        if anchors.len() != w.anchors.len() {
            continue;
        }
        let Some(raster) = raster_region(template, &regions, size) else {
            report.missing.push(w.driver);
            continue;
        };
        let height = author_heights(w, &raster, &anchors);
        let cavity = height.iter().map(|h| h * CAVITY_PER_MM).collect();
        maps.push(WrinkleMap {
            driver,
            tile: u32::from(raster.tile),
            rect: raster.rect,
            height,
            cavity,
            flatten: FLATTEN,
        });
    }
    Ok((maps, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crease_and_line_helpers() {
        assert!((crease(0.0, 0.2, 0.8) + 0.2).abs() < 1e-6);
        assert!(crease(3.0, 0.2, 0.8).abs() < 0.001);
        assert!((nearest_line(0.125, 0.0, 1.0, 4) - 0.0).abs() < 1e-6);
        assert!((nearest_line(0.25, 0.0, 1.0, 4) - 0.125).abs() < 1e-6);
        let (d, t) = segment_distance(Vec3::new(0.5, 1.0, 0.0), Vec3::ZERO, Vec3::X);
        assert!((d - 1.0).abs() < 1e-6 && (t - 0.5).abs() < 1e-6);
    }

    /// todo 1.4 `[SKN-30]`: the generated template gets one map per authored driver, each on its
    /// region's tile, with creases (negative heights) of the authored depth.
    #[test]
    fn todo_1_4_wrinkle_maps_are_authored_on_the_generated_template() {
        let generated =
            crate::generate::build::template(&crate::generate::build::Options { lods: false, ..Default::default() })
                .unwrap();
        let template = &generated.template;
        let names: Vec<&str> = TEMPLATE_WRINKLES.iter().map(|w| w.driver).collect();
        let (maps, report) = author_template_wrinkles(template, &names, 1024).unwrap();
        assert!(report.missing.is_empty(), "{report:?}");
        assert_eq!(maps.len(), TEMPLATE_WRINKLES.len());
        for (m, w) in maps.iter().zip(&TEMPLATE_WRINKLES) {
            let deepest = m.height.iter().copied().fold(0.0_f32, f32::min);
            let creased = m.height.iter().filter(|h| **h < -0.05).count();
            println!(
                "[todo 1.4] {}: tile {} rect {:?} deepest {deepest:.3} mm, {creased} creased texels",
                w.driver, m.tile, m.rect
            );
        }
        for (m, w) in maps.iter().zip(&TEMPLATE_WRINKLES) {
            let deepest = m.height.iter().copied().fold(0.0_f32, f32::min);
            let creased = m.height.iter().filter(|h| **h < -0.05).count();
            assert!(deepest < -0.1, "{} has creases", w.driver);
            assert!(creased > 20, "{} has visible creases", w.driver);
            // The deepest creases are the nose lines (three line depths).
            assert!(m.height.iter().all(|h| *h <= 0.0 && *h >= -3.0 * LINE_DEPTH_MM - 1e-6));
        }
        assert!(author_template_wrinkles(template, &names, MAX_WRINKLE_SIZE + 1).is_err());
        let (none, report) = author_template_wrinkles(template, &[], 64).unwrap();
        assert!(none.is_empty() && report.missing.len() == TEMPLATE_WRINKLES.len());
    }
}
