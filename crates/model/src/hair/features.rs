//! Brow and lash generators `[HAIR-04]`.
//!
//! Brows and lashes are grooms whose every hair is authored: each hair is one guide and the
//! groom's layout is [`GroomLayout::one_per_guide`](super::GroomLayout::one_per_guide). The
//! generators take the feature's line on the skin (the brow line from inner to outer end, the lid
//! margin from the inner to the outer canthus, in world space, from landmarks or the proxy head)
//! and bind every root to the closest point of the skin, so the hairs ride the deformed surface
//! through the attach step.
//!
//! **Brows** ([`brow_groom`]): roots in a band around the brow line whose height follows a
//! [`BrowShape`] profile (head, body, tail) and arch; hairs at the head point up, the rest toward
//! the tail, upper hairs angled down and lower hairs up (the herringbone of a real brow), lying
//! at a small lift angle above the skin, each with its own length.
//!
//! **Lashes** ([`lash_groom`]): roots along the lid margin in one or more rows; each lash leaves
//! the lid along the surface normal tilted away from the eye opening, fans toward the outer
//! corner, is longest past the middle of the lid and curls by its own amount.
//!
//! **Lid-following hooks** `[EXP-05]`: lashes follow the lid because their roots are bound to lid
//! triangles. For the lash direction change of a blink, [`FeatureGroom::along`] gives each lash's
//! position along the lid and [`FeatureGroom::axis`] its lid tangent in the root frame;
//! [`lift_lashes`] turns the lashes about it by an angle that the expression system drives.
//!
//! Lengths, counts and angles of the shapes are styling choices for these generators, not
//! measurements of real brows or lashes.

use humanart_core::rng::Rng;
use humanart_math::{Quat, Vec3};

use super::collide::TriangleGrid;
use super::error::{HairError, HairResult};
use super::params::GroomParams;
use super::strands::{Guides, resample_polyline};
use super::surface::{HairSurface, normalize_or};

/// The most hairs one brow or lash row set may have.
pub const MAX_FEATURE_HAIRS: usize = 8192;

/// Control points per brow or lash hair.
pub const FEATURE_POINTS: u32 = 6;

/// A brow or lash groom: one guide per hair.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureGroom {
    /// One guide per hair.
    pub guides: Guides,
    /// Generation parameters (one strand per guide, no clumping, no interpolation).
    pub params: GroomParams,
    /// Per hair, its position along the feature line, 0 at the inner end to 1 at the outer end.
    pub along: Vec<f32>,
    /// Per hair, the unit tangent of the feature line at the root in the root frame (the axis
    /// [`lift_lashes`] turns about).
    pub axis: Vec<Vec3>,
}

/// A brow's band profile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrowShape {
    /// The stable id.
    pub name: &'static str,
    /// Band height at the head (inner end), metres.
    pub head: f32,
    /// Band height at the body (the widest part, a third of the way out), metres.
    pub body: f32,
    /// Band height at the tail, metres.
    pub tail: f32,
    /// Extra lift of the band's middle above the given line, metres.
    pub arch: f32,
}

/// The brow shapes (stable ids).
pub const BROW_SHAPES: &[BrowShape] = &[
    BrowShape { name: "straight", head: 0.009, body: 0.009, tail: 0.004, arch: 0.0 },
    BrowShape { name: "soft_arch", head: 0.009, body: 0.008, tail: 0.003, arch: 0.003 },
    BrowShape { name: "high_arch", head: 0.008, body: 0.007, tail: 0.003, arch: 0.007 },
    BrowShape { name: "thick", head: 0.013, body: 0.012, tail: 0.006, arch: 0.002 },
    BrowShape { name: "thin", head: 0.005, body: 0.004, tail: 0.002, arch: 0.003 },
];

/// The brow shape called `name`.
#[must_use]
pub fn brow_shape(name: &str) -> Option<BrowShape> {
    BROW_SHAPES.iter().copied().find(|s| s.name == name)
}

/// Settings of [`brow_groom`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrowSpec {
    /// The band profile.
    pub shape: BrowShape,
    /// Hairs in the brow.
    pub count: usize,
    /// Mean hair length, metres.
    pub length: f32,
    /// Random per-hair length change, ± fraction.
    pub length_variation: f32,
    /// Angle between a hair and the skin at the root, radians.
    pub lift: f32,
}

impl Default for BrowSpec {
    fn default() -> Self {
        Self { shape: BROW_SHAPES[1], count: 520, length: 0.008, length_variation: 0.35, lift: 0.12 }
    }
}

/// Settings of [`lash_groom`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LashSpec {
    /// Upper lid (lashes turn up) or lower lid (lashes turn down).
    pub upper: bool,
    /// Lashes on the lid.
    pub count: usize,
    /// Rows the lashes are spread over (1..=4), each 0.4 mm further behind the margin.
    pub rows: u32,
    /// Length of the longest lashes, metres.
    pub length: f32,
    /// Random per-lash length change, ± fraction.
    pub length_variation: f32,
    /// Angle the lash leaves the lid at, away from the eye opening, radians.
    pub tilt: f32,
    /// Total bend of a lash from root to tip, radians.
    pub curl: f32,
    /// Random per-lash curl change, ± fraction.
    pub curl_variation: f32,
}

impl LashSpec {
    /// Defaults for the upper lid.
    #[must_use]
    pub const fn upper() -> Self {
        Self {
            upper: true,
            count: 120,
            rows: 2,
            length: 0.010,
            length_variation: 0.2,
            tilt: 0.05,
            curl: 0.6,
            curl_variation: 0.3,
        }
    }

    /// Defaults for the lower lid.
    #[must_use]
    pub const fn lower() -> Self {
        Self {
            upper: false,
            count: 55,
            rows: 1,
            length: 0.006,
            length_variation: 0.25,
            tilt: 0.15,
            curl: 0.25,
            curl_variation: 0.3,
        }
    }
}

/// A feature line resampled densely for lookups by arc-length fraction.
struct Line {
    points: Vec<Vec3>,
}

impl Line {
    const SAMPLES: usize = 129;

    fn new(curve: &[Vec3]) -> HairResult<Self> {
        if curve.len() < 2 || curve.len() > 4096 || curve.iter().any(|p| !p.is_finite()) {
            return Err(HairError::invalid("feature line", 0, "needs 2 to 4096 finite points"));
        }
        let mut points = vec![Vec3::ZERO; Self::SAMPLES];
        resample_polyline(curve, &mut points);
        if (points[Self::SAMPLES - 1] - points[0]).length() <= 1e-5 {
            return Err(HairError::invalid("feature line", 0, "has no length"));
        }
        Ok(Self { points })
    }

    /// The point and unit tangent at fraction `s` of the length.
    fn at(&self, s: f32) -> (Vec3, Vec3) {
        let n = Self::SAMPLES - 1;
        let x = s.clamp(0.0, 1.0) * n as f32;
        let i = (x.floor() as usize).min(n - 1);
        let f = x - i as f32;
        let (a, b) = (self.points[i], self.points[i + 1]);
        (a.lerp(b, f), normalize_or(b - a, Vec3::X))
    }
}

/// The parameters of a groom of authored hairs.
fn feature_params(count: usize, thickness: f32, tip_ratio: f32) -> GroomParams {
    GroomParams {
        strand_count: count as f32,
        points: FEATURE_POINTS as f32,
        interpolation: 1.0,
        length_variation: 0.0,
        noise: 0.0,
        curl_radius: 0.0,
        wave_amplitude: 0.0,
        frizz: 0.0,
        flyaway_fraction: 0.0,
        clump_count: 0.0,
        clump_strength: 0.0,
        hairline_softness: 0.0,
        thickness,
        tip_ratio,
        collision_offset: 0.0,
        ..GroomParams::default()
    }
}

fn check_count(count: usize) -> HairResult<()> {
    if count == 0 || count > MAX_FEATURE_HAIRS {
        return Err(HairError::limit("feature hair count", MAX_FEATURE_HAIRS as u64, count as u64));
    }
    Ok(())
}

/// Binds a world point to the skin and adds a hair with world points `world` (first = root).
fn push_hair(
    guides: &mut Guides,
    surface: &HairSurface,
    grid: &TriangleGrid,
    world: &[Vec3],
    tangent: Vec3,
    axis: &mut Vec<Vec3>,
) -> HairResult<bool> {
    let Some(hit) = world.first().and_then(|&p| grid.closest(p)) else { return Ok(false) };
    let Some(frame) = surface.frame(hit.triangle, hit.bary) else { return Ok(false) };
    let shift = frame.origin - world[0];
    let local: Vec<Vec3> = world.iter().map(|&w| frame.to_local(w + shift)).collect();
    guides.push(hit.triangle, hit.bary, &local)?;
    axis.push(normalize_or(frame.dir_to_local(tangent), Vec3::X));
    Ok(true)
}

/// The brow on the skin along `line` (world points from the inner to the outer end). `grid` is
/// the closest-point grid of `surface`'s triangles; `up` the world direction the brow's top
/// faces (`+Y` for an upright head).
pub fn brow_groom(
    surface: &HairSurface,
    grid: &TriangleGrid,
    line: &[Vec3],
    up: Vec3,
    spec: &BrowSpec,
    seed: u64,
) -> HairResult<FeatureGroom> {
    check_count(spec.count)?;
    let line = Line::new(line)?;
    let mut rng = Rng::new(seed);
    let mut guides = Guides::new(FEATURE_POINTS)?;
    let (mut along, mut axis) = (Vec::with_capacity(spec.count), Vec::with_capacity(spec.count));
    let shape = spec.shape;
    for _ in 0..spec.count * 4 {
        if guides.len() == spec.count {
            break;
        }
        // Denser toward the body: s drawn uniformly, the band height carries the shape.
        let s = rng.f32();
        // Denser along the lower edge, feathered toward the top (v = -1 is the lower edge).
        let v = 2.0 * rng.f32().powf(1.4) - 1.0;
        let height = if s < 0.33 {
            shape.head + (shape.body - shape.head) * (s / 0.33)
        } else {
            shape.body + (shape.tail - shape.body) * ((s - 0.33) / 0.67)
        };
        // Thin tails hold fewer hairs: keep a hair with probability height / body height.
        if rng.f32() * shape.head.max(shape.body) > height {
            continue;
        }
        let (p, tangent) = line.at(s);
        let Some(hit) = grid.closest(p) else { continue };
        let n = grid.face_normal(hit.triangle);
        let band_up = normalize_or(up.reject_from(n).reject_from(tangent), up);
        let arch = shape.arch * (std::f32::consts::PI * s).sin();
        let root = p + band_up * (arch + 0.5 * v * height);
        // Direction in the skin plane: up at the head, toward the tail elsewhere, herringbone.
        let toward_tail = smooth(0.0, 0.3, s);
        // The head's hairs lean a little toward the tail too (a fan, not a vertical comb).
        let flat = band_up * (1.0 - toward_tail) + tangent * (toward_tail + 0.35 * (1.0 - toward_tail))
            - band_up * (0.35 * v * toward_tail);
        let flat = normalize_or(flat.reject_from(n), tangent);
        // Short at the head (it would stand above the band as a wisp), full along the body.
        let length = spec.length
            * (1.0 + spec.length_variation * rng.range_f32(-1.0, 1.0))
            * (0.7 + 0.3 * height / shape.body.max(1e-4))
            * (0.6 + 0.4 * toward_tail);
        let lift = spec.lift * rng.range_f32(0.7, 1.3);
        let world: Vec<Vec3> = (0..FEATURE_POINTS)
            .map(|k| {
                let t = k as f32 / (FEATURE_POINTS - 1) as f32;
                // Rises at the lift angle, then lies back toward the skin.
                root + flat * (length * t * lift.cos()) + n * (length * lift.sin() * t * (1.0 - 0.5 * t))
            })
            .collect();
        if push_hair(&mut guides, surface, grid, &world, tangent, &mut axis)? {
            along.push(s);
        }
    }
    if guides.is_empty() {
        return Err(HairError::invalid("brow", 0, "no hair could be bound to the skin"));
    }
    let params = feature_params(guides.len(), 0.000_1, 0.35);
    Ok(FeatureGroom { guides, params, along, axis })
}

fn smooth(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The eyeball behind a lid: the exact closest-point grid of its closed surface (outward face
/// normals) and its centre. [`lash_groom`] lets the lashes leave the lid radially from the
/// centre and keeps every lash point [`LASH_EYE_CLEARANCE`] outside the surface.
#[derive(Clone, Debug)]
pub struct Eyeball {
    /// The eyeball's closed surface.
    pub grid: TriangleGrid,
    /// Its centre (the mean of its vertices), world space.
    pub centre: Vec3,
}

/// The least distance between a lash and the eyeball's surface, metres.
pub const LASH_EYE_CLEARANCE: f32 = 0.000_4;

/// The least distance a lash point past the root keeps above the lid skin, metres.
pub const LASH_SKIN_CLEARANCE: f32 = 0.000_1;

/// Lashes per clump: tips of neighbouring lashes gather in clumps of this many (inclusive range).
pub const LASH_CLUMP: (u32, u32) = (3, 5);

/// The lashes of one lid along `margin` (world points from the inner to the outer canthus).
/// `opening` is the world direction from the lid margin into the eye opening (down for an upper
/// lid, up for a lower lid); lashes leave away from it.
///
/// With an `eyeball`, a lash leaves the margin along the eyeball's radius (rejected from the lid
/// tangent) tilted by [`LashSpec::tilt`] away from the opening; without one, along the skin
/// normal minus its part along the lid's up/down direction. Neighbouring lashes are gathered into
/// clumps of [`LASH_CLUMP`] whose tips converge, then every point past the root is walked out of
/// the eyeball and the skin (keeping each segment's length), so no lash pierces the globe or runs
/// inside the lid where the margin wraps around the eye at the canthi.
pub fn lash_groom(
    surface: &HairSurface,
    grid: &TriangleGrid,
    margin: &[Vec3],
    opening: Vec3,
    eyeball: Option<&Eyeball>,
    spec: &LashSpec,
    seed: u64,
) -> HairResult<FeatureGroom> {
    check_count(spec.count)?;
    if !(1..=4).contains(&spec.rows) {
        return Err(HairError::limit("lash rows", 4, u64::from(spec.rows)));
    }
    let line = Line::new(margin)?;
    let mut rng = Rng::new(seed);
    let away = -normalize_or(opening, if spec.upper { -Vec3::Y } else { Vec3::Y });
    // 1. One polyline per lash, in order along the lid.
    let mut lashes: Vec<(f32, Vec3, Vec<Vec3>)> = Vec::with_capacity(spec.count);
    for i in 0..spec.count {
        // Stratified along the lid, avoiding the very corners.
        let s = 0.03 + 0.94 * (i as f32 + rng.f32()) / spec.count as f32;
        let row = rng.below(u64::from(spec.rows)) as f32;
        let (p, tangent) = line.at(s);
        let Some(on_margin) = grid.closest(p) else { continue };
        let n = grid.face_normal(on_margin.triangle);
        let back = normalize_or(away.reject_from(n), away);
        let Some(hit) = grid.closest(p + back * (0.000_4 * row)) else { continue };
        let root = hit.point;
        let forward = match eyeball {
            Some(eye) => normalize_or((root - eye.centre).reject_from(tangent), n),
            // The margin's normal turns into the opening; lashes still leave forward.
            None => normalize_or(n.reject_from(away).reject_from(tangent), n),
        };
        // Longest past the middle, short at the inner corner.
        let profile = 0.45 + 0.55 * (std::f32::consts::PI * s.powf(0.8)).sin();
        let length = spec.length * profile * (1.0 + spec.length_variation * rng.range_f32(-1.0, 1.0));
        let curl = spec.curl * (1.0 + spec.curl_variation * rng.range_f32(-1.0, 1.0));
        let fan = tangent * (0.6 * (s - 0.4));
        let start = normalize_or(forward * spec.tilt.cos() + away * spec.tilt.sin() + fan, forward);
        // Curl: the direction turns toward `away` (away from the eye) by `curl` over the length.
        let bend_axis = normalize_or(start.cross(away), tangent);
        let step = length / (FEATURE_POINTS - 1) as f32;
        let mut q = root;
        let mut world = Vec::with_capacity(FEATURE_POINTS as usize);
        world.push(q);
        for k in 1..FEATURE_POINTS {
            let t = (k as f32 - 0.5) / (FEATURE_POINTS - 1) as f32;
            let d = Quat::from_axis_angle(bend_axis, curl * t).mul_vec3(start);
            q += d * step;
            world.push(q);
        }
        lashes.push((s, tangent, world));
    }
    // 2. Clumps: runs of 3..=5 neighbours whose points converge on the run's mean toward the tip.
    let mut first = 0;
    while first < lashes.len() {
        let size = LASH_CLUMP.0 + rng.below(u64::from(LASH_CLUMP.1 - LASH_CLUMP.0 + 1)) as u32;
        let end = (first + size as usize).min(lashes.len());
        let strength = rng.range_f32(0.5, 0.85);
        let members = &mut lashes[first..end];
        let inv = 1.0 / members.len() as f32;
        for k in 1..FEATURE_POINTS as usize {
            let mean = members.iter().fold(Vec3::ZERO, |a, l| a + l.2[k]) * inv;
            let w = strength * (k as f32 / (FEATURE_POINTS - 1) as f32).powf(1.5);
            for l in members.iter_mut() {
                l.2[k] = l.2[k].lerp(mean, w);
            }
        }
        first = end;
    }
    // 3. Out of the eyeball and the skin, then bound to the skin.
    let mut guides = Guides::new(FEATURE_POINTS)?;
    let (mut along, mut axis) = (Vec::with_capacity(lashes.len()), Vec::with_capacity(lashes.len()));
    for (s, tangent, mut world) in lashes {
        keep_outside(&mut world, grid, eyeball);
        if push_hair(&mut guides, surface, grid, &world, tangent, &mut axis)? {
            along.push(s);
        }
    }
    if guides.is_empty() {
        return Err(HairError::invalid("lashes", 0, "no lash could be bound to the skin"));
    }
    let params = feature_params(guides.len(), 0.000_12, 0.25);
    Ok(FeatureGroom { guides, params, along, axis })
}

/// Walks the points of a lash past its root out of `eyeball` (to [`LASH_EYE_CLEARANCE`]) and the
/// `skin` (to [`LASH_SKIN_CLEARANCE`]), each segment keeping its length: a point too close is
/// moved to the clearance along the outward normal of the closest surface triangle, then put back
/// at its segment's length from the (already moved) point before it.
fn keep_outside(points: &mut [Vec3], skin: &TriangleGrid, eyeball: Option<&Eyeball>) {
    let lengths: Vec<f32> = points.windows(2).map(|w| (w[1] - w[0]).length()).collect();
    for k in 1..points.len() {
        let prev = points[k - 1];
        let length = lengths[k - 1];
        let mut q = points[k];
        // A few rounds: a push out of one surface (or the length fix) can move the point toward
        // the other or back into a fold of the lid at the canthus.
        for _ in 0..8 {
            let mut moved = false;
            let surfaces = [(Some(skin), LASH_SKIN_CLEARANCE), (eyeball.map(|e| &e.grid), LASH_EYE_CLEARANCE)];
            for (grid, clearance) in surfaces {
                let Some(hit) = grid.and_then(|g| g.closest(q)) else { continue };
                if hit.signed_distance < clearance {
                    let normal = grid.map_or(Vec3::ZERO, |g| g.face_normal(hit.triangle));
                    q = hit.point + normal * (clearance * 1.05);
                    moved = true;
                }
            }
            q = prev + normalize_or(q - prev, points[k] - prev) * length;
            if !moved {
                break;
            }
        }
        // The eyeball last and without the length fix: a lash may come out a little longer, but
        // never inside the globe.
        if let Some(eye) = eyeball
            && let Some(hit) = eye.grid.closest(q)
            && hit.signed_distance < LASH_EYE_CLEARANCE
        {
            q = hit.point + eye.grid.face_normal(hit.triangle) * (LASH_EYE_CLEARANCE * 1.05);
        }
        points[k] = q;
    }
}

/// The lid-following hook `[EXP-05]`: turns every lash of `groom` about its lid tangent by
/// `angle` radians (positive curls away from the eye) weighted by `sin(π along)` (most in the
/// middle of the lid), in place.
pub fn lift_lashes(groom: &mut FeatureGroom, angle: f32) {
    if !angle.is_finite() {
        return;
    }
    let n = groom.guides.len().min(groom.along.len()).min(groom.axis.len());
    for i in 0..n {
        let w = (std::f32::consts::PI * groom.along[i].clamp(0.0, 1.0)).sin();
        let q = Quat::from_axis_angle(groom.axis[i], angle * w);
        for o in groom.guides.offsets_mut(i) {
            *o = q.mul_vec3(*o);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::collide::penetration;
    use crate::hair::surface::ProxyHead;

    fn fixture() -> (ProxyHead, TriangleGrid) {
        let head = ProxyHead::new(96).unwrap();
        let grid = TriangleGrid::new(head.surface.positions(), head.surface.triangles(), 0.01).unwrap();
        (head, grid)
    }

    fn world(g: &FeatureGroom, s: &HairSurface, i: usize) -> Vec<Vec3> {
        g.guides.world_points(i, &g.guides.frame(i, s).unwrap())
    }

    #[test]
    fn hair_04_brows_follow_the_line_and_lie_on_the_skin() {
        let (head, grid) = fixture();
        let s = &head.surface;
        let line = head.brow_curve(1.0);
        let brow = brow_groom(s, &grid, &line, Vec3::Y, &BrowSpec::default(), 7).unwrap();
        assert!(brow.guides.len() > 300, "{}", brow.guides.len());
        assert_eq!(brow.along.len(), brow.guides.len());
        assert_eq!(brow.axis.len(), brow.guides.len());
        let mut pts = Vec::new();
        let (mut head_up, mut tail_lateral) = (0, 0);
        let (mut heads, mut tails) = (0, 0);
        for i in 0..brow.guides.len() {
            let w = world(&brow, s, i);
            // Roots within the band around the line.
            let d = line.iter().map(|&p| (p - w[0]).length()).fold(f32::INFINITY, f32::min);
            assert!(d < 0.02, "{d}");
            let dir = normalize_or(w[w.len() - 1] - w[0], Vec3::ZERO);
            if brow.along[i] < 0.08 {
                heads += 1;
                head_up += usize::from(dir.y > dir.x.abs());
            } else if brow.along[i] > 0.6 {
                tails += 1;
                tail_lateral += usize::from(dir.x > dir.y.abs());
            }
            // Short hairs, close to the skin.
            assert!(brow.guides.length(i) < 0.015);
            let tip = grid.closest(w[w.len() - 1]).unwrap();
            assert!(tip.signed_distance.abs() < 0.004, "{}", tip.signed_distance);
            pts.extend(w);
        }
        assert!(head_up * 10 > heads * 8, "{head_up}/{heads}");
        assert!(tail_lateral * 10 > tails * 8, "{tail_lateral}/{tails}");
        assert_eq!(penetration(&grid, None, &pts, 0.000_5).inside, 0);
        // Deterministic; shapes differ.
        assert_eq!(brow, brow_groom(s, &grid, &line, Vec3::Y, &BrowSpec::default(), 7).unwrap());
        let thin = BrowSpec { shape: brow_shape("thin").unwrap(), ..BrowSpec::default() };
        assert_ne!(brow.guides, brow_groom(s, &grid, &line, Vec3::Y, &thin, 7).unwrap().guides);
    }

    #[test]
    fn hair_04_lashes_curl_away_from_the_eye_and_follow_the_lid() {
        let (head, grid) = fixture();
        let s = &head.surface;
        let margin = head.lid_curve(-1.0);
        let mut upper = lash_groom(s, &grid, &margin, -Vec3::Y, None, &LashSpec::upper(), 3).unwrap();
        assert_eq!(upper.guides.len(), 120);
        let (mut mid, mut ends) = (Vec::new(), Vec::new());
        for i in 0..upper.guides.len() {
            let w = world(&upper, s, i);
            // Upper lashes rise from root to tip and leave the skin.
            assert!(w[w.len() - 1].y > w[0].y, "lash {i}");
            assert!(grid.closest(w[w.len() - 1]).unwrap().signed_distance > 0.002);
            let a = upper.along[i];
            if (0.4..0.75).contains(&a) {
                mid.push(upper.guides.length(i));
            } else if !(0.1..0.9).contains(&a) {
                ends.push(upper.guides.length(i));
            }
        }
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        assert!(mean(&mid) > 1.2 * mean(&ends), "{} {}", mean(&mid), mean(&ends));
        let lower = lash_groom(s, &grid, &margin, Vec3::Y, None, &LashSpec::lower(), 3).unwrap();
        let w = world(&lower, s, lower.guides.len() / 2);
        assert!(w[w.len() - 1].y < w[0].y);
        // The lid hook turns the tips, most in the middle; zero angle is the identity.
        let before = upper.clone();
        lift_lashes(&mut upper, 0.0);
        assert_eq!(before, upper);
        lift_lashes(&mut upper, 0.3);
        let moved = |i: usize| (world(&upper, s, i)[5] - world(&before, s, i)[5]).length();
        let middle = (0..upper.guides.len())
            .min_by(|&a, &b| (upper.along[a] - 0.5).abs().total_cmp(&(upper.along[b] - 0.5).abs()));
        let middle = middle.unwrap();
        assert!(moved(middle) > 0.002, "{}", moved(middle));
        assert!(moved(0) < moved(middle));
        // Roots stay bound to the lid.
        assert_eq!(upper.guides.triangles(), before.guides.triangles());
    }

    #[test]
    fn hair_04_bad_input_is_refused() {
        let (head, grid) = fixture();
        let s = &head.surface;
        let line = head.brow_curve(1.0);
        let none = BrowSpec { count: 0, ..BrowSpec::default() };
        assert!(brow_groom(s, &grid, &line, Vec3::Y, &none, 1).is_err());
        assert!(brow_groom(s, &grid, &line[..1], Vec3::Y, &BrowSpec::default(), 1).is_err());
        let nan = [Vec3::splat(f32::NAN), Vec3::ONE];
        assert!(lash_groom(s, &grid, &nan, -Vec3::Y, None, &LashSpec::upper(), 1).is_err());
        let rows = LashSpec { rows: 9, ..LashSpec::upper() };
        assert!(lash_groom(s, &grid, &line, -Vec3::Y, None, &rows, 1).is_err());
        assert!(brow_shape("unibrow").is_none());
    }
}
