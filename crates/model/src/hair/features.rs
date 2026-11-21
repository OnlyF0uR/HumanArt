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
//! **Lashes** ([`lash_groom`]): roots unevenly spaced along the lid margin in irregular rows;
//! each lash leaves the lid forward, tilted away from the eye opening, the outer ones fanning
//! toward the outer corner; it is longest past the middle of the lid, curls as one smooth arc by
//! its own amount, gathers with its neighbours into clumps, and some are short (still growing).
//! A lash that would touch the eyeball or the lid is lifted away whole, never bent point by
//! point.
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

/// Control points per brow hair (lashes: [`LASH_POINTS`]).
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
///
/// The defaults follow the anatomy texts' lash counts and lengths (upper lid about 90 to 160
/// lashes in 3 to 6 irregular rows, 8 to 12 mm long; lower lid about 75 to 80, 6 to 8 mm); the
/// angles, the curl, the fan and the share of growing lashes are styling choices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LashSpec {
    /// Upper lid (lashes turn up) or lower lid (lashes turn down).
    pub upper: bool,
    /// Lashes on the lid.
    pub count: usize,
    /// Rows the lashes are spread over (1..=6), each about [`LASH_ROW_STEP`] further behind the
    /// margin, irregular.
    pub rows: u32,
    /// Length of the longest lashes, metres.
    pub length: f32,
    /// Random per-lash length change, ± fraction.
    pub length_variation: f32,
    /// Angle the lash leaves the lid at, above the plane of the lid margin (away from the eye
    /// opening), radians; negative toward the opening (upper lashes leave forward and a little
    /// down, so their bases crowd into a dark line along the lid in front view, then curl up).
    pub tilt: f32,
    /// Total bend of a lash from root to tip, radians.
    pub curl: f32,
    /// Random per-lash curl change, ± fraction.
    pub curl_variation: f32,
    /// How far the lashes of the outer part of the lid sweep toward the outer corner (and the
    /// innermost toward the nose), radians at the ends.
    pub fan: f32,
    /// Share of lashes still growing (a quarter to two thirds of their full length): they crowd
    /// the lash line, which reads darker than the lashes' tips.
    pub growing: f32,
}

impl LashSpec {
    /// Defaults for the upper lid.
    #[must_use]
    pub const fn upper() -> Self {
        Self {
            upper: true,
            count: 150,
            rows: 4,
            length: 0.010_5,
            length_variation: 0.18,
            tilt: -0.2,
            curl: 1.05,
            curl_variation: 0.3,
            fan: 0.75,
            growing: 0.3,
        }
    }

    /// Defaults for the lower lid.
    #[must_use]
    pub const fn lower() -> Self {
        Self {
            upper: false,
            count: 75,
            rows: 2,
            length: 0.006_8,
            length_variation: 0.25,
            tilt: 0.05,
            curl: 0.4,
            curl_variation: 0.35,
            fan: 0.6,
            growing: 0.25,
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
fn feature_params(count: usize, points: u32, thickness: f32, tip_ratio: f32) -> GroomParams {
    GroomParams {
        strand_count: count as f32,
        points: points as f32,
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
    let params = feature_params(guides.len(), FEATURE_POINTS, 0.000_1, 0.35);
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

/// The least distance between a lash root and the eyeball's surface, metres (the margin rests
/// on the globe; the lashes leave from its front edge).
pub const LASH_ROOT_EYE_CLEARANCE: f32 = 0.000_25;

/// The least distance a lash point past the root keeps above the lid skin, metres.
pub const LASH_SKIN_CLEARANCE: f32 = 0.000_1;

/// Lashes per clump (inclusive range); about a quarter of the lashes stand alone between clumps.
pub const LASH_CLUMP: (u32, u32) = (2, 6);

/// Depth between two lash rows, behind the margin along the lid's front surface, metres.
pub const LASH_ROW_STEP: f32 = 0.000_15;

/// Control points per lash: a curled lash needs more than a brow hair.
pub const LASH_POINTS: u32 = 8;

/// Shapes a lash is tried in before it is left out: each lifts it further from the eye and
/// shortens it a little, so lashes at the canthi clear the globe and the other lid whole
/// instead of being bent point by point.
const LASH_ATTEMPTS: u32 = 10;

/// Where the lashes begin along the margin, from the inner canthus (upper, lower lid): the
/// lacrimal part of the lid, medial to the punctum, bears none.
const LASH_START: (f32, f32) = (0.07, 0.12);

/// Where the lashes end along the margin: the last millimetre or two before the outer canthus,
/// where the lids meet, bears none (lashes there ran into the other lid).
const LASH_END: f32 = 0.94;

/// One lash being built: its place along the lid, the lid tangent there, its points.
struct Lash {
    along: f32,
    tangent: Vec3,
    points: Vec<Vec3>,
}

/// Every point of a lash clear of the eyeball and (past the root) the lid skin.
fn lash_clear(points: &[Vec3], skin: &TriangleGrid, eyeball: Option<&Eyeball>) -> bool {
    points.iter().enumerate().all(|(k, &q)| {
        let skin_ok = k == 0 || skin.closest(q).is_some_and(|c| c.signed_distance >= LASH_SKIN_CLEARANCE);
        let need = if k == 0 { LASH_ROOT_EYE_CLEARANCE } else { LASH_EYE_CLEARANCE };
        let eye_ok = eyeball.is_none_or(|e| e.grid.closest(q).is_none_or(|c| c.signed_distance >= need));
        skin_ok && eye_ok
    })
}

/// The points of a lash from `root` leaving along `start`, bending toward `away` by `curl` over
/// `length` (evenly, so the lash is one smooth arc).
fn lash_points(root: Vec3, start: Vec3, away: Vec3, tangent: Vec3, length: f32, curl: f32) -> Vec<Vec3> {
    let bend_axis = normalize_or(start.cross(away), tangent);
    let step = length / (LASH_POINTS - 1) as f32;
    let mut q = root;
    let mut out = Vec::with_capacity(LASH_POINTS as usize);
    out.push(q);
    for k in 1..LASH_POINTS {
        let t = (k as f32 - 0.5) / (LASH_POINTS - 1) as f32;
        q += Quat::from_axis_angle(bend_axis, curl * t).mul_vec3(start) * step;
        out.push(q);
    }
    out
}

/// The lashes of one lid along `margin` (world points from the inner to the outer canthus).
/// `opening` is the world direction from the lid margin into the eye opening (down for an upper
/// lid, up for a lower lid); lashes leave away from it.
///
/// Roots: unevenly spaced along the margin (strongly jittered strata, sparser toward the inner
/// corner, none on the lacrimal part), in [`LashSpec::rows`] irregular rows behind the margin.
/// Direction: with an `eyeball`, a lash leaves forward (halfway between the eyeball's radius at
/// the root and through the middle of the lid, without their parts along the lid and along the
/// lid's up/down direction) tilted [`LashSpec::tilt`] away from the opening; without one, along
/// the skin normal with the same parts removed. The outer lashes fan toward the outer corner,
/// each lash turns a little at random, and it curls away from the eye as one smooth arc. A share
/// of lashes is still growing (short). A lash whose shape would come closer than
/// [`LASH_EYE_CLEARANCE`] to the eyeball or [`LASH_SKIN_CLEARANCE`] to the skin is tried again
/// turned further toward the skin's normal at its root, a little higher and shorter, and left
/// out when no try clears (no point is ever pushed, so no lash is crumpled). Runs of
/// [`LASH_CLUMP`] neighbours then gather toward their tips, where that keeps them clear.
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
    if !(1..=6).contains(&spec.rows) {
        return Err(HairError::limit("lash rows", 6, u64::from(spec.rows)));
    }
    let line = Line::new(margin)?;
    let mut rng = Rng::new(seed);
    let away = -normalize_or(opening, if spec.upper { -Vec3::Y } else { Vec3::Y });
    // The eye's forward: the eyeball's radius through the middle of the lid, without its part
    // along the lid's up/down direction. Lashes leave between it and their own radius, so the
    // ones at the inner corner do not run into the side of the nose.
    let gaze = eyeball.map(|eye| normalize_or((line.at(0.5).0 - eye.centre).reject_from(away), Vec3::Z));
    let start_at = if spec.upper { LASH_START.0 } else { LASH_START.1 };
    // 1. Places along the lid: jittered strata (gaps and pairs), warped so the lashes thin out
    //    toward the inner corner; in order along the lid.
    let mut places: Vec<f32> = (0..spec.count)
        .map(|i| {
            let u = ((i as f32 + rng.range_f32(-0.4, 1.4)) / spec.count as f32).clamp(0.0, 1.0);
            start_at + (LASH_END - start_at) * (1.0 - (1.0 - u).powf(1.35))
        })
        .collect();
    places.sort_by(f32::total_cmp);
    // 2. One lash per place, in the first shape that clears the eye and the skin.
    let mut lashes: Vec<Lash> = Vec::with_capacity(spec.count);
    for s in places {
        let (p, tangent) = line.at(s);
        let row = rng.below(u64::from(spec.rows)) as f32;
        let depth = (LASH_ROW_STEP * (row + rng.range_f32(-0.35, 0.35))).max(0.0);
        let growing = rng.f32() < spec.growing;
        let grow = if growing { rng.range_f32(0.25, 0.65) } else { 1.0 };
        let length_jitter = 1.0 + spec.length_variation * rng.range_f32(-1.0, 1.0);
        let curl_jitter = 1.0 + spec.curl_variation * rng.range_f32(-1.0, 1.0);
        let (turn, lift_jitter) = (rng.range_f32(-0.14, 0.14), rng.range_f32(-0.1, 0.1));
        let Some(on_margin) = grid.closest(p) else { continue };
        let n = grid.face_normal(on_margin.triangle);
        let back = normalize_or(away.reject_from(n), away);
        let Some(hit) = grid.closest(p + back * depth) else { continue };
        let root = hit.point;
        let hit_normal = grid.face_normal(hit.triangle);
        let forward = match eyeball {
            Some(eye) => {
                let radial = normalize_or((root - eye.centre).reject_from(away), n);
                normalize_or((radial + gaze.unwrap_or(radial)).reject_from(tangent), n)
            }
            None => normalize_or(n.reject_from(away).reject_from(tangent), n),
        };
        // Short and sparse at the inner corner, longest past the middle, a little shorter at
        // the outer corner.
        let profile = 0.5 + 0.5 * smooth(0.0, 0.45, s) - 0.3 * smooth(0.75, 1.0, s);
        let length = spec.length * profile * length_jitter * grow;
        let curl = spec.curl * curl_jitter;
        // The fan: the outer third sweeps toward the outer corner, the innermost lashes a
        // little toward the nose.
        let fan = spec.fan * (smooth(0.45, 1.0, s) - 0.3 * (1.0 - smooth(0.1, 0.35, s))) + turn;
        let mut found = None;
        for attempt in 0..LASH_ATTEMPTS {
            // Each try turns the lash further toward the skin's normal at the root (straight out
            // of the lid) and lifts it a little further from the eye.
            let k = attempt as f32;
            let tilt = spec.tilt + lift_jitter + 0.05 * k;
            let start = forward * tilt.cos() + away * tilt.sin();
            let start = normalize_or(start * fan.cos() + tangent * fan.sin(), forward);
            let start = normalize_or(start + hit_normal * (0.25 * k), start);
            let shortened = length * (1.0 - 0.07 * k);
            let points = lash_points(root, start, away, tangent, shortened, curl);
            if lash_clear(&points, grid, eyeball) {
                found = Some(points);
                break;
            }
        }
        if let Some(points) = found {
            lashes.push(Lash { along: s, tangent, points });
        }
    }
    // 3. Clumps: runs of neighbours whose points gather toward the run's mean near the tips;
    //    a lash the gathering would bring too close to the eye or the skin stays as it was.
    let mut first = 0;
    while first < lashes.len() {
        let size = if rng.f32() < 0.25 {
            1
        } else {
            LASH_CLUMP.0 + rng.below(u64::from(LASH_CLUMP.1 - LASH_CLUMP.0 + 1)) as u32
        };
        let end = (first + size as usize).min(lashes.len());
        let strength = rng.range_f32(0.35, 0.8);
        let members = &mut lashes[first..end];
        if members.len() > 1 {
            let inv = 1.0 / members.len() as f32;
            let mean: Vec<Vec3> = (0..LASH_POINTS as usize)
                .map(|k| members.iter().fold(Vec3::ZERO, |a, l| a + l.points[k]) * inv)
                .collect();
            for l in members.iter_mut() {
                let gathered: Vec<Vec3> = l
                    .points
                    .iter()
                    .zip(&mean)
                    .enumerate()
                    .map(|(k, (&q, &m))| q.lerp(m, strength * (k as f32 / (LASH_POINTS - 1) as f32).powf(1.5)))
                    .collect();
                if lash_clear(&gathered, grid, eyeball) {
                    l.points = gathered;
                }
            }
        }
        first = end;
    }
    // 4. Bound to the skin.
    let mut guides = Guides::new(LASH_POINTS)?;
    let (mut along, mut axis) = (Vec::with_capacity(lashes.len()), Vec::with_capacity(lashes.len()));
    for lash in lashes {
        if push_hair(&mut guides, surface, grid, &lash.points, lash.tangent, &mut axis)? {
            along.push(lash.along);
        }
    }
    if guides.is_empty() {
        return Err(HairError::invalid("lashes", 0, "no lash could be bound to the skin"));
    }
    // Thick at the root, fine at the tip: the bases crowd into a dark lash line.
    let params = feature_params(guides.len(), LASH_POINTS, 0.000_14, 0.2);
    Ok(FeatureGroom { guides, params, along, axis })
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
        assert!((120..=150).contains(&upper.guides.len()), "{}", upper.guides.len());
        let (mut mid, mut ends) = (Vec::new(), Vec::new());
        for i in 0..upper.guides.len() {
            let w = world(&upper, s, i);
            // Upper lashes rise from root to tip and leave the skin.
            assert!(w[w.len() - 1].y > w[0].y, "lash {i}");
            assert!(grid.closest(w[w.len() - 1]).unwrap().signed_distance > 0.002);
            let a = upper.along[i];
            if (0.4..0.75).contains(&a) {
                mid.push(upper.guides.length(i));
            } else if a < 0.2 {
                ends.push(upper.guides.length(i));
            }
        }
        // Short at the inner corner.
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
