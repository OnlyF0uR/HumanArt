//! Preset grooms from guide templates `[HAIR-03]`: short crop, fade, long straight, wavy, curly,
//! coily, braided and bald with stubble.
//!
//! A preset is a set of guides grown on a scalp mask by a template, plus the [`GroomParams`] the
//! style needs (length, curl, wave, frizz, clumps, interpolation). Guides are grown by
//! [`grow_guide`]: from the root along the surface normal, turning toward a comb direction and
//! falling under a gravity weight, and pushed out of the head by the distance grid when one is
//! given. The lengths and modifier values are styling choices made for these presets, not
//! measurements of real hair.
//!
//! Everything is deterministic for a given seed: roots are drawn from
//! [`humanart_core::rng::Rng`], seeded by the caller.

use humanart_core::rng::Rng;
use humanart_math::Vec3;

use super::collide::SdfGrid;
use super::error::{HairError, HairResult};
use super::params::GroomParams;
use super::placement::{PartLine, ScalpMask};
use super::strands::{Guides, MAX_GUIDE_POINTS, MAX_GUIDES};
use super::surface::{HairSurface, RootFrame, normalize_or};

/// A preset groom's id and one-line description.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresetSpec {
    /// The stable id.
    pub name: &'static str,
    /// What it looks like.
    pub doc: &'static str,
}

/// Every preset groom (stable ids, never renamed).
pub const PRESETS: &[PresetSpec] = &[
    PresetSpec { name: "short_crop", doc: "Short hair, about 3 cm, combed back." },
    PresetSpec { name: "fade", doc: "Clipped sides and back fading into about 4 cm on top." },
    PresetSpec { name: "long_straight", doc: "Straight hair about 35 cm long falling to the shoulders." },
    PresetSpec { name: "wavy", doc: "Hair about 25 cm long with a loose wave." },
    PresetSpec { name: "curly", doc: "Hair about 18 cm long in 6 mm curls." },
    PresetSpec { name: "coily", doc: "Hair about 8 cm long in tight 2 mm coils." },
    PresetSpec { name: "braided", doc: "Hair gathered at the nape into one three-strand braid." },
    PresetSpec { name: "bald_stubble", doc: "A shaved scalp with 1.5 mm stubble." },
    PresetSpec { name: "medium_straight", doc: "Straight hair about 18 cm long, to the jaw and the nape." },
    PresetSpec { name: "long_wavy", doc: "Hair about 40 cm long with a loose wave, past the shoulders." },
    PresetSpec { name: "afro", doc: "Tightly coiled hair about 12 cm long standing out in a rounded volume." },
];

/// A preset's guides and parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct PresetGroom {
    /// The preset's id.
    pub name: &'static str,
    /// The guides.
    pub guides: Guides,
    /// The generation parameters.
    pub params: GroomParams,
    /// The triangles the groom grows on: the mask to place its strands with.
    pub area: Vec<u32>,
    /// The parting combed into the guides, if the style has one: place the strands with
    /// [`place_parted`](super::placement::place_parted) so they do not blend across it.
    pub part: Option<PartLine>,
}

/// How [`grow_guide`] shapes a guide.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GrowSpec {
    /// The guide length in metres.
    pub length: f32,
    /// The world direction the guide is combed toward (projected onto the root's tangent plane).
    pub comb: Vec3,
    /// How strongly the guide turns toward the comb direction, per unit length (1/m).
    pub comb_rate: f32,
    /// How strongly it falls toward −Y, per unit length (1/m), on average: the weight grows
    /// linearly from 0 at the root to twice this at the tip, so hair first follows the comb over
    /// the scalp and then falls.
    pub gravity: f32,
    /// How far above the skin the distance grid keeps the points.
    pub offset: f32,
}

/// Grows one guide from `frame` with `points` points (2..=64), in world space. With `sdf`, every
/// point is pushed out of the head to at least `spec.offset` above the surface.
#[must_use]
pub fn grow_guide(frame: &RootFrame, spec: &GrowSpec, points: u32, sdf: Option<&SdfGrid>) -> Vec<Vec3> {
    let points = points.clamp(2, MAX_GUIDE_POINTS) as usize;
    let step = spec.length.max(0.0) / (points - 1) as f32;
    let comb = spec.comb.reject_from(frame.n);
    let comb = normalize_or(comb, frame.b);
    let mut dir = frame.n;
    let mut p = frame.origin;
    let mut out = Vec::with_capacity(points);
    out.push(p);
    for k in 1..points {
        let ramp = 2.0 * k as f32 / (points - 1) as f32;
        let pull = comb * (spec.comb_rate * step) + (-Vec3::Y) * (spec.gravity * ramp * step);
        dir = normalize_or(dir + pull, dir);
        let mut next = p + dir * step;
        if let Some(grid) = sdf {
            let (d, g) = grid.sample(next);
            if d < spec.offset {
                next += normalize_or(g, frame.n) * (spec.offset - d);
                // Keep the step length so the guide length stays close to the target.
                next = p + normalize_or(next - p, dir) * step;
                let (d2, g2) = grid.sample(next);
                if d2 < spec.offset {
                    next += normalize_or(g2, frame.n) * (spec.offset - d2);
                }
                dir = normalize_or(next - p, dir);
            }
        }
        p = next;
        out.push(p);
    }
    out
}

/// Distance above the skin over which a falling guide still follows the scalp, metres.
const HUG_DISTANCE: f32 = 0.02;
/// Arc length over which a falling guide rises from the skin to its offset, metres: the hair
/// leaves the scalp at a low angle instead of standing up at the hairline.
const RISE_LENGTH: f32 = 0.025;

/// Grows a guide of a style that lies on the scalp and then falls (`medium_straight`,
/// `long_wavy`, ...), in world space, with `points` points (2..=64). Unlike [`grow_guide`], the
/// comb direction is projected onto the head's tangent plane at every point (from `sdf`'s
/// gradient) and fades out within [`HUG_DISTANCE`] of the skin, so the hair is combed while it
/// lies on the head and then hangs under gravity instead of carrying on in the root's comb
/// direction; the guide leaves the skin at `lift` (the tangent of the angle to the skin) and
/// reaches `spec.offset` over [`RISE_LENGTH`]. Without `sdf` the comb fades with the length.
#[must_use]
pub fn grow_falling_guide(
    frame: &RootFrame,
    spec: &GrowSpec,
    lift: f32,
    points: u32,
    sdf: Option<&SdfGrid>,
) -> Vec<Vec3> {
    let points = points.clamp(2, MAX_GUIDE_POINTS) as usize;
    let step = spec.length.max(0.0) / (points - 1) as f32;
    let lift = if lift.is_finite() { lift.max(0.0) } else { 0.0 };
    let root_comb = normalize_or(spec.comb.reject_from(frame.n), frame.b);
    let mut dir = normalize_or(frame.n * lift + root_comb, frame.n);
    let mut p = frame.origin;
    let mut out = Vec::with_capacity(points);
    out.push(p);
    for k in 1..points {
        let t = k as f32 / (points - 1) as f32;
        let arc = step * k as f32;
        // The head's outward normal at the current point and how close the point is to it.
        let (n, near) = match sdf {
            Some(grid) => {
                let (d, g) = grid.sample(p);
                (normalize_or(g, frame.n), 1.0 - smoothstep(spec.offset, spec.offset + HUG_DISTANCE, d))
            }
            None => (frame.n, 1.0 - t),
        };
        let comb = normalize_or(spec.comb.reject_from(n), dir);
        let pull = comb * (spec.comb_rate * step * near) + (-Vec3::Y) * (spec.gravity * 2.0 * t * step);
        dir = normalize_or(dir + pull, dir);
        // On the head the hair slides along it instead of diving in.
        let inward = dir.dot(n);
        if inward < 0.0 {
            dir = normalize_or(dir - n * (inward * near), dir);
        }
        let mut next = p + dir * step;
        if let Some(grid) = sdf {
            let offset = spec.offset * smoothstep(0.0, RISE_LENGTH, arc);
            let (d, g) = grid.sample(next);
            if d < offset {
                next += normalize_or(g, n) * (offset - d);
                next = p + normalize_or(next - p, dir) * step;
                let (d2, g2) = grid.sample(next);
                if d2 < offset {
                    next += normalize_or(g2, n) * (offset - d2);
                }
                dir = normalize_or(next - p, dir);
            }
        }
        p = next;
        out.push(p);
    }
    out
}

/// The bounding box of a mask's vertices (min, max).
fn mask_bounds(surface: &HairSurface, mask: &ScalpMask) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for &t in mask.triangles() {
        for p in surface.corners(t).into_iter().flatten() {
            lo = lo.min(p);
            hi = hi.max(p);
        }
    }
    (lo, hi)
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The parting of the falling styles on a scalp mask with bounds `lo`..`hi`: a side part above
/// the character's left eye (`side`) or a centre part, over the top of the head from the front
/// hairline to the crown (a styling choice).
fn part_line(lo: Vec3, hi: Vec3, side: bool) -> PartLine {
    let (centre, size) = ((lo + hi) * 0.5, hi - lo);
    let x = if side { centre.x + 0.18 * size.x } else { centre.x };
    PartLine {
        point: Vec3::new(x, hi.y, centre.z),
        normal: Vec3::X,
        top_y: lo.y + 0.55 * size.y,
        // Ends before the crown: a part combed through it left a bare patch there.
        crown_z: centre.z,
    }
}

/// How strongly hair near a parting is combed away from it, relative to the style's comb.
const PART_COMB: f32 = 2.0;

/// The comb away from `part` at `origin`: toward the part's normal on its positive side and
/// away on the other, strongest at the part and gone 8 cm from it, faded out at the region's
/// edges.
fn part_comb(part: &PartLine, origin: Vec3) -> Vec3 {
    if !part.in_region(origin) {
        return Vec3::ZERO;
    }
    let d = part.side(origin);
    let side = if d >= 0.0 { 1.0 } else { -1.0 };
    let near = 1.0 - smoothstep(0.0, 0.08, d.abs());
    let fade =
        smoothstep(part.crown_z, part.crown_z + 0.06, origin.z) * smoothstep(part.top_y, part.top_y + 0.03, origin.y);
    part.normal * (side * PART_COMB * near * fade)
}

/// The afro's volume on a scalp mask with bounds `lo`..`hi`: an ellipsoid (centre, semi-axes)
/// a little above and behind the head's centre, about 7 cm out from the scalp at the sides and on
/// top and less in front; its rounded top keeps the volume from rising into a peak (a styling
/// choice).
fn afro_volume(lo: Vec3, hi: Vec3) -> (Vec3, Vec3) {
    let (centre, size) = ((lo + hi) * 0.5, hi - lo);
    let c = centre + Vec3::new(0.0, 0.12 * size.y, -0.08 * size.z);
    let axes = Vec3::new(0.5 * size.x + 0.075, 0.5 * size.y + 0.04, 0.5 * size.z + 0.065);
    (c, axes)
}

/// The distance from `origin` along unit `d` to the surface of the ellipsoid (`c`, `axes`); 0
/// when `origin` is outside it.
fn to_ellipsoid(origin: Vec3, d: Vec3, c: Vec3, axes: Vec3) -> f32 {
    let inv = Vec3::new(1.0 / axes.x.max(1e-6), 1.0 / axes.y.max(1e-6), 1.0 / axes.z.max(1e-6));
    let (e, dd) = ((origin - c) * inv, d * inv);
    let (a, b, cc) = (dd.dot(dd), 2.0 * e.dot(dd), e.dot(e) - 1.0);
    let disc = b * b - 4.0 * a * cc;
    if cc >= 0.0 || disc < 0.0 || a <= 0.0 {
        return 0.0;
    }
    (-b + disc.sqrt()) / (2.0 * a)
}

/// The three-strand braid: lane `lane` (0..3) at braid parameter `s` metres below the nape, as an
/// offset in the plane facing −Z (x across, z depth) plus the drop.
fn braid_offset(lane: usize, s: f32, width: f32, period: f32) -> Vec3 {
    let phase = std::f32::consts::TAU * (s / period) + std::f32::consts::TAU * lane as f32 / 3.0;
    Vec3::new(width * phase.sin(), -s, -0.5 * width * (2.0 * phase).sin())
}

/// Builds preset `name` on `mask` with `guide_count` guides (1..=[`MAX_GUIDES`]). `sdf` (the
/// head's distance grid) keeps guides outside the head; without it long styles may pass through.
pub fn preset_groom(
    name: &str,
    surface: &HairSurface,
    mask: &ScalpMask,
    guide_count: usize,
    sdf: Option<&SdfGrid>,
    seed: u64,
) -> HairResult<PresetGroom> {
    let spec = PRESETS
        .iter()
        .find(|p| p.name == name)
        .ok_or_else(|| HairError::Param { name: name.chars().take(64).collect(), what: "unknown groom preset" })?;
    if guide_count == 0 || guide_count > MAX_GUIDES {
        return Err(HairError::limit("preset guide count", MAX_GUIDES as u64, guide_count as u64));
    }
    if mask.triangles().is_empty() || mask.area() <= 0.0 {
        return Err(HairError::invalid("groom", 0, "the scalp mask has no area"));
    }
    let (lo, hi) = mask_bounds(surface, mask);
    let centre = (lo + hi) * 0.5;
    let mut params = GroomParams::default();
    let points: u32 = match spec.name {
        "short_crop" | "fade" | "bald_stubble" => 8,
        "braided" => 48,
        // Six or more points per wave period of `wave_guide`.
        "wavy" | "long_wavy" => 32,
        _ => 24,
    };
    let back = Vec3::new(0.0, -0.3, -1.0);
    let spec_name = spec.name;
    let part = match spec.name {
        "medium_straight" | "long_straight" => Some(part_line(lo, hi, true)),
        "wavy" | "long_wavy" => Some(part_line(lo, hi, false)),
        _ => None,
    };
    let afro = afro_volume(lo, hi);
    match spec.name {
        "short_crop" => {
            params.noise = 0.1;
            params.clump_strength = 0.2;
        }
        "fade" => {
            params.noise = 0.1;
            params.clump_strength = 0.15;
            params.hairline_softness = 0.012;
        }
        "long_straight" | "medium_straight" => {
            params.clump_strength = 0.3;
            params.frizz = 0.000_3;
        }
        "wavy" | "long_wavy" => {
            // The wave itself is in the guides (`wave_guide`, in phase across neighbouring guides,
            // so the strands blended from them wave together like real locks); the strands add
            // only a slight wave of their own (out of phase from strand to strand, a stronger one
            // read as crimp at head distance).
            params.wave_amplitude = 0.000_5;
            params.wave_frequency = 9.0;
            params.clump_strength = 0.5;
        }
        "curly" => {
            params.curl_radius = 0.006;
            params.curl_frequency = 60.0;
            params.curl_variation = 0.3;
            params.clump_strength = 0.7;
            params.clump_tip_spread = 0.1;
            params.frizz = 0.001;
        }
        "afro" => {
            // Loose, fluffy coils: little clumping, much frizz, so the volume reads as a soft mass.
            params.curl_radius = 0.003;
            params.curl_frequency = 220.0;
            params.curl_variation = 0.4;
            params.clump_strength = 0.25;
            params.frizz = 0.002_5;
            params.flyaway_fraction = 0.05;
            params.noise = 0.2;
        }
        "coily" => {
            params.curl_radius = 0.002;
            params.curl_frequency = 250.0;
            params.curl_variation = 0.3;
            params.clump_strength = 0.6;
            params.frizz = 0.001_5;
            params.flyaway_fraction = 0.03;
        }
        "braided" => {
            // Each strand follows its nearest guide so the three lanes stay apart.
            params.interpolation = 1.0;
            params.clump_strength = 0.8;
            params.clump_spread = 0.3;
            params.clump_tip_spread = 0.05;
            params.flyaway_fraction = 0.02;
        }
        _ => {
            // bald_stubble
            params.strand_count = 60_000.0;
            params.thickness = 0.000_05;
            params.tip_ratio = 0.8;
            params.length_variation = 0.3;
            params.clump_count = 0.0;
            params.clump_strength = 0.0;
            params.flyaway_fraction = 0.0;
            params.frizz = 0.0;
        }
    }
    params.points = points.min(32) as f32;
    let mut guides = Guides::new(points)?;
    let mut rng = Rng::new(seed);
    // The braid gathers at the nape, a little behind the back of the head.
    let nape = Vec3::new(centre.x, lo.y + 0.3 * (hi.y - lo.y), lo.z - 0.01);
    let lane_width = (hi.x - lo.x) / 6.0;
    let mut attempts = 0usize;
    while guides.len() < guide_count && attempts < guide_count.saturating_mul(8) {
        attempts += 1;
        let Some((t, bary)) = mask.sample_root(&mut rng) else { break };
        let Some(frame) = surface.frame(t, bary) else { continue };
        let jitter = rng.range_f32(0.85, 1.15);
        let world = match spec.name {
            "short_crop" => grow_guide(
                &frame,
                // Combed flat quickly: short hair lies on the scalp instead of standing out.
                &GrowSpec { length: 0.03 * jitter, comb: back, comb_rate: 150.0, gravity: 5.0, offset: 0.002 },
                points,
                sdf,
            ),
            "fade" => {
                let height = (frame.origin.y - lo.y) / (hi.y - lo.y).max(1e-6);
                let length = 0.002 + 0.038 * smoothstep(0.45, 0.8, height);
                let spec =
                    GrowSpec { length: length * jitter, comb: back, comb_rate: 60.0, gravity: 5.0, offset: 0.001 };
                grow_guide(&frame, &spec, points, sdf)
            }
            "afro" => {
                // A rounded volume (`afro_volume`): every guide leaves along the skin normal,
                // turns out from the volume's centre and ends near its surface, so roots near the
                // surface (front hairline, temples) grow short and the crown and back long, and
                // the top is a dome instead of a peak.
                let (ball, axes) = afro;
                let radial = normalize_or(frame.origin - ball, frame.n);
                let out = normalize_or(radial + frame.n * 0.5, frame.n);
                let length = to_ellipsoid(frame.origin, out, ball, axes).clamp(0.025, 0.14) * jitter;
                let spec = GrowSpec { length, comb: radial, comb_rate: 25.0, gravity: 2.0, offset: 0.006 };
                grow_guide(&frame, &spec, points, sdf)
            }
            "long_straight" | "medium_straight" | "wavy" | "long_wavy" => {
                let (length, gravity, rate) = match spec.name {
                    "long_straight" => (0.35, 40.0, 30.0),
                    "medium_straight" => (0.18, 40.0, 30.0),
                    "wavy" => (0.25, 30.0, 30.0),
                    _ => (0.40, 30.0, 30.0),
                };
                let outward = normalize_or(frame.origin - centre, frame.n);
                // Back over the top, down and back over the sides (no horizontal streaks at the
                // temples), straight down over the back of the head (where `back` is nearly the
                // skin normal, so its tangent part alone would point anywhere).
                let down = 0.9 * outward.x.abs() + 1.5 * (-outward.z).max(0.0);
                // Medium straight hair falls straight down at the sides, over the ears and temples
                // (combed back there, it left the ears and the corners of the forehead bare).
                // It also hangs close along the cheeks (combed slightly inward; the head's distance
                // grid keeps it on the skin), not straight down from the widest part of the skull.
                let (back_share, out) =
                    if spec_name == "medium_straight" { (1.0 - 0.75 * outward.x.abs(), -0.15) } else { (1.0, 0.3) };
                let mut comb = Vec3::new(outward.x, 0.0, outward.z) * out + back * back_share + (-Vec3::Y) * down;
                // Away from the parting on both sides of it.
                if let Some(line) = &part {
                    comb += part_comb(line, frame.origin);
                }
                let spec = GrowSpec { length: length * jitter, comb, comb_rate: rate, gravity, offset: 0.004 };
                let mut world = grow_falling_guide(&frame, &spec, 0.25, points, sdf);
                if matches!(spec_name, "wavy" | "long_wavy") {
                    wave_guide(&mut world, centre, frame.origin, WAVE);
                }
                world
            }
            "curly" | "coily" => {
                let (length, gravity, rate) = if spec.name == "curly" { (0.18, 12.0, 30.0) } else { (0.08, 4.0, 20.0) };
                let outward = normalize_or(frame.origin - centre, frame.n);
                let comb = Vec3::new(outward.x, 0.0, outward.z) * 0.4 + back;
                let spec = GrowSpec { length: length * jitter, comb, comb_rate: rate, gravity, offset: 0.004 };
                grow_guide(&frame, &spec, points, sdf)
            }
            "braided" => braid_guide(&frame, nape, lane_width, centre.x, points, sdf, &mut rng),
            _ => {
                let spec =
                    GrowSpec { length: 0.001_5 * jitter, comb: back, comb_rate: 400.0, gravity: 0.0, offset: 0.0 };
                grow_guide(&frame, &spec, points, None)
            }
        };
        let local: Vec<Vec3> = world.iter().map(|&w| frame.to_local(w)).collect();
        guides.push(t, bary, &local)?;
    }
    if guides.is_empty() {
        return Err(HairError::invalid("groom", 0, "no guide could be placed on the mask"));
    }
    Ok(PresetGroom { name: spec.name, guides, params, area: mask.triangles().to_vec(), part })
}

/// The guide wave of the wavy styles: period, amplitude (metres), and where it starts and reaches
/// its full amplitude (metres of length from the root).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GuideWave {
    /// Length of one period along the strand, metres.
    pub period: f32,
    /// Sideways amplitude, metres.
    pub amplitude: f32,
    /// Length from the root where the wave begins.
    pub start: f32,
    /// Length from the root where it reaches its full amplitude.
    pub full: f32,
}

/// The wave of `wavy` and `long_wavy`: loose 11 cm periods of 1.5 cm, from 5 cm below the root,
/// full at 14 cm (a styling choice, not a measurement; 8 cm periods of 1 cm read as crimp at
/// head distance).
pub const WAVE: GuideWave = GuideWave { period: 0.11, amplitude: 0.015, start: 0.05, full: 0.14 };

/// Adds `wave` to a hanging guide (`points` in world space, root first, `centre` the head's):
/// each point moves sideways (across the strand, along the head's surface) by a sine of its
/// length from the root. The phase varies slowly with the root position (a few radians across
/// the head), so neighbouring guides wave nearly in phase and the strands blended from them keep
/// the wave instead of averaging it away; the per-strand wave of the groom parameters only adds
/// small differences.
pub fn wave_guide(points: &mut [Vec3], centre: Vec3, root: Vec3, wave: GuideWave) {
    if points.len() < 3 || !(wave.period > 0.0 && wave.amplitude.is_finite() && wave.full > wave.start) {
        return;
    }
    let phase = 2.0 * (13.0 * root.x).sin() + 1.5 * (9.0 * root.y + 7.0 * root.z).sin();
    let original = points.to_vec();
    let mut length = 0.0;
    for k in 1..points.len() {
        length += (original[k] - original[k - 1]).length();
        let along = normalize_or(original[k] - original[k - 1], -Vec3::Y);
        // Outward in the horizontal plane: below the head the full outward direction turns
        // parallel to the hanging hair and its cross product flips from point to point (loops).
        let mut horizontal = original[k] - centre;
        horizontal.y = 0.0;
        let outward = normalize_or(horizontal, Vec3::Z);
        let side = normalize_or(along.cross(outward), Vec3::X);
        let ramp = ((length - wave.start) / (wave.full - wave.start)).clamp(0.0, 1.0);
        let ramp = ramp * ramp * (3.0 - 2.0 * ramp);
        let w = std::f32::consts::TAU * length / wave.period + phase;
        points[k] = original[k] + side * (wave.amplitude * ramp * w.sin());
    }
}

/// A braided guide: from the root over the scalp to the nape, then down one of three lanes.
fn braid_guide(
    frame: &RootFrame,
    nape: Vec3,
    lane_width: f32,
    centre_x: f32,
    points: u32,
    sdf: Option<&SdfGrid>,
    rng: &mut Rng,
) -> Vec<Vec3> {
    let points = points.clamp(4, MAX_GUIDE_POINTS) as usize;
    let lane = if frame.origin.x < centre_x - lane_width {
        0
    } else if frame.origin.x > centre_x + lane_width {
        2
    } else {
        1
    };
    let gather = points / 3;
    let jitter = Vec3::new(rng.range_f32(-1.0, 1.0), rng.range_f32(-1.0, 1.0), rng.range_f32(-1.0, 1.0)) * 0.004;
    let start = braid_offset(lane, 0.0, lane_width * 0.5, 0.08);
    let mut out = Vec::with_capacity(points);
    out.push(frame.origin);
    let lift = frame.origin + frame.n * 0.01;
    for k in 1..=gather {
        let s = k as f32 / gather as f32;
        let mut p = lift.lerp(nape + start + jitter, s);
        if let Some(grid) = sdf {
            let (d, g) = grid.sample(p);
            if d < 0.006 {
                p += normalize_or(g, frame.n) * (0.006 - d);
            }
        }
        out.push(p);
    }
    let braid_len = 0.24;
    let rest = points - 1 - gather;
    for k in 1..=rest {
        let s = braid_len * k as f32 / rest as f32;
        let mut p = nape + braid_offset(lane, s, lane_width * 0.5, 0.08) + jitter * (1.0 - s / braid_len * 0.5);
        if let Some(grid) = sdf {
            let (d, g) = grid.sample(p);
            if d < 0.004 {
                p += normalize_or(g, -Vec3::Z) * (0.004 - d);
            }
        }
        out.push(p);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::collide::{TriangleGrid, penetration};
    use crate::hair::placement::place;
    use crate::hair::surface::ProxyHead;
    use crate::hair::{GroomMaps, generate_groom};

    struct Fixture {
        head: ProxyHead,
        mask: ScalpMask,
        grid: TriangleGrid,
        sdf: SdfGrid,
    }

    fn head() -> Fixture {
        let head = ProxyHead::new(48).unwrap();
        let mask = ScalpMask::new(&head.surface, head.scalp.clone()).unwrap();
        let grid = TriangleGrid::new(head.surface.positions(), head.surface.triangles(), 0.01).unwrap();
        let sdf = SdfGrid::build(&grid, 0.004, 0.05).unwrap();
        Fixture { head, mask, grid, sdf }
    }

    fn world(g: &Guides, s: &HairSurface) -> Vec<Vec<Vec3>> {
        (0..g.len()).map(|i| g.world_points(i, &g.frame(i, s).unwrap())).collect()
    }

    #[test]
    fn hair_03_every_preset_builds_outside_the_head_and_generates() {
        let h = head();
        for spec in PRESETS {
            let p = preset_groom(spec.name, &h.head.surface, &h.mask, 120, Some(&h.sdf), 5).unwrap();
            assert_eq!(p.guides.len(), 120, "{}", spec.name);
            p.guides.check(&h.head.surface).unwrap();
            p.params.check().unwrap();
            let pts: Vec<Vec3> = world(&p.guides, &h.head.surface).concat();
            let report = penetration(&h.grid, Some(&h.sdf), &pts, 0.002);
            assert_eq!(report.inside, 0, "{}: {report:?}", spec.name);
            // A small groom from the preset generates.
            let params = GroomParams { strand_count: 500.0, ..p.params };
            let layout = place(&h.head.surface, &h.mask, &p.guides, &params, &GroomMaps::default(), 1).unwrap();
            let g = generate_groom(&h.head.surface, &layout, &p.guides, &params, Some(&h.sdf)).unwrap();
            assert_eq!(g.strands.len(), layout.len(), "{}", spec.name);
        }
    }

    #[test]
    fn hair_03_presets_have_their_shape() {
        let h = head();
        let s = &h.head.surface;
        let mean_len = |name: &str| {
            let p = preset_groom(name, s, &h.mask, 200, Some(&h.sdf), 9).unwrap();
            (0..p.guides.len()).map(|i| p.guides.length(i)).sum::<f32>() / p.guides.len() as f32
        };
        let stubble = mean_len("bald_stubble");
        let crop = mean_len("short_crop");
        let long = mean_len("long_straight");
        assert!(stubble < 0.002 && crop > 0.02 && crop < 0.04 && long > 0.3, "{stubble} {crop} {long}");
        // The fade is short low on the head and long on top.
        let fade = preset_groom("fade", s, &h.mask, 300, Some(&h.sdf), 9).unwrap();
        let (lo, hi) = mask_bounds(s, &h.mask);
        let (mut low, mut top) = (Vec::new(), Vec::new());
        for i in 0..fade.guides.len() {
            let y = (fade.guides.frame(i, s).unwrap().origin.y - lo.y) / (hi.y - lo.y);
            if y < 0.3 {
                low.push(fade.guides.length(i));
            } else if y > 0.85 {
                top.push(fade.guides.length(i));
            }
        }
        assert!(!low.is_empty() && !top.is_empty());
        assert!(low.iter().all(|&l| l < 0.004) && top.iter().all(|&l| l > 0.025));
        // Long hair hangs: the tips are below the roots.
        let lp = preset_groom("long_straight", s, &h.mask, 100, Some(&h.sdf), 9).unwrap();
        let w = world(&lp.guides, s);
        let below = w.iter().filter(|c| c[c.len() - 1].y < c[0].y - 0.1).count();
        assert!(below > 90, "{below}");
        // Hair combed back over the crown falls behind the head instead of standing out backward:
        // the tips hang below their roots and stay within a few centimetres behind the skull.
        let mp = preset_groom("medium_straight", s, &h.mask, 200, Some(&h.sdf), 9).unwrap();
        let w = world(&mp.guides, s);
        let hanging = w.iter().filter(|c| c[c.len() - 1].y < c[0].y - 0.05).count();
        let flaring = w.iter().filter(|c| c[c.len() - 1].z < lo.z - 0.06).count();
        assert!(hanging > 180 && flaring < 10, "{hanging} hang, {flaring} stand out behind");
        // The side part: guides rooted in the parted region within 3 cm of it leave it on their
        // own side.
        let line = mp.part.expect("medium straight has a part");
        let (mut near, mut away) = (0, 0);
        for c in &w {
            let d = line.side(c[0]);
            if line.in_region(c[0]) && d.abs() < 0.03 {
                near += 1;
                away += usize::from(line.side(c[4]) * d.signum() > d.abs());
            }
        }
        assert!(near > 3 && away * 10 >= near * 8, "{away} of {near} guides leave the part on their side");
        // The afro stands out in a round volume: tips further out than their roots and at
        // similar distances from the volume's centre.
        let af = preset_groom("afro", s, &h.mask, 200, Some(&h.sdf), 9).unwrap();
        let (ball, axes) = afro_volume(lo, hi);
        let w = world(&af.guides, s);
        let out = w.iter().filter(|c| (c[c.len() - 1] - ball).length() > (c[0] - ball).length() + 0.02).count();
        // Tips near the volume's surface: their ellipsoid radius (1 on the surface).
        let radius = |p: Vec3| {
            let e = p - ball;
            Vec3::new(e.x / axes.x, e.y / axes.y, e.z / axes.z).length()
        };
        let tips: Vec<f32> = w.iter().map(|c| radius(c[c.len() - 1])).collect();
        let mean = tips.iter().sum::<f32>() / tips.len() as f32;
        let spread = (tips.iter().map(|d| (d - mean) * (d - mean)).sum::<f32>() / tips.len() as f32).sqrt();
        assert!(out > 180 && spread < 0.25 * mean, "{out} out, tip radius {mean} +- {spread}");
        // No peak on top: the highest tip is not far above the tips around it (the highest
        // tenth's spread in height is small).
        let mut heights: Vec<f32> = w.iter().map(|c| c[c.len() - 1].y).collect();
        heights.sort_by(f32::total_cmp);
        let top = heights[heights.len() - 1] - heights[heights.len() * 9 / 10];
        assert!(top < 0.03, "the afro rises {top} m above its top tenth: a peak");
        // No tuft over the forehead: the volume reaches less far in front than behind.
        let front = w.iter().map(|c| c[c.len() - 1].z).fold(f32::MIN, f32::max) - hi.z;
        let behind = lo.z - w.iter().map(|c| c[c.len() - 1].z).fold(f32::MAX, f32::min);
        assert!(front < behind, "the afro reaches {front} m in front, {behind} m behind");
        // The braid ends below and behind the nape in three lanes that cross.
        let b = preset_groom("braided", s, &h.mask, 90, Some(&h.sdf), 9).unwrap();
        assert!((b.params.interpolation - 1.0).abs() < f32::EPSILON);
        let tips: Vec<Vec3> = world(&b.guides, s).iter().map(|c| c[c.len() - 1]).collect();
        assert!(tips.iter().all(|t| t.y < lo.y + 0.3 * (hi.y - lo.y) - 0.15 && t.z < lo.z));
    }

    #[test]
    fn hair_03_presets_are_deterministic_and_checked() {
        let h = head();
        let s = &h.head.surface;
        let a = preset_groom("curly", s, &h.mask, 50, Some(&h.sdf), 3).unwrap();
        let b = preset_groom("curly", s, &h.mask, 50, Some(&h.sdf), 3).unwrap();
        let c = preset_groom("curly", s, &h.mask, 50, Some(&h.sdf), 4).unwrap();
        assert_eq!(a, b);
        assert_ne!(a.guides, c.guides);
        assert!(matches!(preset_groom("mohawk", s, &h.mask, 5, None, 1), Err(HairError::Param { .. })));
        assert!(matches!(preset_groom("fade", s, &h.mask, 0, None, 1), Err(HairError::Limit { .. })));
        let empty = ScalpMask::new(s, Vec::new()).unwrap();
        assert!(preset_groom("fade", s, &empty, 5, None, 1).is_err());
    }

    // Contact sheet: every preset, 3000 CPU-generated strands each, seen from the character's left,
    // written to target/agents/hair/presets.png for a human (and the agent) to look at.
    #[test]
    fn hair_03_preset_contact_sheet() {
        let h = head();
        let s = &h.head.surface;
        let mut grooms = Vec::new();
        for spec in PRESETS {
            let p = preset_groom(spec.name, s, &h.mask, 300, Some(&h.sdf), 11).unwrap();
            let params = GroomParams { strand_count: 3000.0, ..p.params };
            let layout = place(s, &h.mask, &p.guides, &params, &GroomMaps::default(), 2).unwrap();
            grooms.push(generate_groom(s, &layout, &p.guides, &params, Some(&h.sdf)).unwrap());
        }
        let panels: Vec<crate::hair::sheet::Panel<'_>> = grooms
            .iter()
            .map(|g| crate::hair::sheet::Panel {
                dots: s.positions(),
                strands: &g.world,
                points: g.strands.points as usize,
                centre: h.head.centre,
                scale: 700.0,
                front: false,
            })
            .collect();
        crate::hair::sheet::write("presets.png", &panels, 4, 240, 300);
    }

    /// The guide wave: full amplitude past `WAVE.full`, none above `WAVE.start`, and two guides
    /// rooted 1 cm apart wave nearly in phase (so the strands blended from them keep it).
    #[test]
    fn hair_03_guide_wave_is_coherent_between_neighbours() {
        let centre = Vec3::new(0.0, 1.6, 0.0);
        let hanging = |x: f32| -> Vec<Vec3> {
            (0..32).map(|k| Vec3::new(x, 1.6 - 0.4 * k as f32 / 31.0, -0.09)).collect::<Vec<Vec3>>()
        };
        let (mut a, mut b) = (hanging(0.0), hanging(0.01));
        let (ra, rb) = (a[0], b[0]);
        wave_guide(&mut a, centre, ra, WAVE);
        wave_guide(&mut b, centre, rb, WAVE);
        let side = |p: &[Vec3], x: f32| -> Vec<f32> { p.iter().map(|q| q.x - x).collect() };
        let (sa, sb) = (side(&a, 0.0), side(&b, 0.01));
        // Nothing near the root, about the full amplitude further down.
        assert!(sa[..3].iter().all(|d| d.abs() < 1e-4), "{sa:?}");
        let peak = sa[12..].iter().fold(0.0_f32, |m, d| m.max(d.abs()));
        assert!((0.7 * WAVE.amplitude..=1.01 * WAVE.amplitude).contains(&peak), "{peak}");
        // Correlation of the two sideways offsets over the waved part.
        let dot: f32 = sa[12..].iter().zip(&sb[12..]).map(|(x, y)| x * y).sum();
        let na: f32 = sa[12..].iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = sb[12..].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(dot / (na * nb) > 0.9, "{}", dot / (na * nb));
    }
}
