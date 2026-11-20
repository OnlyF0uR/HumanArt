//! The separate closed parts of `[MOD-01]`: eyeballs, tear-line strips, eye-occlusion shells, upper
//! and lower teeth, gums, tongue, and the lip-line helper.
//!
//! Every part is a closed cage, fitted and subdivided like the body, so it is watertight and
//! quad-only. Sizes: the eyeball radius is the anatomy's (12 mm), its corneal sphere and limbus
//! come from [`EyeParams`] (7.8 mm and 11.7 mm by default) scaled to that radius, the way the
//! renderer's analytic eye scales them `[EYE-07]`; tooth crown sizes are textbook means of the
//! permanent dentition (Wheeler's dental anatomy) scaled by [`MouthParams`] `[MTH-01]`, the tongue
//! by its length and width `[MTH-03]`; the arch is an ellipse through the incisors and second
//! molars. [`build`] uses the default parameters, [`build_with`] any valid ones.

use std::f32::consts::{PI, TAU};

use humanart_math::Vec3;

use super::anatomy::{Anatomy, sided};
use super::body::{BodyCage, head_point};
use super::cage::{Cage, Tag, fit, subdivide_levels};
use super::face::{fissure, on_sphere};
use super::regions::{C, R};
use super::tube::Frame;
use crate::error::{ModelError, Result};
use crate::eye::EyeParams;
use crate::mouth::MouthParams;

/// One separate part: stable name, region, chart and its final (subdivided) mesh.
#[derive(Clone, Debug)]
pub struct Part {
    /// Stable part name (also the mesh part name).
    pub name: &'static str,
    /// The part's region.
    pub region: R,
    /// The part's UV chart group.
    pub chart: C,
    /// The surface.
    pub mesh: Cage,
}

/// A ring walked the other way round from the same first vertex (so a cap on it has its corners
/// at the same ring vertices as a cap on the ring itself, and mirrored parts stay mirrored).
fn reversed(ring: &[u32]) -> Vec<u32> {
    ring.iter().take(1).chain(ring.iter().skip(1).rev()).copied().collect()
}

/// Fits and subdivides a part cage.
fn finish(name: &'static str, region: R, chart: C, mut cage: Cage, levels: usize) -> Part {
    let targets = cage.pos.clone();
    fit(&mut cage, &targets, levels, 8, &|_, _| false);
    let mesh = subdivide_levels(&cage, levels, &|_, _| false).pop().map_or(cage, |s| s.cage);
    Part { name, region, chart, mesh }
}

/// A closed tube: rings (each counter-clockwise around the tube direction) bridged in order, the
/// first ring closed facing backward and the last facing forward with grid caps whose first corner
/// is ring slot `start` (chosen so that mirrored parts get mirrored caps).
fn closed_tube(
    cage: &mut Cage,
    rings: &[Vec<Vec3>],
    tag: Tag,
    caps: (usize, usize),
    start: usize,
) -> Result<Vec<Vec<u32>>> {
    let ids: Vec<Vec<u32>> = rings.iter().map(|r| cage.add_all(r)).collect();
    for w in ids.windows(2) {
        cage.bridge(&w[0], &w[1], tag);
    }
    let (Some(first), Some(last)) = (ids.first(), ids.last()) else {
        return Err(ModelError::invalid("part", 0, "tube without rings"));
    };
    let rev = reversed(first);
    let n = first.len().max(1);
    cage.cap(&rev, caps.0, caps.1, start % n, tag, &|_, _, p| p)?;
    cage.cap(last, caps.0, caps.1, start % n, tag, &|_, _, p| p)?;
    Ok(ids)
}

/// A closed torus along a closed centreline: 4-slot rings across it with half-sizes `(a, b)` along
/// `normal(i)` and the in-plane cross direction.
fn torus(cage: &mut Cage, centre: &[Vec3], normal: &dyn Fn(usize) -> Vec3, a: f32, b: f32, tag: Tag) {
    let n = centre.len();
    let rings: Vec<Vec<u32>> = (0..n)
        .map(|i| {
            let t = (centre[(i + 1) % n] - centre[(i + n - 1) % n]).normalize_or_zero();
            let f = Frame::new(centre[i], t, normal(i), 1.0);
            let pts: Vec<Vec3> = (0..4).map(|k| f.at((45.0 + 90.0 * k as f32).to_radians(), a, b)).collect();
            cage.add_all(&pts)
        })
        .collect();
    for i in 0..n {
        cage.bridge(&rings[i], &rings[(i + 1) % n], tag);
    }
}

/// The eyeball's outer surface `[EYE-07]`: the scleral sphere and, in front of the limbus, the
/// corneal sphere. Millimetres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeballShape {
    /// Scleral radius.
    pub radius: f32,
    /// Limbus radius (half the visible iris diameter).
    pub limbus: f32,
    /// Anterior corneal radius of curvature.
    pub cornea: f32,
}

impl EyeballShape {
    /// The surface of an eyeball of radius `radius` mm with the cornea and limbus of `eye` scaled by
    /// `radius / eye.eyeball_radius_mm` (as `humanart_render::eye` scales the analytic eye to the
    /// fitted eyeball). The limbus is kept inside the sphere and the cornea at least as wide as it.
    pub fn new(eye: &EyeParams, radius: f32) -> Self {
        let k = radius / eye.eyeball_radius_mm.max(1e-3);
        let limbus = (eye.limbus_radius_mm * k).min(radius * 0.95);
        Self { radius, limbus, cornea: (eye.cornea_radius_mm * k).max(limbus * 1.01) }
    }

    /// Distance from the eyeball centre to its surface along a unit direction `d` (local, +Z
    /// forward), millimetres: the scleral sphere, or the cornea in front of the limbus.
    pub fn surface(&self, d: Vec3) -> f32 {
        let (r, l, c) = (self.radius, self.limbus, self.cornea);
        let limbus_z = (r * r - l * l).sqrt();
        let centre = limbus_z - (c * c - l * l).sqrt();
        if d.z * r <= limbus_z {
            return r;
        }
        // Ray from the eyeball centre: |t d - (0,0,c)| = rc.
        let b = d.z * centre;
        let disc = b * b - (centre * centre - c * c);
        if disc < 0.0 {
            return r;
        }
        (b + disc.sqrt()).max(r)
    }
}

fn eyeball(a: &Anatomy, shape: &EyeballShape, side: f32) -> Result<Part> {
    let e = sided(a.eye_center, side);
    let n = 16;
    let point = |polar: f32, k: usize| -> Vec3 {
        let psi = TAU * k as f32 / n as f32;
        let d = Vec3::new(polar.sin() * psi.cos(), polar.sin() * psi.sin(), polar.cos());
        e + d * (shape.surface(d) * 0.001)
    };
    let polars = [150.0_f32, 120.0, 90.0, 62.0, 42.0, 29.0, 18.0];
    let rings: Vec<Vec<Vec3>> = polars.iter().map(|&p| (0..n).map(|k| point(p.to_radians(), k)).collect()).collect();
    let region = R::sided(R::EyeballL, R::EyeballR, side);
    let chart = C::sided(C::EyeballL, C::EyeballR, side);
    let tag = Tag::new(region.id(), chart.id());
    let mut cage = Cage::new();
    let ids: Vec<Vec<u32>> = rings.iter().map(|r| cage.add_all(r)).collect();
    for w in ids.windows(2) {
        cage.bridge(&w[0], &w[1], tag);
    }
    let project = |p: Vec3| -> Vec3 {
        let d = (p - e).normalize_or_zero();
        e + d * (shape.surface(d) * 0.001)
    };
    let (Some(first), Some(last)) = (ids.first(), ids.last()) else {
        return Err(ModelError::invalid("eyeball", 0, "no rings"));
    };
    let rev = reversed(first);
    cage.cap(&rev, 4, 4, 0, tag, &|_, _, p| {
        let d = (p - e).normalize_or_zero();
        project(e + Vec3::new(d.x, d.y, d.z.min(-0.5)))
    })?;
    cage.cap(last, 4, 4, 0, tag, &|_, _, p| {
        let d = (p - e).normalize_or_zero();
        project(e + Vec3::new(d.x, d.y, d.z.max(0.5)))
    })?;
    Ok(finish(if side > 0.0 { "eyeball-L" } else { "eyeball-R" }, region, chart, cage, 2))
}

fn tear_line(a: &Anatomy, body: &BodyCage, side: f32) -> Part {
    let e = sided(a.eye_center, side);
    let r = a.eye_radius * 1000.0;
    let params = &body.loops.eyes[usize::from(side < 0.0)].params;
    let local = |x: f32, y: f32, z: f32| e + Vec3::new(side * x, y, z) * 0.001;
    let centre: Vec<Vec3> = params
        .iter()
        .map(|&t| {
            let (x, y) = fissure(t);
            let (x, y) = (-0.5 + (x + 0.5) * 0.9, 0.5 + (y - 0.5) * 0.93);
            local(x, y, on_sphere(x, y, r + 0.45))
        })
        .collect();
    let region = R::sided(R::TearLineL, R::TearLineR, side);
    let chart = C::sided(C::TearLineL, C::TearLineR, side);
    let mut cage = Cage::new();
    torus(
        &mut cage,
        &centre,
        &|i| (centre[i] - e).normalize_or_zero(),
        0.00035,
        0.0007,
        Tag::new(region.id(), chart.id()),
    );
    finish(if side > 0.0 { "tear-line-L" } else { "tear-line-R" }, region, chart, cage, 2)
}

fn eye_occlusion(a: &Anatomy, shape: &EyeballShape, body: &BodyCage, side: f32) -> Result<Part> {
    let e = sided(a.eye_center, side);
    let r = a.eye_radius * 1000.0;
    let params = &body.loops.eyes[usize::from(side < 0.0)].params;
    let shell = |scale: f32, offset: f32| -> Vec<Vec3> {
        params
            .iter()
            .map(|&t| {
                let (x, y) = fissure(t);
                let (x, y) = (-0.5 + (x + 0.5) * scale, 0.5 + (y - 0.5) * scale);
                let d = Vec3::new(side * x, y, on_sphere(x, y, r)).normalize_or_zero();
                e + d * ((shape.surface(d) + offset) * 0.001)
            })
            .collect()
    };
    let region = R::sided(R::EyeOcclusionL, R::EyeOcclusionR, side);
    let chart = C::sided(C::OcclusionL, C::OcclusionR, side);
    let tag = Tag::new(region.id(), chart.id());
    let mut cage = Cage::new();
    // A thin lens: back sheet ring, rim, front sheet ring, with grid caps on both sheets.
    let rings = [shell(0.55, 0.25), shell(1.0, 0.35), shell(0.55, 0.45)];
    // The slot order runs counter-clockwise seen from the front on both sides only in world
    // terms, so check it against +Z and reverse if needed.
    let mut rings: Vec<Vec<Vec3>> = rings.to_vec();
    let c = rings[1].iter().fold(Vec3::ZERO, |s, p| s + *p) / rings[1].len() as f32;
    let mut turn = 0.0;
    for i in 0..rings[1].len() {
        turn += (rings[1][i] - c).cross(rings[1][(i + 1) % rings[1].len()] - c).z;
    }
    if turn < 0.0 {
        // Keep the first slot first: the caps start there, and the fan triangulation of every
        // face starts at its first corner, so both sides must start at mirrored slots.
        for ring in &mut rings {
            ring[1..].reverse();
        }
    }
    let project_front = |p: Vec3, off: f32| -> Vec3 {
        let d = (p - e).normalize_or_zero();
        e + d * ((shape.surface(d) + off) * 0.001)
    };
    let ids: Vec<Vec<u32>> = rings.iter().map(|ring| cage.add_all(ring)).collect();
    cage.bridge(&ids[0], &ids[1], tag);
    cage.bridge(&ids[1], &ids[2], tag);
    let rev = reversed(&ids[0]);
    cage.cap(&rev, 4, 4, 0, tag, &|_, _, p| project_front(p, 0.25))?;
    cage.cap(&ids[2], 4, 4, 0, tag, &|_, _, p| project_front(p, 0.45))?;
    let mut part = finish(if side > 0.0 { "eye-occlusion-L" } else { "eye-occlusion-R" }, region, chart, cage, 1);
    // The subdivided sheets sag toward the eyeball between their cage rings (over the strongly
    // curved canthi most); the shell is drawn as a film over the eyeball, so no point of it may
    // lie inside the eyeball or closer than the clearance.
    for p in &mut part.mesh.pos {
        let d = (*p - e).normalize_or_zero();
        let least = (shape.surface(d) + OCCLUSION_CLEARANCE) * 0.001;
        if (*p - e).length() < least {
            *p = e + d * least;
        }
    }
    Ok(part)
}

/// Least distance of the eye-occlusion shell from the eyeball, millimetres.
const OCCLUSION_CLEARANCE: f32 = 0.2;

/// The dental arch: labial line through the incisors (`z0`) and molars.
struct Arch {
    a: f32,
    b: f32,
    z0: f32,
    table: Vec<(f32, f32)>,
}

impl Arch {
    fn new(a: f32, b: f32, z0: f32) -> Self {
        let mut table = vec![(0.0, 0.0)];
        let steps = 400;
        let mut s = 0.0;
        let mut prev = (0.0_f32, z0);
        for i in 1..=steps {
            let th = (PI * 0.62) * i as f32 / steps as f32;
            let p = (a * th.sin(), z0 - b * (1.0 - th.cos()));
            s += ((p.0 - prev.0).powi(2) + (p.1 - prev.1).powi(2)).sqrt();
            table.push((s, th));
            prev = p;
        }
        Self { a, b, z0, table }
    }

    /// Point (x, z) and outward normal at signed arc length `s` (positive to the left).
    fn at(&self, s: f32) -> (f32, f32, Vec3) {
        let sign = if s < 0.0 { -1.0 } else { 1.0 };
        let s = s.abs();
        let th = super::tube::lookup(&self.table, s);
        let (x, z) = (self.a * th.sin(), self.z0 - self.b * (1.0 - th.cos()));
        let tangent = Vec3::new(self.a * th.cos(), 0.0, -self.b * th.sin()).normalize_or_zero();
        let outward = Vec3::new(-tangent.z, 0.0, tangent.x);
        (sign * x, z, Vec3::new(sign * outward.x, 0.0, outward.z))
    }
}

/// Mesiodistal width, labiolingual thickness, crown height (mm) from the midline back: central,
/// lateral, canine, two premolars, two molars.
const UPPER_TEETH: [(f32, f32, f32); 7] = [
    (8.6, 7.0, 10.5),
    (6.6, 6.0, 9.0),
    (7.6, 8.0, 10.0),
    (7.0, 9.0, 8.5),
    (6.6, 9.0, 8.0),
    (10.0, 11.0, 7.5),
    (9.0, 11.0, 7.0),
];
const LOWER_TEETH: [(f32, f32, f32); 7] = [
    (5.0, 6.0, 9.0),
    (5.5, 6.2, 9.5),
    (7.0, 7.5, 11.0),
    (7.0, 7.7, 8.5),
    (7.0, 8.0, 8.0),
    (11.0, 10.5, 7.5),
    (10.5, 10.0, 7.0),
];

/// The crown sizes of one arch scaled by the tooth proportions of `mouth` `[MTH-01]`: incisors,
/// canine, then premolars and molars (width and crown height; the labiolingual depth follows the
/// width).
fn scaled_teeth(mouth: &MouthParams, upper: bool) -> [(f32, f32, f32); 7] {
    let t = &mouth.teeth;
    let mut sizes = if upper { UPPER_TEETH } else { LOWER_TEETH };
    for (k, size) in sizes.iter_mut().enumerate() {
        let (w, h) = match k {
            0 | 1 => (t.incisor_width, t.incisor_height),
            2 => (t.canine_width, t.canine_height),
            _ => (t.molar_width, t.molar_height),
        };
        *size = (size.0 * w, size.1 * w, size.2 * h);
    }
    sizes
}

fn teeth(a: &Anatomy, mouth: &MouthParams, upper: bool) -> Result<(Part, Part)> {
    let sizes = &scaled_teeth(mouth, upper);
    let arch = if upper { Arch::new(27.0, 40.0, 91.5) } else { Arch::new(25.0, 38.0, 89.0) };
    let (region, gum_region) = if upper { (R::TeethUpper, R::GumsUpper) } else { (R::TeethLower, R::GumsLower) };
    let tag = Tag::new(region.id(), C::Teeth.id());
    let gum_tag = Tag::new(gum_region.id(), C::Gums.id());
    // Occlusal direction: down for the upper teeth, up for the lower.
    let occlusal = if upper { -Vec3::Y } else { Vec3::Y };
    let mut cage = Cage::new();
    let mut necks: Vec<(f32, f32, f32)> = Vec::new();
    for side in [1.0_f32, -1.0] {
        let mut s = 0.0;
        for (k, &(w, d, h)) in sizes.iter().enumerate() {
            let mid = s + w * 0.5;
            s += w;
            let (x, z, out) = arch.at(side * mid);
            let incisor = k <= 2;
            let edge_y = if upper {
                if incisor { -50.5 } else { -49.0 }
            } else if incisor {
                -47.5
            } else {
                -49.0
            };
            let centre_xz = Vec3::new(x, 0.0, z) - out * (d * 0.5);
            let tangent = Vec3::Y.cross(out).normalize_or_zero();
            // Rings from the root end to the occlusal end (tube direction = occlusal).
            let profile: [(f32, f32, f32); 5] = if incisor {
                [(h + 6.0, 0.55, 0.65), (h, 0.78, 0.85), (h * 0.5, 1.0, 1.0), (1.2, 0.95, 0.55), (0.2, 0.9, 0.22)]
            } else {
                [(h + 6.0, 0.6, 0.65), (h, 0.85, 0.85), (h * 0.5, 1.0, 1.0), (1.2, 0.97, 0.95), (0.2, 0.85, 0.8)]
            };
            let rings: Vec<Vec<Vec3>> = profile
                .iter()
                .map(|&(above, sw, sd)| {
                    let y = edge_y - occlusal.y * above;
                    let o = head_point(a, centre_xz.x, y, centre_xz.z);
                    let f = Frame::new(o, occlusal, tangent, 1.0);
                    (0..8)
                        .map(|j| {
                            let th = TAU * j as f32 / 8.0 + PI / 8.0;
                            // Rounded rectangle: superellipse with exponent 3.
                            let (c, sn) = (th.cos(), th.sin());
                            let rr = 1.0 / (c.abs().powf(3.0) + sn.abs().powf(3.0)).powf(1.0 / 3.0);
                            f.at(th, w * 0.5 * sw * rr * 0.001, d * 0.5 * sd * rr * 0.001)
                        })
                        .collect()
                })
                .collect();
            // Mirroring maps slot j of a tooth to slot 3 - j of its partner: the right tooth's caps
            // start one slot later.
            closed_tube(&mut cage, &rings, tag, (2, 2), usize::from(side < 0.0))?;
            necks.push((side * mid, edge_y - occlusal.y * h, d));
        }
    }
    let name = if upper { "teeth-upper" } else { "teeth-lower" };
    let teeth = finish(name, region, C::Teeth, cage, 1);
    // Gums: a horseshoe tube along the arch around the tooth necks.
    let total: f32 = sizes.iter().map(|t| t.0).sum();
    let stations = 18;
    let mut rings: Vec<Vec<Vec3>> = Vec::with_capacity(stations);
    for i in 0..stations {
        let s = -total + 2.0 * total * (i as f32 + 0.5) / stations as f32;
        let (x, z, out) = arch.at(s);
        let tooth = necks
            .iter()
            .min_by(|p, q| (p.0 - s).abs().total_cmp(&(q.0 - s).abs()))
            .copied()
            .unwrap_or((0.0, -45.0, 7.0));
        let thick = tooth.2 * 0.5 + 1.6;
        let centre = Vec3::new(x, 0.0, z) - out * (tooth.2 * 0.5);
        let neck_y = tooth.1;
        let y_mid = neck_y - occlusal.y * 3.5;
        let o = head_point(a, centre.x, y_mid, centre.z);
        let tangent = {
            // Central difference: the same at mirrored stations.
            let ((x1, z1, _), (x2, z2, _)) = (arch.at(s - 0.25), arch.at(s + 0.25));
            Vec3::new(x2 - x1, 0.0, z2 - z1).normalize_or_zero()
        };
        let f = Frame::new(o, tangent, out, 1.0);
        rings.push((0..8).map(|j| f.at(TAU * j as f32 / 8.0, thick * 0.001, 6.0 * 0.001)).collect());
    }
    let mut gum = Cage::new();
    closed_tube(&mut gum, &rings, gum_tag, (2, 2), 0)?;
    let gum_name = if upper { "gums-upper" } else { "gums-lower" };
    Ok((teeth, finish(gum_name, gum_region, C::Gums, gum, 2)))
}

fn tongue(a: &Anatomy, mouth: &MouthParams) -> Result<Part> {
    // Length scales forward from the root station, width across `[MTH-03]`.
    let (length, width) = (mouth.tongue.length, mouth.tongue.width);
    // (z, half width, half height, centre y) from the root to the tip, head millimetres.
    let stations: [(f32, f32, f32, f32); 6] = [
        (42.0, 17.0, 10.0, -57.0),
        (52.0, 18.0, 10.0, -57.5),
        (62.0, 17.0, 9.0, -57.5),
        (72.0, 15.0, 7.5, -57.0),
        (80.0, 11.5, 6.0, -56.5),
        (85.5, 7.0, 4.0, -56.0),
    ];
    let rings: Vec<Vec<Vec3>> = stations
        .iter()
        .map(|&(z, w, h, y)| {
            let (z, w) = (42.0 + (z - 42.0) * length, w * width);
            let f = Frame::new(head_point(a, 0.0, y, z), Vec3::Z, Vec3::X, 1.0);
            (0..12)
                .map(|j| {
                    let th = TAU * j as f32 / 12.0;
                    let (c, sn) = (th.cos(), th.sin());
                    // Flatter on top (the dorsum), rounder below.
                    let hh = if sn > 0.0 { h * (0.75 + 0.25 * sn) } else { h };
                    f.at(th, w * 0.001, hh * 0.001) + Vec3::Y * (0.001 * if sn > 0.0 { -0.8 * c * c } else { 0.0 })
                })
                .collect()
        })
        .collect();
    let mut cage = Cage::new();
    // Mirroring maps slot j to 6 - j: corners at slots 11, 1, 5, 7 are a mirror-symmetric set.
    closed_tube(&mut cage, &rings, Tag::new(R::Tongue.id(), C::Tongue.id()), (2, 4), 11)?;
    Ok(finish("tongue", R::Tongue, C::Tongue, cage, 2))
}

fn lip_line(body: &BodyCage, targets: &[Vec3]) -> Part {
    let ring = &body.loops.mouth.rings[4];
    let inner = &body.loops.mouth.rings[5];
    let centre: Vec<Vec3> =
        ring.iter().zip(inner).map(|(&o, &i)| targets[o as usize] * 0.75 + targets[i as usize] * 0.25).collect();
    let mid = centre.iter().fold(Vec3::ZERO, |s, p| s + *p) / centre.len().max(1) as f32;
    let mut cage = Cage::new();
    let tag = Tag::new(R::MouthHelper.id(), C::MouthHelper.id());
    torus(
        &mut cage,
        &centre,
        &|i| Vec3::new(centre[i].x - mid.x, centre[i].y - mid.y, 0.0).normalize_or(Vec3::Y),
        0.0004,
        0.0008,
        tag,
    );
    finish("mouth-helper", R::MouthHelper, C::MouthHelper, cage, 2)
}

/// Builds all separate parts with the default eye and mouth parameters. `targets` are the designed
/// (unfitted) cage positions of the body.
pub fn build(a: &Anatomy, body: &BodyCage, targets: &[Vec3]) -> Result<Vec<Part>> {
    build_with(a, body, targets, &EyeParams::default(), &MouthParams::default())
}

/// Builds all separate parts with the eyeball shaped by `eye` `[EYE-07]` and the teeth and tongue
/// sized by `mouth` `[MTH-01]` `[MTH-03]`; both are validated first.
pub fn build_with(
    a: &Anatomy,
    body: &BodyCage,
    targets: &[Vec3],
    eye: &EyeParams,
    mouth: &MouthParams,
) -> Result<Vec<Part>> {
    eye.validate()?;
    mouth.validate()?;
    let shape = EyeballShape::new(eye, a.eye_radius * 1000.0);
    let mut parts = Vec::new();
    for side in [1.0_f32, -1.0] {
        parts.push(eyeball(a, &shape, side)?);
    }
    for side in [1.0_f32, -1.0] {
        parts.push(tear_line(a, body, side));
    }
    for side in [1.0_f32, -1.0] {
        parts.push(eye_occlusion(a, &shape, body, side)?);
    }
    let (tu, gu) = teeth(a, mouth, true)?;
    let (tl, gl) = teeth(a, mouth, false)?;
    parts.extend([tu, tl, gu, gl]);
    parts.push(tongue(a, mouth)?);
    parts.push(lip_line(body, targets));
    Ok(parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eye_07_the_default_eyeball_has_the_textbook_corneal_bulge() {
        let shape = EyeballShape::new(&EyeParams::default(), 12.0);
        assert!((shape.limbus - 5.85).abs() < 1e-5 && (shape.cornea - 7.8).abs() < 1e-5);
        // Apex: limbus plane 10.478 mm in front of the centre, corneal centre 5.159 mm behind it.
        let apex = shape.surface(Vec3::Z);
        assert!((apex - 13.118).abs() < 2e-3, "apex {apex}");
        assert!((shape.surface(-Vec3::Z) - 12.0).abs() < 1e-6);
        // Continuous at the limbus: the corneal sphere meets the sclera there.
        let limbus_z = (144.0_f32 - 5.85 * 5.85).sqrt();
        let just_inside = Vec3::new(5.849, 0.0, limbus_z).normalize_or_zero();
        assert!((shape.surface(just_inside) - 12.0).abs() < 0.01);
    }

    #[test]
    fn eye_07_the_eyeball_follows_its_parameters() {
        let flat = EyeParams { cornea_radius_mm: 8.6, ..EyeParams::default() };
        let steep = EyeParams { cornea_radius_mm: 7.0, ..EyeParams::default() };
        let (flat, steep) = (EyeballShape::new(&flat, 12.0), EyeballShape::new(&steep, 12.0));
        assert!(steep.surface(Vec3::Z) > flat.surface(Vec3::Z) + 0.2);
        // Scaled to a larger eyeball: the same shape, larger.
        let big = EyeballShape::new(&EyeParams::default(), 13.2);
        assert!(
            (big.surface(Vec3::Z) / 13.2 - EyeballShape::new(&EyeParams::default(), 12.0).surface(Vec3::Z) / 12.0)
                .abs()
                < 1e-4
        );
    }

    #[test]
    fn mth_01_tooth_proportions_scale_their_class_only() {
        let mut mouth = MouthParams::default();
        assert_eq!(scaled_teeth(&mouth, true), UPPER_TEETH);
        assert_eq!(scaled_teeth(&mouth, false), LOWER_TEETH);
        mouth.teeth.incisor_width = 1.2;
        mouth.teeth.molar_height = 0.8;
        let s = scaled_teeth(&mouth, true);
        assert!((s[0].0 - UPPER_TEETH[0].0 * 1.2).abs() < 1e-5 && (s[1].1 - UPPER_TEETH[1].1 * 1.2).abs() < 1e-5);
        assert_eq!(s[0].2, UPPER_TEETH[0].2);
        assert_eq!(s[2], UPPER_TEETH[2]);
        assert!((s[6].2 - UPPER_TEETH[6].2 * 0.8).abs() < 1e-5 && s[6].0 == UPPER_TEETH[6].0);
    }
}
