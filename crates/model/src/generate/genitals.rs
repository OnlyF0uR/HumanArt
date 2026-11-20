//! External genitalia, both variants (ADR-039).
//!
//! The groin carries a fixed attachment loop: a ring of 12 body-cage vertices whose edges are
//! creases ([`attachment_loop`], placed by `body::build`). A crease edge and a vertex on exactly
//! two crease edges subdivide from the loop alone, and the body faces outside the loop never see
//! the faces inside, so the body surface is the same, bit for bit, with either variant (test
//! `adr_039_body_is_identical_with_either_variant`) and the variant's surface meets it exactly.
//! The variant's faces carry the genital regions ([`R::is_genital`]) and stay in the body mesh
//! part, so the skin is one closed surface with either variant.
//!
//! Both variants are built in the site frame ([`site`]): `s` runs along the midline from the top
//! of the loop (below the mons pubis, 0) to its bottom (in front of the perineum, 1), `x` across
//! (+X is the left side), `h` out of the body along the surface normal.
//!
//! Sizes are design values chosen inside published ranges, transcribed from memory and **to be
//! checked against the papers**: the vulva after Lloyd et al. 2005 (BJOG 112: labia minora 20 to
//! 100 mm long, clitoral glans 3 to 10 mm wide, perineum 15 to 55 mm), the penis after Veale et al.
//! 2015 (BJU Int 115: flaccid length 91.6 mm, flaccid girth 93.1 mm). The scrotum (each testis
//! about 45 x 25 x 30 mm) and every height are **unsourced**.

use std::f32::consts::TAU;

use humanart_math::Vec3;

use super::anatomy::CROTCH_HEIGHT;
use super::cage::{Cage, Tag};
use super::regions::{C, R};
use crate::error::{ModelError, Result};

/// Which external genitalia the template carries (ADR-039): a discrete choice, never a blend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Variant {
    /// Vulva: mons-side commissure, labia majora and minora, clitoral hood and glans, vestibule.
    #[default]
    Female,
    /// Penis (flaccid, uncircumcised) and scrotum.
    Male,
}

impl Variant {
    /// Both variants.
    pub const ALL: [Self; 2] = [Self::Female, Self::Male];

    /// A stable name of the variant.
    pub const fn part_name(self) -> &'static str {
        match self {
            Self::Female => "genitals-female",
            Self::Male => "genitals-male",
        }
    }
}

/// The midline of the site as `(y, z)`: top of the loop, the bend under the pubis, bottom of the
/// loop in front of the perineum (a quadratic Bézier).
const TOP: [f32; 2] = [0.858, 0.071];
const BEND: [f32; 2] = [0.817, 0.064];
const BOTTOM: [f32; 2] = [CROTCH_HEIGHT, -0.004];
/// How fast the site rises into the thighs away from the midline (1/m): `h` falls by `GROOVE x²`.
const GROOVE: f32 = 10.0;
/// Half-width of the attachment loop at its middle, metres.
pub const LOOP_HALF_WIDTH: f32 = 0.020;
/// Vertices of the attachment loop.
pub const LOOP_LEN: usize = 12;

fn midline(s: f32) -> Vec3 {
    let (a, b, c) = ((1.0 - s) * (1.0 - s), 2.0 * s * (1.0 - s), s * s);
    Vec3::new(0.0, a * TOP[0] + b * BEND[0] + c * BOTTOM[0], a * TOP[1] + b * BEND[1] + c * BOTTOM[1])
}

/// The unit tangent of the midline (pointing toward the perineum).
fn tangent(s: f32) -> Vec3 {
    let (a, b) = (2.0 * (1.0 - s), 2.0 * s);
    Vec3::new(
        0.0,
        a * (BEND[0] - TOP[0]) + b * (BOTTOM[0] - BEND[0]),
        a * (BEND[1] - TOP[1]) + b * (BOTTOM[1] - BEND[1]),
    )
    .normalize_or(Vec3::new(0.0, -1.0, 0.0))
}

/// The outward unit normal of the site on the midline (forward at the top, down at the bottom).
fn normal(s: f32) -> Vec3 {
    let t = tangent(s);
    Vec3::new(0.0, t.z, -t.y)
}

/// Length of the midline from `s = 0` to `s = 1`, metres.
pub fn site_length() -> f32 {
    let n = 64;
    (0..n).map(|i| midline((i + 1) as f32 / n as f32).distance(midline(i as f32 / n as f32))).sum()
}

/// A point of the site frame: `s` along the midline (0 top, 1 bottom), `x` across (metres, +X
/// left), `h` out of the body along the site's normal (metres). Mirror-exact in `x`.
pub fn site(s: f32, x: f32, h: f32) -> Vec3 {
    let n0 = normal(s);
    let base = midline(s) + Vec3::X * x - n0 * (GROOVE * x * x);
    let n = (n0 + Vec3::X * (2.0 * GROOVE * x)).normalize_or(n0);
    base + n * h
}

/// The angle of slot `j` of an `n`-slot ring (0 at the bottom of the loop, a quarter turn at the
/// left side) as `(cos, |sin|, side)`, computed from the slot's left-side mirror so mirrored slots
/// get exactly mirrored values.
fn slot(j: usize, n: usize) -> (f32, f32, f32) {
    let (k, side) = if j <= n / 2 { (j, 1.0) } else { (n - j, -1.0) };
    let th = TAU * k as f32 / n as f32;
    let sin = if k == 0 || 2 * k == n { 0.0 } else { th.sin().abs() };
    (th.cos(), sin, side)
}

/// The designed attachment loop: an oval on the site, counter-clockwise seen from outside from
/// the bottom (slot 0) through the left side (slot 3) and the top (slot 6).
pub fn attachment_loop() -> Vec<Vec3> {
    (0..LOOP_LEN)
        .map(|j| {
            let (c, sn, side) = slot(j, LOOP_LEN);
            site(0.5 + 0.5 * c, side * LOOP_HALF_WIDTH * sn, 0.0)
        })
        .collect()
}

/// The genital rings, outermost first (the attachment loop is ring 0).
#[derive(Clone, Debug, Default)]
pub struct GenitalLoops {
    /// The variant built.
    pub variant: Variant,
    /// The rings from the attachment loop inward (female) or from the loop over the scrotum to the
    /// penis root and along the shaft to the tip (male).
    pub rings: Vec<Vec<u32>>,
}

/// Bridges two closed rings of equal length, `a` used forward, with a tag per face (face `i` is
/// on edge `a[i] -> a[i + 1]`).
fn bridge_tagged(cage: &mut Cage, a: &[u32], b: &[u32], tag: &dyn Fn(usize) -> Tag) {
    let n = a.len().min(b.len());
    for i in 0..n {
        let j = (i + 1) % n;
        cage.face(&[a[i], a[j], b[j], b[i]], tag(i));
    }
}

/// A ring walked the other way from the same first vertex.
fn reversed(ring: &[u32]) -> Vec<u32> {
    ring.iter().take(1).chain(ring.iter().skip(1).rev()).copied().collect()
}

/// Builds the variant's faces inside the attachment loop `ring` (forward order, as returned by
/// [`attachment_loop`]). The new faces are counter-clockwise seen from outside.
pub fn build(cage: &mut Cage, variant: Variant, ring: &[u32]) -> Result<GenitalLoops> {
    if ring.len() != LOOP_LEN {
        return Err(ModelError::invalid("genital attachment loop", ring.len(), "loop must have 12 vertices"));
    }
    let rings = match variant {
        Variant::Female => vulva(cage, ring)?,
        Variant::Male => penis_and_scrotum(cage, ring)?,
    };
    Ok(GenitalLoops { variant, rings })
}

/// One oval ring of the vulva: `s` of its top and bottom ends, half-width, height at the sides,
/// at the top end and at the bottom end (mm), and the region of the faces from the previous ring.
struct Oval {
    top: f32,
    bottom: f32,
    w: f32,
    h: f32,
    h_top: f32,
    h_bottom: f32,
    region: R,
}

const fn oval(top: f32, bottom: f32, w: f32, h: f32, h_top: f32, h_bottom: f32, region: R) -> Oval {
    Oval { top, bottom, w, h, h_top, h_bottom, region }
}

/// The vulva, outside in: the outer slope and the crest of the labia majora (about 10 mm proud),
/// their inner face and the interlabial sulcus, the crest of the labia minora (their top ends
/// meet as the clitoral hood), their inner face over the glans clitoris, the vestibule.
const VULVA: [Oval; 7] = [
    oval(0.04, 0.96, 16.5, 2.5, 1.0, 1.0, R::LabiaMajora),
    oval(0.08, 0.92, 11.0, 6.5, 3.0, 2.5, R::LabiaMajora),
    oval(0.11, 0.89, 5.5, 5.0, 2.0, 1.0, R::LabiaMajora),
    oval(0.13, 0.86, 3.8, -0.5, -0.5, -2.0, R::LabiaMajora),
    oval(0.15, 0.81, 2.0, 2.5, 3.5, -2.0, R::LabiaMinora),
    oval(0.20, 0.77, 2.8, -3.5, 1.0, -5.0, R::LabiaMinora),
    oval(0.25, 0.73, 3.4, -7.0, -4.0, -8.5, R::Vestibule),
];

/// Ring slots of the vulva and of the scrotum.
const RING: usize = 24;

fn oval_point(o: &Oval, j: usize) -> Vec3 {
    let (c, sn, side) = slot(j, RING);
    let s = o.top.midpoint(o.bottom) + 0.5 * (o.bottom - o.top) * c;
    let x = side * o.w * 0.001 * sn.powf(0.75);
    let end = if c > 0.0 { o.h_bottom } else { o.h_top };
    let h = o.h + (end - o.h) * c.abs().powi(3);
    site(s, x, h * 0.001)
}

fn tag(region: R, chart: C) -> Tag {
    Tag::new(region.id(), chart.id())
}

fn vulva(cage: &mut Cage, ring: &[u32]) -> Result<Vec<Vec<u32>>> {
    let mut rings = vec![ring.to_vec()];
    for (k, o) in VULVA.iter().enumerate() {
        let points: Vec<Vec3> = (0..RING).map(|j| oval_point(o, j)).collect();
        let next = cage.add_all(&points);
        let prev = rings.last().cloned().unwrap_or_default();
        // The top ends of the minora's inner face and of the vestibule: hood and glans clitoris.
        // `c` is the cosine of the face's middle angle (mirrored faces get the same value).
        let region = |c: f32| if k >= 4 && c < -0.85 { R::Clitoris } else { o.region };
        if k == 0 {
            cage.bridge_double(&prev, &next, &|i| tag(region(slot(i, RING).0), C::Vulva));
        } else {
            let mid = |i: usize| f32::midpoint(slot(i, RING).0, slot((i + 1) % RING, RING).0);
            bridge_tagged(cage, &prev, &next, &|i| tag(region(mid(i)), C::Vulva));
        }
        rings.push(next);
    }
    // The floor of the vestibule: an 8 x 4 grid whose long side runs along the midline, sinking to
    // the vaginal introitus in its lower half.
    let last = rings.last().cloned().unwrap_or_default();
    let n = normal(0.5);
    cage.cap(&last, 8, 4, 2, tag(R::Vestibule, C::Vulva), &|s, t, p| {
        let across = 1.0 - (2.0 * t - 1.0).powi(2);
        let along = (std::f32::consts::PI * s).sin();
        // s runs from the bottom end (0) to the top end (1): the introitus sits at s ~ 0.3.
        let pit = (-((s - 0.32) / 0.22).powi(2)).exp();
        p - n * (0.002 * across * along + 0.006 * across * pit)
    })?;
    Ok(rings)
}

/// Penis root: `s` of its centre, height of the centre (mm), radius (mm).
const ROOT_S: f32 = 0.2;
const ROOT_H: f32 = 3.0;
/// The scrotum as an ellipsoid the meridians are pulled onto: centre and radii, metres.
const SCROTUM_CENTRE: [f32; 3] = [0.0, 0.792, 0.050];
const SCROTUM_RADII: [f32; 3] = [0.022, 0.032, 0.024];
/// Where the scrotum rings sit along each meridian from the loop (0) to the penis root (1).
const SCROTUM_RINGS: [f32; 5] = [0.10, 0.30, 0.50, 0.70, 0.88];
/// Shaft rings: position along the centreline (0 root, 1 tip), radius (mm), region. The flaccid,
/// uncircumcised penis: the prepuce covers the glans, whose corona shows as a low ridge; the
/// prepuce narrows to the preputial orifice at the tip.
const SHAFT: [(f32, f32, R); 10] = [
    (0.10, 15.2, R::Penis),
    (0.28, 14.9, R::Penis),
    (0.46, 14.7, R::Penis),
    (0.62, 14.5, R::Penis),
    (0.72, 14.1, R::Penis),
    (0.80, 15.3, R::Glans),
    (0.87, 14.6, R::Glans),
    (0.93, 12.0, R::Glans),
    (0.975, 7.0, R::Glans),
    (0.995, 2.6, R::Glans),
];
/// Radius of the root ring, mm (flaccid girth 93 mm).
const ROOT_RADIUS: f32 = 15.5;
/// Slots of the shaft rings.
const SHAFT_SLOTS: usize = 12;

/// The penis centreline: a cubic Bézier from the root, leaving the body forward and down, to the
/// tip hanging in front of the scrotum.
struct Shaft {
    p: [Vec3; 4],
}

impl Shaft {
    fn new() -> Self {
        let root = site(ROOT_S, 0.0, ROOT_H * 0.001);
        // The shaft leaves the skin square to it, then bends down under its weight.
        let leave = normal(ROOT_S);
        let tip = root + Vec3::new(0.0, -0.080, 0.034);
        Self { p: [root, root + leave * 0.028, tip + Vec3::new(0.0, 0.036, 0.0), tip] }
    }

    fn at(&self, t: f32) -> Vec3 {
        let u = 1.0 - t;
        self.p[0] * (u * u * u)
            + self.p[1] * (3.0 * u * u * t)
            + self.p[2] * (3.0 * u * t * t)
            + self.p[3] * (t * t * t)
    }

    fn tangent(&self, t: f32) -> Vec3 {
        let u = 1.0 - t;
        ((self.p[1] - self.p[0]) * (3.0 * u * u)
            + (self.p[2] - self.p[1]) * (6.0 * u * t)
            + (self.p[3] - self.p[2]) * (3.0 * t * t))
            .normalize_or(Vec3::new(0.0, -1.0, 0.0))
    }

    /// Slot `j` of an `n`-slot ring of radius `r` (metres) at `t`: slot 0 ventral (toward the
    /// scrotum), a quarter turn on the left side, half a turn dorsal.
    fn ring_point(&self, t: f32, r: f32, j: usize, n: usize) -> Vec3 {
        let tangent = self.tangent(t);
        let ventral = Vec3::X.cross(tangent).normalize_or(Vec3::new(0.0, 0.0, -1.0));
        let (c, sn, side) = slot(j, n);
        self.at(t) + ventral * (r * c) + Vec3::X * (side * r * sn)
    }
}

/// Length of the penis centreline, metres.
pub fn shaft_length() -> f32 {
    let shaft = Shaft::new();
    let n = 64;
    (0..n).map(|i| shaft.at((i + 1) as f32 / n as f32).distance(shaft.at(i as f32 / n as f32))).sum()
}

fn smoothstep(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// The middle of the meridian at slot `j`: on the scrotal ellipsoid for the lower meridians, just
/// proud of the straight path for the upper ones (the skin above the penis root).
fn meridian_middle(j: usize, from: Vec3, to: Vec3) -> Vec3 {
    let (c, sn, side) = slot(j, RING);
    let mid = (from + to) * 0.5;
    let flat = mid + normal(ROOT_S) * 0.002;
    let dir = Vec3::new(side * 1.15 * sn, -(0.55 + 0.45 * c), -0.25 * c);
    let [rx, ry, rz] = SCROTUM_RADII;
    let scale = 1.0 / ((dir.x / rx).powi(2) + (dir.y / ry).powi(2) + (dir.z / rz).powi(2)).sqrt();
    // The raphe: a shallow groove between the testes along the bottom midline.
    let raphe = 1.0 - 0.12 * (1.0 - smoothstep(sn / 0.35)) * smoothstep(c / 0.5);
    let on = Vec3::from(SCROTUM_CENTRE) + dir * (scale * raphe);
    let w = smoothstep((c + 0.55) / 0.6);
    flat + (on - flat) * w
}

fn penis_and_scrotum(cage: &mut Cage, ring: &[u32]) -> Result<Vec<Vec<u32>>> {
    let shaft = Shaft::new();
    let root_r = ROOT_RADIUS * 0.001;
    let mut rings = vec![ring.to_vec()];
    // Meridians from the designed loop (24 slots) to the root ring.
    let ends: Vec<(Vec3, Vec3, Vec3)> = (0..RING)
        .map(|j| {
            let (c, sn, side) = slot(j, RING);
            let from = site(0.5 + 0.5 * c, side * LOOP_HALF_WIDTH * sn, 0.0);
            let to = shaft.ring_point(0.0, root_r, j, RING);
            let mid = meridian_middle(j, from, to);
            (from, mid * 2.0 - (from + to) * 0.5, to)
        })
        .collect();
    // By the cosine of the face's middle angle: the lower faces are the scrotum, the upper ones the
    // skin around the penis root.
    let region = |c: f32| if c > -0.5 { R::Scrotum } else { R::Genitals };
    let mid = |i: usize| f32::midpoint(slot(i, RING).0, slot((i + 1) % RING, RING).0);
    for (k, &u) in SCROTUM_RINGS.iter().enumerate() {
        let points: Vec<Vec3> = ends
            .iter()
            .map(|&(a, b, c)| a * ((1.0 - u) * (1.0 - u)) + b * (2.0 * u * (1.0 - u)) + c * (u * u))
            .collect();
        let next = cage.add_all(&points);
        let prev = rings.last().cloned().unwrap_or_default();
        if k == 0 {
            cage.bridge_double(&prev, &next, &|i| tag(region(slot(i, RING).0), C::Scrotum));
        } else {
            bridge_tagged(cage, &prev, &next, &|i| tag(region(mid(i)), C::Scrotum));
        }
        rings.push(next);
    }
    // The root ring (12 slots) and the pentagons from the last scrotum ring to it.
    let root: Vec<Vec3> = (0..SHAFT_SLOTS).map(|j| shaft.ring_point(0.0, root_r, j, SHAFT_SLOTS)).collect();
    let root = cage.add_all(&root);
    let last = rings.last().cloned().unwrap_or_default();
    // The pentagon tagged with index `i` covers slots `24 - i - 1 ..= 24 - i + 1` of the last ring.
    cage.bridge_double(&reversed(&root), &reversed(&last), &|i| {
        tag(region(slot((RING - i) % RING, RING).0), C::Scrotum)
    });
    rings.push(root);
    for &(t, r, region) in &SHAFT {
        let points: Vec<Vec3> = (0..SHAFT_SLOTS).map(|j| shaft.ring_point(t, r * 0.001, j, SHAFT_SLOTS)).collect();
        let next = cage.add_all(&points);
        let prev = rings.last().cloned().unwrap_or_default();
        bridge_tagged(cage, &prev, &next, &|_| tag(region, C::Penis));
        rings.push(next);
    }
    // The preputial orifice: a small pit closing the tip.
    let last = rings.last().cloned().unwrap_or_default();
    let axis = shaft.tangent(1.0);
    cage.cap(&last, 3, 3, 0, tag(R::Glans, C::Penis), &|s, t, p| {
        let w = (std::f32::consts::PI * s).sin() * (std::f32::consts::PI * t).sin();
        p - axis * (0.0025 * w)
    })?;
    Ok(rings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adr_039_site_and_loop_are_mirror_exact() {
        let lp = attachment_loop();
        assert_eq!(lp.len(), LOOP_LEN);
        for j in 1..LOOP_LEN {
            let (p, q) = (lp[j], lp[LOOP_LEN - j]);
            assert!(p.x == -q.x && p.y == q.y && p.z == q.z, "{j}: {p:?} {q:?}");
        }
        assert!(lp[0].x == 0.0 && lp[6].x == 0.0);
        // Top below the mons, bottom in front of the perineum, left side at +X.
        assert!(lp[6].y > 0.85 && lp[0].y < 0.82 && lp[3].x > 0.015);
        let len = site_length();
        assert!((0.08..0.12).contains(&len), "{len}");
    }

    #[test]
    fn adr_039_penis_length_and_girth_match_the_sources() {
        // Veale et al. 2015: flaccid length 91.6 mm, girth 93.1 mm.
        let len = shaft_length();
        assert!((len - 0.0916).abs() < 0.006, "{len}");
        let girth = TAU * ROOT_RADIUS;
        assert!((girth - 93.1).abs() < 5.0, "{girth}");
    }
}
