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
pub const LOOP_HALF_WIDTH: f32 = 0.023;
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
/// [`attachment_loop`]). `from[k]` is a body point outside the loop across from loop vertex `k`
/// (the cage vertex the body surface comes from); the variant's first ring continues the body
/// surface across the crease loop from there ([`continuation`]). The new faces are
/// counter-clockwise seen from outside.
pub fn build(cage: &mut Cage, variant: Variant, ring: &[u32], from: &[Vec3]) -> Result<GenitalLoops> {
    if ring.len() != LOOP_LEN || from.len() != LOOP_LEN {
        return Err(ModelError::invalid("genital attachment loop", ring.len(), "loop must have 12 vertices"));
    }
    let first = continuation(cage, ring, from);
    let rings = match variant {
        Variant::Female => vulva(cage, ring, &first)?,
        Variant::Male => penis_and_scrotum(cage, ring, &first)?,
    };
    Ok(GenitalLoops { variant, rings })
}

/// How far inside the attachment loop the variant's first ring lies, metres.
const CONTINUATION: f32 = 0.004;

/// The variant's first ring (24 slots, slot `2k` next to loop vertex `k`): the loop points moved
/// [`CONTINUATION`] further along the direction the body surface crosses the loop in (from
/// `from[k]` to the loop vertex, square to the loop). The loop is a crease, so the subdivided
/// body never sees the variant; with the surface on both sides running the same way across it,
/// the crease does not show as an edge. Mirror-exact: every value is computed from sums and
/// differences that are exact mirrors of their mirror slot's.
fn continuation(cage: &Cage, ring: &[u32], from: &[Vec3]) -> Vec<Vec3> {
    let n = ring.len();
    let at = |k: usize| cage.pos[ring[k % n] as usize];
    let across: Vec<Vec3> = (0..n)
        .map(|k| {
            let d = at(k) - from[k];
            let t = at(k + 1) - at(k + n - 1);
            let d = d - t * (d.dot(t) / t.dot(t).max(1e-12));
            d.normalize_or(Vec3::Y)
        })
        .collect();
    (0..2 * n)
        .map(|j| {
            let k = j / 2;
            if j % 2 == 0 {
                at(k) + across[k] * CONTINUATION
            } else {
                let k1 = (k + 1) % n;
                (at(k) + at(k1)) * 0.5 + (across[k] + across[k1]).normalize_or(across[k]) * CONTINUATION
            }
        })
        .collect()
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

/// The vulva inside the first ring ([`continuation`]), outside in: the outer slope and the crest
/// of the labia majora (about 7 mm proud), their inner faces closing the pudendal cleft to a slit
/// (the standing, closed posture), the interlabial sulcus, the crest of the labia minora (their
/// top ends meet as the clitoral hood), their inner face over the glans clitoris, the vestibule.
const VULVA: [Oval; 7] = [
    oval(0.075, 0.93, 14.0, 3.0, 1.5, 1.0, R::LabiaMajora),
    oval(0.10, 0.91, 10.0, 7.0, 3.0, 2.5, R::LabiaMajora),
    oval(0.12, 0.89, 4.0, 5.0, 2.0, 1.0, R::LabiaMajora),
    oval(0.13, 0.86, 2.6, -0.5, -0.5, -2.0, R::LabiaMajora),
    oval(0.15, 0.81, 1.6, 2.0, 3.5, -2.0, R::LabiaMinora),
    oval(0.20, 0.77, 2.4, -3.5, 1.0, -5.0, R::LabiaMinora),
    oval(0.25, 0.73, 3.0, -7.0, -4.0, -8.5, R::Vestibule),
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

fn vulva(cage: &mut Cage, ring: &[u32], first: &[Vec3]) -> Result<Vec<Vec<u32>>> {
    let mut rings = vec![ring.to_vec()];
    let first = cage.add_all(first);
    cage.bridge_double(ring, &first, &|_| tag(R::LabiaMajora, C::Vulva));
    rings.push(first);
    for (k, o) in VULVA.iter().enumerate() {
        let points: Vec<Vec3> = (0..RING).map(|j| oval_point(o, j)).collect();
        let next = cage.add_all(&points);
        let prev = rings.last().cloned().unwrap_or_default();
        // The top ends of the minora's inner face and of the vestibule: hood and glans clitoris.
        // `c` is the cosine of the face's middle angle (mirrored faces get the same value).
        let region = |c: f32| if k >= 4 && c < -0.85 { R::Clitoris } else { o.region };
        let mid = |i: usize| f32::midpoint(slot(i, RING).0, slot((i + 1) % RING, RING).0);
        bridge_tagged(cage, &prev, &next, &|i| tag(region(mid(i)), C::Vulva));
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
const ROOT_S: f32 = 0.32;
const ROOT_H: f32 = -1.0;
/// The scrotum as an ellipsoid the meridians are pulled onto: centre and radii, metres.
const SCROTUM_CENTRE: [f32; 3] = [0.0, 0.768, 0.064];
const SCROTUM_RADII: [f32; 3] = [0.026, 0.036, 0.026];
/// Where the scrotum rings sit along each meridian from the loop (0) to the penis root (1).
const SCROTUM_RINGS: [f32; 7] = [0.08, 0.20, 0.35, 0.50, 0.65, 0.80, 0.92];
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
/// Radius of the root ring, mm (girth 91 mm; flaccid mid-shaft girth 93 mm). It must stay inside
/// the first ring around it, or the skin between them folds over.
const ROOT_RADIUS: f32 = 14.5;
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

/// The point of the scrotal ellipsoid in direction `dir` from its centre, with the raphe: a
/// shallow groove between the testes along the lower midline.
fn on_scrotum(dir: Vec3) -> Vec3 {
    let d = dir.normalize_or(Vec3::new(0.0, -1.0, 0.0));
    let [rx, ry, rz] = SCROTUM_RADII;
    let scale = 1.0 / ((d.x / rx).powi(2) + (d.y / ry).powi(2) + (d.z / rz).powi(2)).sqrt();
    let raphe = 1.0 - 0.10 * (1.0 - smoothstep(d.x.abs() / 0.3)) * smoothstep(-d.y / 0.6);
    Vec3::from(SCROTUM_CENTRE) + d * (scale * raphe)
}

/// The meridian at slot `j` from `start` (first ring) to `to` (root ring) at `u` in `[0, 1]`: a
/// cubic that leaves `start` along `leave` and runs just proud of the straight path (the skin
/// above the penis root), pulled for the lower slots onto the scrotal ellipsoid, which it follows
/// from the back over the side or the bottom to the front (the sac around the testes).
fn meridian(j: usize, start: Vec3, leave: Vec3, to: Vec3, u: f32) -> Vec3 {
    let (c, sn, side) = slot(j, RING);
    let mid = (start + to) * 0.5 + normal(ROOT_S) * 0.002;
    // A short handle: long ones fold the skin over at the upper slots, next to the root.
    let p1 = start + leave * (0.2 * start.distance(mid).min(start.distance(to)));
    let p2 = (mid * 8.0 - start - p1 * 3.0 - to) / 3.0;
    let v = 1.0 - u;
    let flat = start * (v * v * v) + p1 * (3.0 * v * v * u) + p2 * (3.0 * v * u * u) + to * (u * u * u);
    let centre = Vec3::from(SCROTUM_CENTRE);
    let (a, b) = ((start - centre).normalize_or(Vec3::Y), (to - centre).normalize_or(Vec3::Y));
    let m = Vec3::new(side * 1.4 * sn, -(0.55 + 0.45 * c), -0.25 * c).normalize_or(Vec3::new(0.0, -1.0, 0.0));
    let sac = on_scrotum(a * (v * v) + m * (4.0 * u * v) + b * (u * u));
    let w = smoothstep((c + 0.55) / 0.6) * smoothstep(u / 0.22) * smoothstep(v / 0.3);
    flat + (sac - flat) * w
}

/// Fairing passes over the scrotum rings, and how far each pass moves a point toward the mean of
/// its four neighbours.
const SAC_FAIRING: usize = 10;
const SAC_LAMBDA: f32 = 0.5;

/// Fairs the scrotum rings `grid` (outermost first, [`RING`] slots each) between the first ring
/// `inner` and the root ring `outer` (both at [`RING`] slots): smooths away the folds where the
/// meridians leave the attachment and meet the sac, on the lower slots only (the skin above the
/// penis root keeps its designed shape). Mirror-exact: the side neighbours enter as one sum.
fn fair_sac(grid: &mut [Vec<Vec3>], inner: &[Vec3], outer: &[Vec3]) {
    let n = grid.len();
    for _ in 0..SAC_FAIRING {
        let prev = grid.to_vec();
        for (k, row) in grid.iter_mut().enumerate() {
            for (j, p) in row.iter_mut().enumerate() {
                let w = smoothstep((slot(j, RING).0 + 0.55) / 0.6);
                if w == 0.0 {
                    continue;
                }
                let before = if k == 0 { inner[j] } else { prev[k - 1][j] };
                let after = if k + 1 == n { outer[j] } else { prev[k + 1][j] };
                let side = prev[k][(j + 1) % RING] + prev[k][(j + RING - 1) % RING];
                let mean = ((before + after) + side) * 0.25;
                *p += (mean - *p) * (SAC_LAMBDA * w);
            }
        }
    }
}

fn penis_and_scrotum(cage: &mut Cage, ring: &[u32], first: &[Vec3]) -> Result<Vec<Vec<u32>>> {
    let shaft = Shaft::new();
    let root_r = ROOT_RADIUS * 0.001;
    let mut rings = vec![ring.to_vec()];
    let first_ring = cage.add_all(first);
    // Meridians ([`meridian`]) from the first ring, leaving it the way the body crosses the loop,
    // to the root ring.
    let ends: Vec<(Vec3, Vec3, Vec3)> = (0..RING)
        .map(|j| {
            let (c, sn, side) = slot(j, RING);
            let on_loop = site(0.5 + 0.5 * c, side * LOOP_HALF_WIDTH * sn, 0.0);
            let leave = (first[j] - on_loop).normalize_or(normal(0.5));
            (first[j], leave, shaft.ring_point(0.0, root_r, j, RING))
        })
        .collect();
    // By the cosine of the face's middle angle: the lower faces are the scrotum, the upper ones the
    // skin around the penis root.
    let region = |c: f32| if c > -0.5 { R::Scrotum } else { R::Genitals };
    let mid = |i: usize| f32::midpoint(slot(i, RING).0, slot((i + 1) % RING, RING).0);
    cage.bridge_double(ring, &first_ring, &|i| tag(region(slot(i, RING).0), C::Scrotum));
    rings.push(first_ring);
    let mut grid: Vec<Vec<Vec3>> = SCROTUM_RINGS
        .iter()
        .map(|&u| ends.iter().enumerate().map(|(j, &(start, leave, to))| meridian(j, start, leave, to, u)).collect())
        .collect();
    let roots: Vec<Vec3> = ends.iter().map(|e| e.2).collect();
    fair_sac(&mut grid, first, &roots);
    for points in &grid {
        let next = cage.add_all(points);
        let prev = rings.last().cloned().unwrap_or_default();
        bridge_tagged(cage, &prev, &next, &|i| tag(region(mid(i)), C::Scrotum));
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
