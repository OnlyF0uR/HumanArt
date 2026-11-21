//! Surface anatomy of the starter template (ADR-039): the bony and muscular landmarks a nude adult
//! of average build shows under the skin, applied as a small displacement along the normal of the
//! subdivided body surface.
//!
//! Each feature is a smooth bump or groove: a capsule (a segment with a radius) or an ellipse,
//! measured in the plane across the direction it faces, so its anchor points need not lie exactly
//! on the skin. A feature acts only on skin facing its direction (the normal within about 70
//! degrees of it) and, where a depth is given, only within that distance of its anchor along the
//! facing direction. Heights are a few millimetres: an average build, not a bodybuilder. All
//! features are authored for the left side and evaluated on the mirror image of right-side
//! points, so the surface stays exactly mirror-symmetric.
//!
//! Positions are design values read off standard surface-anatomy references (clavicle, pectoral
//! border, costal margin, linea alba and semilunaris, iliac crest and spines, spine furrow,
//! scapular borders, gluteal fold, patella, malleoli, Achilles tendon, ulnar head, knuckles): they
//! are **unsourced** numbers, placed relative to [`Anatomy`]'s joints and the torso sections.

use humanart_math::{Vec2, Vec3};

use super::anatomy::{Anatomy, Auricle};
use super::cage::Cage;
use super::limbs::finger_joints;
use super::regions::R;

/// A bump or groove of the surface.
#[derive(Clone, Copy, Debug)]
struct Feature {
    /// The direction the skin it acts on faces (left side, unit).
    facing: Vec3,
    /// Segment start (a point for an ellipse).
    a: Vec3,
    /// Segment end (equal to `a` for an ellipse).
    b: Vec3,
    /// Radii across: (along the in-plane horizontal axis, along the in-plane vertical axis) for an
    /// ellipse; both the capsule radius for a segment.
    radii: Vec2,
    /// Height at `a` and at `b` (metres; negative is a groove).
    height: (f32, f32),
    /// Only within this distance of `a` along `facing` (metres); `None`: everywhere.
    depth: Option<f32>,
}

/// Scale of every authored height: the table is written for soft tissue over the forms, this
/// brings them to what an average adult shows in raking light.
pub const STRENGTH: f32 = 2.2;

const FRONT: Vec3 = Vec3::Z;
const BACK: Vec3 = Vec3::new(0.0, 0.0, -1.0);
const LATERAL: Vec3 = Vec3::X;
const MEDIAL: Vec3 = Vec3::new(-1.0, 0.0, 0.0);

/// A capsule from `a` to `b` (x, y; z only matters with a depth) of radius `r`, heights in mm.
fn ridge(facing: Vec3, a: [f32; 3], b: [f32; 3], r: f32, h0: f32, h1: f32) -> Feature {
    Feature {
        facing: facing.normalize(),
        a: Vec3::from(a),
        b: Vec3::from(b),
        radii: Vec2::splat(r),
        height: (h0 * 0.001, h1 * 0.001),
        depth: None,
    }
}

/// An ellipse at `c` with radii `rx`, `ry` across `facing`, height in mm.
fn blob(facing: Vec3, c: [f32; 3], rx: f32, ry: f32, h: f32) -> Feature {
    Feature {
        facing: facing.normalize(),
        a: Vec3::from(c),
        b: Vec3::from(c),
        radii: Vec2::new(rx, ry),
        height: (h * 0.001, h * 0.001),
        depth: None,
    }
}

impl Feature {
    fn within(mut self, depth: f32) -> Self {
        self.depth = Some(depth);
        self
    }

    /// The in-plane axes across `facing`: horizontal-ish and vertical-ish.
    fn axes(&self) -> (Vec3, Vec3) {
        let up = if self.facing.y.abs() > 0.9 { Vec3::Z } else { Vec3::Y };
        let u = up.cross(self.facing).normalize_or(Vec3::X);
        (u, self.facing.cross(u))
    }

    /// The displacement at left-side point `p` with left-side unit normal `n`.
    fn at(&self, p: Vec3, n: Vec3) -> f32 {
        let facing = smoothstep((n.dot(self.facing) - 0.15) / 0.45);
        if facing <= 0.0 {
            return 0.0;
        }
        if let Some(depth) = self.depth {
            let along = (p - self.a).dot(self.facing).abs();
            if along > depth + 0.01 {
                return 0.0;
            }
        }
        let depth_w = self.depth.map_or(1.0, |d| 1.0 - smoothstep(((p - self.a).dot(self.facing).abs() - d) / 0.01));
        let (u, v) = self.axes();
        let flat = |q: Vec3| Vec2::new(q.dot(u), q.dot(v));
        let (pa, ab) = (flat(p - self.a), flat(self.b - self.a));
        let len2 = ab.length_squared();
        let t = if len2 > 0.0 { (pa.dot(ab) / len2).clamp(0.0, 1.0) } else { 0.0 };
        let d = pa - ab * t;
        let s = (d / self.radii).length_squared();
        if s >= 1.0 {
            return 0.0;
        }
        let profile = (1.0 - s) * (1.0 - s);
        let h = self.height.0 + (self.height.1 - self.height.0) * t;
        h * profile * facing * depth_w
    }
}

fn smoothstep(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// The spinous process of C7 (vertebra prominens) at the base of the nape. The fitting base keeps
/// its knob (4.4 mm high, 12 mm round: `build::BASE_VERSION`); the finished head
/// ([`Auricle::Finished`], todo 1.1) has the low, broad bump a lean adult shows with the head
/// upright (2.2 mm over 16 x 22 mm), which reads as part of the nape's outline instead of a knob on
/// it in profile.
fn c7(a: &Anatomy) -> Feature {
    if a.auricle == Auricle::Finished {
        blob(BACK, [0.0, 1.452, 0.0], 0.016, 0.022, 1.0)
    } else {
        blob(BACK, [0.0, 1.452, 0.0], 0.012, 0.012, 2.0)
    }
}

/// The feature list (left side).
fn features(a: &Anatomy) -> Vec<Feature> {
    let fl = (LATERAL + FRONT * 1.7).normalize();
    let mut f = vec![
        // Clavicle, the hollow below its lateral third, the jugular notch.
        ridge(FRONT, [0.022, 1.383, 0.0], [0.150, 1.400, 0.0], 0.011, 3.0, 2.2),
        blob(FRONT, [0.128, 1.362, 0.0], 0.018, 0.013, -1.4),
        blob(FRONT, [0.0, 1.391, 0.0], 0.016, 0.012, -3.0),
        // Pectoral mass, its lower border, the sternal furrow between the two.
        blob(FRONT, [0.072, 1.29, 0.0], 0.075, 0.055, 2.4),
        ridge(FRONT, [0.035, 1.213, 0.0], [0.130, 1.236, 0.0], 0.013, -0.8, -1.0),
        ridge(FRONT, [0.0, 1.372, 0.0], [0.0, 1.215, 0.0], 0.014, -1.0, -1.3),
        // Costal margin and the epigastric hollow under the xiphoid.
        ridge(FRONT, [0.010, 1.200, 0.0], [0.105, 1.125, 0.0], 0.013, 0.9, 1.1),
        blob(FRONT, [0.0, 1.185, 0.0], 0.025, 0.020, -1.0),
        // Rectus abdominis: linea alba, two tendinous intersections, linea semilunaris; the soft
        // lower belly.
        ridge(FRONT, [0.0, 1.180, 0.0], [0.0, 1.070, 0.0], 0.009, -1.0, -0.8),
        ridge(FRONT, [0.0, 1.020, 0.0], [0.0, 0.930, 0.0], 0.010, -0.5, -0.2),
        ridge(FRONT, [0.012, 1.145, 0.0], [0.058, 1.150, 0.0], 0.006, -0.45, -0.35),
        ridge(FRONT, [0.012, 1.097, 0.0], [0.060, 1.099, 0.0], 0.006, -0.40, -0.30),
        ridge(FRONT, [0.076, 1.170, 0.0], [0.070, 1.000, 0.0], 0.012, -0.6, -0.5),
        blob(FRONT, [0.0, 0.985, 0.0], 0.085, 0.045, 1.4),
        // Anterior superior iliac spine and the inguinal groove down to the pubis.
        blob(fl, [0.112, 0.975, 0.0], 0.013, 0.013, 1.5),
        ridge(FRONT, [0.104, 0.955, 0.0], [0.035, 0.877, 0.0], 0.012, -2.0, -1.2),
        // Iliac crest on the flank.
        ridge(LATERAL, [0.15, 1.030, 0.06], [0.15, 1.045, -0.07], 0.014, 1.2, 1.0).within(0.05),
        // Back: the spinal furrow (deeper in the lumbar region), C7, the scapulae (plate, medial
        // border, spine, inferior angle), the dimples over the posterior iliac spines, the natal
        // cleft and the gluteal fold.
        ridge(BACK, [0.0, 1.440, 0.0], [0.0, 1.120, 0.0], 0.013, -1.2, -2.2),
        ridge(BACK, [0.0, 1.120, 0.0], [0.0, 0.990, 0.0], 0.016, -2.2, -2.6),
        c7(a),
        blob(BACK, [0.095, 1.31, 0.0], 0.055, 0.070, 2.0),
        ridge(BACK, [0.060, 1.370, 0.0], [0.076, 1.235, 0.0], 0.010, 1.3, 1.4),
        ridge(BACK, [0.065, 1.366, 0.0], [0.160, 1.395, 0.0], 0.010, 1.5, 1.1),
        blob(BACK, [0.080, 1.230, 0.0], 0.014, 0.014, 1.0),
        blob(BACK, [0.042, 0.985, 0.0], 0.011, 0.011, -2.0),
        ridge(BACK, [0.0, 0.955, 0.0], [0.0, 0.835, 0.0], 0.012, -2.0, -4.5),
        ridge(BACK, [0.015, 0.800, 0.0], [0.130, 0.795, 0.0], 0.012, -3.0, -1.0),
    ];
    // Knee: patella, patellar ligament with the hollows beside it, vastus medialis.
    let k = a.knee;
    let kz = k.z + 0.055;
    f.extend([
        blob(FRONT, [k.x, k.y + 0.024, kz], 0.024, 0.027, 3.0).within(0.03),
        ridge(FRONT, [k.x, k.y - 0.006, kz], [k.x, k.y - 0.050, kz], 0.008, 1.2, 0.8).within(0.03),
        blob(FRONT, [k.x - 0.024, k.y - 0.006, kz], 0.012, 0.012, -1.4).within(0.03),
        blob(FRONT, [k.x + 0.024, k.y - 0.006, kz], 0.012, 0.012, -1.4).within(0.03),
        blob(FRONT, [k.x - 0.024, k.y + 0.070, kz], 0.025, 0.035, 1.4).within(0.04),
    ]);
    // Ankle: malleoli (the medial one higher and further forward), Achilles tendon and the
    // hollows beside it.
    let n = a.ankle;
    f.extend([
        blob(LATERAL, [n.x + 0.030, n.y, n.z], 0.011, 0.012, 3.0).within(0.03),
        blob(MEDIAL, [n.x - 0.028, n.y + 0.010, n.z + 0.006], 0.011, 0.012, 2.8).within(0.03),
        ridge(BACK, [n.x, n.y + 0.100, n.z - 0.035], [n.x, n.y - 0.010, n.z - 0.045], 0.009, 1.0, 1.6).within(0.03),
        ridge(BACK, [n.x + 0.020, n.y + 0.060, n.z - 0.035], [n.x + 0.020, n.y - 0.005, n.z - 0.04], 0.009, -1.8, -1.8)
            .within(0.03),
        ridge(BACK, [n.x - 0.020, n.y + 0.060, n.z - 0.035], [n.x - 0.020, n.y - 0.005, n.z - 0.04], 0.009, -1.8, -1.8)
            .within(0.03),
    ]);
    // Wrist and hand: the head of the ulna, the knuckles and the extensor tendons on the back of
    // the hand.
    let [_, palmar, radial] = a.hand_axes;
    let dorsal = -palmar;
    let w = a.wrist;
    let ulnar = (dorsal * 0.6 - radial * 0.8).normalize();
    let head = w - radial * 0.020 + dorsal * 0.010;
    f.push(blob(ulnar, head.into(), 0.008, 0.008, 2.0).within(0.02));
    for finger in 0..4 {
        let Some(&mcp) = finger_joints(a, finger, 1.0).first() else { continue };
        let top = mcp + dorsal * 0.011;
        f.push(blob(dorsal, top.into(), 0.008, 0.008, 1.4).within(0.012));
        let start = w + dorsal * 0.014 + radial * (0.006 - 0.004 * finger as f32);
        f.push(ridge(dorsal, start.into(), (top - (top - start) * 0.15).into(), 0.003, 0.2, 0.5).within(0.012));
    }
    f
}

/// Displaces the body surface `mesh` (the last subdivision level) by the surface-anatomy
/// features of `a`. The genital faces (ADR-039) are left alone and do not count toward the
/// normals of the attachment loop, so the body moves the same with either variant.
pub fn apply(mesh: &mut Cage, a: &Anatomy) {
    let mut normals = vec![Vec3::ZERO; mesh.pos.len()];
    for f in 0..mesh.face_count() {
        if R::from_id(mesh.tags[f].region).is_some_and(R::is_genital) {
            continue;
        }
        let c = mesh.corners_of(f);
        let mut n = Vec3::ZERO;
        for k in 0..c.len() {
            let (p, q) = (mesh.pos[c[k] as usize], mesh.pos[c[(k + 1) % c.len()] as usize]);
            n += Vec3::new((p.y - q.y) * (p.z + q.z), (p.z - q.z) * (p.x + q.x), (p.x - q.x) * (p.y + q.y));
        }
        for &v in c {
            normals[v as usize] += n;
        }
    }
    let list = features(a);
    for (p, n) in mesh.pos.iter_mut().zip(&normals) {
        let n = n.normalize_or_zero();
        let side = if p.x >= 0.0 { 1.0 } else { -1.0 };
        let (lp, ln) = (Vec3::new(p.x * side, p.y, p.z), Vec3::new(n.x * side, n.y, n.z));
        let h: f32 = list.iter().map(|f| f.at(lp, ln)).sum();
        *p += n * (h * STRENGTH);
    }
}

/// Taubin fairing passes over the side of the face ([`fair_cheeks`]).
const FAIR_ITERATIONS: usize = 200;

/// How much of the side of the face [`fair_cheeks`] smooths at a head-frame point (mm): the cheek
/// and the side of the face between the lower lid and the mouth corner, away from the eye, the
/// nose, the mouth and the ear.
fn cheek_mask(h: Vec3, eye: Vec3) -> f32 {
    let x = h.x.abs();
    let lateral = smoothstep((x - 34.0) / 8.0);
    let band = smoothstep((h.y + 40.0) / 8.0) * (1.0 - smoothstep((h.y - 8.0) / 8.0));
    let front_of_ear = smoothstep((h.z - 2.0) / 10.0);
    let e = Vec3::new(x, h.y, h.z) - eye;
    let off_eye = smoothstep((e.length() - 20.0) / 6.0);
    lateral * band * front_of_ear * off_eye
}

/// Fairs the side of the face of the body surface `mesh` (the last subdivision level): Taubin
/// smoothing (shrink-free) weighted by [`cheek_mask`]. The authored head rows meet with slightly
/// different slopes between the lower lid and the cheekbone, which the subdivision keeps as a
/// faint horizontal band from the nose to the ear; this removes it without moving the features.
pub fn fair_cheeks(mesh: &mut Cage, a: &Anatomy) {
    let n = mesh.pos.len();
    let mut neighbours: Vec<Vec<u32>> = vec![Vec::new(); n];
    for f in 0..mesh.face_count() {
        let c = mesh.corners_of(f);
        for k in 0..c.len() {
            let (p, q) = (c[k], c[(k + 1) % c.len()]);
            for (x, y) in [(p, q), (q, p)] {
                let list = &mut neighbours[x as usize];
                if !list.contains(&y) {
                    list.push(y);
                }
            }
        }
    }
    let eye = (a.eye_center - a.head_origin) * 1000.0;
    let weights: Vec<f32> = mesh.pos.iter().map(|&p| cheek_mask((p - a.head_origin) * 1000.0, eye)).collect();
    let active: Vec<usize> = (0..n).filter(|&v| weights[v] > 0.0).collect();
    let mut next = mesh.pos.clone();
    for it in 0..FAIR_ITERATIONS {
        let step = if it % 2 == 0 { 0.6 } else { -0.62 };
        for &v in &active {
            let nb = &neighbours[v];
            if nb.is_empty() {
                continue;
            }
            let mean = nb.iter().fold(Vec3::ZERO, |s, &u| s + mesh.pos[u as usize]) / nb.len() as f32;
            next[v] = mesh.pos[v] + (mean - mesh.pos[v]) * (step * weights[v]);
        }
        for &v in &active {
            mesh.pos[v] = next[v];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_are_smooth_bounded_and_face_their_side() {
        let f = ridge(FRONT, [0.0, 1.0, 0.0], [0.1, 1.0, 0.0], 0.01, -2.0, -2.0);
        // Full depth on the axis facing the feature, nothing outside the radius or facing away.
        assert!((f.at(Vec3::new(0.05, 1.0, 0.1), Vec3::Z) + 0.002).abs() < 1e-6);
        assert!(f.at(Vec3::new(0.05, 1.011, 0.1), Vec3::Z).abs() < 1e-9);
        assert!(f.at(Vec3::new(0.05, 1.0, -0.1), -Vec3::Z).abs() < 1e-9);
        // Continuous: a small step moves the value a little.
        let (x, y) = (f.at(Vec3::new(0.05, 1.005, 0.1), Vec3::Z), f.at(Vec3::new(0.05, 1.0051, 0.1), Vec3::Z));
        assert!((x - y).abs() < 5e-5);
        // Depth limits.
        let g = blob(FRONT, [0.0, 0.0, 0.0], 0.01, 0.01, 1.0).within(0.01);
        assert!(g.at(Vec3::new(0.0, 0.0, 0.005), Vec3::Z) > 0.0009);
        assert!(g.at(Vec3::new(0.0, 0.0, 0.05), Vec3::Z).abs() < 1e-9);
        // Every authored feature stays within 5 mm (11 mm with the strength).
        for f in features(&Anatomy::neutral()) {
            assert!(f.height.0.abs() <= 0.005_01 && f.height.1.abs() <= 0.005_01);
        }
    }
}
