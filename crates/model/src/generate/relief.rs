//! Surface anatomy of the starter template (ADR-039, todo 1.3): the bony and muscular landmarks a
//! lean adult of average build shows under the skin, applied as a small displacement along the
//! normal of the subdivided body surface.
//!
//! Each feature is a smooth bump or groove around a path (a point, or a polyline of up to four
//! points) measured in the plane across the direction it faces, so its anchor points need not lie
//! exactly on the skin. Across the path the two sides have their own radius, so a border can fall
//! off steeply on one side and fade out slowly on the other (the lower edge of the pectoral, the
//! costal margin, the medial border of the scapula); beyond the path's ends the feature rounds off
//! with its own end radius. The profile is `h (1 - s)^2` with `s` the squared normalised distance:
//! flat at the path and at the rim, so no feature leaves a crease. A feature acts only on skin
//! facing its direction (the normal within about 70 degrees of it) and, where a depth is given,
//! only within that distance of its first point along the facing direction.
//!
//! **Why the forms are broad (wave 10).** The torso of the last subdivision level has edges of
//! 10 to 16 mm (median 12.4 mm, diagnostic `diag_relief_folds_and_spacing`). The first table used
//! capsules of 6 to 14 mm radius, which the mesh can only draw as one displaced row of vertices:
//! the pectoral border, the costal margin and the clavicles read as drawn lines, and two capsules
//! meeting end to end (the spinal furrow at T12) added up to twice the depth at the joint. Every
//! torso feature now reaches at least 16 mm on its steep side and rises at most about 3.6 mm per
//! 10 mm (test `torso_relief_is_broad_and_gentle`), joined segments are one polyline, and no
//! off-midline feature reaches the midline with a slope (test `relief_has_no_crease_on_the_midline`).
//!
//! **Placement.** Heights above the floor follow the template's own landmarks (suprasternale
//! 1.384 m, xiphoid 1.225 m, navel 1.018 m, nipple 1.233 m, ASIS 0.97 m) and the bones and
//! muscles of BodyParts3D 4.0 (CC BY 4.0, "BodyParts3D, (c) The Database Center for Life Science
//! licensed under CC Attribution 4.0 International"; a 1.719 m male, scaled by 1.692 / 1.719; read
//! with `target/agents/template-body/scripts/bp3d_landmarks.py`, nothing of its mesh is copied):
//! manubrium top 1.381 m, xiphoid tip 1.222 m, clavicle from x 8..24 mm at 1.37..1.39 m to its
//! lateral end at x 142 mm, 1.407 m; lowest point of the sternocostal pectoralis at x 74 mm,
//! 1.200 m, its border rising to the axilla at x 180 mm, 1.29 m; scapula medial border at x 56 mm,
//! inferior angle at x 92 mm, 1.242 m; sternocleidomastoid from x 13 mm at the manubrium to the
//! mastoid at x 56 mm; iliac crest highest at x 103 mm, 1.019 m, tubercle at x 141 mm, 0.988 m.
//! Which borders are steep and which fade follows the surface-anatomy chapters of Gray's Anatomy
//! (41st ed., 2016: thorax, back, abdomen) and Peck, *Atlas of Human Anatomy for the Artist*
//! (1951); heights are design values for a lean, not athletic, build: 0.7 to 4.8 mm on the torso.
//!
//! All features are authored for the left side and evaluated on the mirror image of right-side
//! points, so the surface stays exactly mirror-symmetric.

use humanart_math::{Vec2, Vec3};

use super::anatomy::{Anatomy, Auricle};
use super::cage::Cage;
use super::limbs::{TOE_WIDTHS, finger_joints, foot_point, thumb_joints, toe_joints};
use super::regions::R;

/// Most points in one feature's path.
const MAX_PATH: usize = 4;

/// A bump or groove of the surface.
#[derive(Clone, Copy, Debug)]
struct Feature {
    /// The direction the skin it acts on faces (left side, unit).
    facing: Vec3,
    /// The path: `len` points (one for a blob).
    path: [Vec3; MAX_PATH],
    /// Height at each path point (metres; negative is a groove), linear between them.
    height: [f32; MAX_PATH],
    /// Number of path points (1 to [`MAX_PATH`]).
    len: usize,
    /// A direction that marks the "toward" side of the path (projected into the plane).
    toward: Vec3,
    /// Radius across the path on the `toward` side and on the other side (metres).
    across: (f32, f32),
    /// Radius beyond the path's ends (metres).
    end: f32,
    /// Only within this distance of the first path point along `facing` (metres); `None`:
    /// everywhere.
    depth: Option<f32>,
    /// Exponent of the normalised distance in the profile `h (1 - s^k)^2`: 1 is a dome, higher
    /// values a plateau with a steeper rim (the nail plates).
    flat: i32,
}

const FRONT: Vec3 = Vec3::Z;
const BACK: Vec3 = Vec3::new(0.0, 0.0, -1.0);
const LATERAL: Vec3 = Vec3::X;
const MEDIAL: Vec3 = Vec3::new(-1.0, 0.0, 0.0);
const UP: Vec3 = Vec3::Y;

/// A feature along `points` (metres) with `heights` (mm) at them; radii in metres.
fn path(facing: Vec3, points: &[[f32; 3]], heights: &[f32], toward: Vec3, across: (f32, f32), end: f32) -> Feature {
    let mut f = Feature {
        facing: facing.normalize(),
        path: [Vec3::ZERO; MAX_PATH],
        height: [0.0; MAX_PATH],
        len: points.len().clamp(1, MAX_PATH),
        toward,
        across,
        end,
        depth: None,
        flat: 1,
    };
    for (k, (p, h)) in points.iter().zip(heights).take(MAX_PATH).enumerate() {
        f.path[k] = Vec3::from(*p);
        f.height[k] = h * 0.001;
    }
    f
}

/// A symmetric capsule from `a` to `b` of radius `r`, heights in mm.
fn ridge(facing: Vec3, a: [f32; 3], b: [f32; 3], r: f32, h0: f32, h1: f32) -> Feature {
    path(facing, &[a, b], &[h0, h1], UP, (r, r), r)
}

/// An ellipse at `c` with radii `rx` (horizontal in the plane) and `ry`, height in mm.
fn blob(facing: Vec3, c: [f32; 3], rx: f32, ry: f32, h: f32) -> Feature {
    path(facing, &[c], &[h], UP, (ry, ry), rx)
}

/// An egg at `c`: radius `rx` sideways, `up` above and `down` below, height in mm.
fn lobe(facing: Vec3, c: [f32; 3], rx: f32, up: f32, down: f32, h: f32) -> Feature {
    path(facing, &[c], &[h], UP, (up, down), rx)
}

impl Feature {
    fn within(mut self, depth: f32) -> Self {
        self.depth = Some(depth);
        self
    }

    /// A plateau profile of exponent `k` (see [`Feature::flat`]).
    fn plateau(mut self, k: i32) -> Self {
        self.flat = k.max(1);
        self
    }

    /// The in-plane axes across `facing`: horizontal-ish and vertical-ish.
    fn axes(&self) -> (Vec3, Vec3) {
        let up = if self.facing.y.abs() > 0.9 { Vec3::Z } else { Vec3::Y };
        let u = up.cross(self.facing).normalize_or(Vec3::X);
        (u, self.facing.cross(u))
    }

    /// The squared normalised distance of the in-plane point `p` from segment `k` (path point `k`
    /// to `k + 1`, or the single point) and the parameter along the segment.
    fn segment(&self, p: Vec2, k: usize, flat: &impl Fn(Vec3) -> Vec2) -> (f32, f32) {
        let a = flat(self.path[k]);
        let b = if k + 1 < self.len { flat(self.path[k + 1]) } else { a };
        let ab = b - a;
        let len = ab.length();
        let dir = if len > 1e-6 { ab / len } else { Vec2::X };
        let mut across = Vec2::new(-dir.y, dir.x);
        if across.dot(flat(self.toward)) < 0.0 {
            across = -across;
        }
        let pa = p - a;
        let x = pa.dot(dir);
        let beyond = x - x.clamp(0.0, len);
        let q = pa.dot(across);
        let r = if q >= 0.0 { self.across.0 } else { self.across.1 };
        let s = (beyond / self.end).powi(2) + (q / r).powi(2);
        (s, if len > 1e-6 { (x / len).clamp(0.0, 1.0) } else { 0.0 })
    }

    /// The displacement at left-side point `p` with left-side unit normal `n`.
    fn at(&self, p: Vec3, n: Vec3) -> f32 {
        let facing = smoothstep((n.dot(self.facing) - 0.15) / 0.45);
        if facing <= 0.0 {
            return 0.0;
        }
        let depth_w = match self.depth {
            Some(d) => {
                let along = (p - self.path[0]).dot(self.facing).abs();
                if along > d + 0.01 {
                    return 0.0;
                }
                1.0 - smoothstep((along - d) / 0.01)
            }
            None => 1.0,
        };
        let (s, t, k) = self.nearest(p);
        if s >= 1.0 {
            return 0.0;
        }
        let h0 = self.height[k];
        let h1 = if k + 1 < self.len { self.height[k + 1] } else { h0 };
        let shape = 1.0 - s.powi(self.flat);
        (h0 + (h1 - h0) * t) * shape * shape * facing * depth_w
    }

    /// The squared normalised in-plane distance of `p` from the path (below 1 inside the
    /// feature's outline), the parameter along the nearest segment and that segment.
    fn nearest(&self, p: Vec3) -> (f32, f32, usize) {
        let (u, v) = self.axes();
        let flat = |q: Vec3| Vec2::new(q.dot(u), q.dot(v));
        let fp = flat(p);
        let mut best = (f32::INFINITY, 0.0, 0);
        for k in 0..self.len.saturating_sub(1).max(1) {
            let (s, t) = self.segment(fp, k, &flat);
            if s < best.0 {
                best = (s, t, k);
            }
        }
        best
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
        blob(BACK, [0.0, 1.452, 0.0], 0.016, 0.022, 2.2)
    } else {
        blob(BACK, [0.0, 1.452, 0.0], 0.012, 0.012, 4.4)
    }
}

/// The broad forms of the neck, chest, abdomen and back (left side), checked by the test
/// `torso_relief_is_broad_and_gentle`.
///
/// **Heights (round 2).** The first broad table (0.4 to 2.6 mm) read almost featureless in studio
/// light. These are the reliefs a lean adult shows in a photograph (Peck 1951, plates of the
/// torso; Gray's 41st ed., surface anatomy of the thorax and back): the clavicle stands about
/// 4 mm over the hollows above and below it, the jugular notch sinks about 5 mm between the
/// sternal heads, the lower pectoral border is a 4 mm step, the lumbar spine groove lies 8 to
/// 9 mm below the crests of the erector columns, and the medial border of the scapula stands
/// about 4 mm off the back. Each is still at least 18 mm wide on its steep side, so the 12 mm
/// torso mesh draws it as a form, not a line.
fn torso_features() -> Vec<Feature> {
    let fl = (LATERAL + FRONT * 1.7).normalize();
    let neck = (LATERAL * 0.75 + FRONT * 0.66).normalize();
    let above_clavicle = (UP * 0.6 + FRONT * 0.8).normalize();
    let medial_up = Vec3::new(-1.0, 1.0, 0.0);
    let lateral_up = Vec3::new(1.0, 1.0, 0.0);
    vec![
        // The jugular notch: a broad hollow between the sternal heads of the two
        // sternocleidomastoids, reaching further up the neck than down onto the manubrium.
        lobe(FRONT, [0.0, 1.386, 0.0], 0.026, 0.034, 0.020, -4.6),
        // Sternocleidomastoid, sternal head: from beside the notch up and back toward the
        // mastoid. It stops 18 mm under the scanned face's mask (`scan_face::FACE_BOTTOM`,
        // 1.443 m): the face model was measured on this surface and must not move.
        path(
            neck,
            &[[0.030, 1.397, 0.044], [0.036, 1.412, 0.036], [0.042, 1.425, 0.026]],
            &[3.0, 2.8, 1.6],
            UP,
            (0.019, 0.019),
            0.016,
        ),
        // Clavicle: the bone's front from the sternoclavicular joint to its lateral third, the
        // lower edge a little steeper than the upper; the supraclavicular hollow above it and
        // the infraclavicular (deltopectoral) hollow under its lateral third.
        path(
            FRONT,
            &[[0.024, 1.388, 0.0], [0.080, 1.393, 0.0], [0.138, 1.405, 0.0]],
            &[4.4, 4.0, 2.0],
            UP,
            (0.020, 0.019),
            0.019,
        ),
        blob(above_clavicle, [0.066, 1.407, 0.028], 0.028, 0.018, -3.2).within(0.04),
        lobe(FRONT, [0.120, 1.366, 0.0], 0.024, 0.018, 0.024, -2.4),
        // Pectoralis major: the mass, then its lower border, steep below and fading upward,
        // lowest under the nipple and rising into the anterior axillary fold.
        lobe(FRONT, [0.074, 1.285, 0.0], 0.074, 0.090, 0.056, 3.0),
        path(
            FRONT,
            &[[0.030, 1.222, 0.0], [0.085, 1.208, 0.0], [0.126, 1.226, 0.0], [0.150, 1.268, 0.0]],
            &[2.0, 4.4, 3.8, 1.2],
            medial_up,
            (0.056, 0.020),
            0.022,
        ),
        // The sternum between the two muscles: a furrow down to the xiphoid.
        ridge(FRONT, [0.0, 1.362, 0.0], [0.0, 1.228, 0.0], 0.024, -2.2, -2.6),
        // Costal margin: the rib cage's lower edge from beside the xiphoid down and out,
        // standing over the softer abdomen; the epigastric hollow under the xiphoid.
        path(
            FRONT,
            &[[0.036, 1.200, 0.0], [0.076, 1.150, 0.0], [0.112, 1.098, 0.0]],
            &[1.6, 2.6, 2.0],
            lateral_up,
            (0.040, 0.020),
            0.020,
        ),
        blob(FRONT, [0.0, 1.182, 0.0], 0.032, 0.032, -1.6),
        // Rectus abdominis: the linea alba above the navel, the fullness of the muscle, its
        // lateral border (linea semilunaris); the soft lower belly.
        ridge(FRONT, [0.0, 1.196, 0.0], [0.0, 1.042, 0.0], 0.018, -1.2, -0.7),
        lobe(FRONT, [0.040, 1.110, 0.0], 0.030, 0.070, 0.090, 1.0),
        ridge(FRONT, [0.077, 1.168, 0.0], [0.072, 1.000, 0.0], 0.020, -1.1, -0.8),
        lobe(FRONT, [0.0, 0.982, 0.0], 0.090, 0.050, 0.045, 1.4),
        // Anterior superior iliac spine and the inguinal groove from it down to the pubis.
        blob(fl, [0.110, 0.972, 0.080], 0.022, 0.024, 3.0),
        ridge(FRONT, [0.100, 0.952, 0.0], [0.036, 0.880, 0.0], 0.019, -2.8, -1.4),
        // Iliac crest on the flank, rising from the tubercle toward the back.
        ridge(LATERAL, [0.15, 0.998, 0.06], [0.15, 1.024, -0.06], 0.022, 2.6, 2.0).within(0.05),
        // Back: the spinal furrow from the nape to the sacrum, deepest in the lumbar region where
        // the erector spinae columns stand either side of it; the columns fade out upward over
        // the lower thoracic spine.
        path(
            BACK,
            &[[0.0, 1.430, 0.0], [0.0, 1.250, 0.0], [0.0, 1.120, 0.0], [0.0, 0.990, 0.0]],
            &[-1.6, -3.0, -4.4, -4.8],
            UP,
            (0.021, 0.021),
            0.022,
        ),
        path(
            BACK,
            &[[0.030, 1.310, 0.0], [0.034, 1.180, 0.0], [0.036, 0.995, 0.0]],
            &[1.0, 3.0, 3.8],
            LATERAL,
            (0.026, 0.024),
            0.026,
        ),
        // Scapula: the plate, its medial border down to the inferior angle (steep toward the
        // spine), the spine of the scapula out toward the acromion (low: it lies under the
        // trapezius and must not bump the shoulder's outline).
        lobe(BACK, [0.100, 1.312, 0.0], 0.052, 0.062, 0.072, 3.0),
        path(
            BACK,
            &[[0.060, 1.370, 0.0], [0.072, 1.300, 0.0], [0.090, 1.250, 0.0]],
            &[2.6, 3.8, 4.2],
            LATERAL,
            (0.044, 0.019),
            0.020,
        ),
        ridge(BACK, [0.064, 1.362, 0.0], [0.140, 1.388, 0.0], 0.019, 1.4, 0.6),
        // The dimples over the posterior superior iliac spines.
        blob(BACK, [0.045, 0.990, 0.0], 0.020, 0.020, -2.4),
    ]
}

/// The creases that are lines in life too (left side): the natal cleft and the gluteal fold.
fn creases() -> Vec<Feature> {
    vec![
        ridge(BACK, [0.0, 0.955, 0.0], [0.0, 0.835, 0.0], 0.012, -4.4, -9.9),
        ridge(BACK, [0.015, 0.800, 0.0], [0.130, 0.795, 0.0], 0.012, -6.6, -2.2),
    ]
}

/// Knee, hand and foot (left side): the limbs' finer mesh draws smaller forms.
fn limb_features(a: &Anatomy) -> Vec<Feature> {
    let mut f = Vec::new();
    // Knee: patella, patellar ligament with the hollows beside it, vastus medialis.
    let k = a.knee;
    let kz = k.z + 0.055;
    f.extend([
        blob(FRONT, [k.x, k.y + 0.024, kz], 0.024, 0.027, 6.6).within(0.03),
        ridge(FRONT, [k.x, k.y - 0.006, kz], [k.x, k.y - 0.050, kz], 0.008, 2.64, 1.76).within(0.03),
        blob(FRONT, [k.x - 0.024, k.y - 0.006, kz], 0.012, 0.012, -3.08).within(0.03),
        blob(FRONT, [k.x + 0.024, k.y - 0.006, kz], 0.012, 0.012, -3.08).within(0.03),
        blob(FRONT, [k.x - 0.024, k.y + 0.070, kz], 0.025, 0.035, 3.08).within(0.04),
    ]);
    f.extend(hand_features(a));
    f.extend(foot_features(a));
    f
}

/// The unit direction `up` made perpendicular to the unit direction `along`.
fn across_of(up: Vec3, along: Vec3) -> Vec3 {
    (up - along * up.dot(along)).normalize_or(up)
}

/// The nail of one digit (left side): its plate as a low plateau along the distal phalanx, the
/// lunula at the plate's root, and the digit's frame there. The plate and the lunula are also the
/// outlines [`tag_details`] gives their regions.
#[derive(Clone, Copy, Debug)]
struct Nail {
    /// The nail plate: a plateau about 0.6 mm high with a rounded rim.
    plate: Feature,
    /// The lunula: the pale half-moon at the root of the plate (a region only, no relief).
    lunula: Feature,
    /// The proximal nail fold: the skin ridge over the plate's root.
    fold: Feature,
    /// The pulp of the digit's tip on the other side (the finger or toe pad).
    pad: Feature,
    /// A finger or thumb (`true`) or a toe.
    hand: bool,
}

/// A nail on the distal segment `base` (the last joint) to `tip`: `up` is the dorsal side,
/// `half_width` the digit's half width there and `thickness` the distance from the axis to the
/// dorsal skin. The plate covers the distal 60 % of the segment, a little narrower than the
/// digit (Zaias, *The Nail in Health and Disease*, 1990: the plate is about as long as the free
/// part of the distal phalanx and covers about two thirds of the digit's width).
fn nail(base: Vec3, tip: Vec3, up: Vec3, half_width: f32, thickness: f32, hand: bool) -> Nail {
    let len = (tip - base).length().max(1e-4);
    let along = (tip - base) / len;
    let dorsal = across_of(up, along);
    let side = along.cross(dorsal).normalize_or(Vec3::X);
    let on = |t: f32, lift: f32| -> [f32; 3] { (base + along * (t * len) + dorsal * (thickness * lift)).into() };
    let hw = half_width * if hand { 0.64 } else { 0.66 };
    let (start, stop) = if hand { (0.36, 0.90) } else { (0.30, 0.86) };
    let plate = path(dorsal, &[on(start, 1.0), on(stop, 0.92)], &[0.6, 0.6], side, (hw, hw), 0.0024)
        .within(thickness * 0.8)
        .plateau(4);
    let lunula = path(
        dorsal,
        &[on(start + 0.02, 1.0), on(start + 0.12, 1.0)],
        &[0.0, 0.0],
        side,
        (hw * 0.72, hw * 0.72),
        0.0022,
    )
    .within(thickness * 0.8);
    let fold_at = start - 0.0024 / len;
    let fold = path(
        dorsal,
        &[
            (Vec3::from(on(fold_at, 1.0)) - side * (hw * 0.9)).into(),
            (Vec3::from(on(fold_at, 1.0)) + side * (hw * 0.9)).into(),
        ],
        &[0.45, 0.45],
        along,
        (0.0020, 0.0020),
        0.0020,
    )
    .within(thickness * 0.8);
    let pad = path(
        -dorsal,
        &[on(0.40, -1.0), on(0.70, -1.0)],
        &[0.9, 0.9],
        side,
        (half_width * 0.8, half_width * 0.8),
        0.006,
    )
    .within(thickness * 0.9);
    Nail { plate, lunula, fold, pad, hand }
}

/// The nails of the left hand (thumb first) and foot (big toe first).
fn nails(a: &Anatomy) -> Vec<Nail> {
    let palmar = a.hand_axes[1];
    let mut list = Vec::with_capacity(10);
    let thumb = thumb_joints(a, 1.0);
    if let [.., ip, tip] = thumb[..] {
        list.push(nail(ip, tip, -palmar, 0.0090, 0.0080, true));
    }
    for f in 0..4 {
        let scale = [0.97, 1.0, 0.94, 0.83][f];
        if let [.., dip, tip] = finger_joints(a, f, 1.0)[..] {
            list.push(nail(dip, tip, -palmar, 0.0079 * scale, 0.0071 * scale, true));
        }
    }
    for (t, &w) in TOE_WIDTHS.iter().enumerate() {
        if let [.., mid, tip] = toe_joints(a, t, 1.0)[..] {
            list.push(nail(mid, tip, Vec3::Y, w * 0.92, w * 0.76, false));
        }
    }
    list
}

/// The hand (left side): the head of the ulna, the knuckles of the metacarpal heads with the
/// hollows between them and the extensor tendons, the finger joints on the back, the palm's
/// eminences, hollow and creases, the finger creases on the palmar side, the nails and pads.
///
/// The palm creases follow the usual pattern (Kimura 1991; Gray's 41st ed., the hand): the thenar
/// crease curving round the ball of the thumb from the radial border to the wrist, the proximal
/// transverse crease from the same start obliquely across the palm, the distal transverse crease
/// from the ulnar border to between the index and middle fingers. They are grooves about 1.3 mm deep
/// and 8 mm wide (4 mm either side of the line), the narrowest the palm's 5 mm mesh draws.
fn hand_features(a: &Anatomy) -> Vec<Feature> {
    let [_, palmar, radial] = a.hand_axes;
    let dorsal = -palmar;
    let w = a.wrist;
    let hp = |r: f32, d: f32, z: f32| -> [f32; 3] { a.hand_point([r, d, z]).into() };
    let mut f = Vec::new();
    let ulnar = (dorsal * 0.6 - radial * 0.8).normalize();
    let head = w - radial * 0.020 + dorsal * 0.010;
    f.push(blob(ulnar, head.into(), 0.008, 0.008, 4.4).within(0.02));
    let joints: Vec<Vec<Vec3>> = (0..4).map(|i| finger_joints(a, i, 1.0)).collect();
    for (finger, chain) in joints.iter().enumerate() {
        let scale = [0.97, 1.0, 0.94, 0.83][finger];
        let [mcp, pip, dip, tip] = chain[..] else { continue };
        // The metacarpal head and the extensor tendon running to it.
        let top = mcp + dorsal * 0.011;
        f.push(blob(dorsal, top.into(), 0.0085, 0.0095, 3.4).within(0.012));
        let start = w + dorsal * 0.014 + radial * (0.006 - 0.004 * finger as f32);
        f.push(ridge(dorsal, start.into(), (top - (top - start) * 0.15).into(), 0.003, 0.6, 1.3).within(0.012));
        // The hollow between this knuckle and the next, running into the web.
        if let Some(&next) = joints.get(finger + 1).and_then(|c| c.first()) {
            let mid = (mcp + next) * 0.5 + dorsal * 0.009 + (pip - mcp).normalize_or_zero() * 0.006;
            f.push(blob(dorsal, mid.into(), 0.005, 0.009, -1.6).within(0.012));
        }
        // The finger joints on the back: the head of the proximal and middle phalanges, with
        // the slack skin over them in the relaxed, flexed hand: a band of shallow creases.
        let d_prox = across_of(dorsal, (pip - mcp).normalize_or_zero());
        let d_mid = across_of(dorsal, (dip - pip).normalize_or_zero());
        let d_dist = across_of(dorsal, (tip - dip).normalize_or_zero());
        let d_pip = (d_prox + d_mid).normalize_or(dorsal);
        let d_dip = (d_mid + d_dist).normalize_or(dorsal);
        let mid_dir = (dip - pip).normalize_or_zero();
        let pip_top = pip + d_pip * (0.0088 * scale);
        f.push(blob(d_pip, pip_top.into(), 0.0055 * scale, 0.0055 * scale, 1.2).within(0.006));
        let side = mid_dir.cross(d_pip).normalize_or(radial);
        for (offset, depth) in [(-0.0045, -0.45), (0.0040, -0.40)] {
            let c = pip_top + mid_dir * (offset * scale);
            let (l, r) = (c - side * (0.0055 * scale), c + side * (0.0055 * scale));
            f.push(path(d_pip, &[l.into(), r.into()], &[depth, depth], mid_dir, (0.0026, 0.0026), 0.003).within(0.006));
        }
        let dip_top = dip + d_dip * (0.0075 * scale);
        f.push(blob(d_dip, dip_top.into(), 0.0045 * scale, 0.0045 * scale, 0.8).within(0.006));
        // The palmar flexion creases of the two finger joints.
        for (joint, up, r) in [(pip, d_pip, 0.0088), (dip, d_dip, 0.0075)] {
            let c = joint - up * (r * scale);
            let side = mid_dir.cross(up).normalize_or(radial);
            let (l, rr) = (c - side * (0.006 * scale), c + side * (0.006 * scale));
            f.push(path(-up, &[l.into(), rr.into()], &[-0.7, -0.7], mid_dir, (0.0022, 0.0022), 0.0025).within(0.006));
        }
    }
    // The palm: the thenar and hypothenar eminences, the hollow between them, the creases.
    f.extend([
        lobe(palmar, hp(0.020, 0.040, 0.014), 0.014, 0.020, 0.022, 1.6).within(0.012),
        lobe(palmar, hp(-0.029, 0.045, 0.012), 0.011, 0.025, 0.025, 1.2).within(0.012),
        blob(palmar, hp(-0.003, 0.058, 0.012), 0.015, 0.015, -1.4).within(0.012),
        path(
            palmar,
            &[hp(0.030, 0.074, 0.012), hp(0.016, 0.062, 0.013), hp(0.008, 0.042, 0.014), hp(0.008, 0.020, 0.014)],
            &[-0.8, -1.4, -1.3, -0.7],
            radial,
            (0.0042, 0.0042),
            0.0040,
        )
        .within(0.012),
        path(
            palmar,
            &[hp(0.030, 0.075, 0.012), hp(0.012, 0.070, 0.012), hp(-0.010, 0.066, 0.012), hp(-0.026, 0.061, 0.012)],
            &[-0.8, -1.3, -1.2, -0.4],
            radial,
            (0.0042, 0.0042),
            0.0040,
        )
        .within(0.012),
        path(
            palmar,
            &[hp(-0.041, 0.076, 0.010), hp(-0.022, 0.079, 0.011), hp(-0.002, 0.083, 0.011), hp(0.012, 0.090, 0.010)],
            &[-0.5, -1.3, -1.3, -0.5],
            radial,
            (0.0042, 0.0042),
            0.0040,
        )
        .within(0.012),
        ridge(palmar, hp(-0.022, 0.006, 0.014), hp(0.020, 0.006, 0.014), 0.0026, -0.6, -0.6).within(0.012),
    ]);
    for n in nails(a).iter().filter(|n| n.hand) {
        f.extend([n.plate, n.fold, n.pad]);
    }
    f
}

/// The foot (left side): the malleoli, the Achilles tendon with the hollows beside it, the
/// longitudinal arch lifting the medial sole off the floor, the groove between the ball and the
/// toes, the toe nails and pads.
fn foot_features(a: &Anatomy) -> Vec<Feature> {
    let n = a.ankle;
    let mut f = vec![
        blob(LATERAL, [n.x + 0.030, n.y, n.z], 0.015, 0.017, 5.2).within(0.03),
        blob(MEDIAL, [n.x - 0.028, n.y + 0.010, n.z + 0.006], 0.015, 0.017, 4.8).within(0.03),
        ridge(BACK, [n.x, n.y + 0.100, n.z - 0.035], [n.x, n.y - 0.010, n.z - 0.045], 0.010, 2.2, 3.2).within(0.03),
        ridge(BACK, [n.x + 0.020, n.y + 0.060, n.z - 0.035], [n.x + 0.020, n.y - 0.005, n.z - 0.04], 0.011, -3.2, -3.2)
            .within(0.03),
        ridge(BACK, [n.x - 0.020, n.y + 0.060, n.z - 0.035], [n.x - 0.020, n.y - 0.005, n.z - 0.04], 0.011, -3.2, -3.2)
            .within(0.03),
    ];
    let fp = |m: f32, l: f32, h: f32| -> [f32; 3] { foot_point(a, 1.0, m, l, h).into() };
    let dir = a.foot_dir;
    let medial = dir.cross(Vec3::Y);
    let down_medial = (medial * 0.55 - Vec3::Y).normalize();
    // The medial longitudinal arch: the sole between the heel and the ball lifts off the floor on
    // the medial side, most under the navicular (Gray's 41st ed., the foot: the medial arch is
    // highest at the talus and navicular, the lateral border rests on the ground).
    f.push(
        path(
            down_medial,
            &[fp(0.000, 0.030, 0.006), fp(0.040, 0.034, 0.006), fp(0.075, 0.034, 0.006), fp(0.095, 0.032, 0.006)],
            &[-1.5, -6.0, -5.5, -1.5],
            -medial,
            (0.030, 0.020),
            0.020,
        )
        .within(0.03),
    );
    // The sulcus between the ball of the foot and the toe pads.
    f.push(
        path(
            -Vec3::Y,
            &[fp(0.170, 0.045, 0.004), fp(0.166, 0.005, 0.003), fp(0.150, -0.040, 0.004)],
            &[-1.2, -1.4, -1.0],
            dir,
            (0.004, 0.005),
            0.005,
        )
        .within(0.012),
    );
    for n in nails(a).iter().filter(|n| !n.hand) {
        f.extend([n.plate, n.fold, n.pad]);
    }
    f
}

/// The fine regions of the hands and feet on the final surface (todo 1.3): the nail plates and
/// their lunulae on the fingers and toes, and the arch of each sole (the instep, whose skin is
/// thinner than that of the heel and the ball). Faces keep their chart; only the region changes.
fn tag_details(mesh: &mut Cage, a: &Anatomy) {
    let nails = nails(a);
    let dir = a.foot_dir;
    let medial = dir.cross(Vec3::Y);
    let foot_base = Vec3::new(a.ankle.x, 0.0, a.ankle.z);
    for f in 0..mesh.face_count() {
        let Some(region) = R::from_id(mesh.tags[f].region) else { continue };
        let digit = matches!(
            region,
            R::HandLThumb
                | R::HandRThumb
                | R::HandLIndex
                | R::HandRIndex
                | R::HandLMiddle
                | R::HandRMiddle
                | R::HandLRing
                | R::HandRRing
                | R::HandLLittle
                | R::HandRLittle
        );
        let toe = matches!(region, R::FootLToes | R::FootRToes);
        let sole = matches!(region, R::FootLSole | R::FootRSole);
        if !(digit || toe || sole) {
            continue;
        }
        let c = mesh.corners_of(f);
        let o = mesh.pos[c[0] as usize];
        let mut normal = Vec3::ZERO;
        let mut centre = Vec3::ZERO;
        for k in 0..c.len() {
            let (p, q) = (mesh.pos[c[k] as usize] - o, mesh.pos[c[(k + 1) % c.len()] as usize] - o);
            normal += p.cross(q);
            centre += mesh.pos[c[k] as usize];
        }
        centre /= c.len().max(1) as f32;
        let side = if centre.x >= 0.0 { 1.0 } else { -1.0 };
        let p = Vec3::new(centre.x * side, centre.y, centre.z);
        let n = normal.normalize_or_zero();
        let n = Vec3::new(n.x * side, n.y, n.z);
        let tag = &mut mesh.tags[f].region;
        if sole {
            let m = (p - foot_base).dot(dir);
            let l = (p - foot_base).dot(medial);
            if (0.020..0.100).contains(&m) && l > 0.008 {
                *tag = R::sided(R::FootLArch, R::FootRArch, side).id();
            }
            continue;
        }
        for nl in nails.iter().filter(|nl| nl.hand == digit) {
            let up = nl.plate.facing;
            let along = (p - nl.plate.path[0]).dot(up);
            if n.dot(up) < 0.45 || along.abs() > 0.004 || nl.plate.nearest(p).0 >= 0.85 {
                continue;
            }
            let lunula = nl.lunula.nearest(p).0 < 1.0;
            *tag = match (digit, lunula) {
                (true, true) => R::sided(R::HandLLunula, R::HandRLunula, side),
                (true, false) => R::sided(R::HandLNail, R::HandRNail, side),
                (false, _) => R::sided(R::FootLNail, R::FootRNail, side),
            }
            .id();
            break;
        }
    }
}

/// The feature list (left side).
fn features(a: &Anatomy) -> Vec<Feature> {
    let mut f = torso_features();
    f.push(c7(a));
    f.extend(creases());
    f.extend(limb_features(a));
    f
}

/// The small details the relief carries along instead of reshaping: a vertex of any face of
/// these regions moves along the mean normal of its detail (0: none).
fn detail_of(region: R) -> usize {
    match region {
        R::AreolaL | R::NippleL => 1,
        R::AreolaR | R::NippleR => 2,
        R::Navel => 3,
        _ => 0,
    }
}

/// Displaces the body surface `mesh` (the last subdivision level) by the surface-anatomy
/// features of `a`. The genital faces (ADR-039) are left alone and do not count toward the
/// normals of the attachment loop, so the body moves the same with either variant. The nipples,
/// areolae and the navel take the field at their own points but the mean normal of their detail
/// (for the facing test and the direction): their millimetre-sized walls face every way, and
/// their own normals would tear them apart (wave 10: folds around the nipples). Last, the nails,
/// lunulae and sole arches get their regions ([`tag_details`]).
pub fn apply(mesh: &mut Cage, a: &Anatomy) {
    let mut normals = vec![Vec3::ZERO; mesh.pos.len()];
    let mut detail = vec![0_usize; mesh.pos.len()];
    for f in 0..mesh.face_count() {
        let region = R::from_id(mesh.tags[f].region);
        if region.is_some_and(R::is_genital) {
            continue;
        }
        let c = mesh.corners_of(f);
        let mut n = Vec3::ZERO;
        for k in 0..c.len() {
            let (p, q) = (mesh.pos[c[k] as usize], mesh.pos[c[(k + 1) % c.len()] as usize]);
            n += Vec3::new((p.y - q.y) * (p.z + q.z), (p.z - q.z) * (p.x + q.x), (p.x - q.x) * (p.y + q.y));
        }
        let d = region.map_or(0, detail_of);
        for &v in c {
            normals[v as usize] += n;
            if d > 0 {
                detail[v as usize] = d;
            }
        }
    }
    let mut detail_normal = [Vec3::ZERO; 4];
    for (n, &d) in normals.iter().zip(&detail) {
        detail_normal[d] += *n;
    }
    let detail_normal = detail_normal.map(Vec3::normalize_or_zero);
    let list = features(a);
    for ((p, n), &d) in mesh.pos.iter_mut().zip(&normals).zip(&detail) {
        let n = if d > 0 { detail_normal[d] } else { n.normalize_or_zero() };
        let side = if p.x >= 0.0 { 1.0 } else { -1.0 };
        let (lp, ln) = (Vec3::new(p.x * side, p.y, p.z), Vec3::new(n.x * side, n.y, n.z));
        let h: f32 = list.iter().map(|f| f.at(lp, ln)).sum();
        *p += n * h;
    }
    tag_details(mesh, a);
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
        // Every authored feature stays within 10 mm (the creases and the ankle are the deepest).
        for f in features(&Anatomy::neutral()) {
            assert!(f.height.iter().all(|h| h.abs() <= 0.010_01));
        }
    }

    #[test]
    fn borders_fall_off_on_their_own_side_and_polylines_join_smoothly() {
        // An edge along y = 1 from x 0 to 0.1 and on to (0.2, 1.1): wide above, steep below.
        let f =
            path(FRONT, &[[0.0, 1.0, 0.0], [0.1, 1.0, 0.0], [0.2, 1.1, 0.0]], &[2.0, 2.0, 2.0], UP, (0.05, 0.02), 0.02);
        let at = |x: f32, y: f32| f.at(Vec3::new(x, y, 0.1), Vec3::Z);
        assert!(at(0.05, 1.03) > 0.0005, "the wide side reaches 30 mm");
        assert!(at(0.05, 0.97).abs() < 1e-9, "the steep side ends at 20 mm");
        // No doubling where two segments meet: the joint has the path's height.
        assert!((at(0.1, 1.0) - 0.002).abs() < 1e-6);
        // Continuous along lines crossing the path, the joint and the ends.
        for (x0, y0, dx, dy) in
            [(0.05, 0.95, 0.0, 1.0), (0.1, 0.95, 0.0, 1.0), (-0.05, 1.0, 1.0, 0.0), (0.06, 1.06, 1.0, -1.0)]
        {
            let mut last = at(x0, y0);
            for i in 1..=2000 {
                let t = i as f32 * 0.0001;
                let v = at(x0 + dx * t, y0 + dy * t);
                assert!((v - last).abs() < 4e-5, "jump at ({}, {})", x0 + dx * t, y0 + dy * t);
                last = v;
            }
        }
    }

    /// The torso features must be broad and gentle enough for the 10 to 16 mm edges of the
    /// subdivided torso (wave 10: the narrow table read as drawn lines): at least 16 mm on every
    /// side (18 mm for anything over 3 mm high), at most 5 mm high, and a steepest slope
    /// `1.54 h / r` of at most 0.36 (20 degrees).
    #[test]
    fn torso_relief_is_broad_and_gentle() {
        for (i, f) in torso_features().iter().enumerate() {
            let r = f.across.0.min(f.across.1).min(f.end);
            let h = f.height[..f.len].iter().fold(0.0_f32, |m, h| m.max(h.abs()));
            assert!(r >= 0.016 - 1e-6, "feature {i}: radius {r}");
            assert!(h <= 0.005 + 1e-6, "feature {i}: height {h}");
            assert!(h <= 0.003 + 1e-6 || r >= 0.018 - 1e-6, "feature {i}: {h} m high on a {r} m radius");
            assert!(1.54 * h / r <= 0.36, "feature {i}: slope {}", 1.54 * h / r);
        }
    }

    /// The right side is the mirror image of the left, so a feature that reaches the midline
    /// with a slope across it would fold the surface into a crease there: every feature is
    /// flat across x = 0 (one-sided slope, Richardson-extrapolated, under 0.005).
    #[test]
    fn relief_has_no_crease_on_the_midline() {
        let a = Anatomy::neutral();
        for (i, f) in features(&a).iter().enumerate() {
            let z = f.path[0].z;
            for k in 0..=800 {
                let y = 0.70 + k as f32 * 0.001;
                let v = |x: f32| f64::from(f.at(Vec3::new(x, y, z), f.facing));
                let e = 0.0005_f32;
                let d1 = (v(2.0 * e) - v(0.0)) / f64::from(2.0 * e);
                let d2 = (v(e) - v(0.0)) / f64::from(e);
                let slope = 2.0 * d2 - d1;
                assert!(slope.abs() < 0.005, "feature {i} at y {y}: slope {slope}");
            }
        }
    }
    /// How far each hand and foot feature moves the finished surface (mm): the largest
    /// displacement over the vertices of the left side, with the count of vertices it reaches.
    #[test]
    #[ignore = "diagnostic"]
    fn diag_limb_feature_reach() {
        let s = super::super::body_surface().unwrap();
        let mesh = s.mesh();
        let mut normals = vec![Vec3::ZERO; mesh.pos.len()];
        for f in 0..mesh.face_count() {
            let c = mesh.corners_of(f);
            let o = mesh.pos[c[0] as usize];
            let mut n = Vec3::ZERO;
            for k in 0..c.len() {
                n += (mesh.pos[c[k] as usize] - o).cross(mesh.pos[c[(k + 1) % c.len()] as usize] - o);
            }
            for &v in c {
                normals[v as usize] += n;
            }
        }
        let list: Vec<Feature> = hand_features(&s.anatomy).into_iter().chain(foot_features(&s.anatomy)).collect();
        for (i, f) in list.iter().enumerate() {
            let (mut most, mut count) = (0.0_f32, 0);
            for (p, n) in mesh.pos.iter().zip(&normals) {
                if p.x <= 0.0 {
                    continue;
                }
                let h = f.at(*p, n.normalize_or_zero());
                if h.abs() > 1e-6 {
                    count += 1;
                }
                most = most.max(h.abs());
            }
            println!("feature {i}: up to {:.2} mm on {count} vertices (authored {:.2})", most * 1e3, f.height[0] * 1e3);
        }
    }
}
