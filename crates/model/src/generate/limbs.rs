//! Arms with hands and fingers, legs with feet and toes, as ring tubes on the body cage.
//!
//! Every limb starts from a ring that is already part of the cage (the arm socket of the grid, the
//! leg ring made of the hip row and the crotch path) and keeps its slot order. Cross-sections are
//! ellipses with muscle and bone bumps given as angular Gaussians in the limb frame: angle 0 points
//! forward (radial on the forearm, dorsal on the foot), 90 degrees medially (palmar on the forearm,
//! medial on the leg).

use std::f32::consts::PI;

use humanart_math::Vec3;

use super::anatomy::{Anatomy, FINGERS, THUMB, rotate, sided};
use super::cage::{Cage, Tag};
use super::regions::{C, R};
use super::tube::{Frame, match_pattern, mix, ring_angles, smooth, unwrap};
use crate::error::{ModelError, Result};

/// The cage loops of one arm.
#[derive(Clone, Debug, Default)]
pub struct ArmLoops {
    /// Arm rings from the socket to the wrist (the socket ring first).
    pub rings: Vec<Vec<u32>>,
    /// Hand rings from the wrist to the knuckles.
    pub hand: Vec<Vec<u32>>,
    /// Digit rings: thumb, index, middle, ring, little.
    pub digits: Vec<Vec<Vec<u32>>>,
}

/// The cage loops of one leg.
#[derive(Clone, Debug, Default)]
pub struct LegLoops {
    /// Leg rings from the hip to the ankle (the start ring first).
    pub rings: Vec<Vec<u32>>,
    /// Foot rings from the ankle to the toe bases.
    pub foot: Vec<Vec<u32>>,
    /// Toe rings, big toe first.
    pub toes: Vec<Vec<Vec<u32>>>,
}

/// An angular bump: centre (degrees), width (degrees, standard deviation) and height (metres).
type Bump = (f32, f32, f32);

/// The polar radius of an ellipse with semi-axes `a` (angle 0) and `b` (angle 90 degrees), plus bumps.
fn radius(theta: f32, a: f32, b: f32, bumps: &[Bump]) -> f32 {
    let (s, c) = theta.sin_cos();
    let mut r = 1.0 / ((c / a).powi(2) + (s / b).powi(2)).sqrt();
    for &(centre, width, height) in bumps {
        let mut d = theta.to_degrees() - centre;
        d = (d + 540.0).rem_euclid(360.0) - 180.0;
        r += height * (-0.5 * (d / width).powi(2)).exp();
    }
    r
}

/// One arm ring: distance from the shoulder joint along the arm axis, semi-axes, bumps.
struct ArmRing {
    s: f32,
    a: f32,
    b: f32,
    bumps: &'static [Bump],
}

/// The arm rings (upper-arm circumference ~300 mm, forearm ~265 mm, wrist ~165 mm: ANSUR II means
/// averaged over the sexes).
const ARM: [ArmRing; 10] = [
    ArmRing { s: 0.035, a: 0.049, b: 0.050, bumps: &[(270.0, 45.0, 0.008), (0.0, 30.0, 0.003)] },
    ArmRing { s: 0.090, a: 0.047, b: 0.047, bumps: &[(270.0, 50.0, 0.007), (190.0, 30.0, 0.002)] },
    ArmRing { s: 0.150, a: 0.045, b: 0.043, bumps: &[(10.0, 35.0, 0.003), (190.0, 35.0, 0.004), (275.0, 30.0, 0.002)] },
    ArmRing { s: 0.205, a: 0.044, b: 0.040, bumps: &[(5.0, 35.0, 0.005), (185.0, 40.0, 0.003)] },
    ArmRing { s: 0.258, a: 0.039, b: 0.038, bumps: &[(5.0, 30.0, 0.003)] },
    ArmRing { s: 0.302, a: 0.036, b: 0.041, bumps: &[(180.0, 20.0, 0.004), (95.0, 20.0, 0.003)] },
    ArmRing { s: 0.348, a: 0.042, b: 0.040, bumps: &[(330.0, 35.0, 0.004), (130.0, 40.0, 0.003)] },
    ArmRing { s: 0.405, a: 0.041, b: 0.034, bumps: &[(320.0, 35.0, 0.003), (120.0, 40.0, 0.002)] },
    ArmRing { s: 0.475, a: 0.034, b: 0.026, bumps: &[] },
    ArmRing { s: 0.548, a: 0.029, b: 0.019, bumps: &[(180.0, 18.0, 0.0015)] },
];

/// Hand rings in hand coordinates: distal position of the dorsal and palmar slots, the radial
/// position of the five column boundaries (ulnar to radial), dorsal and palmar depth per column.
struct HandRing {
    y_dorsal: f32,
    y_palmar: f32,
    x: [f32; 5],
    z_dorsal: [f32; 5],
    z_palmar: [f32; 5],
}

const HAND: [HandRing; 4] = [
    HandRing {
        y_dorsal: 0.028,
        y_palmar: 0.030,
        x: [-0.036, -0.019, 0.000, 0.019, 0.034],
        z_dorsal: [-0.008, -0.013, -0.014, -0.013, -0.008],
        z_palmar: [0.012, 0.017, 0.013, 0.019, 0.016],
    },
    HandRing {
        y_dorsal: 0.058,
        y_palmar: 0.060,
        x: [-0.040, -0.021, -0.002, 0.017, 0.035],
        z_dorsal: [-0.007, -0.012, -0.013, -0.012, -0.007],
        z_palmar: [0.011, 0.015, 0.010, 0.015, 0.012],
    },
    HandRing {
        y_dorsal: 0.080,
        y_palmar: 0.086,
        x: [-0.041, -0.022, -0.004, 0.015, 0.034],
        z_dorsal: [-0.006, -0.012, -0.0125, -0.012, -0.006],
        z_palmar: [0.009, 0.014, 0.013, 0.014, 0.010],
    },
    HandRing {
        y_dorsal: 0.093,
        y_palmar: 0.110,
        x: [-0.040, -0.0225, -0.0055, 0.0135, 0.033],
        z_dorsal: [-0.004, -0.010, -0.011, -0.010, -0.004],
        z_palmar: [0.006, 0.010, 0.010, 0.010, 0.006],
    },
];

/// The semantic slots of the hand rings: `(palmar, column)`, ordered by increasing angle in the
/// forearm frame (radial palmar first).
const HAND_SLOTS: [(bool, usize); 10] =
    [(true, 4), (true, 3), (true, 2), (true, 1), (true, 0), (false, 0), (false, 1), (false, 2), (false, 3), (false, 4)];

fn hand_slot_point(a: &Anatomy, ring: &HandRing, palmar: bool, col: usize, side: f32) -> Vec3 {
    let (y, z) = if palmar { (ring.y_palmar, ring.z_palmar[col]) } else { (ring.y_dorsal, ring.z_dorsal[col]) };
    sided(a.hand_point([ring.x[col], y, z]), side)
}

fn add_ring(cage: &mut Cage, points: &[Vec3]) -> Vec<u32> {
    cage.add_all(points)
}

/// The thumb's joint chain of the `side` hand: carpometacarpal, metacarpophalangeal and
/// interphalangeal joints and the tip (the skeleton uses the same points).
pub fn thumb_joints(a: &Anatomy, side: f32) -> Vec<Vec3> {
    let [distal, palmar_axis, radial] = a.hand_axes.map(|v| sided(v, side));
    let (cmc, lengths) = THUMB;
    let cmc = sided(a.hand_point(cmc), side);
    let thumb_bend = radial.cross(palmar_axis).normalize();
    let mut joints = vec![cmc];
    // Relaxed: the thumb lies in front of the index finger's base, turned about 35 degrees out of
    // the palm plane, not hanging down from the palm.
    let mut dir = (radial * 0.55 + distal * 0.72 + palmar_axis * 0.42).normalize();
    for (i, &len) in lengths.iter().enumerate() {
        let last = *joints.last().unwrap_or(&cmc);
        joints.push(last + dir * len);
        if i == 0 {
            // `thumb_bend` is a cross product of mirrored axes, so it already flips with the side.
            dir = rotate(dir, thumb_bend, 0.10).normalize();
        } else {
            dir = rotate(dir, palmar_axis.cross(dir).normalize(), -0.15).normalize();
        }
    }
    joints
}

/// Finger `f`'s joint chain (0 = index .. 3 = little) of the `side` hand: metacarpophalangeal,
/// proximal and distal interphalangeal joints and the tip.
pub fn finger_joints(a: &Anatomy, f: usize, side: f32) -> Vec<Vec3> {
    let Some(&(_, mcp, phalanges)) = FINGERS.get(f) else {
        return Vec::new();
    };
    let [distal, palmar_axis, radial] = a.hand_axes.map(|v| sided(v, side));
    let mcp = sided(a.hand_point(mcp), side);
    // Relaxed fingers fan out a little and flex more from the index to the little finger (the
    // resting cascade of the hand).
    let spread = (f as f32 - 1.3) * -0.11 * side;
    let mut dir = rotate(distal, palmar_axis, spread).normalize();
    let cascade = 1.0 + 0.22 * f as f32;
    let flexion = [0.20 * cascade, 0.30 * cascade, 0.18 * cascade];
    let mut joints = vec![mcp];
    for (i, &len) in phalanges.iter().enumerate() {
        // `radial` is mirrored already; mirroring a rotation negates its angle.
        dir = rotate(dir, radial, -flexion[i] * side).normalize();
        let last = *joints.last().unwrap_or(&mcp);
        joints.push(last + dir * len);
    }
    joints
}

/// Builds an arm from its socket ring (10 vertices, counter-clockwise around the outward direction).
pub fn arm(cage: &mut Cage, a: &Anatomy, socket: &[u32], side: f32) -> Result<ArmLoops> {
    if socket.len() != 10 {
        return Err(ModelError::invalid("arm socket", socket.len(), "arm socket ring must have 10 vertices"));
    }
    let shoulder = sided(a.shoulder, side);
    let elbow = sided(a.elbow, side);
    let wrist = sided(a.wrist, side);
    let up_dir = sided(a.upper_arm_dir, side);
    let fore_dir = sided(a.forearm_dir, side);
    let frame0 = Frame::new(shoulder, up_dir, Vec3::Z, side);
    let start = ring_angles(cage, socket, &frame0);
    // The wrist pattern: the angles of the hand's semantic slots in the wrist frame.
    let elbow_len = (elbow - shoulder).length();
    let axis_at = |s: f32| -> (Vec3, Vec3) {
        if s <= elbow_len {
            (shoulder + up_dir * s, if s > elbow_len - 0.03 { (up_dir + fore_dir).normalize() } else { up_dir })
        } else {
            (elbow + fore_dir * (s - elbow_len), fore_dir)
        }
    };
    let mut frames = Vec::with_capacity(ARM.len());
    let mut frame = frame0;
    for ring in &ARM {
        let (o, w) = axis_at(ring.s);
        frame = frame.carry(o, w);
        frames.push(frame);
    }
    let wrist_frame = frame.carry(wrist, fore_dir);
    let first_hand = &HAND[0];
    let mut pattern: Vec<f32> = HAND_SLOTS
        .iter()
        .map(|&(palmar, col)| {
            let p = hand_slot_point(a, first_hand, palmar, col, side);
            wrist_frame.angle(p)
        })
        .collect();
    // The pattern must increase with the slot order for `match_pattern`; on the right side the
    // mirrored frame already measures the mirrored slots at the same angles.
    unwrap(&mut pattern);
    if pattern[9] < pattern[0] {
        return Err(ModelError::invalid("hand pattern", 0, "hand slots are not counter-clockwise"));
    }
    let matched = match_pattern(&start, &pattern);
    let arm_tag = |s: f32| -> Tag {
        let region =
            if s < 0.3 { R::sided(R::UpperArmL, R::UpperArmR, side) } else { R::sided(R::ForearmL, R::ForearmR, side) };
        Tag::new(region.id(), C::sided(C::ArmL, C::ArmR, side).id())
    };
    let mut rings = vec![socket.to_vec()];
    let n = ARM.len();
    for (k, spec) in ARM.iter().enumerate() {
        let t = smooth((k as f32 + 1.0) / n as f32);
        let f = &frames[k];
        let points: Vec<Vec3> = (0..10)
            .map(|j| {
                let theta = mix(start[j], matched[j].1, t);
                let r = radius(theta, spec.a, spec.b, spec.bumps);
                f.at(theta, r, r)
            })
            .collect();
        let ring = add_ring(cage, &points);
        let prev = rings.last().cloned().unwrap_or_default();
        cage.bridge(&prev, &ring, arm_tag(spec.s));
        rings.push(ring);
    }
    // The hand: rings in hand coordinates, slot j is the semantic slot matched to it.
    let slot_of: Vec<(bool, usize)> = matched.iter().map(|&(k, _)| HAND_SLOTS[k]).collect();
    let hand_region = R::sided(R::HandLPalm, R::HandRPalm, side);
    let dorsal_chart = C::sided(C::HandLDorsal, C::HandRDorsal, side);
    let palmar_chart = C::sided(C::HandLPalmar, C::HandRPalmar, side);
    let chart_between = |p: (bool, usize), q: (bool, usize)| -> C {
        match (p.0, q.0) {
            (true, true) => palmar_chart,
            (false, false) => dorsal_chart,
            // Side faces: radial side to the dorsal chart, ulnar side to the palmar chart.
            _ => {
                if p.1 == 4 || q.1 == 4 {
                    dorsal_chart
                } else {
                    palmar_chart
                }
            }
        }
    };
    let wrist_ring = rings.last().cloned().unwrap_or_default();
    let mut hand = vec![wrist_ring];
    let mut thumb_start: Option<[u32; 4]> = None;
    for (h, spec) in HAND.iter().enumerate() {
        let points: Vec<Vec3> = slot_of.iter().map(|&(p, c)| hand_slot_point(a, spec, p, c, side)).collect();
        let ring = add_ring(cage, &points);
        let prev = hand.last().cloned().unwrap_or_default();
        for j in 0..10 {
            let j1 = (j + 1) % 10;
            let quad = [prev[j], prev[j1], ring[j1], ring[j]];
            // The thumb leaves from the radial side quad between the first two hand rings.
            let radial_side = slot_of[j].0 && slot_of[j1].0 && slot_of[j].1 + slot_of[j1].1 == 7;
            if h == 1 && radial_side {
                thumb_start = Some(quad);
                continue;
            }
            cage.face(&quad, Tag::new(hand_region.id(), chart_between(slot_of[j], slot_of[j1]).id()));
        }
        hand.push(ring);
    }
    let knuckles = hand.last().cloned().unwrap_or_default();
    let find = |palmar: bool, col: usize| -> Option<u32> {
        slot_of.iter().position(|&s| s == (palmar, col)).map(|j| knuckles[j])
    };
    let [_, palmar_axis, radial] = a.hand_axes.map(|v| sided(v, side));
    let mut digits = Vec::with_capacity(5);
    // Thumb.
    let Some(thumb_quad) = thumb_start else {
        return Err(ModelError::invalid("hand", 0, "no radial quad for the thumb"));
    };
    let thumb_tag = Tag::new(R::sided(R::HandLThumb, R::HandRThumb, side).id(), dorsal_chart.id());
    let thumb_palmar_tag = Tag::new(R::sided(R::HandLThumb, R::HandRThumb, side).id(), palmar_chart.id());
    let joints = thumb_joints(a, side);
    let thumb = digit(
        cage,
        &thumb_quad,
        &joints,
        &[
            (0.30, 0.0150, 0.0135),
            (0.55, 0.0118, 0.0102),
            (0.72, 0.0106, 0.0094),
            (0.85, 0.0098, 0.0086),
            (0.93, 0.0090, 0.0077),
            (0.975, 0.0070, 0.0060),
            (0.996, 0.0040, 0.0034),
        ],
        -palmar_axis,
        side,
        &|n: Vec3| if n.dot(palmar_axis) > 0.0 { thumb_palmar_tag } else { thumb_tag },
    )?;
    digits.push(thumb);
    // Fingers: index (columns 3-4) to little (0-1).
    let finger_regions = [
        R::sided(R::HandLIndex, R::HandRIndex, side),
        R::sided(R::HandLMiddle, R::HandRMiddle, side),
        R::sided(R::HandLRing, R::HandRRing, side),
        R::sided(R::HandLLittle, R::HandRLittle, side),
    ];
    for f in 0..FINGERS.len() {
        let col = 3 - f;
        let (Some(d0), Some(d1), Some(p1), Some(p0)) =
            (find(false, col), find(false, col + 1), find(true, col + 1), find(true, col))
        else {
            return Err(ModelError::invalid("hand", f, "knuckle slot missing"));
        };
        let start = [d0, d1, p1, p0];
        let joints = finger_joints(a, f, side);
        let scale = [0.97, 1.0, 0.94, 0.83][f];
        let radii: Vec<(f32, f32, f32)> = [
            // Web, proximal phalanx waist, PIP knuckle, middle phalanx, DIP, pad, tip dome.
            (0.15, 0.0102, 0.0098),
            (0.33, 0.0092, 0.0087),
            (0.48, 0.0095, 0.0088),
            (0.63, 0.0084, 0.0077),
            (0.78, 0.0083, 0.0073),
            (0.89, 0.0079, 0.0069),
            (0.96, 0.0065, 0.0057),
            (0.993, 0.0038, 0.0034),
        ]
        .iter()
        .map(|&(t, ra, rb)| (t, ra * scale, rb * scale))
        .collect();
        let region = finger_regions[f];
        let dorsal = Tag::new(region.id(), dorsal_chart.id());
        let palmar = Tag::new(region.id(), palmar_chart.id());
        let rings = digit(cage, &start, &joints, &radii, -palmar_axis, side, &|n: Vec3| {
            // Dorsal and radial sides to the dorsal chart, palmar and ulnar sides to the palmar one.
            let d = n.dot(-palmar_axis);
            let r = n.dot(radial);
            if d > 0.5 || (d > -0.5 && r > 0.0) { dorsal } else { palmar }
        })?;
        digits.push(rings);
    }
    Ok(ArmLoops { rings, hand, digits })
}

/// Builds a digit (finger, thumb or toe) from a 4-vertex start ring along the joint polyline
/// `joints` (base joint first, tip last). `radii` gives, at fractions of the polyline length, the
/// half-width and half-thickness of the rings; the last ring is closed with one quad. `up` is the
/// dorsal direction; `tag` picks the tag from a face's outward normal.
fn digit(
    cage: &mut Cage,
    start: &[u32; 4],
    joints: &[Vec3],
    radii: &[(f32, f32, f32)],
    up: Vec3,
    side: f32,
    tag: &dyn Fn(Vec3) -> Tag,
) -> Result<Vec<Vec<u32>>> {
    let total: f32 = joints.windows(2).map(|w| (w[1] - w[0]).length()).sum();
    if total <= 0.0 {
        return Err(ModelError::invalid("digit", 0, "digit has no length"));
    }
    if joints.len() < 2 {
        return Err(ModelError::invalid("digit", 0, "digit needs at least two joints"));
    }
    // The digit leaves from the start ring's centre: joints behind it (inside the hand or foot)
    // are dropped.
    let centre = start.iter().fold(Vec3::ZERO, |s, &v| s + cage.pos[v as usize]) * 0.25;
    // The hole's orientation comes from the topology: the faces around it already use its edges
    // in the opposite direction, so the ring that runs the other way is the hole's own boundary
    // and its normal points out of the hand or foot (a geometric guess from the joints fails for
    // strongly flexed digits).
    let used = |a: u32, b: u32| -> bool {
        (0..cage.face_count()).any(|f| {
            let cs = cage.corners_of(f);
            (0..cs.len()).any(|k| cs[k] == a && cs[(k + 1) % cs.len()] == b)
        })
    };
    let taken = (0..4).any(|k| used(start[k], start[(k + 1) % 4]));
    let reversed_taken = (0..4).any(|k| used(start[(k + 1) % 4], start[k]));
    if taken == reversed_taken {
        return Err(ModelError::invalid("digit", 0, "the start ring is not the boundary of one hole"));
    }
    let mut start = *start;
    if taken {
        start.reverse();
    }
    let mut outward = Vec3::ZERO;
    for i in 0..4 {
        let (p, q) = (cage.pos[start[i] as usize] - centre, cage.pos[start[(i + 1) % 4] as usize] - centre);
        outward += p.cross(q);
    }
    let outward = outward.normalize_or_zero();
    // The digit leaves the ring along a lead-in that never runs along the ring's plane: a flexed
    // digit's first joint can lie almost in that plane, and a tube starting along it would twist
    // inside out.
    // Joints count when they lie ahead of the ring along the digit (towards its tip); the lead-in
    // runs half way between the ring's normal and the first such joint.
    let along = (joints[joints.len() - 1] - centre).normalize_or_zero();
    let ahead: Vec<Vec3> = joints.iter().copied().filter(|&j| (j - centre).dot(along) > 0.008).collect();
    let first = ahead.first().map_or(along, |&j| (j - centre).normalize_or_zero());
    let lead = (outward + first).normalize_or_zero();
    let lead = if lead.dot(outward) < 0.3 { outward } else { lead };
    let mut path = vec![centre, centre + lead * 0.005];
    path.extend(ahead);
    if path.len() < 2 {
        return Err(ModelError::invalid("digit", 0, "no joint lies beyond the start ring of a digit"));
    }
    let joints = &path;
    let total: f32 = joints.windows(2).map(|w| (w[1] - w[0]).length()).sum();
    let segments = joints.len() - 1;
    let at = |t: f32| -> (Vec3, Vec3) {
        let mut d = t * total;
        for (i, w) in joints.windows(2).enumerate() {
            let len = (w[1] - w[0]).length().max(1e-9);
            let dir = (w[1] - w[0]) / len;
            if d <= len || i + 1 == segments {
                return (w[0] + dir * d, dir);
            }
            d -= len;
        }
        (joints[0], Vec3::Z)
    };
    let (o0, w0) = at(0.0);
    // Frame: u across the digit (radial/medial), v toward the dorsal side, mirrored per side.
    let base = Frame::new(o0, w0, up.cross(w0) * side, side);
    let start_angles = ring_angles(cage, &start, &base);
    let diag: Vec<f32> = (0..4).map(|k| (45.0 + 90.0 * k as f32).to_radians()).collect();
    let matched = match_pattern(&start_angles, &diag);
    let mut rings = vec![start.to_vec()];
    let mut frame = base;
    let n = radii.len();
    let face_tag = |cage: &Cage, quad: &[u32]| -> Tag {
        let p: Vec<Vec3> = quad.iter().map(|&v| cage.pos[v as usize]).collect();
        let mut nrm = Vec3::ZERO;
        for i in 0..p.len() {
            nrm += p[i].cross(p[(i + 1) % p.len()]);
        }
        tag(nrm.normalize_or_zero())
    };
    for (k, &(t, ra, rb)) in radii.iter().enumerate() {
        let (o, w) = at(t);
        frame = frame.carry(o, w);
        let blend = smooth(f32::midpoint(k as f32, 1.0));
        let points: Vec<Vec3> = (0..4)
            .map(|j| {
                let theta = mix(start_angles[j], matched[j].1, blend);
                frame.at(theta, ra * 1.0, rb * 1.0)
            })
            .collect();
        let ring = add_ring(cage, &points);
        let prev = rings.last().cloned().unwrap_or_default();
        for j in 0..4 {
            let quad = [prev[j], prev[(j + 1) % 4], ring[(j + 1) % 4], ring[j]];
            let tg = face_tag(cage, &quad);
            cage.face(&quad, tg);
        }
        rings.push(ring);
        if k + 1 == n {
            let last = rings.last().cloned().unwrap_or_default();
            let tg = face_tag(cage, &last);
            cage.face(&last, tg);
        }
    }
    Ok(rings)
}

/// One leg ring: height of the axis point, semi-axes (front, medial), bumps, and the vertical tilt
/// (front-back and medial-lateral height differences of the ring).
struct LegRing {
    y: f32,
    a: f32,
    b: f32,
    bumps: &'static [Bump],
    tilt: (f32, f32),
}

/// Leg rings (thigh circumference ~620 mm at the gluteal furrow, knee ~385 mm, calf ~375 mm,
/// ankle ~225 mm: ANSUR II means averaged over the sexes). The first ring hangs just below the
/// gluteal fold: deep front to back with the lower pole of the buttock behind, its medial side
/// a few millimetres off the midline so the thighs meet only near the crotch (the soft tissue of
/// real thighs flattens where they touch; a cage that crossed the midline would interpenetrate).
const LEG: [LegRing; 11] = [
    LegRing {
        y: 0.785,
        a: 0.095,
        b: 0.086,
        bumps: &[(180.0, 50.0, 0.022), (90.0, 30.0, -0.009), (270.0, 40.0, -0.002)],
        tilt: (0.018, -0.010),
    },
    LegRing {
        y: 0.700,
        a: 0.086,
        b: 0.080,
        bumps: &[(10.0, 40.0, 0.005), (90.0, 30.0, -0.004), (180.0, 50.0, 0.006)],
        tilt: (0.006, -0.004),
    },
    LegRing { y: 0.625, a: 0.076, b: 0.073, bumps: &[(10.0, 40.0, 0.005), (110.0, 35.0, 0.003)], tilt: (0.0, 0.0) },
    LegRing { y: 0.556, a: 0.066, b: 0.064, bumps: &[(20.0, 35.0, 0.004), (80.0, 30.0, 0.006)], tilt: (0.0, 0.0) },
    LegRing { y: 0.505, a: 0.057, b: 0.057, bumps: &[(70.0, 30.0, 0.005), (0.0, 25.0, 0.003)], tilt: (0.0, 0.0) },
    LegRing { y: 0.468, a: 0.054, b: 0.055, bumps: &[(0.0, 25.0, 0.006), (180.0, 40.0, -0.004)], tilt: (0.0, 0.0) },
    LegRing { y: 0.425, a: 0.050, b: 0.050, bumps: &[(0.0, 22.0, 0.003), (160.0, 40.0, 0.004)], tilt: (0.0, 0.0) },
    LegRing { y: 0.355, a: 0.047, b: 0.052, bumps: &[(170.0, 45.0, 0.013), (120.0, 25.0, 0.004)], tilt: (0.0, 0.0) },
    LegRing { y: 0.280, a: 0.043, b: 0.046, bumps: &[(165.0, 45.0, 0.009)], tilt: (0.0, 0.0) },
    LegRing { y: 0.195, a: 0.036, b: 0.036, bumps: &[(175.0, 35.0, 0.003)], tilt: (0.0, 0.0) },
    LegRing { y: 0.130, a: 0.032, b: 0.031, bumps: &[(180.0, 22.0, 0.004)], tilt: (0.0, 0.0) },
];

/// Builds a leg from its start ring (16 vertices, counter-clockwise around -Y).
pub fn leg(cage: &mut Cage, a: &Anatomy, start: &[u32], side: f32) -> Result<LegLoops> {
    if start.len() != 16 {
        return Err(ModelError::invalid("leg start", start.len(), "leg start ring must have 16 vertices"));
    }
    let hip = sided(a.hip, side);
    let knee = sided(a.knee, side);
    let ankle = sided(a.ankle, side);
    let axis = |y: f32| -> (Vec3, Vec3) {
        if y >= knee.y {
            let t = (hip.y - y) / (hip.y - knee.y);
            (hip + (knee - hip) * t, (knee - hip).normalize())
        } else {
            let t = (knee.y - y) / (knee.y - ankle.y);
            (knee + (ankle - knee) * t, (ankle - knee).normalize())
        }
    };
    let (o0, w0) = axis(LEG[0].y + 0.04);
    let frame0 = Frame::new(o0, w0, Vec3::Z, side);
    let start_angles = ring_angles(cage, start, &frame0);
    let even: Vec<f32> = (0..16).map(|k| (k as f32 * 22.5).to_radians()).collect();
    let matched = match_pattern(&start_angles, &even);
    let leg_chart = C::sided(C::LegL, C::LegR, side);
    let mut rings = vec![start.to_vec()];
    let mut frame = frame0;
    for (k, spec) in LEG.iter().enumerate() {
        let (o, w) = axis(spec.y);
        frame = frame.carry(o, w);
        let t = smooth((k as f32 + 1.0) / 3.0);
        let points: Vec<Vec3> = (0..16)
            .map(|j| {
                let theta = mix(start_angles[j], matched[j].1, t);
                let r = radius(theta, spec.a, spec.b, spec.bumps);
                let p = frame.at(theta, r, r);
                p + Vec3::Y * (spec.tilt.0 * theta.cos() + spec.tilt.1 * theta.sin())
            })
            .collect();
        let ring = add_ring(cage, &points);
        let region =
            if spec.y > 0.49 { R::sided(R::ThighL, R::ThighR, side) } else { R::sided(R::ShinL, R::ShinR, side) };
        let prev = rings.last().cloned().unwrap_or_default();
        cage.bridge(&prev, &ring, Tag::new(region.id(), leg_chart.id()));
        rings.push(ring);
    }
    let angles: Vec<f32> = (0..16).map(|j| matched[j].1).collect();
    let (foot, toes) = foot(cage, a, &rings, &angles, frame, side)?;
    Ok(LegLoops { rings, foot, toes })
}

/// Distance of the ball of the foot (toe base centre) in front of the ankle, metres: with the heel
/// ~58 mm behind the ankle and the big toe 37 mm long the foot is ANSUR II's `footlength`.
pub const BALL: f32 = 0.156;

/// Foot coordinates (along the foot from the ankle, medial offset, height) to world.
pub fn foot_point(a: &Anatomy, side: f32, m: f32, l: f32, h: f32) -> Vec3 {
    let dir = sided(a.foot_dir, side);
    let medial = dir.cross(Vec3::Y) * side;
    let base = sided(Vec3::new(a.ankle.x, 0.0, a.ankle.z), side);
    base + dir * m + medial * l + Vec3::Y * h
}

/// The foot's rings (ankle to toe base) and the rings of each toe.
type FootLoops = (Vec<Vec<u32>>, Vec<Vec<Vec<u32>>>);

/// The foot: rings from the ankle (bending forward) to the toe bases, then five toes.
fn foot(
    cage: &mut Cage,
    a: &Anatomy,
    leg: &[Vec<u32>],
    angles: &[f32],
    leg_frame: Frame,
    side: f32,
) -> Result<FootLoops> {
    let top = C::sided(C::FootLTop, C::FootRTop, side);
    let sole = C::sided(C::FootLSole, C::FootRSole, side);
    let dorsum = R::sided(R::FootLDorsum, R::FootRDorsum, side);
    let sole_region = R::sided(R::FootLSole, R::FootRSole, side);
    let first_face = cage.face_count();
    // Ring points by angle: 0 = top (front of the shin), 90 = medial, 180 = sole/back, 270 = lateral.
    // Each ring: (along, height, half-length of the section along its "up" direction, half-width,
    // tilt of the section from vertical in radians: 0 = horizontal leg ring, PI/2 = upright).
    let specs: [(f32, f32, f32, f32, f32); 4] = [
        (0.002, 0.058, 0.040, 0.034, 0.25),
        (0.000, 0.040, 0.060, 0.036, 0.80),
        (0.068, 0.033, 0.034, 0.041, 1.48),
        (0.136, 0.024, 0.024, 0.049, 1.55),
    ];
    let dir = sided(a.foot_dir, side);
    let mut rings: Vec<Vec<u32>> = vec![leg.last().cloned().unwrap_or_default()];
    let mut frame = leg_frame;
    for (k, &(m, h, half_up, half_w, tilt)) in specs.iter().enumerate() {
        let o = foot_point(a, side, m, 0.0, h);
        // The section normal rotates from straight down (leg) to the foot direction.
        let w = (-Vec3::Y * tilt.cos() + dir * tilt.sin()).normalize();
        frame = frame.carry(o, w);
        let points: Vec<Vec3> = angles
            .iter()
            .map(|&theta| {
                let (s, c) = theta.sin_cos();
                // Heel: the back-bottom of the oblique rings reaches back and down.
                let heel = if k <= 1 && c < 0.0 { 1.0 + 0.22 * c * c } else { 1.0 };
                // Arch: the medial sole is lifted on the midfoot ring.
                let mut p = frame.at(theta, half_up * heel, half_w);
                if k == 2 && c < 0.0 && s > 0.0 {
                    p += Vec3::Y * (0.010 * s * -c);
                }
                // Nothing below the floor: the designed points stay 3 mm above it so the fitted
                // surface (which overshoots a little between them) rests on y = 0.
                p.y = p.y.max(0.003);
                p
            })
            .collect();
        let ring = add_ring(cage, &points);
        let prev = rings.last().cloned().unwrap_or_default();
        for j in 0..prev.len() {
            let j1 = (j + 1) % prev.len();
            let mid = f32::midpoint(angles[j], angles[j1]);
            let is_sole = mid.cos() < -0.2;
            let chart = if is_sole && k >= 1 { sole } else { top };
            let region = if is_sole && k >= 1 { sole_region } else { dorsum };
            cage.face(&[prev[j], prev[j1], ring[j1], ring[j]], Tag::new(region.id(), chart.id()));
        }
        rings.push(ring);
    }
    // Toe base ring: 12 slots, six dorsal and six plantar column boundaries (medial to lateral).
    let cols: [f32; 6] = [0.047, 0.019, 0.002, -0.013, -0.027, -0.041];
    let base_m: [f32; 6] = [0.166, 0.164, 0.158, 0.150, 0.141, 0.131];
    let dorsal_h: [f32; 6] = [0.030, 0.034, 0.029, 0.026, 0.023, 0.018];
    let plantar_h: [f32; 6] = [0.004, 0.003, 0.002, 0.002, 0.002, 0.003];
    let mut semantic: Vec<(bool, usize, Vec3)> = Vec::with_capacity(12);
    for i in 0..6 {
        semantic.push((false, i, foot_point(a, side, base_m[i] - 0.004, cols[i], dorsal_h[i])));
        semantic.push((true, i, foot_point(a, side, base_m[i] + 0.006, cols[i], plantar_h[i])));
    }
    let o = foot_point(a, side, BALL, 0.0, 0.018);
    frame = frame.carry(o, dir);
    let prev = rings.last().cloned().unwrap_or_default();
    let prev_angles = ring_angles(cage, &prev, &frame);
    let increasing = prev_angles[prev_angles.len() - 1] > prev_angles[0];
    let mut by_angle: Vec<(f32, usize)> = semantic.iter().enumerate().map(|(i, s)| (frame.angle(s.2), i)).collect();
    by_angle.sort_by(|x, y| x.0.total_cmp(&y.0));
    if !increasing {
        by_angle.reverse();
    }
    let a0 = prev_angles[0];
    let first = by_angle
        .iter()
        .enumerate()
        .min_by(|x, y| {
            let dx = ((x.1.0 - a0 + 3.0 * PI).rem_euclid(2.0 * PI) - PI).abs();
            let dy = ((y.1.0 - a0 + 3.0 * PI).rem_euclid(2.0 * PI) - PI).abs();
            dx.total_cmp(&dy)
        })
        .map_or(0, |x| x.0);
    by_angle.rotate_left(first);
    let order: Vec<usize> = by_angle.iter().map(|x| x.1).collect();
    let points: Vec<Vec3> = order.iter().map(|&i| semantic[i].2).collect();
    let ring = add_ring(cage, &points);
    let f_start = cage.face_count();
    // A loft depends on where it starts and which way it walks. Both feet start at the same
    // (mirror-invariant) top-front vertex of each ring and walk the same way round in space: the
    // right foot's rings run the other way, so it walks them backwards and turns its faces.
    let score = |v: u32| {
        let p = cage.pos[v as usize];
        p.y + 0.37 * p.z + 0.013 * p.x.abs()
    };
    let walk = |r: &[u32]| -> Vec<u32> {
        let n = r.len();
        let s0 = (0..n).max_by(|&i, &j| score(r[i]).total_cmp(&score(r[j]))).unwrap_or(0);
        (0..n).map(|k| r[if side > 0.0 { (s0 + k) % n } else { (s0 + n - k) % n }]).collect()
    };
    let (prev_w, ring_w) = (walk(&prev), walk(&ring));
    cage.loft(&prev_w, &ring_w, Tag::new(dorsum.id(), top.id()));
    if side < 0.0 {
        for f in f_start..cage.face_count() {
            let (lo, hi) = (cage.offsets[f] as usize, cage.offsets[f + 1] as usize);
            cage.corners[lo..hi].reverse();
        }
    }
    // Re-tag the loft faces on the sole side.
    for f in f_start..cage.face_count() {
        if cage.centroid(f).y < 0.012 {
            cage.tags[f] = Tag::new(sole_region.id(), sole.id());
        }
    }
    rings.push(ring.clone());
    let find = |plantar: bool, col: usize| -> Option<u32> {
        order.iter().position(|&i| semantic[i].0 == plantar && semantic[i].1 == col).map(|j| ring[j])
    };
    let toe_region = R::sided(R::FootLToes, R::FootRToes, side);
    let toe_top = Tag::new(toe_region.id(), top.id());
    let toe_sole = Tag::new(toe_region.id(), sole.id());
    let lengths: [f32; 5] = [0.037, 0.032, 0.028, 0.024, 0.020];
    let widths: [f32; 5] = [0.0125, 0.0082, 0.0076, 0.0072, 0.0068];
    let mut toes = Vec::with_capacity(5);
    for t in 0..5 {
        let (Some(d0), Some(d1), Some(p1), Some(p0)) =
            (find(false, t), find(false, t + 1), find(true, t + 1), find(true, t))
        else {
            return Err(ModelError::invalid("foot", t, "toe slot missing"));
        };
        let q = [d0, d1, p1, p0];
        let base =
            (cage.pos[d0 as usize] + cage.pos[d1 as usize] + cage.pos[p0 as usize] + cage.pos[p1 as usize]) * 0.25;
        let spread = (2.0 - t as f32) * 0.035 * side;
        let tdir = rotate(dir, Vec3::Y, spread).normalize();
        let len = lengths[t];
        let droop = rotate(tdir, tdir.cross(Vec3::Y).normalize(), -0.25).normalize();
        let joints =
            vec![base, base + tdir * (len * 0.45) + Vec3::Y * 0.002, base + tdir * (len * 0.8) + droop * (len * 0.15)];
        let w = widths[t];
        let radii = [
            (0.25, w * 0.95, w * 0.80),
            (0.55, w * 0.92, w * 0.78),
            (0.82, w * 0.90, w * 0.72),
            (0.97, w * 0.65, w * 0.50),
        ];
        let rings_t =
            digit(cage, &q, &joints, &radii, Vec3::Y, side, &|n: Vec3| if n.y < -0.3 { toe_sole } else { toe_top })?;
        toes.push(rings_t);
    }
    // Charts by the cage face normal: faces looking down are the sole, the rest the top (which
    // wraps round the heel and is cut along its back by the UV stage).
    let toe_region = R::sided(R::FootLToes, R::FootRToes, side);
    for f in first_face..cage.face_count() {
        let c = cage.corners_of(f);
        let mut n = Vec3::ZERO;
        for k in 0..c.len() {
            n += cage.pos[c[k] as usize].cross(cage.pos[c[(k + 1) % c.len()] as usize]);
        }
        let down = n.normalize_or_zero().y < -0.55;
        let region = if cage.tags[f].region == toe_region.id() {
            toe_region
        } else if down {
            sole_region
        } else {
            dorsum
        };
        cage.tags[f] = Tag::new(region.id(), if down { sole } else { top }.id());
    }
    Ok((rings, toes))
}
