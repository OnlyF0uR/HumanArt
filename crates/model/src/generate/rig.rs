//! The skeleton and skin weights of the template `[MOD-07]` `[SKIN-01]`.
//!
//! **Joints.** Pelvis, four spine joints, two neck joints, head, jaw, a four-joint tongue chain, the
//! eyes, upper and lower eyelid helpers, clavicles, upper arms and forearms with twist joints,
//! hands with three joints per digit, thighs, shins, feet and toes: 68 joints. Positions come from
//! [`Anatomy`] and from the same digit chains the surface is built around
//! ([`limbs::finger_joints`], [`limbs::thumb_joints`]).
//!
//! **Rest frames.** Every joint's local Y axis points along its bone (to its child) and its local
//! Z axis toward the side the joint flexes to, so a positive rotation about local X is flexion
//! (elbow and finger bending, knee bending, jaw opening, looking down, closing an upper lid).
//! Limits are Euler angles (Y-X-Z) in that frame; on the right side the Y and Z ranges are mirrored.
//!
//! **Regressor.** The template has no delta components yet, so the joint regressor is empty
//! (`[MOD-07]`'s fitted regressor needs the identity components, a later work package).
//!
//! **Skin weights.** Authored procedurally per vertex from its region and position: body regions
//! blend the bone segments their region may use by inverse distance (power 4), which gives smooth
//! blends about the size of the limb's radius at each joint; twist joints take a growing share
//! along their segment; the lower face, lower teeth, gums and tongue follow the jaw by height; the
//! lid regions follow the eyelid helpers near the lid margin; eyeballs follow the eyes. Four
//! influences per vertex. Dual quaternions are used on the neck, shoulders, forearms and palms.

use std::collections::HashMap;

use humanart_math::{Mat3, Quat, Transform, Vec3};

use super::anatomy::{Anatomy, sided};
use super::body::head_point;
use super::limbs::{BALL, finger_joints, foot_point, thumb_joints};
use super::regions::R;
use crate::error::{ModelError, Result};
use crate::mesh::Mesh;
use crate::region::RegionMap;
use crate::skeleton::{Joint, JointKind, JointLimits, JointRegressor, Skeleton};
use crate::skin::SkinWeights;

/// Influences per vertex.
pub const INFLUENCES: usize = 4;

/// Names of the finger chains, index to little (matches [`super::anatomy::FINGERS`]).
const FINGER_NAMES: [&str; 4] = ["index", "middle", "ring", "little"];

/// One joint being designed: world position, bone direction (local Y), flexion direction (local Z
/// hint), the end of its bone segment for skinning, kind and limits (left-side convention).
struct Spec {
    name: String,
    parent: Option<usize>,
    kind: JointKind,
    pos: Vec3,
    dir: Vec3,
    flex: Vec3,
    end: Vec3,
    limits: JointLimits,
}

/// Limits from `(x min, x max)`, `(y min, y max)`, `(z min, z max)`, mirrored for the right side.
fn limits(x: (f32, f32), y: (f32, f32), z: (f32, f32), side: f32) -> JointLimits {
    let (y, z) = if side < 0.0 { ((-y.1, -y.0), (-z.1, -z.0)) } else { (y, z) };
    JointLimits { min: Vec3::new(x.0, y.0, z.0), max: Vec3::new(x.1, y.1, z.1) }
}

/// The designed joints and the bone segments used for skinning.
pub struct Rig {
    /// The skeleton (rest transforms relative to the parents, limits, empty regressor).
    pub skeleton: Skeleton,
    /// Per joint: the bone segment `(start, end)` in world space at rest.
    pub segments: Vec<(Vec3, Vec3)>,
}

impl Rig {
    /// The index of joint `name`.
    fn index(&self, name: &str) -> Option<u16> {
        self.skeleton.find(name)
    }
}

struct Builder {
    specs: Vec<Spec>,
}

impl Builder {
    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        name: String,
        parent: Option<usize>,
        kind: JointKind,
        pos: Vec3,
        end: Vec3,
        flex: Vec3,
        limits: JointLimits,
    ) -> usize {
        let dir = (end - pos).normalize_or(Vec3::Y);
        self.specs.push(Spec { name, parent, kind, pos, dir, flex, end, limits });
        self.specs.len() - 1
    }
}

/// The world rest rotation of a joint: Y along the bone, Z toward the flexion side.
fn frame(dir: Vec3, flex: Vec3) -> Quat {
    let y = dir.normalize_or(Vec3::Y);
    let mut z = flex - y * flex.dot(y);
    if z.length_squared() < 1e-10 {
        // The flexion hint is along the bone: any perpendicular does.
        let other = if y.x.abs() < 0.9 { Vec3::X } else { Vec3::Z };
        z = other - y * other.dot(y);
    }
    let z = z.normalize_or(Vec3::Z);
    let x = y.cross(z);
    Quat::from_mat3(Mat3::from_cols(x, y, z))
}

fn arm(b: &mut Builder, a: &Anatomy, side: f32, spine_top: usize) {
    let l = if side > 0.0 { "L" } else { "R" };
    let s = |p: Vec3| sided(p, side);
    let (shoulder, elbow, wrist) = (s(a.shoulder), s(a.elbow), s(a.wrist));
    let [distal, palmar, _] = a.hand_axes.map(s);
    let fwd = Vec3::Z;
    let clav = b.add(
        format!("{l}.clavicle"),
        Some(spine_top),
        JointKind::Deform,
        s(a.clavicle),
        shoulder,
        fwd,
        limits((-0.35, 0.35), (-0.2, 0.2), (-0.3, 0.45), side),
    );
    let upper = b.add(
        format!("{l}.upper-arm"),
        Some(clav),
        JointKind::Deform,
        shoulder,
        elbow,
        fwd,
        limits((-1.2, 2.8), (-1.6, 1.6), (-1.6, 1.4), side),
    );
    b.add(
        format!("{l}.upper-arm.twist"),
        Some(upper),
        JointKind::Twist,
        shoulder + (elbow - shoulder) * 0.5,
        elbow,
        fwd,
        limits((0.0, 0.0), (-1.6, 1.6), (0.0, 0.0), side),
    );
    let fore = b.add(
        format!("{l}.forearm"),
        Some(upper),
        JointKind::Deform,
        elbow,
        wrist,
        fwd,
        limits((-0.25, 2.5), (0.0, 0.0), (0.0, 0.0), side),
    );
    b.add(
        format!("{l}.forearm.twist"),
        Some(fore),
        JointKind::Twist,
        elbow + (wrist - elbow) * 0.6,
        wrist,
        fwd,
        limits((0.0, 0.0), (-1.6, 1.6), (0.0, 0.0), side),
    );
    let middle_mcp = finger_joints(a, 1, side).first().copied().unwrap_or(wrist + distal * 0.09);
    let hand = b.add(
        format!("{l}.hand"),
        Some(fore),
        JointKind::Deform,
        wrist,
        middle_mcp,
        palmar,
        limits((-1.2, 1.3), (-0.3, 0.3), (-0.4, 0.6), side),
    );
    let digit_limits = [
        limits((-0.6, 1.6), (-0.1, 0.1), (-0.35, 0.35), side),
        limits((-0.1, 1.9), (0.0, 0.0), (-0.05, 0.05), side),
        limits((-0.2, 1.5), (0.0, 0.0), (-0.05, 0.05), side),
    ];
    let thumb_limits = [
        limits((-0.8, 0.8), (-0.6, 0.6), (-0.8, 0.8), side),
        limits((-0.3, 1.3), (-0.1, 0.1), (-0.1, 0.1), side),
        limits((-0.3, 1.4), (0.0, 0.0), (-0.05, 0.05), side),
    ];
    let chains = std::iter::once(("thumb".to_owned(), thumb_joints(a, side), thumb_limits)).chain(
        FINGER_NAMES.iter().enumerate().map(|(f, n)| ((*n).to_owned(), finger_joints(a, f, side), digit_limits)),
    );
    for (name, joints, lim) in chains {
        let mut parent = hand;
        for k in 0..joints.len().saturating_sub(1).min(3) {
            parent = b.add(
                format!("{l}.{name}.{:02}", k + 1),
                Some(parent),
                JointKind::Deform,
                joints[k],
                joints[k + 1],
                palmar,
                lim[k],
            );
        }
    }
}

fn leg(b: &mut Builder, a: &Anatomy, side: f32, pelvis: usize) {
    let l = if side > 0.0 { "L" } else { "R" };
    let s = |p: Vec3| sided(p, side);
    let (hip, knee, ankle) = (s(a.hip), s(a.knee), s(a.ankle));
    let ball = foot_point(a, side, BALL, 0.0, 0.018);
    let toe_tip = foot_point(a, side, BALL + 0.042, 0.0, 0.012);
    let thigh = b.add(
        format!("{l}.thigh"),
        Some(pelvis),
        JointKind::Deform,
        hip,
        knee,
        Vec3::Z,
        limits((-0.5, 2.1), (-0.7, 0.8), (-0.5, 0.8), side),
    );
    let shin = b.add(
        format!("{l}.shin"),
        Some(thigh),
        JointKind::Deform,
        knee,
        ankle,
        -Vec3::Z,
        limits((-0.1, 2.5), (-0.1, 0.1), (-0.05, 0.05), side),
    );
    let foot = b.add(
        format!("{l}.foot"),
        Some(shin),
        JointKind::Deform,
        ankle,
        ball,
        -Vec3::Y,
        limits((-0.5, 0.9), (-0.3, 0.3), (-0.4, 0.4), side),
    );
    b.add(
        format!("{l}.toes"),
        Some(foot),
        JointKind::Deform,
        ball,
        toe_tip,
        -Vec3::Y,
        limits((-1.0, 0.6), (0.0, 0.0), (-0.1, 0.1), side),
    );
}

/// Builds the skeleton `[MOD-07]`; `components` is the number of delta components (for the empty
/// regressor).
pub fn skeleton(a: &Anatomy, components: usize) -> Result<Rig> {
    let mut b = Builder { specs: Vec::with_capacity(80) };
    let fwd = Vec3::Z;
    let free3 = |x: f32, y: f32, z: f32| limits((-x, x), (-y, y), (-z, z), 1.0);
    let pelvis = b.add("pelvis".into(), None, JointKind::Deform, a.pelvis, a.spine[0], fwd, JointLimits::FREE);
    let mut parent = pelvis;
    for k in 0..4 {
        let end = a.spine.get(k + 1).copied().unwrap_or(a.neck[0]);
        parent = b.add(
            format!("spine.{:02}", k + 1),
            Some(parent),
            JointKind::Deform,
            a.spine[k],
            end,
            fwd,
            free3(0.45, 0.35, 0.35),
        );
    }
    let spine_top = parent;
    let neck1 =
        b.add("neck.01".into(), Some(spine_top), JointKind::Deform, a.neck[0], a.neck[1], fwd, free3(0.6, 0.7, 0.5));
    let neck2 =
        b.add("neck.02".into(), Some(neck1), JointKind::Deform, a.neck[1], a.neck[2], fwd, free3(0.5, 0.6, 0.4));
    let head_top = head_point(a, 0.0, 133.0, 0.0);
    let head = b.add("head".into(), Some(neck2), JointKind::Deform, a.neck[2], head_top, fwd, free3(0.45, 0.5, 0.35));
    let chin = head_point(a, 0.0, -92.0, 85.0);
    let jaw = b.add(
        "jaw".into(),
        Some(head),
        JointKind::Deform,
        a.jaw_pivot,
        chin,
        -Vec3::Z,
        limits((-0.05, 0.5), (-0.12, 0.12), (-0.1, 0.1), 1.0),
    );
    let tongue = [(45.0, -57.0), (57.0, -57.5), (68.0, -57.0), (78.0, -56.5), (86.0, -56.0)];
    let mut parent = jaw;
    for k in 0..4 {
        let (z0, y0) = tongue[k];
        let (z1, y1) = tongue[k + 1];
        parent = b.add(
            format!("tongue.{:02}", k + 1),
            Some(parent),
            JointKind::Deform,
            head_point(a, 0.0, y0, z0),
            head_point(a, 0.0, y1, z1),
            Vec3::Y,
            free3(0.6, 0.5, 0.5),
        );
    }
    for side in [1.0_f32, -1.0] {
        let l = if side > 0.0 { "L" } else { "R" };
        let e = sided(a.eye_center, side);
        let front = e + Vec3::Z * a.eye_radius;
        b.add(
            format!("{l}.eye"),
            Some(head),
            JointKind::Deform,
            e,
            front,
            -Vec3::Y,
            limits((-0.6, 0.6), (-0.8, 0.8), (0.0, 0.0), side),
        );
        b.add(
            format!("{l}.eyelid.upper"),
            Some(head),
            JointKind::Helper,
            e,
            front,
            -Vec3::Y,
            limits((-0.3, 0.7), (0.0, 0.0), (0.0, 0.0), side),
        );
        b.add(
            format!("{l}.eyelid.lower"),
            Some(head),
            JointKind::Helper,
            e,
            front,
            -Vec3::Y,
            limits((-0.5, 0.3), (0.0, 0.0), (0.0, 0.0), side),
        );
    }
    for side in [1.0_f32, -1.0] {
        arm(&mut b, a, side, spine_top);
    }
    for side in [1.0_f32, -1.0] {
        leg(&mut b, a, side, pelvis);
    }
    // World frames, then rest transforms relative to the parents.
    let world: Vec<Quat> = b.specs.iter().map(|s| frame(s.dir, s.flex)).collect();
    let mut joints = Vec::with_capacity(b.specs.len());
    for (i, s) in b.specs.iter().enumerate() {
        let rest = match s.parent {
            Some(p) => {
                let inv = world[p].conjugate();
                Transform::from_rotation_translation((inv * world[i]).normalize(), inv * (s.pos - b.specs[p].pos))
            }
            None => Transform::from_rotation_translation(world[i], s.pos),
        };
        let parent = match s.parent {
            Some(p) => Some(u16::try_from(p).map_err(|_| ModelError::invalid("joint", i, "joint index exceeds u16"))?),
            None => None,
        };
        joints.push(Joint { name: s.name.clone(), parent, kind: s.kind, rest, limits: s.limits });
    }
    let segments = b.specs.iter().map(|s| (s.pos, s.end)).collect();
    let skeleton = Skeleton { joints, regressor: JointRegressor::empty(components) };
    skeleton.check(components)?;
    Ok(Rig { skeleton, segments })
}

/// Distance from `p` to the segment `(a, b)` and the segment parameter of the closest point.
fn segment_distance(p: Vec3, (a, b): (Vec3, Vec3)) -> (f32, f32) {
    let ab = b - a;
    let len2 = ab.length_squared();
    let t = if len2 > 0.0 { ((p - a).dot(ab) / len2).clamp(0.0, 1.0) } else { 0.0 };
    ((a + ab * t).distance(p), t)
}

fn smoothstep(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// The axilla blend `[SKIN-01]`: how much a body point below the shoulder follows the upper arm
/// when the arm swings toward the body, and how strongly that rule applies there.
///
/// Inverse-distance weights put the whole armpit fold on one side of a narrow band, so an arm
/// coming down from the A-pose drives its inner surface through the side of the chest. Here the
/// point's angle `phi` about the shoulder, in the frontal plane (0 straight down the side of the
/// chest, the arm axis at [`super::anatomy::ARM_ABDUCTION`]), sets the arm share linearly: the
/// wedge between the chest wall and the arm then closes evenly as the arm comes down, and as the
/// share's slope (1 / [`AXILLA_SPREAD`]) stays below 1 / (largest adduction), no point passes the
/// one below it. The rule fades out above the arm axis, in front of the chest midline and beyond
/// [`AXILLA_REACH`] from the joint, where the segment weights take over.
fn axilla(p: Vec3, shoulder: Vec3, side: f32) -> (f32, f32) {
    let d = p - shoulder;
    let (out, down) = (d.x * side, -d.y);
    let r = out.hypot(down);
    if down <= 0.0 || r <= 0.0 {
        return (0.0, 0.0);
    }
    let phi = out.atan2(down);
    let share = (phi / AXILLA_SPREAD).clamp(0.0, 1.0);
    let applies = smoothstep((phi + 0.6) / 0.4)
        * (1.0 - smoothstep((phi - super::anatomy::ARM_ABDUCTION) / 0.4))
        * (1.0 - smoothstep((r - AXILLA_REACH) / 0.10));
    (share, applies)
}

/// Angle over which the axilla's arm share grows from 0 (chest wall) to 1, radians. Larger than the
/// largest adduction from the A-pose that should stay fold-free (arms at the sides, ~0.7 rad).
const AXILLA_SPREAD: f32 = 0.78;
/// Distance from the shoulder joint up to which the axilla blend applies fully, metres.
const AXILLA_REACH: f32 = 0.14;

/// How a region is skinned.
enum Rule {
    /// Inverse-distance blend of these joints' bone segments.
    Segments(Vec<String>),
    /// Fully bound to one joint.
    Joint(String),
    /// Head, with the jaw taking the share below the mouth.
    LowerFace { full_jaw: bool },
    /// Head, with an eyelid helper near the lid margin (`scale` of its share at the canthi).
    Lid { helper: String, scale: f32 },
}

/// The trunk regions share one joint set, so the weights are continuous across their borders.
const TRUNK: [&str; 13] = [
    "pelvis",
    "spine.01",
    "spine.02",
    "spine.03",
    "spine.04",
    "neck.01",
    "neck.02",
    "head",
    "L.clavicle",
    "R.clavicle",
    "L.upper-arm",
    "R.upper-arm",
    "L.thigh",
];

fn rule(r: R) -> Rule {
    let n = r.name();
    let side = if n.ends_with("-L") || n.contains("-L.") { "L" } else { "R" };
    let seg = |names: &[&str]| Rule::Segments(names.iter().map(|s| s.replace('#', side)).collect());
    match r {
        R::NeckFront
        | R::NeckBack
        | R::NeckSideL
        | R::NeckSideR
        | R::Neck
        | R::ShoulderL
        | R::ShoulderR
        | R::Shoulder
        | R::Chest
        | R::Abdomen
        | R::Back
        | R::Torso
        | R::Body
        | R::Pelvis
        | R::Navel
        | R::AreolaL
        | R::AreolaR
        | R::NippleL
        | R::NippleR
        | R::Perineum
        | R::Genitals
        | R::LabiaMajora
        | R::LabiaMinora
        | R::Clitoris
        | R::Vestibule
        | R::Penis
        | R::Glans
        | R::Scrotum => {
            Rule::Segments(TRUNK.iter().map(|s| (*s).to_owned()).chain(std::iter::once("R.thigh".to_owned())).collect())
        }
        R::UpperArmL | R::UpperArmR | R::UpperArm => {
            seg(&["spine.03", "spine.04", "#.clavicle", "#.upper-arm", "#.forearm"])
        }
        R::ForearmL | R::ForearmR | R::Forearm => seg(&["#.upper-arm", "#.forearm", "#.hand"]),
        R::HandLPalm | R::HandRPalm | R::HandL | R::HandR | R::Hand => {
            seg(&["#.forearm", "#.hand", "#.thumb.01", "#.index.01", "#.middle.01", "#.ring.01", "#.little.01"])
        }
        R::HandLThumb | R::HandRThumb => seg(&["#.hand", "#.thumb.01", "#.thumb.02", "#.thumb.03"]),
        R::HandLIndex | R::HandRIndex => seg(&["#.hand", "#.index.01", "#.index.02", "#.index.03"]),
        R::HandLMiddle | R::HandRMiddle => seg(&["#.hand", "#.middle.01", "#.middle.02", "#.middle.03"]),
        R::HandLRing | R::HandRRing => seg(&["#.hand", "#.ring.01", "#.ring.02", "#.ring.03"]),
        R::HandLLittle | R::HandRLittle => seg(&["#.hand", "#.little.01", "#.little.02", "#.little.03"]),
        R::ThighL | R::ThighR | R::Thigh => seg(&["pelvis", "spine.01", "#.thigh", "#.shin"]),
        R::ShinL | R::ShinR | R::Shin => seg(&["#.thigh", "#.shin", "#.foot"]),
        R::FootL | R::FootR | R::FootLDorsum | R::FootRDorsum | R::FootLSole | R::FootRSole | R::Foot => {
            seg(&["#.shin", "#.foot", "#.toes"])
        }
        R::FootLToes | R::FootRToes => seg(&["#.foot", "#.toes"]),
        R::Tongue => seg(&["tongue.01", "tongue.02", "tongue.03", "tongue.04"]),
        R::EyeballL | R::EyeballR => Rule::Joint(format!("{side}.eye")),
        R::UpperLidL | R::UpperLidR => Rule::Lid { helper: format!("{side}.eyelid.upper"), scale: 1.0 },
        R::LowerLidL | R::LowerLidR => Rule::Lid { helper: format!("{side}.eyelid.lower"), scale: 1.0 },
        R::InnerCanthusL | R::InnerCanthusR | R::OuterCanthusL | R::OuterCanthusR => {
            Rule::Lid { helper: format!("{side}.eyelid.upper"), scale: 0.5 }
        }
        R::LowerLip | R::TeethLower | R::GumsLower => Rule::LowerFace { full_jaw: true },
        R::Chin
        | R::JawL
        | R::JawR
        | R::CheekL
        | R::CheekR
        | R::LipCornerL
        | R::LipCornerR
        | R::MouthInterior
        | R::MouthHelper => Rule::LowerFace { full_jaw: false },
        _ => Rule::Joint("head".into()),
    }
}

/// Laplacian smoothing steps of the body's weights (each moves a vertex half way to the mean of its
/// neighbours): removes the steps where neighbouring regions use different rules.
pub const SMOOTHING_ITERATIONS: usize = 10;

/// Adds `w` times `list` into `acc` (merging joints).
fn accumulate(acc: &mut Vec<(u16, f32)>, list: &[(u16, f32)], w: f32) {
    for &(j, x) in list {
        match acc.iter_mut().find(|e| e.0 == j) {
            Some(e) => e.1 += x * w,
            None => acc.push((j, x * w)),
        }
    }
}

/// Normalises a weight list to sum 1 (dropping non-positive entries).
fn normalised(list: &mut Vec<(u16, f32)>) {
    list.retain(|e| e.1 > 0.0);
    let sum: f32 = list.iter().map(|e| e.1).sum();
    if sum > 0.0 {
        for e in list.iter_mut() {
            e.1 /= sum;
        }
    }
}

/// Keeps the `k` largest weights continuously: subtracts the (k+1)-th largest from all of them, so
/// a joint that changes rank with the one it replaces has (almost) no weight at that moment and
/// the truncation leaves no step in the surface.
fn sparsify(list: &mut Vec<(u16, f32)>, k: usize) {
    normalised(list);
    list.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    if list.len() > k {
        let floor = list[k].1;
        list.truncate(k);
        for e in list.iter_mut() {
            e.1 -= floor;
        }
    }
    normalised(list);
}

/// Smooths the weights of the `body` part over its welded surface (UV-seam copies count as one
/// vertex, so they keep equal weights).
fn smooth(mesh: &Mesh, lists: &mut [Vec<(u16, f32)>], iterations: usize) {
    let Some(body) = mesh.parts().iter().find(|p| p.name == "body") else { return };
    let keys = mesh.position_keys();
    let key_count = keys.iter().map(|&k| k as usize + 1).max().unwrap_or(0);
    // Welded neighbour lists of the body.
    let mut neighbours: Vec<Vec<u32>> = vec![Vec::new(); key_count];
    let mut member = vec![u32::MAX; key_count];
    for f in body.faces() {
        let c = mesh.face(f);
        for k in 0..c.len() {
            let (a, b) = (keys[c[k] as usize], keys[c[(k + 1) % c.len()] as usize]);
            member[a as usize] = c[k];
            if a != b {
                for (x, y) in [(a, b), (b, a)] {
                    let list = &mut neighbours[x as usize];
                    if !list.contains(&y) {
                        list.push(y);
                    }
                }
            }
        }
    }
    let mut current: Vec<Vec<(u16, f32)>> = vec![Vec::new(); key_count];
    for (k, &v) in member.iter().enumerate() {
        if v != u32::MAX
            && let Some(list) = lists.get(v as usize)
        {
            current[k].clone_from(list);
            normalised(&mut current[k]);
        }
    }
    let mut next = current.clone();
    for _ in 0..iterations {
        for (k, out) in next.iter_mut().enumerate() {
            let nb = &neighbours[k];
            if nb.is_empty() {
                continue;
            }
            out.clear();
            accumulate(out, &current[k], 0.5);
            let w = 0.5 / nb.len() as f32;
            for &n in nb {
                accumulate(out, &current[n as usize], w);
            }
        }
        std::mem::swap(&mut current, &mut next);
    }
    for f in body.faces() {
        for &v in mesh.face(f) {
            let k = keys[v as usize] as usize;
            if let Some(list) = lists.get_mut(v as usize) {
                list.clone_from(&current[k]);
            }
        }
    }
}

/// Applies [`axilla`] to a segment-rule weight list: the arm joints' share becomes the axilla's
/// arm share (all on the upper arm), the trunk joints keep their proportions in the rest.
fn blend_axilla(list: &mut Vec<(u16, f32)>, p: Vec3, a: &Anatomy, arms: &[(u16, Vec<u16>); 2]) {
    let side = if p.x >= 0.0 { 1.0 } else { -1.0 };
    let (share, applies) = axilla(p, sided(a.shoulder, side), side);
    if applies <= 0.0 {
        return;
    }
    let upper = arms[usize::from(side < 0.0)].0;
    normalised(list);
    let mut trunk: Vec<(u16, f32)> =
        list.iter().copied().filter(|e| arms.iter().all(|(_, arm)| !arm.contains(&e.0))).collect();
    normalised(&mut trunk);
    let original = std::mem::take(list);
    accumulate(list, &original, 1.0 - applies);
    if trunk.is_empty() {
        // No trunk joint in this rule: the rule only moves weight between the arm joints.
        accumulate(list, &[(upper, 1.0)], applies);
    } else {
        accumulate(list, &trunk, applies * (1.0 - share));
        accumulate(list, &[(upper, 1.0)], applies * share);
    }
    normalised(list);
}

/// The submental skin (the floor under the chin, `h` in the head frame, mm) follows the jaw in
/// part: full share at the chin's border fading to none on the throat and toward the neck angle,
/// so an open jaw stretches the skin under the chin instead of folding it into the neck.
fn blend_submental(list: &mut Vec<(u16, f32)>, h: Vec3, jaw: u16) {
    let share = 0.75 * smoothstep((h.y + 140.0) / 30.0) * smoothstep((h.z - 5.0) / 35.0);
    if share <= 0.0 {
        return;
    }
    normalised(list);
    for e in list.iter_mut() {
        e.1 *= 1.0 - share;
    }
    list.push((jaw, share));
}

/// Authors the skin weights of `mesh` `[SKIN-01]`.
pub fn skin(rig: &Rig, a: &Anatomy, mesh: &Mesh, regions: &RegionMap) -> Result<SkinWeights> {
    let joint =
        |name: &str| rig.index(name).ok_or(ModelError::invalid("skin joint", 0, "joint missing from the skeleton"));
    let (head, jaw) = (joint("head")?, joint("jaw")?);
    // Rules per region, joint names resolved once.
    let mut rules: HashMap<u16, (Rule, Vec<u16>)> = HashMap::new();
    for (i, _) in regions.regions.iter().enumerate() {
        let id = u16::try_from(i).map_err(|_| ModelError::invalid("region", i, "region index exceeds u16"))?;
        let Some(r) = R::from_id(id) else { continue };
        let rl = rule(r);
        let ids = match &rl {
            Rule::Segments(names) => names.iter().map(|n| joint(n)).collect::<Result<Vec<u16>>>()?,
            Rule::Joint(n) => vec![joint(n)?],
            Rule::Lid { helper, .. } => vec![joint(helper)?],
            Rule::LowerFace { .. } => Vec::new(),
        };
        rules.insert(id, (rl, ids));
    }
    let twist: HashMap<u16, u16> = ["L.upper-arm", "R.upper-arm", "L.forearm", "R.forearm"]
        .iter()
        .filter_map(|n| Some((rig.index(n)?, rig.index(&format!("{n}.twist"))?)))
        .collect();
    let to_head = |p: Vec3| (p - a.head_origin) * 1000.0;
    // Per side (left, right): the upper-arm joint and every arm joint that is not trunk.
    let arm_side = |l: &str| -> Result<(u16, Vec<u16>)> {
        let upper = joint(&format!("{l}.upper-arm"))?;
        let all = ["upper-arm", "upper-arm.twist", "forearm", "forearm.twist"]
            .iter()
            .map(|n| joint(&format!("{l}.{n}")))
            .collect::<Result<Vec<u16>>>()?;
        Ok((upper, all))
    };
    let arms = [arm_side("L")?, arm_side("R")?];
    let mut lists: Vec<Vec<(u16, f32)>> = Vec::with_capacity(mesh.vertex_count());
    for (v, &p) in mesh.positions().iter().enumerate() {
        let r = regions.vertex_regions.get(v).copied().unwrap_or(0);
        let mut list: Vec<(u16, f32)> = Vec::with_capacity(8);
        match rules.get(&r) {
            Some((Rule::Segments(_), ids)) => {
                for &j in ids {
                    let seg = rig.segments.get(j as usize).copied().unwrap_or((p, p));
                    let (d, t) = segment_distance(p, seg);
                    let w = (0.004 + d).powi(-4);
                    match twist.get(&j) {
                        Some(&tw) => {
                            let share = smoothstep((t - 0.15) / 0.75);
                            list.push((j, w * (1.0 - share)));
                            list.push((tw, w * share));
                        }
                        None => list.push((j, w)),
                    }
                }
            }
            Some((Rule::Joint(_), ids)) => list.extend(ids.iter().map(|&j| (j, 1.0))),
            Some((Rule::Lid { scale, .. }, ids)) => {
                let side = if p.x >= 0.0 { 1.0 } else { -1.0 };
                let d = sided(a.eye_center, side).distance(p) * 1000.0;
                let w = scale * smoothstep((a.eye_radius * 1000.0 + 5.0 - d) / 3.3);
                if let Some(&h) = ids.first() {
                    list.push((h, w));
                }
                list.push((head, 1.0 - w));
            }
            Some((Rule::LowerFace { full_jaw }, _)) => {
                let h = to_head(p);
                let mut w = if *full_jaw { 1.0 } else { smoothstep((-46.0 - h.y) / 10.0) };
                if !*full_jaw && h.z < 30.0 {
                    // Toward the ear the jaw's share fades: the ramus moves little near the joint.
                    w *= smoothstep((h.z + 10.0) / 40.0);
                }
                list.push((jaw, w));
                list.push((head, 1.0 - w));
            }
            None => list.push((head, 1.0)),
        }
        if matches!(rules.get(&r), Some((Rule::Segments(_), _))) {
            blend_axilla(&mut list, p, a, &arms);
        }
        if matches!(R::from_id(r), Some(R::NeckFront | R::NeckSideL | R::NeckSideR)) {
            blend_submental(&mut list, to_head(p), jaw);
        }
        lists.push(list);
    }
    smooth(mesh, &mut lists, SMOOTHING_ITERATIONS);
    for list in &mut lists {
        sparsify(list, INFLUENCES);
    }
    let mut skin = SkinWeights::from_lists(INFLUENCES, lists.iter().map(Vec::as_slice), 0)?;
    let mut dq: Vec<u16> = [R::Neck, R::ShoulderL, R::ShoulderR, R::ForearmL, R::ForearmR, R::HandLPalm, R::HandRPalm]
        .iter()
        .map(|r| r.id())
        .collect();
    dq.sort_unstable();
    skin.dual_quaternion_regions = dq;
    Ok(skin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skin_01_sparsify_is_continuous_when_ranks_swap() {
        // Joints 3 and 4 swap rank 4/5: just before and after the swap the result barely differs.
        let base = [(0_u16, 0.4_f32), (1, 0.25), (2, 0.15), (3, 0.1), (4, 0.1)];
        let mut a: Vec<(u16, f32)> = base.iter().map(|&(j, w)| (j, if j == 3 { w + 1e-4 } else { w })).collect();
        let mut b: Vec<(u16, f32)> = base.iter().map(|&(j, w)| (j, if j == 4 { w + 1e-4 } else { w })).collect();
        sparsify(&mut a, 4);
        sparsify(&mut b, 4);
        assert_eq!((a.len(), b.len()), (4, 4));
        let get = |l: &[(u16, f32)], j: u16| l.iter().find(|e| e.0 == j).map_or(0.0, |e| e.1);
        for j in 0..5 {
            assert!((get(&a, j) - get(&b, j)).abs() < 1e-3, "joint {j}: {} vs {}", get(&a, j), get(&b, j));
        }
        assert!((a.iter().map(|e| e.1).sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mod_07_frames_are_rotations_with_y_along_the_bone() {
        let q = frame(Vec3::new(1.0, -1.0, 0.0), Vec3::Z);
        let y = q.mul_vec3(Vec3::Y);
        assert!((y - Vec3::new(1.0, -1.0, 0.0).normalize()).length() < 1e-5);
        // A positive rotation about local X moves the bone toward the flexion side.
        let bent = (q * Quat::from_rotation_x(0.3)).mul_vec3(Vec3::Y);
        assert!(bent.z > 0.0);
        // A flexion hint along the bone still gives a valid frame.
        assert!(frame(Vec3::Y, Vec3::Y).is_normalized());
    }

    #[test]
    fn mod_07_right_side_limits_are_mirrored() {
        let l = limits((-0.1, 0.5), (-0.2, 0.3), (-0.4, 0.6), 1.0);
        let r = limits((-0.1, 0.5), (-0.2, 0.3), (-0.4, 0.6), -1.0);
        assert_eq!((l.min.x, l.max.x), (r.min.x, r.max.x));
        assert_eq!((r.min.y, r.max.y, r.min.z, r.max.z), (-0.3, 0.2, -0.6, 0.4));
        assert_eq!(segment_distance(Vec3::new(1.0, 0.5, 0.0), (Vec3::ZERO, Vec3::Y)), (1.0, 0.5));
    }
}
