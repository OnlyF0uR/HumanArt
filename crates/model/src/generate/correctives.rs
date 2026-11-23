//! Pose correctives for the body's big joints (todo 1.3): elbows, knees, shoulders and hips.
//!
//! Linear blend skinning collapses the inside of a bent joint (the elbow pit pinches, the knee
//! loses its back, the armpit tears when the arm comes down) and twists the skin like a sweet
//! wrapper when a limb rolls. Each corrective here is a rest-space delta that, fully on at its
//! example pose, turns the skinned surface into a better target ([`TargetKind`]):
//!
//! - **dual-quaternion skinning** of the whole body at that pose (no collapse; it bulges a little);
//! - at the elbow, knee and shoulder, that surface **smoothed by Delta Mush** (Mancewicz, Derksen,
//!   Rijpkema and Wilson, "Delta Mush: smoothing deformations while preserving detail", DigiPro
//!   2014): the rest surface is smoothed `k` times (uniform Laplacian over the weld keys of the
//!   skin part) and each vertex's offset from its smoothed position is kept in the smoothed
//!   surface's local frame; the posed surface is smoothed the same way and the offsets added back
//!   in the posed frames. Smoothing commutes with rigid motion, so the rigidly moving skin is left
//!   as it is, while the skinning's folds around the joint are smoothed away and the surface
//!   detail comes back. Delta Mush shrinks a bend, so the body's enclosed volume is then put back
//!   to the rest volume by inflating the non-rigidly skinned skin along its normals.
//!
//! The difference to the default skinning's surface is taken back to rest space through each
//! vertex's skinning map (affine in the rest position for both linear and dual-quaternion
//! skinning), so that skinning the corrected rest shape lands exactly on the target at the
//! example pose. Measured against plain skinning (test
//! `todo_1_3_correctives_keep_volume_and_smooth_folds_at_bent_joints`): the knee bent 126 degrees
//! keeps its volume (plain skinning loses 709 cm³) and its surface is smoother; the elbow, the arm
//! raised forward and the hip flexed have fewer faces turned over.
//!
//! The correctives of one joint are grouped in a [`PoseDriver`] (swing metric: the roll of the
//! bone does not switch them), so several examples blend by radial basis interpolation and every
//! one is off at rest. The right side's example rotations are the mirror images of the left's.
//! The amounts are what the smoothing gives, not sculpted shapes: a starting point the
//! sculpt tools can refine.

use humanart_math::{Mat3, Quat, Vec3};

use crate::delta::DeltaKind;
use crate::error::{ModelError, Result};
use crate::eval::{EvalInput, Evaluator, SkinningMode};
use crate::expression::PoseCorrective;
use crate::expression::pose_space::{PoseDriver, PoseMetric};
use crate::identity::{IdentityComponent, Symmetry};
use crate::template::{ProvenanceEntry, Template};

/// Rest-space displacements below this are dropped from the deltas, metres.
pub const EPSILON: f32 = 2e-5;
/// The kernel radius of every driver, radians: wide enough that the examples of one joint overlap.
pub const DRIVER_RADIUS: f32 = 1.2;

/// One example pose of a left-side joint (the right side is its mirror image).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExamplePose {
    /// The corrective's name with `#` for the side letter (`#.elbow.bend-90`).
    pub name: &'static str,
    /// The left joint with `#` for the side letter (`#.forearm`).
    pub joint: &'static str,
    /// Axis in the joint's frame (Y along the bone, Z toward flexion; see `generate::rig`).
    pub axis: Vec3,
    /// Angle about [`ExamplePose::axis`], radians.
    pub angle: f32,
    /// How its target surface is made.
    pub target: TargetKind,
}

const SMOOTHED: TargetKind = TargetKind::SmoothedDualQuat { iterations: 20, volume_percent: 100 };

const fn example(name: &'static str, joint: &'static str, axis: Vec3, angle: f32, target: TargetKind) -> ExamplePose {
    ExamplePose { name, joint, axis, angle, target }
}

/// The example poses, left side. Flexion is about the joint frame's X axis for the elbow, knee,
/// shoulder (forward raise) and hip; the shoulder's Z turns the arm up out of the A-pose (positive)
/// or down to the side (negative); the hip's Z spreads the leg. Every angle is inside the joint's
/// limits (`generate::rig`).
pub const EXAMPLES: &[ExamplePose] = &[
    example("#.elbow.bend-90", "#.forearm", Vec3::X, 1.57, SMOOTHED),
    example("#.elbow.bend-140", "#.forearm", Vec3::X, 2.45, SMOOTHED),
    example("#.knee.bend-90", "#.shin", Vec3::X, 1.57, SMOOTHED),
    // Not deeper: past about 126 degrees the calf meets the back of the thigh, and without contact
    // handling the target pushes the calf 12 cm (in rest space) into it.
    example("#.knee.bend-125", "#.shin", Vec3::X, 2.2, SMOOTHED),
    // A quarter of the volume for the arm lowered against the side: all of it folded faces at the
    // front of the armpit (2 at the example, against none for a quarter).
    example(
        "#.shoulder.down",
        "#.upper-arm",
        Vec3::Z,
        -0.75,
        TargetKind::SmoothedDualQuat { iterations: 20, volume_percent: 25 },
    ),
    example("#.shoulder.up", "#.upper-arm", Vec3::Z, 1.3, SMOOTHED),
    example("#.shoulder.forward", "#.upper-arm", Vec3::X, 1.5, SMOOTHED),
    example("#.shoulder.back", "#.upper-arm", Vec3::X, -0.9, SMOOTHED),
    example("#.hip.flex-90", "#.thigh", Vec3::X, 1.57, TargetKind::DualQuat),
    example("#.hip.extend", "#.thigh", Vec3::X, -0.45, TargetKind::DualQuat),
    example("#.hip.spread", "#.thigh", Vec3::Z, 0.7, TargetKind::DualQuat),
];

/// What [`with_pose_correctives`] built, per corrective: name, moved vertices, largest
/// displacement (metres).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CorrectiveReport {
    /// One line per corrective.
    pub correctives: Vec<(String, usize, f32)>,
}

/// The rotation of a right-side joint that mirrors the rotation `left` of its left twin across
/// the body's midplane (x → −x), both relative to their rest frames.
#[must_use]
pub fn mirrored(left: Quat, left_rest: Quat, right_rest: Quat) -> Quat {
    let world = left_rest * left * left_rest.conjugate();
    // A reflection through the YZ plane keeps the x component of a rotation's axis and flips y, z.
    let mirrored = Quat::from_xyzw(world.x, -world.y, -world.z, world.w);
    (right_rest.conjugate() * mirrored * right_rest).normalize()
}

/// The smoothed positions of the weld keys of the parts selected by `on_key` (others keep their
/// positions). `positions` is per render vertex; the result is per render vertex too, duplicates
/// sharing their key's value.
struct Smoother {
    /// Per render vertex: its weld key.
    keys: Vec<u32>,
    /// The weld keys that take part, in order.
    active: Vec<u32>,
    /// Neighbour lists of the active keys (keys).
    neighbours: crate::topology::Csr,
    /// The first neighbour of each active key (the frame's tangent direction).
    first: Vec<u32>,
    /// The part's triangles as weld keys (for the smoothed surfaces' normals).
    triangles: Vec<[u32; 3]>,
}

impl Smoother {
    fn new(template: &Template, part: &str) -> Result<Self> {
        let mesh = template.mesh();
        let topology = mesh.topology();
        let keys = mesh.position_keys().to_vec();
        let part = mesh
            .parts()
            .iter()
            .find(|p| p.name == part)
            .ok_or_else(|| ModelError::invalid("pose correctives", 0, "the template has no body part"))?;
        let mut on = vec![false; mesh.vertex_count()];
        for f in part.faces() {
            for &v in mesh.face(f) {
                if let Some(&k) = keys.get(v as usize)
                    && let Some(slot) = on.get_mut(k as usize)
                {
                    *slot = true;
                }
            }
        }
        let active: Vec<u32> = (0..mesh.vertex_count() as u32).filter(|&k| on[k as usize]).collect();
        let mut pairs = Vec::new();
        let mut first = Vec::with_capacity(active.len());
        for (row, &k) in active.iter().enumerate() {
            let ring: Vec<u32> =
                topology.neighbors.row(k as usize).iter().copied().filter(|&n| on[n as usize]).collect();
            first.push(ring.first().copied().unwrap_or(k));
            pairs.extend(ring.into_iter().map(|n| (row as u32, n)));
        }
        let neighbours = crate::topology::Csr::from_pairs(active.len(), &pairs);
        let tri = mesh.triangulate();
        let faces = part.faces();
        let triangles = tri
            .indices
            .as_chunks::<3>()
            .0
            .iter()
            .zip(&tri.triangle_faces)
            .filter(|(_, f)| faces.contains(&(**f as usize)))
            .map(|(t, _)| [keys[t[0] as usize], keys[t[1] as usize], keys[t[2] as usize]])
            .collect();
        Ok(Self { keys, active, neighbours, first, triangles })
    }

    /// `iterations` Laplacian steps of the active keys of `positions` (per render vertex), returned
    /// per render vertex.
    fn smooth(&self, positions: &[Vec3], iterations: usize) -> Vec<Vec3> {
        let mut cur: Vec<Vec3> = positions.to_vec();
        let mut next = cur.clone();
        for _ in 0..iterations {
            for (row, &k) in self.active.iter().enumerate() {
                let ring = self.neighbours.row(row);
                if ring.is_empty() {
                    continue;
                }
                let mean = ring.iter().fold(Vec3::ZERO, |a, &n| a + cur[n as usize]) / ring.len() as f32;
                next[k as usize] = (cur[k as usize] + mean) * 0.5;
            }
            std::mem::swap(&mut cur, &mut next);
        }
        for (v, &k) in self.keys.iter().enumerate() {
            if k as usize != v {
                cur[v] = cur[k as usize];
            }
        }
        cur
    }

    /// The area-weighted vertex normals (unnormalised) of the surface `s` (per render vertex), per
    /// weld key.
    fn normals(&self, s: &[Vec3]) -> Vec<Vec3> {
        let mut n = vec![Vec3::ZERO; s.len()];
        for &[a, b, c] in &self.triangles {
            let (pa, pb, pc) = (s[a as usize], s[b as usize], s[c as usize]);
            let f = (pb - pa).cross(pc - pa);
            for k in [a, b, c] {
                n[k as usize] += f;
            }
        }
        n
    }

    /// The local frame (tangent, bitangent, normal as columns) of active key `row` on the smoothed
    /// surface `s` (per render vertex) with normals `normals`: the tangent toward the first
    /// neighbour.
    fn frame(&self, s: &[Vec3], normals: &[Vec3], row: usize) -> Mat3 {
        let k = self.active[row] as usize;
        let n = normals[k].normalize_or(Vec3::Z);
        let t = s[self.first[row] as usize] - s[k];
        let t = t - n * t.dot(n);
        let t = t.normalize_or(n.any_orthonormal_vector());
        Mat3::from_cols(t, n.cross(t), n)
    }
}

/// Builds the Delta Mush target of the posed surface `posed` (per render vertex) given the rest
/// surface `rest`, both on the smoother's keys.
fn delta_mush(smoother: &Smoother, rest: &[Vec3], posed: &[Vec3], iterations: usize) -> Vec<Vec3> {
    let s_rest = smoother.smooth(rest, iterations);
    let s_posed = smoother.smooth(posed, iterations);
    let (n_rest, n_posed) = (smoother.normals(&s_rest), smoother.normals(&s_posed));
    let mut out = posed.to_vec();
    for (row, &k) in smoother.active.iter().enumerate() {
        let k = k as usize;
        let fr = smoother.frame(&s_rest, &n_rest, row);
        let fp = smoother.frame(&s_posed, &n_posed, row);
        let local = fr.transpose() * (rest[k] - s_rest[k]);
        out[k] = s_posed[k] + fp * local;
    }
    for (v, &k) in smoother.keys.iter().enumerate() {
        if k as usize != v {
            out[v] = out[k as usize];
        }
    }
    out
}

/// How the target surface of an example pose is made (measured on the template against plain
/// skinning, `target/agents/correctives/metrics.txt`: enclosed volume, faces turned over and the
/// mean change of the angle between neighbouring faces near the joint).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetKind {
    /// Dual-quaternion skinning everywhere: no collapse, but it bulges. Best at the hip, where
    /// putting the volume back folds the groin.
    DualQuat,
    /// Delta Mush with `iterations` smoothing iterations of the dual-quaternion surface, then
    /// `volume_percent` of the enclosed volume it lost put back by inflating the skin where the
    /// moving and the still bones blend along its normals. Best at the elbow, knee and shoulder
    /// (all the volume; a quarter for the arm lowered against the side, where more folds the armpit).
    SmoothedDualQuat {
        /// Smoothing iterations.
        iterations: usize,
        /// Share of the lost volume put back, percent.
        volume_percent: u8,
    },
}

/// The signed volume enclosed by the smoother's triangles over `p` (holes cancel in differences).
fn enclosed_volume(smoother: &Smoother, p: &[Vec3]) -> f32 {
    smoother.triangles.iter().map(|&[a, b, c]| p[a as usize].dot(p[b as usize].cross(p[c as usize])) / 6.0).sum()
}

/// `target` with `share` of the enclosed volume it lost against `rest` put back by moving each key along
/// the target's normal by `δ · w`, δ found by two secant steps. `w` is where the skinning blends
/// the bones that move in this pose (`moving`: the posed joint and its descendants) with those
/// that stay: `4 s (1 − s)` for the vertex's weight `s` on the moving bones, smoothed, so the
/// volume comes back around the bent joint and nowhere else.
fn restore_volume(
    smoother: &Smoother,
    template: &Template,
    moving: &[bool],
    rest: &[Vec3],
    target: &[Vec3],
    share: f32,
) -> Vec<Vec3> {
    let skin = template.skin();
    let raw: Vec<Vec3> = (0..rest.len())
        .map(|v| {
            let (js, ws) = skin.vertex(v);
            let s: f32 = js
                .iter()
                .zip(ws)
                .filter(|(j, _)| moving.get(**j as usize).copied().unwrap_or(false))
                .map(|x| *x.1)
                .sum();
            let s = s.clamp(0.0, 1.0);
            Vec3::new(4.0 * s * (1.0 - s), 0.0, 0.0)
        })
        .collect();
    let w = smoother.smooth(&raw, 10);
    let normals = smoother.normals(target);
    let moved = |delta: f32| -> Vec<Vec3> {
        let mut out = target.to_vec();
        for &k in &smoother.active {
            let k = k as usize;
            out[k] += normals[k].normalize_or(Vec3::ZERO) * (delta * w[k].x.clamp(0.0, 1.0));
        }
        for (v, &k) in smoother.keys.iter().enumerate() {
            if k as usize != v {
                out[v] = out[k as usize];
            }
        }
        out
    };
    let now = enclosed_volume(smoother, target);
    let goal = now + (enclosed_volume(smoother, rest) - now) * share.clamp(0.0, 1.0);
    let f = |delta: f32| enclosed_volume(smoother, &moved(delta)) - goal;
    let (mut a, mut fa) = (0.0_f32, f(0.0));
    let (mut b, mut fb) = (0.001_f32, f(0.001));
    for _ in 0..3 {
        if (fb - fa).abs() < 1e-12 {
            break;
        }
        let c = (b - (fb * (b - a) / (fb - fa))).clamp(-0.02, 0.02);
        (a, fa) = (b, fb);
        (b, fb) = (c, f(c));
    }
    moved(b)
}

/// The template with the body pose correctives of [`EXAMPLES`] (both sides) added, grouped in one
/// swing driver per joint. Refuses a template that already has pose correctives.
pub fn with_pose_correctives(template: &Template) -> Result<(Template, CorrectiveReport)> {
    if !template.expressions().pose_correctives.is_empty() {
        return Err(ModelError::invalid("pose correctives", 0, "the template already has some"));
    }
    let skeleton = template.skeleton();
    let rest_world = skeleton.rest_world();
    let smoother = Smoother::new(template, "body")?;
    let mut eval = Evaluator::new(std::sync::Arc::new(template.clone()));
    let rest = eval.positions().to_vec();
    let mut pose = vec![Quat::IDENTITY; skeleton.joints.len()];
    let mut components = Vec::new();
    let mut correctives = Vec::new();
    let mut report = CorrectiveReport::default();
    for side in ["L", "R"] {
        for ex in EXAMPLES {
            let joint_name = ex.joint.replace('#', side);
            let name = ex.name.replace('#', side);
            let j = skeleton
                .find(&joint_name)
                .ok_or_else(|| ModelError::invalid("pose correctives", 0, "joint missing"))?;
            let left = Quat::from_axis_angle(ex.axis, ex.angle);
            let rotation = if side == "L" {
                left
            } else {
                let lj = skeleton
                    .find(&ex.joint.replace('#', "L"))
                    .ok_or_else(|| ModelError::invalid("pose correctives", 0, "left joint missing"))?;
                let (lr, rr) = (rest_world[lj as usize].rotation, rest_world[j as usize].rotation);
                mirrored(left, lr, rr)
            };
            pose.fill(Quat::IDENTITY);
            pose[j as usize] = rotation;
            let mut evaluate = |skinning: SkinningMode| {
                eval.evaluate(&EvalInput { pose: &pose, skinning, ..EvalInput::default() })
                    .map_err(|_| ModelError::invalid("pose correctives", 0, "an example pose could not be evaluated"))
                    .map(|()| eval.positions().to_vec())
            };
            let dual = evaluate(SkinningMode::DualQuaternion)?;
            let target = match ex.target {
                TargetKind::DualQuat => dual,
                TargetKind::SmoothedDualQuat { iterations, volume_percent } => {
                    let moving = subtree(skeleton, j);
                    let smoothed = delta_mush(&smoother, &rest, &dual, iterations);
                    restore_volume(&smoother, template, &moving, &rest, &smoothed, f32::from(volume_percent) / 100.0)
                }
            };
            // The corrective is applied under the default skinning: its rest-space delta inverts
            // that skinning's map at the example pose.
            let posed = evaluate(SkinningMode::ByRegion)?;
            let entries = rest_deltas(&eval, &rest, &posed, &target);
            let max = entries.iter().map(|e| e.1.length()).fold(0.0, f32::max);
            report.correctives.push((name.clone(), entries.len(), max));
            let region = crate::expression::build::bounding_region(template, entries.iter().map(|e| e.0));
            components.push(IdentityComponent {
                name: name.clone(),
                kind: DeltaKind::PoseCorrective,
                region,
                entries,
                joints: Vec::new(),
                symmetry: Symmetry::Free,
                allowed: Vec::new(),
            });
            correctives.push((name, j, rotation));
        }
    }
    let with = crate::identity::add_components(template, &components)?;
    let mut parts = with.into_parts();
    let first = parts.expressions.pose_correctives.len() as u32;
    for (name, joint, target) in &correctives {
        let component = parts
            .deltas
            .find(name)
            .ok_or_else(|| ModelError::invalid("pose correctives", 0, "component missing after the build"))?;
        parts.expressions.pose_correctives.push(PoseCorrective {
            name: name.clone(),
            joint: *joint,
            target: *target,
            radius: DRIVER_RADIUS,
            component,
        });
    }
    // One driver per joint, its correctives in example order.
    let mut drivers: Vec<PoseDriver> = Vec::new();
    for (i, (_, joint, _)) in correctives.iter().enumerate() {
        let index = first + i as u32;
        if let Some(d) = drivers.iter_mut().find(|d| d.joint == *joint) {
            d.correctives.push(index);
        } else {
            let name = skeleton.joints.get(*joint as usize).map_or_else(String::new, |j| format!("{}.pose", j.name));
            drivers.push(PoseDriver {
                name,
                joint: *joint,
                metric: PoseMetric::Swing,
                radius: DRIVER_RADIUS,
                correctives: vec![index],
            });
        }
    }
    parts.expressions.pose_drivers.extend(drivers);
    parts.info.provenance.push(ProvenanceEntry {
        item: "body pose correctives (todo 1.3)".into(),
        source: "derived by humanart-model generate::correctives: Delta Mush targets (Mancewicz et al. 2014) of the skinned template at authored example poses, solved by radial basis interpolation after OpenRigLogic (MIT, Epic Games)".into(),
        licence: parts.info.licence.clone(),
    });
    Ok((Template::new(parts)?, report))
}

/// The template without its pose correctives and drivers (their delta components removed, every
/// other component kept and re-indexed): the base [`with_pose_correctives`] derives them on, e.g.
/// again after the rest shape or the skin weights changed.
pub fn without_pose_correctives(template: &Template) -> Result<Template> {
    let names: Vec<String> = template
        .expressions()
        .pose_correctives
        .iter()
        .filter_map(|p| template.deltas().components().get(p.component as usize).map(|c| c.name.clone()))
        .collect();
    let mut parts = template.clone().into_parts();
    parts.expressions.pose_correctives.clear();
    parts.expressions.pose_drivers.clear();
    let bare = Template::new(parts)?;
    Ok(crate::identity::rebuild_keeping(&bare, |name| !names.iter().any(|n| n == name))?)
}

/// Per joint: whether it is `root` or one of its descendants (parents come before children).
fn subtree(skeleton: &crate::skeleton::Skeleton, root: u16) -> Vec<bool> {
    let mut out = vec![false; skeleton.joints.len()];
    for (i, joint) in skeleton.joints.iter().enumerate() {
        out[i] = i == root as usize || joint.parent.is_some_and(|p| out.get(p as usize).copied().unwrap_or(false));
    }
    out
}

/// The rest-space deltas that move the skinned surface `posed` onto `target`: per vertex, the
/// inverse of the vertex's skinning map applied to the difference (found by finite differences:
/// both skinning methods are affine in the rest position).
fn rest_deltas(eval: &Evaluator, rest: &[Vec3], posed: &[Vec3], target: &[Vec3]) -> Vec<(u32, Vec3)> {
    let skin = eval.template().skin();
    let joints = eval.joints();
    let mut out = Vec::new();
    for (v, ((&p, &x), &t)) in rest.iter().zip(posed).zip(target).enumerate() {
        let d = t - x;
        if d.length() < EPSILON * 0.25 {
            continue;
        }
        let (js, ws) = skin.vertex(v);
        if js.is_empty() {
            continue;
        }
        let map = |q: Vec3| {
            if eval.uses_dual_quaternions(v) { joints.skin_dual_quat(q, js, ws) } else { joints.skin_linear(q, js, ws) }
        };
        let o = map(p);
        let m = Mat3::from_cols(map(p + Vec3::X) - o, map(p + Vec3::Y) - o, map(p + Vec3::Z) - o);
        let Some(inv) = m.inverse() else { continue };
        let r = inv * d;
        if r.length() >= EPSILON && r.is_finite() {
            out.push((v as u32, r));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, OnceLock};

    use super::*;
    use crate::generate::build::{Options, template};

    struct Fixture {
        plain: Arc<Template>,
        corrected: Arc<Template>,
        report: CorrectiveReport,
    }

    fn fixture() -> &'static Fixture {
        static F: OnceLock<Fixture> = OnceLock::new();
        F.get_or_init(|| {
            // The generator adds them: strip them and derive them again.
            let built = template(&Options { lods: false, ..Options::default() }).unwrap().template;
            let plain = without_pose_correctives(&built).unwrap();
            assert!(plain.expressions().pose_correctives.is_empty());
            assert_eq!(plain.deltas().len() + 2 * EXAMPLES.len(), built.deltas().len());
            let (corrected, report) = with_pose_correctives(&plain).unwrap();
            assert_eq!(corrected.deltas().len(), built.deltas().len());
            Fixture { plain: Arc::new(plain), corrected: Arc::new(corrected), report }
        })
    }

    /// Signed enclosed volume of the body part (m^3); holes elsewhere cancel in differences.
    fn body_volume(t: &Template, p: &[Vec3]) -> f32 {
        let mesh = t.mesh();
        let tri = mesh.triangulate();
        let body = mesh.parts().iter().find(|x| x.name == "body").unwrap().faces();
        tri.indices
            .as_chunks::<3>()
            .0
            .iter()
            .zip(&tri.triangle_faces)
            .filter(|(_, f)| body.contains(&(**f as usize)))
            .map(|(c, _)| p[c[0] as usize].dot(p[c[1] as usize].cross(p[c[2] as usize])) / 6.0)
            .sum()
    }

    /// Faces of the body part whose normal turned more than 90 degrees against the rest shape's,
    /// and the mean change of the angle between neighbouring face normals (degrees) within
    /// `radius` of `centre`.
    fn folds(t: &Template, rest: &[Vec3], posed: &[Vec3], centre: Vec3, radius: f32) -> (usize, f32) {
        let mesh = t.mesh();
        let body = mesh.parts().iter().find(|x| x.name == "body").unwrap().faces();
        let normal = |p: &[Vec3], f: usize| crate::mesh::polygon_vector_area(mesh.face(f), p);
        let mut turned = 0;
        for f in body.clone() {
            let c = mesh.face(f);
            if rest[c[0] as usize].distance(centre) > radius {
                continue;
            }
            if normal(rest, f).dot(normal(posed, f)) < 0.0 {
                turned += 1;
            }
        }
        let topo = mesh.topology();
        let (mut sum, mut n) = (0.0, 0);
        for (e, &[a, _]) in topo.edges.iter().enumerate() {
            let faces = topo.edge_faces.row(e);
            if faces.len() != 2 || rest[a as usize].distance(centre) > radius {
                continue;
            }
            let (f0, f1) = (faces[0] as usize, faces[1] as usize);
            if !body.contains(&f0) || !body.contains(&f1) {
                continue;
            }
            let angle = |p: &[Vec3]| {
                let (u, v) = (normal(p, f0).normalize_or(Vec3::Z), normal(p, f1).normalize_or(Vec3::Z));
                u.dot(v).clamp(-1.0, 1.0).acos().to_degrees()
            };
            sum += (angle(posed) - angle(rest)).abs();
            n += 1;
        }
        (turned, sum / n.max(1) as f32)
    }

    fn pose_one(t: &Arc<Template>, joint: &str, q: Quat) -> (Vec<Vec3>, Vec<Vec3>) {
        let mut e = Evaluator::new(t.clone());
        let rest = e.positions().to_vec();
        let mut pose = vec![Quat::IDENTITY; t.skeleton().joints.len()];
        pose[t.skeleton().find(joint).unwrap() as usize] = q;
        e.evaluate(&EvalInput { pose: &pose, ..EvalInput::default() }).unwrap();
        (rest, e.positions().to_vec())
    }

    #[test]
    #[ignore = "diagnostic"]
    fn diag_target_kinds() {
        let f = fixture();
        let t = &f.plain;
        let smoother = Smoother::new(t, "body").unwrap();
        for (joint, axis, angle, centre_name) in [
            ("L.forearm", Vec3::X, 2.2_f32, "L.elbow.lateral"),
            ("L.shin", Vec3::X, 2.2, "L.knee.center"),
            ("L.upper-arm", Vec3::Z, -0.75, "L.shoulder.acromion"),
            ("L.upper-arm", Vec3::X, 1.3, "L.shoulder.acromion"),
            ("L.thigh", Vec3::X, 1.4, "torso.pubis"),
        ] {
            let q = Quat::from_axis_angle(axis, angle);
            let centre = t.landmarks().iter().find(|l| l.name == centre_name).unwrap();
            let centre = centre.position(t.mesh().positions()).unwrap();
            let (rest, posed) = pose_one(t, joint, q);
            let mut e = Evaluator::new(t.clone());
            let mut pose = vec![Quat::IDENTITY; t.skeleton().joints.len()];
            pose[t.skeleton().find(joint).unwrap() as usize] = q;
            e.evaluate(&EvalInput { pose: &pose, skinning: SkinningMode::DualQuaternion, ..EvalInput::default() })
                .unwrap();
            let dq = e.positions().to_vec();
            let v0 = enclosed_volume(&smoother, &rest);
            let report = |name: &str, p: &[Vec3]| {
                let (turned, dihedral) = folds(t, &rest, p, centre, 0.15);
                println!(
                    "{joint} {angle}: {name}: volume {:+.1} cm3, turned {turned}, dihedral {dihedral:.2}",
                    (enclosed_volume(&smoother, p) - v0) * 1e6
                );
            };
            report("skinned", &posed);
            report("dual quat", &dq);
            for k in [10, 20, 40] {
                let dm = delta_mush(&smoother, &rest, &posed, k);
                report(&format!("delta mush {k}"), &dm);
                report(
                    &format!("delta mush {k} + volume"),
                    &restore_volume(
                        &smoother,
                        t,
                        &subtree(t.skeleton(), t.skeleton().find(joint).unwrap()),
                        &rest,
                        &dm,
                        1.0,
                    ),
                );
            }
            report(
                "dual quat + volume",
                &restore_volume(
                    &smoother,
                    t,
                    &subtree(t.skeleton(), t.skeleton().find(joint).unwrap()),
                    &rest,
                    &dq,
                    1.0,
                ),
            );
            for k in [3, 6] {
                let d = delta_mush(&smoother, &rest, &dq, k);
                report(
                    &format!("delta mush {k} of dual quat + volume"),
                    &restore_volume(
                        &smoother,
                        t,
                        &subtree(t.skeleton(), t.skeleton().find(joint).unwrap()),
                        &rest,
                        &d,
                        1.0,
                    ),
                );
            }
            let dmq = delta_mush(&smoother, &rest, &dq, 20);
            report("delta mush 20 of dual quat", &dmq);
            report(
                "delta mush 20 of dual quat + volume",
                &restore_volume(
                    &smoother,
                    t,
                    &subtree(t.skeleton(), t.skeleton().find(joint).unwrap()),
                    &rest,
                    &dmq,
                    1.0,
                ),
            );
            for share in [0.25, 0.5, 0.75] {
                report(
                    &format!("delta mush 20 of dual quat + {share} volume"),
                    &restore_volume(
                        &smoother,
                        t,
                        &subtree(t.skeleton(), t.skeleton().find(joint).unwrap()),
                        &rest,
                        &dmq,
                        share,
                    ),
                );
            }
        }
    }

    #[test]
    fn todo_1_3_every_example_has_a_corrective_and_one_driver_per_joint() {
        let f = fixture();
        let set = f.corrected.expressions();
        assert_eq!(set.pose_correctives.len(), 2 * EXAMPLES.len());
        assert_eq!(set.pose_drivers.len(), 8, "elbow, knee, shoulder, hip on both sides");
        for (name, moved, max) in &f.report.correctives {
            println!("{name}: {moved} vertices, at most {:.1} mm", max * 1e3);
            assert!(*moved > 50, "{name} moves only {moved} vertices");
            // A sanity bound against runaway targets: the largest real one is the calf at the
            // 126-degree knee (74 mm), where plain skinning collapses it toward the knee axis.
            assert!(*max > 5e-4 && *max < 0.08, "{name}: {max} m");
        }
        // The right side is the mirror image of the left.
        for ex in EXAMPLES {
            let line = |side: &str| {
                let n = ex.name.replace('#', side);
                f.report.correctives.iter().find(|c| c.0 == n).unwrap().clone()
            };
            let (l, r) = (line("L"), line("R"));
            assert!((l.2 - r.2).abs() < 0.1 * l.2.max(r.2) + 2e-4, "{} {:?} vs {:?}", ex.name, l, r);
        }
        // A package round trip keeps the drivers.
        let bytes = crate::write_package(&f.corrected).unwrap();
        let back = crate::read_package(&bytes, &crate::PackageLimits::DEFAULT).unwrap();
        assert_eq!(back.expressions().pose_drivers, set.pose_drivers);
    }

    #[test]
    fn todo_1_3_correctives_are_off_at_rest_and_full_at_their_example() {
        let f = fixture();
        let t = &f.corrected;
        let mut e = Evaluator::new(t.clone());
        e.evaluate(&EvalInput::default()).unwrap();
        assert!(e.pose_activations().iter().all(|&a| a.abs() < 1e-4));
        let set = t.expressions();
        let mut pose = vec![Quat::IDENTITY; t.skeleton().joints.len()];
        for (i, pc) in set.pose_correctives.iter().enumerate() {
            pose.fill(Quat::IDENTITY);
            pose[pc.joint as usize] = pc.target;
            e.evaluate(&EvalInput { pose: &pose, ..EvalInput::default() }).unwrap();
            let a = e.pose_activations();
            assert!((a[i] - 1.0).abs() < 2e-3, "{} at its example: {}", pc.name, a[i]);
            for (j, &x) in a.iter().enumerate() {
                if j != i {
                    assert!(x.abs() < 2e-3, "{} on at the example of {}: {x}", set.pose_correctives[j].name, pc.name);
                }
            }
        }
    }

    #[test]
    fn todo_1_3_correctives_keep_volume_and_smooth_folds_at_bent_joints() {
        let f = fixture();
        let mut lines = Vec::new();
        let mut worse = Vec::new();
        for (joint, axis, angle, centre_name) in [
            ("L.forearm", Vec3::X, 2.2_f32, "L.elbow.lateral"),
            ("L.forearm", Vec3::X, 1.2, "L.elbow.lateral"),
            ("L.shin", Vec3::X, 2.2, "L.knee.center"),
            ("L.upper-arm", Vec3::Z, -0.65, "L.shoulder.acromion"),
            ("L.upper-arm", Vec3::X, 1.3, "L.shoulder.acromion"),
            ("L.thigh", Vec3::X, 1.4, "torso.pubis"),
            ("R.forearm", Vec3::X, 2.2, "R.elbow.lateral"),
        ] {
            let q = if joint.starts_with('R') {
                let s = f.plain.skeleton();
                let rw = s.rest_world();
                let l = s.find(&joint.replacen('R', "L", 1)).unwrap() as usize;
                let r = s.find(joint).unwrap() as usize;
                mirrored(Quat::from_axis_angle(axis, angle), rw[l].rotation, rw[r].rotation)
            } else {
                Quat::from_axis_angle(axis, angle)
            };
            let centre = f.plain.landmarks().iter().find(|l| l.name == centre_name).unwrap();
            let centre = centre.position(f.plain.mesh().positions()).unwrap();
            let (rest, plain) = pose_one(&f.plain, joint, q);
            let (_, fixed) = pose_one(&f.corrected, joint, q);
            let v0 = body_volume(&f.plain, &rest);
            let (dv_plain, dv_fixed) = (body_volume(&f.plain, &plain) - v0, body_volume(&f.plain, &fixed) - v0);
            let (t_plain, r_plain) = folds(&f.plain, &rest, &plain, centre, 0.15);
            let (t_fixed, r_fixed) = folds(&f.plain, &rest, &fixed, centre, 0.15);
            let line = format!(
                "{joint} {axis:?} {angle}: volume {:+.1} -> {:+.1} cm3, turned faces {t_plain} -> {t_fixed}, mean dihedral change {r_plain:.2} -> {r_fixed:.2} deg",
                dv_plain * 1e6,
                dv_fixed * 1e6
            );
            println!("{line}");
            if dv_fixed.abs() > dv_plain.abs() + 2e-6 || t_fixed > t_plain || r_fixed > r_plain * 1.05 {
                worse.push(line.clone());
            }
            lines.push(line);
        }
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/agents/correctives");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("metrics.txt"), lines.join("\n")).unwrap();
        assert!(worse.is_empty(), "{worse:#?}");
    }
}
