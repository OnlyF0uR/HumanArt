//! Synthetic templates for the evaluation tests, and the tests that span the whole pipeline
//! (`[AC-WP09-2]` sweeps, the allocation check, timings).

use std::sync::Arc;

use humanart_core::rng::Rng;
use humanart_math::{Quat, Vec2, Vec3};

use super::evaluator::{EvalInput, Evaluator, FULL_REFRESH_INTERVAL};
use super::skinning::SkinningMode;
use super::*;
use crate::delta::{DeltaBases, DeltaBasesBuilder, DeltaKind};
use crate::expression::{CombinationCorrective, CombineOp, Expression, ExpressionKind, ExpressionSet, PoseCorrective};
use crate::landmark::{Landmark, LandmarkGroup};
use crate::mesh::MeshBuilder;
use crate::names::Side;
use crate::region::{Region, RegionMap};
use crate::skeleton::{JointRegressor, Skeleton, simple_joint};
use crate::skin::SkinWeights;
use crate::template::{Template, TemplateInfo, TemplateParts, UvTile};

// [AC-WP09-2] / [PERF-R1]: this test binary counts allocations per thread.
#[global_allocator]
static ALLOCATOR: humanart_core::alloc::CountingAllocator = humanart_core::alloc::CountingAllocator;

/// A smooth bump: 1 at `centre`, 0 from `radius` on.
fn bump(p: Vec3, centre: Vec2, radius: f32) -> f32 {
    let t = (Vec2::new(p.x, p.y) - centre).length() / radius;
    if t >= 1.0 { 0.0 } else { (1.0 - t * t) * (1.0 - t * t) }
}

/// Names of the identity components of [`face_patch`] (`id.0` ...).
pub fn identity_name(k: usize) -> String {
    format!("id.{k}")
}

/// A synthetic "face patch": an `n` x `n` grid on the unit square (normal +Z) with three regions
/// (face, left, right), `identity` smooth global identity components, a semantic and a regional
/// component, four expressions (smile L/R, brow, jaw), a product combination corrective, a pose
/// corrective on the jaw, a three-joint skeleton (root, head, jaw) with a regressor, skin weights
/// with the right region in dual quaternions, and a landmark on every 4th face along the diagonal
/// and the borders.
pub fn face_patch(n: usize, identity: usize) -> Template {
    let mut b = MeshBuilder::new();
    b.begin_part("patch", false).unwrap();
    for j in 0..=n {
        for i in 0..=n {
            let (x, y) = (i as f32 / n as f32, j as f32 / n as f32);
            b.vertex(Vec3::new(x, y, 0.0), Vec2::new(x, 1.0 - y));
        }
    }
    let id = |i: usize, j: usize| (j * (n + 1) + i) as u32;
    for j in 0..n {
        for i in 0..n {
            b.face(&[id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)], 0).unwrap();
        }
    }
    let mesh = b.finish().unwrap();
    let pos = mesh.positions().to_vec();

    let side = |p: Vec3| if p.x < 0.5 { 1_u16 } else { 2 };
    let regions = RegionMap {
        regions: vec![
            Region { name: "face".into(), parent: None, side: Side::Center },
            Region { name: "face-L".into(), parent: Some(0), side: Side::Left },
            Region { name: "face-R".into(), parent: Some(0), side: Side::Right },
        ],
        vertex_regions: pos.iter().map(|&p| side(p)).collect(),
        soft: None,
    };
    let field = |f: &dyn Fn(Vec3) -> Vec3| -> Vec<(u32, Vec3)> {
        pos.iter().enumerate().map(|(i, &p)| (i as u32, f(p))).collect()
    };
    let mut d = DeltaBasesBuilder::new();
    for k in 0..identity {
        let (fx, fy) = ((k % 3 + 1) as f32, (k / 3 % 3 + 1) as f32);
        let phase = k as f32 * 0.7;
        let amp = 0.04 / (1.0 + k as f32 * 0.1);
        let dir = match k % 3 {
            0 => Vec3::Z,
            1 => Vec3::new(0.3, 0.0, 1.0),
            _ => Vec3::new(0.0, 0.4, 1.0),
        };
        d.add(
            &identity_name(k),
            DeltaKind::IdentityGlobal,
            0,
            &field(&|p| dir * (amp * (fx * 3.0 * p.x + phase).sin() * (fy * 2.0 * p.y + phase * 0.5).cos())),
        )
        .unwrap();
    }
    d.add("nose.height", DeltaKind::Semantic, 0, &field(&|p| Vec3::Z * 0.08 * bump(p, Vec2::new(0.5, 0.5), 0.3)))
        .unwrap();
    let right: Vec<(u32, Vec3)> = field(&|p| Vec3::new(0.03, 0.0, 0.02) * bump(p, Vec2::new(0.75, 0.6), 0.2))
        .into_iter()
        .filter(|&(i, _)| side(pos[i as usize]) == 2)
        .collect();
    d.add("cheek-R.width", DeltaKind::IdentityRegional, 2, &right).unwrap();
    let in_side = |s: u16, f: &dyn Fn(Vec3) -> Vec3| -> Vec<(u32, Vec3)> {
        field(f).into_iter().filter(|&(i, _)| side(pos[i as usize]) == s).collect()
    };
    let smile_l = |p: Vec3| Vec3::new(-0.02, 0.05, 0.02) * bump(p, Vec2::new(0.25, 0.4), 0.2);
    let smile_r = |p: Vec3| Vec3::new(0.02, 0.05, 0.02) * bump(p, Vec2::new(0.75, 0.4), 0.2);
    d.add("smile-L", DeltaKind::Expression, 1, &in_side(1, &smile_l)).unwrap();
    d.add("smile-R", DeltaKind::Expression, 2, &in_side(2, &smile_r)).unwrap();
    d.add(
        "brow-raise",
        DeltaKind::Expression,
        0,
        &field(&|p| Vec3::new(0.0, 0.04, 0.01) * bump(p, Vec2::new(0.5, 0.85), 0.3)),
    )
    .unwrap();
    d.add(
        "jaw-drop",
        DeltaKind::Expression,
        0,
        &field(&|p| Vec3::new(0.0, -0.05, 0.01) * bump(p, Vec2::new(0.5, 0.1), 0.35)),
    )
    .unwrap();
    d.add(
        "smile-L.brow-raise",
        DeltaKind::CombinationCorrective,
        1,
        &in_side(1, &|p| Vec3::new(0.0, 0.0, 0.015) * bump(p, Vec2::new(0.3, 0.6), 0.15)),
    )
    .unwrap();
    d.add(
        "jaw.open-fix",
        DeltaKind::PoseCorrective,
        0,
        &field(&|p| Vec3::Z * 0.01 * bump(p, Vec2::new(0.5, 0.2), 0.3)),
    )
    .unwrap();
    let deltas = d.finish().unwrap();
    let find = |name: &str| deltas.find(name).unwrap();
    let mut regressor = JointRegressor::empty(deltas.len());
    // id.0 (when present) lifts the head joint, nose.height pushes the jaw forward.
    let mut entries: Vec<(usize, u16, Vec3)> = vec![(find("nose.height") as usize, 2, Vec3::Z * 0.01)];
    if identity > 0 {
        entries.push((find(&identity_name(0)) as usize, 1, Vec3::Y * 0.02));
    }
    entries.sort_by_key(|e| e.0);
    for c in 0..deltas.len() {
        for &(_, j, o) in entries.iter().filter(|e| e.0 == c) {
            regressor.joints.push(j);
            regressor.deltas.push(o);
        }
        regressor.offsets[c + 1] = regressor.joints.len() as u32;
    }
    let skeleton = Skeleton {
        joints: vec![
            simple_joint("root", None, Vec3::new(0.5, -0.2, -0.2)),
            simple_joint("head", Some(0), Vec3::new(0.0, 0.7, 0.0)),
            simple_joint("jaw", Some(1), Vec3::new(0.0, -0.15, 0.0)),
        ],
        regressor,
    };
    let lists: Vec<Vec<(u16, f32)>> = pos
        .iter()
        .map(|p| {
            let jaw = ((0.4 - p.y) / 0.3).clamp(0.0, 1.0);
            vec![(2, jaw), (1, 1.0 - jaw)]
        })
        .collect();
    let mut skin = SkinWeights::from_lists(4, lists.iter().map(Vec::as_slice), 1).unwrap();
    skin.dual_quaternion_regions = vec![2];
    let expression = |name: &str, side: Side, facs: u16| Expression {
        name: name.into(),
        kind: ExpressionKind::ActionUnit,
        side,
        facs,
        component: find(name),
    };
    let expressions = ExpressionSet {
        expressions: vec![
            expression("smile-L", Side::Left, 12),
            expression("smile-R", Side::Right, 12),
            expression("brow-raise", Side::Center, 1),
            expression("jaw-drop", Side::Center, 26),
        ],
        combinations: vec![CombinationCorrective {
            name: "smile-L.brow-raise".into(),
            op: CombineOp::Product,
            inputs: vec![0, 2],
            component: find("smile-L.brow-raise"),
        }],
        pose_correctives: vec![PoseCorrective {
            name: "jaw.open-fix".into(),
            joint: 2,
            target: Quat::from_rotation_x(0.6),
            radius: 0.5,
            component: find("jaw.open-fix"),
        }],
    };
    let mut landmarks = Vec::new();
    let mut add_landmark = |name: String, i: usize, j: usize| {
        landmarks.push(Landmark {
            name,
            group: LandmarkGroup::Face,
            vertices: [id(i, j), id(i + 1, j), id(i + 1, j + 1)],
            weights: [0.3, 0.3, 0.4],
        });
    };
    let step = (n / 6).max(1);
    let mut k = 0;
    for j in (0..n).step_by(step) {
        for i in (0..n).step_by(step) {
            add_landmark(format!("lm.{k}"), i, j);
            k += 1;
        }
    }
    let parts = TemplateParts {
        info: TemplateInfo { name: "face-patch".into(), ..TemplateInfo::default() },
        mesh,
        tiles: vec![UvTile { name: "face".into(), resolution: 1024, target_texels_per_mm: 0.0 }],
        regions,
        landmarks,
        skeleton,
        skin,
        deltas,
        expressions,
        lods: Vec::new(),
    };
    Template::new(parts).unwrap()
}

/// The template's cube fixture.
pub fn cube() -> Template {
    Template::new(crate::template::fixtures::cube_template()).unwrap()
}

/// A random rotation of at most `max_angle` radians.
pub fn random_rotation(rng: &mut Rng, max_angle: f32) -> Quat {
    let axis = Vec3::new(rng.normal_f32(), rng.normal_f32(), rng.normal_f32()).normalize_or(Vec3::Y);
    Quat::from_axis_angle(axis, rng.range_f32(-max_angle, max_angle))
}

fn all_finite(e: &Evaluator) -> bool {
    e.positions().iter().all(|p| p.is_finite())
        && e.normals().iter().all(|n| n.is_finite() && (n.length() - 1.0).abs() < 1e-3)
        && e.tangents().iter().all(|t| t.is_finite())
        && e.strain().edge_strain().iter().all(|s| s.is_finite())
        && e.strain().region_stretch().iter().all(|s| s.is_finite())
        && e.strain().vertex_strain().iter().all(|s| s.is_finite())
}

#[test]
fn neutral_evaluation_is_the_template() {
    for t in [face_patch(8, 4), cube()] {
        let e = Evaluator::new(Arc::new(t));
        let t = e.template();
        for (a, b) in e.positions().iter().zip(t.mesh().positions()) {
            assert!(a.distance(*b) < 1e-6, "{a:?} {b:?}");
        }
        assert!(e.strain().edge_strain().iter().all(|s| s.abs() < 1e-6));
        assert!(all_finite(&e));
    }
}

#[test]
fn eval_01_steps_add_up() {
    let t = Arc::new(face_patch(8, 3));
    let mut e = Evaluator::new(t.clone());
    let deltas = t.deltas();
    let sem = deltas.find("nose.height").unwrap() as usize;
    let smile = deltas.find("smile-L").unwrap() as usize;
    let mut shape = vec![0.0; deltas.len()];
    shape[sem] = 1.5;
    // An expression component weight in the shape array is ignored.
    shape[smile] = 7.0;
    e.evaluate(&EvalInput { shape: &shape, ..EvalInput::default() }).unwrap();
    let mut expect = t.mesh().positions().to_vec();
    deltas.accumulate(sem, 1.5, &mut expect).unwrap();
    for (a, b) in e.rest_positions().iter().zip(&expect) {
        assert!(a.distance(*b) < 1e-6);
    }
    assert_eq!(e.shape_weights()[smile], 0.0);
    // Expressions + the product combination corrective.
    let ex = [0.5, 0.0, 0.8, 0.0];
    e.evaluate(&EvalInput { shape: &shape, expressions: &ex, ..EvalInput::default() }).unwrap();
    deltas.accumulate(smile, 0.5, &mut expect).unwrap();
    deltas.accumulate(deltas.find("brow-raise").unwrap() as usize, 0.8, &mut expect).unwrap();
    deltas.accumulate(deltas.find("smile-L.brow-raise").unwrap() as usize, 0.4, &mut expect).unwrap();
    assert!((e.combination_activations()[0] - 0.4).abs() < 1e-6);
    for (a, b) in e.unposed_positions().iter().zip(&expect) {
        assert!(a.distance(*b) < 1e-6, "{a:?} {b:?}");
    }
    // Rest pose: skinning is the identity.
    for (a, b) in e.positions().iter().zip(&expect) {
        assert!(a.distance(*b) < 1e-5);
    }
    // The pose corrective turns fully on at its target rotation.
    let pose = [Quat::IDENTITY, Quat::IDENTITY, Quat::from_rotation_x(0.6)];
    e.evaluate(&EvalInput { shape: &shape, expressions: &ex, pose: &pose, ..EvalInput::default() }).unwrap();
    assert!((e.pose_activations()[0] - 1.0).abs() < 1e-5);
    deltas.accumulate(deltas.find("jaw.open-fix").unwrap() as usize, e.pose_activations()[0], &mut expect).unwrap();
    for (a, b) in e.unposed_positions().iter().zip(&expect) {
        assert!(a.distance(*b) < 1e-6);
    }
}

#[test]
fn mod_07_regressor_moves_the_bind_joints() {
    let t = Arc::new(face_patch(6, 2));
    let mut e = Evaluator::new(t.clone());
    let id0 = t.deltas().find("id.0").unwrap() as usize;
    let mut shape = vec![0.0; t.deltas().len()];
    shape[id0] = 2.0;
    e.evaluate(&EvalInput { shape: &shape, ..EvalInput::default() }).unwrap();
    let rest = t.skeleton().rest_world();
    assert!(e.joints().bind_positions[1].distance(rest[1].translation + Vec3::Y * 0.04) < 1e-6);
    // Children follow in the posed frames even though the regressor moved only the head.
    assert!(e.joints().world_positions[2].distance(rest[2].translation) < 1e-6);
    // Rotating the jaw about its regressed bind position: a vertex fully bound to the jaw turns about it.
    let q = Quat::from_rotation_x(0.3);
    e.evaluate(&EvalInput { shape: &shape, pose: &[Quat::IDENTITY, Quat::IDENTITY, q], ..EvalInput::default() })
        .unwrap();
    let jaw = e.joints().bind_positions[2];
    let v0 = 0; // (0, 0): fully on the jaw.
    let p = e.unposed_positions()[v0];
    let expect = q * (p - jaw) + jaw;
    assert!(e.positions()[v0].distance(expect) < 1e-5);
}

#[test]
fn skin_01_dual_quaternions_only_in_their_regions() {
    let t = Arc::new(face_patch(8, 0));
    let mut e = Evaluator::new(t.clone());
    let n_dq = (0..t.mesh().vertex_count()).filter(|&v| e.uses_dual_quaternions(v)).count();
    assert!(n_dq > 0 && n_dq < t.mesh().vertex_count());
    let pose = [Quat::IDENTITY, Quat::from_rotation_y(0.5), Quat::from_rotation_x(0.6)];
    let mut run = |mode| {
        e.evaluate(&EvalInput { pose: &pose, skinning: mode, ..EvalInput::default() }).unwrap();
        e.positions().to_vec()
    };
    let (by_region, linear, dual) =
        (run(SkinningMode::ByRegion), run(SkinningMode::Linear), run(SkinningMode::DualQuaternion));
    let e = Evaluator::new(t.clone());
    for v in 0..by_region.len() {
        let want = if e.uses_dual_quaternions(v) { dual[v] } else { linear[v] };
        assert_eq!(by_region[v], want);
    }
    assert!(linear.iter().zip(&dual).any(|(a, b)| a.distance(*b) > 1e-4), "blended vertices differ");
}

#[test]
fn eval_03_incremental_rest_matches_full_accumulation() {
    let t = Arc::new(face_patch(10, 12));
    let mut e = Evaluator::new(t.clone());
    let mut rng = Rng::new(0x5EED_0003);
    let mut shape = vec![0.0_f32; t.deltas().len()];
    let mut worst = 0.0_f32;
    for step in 0..(3 * FULL_REFRESH_INTERVAL as usize) {
        // One slider moves per step, as when a user drags.
        let c = rng.index(shape.len());
        shape[c] = rng.range_f32(-3.0, 3.0);
        e.update_rest(&shape, &[]).unwrap();
        if step % 37 == 0 {
            let full = t.deformed_positions(
                &shape
                    .iter()
                    .enumerate()
                    .map(
                        |(i, &w)| {
                            if super::evaluator::is_shape_kind(t.deltas().components()[i].kind) { w } else { 0.0 }
                        },
                    )
                    .collect::<Vec<_>>(),
            );
            for (a, b) in e.rest_positions().iter().zip(&full) {
                worst = worst.max(a.distance(*b));
            }
        }
    }
    assert!(worst < 1e-5, "drift {worst}");
    e.refresh_rest().unwrap();
}

#[test]
fn layers_add_replace_and_refuse() {
    let t = Arc::new(face_patch(4, 0));
    let mut e = Evaluator::new(t.clone());
    let v = t.mesh().vertex_count();
    let l = e.add_layer(SculptLayer::new(vec![3], vec![Vec3::Z], v).unwrap()).unwrap();
    e.evaluate(&EvalInput { layers: &[0.5], ..EvalInput::default() }).unwrap();
    assert!((e.rest_positions()[3].z - 0.5).abs() < 1e-6);
    e.replace_layer(l, SculptLayer::new(vec![4], vec![Vec3::X], v).unwrap()).unwrap();
    assert!(e.rest_positions()[3].z.abs() < 1e-6);
    assert!((e.rest_positions()[4].x - (t.mesh().positions()[4].x + 0.5)).abs() < 1e-6);
    assert!(e.replace_layer(9, SculptLayer::default()).is_err());
    let big = SculptLayer::new(vec![(v + 5) as u32], vec![Vec3::X], v + 10).unwrap();
    assert!(e.add_layer(big).is_err());
    assert_eq!(e.layers().len(), 1);
}

#[test]
fn bad_inputs_are_refused_without_change() {
    type Case = (&'static str, Vec<f32>, Vec<f32>, Vec<Quat>);
    let t = Arc::new(face_patch(4, 2));
    let mut e = Evaluator::new(t.clone());
    let before = e.positions().to_vec();
    let c = t.deltas().len();
    let bad: [Case; 6] = [
        ("long shape", vec![0.0; c + 1], vec![], vec![]),
        ("nan shape", vec![f32::NAN], vec![], vec![]),
        ("huge expression", vec![], vec![2.0e4], vec![]),
        ("long expressions", vec![], vec![0.0; 5], vec![]),
        ("zero rotation", vec![], vec![], vec![Quat::from_xyzw(0.0, 0.0, 0.0, 0.0)]),
        ("long pose", vec![], vec![], vec![Quat::IDENTITY; 4]),
    ];
    for (what, shape, ex, pose) in bad {
        let r = e.evaluate(&EvalInput { shape: &shape, expressions: &ex, pose: &pose, ..EvalInput::default() });
        assert!(r.is_err(), "{what}");
        assert_eq!(e.positions(), &before[..], "{what}");
    }
    assert!(e.evaluate(&EvalInput { layers: &[1.0], ..EvalInput::default() }).is_err());
}

/// `[AC-WP09-2]`: randomised sweeps over every input give finite positions, unit normals, finite
/// tangents and strain.
#[test]
fn ac_wp09_2_no_nan_or_inf_over_random_sweeps() {
    for (seed, t) in [(1_u64, face_patch(12, 8)), (2, cube())] {
        let t = Arc::new(t);
        let mut e = Evaluator::new(t.clone());
        let v = t.mesh().vertex_count();
        let layer: Vec<(u32, Vec3)> = (0..v as u32).step_by(3).map(|i| (i, Vec3::new(0.0, 0.0, 0.01))).collect();
        e.add_layer(SculptLayer::from_pairs(&layer, v).unwrap().spread_to_duplicates(t.mesh()).unwrap()).unwrap();
        let mut rng = Rng::new(0xAC09_0002 ^ seed);
        let joints = t.skeleton().joints.len();
        let modes = [SkinningMode::ByRegion, SkinningMode::Linear, SkinningMode::DualQuaternion];
        for case in 0..400 {
            // Wide ranges, including extreme ones every 10th case.
            let scale = if case % 10 == 0 { 100.0 } else { 3.0 };
            let shape: Vec<f32> = (0..t.deltas().len()).map(|_| rng.range_f32(-scale, scale)).collect();
            let ex: Vec<f32> = (0..t.expressions().expressions.len()).map(|_| rng.range_f32(-0.5, 1.5)).collect();
            let pose: Vec<Quat> = (0..joints).map(|_| random_rotation(&mut rng, std::f32::consts::PI)).collect();
            let layers = [rng.range_f32(-scale, scale)];
            let input =
                EvalInput { shape: &shape, expressions: &ex, layers: &layers, pose: &pose, skinning: modes[case % 3] };
            e.evaluate(&input).unwrap();
            assert!(all_finite(&e), "case {case} of template {seed}");
        }
    }
}

/// Signed area normals of every triangle of `positions`.
fn triangle_normals_of(e: &Evaluator, positions: &[Vec3]) -> Vec<Vec3> {
    e.triangles()
        .as_chunks::<3>()
        .0
        .iter()
        .map(|t| {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| positions[i as usize]);
            (b - a).cross(c - a)
        })
        .collect()
}

/// `[AC-WP09-2]`: every expression alone at default strength (1.0), and all of them together, leave
/// every triangle facing the way it faced in the rest shape. Run on the synthetic templates; the
/// generated human template gets the same test once it exists (see STATUS).
#[test]
fn ac_wp09_2_expression_sweep_inverts_no_triangle() {
    for t in [face_patch(24, 4), cube()] {
        let t = Arc::new(t);
        let mut e = Evaluator::new(t.clone());
        let rest = triangle_normals_of(&e, t.mesh().positions());
        let count = t.expressions().expressions.len();
        let mut sweeps: Vec<Vec<f32>> = (0..count)
            .map(|k| {
                let mut w = vec![0.0; count];
                w[k] = 1.0;
                w
            })
            .collect();
        sweeps.push(vec![1.0; count]);
        for (k, ex) in sweeps.iter().enumerate() {
            e.evaluate(&EvalInput { expressions: ex, ..EvalInput::default() }).unwrap();
            let now = triangle_normals_of(&e, e.unposed_positions());
            for (tri, (a, b)) in rest.iter().zip(&now).enumerate() {
                assert!(a.dot(*b) > 0.0, "sweep {k}: triangle {tri} inverted");
            }
        }
    }
}

/// `[PERF-R1]`: after set-up, an evaluation allocates nothing.
#[test]
fn evaluation_is_allocation_free() {
    let t = Arc::new(face_patch(16, 6));
    let mut e = Evaluator::new(t.clone());
    e.add_layer(SculptLayer::new(vec![1, 2], vec![Vec3::Z; 2], t.mesh().vertex_count()).unwrap()).unwrap();
    let shape: Vec<f32> = (0..t.deltas().len()).map(|c| c as f32 * 0.1).collect();
    let pose = [Quat::IDENTITY, Quat::from_rotation_y(0.2), Quat::from_rotation_x(0.3)];
    let ex = [0.5, 0.2, 0.9, 0.4];
    let input =
        EvalInput { shape: &shape, expressions: &ex, layers: &[0.3], pose: &pose, skinning: SkinningMode::ByRegion };
    e.evaluate(&input).unwrap();
    let stats = humanart_core::alloc::count_in(|| {
        for k in 0..20 {
            let shape2: [f32; 3] = [k as f32 * 0.01, 0.0, 1.0];
            e.evaluate(&input).unwrap();
            e.update_rest(&shape2, &[]).unwrap();
        }
        e.refresh_rest().unwrap();
    });
    assert!(humanart_core::alloc::is_installed());
    assert_eq!(stats.allocations, 0, "{stats:?}");
}

/// Timing of a full evaluation on a 40k-vertex synthetic template (printed; recorded in STATUS).
#[test]
fn timing_full_evaluation() {
    let t = Arc::new(face_patch(200, 64));
    let mut e = Evaluator::new(t.clone());
    let v = t.mesh().vertex_count();
    let shape: Vec<f32> = (0..t.deltas().len()).map(|c| (c % 7) as f32 * 0.1).collect();
    let pose = [Quat::IDENTITY, Quat::from_rotation_y(0.2), Quat::from_rotation_x(0.3)];
    let ex = [0.5, 0.2, 0.9, 0.4];
    let input = EvalInput { shape: &shape, expressions: &ex, pose: &pose, ..EvalInput::default() };
    e.evaluate(&input).unwrap();
    let runs: u32 = 10;
    let start = std::time::Instant::now();
    for _ in 0..runs {
        e.evaluate(&input).unwrap();
    }
    let full = start.elapsed().as_secs_f64() * 1e3 / f64::from(runs);
    // One slider moving: the incremental rest update.
    let mut shape2 = shape.clone();
    let start = std::time::Instant::now();
    for k in 0..runs {
        shape2[k as usize] += 0.1;
        e.update_rest(&shape2, &[]).unwrap();
    }
    let one = start.elapsed().as_secs_f64() * 1e3 / f64::from(runs);
    println!(
        "[EVAL-01] timing: {v} vertices, {} components: evaluate {full:.3} ms, one-slider rest update {one:.4} ms",
        t.deltas().len()
    );
    assert!(full.is_finite());
}

#[test]
fn bases_without_components_evaluate() {
    let mut parts = crate::template::fixtures::bare_grid(3);
    parts.deltas = DeltaBases::empty();
    let t = Arc::new(Template::new(parts).unwrap());
    let mut e = Evaluator::new(t);
    e.evaluate(&EvalInput::default()).unwrap();
    assert!(all_finite(&e));
}

#[test]
fn branch_free_half_decode_is_exact_for_every_finite_half() {
    for bits in 0..=u16::MAX {
        let h = humanart_math::F16::from_bits(bits);
        if h.to_f32().is_finite() {
            assert_eq!(super::evaluator::decode_finite_f16(h).to_bits(), h.to_f32().to_bits(), "{bits:#x}");
        }
    }
}

#[test]
fn perf_accumulation_gives_the_same_bits_as_the_scaled_subnormal_form() {
    // [EVAL-02] the denormal-free accumulation adds exactly what the earlier form (half bits moved
    // into a possibly subnormal f32, times 2^112 * weight) added, for every finite half; the
    // contiguous-range path and the indexed path agree.
    let old = |bits: u16, w: f32| {
        let m = f32::from_bits(u32::from(bits & 0x7fff) << 13) * (f32::from_bits(0x7780_0000) * w);
        f32::from_bits(m.to_bits() ^ (u32::from(bits & 0x8000) << 16))
    };
    for w in [1.0_f32, 0.37, -1.5, 1e-3, 3.0, 0.0] {
        let finite: Vec<u16> =
            (0..=u16::MAX).filter(|&b| humanart_math::F16::from_bits(b).to_f32().is_finite()).collect();
        let mut b = DeltaBasesBuilder::new();
        let entries: Vec<(u32, Vec3)> = finite
            .chunks(3)
            .enumerate()
            .map(|(i, c)| {
                let f = |k: usize| humanart_math::F16::from_bits(c.get(k).copied().unwrap_or(0)).to_f32();
                (i as u32, Vec3::new(f(0), f(1), f(2)))
            })
            .collect();
        b.add("test.all_halves", DeltaKind::Semantic, 0, &entries).unwrap();
        // The same entries over every other vertex: the indexed path.
        let spread: Vec<(u32, Vec3)> = entries.iter().map(|&(i, v)| (2 * i, v)).collect();
        b.add("test.all_halves_spread", DeltaKind::Semantic, 0, &spread).unwrap();
        // Normal and zero halves only: the fused path's cheaper decode.
        let plain_bits: Vec<u16> =
            finite.iter().copied().filter(|b| b & 0x7c00 != 0 || b.trailing_zeros() >= 10).collect();
        let plain_entries: Vec<(u32, Vec3)> = plain_bits
            .chunks(3)
            .enumerate()
            .map(|(i, c)| {
                let f = |k: usize| humanart_math::F16::from_bits(c.get(k).copied().unwrap_or(0)).to_f32();
                (i as u32, Vec3::new(f(0), f(1), f(2)))
            })
            .collect();
        b.add("test.plain_halves", DeltaKind::Semantic, 0, &plain_entries).unwrap();
        let bases = b.finish().unwrap();
        for plain in [false, true] {
            let mut fused = vec![Vec3::ZERO; plain_entries.len()];
            let weight = |c: usize| if c == 2 { w } else { 0.0 };
            super::evaluator::accumulate_all(&bases, 3, weight, &[false, false, plain], &mut fused).unwrap();
            for (i, c) in plain_bits.chunks(3).enumerate() {
                for (k, &bits) in c.iter().enumerate() {
                    assert_eq!(fused[i].to_array()[k].to_bits(), (0.0 + old(bits, w)).to_bits(), "{bits:#x} * {w}");
                }
            }
        }
        let mut dense = vec![Vec3::ZERO; entries.len()];
        super::evaluator::accumulate(&bases, 0, w, &mut dense).unwrap();
        let mut sparse = vec![Vec3::ZERO; 2 * entries.len()];
        super::evaluator::accumulate(&bases, 1, w, &mut sparse).unwrap();
        for (i, c) in finite.chunks(3).enumerate() {
            let got = dense[i].to_array();
            for (k, &bits) in c.iter().enumerate() {
                assert_eq!(got[k].to_bits(), (0.0 + old(bits, w)).to_bits(), "{bits:#x} * {w}");
            }
            assert_eq!(got.map(f32::to_bits), sparse[2 * i].to_array().map(f32::to_bits));
        }
    }
}

/// The accumulation kernel against the plain f32 loop (printed; recorded in STATUS).
#[test]
fn timing_delta_accumulation() {
    let t = face_patch(200, 4);
    let mut p = t.mesh().positions().to_vec();
    let d = t.deltas();
    let start = std::time::Instant::now();
    for k in 0..100 {
        super::evaluator::accumulate(d, 0, 0.001 * k as f32, &mut p).unwrap();
    }
    let a = start.elapsed().as_secs_f64() * 1e9 / (100.0 * d.indices(0).len() as f64);
    let start = std::time::Instant::now();
    for k in 0..100 {
        d.accumulate(0, 0.001 * k as f32, &mut p).unwrap();
    }
    let b = start.elapsed().as_secs_f64() * 1e9 / (100.0 * d.indices(0).len() as f64);
    println!("[EVAL-02] timing: delta accumulation {a:.2} ns/entry (DeltaBases::accumulate {b:.2} ns/entry)");
    let mut q = t.mesh().positions().to_vec();
    let mut r = q.clone();
    super::evaluator::accumulate(d, 1, 0.7, &mut q).unwrap();
    d.accumulate(1, 0.7, &mut r).unwrap();
    assert_eq!(q, r);
    assert!(super::evaluator::accumulate(d, 99, 1.0, &mut q).is_err());
    assert!(super::evaluator::accumulate(d, 0, 1.0, &mut q[..3]).is_err());
}

/// D-003 `[LM-03]`: the rest rebuild after a landmark solve (every dense component changes) with
/// the kernel this module had before (one pass per component, the half moved into a possibly
/// subnormal `f32` and scaled by `2^112 * weight`) against [`super::evaluator::Evaluator::update_rest`],
/// alternating in one run so both see the same machine load. Run with
/// `cargo test -p humanart-model --release --lib d003_rest_rebuild -- --ignored --nocapture`.
#[test]
#[ignore = "timing: run explicitly in release"]
fn d003_rest_rebuild_before_and_after() {
    let t = Arc::new(face_patch(100, 100));
    let deltas = t.deltas();
    let old_accumulate = |c: usize, weight: f32, positions: &mut [Vec3]| {
        let scale = f32::from_bits(0x7780_0000) * weight;
        let term = |h: humanart_math::F16| {
            let bits = h.to_bits();
            let m = f32::from_bits(u32::from(bits & 0x7fff) << 13) * scale;
            f32::from_bits(m.to_bits() ^ (u32::from(bits & 0x8000) << 16))
        };
        for (&v, d) in deltas.indices(c).iter().zip(deltas.deltas(c)) {
            if let Some(p) = positions.get_mut(v as usize) {
                *p += Vec3::new(term(d[0]), term(d[1]), term(d[2]));
            }
        }
    };
    let mut e = Evaluator::new(t.clone());
    let mut old_rest = e.rest_positions().to_vec();
    let mut rng = Rng::new(3);
    let (mut before, mut after) = (Vec::new(), Vec::new());
    let mut shape = vec![0.0_f32; deltas.len()];
    for run in 0..41 {
        let previous = shape.clone();
        for w in &mut shape {
            *w = rng.range_f32(-0.5, 0.5);
        }
        let start = std::time::Instant::now();
        for (c, (&w, &old)) in shape.iter().zip(&previous).enumerate() {
            if w != old && super::evaluator::is_shape_kind(deltas.components()[c].kind) {
                old_accumulate(c, w - old, &mut old_rest);
            }
        }
        let old_ms = start.elapsed().as_secs_f64() * 1e3;
        let start = std::time::Instant::now();
        e.update_rest(&shape, &[]).unwrap();
        let new_ms = start.elapsed().as_secs_f64() * 1e3;
        if run > 0 {
            before.push(old_ms);
            after.push(new_ms);
        }
        // Same arithmetic in the same order: the same bits (until the periodic full refresh after
        // FULL_REFRESH_INTERVAL updates; two runs of 108 stay below it).
        if run < 2 {
            assert!(old_rest.iter().zip(e.rest_positions()).all(|(a, b)| a.to_array() == b.to_array()));
        }
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    println!(
        "[D-003] rest rebuild, {} vertices, {} delta components (the shape ones change): before {:.3} ms, after {:.3} ms (medians of 40)",
        t.mesh().positions().len(),
        deltas.len(),
        median(&mut before),
        median(&mut after)
    );
}
