//! GPU tests of the deformation on this machine's GPU (headless): parity with the CPU reference
//! `[AC-WP10-1]`, timings against section 14.2 `[AC-WP10-2]`, incremental drift `[AC-WP10-3]`,
//! motion vectors, sculpt layers, levels of detail and the allocation-free plan.

use std::sync::{Arc, OnceLock};

use humanart_core::rng::Rng;
use humanart_gpu::resources::{Budget, Resources};
use humanart_gpu::{DeviceRequest, Gpu};
use humanart_math::{Quat, Transform, Vec2, Vec3, Vec4};
use humanart_model::delta::{DeltaBasesBuilder, DeltaKind};
use humanart_model::eval::normals::{TriangleFrames, normal_corner_gather};
use humanart_model::eval::{EvalInput, Evaluator, SculptLayer, SkinningMode};
use humanart_model::expression::{
    CombinationCorrective, CombineOp, Expression, ExpressionKind, ExpressionSet, PoseCorrective,
};
use humanart_model::lod::LodLevel;
use humanart_model::region::{Region, RegionMap};
use humanart_model::skeleton::{Joint, JointKind, JointLimits, JointRegressor, Skeleton};
use humanart_model::skin::SkinWeights;
use humanart_model::{MeshBuilder, Side, Template, TemplateInfo, TemplateParts, UvTile};

use super::deformer::{DeformSettings, GpuDeformer, Stage};

// [PERF-R1]: this test binary counts allocations per thread.
#[global_allocator]
static ALLOCATOR: humanart_core::alloc::CountingAllocator = humanart_core::alloc::CountingAllocator;

/// Synthetic templates for the deformation tests.
pub(crate) mod fixtures {
    use super::*;

    /// What a synthetic surface template holds.
    #[derive(Clone, Copy, Debug)]
    pub struct Spec {
        /// Grid cells per side.
        pub n: usize,
        /// Global identity components.
        pub identity: usize,
        /// Semantic components (local bumps).
        pub semantic: usize,
        /// Expressions.
        pub expressions: usize,
        /// Combination correctives (of two expressions each).
        pub combinations: usize,
        /// Pose correctives.
        pub pose_correctives: usize,
        /// Joints in the chain.
        pub joints: usize,
        /// Skin influences per vertex.
        pub influences: usize,
        /// Add a reduced level of detail.
        pub lod: bool,
    }

    /// A smooth bump: 1 at `centre`, 0 from `radius` on.
    fn bump(p: Vec3, centre: Vec2, radius: f32) -> f32 {
        let t = (Vec2::new(p.x, p.y) - centre).length() / radius;
        if t >= 1.0 { 0.0 } else { (1.0 - t * t) * (1.0 - t * t) }
    }

    fn grid_position(x: f32, y: f32) -> Vec3 {
        Vec3::new(x, y, 0.15 * (std::f32::consts::PI * x).sin() * (0.5 * std::f32::consts::PI * y).cos())
    }

    /// A curved `n` x `n` sheet over the unit square with a UV seam (smooth duplicates) at column
    /// `n / 2` and a hard edge (split normals) at column `3n / 4`, four regions (face, face-L,
    /// face-R, face-L.cheek), and the components, rig and skin of `spec`.
    #[allow(clippy::too_many_lines)]
    pub fn surface(spec: &Spec) -> Template {
        let n = spec.n;
        let mut b = MeshBuilder::new();
        b.begin_part("sheet", false).unwrap();
        let id = |i: usize, j: usize| (j * (n + 1) + i) as u32;
        let (seam, hard) = (n / 2, 3 * n / 4);
        // Columns right of the seam use the second half of the UV square.
        for j in 0..=n {
            for i in 0..=n {
                let (x, y) = (i as f32 / n as f32, j as f32 / n as f32);
                let u = if i > seam { 0.5 + x * 0.5 } else { x * 0.5 };
                b.vertex(grid_position(x, y), Vec2::new(u, 1.0 - y));
            }
        }
        let seam_dups: Vec<u32> = (0..=n)
            .map(|j| {
                let (x, y) = (seam as f32 / n as f32, j as f32 / n as f32);
                b.duplicate(id(seam, j), Vec2::new(0.5 + x * 0.5, 1.0 - y), true).unwrap()
            })
            .collect();
        let hard_dups: Vec<u32> = (0..=n)
            .map(|j| {
                let (x, y) = (hard as f32 / n as f32, j as f32 / n as f32);
                b.duplicate(id(hard, j), Vec2::new(0.5 + x * 0.5, 1.0 - y), false).unwrap()
            })
            .collect();
        // Faces right of the seam use the seam duplicates, faces right of the hard edge its duplicates.
        let corner = |ii: usize, j: usize, column: usize| -> u32 {
            if ii == seam && column == seam {
                seam_dups[j]
            } else if ii == hard && column == hard {
                hard_dups[j]
            } else {
                id(ii, j)
            }
        };
        for j in 0..n {
            for i in 0..n {
                b.face(&[corner(i, j, i), corner(i + 1, j, i), corner(i + 1, j + 1, i), corner(i, j + 1, i)], 0)
                    .unwrap();
            }
        }
        let mesh = b.finish().unwrap();
        let pos = mesh.positions().to_vec();

        let region_of = |p: Vec3| -> u16 { if p.x < 0.5 { if p.y < 0.5 { 3 } else { 1 } } else { 2 } };
        let regions = RegionMap {
            regions: vec![
                Region { name: "face".into(), parent: None, side: Side::Center },
                Region { name: "face-L".into(), parent: Some(0), side: Side::Left },
                Region { name: "face-R".into(), parent: Some(0), side: Side::Right },
                Region { name: "face-L.cheek".into(), parent: Some(1), side: Side::Left },
            ],
            vertex_regions: pos.iter().map(|&p| region_of(p)).collect(),
            soft: None,
        };
        let field = |f: &dyn Fn(Vec3) -> Vec3| -> Vec<(u32, Vec3)> {
            pos.iter().enumerate().map(|(i, &p)| (i as u32, f(p))).filter(|(_, d)| *d != Vec3::ZERO).collect()
        };
        let mut rng = Rng::new(0x5eed_0010);
        let mut d = DeltaBasesBuilder::new();
        for k in 0..spec.identity {
            let (fx, fy) = ((k % 3 + 1) as f32, (k / 3 % 3 + 1) as f32);
            let phase = k as f32 * 0.7;
            let amp = 0.04 / (1.0 + k as f32 * 0.1);
            let dir = match k % 3 {
                0 => Vec3::Z,
                1 => Vec3::new(0.3, 0.0, 1.0),
                _ => Vec3::new(0.0, 0.4, 1.0),
            };
            let f = |p: Vec3| dir * (amp * (fx * 3.0 * p.x + phase).sin() * (fy * 2.0 * p.y + phase * 0.5).cos());
            d.add(&format!("id.{k}"), DeltaKind::IdentityGlobal, 0, &field(&f)).unwrap();
        }
        let local = |rng: &mut Rng, scale: f32| {
            let centre = Vec2::new(rng.range_f32(0.1, 0.9), rng.range_f32(0.1, 0.9));
            let radius = rng.range_f32(0.1, 0.25);
            let dir = Vec3::new(rng.range_f32(-1.0, 1.0), rng.range_f32(-1.0, 1.0), rng.range_f32(0.2, 1.0)) * scale;
            field(&|p: Vec3| dir * bump(p, centre, radius))
        };
        for k in 0..spec.semantic {
            let entries = local(&mut rng, 0.05);
            d.add(&format!("sem.{k}"), DeltaKind::Semantic, 0, &entries).unwrap();
        }
        let right: Vec<(u32, Vec3)> = field(&|p| Vec3::new(0.03, 0.0, 0.02) * bump(p, Vec2::new(0.75, 0.6), 0.2))
            .into_iter()
            .filter(|&(i, _)| region_of(pos[i as usize]) == 2)
            .collect();
        d.add("cheek-R.width", DeltaKind::IdentityRegional, 2, &right).unwrap();
        for k in 0..spec.expressions {
            let entries = local(&mut rng, 0.04);
            d.add(&format!("expr.{k}"), DeltaKind::Expression, 0, &entries).unwrap();
        }
        for k in 0..spec.combinations {
            let entries = local(&mut rng, 0.015);
            d.add(&format!("comb.{k}"), DeltaKind::CombinationCorrective, 0, &entries).unwrap();
        }
        for k in 0..spec.pose_correctives {
            let entries = local(&mut rng, 0.01);
            d.add(&format!("pose.{k}"), DeltaKind::PoseCorrective, 0, &entries).unwrap();
        }
        let deltas = d.finish().unwrap();
        let find = |name: &str| deltas.find(name).unwrap();

        // A chain of joints up the sheet; joint 1 has a rest rotation so bind rotations matter.
        let joints = spec.joints.max(2);
        let step = 1.1 / (joints - 1) as f32;
        let mut skeleton_joints = vec![Joint {
            name: "root".into(),
            parent: None,
            kind: JointKind::Deform,
            rest: Transform::from_rotation_translation(Quat::IDENTITY, Vec3::new(0.5, -0.1, -0.2)),
            limits: JointLimits::FREE,
        }];
        for j in 1..joints {
            let rotation = if j == 1 { Quat::from_rotation_z(0.3) } else { Quat::IDENTITY };
            let translation = if j == 2 {
                Quat::from_rotation_z(-0.3) * Vec3::new(0.0, step, 0.05)
            } else {
                Vec3::new(0.0, step, 0.05)
            };
            skeleton_joints.push(Joint {
                name: format!("j.{j}"),
                parent: Some(j as u16 - 1),
                kind: JointKind::Deform,
                rest: Transform::from_rotation_translation(rotation, translation),
                limits: JointLimits::FREE,
            });
        }
        let mut regressor = JointRegressor::empty(deltas.len());
        let mut entries: Vec<(usize, u16, Vec3)> = Vec::new();
        if spec.identity > 0 {
            entries.push((find("id.0") as usize, 1, Vec3::Y * 0.02));
        }
        if spec.semantic > 0 {
            entries.push((find("sem.0") as usize, (joints - 1) as u16, Vec3::Z * 0.01));
        }
        entries.sort_by_key(|e| (e.0, e.1));
        for c in 0..deltas.len() {
            for &(_, j, o) in entries.iter().filter(|e| e.0 == c) {
                regressor.joints.push(j);
                regressor.deltas.push(o);
            }
            regressor.offsets[c + 1] = regressor.joints.len() as u32;
        }
        let skeleton = Skeleton { joints: skeleton_joints, regressor };
        let rest = skeleton.rest_world();
        let lists: Vec<Vec<(u16, f32)>> = pos
            .iter()
            .map(|p| {
                rest.iter()
                    .enumerate()
                    .map(|(j, t)| (j as u16, (1.0 - (p.y - t.translation.y).abs() / (1.5 * step)).max(0.0)))
                    .collect()
            })
            .collect();
        let mut skin = SkinWeights::from_lists(spec.influences, lists.iter().map(Vec::as_slice), 0).unwrap();
        skin.dual_quaternion_regions = vec![2];

        let expressions = ExpressionSet {
            expressions: (0..spec.expressions)
                .map(|k| Expression {
                    name: format!("expr.{k}"),
                    kind: ExpressionKind::ActionUnit,
                    side: Side::Center,
                    facs: 1,
                    component: find(&format!("expr.{k}")),
                })
                .collect(),
            combinations: (0..spec.combinations)
                .map(|k| {
                    let a = (k * 7 + 1) % spec.expressions.max(1);
                    let b = (k * 3 + 2) % spec.expressions.max(1);
                    let b = if a == b { (b + 1) % spec.expressions.max(1) } else { b };
                    CombinationCorrective {
                        name: format!("comb.{k}"),
                        op: if k % 2 == 0 { CombineOp::Product } else { CombineOp::Min },
                        inputs: vec![a as u32, b as u32],
                        component: find(&format!("comb.{k}")),
                    }
                })
                .collect(),
            pose_correctives: (0..spec.pose_correctives)
                .map(|k| PoseCorrective {
                    name: format!("pose.{k}"),
                    joint: (1 + k % (joints - 1)) as u16,
                    target: Quat::from_rotation_x(0.4 + 0.1 * (k % 3) as f32),
                    radius: 0.6,
                    component: find(&format!("pose.{k}")),
                })
                .collect(),
        };
        let lods = if spec.lod { vec![lod_level(n, &id)] } else { Vec::new() };
        let parts = TemplateParts {
            info: TemplateInfo { name: "deform-sheet".into(), ..TemplateInfo::default() },
            mesh,
            tiles: vec![UvTile { name: "face".into(), resolution: 1024, target_texels_per_mm: 0.0 }],
            regions,
            landmarks: Vec::new(),
            skeleton,
            skin,
            deltas,
            expressions,
            lods,
        };
        Template::new(parts).unwrap()
    }

    /// A reduced level: the grid at half resolution, each vertex a blend of three LOD0 vertices.
    fn lod_level(n: usize, id: &dyn Fn(usize, usize) -> u32) -> LodLevel {
        let m = n / 2;
        let mut b = MeshBuilder::new();
        b.begin_part("sheet", false).unwrap();
        let mut sources = Vec::new();
        let mut weights = Vec::new();
        for j in 0..=m {
            for i in 0..=m {
                let (fi, fj) = (2 * i, 2 * j);
                let ni = if fi < n { fi + 1 } else { fi - 1 };
                let nj = if fj < n { fj + 1 } else { fj - 1 };
                sources.push([id(fi, fj), id(ni, fj), id(fi, nj)]);
                weights.push([0.6, 0.2, 0.2]);
                let p = grid_position(fi as f32 / n as f32, fj as f32 / n as f32);
                b.vertex(p, Vec2::new(p.x * 0.5, 1.0 - p.y));
            }
        }
        let lid = |i: usize, j: usize| (j * (m + 1) + i) as u32;
        for j in 0..m {
            for i in 0..m {
                b.face(&[lid(i, j), lid(i + 1, j), lid(i + 1, j + 1), lid(i, j + 1)], 0).unwrap();
            }
        }
        LodLevel {
            mesh: b.finish().unwrap(),
            sources,
            source_weights: weights,
            declared_error: 0.01,
            switch_size: 0.5,
            hysteresis: 0.1,
        }
    }

    /// A small sheet with every feature: `identity` identity components, 3 semantic, 6
    /// expressions, 2 combinations, 2 pose correctives, 4 joints, 3 influences, one LOD.
    pub fn sheet(n: usize, identity: usize) -> Template {
        surface(&Spec {
            n,
            identity,
            semantic: 3,
            expressions: 6,
            combinations: 2,
            pose_correctives: 2,
            joints: 4,
            influences: 3,
            lod: true,
        })
    }

    /// A hero-scale surface for timings: 201 x 201 grid plus seam and hard-edge duplicates
    /// (40 803 vertices, about LOD0 of the generated body), 100 global identity components, 30
    /// semantic, 150 expressions, 30 combinations, 12 pose correctives, 24 joints, 4 influences.
    pub fn hero() -> Template {
        surface(&Spec {
            n: 200,
            identity: 100,
            semantic: 30,
            expressions: 150,
            combinations: 30,
            pose_correctives: 12,
            joints: 24,
            influences: 4,
            lod: false,
        })
    }
}

fn gpu() -> &'static Gpu {
    static GPU: OnceLock<Gpu> = OnceLock::new();
    GPU.get_or_init(|| Gpu::headless_blocking(&DeviceRequest::default()).expect("this machine has a GPU"))
}

fn resources() -> Resources {
    Resources::new(gpu(), Budget::UNLIMITED)
}

fn submit(deformer: &mut GpuDeformer) -> bool {
    let gpu = gpu();
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("deform test") });
    let recorded = deformer.encode(&mut encoder);
    gpu.queue.submit([encoder.finish()]);
    recorded
}

/// Reads a whole buffer as f32 words (blocking: tests only).
fn read_f32(buffer: &wgpu::Buffer) -> Vec<f32> {
    let gpu = gpu();
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("deform readback"),
        size: buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    gpu.queue.submit([encoder.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("the readback maps"));
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("the device answers");
    let words = {
        let bytes = staging.slice(..).get_mapped_range().expect("mapped").to_vec();
        bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
    };
    staging.unmap();
    words
}

fn vec3s(words: &[f32], n: usize) -> Vec<Vec3> {
    words.as_chunks::<3>().0.iter().take(n).map(|c| Vec3::from_array(*c)).collect()
}

/// Everything the GPU produced.
struct GpuState {
    rest: Vec<Vec3>,
    positions: Vec<Vec3>,
    prev: Vec<Vec3>,
    normals: Vec<Vec3>,
    tangents: Vec<Vec4>,
    edge_strain: Vec<f32>,
    region_strain: Vec<[f32; 2]>,
    vertex_strain: Vec<f32>,
}

fn read_state(d: &GpuDeformer) -> GpuState {
    let o = d.outputs();
    let v = d.vertex_count() as usize;
    GpuState {
        rest: vec3s(&read_f32(o.rest), v),
        positions: vec3s(&read_f32(o.positions), v),
        prev: vec3s(&read_f32(o.prev_positions), v),
        normals: vec3s(&read_f32(o.normals), v),
        tangents: read_f32(o.tangents).as_chunks::<4>().0.iter().take(v).map(|c| Vec4::from_array(*c)).collect(),
        edge_strain: read_f32(o.edge_strain)
            .as_chunks::<2>()
            .0
            .iter()
            .take(d.edge_count() as usize)
            .map(|c| c[0])
            .collect(),
        region_strain: read_f32(o.region_strain)
            .as_chunks::<2>()
            .0
            .iter()
            .take(d.region_count() as usize)
            .copied()
            .collect(),
        vertex_strain: read_f32(o.vertex_strain).into_iter().take(v).collect(),
    }
}

/// `[EVAL-01]`'s position tolerance: 1e-4 of the head height. A synthetic template has no head;
/// the smaller of its height and 0.25 m (a head) stands in, which is the stricter choice here.
fn position_tolerance(t: &Template) -> f32 {
    let (lo, hi) = t.mesh().positions().iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p.y), hi.max(p.y)));
    1e-4 * (hi - lo).min(0.25)
}

/// The largest deviations between the GPU and the CPU reference.
#[derive(Debug, Default, Clone, Copy)]
struct Deviation {
    rest: f32,
    position: f32,
    normal_degrees: f32,
    tangent_degrees: f32,
    handedness: usize,
    edge_strain: f32,
    region_strain: f32,
    vertex_strain: f32,
}

impl Deviation {
    fn max(self, o: Self) -> Self {
        Self {
            rest: self.rest.max(o.rest),
            position: self.position.max(o.position),
            normal_degrees: self.normal_degrees.max(o.normal_degrees),
            tangent_degrees: self.tangent_degrees.max(o.tangent_degrees),
            handedness: self.handedness + o.handedness,
            edge_strain: self.edge_strain.max(o.edge_strain),
            region_strain: self.region_strain.max(o.region_strain),
            vertex_strain: self.vertex_strain.max(o.vertex_strain),
        }
    }
}

fn degrees(a: Vec3, b: Vec3) -> f32 {
    a.angle_between(b).to_degrees()
}

fn compare(cpu: &Evaluator, g: &GpuState) -> Deviation {
    let max = |it: &mut dyn Iterator<Item = f32>| it.fold(0.0_f32, f32::max);
    let s = cpu.strain();
    Deviation {
        rest: max(&mut cpu.rest_positions().iter().zip(&g.rest).map(|(a, b)| a.distance(*b))),
        position: max(&mut cpu.positions().iter().zip(&g.positions).map(|(a, b)| a.distance(*b))),
        normal_degrees: max(&mut cpu.normals().iter().zip(&g.normals).map(|(a, b)| degrees(*a, *b))),
        tangent_degrees: max(&mut cpu
            .tangents()
            .iter()
            .zip(&g.tangents)
            .map(|(a, b)| degrees(a.truncate(), b.truncate()))),
        handedness: cpu.tangents().iter().zip(&g.tangents).filter(|(a, b)| a.w != b.w).count(),
        edge_strain: max(&mut s.edge_strain().iter().zip(&g.edge_strain).map(|(a, b)| (a - b).abs())),
        region_strain: max(&mut s
            .region_stretch()
            .iter()
            .zip(s.region_compression())
            .zip(&g.region_strain)
            .map(|((st, co), gs)| (st - gs[0]).abs().max((co - gs[1]).abs()))),
        vertex_strain: max(&mut s.vertex_strain().iter().zip(&g.vertex_strain).map(|(a, b)| (a - b).abs())),
    }
}

/// Strain is a ratio of lengths; an absolute deviation of 1e-3 (0.1 % stretch) is far below what
/// the wrinkle drivers resolve. The spec gives no strain tolerance; this is the documented one.
const STRAIN_TOLERANCE: f32 = 1e-3;
/// `[EVAL-01]`: normals within 0.5 degrees; tangents are held to the same bound.
const ANGLE_TOLERANCE: f32 = 0.5;

fn assert_within(dev: &Deviation, tolerance: f32, what: &str) {
    assert!(dev.rest <= tolerance, "{what}: rest {dev:?} > {tolerance}");
    assert!(dev.position <= tolerance, "{what}: positions {dev:?} > {tolerance}");
    assert!(dev.normal_degrees <= ANGLE_TOLERANCE, "{what}: normals {dev:?}");
    assert!(dev.tangent_degrees <= ANGLE_TOLERANCE, "{what}: tangents {dev:?}");
    assert_eq!(dev.handedness, 0, "{what}: handedness {dev:?}");
    assert!(dev.edge_strain <= STRAIN_TOLERANCE, "{what}: edge strain {dev:?}");
    assert!(dev.region_strain <= STRAIN_TOLERANCE, "{what}: region strain {dev:?}");
    assert!(dev.vertex_strain <= STRAIN_TOLERANCE, "{what}: vertex strain {dev:?}");
}

/// A random rotation of at most `max_angle` radians.
fn random_rotation(rng: &mut Rng, max_angle: f32) -> Quat {
    let axis = Vec3::new(rng.normal_f32(), rng.normal_f32(), rng.normal_f32()).normalize_or(Vec3::Y);
    Quat::from_axis_angle(axis, rng.range_f32(-max_angle, max_angle))
}

/// Random inputs for `t`: shape weights in [-2, 2] (shape components only matter), expression
/// weights in [0, 1], a random pose.
struct RandomInput {
    shape: Vec<f32>,
    expressions: Vec<f32>,
    pose: Vec<Quat>,
}

impl RandomInput {
    fn new(rng: &mut Rng, t: &Template) -> Self {
        Self {
            shape: (0..t.deltas().len()).map(|_| rng.range_f32(-2.0, 2.0)).collect(),
            expressions: (0..t.expressions().expressions.len()).map(|_| rng.range_f32(0.0, 1.0)).collect(),
            pose: (0..t.skeleton().joints.len()).map(|_| random_rotation(rng, 0.8)).collect(),
        }
    }

    fn input(&self, skinning: SkinningMode) -> EvalInput<'_> {
        EvalInput { shape: &self.shape, expressions: &self.expressions, layers: &[], pose: &self.pose, skinning }
    }
}

// [AC-WP10-1] [EVAL-01]: the GPU matches the CPU reference across random parameter sets and poses,
// in every skinning mode, through UV seams and hard edges.
#[test]
fn ac_wp10_1_parity_across_random_parameters_and_poses() {
    let template = Arc::new(fixtures::sheet(24, 6));
    let tolerance = position_tolerance(&template);
    let mut res = resources();
    let mut gpu_d = GpuDeformer::new(gpu(), &mut res, template.clone(), DeformSettings::default()).unwrap();
    let mut cpu = Evaluator::new(template.clone());
    // The neutral state first.
    assert!(submit(&mut gpu_d));
    assert_within(&compare(&cpu, &read_state(&gpu_d)), tolerance, "neutral");
    let mut rng = Rng::new(0xac10_0001);
    let mut worst = Deviation::default();
    let modes = [SkinningMode::ByRegion, SkinningMode::Linear, SkinningMode::DualQuaternion];
    for round in 0..24 {
        let r = RandomInput::new(&mut rng, &template);
        let input = r.input(modes[round % 3]);
        cpu.evaluate(&input).unwrap();
        gpu_d.update(&mut res, &input).unwrap();
        assert!(submit(&mut gpu_d));
        let dev = compare(&cpu, &read_state(&gpu_d));
        assert_within(&dev, tolerance, &format!("round {round}"));
        worst = worst.max(dev);
    }
    eprintln!("[AC-WP10-1] worst deviation over 24 random sets (tolerance {tolerance:e} m): {worst:?}");
    assert_eq!(gpu_d.late_pipelines(), 0);
}

// [EVAL-04]: seam duplicates share their normal, hard-edge duplicates do not.
#[test]
fn eval_04_split_normals_on_hard_edges_only() {
    let template = Arc::new(fixtures::sheet(16, 2));
    let mut res = resources();
    let mut d = GpuDeformer::new(gpu(), &mut res, template.clone(), DeformSettings::default()).unwrap();
    submit(&mut d);
    let g = read_state(&d);
    let mesh = template.mesh();
    let (mut seams, mut hard) = (0, 0);
    for v in 0..mesh.vertex_count() {
        let k = mesh.position_keys()[v] as usize;
        if k == v {
            continue;
        }
        let angle = degrees(g.normals[v], g.normals[k]);
        if mesh.normal_keys()[v] as usize == k {
            assert!(angle < 1e-3, "seam duplicate {v} differs by {angle}");
            seams += 1;
        } else {
            assert!(angle > 1.0, "hard-edge duplicate {v} shares the normal");
            hard += 1;
        }
    }
    assert!(seams > 0 && hard > 0);
}

/// Runs `moves` random slider moves (one shape component to a new weight each), submitting every
/// 50 moves, and returns the deformer and the final weights.
fn slider_moves(
    template: &Arc<Template>,
    settings: DeformSettings,
    moves: usize,
    seed: u64,
) -> (GpuDeformer, Resources, Vec<f32>) {
    let mut res = resources();
    let mut d = GpuDeformer::new(gpu(), &mut res, template.clone(), settings).unwrap();
    let mut weights = vec![0.0_f32; template.deltas().len()];
    let shape: Vec<usize> = (0..weights.len())
        .filter(|&c| humanart_model::eval::evaluator::is_shape_kind(template.deltas().components()[c].kind))
        .collect();
    let mut rng = Rng::new(seed);
    for m in 0..moves {
        let c = shape[rng.index(shape.len())];
        weights[c] = rng.range_f32(-3.0, 3.0);
        let plan = d.update(&mut res, &EvalInput { shape: &weights, ..EvalInput::default() }).unwrap();
        assert!(plan.deform);
        if m % 50 == 49 {
            submit(&mut d);
        } else {
            // Several updates per submission: the jobs accumulate until the encode.
            let mut encoder = gpu().device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            d.encode(&mut encoder);
            gpu().queue.submit([encoder.finish()]);
        }
    }
    submit(&mut d);
    (d, res, weights)
}

// [AC-WP10-3] [EVAL-03]: 10 000 random slider moves applied incrementally equal a full
// re-evaluation, with the periodic refresh and (measured) without it.
#[test]
fn ac_wp10_3_incremental_equals_full_after_10000_slider_moves() {
    let template = Arc::new(fixtures::sheet(32, 9));
    let tolerance = position_tolerance(&template);
    let mut cpu = Evaluator::new(template.clone());
    for (label, interval) in
        [("periodic refresh", humanart_model::eval::evaluator::FULL_REFRESH_INTERVAL), ("no refresh", u32::MAX)]
    {
        let settings = DeformSettings { refresh_interval: interval, ..DeformSettings::default() };
        let (d, _res, weights) = slider_moves(&template, settings, 10_000, 0xd71f_7000);
        let incremental = read_state(&d);
        // The full re-evaluation: a fresh deformer that gathers the final weights in one pass.
        let mut res = resources();
        let mut full = GpuDeformer::new(gpu(), &mut res, template.clone(), settings).unwrap();
        full.update(&mut res, &EvalInput { shape: &weights, ..EvalInput::default() }).unwrap();
        submit(&mut full);
        let reference = read_state(&full);
        let drift = incremental.rest.iter().zip(&reference.rest).map(|(a, b)| a.distance(*b)).fold(0.0_f32, f32::max);
        let drift_positions =
            incremental.positions.iter().zip(&reference.positions).map(|(a, b)| a.distance(*b)).fold(0.0_f32, f32::max);
        cpu.evaluate(&EvalInput { shape: &weights, ..EvalInput::default() }).unwrap();
        cpu.refresh_rest().unwrap();
        let vs_cpu =
            cpu.rest_positions().iter().zip(&incremental.rest).map(|(a, b)| a.distance(*b)).fold(0.0_f32, f32::max);
        eprintln!(
            "[AC-WP10-3] {label}: max rest drift after 10 000 moves {drift:e} m (positions {drift_positions:e}), vs CPU full {vs_cpu:e}, tolerance {tolerance:e}"
        );
        assert!(drift <= tolerance && drift_positions <= tolerance && vs_cpu <= tolerance, "{label}");
    }
}

// [RND-04]: the previous positions hold the last frame's positions; a frame without change makes
// them equal (zero motion).
#[test]
fn motion_vectors_keep_the_previous_positions() {
    let template = Arc::new(fixtures::sheet(8, 2));
    let mut res = resources();
    let mut d = GpuDeformer::new(gpu(), &mut res, template.clone(), DeformSettings::default()).unwrap();
    submit(&mut d);
    let first = read_state(&d);
    let expressions = vec![0.8; template.expressions().expressions.len()];
    d.update(&mut res, &EvalInput { expressions: &expressions, ..EvalInput::default() }).unwrap();
    assert!(submit(&mut d));
    let second = read_state(&d);
    assert_eq!(second.prev, first.positions);
    assert_ne!(second.positions, first.positions);
    // Nothing changed: one copy so motion becomes zero, then nothing at all.
    d.update(&mut res, &EvalInput { expressions: &expressions, ..EvalInput::default() }).unwrap();
    assert!(submit(&mut d));
    let third = read_state(&d);
    assert_eq!(third.prev, third.positions);
    assert!(!submit(&mut d));
}

// Sculpt layers: added, weighted, re-weighted incrementally and replaced, against the CPU.
#[test]
fn sculpt_layers_match_the_cpu() {
    let template = Arc::new(fixtures::sheet(12, 3));
    let tolerance = position_tolerance(&template);
    let mut res = resources();
    let mut d = GpuDeformer::new(gpu(), &mut res, template.clone(), DeformSettings::default()).unwrap();
    let mut cpu = Evaluator::new(template.clone());
    let v = template.mesh().vertex_count();
    let pairs: Vec<(u32, Vec3)> =
        (0..v as u32).step_by(3).map(|i| (i, Vec3::new(0.0, 0.01, 0.02) * (i % 7) as f32 * 0.1)).collect();
    let layer = SculptLayer::from_pairs(&pairs, v).unwrap();
    assert_eq!(d.add_layer(gpu(), &mut res, &layer).unwrap(), 0);
    assert_eq!(cpu.add_layer(layer).unwrap(), 0);
    let shape = vec![0.5; template.deltas().len()];
    for w in [1.0, -0.5, 2.0] {
        let input = EvalInput { shape: &shape, layers: &[w], ..EvalInput::default() };
        cpu.evaluate(&input).unwrap();
        let plan = d.update(&mut res, &input).unwrap();
        submit(&mut d);
        assert_within(&compare(&cpu, &read_state(&d)), tolerance, &format!("layer weight {w} ({plan:?})"));
    }
    let pairs: Vec<(u32, Vec3)> = (1..v as u32).step_by(5).map(|i| (i, Vec3::new(0.01, 0.0, -0.01))).collect();
    let replacement = SculptLayer::from_pairs(&pairs, v).unwrap();
    d.replace_layer(gpu(), &mut res, 0, &replacement).unwrap();
    cpu.replace_layer(0, replacement).unwrap();
    let input = EvalInput { shape: &shape, layers: &[2.0], ..EvalInput::default() };
    cpu.evaluate(&input).unwrap();
    assert!(d.update(&mut res, &input).unwrap().full);
    submit(&mut d);
    assert_within(&compare(&cpu, &read_state(&d)), tolerance, "replaced layer");
}

// [MOD-05]: a reduced level follows LOD0 through its barycentric sources, with its own normals.
#[test]
fn lod_levels_follow_lod0() {
    let template = Arc::new(fixtures::sheet(16, 3));
    let mut res = resources();
    let mut d = GpuDeformer::new(gpu(), &mut res, template.clone(), DeformSettings::default()).unwrap();
    assert_eq!(d.lod_count(), 1);
    let mut cpu = Evaluator::new(template.clone());
    let mut rng = Rng::new(0x10d);
    let r = RandomInput::new(&mut rng, &template);
    let input = r.input(SkinningMode::ByRegion);
    cpu.evaluate(&input).unwrap();
    d.update(&mut res, &input).unwrap();
    submit(&mut d);
    let level = &template.lods()[0];
    let out = d.lod_outputs(1).unwrap();
    let n = out.vertex_count as usize;
    let positions = vec3s(&read_f32(out.positions), n);
    let normals = vec3s(&read_f32(out.normals), n);
    let expected = level.transfer_positions(cpu.positions()).unwrap();
    let tolerance = position_tolerance(&template);
    for (a, b) in positions.iter().zip(&expected) {
        assert!(a.distance(*b) <= tolerance, "{a:?} {b:?}");
    }
    let tri = level.mesh.triangulate();
    let mut frames = TriangleFrames::new(tri.len());
    frames.compute(&expected, level.mesh.uvs(), &tri.indices);
    let mut cpu_normals = vec![Vec3::ZERO; n];
    frames.gather_normals(&normal_corner_gather(&level.mesh, &tri.indices), &mut cpu_normals);
    for (a, b) in normals.iter().zip(&cpu_normals) {
        assert!(degrees(*a, *b) <= ANGLE_TOLERANCE);
    }
    assert!(d.lod_outputs(0).is_none() && d.lod_outputs(2).is_none());
}

// Refused inputs change nothing and record nothing.
#[test]
fn invalid_input_is_refused_without_change() {
    let template = Arc::new(fixtures::sheet(6, 2));
    let mut res = resources();
    let mut d = GpuDeformer::new(gpu(), &mut res, template.clone(), DeformSettings::default()).unwrap();
    assert!(submit(&mut d));
    // The frame after a change only keeps the previous positions; then nothing is recorded.
    assert!(submit(&mut d));
    assert!(!submit(&mut d));
    let too_long = vec![0.0; template.deltas().len() + 1];
    assert!(d.update(&mut res, &EvalInput { shape: &too_long, ..EvalInput::default() }).is_err());
    let nan = [f32::NAN];
    assert!(d.update(&mut res, &EvalInput { expressions: &nan, ..EvalInput::default() }).is_err());
    let huge = [2.0e4];
    assert!(d.update(&mut res, &EvalInput { expressions: &huge, ..EvalInput::default() }).is_err());
    let zero = [Quat::from_xyzw(0.0, 0.0, 0.0, 0.0)];
    assert!(d.update(&mut res, &EvalInput { pose: &zero, ..EvalInput::default() }).is_err());
    assert!(!submit(&mut d), "a refused input scheduled work");
    assert!(
        GpuDeformer::new(
            gpu(),
            &mut res,
            template,
            DeformSettings { refresh_interval: 0, ..DeformSettings::default() }
        )
        .is_err()
    );
}

// [PERF-R1]: the CPU half of an update allocates nothing in steady state.
#[test]
fn perf_r1_update_plan_allocates_nothing() {
    let template = Arc::new(fixtures::sheet(16, 4));
    let mut res = resources();
    let mut d = GpuDeformer::new(gpu(), &mut res, template.clone(), DeformSettings::default()).unwrap();
    let mut rng = Rng::new(7);
    let inputs: Vec<RandomInput> = (0..8).map(|_| RandomInput::new(&mut rng, &template)).collect();
    // Warm up (first full pass), then measure.
    d.update(&mut res, &inputs[0].input(SkinningMode::ByRegion)).unwrap();
    submit(&mut d);
    let stats = humanart_core::alloc::count_in(|| {
        for r in &inputs {
            d.plan(&r.input(SkinningMode::ByRegion)).unwrap();
        }
    });
    assert_eq!(stats.events(), 0, "{stats:?}");
    // The whole update + encode, including wgpu's own queue writes and command recording, for the
    // record: wgpu allocates inside `write_buffer` and command recording, outside this crate.
    let mut encoder = gpu().device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let stats = humanart_core::alloc::count_in(|| {
        for r in &inputs {
            d.update(&mut res, &r.input(SkinningMode::Linear)).unwrap();
            d.encode(&mut encoder);
        }
    });
    gpu().queue.submit([encoder.finish()]);
    eprintln!("[PERF-R1] 8 updates + encodes, allocator events including wgpu: {}", stats.events());
}

fn slider_input<'a>(shape: &'a [f32], expressions: &'a [f32], pose: &'a [Quat]) -> EvalInput<'a> {
    EvalInput { shape, expressions, layers: &[], pose, skinning: SkinningMode::ByRegion }
}

/// Median of `samples` (sorted in place).
fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples.get(samples.len() / 2).copied().unwrap_or(0.0)
}

// [AC-WP10-2]: deformation timings on this GPU against section 14.2, by timestamp queries per stage.
// The numbers are printed (and recorded in the agent status); the test asserts only that they were
// measured, since a miss is reported as a number, not hidden behind a skipped test.
#[test]
fn ac_wp10_2_deformation_timings() {
    let gpu = gpu();
    assert!(
        gpu.features().contains(wgpu::Features::TIMESTAMP_QUERY),
        "[AC-WP10-2] this adapter has no timestamp queries: {}",
        gpu.report()
    );
    let template = Arc::new(fixtures::hero());
    let mut res = resources();
    let mut d = GpuDeformer::new(gpu, &mut res, template.clone(), DeformSettings::default()).unwrap();
    let stages = Stage::ALL.len() as u32;
    let queries = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("deform timings"),
        ty: wgpu::QueryType::Timestamp,
        count: 2 * stages,
    });
    let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("deform timings"),
        size: u64::from(2 * stages) * 8,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let period = f64::from(gpu.queue.get_timestamp_period());
    // Milliseconds per stage (`total` false) or of the single pass (`total` true, in slot 0).
    let timed = |d: &mut GpuDeformer, total: bool| -> [f64; 5] {
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let recorded = if total {
            d.encode_timed_total(&mut encoder, &queries, 0)
        } else {
            d.encode_timed(&mut encoder, &queries, 0)
        };
        assert!(recorded);
        let used = if total { 2 } else { 2 * stages };
        encoder.resolve_query_set(&queries, 0..used, &resolve, 0);
        gpu.queue.submit([encoder.finish()]);
        let words = read_f32(&resolve);
        let ticks: Vec<u64> = words
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u64::from(c[0].to_bits()) | (u64::from(c[1].to_bits()) << 32))
            .collect();
        let mut ms = [0.0; 5];
        for (k, slot) in ms.iter_mut().enumerate().take(used as usize / 2) {
            *slot = ticks[2 * k + 1].saturating_sub(ticks[2 * k]) as f64 * period * 1e-6;
        }
        ms
    };
    let shape_components: Vec<usize> = (0..template.deltas().len())
        .filter(|&c| humanart_model::eval::evaluator::is_shape_kind(template.deltas().components()[c].kind))
        .collect();
    let mut weights = vec![0.0_f32; template.deltas().len()];
    let mut rng = Rng::new(0x14_02);
    let expressions: Vec<f32> =
        (0..template.expressions().expressions.len()).map(|_| rng.range_f32(0.0, 0.6)).collect();
    let pose: Vec<Quat> = (0..template.skeleton().joints.len()).map(|_| random_rotation(&mut rng, 0.3)).collect();
    // Every shape component non-zero, so a full pass gathers them all.
    for &c in &shape_components {
        weights[c] = rng.range_f32(-1.0, 1.0);
    }
    let (mut full_total, mut one_total, mut cpu_us) = (vec![], vec![], vec![]);
    let (mut full_stages, mut one_stages) = ([0.0_f64; 5], [0.0_f64; 5]);
    let rounds: u32 = 40;
    for round in 0..rounds {
        // Full re-evaluation: more changed components than job slots.
        for total in [true, false] {
            for &c in shape_components.iter().take(70) {
                weights[c] += 0.01;
            }
            assert!(d.update(&mut res, &slider_input(&weights, &expressions, &pose)).unwrap().full);
            let ms = timed(&mut d, total);
            if total {
                full_total.push(ms[0]);
            } else {
                full_stages.iter_mut().zip(ms).for_each(|(s, m)| *s += m);
            }
        }
        // One slider: one shape component changes. CPU time of update + encode first.
        let c = shape_components[round as usize % shape_components.len()];
        weights[c] += 0.05;
        let start = std::time::Instant::now();
        let plan = d.update(&mut res, &slider_input(&weights, &expressions, &pose)).unwrap();
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        d.encode(&mut encoder);
        cpu_us.push(start.elapsed().as_secs_f64() * 1e6);
        gpu.queue.submit([encoder.finish()]);
        assert_eq!((plan.jobs, plan.full), (1, false));
        for total in [true, false] {
            weights[c] += 0.05;
            assert_eq!(d.update(&mut res, &slider_input(&weights, &expressions, &pose)).unwrap().jobs, 1);
            let ms = timed(&mut d, total);
            if total {
                one_total.push(ms[0]);
            } else {
                one_stages.iter_mut().zip(ms).for_each(|(s, m)| *s += m);
            }
        }
    }
    let n = f64::from(rounds);
    let stages_text = |s: &[f64; 5]| {
        format!("accumulate {:.4}, skin {:.4}, normals {:.4}, strain {:.4}", s[0] / n, s[1] / n, s[2] / n, s[3] / n)
    };
    eprintln!("[AC-WP10-2] adapter: {} ({:?})", gpu.adapter_info().name, gpu.adapter_info().backend);
    eprintln!(
        "[AC-WP10-2] hero surface: {} vertices, {} triangles, {} edges, {} components ({} shape), {} expressions, {} joints",
        d.vertex_count(),
        d.triangle_count(),
        d.edge_count(),
        template.deltas().len(),
        shape_components.len(),
        template.expressions().expressions.len(),
        template.skeleton().joints.len()
    );
    eprintln!(
        "[AC-WP10-2] full re-evaluation, one pass: median {:.4} ms GPU (target 0.6 ms); per stage (mean, ms): {}",
        median(&mut full_total),
        stages_text(&full_stages)
    );
    eprintln!(
        "[AC-WP10-2] one slider, one pass: median {:.4} ms GPU (target 0.15 ms); per stage (mean, ms): {}",
        median(&mut one_total),
        stages_text(&one_stages)
    );
    eprintln!("[AC-WP10-2] CPU update + encode for one slider: median {:.1} us (target 100 us)", median(&mut cpu_us));
    assert!(full_total.len() == rounds as usize && one_total.len() == rounds as usize);
}
