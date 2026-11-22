//! Crate-level GPU tests of the renderer core on this machine's GPU (headless): the GPU deformer
//! wired into the passes `[EVAL-05]` with correct motion vectors `[RND-04]`, and blocking GPU waits
//! that end with an error instead of hanging `[ERR-05]`.

use std::sync::Arc;

use humanart_gpu::Gpu;
use humanart_gpu::resources::{Budget, Resources};
use humanart_math::{Vec2, Vec3, f16_bits_to_f32, ndc_to_screen};
use humanart_model::eval::EvalInput;

use crate::camera::{Camera, Exposure, ViewMatrices};
use crate::capture;
use crate::deform::tests::fixtures;
use crate::deform::{DeformSettings, GpuDeformer};
use crate::error::RenderError;
use crate::light::{Background, LightRig, ProceduralEnvironment};
use crate::renderer::{FrameInput, RenderConfig, Renderer};
use crate::scene::{GpuScene, Material, Mesh, Scene};

/// Held by the unit tests that time GPU work against a bound (`skin::tests::ac_wp13_3_*`) and by
/// the heaviest GPU tests beside them (`deform::tests::ac_wp10_2_*`), so two of them never share the
/// device: with two test threads a ratio bound of 2.0x read 2.05x and 2.39x (todo section 6).
static GPU_TIMING: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The lock of [`GPU_TIMING`] (a test that panicked while holding it leaves it usable).
pub(crate) fn gpu_timing_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_TIMING.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn gpu() -> &'static Gpu {
    // The process's shared device (ADR-040).
    humanart_gpu::shared_headless().expect("this machine has a GPU")
}

/// Reads a whole buffer as f32 words (blocking: tests only).
fn read_f32(buffer: &wgpu::Buffer) -> Vec<f32> {
    let gpu = gpu();
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test readback"),
        size: buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    gpu.queue.submit([encoder.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("the readback maps"));
    gpu.device
        .poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(30)) })
        .expect("the device answers within 30 s");
    let words = {
        let bytes = staging.slice(..).get_mapped_range().expect("mapped").to_vec();
        bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
    };
    staging.unmap();
    words
}

fn vec3s(words: &[f32]) -> Vec<Vec3> {
    words.as_chunks::<3>().0.iter().map(|c| Vec3::new(c[0], c[1], c[2])).collect()
}

fn rig() -> LightRig {
    LightRig {
        name: "test",
        lights: Vec::new(),
        environment: ProceduralEnvironment::Uniform,
        environment_nits: 1.0,
        environment_rotation: 0.0,
        background: Background::Gray(0.18),
        ev100: 0.0,
    }
}

/// The motion vectors (pixels) of the surface target.
fn motion(resources: &mut Resources, renderer: &Renderer) -> Vec<Vec2> {
    let surface = renderer.target_texture("surface").expect("a surface target");
    let (_, _, bytes) = capture::read_texture(resources, surface, 0).expect("readback");
    bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|t| {
            let w = u32::from_le_bytes([t[12], t[13], t[14], t[15]]);
            Vec2::new(f16_bits_to_f32(w as u16), f16_bits_to_f32((w >> 16) as u16))
        })
        .collect()
}

/// `[EVAL-05]` `[RND-04]`: a template deformed by the GPU deformer is drawn straight from the
/// deformer's buffers (behind another mesh, so its streams start at a non-zero base vertex in the
/// scene), its motion vectors equal the projected motion of the deformed vertices, a frame without
/// change has zero motion, and the static mesh beside it never moves.
#[test]
fn eval_05_deformed_mesh_is_pulled_from_the_deformer_with_correct_motion() {
    let gpu = gpu();
    let mut res = Resources::new(gpu, Budget::UNLIMITED);
    let template = Arc::new(fixtures::sheet(24, 2));
    let tm = template.mesh();
    let mut sheet = Mesh::from_polygons(tm.positions(), tm.faces()).unwrap();
    sheet.uvs = tm.uvs().to_vec();
    let mut scene = Scene::default();
    let ball = scene.add(Mesh::sphere(Vec3::new(-0.6, 0.5, 0.0), 0.2, 32, 16), Material::default());
    let triangles: Vec<[usize; 3]> =
        sheet.indices.as_chunks::<3>().0.iter().map(|t| [t[0] as usize, t[1] as usize, t[2] as usize]).collect();
    let deformed = scene.add(sheet, Material::default());
    let mesh = scene.objects[deformed].mesh;
    let (w, h) = (400, 300);
    let config = RenderConfig { taa: false, overlays: true, picking: true, ..RenderConfig::new(w, h) };
    let mut renderer = Renderer::new(gpu, &mut res, config).unwrap();
    let mut gpu_scene = GpuScene::new(&mut res, &scene).unwrap();
    let mut deformer = GpuDeformer::new(gpu, &mut res, template.clone(), DeformSettings::default()).unwrap();
    assert_ne!(gpu_scene.ranges()[mesh].base_vertex, 0, "the sheet is not the first mesh");
    let camera = Camera {
        position: Vec3::new(0.2, 0.5, 2.2),
        target: Vec3::new(0.2, 0.5, 0.0),
        focal_mm: 35.0,
        near: 0.05,
        exposure: Exposure::ev100(0.0),
        ..Camera::default()
    };
    let light_rig = rig();
    let mut frame = |res: &mut Resources, scene: &mut GpuScene, deformer: &mut GpuDeformer, attach: bool| {
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        if attach {
            scene.attach_deformer(res, &mut encoder, mesh, deformer).unwrap();
        }
        deformer.encode(&mut encoder);
        let input = FrameInput {
            camera: &camera,
            rig: &light_rig,
            scene,
            overlay: &[],
            wireframe: true,
            still: None,
            pick: None,
            reset_history: false,
        };
        renderer.render(res, &mut encoder, input).unwrap();
        gpu.queue.submit([encoder.finish()]);
        renderer.after_submit();
        motion(res, &renderer)
    };
    frame(&mut res, &mut gpu_scene, &mut deformer, true);
    assert!(gpu_scene.set_positions(&mut res, mesh, &[]).is_err(), "a deformed mesh refuses CPU positions");
    let rest = vec3s(&read_f32(deformer.outputs().positions));

    let expressions = vec![0.9; template.expressions().expressions.len()];
    deformer.update(&mut res, &EvalInput { expressions: &expressions, ..EvalInput::default() }).unwrap();
    let m = frame(&mut res, &mut gpu_scene, &mut deformer, false);
    let now = vec3s(&read_f32(deformer.outputs().positions));
    let prev = vec3s(&read_f32(deformer.outputs().prev_positions));
    assert_eq!(prev, rest, "the deformer kept the previous positions");

    let vp = ViewMatrices::new(&camera, w, h, Vec2::ZERO, None).view_projection;
    let screen = |p: Vec3| ndc_to_screen(vp.project_point3(p).xy(), Vec2::new(w as f32, h as f32));
    let clip_w = |p: Vec3| (vp.cols[0] * p.x + vp.cols[1] * p.y + vp.cols[2] * p.z + vp.cols[3]).w;
    // The expected motion at the centre of the pixel under vertex `i`: the previous position of the
    // surface point seen there, interpolated perspective-correctly in the triangle that covers it.
    let expected = |i: usize| -> Option<(usize, Vec2)> {
        let c = screen(now[i]).floor() + Vec2::splat(0.5);
        for t in triangles.iter().filter(|t| t.contains(&i)) {
            let [p0, p1, p2] = t.map(|k| screen(now[k]));
            let area = (p1 - p0).perp_dot(p2 - p0);
            if area.abs() < 1e-6 {
                continue;
            }
            let l = [(p2 - p1).perp_dot(c - p1) / area, (p0 - p2).perp_dot(c - p2) / area];
            let l = [l[0], l[1], 1.0 - l[0] - l[1]];
            if l.iter().any(|&v| v < -1e-4) {
                continue;
            }
            let mu = [0, 1, 2].map(|k| l[k] / clip_w(now[t[k]]));
            let sum = mu[0] + mu[1] + mu[2];
            let before = prev[t[0]] * (mu[0] / sum) + prev[t[1]] * (mu[1] / sum) + prev[t[2]] * (mu[2] / sum);
            return Some((c.y as usize * w as usize + c.x as usize, c - screen(before)));
        }
        None
    };
    let (mut checked, mut worst, mut largest) = (0, 0.0_f32, 0.0_f32);
    for i in 0..now.len() {
        // Interior vertices only (edges of the sheet and of its seams sit on pixel boundaries).
        let (x, y) = (tm.positions()[i].x, tm.positions()[i].y);
        if !(0.1..0.9).contains(&x) || !(0.1..0.9).contains(&y) || i % 7 != 0 {
            continue;
        }
        let Some((pixel, want)) = expected(i) else { continue };
        let got = m[pixel];
        worst = worst.max((got - want).length());
        largest = largest.max(want.length());
        checked += 1;
    }
    println!("deformed motion: {checked} vertices, worst error {worst:.4} px, largest motion {largest:.2} px");
    assert!(checked > 20, "enough probes: {checked}");
    assert!(largest > 2.0, "a real deformation is tested: {largest}");
    assert!(worst < 0.02, "motion vectors follow the deformer: {worst} px");
    let ball_px = screen(Vec3::new(-0.6, 0.5, 0.2));
    assert!(m[ball_px.y as usize * w as usize + ball_px.x as usize].length() < 1e-3, "the static ball stays");

    // A frame without change: the deformer copies once more, motion is zero everywhere.
    deformer.update(&mut res, &EvalInput { expressions: &expressions, ..EvalInput::default() }).unwrap();
    let m = frame(&mut res, &mut gpu_scene, &mut deformer, false);
    let still = m.iter().map(|v| v.length()).fold(0.0_f32, f32::max);
    assert!(still < 1e-3, "no motion once the deformation stopped: {still}");

    // Detached: the scene's own (rest) streams come back with their base vertex.
    gpu_scene.detach_deformer(&mut res, mesh);
    assert!(gpu_scene.external_streams(mesh).is_none());
    assert_eq!(gpu_scene.draws()[deformed].base_vertex, gpu_scene.ranges()[mesh].base_vertex as i32);
    let _ = ball;
    deformer.destroy(&mut res);
    gpu_scene.destroy(&mut res);
    renderer.destroy(&mut res);
}

/// `[ERR-05]`: a blocking wait the GPU cannot meet in time returns `GpuTimeout` instead of
/// blocking (64 copies of 256 MB cannot finish within zero time); the same submission is then
/// waited for within the default limit.
#[test]
fn err_05_blocking_gpu_waits_are_bounded() {
    let gpu = gpu();
    let size = gpu.device.limits().max_buffer_size.min(256 << 20);
    let buffer = |label| {
        gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let (a, b) = (buffer("wait test a"), buffer("wait test b"));
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("wait test") });
    for _ in 0..32 {
        encoder.copy_buffer_to_buffer(&a, 0, &b, 0, size);
        encoder.copy_buffer_to_buffer(&b, 0, &a, 0, size);
    }
    let submission = gpu.queue.submit([encoder.finish()]);
    let start = std::time::Instant::now();
    let early = capture::wait_for_submission(&gpu.device, submission.clone(), std::time::Duration::ZERO, "test copies");
    let returned_after = start.elapsed();
    assert!(
        matches!(early, Err(RenderError::GpuTimeout { what: "test copies", limit_ms: 0 })),
        "a zero limit times out: {early:?}"
    );
    assert!(returned_after.as_millis() < 1000, "the timed-out wait returned at once: {returned_after:?}");
    capture::wait_for_submission(&gpu.device, submission, capture::GPU_WAIT_LIMIT, "test copies")
        .expect("the copies finish within the default limit");
    println!(
        "copies of {} MB: zero-limit wait returned after {returned_after:?}, then finished in {:?}",
        size >> 20,
        start.elapsed()
    );
}
