//! `[PERF-R6]` GPU pass timings of the core renderer at 1080p and 1440p, Review quality, TAA on,
//! measured with the timestamp-query profiler on the material test scene (a bust, ten spheres and
//! a ground plane; the test prints its size). The numbers are printed for the record;
//! the assertion is the 8 ms whole-viewport budget of spec 14.2, which a scene this small must
//! meet with a wide margin (the budget is for the hero character, which these passes do not draw
//! yet).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

mod common;

use std::collections::BTreeMap;

use humanart_math::Vec3;
use humanart_render::camera::{Camera, Exposure};
use humanart_render::light::{BuiltinRig, LightRig};
use humanart_render::renderer::RenderConfig;
use humanart_render::scene::Scene;

const WARM_UP: u32 = 30;
const MEASURED: u32 = 120;

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values.get(values.len() / 2).copied().unwrap_or(0.0)
}

/// Renders `WARM_UP + MEASURED` frames of a slowly orbiting camera at `w` x `h` and returns the
/// median GPU milliseconds per pass, the median total and the median CPU encode milliseconds.
fn measure(w: u32, h: u32) -> (Vec<(String, f64)>, f64, f64) {
    let scene = Scene::material_test();
    let mut harness = common::Harness::new(RenderConfig::new(w, h), &scene);
    let (vertices, indices) = harness.scene.counts();
    println!("scene: {vertices} vertices, {} triangles", indices / 3);
    assert!(
        harness.gpu.features().contains(wgpu::Features::TIMESTAMP_QUERY),
        "this machine's GPU has timestamp queries"
    );
    let rig = LightRig::builtin(BuiltinRig::NeutralStudio, Vec3::new(0.0, 1.6, -0.05), 0.12);
    let mut camera = Camera {
        position: Vec3::new(0.0, 1.55, 1.1),
        target: Vec3::new(0.0, 1.45, 0.0),
        focal_mm: 50.0,
        exposure: Exposure::ev100(rig.ev100),
        ..Camera::default()
    };
    let mut per_pass: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    let mut totals = Vec::new();
    let mut cpu = Vec::new();
    for i in 0..WARM_UP + MEASURED {
        camera.orbit(0.002, 0.0);
        let start = std::time::Instant::now();
        harness.frame(&camera, &rig, &[], false, None, i == 0);
        let encode = start.elapsed().as_secs_f64() * 1e3;
        common::wait(&harness.gpu.device);
        if harness.renderer.collect_timings() && i >= WARM_UP {
            let report = harness.renderer.timings();
            assert!(report.gpu_timed);
            for scope in &report.scopes {
                per_pass.entry(scope.name).or_default().push(scope.gpu_nanos.unwrap_or(0) as f64 / 1e6);
            }
            totals.push(report.gpu_total_nanos().unwrap_or(0) as f64 / 1e6);
            cpu.push(encode);
        }
    }
    assert!(totals.len() as u32 >= MEASURED / 2, "only {} reports arrived", totals.len());
    let passes = per_pass.into_iter().map(|(name, mut v)| (name.to_owned(), median(&mut v))).collect();
    (passes, median(&mut totals), median(&mut cpu))
}

#[test]
fn perf_r6_core_pass_timings_at_1080p_and_1440p() {
    for (w, h) in [(1920, 1080), (2560, 1440)] {
        let (passes, total, cpu) = measure(w, h);
        println!("{w}x{h} Review, TAA: GPU total {total:.3} ms (median), CPU encode + submit {cpu:.3} ms");
        for (name, ms) in &passes {
            println!("  {name:<12} {ms:.3} ms");
        }
        assert!(total > 0.0, "the passes take time");
        assert!(total <= 8.0, "{w}x{h}: {total:.3} ms is over the 8 ms viewport budget");
    }
}
