//! Acceptance tests of WP-14 on the real GPU (headless): the skin furnace and the subsurface
//! kernel against the CPU reference (`AC-WP14-1`), translucency through thin geometry and the
//! softened terminator (`[SHD-03]`, `[SHD-05]`, the measurable part of `AC-WP14-2`).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

mod common;

use humanart_math::{Vec2, Vec3, ndc_to_screen};
use humanart_render::camera::{Camera, Exposure, ViewMatrices};
use humanart_render::capture;
use humanart_render::ibl::EnvironmentSource;
use humanart_render::light::{Background, BuiltinRig, Light, LightColor, LightRig, ProceduralEnvironment};
use humanart_render::renderer::{DebugView, Quality, RenderConfig};
use humanart_render::scene::{Material, Mesh, Scene};
use humanart_render::sss::{self, SKIN_CLASS, SkinShading};

fn rig(environment_nits: f32, lights: Vec<Light>) -> LightRig {
    LightRig {
        name: "skin test",
        lights,
        environment: ProceduralEnvironment::Uniform,
        environment_nits,
        environment_rotation: 0.0,
        background: Background::Gray(0.0),
        ev100: 0.0,
    }
}

fn skin_material(albedo: f32) -> Material {
    Material { base_color: [albedo; 3], roughness: 0.45, f0: 0.028, class: SKIN_CLASS, ..Material::default() }
}

fn camera(position: Vec3, target: Vec3, focal_mm: f32) -> Camera {
    Camera { position, target, focal_mm, near: 0.05, exposure: Exposure::ev100(0.0), ..Camera::default() }
}

fn read(h: &mut common::Harness, name: &str) -> Vec<[f32; 4]> {
    let t = h.renderer.target_texture(name).unwrap();
    let (_, _, bytes) = h.read(t);
    capture::rgba16f_to_f32(&bytes)
}

fn screen(camera: &Camera, w: u32, h: u32, p: Vec3) -> Vec2 {
    let m = ViewMatrices::new(camera, w, h, Vec2::ZERO, None).view_projection;
    ndc_to_screen(m.project_point3(p).xy(), Vec2::new(w as f32, h as f32))
}

fn white_light(direction: Vec3, lux: f32) -> Light {
    let mut light = Light::directional(direction, lux, 6500.0);
    light.color = LightColor::Linear([1.0; 3]);
    light
}

/// A quad facing +Z, `half` metres from its centre to each side.
fn quad(center: Vec3, half: f32) -> Mesh {
    let p = [
        center + Vec3::new(-half, -half, 0.0),
        center + Vec3::new(half, -half, 0.0),
        center + Vec3::new(half, half, 0.0),
        center + Vec3::new(-half, half, 0.0),
    ];
    Mesh::from_polygons(&p, [[0_u32, 1, 2, 3].as_slice()]).unwrap()
}

/// `[AC-WP14-1]` the skin furnace: a white skin sphere in a uniform environment of radiance 1
/// reflects 1 (diffuse weighted by what the two specular lobes do not reflect, plus those lobes),
/// in both diffusion modes; nothing gains energy.
#[test]
fn ac_wp14_1_skin_furnace_gains_no_energy() {
    let mut scene = Scene::default();
    scene.add(Mesh::sphere(Vec3::ZERO, 0.1, 96, 64), skin_material(1.0));
    let (w, h) = (256, 256);
    let cam = camera(Vec3::new(0.0, 0.0, 0.6), Vec3::ZERO, 50.0);
    let centre = screen(&cam, w, h, Vec3::ZERO);
    let radius_px = (screen(&cam, w, h, Vec3::new(0.1, 0.0, 0.0)) - centre).length();
    for quality in [Quality::Review, Quality::Interactive] {
        let config = RenderConfig { taa: false, quality, ..RenderConfig::new(w, h) };
        let mut harness = common::Harness::new(config, &scene);
        harness
            .renderer
            .set_environment(&mut harness.resources, EnvironmentSource::Procedural(ProceduralEnvironment::Uniform))
            .unwrap();
        harness.frames(&cam, &rig(1.0, Vec::new()), 1);
        let hdr = read(&mut harness, "hdr");
        let (mut lo, mut hi, mut sum, mut n) = (f32::MAX, f32::MIN, 0.0_f64, 0);
        for y in 0..h {
            for x in 0..w {
                let d = (Vec2::new(x as f32 + 0.5, y as f32 + 0.5) - centre).length();
                // Away from the silhouette (grazing pixels mix with the backdrop).
                if d > 0.85 * radius_px {
                    continue;
                }
                for &c in &hdr[(y * w + x) as usize][..3] {
                    lo = lo.min(c);
                    hi = hi.max(c);
                    sum += f64::from(c);
                    n += 1;
                }
            }
        }
        let mean = sum / f64::from(n);
        println!("skin furnace {quality:?}: {n} samples, min {lo:.4}, max {hi:.4}, mean {mean:.4}");
        assert!(n > 10_000);
        assert!(hi < 1.015, "{quality:?}: energy gain {hi}");
        assert!(lo > 0.97 && (mean - 1.0).abs() < 0.01, "{quality:?}: energy loss {lo} / {mean}");
    }
}

/// `[AC-WP14-1]` the separable subsurface passes against the CPU reference: a skin quad facing
/// the camera, lit through a hard-edged shadow (a corner of a plate outside the view), blurred on
/// the GPU (the diffuse after both passes) and on the CPU from the GPU's own input (the irradiance
/// before the passes, `sss::blur_pass_reference` along x and then y).
#[test]
fn ac_wp14_1_subsurface_passes_match_the_cpu_kernel() {
    let mut scene = Scene::default();
    scene.add(quad(Vec3::ZERO, 0.3), skin_material(0.6));
    // The occluder: a plate at z = 0.3 above and right of the view; its shadow covers x > 0, y > 0.
    let plate = Mesh::from_polygons(
        &[Vec3::new(0.0, 0.3, 0.3), Vec3::new(1.0, 0.3, 0.3), Vec3::new(1.0, 2.0, 0.3), Vec3::new(0.0, 2.0, 0.3)],
        [[0_u32, 1, 2, 3].as_slice()],
    )
    .unwrap();
    scene.add(plate, Material::default());
    let (w, h) = (256, 256);
    // About 0.2 m across at 1 m: 1.28 pixels per mm, the kernel spans about ±10 pixels.
    let cam = camera(Vec3::new(0.0, 0.0, 1.0), Vec3::ZERO, 120.0);
    let config = RenderConfig { taa: false, debug: DebugView::Irradiance, ..RenderConfig::new(w, h) };
    let mut harness = common::Harness::new(config, &scene);
    let mut light = white_light(Vec3::new(0.0, -1.0, -1.0), 1000.0);
    light.kind = humanart_render::light::LightKind::Directional { angular_diameter: 0.002 };
    harness.frames(&cam, &rig(0.0, vec![light]), 1);
    let input = read(&mut harness, "irradiance");
    let output = read(&mut harness, "diffuse");
    let px_per_mm = (screen(&cam, w, h, Vec3::new(0.001, 0.0, 0.0)) - screen(&cam, w, h, Vec3::ZERO)).x;
    let (wu, hu) = (w as usize, h as usize);
    assert!(input.iter().all(|p| (p[3] - 1.0).abs() < 0.01), "every pixel is skin at 1 m");
    let horizontal = sss::blur_pass_reference(&input, wu, hu, true, px_per_mm, 1.0);
    let reference = sss::blur_pass_reference(&horizontal, wu, hu, false, px_per_mm, 1.0);
    let margin = (sss::KERNEL_RANGE_MM * px_per_mm).ceil() as usize + 2;
    let peak = input.iter().map(|p| p[0]).fold(0.0_f32, f32::max);
    let (mut worst, mut edge_change) = (0.0_f32, 0.0_f32);
    for y in margin..hu - margin {
        for x in margin..wu - margin {
            let i = y * wu + x;
            for c in 0..3 {
                worst = worst.max((output[i][c] - reference[i][c]).abs());
                edge_change = edge_change.max((output[i][c] - input[i][c]).abs());
            }
        }
    }
    println!(
        "subsurface vs CPU: {px_per_mm:.3} px/mm, peak irradiance {peak:.3}, worst difference {worst:.5}, largest change by the blur {edge_change:.4}"
    );
    assert!(peak > 0.1 && edge_change > 0.1 * peak, "the shadow edge gives the blur something to do");
    assert!(worst < 0.01 * peak, "GPU and CPU kernels differ by {worst}");
}

/// Todo 1.4 (the red ring in the eye socket): the subsurface passes across a depth step against the
/// CPU reference. A lit skin shelf 4 mm in front of a skin floor, lit from the side 30 degrees
/// above the floor, casts a shadow 6.9 mm long on the floor beside it (a shadowed pit next to a
/// lit rim). The step is deeper than the shadow lookup's normal offset at that angle (about 1 mm,
/// `SHADOW_TERMINATOR_OFFSET` in lighting.wgsl) plus its least penumbra (0.8 mm): with a 2.5 mm
/// step lit at 45 or 14 degrees the strip beside the shelf kept 6 to 12 % of the light. The
/// GPU blur matches `sss::blur_pass_reference` on its own input (depths included), and in the
/// shadowed strip it carries clearly less light over the step than the same blur on a flat surface
/// would (the reference with the floor's depth everywhere), red the most.
#[test]
fn todo_1_4_subsurface_passes_attenuate_across_a_depth_step() {
    let mut scene = Scene::default();
    scene.add(quad(Vec3::ZERO, 0.3), skin_material(0.6));
    scene.add(quad(Vec3::new(0.0, 0.0, 0.004), 0.015), skin_material(0.6));
    let (w, h) = (256, 256);
    // About 3.8 pixels per mm from 0.83 m (300 mm, the longest lens the camera takes): the view is
    // 67 mm wide, the shelf 30 mm.
    let cam = camera(Vec3::new(0.0, 0.0, 0.8333), Vec3::ZERO, 300.0);
    let config = RenderConfig { taa: false, debug: DebugView::Irradiance, ..RenderConfig::new(w, h) };
    let mut harness = common::Harness::new(config, &scene);
    // Travelling towards -x and into the floor at 30 degrees (tan 30 = 0.577).
    let mut light = white_light(Vec3::new(-1.0, 0.0, -0.577), 1000.0);
    light.kind = humanart_render::light::LightKind::Directional { angular_diameter: 0.002 };
    harness.frames(&cam, &rig(0.0, vec![light]), 1);
    let input = read(&mut harness, "irradiance");
    let output = read(&mut harness, "diffuse");
    let px_per_mm = (screen(&cam, w, h, Vec3::new(0.001, 0.0, 0.0)) - screen(&cam, w, h, Vec3::ZERO)).x;
    let (wu, hu) = (w as usize, h as usize);
    let horizontal = sss::blur_pass_reference(&input, wu, hu, true, px_per_mm, 1.0);
    let reference = sss::blur_pass_reference(&horizontal, wu, hu, false, px_per_mm, 1.0);
    let flat_input: Vec<[f32; 4]> =
        input.iter().map(|p| [p[0], p[1], p[2], if p[3] > 0.0 { 1.0 } else { 0.0 }]).collect();
    let flat_h = sss::blur_pass_reference(&flat_input, wu, hu, true, px_per_mm, 1.0);
    let flat = sss::blur_pass_reference(&flat_h, wu, hu, false, px_per_mm, 1.0);
    let margin = (sss::KERNEL_RANGE_MM * px_per_mm).ceil() as usize + 2;
    let peak = input.iter().map(|p| p[0]).fold(0.0_f32, f32::max);
    let mut worst = 0.0_f32;
    for y in margin..hu - margin {
        for x in margin..wu - margin {
            let i = y * wu + x;
            for c in 0..3 {
                worst = worst.max((output[i][c] - reference[i][c]).abs());
            }
        }
    }
    // The shadowed strip on the floor left of the shelf (the light travels towards -x), 0.3 to
    // 1.5 mm from the shelf's edge, along the middle of its height.
    let edge = screen(&cam, w, h, Vec3::new(-0.015, 0.0, 0.004)).x;
    let (mut gpu_strip, mut flat_strip, mut lit_shelf) = ([0.0_f64; 3], [0.0_f64; 3], 0.0_f64);
    let mut n = 0;
    for y in hu / 2 - 20..hu / 2 + 20 {
        for x in (edge - 1.5 * px_per_mm) as usize..(edge - 0.3 * px_per_mm) as usize {
            let i = y * wu + x;
            for c in 0..3 {
                gpu_strip[c] += f64::from(output[i][c]);
                flat_strip[c] += f64::from(flat[i][c]);
            }
            n += 1;
        }
        lit_shelf += f64::from(input[y * wu + edge as usize + 8][1]);
    }
    let strip_dark = input[hu / 2 * wu + (edge - 0.9 * px_per_mm) as usize];
    println!(
        "depth step: {px_per_mm:.3} px/mm, worst GPU vs CPU {worst:.5} (peak {peak:.3}); shadowed strip GPU {:?} vs flat blur {:?} over {n} px, input there {strip_dark:?}, shelf {:.3}",
        gpu_strip.map(|v| v / f64::from(n)),
        flat_strip.map(|v| v / f64::from(n)),
        lit_shelf / 40.0
    );
    assert!(peak > 0.05 && strip_dark[1] < 0.02 * peak, "the strip beside the shelf is in shadow");
    assert!(worst < 0.01 * peak, "GPU and CPU kernels differ by {worst}");
    for c in 0..3 {
        assert!(gpu_strip[c] < 0.8 * flat_strip[c], "channel {c}: {} vs flat {}", gpu_strip[c], flat_strip[c]);
    }
    assert!(gpu_strip[0] > gpu_strip[1] && gpu_strip[1] >= gpu_strip[2], "red carries farthest: {gpu_strip:?}");
}

/// The whole template as a scene (the skin part in a skin material, the eyeballs, teeth and the
/// other parts plain) and the centre of the left eyeball.
fn template_scene() -> (Scene, Vec3) {
    let generated = humanart_model::generate::build::template(&humanart_model::generate::build::Options {
        lods: false,
        ..Default::default()
    })
    .unwrap();
    let template = generated.template.mesh();
    let mut scene = Scene::default();
    let mut eye = Vec3::ZERO;
    for part in template.parts() {
        let mesh = Mesh::from_template(template, Some(&part.name)).unwrap();
        if part.name == "eyeball-L" {
            let mut used = mesh.indices.clone();
            used.sort_unstable();
            used.dedup();
            eye = used.iter().fold(Vec3::ZERO, |sum, &i| sum + mesh.positions[i as usize]) / used.len() as f32;
        }
        let material = if part.name == "body" {
            Material { base_color: [0.56, 0.38, 0.30], ..skin_material(1.0) }
        } else {
            Material { base_color: [0.6; 3], roughness: 0.3, ..Material::default() }
        };
        scene.add(mesh, material);
    }
    (scene, eye)
}

/// Both subsurface passes on the CPU with the off-plane `rates` given (`sss::blur_pass_with_rates`
/// along x and then y): all zero is the blur that weighs taps by their screen offset alone.
fn blur_on_cpu(
    input: &[[f32; 4]],
    w: usize,
    h: usize,
    px_per_mm: f32,
    rates: &[[f32; 3]; sss::KERNEL_TAPS],
) -> Vec<[f32; 4]> {
    let horizontal = sss::blur_pass_with_rates(input, w, h, true, px_per_mm, 1.0, rates);
    sss::blur_pass_with_rates(&horizontal, w, h, false, px_per_mm, 1.0, rates)
}

/// The value below which `fraction` of `values` lie.
fn percentile(mut values: Vec<f32>, fraction: f32) -> f32 {
    values.sort_by(f32::total_cmp);
    values[((values.len() - 1) as f32 * fraction) as usize]
}

/// The light the blur carries into shadowed skin: over the skin pixels `margin` inside the image
/// whose `input` (the lighting before the passes) is below 3 % of the lit level (the 95th
/// percentile of the skin, per channel), the mean of `output` relative to that level per channel,
/// and the number of such pixels.
fn light_in_shadow(input: &[[f32; 4]], output: &[[f32; 4]], w: usize, h: usize, margin: usize) -> ([f64; 3], usize) {
    let lit: [f32; 3] =
        std::array::from_fn(|c| percentile(input.iter().filter(|p| p[3] > 0.0).map(|p| p[c]).collect(), 0.95));
    let (mut sum, mut n) = ([0.0_f64; 3], 0_usize);
    for y in margin..h - margin {
        for x in margin..w - margin {
            let i = y * w + x;
            if input[i][3] > 0.0 && input[i][1] < 0.03 * lit[1] {
                for c in 0..3 {
                    sum[c] += f64::from(output[i][c] / lit[c]);
                }
                n += 1;
            }
        }
    }
    (sum.map(|v| v / n.max(1) as f64), n)
}

/// Two lightings of unit albedo side by side as a picture for the review folder: skin-coloured,
/// the lit level at 0.7 of white, display gamma 2.2.
fn save_pair(name: &str, left: &[[f32; 4]], right: &[[f32; 4]], w: usize, h: usize, lit: f32) {
    let albedo = [0.56_f32, 0.38, 0.30];
    let mut rgba = vec![255_u8; 2 * w * h * 4];
    for y in 0..h {
        for x in 0..2 * w {
            let p = if x < w { left[y * w + x] } else { right[y * w + x - w] };
            for c in 0..3 {
                let v = (0.7 * albedo[c] / albedo[0] * p[c] / lit).clamp(0.0, 1.0).powf(1.0 / 2.2);
                rgba[(y * 2 * w + x) * 4 + c] = (v * 255.0).round() as u8;
            }
        }
    }
    common::save_png(name, 2 * w as u32, h as u32, &rgba);
}

/// Todo 1.4, the red ring in the eye socket under a side light, measured on the template: the left
/// eye from the front, the built-in harsh side light and a 5 nit uniform fill. "Before" is the blur
/// that weighs taps by their screen offset alone, run on the CPU on the GPU's own input
/// (`sss::blur_pass_with_rates` with zero rates); "after" is the GPU's output, which the CPU
/// reference with the off-plane rates follows. The ring is the light the passes carry into the
/// shadowed skin of the socket (`light_in_shadow`): red far above green. Picture (before left,
/// after right): target/agents/render/shots/skin_eye_socket_ring.png.
#[test]
fn todo_1_4_eye_socket_ring_under_side_light() {
    let (scene, eye) = template_scene();
    assert!(eye.x > 0.01 && eye.y > 1.3, "the left eyeball: {eye:?}");
    let (w, h) = (512_usize, 512_usize);
    // 40 mm across from half a metre: the depth of the socket changes the scale by 2 %.
    let cam = camera(eye + Vec3::new(0.0, 0.0, 0.5), eye, 300.0);
    let config = RenderConfig { taa: false, debug: DebugView::Irradiance, ..RenderConfig::new(w as u32, h as u32) };
    let mut harness = common::Harness::new(config, &scene);
    let side = LightRig::builtin(BuiltinRig::HarshSidelight, eye, 0.12).lights;
    harness.frames(&cam, &rig(5.0, side), 1);
    let input = read(&mut harness, "irradiance");
    let output = read(&mut harness, "diffuse");
    let px_per_mm =
        (screen(&cam, w as u32, h as u32, eye + Vec3::new(0.001, 0.0, 0.0)) - screen(&cam, w as u32, h as u32, eye)).x;
    let before = blur_on_cpu(&input, w, h, px_per_mm, &[[0.0; 3]; sss::KERNEL_TAPS]);
    let reference = blur_on_cpu(&input, w, h, px_per_mm, &sss::off_plane_rates());
    let margin = 8;
    let (unblurred, n) = light_in_shadow(&input, &input, w, h, margin);
    let (ring_before, _) = light_in_shadow(&input, &before, w, h, margin);
    let (ring_reference, _) = light_in_shadow(&input, &reference, w, h, margin);
    let (ring_after, _) = light_in_shadow(&input, &output, w, h, margin);
    let skin = input.iter().filter(|p| p[3] > 0.0).count();
    let excess = |ring: [f64; 3]| ring[0] - ring[1];
    println!(
        "eye socket, {px_per_mm:.2} px/mm, {n} shadowed of {skin} skin pixels; light in the shadow relative to the lit skin (r, g, b): unblurred {unblurred:.4?}, before {ring_before:.4?}, after (GPU) {ring_after:.4?}, after (CPU reference) {ring_reference:.4?}; red excess before {:.4}, after {:.4} ({:.0} % of before)",
        excess(ring_before),
        excess(ring_after),
        100.0 * excess(ring_after) / excess(ring_before)
    );
    let lit = percentile(input.iter().filter(|p| p[3] > 0.0).map(|p| p[0]).collect(), 0.95);
    save_pair("skin_eye_socket_ring.png", &before, &output, w, h, lit);
    assert!(n > skin / 50 && n > 2000, "the socket has shadowed skin: {n} of {skin}");
    assert!(excess(ring_before) > 0.01, "the ring is there without the off-plane weights: {ring_before:?}");
    assert!(
        (ring_after[0] - ring_reference[0]).abs() < 0.15 * ring_reference[0],
        "GPU {ring_after:?} against the CPU reference {ring_reference:?}"
    );
    assert!(
        excess(ring_after) < RING_LEFT * excess(ring_before),
        "red excess {} against {} before",
        excess(ring_after),
        excess(ring_before)
    );
}

/// The share of the red excess in the shadowed eye socket the off-plane weights may leave
/// (`todo_1_4_eye_socket_ring_under_side_light`; measured 0.79: the ring gets weaker, the profile's
/// own bleed along the skin from the lit border stays).
const RING_LEFT: f64 = 0.85;

/// Todo 1.4, what the off-plane weights must not take away: the glow past the terminator of a
/// convex curve. Skin spheres of 30 mm (a chin, a cheek) and 10 mm radius (the tip of a nose, the
/// rim of an ear) lit from the side: the red light the passes carry into the shadowed side on the
/// GPU against the blur that weighs taps by their screen offset alone (on the CPU, from the GPU's
/// input). A tap on a curve of radius R lies x² / 2R off the tangent plane, a fraction of a
/// millimetre within red's reach, so nearly all of the glow stays.
#[test]
fn todo_1_4_off_plane_weights_keep_the_terminator_glow() {
    let (w, h) = (256_usize, 256_usize);
    let light = white_light(Vec3::new(-1.0, 0.0, 0.0), 1000.0);
    for (radius, distance, kept_at_least) in [(0.03_f32, 1.2_f32, GLOW_KEPT[0]), (0.01, 0.4, GLOW_KEPT[1])] {
        let mut scene = Scene::default();
        scene.add(Mesh::sphere(Vec3::ZERO, radius, 192, 128), skin_material(1.0));
        let cam = camera(Vec3::new(0.0, 0.0, distance), Vec3::ZERO, 300.0);
        let config = RenderConfig { taa: false, debug: DebugView::Irradiance, ..RenderConfig::new(w as u32, h as u32) };
        let mut harness = common::Harness::new(config, &scene);
        harness.frames(&cam, &rig(0.0, vec![light]), 1);
        let input = read(&mut harness, "irradiance");
        let output = read(&mut harness, "diffuse");
        let px_per_mm = (screen(&cam, w as u32, h as u32, Vec3::new(0.001, 0.0, 0.0))
            - screen(&cam, w as u32, h as u32, Vec3::ZERO))
        .x;
        let before = blur_on_cpu(&input, w, h, px_per_mm, &[[0.0; 3]; sss::KERNEL_TAPS]);
        let reference = blur_on_cpu(&input, w, h, px_per_mm, &sss::off_plane_rates());
        let (glow_before, n) = light_in_shadow(&input, &before, w, h, 2);
        let (glow_reference, _) = light_in_shadow(&input, &reference, w, h, 2);
        let (glow_after, _) = light_in_shadow(&input, &output, w, h, 2);
        println!(
            "sphere of {:.0} mm, {px_per_mm:.2} px/mm, {n} shadowed pixels; light past the terminator (r, g, b): before {glow_before:.4?}, after (GPU) {glow_after:.4?}, after (CPU reference) {glow_reference:.4?}; red kept {:.1} %",
            radius * 1000.0,
            100.0 * glow_after[0] / glow_before[0]
        );
        assert!(n > 1000 && glow_before[0] > 1e-3, "the shadowed side takes red light: {glow_before:?} over {n}");
        assert!(glow_after[0] > glow_after[1] && glow_after[1] >= glow_after[2] * 0.99, "red the farthest");
        assert!(
            glow_after[0] > kept_at_least * glow_before[0],
            "{radius} m: red glow {} of {} before",
            glow_after[0],
            glow_before[0]
        );
    }
}

/// The share of the red glow past the terminator the off-plane weights must keep on a sphere of
/// 30 mm and of 10 mm radius (`todo_1_4_off_plane_weights_keep_the_terminator_glow`; measured 1.01
/// and 1.00).
const GLOW_KEPT: [f64; 2] = [0.97, 0.95];

/// `[SHD-03]` light through a thin part: a 6 mm skin disc lit from behind glows on the front,
/// red far more than green and blue, as much as the profile's transmittance through its thickness
/// says; with translucency 0 the front stays dark.
#[test]
fn shd_03_backlit_thin_skin_glows_red() {
    let disc = || Mesh::ellipsoid(Vec3::ZERO, Vec3::new(0.05, 0.05, 0.003), 96, 48);
    let (w, h) = (128, 128);
    let cam = camera(Vec3::new(0.0, 0.0, 0.5), Vec3::ZERO, 120.0);
    let lux = 10_000.0;
    // Travelling towards the camera: the light hits the back of the disc.
    let light = white_light(Vec3::new(0.0, 0.0, 1.0), lux);
    let mut glow = Vec::new();
    for translucency in [1.0, 0.0] {
        let mut scene = Scene::default();
        let o = scene.add(disc(), skin_material(1.0));
        scene.objects[o].skin = SkinShading { translucency, ..SkinShading::default() };
        let config = RenderConfig { taa: false, ..RenderConfig::new(w, h) };
        let mut harness = common::Harness::new(config, &scene);
        harness.frames(&cam, &rig(0.0, vec![light]), 1);
        let diffuse = read(&mut harness, "diffuse");
        glow.push(diffuse[(h / 2 * w + w / 2) as usize]);
    }
    let [lit, dark] = [glow[0], glow[1]];
    // Expected at the centre: E T(d) / pi over the 6 mm (minus the 0.5 mm shrink and the depth
    // bias), times the diffuse weight of the lobes (about 0.97 head-on).
    let t = sss::transmittance(5.5, 1.0);
    let expected = lux * t[0] / std::f32::consts::PI;
    println!("backlit disc centre: {lit:?} (translucency 0: {dark:?}); expected red about {expected:.2}");
    assert!(dark[0] < 1e-3 && dark[1] < 1e-3, "no translucency, no glow: {dark:?}");
    assert!(lit[0] > 3.0 * lit[1] && lit[1] >= lit[2], "red glows: {lit:?}");
    assert!((lit[0] / expected - 1.0).abs() < 0.35, "red glow {} vs the profile's {expected}", lit[0]);
}

/// `[SHD-05]` `[SHD-04]` the terminator: a skin sphere lit from the side carries light past the
/// shadow terminator, red the farthest, in both diffusion modes, where a plain white sphere is
/// dark at the same pixels.
#[test]
fn shd_05_terminator_softens_through_the_profile() {
    let (w, h) = (256, 256);
    let cam = camera(Vec3::new(0.0, 0.0, 0.5), Vec3::ZERO, 50.0);
    // A small sphere: 3 cm radius, about the curvature of a nose or a chin.
    let radius = 0.03;
    let light = white_light(Vec3::new(-1.0, 0.0, 0.0), 1000.0);
    // Probe just past the terminator on the side facing the camera: normal 4 degrees beyond it.
    let angle = 94.0_f32.to_radians();
    let normal = Vec3::new(angle.cos(), 0.0, angle.sin());
    let probe = screen(&cam, w, h, normal * radius);
    let index = (probe.y as u32 * w + probe.x as u32) as usize;
    let mut results = Vec::new();
    for (quality, class) in [(Quality::Review, SKIN_CLASS), (Quality::Interactive, SKIN_CLASS), (Quality::Review, 0)] {
        let mut scene = Scene::default();
        let material = Material { class, ..skin_material(1.0) };
        scene.add(Mesh::sphere(Vec3::ZERO, radius, 128, 96), material);
        let config = RenderConfig { taa: false, quality, ..RenderConfig::new(w, h) };
        let mut harness = common::Harness::new(config, &scene);
        harness.frames(&cam, &rig(0.0, vec![light]), 1);
        let diffuse = read(&mut harness, "diffuse");
        // Normalised by the lit side facing the light.
        let lit_px = screen(&cam, w, h, Vec3::new(radius * 0.7, 0.0, radius * 0.7));
        let lit = diffuse[(lit_px.y as u32 * w + lit_px.x as u32) as usize];
        let d = diffuse[index];
        let ratio = [d[0] / lit[0], d[1] / lit[1], d[2] / lit[2]];
        println!("terminator {quality:?} class {class}: past the terminator {ratio:?} of the lit side");
        results.push(ratio);
    }
    for ratio in &results[..2] {
        assert!(ratio[0] > 1e-3, "light reaches past the terminator: {ratio:?}");
        assert!(ratio[0] > ratio[1] && ratio[1] >= ratio[2] * 0.99, "red the farthest: {ratio:?}");
    }
    assert!(results[2][0] < 1e-3, "a plain material ends at the terminator: {:?}", results[2]);
}

/// The shipped template's skin as a render mesh: the `body` part of LOD0 of
/// `humanart_model::generate::build::template` with the default options (the scanned face and the
/// finished head; the LODs are not built, they do not change LOD0), not the scan-fitting base
/// `generate::body_surface`.
fn template_mesh() -> Mesh {
    let generated = humanart_model::generate::build::template(&humanart_model::generate::build::Options {
        lods: false,
        ..Default::default()
    })
    .unwrap();
    Mesh::from_template(generated.template.mesh(), Some("body")).unwrap()
}

/// `[AC-WP14-3]` the subsurface passes at 1440p with the template's face filling the frame
/// height (16:9, so skin covers about 58 % of the pixels; review image
/// `skin_sss_timing_1440p.png`), Review, a key light and a uniform environment: the median GPU
/// time of `sss_h` + `sss_v` over 60 frames against the 1.2 ms of spec 14.2.
#[test]
fn ac_wp14_3_subsurface_passes_at_1440p() {
    let mut scene = Scene::default();
    scene.add(template_mesh(), Material { base_color: [0.56, 0.38, 0.30], ..skin_material(1.0) });
    let (w, h) = (2560, 1440);
    let config = RenderConfig { taa: false, ..RenderConfig::new(w, h) };
    let mut harness = common::Harness::new(config, &scene);
    let cam =
        Camera { exposure: Exposure::ev100(9.0), ..camera(Vec3::new(0.0, 1.6, 0.3), Vec3::new(0.0, 1.6, 0.0), 50.0) };
    let rig = rig(0.3, vec![white_light(Vec3::new(-0.5, -0.3, -1.0), 2000.0)]);
    let (mut sss, mut total, mut covered) = (Vec::new(), Vec::new(), 0.0);
    for i in 0..80 {
        harness.frame(&cam, &rig, &[], false, None, i == 0);
        common::wait(&harness.gpu.device);
        if i == 0 {
            let irradiance = read(&mut harness, "diffuse");
            covered = irradiance.iter().filter(|p| p[3] > 0.0).count() as f64 / irradiance.len() as f64;
            harness.save_display("skin_sss_timing_1440p.png");
        }
        if harness.renderer.collect_timings() && i >= 20 {
            let report = harness.renderer.timings();
            let ms = |name: &str| {
                report.scopes.iter().find(|s| s.name == name).and_then(|s| s.gpu_nanos).unwrap_or(0) as f64 / 1e6
            };
            sss.push(ms("sss_h") + ms("sss_v"));
            total.push(report.gpu_total_nanos().unwrap_or(0) as f64 / 1e6);
        }
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied().unwrap_or(0.0)
    };
    let (sss_ms, total_ms) = (median(&mut sss), median(&mut total));
    println!(
        "1440p, skin covering {:.0} % of the frame: subsurface passes {sss_ms:.3} ms (median of {}), whole frame {total_ms:.3} ms",
        covered * 100.0,
        sss.len()
    );
    assert!(covered > 0.5, "the face fills the frame");
    assert!(sss_ms > 0.0 && sss_ms <= 1.2, "subsurface passes {sss_ms:.3} ms over the 1.2 ms budget");
}

/// The template's head and neck (the triangles above `cut`), so shadow maps fit the head.
fn head_mesh(cut: f32) -> Mesh {
    let body = template_mesh();
    let mut map = vec![u32::MAX; body.positions.len()];
    let mut head = Mesh::default();
    for t in body.indices.as_chunks::<3>().0 {
        if t.iter().any(|&i| body.positions[i as usize].y < cut) {
            continue;
        }
        for &i in t {
            let i = i as usize;
            if map[i] == u32::MAX {
                map[i] = head.positions.len() as u32;
                head.positions.push(body.positions[i]);
                head.normals.push(body.normals[i]);
                head.uvs.push(body.uvs.get(i).copied().unwrap_or(Vec2::ZERO));
            }
            head.indices.push(map[i]);
        }
    }
    head
}

/// `[AC-WP14-2]` the review stills for the owner's rubric sign-off (the rubric scores are his, not
/// this test's): the template head with three skin types from the pigment model (melanin 0.1,
/// 0.4, 0.8; albedo and profile scale from `PigmentModel`), under a hard sidelight (terminator
/// softness, front three-quarter view) and a backlight grazing the back of the head (ear
/// translucency, side view), Review quality, 4096 shadow maps, TAA converged over 32 frames;
/// plus the medium skin without skin shading (class 0) under both for comparison. Written to
/// target/agents/render/shots/wp14_*.png. The test checks what can be measured: the frames are
/// finite and lit, and under the backlight the skin's lighting is redder than the white light
/// (the plain material's lighting has red/green 1).
#[test]
fn ac_wp14_2_sidelight_and_backlight_stills() {
    use humanart_model::skin_appearance::{Mapping, PigmentModel, PigmentParams};
    let mesh = head_mesh(1.40);
    let model = PigmentModel::new(Mapping::default());
    let (w, h) = (960, 1200);
    let front = Camera {
        exposure: Exposure::ev100(7.5),
        ..camera(Vec3::new(0.30, 1.62, 0.66), Vec3::new(0.0, 1.585, 0.0), 85.0)
    };
    let profile = Camera {
        exposure: Exposure::ev100(7.5),
        ..camera(Vec3::new(0.68, 1.6, 0.22), Vec3::new(0.0, 1.575, -0.01), 85.0)
    };
    // The key from the character's left, slightly above; the back light from behind the head,
    // grazing its right side and the right ear.
    let side = white_light(Vec3::new(1.0, -0.2, -0.1), 1000.0);
    let back = white_light(Vec3::new(-0.15, -0.2, 1.0), 2000.0);
    let lightings = [("sidelight", side, front, 8.0), ("backlight", back, profile, 4.0)];
    let mut cases = Vec::new();
    for (skin_name, melanin) in [("light", 0.1), ("medium", 0.4), ("dark", 0.8)] {
        let response = model.evaluate(&PigmentParams { melanin, ..PigmentParams::default() });
        let albedo = response.albedo.map(|v| v as f32);
        let radius = response.sss_radius_mm.map(|v| v as f32);
        cases.push((skin_name, albedo, SkinShading::scale_for_radius_mm(radius), SKIN_CLASS));
    }
    cases.push(("medium-no-skin-shading", cases[1].1, 1.0, 0));
    let config = RenderConfig { shadow_map_size: 4096, ..RenderConfig::new(w, h) };
    for (name, albedo, scale, class) in cases {
        let mut scene = Scene::default();
        let o = scene.add(
            mesh.clone(),
            Material { base_color: albedo, roughness: 0.45, f0: 0.028, class, ..Material::default() },
        );
        scene.objects[o].skin = SkinShading { scatter_scale: scale, ..SkinShading::default() };
        let mut harness = common::Harness::new(config, &scene);
        for (lighting, light, cam, ambient) in lightings {
            let mut light_rig = rig(ambient, vec![light]);
            light_rig.background = Background::Gray(0.02);
            harness.frames(&cam, &light_rig, 32);
            let file = format!("wp14_{name}_{lighting}.png");
            let (_, _, rgba) = harness.save_display(&file);
            let lit = rgba.as_chunks::<4>().0.iter().filter(|p| p[0] > 40).count() as f64 / (rgba.len() / 4) as f64;
            let diffuse = read(&mut harness, "diffuse");
            let (mut r, mut g) = (0.0_f64, 0.0_f64);
            for p in &diffuse {
                r += f64::from(p[0]);
                g += f64::from(p[1]);
            }
            let redness = (r / g.max(1e-9)) as f32;
            println!(
                "{file}: albedo {albedo:.3?}, profile scale {scale:.2}, {:.0} % of pixels lit, lighting red/green {redness:.3}",
                lit * 100.0
            );
            assert!(diffuse.iter().all(|p| p.iter().all(|v| v.is_finite())), "{file}: finite");
            assert!(lit > 0.05, "{file}: the head is lit");
            if lighting == "backlight" && class == SKIN_CLASS {
                assert!(redness > 1.01, "{file}: light through the skin is red ({redness})");
            }
        }
    }
}
