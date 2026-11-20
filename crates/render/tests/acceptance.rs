//! Acceptance tests of WP-11 on the real GPU (headless): IBL, shadows and the tone mapper
//! (`AC-WP11-1`), reversed-Z, TAA and motion vectors (`AC-WP11-2`), picking.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use humanart_math::color::tonemap::ToneMapper;
use humanart_math::color::transfer::linear_to_srgb_f32;
use humanart_math::{Mat4, Vec2, Vec3, ndc_to_screen};
use humanart_render::camera::{Camera, Exposure, ViewMatrices};
use humanart_render::capture;
use humanart_render::ibl::EnvironmentSource;
use humanart_render::light::{Background, BuiltinRig, Light, LightRig, ProceduralEnvironment};
use humanart_render::renderer::{DebugView, FrameInput, RenderConfig};
use humanart_render::scene::{Material, Mesh, Scene};

fn empty_rig(environment: ProceduralEnvironment, nits: f32) -> LightRig {
    LightRig {
        name: "test",
        lights: Vec::new(),
        environment,
        environment_nits: nits,
        environment_rotation: 0.0,
        background: Background::Gray(0.18),
        ev100: 0.0,
    }
}

fn camera_at(position: Vec3, target: Vec3) -> Camera {
    Camera { position, target, focal_mm: 35.0, near: 0.05, exposure: Exposure::ev100(0.0), ..Camera::default() }
}

/// The pixel (column, row) a world point lands on.
fn pixel_of(camera: &Camera, w: u32, h: u32, p: Vec3) -> (usize, usize) {
    let m = ViewMatrices::new(camera, w, h, Vec2::ZERO, None).view_projection;
    let ndc = m.project_point3(p);
    let s = ndc_to_screen(ndc.xy(), Vec2::new(w as f32, h as f32));
    (s.x as usize, s.y as usize)
}

fn texel(data: &[[f32; 4]], w: u32, (x, y): (usize, usize)) -> [f32; 4] {
    data[y * w as usize + x]
}

fn read_target(h: &mut common::Harness, name: &str) -> Vec<[f32; 4]> {
    let t = h.renderer.target_texture(name).unwrap();
    let (_, _, bytes) = h.read(t);
    capture::rgba16f_to_f32(&bytes)
}

fn no_taa(w: u32, h: u32) -> RenderConfig {
    RenderConfig { taa: false, ..RenderConfig::new(w, h) }
}

/// `[AC-WP11-1]` white furnace and an analytic sky: in a uniform environment of radiance 1 the
/// diffuse lighting of every surface is 1 (SH irradiance pi, divided by pi) and a white metal
/// reflects 1 (prefiltered radiance 1 times the energy-compensated split sum); under the CIE
/// overcast sky an upward surface gets 7 pi Lz / 9, so its diffuse lighting is 7/9 Lz.
#[test]
fn ac_wp11_1_ibl_white_furnace_and_overcast_sky() {
    let mut scene = Scene::default();
    scene.add(
        Mesh::ground(Vec3::ZERO, 4.0, 4),
        Material { base_color: [1.0; 3], roughness: 1.0, f0: 0.0, ..Material::default() },
    );
    scene.add(
        Mesh::sphere(Vec3::new(-0.4, 0.4, 0.0), 0.3, 64, 48),
        Material { base_color: [1.0; 3], roughness: 0.6, f0: 0.0, ..Material::default() },
    );
    scene.add(
        Mesh::sphere(Vec3::new(0.4, 0.4, 0.0), 0.3, 64, 48),
        Material { base_color: [1.0; 3], roughness: 0.35, metallic: 1.0, ..Material::default() },
    );
    let (w, hgt) = (480, 320);
    let mut h = common::Harness::new(no_taa(w, hgt), &scene);
    h.renderer
        .set_environment(&mut h.resources, EnvironmentSource::Procedural(ProceduralEnvironment::Uniform))
        .unwrap();
    let camera = camera_at(Vec3::new(0.0, 1.0, 2.0), Vec3::new(0.0, 0.35, 0.0));
    let rig = empty_rig(ProceduralEnvironment::Uniform, 1.0);
    h.frames(&camera, &rig, 1);
    let diffuse = read_target(&mut h, "diffuse");
    let specular = read_target(&mut h, "specular");
    let mut worst_diffuse: f32 = 0.0;
    let mut worst_metal: f32 = 0.0;
    for p in
        [Vec3::new(-0.4, 0.7, 0.0), Vec3::new(-0.4, 0.4, 0.3), Vec3::new(-0.65, 0.45, 0.15), Vec3::new(0.0, 0.0, 0.45)]
    {
        let d = texel(&diffuse, w, pixel_of(&camera, w, hgt, p));
        for c in &d[..3] {
            worst_diffuse = worst_diffuse.max((c - 1.0).abs());
        }
    }
    for p in [Vec3::new(0.4, 0.7, 0.0), Vec3::new(0.4, 0.4, 0.3), Vec3::new(0.62, 0.5, 0.15)] {
        let s = texel(&specular, w, pixel_of(&camera, w, hgt, p));
        let d = texel(&diffuse, w, pixel_of(&camera, w, hgt, p));
        assert!(d[0].abs() < 1e-3, "a metal has no diffuse: {d:?}");
        for c in &s[..3] {
            worst_metal = worst_metal.max((c - 1.0).abs());
        }
    }
    println!("furnace: worst diffuse error {worst_diffuse:.4}, worst metal error {worst_metal:.4}");
    assert!(worst_diffuse < 0.015, "diffuse furnace error {worst_diffuse}");
    assert!(worst_metal < 0.03, "specular furnace error {worst_metal}");

    h.renderer
        .set_environment(&mut h.resources, EnvironmentSource::Procedural(ProceduralEnvironment::Overcast))
        .unwrap();
    let rig = empty_rig(ProceduralEnvironment::Overcast, 1.0);
    h.frames(&camera, &rig, 1);
    let diffuse = read_target(&mut h, "diffuse");
    let up = texel(&diffuse, w, pixel_of(&camera, w, hgt, Vec3::new(0.0, 0.0, 0.45)));
    let top = texel(&diffuse, w, pixel_of(&camera, w, hgt, Vec3::new(-0.4, 0.7, 0.0)));
    let expected = 7.0 / 9.0;
    println!("overcast: ground {:.4}, sphere top {:.4}, expected {expected:.4}", up[1], top[1]);
    assert!((up[1] - expected).abs() / expected < 0.04, "ground under the overcast sky: {up:?}");
    assert!((top[1] - expected).abs() / expected < 0.04, "sphere top under the overcast sky: {top:?}");
}

/// `[AC-WP11-1]` a hard directional light at 45 degrees: the sphere's shadow lies where the light
/// says (its centre fully dark), the lit ground gets E cos(45) / pi, and the shadow ends within
/// the filter width of the analytic outline.
#[test]
fn ac_wp11_1_shadows_fall_where_the_light_says() {
    let mut scene = Scene::default();
    let ground = scene.add(
        Mesh::ground(Vec3::ZERO, 4.0, 8),
        Material { base_color: [1.0; 3], roughness: 1.0, f0: 0.0, ..Material::default() },
    );
    scene.objects[ground].casts_shadow = false;
    scene.add(
        Mesh::sphere(Vec3::new(0.0, 1.0, 0.0), 0.25, 64, 48),
        Material { base_color: [0.8, 0.3, 0.2], ..Material::default() },
    );
    let (w, hgt) = (640, 480);
    let mut h = common::Harness::new(no_taa(w, hgt), &scene);
    let mut light = Light::directional(Vec3::new(1.0, -1.0, 0.0), 1000.0, 6500.0);
    light.shadow.soft = false;
    light.color = humanart_render::light::LightColor::Linear([1.0, 1.0, 1.0]);
    let mut rig = empty_rig(ProceduralEnvironment::Uniform, 0.0);
    rig.lights.push(light);
    let camera = camera_at(Vec3::new(0.0, 4.0, 3.0), Vec3::new(0.5, 0.0, 0.0));
    h.frames(&camera, &rig, 1);
    let diffuse = read_target(&mut h, "diffuse");
    let lit = 1000.0 * std::f32::consts::FRAC_1_SQRT_2 / std::f32::consts::PI;
    let at = |p: Vec3| texel(&diffuse, w, pixel_of(&camera, w, hgt, p))[1];
    let center = at(Vec3::new(1.0, 0.0, 0.0));
    let inside = at(Vec3::new(1.2, 0.0, 0.1));
    let outside_z = at(Vec3::new(1.0, 0.0, 0.4));
    let outside_x = at(Vec3::new(1.5, 0.0, 0.0));
    let under = at(Vec3::new(0.0, 0.0, 0.0));
    println!(
        "shadow: centre {center:.2}, inside {inside:.2}, outside {outside_z:.2} {outside_x:.2}, under {under:.2}, lit {lit:.2}"
    );
    assert!(center < 0.01 * lit && inside < 0.01 * lit, "the umbra is dark");
    for v in [outside_z, outside_x, under] {
        assert!((v - lit).abs() / lit < 0.01, "lit ground {v} vs {lit}");
    }
    let (_, _, rgba) = h.display();
    common::save_png("shadow_directional.png", w, hgt, &rgba);
}

/// `[AC-WP11-1]` `[COL-02]` the display transform matches the CPU tone mappers of
/// `humanart_math` pixel for pixel: the display pixel is the 8-bit sRGB encoding of the
/// reference mapper applied to the exposed HDR value the pass read (within one code value).
#[test]
fn ac_wp11_1_tone_mappers_match_the_reference() {
    let scene = Scene::material_test();
    for mapper in [ToneMapper::PbrNeutral, ToneMapper::AcesFitted, ToneMapper::Clamp] {
        let (w, hgt) = (320, 200);
        let mut h = common::Harness::new(RenderConfig { tone_mapper: mapper, ..no_taa(w, hgt) }, &scene);
        let rig = LightRig::builtin(BuiltinRig::NeutralStudio, Vec3::new(0.0, 1.6, -0.05), 0.12);
        let mut camera = camera_at(Vec3::new(0.0, 1.6, 1.4), Vec3::new(0.0, 1.45, 0.0));
        // Over-expose a stop so the shoulder of every mapper is exercised.
        camera.exposure = Exposure::ev100(rig.ev100 - 1.0);
        h.frames(&camera, &rig, 1);
        let hdr = h.resolved();
        let (_, _, display) = h.display();
        let exposure = camera.exposure.scale();
        let mut worst = 0_i32;
        let mut over_one = 0;
        for (i, px) in hdr.iter().enumerate() {
            let mapped = mapper.apply_f32([px[0], px[1], px[2]], exposure);
            for c in 0..3 {
                if px[c] * exposure > 1.0 {
                    over_one += 1;
                }
                let expected = (linear_to_srgb_f32(mapped[c].clamp(0.0, 1.0)) * 255.0).round() as i32;
                worst = worst.max((i32::from(display[i * 4 + c]) - expected).abs());
            }
        }
        println!("{}: worst difference {worst} code values, {over_one} channels above 1 after exposure", mapper.name());
        assert!(worst <= 1, "{} differs by {worst}", mapper.name());
        assert!(over_one > 0, "the test exercises the shoulder");
    }
}

fn quad(center: Vec3, half: f32) -> Mesh {
    // A square facing +Z.
    let p = [
        Vec3::new(-half, -half, 0.0),
        Vec3::new(half, -half, 0.0),
        Vec3::new(half, half, 0.0),
        Vec3::new(-half, half, 0.0),
    ];
    Mesh {
        positions: p.iter().map(|&v| v + center).collect(),
        normals: vec![Vec3::Z; 4],
        uvs: vec![Vec2::ZERO; 4],
        indices: vec![0, 1, 2, 0, 2, 3],
    }
}

/// `[AC-WP11-2]` `[RND-02]` reversed-Z with a 32-bit float buffer separates two walls 5 cm apart
/// at 1 km with a 1 cm near plane (a standard-Z float buffer cannot: depth differences there are
/// below its precision); the result is bit-identical across runs.
#[test]
fn ac_wp11_2_reversed_z_separates_5cm_at_1km() {
    let mut scene = Scene::default();
    scene.add(
        quad(Vec3::new(0.0, 0.0, -1000.0), 300.0),
        Material { base_color: [1.0, 0.0, 0.0], ..Material::default() },
    );
    scene.add(
        quad(Vec3::new(0.0, 0.0, -1000.05), 300.0),
        Material { base_color: [0.0, 0.0, 1.0], ..Material::default() },
    );
    let (w, hgt) = (256, 256);
    let config = RenderConfig { debug: DebugView::Albedo, ..no_taa(w, hgt) };
    let camera = Camera { position: Vec3::ZERO, target: -Vec3::Z, near: 0.01, focal_mm: 50.0, ..Camera::default() };
    let rig = empty_rig(ProceduralEnvironment::Uniform, 1.0);
    let mut runs = Vec::new();
    for _ in 0..2 {
        let mut h = common::Harness::new(config, &scene);
        h.frames(&camera, &rig, 1);
        let (_, _, rgba) = h.display();
        let depth = h.renderer.target_texture("depth").unwrap();
        let (_, _, depth_bytes) = h.read(depth);
        runs.push((rgba, depth_bytes));
    }
    let (rgba, depth) = &runs[0];
    assert_eq!(runs[0], runs[1], "deterministic");
    let blue = rgba.as_chunks::<4>().0.iter().filter(|p| p[2] > p[0]).count();
    let red = rgba.as_chunks::<4>().0.iter().filter(|p| p[0] > 128 && p[2] < 16).count();
    println!("reversed-Z: {red} red pixels, {blue} blue pixels of {}", w * hgt);
    assert_eq!(blue, 0, "the far wall shows through");
    assert_eq!(red, (w * hgt) as usize);
    let d = f32::from_le_bytes([depth[0], depth[1], depth[2], depth[3]]);
    // Infinite reversed-Z: depth = near / distance.
    assert!((d - 0.01 / 1000.0).abs() / (0.01 / 1000.0) < 0.01, "depth {d}");
    // In a standard-Z float buffer (near 0 -> 0, far 1000.05 -> 1) the two walls would map to
    // 1 - near/z values that differ by 5e-10, under half an ulp of 1.0 (6e-8): one value.
    let standard = |z: f64| (1.0 - 0.01 / z) as f32;
    assert_eq!(standard(1000.0), standard(1000.05));
    let reversed = |z: f64| (0.01 / z) as f32;
    assert_ne!(reversed(1000.0), reversed(1000.05));
    common::save_png("reversed_z_walls.png", w, hgt, rgba);
}

fn studio_scene_setup(w: u32, hgt: u32, taa: bool) -> (common::Harness, Camera, LightRig) {
    let scene = Scene::material_test();
    let h = common::Harness::new(RenderConfig { taa, ..RenderConfig::new(w, hgt) }, &scene);
    let rig = LightRig::builtin(BuiltinRig::NeutralStudio, Vec3::new(0.0, 1.6, -0.05), 0.12);
    let mut camera = camera_at(Vec3::new(0.25, 1.55, 1.1), Vec3::new(0.0, 1.45, 0.0));
    camera.exposure = Exposure::ev100(rig.ev100);
    (h, camera, rig)
}

/// `[AC-WP11-2]` TAA on a still camera: the same frame sequence gives bit-identical images in two
/// renderers, and after 64 frames the resolved image matches the 64-sample supersampled reference
/// (the exact mean of the same jitter positions) within a small tolerance.
#[test]
fn ac_wp11_2_taa_is_deterministic_and_converges_to_the_supersampled_image() {
    let (w, hgt) = (320, 240);
    let mut outputs = Vec::new();
    for _ in 0..2 {
        let (mut h, camera, rig) = studio_scene_setup(w, hgt, true);
        h.frames(&camera, &rig, 64);
        outputs.push((h.resolved(), h.display().2));
    }
    assert_eq!(outputs[0].1, outputs[1].1, "TAA is deterministic");
    // Reference: accumulate the 16-position jitter cycle (the TAA's sequence) as an exact mean.
    let (mut h, camera, rig) = studio_scene_setup(w, hgt, true);
    let rect = humanart_render::camera::TileRect { x: 0, y: 0, width: w, height: hgt };
    for i in 0..64_u32 {
        let jitter = humanart_math::taa_jitter_pixels(i % 16 + 1, 16);
        let still =
            humanart_render::renderer::StillFrame { full_width: w, full_height: hgt, tile: rect, jitter, sample: i };
        h.frame(&camera, &rig, &[], false, Some(still), i == 0);
    }
    let reference = h.resolved();
    let (_, _, reference_display) = h.display();
    let taa = &outputs[0].0;
    let exposure = camera.exposure.scale();
    let mut sum = 0.0_f64;
    let mut big = 0;
    for (a, b) in taa.iter().zip(&reference) {
        for c in 0..3 {
            let d = f64::from(((a[c] - b[c]) * exposure).abs());
            sum += d;
            if d > 0.05 {
                big += 1;
            }
        }
    }
    let mean = sum / (taa.len() * 3) as f64;
    println!("TAA vs supersampled: mean |difference| {mean:.5} (exposed), {big} channels above 0.05");
    assert!(mean < 0.004, "mean difference {mean}");
    assert!(big < taa.len() / 100, "{big} large differences");
    common::save_png("taa_converged.png", w, hgt, &outputs[0].1);
    common::save_png("taa_reference_supersampled.png", w, hgt, &reference_display);
}

/// `[AC-WP11-2]` `[RND-04]` motion vectors under deformation: the vertices of a sphere are moved
/// through the position stream (what the GPU deformation writes) and an object is moved by its
/// transform; the motion target holds the analytic screen displacement, and a mesh that stopped
/// moving has zero motion on the next frame.
#[test]
fn ac_wp11_2_motion_vectors_follow_deformation() {
    let mut scene = Scene::default();
    let sphere = Mesh::sphere(Vec3::new(-0.3, 0.0, 0.0), 0.2, 48, 32);
    let original = sphere.positions.clone();
    scene.add(sphere, Material::default());
    let moved_object = scene.add(Mesh::sphere(Vec3::new(0.35, 0.0, 0.0), 0.15, 48, 32), Material::default());
    let (w, hgt) = (400, 300);
    let mut h = common::Harness::new(no_taa(w, hgt), &scene);
    let camera = camera_at(Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO);
    let rig = empty_rig(ProceduralEnvironment::Uniform, 1.0);
    h.frames(&camera, &rig, 1);
    // Deform: shift every vertex by d (a rigid "expression" keeps the test analytic).
    let d = Vec3::new(0.02, 0.03, 0.0);
    let shifted: Vec<Vec3> = original.iter().map(|&p| p + d).collect();
    h.scene.set_positions(&mut h.resources, 0, &shifted).unwrap();
    let t = Vec3::new(-0.01, 0.025, 0.0);
    h.scene.set_transform(moved_object, Mat4::from_translation(t)).unwrap();
    h.frame(&camera, &rig, &[], false, None, false);
    let motion = |h: &mut common::Harness| {
        let surface = h.renderer.target_texture("surface").unwrap();
        let (_, _, bytes) = h.read(surface);
        bytes
            .as_chunks::<16>()
            .0
            .iter()
            .map(|t| {
                let wv = u32::from_le_bytes([t[12], t[13], t[14], t[15]]);
                let lo = humanart_math::f16_bits_to_f32(wv as u16);
                let hi = humanart_math::f16_bits_to_f32((wv >> 16) as u16);
                Vec2::new(lo, hi)
            })
            .collect::<Vec<_>>()
    };
    let m = motion(&mut h);
    let expected = |from: Vec3, to: Vec3| {
        let (a, b) = (pixel_of_f(&camera, w, hgt, from), pixel_of_f(&camera, w, hgt, to));
        b - a
    };
    let probe_deformed = Vec3::new(-0.3, 0.0, 0.2) + d;
    let probe_moved = Vec3::new(0.35, 0.0, 0.15) + t;
    let got_deformed = m[index(&camera, w, hgt, probe_deformed)];
    let got_moved = m[index(&camera, w, hgt, probe_moved)];
    let want_deformed = expected(probe_deformed - d, probe_deformed);
    let want_moved = expected(probe_moved - t, probe_moved);
    println!(
        "motion: deformed {got_deformed:?} expected {want_deformed:?}; moved {got_moved:?} expected {want_moved:?}"
    );
    assert!((got_deformed - want_deformed).length() < 0.05, "deformation motion");
    assert!((got_moved - want_moved).length() < 0.05, "transform motion");
    assert!(want_deformed.length() > 5.0, "a real displacement is tested");
    // Next frame without changes: zero motion everywhere.
    h.frame(&camera, &rig, &[], false, None, false);
    let m = motion(&mut h);
    let worst = m.iter().map(|v| v.length()).fold(0.0_f32, f32::max);
    assert!(worst < 1e-3, "stopped meshes have no motion: {worst}");
}

fn pixel_of_f(camera: &Camera, w: u32, h: u32, p: Vec3) -> Vec2 {
    let m = ViewMatrices::new(camera, w, h, Vec2::ZERO, None).view_projection;
    ndc_to_screen(m.project_point3(p).xy(), Vec2::new(w as f32, h as f32))
}

fn index(camera: &Camera, w: u32, h: u32, p: Vec3) -> usize {
    let (x, y) = pixel_of(camera, w, h, p);
    y * w as usize + x
}

/// Picking: the id under a pixel is the object's pick id, read back without blocking the frame.
#[test]
fn picking_reports_the_object_under_the_cursor() {
    let scene = Scene::material_test();
    let (w, hgt) = (320, 240);
    let mut h = common::Harness::new(RenderConfig { picking: true, ..RenderConfig::new(w, hgt) }, &scene);
    let rig = LightRig::builtin(BuiltinRig::NeutralStudio, Vec3::new(0.0, 1.6, -0.05), 0.12);
    let camera = camera_at(Vec3::new(0.0, 1.5, 1.6), Vec3::new(0.0, 1.45, 0.0));
    let head = scene.objects.len() - 1;
    let probes = [
        (pixel_of(&camera, w, hgt, Vec3::new(0.0, 1.62, 0.04)), scene.objects[head].pick_id),
        (pixel_of(&camera, w, hgt, Vec3::new(-0.24, 1.28, 0.3)), scene.objects[1].pick_id),
        ((2, 2), 0),
    ];
    for ((x, y), want) in probes {
        let mut encoder = h.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        h.renderer
            .render(
                &mut h.resources,
                &mut encoder,
                FrameInput {
                    camera: &camera,
                    rig: &rig,
                    scene: &mut h.scene,
                    overlay: &[],
                    wireframe: false,
                    still: None,
                    pick: Some([x as u32, y as u32]),
                    reset_history: false,
                },
            )
            .unwrap();
        h.gpu.queue.submit([encoder.finish()]);
        h.renderer.after_submit();
        let mut hit = None;
        for _ in 0..100 {
            common::wait(&h.gpu.device);
            hit = h.renderer.poll_pick(&h.gpu.device).unwrap();
            if hit.is_some() {
                break;
            }
        }
        let hit = hit.expect("the readback arrives");
        assert_eq!(hit.pixel, [x as u32, y as u32]);
        assert_eq!(hit.pick_id, want, "pixel ({x}, {y})");
        assert_eq!(hit.depth > 0.0, want != 0);
    }
}
