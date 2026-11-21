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
use humanart_render::light::{Background, Light, LightColor, LightRig, ProceduralEnvironment};
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
