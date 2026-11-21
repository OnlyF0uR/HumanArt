//! `[AC-WP11-1]` the generated template renders with correct IBL and shadows. The mesh is the
//! skin that ships: the `body` part of LOD0 of `generate::build::template` with the default
//! options (the scanned face, the finished cranium, ears and lips; todo 1.1 and 1.3), not the
//! scan-fitting base `generate::body_surface`. The checks: in the white furnace every visible point of the head has diffuse
//! lighting 1, and under a directional key light the points a CPU ray cast finds occluded are
//! dark while the others match the unshadowed render (points within about 6 mm of a shadow
//! boundary, where the shadow map's resolution and filter decide, are left out). Review shots of the head and the body are
//! written to target/agents/render/shots/.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

mod common;

use humanart_math::{Vec2, Vec3, ndc_to_screen};
use humanart_model::generate::anatomy::HEAD_HEIGHT;
use humanart_model::generate::build::{Options, template as build_template};
use humanart_render::camera::{Camera, Exposure, ViewMatrices};
use humanart_render::capture;
use humanart_render::ibl::EnvironmentSource;
use humanart_render::light::{Background, BuiltinRig, Light, LightRig, ProceduralEnvironment};
use humanart_render::renderer::RenderConfig;
use humanart_render::scene::{Material, Mesh, Scene};

/// The shipped template's skin as a render mesh (the `body` part of LOD0 of
/// `generate::build::template(&Options::default())`; the LODs are not built, they do not change
/// LOD0), only the vertices its triangles use, with its head centre and its head top.
fn template() -> (Mesh, Vec3, f32) {
    let start = std::time::Instant::now();
    let generated = build_template(&Options { lods: false, ..Options::default() }).unwrap();
    let skin = Mesh::from_template(generated.template.mesh(), Some("body")).unwrap();
    // The template mesh keeps every vertex of every part: drop those the skin does not use.
    let mut map = vec![u32::MAX; skin.positions.len()];
    let mut mesh = Mesh::default();
    for &i in &skin.indices {
        let i = i as usize;
        if map[i] == u32::MAX {
            map[i] = mesh.positions.len() as u32;
            mesh.positions.push(skin.positions[i]);
            mesh.normals.push(skin.normals[i]);
            mesh.uvs.push(skin.uvs[i]);
        }
        mesh.indices.push(map[i]);
    }
    let top = mesh.positions.iter().map(|p| p.y).fold(f32::MIN, f32::max);
    let head: Vec<Vec3> = mesh.positions.iter().copied().filter(|p| p.y > top - HEAD_HEIGHT).collect();
    let (lo, hi) =
        head.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), &p| (lo.min(p), hi.max(p)));
    println!(
        "template: {} vertices, {} triangles, generated in {:.2} s; head box {lo:?}..{hi:?}",
        mesh.positions.len(),
        mesh.indices.len() / 3,
        start.elapsed().as_secs_f64()
    );
    (mesh, (lo + hi) * 0.5, top)
}

fn skin() -> Material {
    Material { base_color: [0.56, 0.38, 0.30], roughness: 0.5, f0: 0.028, ..Material::default() }
}

fn white() -> Material {
    Material { base_color: [1.0; 3], roughness: 1.0, f0: 0.0, ..Material::default() }
}

fn rig_without_lights(environment: ProceduralEnvironment, nits: f32) -> LightRig {
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

fn pixel_of(camera: &Camera, w: u32, h: u32, p: Vec3) -> (u32, u32) {
    let m = ViewMatrices::new(camera, w, h, Vec2::ZERO, None).view_projection;
    let s = ndc_to_screen(m.project_point3(p).xy(), Vec2::new(w as f32, h as f32));
    (s.x as u32, s.y as u32)
}

fn target(h: &mut common::Harness, name: &str) -> Vec<[f32; 4]> {
    let t = h.renderer.target_texture(name).unwrap();
    let (_, _, bytes) = h.read(t);
    capture::rgba16f_to_f32(&bytes)
}

/// Möller-Trumbore: does the ray from `origin` along unit `dir` hit a triangle before `max_t`?
fn occluded(triangles: &[[Vec3; 3]], origin: Vec3, dir: Vec3, max_t: f32) -> bool {
    triangles.iter().any(|&[a, b, c]| {
        let (e1, e2) = (b - a, c - a);
        let p = dir.cross(e2);
        let det = e1.dot(p);
        if det.abs() < 1e-12 {
            return false;
        }
        let inv = 1.0 / det;
        let s = origin - a;
        let u = s.dot(p) * inv;
        if !(0.0..=1.0).contains(&u) {
            return false;
        }
        let q = s.cross(e1);
        let v = dir.dot(q) * inv;
        if v < 0.0 || u + v > 1.0 {
            return false;
        }
        let t = e2.dot(q) * inv;
        t > 1e-5 && t < max_t
    })
}

/// A head point the test samples: where it lands, and what the CPU ray cast says.
struct Sample {
    point: Vec3,
    pixel: (u32, u32),
    shadowed: bool,
}

#[test]
fn ac_wp11_1_template_head_ibl_and_shadows() {
    let (mesh, head, top) = template();
    let (w, hgt) = (720, 720);
    // The head and the neck down to the shoulders.
    let aim = Vec3::new(head.x, top - 0.16, head.z);
    let camera = Camera {
        position: aim + Vec3::new(0.0, 0.0, 1.0),
        target: aim,
        focal_mm: 70.0,
        near: 0.05,
        exposure: Exposure::ev100(0.0),
        ..Camera::default()
    };
    let mut scene = Scene::default();
    scene.add(mesh.clone(), white());
    let config = RenderConfig { taa: false, ..RenderConfig::new(w, hgt) };
    let mut h = common::Harness::new(config, &scene);

    // The white furnace: diffuse lighting 1 everywhere, whatever the shape.
    h.renderer
        .set_environment(&mut h.resources, EnvironmentSource::Procedural(ProceduralEnvironment::Uniform))
        .unwrap();
    h.frames(&camera, &rig_without_lights(ProceduralEnvironment::Uniform, 1.0), 1);
    let furnace = target(&mut h, "diffuse");

    // A directional key from above and in front, travelling down and backwards.
    let direction = Vec3::new(0.15, -0.8, -0.55).normalize();
    let to_light = -direction;
    let mut rig = rig_without_lights(ProceduralEnvironment::Uniform, 0.0);
    rig.lights.push(Light::directional(direction, 1000.0, 6500.0));
    h.renderer
        .set_environment(&mut h.resources, EnvironmentSource::Procedural(ProceduralEnvironment::Uniform))
        .unwrap();
    h.frames(&camera, &rig, 1);
    let shadowed_render = target(&mut h, "diffuse");
    rig.lights[0].shadow.enabled = false;
    h.frames(&camera, &rig, 1);
    let unshadowed_render = target(&mut h, "diffuse");

    // Samples: head and neck vertices facing the camera and the light, visible, whose shadow
    // rays (from the point and from four points 1 cm around it) all agree.
    let bottom = top - HEAD_HEIGHT - 0.1;
    let triangles: Vec<[Vec3; 3]> = mesh
        .indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|t| t.map(|i| mesh.positions[i as usize]))
        .filter(|t| t.iter().any(|p| p.y > bottom - 0.1))
        .collect();
    let (side, up) = {
        let s = to_light.cross(Vec3::Y).normalize();
        (s, s.cross(to_light).normalize())
    };
    let mut samples = Vec::new();
    for (i, (&p, &n)) in mesh.positions.iter().zip(&mesh.normals).enumerate() {
        if i % 3 != 0 || p.y < bottom {
            continue;
        }
        let to_camera = (camera.position - p).normalize();
        if n.dot(to_camera) < 0.4 || n.dot(to_light) < 0.3 {
            continue;
        }
        let start = p + n * 1e-3;
        if occluded(&triangles, start, to_camera, (camera.position - start).length()) {
            continue;
        }
        // The neighbours lie in the tangent plane, lifted 2 mm off the surface (the sagitta of
        // 6 mm on the head's curvature is well below that).
        let mut votes = [false; 5];
        for (k, offset) in [Vec3::ZERO, side, -side, up, -up].into_iter().enumerate() {
            let tangent = offset - n * offset.dot(n);
            let origin = if k == 0 { start } else { p + tangent * 0.006 + n * 2e-3 };
            votes[k] = occluded(&triangles, origin, to_light, 10.0);
        }
        if votes.iter().any(|&v| v != votes[0]) {
            continue;
        }
        let pixel = pixel_of(&camera, w, hgt, p);
        if pixel.0 < w && pixel.1 < hgt {
            samples.push(Sample { point: p, pixel, shadowed: votes[0] });
        }
    }
    let index = |(x, y): (u32, u32)| (y * w + x) as usize;
    let mut worst_furnace: f32 = 0.0;
    let (mut shadowed, mut lit, mut wrong) = (0, 0, 0);
    let mut lit_ratio_worst: f32 = 1.0;
    for s in &samples {
        let i = index(s.pixel);
        worst_furnace = worst_furnace.max((furnace[i][1] - 1.0).abs());
        let on = shadowed_render[i][1];
        let off = unshadowed_render[i][1];
        let ratio = on / off.max(1e-6);
        if s.shadowed {
            shadowed += 1;
        } else {
            lit += 1;
            lit_ratio_worst = lit_ratio_worst.min(ratio);
        }
        if (s.shadowed && ratio > 0.1) || (!s.shadowed && ratio < 0.9) {
            wrong += 1;
            println!(
                "  disagreement at {:?} (pixel {:?}): ray cast {}, on/off {ratio:.3}",
                s.point,
                s.pixel,
                if s.shadowed { "occluded" } else { "lit" }
            );
        }
    }
    println!(
        "{} samples: furnace worst |diffuse - 1| {worst_furnace:.4}; shadow rays: {shadowed} occluded, {lit} lit, {wrong} disagree with the render; worst lit on/off ratio {lit_ratio_worst:.3}",
        samples.len()
    );
    assert!(samples.len() > 200 && shadowed >= 20 && lit >= 100, "the sample covers lit and shadowed points");
    assert!(worst_furnace < 0.015, "furnace error {worst_furnace}");
    assert!(wrong * 50 <= samples.len(), "{wrong} of {} samples disagree with the CPU ray cast", samples.len());
    // The shots at an exposure that suits 1000 lx on white (the measurements above are taken
    // before exposure).
    let shot_camera = Camera { exposure: Exposure::ev100(9.0), ..camera };
    h.frames(&shot_camera, &rig, 1);
    common::save_png("template_head_unshadowed_key.png", w, hgt, &h.display().2);
    rig.lights[0].shadow.enabled = true;
    h.frames(&shot_camera, &rig, 1);
    common::save_png("template_head_shadowed_key.png", w, hgt, &h.display().2);
}

/// Review shots of the template in the neutral studio (TAA converged): the face from the front
/// and three-quarters, and the whole body.
#[test]
fn template_review_shots() {
    let (mesh, head, _) = template();
    let mut scene = Scene::default();
    let feet = mesh.positions.iter().map(|p| p.y).fold(f32::MAX, f32::min);
    scene.add(
        Mesh::ground(Vec3::new(0.0, feet, 0.0), 4.0, 8),
        Material { base_color: [0.3; 3], roughness: 0.8, ..Material::default() },
    );
    scene.add(mesh, skin());
    let rig = LightRig::builtin(BuiltinRig::NeutralStudio, head, 0.13);
    let shots = [
        (
            "template_face_front.png",
            head + Vec3::new(0.0, -0.02, 1.15),
            head - Vec3::new(0.0, 0.02, 0.0),
            85.0,
            (720, 900),
        ),
        ("template_face_34.png", head + Vec3::new(0.7, 0.05, 0.88), head - Vec3::new(0.0, 0.02, 0.0), 85.0, (720, 900)),
        ("template_body_front.png", Vec3::new(0.0, 1.0, 4.2), Vec3::new(0.0, 0.88, 0.0), 50.0, (720, 1080)),
    ];
    for (name, position, target, focal_mm, (w, hgt)) in shots {
        let mut h = common::Harness::new(RenderConfig::new(w, hgt), &scene);
        let camera = Camera {
            position,
            target,
            focal_mm,
            near: 0.05,
            exposure: Exposure::ev100(rig.ev100),
            ..Camera::default()
        };
        h.frames(&camera, &rig, 16);
        let (_, _, rgba) = h.save_display(name);
        let mean = rgba.iter().map(|&v| f64::from(v)).sum::<f64>() / rgba.len() as f64;
        assert!(mean > 20.0 && mean < 235.0, "{name} is not blank: {mean}");
    }
}
