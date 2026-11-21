//! todo 1.9 (light and camera) on the real GPU (headless): what each new pass claims.
//!
//! - shadows: a penumbra that grows with the distance from the occluder, no acne on lit surfaces,
//!   a shadow that starts at the occluder's foot, a map narrowed to what the camera sees that
//!   still holds the casters outside the view;
//! - the occlusion pass: creases close, flat and convex surfaces stay exactly open, nothing
//!   darkens around a silhouette; a contact shadow per shadowed light;
//! - the depth of field: sharp at the focus distance, blurred behind it, no halo of the sharp
//!   subject in the blurred background;
//! - the studio paper backdrop: lit by the rig as a matte surface, the subject's shadow on the
//!   floor, no edge between floor and wall; the flat backdrop is still there;
//! - the generated environments (sunset, the studio room without panels);
//! - the cost of the new passes at 1440p (printed, not asserted: the GPU is shared).
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use std::collections::BTreeMap;

use humanart_math::{Vec2, Vec3, ndc_to_screen};
use humanart_render::camera::{Camera, Exposure, Lens, ViewMatrices};
use humanart_render::capture;
use humanart_render::ibl::EnvironmentSource;
use humanart_render::light::{
    Backdrop, Background, BuiltinRig, Light, LightColor, LightKind, LightRig, ProceduralEnvironment,
};
use humanart_render::renderer::{Quality, RenderConfig};
use humanart_render::scene::{Material, Mesh, Scene};

fn rig(lights: Vec<Light>, environment: ProceduralEnvironment, nits: f32, background: Background) -> LightRig {
    LightRig {
        name: "test",
        lights,
        environment,
        environment_nits: nits,
        environment_rotation: 0.0,
        background,
        ev100: 0.0,
    }
}

/// A white directional light of `lux` travelling along `direction` with a source `diameter` rad wide.
fn sun(direction: Vec3, lux: f32, diameter: f32) -> Light {
    let mut light = Light::directional(direction, lux, 5600.0);
    light.kind = LightKind::Directional { angular_diameter: diameter };
    light.color = LightColor::Linear([1.0; 3]);
    light
}

fn camera(position: Vec3, target: Vec3, focal_mm: f32) -> Camera {
    Camera { position, target, focal_mm, near: 0.02, exposure: Exposure::ev100(0.0), ..Camera::default() }
}

fn matte(value: f32) -> Material {
    Material { base_color: [value; 3], roughness: 1.0, metallic: 0.0, ..Material::default() }
}

/// The pixel (column, row) a world point lands on.
fn pixel_of(camera: &Camera, w: u32, h: u32, p: Vec3) -> (usize, usize) {
    let m = ViewMatrices::new(camera, w, h, Vec2::ZERO, None).view_projection;
    let s = ndc_to_screen(m.project_point3(p).xy(), Vec2::new(w as f32, h as f32));
    assert!(s.x >= 0.0 && s.y >= 0.0 && s.x < w as f32 && s.y < h as f32, "{p:?} is outside the image: {s:?}");
    (s.x as usize, s.y as usize)
}

fn luminance(c: [f32; 4]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

fn hdr_target(h: &mut common::Harness, name: &str) -> Vec<[f32; 4]> {
    let t = h.renderer.target_texture(name).unwrap();
    let (_, _, bytes) = h.read(t);
    capture::rgba16f_to_f32(&bytes)
}

fn byte_target(h: &mut common::Harness, name: &str) -> Vec<u8> {
    let t = h.renderer.target_texture(name).unwrap();
    h.read(t).2
}

/// A ball resting on a ground that does not cast, and a small ball floating 7 cm above it.
fn crease_scene() -> Scene {
    let mut scene = Scene::default();
    let ground = scene.add(Mesh::ground(Vec3::ZERO, 0.4, 8), matte(0.8));
    scene.objects[ground].casts_shadow = false;
    scene.add(Mesh::sphere(Vec3::new(0.0, 0.03, 0.0), 0.03, 96, 48), matte(0.8));
    scene.add(Mesh::sphere(Vec3::new(-0.03, 0.07, 0.05), 0.008, 48, 24), matte(0.8));
    scene
}

const SIDE: u32 = 256;

/// todo 1.9 item 2: the ambient term closes in the crease where the ball meets the ground and
/// stays exactly open on the flat ground, on the convex ball and on the ground seen right next to
/// the floating ball's outline (no halo); the geometry pass darkens the environment's light by it
/// and leaves open surfaces as they were. Still tier: the occlusion targets have the image's
/// resolution (the half-resolution tiers: the next test).
#[test]
fn todo_1_9_ambient_occlusion_closes_creases_and_nothing_else() {
    let scene = crease_scene();
    let cam = camera(Vec3::new(0.0, 0.15, 0.3), Vec3::new(0.0, 0.02, 0.0), 50.0);
    let furnace = rig(Vec::new(), ProceduralEnvironment::Uniform, 1.0, Background::Gray(0.18));
    let off = RenderConfig { taa: false, quality: Quality::Still, ..RenderConfig::new(SIDE, SIDE) };
    let on = RenderConfig { occlusion: true, ..off };
    let at = |p: Vec3| {
        let (x, y) = pixel_of(&cam, SIDE, SIDE, p);
        y * SIDE as usize + x
    };
    let crease = at(Vec3::new(-0.01, 0.0, 0.01));
    // The mean over the 5 x 5 pixels around `i`: one frame has one slice rotation per pixel.
    let around = |values: &dyn Fn(usize) -> f32, i: usize| {
        let mut sum = 0.0;
        for dy in 0..5 {
            for dx in 0..5 {
                sum += values(i + dx + dy * SIDE as usize - 2 - 2 * SIDE as usize);
            }
        }
        sum / 25.0
    };
    let open_ground = at(Vec3::new(0.035, 0.0, 0.06));
    let ball_top = at(Vec3::new(0.0, 0.06, 0.0));
    // The ground seen 4 mm outside the floating ball's outline: 7 cm under it.
    let beside_floating = at(Vec3::new(-0.042, 0.07, 0.05));

    let mut plain = common::Harness::new(off, &scene);
    assert!(plain.renderer.target_texture("ambient").is_none(), "no occlusion pass unless asked for");
    plain.frames(&cam, &furnace, 1);
    let before = hdr_target(&mut plain, "diffuse");

    let mut h = common::Harness::new(on, &scene);
    assert!(h.renderer.graph().unwrap().is_live("prepass") && h.renderer.graph().unwrap().is_live("occlusion"));
    h.frames(&cam, &furnace, 1);
    let ambient = byte_target(&mut h, "ambient");
    assert_eq!(ambient.len(), (SIDE * SIDE) as usize, "one byte per pixel");
    let after = hdr_target(&mut h, "diffuse");
    println!(
        "ambient: crease {}, open ground {}, ball top {}, beside the floating ball {}",
        ambient[crease], ambient[open_ground], ambient[ball_top], ambient[beside_floating]
    );
    let closed = around(&|i| f32::from(ambient[i]), crease);
    println!("ambient around the crease: {closed:.1} of 255");
    // The crease's mouth, 3 mm from the ball's outline: the depth buffer holds the ball's front, not
    // its underside, so the horizon there reaches about 50 degrees on one side of four.
    assert!(closed < 235.0, "the crease under the ball is closed: {closed}");
    for (name, i) in
        [("open ground", open_ground), ("ball top", ball_top), ("beside the floating ball", beside_floating)]
    {
        assert_eq!(ambient[i], 255, "{name} is exactly open");
        assert!((after[i][1] - before[i][1]).abs() <= 1e-3 * before[i][1], "{name} is lit as without the pass");
    }
    let ratio = around(&|i| after[i][1], crease) / around(&|i| before[i][1], crease);
    assert!(ratio < 0.9 && ratio > 0.05, "the environment's light in the crease falls with it: {ratio}");
    assert!((ratio - closed / 255.0).abs() < 0.02, "by the ambient term: {ratio} against {closed}");
    // Open over the whole image except near the two contacts: most pixels are exactly 255.
    #[allow(clippy::naive_bytecount)] // No dependencies [DEP-05]: the bytecount crate is not one.
    let open = ambient.iter().filter(|&&a| a == 255).count() as f32 / ambient.len() as f32;
    assert!(open > 0.6, "{open} of the image is exactly open");

    // A light without a shadow map (a fill) has nothing else that closes the crease: the ambient
    // term darkens it too; a light with a shadow map is left to its map and contact shadow.
    let mut fill = sun(Vec3::new(0.3, -1.0, -0.4), 1000.0, 0.02);
    fill.shadow.enabled = false;
    let filled = rig(vec![fill], ProceduralEnvironment::Uniform, 0.0, Background::Gray(0.18));
    plain.frames(&cam, &filled, 1);
    let before = hdr_target(&mut plain, "diffuse");
    h.frames(&cam, &filled, 1);
    let ambient = byte_target(&mut h, "ambient");
    let after = hdr_target(&mut h, "diffuse");
    let closed = around(&|i| f32::from(ambient[i]), crease);
    let ratio = around(&|i| after[i][1], crease) / around(&|i| before[i][1], crease);
    println!("unshadowed light in the crease: {ratio:.3} of the light without the pass (ambient {closed:.1})");
    assert!(ratio < 0.97 && (ratio - closed / 255.0).abs() < 0.03, "the fill in the crease: {ratio}");
    for i in [open_ground, ball_top] {
        assert!((after[i][1] - before[i][1]).abs() <= 1e-3 * before[i][1], "open surfaces keep the fill");
    }
}

/// todo 1.9 item 2: the contact shadow of a shadowed light closes next to the ball's foot on the
/// side away from the light, in that light's layer, and is open on open ground and for a light
/// without a shadow map (Still tier: full-resolution targets).
#[test]
fn todo_1_9_contact_shadows_close_at_the_foot_of_an_occluder() {
    let scene = crease_scene();
    let cam = camera(Vec3::new(0.0, 0.15, 0.3), Vec3::new(0.0, 0.02, 0.0), 50.0);
    let mut unshadowed = sun(Vec3::new(1.0, -1.0, 0.0), 1000.0, 0.02);
    unshadowed.shadow.enabled = false;
    let lights = vec![unshadowed, sun(Vec3::new(-1.0, -1.0, 0.0), 1000.0, 0.02)];
    let lit = rig(lights, ProceduralEnvironment::Uniform, 0.0, Background::Gray(0.18));
    let config = RenderConfig { taa: false, occlusion: true, quality: Quality::Still, ..RenderConfig::new(SIDE, SIDE) };
    let mut h = common::Harness::new(config, &scene);
    h.frames(&cam, &lit, 1);
    let contact = byte_target(&mut h, "contact");
    assert_eq!(contact.len(), (SIDE * SIDE * 4) as usize);
    let at = |p: Vec3| {
        let (x, y) = pixel_of(&cam, SIDE, SIDE, p);
        4 * (y * SIDE as usize + x)
    };
    // The second light (the first with a shadow map) is layer 0: the red channel.
    let foot = at(Vec3::new(-0.012, 0.0, 0.012));
    let open = at(Vec3::new(0.035, 0.0, 0.06));
    println!("contact: foot {:?}, open {:?}", &contact[foot..foot + 4], &contact[open..open + 4]);
    assert!(contact[foot] < 100, "the ground at the ball's foot, away from the light: {}", contact[foot]);
    assert_eq!(&contact[open..open + 4], &[255; 4], "open ground");
    assert_eq!(&contact[foot + 1..foot + 4], &[255; 3], "the layers of no light stay open");
}

/// todo 1.9 / todo 5 (viewport budget): below the Still tier the occlusion targets have half the
/// resolution and each texel takes one pixel of its 2 x 2 block per frame, in turn. Over four frames
/// the crease closes as much as at full resolution, open surfaces stay exactly open, and the
/// geometry pass reads the right texel (the environment's light in the crease falls, open ground
/// keeps it).
#[test]
fn todo_1_9_half_resolution_occlusion_matches_the_full_one() {
    let scene = crease_scene();
    let cam = camera(Vec3::new(0.0, 0.15, 0.3), Vec3::new(0.0, 0.02, 0.0), 50.0);
    let furnace = rig(Vec::new(), ProceduralEnvironment::Uniform, 1.0, Background::Gray(0.18));
    let pixel = |p: Vec3| pixel_of(&cam, SIDE, SIDE, p);
    let (crease, open_ground, ball_top) =
        (pixel(Vec3::new(-0.01, 0.0, 0.01)), pixel(Vec3::new(0.035, 0.0, 0.06)), pixel(Vec3::new(0.0, 0.06, 0.0)));
    // The mean of the ambient term over the 6 x 6 pixels around a pixel and four frames.
    let mut means = Vec::new();
    for quality in [Quality::Still, Quality::Review] {
        let scale = quality.occlusion_scale() as usize;
        let side = SIDE as usize / scale;
        let config = RenderConfig { taa: false, occlusion: true, quality, ..RenderConfig::new(SIDE, SIDE) };
        let mut h = common::Harness::new(config, &scene);
        let mut sum = 0.0;
        for _ in 0..4 {
            h.frames(&cam, &furnace, 1);
            let ambient = byte_target(&mut h, "ambient");
            assert_eq!(ambient.len(), side * side, "{quality:?}: 1 / {scale} of the resolution");
            let at = |(x, y): (usize, usize)| usize::from(ambient[(y / scale) * side + x / scale]);
            for (name, p) in [("open ground", open_ground), ("ball top", ball_top)] {
                assert_eq!(at(p), 255, "{quality:?}: {name} is exactly open");
            }
            for dy in 0..6 {
                for dx in 0..6 {
                    sum += at((crease.0 + dx - 3, crease.1 + dy - 3)) as f32;
                }
            }
        }
        means.push(sum / 144.0);
        if quality == Quality::Review {
            let mut plain = common::Harness::new(RenderConfig { occlusion: false, ..config }, &scene);
            plain.frames(&cam, &furnace, 1);
            let before = hdr_target(&mut plain, "diffuse");
            let after = hdr_target(&mut h, "diffuse");
            let i = |(x, y): (usize, usize)| y * SIDE as usize + x;
            let around = |image: &[[f32; 4]]| {
                let mut s = 0.0;
                for dy in 0..6 {
                    for dx in 0..6 {
                        s += image[i((crease.0 + dx - 3, crease.1 + dy - 3))][1];
                    }
                }
                s
            };
            let ratio = around(&after) / around(&before);
            let open = after[i(open_ground)][1] / before[i(open_ground)][1];
            println!(
                "Review: the environment's light in the crease {ratio:.3} of the light without the pass, open {open:.4}"
            );
            assert!(ratio < 0.95 && (open - 1.0).abs() < 1e-3, "the geometry pass reads the half-resolution texel");
        }
    }
    println!(
        "ambient around the crease over four frames: full resolution {:.1}, half {:.1} of 255",
        means[0], means[1]
    );
    assert!(means[0] < 235.0 && means[1] < 235.0, "the crease closes at both resolutions: {means:?}");
    assert!((means[0] - means[1]).abs() < 8.0, "and by as much: {means:?}");
}

/// todo 1.9 item 1: a pole on a ground under a light 0.06 rad wide. The shadow's edge is a few
/// millimetres wide 5 cm from the pole's foot and several times wider 45 cm from it; it starts at
/// the foot; the lit ground is evenly lit and a lit ball follows the cosine (no acne).
#[test]
fn todo_1_9_soft_shadow_penumbra_grows_with_distance() {
    const W: u32 = 512;
    let mut scene = Scene::default();
    let ground = scene.add(Mesh::ground(Vec3::new(-0.25, 0.0, 0.0), 1.2, 8), matte(0.8));
    scene.objects[ground].casts_shadow = false;
    scene.add(Mesh::cylinder(Vec3::ZERO, 0.02, 0.5, 48), matte(0.8));
    let ball = Vec3::new(-0.1, 0.06, 0.25);
    scene.add(Mesh::sphere(ball, 0.06, 96, 48), matte(0.8));
    let direction = Vec3::new(-1.0, -1.0, 0.0).normalize_or(-Vec3::Y);
    let lit = rig(vec![sun(direction, 1000.0, 0.06)], ProceduralEnvironment::Uniform, 0.0, Background::Gray(0.18));
    let mut cam = camera(Vec3::new(-0.25, 1.5, 0.0), Vec3::new(-0.25, 0.0, 0.0), 50.0);
    cam.up = -Vec3::Z;
    let mut h = common::Harness::new(RenderConfig::new(W, W), &scene);
    h.frames(&cam, &lit, 32);
    let image = h.resolved();
    let value = |p: Vec3| {
        let (x, y) = pixel_of(&cam, W, W, p);
        luminance(image[y * W as usize + x])
    };
    let lit_ground = value(Vec3::new(-0.05, 0.0, -0.14));
    assert!(lit_ground > 1.0, "the lit ground: {lit_ground}");
    // No acne: the lit ground is even.
    for p in [Vec3::new(-0.45, 0.0, -0.14), Vec3::new(0.05, 0.0, 0.1), Vec3::new(-0.3, 0.0, 0.2)] {
        let v = value(p);
        assert!((v / lit_ground - 1.0).abs() < 0.02, "lit ground at {p:?}: {v} of {lit_ground}");
    }
    // No acne on a curved surface: the ball follows the cosine where it faces the light.
    let to_light = -direction;
    let facing = value(ball + to_light * 0.06) / lit_ground * to_light.y;
    let n = (to_light + Vec3::Z * 1.2).normalize_or(Vec3::Y);
    let tilted = value(ball + n * 0.06) / lit_ground * to_light.y;
    println!("ball: facing the light {facing:.3} (1), tilted {tilted:.3} ({:.3})", n.dot(to_light));
    assert!((facing - 1.0).abs() < 0.08 && (tilted - n.dot(to_light)).abs() < 0.08);
    // The shadow starts at the foot: 1 cm behind the pole it is dark.
    let foot = value(Vec3::new(-0.03, 0.0, 0.0)) / lit_ground;
    assert!(foot < 0.15, "the ground right behind the pole's foot: {foot}");
    // The width of the edge (10 % to 90 % of the light) across the shadow, metres.
    let width = |x: f32| {
        let (mut low, mut high) = (f32::NAN, f32::NAN);
        for i in 0..150 {
            let z = i as f32 * 0.001;
            let v = value(Vec3::new(x, 0.0, z)) / lit_ground;
            if low.is_nan() && v > 0.1 {
                low = z;
            }
            if high.is_nan() && v > 0.9 {
                high = z;
            }
        }
        assert!(low.is_finite() && high.is_finite(), "the shadow's edge at x = {x} was not found");
        high - low
    };
    let (near, far) = (width(-0.05), width(-0.45));
    println!("shadow edge: {:.1} mm at 5 cm, {:.1} mm at 45 cm (0.06 rad: 4 and 38 mm)", near * 1e3, far * 1e3);
    assert!(near < 0.012, "close to the pole the edge is narrow: {near}");
    assert!(far > 2.5 * near && far > 0.015 && far < 0.08, "far from it the edge is wide: {far}");
}

/// todo 1.9 item 1: with every object a caster the map is narrowed to what the camera sees, and a
/// caster outside the view still shadows what is inside it.
#[test]
fn todo_1_9_focused_shadow_map_keeps_casters_outside_the_view() {
    let mut scene = Scene::default();
    scene.add(Mesh::ground(Vec3::ZERO, 4.0, 8), matte(0.8));
    scene.add(Mesh::sphere(Vec3::new(0.0, 0.05, 0.0), 0.05, 48, 24), matte(0.8));
    // A ball 1.4 m up the light's path, out of the camera's sight, whose shadow covers the origin.
    scene.add(Mesh::sphere(Vec3::new(1.0, 1.0, 0.0), 0.3, 48, 24), matte(0.8));
    let lit = rig(
        vec![sun(Vec3::new(-1.0, -1.0, 0.0), 1000.0, 0.01)],
        ProceduralEnvironment::Uniform,
        0.0,
        Background::Gray(0.18),
    );
    let cam = camera(Vec3::new(0.0, 0.5, 0.9), Vec3::new(0.0, 0.03, 0.0), 85.0);
    let focused = RenderConfig { taa: false, ..RenderConfig::new(SIDE, SIDE) };
    let whole = RenderConfig { shadow_focus: false, ..focused };
    let shadowed = Vec3::new(0.1, 0.0, 0.05);
    let lit_point = Vec3::new(0.0, 0.0, -0.44);
    let mut values = Vec::new();
    for config in [focused, whole] {
        let mut h = common::Harness::new(config, &scene);
        assert!(h.scene.shadow_focusable());
        let bounds = h.scene.bounds();
        let seen = h.scene.caster_slabs().visible(&cam, 1.0).unwrap();
        assert!(seen.radius < 0.5 * bounds.radius, "the camera sees {} of {} m", seen.radius, bounds.radius);
        h.frames(&cam, &lit, 1);
        let image = hdr_target(&mut h, "diffuse");
        let value = |p: Vec3| {
            let (x, y) = pixel_of(&cam, SIDE, SIDE, p);
            image[y * SIDE as usize + x][1]
        };
        values.push((value(shadowed), value(lit_point)));
    }
    println!("(in the far ball's shadow, lit): focused {:?}, whole {:?}", values[0], values[1]);
    for (dark, bright) in &values {
        assert!(*bright > 100.0 && *dark < 0.02 * bright, "shadow {dark}, lit {bright}");
    }
}

/// todo 1.9 item 4: a small ball at the focus distance and a larger one a metre behind it, f/2.
/// The ball in focus is unchanged up to its edge, the background next to it takes none of its
/// light (no halo), and the far ball's edge spreads.
#[test]
fn todo_1_9_depth_of_field_blurs_what_is_out_of_focus_without_a_halo() {
    let mut scene = Scene::default();
    scene.add(Mesh::sphere(Vec3::new(-0.035, 0.0, 0.0), 0.02, 96, 48), matte(0.9));
    scene.add(Mesh::sphere(Vec3::new(0.12, 0.0, -1.0), 0.06, 96, 48), matte(0.9));
    let cam = camera(Vec3::new(0.0, 0.0, 0.5), Vec3::ZERO, 85.0);
    let furnace = rig(Vec::new(), ProceduralEnvironment::Uniform, 1.0, Background::Gray(0.18));
    let mut h = common::Harness::new(RenderConfig::new(SIDE, SIDE), &scene);
    h.frames(&cam, &furnace, 24);
    let sharp = h.resolved();
    assert!(!h.renderer.graph().unwrap().is_live("dof"));
    let lens = Lens { max_blur: 0.1, ..Lens::new(2.0, 0.5) };
    assert!((lens.confusion_at_infinity(&cam, SIDE as f32) - 46.4).abs() < 0.5, "85 mm f/2 at 0.5 m on 256 px");
    h.renderer.set_lens(&mut h.resources, Some(lens)).unwrap();
    assert!(h.renderer.graph().unwrap().is_live("dof"));
    h.frames(&cam, &furnace, 48);
    let soft = h.resolved();
    let row = |image: &[[f32; 4]], x: usize| luminance(image[(SIDE as usize / 2) * SIDE as usize + x]);
    let background = row(&sharp, 132);
    let ball = row(&sharp, 64);
    assert!(ball > 3.0 * background, "ball {ball}, background {background}");
    // The ball in focus spans columns 28..100, the far one 164..238.
    for x in [31, 64, 97] {
        assert!((row(&soft, x) / row(&sharp, x) - 1.0).abs() < 0.02, "the ball in focus at column {x}");
    }
    for x in [22, 24, 104, 106] {
        let v = row(&soft, x);
        assert!((v / background - 1.0).abs() < 0.04, "no halo at column {x}: {v} beside {background}");
    }
    let far_ball = row(&sharp, 174);
    let outside = (row(&soft, 154) - background) / (far_ball - background);
    let inside = (row(&soft, 174) - background) / (far_ball - background);
    for x in (136..=212).step_by(4) {
        println!("  column {x}: sharp {:.3}, with the lens {:.3}", row(&sharp, x), row(&soft, x));
    }
    println!("far ball's edge with the lens: {outside:.3} 10 px outside, {inside:.3} 10 px inside (0, 1 sharp)");
    // The far ball's edge is darker than its middle (grazing Fresnel): twice the background is
    // contrast enough to measure the spread.
    assert!((row(&sharp, 154) / background - 1.0).abs() < 0.02 && far_ball > 2.0 * background);
    assert!(outside > 0.08 && outside < 0.6, "the far ball spreads outwards: {outside}");
    assert!(inside > 0.4 && inside < 0.95, "and loses light inside its edge: {inside}");
    // Without the lens again the image is the sharp one.
    h.renderer.set_lens(&mut h.resources, None).unwrap();
    h.frames(&cam, &furnace, 24);
    let again = h.resolved();
    assert!((row(&again, 154) / row(&sharp, 154) - 1.0).abs() < 0.01);
    assert!(h.renderer.set_lens(&mut h.resources, Some(Lens::new(0.1, 1.0))).is_err(), "an f-number out of range");
}

/// todo 1.9 item 5: the studio paper is a matte surface lit by the rig (floor: the light's cosine
/// plus the environment; wall, parallel to this light: the environment alone), carries the
/// subject's shadow and has no edge where the floor turns into the wall. The flat backdrop is
/// unchanged.
#[test]
fn todo_1_9_studio_paper_is_lit_by_the_rig() {
    let mut scene = Scene::default();
    scene.add(Mesh::sphere(Vec3::new(0.0, 0.1, 0.0), 0.1, 64, 32), matte(0.8));
    let direction = Vec3::new(-1.0, -0.5, 0.0).normalize_or(-Vec3::Y);
    let paper = Backdrop { falloff: 0.0, ..Backdrop::GRAY };
    let mut lit = rig(
        vec![sun(direction, std::f32::consts::PI, 0.02)],
        ProceduralEnvironment::Uniform,
        0.2,
        Background::Backdrop(paper),
    );
    let cam = camera(Vec3::new(0.0, 0.3, 1.5), Vec3::new(0.0, 0.15, 0.0), 50.0);
    let mut h = common::Harness::new(RenderConfig { taa: false, ..RenderConfig::new(SIDE, SIDE) }, &scene);
    h.frames(&cam, &lit, 1);
    let image = h.resolved();
    let at = |image: &[[f32; 4]], p: Vec3| {
        let (x, y) = pixel_of(&cam, SIDE, SIDE, p);
        luminance(image[y * SIDE as usize + x])
    };
    let albedo = luminance([paper.color[0], paper.color[1], paper.color[2], 1.0]);
    let floor = at(&image, Vec3::new(0.3, 0.0, 0.2));
    let shadow = at(&image, Vec3::new(-0.2, 0.0, 0.0));
    let wall = luminance(image[20 * SIDE as usize + 128]);
    println!(
        "paper: floor {floor:.4} ({:.4}), shadow {shadow:.4} ({:.4}), wall {wall:.4} ({:.4})",
        albedo * (-direction.y + 0.2),
        albedo * 0.2,
        albedo * 0.2
    );
    assert!((floor / (albedo * (-direction.y + 0.2)) - 1.0).abs() < 0.04, "the lit floor");
    assert!((shadow / (albedo * 0.2) - 1.0).abs() < 0.1, "the ball's shadow on the floor");
    assert!((wall / (albedo * 0.2) - 1.0).abs() < 0.05, "the wall, edge-on to the light");
    // Down a column beside the ball: no step anywhere between wall and floor.
    let column: Vec<f32> = (0..SIDE as usize).map(|y| luminance(image[y * SIDE as usize + 236])).collect();
    let step = column.windows(2).map(|w| (w[1] - w[0]).abs() / w[0].max(1e-6)).fold(0.0_f32, f32::max);
    println!("the largest step between two rows down the paper: {step:.4}");
    assert!(step < 0.12, "the largest step down the paper: {step}");
    assert!(column[250] > 1.5 * column[5], "the floor in front is brighter than the wall: {column:?}");
    // The flat grey is what it was: the display value over the exposure.
    lit.background = Background::Gray(0.18);
    h.frames(&cam, &lit, 1);
    let flat = h.resolved();
    let expected = 0.18 / cam.exposure.scale();
    assert!((luminance(flat[20 * SIDE as usize + 128]) / expected - 1.0).abs() < 1e-3);
}

/// todo 1.9 item 3: the generated environments. The sunset is orange and bright along the horizon
/// under the sun, darker opposite it and blue overhead; the studio room is the studio without its
/// painted softbox panels.
#[test]
fn todo_1_9_generated_environments() {
    let mut scene = Scene::default();
    scene.add(Mesh::sphere(Vec3::new(0.0, -50.0, 0.0), 0.01, 8, 4), matte(0.5));
    let mut h = common::Harness::new(RenderConfig { taa: false, ..RenderConfig::new(64, 64) }, &scene);
    let look = |h: &mut common::Harness, environment: ProceduralEnvironment, towards: Vec3| {
        h.renderer.set_environment(&mut h.resources, EnvironmentSource::Procedural(environment)).unwrap();
        let sky = rig(Vec::new(), environment, 1.0, Background::Environment);
        let cam = camera(Vec3::ZERO, towards, 200.0);
        h.frames(&cam, &sky, 1);
        h.resolved()[32 * 64 + 32]
    };
    let sun_direction = Vec3::new(0.7, 0.1, 0.7).normalize_or(Vec3::Y);
    let sunset = ProceduralEnvironment::Sunset { sun_direction };
    let under_sun = look(&mut h, sunset, Vec3::new(0.7, 0.02, 0.7));
    let opposite = look(&mut h, sunset, Vec3::new(-0.7, 0.02, -0.7));
    let overhead = look(&mut h, sunset, Vec3::new(0.2, 1.0, 0.2));
    let ground = look(&mut h, sunset, Vec3::new(0.7, -0.5, 0.7));
    println!("sunset: under the sun {under_sun:?}, opposite {opposite:?}, overhead {overhead:?}, ground {ground:?}");
    assert!(under_sun[0] > 2.0 * under_sun[2], "orange under the sun");
    assert!(luminance(under_sun) > 2.0 * luminance(opposite), "brighter than the far horizon");
    assert!(overhead[2] > 1.5 * overhead[0], "blue overhead");
    assert!(luminance(ground) < 0.4 * luminance(opposite), "a dark ground");
    let panel = Vec3::new(0.8, 0.45, 0.9);
    let studio = look(&mut h, ProceduralEnvironment::Studio, panel);
    let room = look(&mut h, ProceduralEnvironment::StudioRoom, panel);
    println!("towards the studio's key panel: studio {studio:?}, room {room:?}");
    assert!(luminance(studio) > 10.0 && luminance(room) < 0.4 && luminance(room) > 0.15);
}

/// todo 1.9: what the new passes cost at 2560 x 1440 on the material test scene (Review tier),
/// printed for the status file. Not asserted: the GPU of this PC is shared with whatever the
/// owner runs, and a timing taken then is not a measurement.
#[test]
fn todo_1_9_pass_costs_at_1440p() {
    let scene = Scene::material_test();
    let rig = LightRig::builtin(BuiltinRig::NeutralStudio, Vec3::new(0.0, 1.6, -0.05), 0.12);
    let mut cam = Camera {
        position: Vec3::new(0.0, 1.55, 1.1),
        target: Vec3::new(0.0, 1.45, 0.0),
        focal_mm: 50.0,
        exposure: Exposure::ev100(rig.ev100),
        ..Camera::default()
    };
    let mut unshadowed = rig.clone();
    for light in &mut unshadowed.lights {
        light.shadow.enabled = false;
    }
    for (label, occlusion, lens, shadows, quality) in [
        ("plain", false, false, true, Quality::Review),
        ("plain, no shadow maps", false, false, false, Quality::Review),
        ("plain, whole-scene shadow maps", false, false, true, Quality::Review),
        ("occlusion + lens", true, true, true, Quality::Review),
        ("Interactive, occlusion + lens", true, true, true, Quality::Interactive),
        ("Interactive, no shadow maps", false, false, false, Quality::Interactive),
    ] {
        let rig = if shadows { &rig } else { &unshadowed };
        let shadow_focus = !label.contains("whole-scene");
        let config = RenderConfig { occlusion, shadow_focus, quality, ..RenderConfig::new(2560, 1440) };
        let mut h = common::Harness::new(config, &scene);
        if lens {
            h.renderer.set_lens(&mut h.resources, Some(Lens::new(2.8, 1.1))).unwrap();
        }
        let mut per_pass: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
        let mut totals = Vec::new();
        // 20 frames of warm-up: the first configuration otherwise meets an idle GPU's low clock.
        for i in 0..60 {
            cam.orbit(0.002, 0.0);
            h.frame(&cam, rig, &[], false, None, i == 0);
            common::wait(&h.gpu.device);
            if h.renderer.collect_timings() && i >= 20 {
                let report = h.renderer.timings();
                for scope in &report.scopes {
                    per_pass.entry(scope.name).or_default().push(scope.gpu_nanos.unwrap_or(0) as f64 / 1e6);
                }
                totals.push(report.gpu_total_nanos().unwrap_or(0) as f64 / 1e6);
            }
        }
        let median = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v.get(v.len() / 2).copied().unwrap_or(0.0)
        };
        println!("2560x1440 {label}: GPU total {:.3} ms (median of {})", median(&mut totals), totals.len());
        for (name, mut v) in per_pass {
            println!("  {name:<12} {:.3} ms", median(&mut v));
        }
    }
}
