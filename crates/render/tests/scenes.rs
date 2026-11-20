//! Renders of the procedural test scenes, written to target/agents/render/shots/ for review.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use humanart_math::Vec3;
use humanart_render::camera::{Camera, ReviewCamera};
use humanart_render::light::{BuiltinRig, LightRig};
use humanart_render::renderer::RenderConfig;
use humanart_render::scene::Scene;

#[test]
fn material_test_scene_studio() {
    let scene = Scene::material_test();
    let mut h = common::Harness::new(RenderConfig::new(960, 640), &scene);
    let center = Vec3::new(0.0, 1.45, 0.0);
    let mut camera = Camera::review(ReviewCamera::Front, center, 0.4, 1.5);
    camera.position += Vec3::new(0.0, 0.25, 0.0);
    let rig = LightRig::builtin(BuiltinRig::NeutralStudio, Vec3::new(0.0, 1.6, -0.05), 0.12);
    camera.exposure = humanart_render::camera::Exposure::ev100(rig.ev100);
    h.frames(&camera, &rig, 16);
    let (_, _, rgba) = h.save_display("material_test_studio.png");
    let mean: f64 = rgba.iter().map(|&v| f64::from(v)).sum::<f64>() / rgba.len() as f64;
    assert!(mean > 20.0 && mean < 235.0, "the image is not blank: {mean}");
}
