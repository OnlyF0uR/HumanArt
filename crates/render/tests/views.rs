//! Debug views (spec 8.5) and overlays (lines, points, wireframe, selection highlight), checked
//! on offscreen renders and written to target/agents/render/shots/ for review.
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
use humanart_render::light::{BuiltinRig, LightRig};
use humanart_render::renderer::{DebugView, OverlayPrimitive, RenderConfig};
use humanart_render::scene::Scene;

const W: u32 = 320;
const H: u32 = 240;

fn camera() -> Camera {
    Camera {
        position: Vec3::new(0.0, 1.5, 1.6),
        target: Vec3::new(0.0, 1.45, 0.0),
        focal_mm: 35.0,
        near: 0.05,
        exposure: Exposure::ev100(rig().ev100),
        ..Camera::default()
    }
}

fn rig() -> LightRig {
    LightRig::builtin(BuiltinRig::NeutralStudio, Vec3::new(0.0, 1.6, -0.05), 0.12)
}

/// The pixel (column, row) a world point lands on.
fn pixel_of(camera: &Camera, w: u32, h: u32, p: Vec3) -> (u32, u32) {
    let m = ViewMatrices::new(camera, w, h, Vec2::ZERO, None).view_projection;
    let s = ndc_to_screen(m.project_point3(p).xy(), Vec2::new(w as f32, h as f32));
    (s.x as u32, s.y as u32)
}

fn rgb(image: &[u8], w: u32, (x, y): (u32, u32)) -> [u8; 3] {
    let i = ((y * w + x) * 4) as usize;
    [image[i], image[i + 1], image[i + 2]]
}

fn close(a: [u8; 3], b: [u8; 3], tolerance: u8) -> bool {
    a.iter().zip(&b).all(|(x, y)| x.abs_diff(*y) <= tolerance)
}

fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
    a.iter().zip(b).map(|(x, y)| f64::from(x.abs_diff(*y))).sum::<f64>() / a.len() as f64
}

/// Every debug view of spec 8.5 that the core renderer has, on the material test scene after a
/// small camera move (so motion vectors are not zero): each is a real picture (not one flat
/// colour) and no two views show the same image. The grid is `debug_views.png`, in the order of
/// `DebugView::ALL`, five per row.
#[test]
fn debug_view_grid_every_view_is_distinct() {
    let scene = Scene::material_test();
    let config = RenderConfig { taa: false, ..RenderConfig::new(W, H) };
    let mut h = common::Harness::new(config, &scene);
    let rig = rig();
    let first = camera();
    let mut second = first;
    second.orbit(0.03, 0.01);
    let mut images = Vec::new();
    for view in DebugView::ALL {
        h.renderer.configure(&mut h.resources, RenderConfig { debug: view, ..config }).unwrap();
        h.frame(&first, &rig, &[], false, None, true);
        h.frame(&second, &rig, &[], false, None, false);
        let (_, _, rgba) = h.display();
        let mut colours: Vec<[u8; 3]> = rgba.as_chunks::<4>().0.iter().map(|p| [p[0], p[1], p[2]]).collect();
        colours.sort_unstable();
        colours.dedup();
        println!("{view:?}: {} distinct colours", colours.len());
        // Albedo, metalness and the material classes are flat per material: a handful of colours
        // (metalness: 0, 1 and the background).
        assert!(colours.len() >= 2, "{view:?} is flat: {} colours", colours.len());
        images.push((view, rgba));
    }
    for (i, (va, a)) in images.iter().enumerate() {
        for (vb, b) in &images[i + 1..] {
            let d = mean_abs_diff(a, b);
            assert!(d > 0.5, "{va:?} and {vb:?} look the same (mean difference {d:.3})");
        }
    }
    let (cols, rows) = (5_u32, images.len().div_ceil(5) as u32);
    let (gw, gh) = (cols * W, rows * H);
    let mut grid = vec![0_u8; (gw * gh * 4) as usize];
    for (n, (_, rgba)) in images.iter().enumerate() {
        let (ox, oy) = ((n as u32 % cols) * W, (n as u32 / cols) * H);
        for y in 0..H {
            let src = (y * W * 4) as usize;
            let dst = (((oy + y) * gw + ox) * 4) as usize;
            grid[dst..dst + (W * 4) as usize].copy_from_slice(&rgba[src..src + (W * 4) as usize]);
        }
    }
    common::save_png("debug_views.png", gw, gh, &grid);
}

/// Overlays draw over the tone-mapped image without touching it elsewhere: a depth-tested line
/// is hidden behind the head and visible beside it, an on-top line and point show through, the
/// selection highlight tints only the selected object, and the wireframe only covers geometry.
#[test]
fn overlays_lines_points_selection_and_wireframe() {
    let scene = Scene::material_test();
    let (w, hgt) = (480, 360);
    let config = RenderConfig { taa: false, overlays: true, ..RenderConfig::new(w, hgt) };
    let mut h = common::Harness::new(config, &scene);
    let rig = rig();
    let camera = camera();
    let head = scene.objects.len() - 1;
    h.frame(&camera, &rig, &[], false, None, true);
    let (_, _, base) = h.display();

    let red = [1.0, 0.0, 0.0, 1.0];
    let green = [0.0, 1.0, 0.0, 1.0];
    let blue = [0.0, 0.0, 1.0, 1.0];
    let behind = -0.35;
    let overlay = [
        OverlayPrimitive::line(Vec3::new(-0.4, 1.62, behind), Vec3::new(0.4, 1.62, behind), 5.0, red, false),
        OverlayPrimitive::line(Vec3::new(-0.4, 1.56, behind), Vec3::new(0.4, 1.56, behind), 5.0, green, true),
        OverlayPrimitive::point(Vec3::new(0.0, 1.62, 0.04), 11.0, blue, true),
    ];
    h.frame(&camera, &rig, &overlay, false, None, true);
    let (_, _, lines) = h.display();
    let hidden = pixel_of(&camera, w, hgt, Vec3::new(0.0, 1.62, behind));
    let beside = pixel_of(&camera, w, hgt, Vec3::new(-0.3, 1.62, behind));
    let through = pixel_of(&camera, w, hgt, Vec3::new(0.0, 1.56, behind));
    let point = pixel_of(&camera, w, hgt, Vec3::new(0.0, 1.62, 0.04));
    println!(
        "hidden {:?} (base {:?}), beside {:?}, through {:?}, point {:?}",
        rgb(&lines, w, hidden),
        rgb(&base, w, hidden),
        rgb(&lines, w, beside),
        rgb(&lines, w, through),
        rgb(&lines, w, point)
    );
    assert!(close(rgb(&lines, w, hidden), rgb(&base, w, hidden), 1), "the depth-tested line shows through the head");
    assert!(close(rgb(&lines, w, beside), [255, 0, 0], 2), "the depth-tested line beside the head");
    assert!(close(rgb(&lines, w, through), [0, 255, 0], 2), "the on-top line over the head");
    assert!(close(rgb(&lines, w, point), [0, 0, 255], 2), "the on-top point");
    assert!(close(rgb(&lines, w, (2, 2)), rgb(&base, w, (2, 2)), 0), "the overlay touches nothing else");
    common::save_png("overlay_lines.png", w, hgt, &lines);

    h.scene.set_selected(head, true).unwrap();
    h.frame(&camera, &rig, &[], false, None, true);
    let (_, _, selected) = h.display();
    h.scene.set_selected(head, false).unwrap();
    let face = pixel_of(&camera, w, hgt, Vec3::new(0.0, 1.66, 0.0));
    let sphere = pixel_of(&camera, w, hgt, Vec3::new(-0.24, 1.28, 0.3));
    let (f0, f1) = (rgb(&base, w, face), rgb(&selected, w, face));
    println!(
        "selection: face {f0:?} -> {f1:?}, sphere {sphere:?} {:?} -> {:?}",
        rgb(&base, w, sphere),
        rgb(&selected, w, sphere)
    );
    common::save_png("overlay_selection.png", w, hgt, &selected);
    // The highlight is orange (1, 0.62, 0.15): it lowers blue against the red channel.
    assert!(
        f1 != f0 && i32::from(f1[0]) - i32::from(f1[2]) > i32::from(f0[0]) - i32::from(f0[2]),
        "the head is tinted"
    );
    // One code of tolerance: the shading noise is seeded with the frame number.
    assert!(close(rgb(&selected, w, sphere), rgb(&base, w, sphere), 1), "an unselected sphere is untouched");
    assert!(close(rgb(&selected, w, (2, 2)), rgb(&base, w, (2, 2)), 0), "the background is untouched");

    h.frame(&camera, &rig, &[], true, None, true);
    let (_, _, wire) = h.display();
    let changed = base.as_chunks::<4>().0.iter().zip(wire.as_chunks::<4>().0).filter(|(a, b)| a[..3] != b[..3]).count();
    let fraction = changed as f64 / f64::from(w * hgt);
    println!("wireframe changes {:.1} % of the pixels", fraction * 100.0);
    assert!(fraction > 0.02, "the wireframe is visible");
    assert!(close(rgb(&wire, w, (2, 2)), rgb(&base, w, (2, 2)), 0), "no wire over the background");
    common::save_png("overlay_wireframe.png", w, hgt, &wire);
}
