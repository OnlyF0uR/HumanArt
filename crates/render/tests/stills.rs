//! `[AC-WP11-3]` `[QUAL-04]`: an 8K still rendered in tiles with overlap equals the untiled render.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use humanart_gpu::resources::{Budget, Resources};
use humanart_math::Vec3;
use humanart_render::camera::{Camera, Exposure};
use humanart_render::light::{BuiltinRig, LightRig};
use humanart_render::renderer::{Quality, RenderConfig, Renderer};
use humanart_render::scene::{GpuScene, Scene};
use humanart_render::still::{StillSettings, render_still};

const W: u32 = 7680;
const H: u32 = 4320;

fn still(
    resources: &mut Resources,
    gpu: &humanart_gpu::Gpu,
    scene: &mut GpuScene,
    settings: StillSettings,
) -> (Vec<u8>, f64) {
    let (tw, th) = settings.target_size();
    let mut renderer =
        Renderer::new(gpu, resources, RenderConfig { quality: Quality::Still, ..RenderConfig::new(tw, th) }).unwrap();
    let rig = LightRig::builtin(BuiltinRig::NeutralStudio, Vec3::new(0.0, 1.6, -0.05), 0.12);
    let camera = Camera {
        position: Vec3::new(0.3, 1.62, 0.9),
        target: Vec3::new(0.0, 1.45, 0.0),
        focal_mm: 50.0,
        exposure: Exposure::ev100(rig.ev100),
        ..Camera::default()
    };
    let start = std::time::Instant::now();
    let image = render_still(&mut renderer, resources, scene, &camera, &rig, &settings).unwrap();
    let seconds = start.elapsed().as_secs_f64();
    renderer.destroy(resources);
    (image, seconds)
}

/// A box-filtered copy at 1/`f` of the size.
fn downscale(rgba: &[u8], w: u32, h: u32, f: u32) -> (u32, u32, Vec<u8>) {
    let (ow, oh) = (w / f, h / f);
    let mut out = vec![0_u8; (ow * oh * 4) as usize];
    for y in 0..oh {
        for x in 0..ow {
            for c in 0..4 {
                let mut sum = 0_u32;
                for dy in 0..f {
                    for dx in 0..f {
                        sum += u32::from(rgba[(((y * f + dy) * w + x * f + dx) * 4 + c) as usize]);
                    }
                }
                out[((y * ow + x) * 4 + c) as usize] = (sum / (f * f)) as u8;
            }
        }
    }
    (ow, oh, out)
}

fn crop(rgba: &[u8], w: u32, x0: u32, y0: u32, cw: u32, ch: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((cw * ch * 4) as usize);
    for y in y0..y0 + ch {
        let start = ((y * w + x0) * 4) as usize;
        out.extend_from_slice(&rgba[start..start + (cw * 4) as usize]);
    }
    out
}

#[test]
fn ac_wp11_3_tiled_8k_still_equals_the_untiled_render() {
    let gpu = common::gpu();
    let mut resources = Resources::new(gpu, Budget::UNLIMITED);
    let mut scene = GpuScene::new(&mut resources, &Scene::material_test()).unwrap();
    let samples = 4;
    let (untiled, untiled_s) =
        still(&mut resources, gpu, &mut scene, StillSettings { width: W, height: H, tile: W, overlap: 0, samples });
    let tiled_settings = StillSettings { width: W, height: H, tile: 2048, overlap: 32, samples };
    let tiles = humanart_render::still::plan_tiles(&tiled_settings).len();
    let (tiled, tiled_s) = still(&mut resources, gpu, &mut scene, tiled_settings);
    let mut max = 0_u8;
    let mut differing = 0_usize;
    let mut sum = 0_u64;
    for (a, b) in untiled.iter().zip(&tiled) {
        let d = a.abs_diff(*b);
        max = max.max(d);
        sum += u64::from(d);
        if d > 1 {
            differing += 1;
        }
    }
    let mean = sum as f64 / untiled.len() as f64;
    println!(
        "8K still: untiled {untiled_s:.2} s, {tiles} tiles {tiled_s:.2} s; max difference {max} code values, mean {mean:.6}, {differing} samples differ by more than 1"
    );
    let (sw, sh, small) = downscale(&tiled, W, H, 4);
    common::save_png("still_8k_tiled_quarter.png", sw, sh, &small);
    // 1:1 crops across the first vertical tile seam (x = 2048) of both renders.
    let (cx, cy, cw, ch) = (2048 - 256, 1900, 512, 400);
    common::save_png("still_8k_seam_tiled.png", cw, ch, &crop(&tiled, W, cx, cy, cw, ch));
    common::save_png("still_8k_seam_untiled.png", cw, ch, &crop(&untiled, W, cx, cy, cw, ch));
    // Where do the larger differences lie? Rasterisation through the tile's sub-frustum rounds
    // vertex positions differently in the last bit, so a sample whose centre lies on a triangle
    // edge may change sides. That moves the pixel towards the colour of the surface on the other
    // side of the edge, which its neighbours show: a tiled value outside the range of the
    // untiled 5x5 neighbourhood (plus one code of rounding) is a real shading difference, not an
    // edge sample. (5x5 rather than 3x3: the tapering tip of a thin specular highlight along a
    // silhouette is sub-pixel, and one sample flipping onto it lands outside the 3x3 range;
    // still_8k_outlier_*.png show such a case.) A handful of single pixels on silhouettes still
    // leave the 5x5 range by a few codes (3 of 132 M samples on the RTX 4060, at most 10 codes,
    // each next to a silhouette); the bound allows one per ten million samples, which a shading
    // difference (shadow or noise not matching across tiles) would exceed by orders of magnitude. Count those, and the larger differences next to a tile seam.
    let outside_neighbourhood = |x: u32, y: u32, k: usize, value: u8| {
        let (mut lo, mut hi) = (u8::MAX, u8::MIN);
        for ny in y.saturating_sub(2)..=(y + 2).min(H - 1) {
            for nx in x.saturating_sub(2)..=(x + 2).min(W - 1) {
                let v = untiled[((ny * W + nx) * 4) as usize + k];
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        value.saturating_add(1) < lo || value > hi.saturating_add(1)
    };
    let mut off_edge = 0;
    let mut at_seam = 0;
    for (i, (a, b)) in untiled.iter().zip(&tiled).enumerate() {
        let d = a.abs_diff(*b);
        if d <= 2 {
            continue;
        }
        let pixel = (i / 4) as u32;
        let (x, y) = (pixel % W, pixel / W);
        if outside_neighbourhood(x, y, i % 4, *b) {
            off_edge += 1;
            if off_edge <= 12 {
                if off_edge == 1 {
                    let (cx, cy) = (x.saturating_sub(16).min(W - 32), y.saturating_sub(16).min(H - 32));
                    common::save_png("still_8k_outlier_tiled.png", 32, 32, &crop(&tiled, W, cx, cy, 32, 32));
                    common::save_png("still_8k_outlier_untiled.png", 32, 32, &crop(&untiled, W, cx, cy, 32, 32));
                }
                println!("  outside the neighbourhood: ({x}, {y}) channel {} untiled {a} tiled {b}", i % 4);
            }
        }
        if x % 2048 < 2 || x % 2048 > 2045 || y % 2048 < 2 || y % 2048 > 2045 {
            at_seam += 1;
        }
    }
    println!(
        "differences above 2 codes: {off_edge} outside the untiled neighbourhood, {at_seam} within 2 px of a tile seam"
    );
    assert!(mean < 0.002, "mean difference {mean}");
    assert!(differing < untiled.len() / 100_000, "{differing} samples differ by more than one code value");
    assert!(off_edge <= untiled.len() / 10_000_000, "{off_edge} tiled values outside the untiled neighbourhood");
    assert!(at_seam <= differing / 20 + 2, "tile seams show: {at_seam}");
}
