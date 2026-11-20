//! Tests of the groom tools `[HAIR-03]` `[HAIR-10]`.

use humanart_math::{Vec2, Vec3};
use humanart_model::hair::cards::{CardSet, CardSpec, build_cards};
use humanart_model::hair::collide::{SdfGrid, TriangleGrid};
use humanart_model::hair::presets::{GrowSpec, preset_groom};
use humanart_model::hair::{GroomMap, GroomMaps, GroomParams, Guides, ProxyHead, ScalpMask, generate_groom, place};

use super::cards::{CardAtlas, CardStrands, bake_card_atlas, export_cards_glb};
use super::ops::{Brush, comb, cut, edit_hairline, mirror, paint_map, place_cowlick, place_guide, set_parting};

struct Fixture {
    head: ProxyHead,
    mask: ScalpMask,
    grid: TriangleGrid,
    sdf: SdfGrid,
}

fn fixture() -> Fixture {
    let head = ProxyHead::new(48).unwrap();
    let mask = ScalpMask::new(&head.surface, head.scalp.clone()).unwrap();
    let grid = TriangleGrid::new(head.surface.positions(), head.surface.triangles(), 0.01).unwrap();
    let sdf = SdfGrid::build(&grid, 0.004, 0.05).unwrap();
    Fixture { head, mask, grid, sdf }
}

fn world(g: &Guides, f: &Fixture, i: usize) -> Vec<Vec3> {
    g.world_points(i, &g.frame(i, &f.head.surface).unwrap())
}

#[test]
fn hair_03_comb_turns_tips_and_keeps_lengths() {
    let f = fixture();
    let mut g = preset_groom("short_crop", &f.head.surface, &f.mask, 200, Some(&f.sdf), 1).unwrap().guides;
    let before = g.clone();
    let top = f.head.surface_point(Vec3::new(0.0, 1.0, 0.0));
    let brush = Brush { centre: top, radius: 0.04, strength: 1.0 };
    let report = comb(&mut g, &f.head.surface, &brush, Vec3::X, Some(&f.sdf), 0.002).unwrap();
    assert!(!report.guides.is_empty() && report.elements > 0);
    for i in 0..g.len() {
        let touched = report.guides.binary_search(&(i as u32)).is_ok();
        if touched {
            let (a, b) = (world(&before, &f, i), world(&g, &f, i));
            assert!((g.length(i) - before.length(i)).abs() < 0.01 * before.length(i) + 1e-6);
            assert_eq!(a[0], b[0], "the root stays");
            let (ta, tb) = (a[a.len() - 1] - a[0], b[b.len() - 1] - b[0]);
            assert!(tb.x >= ta.x - 1e-5, "guide {i} turned away from the comb");
        } else {
            assert_eq!(g.offsets(i), before.offsets(i));
        }
    }
    let shift = |i: u32| {
        let (a, b) = (world(&before, &f, i as usize), world(&g, &f, i as usize));
        (b[b.len() - 1] - a[a.len() - 1]).x
    };
    let mean = report.guides.iter().map(|&i| shift(i)).sum::<f32>() / report.guides.len() as f32;
    let best = report.guides.iter().map(|&i| shift(i)).fold(0.0_f32, f32::max);
    assert!(mean > 0.0 && best > 0.005, "mean {mean} best {best}");
    assert!(comb(&mut g, &f.head.surface, &brush, Vec3::ZERO, None, 0.0).is_err());
}

#[test]
fn hair_03_cut_shortens_guides_at_the_brush() {
    let f = fixture();
    let mut g = preset_groom("long_straight", &f.head.surface, &f.mask, 150, Some(&f.sdf), 2).unwrap().guides;
    let before = g.clone();
    // A huge sphere whose top is at the median tip height acts as a cutting plane there.
    let mut tips: Vec<f32> = (0..g.len()).map(|i| world(&g, &f, i).last().unwrap().y).collect();
    tips.sort_by(f32::total_cmp);
    let level = tips[tips.len() / 2];
    let brush = Brush { centre: Vec3::new(0.0, level - 2.0, 0.0), radius: 2.0, strength: 1.0 };
    let report = cut(&mut g, &f.head.surface, &brush, 0.01).unwrap();
    assert!(report.guides.len() * 10 > g.len() * 4, "{} of {}", report.guides.len(), g.len());
    for &i in &report.guides {
        let i = i as usize;
        assert!(g.length(i) < before.length(i));
        let w = world(&g, &f, i);
        assert!((w[w.len() - 1] - brush.centre).length() > brush.radius - 0.002);
        assert_eq!(w[0], world(&before, &f, i)[0]);
    }
    let bad = Brush { radius: 0.0, ..brush };
    assert!(cut(&mut g, &f.head.surface, &bad, 0.0).is_err());
}

#[test]
fn hair_03_paint_hairline_parting_cowlick_place() {
    let f = fixture();
    // Density painting.
    let mut density = GroomMap::new(16, 16, 1, &[1.0]).unwrap();
    let n = paint_map(&mut density, Vec2::new(0.5, 0.5), 0.2, 1.0, &[0.0]).unwrap();
    assert!(n > 20);
    assert!(density.sample(Vec2::new(0.5, 0.5), 0) < 0.1);
    assert!((density.sample(Vec2::new(0.02, 0.02), 0) - 1.0).abs() < 1e-6);
    assert!(paint_map(&mut density, Vec2::new(0.5, 0.5), 0.2, 1.0, &[0.0, 1.0]).is_err());
    // Clump painting on a two-channel map works the same way.
    let mut direction = GroomMap::new(8, 8, 2, &[0.0, 0.0]).unwrap();
    assert!(paint_map(&mut direction, Vec2::new(0.1, 0.1), 0.3, 0.5, &[1.0, -1.0]).unwrap() > 0);
    // Hairline: add a patch of forehead, then remove it again.
    let mut scalp = f.head.scalp.clone();
    let forehead = f.head.surface_point(Vec3::new(0.0, 0.1, 1.0));
    let brush = Brush { centre: forehead, radius: 0.02, strength: 1.0 };
    let added = edit_hairline(&mut scalp, &f.head.surface, &brush, true).unwrap();
    assert!(added > 0);
    let removed = edit_hairline(&mut scalp, &f.head.surface, &brush, false).unwrap();
    assert!(removed >= added);
    // Parting and cowlick.
    let mut params = GroomParams::default();
    let top = f.head.surface_point(Vec3::new(0.02, 1.0, 0.2));
    set_parting(&mut params, top, Vec3::Y, Vec3::Z, 0.02, 0.8).unwrap();
    let p = params.parting();
    assert!((p.normal + Vec3::X).length() < 1e-5 && (p.point - top).length() < 1e-5);
    assert!(set_parting(&mut params, top, Vec3::Y, Vec3::Y, 0.02, 0.8).is_err());
    place_cowlick(&mut params, 1, top, 0.03, -0.5).unwrap();
    assert_eq!(params.cowlick(1).unwrap().radius, 0.03);
    assert!(place_cowlick(&mut params, 9, top, 0.03, 0.5).is_err());
    assert!(place_cowlick(&mut params, 0, Vec3::splat(f32::NAN), 0.03, 0.5).is_err());
    // Guide placement.
    let mut g = Guides::new(8).unwrap();
    let spec = GrowSpec { length: 0.05, comb: Vec3::new(0.0, 0.0, -1.0), comb_rate: 40.0, gravity: 5.0, offset: 0.002 };
    let i =
        place_guide(&mut g, &f.head.surface, &f.grid, top + Vec3::new(0.0, 0.01, 0.0), &spec, Some(&f.sdf)).unwrap();
    assert_eq!(i, 0);
    assert!((world(&g, &f, 0)[0] - top).length() < 0.012);
    assert!((g.length(0) - 0.05).abs() < 0.005);
}

#[test]
fn hair_03_mirror_makes_the_groom_symmetric() {
    let f = fixture();
    let mut g = preset_groom("wavy", &f.head.surface, &f.mask, 120, Some(&f.sdf), 5).unwrap().guides;
    let sources: Vec<Vec<Vec3>> = (0..g.len()).map(|i| world(&g, &f, i)).filter(|w| w[0].x > 1e-4).collect();
    let report = mirror(&mut g, &f.head.surface, &f.grid, true).unwrap();
    assert_eq!(report.guides.len(), sources.len());
    let mirrored: Vec<Vec<Vec3>> = report.guides.iter().map(|&i| world(&g, &f, i as usize)).collect();
    for (src, m) in sources.iter().zip(&mirrored) {
        assert!(m[0].x < 0.0);
        for (a, b) in src.iter().zip(m) {
            assert!((Vec3::new(-a.x, a.y, a.z) - *b).length() < 0.003, "{a:?} {b:?}");
        }
    }
    // No guide is left on the destination side except the mirrored ones.
    let left = (0..g.len()).filter(|&i| world(&g, &f, i)[0].x < -1e-4).count();
    assert_eq!(left, sources.len());
}

struct Cards {
    sets: Vec<(CardSet, CardAtlas)>,
    world: Vec<Vec3>,
    width_root: Vec<f32>,
    width_tip: Vec<f32>,
    per: usize,
}

fn cards(f: &Fixture, strands: f32) -> Cards {
    let s = &f.head.surface;
    let p = preset_groom("long_straight", s, &f.mask, 300, Some(&f.sdf), 3).unwrap();
    let params = GroomParams { strand_count: strands, ..p.params };
    let layout = place(s, &f.mask, &p.guides, &params, &GroomMaps::default(), 4).unwrap();
    let g = generate_groom(s, &layout, &p.guides, &params, Some(&f.sdf)).unwrap();
    let normals: Vec<Vec3> =
        (0..layout.len()).map(|i| s.frame(layout.triangle[i], layout.bary[i]).unwrap().n).collect();
    let input = CardStrands {
        world: &g.world,
        per: g.strands.points,
        width_root: &g.strands.width_root,
        width_tip: &g.strands.width_tip,
    };
    let sets = CardSpec::LODS
        .iter()
        .map(|spec| {
            let set = build_cards(&g.world, g.strands.points, &normals, spec, 9).unwrap();
            let atlas = bake_card_atlas(&set, &input, [32, 128]).unwrap();
            (set, atlas)
        })
        .collect();
    Cards {
        sets,
        world: g.world.clone(),
        width_root: g.strands.width_root.clone(),
        width_tip: g.strands.width_tip.clone(),
        per: g.strands.points as usize,
    }
}

/// Coverage images (`w x h`, side view from +X, `scale` pixels per metre) of the strands, each
/// strand covering `min(1, width in pixels)` of the pixels it crosses, and of every card LOD, each
/// card covering its baked alpha; layers combine as `1 − Π(1 − c)`.
fn side_coverage(c: &Cards, centre: Vec3, w: usize, h: usize, scale: f32) -> (Vec<f32>, Vec<Vec<f32>>) {
    let project =
        |p: Vec3| Vec2::new(w as f32 * 0.65 + (p.z - centre.z) * scale, h as f32 * 0.25 - (p.y - centre.y) * scale);
    // Transmittance products, turned into coverage at the end.
    let mut strands = vec![1.0_f32; w * h];
    let mut seen = vec![u32::MAX; w * h];
    for (i, s) in c.world.chunks(c.per).enumerate() {
        for k in 1..s.len() {
            let t = k as f32 / (s.len() - 1) as f32;
            let width = (c.width_root[i] + (c.width_tip[i] - c.width_root[i]) * t) * scale;
            let (a, b) = (project(s[k - 1]), project(s[k]));
            let steps = (b - a).length().ceil().max(1.0) as usize;
            for step in 0..=steps {
                let q = a + (b - a) * (step as f32 / steps as f32);
                if q.x >= 0.0 && q.y >= 0.0 && (q.x as usize) < w && (q.y as usize) < h {
                    let at = q.y as usize * w + q.x as usize;
                    // A strand counts once per pixel.
                    if seen[at] != i as u32 {
                        seen[at] = i as u32;
                        strands[at] *= 1.0 - width.min(1.0);
                    }
                }
            }
        }
    }
    let strands = strands.iter().map(|t| 1.0 - t).collect();
    let mut lods = Vec::new();
    for (set, atlas) in &c.sets {
        let mut trans = vec![1.0_f32; w * h];
        let m = &set.mesh;
        for tri in m.indices.chunks(3) {
            let v: Vec<Vec2> = tri.iter().map(|&i| project(m.positions[i as usize])).collect();
            let uv: Vec<Vec2> = tri.iter().map(|&i| m.uvs[i as usize]).collect();
            let lo = v[0].min(v[1]).min(v[2]).max(Vec2::ZERO);
            let hi = v[0].max(v[1]).max(v[2]).min(Vec2::new(w as f32 - 1.0, h as f32 - 1.0));
            let area = (v[1] - v[0]).perp_dot(v[2] - v[0]);
            if area.abs() < 1e-9 || hi.x < lo.x || hi.y < lo.y {
                continue;
            }
            for y in lo.y as usize..=hi.y as usize {
                for x in lo.x as usize..=hi.x as usize {
                    let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                    let b1 = (p - v[0]).perp_dot(v[2] - v[0]) / area;
                    let b2 = (v[1] - v[0]).perp_dot(p - v[0]) / area;
                    let b0 = 1.0 - b1 - b2;
                    if b0 < 0.0 || b1 < 0.0 || b2 < 0.0 {
                        continue;
                    }
                    let t = uv[0] * b0 + uv[1] * b1 + uv[2] * b2;
                    let tx = ((t.x * atlas.alpha.width as f32) as u32).min(atlas.alpha.width - 1);
                    let ty = ((t.y * atlas.alpha.height as f32) as u32).min(atlas.alpha.height - 1);
                    let a = atlas.alpha.at(tx, ty).map_or(0.0, |a| a[0]);
                    trans[y * w + x] *= 1.0 - a.clamp(0.0, 1.0);
                }
            }
        }
        lods.push(trans.iter().map(|t| 1.0 - t).collect());
    }
    (strands, lods)
}

#[test]
fn hair_10_card_lods_bake_export_and_match_the_strands() {
    let f = fixture();
    let c = cards(&f, 30_000.0);
    for (set, atlas) in &c.sets {
        let (empty, covered, _) = atlas.alpha.coverage_counts();
        assert!(covered > 0 && empty > 0);
        assert_eq!(atlas.alpha.width, set.cols * 32);
        assert!(atlas.flow.data.iter().all(|v| v.is_finite()));
    }
    let lods: Vec<(&CardSet, &CardAtlas)> = c.sets.iter().map(|(s, a)| (s, a)).collect();
    let glb = export_cards_glb(&lods).unwrap();
    let back = humanart_io::gltf::read_self_contained(&glb, &humanart_io::Limits::DEFAULT).unwrap();
    assert_eq!(back.document.meshes.len(), 3);
    assert_eq!(back.document.materials.len(), 3);
    assert_eq!(back.document.images.len(), 15);
    assert!(export_cards_glb(&[]).is_err());
    // Side-by-side comparison scene: the strand groom and each card LOD seen from the side at
    // 2.5 mm per pixel; agreement is the soft intersection over union of the coverage images,
    // sum(min) / sum(max).
    let (w, h) = (300, 420);
    let (strands, lods) = side_coverage(&c, f.head.centre, w, h, 400.0);
    let soft_iou = |a: &[f32], b: &[f32]| {
        let (mut lo, mut hi) = (0.0_f64, 0.0_f64);
        for (x, y) in a.iter().zip(b) {
            lo += f64::from(x.min(*y));
            hi += f64::from(x.max(*y));
        }
        lo / hi.max(1e-9)
    };
    let ious: Vec<f64> = lods.iter().map(|l| soft_iou(&strands, l)).collect();
    let total = |v: &[f32]| v.iter().map(|&x| f64::from(x)).sum::<f64>();
    let ratio: Vec<f64> = lods.iter().map(|l| total(l) / total(&strands)).collect();
    println!(
        "[HAIR-10] 30k strands, side view 2.5 mm/px: soft IoU cards vs strands, LOD 256/96/32 cards: {ious:.3?}; \
         total coverage ratio {ratio:.3?}"
    );
    // The picture: strands, then each LOD, coverage as brightness.
    let mut rgb = vec![28_u8; w * 4 * h * 3];
    for (panel, cov) in std::iter::once(&strands).chain(&lods).enumerate() {
        for y in 0..h {
            for x in 0..w {
                let v = cov[y * w + x].clamp(0.0, 1.0);
                let i = (y * w * 4 + panel * w + x) * 3;
                for (k, base) in [230.0, 170.0, 100.0].iter().enumerate() {
                    rgb[i + k] = (28.0 + (base - 28.0) * v) as u8;
                }
            }
        }
    }
    let image = humanart_io::Image::new(
        humanart_io::Format::Png,
        (w * 4) as u32,
        h as u32,
        3,
        humanart_io::ColorSpace::Srgb,
        humanart_io::Samples::U8(rgb),
    )
    .unwrap();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/agents/hair");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("cards_compare.png"), humanart_io::png::encode(&image).unwrap()).unwrap();
    let alpha = c.sets[0].1.alpha.to_unorm8(humanart_io::ColorSpace::Data, 1.0, 0.0).unwrap();
    let alpha = humanart_io::Image::new(
        humanart_io::Format::Png,
        alpha.width,
        alpha.height,
        1,
        humanart_io::ColorSpace::Data,
        alpha.samples,
    )
    .unwrap();
    std::fs::write(dir.join("cards_alpha_lod0.png"), humanart_io::png::encode(&alpha).unwrap()).unwrap();
    assert!(ious[0] > 0.5, "{ious:?}");
    assert!(ious.iter().all(|&x| x > 0.3), "{ious:?}");
}
