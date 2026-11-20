//! Synthetic test shapes for the bakers, and the cross-module tests: seam continuity through the
//! mip chain `[AC-WP15-1]` and the regression images `[AC-WP15-2]`.

use std::collections::HashMap;
use std::path::PathBuf;

use humanart_io::{ColorSpace, Format, Image, Samples};
use humanart_math::{Vec2, Vec3};
use humanart_model::{Mesh, MeshBuilder};

use super::BakeMesh;

/// Adds vertices, reusing a position seen before as a duplicate (smooth across UV seams, or hard).
struct Welder {
    b: MeshBuilder,
    seen: HashMap<[i64; 3], u32>,
    smooth: bool,
}

impl Welder {
    fn new(part: &str, smooth: bool) -> Self {
        let mut b = MeshBuilder::new();
        b.begin_part(part, true).unwrap();
        Self { b, seen: HashMap::new(), smooth }
    }

    fn vertex(&mut self, p: Vec3, uv: Vec2) -> u32 {
        let key = [(p.x * 1e7).round() as i64, (p.y * 1e7).round() as i64, (p.z * 1e7).round() as i64];
        if let Some(&v) = self.seen.get(&key) {
            self.b.duplicate(v, uv, self.smooth).unwrap()
        } else {
            let v = self.b.vertex(p, uv);
            self.seen.insert(key, v);
            v
        }
    }
}

/// A grid of `(n + 1)²` vertices of one face patch; returns the vertex ids row by row.
fn patch(w: &mut Welder, n: usize, at: impl Fn(f32, f32) -> (Vec3, Vec2)) -> Vec<u32> {
    let mut ids = Vec::new();
    // Vertices shared inside the patch are added once; across patches they become duplicates.
    let mut local: HashMap<[i64; 3], u32> = HashMap::new();
    for j in 0..=n {
        for i in 0..=n {
            let (p, uv) = at(i as f32 / n as f32, j as f32 / n as f32);
            let key = [(p.x * 1e7).round() as i64, (p.y * 1e7).round() as i64, (p.z * 1e7).round() as i64];
            let v = if let Some(&v) = local.get(&key) {
                v
            } else {
                let v = w.vertex(p, uv);
                local.insert(key, v);
                v
            };
            ids.push(v);
        }
    }
    ids
}

/// An axis-aligned box `sx × sy × sz` centred at the origin, each face `n × n` quads and its own
/// UV island in a 3 × 2 layout (+X, −X, +Y on the top row, −Y, +Z, −Z below), hard edges, every
/// vertex in region 0.
pub fn box_mesh(sx: f32, sy: f32, sz: f32, n: usize) -> (BakeMesh, Mesh) {
    let h = Vec3::new(sx, sy, sz) * 0.5;
    // (outward normal, u axis, v axis) with u × v = −normal, so the corner order (i, j), (i, j+1),
    // (i+1, j+1), (i+1, j) turns counter-clockwise seen from outside.
    let faces: [(Vec3, Vec3, Vec3); 6] = [
        (Vec3::X, -Vec3::Z, -Vec3::Y),
        (-Vec3::X, Vec3::Z, -Vec3::Y),
        (Vec3::Y, Vec3::X, Vec3::Z),
        (-Vec3::Y, Vec3::X, -Vec3::Z),
        (Vec3::Z, Vec3::X, -Vec3::Y),
        (-Vec3::Z, -Vec3::X, -Vec3::Y),
    ];
    let mut w = Welder::new("box", false);
    for (f, &(nrm, u, v)) in faces.iter().enumerate() {
        let (cx, cy) = ((f % 3) as f32 / 3.0, (f / 3) as f32 / 2.0);
        let ids = patch(&mut w, n, |a, b| {
            let p = nrm * h + u * h * (2.0 * a - 1.0) + v * h * (2.0 * b - 1.0);
            // Island inside its cell with a 10 % margin; V of the island runs with b (down).
            (p, Vec2::new(cx + (0.1 + 0.8 * a) / 3.0, cy + f32::midpoint(0.1, 0.8 * b)))
        });
        for j in 0..n {
            for i in 0..n {
                let k = |di: usize, dj: usize| ids[(j + dj) * (n + 1) + i + di];
                w.b.face(&[k(0, 0), k(0, 1), k(1, 1), k(1, 0)], 0).unwrap();
            }
        }
    }
    let mesh = w.b.finish().unwrap();
    let regions = vec![0_u16; mesh.vertex_count()];
    (BakeMesh::new(&mesh, mesh.positions(), Some(&regions)).unwrap(), mesh)
}

/// A UV-sphere-like ellipsoid with radii `r`, `seg` columns and `rings` rows, one UV island
/// (seam at u = 0/1, smooth), triangles at the poles.
pub fn ellipsoid_mesh(r: Vec3, seg: usize, rings: usize) -> Mesh {
    let mut w = Welder::new("ellipsoid", true);
    let margin = 0.05;
    let mut ids = vec![0_u32; (seg + 1) * (rings + 1)];
    for j in 0..=rings {
        for i in 0..=seg {
            let (a, b) = (i as f32 / seg as f32, j as f32 / rings as f32);
            let theta = b * std::f32::consts::PI;
            let phi = a * std::f32::consts::TAU;
            // Poles collapse; every pole copy gets the UV of its column.
            let p = Vec3::new(theta.sin() * phi.cos() * r.x, theta.cos() * r.y, -theta.sin() * phi.sin() * r.z);
            let uv = Vec2::new(margin + (1.0 - 2.0 * margin) * a, margin + (1.0 - 2.0 * margin) * b);
            ids[j * (seg + 1) + i] = w.vertex(p, uv);
        }
    }
    for j in 0..rings {
        for i in 0..seg {
            let k = |di: usize, dj: usize| ids[(j + dj) * (seg + 1) + i + di];
            if j == 0 {
                w.b.face(&[k(0, 0), k(0, 1), k(1, 1)], 0).unwrap();
            } else if j == rings - 1 {
                w.b.face(&[k(0, 0), k(0, 1), k(1, 0)], 0).unwrap();
            } else {
                w.b.face(&[k(0, 0), k(0, 1), k(1, 1), k(1, 0)], 0).unwrap();
            }
        }
    }
    w.b.finish().unwrap()
}

/// A sphere of radius `r` as a [`BakeMesh`] (regions all 0).
pub fn sphere_mesh(r: f32, seg: usize, rings: usize) -> BakeMesh {
    let mesh = ellipsoid_mesh(Vec3::splat(r), seg, rings);
    let regions = vec![0_u16; mesh.vertex_count()];
    BakeMesh::new(&mesh, mesh.positions(), Some(&regions)).unwrap()
}

/// A capsule-ended tube along Y (a finger): radius `r`, straight length `len`, one UV island.
pub fn tube_mesh(r: f32, len: f32, seg: usize, rings: usize) -> Mesh {
    let mut w = Welder::new("tube", true);
    let margin = 0.05;
    let mut ids = vec![0_u32; (seg + 1) * (rings + 1)];
    for j in 0..=rings {
        for i in 0..=seg {
            let (a, b) = (i as f32 / seg as f32, j as f32 / rings as f32);
            // Arc length parameter over cap, body, cap.
            let total = std::f32::consts::PI * r + len;
            let s = b * total;
            let cap = std::f32::consts::FRAC_PI_2 * r;
            let (y, rad) = if s < cap {
                let t = s / r;
                (len * 0.5 + r * t.cos(), r * t.sin())
            } else if s > cap + len {
                let t = (s - cap - len) / r + std::f32::consts::FRAC_PI_2;
                (-len * 0.5 + r * t.cos(), r * t.sin())
            } else {
                (len * 0.5 - (s - cap), r)
            };
            let phi = a * std::f32::consts::TAU;
            let p = Vec3::new(rad * phi.cos(), y, -rad * phi.sin());
            let uv = Vec2::new(margin + (1.0 - 2.0 * margin) * a, margin + (1.0 - 2.0 * margin) * b);
            ids[j * (seg + 1) + i] = w.vertex(p, uv);
        }
    }
    for j in 0..rings {
        for i in 0..seg {
            let k = |di: usize, dj: usize| ids[(j + dj) * (seg + 1) + i + di];
            if j == 0 {
                w.b.face(&[k(0, 0), k(0, 1), k(1, 1)], 0).unwrap();
            } else if j == rings - 1 {
                w.b.face(&[k(0, 0), k(0, 1), k(1, 0)], 0).unwrap();
            } else {
                w.b.face(&[k(0, 0), k(0, 1), k(1, 1), k(1, 0)], 0).unwrap();
            }
        }
    }
    w.b.finish().unwrap()
}

/// Where the regression images go: `target/agents/bakers/images/`.
pub fn image_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/agents/bakers/images");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Writes an 8-bit image as a PNG into [`image_dir`].
pub fn write_png(name: &str, image: &Image) {
    let bytes = humanart_io::png::encode(image).unwrap();
    std::fs::write(image_dir().join(name), bytes).unwrap();
}

/// A false-colour ramp (dark blue, red, yellow, white) of a value in `[0, 1]`; `None` is grey.
pub fn ramp(v: Option<f32>) -> [u8; 3] {
    let Some(v) = v else { return [40, 40, 40] };
    let v = v.clamp(0.0, 1.0);
    let stops = [[0.0, 0.0, 0.25], [0.8, 0.05, 0.05], [1.0, 0.85, 0.1], [1.0, 1.0, 1.0]];
    let x = v * 3.0;
    let k = (x.floor() as usize).min(2);
    let f = x - k as f32;
    let c: Vec<u8> = (0..3).map(|i| ((stops[k][i] * (1.0 - f) + stops[k + 1][i] * f) * 255.0).round() as u8).collect();
    [c[0], c[1], c[2]]
}

/// An RGB image from per-texel colours.
pub fn rgb_image(width: u32, height: u32, texels: impl Iterator<Item = [u8; 3]>) -> Image {
    let samples: Vec<u8> = texels.flatten().collect();
    Image::new(Format::Png, width, height, 3, ColorSpace::Srgb, Samples::U8(samples)).unwrap()
}

/// Pastes single-channel maps side by side through `colour` (`None` for empty texels).
pub fn montage(maps: &[&super::BakeMap], colour: impl Fn(&[f32]) -> [u8; 3]) -> Image {
    let h = maps.iter().map(|m| m.height).max().unwrap();
    let w: u32 = maps.iter().map(|m| m.width).sum();
    let mut px = vec![[40_u8, 40, 40]; (w * h) as usize];
    let mut x0 = 0;
    for m in maps {
        for y in 0..m.height {
            for x in 0..m.width {
                let i = (y * m.width + x) as usize;
                if m.coverage[i] != super::Coverage::Empty {
                    px[(y * w + x0 + x) as usize] = colour(m.texel(i).unwrap());
                }
            }
        }
        x0 += m.width;
    }
    rgb_image(w, h, px.into_iter())
}

mod regression {
    use super::*;
    use crate::bake::dilate::{DilateMode, dilate, padding_for_levels};
    use crate::bake::occlusion::{OcclusionSettings, bake_occlusion};
    use crate::bake::raster::rasterize;
    use crate::bake::surface::bake_curvature;
    use crate::bake::thickness::{ThicknessSettings, bake_thickness};
    use crate::bake::{BakeMap, BakeTarget};

    /// The mean of channel `c` over the covered texels whose surface point passes `keep`.
    fn mean_where(
        mesh: &BakeMesh,
        surface: &crate::bake::raster::SurfaceMap,
        map: &BakeMap,
        c: usize,
        keep: impl Fn(Vec3) -> bool,
    ) -> f32 {
        let (mut sum, mut n) = (0.0, 0);
        for (i, t, b1, b2) in surface.covered() {
            if keep(mesh.point(t, b1, b2).unwrap().position) {
                sum += map.texel(i).unwrap()[c];
                n += 1;
            }
        }
        assert!(n > 0);
        sum / n as f32
    }

    /// `[AC-WP15-2]` on synthetic stand-ins: an ear-like flap (a 3 x 6 cm ellipsoid 5 mm thick),
    /// a finger (a 16 mm capsule) and a head-sized sphere. Writes
    /// `target/agents/bakers/images/thickness_synthetic.png` (translucency, false colour) and
    /// `thickness_synthetic_mm.png` (thickness, 0 to 20 mm).
    #[test]
    fn ac_wp15_2_thickness_on_synthetic_shapes() {
        let flap_mesh = ellipsoid_mesh(Vec3::new(0.015, 0.03, 0.0025), 48, 32);
        let flap = BakeMesh::new(&flap_mesh, flap_mesh.positions(), None).unwrap();
        let finger_mesh = tube_mesh(0.008, 0.05, 32, 48);
        let finger = BakeMesh::new(&finger_mesh, finger_mesh.positions(), None).unwrap();
        let head = sphere_mesh(0.09, 32, 16);
        let settings = ThicknessSettings { rays: 64, ..ThicknessSettings::default() };
        let target = BakeTarget::new(0, 160, 160);
        let mut maps = Vec::new();
        let mut means = Vec::new();
        for mesh in [&flap, &finger, &head] {
            let surface = rasterize(mesh, &target).unwrap();
            let mut map = bake_thickness(mesh, &surface, &settings).unwrap();
            let middle = mean_where(mesh, &surface, &map, 0, |p| p.x.abs() < 0.004 && p.y.abs() < 0.008);
            means.push(middle);
            dilate(&mut map, padding_for_levels(4), DilateMode::Average).unwrap();
            maps.push(map);
        }
        // Flap middle: the 5 mm slab seen through the 30 degree cone (1.07 x), plus curvature.
        assert!((0.0045..0.0065).contains(&means[0]), "flap {}", means[0]);
        // Finger middle: the mean chord of a 16 mm circle in the cone (0.93 x 16 mm).
        assert!((0.013..0.0165).contains(&means[1]), "finger {}", means[1]);
        // The head is thicker than the reach: every ray misses.
        assert!((means[2] - settings.max_distance).abs() < 1e-3, "head {}", means[2]);
        let refs: Vec<&BakeMap> = maps.iter().collect();
        write_png("thickness_synthetic.png", &montage(&refs, |v| ramp(Some(v[1]))));
        write_png("thickness_synthetic_mm.png", &montage(&refs, |v| ramp(Some(v[0] / 0.02))));
    }

    /// Bilinear lookup with clamp-to-edge of a 1-channel float image at `uv`.
    fn bilinear(image: &Image, u: f32, v: f32) -> f32 {
        let data = image.as_f32().unwrap();
        let (w, h) = (image.width as i32, image.height as i32);
        let (x, y) = (u * w as f32 - 0.5, v * h as f32 - 0.5);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let at = |i: i32, j: i32| data[(j.clamp(0, h - 1) * w + i.clamp(0, w - 1)) as usize];
        let (i, j) = (x0 as i32, y0 as i32);
        let top = at(i, j) * (1.0 - fx) + at(i + 1, j) * fx;
        let bottom = at(i, j + 1) * (1.0 - fx) + at(i + 1, j + 1) * fx;
        top * (1.0 - fy) + bottom * fy
    }

    /// `[AC-WP15-1]` across a real UV seam: a function continuous on the surface (`2 + y / r`)
    /// baked on a sphere whose seam runs along u = 0.05 | 0.95. Through the mip chain of the
    /// exporter (`humanart_io::mip`), bilinear lookups at both images of each seam point read the
    /// surface value at levels 0 to 3 once padded (no line along the seam), and do not without
    /// padding (the background bleeds in).
    #[test]
    fn ac_wp15_1_no_visible_seam_at_mips_0_to_3() {
        let r = 0.03;
        let m = ellipsoid_mesh(Vec3::splat(r), 48, 24);
        let mesh = BakeMesh::new(&m, m.positions(), None).unwrap();
        let surface = rasterize(&mesh, &BakeTarget::new(0, 256, 256)).unwrap();
        let bare = surface.bake(&mesh, 1, 0.0, |_, _, p, out| out[0] = 2.0 + p.position.y / r).unwrap();
        let mut padded = bare.clone();
        dilate(&mut padded, padding_for_levels(4), DilateMode::Average).unwrap();
        let worst = |map: &BakeMap| -> Vec<f32> {
            let mips = humanart_io::mip::generate(&map.to_image(ColorSpace::Data).unwrap(), 4).unwrap();
            mips.iter()
                .map(|level| {
                    // Seam points away from the poles.
                    (0..64)
                        .map(|k| {
                            let v = 0.15 + 0.7 * k as f32 / 63.0;
                            // The sphere's height at this V (rings run from the top pole down).
                            let truth = 2.0 + (std::f32::consts::PI * (v - 0.05) / 0.9).cos();
                            (bilinear(level, 0.05, v) - truth).abs().max((bilinear(level, 0.95, v) - truth).abs())
                        })
                        .fold(0.0, f32::max)
                })
                .collect()
        };
        let (with, without) = (worst(&padded), worst(&bare));
        println!("largest seam error per level, padded {with:?}, bare {without:?}");
        // A level-3 texel spans 8 / 230 of the island's height, where y / r changes by < 0.11;
        // the faceted sphere is within 0.3 % of the round one.
        assert!(with.iter().all(|&d| d < 0.05), "{with:?}");
        assert!(without.iter().all(|&d| d > 0.5), "{without:?}");
    }

    /// Occlusion, cavity and curvature of the flap and finger, for looking at
    /// (`occlusion_synthetic.png`: AO | cavity per shape; `curvature_synthetic.png`).
    #[test]
    fn bake_02_03_images_on_synthetic_shapes() {
        let flap_mesh = ellipsoid_mesh(Vec3::new(0.015, 0.03, 0.0025), 48, 32);
        let flap = BakeMesh::new(&flap_mesh, flap_mesh.positions(), None).unwrap();
        let finger_mesh = tube_mesh(0.008, 0.05, 32, 48);
        let finger = BakeMesh::new(&finger_mesh, finger_mesh.positions(), None).unwrap();
        let target = BakeTarget::new(0, 128, 128);
        let mut ao = Vec::new();
        let mut curvature = Vec::new();
        for mesh in [&flap, &finger] {
            let surface = rasterize(mesh, &target).unwrap();
            let occ = bake_occlusion(mesh, &surface, &OcclusionSettings { rays: 32, ..OcclusionSettings::default() })
                .unwrap();
            ao.push(occ.occlusion.channel(0).unwrap());
            ao.push(occ.occlusion.channel(1).unwrap());
            curvature.push(bake_curvature(mesh, &surface).unwrap());
        }
        let refs: Vec<&BakeMap> = ao.iter().collect();
        write_png(
            "occlusion_synthetic.png",
            &montage(&refs, |v| {
                let g = (v[0] * 255.0).round() as u8;
                [g, g, g]
            }),
        );
        // Convex shapes: nothing occludes them.
        assert!(
            ao.iter().all(|m| m
                .data
                .iter()
                .zip(&m.coverage)
                .all(|(&v, c)| *c == crate::bake::Coverage::Empty || v == 1.0))
        );
        let refs: Vec<&BakeMap> = curvature.iter().collect();
        write_png("curvature_synthetic.png", &montage(&refs, |v| ramp(Some(0.5 + v[0] / 400.0))));
    }
}

/// `[AC-WP15-2]` on the generated template (`humanart_model::generate`, the in-house template):
/// GPU thickness of the face tile and the hands-and-feet tile, statistics per anatomical group
/// and the regression images `thickness_template_face.png` / `thickness_template_hands.png`
/// (translucency, false colour) in `target/agents/bakers/images/`.
mod template {
    use humanart_model::generate::{assemble, body_surface, cage::Cage, parts, uv};

    use super::*;
    use crate::bake::BakeTarget;
    use crate::bake::dilate::padding_for_levels;
    use crate::bake::gpu::tests::{baker, gpu};
    use crate::bake::gpu::{GpuMesh, GpuTarget};
    use crate::bake::thickness::ThicknessSettings;

    /// The median of `v` (sorted in place).
    fn median(v: &mut [f32]) -> f32 {
        v.sort_by(f32::total_cmp);
        v[v.len() / 2]
    }

    #[test]
    fn ac_wp15_2_thickness_on_the_template() {
        let surface = body_surface().unwrap();
        let parts = parts::build(&surface.anatomy, &surface.body, &surface.targets).unwrap();
        let pieces: Vec<&Cage> = std::iter::once(surface.mesh()).chain(parts.iter().map(|p| &p.mesh)).collect();
        let layout = uv::layout(&pieces).unwrap();
        let asm = assemble::assemble(&surface, &parts, &layout).unwrap();
        let regions = &asm.regions;
        let mesh = BakeMesh::new(&asm.mesh, asm.mesh.positions(), Some(&regions.vertex_regions)).unwrap();
        let t0 = std::time::Instant::now();
        let curvature = crate::bake::surface::vertex_mean_curvature(&mesh);
        println!(
            "template: {} vertices, {} triangles; CPU vertex curvature {:.1} ms",
            curvature.len(),
            mesh.triangle_count(),
            t0.elapsed().as_secs_f64() * 1e3
        );
        let gpu = gpu();
        let gm = GpuMesh::upload(gpu, &mesh).unwrap();
        let settings = ThicknessSettings { rays: 64, ..ThicknessSettings::default() };
        let groups: [(&str, &[&str]); 7] = [
            ("ear", &["ear-L", "ear-R"]),
            ("nostril", &["nostril-L", "nostril-R"]),
            ("nose", &["nose"]),
            (
                "finger",
                &[
                    "hand-L.index",
                    "hand-L.middle",
                    "hand-L.ring",
                    "hand-L.little",
                    "hand-R.index",
                    "hand-R.middle",
                    "hand-R.ring",
                    "hand-R.little",
                ],
            ),
            ("thumb", &["hand-L.thumb", "hand-R.thumb"]),
            ("palm", &["hand-L.palm", "hand-R.palm"]),
            ("head", &["head"]),
        ];
        let ids: Vec<Vec<u16>> =
            groups.iter().map(|(_, names)| names.iter().filter_map(|n| regions.find(n)).collect()).collect();
        let mut values: Vec<Vec<f32>> = vec![Vec::new(); groups.len()];
        for (tile, name) in [(0_u16, "face"), (4, "hands")] {
            let target = BakeTarget::new(tile, 1024, 1024);
            let gt = GpuTarget::new(gpu, &target).unwrap();
            let surf = baker().surface_blocking(gpu, &gm, &gt).unwrap();
            let t0 = std::time::Instant::now();
            let map = baker().bake_thickness_blocking(gpu, &mesh, &gm, &gt, &settings, padding_for_levels(4)).unwrap();
            println!(
                "{name} tile 1024^2, 64 rays, 16 padding passes, readback: {:.0} ms",
                t0.elapsed().as_secs_f64() * 1e3
            );
            for (i, t, b1, b2) in surf.covered() {
                let p = mesh.point(t, b1, b2).unwrap();
                for (g, list) in ids.iter().enumerate() {
                    if list.iter().any(|&r| regions.is_within(p.region, r)) {
                        values[g].push(map.texel(i).unwrap()[0]);
                    }
                }
            }
            write_png(&format!("thickness_template_{name}.png"), &montage(&[&map], |v| ramp(Some(v[1]))));
            write_png(&format!("thickness_template_{name}_mm.png"), &montage(&[&map], |v| ramp(Some(v[0] / 0.025))));
        }
        // Occlusion and cavity of the face tile, for looking at.
        let gt = GpuTarget::new(gpu, &BakeTarget::new(0, 1024, 1024)).unwrap();
        let settings = crate::bake::occlusion::OcclusionSettings { rays: 64, ..Default::default() };
        let (occlusion, _) = baker().bake_occlusion_blocking(gpu, &mesh, &gm, &gt, &settings, 16).unwrap();
        let grey = |v: f32| {
            let g = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            [g, g, g]
        };
        write_png("occlusion_template_face.png", &montage(&[&occlusion], |v| grey(v[0])));
        write_png("cavity_template_face.png", &montage(&[&occlusion], |v| grey(v[1])));
        let mut medians = Vec::new();
        for ((name, _), v) in groups.iter().zip(&mut values) {
            let n = v.len();
            let m = if n > 0 { median(v) } else { f32::NAN };
            println!("{name}: {n} texels, median thickness {:.2} mm", m * 1000.0);
            assert!(n > 1000, "{name}: {n} texels");
            medians.push(m * 1000.0);
        }
        // Plausible anatomy (adult ear 2-6 mm, finger 14-20 mm, thumb up to 25 mm), with margins for
        // the template's still-changing shape; and thin features thinner than thick ones.
        let (ear, nostril, nose, finger) = (medians[0], medians[1], medians[2], medians[3]);
        let (thumb, palm, head) = (medians[4], medians[5], medians[6]);
        assert!((2.0..8.0).contains(&ear), "ear {ear}");
        assert!(nostril < 12.0, "nostril {nostril}");
        assert!((12.0..22.0).contains(&finger), "finger {finger}");
        assert!((14.0..28.0).contains(&thumb), "thumb {thumb}");
        assert!(ear < nostril && nostril < finger && finger < palm && finger < nose && palm <= head, "{medians:?}");
    }
}
