//! Tests of the LOD generator `[MOD-05]` `[AC-WP28-1]` on synthetic meshes: cube-spheres with UV
//! seams on every cube edge, two regions, a protected loop, skin weights, global and local deltas
//! and landmarks; flat grids with borders; random degenerate soups.

use humanart_core::rng::Rng;
use humanart_math::{Vec2, Vec3};

use super::*;
use crate::delta::{DeltaBasesBuilder, DeltaKind};
use crate::expression::ExpressionSet;
use crate::landmark::LandmarkGroup;
use crate::mesh::MeshBuilder;
use crate::names::Side;
use crate::region::Region;
use crate::skeleton::{JointRegressor, Skeleton, simple_joint};
use crate::template::{TemplateInfo, TemplateParts, UvTile};
use crate::validate::{Check, validate};

/// The six cube faces: (normal axis, sign, u axis, v axis) with `u × v = sign * axis`.
const FACES: [(usize, f32, usize, usize); 6] =
    [(0, 1.0, 1, 2), (0, -1.0, 2, 1), (1, 1.0, 2, 0), (1, -1.0, 0, 2), (2, 1.0, 0, 1), (2, -1.0, 1, 0)];

/// A cube-sphere of radius `radius` around `centre` with `n` quads per cube-face side, each cube
/// face its own UV island of a 3x2 atlas (seams on every cube edge). Returns the builder's mesh.
fn add_cube_sphere(b: &mut MeshBuilder, name: &str, n: usize, radius: f32, centre: Vec3, bump: f32) {
    b.begin_part(name, true).unwrap();
    let side = n + 1;
    let mut first = vec![u32::MAX; side * side * side];
    for (f, &(axis, sign, ua, va)) in FACES.iter().enumerate() {
        let (cx, cy) = ((f % 3) as f32 / 3.0, (f / 3) as f32 / 2.0);
        let mut ids = vec![0_u32; side * side];
        for j in 0..side {
            for i in 0..side {
                let mut lat = [0_usize; 3];
                lat[axis] = if sign > 0.0 { n } else { 0 };
                lat[ua] = i;
                lat[va] = j;
                let key = (lat[2] * side + lat[1]) * side + lat[0];
                let q = Vec3::new(lat[0] as f32, lat[1] as f32, lat[2] as f32) * (2.0 / n as f32) - Vec3::ONE;
                let dir = q.normalize();
                // A smooth bump on +Z, upper half (the "eye").
                let e = (dir - Vec3::new(0.0, 0.45, 0.89).normalize()).length();
                let r = radius * (1.0 + bump * (-(e * e) / 0.02).exp());
                let p = centre + dir * r;
                let (s, t) = (i as f32 / n as f32, j as f32 / n as f32);
                let uv = Vec2::new(cx + 0.01 + s * (1.0 / 3.0 - 0.02), cy + 0.01 + (1.0 - t) * (0.5 - 0.02));
                ids[j * side + i] = if first[key] == u32::MAX {
                    let v = b.vertex(p, uv);
                    first[key] = v;
                    v
                } else {
                    b.duplicate(first[key], uv, true).unwrap()
                };
            }
        }
        for j in 0..n {
            for i in 0..n {
                let c =
                    [ids[j * side + i], ids[j * side + i + 1], ids[(j + 1) * side + i + 1], ids[(j + 1) * side + i]];
                b.face(&c, 0).unwrap();
            }
        }
    }
}

/// The loop of grid vertices at Chebyshev distance `k` around grid point `(ci, cj)` of the +Z face.
fn plus_z_loop(mesh: &Mesh, centre: Vec3, n: usize, ci: usize, cj: usize, k: usize) -> Vec<u32> {
    let mut ring = Vec::new();
    let (i0, i1, j0, j1) = (ci - k, ci + k, cj - k, cj + k);
    for i in i0..i1 {
        ring.push((i, j0));
    }
    for j in j0..j1 {
        ring.push((i1, j));
    }
    for i in (i0 + 1..=i1).rev() {
        ring.push((i, j1));
    }
    for j in (j0 + 1..=j1).rev() {
        ring.push((i0, j));
    }
    let target = |i: usize, j: usize| {
        let q = Vec3::new(i as f32, j as f32, n as f32) * (2.0 / n as f32) - Vec3::ONE;
        q.normalize()
    };
    ring.iter()
        .map(|&(i, j)| {
            let d = target(i, j);
            (0..mesh.vertex_count() as u32)
                .filter(|&v| mesh.position_keys()[v as usize] == v)
                .min_by(|&a, &b| {
                    let da = (mesh.positions()[a as usize] - centre).normalize().distance(d);
                    let db = (mesh.positions()[b as usize] - centre).normalize().distance(d);
                    da.total_cmp(&db)
                })
                .unwrap()
        })
        .collect()
}

/// A complete template on a cube-sphere: regions upper/lower, a two-joint skin, a global
/// semantic delta, a local expression delta on the bump, landmarks.
fn sphere_template(n: usize) -> TemplateParts {
    let mut b = MeshBuilder::new();
    add_cube_sphere(&mut b, "head", n, 0.1, Vec3::new(0.0, 1.6, 0.0), 0.08);
    let mesh = b.finish().unwrap();
    let centre_y = 1.6;
    let up = |p: Vec3| p.y >= centre_y;
    let regions = RegionMap {
        regions: vec![
            Region { name: "head".into(), parent: None, side: Side::Center },
            Region { name: "lower".into(), parent: Some(0), side: Side::Center },
            Region { name: "upper".into(), parent: Some(0), side: Side::Center },
        ],
        vertex_regions: mesh.positions().iter().map(|&p| if up(p) { 2 } else { 1 }).collect(),
        soft: None,
    };
    let lists: Vec<Vec<(u16, f32)>> = mesh
        .positions()
        .iter()
        .map(|p| {
            let w = ((p.y - centre_y) / 0.1 * 0.5 + 0.5).clamp(0.0, 1.0);
            vec![(0, 1.0 - w), (1, w)]
        })
        .collect();
    let skin = SkinWeights::from_lists(4, lists.iter().map(Vec::as_slice), 0).unwrap();
    let mut db = DeltaBasesBuilder::new();
    let wide: Vec<(u32, Vec3)> = (0..mesh.vertex_count() as u32)
        .map(|v| {
            let p = mesh.positions()[v as usize];
            (v, Vec3::new(p.x * 0.05, 0.0, 0.0))
        })
        .collect();
    db.add("wide", DeltaKind::Semantic, 0, &wide).unwrap();
    let eye = Vec3::new(0.0, 0.45, 0.89).normalize();
    let blink: Vec<(u32, Vec3)> = (0..mesh.vertex_count() as u32)
        .filter_map(|v| {
            let p = mesh.positions()[v as usize];
            let e = ((p - Vec3::new(0.0, centre_y, 0.0)).normalize() - eye).length();
            (up(p) && e < 0.35).then_some((v, Vec3::new(0.0, -0.004, 0.0) * (1.0 - e / 0.35)))
        })
        .collect();
    db.add("blink", DeltaKind::Expression, 2, &blink).unwrap();
    let deltas = db.finish().unwrap();
    let landmarks = (0..4)
        .map(|i| {
            let f = (i * 37 + 5) % mesh.face_count();
            let c = mesh.face(f);
            Landmark {
                name: format!("lm{i}"),
                group: if i % 2 == 0 { LandmarkGroup::Face } else { LandmarkGroup::Body },
                vertices: [c[0], c[1], c[2]],
                weights: [0.2, 0.3, 0.5],
            }
        })
        .collect();
    TemplateParts {
        info: TemplateInfo { name: "sphere".into(), ..TemplateInfo::default() },
        tiles: vec![UvTile { name: "head".into(), resolution: 1024, target_texels_per_mm: 0.0 }],
        regions,
        landmarks,
        skeleton: Skeleton {
            joints: vec![
                simple_joint("neck", None, Vec3::new(0.0, 1.5, 0.0)),
                simple_joint("head", Some(0), Vec3::new(0.0, 0.1, 0.0)),
            ],
            regressor: JointRegressor::empty(deltas.len()),
        },
        skin,
        deltas,
        expressions: ExpressionSet::default(),
        lods: Vec::new(),
        mesh,
    }
}

/// A template made from one generated level, with every attribute transferred.
fn level_as_template(parts: &TemplateParts, g: &GeneratedLod) -> Template {
    let l = &g.level;
    let regions = RegionMap {
        regions: parts.regions.regions.clone(),
        vertex_regions: l.transfer_regions(&parts.regions.vertex_regions).unwrap(),
        soft: None,
    };
    Template::new(TemplateParts {
        info: parts.info.clone(),
        mesh: l.mesh.clone(),
        tiles: parts.tiles.clone(),
        regions,
        landmarks: g.landmarks.clone(),
        skeleton: parts.skeleton.clone(),
        skin: l.transfer_skin(&parts.skin).unwrap(),
        deltas: l.transfer_deltas(&parts.deltas).unwrap(),
        expressions: parts.expressions.clone(),
        lods: Vec::new(),
    })
    .unwrap()
}

fn generate(parts: &TemplateParts, loops: &[ProtectedLoop], settings: &LodSettings) -> Vec<GeneratedLod> {
    let t = Template::new(parts.clone()).unwrap();
    generate_lods(&LodInput::from_template(&t, loops), settings).unwrap()
}

/// Welded edges whose two triangles use different wedges at an end (UV seams).
fn seam_vertices(mesh: &Mesh) -> Vec<u32> {
    let tri = mesh.triangulate();
    let keys = mesh.position_keys();
    let mut edges: Vec<(u32, u32, u32, u32)> = Vec::new();
    for t in tri.indices.as_chunks::<3>().0 {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            let (ka, kb) = (keys[a as usize], keys[b as usize]);
            if ka < kb { edges.push((ka, kb, a, b)) } else { edges.push((kb, ka, b, a)) }
        }
    }
    edges.sort_unstable();
    let mut out: Vec<u32> = Vec::new();
    for g in edges.chunk_by(|x, y| (x.0, x.1) == (y.0, y.1)) {
        if g.len() == 2 && (g[0].2 != g[1].2 || g[0].3 != g[1].3) {
            out.extend([g[0].0, g[0].1]);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

#[test]
fn mod_05_sphere_levels_pass_template_validation() {
    let parts = sphere_template(24);
    let base_tris = parts.mesh.triangulate().len();
    let levels = generate(&parts, &[], &LodSettings::default());
    assert_eq!(levels.len(), 4);
    let mut with_lods = parts.clone();
    with_lods.lods = levels.iter().map(|g| g.level.clone()).collect();
    let t = Template::new(with_lods).unwrap();
    let report = validate(&t);
    assert!(report.passed(), "{report}");
    assert_eq!(report.count(Check::LodError), 0);
    let mut prev = base_tris;
    for (g, v) in levels.iter().zip(&report.lods) {
        let r = &g.report;
        println!("{r}");
        // The validator's measure is bounded by the declared error, the sources are exact.
        assert!(v.surface_error <= g.level.declared_error + 1e-7, "{v:?}");
        assert_eq!(v.source_error, 0.0);
        assert!(r.triangles < prev, "{r}");
        prev = r.triangles;
        // [AC-WP28-1] The measured errors stay within the stated bounds.
        assert!(r.geometric_error <= r.geometric_budget, "{r}");
        assert!(r.delta_error <= r.delta_budget && r.skin_error <= r.skin_budget, "{r}");
        assert!(r.geometric_error < 0.01 && r.landmark_error < 0.01, "{r}");
    }
    assert!(levels[0].report.triangle_ratio <= 0.5 + 1e-3, "{}", levels[0].report);
    assert!(levels[3].report.triangle_ratio < 0.2, "{}", levels[3].report);
}

#[test]
fn mod_05_each_level_is_a_valid_template_with_seams_and_regions() {
    let parts = sphere_template(16);
    let levels = generate(&parts, &[], &LodSettings::default());
    let seams0 = seam_vertices(&parts.mesh);
    let lod0_regions = &parts.regions.vertex_regions;
    for g in &levels {
        let t = level_as_template(&parts, g);
        let report = validate(&t);
        // Watertight, manifold, UVs in range, unflipped, not overlapping, one tile per vertex,
        // skin normalised and seam-consistent, deltas inside their regions, landmarks valid.
        assert!(report.passed(), "LOD{}: {report}", g.report.level);
        assert!(report.parts.iter().all(crate::validate::PartReport::watertight), "{report}");
        // Seams: every seam vertex of the level was a seam vertex of LOD0 (no new cuts), and the
        // eight cube corners (seam junctions) survive.
        let lod = &g.level.mesh;
        let src = |v: u32| g.level.sources[v as usize][0];
        for v in seam_vertices(lod) {
            assert!(seams0.binary_search(&parts.mesh.position_keys()[src(v) as usize]).is_ok());
        }
        let corners = lod
            .positions()
            .iter()
            .enumerate()
            .filter(|&(v, p)| {
                lod.position_keys()[v] as usize == v && {
                    let d = (*p - Vec3::new(0.0, 1.6, 0.0)).normalize();
                    (d.x.abs() - d.y.abs()).abs() < 1e-4 && (d.y.abs() - d.z.abs()).abs() < 1e-4
                }
            })
            .count();
        assert_eq!(corners, 8, "LOD{}", g.report.level);
        // Regions transfer exactly.
        for (v, &r) in t.regions().vertex_regions.iter().enumerate() {
            assert_eq!(r, lod0_regions[src(v as u32) as usize]);
        }
        assert!(g.report.feature_error <= g.report.geometric_budget, "{}", g.report);
    }
}

#[test]
fn mod_05_protected_loop_is_kept() {
    let n = 24;
    let parts = sphere_template(n);
    let lp = ProtectedLoop {
        name: "eyelid".into(),
        vertices: plus_z_loop(&parts.mesh, Vec3::new(0.0, 1.6, 0.0), n, n / 2, 3 * n / 4, 3),
        lock_through: 1,
        keep_through: 3,
    };
    let keys: Vec<u32> = lp.vertices.iter().map(|&v| parts.mesh.position_keys()[v as usize]).collect();
    let levels = generate(&parts, std::slice::from_ref(&lp), &LodSettings::default());
    for g in &levels {
        let lod = &g.level.mesh;
        let kept: Vec<u32> = keys
            .iter()
            .filter_map(|&k| g.level.sources.iter().position(|s| s[0] == k).map(|v| lod.position_keys()[v]))
            .collect();
        let level = g.report.level;
        if level == 1 {
            assert_eq!(kept.len(), keys.len(), "LOD1 keeps every loop vertex");
        }
        if level <= 3 {
            assert!(kept.len() >= 3);
            // Consecutive kept loop vertices share an edge of the level: the loop is a line of it.
            let topo = lod.topology();
            for i in 0..kept.len() {
                let (a, b) = (kept[i], kept[(i + 1) % kept.len()]);
                assert!(topo.edges.binary_search(&[a.min(b), a.max(b)]).is_ok(), "LOD{level}");
            }
        }
    }
    // Without protection, the same level drops loop vertices.
    let free = generate(&parts, &[], &LodSettings::default());
    let kept_free = keys.iter().filter(|&&k| free[0].level.sources.iter().any(|s| s[0] == k)).count();
    assert!(kept_free < keys.len());
}

#[test]
fn mod_05_delta_budget_bounds_the_delta_error() {
    let parts = sphere_template(20);
    let mut tight = LodSettings::default();
    for l in &mut tight.levels {
        l.max_delta_error = 0.0004;
    }
    let mut loose = tight.clone();
    for l in &mut loose.levels {
        l.max_delta_error = f32::INFINITY;
        l.max_geometric_error = f32::INFINITY;
    }
    loose.delta_weight = 0.0;
    let t = generate(&parts, &[], &tight);
    let l = generate(&parts, &[], &loose);
    for (a, b) in t.iter().zip(&l) {
        println!("tight {}\nloose {}", a.report, b.report);
        assert!(a.report.delta_error <= 0.0004, "{}", a.report);
        assert!(a.report.delta_error <= b.report.delta_error + 1e-6);
    }
    assert!(t[3].report.delta_error < l[3].report.delta_error);
    // The measured error names the local expression as the worst component.
    let blink = parts.deltas.find("blink");
    assert_eq!(l[3].report.delta_worst_component, blink);
}

#[test]
fn mod_05_measured_errors_are_enforced_by_refusing_collapses() {
    // Budgets where the run's estimates let collapses through that the measurement refuses.
    let parts = sphere_template(24);
    let mut s = LodSettings::default();
    for l in &mut s.levels {
        l.max_delta_error = 0.0003;
        l.max_geometric_error = l.max_geometric_error.min(0.002);
    }
    let levels = generate(&parts, &[], &s);
    for g in &levels {
        let r = &g.report;
        println!("{r}");
        assert!(r.delta_error <= 0.0003 && r.geometric_error <= r.geometric_budget, "{r}");
        assert!(r.feature_error <= r.geometric_budget, "{r}");
    }
    assert!(levels.iter().any(|g| g.report.refused_collapses > 0));
    assert!(levels.iter().all(|g| g.report.undone_collapses == 0));
}

#[test]
fn mod_05_flat_grid_reduces_without_error() {
    let mesh = crate::mesh::fixtures::grid(32);
    let input =
        LodInput { mesh: &mesh, regions: None, skin: None, deltas: None, landmarks: &[], loops: &[], locked: &[] };
    let levels = generate_lods(&input, &LodSettings::default()).unwrap();
    let last = &levels[3];
    assert!(last.report.geometric_error < 1e-6, "{}", last.report);
    assert!(last.report.triangle_ratio <= 0.05 + 1e-3, "{}", last.report);
    // The border stays the unit square: every border vertex keeps x or y at 0 or 1, the corners stay.
    let lod = &last.level.mesh;
    let on_border =
        |p: Vec3| p.x.abs() < 1e-6 || p.y.abs() < 1e-6 || (p.x - 1.0).abs() < 1e-6 || (p.y - 1.0).abs() < 1e-6;
    let topo = lod.topology();
    for (i, e) in topo.edges.iter().enumerate() {
        if topo.edge_faces.row(i).len() == 1 {
            let (a, b) = (lod.positions()[e[0] as usize], lod.positions()[e[1] as usize]);
            assert!(on_border(a) && on_border(b));
        }
    }
    for c in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::new(1.0, 1.0, 0.0)] {
        assert!(lod.positions().iter().any(|&p| p.distance(c) < 1e-6), "{c:?}");
    }
}

#[test]
fn mod_05_output_is_deterministic() {
    let parts = sphere_template(12);
    let a = generate(&parts, &[], &LodSettings::default());
    let b = generate(&parts, &[], &LodSettings::default());
    assert_eq!(a, b);
    let mut p = parts.clone();
    p.lods = a.iter().map(|g| g.level.clone()).collect();
    let t = Template::new(p).unwrap();
    assert_eq!(crate::write_package(&t).unwrap(), crate::write_package(&t.clone()).unwrap());
}

#[test]
fn mod_05_two_parts_keep_their_faces() {
    let mut b = MeshBuilder::new();
    add_cube_sphere(&mut b, "head", 10, 0.1, Vec3::ZERO, 0.0);
    add_cube_sphere(&mut b, "eye", 4, 0.01, Vec3::new(0.03, 0.02, 0.09), 0.0);
    let mesh = b.finish().unwrap();
    let input =
        LodInput { mesh: &mesh, regions: None, skin: None, deltas: None, landmarks: &[], loops: &[], locked: &[] };
    let mut s = LodSettings::default();
    for l in &mut s.levels {
        l.max_geometric_error = f32::INFINITY;
        l.triangle_ratio *= 0.1;
    }
    for g in generate_lods(&input, &s).unwrap() {
        let parts = g.level.mesh.parts();
        assert_eq!(parts.len(), 2);
        assert!(parts.iter().all(|p| p.face_count >= 4 && p.closed));
    }
}

#[test]
fn mod_05_bad_input_and_settings_are_refused() {
    let mesh = crate::mesh::fixtures::grid(4);
    let ok = LodInput { mesh: &mesh, regions: None, skin: None, deltas: None, landmarks: &[], loops: &[], locked: &[] };
    let s = LodSettings::default();
    generate_lods(&ok, &s).unwrap();
    for bad in [
        LodSettings { levels: Vec::new(), ..s.clone() },
        LodSettings { levels: vec![s.levels[0]; 5], ..s.clone() },
        LodSettings { uv_weight: f32::NAN, ..s.clone() },
        LodSettings { min_normal_dot: 2.0, ..s.clone() },
        LodSettings { levels: vec![s.levels[1], s.levels[0]], ..s.clone() },
        LodSettings { levels: vec![LevelSettings { triangle_ratio: 0.0, ..s.levels[0] }], ..s.clone() },
        LodSettings { levels: vec![LevelSettings { max_delta_error: -1.0, ..s.levels[0] }], ..s.clone() },
        LodSettings { levels: vec![LevelSettings { hysteresis: 0.7, ..s.levels[0] }], ..s.clone() },
    ] {
        assert!(generate_lods(&ok, &bad).is_err(), "{bad:?}");
    }
    let locked = [99];
    assert!(generate_lods(&LodInput { locked: &locked, ..ok }, &s).is_err());
    let lp = |v: Vec<u32>| ProtectedLoop { name: "l".into(), vertices: v, lock_through: 1, keep_through: 2 };
    for l in [lp(vec![0, 1]), lp(vec![0, 1, 99]), lp(vec![0, 1, 24]), lp(vec![0, 1, 6, 0])] {
        let loops = [l];
        assert!(generate_lods(&LodInput { loops: &loops, ..ok }, &s).is_err(), "{:?}", loops[0].vertices);
    }
    let lm =
        [Landmark { name: "x".into(), group: LandmarkGroup::Face, vertices: [0, 1, 99], weights: [1.0, 0.0, 0.0] }];
    assert!(generate_lods(&LodInput { landmarks: &lm, ..ok }, &s).is_err());
    let regions = RegionMap {
        regions: vec![Region { name: "a".into(), parent: None, side: Side::Center }],
        vertex_regions: vec![0; 3],
        soft: None,
    };
    assert!(generate_lods(&LodInput { regions: Some(&regions), ..ok }, &s).is_err());
}

/// A random triangle soup over few points: duplicates, zero-area triangles, non-manifold edges,
/// bow-ties, inconsistent winding.
fn soup(rng: &mut Rng) -> Option<Mesh> {
    let mut b = MeshBuilder::new();
    let points = 3 + rng.index(30);
    let (mut ids, mut keys) = (Vec::new(), Vec::new());
    for _ in 0..points {
        let p = match rng.index(4) {
            0 => Vec3::ZERO,
            1 => Vec3::new(rng.f32(), 0.0, 0.0),
            _ => Vec3::new(rng.f32(), rng.f32(), rng.f32() * 0.1),
        };
        let v = b.vertex(p, Vec2::new(rng.f32(), rng.f32()));
        ids.push(v);
        keys.push(v);
        if rng.index(5) == 0 {
            ids.push(b.duplicate(v, Vec2::new(rng.f32(), rng.f32()), rng.index(2) == 0).ok()?);
            keys.push(v);
        }
    }
    for _ in 0..=rng.index(80) {
        let c = [rng.index(ids.len()), rng.index(ids.len()), rng.index(ids.len())];
        if keys[c[0]] == keys[c[1]] || keys[c[1]] == keys[c[2]] || keys[c[0]] == keys[c[2]] {
            continue;
        }
        let c = c.map(|i| ids[i]);
        if rng.index(7) == 0 {
            b.begin_part(&format!("p{}", rng.index(1000)), rng.index(2) == 0).ok()?;
        }
        b.face(&c, 0).ok()?;
    }
    b.finish().ok()
}

#[test]
fn mod_05_bow_tie_and_non_manifold_vertices_are_locked() {
    // Two 6x6 grids touching at one corner (a bow-tie vertex) and a fin triangle on an interior
    // edge of the first (a non-manifold edge).
    let n = 6;
    let mut b = MeshBuilder::new();
    let grid = |b: &mut MeshBuilder, origin: Vec3, corner: Option<u32>| -> Vec<u32> {
        let mut ids = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                let p = origin + Vec3::new(i as f32, j as f32, 0.0) / n as f32;
                ids.push(match corner {
                    Some(c) if i == 0 && j == 0 => c,
                    _ => b.vertex(p, Vec2::new(p.x * 0.5, 1.0 - p.y * 0.5)),
                });
            }
        }
        for j in 0..n {
            for i in 0..n {
                let at = |i: usize, j: usize| ids[j * (n + 1) + i];
                b.face(&[at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)], 0).unwrap();
            }
        }
        ids
    };
    let a = grid(&mut b, Vec3::ZERO, None);
    let tip = a[n * (n + 1) + n];
    grid(&mut b, Vec3::new(1.0, 1.0, 0.0), Some(tip));
    let fin = b.vertex(Vec3::new(0.5, 0.5, 0.3), Vec2::new(0.9, 0.9));
    b.face(&[a[3 * (n + 1) + 3], a[3 * (n + 1) + 4], fin], 0).unwrap();
    let mesh = b.finish().unwrap();
    let input =
        LodInput { mesh: &mesh, regions: None, skin: None, deltas: None, landmarks: &[], loops: &[], locked: &[] };
    let s = LodSettings { uv_weight: 0.0, ..LodSettings::default() };
    let levels = generate_lods(&input, &s).unwrap();
    let last = &levels[3].level.mesh;
    assert!(levels[3].report.triangles * 3 < mesh.triangulate().len(), "{}", levels[3].report);
    for p in [Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.5, 0.5, 0.0), Vec3::new(4.0 / 6.0, 0.5, 0.0)] {
        assert!(last.positions().iter().any(|&q| q.distance(p) < 1e-6), "{p:?} was removed");
    }
}

#[test]
fn mod_05_degenerate_soups_never_panic() {
    let mut rng = Rng::new(0x1d0_5eed);
    let (mut ran, mut built) = (0, 0);
    for _ in 0..3000 {
        let Some(mesh) = soup(&mut rng) else { continue };
        built += 1;
        let regions = RegionMap {
            regions: vec![
                Region { name: "a".into(), parent: None, side: Side::Center },
                Region { name: "b".into(), parent: Some(0), side: Side::Center },
            ],
            vertex_regions: (0..mesh.vertex_count()).map(|v| (v % 3 == 0) as u16).collect(),
            soft: None,
        };
        let input = LodInput {
            mesh: &mesh,
            regions: Some(&regions),
            skin: None,
            deltas: None,
            landmarks: &[],
            loops: &[],
            locked: &[],
        };
        let mut s = LodSettings::default();
        for l in &mut s.levels {
            l.max_geometric_error = f32::INFINITY;
        }
        if let Ok(levels) = generate_lods(&input, &s) {
            ran += 1;
            for g in &levels {
                assert!(g.level.check(mesh.vertex_count(), 1).is_ok());
                assert!(g.report.geometric_error.is_finite());
            }
        }
    }
    assert!(built > 2500 && ran > 2000, "{built} {ran}");
}

/// Bench `[MOD-05]`: a ~100k-vertex cube-sphere through four levels (release:
/// `cargo test -p humanart-model --release -- --ignored mod_05_benchmark --nocapture`).
#[test]
#[ignore = "benchmark; run in release"]
fn mod_05_benchmark_100k_vertices() {
    let parts = sphere_template(128);
    let t = Template::new(parts.clone()).unwrap();
    let input = LodInput::from_template(&t, &[]);
    let start = std::time::Instant::now();
    let levels = generate_lods(&input, &LodSettings::default()).unwrap();
    let elapsed = start.elapsed();
    println!(
        "LOD0: {} vertices ({} welded), {} triangles; four levels in {:.3} s",
        parts.mesh.vertex_count(),
        parts.mesh.position_keys().iter().enumerate().filter(|(i, k)| **k as usize == *i).count(),
        parts.mesh.triangulate().len(),
        elapsed.as_secs_f64()
    );
    for g in &levels {
        println!("{}", g.report);
    }
}

/// Diagnostics `[MOD-05]`: runs the levels on the generated template (without bisection) and prints,
/// per level, the measured maxima and the census of what stops the reduction (release:
/// `cargo test -p humanart-model --release --lib mod_05_template_census -- --ignored --nocapture`).
#[test]
#[ignore = "diagnostics on the generated template; run in release"]
fn mod_05_template_census() {
    use crate::generate::build;
    let generated = build::template(&build::Options { lods: false, ..build::Options::default() }).unwrap();
    let t = &generated.template;
    let input = LodInput::from_template(t, &generated.loops);
    let settings = LodSettings::default();
    let rows = collapse::DeltaRows::new(input.deltas, input.mesh.position_keys()).unwrap();
    let mut engine = collapse::Engine::new(&input, &settings, &rows).unwrap();
    let base = report::Baseline::new(&engine, &input).unwrap();
    let n = engine.tris.len();
    println!("LOD0: {} triangles, {} parts", n, t.mesh().parts().len());
    for (i, l) in settings.levels.iter().enumerate() {
        let rules = collapse::Rules {
            geometric: f64::from(l.max_geometric_error),
            delta: l.max_delta_error,
            skin: l.max_skin_error,
        };
        engine.begin_level(i as u8 + 1, rules);
        let target = (f64::from(l.triangle_ratio) * n as f64).ceil() as usize;
        let start = std::time::Instant::now();
        engine.run(target);
        let m = report::measure(&engine, &base, t.mesh().positions(), &report::live_triangles(&engine));
        println!(
            "LOD{}: {:.3} of LOD0 (target {}) in {:.2} s; measured geo {:.2} mm feature {:.2} mm skin {:.3} (budgets {:.1} mm / {:.2})",
            i + 1,
            engine.alive_tris as f64 / n as f64,
            l.triangle_ratio,
            start.elapsed().as_secs_f64(),
            m.geo_max * 1e3,
            m.feature_max * 1e3,
            m.attr.skin_max,
            l.max_geometric_error * 1e3,
            l.max_skin_error
        );
        println!("  {}", collapse::census::census(&mut engine));
        let (by_skin, by_geo) = report::diag::worst(&engine, &base, t.mesh().positions(), 8);
        let regions = t.regions();
        let name = |v: u32| regions.regions[regions.vertex_regions[v as usize] as usize].name.clone();
        for (label, list) in [("skin", &by_skin), ("geo", &by_geo)] {
            for &(a, d, sk, tri) in list {
                let c = engine.tris[tri as usize].map(|w| engine.weld[w as usize]);
                let (j, w) = t.skin().vertex(a as usize);
                let (j0, w0) = t.skin().vertex(c[0] as usize);
                println!(
                    "   {label} a={a} {} d={:.2}mm skin={sk:.2} rep={} ({:.2}mm) tri {:?} [{}|{}|{}] a:{:?}{:?} c0:{:?}{:?}",
                    name(a),
                    d * 1e3,
                    engine.rep(a),
                    t.mesh().positions()[a as usize].distance(t.mesh().positions()[engine.rep(a) as usize]) * 1e3,
                    c,
                    name(c[0]),
                    name(c[1]),
                    name(c[2]),
                    j,
                    w,
                    j0,
                    w0
                );
            }
        }
    }
}

/// `[MOD-05]` `[AC-WP28-1]` on the generated template (release: `cargo test -p humanart-model
/// --release --lib mod_05_generated_template -- --ignored --nocapture`): LOD1..4 reach their
/// triangle targets with every measured error within its budget and nothing undone; the template
/// with its levels and every level as a standalone template pass validation; no level has a seam
/// vertex LOD0 did not have; the eyelid and lip loops keep every vertex through LOD2.
#[test]
#[ignore = "builds the generated template; run in release"]
fn mod_05_generated_template_meets_level_targets() {
    use crate::generate::build;
    let generated = build::template(&build::Options { lods: false, ..build::Options::default() }).unwrap();
    let parts = generated.template.parts().clone();
    let settings = LodSettings::default();
    let start = std::time::Instant::now();
    let levels = generate_lods(&LodInput::from_template(&generated.template, &generated.loops), &settings).unwrap();
    println!("four levels in {:.2} s", start.elapsed().as_secs_f64());
    let base_tris = parts.mesh.triangulate().len();
    for (g, l) in levels.iter().zip(&settings.levels) {
        let r = &g.report;
        println!("{r}");
        assert!(r.triangles <= (f64::from(l.triangle_ratio) * base_tris as f64).ceil() as usize, "{r}");
        assert!(r.geometric_error <= r.geometric_budget && r.feature_error <= r.geometric_budget, "{r}");
        assert!(r.delta_error <= r.delta_budget && r.skin_error <= r.skin_budget, "{r}");
        assert_eq!(r.undone_collapses, 0, "{r}");
    }
    let mut with_lods = parts.clone();
    with_lods.lods = levels.iter().map(|g| g.level.clone()).collect();
    let report = validate(&Template::new(with_lods).unwrap());
    assert!(report.passed(), "{report}");
    assert_eq!(report.count(Check::LodError), 0);
    for (g, v) in levels.iter().zip(&report.lods) {
        assert!(v.surface_error <= g.level.declared_error + 1e-7, "{v:?}");
        println!(
            "LOD{} validator: surface error {:.3} mm, declared {:.3} mm",
            v.level,
            v.surface_error * 1e3,
            v.declared_error * 1e3
        );
    }
    let seams0 = seam_vertices(&parts.mesh);
    let keys0 = parts.mesh.position_keys();
    for g in &levels {
        let t = level_as_template(&parts, g);
        let report = validate(&t);
        assert!(report.passed(), "LOD{}: {report}", g.report.level);
        let src = |v: u32| g.level.sources[v as usize][0];
        for v in seam_vertices(&g.level.mesh) {
            assert!(seams0.binary_search(&keys0[src(v) as usize]).is_ok(), "LOD{}", g.report.level);
        }
        let mut kept: Vec<u32> = g.level.sources.iter().map(|s| keys0[s[0] as usize]).collect();
        kept.sort_unstable();
        kept.dedup();
        for lp in &generated.loops {
            let present = lp.vertices.iter().filter(|&&v| kept.binary_search(&keys0[v as usize]).is_ok()).count();
            println!("LOD{} {}: {present} of {} loop vertices", g.report.level, lp.name, lp.vertices.len());
            if g.report.level <= usize::from(lp.lock_through) {
                assert_eq!(present, lp.vertices.len(), "LOD{} {}", g.report.level, lp.name);
            }
        }
    }
}
