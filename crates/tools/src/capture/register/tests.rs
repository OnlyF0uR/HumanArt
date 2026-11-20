//! Synthetic ground-truth tests of the registration `[WP-25]`: a known smooth deformation with
//! noise and holes, closed and open eyelid and lip slits, and the failure flags.

use humanart_core::rng::Rng;
use humanart_math::Vec3;

use super::*;
use crate::capture::nearest::TriangleTree;
use crate::capture::scan::ScanMesh;

/// An icosphere of radius `r` subdivided `level` times (the 12 icosahedron corners come first).
fn icosphere(r: f32, level: u32) -> FitMesh {
    let t = f32::midpoint(1.0, 5_f32.sqrt());
    let mut p: Vec<Vec3> = [
        (-1.0, t, 0.0),
        (1.0, t, 0.0),
        (-1.0, -t, 0.0),
        (1.0, -t, 0.0),
        (0.0, -1.0, t),
        (0.0, 1.0, t),
        (0.0, -1.0, -t),
        (0.0, 1.0, -t),
        (t, 0.0, -1.0),
        (t, 0.0, 1.0),
        (-t, 0.0, -1.0),
        (-t, 0.0, 1.0),
    ]
    .iter()
    .map(|&(x, y, z)| Vec3::new(x, y, z).normalize())
    .collect();
    let mut f: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    for _ in 0..level {
        let mut mid = std::collections::HashMap::new();
        let mut next = Vec::new();
        let mut midpoint = |a: u32, b: u32, p: &mut Vec<Vec3>| {
            *mid.entry((a.min(b), a.max(b))).or_insert_with(|| {
                p.push(((p[a as usize] + p[b as usize]) * 0.5).normalize());
                p.len() as u32 - 1
            })
        };
        for [a, b, c] in f {
            let ab = midpoint(a, b, &mut p);
            let bc = midpoint(b, c, &mut p);
            let ca = midpoint(c, a, &mut p);
            next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        f = next;
    }
    FitMesh { positions: p.into_iter().map(|v| v * r).collect(), triangles: f.into_iter().flatten().collect() }
}

/// A latitude-longitude sphere of radius `r` (a different tessellation from the icosphere).
fn uv_sphere(r: f32, rings: u32, segments: u32) -> (Vec<Vec3>, Vec<u32>) {
    let mut p = vec![Vec3::new(0.0, r, 0.0)];
    for i in 1..rings {
        let th = std::f32::consts::PI * i as f32 / rings as f32;
        for j in 0..segments {
            let ph = std::f32::consts::TAU * j as f32 / segments as f32;
            p.push(Vec3::new(th.sin() * ph.cos(), th.cos(), -th.sin() * ph.sin()) * r);
        }
    }
    p.push(Vec3::new(0.0, -r, 0.0));
    let last = p.len() as u32 - 1;
    let at = |i: u32, j: u32| 1 + (i - 1) * segments + j % segments;
    let mut t = Vec::new();
    for j in 0..segments {
        t.extend([0, at(1, j), at(1, j + 1)]);
        t.extend([last, at(rings - 1, j + 1), at(rings - 1, j)]);
    }
    for i in 1..rings - 1 {
        for j in 0..segments {
            t.extend([at(i, j), at(i + 1, j), at(i + 1, j + 1)]);
            t.extend([at(i, j), at(i + 1, j + 1), at(i, j + 1)]);
        }
    }
    (p, t)
}

/// The known smooth deformation: radial bumps up to ~8 mm and a tangential swirl up to ~3 mm.
fn field(p: Vec3) -> Vec3 {
    let d = p.normalize_or(Vec3::Y);
    let radial = 0.006 * (3.0 * d.x * d.y) + 0.004 * (d.z * d.z - 0.3) + 0.003 * (2.0 * d.y).sin();
    let swirl = Vec3::new(d.y, -d.x, 0.0) * 0.003 * d.z;
    d * radial + swirl
}

fn scan_of(positions: Vec<Vec3>, indices: Vec<u32>) -> ScanMesh {
    let n = positions.len();
    let mut s = ScanMesh { positions, normals: Vec::new(), uvs: Vec::new(), weld: (0..n as u32).collect(), indices };
    s.recompute_normals();
    s
}

#[test]
fn wp25_registration_recovers_a_known_smooth_field_through_noise_and_holes() {
    let r = 0.1;
    let template = icosphere(r, 4);
    // The scan: another tessellation, deformed, with 0.3 mm noise and two holes.
    let (sp, st) = uv_sphere(r, 120, 240);
    let truth_positions: Vec<Vec3> = sp.iter().map(|&p| p + field(p)).collect();
    let holes = [Vec3::new(0.6, 0.3, 0.74).normalize(), Vec3::new(-0.5, -0.6, 0.62).normalize()];
    let in_hole = |p: Vec3| holes.iter().any(|h| p.normalize_or(Vec3::Y).dot(*h) > 15_f32.to_radians().cos());
    let mut rng = Rng::new(0x25);
    let noisy: Vec<Vec3> = truth_positions
        .iter()
        .map(|&p| p + Vec3::new(rng.normal_f32(), rng.normal_f32(), rng.normal_f32()) * 0.0003)
        .collect();
    let kept: Vec<u32> = st
        .as_chunks::<3>()
        .0
        .iter()
        .filter(|t| !in_hole(sp[t[0] as usize] + sp[t[1] as usize] + sp[t[2] as usize]))
        .flatten()
        .copied()
        .collect();
    let scan = scan_of(noisy, kept);
    let target = TargetSurface::new(&scan, vec![1.0; scan.triangle_count()]).unwrap();
    // Landmarks at the 42 nodes of the once-subdivided icosahedron, annotated at their true
    // positions (a face annotation has more than 100).
    // No annotation can sit in a hole: those are skipped.
    let landmarks: Vec<FitLandmark> = (0..42)
        .filter(|&i| !in_hole(template.positions[i as usize]))
        .map(|i| FitLandmark {
            nodes: [i, i, i],
            weights: [1.0, 0.0, 0.0],
            target: template.positions[i as usize] + field(template.positions[i as usize]),
        })
        .collect();
    let weights = vec![1.0; template.positions.len()];
    let fit = register(&template, &weights, &landmarks, &target, &Schedule::default(), &Thresholds::default()).unwrap();
    // Measure against the noise-free, hole-free deformed surface.
    let truth = TriangleTree::build(&truth_positions, &st).unwrap();
    let (mut s2, mut max, mut hole_max, mut v2) = (0.0_f64, 0.0_f32, 0.0_f32, 0.0_f64);
    let (mut hole_v2, mut hole_n) = (0.0_f64, 0_usize);
    for (&x, &x0) in fit.positions.iter().zip(&template.positions) {
        let d = truth.nearest(x, 1.0, |_| true).unwrap().distance;
        s2 += f64::from(d * d);
        max = max.max(d);
        let e2 = f64::from(x.distance_squared(x0 + field(x0)));
        if in_hole(x0) {
            hole_max = hole_max.max(d);
            hole_v2 += e2;
            hole_n += 1;
        }
        v2 += e2;
    }
    let n = fit.positions.len() as f64;
    let (surface_rms, vertex_rms) = ((s2 / n).sqrt() as f32, (v2 / n).sqrt() as f32);
    // The field moves nodes by up to ~9 mm (RMS 5.4 mm); the stated tolerances: surface RMS
    // 0.4 mm, max 1.5 mm, inside the holes max 1.5 mm; vertex position RMS (tangential sliding
    // included) 1.5 mm overall and 3 mm inside the holes, where only the smoothness speaks.
    let hole_rms = (hole_v2 / hole_n.max(1) as f64).sqrt() as f32;
    assert!(hole_n > 50 && hole_rms < 0.003, "hole vertex RMS {hole_rms} over {hole_n} nodes");
    assert!(surface_rms < 0.0004, "surface RMS {surface_rms} {:?}", fit.report);
    assert!(max < 0.0015, "surface max {max}");
    assert!(hole_max < 0.0015, "hole max {hole_max}");
    assert!(vertex_rms < 0.0015, "vertex RMS {vertex_rms}");
    let start_rms = (template.positions.iter().map(|&p| f64::from(field(p).length_squared())).sum::<f64>() / n).sqrt();
    assert!(f64::from(vertex_rms) < 0.3 * start_rms, "{vertex_rms} vs {start_rms}");
    assert!(fit.flags.is_empty(), "{:?} {:?}", fit.flags.names(), fit.report);
    assert!(fit.report.matched_fraction > 0.9, "{:?}", fit.report);
}

/// A plane patch in XY with an elliptical opening of half-axes `a` × `b` (none for `b == 0`),
/// grid spacing `h`, offset `shift`, bulged towards +Z.
fn slit_plane(a: f32, b: f32, h: f32, shift: f32) -> (Vec<Vec3>, Vec<u32>) {
    let (w, hgt) = (0.05, 0.03);
    let (nx, ny) = ((2.0 * w / h) as u32, (2.0 * hgt / h) as u32);
    let mut p = Vec::new();
    for j in 0..=ny {
        for i in 0..=nx {
            let (x, y) = (-w + i as f32 * h + shift, -hgt + j as f32 * h + shift * 0.5);
            p.push(Vec3::new(x, y, bulge(x, y)));
        }
    }
    let at = |i: u32, j: u32| j * (nx + 1) + i;
    let inside = |q: Vec3| b > 0.0 && (q.x / a).powi(2) + (q.y / b).powi(2) < 1.0;
    let mut t = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            for tri in [[at(i, j), at(i + 1, j), at(i + 1, j + 1)], [at(i, j), at(i + 1, j + 1), at(i, j + 1)]] {
                let c = (p[tri[0] as usize] + p[tri[1] as usize] + p[tri[2] as usize]) / 3.0;
                if !inside(c) {
                    t.extend(tri);
                }
            }
        }
    }
    (p, t)
}

fn bulge(x: f32, y: f32) -> f32 {
    -4.0 * (x * x + y * y)
}

/// The fit node nearest to `q` among nodes used by a triangle.
fn nearest_node(mesh: &FitMesh, q: Vec3) -> u32 {
    let mut used = vec![false; mesh.positions.len()];
    for &i in &mesh.triangles {
        used[i as usize] = true;
    }
    (0..mesh.positions.len())
        .filter(|&i| used[i])
        .min_by(|&i, &j| mesh.positions[i].distance_squared(q).total_cmp(&mesh.positions[j].distance_squared(q)))
        .unwrap() as u32
}

/// Fits a slit of half-height `b_template` to a scan slit of half-height `b_scan` (0: closed),
/// half-width `a`, guided by margin landmarks; returns the fit, the template and the landmarks.
fn fit_slit(a: f32, b_template: f32, b_scan: f32) -> (Fit, FitMesh, Vec<FitLandmark>) {
    let (tp, tt) = slit_plane(a, b_template, 0.002, 0.0);
    let template = FitMesh { positions: tp, triangles: tt };
    let (sp, st) = slit_plane(a, b_scan, 0.0013, 0.0004);
    let scan = scan_of(sp, st);
    let target = TargetSurface::new(&scan, vec![1.0; scan.triangle_count()]).unwrap();
    // Margin landmarks: the ends, the middle and the quarters of both lids (lips), the template's
    // margin node nearest each point of the template ellipse, pulled to the scan ellipse.
    let mut landmarks = Vec::new();
    for s in [-1.0_f32, 1.0] {
        for k in [0.0_f32, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75] {
            let x = a * k;
            let on = |b: f32| {
                let y = s * b * (1.0 - (x / a).powi(2)).max(0.0).sqrt();
                Vec3::new(x, y, bulge(x, y))
            };
            let node = nearest_node(&template, on(b_template) + Vec3::new(0.0, s * 0.0005, 0.0));
            let p0 = template.positions[node as usize];
            let target_y = s * b_scan * (1.0 - (p0.x / a).powi(2)).max(0.0).sqrt();
            landmarks.push(FitLandmark {
                nodes: [node; 3],
                weights: [1.0, 0.0, 0.0],
                target: Vec3::new(p0.x, target_y, bulge(p0.x, target_y)),
            });
        }
    }
    for x in [-a, a] {
        let node = nearest_node(&template, Vec3::new(x * 1.02, 0.0, bulge(x, 0.0)));
        landmarks.push(FitLandmark {
            nodes: [node; 3],
            weights: [1.0, 0.0, 0.0],
            target: Vec3::new(x, 0.0, bulge(x, 0.0)),
        });
    }
    let schedule = Schedule { smoothing_lengths: vec![0.02, 0.01, 0.005], ..Schedule::default() };
    let thresholds = Thresholds { stretch: 20.0, ..Thresholds::default() };
    let weights = vec![1.0; template.positions.len()];
    let fit = register(&template, &weights, &landmarks, &target, &schedule, &thresholds).unwrap();
    (fit, template, landmarks)
}

#[test]
fn wp25_eyelids_and_lips_close_and_open_with_the_scan() {
    // Eyelid: 30 mm wide; the template opening is 8 mm high. Lips: 50 mm wide, 4 mm.
    for (name, a, b) in [("eyelid", 0.015, 0.004), ("lips", 0.025, 0.002)] {
        // Closed in the scan: the lid margins meet.
        let (fit, _, landmarks) = fit_slit(a, b, 0.0);
        let mid_upper = landmarks[0].position(&fit.positions);
        let mid_lower = landmarks[7].position(&fit.positions);
        assert!(mid_upper.distance(mid_lower) < 0.001, "{name} closed: gap {}", mid_upper.distance(mid_lower));
        assert!(fit.report.folded_fraction == 0.0, "{name} closed: {:?}", fit.report);
        assert!(fit.report.rms_error < 0.0005, "{name} closed: {:?}", fit.report);
        assert!(fit.report.landmark_rms < 0.001, "{name} closed: {:?}", fit.report);
        // Open wider in the scan (twice the template's opening).
        let (fit, _, landmarks) = fit_slit(a, b, 2.0 * b);
        for l in &landmarks {
            let d = l.position(&fit.positions).distance(l.target);
            assert!(d < 0.001, "{name} open: landmark off by {d}");
        }
        assert!(fit.report.folded_fraction == 0.0, "{name} open: {:?}", fit.report);
        assert!(fit.report.rms_error < 0.0005, "{name} open: {:?}", fit.report);
    }
}

#[test]
fn wp25_bad_fits_are_flagged() {
    let template = icosphere(0.1, 3);
    let weights = vec![1.0; template.positions.len()];
    // A scan far bigger than the template, out of reach of every correspondence.
    let (sp, st) = uv_sphere(0.2, 40, 80);
    let far = scan_of(sp, st);
    let target = TargetSurface::new(&far, vec![1.0; far.triangle_count()]).unwrap();
    let fit = register(&template, &weights, &[], &target, &Schedule::default(), &Thresholds::default()).unwrap();
    assert!(fit.flags.contains(FitFlags::FEW_CORRESPONDENCES), "{:?}", fit.report);
    assert!(fit.flags.contains(FitFlags::HIGH_ERROR), "{:?}", fit.report);
    // Landmarks that contradict the surface (pulled 3 cm outwards).
    let (sp, st) = uv_sphere(0.1, 60, 120);
    let near = scan_of(sp, st);
    let target = TargetSurface::new(&near, vec![1.0; near.triangle_count()]).unwrap();
    let landmarks: Vec<FitLandmark> = (0..6)
        .map(|i| FitLandmark { nodes: [i; 3], weights: [1.0, 0.0, 0.0], target: template.positions[i as usize] * 1.3 })
        .collect();
    let fit = register(&template, &weights, &landmarks, &target, &Schedule::default(), &Thresholds::default()).unwrap();
    assert!(
        fit.flags.contains(FitFlags::LANDMARK_MISMATCH) || fit.flags.contains(FitFlags::HIGH_ERROR),
        "{:?} {:?}",
        fit.flags.names(),
        fit.report
    );
    // A fully masked scan pulls nothing.
    let target = TargetSurface::new(&near, vec![0.0; near.triangle_count()]).unwrap();
    let fit = register(&template, &weights, &[], &target, &Schedule::default(), &Thresholds::default()).unwrap();
    assert!(fit.flags.contains(FitFlags::FEW_CORRESPONDENCES));
    assert!(fit.positions.iter().zip(&template.positions).all(|(a, b)| a.distance(*b) < 1e-6));
    // Malformed input.
    assert!(register(&template, &weights[1..], &[], &target, &Schedule::default(), &Thresholds::default()).is_err());
    let bad = FitLandmark { nodes: [u32::MAX; 3], weights: [1.0, 0.0, 0.0], target: Vec3::ZERO };
    assert!(register(&template, &weights, &[bad], &target, &Schedule::default(), &Thresholds::default()).is_err());
    let empty = Schedule { smoothing_lengths: Vec::new(), ..Schedule::default() };
    assert!(register(&template, &weights, &[], &target, &empty, &Thresholds::default()).is_err());
    assert!(TargetSurface::new(&near, vec![2.0; near.triangle_count()]).is_err());
}

/// A square grid in the XY plane, `cells` cells of `h` metres per side, centred on the origin,
/// triangles facing +Z.
fn grid(cells: u32, h: f32) -> (Vec<Vec3>, Vec<u32>) {
    let half = cells as f32 * h * 0.5;
    let mut positions = Vec::new();
    for j in 0..=cells {
        for i in 0..=cells {
            positions.push(Vec3::new(i as f32 * h - half, j as f32 * h - half, 0.0));
        }
    }
    let row = cells + 1;
    let mut indices = Vec::new();
    for j in 0..cells {
        for i in 0..cells {
            let a = j * row + i;
            indices.extend([a, a + 1, a + row + 1, a, a + row + 1, a + row]);
        }
    }
    (positions, indices)
}

#[test]
fn wp25_fold_guard_removes_flips_without_dropping_the_data() {
    // A landmark dragging one node 6 mm along a flat scan at a 5 mm smoothing length: the
    // bi-Laplacian alone lets the triangles around it fold over.
    let (tp, tt) = grid(30, 0.002);
    let template = FitMesh { positions: tp, triangles: tt };
    let (sp, st) = grid(60, 0.0015);
    let scan = scan_of(sp, st);
    let target = TargetSurface::new(&scan, vec![1.0; scan.triangle_count()]).unwrap();
    let centre = nearest_node(&template, Vec3::ZERO);
    let landmark = FitLandmark { nodes: [centre; 3], weights: [1.0, 0.0, 0.0], target: Vec3::new(0.006, 0.0, 0.0) };
    let weights = vec![1.0; template.positions.len()];
    let thresholds = Thresholds { stretch: 100.0, ..Thresholds::default() };
    let base = Schedule { smoothing_lengths: vec![0.005], iterations: 3, landmark_weight: 50.0, ..Schedule::default() };
    let unguarded = Schedule { fold_passes: 0, ..base.clone() };
    let fit = register(&template, &weights, &[landmark], &target, &unguarded, &thresholds).unwrap();
    assert!(fit.report.folded_fraction > 0.0, "the case must fold without the guard: {:?}", fit.report);
    let fit = register(&template, &weights, &[landmark], &target, &base, &thresholds).unwrap();
    assert_eq!(fit.report.folded_fraction, 0.0, "{:?}", fit.report);
    assert!(flipped_triangles(&template, &fit.positions).is_empty());
    assert!(fit.report.fold_passes > 0 && fit.report.stiffened_nodes > 0 && fit.report.max_stiffness > 1.0);
    // The data still holds the surface on the scan; the landmark gives way instead.
    assert!(fit.report.rms_error < 0.0002, "{:?}", fit.report);
    assert!(fit.report.matched_fraction > 0.99, "{:?}", fit.report);
}

#[test]
fn wp25_landmarks_fade_by_level() {
    // A landmark 3 mm off a flat scan: kept at full weight it lifts the surface off the scan; with
    // a decay of 0.1 per level the fine levels leave it to the data.
    let (tp, tt) = grid(30, 0.002);
    let template = FitMesh { positions: tp, triangles: tt };
    let (sp, st) = grid(60, 0.0015);
    let scan = scan_of(sp, st);
    let target = TargetSurface::new(&scan, vec![1.0; scan.triangle_count()]).unwrap();
    let centre = nearest_node(&template, Vec3::ZERO);
    let landmark = FitLandmark { nodes: [centre; 3], weights: [1.0, 0.0, 0.0], target: Vec3::new(0.0, 0.0, 0.003) };
    let weights = vec![1.0; template.positions.len()];
    let lift = |decay: f32| {
        let schedule = Schedule {
            smoothing_lengths: vec![0.02, 0.01, 0.005],
            iterations: 2,
            landmark_decay: decay,
            ..Schedule::default()
        };
        let fit = register(&template, &weights, &[landmark], &target, &schedule, &Thresholds::default()).unwrap();
        fit.positions[centre as usize].z
    };
    let (constant, fading) = (lift(1.0), lift(0.1));
    assert!(constant > 0.002, "constant landmark lift {constant}");
    assert!(fading < 0.5 * constant, "fading landmark lift {fading} vs {constant}");
}
