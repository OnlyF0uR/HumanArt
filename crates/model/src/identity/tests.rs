//! Tests of the authored identity model on the generated template, and the reference-silhouette
//! comparison renders (a small software rasteriser, test-only code).

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use humanart_core::rng::Rng;
use humanart_io::{ColorSpace, Format, Image, Samples};
use humanart_math::{Quat, Vec3};

use super::space::{PartKind, mirror};
use super::*;
use crate::delta::DeltaKind;
use crate::eval::{EvalInput, Evaluator, SkinningMode};
use crate::generate::build::{Options, template};

struct Fixture {
    base: Template,
    comps: Vec<IdentityComponent>,
    full: Arc<Template>,
}

fn fixture() -> &'static Fixture {
    static F: OnceLock<Fixture> = OnceLock::new();
    F.get_or_init(|| {
        // The generated template already has the identity components (`generate::build` calls
        // `with_identity`): strip them, rebuild them, and add them back.
        let generated = template(&Options { lods: false, ..Options::default() }).unwrap().template;
        let base = without_identity(&generated).unwrap();
        assert!(BINDINGS.iter().all(|b| base.deltas().find(b.1).is_none()));
        let comps = build(&base).unwrap();
        let full = Arc::new(add_components(&base, &comps).unwrap());
        assert_eq!(full.deltas().len(), generated.deltas().len());
        Fixture { base, comps, full }
    })
}

fn preview_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/agents/identity/previews");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Per vertex displacement of a component (zero where it stores none).
fn dense(c: &IdentityComponent, vertices: usize) -> Vec<Vec3> {
    let mut d = vec![Vec3::ZERO; vertices];
    for &(v, x) in &c.entries {
        d[v as usize] = x;
    }
    d
}

/// `(parameter id, component target)` of every `component` parameter of the registry files.
fn registry_components() -> BTreeSet<(String, String)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/template/params");
    let mut out = BTreeSet::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let bytes = std::fs::read(&path).unwrap();
        let value = humanart_io::json::parse(&bytes).unwrap();
        let root = value.node();
        let defaults = root.opt_field("defaults").unwrap();
        let default_eval = defaults.as_ref().and_then(|d| d.opt_field("evaluation").unwrap());
        for p in root.field("parameters").unwrap().elements().unwrap() {
            let id = p.field("id").unwrap().str().unwrap().to_owned();
            let eval = p.opt_field("evaluation").unwrap();
            let Some(eval) = eval.as_ref().or(default_eval.as_ref()) else { continue };
            if eval.field("kind").unwrap().str().unwrap() == "component" {
                out.insert((id, eval.field("target").unwrap().str().unwrap().to_owned()));
            }
        }
    }
    out
}

#[test]
fn reg_01_bindings_match_the_registry_files() {
    let registry = registry_components();
    let ours: BTreeSet<(String, String)> =
        BINDINGS.iter().chain(scan::SCAN_BINDINGS).map(|&(p, c)| (p.to_owned(), c.to_owned())).collect();
    assert_eq!(ours, registry, "BINDINGS and SCAN_BINDINGS differ from data/template/params");
    assert_eq!(BINDINGS.len(), 73);
    assert_eq!(scan::SCAN_BINDINGS.len(), 7);
}

#[test]
fn id_07_every_binding_builds_and_moves_only_its_regions() {
    let f = fixture();
    let regions = f.base.regions();
    assert_eq!(f.comps.len(), BINDINGS.len());
    let mut lines = Vec::new();
    for (c, &(_, name)) in f.comps.iter().zip(BINDINGS) {
        assert_eq!(c.name, name);
        assert_eq!(c.kind, DeltaKind::Semantic, "{name}: authored components are semantic [ID-07]");
        let max = c.entries.iter().map(|e| e.1.length()).fold(0.0, f32::max);
        assert!(max >= 1.0e-4, "{name} moves at most {max} m per unit weight");
        assert!(max <= 0.08, "{name} moves {max} m per unit weight");
        for &(v, _) in &c.entries {
            let r = regions.vertex_regions[v as usize];
            assert!(c.allowed.contains(&r), "{name} moves vertex {v} in region {}", regions.regions[r as usize].name);
            assert!(regions.is_within(r, c.region), "{name}: vertex {v} outside its bounding region");
        }
        lines.push(format!(
            "{name}: {} entries, max {:.2} mm, {} joints",
            c.entries.len(),
            max * 1000.0,
            c.joints.len()
        ));
    }
    std::fs::write(preview_dir().join("components.txt"), lines.join("\n")).unwrap();
}

#[test]
fn id_05_symmetric_components_are_mirror_symmetric() {
    let f = fixture();
    let space = Space::new(&f.base).unwrap();
    let field = space.field();
    let reps = field.representatives();
    let n_vertices = f.base.mesh().vertex_count();
    let joints = &f.base.skeleton().joints;
    for c in &f.comps {
        let d = dense(c, n_vertices);
        let mut worst = 0.0_f32;
        for (n, &m) in space.mirror_map().iter().enumerate() {
            if m == u32::MAX {
                continue;
            }
            let exact =
                field.positions()[m as usize].distance(mirror(field.positions()[n])) <= super::space::MIRROR_TOLERANCE;
            if !exact {
                continue;
            }
            let (a, b) = (d[reps[n] as usize], d[reps[m as usize] as usize]);
            let tol = 2.0e-6 + 2.0e-3 * a.length().max(b.length());
            worst = worst.max((b - mirror(a)).length() - tol);
        }
        match c.symmetry {
            Symmetry::Mirror => {
                assert!(worst <= 0.0, "{}: mirror error {worst}", c.name);
                for &(j, o) in &c.joints {
                    let name = &joints[j as usize].name;
                    if let Some(rest) = name.strip_prefix("L.") {
                        let r = f.base.skeleton().find(&format!("R.{rest}")).unwrap();
                        let ro = c.joints.iter().find(|e| e.0 == r).map_or(Vec3::ZERO, |e| e.1);
                        assert!((ro - mirror(o)).length() < 1e-7, "{}: joint {name}", c.name);
                    } else if !name.starts_with("R.") {
                        assert_eq!(o.x, 0.0, "{}: centre joint {name} moves sideways", c.name);
                    }
                }
            }
            Symmetry::Free => assert!(worst > 1.0e-5, "{}: an asymmetry component is symmetric", c.name),
        }
    }
    // Every node has a mirror image.
    let mut unmatched: HashMap<String, usize> = HashMap::new();
    for (n, &m) in space.mirror_map().iter().enumerate() {
        if m == u32::MAX {
            let part = &f.base.mesh().parts()[field.part_of(n) as usize].name;
            *unmatched.entry(part.clone()).or_default() += 1;
        }
    }
    assert!(unmatched.is_empty(), "nodes without a mirror image: {unmatched:?}");
}

/// Triangle pairs across edges of the body surface (welded), and whether the edge's two triangles
/// touch more than one region.
fn edge_pairs(t: &Template, space: &Space<'_>, tris: &[u32]) -> Vec<(usize, usize, bool)> {
    let regions = &t.regions().vertex_regions;
    let mut map: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (ti, c) in tris.chunks(3).enumerate() {
        let n = |v: u32| space.field().node_of(v as usize);
        if space.kind_of(n(c[0]) as usize) != PartKind::Body {
            continue;
        }
        for k in 0..3 {
            let (a, b) = (n(c[k]), n(c[(k + 1) % 3]));
            map.entry((a.min(b), a.max(b))).or_default().push(ti);
        }
    }
    let mut out = Vec::new();
    for list in map.values() {
        if let [a, b] = list[..] {
            let rs: BTreeSet<u16> =
                tris[a * 3..a * 3 + 3].iter().chain(&tris[b * 3..b * 3 + 3]).map(|&v| regions[v as usize]).collect();
            out.push((a, b, rs.len() > 1));
        }
    }
    out
}

fn tri_normals(pos: &[Vec3], tris: &[u32]) -> Vec<Vec3> {
    tris.chunks(3)
        .map(|c| {
            (pos[c[1] as usize] - pos[c[0] as usize]).cross(pos[c[2] as usize] - pos[c[0] as usize]).normalize_or_zero()
        })
        .collect()
}

/// The largest change of the angle between neighbouring triangle normals, in degrees, over
/// region-border edges and over interior edges.
fn dihedral_change(pairs: &[(usize, usize, bool)], rest: &[Vec3], now: &[Vec3]) -> (f32, f32) {
    dihedral_quantile(pairs, rest, now, 1.0)
}

/// The `count` border edges with the largest dihedral change: `(degrees, rest centroid of the
/// first triangle)`, to say where a crease is.
fn worst_border_edges(
    pairs: &[(usize, usize, bool)],
    tris: &[u32],
    pos: &[Vec3],
    rest: &[Vec3],
    now: &[Vec3],
    count: usize,
) -> Vec<(f32, Vec3)> {
    let mut out: Vec<(f32, Vec3)> = pairs
        .iter()
        .filter(|p| p.2 && rest[p.0] != Vec3::ZERO && now[p.0] != Vec3::ZERO && now[p.1] != Vec3::ZERO)
        .map(|&(a, b, _)| {
            let d = (now[a].angle_between(now[b]) - rest[a].angle_between(rest[b])).abs().to_degrees();
            let c = tris[a * 3..a * 3 + 3].iter().fold(Vec3::ZERO, |s, &v| s + pos[v as usize]) / 3.0;
            (d, c)
        })
        .collect();
    out.sort_by(|a, b| b.0.total_cmp(&a.0));
    out.truncate(count);
    out
}

/// The `q` quantile (1 = maximum) of the dihedral changes over border and interior edges.
fn dihedral_quantile(pairs: &[(usize, usize, bool)], rest: &[Vec3], now: &[Vec3], q: f32) -> (f32, f32) {
    let (mut border, mut inner) = (Vec::new(), Vec::new());
    for &(a, b, is_border) in pairs {
        if rest[a] == Vec3::ZERO || rest[b] == Vec3::ZERO || now[a] == Vec3::ZERO || now[b] == Vec3::ZERO {
            continue;
        }
        let d = (now[a].angle_between(now[b]) - rest[a].angle_between(rest[b])).abs().to_degrees();
        if is_border { border.push(d) } else { inner.push(d) }
    }
    let pick = |mut v: Vec<f32>| {
        if v.is_empty() {
            return 0.0;
        }
        v.sort_by(f32::total_cmp);
        v[((v.len() - 1) as f32 * q).round() as usize]
    };
    (pick(border), pick(inner))
}

#[test]
fn id_02_no_normal_discontinuity_at_region_borders() {
    let f = fixture();
    let t = &*f.full;
    let space = Space::new(t).unwrap();
    let tris = t.mesh().triangulate().indices;
    // The genital variants are left out: the switch moves them between their two states and
    // other components carry them as helpers (`todo_1_3_sex_switch_*` test them).
    let mut genital = region_ids(t, &crate::generate::genitals::MALE_REGIONS);
    genital.extend(region_ids(t, &crate::generate::genitals::FEMALE_REGIONS));
    let touches = |tri: usize| {
        tris[tri * 3..tri * 3 + 3].iter().any(|&v| genital.contains(&t.regions().vertex_regions[v as usize]))
    };
    let pairs: Vec<(usize, usize, bool)> =
        edge_pairs(t, &space, &tris).into_iter().filter(|p| !touches(p.0) && !touches(p.1)).collect();
    let rest = tri_normals(t.mesh().positions(), &tris);
    let index: Vec<usize> = BINDINGS.iter().map(|&(_, c)| t.deltas().find(c).unwrap() as usize).collect();
    let mut rng = Rng::new(0x1d02);
    let mut lines = Vec::new();
    let mut failures = Vec::new();
    // The head (fine face mesh, lip and alar folds) and the rest of the body are compared
    // separately: smoothing the body shapes must not let the face's creases pass, nor the reverse.
    let head = t.regions().find("head").unwrap();
    let in_head = |p: &(usize, usize, bool)| {
        tris[p.0 * 3..p.0 * 3 + 3].iter().any(|&v| t.regions().is_within(t.regions().vertex_regions[v as usize], head))
    };
    let zones: [Vec<(usize, usize, bool)>; 2] = [
        pairs.iter().copied().filter(|p| in_head(p)).collect(),
        pairs.iter().copied().filter(|p| !in_head(p)).collect(),
    ];
    let mut worst = [(0.0_f32, 0.0_f32); 2];
    for sample in 0..6 {
        let mut w = vec![0.0; t.deltas().len()];
        // The sex switch takes its two values; the other components are compared with that sex.
        let mut alone = vec![0.0; t.deltas().len()];
        for (&i, &(_, name)) in index.iter().zip(BINDINGS) {
            w[i] = match name {
                "asymmetry.global" => rng.range_f32(0.0, 1.0),
                sex::SWITCH => {
                    alone[i] = if rng.range_f32(-1.0, 1.0) < 0.0 { -1.0 } else { 1.0 };
                    alone[i]
                }
                _ => rng.range_f32(-2.0, 2.0),
            };
        }
        let rest = tri_normals(&t.deformed_positions(&alone), &tris);
        let now = tri_normals(&t.deformed_positions(&w), &tris);
        let (b, i) = dihedral_quantile(&pairs, &rest, &now, 0.999);
        let (bm, im) = dihedral_change(&pairs, &rest, &now);
        lines.push(format!(
            "random sample {sample}: 99.9 % quantile border {b:.2} deg, interior {i:.2} deg; maximum border {bm:.2}, interior {im:.2}"
        ));
        for (d, c) in worst_border_edges(&pairs, &tris, t.mesh().positions(), &rest, &now, 4) {
            lines.push(format!("  border edge {d:.1} deg at ({:.3}, {:.3}, {:.3})", c.x, c.y, c.z));
        }
        for (z, zone) in zones.iter().enumerate() {
            let (b, i) = dihedral_quantile(zone, &rest, &now, 0.999);
            lines.push(format!("  {}: 99.9 % quantile border {b:.2} deg, interior {i:.2} deg", ["head", "body"][z]));
            worst[z] = (worst[z].0.max(b), worst[z].1.max(i));
        }
    }
    // Each component alone at the edge of the usual span.
    for (&i, &(_, name)) in index.iter().zip(BINDINGS) {
        if name == sex::SWITCH {
            // Two values only, each compared with the other components in
            // `todo_1_3_sex_switch_is_fold_free`.
            continue;
        }
        let mut w = vec![0.0; t.deltas().len()];
        w[i] = 2.0;
        let now = tri_normals(&t.deformed_positions(&w), &tris);
        let (b, inner) = dihedral_change(&pairs, &rest, &now);
        lines.push(format!("{name} at 2: border {b:.2} deg, interior {inner:.2} deg"));
        if b > 45.0 || b > inner * 1.5 + 5.0 {
            for (d, c) in worst_border_edges(&pairs, &tris, t.mesh().positions(), &rest, &now, 3) {
                lines.push(format!("  border edge {d:.1} deg at ({:.3}, {:.3}, {:.3})", c.x, c.y, c.z));
            }
        }
        if b > inner * 1.5 + 5.0 {
            failures.push(format!("{name}: border crease {b} deg against {inner} deg inside"));
        }
    }
    std::fs::write(preview_dir().join("id-02-normals.txt"), lines.join("\n")).unwrap();
    assert!(failures.is_empty(), "{failures:#?}");
    // A crease at a region border would show as a border change well above the interior's
    // (random identities: 99.9 % quantiles, robust against the few sliver triangles in folds).
    for (z, &(b, i)) in worst.iter().enumerate() {
        assert!(b <= i * 1.5 + 2.0, "{}: border {b} deg, interior {i} deg", ["head", "body"][z]);
    }
}

/// The rest and deformed positions of landmark `name` under weights `w` of the full template.
fn landmark_shift(t: &Template, pos: &[Vec3], name: &str) -> Vec3 {
    let l = t.landmarks().iter().find(|l| l.name == name).unwrap();
    l.position(pos).unwrap() - l.position(t.mesh().positions()).unwrap()
}

#[test]
fn mod_07_joints_follow_the_shape() {
    let f = fixture();
    let t = &*f.full;
    let sk = t.skeleton();
    // Joint and the landmarks around it on the skin.
    // Written out here (not taken from `space::SKIN_RINGS`) so the test checks the rule itself.
    let rings: [(&str, &[&str]); 8] = [
        ("L.shin", &["L.knee.center", "L.knee.lateral", "L.knee.medial", "L.knee.back"]),
        ("L.foot", &["L.ankle.lateral", "L.ankle.medial"]),
        ("L.forearm", &["L.elbow.lateral", "L.elbow.medial"]),
        ("L.hand", &["L.wrist.radial", "L.wrist.ulnar"]),
        ("R.shin", &["R.knee.center", "R.knee.lateral", "R.knee.medial", "R.knee.back"]),
        ("R.foot", &["R.ankle.lateral", "R.ankle.medial"]),
        ("R.forearm", &["R.elbow.lateral", "R.elbow.medial"]),
        ("R.hand", &["R.wrist.radial", "R.wrist.ulnar"]),
    ];
    let mut checked = 0;
    for &(_, name) in BINDINGS {
        let c = t.deltas().find(name).unwrap() as usize;
        let mut w = vec![0.0; t.deltas().len()];
        w[c] = 1.0;
        let pos = t.deformed_positions(&w);
        let (joints, offsets) = sk.regressor.component(c);
        for (joint, marks) in rings {
            let j = sk.find(joint).unwrap();
            let off = joints.iter().position(|&x| x == j).map_or(Vec3::ZERO, |k| offsets[k]);
            let skin =
                marks.iter().fold(Vec3::ZERO, |a, m| a + landmark_shift(t, &pos, m)) * (1.0 / marks.len() as f32);
            let err = (off - skin).length();
            assert!(
                err <= 1.0e-3 + 0.15 * off.length().max(skin.length()),
                "{name}: {joint} moves {off:?}, its skin {skin:?}"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, BINDINGS.len() * 8);
    // Height raises the head and keeps the ankles; hip width moves the hip joints apart.
    let row = |name: &str, joint: &str| {
        let c = t.deltas().find(name).unwrap() as usize;
        let (j, d) = sk.regressor.component(c);
        let k = sk.find(joint).unwrap();
        j.iter().position(|&x| x == k).map_or(Vec3::ZERO, |i| d[i])
    };
    assert!(row("global.height", "head").y > 0.02);
    // The ankle joint follows its skin ring, which the blended skinning lifts by under 2 mm.
    assert!(row("global.height", "L.foot").y.abs() < 2.0e-3);
    assert!(row("semantic.leg-length", "pelvis").y > 0.02);
    assert!(row("semantic.hip-width", "L.thigh").x > 0.002 && row("semantic.hip-width", "R.thigh").x < -0.002);
    assert!(row("semantic.shoulder-width", "L.upper-arm").x > 0.005);
    assert!(row("semantic.eye-spacing", "L.eye").x > 0.0005);
    assert!(row("semantic.neck-length", "head").y > 0.003);
}

#[test]
fn ac_wp06_template_validates_and_round_trips() {
    let f = fixture();
    let t = &*f.full;
    let report = crate::validate::validate(t);
    assert!(report.passed(), "{report}");
    // [ID-07]: the authored components are not scan-derived; the 7 face-shape components of the
    // scans (todo 1.2) are, and are the only ones.
    assert_eq!(t.scan_component_count(), scan::SCAN_BINDINGS.len());
    for &(_, name) in scan::SCAN_BINDINGS {
        let c = t.deltas().find(name).unwrap() as usize;
        assert_eq!(t.deltas().components()[c].kind, DeltaKind::IdentityGlobal, "{name}");
    }
    assert_eq!(
        t.deltas().count_kind(DeltaKind::Semantic),
        f.base.deltas().count_kind(DeltaKind::Semantic) + BINDINGS.len()
    );
    let bytes = crate::write_package(t).unwrap();
    let back = crate::read_package(&bytes, &crate::PackageLimits::default()).unwrap();
    assert_eq!(&back, t);
    // A second application is refused; the bases keep the old components and re-index them.
    assert!(add_components(t, &f.comps).is_err());
    assert!(t.info().provenance.iter().any(|p| p.item.contains("identity axes")));
}

#[test]
fn missing_names_fail_cleanly() {
    let cube = Template::new(crate::template::fixtures::cube_template()).unwrap();
    let e = build(&cube).unwrap_err();
    assert!(matches!(e, IdentityError::Missing { .. } | IdentityError::Field(_)), "{e}");
    assert!(!e.to_string().is_empty());
}

/// The joint-local rotation that turns joint `j` by the world rotation `world` at rest.
fn local_pose(t: &Template, j: usize, world: Quat) -> Quat {
    let rb = t.skeleton().rest_world()[j].rotation;
    (rb.inverse() * world * rb).normalize()
}

/// Renders the body part from `forward` into a `w` x `h` tile, flat-shaded, linear RGB.
fn render(tris: &[u32], pos: &[Vec3], w: usize, h: usize, forward: Vec3, centre: Vec3, half: f32) -> Vec<[f32; 3]> {
    let mut color = vec![[0.02_f32, 0.02, 0.02]; w * h];
    let mut depth = vec![f32::INFINITY; w * h];
    let f = forward.normalize();
    let r = f.cross(Vec3::Y).normalize();
    let u = r.cross(f);
    let scale = h as f32 / (2.0 * half);
    let key = (-f + u * 0.7 - r * 0.45).normalize();
    let project = |p: Vec3| {
        let d = p - centre;
        Vec3::new(w as f32 * 0.5 + d.dot(r) * scale, h as f32 * 0.5 - d.dot(u) * scale, d.dot(f))
    };
    // Smooth (Gouraud-style per-pixel) normals: area-weighted vertex normals, welded by position so
    // UV seams do not show. Flat per-triangle shading made the 1 to 3 cm quads of the body show as
    // stair-step blocks wherever a shape curves the surface.
    let key_of = |p: Vec3| ((p.x * 2.0e4).round() as i64, (p.y * 2.0e4).round() as i64, (p.z * 2.0e4).round() as i64);
    let mut welded: HashMap<(i64, i64, i64), Vec3> = HashMap::new();
    for c in tris.chunks(3) {
        let (a, b, cc) = (pos[c[0] as usize], pos[c[1] as usize], pos[c[2] as usize]);
        let n = (b - a).cross(cc - a);
        for p in [a, b, cc] {
            *welded.entry(key_of(p)).or_insert(Vec3::ZERO) += n;
        }
    }
    let normal = |p: Vec3| welded.get(&key_of(p)).copied().unwrap_or(Vec3::ZERO).normalize_or_zero();
    for c in tris.chunks(3) {
        let (a, b, cc) = (pos[c[0] as usize], pos[c[1] as usize], pos[c[2] as usize]);
        let face = (b - a).cross(cc - a).normalize_or_zero();
        let ns = [normal(a), normal(b), normal(cc)];
        let s = [project(a), project(b), project(cc)];
        let area = (s[1].x - s[0].x) * (s[2].y - s[0].y) - (s[2].x - s[0].x) * (s[1].y - s[0].y);
        if area.abs() < 1e-12 {
            continue;
        }
        let x0 = s.iter().map(|p| p.x).fold(f32::MAX, f32::min).floor().max(0.0) as usize;
        let x1 = (s.iter().map(|p| p.x).fold(f32::MIN, f32::max).ceil() as isize).min(w as isize - 1);
        let y0 = s.iter().map(|p| p.y).fold(f32::MAX, f32::min).floor().max(0.0) as usize;
        let y1 = (s.iter().map(|p| p.y).fold(f32::MIN, f32::max).ceil() as isize).min(h as isize - 1);
        if x1 < 0 || y1 < 0 {
            continue;
        }
        for y in y0..=y1 as usize {
            for x in x0..=x1 as usize {
                let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                let w0 = ((s[1].x - px) * (s[2].y - py) - (s[2].x - px) * (s[1].y - py)) / area;
                let w1 = ((s[2].x - px) * (s[0].y - py) - (s[0].x - px) * (s[2].y - py)) / area;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                let z = w0 * s[0].z + w1 * s[1].z + w2 * s[2].z;
                let i = y * w + x;
                if z < depth[i] {
                    depth[i] = z;
                    let mut n = (ns[0] * w0 + ns[1] * w1 + ns[2] * w2).normalize_or_zero();
                    if n == Vec3::ZERO {
                        n = face;
                    }
                    let k = 0.15 + 0.75 * n.dot(key).abs();
                    color[i] = [0.80 * k, 0.58 * k, 0.47 * k];
                }
            }
        }
    }
    color
}

/// Writes a 2 x 2 sheet of the four reference views, each as `[reference | render]`, posed with
/// the arms lowered like the reference (front, side, three-quarter back, back).
fn comparison_sheet(name: &str, t: &Arc<Template>, weights: &[(&str, f32)], arms_down: bool) {
    let mut shape = vec![0.0; t.deltas().len()];
    for &(c, w) in weights {
        shape[t.deltas().find(c).unwrap_or_else(|| panic!("{c}")) as usize] = w;
    }
    let mut pose = vec![Quat::IDENTITY; t.skeleton().joints.len()];
    let lower = if arms_down { 0.5_f32 } else { 0.0 };
    for (side, angle) in [("L.upper-arm", -lower), ("R.upper-arm", lower)] {
        let j = t.skeleton().find(side).unwrap() as usize;
        pose[j] = local_pose(t, j, Quat::from_rotation_z(angle));
    }
    let mut eval = Evaluator::new(Arc::clone(t));
    eval.evaluate(&EvalInput {
        shape: &shape,
        expressions: &[],
        layers: &[],
        pose: &pose,
        skinning: SkinningMode::ByRegion,
    })
    .unwrap();
    let pos = eval.positions().to_vec();
    let mesh = t.mesh();
    let tri = mesh.triangulate();
    let body = mesh.parts().iter().position(|p| p.name == "body").unwrap();
    let range = mesh.parts()[body].faces();
    let tris: Vec<u32> = tri
        .indices
        .chunks(3)
        .zip(&tri.triangle_faces)
        .filter(|(_, f)| range.contains(&(**f as usize)))
        .flat_map(|(c, _)| c.to_vec())
        .collect();
    let reference = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/reference/char-01.png");
    let refimg =
        std::fs::read(&reference).ok().and_then(|b| humanart_io::png::decode(&b, &humanart_io::Limits::default()).ok());
    let (tw, th) = (512_usize, 512_usize);
    let views = [Vec3::new(0.0, 0.0, -1.0), Vec3::X, Vec3::new(1.0, 0.0, 1.0), Vec3::Z];
    let (w, h) = (tw * 4, th * 2);
    let mut out = vec![0_u8; w * h * 3];
    for (k, v) in views.iter().enumerate() {
        let img = render(&tris, &pos, tw, th, *v, Vec3::new(0.0, 0.84, 0.0), 0.90);
        let (gx, gy) = (k % 2, k / 2);
        for y in 0..th {
            for x in 0..tw {
                let o1 = ((gy * th + y) * w + gx * 2 * tw + x) * 3;
                if let Some(r) = &refimg {
                    let (rw, ch) = (r.width as usize, r.channels as usize);
                    let (sx, sy) =
                        (gx * rw / 2 + x * rw / 2 / tw, gy * r.height as usize / 2 + y * r.height as usize / 2 / th);
                    if let Some(px) = r.as_u8().and_then(|b| b.get((sy * rw + sx) * ch..(sy * rw + sx) * ch + 3)) {
                        out[o1..o1 + 3].copy_from_slice(px);
                    }
                }
                let c = img[y * tw + x];
                for j in 0..3 {
                    out[o1 + tw * 3 + j] = (c[j].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
                }
            }
        }
    }
    let image = Image::new(Format::Png, w as u32, h as u32, 3, ColorSpace::Srgb, Samples::U8(out)).unwrap();
    std::fs::write(preview_dir().join(format!("{name}.png")), humanart_io::png::encode(&image).unwrap()).unwrap();
}

/// Parameter settings that approach the owner's reference character (AI-generated, four views of
/// one body, `target/reference/char-01.png`, not in the repository), registry parameters only.
const REFERENCE: &[(&str, f32)] = &[
    ("global.sex", -1.2),
    ("global.mass", 0.2),
    ("semantic.hip-width", 0.3),
    ("semantic.shoulder-width", 0.5),
    ("semantic.leg-length", 0.5),
    ("semantic.waist-width", -1.5),
    ("semantic.bust-size", 3.5),
    ("semantic.breast-projection", 1.0),
    ("semantic.breast-ptosis", -0.5),
    ("semantic.glute-size", 2.2),
    ("semantic.hip-flank", 0.3),
    ("semantic.thigh-fullness", 0.8),
];

#[test]
fn preview_reference_comparison() {
    let f = fixture();
    comparison_sheet("neutral", &f.full, &[], true);
    comparison_sheet("reference-round2", &f.full, REFERENCE, true);
    for (name, w) in [
        ("global.mass", 2.0),
        ("global.muscularity", 2.0),
        ("global.sex", 2.0),
        ("global.sex", -2.0),
        ("global.age", 2.0),
        ("global.height", 2.0),
        ("semantic.bust-size", 2.0),
        ("semantic.glute-size", 2.0),
        ("semantic.thigh-fullness", 2.0),
    ] {
        comparison_sheet(&format!("axis-{name}-{w}"), &f.full, &[(name, w)], false);
    }
}

/// Per node displacement of a component (zero where it stores none).
fn node_disp(space: &Space<'_>, c: &IdentityComponent) -> Vec<Vec3> {
    let mut d = vec![Vec3::ZERO; space.node_count()];
    for &(v, x) in &c.entries {
        d[space.field().node_of(v as usize) as usize] = x;
    }
    d
}

fn component<'a>(f: &'a Fixture, name: &str) -> &'a IdentityComponent {
    f.comps.iter().find(|c| c.name == name).unwrap_or_else(|| panic!("{name}"))
}

/// The enclosed volume change of the skin when it moves by `disp` (m^3; holes elsewhere in the
/// skin cancel out of the difference).
fn volume_change(space: &Space<'_>, disp: &[Vec3]) -> f32 {
    let field = space.field();
    let body = space.part("body").unwrap();
    let pos = field.positions();
    let volume = |w: f32| {
        let at = |n: u32| pos[n as usize] + disp[n as usize] * w;
        field
            .triangles()
            .iter()
            .filter(|t| field.part_of(t[0] as usize) == body)
            .map(|t| at(t[0]).dot(at(t[1]).cross(at(t[2]))) / 6.0)
            .sum::<f32>()
    };
    volume(1.0) - volume(0.0)
}

#[test]
fn id_03_ansur_constants_match_the_data_file() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/anthropometry/ansur2.json");
    let bytes = std::fs::read(path).unwrap();
    let value = humanart_io::json::parse(&bytes).unwrap();
    let root = value.node();
    let stats = root.field("stats").unwrap();
    for v in measure::VARIABLES.iter().chain(sex::VARIABLES) {
        for (sex, mean, sd) in [("male", v.male_mean, v.male_sd), ("female", v.female_mean, v.female_sd)] {
            let s = stats.field(sex).unwrap();
            let s = s.field(v.id).unwrap();
            let (m, d) = (s.field("mean").unwrap().f64().unwrap(), s.field("sd").unwrap().f64().unwrap());
            assert!((m - f64::from(mean)).abs() < 0.01 && (d - f64::from(sd)).abs() < 0.01, "{} {sex}", v.id);
        }
    }
    let corr = root.field("correlation").unwrap();
    let ids: Vec<String> =
        corr.field("variables").unwrap().elements().unwrap().map(|e| e.str().unwrap().to_owned()).collect();
    let index = |id: &str| ids.iter().position(|x| x == id).unwrap_or_else(|| panic!("{id}"));
    for c in measure::CORRELATIONS {
        for (sex, r) in [("male", c.male), ("female", c.female)] {
            let m = corr.field(sex).unwrap();
            let row = m.at(index(c.a)).unwrap();
            let got = row.at(index(c.b)).unwrap().f64().unwrap();
            assert!((got - f64::from(r)).abs() < 1.0e-4, "{} {} {sex}", c.a, c.b);
        }
    }
}

#[test]
fn id_03_calibrated_axes_move_their_measurement_by_one_sd() {
    use measure::{Gauge, Variable};
    let f = fixture();
    let space = Space::new(&f.base).unwrap();
    let mut lines = vec!["Calibrated axes: change per unit weight on the template against the target".to_owned()];
    for cal in measure::CALIBRATIONS {
        let d = node_disp(&space, component(f, cal.component));
        let got = cal.gauge.per_unit(&space, &d).unwrap();
        let want = cal.unit.metres().unwrap();
        lines.push(format!("{}: {:.2} mm (target {:.2} mm, {:?})", cal.component, got * 1e3, want * 1e3, cal.unit));
        // Confinement and the mirror copy run after the calibration: allow 3 %.
        assert!((got - want).abs() <= 0.03 * want + 1.0e-4, "{}: {got} against {want}", cal.component);
    }
    // Mass: the least-squares fit makes sum(target x measured) = sum(measured^2).
    let d = node_disp(&space, component(f, "global.mass"));
    let (mut tm, mut mm) = (0.0, 0.0);
    lines.push("global.mass, per unit (1 SD of weightkg at fixed stature):".to_owned());
    for (gauge, target) in measure::mass_targets().unwrap() {
        let got = gauge.per_unit(&space, &d).unwrap();
        lines.push(format!("  {gauge:?}: {:.1} mm (expected {:.1} mm)", got * 1e3, target * 1e3));
        assert!(got > 0.4 * target && got < 2.5 * target, "{gauge:?}: {got} against {target}");
        tm += target * got;
        mm += got * got;
    }
    assert!((tm / mm - 1.0).abs() < 0.03, "mass fit {}", tm / mm);
    lines.push(format!("  mass volume change per unit: {:.2} L", volume_change(&space, &d) * 1e3));
    // Sex shape (within a sex): one unit is one within-sex SD (RMS of the sexes) in the male
    // direction for the calibrated terms (the terms overlap a little, so 25 %); chest and buttock
    // circumference are reported only.
    let d = node_disp(&space, component(f, "global.sex"));
    let stature = Gauge::Stature.measure(&space, &[], 0.0).unwrap();
    lines.push(format!("template stature {:.1} mm; global.sex per unit:", stature * 1e3));
    let check = |gauge: &Gauge, v: &Variable, strict: bool, lines: &mut Vec<String>| {
        let got = gauge.per_unit(&space, &d).unwrap();
        let want = v.sd_m().copysign(v.sex_half_difference(stature));
        lines.push(format!("  {}: {:.2} mm (one within-sex SD toward male {:.2} mm)", v.id, got * 1e3, want * 1e3));
        if strict {
            assert!((got - want).abs() <= 0.25 * want.abs() + 1.5e-3, "{}: {got} against {want}", v.id);
        }
    };
    check(&measure::BIACROMIAL_GAUGE, &measure::BIACROMIAL, true, &mut lines);
    check(&measure::HIP_BREADTH_GAUGE, &measure::HIP_BREADTH, true, &mut lines);
    check(&measure::WAIST_BREADTH_GAUGE, &measure::WAIST_BREADTH, true, &mut lines);
    check(&measure::NECK_GIRTH, &measure::NECK_CIRC, true, &mut lines);
    check(&measure::CHEST_GIRTH, &measure::CHEST_CIRC, false, &mut lines);
    check(&measure::BUTTOCK_GIRTH, &measure::BUTTOCK_CIRC, false, &mut lines);
    // Template measurements against the sex-midpoint means (information).
    lines.push("template rest measurements against the ANSUR II sex-midpoint mean:".to_owned());
    for (gauge, v) in [
        (Gauge::Stature, measure::STATURE),
        (measure::BIACROMIAL_GAUGE, measure::BIACROMIAL),
        (measure::HIP_BREADTH_GAUGE, measure::HIP_BREADTH),
        (measure::WAIST_BREADTH_GAUGE, measure::WAIST_BREADTH),
        (measure::WAIST_GIRTH, measure::WAIST_CIRC),
        (measure::CHEST_GIRTH, measure::CHEST_CIRC),
        (measure::BUTTOCK_GIRTH, measure::BUTTOCK_CIRC),
        (measure::THIGH_GIRTH, measure::THIGH_CIRC),
        (measure::NECK_GIRTH, measure::NECK_CIRC),
    ] {
        let got = gauge.measure(&space, &[], 0.0).unwrap() * 1e3;
        lines.push(format!("  {}: {got:.1} mm (mean {:.1} mm)", v.id, f32::midpoint(v.male_mean, v.female_mean)));
    }
    for name in ["semantic.bust-size", "semantic.glute-size", "semantic.thigh-fullness", "semantic.hip-flank"] {
        let v = volume_change(&space, &node_disp(&space, component(f, name)));
        lines.push(format!("{name}: {:.0} mL per side per unit", v * 1e6 / 2.0));
    }
    std::fs::write(preview_dir().join("calibration.txt"), lines.join("\n") + "\n").unwrap();
}

#[test]
fn id_03_sex_axis_leaves_the_groin_and_genital_parts_alone() {
    let f = fixture();
    let space = Space::new(&f.base).unwrap();
    let crotch = space.landmark("torso.pubis").unwrap() - Vec3::Y * 0.04;
    let field = space.field();
    let parts = f.base.mesh().parts();
    let sex = component(f, "global.sex");
    assert!(!sex.entries.is_empty());
    for &(v, d) in &sex.entries {
        let n = field.node_of(v as usize) as usize;
        let p = field.positions()[n];
        let e = (p.x / 0.06).hypot((p.y - crotch.y) / 0.09);
        assert!(e >= 1.0 || d.length() < 1.0e-6, "vertex {v} at the groin moves {d:?}");
        let part = &parts[field.part_of(n) as usize].name;
        assert!(!part.contains("genital") && !part.contains("vulva") && !part.contains("penis"), "{part} moves");
    }
}

/// Per moved skin node of the trunk and thighs (outside the carried details), the largest second derivative of `d` along a line
/// through the node and two nearly opposite neighbours `a`, `b` (the non-uniform second difference
/// `((d_a - d_n) / l_a + (d_b - d_n) / l_b) * 2 / (l_a + l_b)`, exact zero for a linear field
/// whatever the mesh spacing), times (2 cm)^2 and divided by the largest displacement: a smooth
/// falloff gives a small value, a stepped or noisy one a large value.
fn second_derivatives(space: &Space<'_>, d: &[Vec3]) -> Vec<(f32, usize)> {
    let max = d.iter().map(|x| x.length()).fold(0.0, f32::max).max(1.0e-9);
    let pos = space.field().positions();
    let body = space.reach_flags(&Reach { regions: &["torso", "thigh"], except: &[], margin: 0.0 }).unwrap();
    // The fine details of ADR-039 (nipples, areolae, navel) have millimetre edges and are carried
    // as a whole (`Space::carry_details`): their nodes and the skin within 2 cm are left out, a
    // second difference there measures the template's fine geometry and the carry's rim, not the
    // centimetre-scale shape (ID-02 checks those rims).
    let mut regions = vec![false; body.len()];
    for name in ["nipple-L", "nipple-R", "areola-L", "areola-R", "navel"] {
        if let Ok(flags) = space.reach_flags(&Reach { regions: &[name], except: &[], margin: 0.0 }) {
            for (d, f) in regions.iter_mut().zip(flags) {
                *d |= f;
            }
        }
    }
    let detail_points: Vec<Vec3> =
        (0..space.node_count()).filter(|&n| regions[space.field().region_of(n) as usize]).map(|n| pos[n]).collect();
    let near: Vec<bool> =
        (0..space.node_count()).map(|n| detail_points.iter().any(|q| q.distance(pos[n]) < 0.02)).collect();
    (0..space.node_count())
        .filter(|&n| {
            space.kind_of(n) == PartKind::Body
                && d[n] != Vec3::ZERO
                && body[space.field().region_of(n) as usize]
                && !near[n]
        })
        .map(|n| {
            let nb = space.field().neighbours(n);
            let mut worst = 0.0_f32;
            for (k, &a) in nb.iter().enumerate() {
                for &b in &nb[k + 1..] {
                    let (ea, eb) = (pos[a as usize] - pos[n], pos[b as usize] - pos[n]);
                    let (la, lb) = (ea.length(), eb.length());
                    // Edges under 4 mm belong to the template's fine details (nipple, navel
                    // rims), not to the centimetre-scale shapes measured here.
                    if la < 4.0e-3 || lb < 4.0e-3 || ea.dot(eb) > -0.9 * la * lb {
                        continue;
                    }
                    let dd = ((d[a as usize] - d[n]) / la + (d[b as usize] - d[n]) / lb) * (2.0 / (la + lb));
                    worst = worst.max(dd.length());
                }
            }
            (worst * 0.02 * 0.02 / max, n)
        })
        .collect()
}

/// The 99.9 % quantile of [`second_derivatives`].
fn roughness(space: &Space<'_>, d: &[Vec3]) -> f32 {
    let mut v: Vec<f32> = second_derivatives(space, d).into_iter().map(|x| x.0).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f32::total_cmp);
    v[((v.len() - 1) as f32 * 0.999).round() as usize]
}

/// The largest roughness ([`roughness`]) a trunk or thigh shape may have. Smooth compact bumps a
/// few centimetres wide (the pectoral bump of muscularity, the sex axis in the suprasternal
/// notch) measure up to about 0.6; the glute shape quantised to 8 levels measures about 1.2.
const MAX_ROUGHNESS: f32 = 0.65;

#[test]
fn id_02_body_shapes_have_no_stepped_falloff() {
    let f = fixture();
    let space = Space::new(&f.base).unwrap();
    let mut lines = vec![format!("roughness (99.9 % quantile, limit {MAX_ROUGHNESS}) and the roughest nodes:")];
    let names = [
        "global.mass",
        "global.sex",
        "global.muscularity",
        "semantic.hip-width",
        "semantic.waist-width",
        "semantic.thigh-fullness",
        "semantic.glute-size",
        "semantic.hip-flank",
        "semantic.bust-size",
        "semantic.breast-projection",
        "semantic.breast-ptosis",
        "semantic.breast-spacing",
    ];
    let mut worst = 0.0_f32;
    let mut failures = Vec::new();
    for name in names {
        let d = node_disp(&space, component(f, name));
        let r = roughness(&space, &d);
        lines.push(format!("{name}: {r:.4}"));
        let mut top = second_derivatives(&space, &d);
        top.sort_by(|a, b| b.0.total_cmp(&a.0));
        for &(v, n) in top.iter().take(2) {
            let p = space.field().positions()[n];
            let region = &f.base.regions().regions[space.field().region_of(n) as usize].name;
            lines.push(format!(
                "  {v:.3} at ({:.3}, {:.3}, {:.3}) in {region}, |d| {:.1} mm",
                p.x,
                p.y,
                p.z,
                d[n].length() * 1e3
            ));
        }
        if r > MAX_ROUGHNESS {
            failures.push(format!("{name}: {r}"));
        }
        worst = worst.max(r);
    }
    // The measure sees steps: the glute shape quantised to 8 levels of its largest displacement.
    let d = node_disp(&space, component(f, "semantic.glute-size"));
    let max = d.iter().map(|x| x.length()).fold(0.0, f32::max);
    let stepped: Vec<Vec3> = d
        .iter()
        .map(|x| {
            let l = x.length();
            if l == 0.0 { *x } else { *x * (((l / max) * 8.0).round() / 8.0 * max / l) }
        })
        .collect();
    let rs = roughness(&space, &stepped);
    lines.push(format!("semantic.glute-size quantised to 8 steps: {rs:.4}"));
    for region in ["chest", "pelvis", "thigh-L"] {
        let r = space.region(region).unwrap();
        let pos = space.field().positions();
        let mut edges: Vec<f32> = (0..space.node_count())
            .filter(|&n| space.field().region_of(n) == r)
            .flat_map(|n| space.field().neighbours(n).iter().map(move |&m| pos[n].distance(pos[m as usize])))
            .collect();
        edges.sort_by(f32::total_cmp);
        lines.push(format!("median edge length in {region}: {:.1} mm", edges[edges.len() / 2] * 1e3));
    }
    std::fs::write(
        preview_dir().join("roughness.txt"),
        lines.join(
            "
",
        ) + "
",
    )
    .unwrap();
    assert!(rs > 1.5 * MAX_ROUGHNESS, "the measure does not separate steps: {rs}");
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The `min` and `max` of registry parameter `id` in `data/template/params/face.json`.
fn registry_range(id: &str) -> (f32, f32) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/template/params/face.json");
    let value = humanart_io::json::parse(&std::fs::read(path).unwrap()).unwrap();
    let root = value.node();
    for p in root.field("parameters").unwrap().elements().unwrap() {
        if p.field("id").unwrap().str().unwrap() == id {
            let n = |k: &str| p.field(k).unwrap().f64().unwrap() as f32;
            return (n("min"), n("max"));
        }
    }
    panic!("{id} is not in face.json");
}

/// Triangles whose normal turned over between `rest` and `now` (rest area above 1e-12 m^2).
fn folded(rest: &[Vec3], now: &[Vec3], tris: &[u32]) -> usize {
    tris.chunks(3)
        .filter(|c| {
            let n = |p: &[Vec3]| (p[c[1] as usize] - p[c[0] as usize]).cross(p[c[2] as usize] - p[c[0] as usize]);
            let n0 = n(rest);
            n0.length() > 2.0e-12 && n0.dot(n(now)) < 0.0
        })
        .count()
}

/// `[WP-25]` todo 1.2: the 7 face-shape components of the scans are bound to the registry, are
/// scan-derived, mirror-symmetric, move only the head and neck (with the eye joints), and fold no
/// triangle anywhere inside their registry range. Also measures the no-fold limit of each (steps of
/// 0.05 up to 3), written to `target/agents/face-scan/scan-ranges.txt`.
#[test]
fn todo_1_2_scan_components_are_bound_symmetric_and_fold_free_in_range() {
    let f = fixture();
    let t = &*f.full;
    let space = Space::new(t).unwrap();
    let tris = t.mesh().triangulate().indices;
    let rest = t.mesh().positions();
    let regions = t.regions();
    let (head, neck) = (regions.find("head").unwrap(), regions.find("neck").unwrap());
    let eye = t.skeleton().find("L.eye").unwrap();
    let mut lines = Vec::new();
    let mut failures = Vec::new();
    for &(id, name) in scan::SCAN_BINDINGS {
        let c = t.deltas().find(name).unwrap() as usize;
        assert_eq!(t.deltas().components()[c].kind, DeltaKind::IdentityGlobal);
        let mut w = vec![0.0; t.deltas().len()];
        w[c] = 1.0;
        let d: Vec<Vec3> = t.deformed_positions(&w).iter().zip(rest).map(|(a, b)| *a - *b).collect();
        let max = d.iter().map(|x| x.length()).fold(0.0, f32::max);
        assert!(max > 5.0e-4 && max < 0.03, "{name} moves {max} m at 1 SD");
        for (v, x) in d.iter().enumerate() {
            if x.length() > 1.0e-6 {
                let r = regions.vertex_regions[v];
                assert!(
                    regions.is_within(r, head) || regions.is_within(r, neck),
                    "{name} moves {}",
                    regions.regions[r as usize].name
                );
            }
        }
        // Mirror-symmetric where the mirror map is exact.
        let reps = space.field().representatives();
        let mut worst = 0.0_f32;
        for (n, &m) in space.mirror_map().iter().enumerate() {
            if m == u32::MAX {
                continue;
            }
            let (a, b) = (d[reps[n] as usize], d[reps[m as usize] as usize]);
            if space.field().positions()[m as usize].distance(mirror(space.field().positions()[n]))
                <= super::space::MIRROR_TOLERANCE
            {
                worst = worst.max((b - mirror(a)).length() - 2.0e-6 - 2.0e-3 * a.length().max(b.length()));
            }
        }
        assert!(worst <= 0.0, "{name}: mirror error {worst}");
        // The eye joints move with the eyes.
        let (joints, offsets) = t.skeleton().regressor.component(c);
        assert!(joints.contains(&eye) && offsets.iter().all(|o| o.is_finite()), "{name}: the eye joint stays");
        // The no-fold limit of each sign, and the registry range inside it.
        let mut limit = |sign: f32| {
            let mut last = 0.0;
            for k in 1..=60 {
                let s = 0.05 * k as f32;
                w[c] = sign * s;
                if folded(rest, &t.deformed_positions(&w), &tris) > 0 {
                    return last;
                }
                last = s;
            }
            last
        };
        let (minus, plus) = (limit(-1.0), limit(1.0));
        let (lo, hi) = registry_range(id);
        lines.push(format!(
            "{id} ({name}): max {:.2} mm at 1 SD; folds from -{minus:.2} / +{plus:.2} on; registry {lo} .. {hi}",
            max * 1e3
        ));
        if -lo > minus || hi > plus {
            failures.push(format!("{id}: registry range {lo}..{hi} folds (limits -{minus} / +{plus})"));
        }
    }
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/agents/face-scan");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("scan-ranges.txt"), lines.join("\n")).unwrap();
    println!("{}", lines.join("\n"));
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Positions of `t` with the named components at their weights.
fn posed(t: &Template, weights: &[(&str, f32)]) -> Vec<Vec3> {
    let mut p = t.mesh().positions().to_vec();
    for &(name, w) in weights {
        let c = t.deltas().find(name).unwrap_or_else(|| panic!("{name}")) as usize;
        for (&v, &d) in t.deltas().indices(c).iter().zip(t.deltas().deltas(c)) {
            p[v as usize] += crate::delta::DeltaBases::delta_f32(d) * w;
        }
    }
    p
}

/// The region indices of the given generator regions.
fn region_ids(t: &Template, list: &[crate::generate::regions::R]) -> Vec<u16> {
    let regions = &t.regions().regions;
    list.iter().map(|r| regions.iter().position(|x| x.name == r.name()).unwrap() as u16).collect()
}

/// Per vertex: whether a face of the `body` part uses it.
fn on_body(t: &Template) -> Vec<bool> {
    let mut on = vec![false; t.mesh().vertex_count()];
    let body = t.mesh().parts().iter().find(|p| p.name == "body").unwrap().faces();
    for f in body {
        for &v in t.mesh().face(f) {
            on[v as usize] = true;
        }
    }
    on
}

/// todo 1.3: the switch puts every fitted measurement of both sexes near the ANSUR II mean
/// (stated tolerance: 0.35 SD in the gauge's scale); the bust girth, the face breadth and the arm
/// girth are printed, not checked (their gauges differ from ANSUR's).
#[test]
fn todo_1_3_sex_switch_meets_the_ansur_means() {
    let f = fixture();
    let space = Space::new(&f.base).unwrap();
    let kit = Kit::new(&space).unwrap();
    let (raw, amounts) = sex::fit(&kit).unwrap();
    println!("amounts {amounts:?}");
    let mut failed = Vec::new();
    for (g, t) in &sex::FITTED {
        let (fe, ma) = sex::states(&space, g, &raw).unwrap();
        let rest = g.measure(&space, &[], 0.0).unwrap();
        let (tf, tm) = sex::targets(*t, rest);
        let v = t.variable();
        // The SD in the gauge's own scale (relative and difference gauges are not ANSUR's).
        let k = if matches!(t, sex::Target::Relative(_)) {
            rest / (f32::midpoint(v.male_mean, v.female_mean) * 1e-3)
        } else {
            1.0
        };
        // Hip breadth and buttock depth take an extreme point that the switch moves outward at
        // both settings (the fit matches their difference): 0.6 SD for them.
        let tol = if ["hipbreadth", "buttockdepth"].contains(&v.id) { 0.6 } else { 0.35 };
        let ok = (fe - tf).abs() <= tol * k * v.female_sd * 1e-3 && (ma - tm).abs() <= tol * k * v.male_sd * 1e-3;
        let line = format!(
            "{} ({t:?}): female {:.1} / {:.1}, male {:.1} / {:.1} mm (rest {:.1})",
            v.id,
            fe * 1e3,
            tf * 1e3,
            ma * 1e3,
            tm * 1e3,
            rest * 1e3
        );
        println!("{line}");
        if !ok {
            failed.push(line);
        }
    }
    for (g, v) in &sex::REPORTED_RELATIVE {
        let (fe, ma) = sex::states(&space, g, &raw).unwrap();
        let rest = g.measure(&space, &[], 0.0).unwrap();
        let mid = f32::midpoint(v.male_mean, v.female_mean);
        println!(
            "{} (reported, relative): female {:.3} / {:.3}, male {:.3} / {:.3}",
            v.id,
            fe / rest,
            v.female_mean / mid,
            ma / rest,
            v.male_mean / mid
        );
    }
    for (g, v) in &sex::REPORTED {
        let (fe, ma) = sex::states(&space, g, &raw).unwrap();
        let line = format!(
            "{} (reported): female {:.1} / {:.1}, male {:.1} / {:.1} mm",
            v.id,
            fe * 1e3,
            v.female_mean,
            ma * 1e3,
            v.male_mean
        );
        println!("{line}");
    }
    assert!(failed.is_empty(), "{failed:#?}");
}

/// todo 1.3, ADR-039: one template carries both genital variants; `identity.sex` −1 shows the
/// vulva and keeps the male piece inside the pelvis, +1 shows the male piece (meeting the body at
/// the attachment loop) with the vulva folded under it. No rebuild: only a component weight.
#[test]
fn todo_1_3_sex_switch_shows_one_genital_variant() {
    use crate::generate::genitals;
    let t = &*fixture().full;
    let rest = t.mesh().positions();
    let vr = &t.regions().vertex_regions;
    let (male, female) = (region_ids(t, &genitals::MALE_REGIONS), region_ids(t, &genitals::FEMALE_REGIONS));
    let on = on_body(t);
    let is = |v: usize, list: &[u16]| on[v] && list.contains(&vr[v]);
    let fem = posed(t, &[(sex::SWITCH, -1.0)]);
    let mal = posed(t, &[(sex::SWITCH, 1.0)]);
    // Each variant rides on the groin, which the switch moves with the hips and thighs: its
    // shown state is the real shape carried by the skin around it, so the offsets from the real
    // shape are a few millimetres and nearly the same for the whole variant.
    let (mut n_male, mut n_female, mut worst_fold) = (0, 0, f32::MIN);
    let (mut male_off, mut vulva_off) = (Vec::new(), Vec::new());
    // The stowed piece rides with the groin too (the female groin lies about 3 cm lower than the
    // rest shape's): it stays within 4 cm of the stow centre carried as the vulva is.
    let carried: Vec<Vec3> =
        (0..rest.len()).filter(|&v| is(v, &female)).map(|v| fem[v] - genitals::vulva_states(rest[v]).0).collect();
    let carry = carried.iter().fold(Vec3::ZERO, |s, o| s + *o) / carried.len().max(1) as f32;
    for v in 0..rest.len() {
        if is(v, &male) {
            n_male += 1;
            let stowed = fem[v].distance(genitals::STOW_CENTRE + carry);
            assert!(stowed < 0.04, "male vertex {v} not stowed: {:?} (carry {carry:?})", fem[v]);
            male_off.push(mal[v] - genitals::male_real(rest[v]));
        } else if is(v, &female) {
            n_female += 1;
            vulva_off.push(fem[v] - genitals::vulva_states(rest[v]).0);
            worst_fold = worst_fold.max(genitals::site_coords(mal[v]).2);
        }
    }
    for (name, off) in [("male piece", &male_off), ("vulva", &vulva_off)] {
        let mean = off.iter().fold(Vec3::ZERO, |s, o| s + *o) / off.len().max(1) as f32;
        let spread = off.iter().map(|o| o.distance(mean)).fold(0.0, f32::max);
        println!(
            "{name}: carried {:.2} mm with the groin, shape error at most {:.2} mm",
            mean.length() * 1e3,
            spread * 1e3
        );
        assert!(mean.length() < 0.012, "{name} carried {:.1} mm", mean.length() * 1e3);
        assert!(spread < 0.003, "{name} distorted by {:.2} mm", spread * 1e3);
    }
    assert!(n_male > 500 && n_female > 500, "{n_male} male, {n_female} female vertices");
    assert!(worst_fold < -4.0e-4, "a folded vulva vertex is only {:.2} mm under the site", -worst_fold * 1e3);
    // The male piece meets the body: its border vertices lie on body vertices outside both variants.
    let near_body: Vec<Vec3> = (0..rest.len())
        .filter(|&v| on[v] && !is(v, &male) && !is(v, &female))
        .map(|v| mal[v])
        .filter(|p| p.y < 0.9 && p.y > 0.75 && p.x.abs() < 0.05)
        .collect();
    let met = (0..rest.len())
        .filter(|&v| is(v, &male))
        .filter(|&v| near_body.iter().any(|q| q.distance(mal[v]) < 1.0e-4))
        .count();
    println!(
        "male {n_male}, vulva {n_female}, deepest fold {:.2} mm, male vertices on the body {met}",
        -worst_fold * 1e3
    );
    assert!(met >= 48, "only {met} male vertices meet the body");
}

/// Count of body faces (outside the regions `skip`) whose normal turns over between `a` and `b`.
fn turned_faces(t: &Template, a: &[Vec3], b: &[Vec3], skip: &[u16]) -> usize {
    let vr = &t.regions().vertex_regions;
    let normal = |p: &[Vec3], c: &[u32]| {
        (0..c.len()).fold(Vec3::ZERO, |s, k| s + p[c[k] as usize].cross(p[c[(k + 1) % c.len()] as usize]))
    };
    let body = t.mesh().parts().iter().find(|p| p.name == "body").unwrap().faces();
    body.filter(|&f| {
        let c = t.mesh().face(f);
        !c.iter().any(|&v| skip.contains(&vr[v as usize])) && normal(a, c).dot(normal(b, c)) < 0.0
    })
    .count()
}

/// todo 1.3: both settings are fold-free against the rest shape, and stay so across `sex_shape`
/// −2..2 and the breast parameters at their soft ranges (±2).
#[test]
fn todo_1_3_sex_switch_is_fold_free() {
    use crate::generate::genitals;
    let t = &*fixture().full;
    let rest = t.mesh().positions();
    let mut skip = region_ids(t, &genitals::MALE_REGIONS);
    skip.extend(region_ids(t, &genitals::FEMALE_REGIONS));
    let mut lines = Vec::new();
    let mut bad = 0;
    for w in [-1.0, 1.0] {
        let base = posed(t, &[(sex::SWITCH, w)]);
        let n = turned_faces(t, rest, &base, &skip);
        lines.push(format!("sex {w}: {n} turned faces against the rest shape"));
        bad += n;
        if n > 0 {
            let body = t.mesh().parts().iter().find(|p| p.name == "body").unwrap().faces();
            for f in body {
                let c = t.mesh().face(f);
                let vr = &t.regions().vertex_regions;
                let nrm = |p: &[Vec3]| {
                    (0..c.len()).fold(Vec3::ZERO, |s, k| s + p[c[k] as usize].cross(p[c[(k + 1) % c.len()] as usize]))
                };
                if !c.iter().any(|&v| skip.contains(&vr[v as usize])) && nrm(rest).dot(nrm(&base)) < 0.0 {
                    let r = vr[c[0] as usize];
                    lines.push(format!("  at {:?} in {}", base[c[0] as usize], t.regions().regions[r as usize].name));
                }
            }
        }
        let mut cases: Vec<(&str, f32)> = vec![("global.sex", -2.0), ("global.sex", 2.0)];
        for c in
            ["semantic.bust-size", "semantic.breast-projection", "semantic.breast-ptosis", "semantic.breast-spacing"]
        {
            cases.push((c, -2.0));
            cases.push((c, 2.0));
        }
        for (c, x) in cases {
            let p = posed(t, &[(sex::SWITCH, w), (c, x)]);
            let n = turned_faces(t, &base, &p, &skip);
            // What the parameter turns over on the rest shape without the switch is not the
            // switch's doing.
            let alone = turned_faces(t, rest, &posed(t, &[(c, x)]), &skip);
            lines.push(format!("sex {w}, {c} {x}: {n} (alone on the rest shape {alone})"));
            if n > alone {
                let body = t.mesh().parts().iter().find(|p| p.name == "body").unwrap().faces();
                for f in body {
                    let c = t.mesh().face(f);
                    let nrm = |p: &[Vec3]| {
                        (0..c.len())
                            .fold(Vec3::ZERO, |s, k| s + p[c[k] as usize].cross(p[c[(k + 1) % c.len()] as usize]))
                    };
                    if nrm(&base).dot(nrm(&p)) < 0.0 {
                        let r = t.regions().vertex_regions[c[0] as usize];
                        lines.push(format!(
                            "  at {:?} in {}",
                            base[c[0] as usize],
                            t.regions().regions[r as usize].name
                        ));
                    }
                }
            }
            bad += n.saturating_sub(alone);
        }
    }
    println!("{}", lines.join("\n"));
    assert_eq!(bad, 0, "{lines:#?}");
}

/// The frontmost skin point of each millimetre of height near the midline (|x| < 2 mm) between
/// the heights `lo` and `hi` and in front of `front` (depth), top down: the profile of the face as
/// a camera at the side sees it. Without `front` a height whose few midline vertices all lie on
/// the back of the head would give a back point.
fn midline_profile(p: &[Vec3], on: &[bool], lo: f32, hi: f32, front: f32) -> Vec<Vec3> {
    let mut bins: Vec<Option<Vec3>> = vec![None; ((hi - lo) * 1000.0) as usize + 1];
    for (q, _) in p.iter().zip(on).filter(|(q, on)| **on && q.x.abs() < 0.002 && q.y > lo && q.y < hi && q.z > front) {
        let slot = &mut bins[((hi - q.y) * 1000.0) as usize];
        if slot.is_none_or(|o: Vec3| q.z > o.z) {
            *slot = Some(*q);
        }
    }
    bins.into_iter().flatten().collect()
}

/// The profile point between the heights `lo` and `hi` that lies farthest along `dir`.
fn profile_support(profile: &[Vec3], lo: f32, hi: f32, dir: Vec3) -> Vec3 {
    profile
        .iter()
        .copied()
        .filter(|q| q.y >= lo && q.y <= hi)
        .max_by(|a, b| a.dot(dir).total_cmp(&b.dot(dir)))
        .unwrap_or_else(|| panic!("no profile point between {lo} and {hi}"))
}

/// What the adult-proportion test measures on one posed character, metres.
/// The midline points are found on the posed profile, not taken from the template's landmarks:
/// `forehead.glabella`, `nose.root` and `chin.menton` are found on the procedural head before the
/// scanned face moves it and no longer sit where their names say.
#[derive(Debug)]
struct AdultGauge {
    /// Floor to the top of the head.
    stature: f32,
    /// Vertex to menton, vertical.
    head: f32,
    /// Tragion to vertex, vertical (ANSUR `tragiontopofhead`).
    vault: f32,
    /// Sellion to menton, straight (ANSUR `mentonsellionlength`).
    face: f32,
    /// Sellion to subnasale, vertical (Farkas n-sn).
    nose: f32,
    /// Subnasale to menton, vertical (Farkas sn-gn).
    lower: f32,
    /// Subnasale to pronasale, straight (Farkas sn-prn).
    protrusion: f32,
    /// Between the zygion landmarks (ANSUR `bizygomaticbreadth`).
    bizygomatic: f32,
    /// The widest skin in front of the ears between the mouth and the brows.
    face_breadth: f32,
    /// The widest skin above the ears (ANSUR `headbreadth`).
    head_breadth: f32,
    /// Face breadth at the height of the mouth (stomion), in front of the ears.
    jaw_breadth: f32,
    /// Endocanthion to exocanthion (Farkas en-ex).
    fissure: f32,
    /// Upper to lower lid margin at mid-lid.
    opening: f32,
    /// Between the iris centres (ANSUR `interpupillarybreadth`).
    pupils: f32,
    /// How far the pogonion lies in front of the supramentale.
    sulcus: f32,
    /// How far the pogonion lies behind the subnasale.
    chin_setback: f32,
    /// Menton above the lowest midline point within 50 mm behind it (0: a chin that ends the
    /// face; large: the underside runs down into the neck).
    submental_drop: f32,
    /// Height of the upper plus the lower vermilion at the midline.
    lips: f32,
    /// Superaurale to subaurale (ANSUR `earlength`).
    ear: f32,
    /// The malar landmark's height.
    malar_y: f32,
}

fn adult_gauge(t: &Template, on: &[bool], weights: &[(&str, f32)], print_profile: bool) -> AdultGauge {
    let p = posed(t, weights);
    let lm = |name: &str| {
        t.landmarks().iter().find(|l| l.name == name).unwrap_or_else(|| panic!("{name}")).position(&p).unwrap()
    };
    let floor = p.iter().zip(on).filter(|x| *x.1).map(|x| x.0.y).fold(f32::INFINITY, f32::min);
    let vertex = lm("head.vertex");
    let tragion = lm("ear.L.tragion");
    let (en, ex) = (lm("L.eye.canthus.medial"), lm("L.eye.canthus.lateral"));
    let prn = lm("nose.tip");
    let sn = lm("nose.subnasale");
    let (ls, li, sto) = (lm("lip.upper.outer.06"), lm("lip.lower.outer.06"), lm("lip.upper.inner.06"));
    let brow = lm("L.brow.03");
    let profile = midline_profile(&p, on, sto.y - 0.09, vertex.y, tragion.z);
    if print_profile {
        let line: Vec<String> = profile
            .iter()
            .filter(|q| q.y < brow.y + 0.03)
            .map(|q| format!("{:.0}:{:.1}", (q.y - en.y) * 1e3, (q.z - tragion.z) * 1e3))
            .collect();
        println!("profile (mm above the endocanthion : mm in front of the tragion) {}", line.join(" "));
    }
    let sellion = profile_support(&profile, prn.y + 0.012, brow.y + 0.006, -Vec3::Z);
    let supramentale = profile_support(&profile, li.y - 0.022, li.y - 0.003, -Vec3::Z);
    let pogonion = profile_support(&profile, supramentale.y - 0.03, supramentale.y - 0.003, Vec3::Z);
    // Menton: the lowest point of the chin, taken 60 degrees below the horizontal among the
    // points at most 25 mm behind the pogonion, so that the neck below does not count as chin.
    let chin: Vec<Vec3> = profile.iter().copied().filter(|q| q.z > pogonion.z - 0.025).collect();
    let menton = profile_support(&chin, pogonion.y - 0.035, pogonion.y, Vec3::new(0.0, -0.866, 0.5));
    let behind = p
        .iter()
        .zip(on)
        .filter(|(q, on)| **on && q.x.abs() < 0.002 && q.z < menton.z && q.z > menton.z - 0.05 && q.y < pogonion.y)
        .filter(|(q, _)| q.y > menton.y - 0.06)
        .map(|(q, _)| q.y)
        .fold(menton.y, f32::min);
    let front = |q: &Vec3| q.z > tragion.z + 0.012;
    let widest = |keep: &dyn Fn(&Vec3) -> bool| {
        2.0 * p.iter().zip(on).filter(|(q, on)| **on && keep(q)).map(|(q, _)| q.x.abs()).fold(0.0, f32::max)
    };
    let superaurale = lm("ear.L.superaurale");
    if print_profile {
        println!(
            "sellion {:.1} mm above the endocanthion, subnasale {:.1} below, stomion {:.1}, pogonion {:.1}, menton {:.1} below it",
            (sellion.y - en.y) * 1e3,
            (en.y - sn.y) * 1e3,
            (en.y - sto.y) * 1e3,
            (en.y - pogonion.y) * 1e3,
            (en.y - menton.y) * 1e3
        );
    }
    AdultGauge {
        stature: vertex.y - floor,
        head: vertex.y - menton.y,
        vault: vertex.y - tragion.y,
        face: sellion.distance(menton),
        nose: sellion.y - sn.y,
        lower: sn.y - menton.y,
        protrusion: sn.distance(prn),
        bizygomatic: lm("L.zygion").distance(lm("R.zygion")),
        face_breadth: widest(&|q| front(q) && q.y > sto.y && q.y < brow.y),
        head_breadth: widest(&|q| q.y > superaurale.y + 0.005),
        jaw_breadth: widest(&|q| front(q) && (q.y - sto.y).abs() < 0.003),
        fissure: en.distance(ex),
        opening: lm("L.eye.upper.04").y - lm("L.eye.lower.04").y,
        pupils: lm("L.iris.center").distance(lm("R.iris.center")),
        sulcus: pogonion.z - supramentale.z,
        chin_setback: sn.z - pogonion.z,
        submental_drop: menton.y - behind,
        lips: ls.y - li.y,
        ear: superaurale.distance(lm("ear.L.lobule")),
        malar_y: lm("L.cheek.malar").y,
    }
}

/// `tragiontopofhead` of ANSUR II (checked against the data file below).
const TRAGION_TOP: measure::Variable =
    measure::Variable { id: "tragiontopofhead", male_mean: 131.1, male_sd: 6.2, female_mean: 126.5, female_sd: 6.5 };
/// `headbreadth` of ANSUR II.
const HEAD_BREADTH: measure::Variable =
    measure::Variable { id: "headbreadth", male_mean: 154.3, male_sd: 5.5, female_mean: 147.8, female_sd: 5.2 };

/// todo 1.3 ("sufficiently adult"): both sexes at the registry defaults have the proportions of an
/// adult, by measurement. ANSUR II (adults, mean age 30.2 / 28.9 years, the file in the
/// repository) gives the head and face sizes against the stature; what ANSUR II does not hold
/// (nose, eye fissure, chin) is checked against wide windows around adult values of Farkas,
/// "Anthropometry of the Head and Face" (1994), North American adults 19 to 25, quoted from the
/// secondary literature: nose height n-sn 54.8 (SD 3.3) men, 50.6 (3.1) women; eye fissure en-ex
/// 31.3 / 30.7 (1.2) mm. A child of six has a face height of about 0.78 of the vault height, a
/// nose of about 40 mm and a stature of about six head heights.
#[test]
fn todo_1_3_both_sexes_have_adult_proportions() {
    let t = &*fixture().full;
    let on = on_body(t);
    let mut failed = Vec::new();
    let mut check = |what: String, value: f32, lo: f32, hi: f32| {
        let line = format!("{what}: {value:.3} (adult window {lo:.3} .. {hi:.3})");
        println!("{line}");
        if !(lo..=hi).contains(&value) {
            failed.push(line);
        }
    };
    for (name, w) in [("female", -1.0_f32), ("male", 1.0)] {
        let g = adult_gauge(t, &on, &[(sex::SWITCH, w)], true);
        println!("{name}: {g:#?}");
        let mean = |v: &measure::Variable| 1.0e-3 * if w < 0.0 { v.female_mean } else { v.male_mean };
        let sd = |v: &measure::Variable| 1.0e-3 * if w < 0.0 { v.female_sd } else { v.male_sd };
        // Head to body: the classical 7.5 heads (ANSUR II has no vertex-to-menton height).
        check(format!("{name} stature in head heights"), g.stature / g.head, 7.0, 8.0);
        // Face and vault sizes against ANSUR II, within 1.5 SD.
        for (what, value, v) in [
            ("menton-sellion length", g.face, &sex::MENTON_SELLION),
            ("bizygomatic breadth", g.bizygomatic, &sex::BIZYGOMATIC),
            ("tragion to top of head", g.vault, &TRAGION_TOP),
            ("head breadth", g.head_breadth, &HEAD_BREADTH),
            ("interpupillary breadth", g.pupils, &measure::INTERPUPILLARY),
            ("ear length", g.ear, &measure::EAR_LENGTH),
        ] {
            check(
                format!("{name} {what} (ANSUR mean {:.1}) mm", mean(v) * 1e3),
                value * 1e3,
                (mean(v) - 1.5 * sd(v)) * 1e3,
                (mean(v) + 1.5 * sd(v)) * 1e3,
            );
        }
        // The face against the vault: small in a child, 0.89 (women) and 0.94 (men) in ANSUR II.
        let ratio = mean(&sex::MENTON_SELLION) / mean(&TRAGION_TOP);
        check(format!("{name} face height / vault height"), g.face / g.vault, ratio * 0.92, ratio * 1.1);
        // The face's height against its breadth: 0.845 (women) and 0.860 (men) in ANSUR II.
        let index = mean(&sex::MENTON_SELLION) / mean(&sex::BIZYGOMATIC);
        check(format!("{name} face height / bizygomatic breadth"), g.face / g.bizygomatic, index * 0.93, index * 1.08);
        // Nose (Farkas, within 2 SD) and the lower face's share of the face height.
        let (n, n_sd) = if w < 0.0 { (0.0506, 0.0031) } else { (0.0548, 0.0033) };
        check(format!("{name} nose height n-sn mm"), g.nose * 1e3, (n - 2.0 * n_sd) * 1e3, (n + 2.0 * n_sd) * 1e3);
        check(format!("{name} nose protrusion sn-prn mm"), g.protrusion * 1e3, 16.0, 24.0);
        check(format!("{name} lower face / face height"), g.lower / (g.nose + g.lower), 0.52, 0.62);
        // Eyes against the face breadth: fissure 31.3 / 137.1 and 30.7 / 129.9 in Farkas.
        check(format!("{name} eye fissure / bizygomatic breadth"), g.fissure / g.bizygomatic, 0.20, 0.245);
        check(format!("{name} eye opening / fissure length"), g.opening / g.fissure, 0.28, 0.42);
        // Chin: a labiomental fold of 2 to 8 mm, the pogonion at most 12 mm behind the subnasale,
        // and an underside that does not run down into the neck by more than 10 mm.
        check(format!("{name} labiomental fold depth mm"), g.sulcus * 1e3, 2.0, 8.0);
        check(format!("{name} pogonion behind subnasale mm"), g.chin_setback * 1e3, -4.0, 12.0);
        check(format!("{name} submental drop mm"), g.submental_drop * 1e3, 0.0, 10.0);
        // Jaw against the cheekbones (reported with the face breadth; no ANSUR II variable).
        println!(
            "{name}: face breadth {:.1} mm, at the mouth {:.1} mm ({:.3} of it)",
            g.face_breadth * 1e3,
            g.jaw_breadth * 1e3,
            g.jaw_breadth / g.face_breadth
        );
        // `identity.age` (0 at the default: the adult of the ANSUR II means) moves the face the
        // right way, and the youngest setting of its usual span is still an adult.
        let old = adult_gauge(t, &on, &[(sex::SWITCH, w), ("global.age", 2.0)], false);
        let young = adult_gauge(t, &on, &[(sex::SWITCH, w), ("global.age", -2.0)], false);
        println!(
            "{name} age -2 / 0 / +2: nose {:.1} / {:.1} / {:.1} mm, ear {:.1} / {:.1} / {:.1} mm, lips {:.1} / {:.1} / {:.1} mm, malar height {:+.1} / 0 / {:+.1} mm",
            young.nose * 1e3,
            g.nose * 1e3,
            old.nose * 1e3,
            young.ear * 1e3,
            g.ear * 1e3,
            old.ear * 1e3,
            young.lips * 1e3,
            g.lips * 1e3,
            old.lips * 1e3,
            (young.malar_y - g.malar_y) * 1e3,
            (old.malar_y - g.malar_y) * 1e3
        );
        assert!(old.nose > g.nose && g.nose > young.nose, "{name}: the nose grows with age");
        assert!(old.ear > g.ear && g.ear > young.ear, "{name}: the ears grow with age");
        assert!(old.lips < g.lips && g.lips < young.lips, "{name}: the lips thin with age");
        assert!(old.malar_y < g.malar_y && g.malar_y < young.malar_y, "{name}: the cheeks descend with age");
        check(format!("{name} at age -2: stature in head heights"), young.stature / young.head, 7.0, 8.0);
        check(
            format!("{name} at age -2: face height / vault height"),
            young.face / young.vault,
            ratio * 0.92,
            ratio * 1.1,
        );
        check(
            format!("{name} at age -2: nose height n-sn mm"),
            young.nose * 1e3,
            (n - 2.0 * n_sd) * 1e3,
            (n + 2.0 * n_sd) * 1e3,
        );
    }
    // The body's adult build: shoulders against hips (ANSUR II means: 1.03 women, 1.20 men).
    let f = fixture();
    let space = Space::new(&f.base).unwrap();
    let kit = Kit::new(&space).unwrap();
    let (raw, _) = sex::fit(&kit).unwrap();
    let (sh_f, sh_m) = sex::states(&space, &measure::BIACROMIAL_GAUGE, &raw).unwrap();
    let (hip_f, hip_m) = sex::states(&space, &measure::HIP_BREADTH_GAUGE, &raw).unwrap();
    let (b, h) = (measure::BIACROMIAL, measure::HIP_BREADTH);
    let (want_f, want_m) = (b.female_mean / h.female_mean, b.male_mean / h.male_mean);
    check("female biacromial / hip breadth".to_owned(), sh_f / hip_f, want_f - 0.05, want_f + 0.05);
    check("male biacromial / hip breadth".to_owned(), sh_m / hip_m, want_m - 0.05, want_m + 0.05);
    assert!(failed.is_empty(), "{failed:#?}");
}

/// The ANSUR II constants of the adult-proportion test match the data file, and the survey's
/// mean age is about 30 (what `identity.age` 0 stands for).
#[test]
fn todo_1_3_adult_test_constants_match_the_data_file() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/anthropometry/ansur2.json");
    let value = humanart_io::json::parse(&std::fs::read(path).unwrap()).unwrap();
    let root = value.node();
    let stats = root.field("stats").unwrap();
    for v in [&TRAGION_TOP, &HEAD_BREADTH] {
        for (sex, mean, sd) in [("male", v.male_mean, v.male_sd), ("female", v.female_mean, v.female_sd)] {
            let s = stats.field(sex).unwrap();
            let s = s.field(v.id).unwrap();
            let (m, d) = (s.field("mean").unwrap().f64().unwrap(), s.field("sd").unwrap().f64().unwrap());
            assert!((m - f64::from(mean)).abs() < 0.01 && (d - f64::from(sd)).abs() < 0.01, "{} {sex}", v.id);
        }
    }
    for (sex, lo, hi) in [("male", 29.0, 31.0), ("female", 28.0, 30.0)] {
        let s = stats.field(sex).unwrap();
        let age = s.field("age").unwrap().field("mean").unwrap().f64().unwrap();
        assert!((lo..=hi).contains(&age), "{sex} mean age {age}");
    }
}
