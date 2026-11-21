//! Test-only code of the expression integration tests: a small software rasteriser for preview
//! contact sheets and the integration tests on the generated template.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use humanart_io::{ColorSpace, Format, Image, Samples};
use humanart_math::{Quat, Vec3};

use crate::eval::{EvalInput, Evaluator};
use crate::expression::build::{CorrectiveSpecs, with_expressions};
use crate::expression::field::FaceField;
use crate::generate::build::{Options, template};
use crate::template::Template;

/// The preview folder of this agent.
pub fn preview_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/agents/expression2/previews");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The generated template with the expression model, built once per test process.
pub fn expressive() -> Arc<Template> {
    static T: OnceLock<Arc<Template>> = OnceLock::new();
    T.get_or_init(|| {
        let g = template(&Options { lods: false, ..Options::default() }).unwrap();
        // `build::template` adds the expression model itself; use it as generated.
        if !g.template.expressions().expressions.is_empty() {
            return Arc::new(g.template);
        }
        let start = std::time::Instant::now();
        let (t, report) = with_expressions(&g.template, &CorrectiveSpecs::builtin().unwrap()).unwrap();
        let entries: usize = report.expressions.iter().chain(&report.correctives).map(|e| e.1).sum();
        println!(
            "with_expressions: {:.2} s, {} expressions, {} correctives, {entries} entries",
            start.elapsed().as_secs_f32(),
            report.expressions.len(),
            report.correctives.len()
        );
        Arc::new(t)
    })
    .clone()
}

/// One tile: deformed positions and normals of the template.
pub struct Tile {
    pub pos: Vec<Vec3>,
    pub nrm: Vec<Vec3>,
    /// Per triangle: drawn red (inverted against the neutral); empty for none.
    pub flag: Vec<bool>,
}

/// Which triangles to draw and their base colours (parts drawn as overlays are skipped).
fn triangle_colours(t: &Template, tris: &[u32]) -> Vec<Option<[f32; 3]>> {
    let f = FaceField::new(t).unwrap();
    let parts = t.mesh().parts();
    tris.chunks(3)
        .map(|c| {
            let p = f.part_of(f.node_of(c[0] as usize) as usize) as usize;
            let name = parts.get(p).map_or("", |p| p.name.as_str());
            let region = t.regions().vertex_regions.get(c[0] as usize).copied().unwrap_or(0);
            let interior = t.regions().regions.get(region as usize).is_some_and(|r| r.name == "mouth-interior");
            match name {
                "body" if interior => Some([0.75, 0.25, 0.25]),
                "body" => Some([0.80, 0.56, 0.45]),
                "mouth-helper" => Some([0.2, 0.5, 0.9]),
                n if n.starts_with("eyeball") => Some([0.85, 0.85, 0.82]),
                n if n.starts_with("teeth") => Some([0.92, 0.90, 0.80]),
                n if n.starts_with("gums") => Some([0.75, 0.35, 0.35]),
                "tongue" => Some([0.80, 0.30, 0.32]),
                _ => None,
            }
        })
        .collect()
}

/// Renders tiles side by side (`cols` per row), each an orthographic view along `forward` of the
/// window `(centre, half_height)`, and writes `<name>.png`.
pub fn render_sheet(
    name: &str,
    t: &Template,
    tris: &[u32],
    tiles: &[Tile],
    forward: Vec3,
    centre: Vec3,
    half: f32,
    cols: usize,
) {
    let (tw, th) = (300_usize, 340_usize);
    let rows = tiles.len().div_ceil(cols);
    let (w, h) = (tw * cols, th * rows);
    let mut color = vec![[0.93_f32, 0.93, 0.95]; w * h];
    let colours = triangle_colours(t, tris);
    let f = forward.normalize();
    let r = f.cross(Vec3::Y).normalize();
    let u = r.cross(f);
    let scale = th as f32 / (2.0 * half);
    let key = (-f + u * 0.7 - r * 0.45).normalize();
    let fill = (-f - u * 0.2 + r * 0.6).normalize();
    for (k, tile) in tiles.iter().enumerate() {
        let (ox, oy) = ((k % cols) * tw, (k / cols) * th);
        let mut depth = vec![f32::INFINITY; tw * th];
        let project = |p: Vec3| {
            let d = p - centre;
            Vec3::new(tw as f32 * 0.5 + d.dot(r) * scale, th as f32 * 0.5 - d.dot(u) * scale, d.dot(f))
        };
        for (ti, (c, base)) in tris.chunks(3).zip(&colours).enumerate() {
            let flagged = tile.flag.get(ti).copied().unwrap_or(false);
            let base = if flagged {
                &[1.0, 0.0, 0.0]
            } else {
                match base {
                    Some(b) => b,
                    None => continue,
                }
            };
            let s =
                [project(tile.pos[c[0] as usize]), project(tile.pos[c[1] as usize]), project(tile.pos[c[2] as usize])];
            let shade = |v: u32| {
                let n = tile.nrm[v as usize];
                let k = 0.22 + 0.62 * n.dot(key).max(0.0) + 0.22 * n.dot(fill).max(0.0);
                base.map(|x| x * k)
            };
            let cs = [shade(c[0]), shade(c[1]), shade(c[2])];
            let (minx, maxx) =
                (s.iter().map(|p| p.x).fold(f32::MAX, f32::min), s.iter().map(|p| p.x).fold(f32::MIN, f32::max));
            let (miny, maxy) =
                (s.iter().map(|p| p.y).fold(f32::MAX, f32::min), s.iter().map(|p| p.y).fold(f32::MIN, f32::max));
            let area = (s[1].x - s[0].x) * (s[2].y - s[0].y) - (s[2].x - s[0].x) * (s[1].y - s[0].y);
            if area.abs() < 1e-12 {
                continue;
            }
            let x0 = minx.floor().max(0.0) as usize;
            let x1 = (maxx.ceil() as isize).min(tw as isize - 1);
            let y0 = miny.floor().max(0.0) as usize;
            let y1 = (maxy.ceil() as isize).min(th as isize - 1);
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
                    let i = y * tw + x;
                    if z < depth[i] {
                        depth[i] = z;
                        let mut c = [0.0; 3];
                        for ch in 0..3 {
                            c[ch] = w0 * cs[0][ch] + w1 * cs[1][ch] + w2 * cs[2][ch];
                        }
                        color[(oy + y) * w + ox + x] = c;
                    }
                }
            }
        }
    }
    let mut bytes = Vec::with_capacity(w * h * 3);
    for c in &color {
        for x in c {
            let s = if *x <= 0.003_130_8 { x * 12.92 } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
            bytes.push((s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
    }
    let image = Image::new(Format::Png, w as u32, h as u32, 3, ColorSpace::Srgb, Samples::U8(bytes)).unwrap();
    std::fs::write(preview_dir().join(format!("{name}.png")), humanart_io::png::encode(&image).unwrap()).unwrap();
}

/// The joint-local pose of a world rotation of joint `j` at rest.
pub fn local_pose(t: &Template, j: usize, world: Quat) -> Quat {
    let rb = t.skeleton().rest_world()[j].rotation;
    (rb.conjugate() * world * rb).normalize()
}

#[test]
fn preview_all_targets() {
    let t = expressive();
    let mut ev = Evaluator::new(t.clone());
    let set = t.expressions();
    let jaw = t.skeleton().find("jaw").unwrap() as usize;
    let n = set.expressions.len();
    let mut tiles = Vec::new();
    let mut labels = Vec::new();
    let tris = ev.triangles().to_vec();
    for k in 0..=n {
        let mut w = vec![0.0; n];
        let mut pose = vec![Quat::IDENTITY; t.skeleton().joints.len()];
        let label = if k == 0 { "neutral".to_owned() } else { set.expressions[k - 1].name.clone() };
        if k > 0 {
            w[k - 1] = 1.0;
            let angle = match label.as_str() {
                "AU26" => crate::expression::recipes::JAW_DROP,
                "AU27" => crate::expression::recipes::MOUTH_STRETCH,
                "viseme.jaw-open-round" => crate::expression::recipes::VISEME_ROUND,
                "viseme.jaw-open-wide" => crate::expression::recipes::VISEME_WIDE,
                _ => 0.0,
            };
            pose[jaw] = local_pose(&t, jaw, Quat::from_axis_angle(Vec3::X, angle));
        }
        ev.evaluate(&EvalInput { expressions: &w, pose: &pose, ..EvalInput::default() }).unwrap();
        tiles.push(Tile { pos: ev.positions().to_vec(), nrm: ev.normals().to_vec(), flag: Vec::new() });
        labels.push(label);
    }
    let centre = Vec3::new(0.0, 1.548, 0.1);
    for (s, chunk) in tiles.chunks(18).enumerate() {
        let names = &labels[s * 18..s * 18 + chunk.len()];
        std::fs::write(preview_dir().join(format!("targets-{s}.txt")), names.join("\n")).unwrap();
        render_sheet(&format!("targets-{s}"), &t, &tris, chunk, -Vec3::Z, centre, 0.085, 6);
    }
}

/// The registry's expression target names and control ids (`data/template/params/expression.json`).
fn registry() -> (Vec<String>, Vec<String>) {
    let text = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/template/params/expression.json"),
    )
    .unwrap();
    let v = humanart_io::json::parse_str(&text).unwrap();
    let (mut targets, mut controls) = (Vec::new(), Vec::new());
    for p in v.get("parameters").unwrap().as_array().unwrap() {
        let id = p.get("id").unwrap().as_str().unwrap();
        let t = p.get("evaluation").unwrap().get("target").unwrap().as_str().unwrap();
        if p.get("sides").and_then(|s| if let humanart_io::json::Value::Bool(b) = s { Some(*b) } else { None })
            == Some(true)
        {
            for (l, u) in [("l", "L"), ("r", "R")] {
                controls.push(format!("{id}.{l}"));
                targets.push(t.replace("{side}", u));
            }
        } else {
            controls.push(id.to_owned());
            targets.push(t.to_owned());
        }
    }
    (targets, controls)
}

#[test]
fn exp_01_every_appendix_b_control_has_its_target_and_rig_control() {
    let t = expressive();
    let (targets, controls) = registry();
    assert_eq!(targets.len(), 82);
    let set = t.expressions();
    for name in &targets {
        assert!(set.expressions.iter().any(|e| &e.name == name), "missing target {name}");
    }
    let rig = crate::expression::controls::ControlRig::builtin(&t).unwrap();
    for c in &controls {
        assert!(rig.controls().contains(c), "rig lacks control {c}");
    }
    // Every expression is driven by the rig.
    let driven: Vec<u32> = rig.logic().outputs().map(|(e, _)| e).collect();
    for (i, e) in set.expressions.iter().enumerate() {
        assert!(driven.contains(&(i as u32)), "{} is not driven", e.name);
    }
    // Sided targets are mirror images: equal displacement magnitude sums over welded positions.
    let d = t.deltas();
    let mag = |name: &str| -> f32 {
        let c = d.find(name).unwrap() as usize;
        let keys = t.mesh().position_keys();
        d.indices(c)
            .iter()
            .zip(d.deltas(c))
            .filter(|(v, _)| keys[**v as usize] == **v)
            .map(|(_, &x)| crate::delta::DeltaBases::delta_f32(x).length())
            .sum()
    };
    for e in &set.expressions {
        if let Some(base) = e.name.strip_suffix("-L") {
            let (l, r) = (mag(&e.name), mag(&format!("{base}-R")));
            assert!((l - r).abs() <= 0.02 * l.max(r) + 1e-6, "{} {l} vs {r}", e.name);
        }
    }
    let report = crate::validate::validate(&t);
    println!("{report}");
}

#[test]
fn exp_01_jaw_drivers_match_the_support_shapes() {
    use crate::expression::recipes::{JAW_DROP, JAW_SIDE, MOUTH_STRETCH, VISEME_ROUND, VISEME_WIDE};
    let t = expressive();
    let rig = crate::expression::controls::ControlRig::builtin(&t).unwrap();
    let jaw = t.skeleton().find("jaw").unwrap() as usize;
    let rb = t.skeleton().rest_world()[jaw].rotation;
    for (control, axis, angle) in [
        ("expression.au26_jaw_drop", Vec3::X, JAW_DROP),
        ("expression.au27_mouth_stretch", Vec3::X, MOUTH_STRETCH),
        ("viseme.jaw-open-round", Vec3::X, VISEME_ROUND),
        ("viseme.jaw-open-wide", Vec3::X, VISEME_WIDE),
        ("expression.au30_jaw_sideways.l", Vec3::Y, JAW_SIDE),
        ("expression.au30_jaw_sideways.r", Vec3::Y, -JAW_SIDE),
    ] {
        let mut c = vec![0.0; rig.controls().len()];
        c[rig.controls().iter().position(|x| x == control).unwrap()] = 1.0;
        let mut regs = vec![0.0; rig.register_count()];
        let mut w = vec![0.0; t.expressions().expressions.len()];
        let mut pose = vec![Quat::IDENTITY; t.skeleton().joints.len()];
        rig.evaluate(&c, &mut regs, &mut w, &mut pose).unwrap();
        let world = rb * pose[jaw] * rb.conjugate();
        assert!(world.angle_between(Quat::from_axis_angle(axis, angle)) < 1e-4, "{control}");
    }
    // Opening the jaw moves the chin down and back.
    let mut ev = Evaluator::new(t.clone());
    let mut pose = vec![Quat::IDENTITY; t.skeleton().joints.len()];
    pose[jaw] = local_pose(&t, jaw, Quat::from_axis_angle(Vec3::X, JAW_DROP));
    ev.evaluate(&EvalInput { pose: &pose, ..EvalInput::default() }).unwrap();
    let chin = t.landmarks().iter().find(|l| l.name == "chin.mid").unwrap();
    let (before, after) = (chin.position(t.mesh().positions()).unwrap(), chin.position(ev.positions()).unwrap());
    assert!(after.y < before.y - 0.01 && after.z < before.z, "{before:?} -> {after:?}");
}

#[test]
fn ac_wp19_1_range_of_motion_sweep() {
    let t = expressive();
    let rig = crate::expression::controls::ControlRig::builtin(&t).unwrap();
    let start = std::time::Instant::now();
    let metrics = crate::expression::sweep::sweep(&t, &rig, 4).unwrap();
    let limits = crate::expression::sweep::SweepLimits::default();
    let mut text = String::from("control | inverted | max disp mm | area min | area max | most compressed region\n");
    let mut failed = Vec::new();
    for m in &metrics {
        writeln!(
            text,
            "{} | {} | {:.2} | {:.3} | {:.3} | {} {:.4}",
            m.control,
            m.inverted,
            m.max_displacement * 1e3,
            m.min_area_ratio,
            m.max_area_ratio,
            m.most_compressed.0,
            m.most_compressed.1
        )
        .unwrap();
        if !m.passes(&limits) {
            failed.push(m.control.clone());
        }
    }
    writeln!(text, "{} controls, {} failed, {:.1} s", metrics.len(), failed.len(), start.elapsed().as_secs_f32())
        .unwrap();
    std::fs::write(preview_dir().join("../sweep.txt"), &text).unwrap();
    let rigtext = crate::expression::controls::controls_json(&t, &rig).unwrap();
    std::fs::write(preview_dir().join("../controls.json"), &rigtext).unwrap();
    assert!(failed.is_empty(), "{failed:?}\n{text}");
}

#[test]
fn preview_closeups() {
    let t = expressive();
    let rig = crate::expression::controls::ControlRig::builtin(&t).unwrap();
    let mut ev = Evaluator::new(t.clone());
    ev.evaluate(&EvalInput::default()).unwrap();
    let neutral = ev.positions().to_vec();
    let tris = ev.triangles().to_vec();
    let list = std::env::var("CLOSEUP").unwrap_or_else(|_| "eye:expression.au07_lid_tightener.l,expression.au42_slit.l,expression.au43_eyes_closed.l,expression.au44_squint.l".into());
    let (what, names) = list.split_once(':').unwrap();
    let names: Vec<&str> = std::iter::once("neutral").chain(names.split(',')).collect();
    let half_scale: f32 = std::env::var("CLOSEUP_HALF").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let mut front = Vec::new();
    let mut side = Vec::new();
    for name in &names {
        let mut c = vec![0.0; rig.controls().len()];
        // `a+b=0.5`: several controls, each at 1 or at its `=value`.
        for item in name.split('+') {
            let (control, value) = item.split_once('=').map_or((item, 1.0), |(a, b)| (a, b.parse().unwrap()));
            if let Some(i) = rig.controls().iter().position(|x| x == control) {
                c[i] = value;
            }
        }
        let mut regs = vec![0.0; rig.register_count()];
        let mut w = vec![0.0; t.expressions().expressions.len()];
        let mut pose = vec![Quat::IDENTITY; t.skeleton().joints.len()];
        rig.evaluate(&c, &mut regs, &mut w, &mut pose).unwrap();
        ev.evaluate(&EvalInput { expressions: &w, pose: &pose, ..EvalInput::default() }).unwrap();
        let flag = inverted(&tris, &neutral, ev.positions());
        front.push(Tile { pos: ev.positions().to_vec(), nrm: ev.normals().to_vec(), flag: flag.clone() });
        side.push(Tile { pos: ev.positions().to_vec(), nrm: ev.normals().to_vec(), flag });
    }
    let (centre, half, side_dir) = match what {
        "eye" => (
            t.landmarks().iter().find(|l| l.name == "L.iris.center").unwrap().position(t.mesh().positions()).unwrap(),
            0.016,
            Vec3::new(-1.0, 0.0, -0.6),
        ),
        "mouth" => {
            let at =
                |n: &str| t.landmarks().iter().find(|l| l.name == n).unwrap().position(t.mesh().positions()).unwrap();
            let dx: f32 = std::env::var("CLOSEUP_DX").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            (
                (at("lip.upper.inner.06") + at("lip.lower.inner.06")) * 0.5 + Vec3::X * dx,
                0.035 * half_scale,
                Vec3::new(-1.0, 0.0, -0.4),
            )
        }
        _ => (Vec3::new(0.0, 1.548, 0.1), 0.085, Vec3::new(-1.0, 0.0, -0.6)),
    };
    front.extend(side);
    let tiles = front;
    let mut all = Vec::new();
    let n = names.len();
    for (i, tile) in tiles.into_iter().enumerate() {
        all.push((i, tile));
    }
    let (f, s): (Vec<_>, Vec<_>) = all.into_iter().partition(|(i, _)| *i < n);
    let f: Vec<Tile> = f.into_iter().map(|x| x.1).collect();
    let s: Vec<Tile> = s.into_iter().map(|x| x.1).collect();
    render_sheet(&format!("closeup-{what}-front"), &t, &tris, &f, -Vec3::Z, centre, half, n);
    render_sheet(&format!("closeup-{what}-side"), &t, &tris, &s, side_dir, centre, half, n);
    std::fs::write(preview_dir().join(format!("closeup-{what}.txt")), names.join("\n")).unwrap();
}

#[test]
fn exp_07_presets_resolve_and_preview() {
    let t = expressive();
    let rig = crate::expression::controls::ControlRig::builtin(&t).unwrap();
    let presets = crate::expression::presets::Presets::builtin().unwrap();
    presets.resolve(&rig).unwrap();
    let mut ev = Evaluator::new(t.clone());
    ev.evaluate(&EvalInput::default()).unwrap();
    let neutral = ev.positions().to_vec();
    let tris = ev.triangles().to_vec();
    let mut tiles = Vec::new();
    let mut names = Vec::new();
    let mut report = String::new();
    for p in &presets.presets {
        let mut c = vec![0.0; rig.controls().len()];
        presets.apply(p, &rig, &mut c).unwrap();
        let mut regs = vec![0.0; rig.register_count()];
        let mut w = vec![0.0; t.expressions().expressions.len()];
        let mut pose = vec![Quat::IDENTITY; t.skeleton().joints.len()];
        rig.evaluate(&c, &mut regs, &mut w, &mut pose).unwrap();
        ev.evaluate(&EvalInput { expressions: &w, pose: &pose, ..EvalInput::default() }).unwrap();
        let flag = inverted(&tris, &neutral, ev.positions());
        writeln!(report, "{}: {} inverted", p.name, flag.iter().filter(|&&x| x).count()).unwrap();
        tiles.push(Tile { pos: ev.positions().to_vec(), nrm: ev.normals().to_vec(), flag });
        names.push(p.name.clone());
    }
    std::fs::write(preview_dir().join("../presets.txt"), &report).unwrap();
    let centre = Vec3::new(0.0, 1.548, 0.1);
    for (s, chunk) in tiles.chunks(18).enumerate() {
        std::fs::write(preview_dir().join(format!("presets-{s}.txt")), names[s * 18..s * 18 + chunk.len()].join("\n"))
            .unwrap();
        render_sheet(&format!("presets-{s}"), &t, &tris, chunk, Vec3::new(-0.35, 0.0, -1.0), centre, 0.085, 6);
    }
}

#[test]
fn ac_wp19_2_wrinkle_drivers_fold_at_expected_activations() {
    let t = expressive();
    let rig = crate::expression::controls::ControlRig::builtin(&t).unwrap();
    let drivers = crate::expression::controls::wrinkle_drivers(&t).unwrap();
    let mut ev = Evaluator::new(t.clone());
    let (mut fold, mut stretch) = (vec![0.0; drivers.len()], vec![0.0; drivers.len()]);
    let mut text = String::from(
        "control | driver | fold | stretch | region compression
",
    );
    for (control, driver) in [
        ("neutral", "forehead.raise"),
        ("expression.au01_inner_brow_raiser.l+expression.au01_inner_brow_raiser.r", "forehead.raise"),
        ("expression.au02_outer_brow_raiser.l+expression.au02_outer_brow_raiser.r", "forehead.raise"),
        ("expression.au04_brow_lowerer.l", "glabella.L"),
        ("expression.au06_cheek_raiser.l", "crows-feet.L"),
        ("expression.au12_lip_corner_puller.l", "nasolabial.L"),
        ("expression.au11_nasolabial_deepener.r", "nasolabial.R"),
    ] {
        let mut c = vec![0.0; rig.controls().len()];
        for name in control.split('+') {
            if let Some(i) = rig.controls().iter().position(|x| x == name) {
                c[i] = 1.0;
            }
        }
        let mut regs = vec![0.0; rig.register_count()];
        let mut w = vec![0.0; t.expressions().expressions.len()];
        let mut pose = vec![Quat::IDENTITY; t.skeleton().joints.len()];
        rig.evaluate(&c, &mut regs, &mut w, &mut pose).unwrap();
        ev.evaluate(&EvalInput { expressions: &w, pose: &pose, ..EvalInput::default() }).unwrap();
        let strain = ev.strain();
        drivers.evaluate(&w, strain.region_stretch(), strain.region_compression(), &mut fold, &mut stretch).unwrap();
        let d = drivers.names().iter().position(|n| n == driver).unwrap();
        let r = drivers.regions()[d] as usize;
        writeln!(
            text,
            "{control} | {driver} | {:.3} | {:.3} | {:.4}",
            fold[d],
            stretch[d],
            strain.region_compression()[r]
        )
        .unwrap();
        if control == "neutral" {
            assert!(fold.iter().all(|&f| f < 0.01), "{fold:?}");
        } else {
            assert!(fold[d] >= 0.5, "{control}: {driver} fold {}", fold[d]);
            assert!(strain.region_compression()[r] > 0.0, "{control}: no compression in {driver}'s region");
        }
    }
    std::fs::write(preview_dir().join("../wrinkles.txt"), &text).unwrap();
}

/// Per triangle, whether it is inverted in `pos` against `neutral`.
fn inverted(tris: &[u32], neutral: &[Vec3], pos: &[Vec3]) -> Vec<bool> {
    tris.chunks(3)
        .map(|t| {
            let n = |p: &[Vec3]| (p[t[1] as usize] - p[t[0] as usize]).cross(p[t[2] as usize] - p[t[0] as usize]);
            n(neutral).dot(n(pos)) < 0.0
        })
        .collect()
}

/// Diagnosis of sweep failures: for every control named in `DIAG` (comma separated), the
/// triangles that turn over or leave the area-ratio bounds at full weight, with part, region,
/// position relative to the mouth centre (mm), neutral area (mm²) and ratio.
#[test]
#[ignore = "diagnostic: DIAG=control,control cargo test diagnose -- --ignored --nocapture"]
fn diagnose_sweep_failures() {
    use crate::expression::sweep::{AREA_EXEMPT_PREFIXES, MIN_RATIO_AREA};
    let t = expressive();
    let rig = crate::expression::controls::ControlRig::builtin(&t).unwrap();
    let mut ev = Evaluator::new(t.clone());
    ev.evaluate(&EvalInput::default()).unwrap();
    let neutral = ev.positions().to_vec();
    let tris = ev.triangles().to_vec();
    let mesh = t.mesh();
    let tri_faces = mesh.triangulate().triangle_faces;
    let mc = (t
        .landmarks()
        .iter()
        .find(|l| l.name == "lip.upper.inner.06")
        .unwrap()
        .position(mesh.positions())
        .unwrap()
        + t.landmarks().iter().find(|l| l.name == "lip.lower.inner.06").unwrap().position(mesh.positions()).unwrap())
        * 0.5;
    let list = std::env::var("DIAG").unwrap_or_default();
    let steps: f32 = std::env::var("DIAG_STEP").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    // `DIAG=a,b+c=0.5`: each comma-separated entry is one evaluation of `+`-joined controls, each
    // at `=value` or at `DIAG_STEP`.
    for name in list.split(',').filter(|s| !s.is_empty()) {
        let mut c = vec![0.0; rig.controls().len()];
        let mut missing = false;
        for item in name.split('+') {
            let (control, value) = item.split_once('=').map_or((item, steps), |(a, b)| (a, b.parse().unwrap()));
            match rig.controls().iter().position(|x| x == control) {
                Some(i) => c[i] = value,
                None => missing = true,
            }
        }
        if missing {
            println!("{name}: no such control");
            continue;
        }
        let mut regs = vec![0.0; rig.register_count()];
        let mut w = vec![0.0; t.expressions().expressions.len()];
        let mut pose = vec![Quat::IDENTITY; t.skeleton().joints.len()];
        rig.evaluate(&c, &mut regs, &mut w, &mut pose).unwrap();
        ev.evaluate(&EvalInput { expressions: &w, pose: &pose, ..EvalInput::default() }).unwrap();
        let pos = ev.positions();
        println!("== {name} at {steps}");
        let mut lines = 0;
        for (k, tr) in tris.chunks(3).enumerate() {
            let n0 = (neutral[tr[1] as usize] - neutral[tr[0] as usize])
                .cross(neutral[tr[2] as usize] - neutral[tr[0] as usize]);
            let n1 = (pos[tr[1] as usize] - pos[tr[0] as usize]).cross(pos[tr[2] as usize] - pos[tr[0] as usize]);
            let (a0, a1) = (0.5 * n0.length(), 0.5 * n1.length());
            if a0 <= 1e-12 {
                continue;
            }
            let f = tri_faces[k] as usize;
            let part = mesh.part_of_face(f).and_then(|p| mesh.parts().get(p)).map_or("?", |p| p.name.as_str());
            let exempt = AREA_EXEMPT_PREFIXES.iter().any(|x| part.starts_with(x));
            let r = a1 / a0;
            let inv = n0.dot(n1) < 0.0;
            let bad_ratio = a0 >= MIN_RATIO_AREA && !exempt && !(0.2..=5.0).contains(&r);
            if !(inv || bad_ratio) {
                continue;
            }
            lines += 1;
            if lines > 40 {
                continue;
            }
            let v = tr[0] as usize;
            let reg = t.regions().vertex_regions.get(v).copied().unwrap_or(0);
            let rname = t.regions().regions.get(reg as usize).map_or("?", |r| r.name.as_str());
            let cen = (neutral[tr[0] as usize] + neutral[tr[1] as usize] + neutral[tr[2] as usize]) / 3.0 - mc;
            println!(
                "  {} {part:>12} {rname:>14} at ({:6.1},{:6.1},{:6.1}) mm a0 {:.4} mm2 ratio {r:.3} verts {:?}",
                if inv { "INV" } else { "   " },
                cen.x * 1e3,
                cen.y * 1e3,
                cen.z * 1e3,
                a0 * 1e6,
                tr
            );
        }
        println!("  {lines} bad triangles");
    }
}

/// Diagnosis: skin weights (joint name: weight) and neutral position of the vertices in `DIAG_V`.
#[test]
#[ignore = "diagnostic: DIAG_V=1,2,3 cargo test diagnose_skin -- --ignored --nocapture"]
fn diagnose_skin() {
    let t = expressive();
    let list = std::env::var("DIAG_V").unwrap_or_default();
    let p = t.mesh().positions();
    for v in list.split(',').filter_map(|s| s.trim().parse::<usize>().ok()) {
        let (js, ws) = t.skin().vertex(v);
        let named: Vec<String> =
            js.iter().zip(ws).map(|(&j, w)| format!("{}:{w:.3}", t.skeleton().joints[j as usize].name)).collect();
        println!("{v} ({:.2},{:.2},{:.2}) mm {}", p[v].x * 1e3, p[v].y * 1e3, p[v].z * 1e3, named.join(" "));
    }
}

/// Diagnosis: neutral position (mm, relative to the mouth centre) and displacement (mm) of the
/// vertices in `DIAG_V` under each expression of `DIAG_E` (comma separated) at weight 1 without
/// pose, and under the jaw pose of `DIAG_JAW` radians alone.
#[test]
#[ignore = "diagnostic"]
fn diagnose_target() {
    let t = expressive();
    let mut ev = Evaluator::new(t.clone());
    ev.evaluate(&EvalInput::default()).unwrap();
    let neutral = ev.positions().to_vec();
    let mesh = t.mesh();
    let at = |name: &str| t.landmarks().iter().find(|l| l.name == name).unwrap().position(mesh.positions()).unwrap();
    let mc = (at("lip.upper.inner.06") + at("lip.lower.inner.06")) * 0.5;
    let list: Vec<usize> =
        std::env::var("DIAG_V").unwrap_or_default().split(',').filter_map(|s| s.trim().parse::<usize>().ok()).collect();
    for v in &list {
        let p = (neutral[*v] - mc) * 1e3;
        println!("{v}: neutral ({:.2},{:.2},{:.2})", p.x, p.y, p.z);
    }
    let jaw_angle: f32 = std::env::var("DIAG_JAW").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let jaw = t.skeleton().find("jaw").unwrap() as usize;
    let mut pose = vec![Quat::IDENTITY; t.skeleton().joints.len()];
    pose[jaw] = local_pose(&t, jaw, Quat::from_axis_angle(Vec3::X, jaw_angle));
    ev.evaluate(&EvalInput { pose: &pose, ..EvalInput::default() }).unwrap();
    let b = ev.positions().to_vec();
    for v in &list {
        let q = (b[*v] - neutral[*v]) * 1e3;
        println!("{v}: jaw {jaw_angle} ({:.3},{:.3},{:.3})", q.x, q.y, q.z);
    }
    for e in std::env::var("DIAG_E").unwrap_or_default().split(',').filter(|s| !s.is_empty()) {
        let k = t.expressions().expressions.iter().position(|x| x.name == e).unwrap();
        let mut w = vec![0.0; t.expressions().expressions.len()];
        w[k] = 1.0;
        ev.evaluate(&EvalInput { expressions: &w, ..EvalInput::default() }).unwrap();
        let a = ev.positions().to_vec();
        for v in &list {
            let d = (a[*v] - neutral[*v]) * 1e3;
            println!("{v}: {e} ({:.3},{:.3},{:.3})", d.x, d.y, d.z);
        }
    }
}
