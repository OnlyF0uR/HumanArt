//! Test-only code of the generator: a small software rasteriser for preview PNGs (flat Lambert or
//! region colours, orthographic views, optional wireframe), OBJ export, and the integration tests.

use std::path::PathBuf;

use humanart_io::{ColorSpace, Format, Image, Samples};
use humanart_math::{Vec2, Vec3};

/// The preview folder of this agent.
pub fn preview_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/agents/template-body/previews");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A mesh to draw: positions and polygons with a colour each.
pub struct Scene {
    pub pos: Vec<Vec3>,
    pub faces: Vec<(Vec<u32>, [f32; 3])>,
}

/// An orthographic view.
#[derive(Clone, Copy)]
pub struct View {
    pub name: &'static str,
    /// Direction the camera looks in.
    pub forward: Vec3,
    pub up: Vec3,
    /// Centre and half height of the visible window; `None` fits the scene.
    pub window: Option<(Vec3, f32)>,
    pub wire: bool,
    pub size: (u32, u32),
    /// Light grazing from above (shows low relief).
    pub raking: bool,
}

impl View {
    pub fn new(name: &'static str, forward: Vec3) -> Self {
        Self {
            name,
            forward: forward.normalize(),
            up: Vec3::Y,
            window: None,
            wire: false,
            size: (900, 1200),
            raking: false,
        }
    }
    pub fn window(mut self, centre: Vec3, half_height: f32) -> Self {
        self.window = Some((centre, half_height));
        self
    }
    pub fn raking(mut self) -> Self {
        self.raking = true;
        self
    }
    pub fn wire(mut self) -> Self {
        self.wire = true;
        self
    }
    pub fn size(mut self, w: u32, h: u32) -> Self {
        self.size = (w, h);
        self
    }
}

/// Renders `scene` from `view` and writes `<name>.png` into the preview folder.
pub fn render(scene: &Scene, view: &View) {
    let (w, h) = view.size;
    let f = view.forward;
    let r = f.cross(view.up).normalize();
    let u = r.cross(f);
    let (centre, half) = view.window.unwrap_or_else(|| {
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for p in &scene.pos {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let c = (lo + hi) * 0.5;
        let mut ext = 0.0_f32;
        for p in &scene.pos {
            let d = *p - c;
            ext = ext.max(d.dot(u).abs()).max(d.dot(r).abs() * h as f32 / w as f32);
        }
        (c, ext * 1.04)
    });
    let scale = h as f32 / (2.0 * half);
    let project = |p: Vec3| -> Vec3 {
        let d = p - centre;
        Vec3::new(w as f32 * 0.5 + d.dot(r) * scale, h as f32 * 0.5 - d.dot(u) * scale, d.dot(f))
    };
    let screen: Vec<Vec3> = scene.pos.iter().map(|&p| project(p)).collect();
    let mut depth = vec![f32::INFINITY; (w * h) as usize];
    let mut color = vec![[0.93_f32, 0.93, 0.95]; (w * h) as usize];
    let key =
        if view.raking { (-f * 0.35 + u * 0.9 - r * 0.3).normalize() } else { (-f + u * 0.7 - r * 0.45).normalize() };
    let fill = (-f - u * 0.2 + r * 0.6).normalize();
    // Smooth shading: area-weighted vertex normals (positions shared by seam copies are welded by
    // position so seams do not show).
    let mut weld: std::collections::HashMap<[u32; 3], Vec3> = std::collections::HashMap::new();
    let wkey = |p: Vec3| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
    let face_normal = |corners: &[u32]| {
        // Relative to the first corner: absolute positions lose the normal to f32 cancellation.
        let o = scene.pos[corners[0] as usize];
        let mut n = Vec3::ZERO;
        for i in 0..corners.len() {
            let (p, q) = (scene.pos[corners[i] as usize] - o, scene.pos[corners[(i + 1) % corners.len()] as usize] - o);
            n += p.cross(q);
        }
        n
    };
    for (corners, _) in &scene.faces {
        let n = face_normal(corners);
        for &v in corners {
            *weld.entry(wkey(scene.pos[v as usize])).or_insert(Vec3::ZERO) += n;
        }
    }
    let shade = |n: Vec3| 0.22 + 0.62 * n.dot(key).max(0.0) + 0.22 * n.dot(fill).max(0.0);
    for (corners, base) in &scene.faces {
        let nf = face_normal(corners).normalize_or_zero();
        let front = nf.dot(-f) >= 0.0;
        let vs: Vec<[f32; 3]> = corners
            .iter()
            .map(|&v| {
                let n = weld.get(&wkey(scene.pos[v as usize])).copied().unwrap_or(nf).normalize_or_zero();
                let n = if n.dot(nf) < 0.2 { nf } else { n };
                let k = shade(n);
                if front { base.map(|x| x * k) } else { [0.55 * k, 0.1, 0.55 * k] }
            })
            .collect();
        for i in 1..corners.len() - 1 {
            let tri = [screen[corners[0] as usize], screen[corners[i] as usize], screen[corners[i + 1] as usize]];
            raster(&tri, w, h, &mut depth, &mut color, [vs[0], vs[i], vs[i + 1]]);
        }
    }
    if view.wire {
        let eps = 2.0 / scale;
        for (corners, _) in &scene.faces {
            for i in 0..corners.len() {
                let (a, b) = (screen[corners[i] as usize], screen[corners[(i + 1) % corners.len()] as usize]);
                line(a, b, w, h, &depth, &mut color, eps);
            }
        }
    }
    let mut bytes = Vec::with_capacity((w * h * 3) as usize);
    for c in &color {
        for x in c {
            let s = if *x <= 0.003_130_8 { x * 12.92 } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
            bytes.push((s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
    }
    let image = Image::new(Format::Png, w, h, 3, ColorSpace::Srgb, Samples::U8(bytes)).unwrap();
    let png = humanart_io::png::encode(&image).unwrap();
    std::fs::write(preview_dir().join(format!("{}.png", view.name)), png).unwrap();
}

fn raster(t: &[Vec3; 3], w: u32, h: u32, depth: &mut [f32], color: &mut [[f32; 3]], c: [[f32; 3]; 3]) {
    let min_x = t.iter().map(|p| p.x).fold(f32::INFINITY, f32::min).floor().max(0.0) as i64;
    let max_x = t.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max).ceil().min(w as f32 - 1.0) as i64;
    let min_y = t.iter().map(|p| p.y).fold(f32::INFINITY, f32::min).floor().max(0.0) as i64;
    let max_y = t.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max).ceil().min(h as f32 - 1.0) as i64;
    let area = (t[1].x - t[0].x) * (t[2].y - t[0].y) - (t[2].x - t[0].x) * (t[1].y - t[0].y);
    if area.abs() < 1e-12 {
        return;
    }
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let e = |a: Vec3, b: Vec3| (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x);
            let (w0, w1, w2) = (e(t[1], t[2]) / area, e(t[2], t[0]) / area, e(t[0], t[1]) / area);
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }
            let z = w0 * t[0].z + w1 * t[1].z + w2 * t[2].z;
            let i = (y as u32 * w + x as u32) as usize;
            if z < depth[i] {
                depth[i] = z;
                color[i] = std::array::from_fn(|k| w0 * c[0][k] + w1 * c[1][k] + w2 * c[2][k]);
            }
        }
    }
}

fn line(a: Vec3, b: Vec3, w: u32, h: u32, depth: &[f32], color: &mut [[f32; 3]], eps: f32) {
    let steps = ((b.x - a.x).abs().max((b.y - a.y).abs()).ceil() as usize).max(1);
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let p = a + (b - a) * t;
        if p.x < 0.0 || p.y < 0.0 || p.x >= w as f32 || p.y >= h as f32 {
            continue;
        }
        let i = (p.y as u32 * w + p.x as u32) as usize;
        if p.z <= depth[i] + eps {
            color[i] = color[i].map(|x| x * 0.35);
        }
    }
}

/// A stable colour per id.
pub fn id_color(id: u32) -> [f32; 3] {
    let mut x = id.wrapping_mul(0x9E37_79B9) ^ 0x5bd1_e995;
    x ^= x >> 15;
    x = x.wrapping_mul(0x2c1b_3c6d);
    x ^= x >> 12;
    let c = |s: u32| 0.25 + 0.65 * ((x >> s) & 0xff) as f32 / 255.0;
    [c(0), c(8), c(16)]
}

/// The standard set of views of a full body plus face close-ups.
pub fn body_views(prefix: &str, head_centre: Vec3) -> Vec<View> {
    let leak = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
    let three_quarter = Vec3::new(-0.6, 0.0, -1.0);
    vec![
        View::new(leak(format!("{prefix}-front")), -Vec3::Z),
        View::new(leak(format!("{prefix}-side")), -Vec3::X),
        View::new(leak(format!("{prefix}-back")), Vec3::Z),
        View::new(leak(format!("{prefix}-three-quarter")), three_quarter),
        View::new(leak(format!("{prefix}-face")), -Vec3::Z).window(head_centre, 0.14).size(1000, 1000),
        View::new(leak(format!("{prefix}-face-wire")), -Vec3::Z).window(head_centre, 0.14).size(1000, 1000).wire(),
        View::new(leak(format!("{prefix}-face-side")), -Vec3::X).window(head_centre, 0.14).size(1000, 1000),
        View::new(leak(format!("{prefix}-face-34")), three_quarter).window(head_centre, 0.14).size(1000, 1000),
    ]
}

use super::cage::Cage;
use super::regions::REGIONS;

/// Checks that a cage is closed and consistently oriented: every directed edge has its reverse.
fn closed_and_oriented(c: &Cage) -> (usize, usize) {
    let mut dir: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
    for f in 0..c.face_count() {
        let cs = c.corners_of(f);
        for k in 0..cs.len() {
            *dir.entry((cs[k], cs[(k + 1) % cs.len()])).or_default() += 1;
        }
    }
    let open = dir.keys().filter(|&&(a, b)| !dir.contains_key(&(b, a))).count();
    let doubled = dir.values().filter(|&&n| n > 1).count();
    (open, doubled)
}

fn scene_of(c: &Cage, color: &dyn Fn(usize) -> [f32; 3]) -> Scene {
    Scene { pos: c.pos.clone(), faces: (0..c.face_count()).map(|f| (c.corners_of(f).to_vec(), color(f))).collect() }
}

#[test]
fn mod_01_body_cage_is_closed_and_oriented() {
    let a = super::anatomy::Anatomy::neutral();
    let body = super::body::build(&a).unwrap();
    let (open, doubled) = closed_and_oriented(&body.cage);
    assert_eq!((open, doubled), (0, 0), "cage has open or doubled directed edges");
}

#[test]
fn preview_body_surface() {
    let t0 = std::time::Instant::now();
    let s = super::body_surface().unwrap();
    let mesh = s.mesh();
    eprintln!("surface: {} vertices, {} faces, {:?}", mesh.pos.len(), mesh.face_count(), t0.elapsed());
    let (open, doubled) = closed_and_oriented(mesh);
    assert_eq!((open, doubled), (0, 0));
    let skin = [0.80, 0.66, 0.58];
    let plain = scene_of(mesh, &|_| skin);
    let head = s.anatomy.head_origin + Vec3::new(0.0, -0.01, 0.05);
    for v in body_views("body", head) {
        render(&plain, &v);
    }
    let regions = scene_of(mesh, &|f| id_color(u32::from(mesh.tags[f].region)));
    for v in body_views("regions", head).into_iter().take(5) {
        render(&regions, &v);
    }
    let a = &s.anatomy;
    let hand = a.hand_point([0.0, 0.10, 0.0]);
    // The hand alone (the thigh would hide it in the palm views).
    let near = |c: Vec3, r: f32| Scene {
        pos: plain.pos.clone(),
        faces: plain.faces.iter().filter(|(cs, _)| plain.pos[cs[0] as usize].distance(c) < r).cloned().collect(),
    };
    let only_hand = near(hand, 0.15);
    render(&only_hand, &View::new("hand-L-back", -a.hand_axes[1]).window(hand, 0.105).size(800, 800));
    render(&only_hand, &View::new("hand-L-palm", a.hand_axes[1]).window(hand, 0.105).size(800, 800));
    render(&only_hand, &View::new("hand-L-wire", -a.hand_axes[1]).window(hand, 0.105).size(800, 800).wire());
    render(&only_hand, &View::new("hand-L-side", a.hand_axes[2]).window(hand, 0.105).size(800, 800));
    let foot = Vec3::new(a.ankle.x, 0.05, a.ankle.z + 0.07);
    render(&plain, &View::new("foot-L-top", Vec3::new(0.0, -1.0, -0.6)).window(foot, 0.10).size(800, 800).wire());
    render(&plain, &View::new("foot-L-side", -Vec3::X).window(foot, 0.10).size(800, 800));
    // Round 2 (todo 1.3): the whole foot from both sides and below, nails and regions.
    let only_foot = near(Vec3::new(a.ankle.x, 0.04, a.ankle.z + 0.05), 0.17);
    let mid_foot = Vec3::new(a.ankle.x + 0.01, 0.05, a.ankle.z + 0.05);
    render(&only_foot, &View::new("foot-L-lateral", -Vec3::X).window(mid_foot, 0.085).size(900, 600));
    render(&only_foot, &View::new("foot-L-medial", Vec3::X).window(mid_foot, 0.085).size(900, 600));
    render(
        &only_foot,
        &View::new("foot-L-below", Vec3::new(0.0, 1.0, -0.25)).window(mid_foot, 0.16).size(600, 900).wire(),
    );
    render(
        &only_foot,
        &View::new("foot-L-front", Vec3::new(-0.3, -0.5, -1.0)).window(mid_foot + Vec3::Z * 0.08, 0.07).size(800, 800),
    );
    let region_scene = |c: Vec3, r: f32| Scene {
        pos: mesh.pos.clone(),
        faces: (0..mesh.face_count())
            .filter(|&f| mesh.pos[mesh.corners_of(f)[0] as usize].distance(c) < r)
            .map(|f| (mesh.corners_of(f).to_vec(), id_color(u32::from(mesh.tags[f].region))))
            .collect(),
    };
    let toes = Vec3::new(a.ankle.x + 0.02, 0.02, a.ankle.z + 0.17);
    render(
        &region_scene(Vec3::new(a.ankle.x, 0.04, a.ankle.z + 0.05), 0.17),
        &View::new("foot-L-regions", Vec3::new(0.0, -0.8, -0.6)).window(toes, 0.05).size(800, 800),
    );
    render(&only_foot, &View::new("toes-L", Vec3::new(0.0, -0.8, -0.6)).window(toes, 0.05).size(800, 800).wire());
    let fingertips = a.hand_point([0.0, 0.17, 0.0]);
    render(
        &region_scene(hand, 0.15),
        &View::new("hand-L-regions", a.hand_axes[1]).window(fingertips, 0.05).size(800, 800),
    );
    render(&only_hand, &View::new("fingers-L", a.hand_axes[1]).window(fingertips, 0.05).size(800, 800).raking());
    render(
        &only_hand,
        &View::new("fingers-L-34", (a.hand_axes[1] - a.hand_axes[2]).normalize())
            .window(fingertips, 0.05)
            .size(800, 800),
    );
    render(&only_hand, &View::new("palm-L", -a.hand_axes[1]).window(hand - a.hand_axes[0] * 0.03, 0.07).size(800, 800));
    let eye = a.eye_center + Vec3::new(0.0, 0.0, 0.01);
    render(&plain, &View::new("eye-L", -Vec3::Z).window(eye, 0.025).size(800, 800).wire());
    let corner = s.anatomy.head_origin + Vec3::new(-0.024, -0.048, 0.085);
    render(&plain, &View::new("mouth-corner-R", Vec3::new(0.5, 0.0, -1.0)).window(corner, 0.008).size(800, 800).wire());
    let mouth = s.anatomy.head_origin + Vec3::new(0.0, -0.048, 0.09);
    render(&plain, &View::new("mouth", -Vec3::Z).window(mouth, 0.03).size(900, 900).wire());
    render(&plain, &View::new("mouth-side", -Vec3::X).window(mouth, 0.03).size(900, 900).wire());
    render(
        &plain,
        &View::new("face-raking-34", Vec3::new(-0.75, -0.1, -0.65)).window(head, 0.13).size(900, 900).raking(),
    );
    render(&plain, &View::new("face-raking-side", -Vec3::X).window(head, 0.13).size(900, 900).raking());
    render(
        &plain,
        &View::new("face-wire-34", Vec3::new(-0.75, -0.1, -0.65))
            .window(head + Vec3::new(0.04, 0.0, 0.0), 0.06)
            .size(900, 900)
            .wire(),
    );
    let nose = s.anatomy.head_origin + Vec3::new(0.0, -0.018, 0.105);
    render(&plain, &View::new("nose-below", Vec3::new(0.0, 0.8, -0.6)).window(nose, 0.025).size(800, 800).wire());
    render(&plain, &View::new("nose-side", -Vec3::X).window(nose, 0.025).size(800, 800));
    let groin = Vec3::new(0.0, 0.84, 0.0);
    render(&plain, &View::new("groin-front", -Vec3::Z).window(groin, 0.12).size(900, 900).wire());
    render(&plain, &View::new("groin-below", Vec3::new(0.0, 1.0, -0.35)).window(groin, 0.12).size(900, 900).wire());
    render(&plain, &View::new("groin-side", -Vec3::X).window(groin, 0.14).size(900, 900));
    let pelvis = Vec3::new(0.0, 0.86, 0.0);
    for (name, dir) in [
        ("pelvis-front", -Vec3::Z),
        ("pelvis-side", -Vec3::X),
        ("pelvis-back", Vec3::Z),
        ("pelvis-34", Vec3::new(-0.6, 0.0, -1.0)),
        ("pelvis-back-34", Vec3::new(-0.6, 0.0, 1.0)),
    ] {
        render(&plain, &View::new(name, dir).window(pelvis, 0.22).size(900, 900).raking());
    }
    let chest = Vec3::new(0.0, 1.22, 0.05);
    let nipple = Vec3::new(0.095, 1.233, 0.1);
    render(&plain, &View::new("nipple-L", Vec3::new(-0.3, 0.0, -1.0)).window(nipple, 0.035).size(800, 800).wire());
    render(&plain, &View::new("nipple-L-side", -Vec3::X).window(nipple, 0.035).size(800, 800));
    render(
        &plain,
        &View::new("nipple-L-wide", Vec3::new(-0.3, 0.3, -1.0)).window(nipple, 0.07).size(800, 800).raking(),
    );
    render(&plain, &View::new("nipple-L-34", Vec3::new(-0.7, 0.0, -0.7)).window(nipple, 0.015).size(800, 800).wire());
    let navel = Vec3::new(0.0, 1.018, 0.1);
    render(&plain, &View::new("navel", -Vec3::Z).window(navel, 0.03).size(800, 800).wire());
    render(&plain, &View::new("navel-34", Vec3::new(-0.6, -0.3, -1.0)).window(navel, 0.03).size(800, 800));
    render(&plain, &View::new("torso-front-34", Vec3::new(-0.45, -0.1, -1.0)).window(chest, 0.28).size(900, 900));
    render(
        &plain,
        &View::new("torso-back-34", Vec3::new(-0.45, -0.1, 1.0)).window(chest - Vec3::Z * 0.1, 0.28).size(900, 900),
    );
    render(
        &plain,
        &View::new("relief-front", -Vec3::Z).window(Vec3::new(0.0, 1.12, 0.0), 0.33).size(900, 900).raking(),
    );
    render(&plain, &View::new("relief-back", Vec3::Z).window(Vec3::new(0.0, 1.12, 0.0), 0.33).size(900, 900).raking());
    render(
        &plain,
        &View::new("relief-legs-front", -Vec3::Z).window(Vec3::new(0.0, 0.35, 0.0), 0.35).size(900, 900).raking(),
    );
    render(
        &plain,
        &View::new("relief-legs-back", Vec3::Z).window(Vec3::new(0.0, 0.35, 0.0), 0.35).size(900, 900).raking(),
    );
    let cage = scene_of(&s.body.cage, &|f| id_color(u32::from(s.body.cage.tags[f].chart)));
    render(&cage, &View::new("cage-front", -Vec3::Z).wire());
    render(&cage, &View::new("cage-face", -Vec3::Z).window(head, 0.14).size(1000, 1000).wire());
    let _ = REGIONS.len();
}

#[test]
fn mod_01_parts_are_closed() {
    let s = super::body_surface().unwrap();
    let parts = super::parts::build(&s.anatomy, &s.body, &s.targets).unwrap();
    let mut scene = scene_of(s.mesh(), &|_| [0.80, 0.66, 0.58]);
    for p in &parts {
        let (open, doubled) = closed_and_oriented(&p.mesh);
        assert_eq!((open, doubled), (0, 0), "{}", p.name);
        eprintln!("{}: {} v {} f", p.name, p.mesh.pos.len(), p.mesh.face_count());
        let base = scene.pos.len() as u32;
        scene.pos.extend_from_slice(&p.mesh.pos);
        let color = match p.region {
            super::regions::R::EyeballL | super::regions::R::EyeballR => [0.92, 0.92, 0.9],
            super::regions::R::TeethUpper | super::regions::R::TeethLower => [0.95, 0.93, 0.85],
            super::regions::R::GumsUpper | super::regions::R::GumsLower | super::regions::R::Tongue => {
                [0.75, 0.35, 0.38]
            }
            _ => [0.3, 0.5, 0.9],
        };
        for f in 0..p.mesh.face_count() {
            scene.faces.push((p.mesh.corners_of(f).iter().map(|&v| v + base).collect(), color));
        }
    }
    let head = s.anatomy.head_origin + Vec3::new(0.0, -0.01, 0.05);
    render(&scene, &View::new("parts-face", -Vec3::Z).window(head, 0.14).size(1000, 1000));
    render(&scene, &View::new("parts-face-34", Vec3::new(-0.6, 0.0, -1.0)).window(head, 0.14).size(1000, 1000));
    // Mouth interior alone: parts only, seen from the front and below.
    let only = Scene { pos: scene.pos.clone(), faces: scene.faces[s.mesh().face_count()..].to_vec() };
    let mouth = s.anatomy.head_origin + Vec3::new(0.0, -0.05, 0.07);
    render(&only, &View::new("parts-mouth", Vec3::new(0.0, 0.5, -1.0)).window(mouth, 0.05).size(800, 800));
    let eye = s.anatomy.eye_center + Vec3::new(0.0, 0.0, 0.01);
    render(&scene, &View::new("parts-eye-L", -Vec3::Z).window(eye, 0.025).size(800, 800));
    // Eyes as a viewer sees them: skin and eyeballs only (the tear line and occlusion are shading
    // helpers), front and three-quarter.
    let mut bare = scene_of(s.mesh(), &|_| [0.80, 0.66, 0.58]);
    for p in parts.iter().filter(|p| p.name.starts_with("eyeball")) {
        let base = bare.pos.len() as u32;
        bare.pos.extend_from_slice(&p.mesh.pos);
        let e = super::anatomy::sided(s.anatomy.eye_center, if p.name.ends_with('L') { 1.0 } else { -1.0 });
        for f in 0..p.mesh.face_count() {
            let d = (p.mesh.centroid(f) - e).normalize_or_zero();
            let color = if d.z > 0.975 {
                [0.05, 0.05, 0.05]
            } else if d.z > 0.86 {
                [0.35, 0.25, 0.15]
            } else {
                [0.92, 0.9, 0.86]
            };
            bare.faces.push((p.mesh.corners_of(f).iter().map(|&v| v + base).collect(), color));
        }
    }
    render(
        &bare,
        &View::new("eyes-front", -Vec3::Z)
            .window(s.anatomy.head_origin + Vec3::new(0.0, 0.008, 0.08), 0.04)
            .size(1200, 600),
    );
    render(&bare, &View::new("eye-L-34", Vec3::new(-0.6, 0.0, -1.0)).window(eye, 0.025).size(800, 800));
    render(
        &bare,
        &View::new("eye-R-34", Vec3::new(-0.6, 0.0, -1.0))
            .window(Vec3::new(-eye.x, eye.y, eye.z), 0.025)
            .size(800, 800),
    );
    render(
        &bare,
        &View::new("face-34-eyes", Vec3::new(-0.6, 0.0, -1.0))
            .window(s.anatomy.head_origin + Vec3::new(0.0, -0.01, 0.05), 0.14)
            .size(1000, 1000),
    );
}

fn uv_preview(mesh: &crate::mesh::Mesh, tile: u16, name: &'static str, regions: &[u32]) {
    let pos: Vec<Vec3> = mesh.uvs().iter().map(|uv| Vec3::new(uv.x, 1.0 - uv.y, 0.0)).collect();
    let faces = mesh
        .faces()
        .enumerate()
        .filter(|(f, _)| mesh.face_tiles()[*f] == tile)
        .map(|(_, c)| (c.to_vec(), id_color(regions[c[0] as usize])))
        .collect();
    let scene = Scene { pos, faces };
    render(&scene, &View::new(name, -Vec3::Z).window(Vec3::new(0.5, 0.5, 0.0), 0.5).size(1024, 1024));
}

#[test]
fn mod_04_uv_layout_and_regions_validate() {
    use crate::template::{Template, TemplateInfo, TemplateParts, UvTile};
    let t0 = std::time::Instant::now();
    let s = super::body_surface().unwrap();
    let parts = super::parts::build(&s.anatomy, &s.body, &s.targets).unwrap();
    let pieces: Vec<&super::cage::Cage> = std::iter::once(s.mesh()).chain(parts.iter().map(|p| &p.mesh)).collect();
    let layout = super::uv::layout(&pieces).unwrap();
    let t_uv = t0.elapsed();
    let asm = super::assemble::assemble(&s, &parts, &layout).unwrap();
    let tiles = super::regions::TILES
        .iter()
        .map(|&(_, name, resolution, target)| UvTile { name: name.into(), resolution, target_texels_per_mm: target })
        .collect();
    let template = Template::new(TemplateParts {
        info: TemplateInfo { name: "uv-test".into(), version: 1, ..TemplateInfo::default() },
        mesh: asm.mesh.clone(),
        tiles,
        regions: asm.regions.clone(),
        landmarks: Vec::new(),
        skeleton: crate::skeleton::Skeleton {
            joints: Vec::new(),
            regressor: crate::skeleton::JointRegressor::empty(0),
        },
        skin: crate::skin::SkinWeights::default(),
        deltas: crate::delta::DeltaBases::empty(),
        expressions: crate::expression::ExpressionSet::default(),
        lods: Vec::new(),
    })
    .unwrap();
    let report = crate::validate::validate(&template);
    eprintln!(
        "uv {:?} total {:?}
{report}",
        t_uv,
        t0.elapsed()
    );
    for (i, &(_, name, _, _)) in super::regions::TILES.iter().enumerate() {
        let leaked: &'static str = Box::leak(format!("uv-{name}").into_boxed_str());
        let islands: Vec<u32> = layout.copies.iter().map(|c| c.island).collect();
        uv_preview(&asm.mesh, i as u16, leaked, &islands);
    }
    let regions = Scene {
        pos: asm.mesh.positions().to_vec(),
        faces: asm
            .mesh
            .faces()
            .map(|c| (c.to_vec(), id_color(u32::from(asm.regions.vertex_regions[c[0] as usize]))))
            .collect(),
    };
    let head = s.anatomy.head_origin + Vec3::new(0.0, -0.01, 0.05);
    render(&regions, &View::new("final-regions-face", -Vec3::Z).window(head, 0.14).size(1000, 1000));
    render(&regions, &View::new("final-regions-front", -Vec3::Z));
    for issue in report.issues.iter().filter(|i| i.severity == crate::validate::Severity::Error).take(12) {
        if let Some(f) = issue.index.and_then(|f| usize::try_from(f).ok()).filter(|&f| f < asm.mesh.face_count()) {
            let c = asm.mesh.face(f);
            let p = c.iter().fold(Vec3::ZERO, |s, &v| s + asm.mesh.positions()[v as usize]) / c.len() as f32;
            let r = asm.regions.vertex_regions[c[0] as usize];
            eprintln!(
                "issue face {f} region {} at {:?} uv area {} area mm2 {} uv {:?}",
                super::regions::REGIONS[r as usize].1,
                (p - s.anatomy.head_origin) * 1000.0,
                asm.mesh.face_uv_area(f),
                asm.mesh.face_vector_area(f).length() * 1e6,
                c.iter().map(|&v| asm.mesh.uvs()[v as usize] * 8192.0).collect::<Vec<_>>()
            );
        }
    }
    assert!(report.passed(), "{report}");
}

/// Adds a small octahedron at `p` to the scene.
fn marker(scene: &mut Scene, p: Vec3, r: f32, color: [f32; 3]) {
    let base = scene.pos.len() as u32;
    for d in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z] {
        scene.pos.push(p + d * r);
    }
    // Faces of the octahedron, counter-clockwise from outside.
    for (a, b, c) in [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)] {
        scene.faces.push((vec![base + a, base + b, base + c], color));
    }
}

#[test]
fn mod_06_landmarks_are_valid_symmetric_and_documented() {
    use crate::landmark::LandmarkGroup;
    let s = super::body_surface().unwrap();
    let parts = super::parts::build(&s.anatomy, &s.body, &s.targets).unwrap();
    let pieces: Vec<&super::cage::Cage> = std::iter::once(s.mesh()).chain(parts.iter().map(|p| &p.mesh)).collect();
    let layout = super::uv::layout(&pieces).unwrap();
    let asm = super::assemble::assemble(&s, &parts, &layout).unwrap();
    let t0 = std::time::Instant::now();
    let described = super::landmarks::build(&s, &asm).unwrap();
    eprintln!("landmarks in {:?}", t0.elapsed());
    let face = described.iter().filter(|d| d.landmark.group == LandmarkGroup::Face).count();
    let body = described.len() - face;
    eprintln!("{face} face, {body} body landmarks");
    assert!(face >= 98 && body >= 40, "[MOD-06] counts: {face} face, {body} body");
    let pos = asm.mesh.positions();
    let at = |name: &str| described.iter().find(|d| d.landmark.name == name).and_then(|d| d.landmark.position(pos));
    // Names the parameter registry measures with (data/template/params).
    for name in [
        "head.vertex",
        "L.heel",
        "L.zygion",
        "R.zygion",
        "L.iris.center",
        "R.iris.center",
        "L.lip.corner",
        "R.lip.corner",
        "nose.root",
        "nose.tip",
        "L.shoulder.acromion",
        "R.shoulder.acromion",
    ] {
        assert!(at(name).is_some(), "missing landmark {name}");
    }
    // Left and right mirror each other; midline points lie on the midline.
    let mut worst = (0.0_f32, String::new());
    for d in &described {
        let n = &d.landmark.name;
        let p = d.landmark.position(pos).unwrap();
        let mirrored = if let Some(rest) = n.strip_prefix("L.") {
            Some(format!("R.{rest}"))
        } else if n.contains(".L.") {
            Some(n.replace(".L.", ".R."))
        } else {
            None
        };
        if let Some(m) = mirrored {
            let q = at(&m).unwrap_or_else(|| panic!("{n} has no mirror {m}"));
            let e = (p - Vec3::new(-q.x, q.y, q.z)).length();
            if e > 0.001 {
                eprintln!("mirror {n}: {:.2} mm, L {:?} R {:?}", e * 1000.0, p * 1000.0, q * 1000.0);
            }
            if e > worst.0 {
                worst = (e, n.clone());
            }
        } else if !n.starts_with("R.") && !n.contains(".R.") && !n.starts_with("lip.") && !n.starts_with("hairline.") {
            assert!(p.x.abs() < 0.002, "midline landmark {n} at x = {}", p.x);
        }
    }
    eprintln!("worst mirror error {:.3} mm at {}", worst.0 * 1000.0, worst.1);
    // The validator accepts them.
    let mut t = crate::template::Template::new(crate::template::TemplateParts {
        info: crate::template::TemplateInfo { name: "landmark-test".into(), version: 1, ..Default::default() },
        mesh: asm.mesh.clone(),
        tiles: super::regions::TILES
            .iter()
            .map(|&(_, name, resolution, target)| crate::template::UvTile {
                name: name.into(),
                resolution,
                target_texels_per_mm: target,
            })
            .collect(),
        regions: asm.regions.clone(),
        landmarks: described.iter().map(|d| d.landmark.clone()).collect(),
        skeleton: crate::skeleton::Skeleton {
            joints: Vec::new(),
            regressor: crate::skeleton::JointRegressor::empty(0),
        },
        skin: crate::skin::SkinWeights::default(),
        deltas: crate::delta::DeltaBases::empty(),
        expressions: crate::expression::ExpressionSet::default(),
        lods: Vec::new(),
    })
    .unwrap();
    let report = crate::validate::validate(&t);
    assert_eq!(report.landmarks.invalid, 0, "{report}");
    assert!(report.landmarks.meets_mod06());
    let _ = &mut t;
    // Preview: face landmarks red, body landmarks blue.
    let mut scene = scene_of_mesh(&asm.mesh, 0);
    for d in &described {
        let p = d.landmark.position(pos).unwrap();
        let (r, c) =
            if d.landmark.group == LandmarkGroup::Face { (0.0012, [0.9, 0.1, 0.1]) } else { (0.006, [0.1, 0.2, 0.9]) };
        marker(&mut scene, p, r, c);
    }
    let head = s.anatomy.head_origin + Vec3::new(0.0, -0.01, 0.05);
    render(&scene, &View::new("landmarks-face", -Vec3::Z).window(head, 0.14).size(1000, 1000));
    render(&scene, &View::new("landmarks-face-side", -Vec3::X).window(head, 0.14).size(1000, 1000));
    render(&scene, &View::new("landmarks-face-34", Vec3::new(-0.6, 0.0, -1.0)).window(head, 0.14).size(1000, 1000));
    render(&scene, &View::new("landmarks-front", -Vec3::Z));
    render(&scene, &View::new("landmarks-back", Vec3::Z));
    render(&scene, &View::new("landmarks-side", -Vec3::X));
    let hand = s.anatomy.hand_point([0.0, 0.07, 0.0]);
    let mut feet =
        View::new("landmarks-feet-top", -Vec3::Y).window(Vec3::new(0.0, 0.03, 0.06), 0.16).size(1000, 1000).wire();
    feet.up = Vec3::Z;
    render(&scene, &feet);
    render(&scene, &View::new("landmarks-hands", -Vec3::Z).window(Vec3::new(0.0, 0.86, 0.0), 0.12).size(2400, 400));
    render(&scene, &View::new("landmarks-hand-L", -s.anatomy.hand_axes[1]).window(hand, 0.08).size(800, 800));
    assert!(worst.0 < 0.002, "mirror error {} m at {}", worst.0, worst.1);
    // The documented list equals the generator's output (HUMANART_BLESS=1 rewrites it).
    let text = super::landmarks::json(&described, pos);
    humanart_io::json::parse_str(&text).unwrap();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/template/landmarks.json");
    if std::env::var_os("HUMANART_BLESS").is_some() {
        std::fs::write(&path, &text).unwrap();
    }
    let on_disk = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(on_disk == text, "data/template/landmarks.json is out of date; rerun with HUMANART_BLESS=1");
}

/// The body part of a template mesh as a scene (skin colour).
fn scene_of_mesh(mesh: &crate::mesh::Mesh, part: usize) -> Scene {
    let range = mesh.parts()[part].faces();
    Scene {
        pos: mesh.positions().to_vec(),
        faces: range.map(|f| (mesh.face(f).to_vec(), [0.80, 0.66, 0.58])).collect(),
    }
}

/// Per vertex of `pos`, the distance to the nearest mirrored vertex (capped at `cap`).
fn mirror_distances(pos: &[Vec3], cap: f32) -> Vec<f32> {
    let key = |p: Vec3| ((p.x / cap).floor() as i64, (p.y / cap).floor() as i64, (p.z / cap).floor() as i64);
    let mut grid: std::collections::HashMap<(i64, i64, i64), Vec<u32>> = std::collections::HashMap::new();
    for (i, &p) in pos.iter().enumerate() {
        grid.entry(key(p)).or_default().push(i as u32);
    }
    pos.iter()
        .map(|&p| {
            let m = Vec3::new(-p.x, p.y, p.z);
            let k = key(m);
            let mut best = cap;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        for &j in grid.get(&(k.0 + dx, k.1 + dy, k.2 + dz)).map_or(&[][..], |v| &v[..]) {
                            best = best.min(pos[j as usize].distance(m));
                        }
                    }
                }
            }
            best
        })
        .collect()
}

#[test]
fn mod_01_surface_and_parts_are_mirror_symmetric() {
    // [AC-WP20-3] needs mirror editing: every vertex of the surface (and of the parts taken
    // together) has a mirror image among the vertices.
    let s = super::body_surface().unwrap();
    let d = mirror_distances(&s.mesh().pos, 0.004);
    let max = d.iter().copied().fold(0.0_f32, f32::max);
    eprintln!("surface: max mirror distance {:.4} mm over {} vertices", max * 1000.0, d.len());
    assert!(max < 5e-5, "surface is not mirror-symmetric: {max} m");
    let parts = super::parts::build(&s.anatomy, &s.body, &s.targets).unwrap();
    let pos: Vec<Vec3> = parts.iter().flat_map(|p| p.mesh.pos.iter().copied()).collect();
    let d = mirror_distances(&pos, 0.004);
    let max = d.iter().copied().fold(0.0_f32, f32::max);
    eprintln!("parts: max mirror distance {:.4} mm over {} vertices", max * 1000.0, d.len());
    let mut k = 0;
    for p in &parts {
        let m = d[k..k + p.mesh.pos.len()].iter().copied().fold(0.0_f32, f32::max);
        if m > 5e-5 {
            eprintln!("  part {}: {:.3} mm", p.name, m * 1000.0);
        }
        k += p.mesh.pos.len();
    }
    assert!(max < 5e-5, "parts are not mirror-symmetric: {max} m");
}

/// Linear-blend skinning of `pos` with per-joint local rotations `pose` (test-only reference).
fn pose_lbs(
    skeleton: &crate::skeleton::Skeleton,
    skin: &crate::skin::SkinWeights,
    pos: &[Vec3],
    pose: &[(&str, humanart_math::Quat)],
) -> Vec<Vec3> {
    use humanart_math::{Quat, Transform};
    let rest = skeleton.rest_world();
    let mut posed: Vec<Transform> = Vec::with_capacity(rest.len());
    for j in &skeleton.joints {
        let r = pose.iter().find(|p| p.0 == j.name).map_or(Quat::IDENTITY, |p| p.1);
        let local = Transform::from_rotation_translation((j.rest.rotation * r).normalize(), j.rest.translation);
        let world = j.parent.and_then(|p| posed.get(p as usize).copied()).map_or(local, |p| p * local);
        posed.push(world);
    }
    pos.iter()
        .enumerate()
        .map(|(v, &p)| {
            let (js, ws) = skin.vertex(v);
            js.iter()
                .zip(ws)
                .map(|(&j, &w)| posed[j as usize].transform_point3(rest[j as usize].inverse_transform_point3(p)) * w)
                .fold(Vec3::ZERO, |s, q| s + q)
        })
        .collect()
}

#[test]
fn mod_07_skeleton_and_skin_validate_and_pose() {
    use humanart_math::Quat;
    let s = super::body_surface().unwrap();
    let parts = super::parts::build(&s.anatomy, &s.body, &s.targets).unwrap();
    let pieces: Vec<&super::cage::Cage> = std::iter::once(s.mesh()).chain(parts.iter().map(|p| &p.mesh)).collect();
    let layout = super::uv::layout(&pieces).unwrap();
    let asm = super::assemble::assemble(&s, &parts, &layout).unwrap();
    let t0 = std::time::Instant::now();
    let rig = super::rig::skeleton(&s.anatomy, 0).unwrap();
    let skin = super::rig::skin(&rig, &s.anatomy, &asm.mesh, &asm.regions).unwrap();
    eprintln!("{} joints, skin in {:?}", rig.skeleton.joints.len(), t0.elapsed());
    // Joint rest positions are where the anatomy puts them.
    let world = rig.skeleton.rest_world();
    let at = |n: &str| world[rig.skeleton.find(n).unwrap() as usize].translation;
    assert!(at("L.forearm").distance(s.anatomy.elbow) < 1e-5);
    assert!(at("R.thigh").distance(Vec3::new(-s.anatomy.hip.x, s.anatomy.hip.y, s.anatomy.hip.z)) < 1e-5);
    // Mirror symmetry of the skeleton.
    for j in &rig.skeleton.joints {
        if let Some(rest) = j.name.strip_prefix("L.") {
            let (p, q) = (at(&j.name), at(&format!("R.{rest}")));
            assert!((p - Vec3::new(-q.x, q.y, q.z)).length() < 1e-5, "{} is not mirrored", j.name);
        }
    }
    let t = crate::template::Template::new(crate::template::TemplateParts {
        info: crate::template::TemplateInfo { name: "rig-test".into(), version: 1, ..Default::default() },
        mesh: asm.mesh.clone(),
        tiles: super::regions::TILES
            .iter()
            .map(|&(_, name, resolution, target)| crate::template::UvTile {
                name: name.into(),
                resolution,
                target_texels_per_mm: target,
            })
            .collect(),
        regions: asm.regions.clone(),
        landmarks: Vec::new(),
        skeleton: rig.skeleton.clone(),
        skin: skin.clone(),
        deltas: crate::delta::DeltaBases::empty(),
        expressions: crate::expression::ExpressionSet::default(),
        lods: Vec::new(),
    })
    .unwrap();
    let report = crate::validate::validate(&t);
    eprintln!("{}", report.to_string().lines().filter(|l| l.contains("skin")).collect::<Vec<_>>().join("\n"));
    assert!(report.passed(), "{report}");
    // The rest pose reproduces the rest shape.
    let pos = asm.mesh.positions();
    let same = pose_lbs(&rig.skeleton, &skin, pos, &[]);
    let max = pos.iter().zip(&same).map(|(a, b)| a.distance(*b)).fold(0.0_f32, f32::max);
    assert!(max < 1e-5, "rest pose moves vertices by {max} m");
    // A test pose: bent elbows, knee and hip, curled fingers, open jaw, turned head, half-closed lid.
    let x = |a: f32| Quat::from_rotation_x(a);
    let mut pose: Vec<(&str, Quat)> = vec![
        ("L.forearm", x(1.3)),
        ("R.forearm", x(0.6)),
        ("L.forearm.twist", Quat::from_rotation_y(0.8)),
        ("L.thigh", x(0.7)),
        ("L.shin", x(1.2)),
        ("R.foot", x(0.4)),
        ("jaw", x(0.3)),
        ("head", Quat::from_rotation_y(0.35)),
        ("spine.03", x(0.15)),
        ("L.eyelid.upper", x(0.45)),
        ("L.upper-arm", Quat::from_rotation_z(-0.5)),
    ];
    let curl: Vec<String> = ["index", "middle", "ring", "little"]
        .iter()
        .flat_map(|f| (1..=3).map(move |k| format!("L.{f}.{k:02}")))
        .collect();
    for n in &curl {
        pose.push((n.as_str(), x(0.9)));
    }
    let posed = pose_lbs(&rig.skeleton, &skin, pos, &pose);
    let mut scene = Scene { pos: posed, faces: Vec::new() };
    for part in asm.mesh.parts() {
        let color = if part.name == "body" { [0.80, 0.66, 0.58] } else { [0.9, 0.9, 0.88] };
        scene.faces.extend(part.faces().map(|f| (asm.mesh.face(f).to_vec(), color)));
    }
    render(&scene, &View::new("posed-front", -Vec3::Z));
    render(&scene, &View::new("posed-side", -Vec3::X));
    let head = s.anatomy.head_origin + Vec3::new(0.0, -0.01, 0.05);
    render(&scene, &View::new("posed-face", -Vec3::Z).window(head, 0.14).size(1000, 1000));
    let elbow = s.anatomy.elbow;
    render(&scene, &View::new("posed-elbow-L", -Vec3::Z).window(elbow, 0.15).size(900, 900).wire());
    let knee = s.anatomy.knee;
    render(&scene, &View::new("posed-knee-L", -Vec3::X).window(knee, 0.2).size(900, 900).wire());
    // Weights preview: colour by the strongest joint.
    let regions = Scene {
        pos: pos.to_vec(),
        faces: asm
            .mesh
            .faces()
            .map(|c| (c.to_vec(), id_color(u32::from(skin.vertex(c[0] as usize).0[0]) * 7 + 3)))
            .collect(),
    };
    render(&regions, &View::new("skin-front", -Vec3::Z));
    render(&regions, &View::new("skin-face", -Vec3::Z).window(head, 0.14).size(1000, 1000));
}

/// Newell normal of a polygon (not normalised).
fn newell(pos: &[Vec3], c: &[u32]) -> Vec3 {
    let mut n = Vec3::ZERO;
    for k in 0..c.len() {
        let (a, b) = (pos[c[k] as usize], pos[c[(k + 1) % c.len()] as usize]);
        n += Vec3::new((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y));
    }
    n
}

#[test]
fn skin_02_arms_come_down_to_the_sides_without_folding() {
    use crate::eval::{EvalInput, Evaluator, SkinningMode};
    use humanart_math::Quat;
    let s = super::body_surface().unwrap();
    let parts = super::parts::build(&s.anatomy, &s.body, &s.targets).unwrap();
    let pieces: Vec<&super::cage::Cage> = std::iter::once(s.mesh()).chain(parts.iter().map(|p| &p.mesh)).collect();
    let layout = super::uv::layout(&pieces).unwrap();
    let asm = super::assemble::assemble(&s, &parts, &layout).unwrap();
    let rig = super::rig::skeleton(&s.anatomy, 0).unwrap();
    let skin = super::rig::skin(&rig, &s.anatomy, &asm.mesh, &asm.regions).unwrap();
    let t = std::sync::Arc::new(
        crate::template::Template::new(crate::template::TemplateParts {
            info: crate::template::TemplateInfo { name: "rig-test".into(), version: 1, ..Default::default() },
            mesh: asm.mesh.clone(),
            tiles: super::regions::TILES
                .iter()
                .map(|&(_, name, resolution, target)| crate::template::UvTile {
                    name: name.into(),
                    resolution,
                    target_texels_per_mm: target,
                })
                .collect(),
            regions: asm.regions.clone(),
            landmarks: Vec::new(),
            skeleton: rig.skeleton.clone(),
            skin,
            deltas: crate::delta::DeltaBases::empty(),
            expressions: crate::expression::ExpressionSet::default(),
            lods: Vec::new(),
        })
        .unwrap(),
    );
    let rest = asm.mesh.positions().to_vec();
    let body = asm.mesh.parts().iter().position(|p| p.name == "body").unwrap();
    let mut eval = Evaluator::new(std::sync::Arc::clone(&t));
    let rest_world = t.skeleton().rest_world();
    // The arms come down about the world forward axis (like the identity comparison sheets); 0.5 rad
    // leaves them ~16 degrees from the vertical, 0.65 rad ~8 degrees (hands at the thighs).
    for (angle, name, check) in [(0.5_f32, "posed-arms-down", true), (0.65, "posed-arms-side", true)] {
        let mut pose = vec![Quat::IDENTITY; t.skeleton().joints.len()];
        for (joint, a) in [("L.upper-arm", -angle), ("R.upper-arm", angle)] {
            let j = t.skeleton().find(joint).unwrap() as usize;
            let rb = rest_world[j].rotation;
            pose[j] = (rb.inverse() * Quat::from_rotation_z(a) * rb).normalize();
        }
        eval.evaluate(&EvalInput {
            shape: &[],
            expressions: &[],
            layers: &[],
            pose: &pose,
            skinning: SkinningMode::ByRegion,
        })
        .unwrap();
        let posed = eval.positions().to_vec();
        // A face whose normal turns by more than 90 degrees has turned over (a rigid arm turns its
        // faces by `angle` at most).
        let mut over = Vec::new();
        for f in asm.mesh.parts()[body].faces() {
            let c = asm.mesh.face(f);
            let (n0, n1) = (newell(&rest, c), newell(&posed, c));
            if n0.normalize_or_zero().dot(n1.normalize_or_zero()) < 0.0 {
                over.push(f);
            }
        }
        eprintln!("{name}: {} body faces turned over", over.len());
        let mut scene = Scene { pos: posed.clone(), faces: Vec::new() };
        for part in asm.mesh.parts() {
            let color = if part.name == "body" { [0.80, 0.66, 0.58] } else { [0.9, 0.9, 0.88] };
            scene.faces.extend(part.faces().map(|f| (asm.mesh.face(f).to_vec(), color)));
        }
        for &f in &over {
            scene.faces[f].1 = [0.9, 0.1, 0.1];
        }
        let names: [&'static str; 4] = match name {
            "posed-arms-down" => [
                "posed-arms-down-front",
                "posed-arms-down-back",
                "posed-arms-down-axilla-L",
                "posed-arms-down-axilla-L-34",
            ],
            _ => [
                "posed-arms-side-front",
                "posed-arms-side-back",
                "posed-arms-side-axilla-L",
                "posed-arms-side-axilla-L-34",
            ],
        };
        render(&scene, &View::new(names[0], -Vec3::Z));
        render(&scene, &View::new(names[1], Vec3::Z));
        let axilla = s.anatomy.shoulder + Vec3::new(-0.03, -0.09, 0.0);
        render(&scene, &View::new(names[2], -Vec3::Z).window(axilla, 0.13).size(900, 900).wire());
        let v34 = Vec3::new(-0.6, 0.0, -0.8);
        render(&scene, &View::new(names[3], v34).window(axilla, 0.13).size(900, 900).wire());
        if check {
            assert!(over.is_empty(), "{name}: {} body faces turn over", over.len());
        }
    }
}

#[test]
fn ac_wp06_1_generated_template_validates_and_round_trips() {
    // [AC-WP06-1] [MOD-01]..[MOD-07] [IO-21]: the whole generated template, LODs included, passes
    // validation and survives a package round trip unchanged.
    let t0 = std::time::Instant::now();
    let g = super::build::template(&super::build::Options::default()).unwrap();
    let built = t0.elapsed();
    let t = &g.template;
    let report = crate::validate::validate(t);
    eprintln!("built in {built:?}\n{report}");
    for (i, r) in g.lod_reports.iter().enumerate() {
        eprintln!(
            "LOD{}: {} vertices, {} faces\n{r}",
            i + 1,
            t.lods()[i].mesh.vertex_count(),
            t.lods()[i].mesh.face_count()
        );
    }
    assert!(report.passed(), "{report}");
    assert_eq!(t.lods().len(), 4);
    assert_eq!(g.lod_landmarks.len(), 4);
    assert!(g.lod_landmarks.iter().all(|l| l.len() == t.landmarks().len()));
    let t1 = std::time::Instant::now();
    let bytes = crate::package::write_package(t).unwrap();
    let back = crate::package::read_package(&bytes, &crate::package::PackageLimits::default()).unwrap();
    eprintln!("package: {} bytes, write+read {:?}", bytes.len(), t1.elapsed());
    assert!(back == *t, "package round trip changed the template");
    assert_eq!(crate::package::write_package(&back).unwrap(), bytes, "re-encoding is not byte-identical");
    std::fs::write(preview_dir().join("../template.antmpl"), &bytes).unwrap();
    // Previews of the LOD levels.
    for (i, lod) in t.lods().iter().enumerate() {
        let m = &lod.mesh;
        let scene =
            Scene { pos: m.positions().to_vec(), faces: m.faces().map(|c| (c.to_vec(), [0.80, 0.66, 0.58])).collect() };
        let name: &'static str = Box::leak(format!("lod{}-face", i + 1).into_boxed_str());
        let head = t.mesh().positions().iter().fold(Vec3::splat(f32::NEG_INFINITY), |a, &p| a.max(p)).y;
        render(
            &scene,
            &View::new(name, -Vec3::Z).window(Vec3::new(0.0, head - 0.14, 0.05), 0.14).size(800, 800).wire(),
        );
    }
}

#[test]
fn preview_crease_map() {
    // Signed curvature across every edge (turn angle over the distance of the face centres): valleys
    // tighter than a 12 mm radius are blue, ridges tighter than 12 mm red; the lighter tones mark
    // 25 mm. Creases the subdivision surface should not have show up as thin blue lines.
    let s = super::body_surface().unwrap();
    let m = s.mesh();
    let (_, _, edge_faces) = m.edges();
    let normal = |f: usize| {
        let c = m.corners_of(f);
        let o = m.pos[c[0] as usize];
        let mut n = Vec3::ZERO;
        for k in 0..c.len() {
            n += (m.pos[c[k] as usize] - o).cross(m.pos[c[(k + 1) % c.len()] as usize] - o);
        }
        n.normalize_or_zero()
    };
    let centre = |f: usize| {
        let c = m.corners_of(f);
        c.iter().fold(Vec3::ZERO, |acc, &v| acc + m.pos[v as usize]) / c.len() as f32
    };
    let normals: Vec<Vec3> = (0..m.face_count()).map(normal).collect();
    let centres: Vec<Vec3> = (0..m.face_count()).map(centre).collect();
    let mut valley = vec![0.0_f32; m.face_count()];
    let mut ridge = vec![0.0_f32; m.face_count()];
    for ef in &edge_faces {
        if ef[1] != u32::MAX {
            let (a, b) = (ef[0] as usize, ef[1] as usize);
            let angle = normals[a].dot(normals[b]).clamp(-1.0, 1.0).acos();
            let d = centres[b] - centres[a];
            let k = angle / d.length().max(1e-6);
            let target = if (normals[b] - normals[a]).dot(d) < 0.0 { &mut valley } else { &mut ridge };
            target[a] = target[a].max(k);
            target[b] = target[b].max(k);
        }
    }
    let scene = scene_of(m, &|f| {
        if valley[f] > 1.0 / 0.012 {
            [0.1, 0.25, 0.95]
        } else if valley[f] > 1.0 / 0.025 {
            [0.5, 0.65, 0.95]
        } else if ridge[f] > 1.0 / 0.012 {
            [0.9, 0.1, 0.1]
        } else if ridge[f] > 1.0 / 0.025 {
            [0.95, 0.65, 0.5]
        } else {
            [0.8, 0.8, 0.8]
        }
    });
    let head = s.anatomy.head_origin + Vec3::new(0.0, -0.01, 0.05);
    render(&scene, &View::new("crease-face-34", Vec3::new(-0.6, 0.0, -1.0)).window(head, 0.14).size(1000, 1000));
    render(&scene, &View::new("crease-face", -Vec3::Z).window(head, 0.14).size(1000, 1000));
    render(&scene, &View::new("crease-face-side", -Vec3::X).window(head, 0.14).size(1000, 1000));
    render(&scene, &View::new("crease-front", -Vec3::Z));
    render(&scene, &View::new("crease-back", Vec3::Z));
    render(&scene, &View::new("crease-legs-side", -Vec3::X).window(Vec3::new(0.1, 0.3, 0.0), 0.3).size(900, 900));
}

#[test]
fn zz_probe_cheek_profile() {
    let s = super::body_surface().unwrap();
    let a = &s.anatomy;
    for xs in [40.0_f32, 55.0, 65.0] {
        let mut pts: Vec<Vec3> = s
            .mesh()
            .pos
            .iter()
            .map(|&p| (p - a.head_origin) * 1000.0)
            .filter(|h| (h.x - xs).abs() < 1.2 && h.z > 0.0 && (-40.0..30.0).contains(&h.y))
            .collect();
        pts.sort_by(|p, q| p.y.total_cmp(&q.y));
        let line: Vec<String> = pts.iter().map(|h| format!("{:.0}:{:.1}", h.y, h.z)).collect();
        eprintln!("x={xs}: {}", line.join(" "));
    }
}

/// Probe: depth profiles along rays (head frame, mm). `from_side`: rays along -X at height y and
/// depth z; else rays along -Z at x, y. Returns the hit coordinate (x or z), the largest one.
fn zz_cast(m: &Cage, origin: Vec3, from_side: bool, a: f32, y: f32) -> Option<f32> {
    let mut best: Option<f32> = None;
    for f in 0..m.face_count() {
        let c = m.corners_of(f);
        for i in 1..c.len() - 1 {
            let t = [c[0], c[i], c[i + 1]].map(|v| (m.pos[v as usize] - origin) * 1000.0);
            // 2D point-in-triangle in the plane orthogonal to the ray.
            let pr = |p: Vec3| if from_side { (p.z, p.y, p.x) } else { (p.x, p.y, p.z) };
            let (p0, p1, p2) = (pr(t[0]), pr(t[1]), pr(t[2]));
            let d = (p1.1 - p2.1) * (p0.0 - p2.0) + (p2.0 - p1.0) * (p0.1 - p2.1);
            if d.abs() < 1e-9 {
                continue;
            }
            let l0 = ((p1.1 - p2.1) * (a - p2.0) + (p2.0 - p1.0) * (y - p2.1)) / d;
            let l1 = ((p2.1 - p0.1) * (a - p2.0) + (p0.0 - p2.0) * (y - p2.1)) / d;
            let l2 = 1.0 - l0 - l1;
            if l0 >= 0.0 && l1 >= 0.0 && l2 >= 0.0 {
                let h = l0 * p0.2 + l1 * p1.2 + l2 * p2.2;
                best = Some(best.map_or(h, |b: f32| b.max(h)));
            }
        }
    }
    best
}

#[test]
fn zz_probe_cheek_rays() {
    let s = super::body_surface().unwrap();
    let m = s.mesh();
    let o = s.anatomy.head_origin;
    for (side, a) in [
        (false, 25.0_f32),
        (false, 35.0),
        (false, 40.0),
        (true, 40.0),
        (true, 25.0),
        (true, 10.0),
        (true, 0.0),
        (true, -10.0),
        (true, -25.0),
    ] {
        let prof: Vec<(f32, f32)> =
            (-40..=30).filter_map(|y| zz_cast(m, o, side, a, y as f32).map(|h| (y as f32, h))).collect();
        let mut line = String::new();
        for w in prof.windows(3) {
            let k = w[0].1 - 2.0 * w[1].1 + w[2].1;
            if k.abs() > 0.1 {
                let _ = std::fmt::Write::write_fmt(&mut line, format_args!("{}:{:.1}({:+.2}) ", w[1].0, w[1].1, k));
            }
        }
        eprintln!("{} {a}: {line}", if side { "z" } else { "x" });
    }
}

#[test]
fn zz_probe_eye_fit() {
    let s = super::body_surface().unwrap();
    let m = s.mesh();
    let e = s.anatomy.eye_center;
    let eye = &s.body.loops.eyes[0];
    for k in 1..4 {
        let line: Vec<String> = eye.rings[k]
            .iter()
            .map(|&v| {
                let t = (s.targets[v as usize] - e) * 1000.0;
                let p = (m.pos[v as usize] - e) * 1000.0;
                format!("({:.1},{:.1},{:.1})->({:.1},{:.1},{:.1})", t.x, t.y, t.z, p.x, p.y, p.z)
            })
            .collect();
        eprintln!("ring {k}: {}", line.join(" "));
    }
}

#[test]
fn zz_probe_eye_rays() {
    let s = super::body_surface().unwrap();
    let m = s.mesh();
    let e = s.anatomy.eye_center;
    for x in [-8.0_f32, 0.0, 8.0] {
        let mut open = Vec::new();
        for k in -24..=24 {
            let y = k as f32 * 0.5;
            if zz_cast(m, e, false, x, y).is_some_and(|z| z < 0.0) {
                open.push(y);
            }
        }
        eprintln!("eye x={x}: open {:?}..{:?}", open.first(), open.last());
    }
}

#[test]
fn zz_probe_open_edges() {
    let a = super::anatomy::Anatomy::neutral();
    let body = super::body::build(&a).unwrap();
    let c = &body.cage;
    let mut dir: std::collections::HashMap<(u32, u32), Vec<usize>> = std::collections::HashMap::new();
    for f in 0..c.face_count() {
        let cs = c.corners_of(f);
        for k in 0..cs.len() {
            dir.entry((cs[k], cs[(k + 1) % cs.len()])).or_default().push(f);
        }
    }
    for (&(a0, b0), fs) in &dir {
        if fs.len() > 1 || !dir.contains_key(&(b0, a0)) {
            let regs: Vec<String> =
                fs.iter().map(|&f| format!("f{} r{} n{}", f, c.tags[f].region, c.corners_of(f).len())).collect();
            eprintln!("edge {a0}->{b0}: {regs:?} pos {:?}", c.pos[a0 as usize]);
        }
    }
}

#[test]
fn zz_probe_folds() {
    let s = super::body_surface().unwrap();
    let m = s.mesh();
    let (_, _, edge_faces) = m.edges();
    let normal = |f: usize| {
        let c = m.corners_of(f);
        let o = m.pos[c[0] as usize];
        let mut n = Vec3::ZERO;
        for k in 0..c.len() {
            n += (m.pos[c[k] as usize] - o).cross(m.pos[c[(k + 1) % c.len()] as usize] - o);
        }
        n.normalize_or_zero()
    };
    let mut hits = Vec::new();
    for ef in &edge_faces {
        let (a, b) = (ef[0] as usize, ef[1] as usize);
        let d = normal(a).dot(normal(b));
        if d < -0.2 {
            let c = (m.centroid(a) - s.anatomy.head_origin) * 1000.0;
            if c.y > -200.0 && (c.x.abs() > 60.0 || (c.y > 12.0 && c.z > 80.0)) {
                hits.push((d, c, m.tags[a].region));
            }
        }
    }
    hits.sort_by(|p, q| p.0.total_cmp(&q.0));
    for h in hits.iter().take(30) {
        eprintln!("fold {:.2} at {:.1?} region {}", h.0, h.1, h.2);
    }
    eprintln!("{} folds", hits.len());
}

#[test]
fn zz_probe_grid_bumps() {
    let a = super::anatomy::Anatomy::neutral();
    let g = |r: usize, c: usize| (super::body::grid_point(&a, r, c) - a.head_origin) * 1000.0;
    for r in 16..24 {
        let mut line = String::new();
        for c in 0..9 {
            let p = g(r, c);
            let nb = (g(r - 1, c) + g(r + 1, c)) * 0.5;
            let side = (g(r, c.saturating_sub(1).max(if c == 0 { 23 } else { 0 })) + g(r, c + 1)) * 0.5;
            let side = if c == 0 { (g(r, 23) + g(r, 1)) * 0.5 } else { side };
            // Outward ~ horizontal from the head axis.
            let n = Vec3::new(p.x, 0.0, p.z).normalize_or_zero();
            let _ = std::fmt::Write::write_fmt(
                &mut line,
                format_args!("c{c}:{:+.1}/{:+.1} ", (p - nb).dot(n), (p - side).dot(n)),
            );
        }
        eprintln!("H{} {line}", r - 14);
    }
}

/// The ANSUR II sex-midpoint mean of a variable, metres, from `data/anthropometry/ansur2.json`.
fn ansur_midpoint(id: &str) -> f32 {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/anthropometry/ansur2.json");
    let bytes = std::fs::read(path).unwrap();
    let value = humanart_io::json::parse(&bytes).unwrap();
    let mm = value.node().field("sex_midpoint_mean").unwrap().field(id).unwrap().f64().unwrap();
    (mm * 1e-3) as f32
}

/// Body measurements of the template surface taken the way ANSUR II defines them (Gordon et al.
/// 2014).
pub struct Measured {
    /// `(id, template value, ANSUR II sex-midpoint mean, tolerance)`, metres.
    pub rows: Vec<(String, f32, f32, f32)>,
    /// The lowest height where a left-thigh vertex lies within 1 mm of the midline.
    pub thighs_touch_to: f32,
    /// The most a left-thigh vertex reaches past the midline (negative x), metres.
    pub thigh_crossing: f32,
}

/// Measures the surface (see [`Measured`]).
pub fn measure_surface(
    s: &super::Surface,
    asm: &super::assemble::Assembled,
    lm: &[super::landmarks::Described],
) -> Measured {
    use super::anatomy as an;
    let pos = asm.mesh.positions();
    let at = |n: &str| lm.iter().find(|d| d.landmark.name == n).unwrap().landmark.position(pos).unwrap();
    let body = &s.mesh().pos;
    // The region of each body vertex (its first copy in the assembled mesh).
    let mut copy = vec![u32::MAX; body.len()];
    for (i, &(piece, v)) in asm.source.iter().enumerate() {
        if piece == 0 && copy[v as usize] == u32::MAX {
            copy[v as usize] = i as u32;
        }
    }
    let regions = &asm.regions;
    let in_region = |v: usize, names: &[&str]| {
        let r = regions.vertex_regions[copy[v] as usize];
        names.iter().filter_map(|n| regions.find(n)).any(|x| regions.is_within(r, x))
    };
    let all = || 0..body.len();
    let floor = body.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
    // Crotch: the lowest point of the seam where the legs meet (the cage's crotch path refined
    // through the subdivision levels).
    let mut path = s.body.loops.crotch.to_vec();
    for level in &s.levels {
        path = level.refine(&path, false);
    }
    let crotch = path.iter().map(|&v| body[v as usize].y).fold(f32::INFINITY, f32::min);
    let trunk = ["chest", "abdomen", "back", "pelvis"];
    let hips = ["pelvis", "abdomen", "back", "thigh"];
    let band = |names: &[&str], y: f32, keep: &dyn Fn(Vec3) -> bool| -> Vec<Vec3> {
        all()
            .filter(|&v| (body[v].y - y).abs() <= 0.008 && keep(body[v]) && in_region(v, names))
            .map(|v| body[v])
            .collect()
    };
    // Girth as a tape takes it: the convex hull of a band of +-8 mm around the level.
    let girth = |pts: &[Vec3]| {
        let flat: Vec<(f32, f32)> = pts.iter().map(|p| (p.x, p.z)).collect();
        crate::identity::measure::hull_perimeter(&flat)
    };
    let extent = |pts: &[Vec3], f: &dyn Fn(Vec3) -> f32| {
        let (lo, hi) =
            pts.iter().map(|&p| f(p)).fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), x| (a.min(x), b.max(x)));
        hi - lo
    };
    let any = |_: Vec3| true;
    // Buttock level: the maximum posterior protrusion between the crotch and the iliac crest.
    let buttock_y = all()
        .filter(|&v| body[v].y > crotch && body[v].y < 1.0 && in_region(v, &hips))
        .min_by(|&a, &b| body[a].z.total_cmp(&body[b].z))
        .map_or(0.0, |v| body[v].y);
    let hip = all()
        .filter(|&v| body[v].y > 0.7 && body[v].y < 1.0 && in_region(v, &hips))
        .map(|v| body[v].x.abs())
        .fold(0.0_f32, f32::max)
        * 2.0;
    let chest = band(&["chest", "back", "abdomen"], at("L.chest.nipple").y, &any);
    let waist = band(&trunk, at("torso.navel").y, &any);
    let buttock = band(&hips, buttock_y, &any);
    // Thigh: one thigh at the lowest point of the gluteal furrow, taken 20 mm below the crotch.
    let thigh = band(&["thigh-L"], crotch - 0.020, &|p| p.x > 0.0);
    let mid = ansur_midpoint;
    let mut rows: Vec<(String, f32, f32, f32)> = Vec::new();
    let mut row = |name: &str, got: f32, want: f32, tol: f32| rows.push((name.to_owned(), got, want, tol));
    row("floor", floor, 0.0, 0.003);
    row("stature", at("head.vertex").y - floor, an::STATURE, 0.004);
    row("tragiontopofhead", at("head.vertex").y - at("ear.L.tragion").y, an::TRAGION_TO_VERTEX, 0.001);
    row("suprasternaleheight", at("torso.suprasternal").y, an::SUPRASTERNALE_HEIGHT, 0.001);
    row("cervicaleheight", at("spine.c7").y, an::CERVICALE_HEIGHT, 0.001);
    row("trochanterionheight", at("L.hip.trochanter").y, an::TROCHANTERION_HEIGHT, 0.001);
    row("lateralfemoralepicondyleheight", at("L.knee.lateral").y, an::KNEE_HEIGHT, 0.001);
    row("lateralmalleolusheight", at("L.ankle.lateral").y, an::MALLEOLUS_HEIGHT, 0.001);
    row("footlength", at("L.heel").distance(at("L.foot.tip")), an::FOOT_LENGTH, 0.02 * an::FOOT_LENGTH);
    let wrist = (at("L.wrist.radial") + at("L.wrist.ulnar")) * 0.5;
    // The rest hand is relaxed (fingers flexed), so the straight-line length is a little short.
    row("handlength (relaxed)", wrist.distance(at("L.hand.middle.tip")), an::HAND_LENGTH, 0.04 * an::HAND_LENGTH);
    // Hand breadth across the metacarpal heads (II to V): the extent of the palm along the radial
    // axis in the band of the knuckles (calipers press the soft tissue a little).
    let a = &s.anatomy;
    let [hd, _, hr] = a.hand_axes;
    let knuckle_band: Vec<Vec3> = all()
        .filter(|&v| {
            let q = body[v] - a.wrist;
            body[v].x > 0.0 && (0.080..0.095).contains(&q.dot(hd)) && in_region(v, &["hand-L.palm"])
        })
        .map(|v| body[v])
        .collect();
    row("handbreadth", extent(&knuckle_band, &|p| p.dot(hr)), an::HAND_BREADTH, 0.06 * an::HAND_BREADTH);
    row(
        "footbreadth (mtp1-mtp5)",
        at("L.foot.mtp1").distance(at("L.foot.mtp5")),
        an::FOOT_BREADTH,
        0.05 * an::FOOT_BREADTH,
    );
    row("biacromialbreadth", at("L.shoulder.acromion").distance(at("R.shoulder.acromion")), an::BIACROMIAL, 0.008);
    // ANSUR measures the acromion with the arms hanging; the rest pose's 45-degree abduction lifts
    // the shoulder girdle, so the template's acromion stands about 20 mm higher.
    row("acromialheight (A-pose)", at("L.shoulder.acromion").y, an::ACROMIAL_HEIGHT, 0.030);
    row("crotchheight", crotch, an::CROTCH_HEIGHT, 0.004);
    row("hipbreadth", hip, an::HIP_BREADTH, 0.006);
    row("buttockheight", buttock_y, mid("buttockheight"), 0.020);
    row("buttockcircumference", girth(&buttock), mid("buttockcircumference"), 0.020);
    row("buttockdepth", extent(&buttock, &|p| p.z), mid("buttockdepth"), 0.008);
    row("waistcircumference", girth(&waist), mid("waistcircumference"), 0.015);
    row("waistbreadth", extent(&waist, &|p| p.x), mid("waistbreadth"), 0.010);
    row("waistdepth", extent(&waist, &|p| p.z), mid("waistdepth"), 0.008);
    row("chestcircumference", girth(&chest), mid("chestcircumference"), 0.015);
    row("chestdepth", extent(&chest, &|p| p.z), mid("chestdepth"), 0.008);
    row("thighcircumference", girth(&thigh), mid("thighcircumference"), 0.020);
    // Thighs: where a left-thigh vertex is within 1 mm of the midline, and how far one crosses it.
    let left_thigh: Vec<usize> = all().filter(|&v| in_region(v, &["thigh-L"]) && body[v].x.abs() > 1.0e-5).collect();
    let thighs_touch_to =
        left_thigh.iter().filter(|&&v| body[v].x < 0.001).map(|&v| body[v].y).fold(f32::INFINITY, f32::min);
    let thigh_crossing = left_thigh.iter().map(|&v| -body[v].x).fold(0.0_f32, f32::max);
    Measured { rows, thighs_touch_to, thigh_crossing }
}

#[test]
fn anat_02_surface_meets_the_ansur_constants() {
    let s = super::body_surface().unwrap();
    let parts = super::parts::build(&s.anatomy, &s.body, &s.targets).unwrap();
    let pieces: Vec<&super::cage::Cage> = std::iter::once(s.mesh()).chain(parts.iter().map(|p| &p.mesh)).collect();
    let layout = super::uv::layout(&pieces).unwrap();
    let asm = super::assemble::assemble(&s, &parts, &layout).unwrap();
    let lm = super::landmarks::build(&s, &asm).unwrap();
    let m = measure_surface(&s, &asm, &lm);
    let mut lines = vec!["measure: template / ANSUR II sex midpoint (difference, tolerance), mm".to_owned()];
    let mut bad = Vec::new();
    for (name, got, want, tol) in &m.rows {
        lines.push(format!(
            "{name}: {:.1} / {:.1} ({:+.1}, +-{:.0})",
            got * 1e3,
            want * 1e3,
            (got - want) * 1e3,
            tol * 1e3
        ));
        if (got - want).abs() > *tol {
            bad.push(name.clone());
        }
    }
    lines.push(format!("thighs touch down to: {:.1} mm", m.thighs_touch_to * 1e3));
    lines.push(format!("thigh crossing the midline: {:.1} mm", m.thigh_crossing * 1e3));
    std::fs::write(preview_dir().join("anthropometry.txt"), lines.join("\n") + "\n").unwrap();
    assert!(bad.is_empty(), "out of tolerance: {bad:?}\n{}", lines.join("\n"));
    // The thighs meet only near the crotch and never pass through each other.
    let crotch = m.rows.iter().find(|r| r.0 == "crotchheight").map_or(0.0, |r| r.1);
    assert!(m.thigh_crossing < 0.0005, "{}", m.thigh_crossing);
    assert!(m.thighs_touch_to > crotch - 0.040, "{} below the crotch {crotch}", m.thighs_touch_to);
}

#[test]
fn zz_probe_cheek_slope() {
    let s = super::body_surface().unwrap();
    let m = s.mesh();
    let o = s.anatomy.head_origin;
    for x in [45.0_f32, 55.0, 62.0, 0.0, 30.0] {
        let range = if x < 40.0 { 40..=124 } else { -36..=16 };
        let prof: Vec<(f32, f32)> =
            range.step_by(2).filter_map(|y| zz_cast(m, o, false, x, y as f32).map(|h| (y as f32, h))).collect();
        let line: Vec<String> =
            prof.windows(2).map(|w| format!("{:.0}:{:+.2}", w[0].0, (w[1].1 - w[0].1) / (w[1].0 - w[0].0))).collect();
        eprintln!("x {x}: {}", line.join(" "));
    }
}

/// Open-mouth previews with the parts coloured (teeth, gums and tongue, the mouth bag dark), and
/// the eyes seen through the lids, for ADR-039's mouth and eye check.
#[test]
fn preview_mouth_open_and_eyes() {
    use humanart_math::Quat;
    let s = super::body_surface().unwrap();
    let parts = super::parts::build(&s.anatomy, &s.body, &s.targets).unwrap();
    let pieces: Vec<&super::cage::Cage> = std::iter::once(s.mesh()).chain(parts.iter().map(|p| &p.mesh)).collect();
    let layout = super::uv::layout(&pieces).unwrap();
    let asm = super::assemble::assemble(&s, &parts, &layout).unwrap();
    let rig = super::rig::skeleton(&s.anatomy, 0).unwrap();
    let skin = super::rig::skin(&rig, &s.anatomy, &asm.mesh, &asm.regions).unwrap();
    let pos = asm.mesh.positions();
    let interior = asm.regions.find("mouth-interior").unwrap();
    let face_region = |f: usize| asm.regions.vertex_regions[asm.mesh.face(f)[0] as usize];
    for (name, jaw) in [("mouth-open", 0.22_f32), ("mouth-ajar", 0.06)] {
        let posed = pose_lbs(&rig.skeleton, &skin, pos, &[("jaw", Quat::from_rotation_x(jaw))]);
        {
            let key = |p: Vec3| [(p.x * 1e5).round() as i64, (p.y * 1e5).round() as i64, (p.z * 1e5).round() as i64];
            let map: std::collections::HashMap<[i64; 3], usize> =
                pos.iter().enumerate().map(|(i, &p)| (key(p), i)).collect();
            let mut worst = (0.0_f32, 0usize);
            for (v, &p) in pos.iter().enumerate() {
                let Some(&m) = map.get(&key(Vec3::new(-p.x, p.y, p.z))) else { continue };
                let q = posed[m];
                let e = posed[v].distance(Vec3::new(-q.x, q.y, q.z));
                if e > worst.0 {
                    worst = (e, v);
                }
            }
            let r = asm.regions.vertex_regions[worst.1];
            eprintln!(
                "{name}: worst posed mirror error {:.3} mm at {:?} region {}",
                worst.0 * 1e3,
                pos[worst.1] * 1e3,
                asm.regions.regions[r as usize].name
            );
            assert!(worst.0 < 5e-5, "the posed jaw is not mirror-symmetric");
        }
        let mut scene = Scene { pos: posed, faces: Vec::new() };
        for part in asm.mesh.parts() {
            let color = match part.name.as_str() {
                n if n.starts_with("teeth") => [0.93, 0.9, 0.82],
                n if n.starts_with("gums") || n == "tongue" => [0.78, 0.36, 0.4],
                n if n.starts_with("eyeball") => [0.92, 0.92, 0.9],
                "mouth-helper" | "tear-line-L" | "tear-line-R" | "eye-occlusion-L" | "eye-occlusion-R" => continue,
                _ => [0.80, 0.66, 0.58],
            };
            for f in part.faces() {
                let c = if face_region(f) == interior { [0.35, 0.12, 0.12] } else { color };
                scene.faces.push((asm.mesh.face(f).to_vec(), c));
            }
        }
        let mouth = s.anatomy.head_origin + Vec3::new(0.0, -0.055, 0.09);
        let names: [&'static str; 2] =
            if jaw > 0.1 { ["mouth-open", "mouth-open-34"] } else { ["mouth-ajar", "mouth-ajar-34"] };
        let _ = name;
        render(&scene, &View::new(names[0], -Vec3::Z).window(mouth, 0.035).size(900, 900));
        render(&scene, &View::new(names[1], Vec3::new(-0.6, 0.0, -0.8)).window(mouth, 0.035).size(900, 900));
        if jaw < 0.1 {
            let head = s.anatomy.head_origin + Vec3::new(0.0, -0.01, 0.05);
            render(&scene, &View::new("face-parts-front", -Vec3::Z).window(head, 0.12).size(900, 900));
            render(&scene, &View::new("face-parts-34", Vec3::new(-0.6, 0.0, -0.8)).window(head, 0.12).size(900, 900));
        }
    }
}

/// Every body cage face has the mirror image of its mirror face's region and chart (a face tagged by
/// its first slot instead of its middle once made the lip corners, nose and ears asymmetric, and the
/// skin weights and expressions with them).
#[test]
fn mod_03_body_regions_are_mirror_symmetric() {
    use super::regions::R;
    let s = super::body_surface().unwrap();
    let c = &s.body.cage;
    let key = |p: Vec3| [(p.x * 1e5).round() as i64, (p.y * 1e5).round() as i64, (p.z * 1e5).round() as i64];
    let map: std::collections::HashMap<[i64; 3], usize> =
        (0..c.face_count()).map(|f| (key(c.centroid(f)), f)).collect();
    let unsided = |n: &str| n.replace("-L", "-?").replace("-R", "-?");
    let (mut bad, mut missing) = (Vec::new(), 0);
    for f in 0..c.face_count() {
        let p = c.centroid(f);
        if p.x <= 1e-4 {
            continue;
        }
        // Centroids on a rounding boundary find no partner: counted, a few allowed.
        let Some(&g) = map.get(&key(Vec3::new(-p.x, p.y, p.z))) else {
            missing += 1;
            continue;
        };
        let (a, b) = (R::from_id(c.tags[f].region).unwrap().name(), R::from_id(c.tags[g].region).unwrap().name());
        // The nose's tip / ala / bridge split is knowingly side-dependent (see `face::nose`).
        let nose = |n: &str| ["nasal-tip", "nasal-bridge", "ala-L", "ala-R"].contains(&n);
        if nose(a) && nose(b) {
            continue;
        }
        let charts =
            (super::regions::CHARTS[c.tags[f].chart as usize].1, super::regions::CHARTS[c.tags[g].chart as usize].1);
        if unsided(a) != unsided(b) || unsided(charts.0) != unsided(charts.1) {
            bad.push(format!("{a}/{} vs {b}/{} at {:?}", charts.0, charts.1, (p - s.anatomy.head_origin) * 1000.0));
        }
    }
    assert!(bad.is_empty(), "{} asymmetric faces: {:?}", bad.len(), &bad[..bad.len().min(8)]);
    assert!(missing < c.face_count() / 100, "{missing} faces without a mirror partner");
}

/// Lips without wires (front, front in raking light, side, three-quarter, from below) for the
/// round-3 lip shape: the vermilion border, Cupid's bow, philtrum, tubercle and the corners.
#[test]
fn preview_lips() {
    let s = super::body_surface().unwrap();
    let plain = scene_of(s.mesh(), &|_| [0.80, 0.66, 0.58]);
    let mouth = s.anatomy.head_origin + Vec3::new(0.0, -0.045, 0.09);
    render(&plain, &View::new("lips-front", -Vec3::Z).window(mouth, 0.032).size(800, 800));
    render(&plain, &View::new("lips-front-raking", -Vec3::Z).window(mouth, 0.032).size(800, 800).raking());
    render(&plain, &View::new("lips-side", -Vec3::X).window(mouth, 0.032).size(800, 800));
    render(&plain, &View::new("lips-34", Vec3::new(-0.6, -0.1, -0.8)).window(mouth, 0.032).size(800, 800));
    render(&plain, &View::new("lips-below", Vec3::new(0.0, 0.6, -0.8)).window(mouth, 0.032).size(800, 800));
}

#[test]
fn zz_probe_asym() {
    let s = super::body_surface().unwrap();
    let c = &s.body.cage;
    let d = mirror_distances(&s.targets, 0.004);
    let mut idx: Vec<usize> = (0..d.len()).collect();
    idx.sort_by(|&a, &b| d[b].total_cmp(&d[a]));
    for &i in idx.iter().take(8) {
        eprintln!("target {i}: {:.4} mm at {:?}", d[i] * 1e3, (s.targets[i] - s.anatomy.head_origin) * 1e3);
    }
    let d = mirror_distances(&c.pos, 0.004);
    let mut idx: Vec<usize> = (0..d.len()).collect();
    idx.sort_by(|&a, &b| d[b].total_cmp(&d[a]));
    for &i in idx.iter().take(4) {
        eprintln!("cage {i}: {:.4} mm at {:?}", d[i] * 1e3, (c.pos[i] - s.anatomy.head_origin) * 1e3);
    }
}

#[test]
fn zz_probe_jaw_inversions() {
    use humanart_math::Quat;
    let s = super::body_surface().unwrap();
    let parts = super::parts::build(&s.anatomy, &s.body, &s.targets).unwrap();
    let pieces: Vec<&super::cage::Cage> = std::iter::once(s.mesh()).chain(parts.iter().map(|p| &p.mesh)).collect();
    let layout = super::uv::layout(&pieces).unwrap();
    let asm = super::assemble::assemble(&s, &parts, &layout).unwrap();
    let rig = super::rig::skeleton(&s.anatomy, 0).unwrap();
    let skin = super::rig::skin(&rig, &s.anatomy, &asm.mesh, &asm.regions).unwrap();
    let pos = asm.mesh.positions();
    let o = s.anatomy.head_origin;
    for jaw in [0.1_f32, 0.22, 0.35] {
        let posed = pose_lbs(&rig.skeleton, &skin, pos, &[("jaw", Quat::from_rotation_x(jaw))]);
        let mut n = 0;
        let mut hist = std::collections::BTreeMap::new();
        for f in 0..asm.mesh.face_count() {
            let c = asm.mesh.face(f);
            let a = newell(pos, c);
            let b = newell(&posed, c);
            if a.dot(b) < 0.0 {
                n += 1;
                let r = asm.regions.vertex_regions[c[0] as usize];
                *hist.entry(asm.regions.regions[r as usize].name.clone()).or_insert(0) += 1;
                if n <= 3 {
                    eprintln!("jaw {jaw}: inverted at {:?}", (pos[c[0] as usize] - o) * 1e3);
                }
            }
        }
        eprintln!("jaw {jaw}: {n} inverted faces {hist:?}");
    }
}

#[test]
#[ignore = "probe"]
fn zz_probe_uv_flips() {
    let s = super::body_surface().unwrap();
    let parts = super::parts::build(&s.anatomy, &s.body, &s.targets).unwrap();
    let pieces: Vec<&super::cage::Cage> = std::iter::once(s.mesh()).chain(parts.iter().map(|p| &p.mesh)).collect();
    let layout = super::uv::layout(&pieces).unwrap();
    let asm = super::assemble::assemble(&s, &parts, &layout).unwrap();
    let (uv, pos) = (asm.mesh.uvs(), asm.mesh.positions());
    let body = s.mesh();
    let mut areas = Vec::new();
    let mut sum: std::collections::HashMap<u16, f32> = std::collections::HashMap::new();
    for f in 0..body.face_count() {
        let c = asm.mesh.face(f);
        let mut a = 0.0;
        for k in 0..c.len() {
            let (p, q) = (uv[c[k] as usize], uv[c[(k + 1) % c.len()] as usize]);
            a += p.x * q.y - q.x * p.y;
        }
        areas.push(a);
        *sum.entry(body.tags[f].chart).or_default() += a.signum();
    }
    let mut count: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for f in 0..body.face_count() {
        let tag = body.tags[f];
        let a = areas[f];
        let name = format!(
            "{} / {}",
            super::regions::REGIONS[tag.region as usize].1,
            super::regions::CHARTS[tag.chart as usize].1
        );
        if a.abs() < 1e-14 || a.signum() != sum[&tag.chart].signum() {
            let e = count.entry(name.clone()).or_default();
            if *e < 3 {
                eprintln!("face {f} area {a:e} {name} at {:?}", pos[asm.mesh.face(f)[0] as usize]);
            }
            *e += 1;
        }
    }
    eprintln!("{count:?}");
}

/// The faces of `mesh` whose region is (not) genital.
fn genital_faces(mesh: &Cage, genital: bool) -> Vec<usize> {
    (0..mesh.face_count())
        .filter(|&f| {
            super::regions::R::from_id(mesh.tags[f].region).is_some_and(super::regions::R::is_genital) == genital
        })
        .collect()
}

#[test]
fn adr_039_body_is_identical_with_either_variant() {
    // The body outside the attachment loop is bit for bit the same with either variant, and both
    // variants close the loop (the whole surface is closed and consistently oriented).
    let surfaces: Vec<super::Surface> =
        super::genitals::Variant::ALL.iter().map(|&v| super::body_surface_with(v).unwrap()).collect();
    let body_corners = |s: &super::Surface| -> Vec<[u32; 3]> {
        let m = s.mesh();
        genital_faces(m, false)
            .iter()
            .flat_map(|&f| m.corners_of(f).iter().map(|&v| m.pos[v as usize].to_array().map(f32::to_bits)))
            .collect()
    };
    let (a, b) = (body_corners(&surfaces[0]), body_corners(&surfaces[1]));
    assert_eq!(a.len(), b.len());
    let differ = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    assert_eq!(differ, 0, "{differ} body corners differ between the variants");
    for s in &surfaces {
        let m = s.mesh();
        assert_eq!(closed_and_oriented(m), (0, 0), "{:?}", s.variant);
        let g = genital_faces(m, true);
        assert!(g.len() > 500, "{:?}: {} genital faces", s.variant, g.len());
        // No genital face off the midline reaches across it (the two sides never pass through each
        // other); faces centred on the midline (the cap at the tip) straddle it by design.
        let crossing = g
            .iter()
            .filter(|&&f| {
                let c = m.centroid(f).x;
                c.abs() > 1e-3 && m.corners_of(f).iter().any(|&v| m.pos[v as usize].x * c.signum() < -1e-4)
            })
            .count();
        assert_eq!(crossing, 0, "{:?}: {crossing} genital faces cross the midline", s.variant);
        let d = mirror_distances(&m.pos, 0.004);
        let max = d.iter().copied().fold(0.0_f32, f32::max);
        assert!(max < 5e-5, "{:?}: not mirror-symmetric ({max} m)", s.variant);
    }
}

/// How deep the genital vertices of `s` go into the thighs: per vertex, the signed distance along
/// the nearest thigh vertex's normal (negative inside), the minimum over the genital vertices, and
/// the genital vertex where it is reached.
fn thigh_penetration(s: &super::Surface) -> (f32, Vec3) {
    let m = s.mesh();
    let mut normals = vec![Vec3::ZERO; m.pos.len()];
    for f in 0..m.face_count() {
        let c = m.corners_of(f);
        let o = m.pos[c[0] as usize];
        let mut n = Vec3::ZERO;
        for k in 0..c.len() {
            n += (m.pos[c[k] as usize] - o).cross(m.pos[c[(k + 1) % c.len()] as usize] - o);
        }
        for &v in c {
            normals[v as usize] += n;
        }
    }
    let thigh: Vec<usize> = (0..m.face_count())
        .filter(|&f| {
            matches!(
                super::regions::R::from_id(m.tags[f].region),
                Some(super::regions::R::ThighL | super::regions::R::ThighR)
            )
        })
        .flat_map(|f| m.corners_of(f).iter().map(|&v| v as usize).collect::<Vec<_>>())
        .collect();
    let mut genital: Vec<usize> = genital_faces(m, true)
        .iter()
        .flat_map(|&f| m.corners_of(f).iter().map(|&v| v as usize).collect::<Vec<_>>())
        .collect();
    genital.sort_unstable();
    genital.dedup();
    // Near the attachment loop the genital surface meets the body in a concave fold, where the
    // nearest thigh vertex's tangent plane says nothing about crossing: skip 8 mm around it, and
    // the penis root and mons above 0.825 m (the same fold against the groin).
    let lp = super::genitals::attachment_loop();
    let near_loop = |p: Vec3| {
        (0..lp.len()).any(|k| {
            let (a, b) = (lp[k], lp[(k + 1) % lp.len()]);
            let t = ((p - a).dot(b - a) / (b - a).dot(b - a)).clamp(0.0, 1.0);
            p.distance(a + (b - a) * t) < 0.008
        })
    };
    genital.retain(|&g| !near_loop(m.pos[g]) && m.pos[g].y < 0.825);
    genital
        .iter()
        .map(|&g| {
            let p = m.pos[g];
            let q = thigh.iter().copied().min_by(|&a, &b| m.pos[a].distance(p).total_cmp(&m.pos[b].distance(p)));
            (q.map_or(0.0, |q| (p - m.pos[q]).dot(normals[q].normalize_or_zero()).min(m.pos[q].distance(p))), p)
        })
        .fold((f32::INFINITY, Vec3::ZERO), |a, b| if b.0 < a.0 { b } else { a })
}

/// The gap between the thighs below the crotch: per height and depth, the smallest `x` of a left
/// thigh vertex within 4 mm of that height and depth (mm), as text rows.
fn thigh_gap(s: &super::Surface) -> Vec<String> {
    let m = s.mesh();
    let thigh: Vec<Vec3> = (0..m.face_count())
        .filter(|&f| matches!(super::regions::R::from_id(m.tags[f].region), Some(super::regions::R::ThighL)))
        .flat_map(|f| m.corners_of(f).iter().map(|&v| m.pos[v as usize]).collect::<Vec<_>>())
        .collect();
    let zs: Vec<f32> = (-4..=10).map(|k| k as f32 * 0.01).collect();
    let mut rows = vec![format!("thigh gap x (mm) at z (cm) {:?}", (-4..=10).collect::<Vec<i32>>())];
    for y in [0.70_f32, 0.73, 0.76, 0.78, 0.80, 0.82, 0.84] {
        let row: Vec<String> = zs
            .iter()
            .map(|&z| {
                let x = thigh
                    .iter()
                    .filter(|p| (p.y - y).abs() < 0.004 && (p.z - z).abs() < 0.004)
                    .map(|p| p.x)
                    .fold(f32::INFINITY, f32::min);
                if x.is_finite() { format!("{:5.1}", x * 1e3) } else { "    -".to_string() }
            })
            .collect();
        rows.push(format!("y {y:.2}: {}", row.join(" ")));
    }
    rows
}

#[test]
fn preview_genitals() {
    let mut lines = Vec::new();
    for v in super::genitals::Variant::ALL {
        let s = super::body_surface_with(v).unwrap();
        let m = s.mesh();
        let skin = [0.80, 0.66, 0.58];
        let plain = scene_of(m, &|_| skin);
        let regions = scene_of(m, &|f| id_color(u32::from(m.tags[f].region)));
        let name = |view: &str| -> &'static str {
            Box::leak(format!("genitals-{v:?}-{view}").to_lowercase().into_boxed_str())
        };
        let site = Vec3::new(0.0, 0.815, 0.03);
        render(&plain, &View::new(name("front"), -Vec3::Z).window(site, 0.07).size(900, 900));
        render(&plain, &View::new(name("front-wire"), -Vec3::Z).window(site, 0.07).size(900, 900).wire());
        render(&plain, &View::new(name("below"), Vec3::new(0.0, 1.0, -0.45)).window(site, 0.07).size(900, 900));
        render(
            &plain,
            &View::new(name("below-wire"), Vec3::new(0.0, 1.0, -0.45)).window(site, 0.07).size(900, 900).wire(),
        );
        render(&plain, &View::new(name("side"), -Vec3::X).window(site, 0.07).size(900, 900));
        // The right half seen from the left: the profile of the midline and the right side.
        let half = Scene {
            pos: m.pos.clone(),
            faces: (0..m.face_count())
                .filter(|&f| m.centroid(f).x < 1e-4 && m.centroid(f).x > -0.12)
                .map(|f| (m.corners_of(f).to_vec(), skin))
                .collect(),
        };
        render(&half, &View::new(name("cut"), -Vec3::X).window(site, 0.07).size(900, 900));
        render(&plain, &View::new(name("34"), Vec3::new(-0.6, 0.0, -1.0)).window(site, 0.07).size(900, 900));
        render(&regions, &View::new(name("regions"), Vec3::new(0.0, 0.6, -1.0)).window(site, 0.07).size(900, 900));
        render(
            &plain,
            &View::new(name("pelvis"), -Vec3::Z).window(Vec3::new(0.0, 0.86, 0.0), 0.2).size(900, 900).raking(),
        );
        for region in [super::regions::R::Scrotum, super::regions::R::Penis, super::regions::R::LabiaMajora] {
            let (lo, hi) = (0..m.face_count())
                .filter(|&f| super::regions::R::from_id(m.tags[f].region) == Some(region))
                .flat_map(|f| m.corners_of(f).iter().map(|&v| m.pos[v as usize]).collect::<Vec<_>>())
                .fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), p| (lo.min(p), hi.max(p)));
            if lo.x <= hi.x {
                lines.push(format!(
                    "{v:?} {region:?}: box {:.1?} .. {:.1?} mm",
                    (lo * 1e3).to_array(),
                    (hi * 1e3).to_array()
                ));
            }
        }
        // The kink across the attachment loop: angle between the normals of the body face and the
        // genital face on each loop edge of the surface.
        let is_genital =
            |f: usize| super::regions::R::from_id(m.tags[f].region).is_some_and(super::regions::R::is_genital);
        let face_normal = |f: usize| {
            let c = m.corners_of(f);
            let o = m.pos[c[0] as usize];
            (1..c.len() - 1)
                .map(|k| (m.pos[c[k] as usize] - o).cross(m.pos[c[k + 1] as usize] - o))
                .fold(Vec3::ZERO, |a, b| a + b)
                .normalize_or_zero()
        };
        let (_, _, edge_faces) = m.edges();
        let kinks: Vec<f32> = edge_faces
            .iter()
            .filter(|f| f[1] != u32::MAX && is_genital(f[0] as usize) != is_genital(f[1] as usize))
            .map(|f| face_normal(f[0] as usize).dot(face_normal(f[1] as usize)).clamp(-1.0, 1.0).acos().to_degrees())
            .collect();
        let mean = kinks.iter().sum::<f32>() / kinks.len().max(1) as f32;
        let max = kinks.iter().copied().fold(0.0_f32, f32::max);
        lines.push(format!("{v:?}: loop kink over {} edges: mean {mean:.1} deg, max {max:.1} deg", kinks.len()));
        let (depth, at) = thigh_penetration(&s);
        lines.push(format!("{v:?}: thigh penetration {:.1} mm at {at:?}", depth * 1e3));
        if v == super::genitals::Variant::Female {
            lines.extend(thigh_gap(&s));
        }
    }
    lines.push(format!("penis centreline {:.1} mm", super::genitals::shaft_length() * 1e3));
    lines.push(format!("site midline {:.1} mm", super::genitals::site_length() * 1e3));
    eprintln!("{}", lines.join("\n"));
    std::fs::write(preview_dir().join("genitals.txt"), lines.join("\n") + "\n").unwrap();
}

/// Diagnostic for `ac_wp15_2_thickness_on_the_template` (tools): the face tile's islands, their
/// UV boxes and the UV area of the nostril regions (times 1024² is roughly the test's texels).
#[test]
#[ignore = "diagnostic"]
fn diag_face_tile_islands() {
    let s = super::body_surface().unwrap();
    let parts = super::parts::build(&s.anatomy, &s.body, &s.targets).unwrap();
    let pieces: Vec<&Cage> = std::iter::once(s.mesh()).chain(parts.iter().map(|p| &p.mesh)).collect();
    let layout = super::uv::layout(&pieces).unwrap();
    let mut islands: std::collections::BTreeMap<u32, (usize, Vec2, Vec2)> = std::collections::BTreeMap::new();
    for c in layout.copies.iter().filter(|c| c.tile == 0) {
        let e = islands.entry(c.island).or_insert((0, Vec2::splat(f32::MAX), Vec2::splat(f32::MIN)));
        e.0 += 1;
        e.1 = e.1.min(c.uv);
        e.2 = e.2.max(c.uv);
    }
    let mut used = 0.0;
    for (i, (n, lo, hi)) in &islands {
        let d = *hi - *lo;
        used += d.x * d.y;
        eprintln!("island {i}: {n} copies, box {:.4} x {:.4} at {:.3?}", d.x, d.y, lo.to_array());
    }
    let m = s.mesh();
    let mut uv_area = 0.0_f32;
    let mut area = 0.0_f32;
    for f in 0..m.face_count() {
        if !matches!(
            super::regions::R::from_id(m.tags[f].region),
            Some(super::regions::R::NostrilL | super::regions::R::NostrilR)
        ) {
            continue;
        }
        let start = m.offsets[f] as usize;
        let c = m.corners_of(f);
        let uv: Vec<Vec2> =
            (0..c.len()).map(|k| layout.copies[layout.corner_copies[0][start + k] as usize].uv).collect();
        for k in 1..c.len() - 1 {
            let (a, b) = (uv[k] - uv[0], uv[k + 1] - uv[0]);
            uv_area += (a.x * b.y - a.y * b.x).abs() * 0.5;
            let (p, q) = (m.pos[c[k] as usize] - m.pos[c[0] as usize], m.pos[c[k + 1] as usize] - m.pos[c[0] as usize]);
            area += p.cross(q).length() * 0.5;
        }
    }
    eprintln!(
        "face tile: {} islands, boxes cover {used:.3}; nostril 3D area {:.1} mm², UV area {uv_area:.6} = {:.0} texels at 1024²",
        islands.len(),
        area * 1e6,
        uv_area * 1024.0 * 1024.0
    );
}

/// Writes the template with each genital variant (ADR-039) as a package into a data folder for
/// `humanart-cli render --data` (review stills of the current generator with any CLI build).
#[test]
#[ignore = "writes review data"]
fn write_variant_review_packages() {
    for (variant, folder) in
        [(super::genitals::Variant::Female, "data-female"), (super::genitals::Variant::Male, "data-male")]
    {
        let options = super::build::Options { lods: false, genitals: variant, ..super::build::Options::default() };
        let generated = super::build::template(&options).unwrap();
        let bytes = crate::package::write_package(&generated.template).unwrap();
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/agents/template-body")
            .join(folder)
            .join("template");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("template.antmpl"), bytes).unwrap();
    }
}

/// Diagnostic for `ac_wp11_1_template_head_ibl_and_shadows` (render): surface faces above the
/// shoulders that face against their vertices' mean normal (folds the GPU culls as holes).
#[test]
#[ignore = "diagnostic"]
fn diag_head_folds() {
    let s = super::body_surface().unwrap();
    let m = s.mesh();
    let face_normal = |f: usize| {
        let c = m.corners_of(f);
        let o = m.pos[c[0] as usize];
        (1..c.len() - 1)
            .map(|k| (m.pos[c[k] as usize] - o).cross(m.pos[c[k + 1] as usize] - o))
            .fold(Vec3::ZERO, |a, b| a + b)
    };
    let mut normals = vec![Vec3::ZERO; m.pos.len()];
    for f in 0..m.face_count() {
        let n = face_normal(f);
        for &v in m.corners_of(f) {
            normals[v as usize] += n;
        }
    }
    let mut found = 0;
    for f in 0..m.face_count() {
        let c = m.corners_of(f);
        if m.centroid(f).y < 1.40 {
            continue;
        }
        let n = face_normal(f).normalize_or_zero();
        let worst = c.iter().map(|&v| n.dot(normals[v as usize].normalize_or_zero())).fold(1.0_f32, f32::min);
        let near = [
            Vec3::new(-0.018_526_666, 1.520_779_4, 0.094_435_53),
            Vec3::new(-0.080_674_66, 1.550_635_7, 0.003_411_182),
        ]
        .iter()
        .any(|p| p.distance(m.centroid(f)) < 0.008);
        if worst < 0.0 {
            found += 1;
            if near {
                eprintln!(
                    "face {f} {} worst {worst:.2} at {:?}",
                    super::regions::REGIONS[m.tags[f].region as usize].1,
                    m.centroid(f)
                );
            }
        }
    }
    eprintln!("{found} folded faces above 1.40 m");
}

/// Diagnostic for `ac_wp11_1_template_head_ibl_and_shadows`: folded faces of the finished
/// template's body part (scanned face on) above the shoulders, by the same measure as
/// [`diag_head_folds`].
#[test]
#[ignore = "diagnostic"]
fn diag_finished_head_folds() {
    let options = super::build::Options { lods: false, ..super::build::Options::default() };
    let generated = super::build::template(&options).unwrap();
    let mesh = generated.template.mesh();
    let pos = mesh.positions();
    let faces: Vec<Vec<u32>> = mesh.parts()[0].faces().map(|f| mesh.face(f).to_vec()).collect();
    let normal = |c: &[u32]| {
        let o = pos[c[0] as usize];
        (1..c.len() - 1)
            .map(|k| (pos[c[k] as usize] - o).cross(pos[c[k + 1] as usize] - o))
            .fold(Vec3::ZERO, |a, b| a + b)
    };
    let mut normals = vec![Vec3::ZERO; pos.len()];
    for c in &faces {
        let n = normal(c);
        for &v in c {
            normals[v as usize] += n;
        }
    }
    let mut found = 0;
    for c in &faces {
        let centre = c.iter().fold(Vec3::ZERO, |s, &v| s + pos[v as usize]) / c.len() as f32;
        if centre.y < 1.40 {
            continue;
        }
        let n = normal(c).normalize_or_zero();
        if c.iter().any(|&v| n.dot(normals[v as usize].normalize_or_zero()) < 0.0) {
            found += 1;
            if found <= 12 {
                eprintln!("folded at {centre:?}");
            }
        }
    }
    eprintln!("finished template: {found} folded faces above 1.40 m (UV seams split vertices, so seams count too)");
}

/// Diagnostic for the red mark at the base of the throat (todo 1.1, 2026-11-21 renders): the
/// finished template's faces around the jugular notch and the shoulders with their fold measure
/// (face normal against its corners' mean normals), UV tile, UV area and vertex regions.
#[test]
#[ignore = "diagnostic"]
fn diag_neck_base_faces() {
    let options = super::build::Options { lods: false, ..super::build::Options::default() };
    let generated = super::build::template(&options).unwrap();
    let t = &generated.template;
    let mesh = t.mesh();
    let pos = mesh.positions();
    let normal = |c: &[u32]| {
        let o = pos[c[0] as usize];
        (1..c.len() - 1)
            .map(|k| (pos[c[k] as usize] - o).cross(pos[c[k + 1] as usize] - o))
            .fold(Vec3::ZERO, |a, b| a + b)
    };
    // Mean normals by position (UV seams split vertices).
    let key = |p: Vec3| ((p.x * 1e5).round() as i64, (p.y * 1e5).round() as i64, (p.z * 1e5).round() as i64);
    let mut by_pos: std::collections::HashMap<(i64, i64, i64), Vec3> = std::collections::HashMap::new();
    for f in mesh.parts()[0].faces() {
        let c = mesh.face(f);
        let n = normal(c);
        for &v in c {
            *by_pos.entry(key(pos[v as usize])).or_insert(Vec3::ZERO) += n;
        }
    }
    let regions = t.regions();
    let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for f in mesh.parts()[0].faces() {
        let c = mesh.face(f);
        let centre = c.iter().fold(Vec3::ZERO, |s, &v| s + pos[v as usize]) / c.len() as f32;
        if !(1.30..1.47).contains(&centre.y) || centre.z < -0.03 || centre.x.abs() > 0.22 {
            continue;
        }
        let n = normal(c).normalize_or_zero();
        let worst = c.iter().map(|&v| n.dot(by_pos[&key(pos[v as usize])].normalize_or_zero())).fold(1.0_f32, f32::min);
        let names: Vec<&str> =
            c.iter().map(|&v| regions.regions[regions.vertex_regions[v as usize] as usize].name.as_str()).collect();
        *counts.entry(names[0].to_string()).or_default() += 1;
        let uv = c.iter().fold(Vec2::ZERO, |s, &v| s + mesh.uvs()[v as usize]) / c.len() as f32;
        let odd = names.iter().any(|n| !matches!(*n, "neck" | "chest" | "shoulder-L" | "shoulder-R" | "upper-back"));
        if worst < 0.5 || odd || mesh.face_uv_area(f).abs() < 1e-9 {
            eprintln!(
                "face {f} at ({:.4} {:.4} {:.4}) worst {worst:.2} tile {} uv ({:.4} {:.4}) uv-area {:.3e} regions {names:?}",
                centre.x,
                centre.y,
                centre.z,
                mesh.face_tiles()[f],
                uv.x,
                uv.y,
                mesh.face_uv_area(f)
            );
        }
    }
    eprintln!("regions in the box: {counts:?}");
}

/// Folded faces of `m` (face normal against a corner's mean normal below zero) between heights
/// `y0` and `y1`, with their centroids.
fn folded_faces(m: &super::cage::Cage, y0: f32, y1: f32) -> Vec<Vec3> {
    let face_normal = |f: usize| {
        let c = m.corners_of(f);
        let o = m.pos[c[0] as usize];
        (1..c.len() - 1)
            .map(|k| (m.pos[c[k] as usize] - o).cross(m.pos[c[k + 1] as usize] - o))
            .fold(Vec3::ZERO, |a, b| a + b)
    };
    let mut normals = vec![Vec3::ZERO; m.pos.len()];
    for f in 0..m.face_count() {
        let n = face_normal(f);
        for &v in m.corners_of(f) {
            normals[v as usize] += n;
        }
    }
    (0..m.face_count())
        .filter(|&f| (y0..y1).contains(&m.centroid(f).y))
        .filter(|&f| {
            let n = face_normal(f).normalize_or_zero();
            m.corners_of(f).iter().any(|&v| n.dot(normals[v as usize].normalize_or_zero()) < 0.0)
        })
        .map(|f| m.centroid(f))
        .collect()
}

/// Diagnostic (wave 10): folded faces of the torso and shoulders with and without the surface
/// anatomy (`relief::apply`), and the torso's edge lengths (the shortest form the relief can draw).
#[test]
#[ignore = "diagnostic"]
fn diag_relief_folds_and_spacing() {
    let anatomy = super::anatomy::Anatomy::neutral();
    let mut body = super::body::build_with(&anatomy, super::genitals::Variant::default()).unwrap();
    let creases: std::collections::HashSet<(u32, u32)> =
        body.loops.creases.iter().map(|&(a, b)| (a.min(b), a.max(b))).collect();
    let sharp = |a: u32, b: u32| creases.contains(&(a.min(b), a.max(b)));
    let targets = body.cage.pos.clone();
    super::cage::fit(&mut body.cage, &targets, super::LEVELS, super::FIT_ITERATIONS, &sharp);
    let mut levels = super::cage::subdivide_levels(&body.cage, super::LEVELS, &sharp);
    let mut m = levels.pop().unwrap().cage;
    let before = folded_faces(&m, 0.75, 1.47);
    super::relief::fair_cheeks(&mut m, &anatomy);
    super::relief::apply(&mut m, &anatomy);
    let after = folded_faces(&m, 0.75, 1.47);
    eprintln!("folded faces 0.75..1.47 m: {} without relief, {} with", before.len(), after.len());
    let mut clusters: std::collections::BTreeMap<(i32, i32, i32), usize> = std::collections::BTreeMap::new();
    for c in &before {
        *clusters
            .entry(((c.x * 50.0).round() as i32, (c.y * 50.0).round() as i32, (c.z * 50.0).round() as i32))
            .or_default() += 1;
    }
    for (k, n) in &clusters {
        eprintln!("  without, 2 cm cell {:?} m: {n}", (k.0 as f32 / 50.0, k.1 as f32 / 50.0, k.2 as f32 / 50.0));
    }
    for c in after.iter().filter(|c| !before.iter().any(|b| b.distance(**c) < 1e-3)).take(20) {
        eprintln!("  new with relief: {c:?}");
    }
    let mut lengths: Vec<f32> = Vec::new();
    for f in 0..m.face_count() {
        let c = m.corners_of(f);
        let centre = m.centroid(f);
        if centre.z > 0.05 && (1.10..1.20).contains(&centre.y) && (0.03..0.10).contains(&centre.x.abs()) {
            for k in 0..c.len() {
                lengths.push(m.pos[c[k] as usize].distance(m.pos[c[(k + 1) % c.len()] as usize]));
            }
        }
    }
    lengths.sort_by(f32::total_cmp);
    if !lengths.is_empty() {
        let at = |q: f32| lengths[((lengths.len() - 1) as f32 * q) as usize] * 1000.0;
        eprintln!("front torso edge length mm: p10 {:.1} median {:.1} p90 {:.1}", at(0.1), at(0.5), at(0.9));
    }
}

/// Diagnostic (wave 10, the axillary folds): the left arm socket, the first arm rings and the
/// grid around the socket, as designed (before the fit).
#[test]
#[ignore = "diagnostic"]
fn diag_arm_socket() {
    let anatomy = super::anatomy::Anatomy::neutral();
    let body = super::body::build_with(&anatomy, super::genitals::Variant::default()).unwrap();
    let p = |v: u32| body.cage.pos[v as usize];
    eprintln!("shoulder {:?} dir {:?}", anatomy.shoulder, anatomy.upper_arm_dir);
    for (k, ring) in body.loops.arms[0].rings.iter().take(4).enumerate() {
        let pts: Vec<String> = ring.iter().map(|&v| format!("({:.3} {:.3} {:.3})", p(v).x, p(v).y, p(v).z)).collect();
        eprintln!("arm ring {k}: {}", pts.join(" "));
    }
    for r in 5..12 {
        let pts: Vec<String> = (2..9)
            .map(|c| {
                let v = body.loops.grid[r][c];
                if v == u32::MAX {
                    return format!("c{c}(-)");
                }
                let q = p(v);
                format!("c{c}({:.3} {:.3} {:.3})", q.x, q.y, q.z)
            })
            .collect();
        eprintln!("row {r}: {}", pts.join(" "));
    }
}

/// Diagnostic (wave 10, the bump in the shoulder's outline in the three-quarter portrait): the
/// top outline of the left shoulder as the three-quarter camera sees it (azimuth 40 degrees
/// toward +X): per 4 mm of screen x, the highest point and where it lies.
#[test]
#[ignore = "diagnostic"]
fn diag_shoulder_outline() {
    let s = super::body_surface().unwrap();
    let m = s.mesh();
    let a = 0.698_f32;
    let (right, toward) = (Vec3::new(a.cos(), 0.0, -a.sin()), Vec3::new(a.sin(), 0.07, a.cos()).normalize());
    let up = toward.cross(right).normalize();
    let mut bins: Vec<Option<(f32, Vec3)>> = vec![None; 60];
    for &p in &m.pos {
        if p.y < 1.25 || p.y > 1.50 || p.x < 0.0 {
            continue;
        }
        let x = p.dot(right);
        let k = ((x - 0.02) / 0.004).floor();
        if !(0.0..60.0).contains(&k) {
            continue;
        }
        let y = p.dot(up);
        let slot = &mut bins[k as usize];
        if slot.is_none_or(|(best, _)| y > best) {
            *slot = Some((y, p));
        }
    }
    for (k, b) in bins.iter().enumerate() {
        if let Some((y, p)) = b {
            eprintln!("x' {:.3}: top {:.4} at ({:.3} {:.3} {:.3})", 0.02 + 0.004 * k as f32, y, p.x, p.y, p.z);
        }
    }
}

/// Diagnostic (wave 10): faces whose UV area `Mesh::face_uv_area` calls mirrored or collapsed,
/// where they are and their area recomputed in f64 relative to the first corner.
#[test]
#[ignore = "diagnostic"]
fn diag_uv_flipped_faces() {
    let options = super::build::Options { lods: false, ..super::build::Options::default() };
    let generated = super::build::template(&options).unwrap();
    let mesh = generated.template.mesh();
    let (pos, uvs) = (mesh.positions(), mesh.uvs());
    let mut n = 0;
    for f in 0..mesh.face_count() {
        let area = mesh.face_uv_area(f);
        if area > 1e-12 {
            continue;
        }
        let c = mesh.face(f);
        let o = uvs[c[0] as usize];
        let mut twice = 0.0_f64;
        for i in 0..c.len() {
            let (a, b) = (uvs[c[i] as usize] - o, uvs[c[(i + 1) % c.len()] as usize] - o);
            twice += f64::from(a.x) * f64::from(b.y) - f64::from(b.x) * f64::from(a.y);
        }
        let centre = c.iter().fold(Vec3::ZERO, |s, &v| s + pos[v as usize]) / c.len() as f32;
        n += 1;
        if n <= 30 {
            eprintln!(
                "face {f} tile {} at ({:.4} {:.4} {:.4}) uv {:?} area f32 {area:.3e} f64-relative {:.3e}",
                mesh.face_tiles()[f],
                centre.x,
                centre.y,
                centre.z,
                o,
                -0.5 * twice
            );
        }
    }
    eprintln!("{n} faces with UV area <= 1e-12");
}
