//! The template's landmarks `[MOD-06]`: 173 face and 68 body points, named per Appendix C.
//!
//! Every landmark is found on the assembled mesh by one of four rules, so it is stored as a
//! barycentric point of one face and follows every deformation:
//!
//! - **loop vertex**: a vertex of a cage loop (lid margin, vermilion border, helix rim). Catmull-Clark
//!   keeps cage vertex indices, and the fit puts the surface through the designed point;
//! - **nearest vertex** to a designed point of the face fills (subnasale, alar base);
//! - **ray**: the first hit of a ray, either from in front of the face along `-Z` (brows, midline
//!   profile) or from a joint centre outward (elbow epicondyles, malleoli, patella);
//! - **extreme**: the vertex of a region set that lies farthest in a direction (head vertex, heel,
//!   fingertips, lobule, zygion).
//!
//! `data/template/landmarks.json` is the documented list written from [`json`]; a test keeps the
//! two equal. Joint centres lie inside the body, so the body set names surface points at the joints
//! (`L.knee.center` is the front of the knee over the joint centre).

use std::collections::HashMap;
use std::f32::consts::{PI, TAU};
use std::fmt::Write as _;
use std::ops::Range;

use humanart_math::Vec3;

use super::Surface;
use super::anatomy::{Anatomy, CERVICALE_HEIGHT, FINGERS, SUPRASTERNALE_HEIGHT, sided};
use super::assemble::Assembled;
use super::body::head_point;
use super::limbs::{finger_joints, thumb_joints};
use super::regions::R;
use crate::error::{ModelError, Result};
use crate::landmark::{Landmark, LandmarkGroup};
use crate::mesh::Mesh;

/// A landmark with the sentence that documents it.
#[derive(Clone, Debug, PartialEq)]
pub struct Described {
    /// The landmark.
    pub landmark: Landmark,
    /// What it marks.
    pub description: String,
}

/// A point found on the mesh: three corners of one face and their weights.
type Hit = ([u32; 3], [f32; 3]);

/// Queries on the assembled mesh.
struct Finder<'a> {
    mesh: &'a Mesh,
    regions: &'a [u16],
    /// The mesh vertex of (piece, piece vertex): its first copy.
    first: HashMap<(u32, u32), u32>,
    /// One face of each vertex (`u32::MAX` for none).
    face_of: Vec<u32>,
}

impl<'a> Finder<'a> {
    fn new(asm: &'a Assembled) -> Self {
        let mesh = &asm.mesh;
        let mut first = HashMap::with_capacity(asm.source.len());
        for (m, &s) in asm.source.iter().enumerate() {
            first.entry(s).or_insert(m as u32);
        }
        let mut face_of = vec![u32::MAX; mesh.vertex_count()];
        for f in 0..mesh.face_count() {
            for &v in mesh.face(f) {
                if let Some(slot) = face_of.get_mut(v as usize).filter(|s| **s == u32::MAX) {
                    *slot = f as u32;
                }
            }
        }
        Self { mesh, regions: &asm.regions.vertex_regions, first, face_of }
    }

    fn part(&self, name: &str) -> Result<Range<usize>> {
        self.mesh.parts().iter().find(|p| p.name == name).map(crate::mesh::MeshPart::faces).ok_or(ModelError::invalid(
            "landmark part",
            0,
            "the mesh has no part of that name",
        ))
    }

    /// The landmark exactly at mesh vertex `m`: `m` and the next two corners of one of its faces.
    fn at_vertex(&self, m: u32) -> Result<Hit> {
        let f = self.face_of.get(m as usize).copied().unwrap_or(u32::MAX);
        if f == u32::MAX {
            return Err(ModelError::invalid("landmark vertex", m as usize, "vertex has no face"));
        }
        let c = self.mesh.face(f as usize);
        let k = c.iter().position(|&v| v == m).unwrap_or(0);
        let n = c.len();
        Ok(([c[k], c[(k + 1) % n], c[(k + 2) % n]], [1.0, 0.0, 0.0]))
    }

    /// The landmark at vertex `v` of piece `piece` (0 = body: cage vertex indices).
    fn piece_vertex(&self, piece: u32, v: u32) -> Result<Hit> {
        let m = self.first.get(&(piece, v)).copied();
        m.map_or(Err(ModelError::invalid("landmark vertex", v as usize, "piece vertex is not in the mesh")), |m| {
            self.at_vertex(m)
        })
    }

    /// Vertices of a part (with repeats).
    fn part_vertices(&self, part: &Range<usize>) -> impl Iterator<Item = u32> + '_ {
        part.clone().flat_map(|f| self.mesh.face(f).iter().copied())
    }

    /// The vertex of `part` nearest to `p`.
    fn nearest(&self, part: &str, p: Vec3) -> Result<Hit> {
        let range = self.part(part)?;
        let pos = self.mesh.positions();
        let best = self
            .part_vertices(&range)
            .min_by(|&a, &b| pos[a as usize].distance_squared(p).total_cmp(&pos[b as usize].distance_squared(p)));
        best.map_or(Err(ModelError::invalid("landmark", 0, "part has no vertices")), |m| self.at_vertex(m))
    }

    /// The vertex of `part` whose region is in `regions` and that passes `keep`, farthest along
    /// `dir`.
    fn extreme(&self, part: &str, regions: &[R], keep: &dyn Fn(Vec3) -> bool, dir: Vec3) -> Result<Hit> {
        let range = self.part(part)?;
        let pos = self.mesh.positions();
        let best = self
            .part_vertices(&range)
            .filter(|&v| {
                let r = self.regions.get(v as usize).copied().unwrap_or(u16::MAX);
                (regions.is_empty() || regions.iter().any(|x| x.id() == r)) && keep(pos[v as usize])
            })
            .max_by(|&a, &b| pos[a as usize].dot(dir).total_cmp(&pos[b as usize].dot(dir)).then(b.cmp(&a)));
        best.map_or(Err(ModelError::invalid("landmark", 0, "no vertex passes the extreme filter")), |m| {
            self.at_vertex(m)
        })
    }

    /// The first hit of the ray `origin + t dir` (t > 0) on `part`, with its distance.
    fn ray_t(&self, part: &str, origin: Vec3, dir: Vec3) -> Result<Option<(Hit, f32)>> {
        let range = self.part(part)?;
        let pos = self.mesh.positions();
        let mut best: Option<(Hit, f32)> = None;
        for f in range {
            let c = self.mesh.face(f);
            for i in 1..c.len().saturating_sub(1) {
                let tri = [c[0], c[i], c[i + 1]];
                let [p0, p1, p2] = tri.map(|v| pos[v as usize]);
                let (e1, e2) = (p1 - p0, p2 - p0);
                let h = dir.cross(e2);
                let det = e1.dot(h);
                if det.abs() < 1e-14 {
                    continue;
                }
                let s = origin - p0;
                let u = s.dot(h) / det;
                if !(0.0..=1.0).contains(&u) {
                    continue;
                }
                let q = s.cross(e1);
                let w = dir.dot(q) / det;
                if w < 0.0 || u + w > 1.0 {
                    continue;
                }
                let t = e2.dot(q) / det;
                if t > 1e-6 && best.as_ref().is_none_or(|b| t < b.1) {
                    best = Some(((tri, [1.0 - u - w, u, w]), t));
                }
            }
        }
        Ok(best)
    }

    /// The first hit of a ray on `part`; an error when the ray misses.
    fn ray(&self, part: &str, origin: Vec3, dir: Vec3) -> Result<Hit> {
        self.ray_t(part, origin, dir.normalize_or_zero())?.map(|h| h.0).ok_or(ModelError::invalid(
            "landmark ray",
            0,
            "the ray misses the part",
        ))
    }

    fn position(&self, h: &Hit) -> Vec3 {
        let pos = self.mesh.positions();
        (0..3).fold(Vec3::ZERO, |s, k| s + pos[h.0[k] as usize] * h.1[k])
    }
}

/// The landmark list under construction.
struct List {
    out: Vec<Described>,
}

impl List {
    fn push(&mut self, name: String, group: LandmarkGroup, description: impl Into<String>, hit: Hit) {
        let (vertices, weights) = hit;
        self.out
            .push(Described { landmark: Landmark { name, group, vertices, weights }, description: description.into() });
    }
}

/// `L` or `R` for a side sign (`> 0` is left, `+X`).
fn side_letter(side: f32) -> &'static str {
    if side > 0.0 { "L" } else { "R" }
}

/// A direction of the left side mirrored to `side`.
fn sided_dir(d: Vec3, side: f32) -> Vec3 {
    Vec3::new(d.x * side.signum(), d.y, d.z)
}

/// The face of the head: rays from in front along `-Z`.
fn front_ray(f: &Finder<'_>, a: &Anatomy, x: f32, y: f32) -> Result<Hit> {
    f.ray("body", head_point(a, x, y, 300.0), -Vec3::Z)
}

/// Midline profile: the hit at height `y` (head mm) and its depth (head mm).
fn profile(f: &Finder<'_>, a: &Anatomy, y: f32) -> Result<(Hit, f32)> {
    let h = front_ray(f, a, 0.0, y)?;
    let z = (f.position(&h).z - a.head_origin.z) * 1000.0;
    Ok((h, z))
}

/// The profile point of greatest (`front`) or least depth for heights in `lo..=hi` (0.5 mm steps).
fn profile_extreme(f: &Finder<'_>, a: &Anatomy, lo: f32, hi: f32, front: bool) -> Result<(Hit, f32)> {
    let steps = ((hi - lo) / 0.5).round().max(0.0) as usize;
    let mut best: Option<(Hit, f32, f32)> = None;
    for k in 0..=steps {
        let y = lo + k as f32 * 0.5;
        let (h, z) = profile(f, a, y)?;
        let better = best.as_ref().is_none_or(|b| if front { z > b.1 } else { z < b.1 });
        if better {
            best = Some((h, z, y));
        }
    }
    best.map(|b| (b.0, b.2)).ok_or(ModelError::invalid("landmark profile", 0, "empty profile range"))
}

/// Head-frame millimetres of a world point.
fn head_mm(a: &Anatomy, p: Vec3) -> Vec3 {
    (p - a.head_origin) * 1000.0
}

/// Ring slots of the 16-slot eye rings in medial-to-lateral order: (upper, lower) slot indices.
fn eye_slots(params: &[f32]) -> (Vec<usize>, Vec<usize>, Option<usize>, Option<usize>) {
    let t = |i: usize| params[i].rem_euclid(TAU);
    let near = |i: usize, x: f32| ((t(i) - x + PI).rem_euclid(TAU) - PI).abs() < 1e-3;
    let lateral = (0..params.len()).find(|&i| near(i, 0.0));
    let medial = (0..params.len()).find(|&i| near(i, PI));
    let mut upper: Vec<usize> = (0..params.len()).filter(|&i| t(i) > 1e-3 && t(i) < PI - 1e-3).collect();
    upper.sort_by(|&i, &j| t(j).total_cmp(&t(i)));
    let mut lower: Vec<usize> = (0..params.len()).filter(|&i| t(i) > PI + 1e-3 && t(i) < TAU - 1e-3).collect();
    lower.sort_by(|&i, &j| t(i).total_cmp(&t(j)));
    (upper, lower, medial, lateral)
}

fn face_midline(f: &Finder<'_>, s: &Surface, list: &mut List) -> Result<()> {
    let a = &s.anatomy;
    let g = LandmarkGroup::Face;
    let head_y = a.head_origin.y;
    let mid = |p: Vec3| p.x.abs() < 0.005;
    list.push(
        "head.vertex".into(),
        g,
        "Vertex: the highest point of the head on the midline.",
        f.extreme("body", &[], &mid, Vec3::Y)?,
    );
    list.push(
        "head.opisthocranion".into(),
        g,
        "Opisthocranion: the rearmost point of the skull on the midline.",
        f.extreme("body", &[R::Scalp, R::Hairline], &|p| mid(p) && p.y > head_y, -Vec3::Z)?,
    );
    let (glabella, gy) = profile_extreme(f, a, 15.0, 40.0, true)?;
    list.push("forehead.glabella".into(), g, "Glabella: the most prominent point between the brows.", glabella);
    let (nasion, ny) = profile_extreme(f, a, 0.0, gy, false)?;
    list.push("nose.root".into(), g, "Nasion (sellion): the deepest point of the nasal root on the midline.", nasion);
    let tip = s.body.loops.nose.tip;
    let tip_hit = f.piece_vertex(0, tip)?;
    let tip_y = head_mm(a, f.position(&tip_hit)).y;
    for (k, w) in [(1, 1.0 / 3.0), (2, 2.0 / 3.0)] {
        let y = ny + (tip_y - ny) * w;
        list.push(
            format!("nose.bridge.{k:02}"),
            g,
            format!("Nasal dorsum at {k}/3 of the way from the root to the tip."),
            profile(f, a, y)?.0,
        );
    }
    list.push("nose.tip".into(), g, "Pronasale: the tip of the nose.", tip_hit);
    let subnasale = f.nearest("body", head_point(a, 0.0, -25.5, 104.0))?;
    let sn_y = head_mm(a, f.position(&subnasale)).y;
    list.push("nose.subnasale".into(), g, "Subnasale: where the columella meets the upper lip.", subnasale);
    let mouth = &s.body.loops.mouth;
    let ls = mouth
        .slots
        .iter()
        .position(|&(x, up)| up && x.abs() < 1e-3)
        .and_then(|i| mouth.rings.get(2).and_then(|r| r.get(i)).copied())
        .ok_or(ModelError::invalid("landmark", 0, "no upper vermilion midline slot"))?;
    let ls_y = head_mm(a, s.mesh().pos[ls as usize]).y;
    let philtrum_y = f32::midpoint(sn_y, ls_y);
    list.push("philtrum.mid".into(), g, "The middle of the philtrum groove.", profile(f, a, philtrum_y)?.0);
    for side in [1.0_f32, -1.0] {
        list.push(
            format!("{}.philtrum.column", side_letter(side)),
            g,
            "Philtrum column half way between the subnasale and the vermilion border.",
            front_ray(f, a, side * 5.0, philtrum_y)?,
        );
    }
    let li = mouth
        .slots
        .iter()
        .position(|&(x, up)| !up && x.abs() < 1e-3)
        .and_then(|i| mouth.rings.get(2).and_then(|r| r.get(i)).copied())
        .ok_or(ModelError::invalid("landmark", 0, "no lower vermilion midline slot"))?;
    let li_y = head_mm(a, s.mesh().pos[li as usize]).y;
    let (supramentale, sm_y) = profile_extreme(f, a, li_y - 22.0, li_y - 3.0, false)?;
    list.push("chin.supramentale".into(), g, "Supramentale: the deepest point of the labiomental fold.", supramentale);
    let (pogonion, _) = profile_extreme(f, a, sm_y - 30.0, sm_y - 3.0, true)?;
    list.push("chin.mid".into(), g, "Pogonion: the most anterior point of the chin.", pogonion);
    list.push(
        "chin.menton".into(),
        g,
        "Menton: the lowest point of the chin on the midline.",
        f.extreme("body", &[R::Chin], &|p| p.x.abs() < 0.004, -Vec3::Y)?,
    );
    Ok(())
}

fn face_sided(f: &Finder<'_>, s: &Surface, list: &mut List, side: f32) -> Result<()> {
    let a = &s.anatomy;
    let g = LandmarkGroup::Face;
    let l = side_letter(side);
    let i = usize::from(side < 0.0);
    let to_head = |p: Vec3| head_mm(a, p);
    let ec = to_head(a.eye_center);
    // Brows: medial to lateral along the brow arch.
    for (k, (dx, dy)) in [(-16.0, 20.0), (-7.0, 23.0), (2.0, 24.0), (11.0, 23.0), (20.0, 20.0)].iter().enumerate() {
        list.push(
            format!("{l}.brow.{:02}", k + 1),
            g,
            format!("Brow contour point {} of 5, medial to lateral.", k + 1),
            front_ray(f, a, side * (ec.x + dx), ec.y + dy)?,
        );
    }
    // Eyelids from the eye loops.
    let eye = &s.body.loops.eyes[i];
    let (upper, lower, medial, lateral) = eye_slots(&eye.params);
    let margin = eye.rings.get(2).ok_or(ModelError::invalid("landmark", 2, "eye has no lid margin ring"))?;
    let crease = eye.rings.get(1).ok_or(ModelError::invalid("landmark", 1, "eye has no crease ring"))?;
    let slot = |ring: &[u32], k: usize| {
        ring.get(k).copied().ok_or(ModelError::invalid("landmark", k, "eye slot out of range"))
    };
    for (k, &j) in upper.iter().enumerate() {
        list.push(
            format!("{l}.eye.upper.{:02}", k + 1),
            g,
            "Upper lid margin at the lash line, medial to lateral.",
            f.piece_vertex(0, slot(margin, j)?)?,
        );
    }
    for (k, &j) in lower.iter().enumerate() {
        list.push(
            format!("{l}.eye.lower.{:02}", k + 1),
            g,
            "Lower lid margin at the lash line, medial to lateral.",
            f.piece_vertex(0, slot(margin, j)?)?,
        );
    }
    for (k, &j) in upper.iter().enumerate() {
        list.push(
            format!("{l}.eye.crease.{:02}", k + 1),
            g,
            "Upper lid crease, medial to lateral.",
            f.piece_vertex(0, slot(crease, j)?)?,
        );
    }
    let (Some(medial), Some(lateral)) = (medial, lateral) else {
        return Err(ModelError::invalid("landmark", i, "eye ring has no canthus slots"));
    };
    list.push(
        format!("{l}.eye.canthus.medial"),
        g,
        "Endocanthion: the inner corner of the eye.",
        f.piece_vertex(0, slot(margin, medial)?)?,
    );
    list.push(
        format!("{l}.eye.canthus.lateral"),
        g,
        "Exocanthion: the outer corner of the eye.",
        f.piece_vertex(0, slot(margin, lateral)?)?,
    );
    let eyeball = if side > 0.0 { "eyeball-L" } else { "eyeball-R" };
    list.push(
        format!("{l}.iris.center"),
        g,
        "Centre of the iris on the cornea (gaze straight ahead).",
        f.ray(eyeball, sided(a.eye_center, side), Vec3::Z)?,
    );
    // Nose.
    let hp = |x: f32, y: f32, z: f32| head_point(a, side * x, y, z);
    list.push(
        format!("{l}.nose.alare"),
        g,
        "Alare: the most lateral point of the ala.",
        f.nearest("body", hp(16.0, -18.5, 101.0))?,
    );
    list.push(
        format!("{l}.nose.alar-crease"),
        g,
        "Alar curvature point in the alar-facial groove.",
        f.nearest("body", hp(17.5, -15.5, 96.5))?,
    );
    list.push(
        format!("{l}.nose.alar-base"),
        g,
        "Subalare: where the alar base meets the upper lip.",
        f.nearest("body", hp(11.5, -26.0, 99.0))?,
    );
    // Cheek.
    let zy = |p: Vec3| {
        let h = to_head(p);
        h.x * side > 0.0 && (-15.0..10.0).contains(&h.y) && (5.0..75.0).contains(&h.z)
    };
    let cheek_regions = [
        R::sided(R::CheekL, R::CheekR, side),
        R::sided(R::MalarL, R::MalarR, side),
        R::sided(R::TempleL, R::TempleR, side),
    ];
    list.push(
        format!("{l}.zygion"),
        g,
        "Zygion: the most lateral point of the zygomatic arch.",
        f.extreme("body", &cheek_regions, &zy, Vec3::X * side)?,
    );
    list.push(
        format!("{l}.cheek.malar"),
        g,
        "Malar prominence below the outer third of the eye.",
        front_ray(f, a, side * (ec.x + 8.0), ec.y - 22.0)?,
    );
    // Jaw line: the lowest face point in sagittal slabs from the gonion forward.
    let jaw_regions = [R::sided(R::JawL, R::JawR, side), R::sided(R::CheekL, R::CheekR, side), R::Chin];
    for (k, x) in [50.0_f32, 44.0, 36.0, 26.0, 14.0].iter().enumerate() {
        let slab = |p: Vec3| {
            let h = to_head(p);
            (h.x * side - x).abs() < 2.0 && h.z > -10.0
        };
        let what = if k == 0 {
            "Gonion: the angle of the jaw.".to_owned()
        } else {
            format!("Jaw line point {} of 5, from the gonion toward the chin.", k + 1)
        };
        list.push(format!("jaw.{l}.{:02}", k + 1), g, what, f.extreme("body", &jaw_regions, &slab, -Vec3::Y)?);
    }
    // Lips: vermilion border (outer) and lip contact (inner), right to left.
    let mouth = &s.body.loops.mouth;
    if side > 0.0 {
        for (ring, which) in [(2_usize, "outer"), (4, "inner")] {
            let r = mouth.rings.get(ring).ok_or(ModelError::invalid("landmark", ring, "mouth ring missing"))?;
            for up in [true, false] {
                let mut slots: Vec<usize> = (0..mouth.slots.len())
                    .filter(|&j| mouth.slots[j].1 == up && mouth.slots[j].0.abs() < 24.99)
                    .collect();
                slots.sort_by(|&p, &q| mouth.slots[p].0.total_cmp(&mouth.slots[q].0));
                let lip = if up { "upper" } else { "lower" };
                let line = if ring == 2 { "vermilion border" } else { "lip contact line" };
                for (k, &j) in slots.iter().enumerate() {
                    list.push(
                        format!("lip.{lip}.{which}.{:02}", k + 1),
                        g,
                        format!(
                            "{} lip {line}, point {} of {}, right to left.",
                            if up { "Upper" } else { "Lower" },
                            k + 1,
                            slots.len()
                        ),
                        f.piece_vertex(0, slot(r, j)?)?,
                    );
                }
            }
        }
    }
    let corner = mouth.slots.iter().position(|&(x, _)| x * side > 24.99).ok_or(ModelError::invalid(
        "landmark",
        i,
        "mouth has no corner slot",
    ))?;
    for (ring, name, what) in [
        (2_usize, "lip.corner", "Cheilion: the corner of the mouth at the vermilion border."),
        (4, "lip.commissure", "The inner corner of the lips at the contact line."),
    ] {
        let r = mouth.rings.get(ring).ok_or(ModelError::invalid("landmark", ring, "mouth ring missing"))?;
        list.push(format!("{l}.{name}"), g, what, f.piece_vertex(0, slot(r, corner)?)?);
    }
    // Ear.
    let ear = &s.body.loops.ears[i];
    let rim = ear.rings.get(3).ok_or(ModelError::invalid("landmark", 3, "ear has no helix rim ring"))?;
    let mut order: Vec<usize> = (0..rim.len().min(ear.params.len())).collect();
    order.sort_by(|&p, &q| ear.params[p].rem_euclid(TAU).total_cmp(&ear.params[q].rem_euclid(TAU)));
    // The finished auricle's rim has twice the block's slots: slot `j` of the block is `2 j`.
    let stride = (rim.len() / ear.params.len().max(1)).max(1);
    for (k, &j) in order.iter().enumerate() {
        list.push(
            format!("ear.{l}.helix.{:02}", k + 1),
            g,
            "Helix rim point, from the back over the top to the front and the lobe.",
            f.piece_vertex(0, slot(rim, j * stride)?)?,
        );
    }
    list.push(format!("ear.{l}.concha"), g, "The floor of the concha.", f.piece_vertex(0, ear.concha)?);
    let ear_regions = if side > 0.0 {
        [R::EarLHelix, R::EarLAntihelix, R::EarLConcha, R::EarLLobe, R::EarLBack]
    } else {
        [R::EarRHelix, R::EarRAntihelix, R::EarRConcha, R::EarRLobe, R::EarRBack]
    };
    list.push(
        format!("ear.{l}.lobule"),
        g,
        "Subaurale: the lowest point of the ear lobe.",
        f.extreme("body", &ear_regions[3..4], &|_| true, Vec3::new(0.0, -1.0, 0.25))?,
    );
    list.push(
        format!("ear.{l}.superaurale"),
        g,
        "Superaurale: the highest point of the auricle.",
        f.extreme("body", &ear_regions, &|_| true, Vec3::Y)?,
    );
    list.push(
        format!("ear.{l}.tragion"),
        g,
        "Tragion: the ear opening seen from the side, on the head frame's ear axis.",
        f.ray("body", a.head_origin, Vec3::X * side)?,
    );
    Ok(())
}

fn hairline(f: &Finder<'_>, a: &Anatomy, list: &mut List) -> Result<()> {
    for k in 0..9_u8 {
        let phi = (-72.0 + 18.0 * f32::from(k)).to_radians();
        let keep = |p: Vec3| {
            let h = head_mm(a, p);
            h.z > 0.0 && (h.x.atan2(h.z) - phi).abs() < 4.0_f32.to_radians()
        };
        list.push(
            format!("hairline.{:02}", k + 1),
            LandmarkGroup::Face,
            format!("Frontal hairline point {} of 9, from the right temple over the forehead to the left.", k + 1),
            f.extreme("body", &[R::Hairline, R::Scalp], &keep, -Vec3::Y)?,
        );
    }
    Ok(())
}

fn body_midline(f: &Finder<'_>, a: &Anatomy, list: &mut List) -> Result<()> {
    let g = LandmarkGroup::Body;
    let rays: [(&str, &str, Vec3, Vec3); 10] = [
        ("neck.front", "Front of the neck at the larynx.", a.neck[1], Vec3::Z),
        ("neck.back", "Back of the neck at mid height.", a.neck[1], -Vec3::Z),
        (
            "spine.c7",
            "Cervicale: the spinous process of C7 at the base of the neck.",
            Vec3::new(0.0, CERVICALE_HEIGHT, a.neck[0].z),
            -Vec3::Z,
        ),
        (
            "torso.suprasternal",
            "Suprasternale: the jugular notch above the sternum.",
            Vec3::new(0.0, SUPRASTERNALE_HEIGHT, -0.03),
            Vec3::Z,
        ),
        ("torso.xiphoid", "The lower end of the sternum.", Vec3::new(0.0, 1.225, -0.03), Vec3::Z),
        ("torso.navel", "Omphalion: the navel.", Vec3::new(0.0, 1.018, -0.03), Vec3::Z),
        ("torso.pubis", "The front of the pubic symphysis.", Vec3::new(0.0, 0.885, -0.02), Vec3::Z),
        ("spine.t8", "The back over the eighth thoracic vertebra.", a.spine[2], -Vec3::Z),
        ("spine.l3", "The back over the third lumbar vertebra.", a.spine[0], -Vec3::Z),
        ("spine.sacrum", "The back over the sacrum.", a.pelvis, -Vec3::Z),
    ];
    for (name, what, origin, dir) in rays {
        list.push(name.into(), g, what, f.ray("body", origin, dir)?);
    }
    Ok(())
}

fn body_sided(f: &Finder<'_>, a: &Anatomy, list: &mut List, side: f32) -> Result<()> {
    let g = LandmarkGroup::Body;
    let l = side_letter(side);
    let u = a.upper_arm_dir;
    let lateral = Vec3::Z.cross(u).normalize_or_zero();
    let back = (-Vec3::Z + u * u.z).normalize_or_zero();
    let [distal, palmar, radial] = a.hand_axes;
    // (name, description, origin, direction), all given for the left side.
    let mut rays: Vec<(String, &str, Vec3, Vec3)> = vec![
        (
            "shoulder.acromion".into(),
            "Acromion: the lateral point of the acromion, at the top corner of the shoulder.",
            a.shoulder,
            Vec3::new(0.8, 1.0, 0.0),
        ),
        ("clavicle.medial".into(), "Over the sternoclavicular joint.", a.clavicle - Vec3::Z * 0.02, Vec3::Z),
        ("chest.nipple".into(), "Thelion: the tip of the nipple.", Vec3::new(0.095, 1.233, 0.0), Vec3::Z),
        ("hip.asis".into(), "Anterior superior iliac spine.", Vec3::new(0.11, 0.97, -0.02), Vec3::Z),
        ("hip.trochanter".into(), "Over the greater trochanter.", a.hip, Vec3::X),
        ("hip.psis".into(), "Posterior superior iliac spine.", Vec3::new(0.045, 0.99, -0.02), -Vec3::Z),
        ("elbow.lateral".into(), "Lateral epicondyle of the humerus.", a.elbow, lateral),
        ("elbow.medial".into(), "Medial epicondyle of the humerus.", a.elbow, -lateral),
        ("elbow.olecranon".into(), "Olecranon: the point of the elbow.", a.elbow, back),
        ("wrist.radial".into(), "Radial styloid.", a.wrist, radial),
        ("wrist.ulnar".into(), "Ulnar styloid.", a.wrist, -radial),
        ("wrist.dorsal".into(), "The back of the wrist over the joint centre.", a.wrist, -palmar),
        ("knee.center".into(), "The front of the knee over the joint centre (patella).", a.knee, Vec3::Z),
        ("knee.lateral".into(), "Lateral side of the knee at the joint line.", a.knee, Vec3::X),
        ("knee.medial".into(), "Medial side of the knee at the joint line.", a.knee, -Vec3::X),
        ("knee.back".into(), "The back of the knee (popliteal fossa).", a.knee, -Vec3::Z),
        ("ankle.lateral".into(), "Lateral malleolus.", a.ankle, Vec3::X),
        ("ankle.medial".into(), "Medial malleolus.", a.ankle, -Vec3::X),
    ];
    for &(finger, mcp, _) in [&FINGERS[0], &FINGERS[3]] {
        rays.push((
            format!("hand.knuckle.{finger}"),
            "The knuckle over the metacarpophalangeal joint.",
            a.hand_point(mcp),
            -palmar,
        ));
    }
    for (name, what, origin, dir) in rays {
        list.push(format!("{l}.{name}"), g, what, f.ray("body", sided(origin, side), sided_dir(dir, side))?);
    }
    let hand_regions: [(&str, R, R, Vec3); 5] = [
        ("thumb", R::HandLThumb, R::HandRThumb, (distal + radial).normalize_or_zero()),
        ("index", R::HandLIndex, R::HandRIndex, distal),
        ("middle", R::HandLMiddle, R::HandRMiddle, distal),
        ("ring", R::HandLRing, R::HandRRing, distal),
        ("little", R::HandLLittle, R::HandRLittle, distal),
    ];
    // The nail plates (todo 1.3) are their own regions: the tip is searched on the digit and its
    // nail, within 12 mm of the digit's end joint.
    let nail = [R::sided(R::HandLNail, R::HandRNail, side), R::sided(R::HandLLunula, R::HandRLunula, side)];
    for (k, (finger, left, right, dir)) in hand_regions.into_iter().enumerate() {
        let chain = if k == 0 { thumb_joints(a, side) } else { finger_joints(a, k - 1, side) };
        let end = chain.last().copied().unwrap_or(Vec3::ZERO);
        list.push(
            format!("{l}.hand.{finger}.tip"),
            g,
            "Fingertip: the farthest point of the digit.",
            f.extreme(
                "body",
                &[R::sided(left, right, side), nail[0], nail[1]],
                &|p| p.distance(end) < 0.012,
                sided_dir(dir, side),
            )?,
        );
    }
    let ankle_y = a.ankle.y;
    let foot = if side > 0.0 {
        [R::FootL, R::FootLDorsum, R::FootLSole, R::FootLArch, R::FootLToes, R::FootLNail]
    } else {
        [R::FootR, R::FootRDorsum, R::FootRSole, R::FootRArch, R::FootRToes, R::FootRNail]
    };
    let fd = sided_dir(a.foot_dir, side);
    // The metatarsal heads: the widest points of the forefoot (behind the toes).
    let ankle = sided(a.ankle, side);
    let forefoot = |p: Vec3| p.y < 0.04 && (0.09..0.17).contains(&(p - ankle).dot(fd));
    let medial = sided_dir(a.foot_dir.cross(Vec3::Y), side);
    list.push(
        format!("{l}.foot.mtp1"),
        g,
        "The head of the first metatarsal: the medial edge of the ball of the foot.",
        f.extreme("body", &foot[..4], &forefoot, medial - Vec3::Y * 0.5)?,
    );
    list.push(
        format!("{l}.foot.mtp5"),
        g,
        "The head of the fifth metatarsal: the lateral edge of the forefoot.",
        f.extreme("body", &foot[..4], &forefoot, -medial - Vec3::Y * 0.5)?,
    );
    list.push(
        format!("{l}.heel"),
        g,
        "Pternion: the rearmost point of the heel.",
        f.extreme("body", &foot, &|p| p.y < ankle_y, -fd)?,
    );
    list.push(
        format!("{l}.foot.tip"),
        g,
        "Acropodion: the tip of the longest toe.",
        f.extreme("body", &foot[4..], &|_| true, fd)?,
    );
    Ok(())
}

/// Finds every landmark on the assembled mesh `[MOD-06]`.
pub fn build(surface: &Surface, asm: &Assembled) -> Result<Vec<Described>> {
    let f = Finder::new(asm);
    let mut list = List { out: Vec::with_capacity(256) };
    face_midline(&f, surface, &mut list)?;
    for side in [1.0, -1.0] {
        face_sided(&f, surface, &mut list, side)?;
    }
    hairline(&f, &surface.anatomy, &mut list)?;
    body_midline(&f, &surface.anatomy, &mut list)?;
    for side in [1.0, -1.0] {
        body_sided(&f, &surface.anatomy, &mut list, side)?;
    }
    let names: Vec<&str> = list.out.iter().map(|d| d.landmark.name.as_str()).collect();
    crate::names::check_names("landmark", names.into_iter())?;
    Ok(list.out)
}

/// Finds the midline profile landmarks again on the finished head (todo 1.3).
///
/// [`build`] runs on the procedural head, which has no brow prominence and whose chin region
/// reaches far down the underside: there `forehead.glabella` and `nose.root` both end at the
/// limit of their search, `chin.menton` lies 47 mm behind the pogonion and `neck.front` sits on
/// the chin. The scanned face ([`super::scan_face`]) then moves the surface; this function looks
/// for the same points on the moved mesh `asm`, by the same rules where they hold and by these
/// where they did not:
///
/// - `chin.menton`: the midline vertex within 35 mm below and 25 mm behind the pogonion that lies
///   farthest along a direction 60 degrees below the horizontal (the lowest point of the chin's
///   outline; the underside that runs back into the neck does not count);
/// - `neck.front`: a ray along `+Z` at 40 % of the height from the menton down to the
///   suprasternale (the laryngeal prominence lies at C4/C5).
///
/// The names stay; only the faces and weights of `forehead.glabella`, `nose.root`,
/// `nose.bridge.01/02`, `chin.supramentale`, `chin.mid`, `chin.menton` and `neck.front` change.
///
/// # Errors
/// A missing landmark name or a search that finds nothing.
pub fn refit_profile(surface: &Surface, asm: &Assembled, described: &mut [Described]) -> Result<()> {
    let f = Finder::new(asm);
    let a = &surface.anatomy;
    let pos = asm.mesh.positions();
    let at = |described: &[Described], name: &str| -> Result<Vec3> {
        described
            .iter()
            .find(|d| d.landmark.name == name)
            .and_then(|d| d.landmark.position(pos))
            .ok_or(ModelError::invalid("landmark refit", 0, "a profile landmark is missing"))
    };
    let mut found: Vec<(&str, Hit)> = Vec::with_capacity(8);
    let (glabella, gy) = profile_extreme(&f, a, 15.0, 55.0, true)?;
    found.push(("forehead.glabella", glabella));
    let (nasion, ny) = profile_extreme(&f, a, 0.0, gy, false)?;
    found.push(("nose.root", nasion));
    let tip_y = head_mm(a, at(described, "nose.tip")?).y;
    for (name, w) in [("nose.bridge.01", 1.0 / 3.0), ("nose.bridge.02", 2.0 / 3.0)] {
        found.push((name, profile(&f, a, ny + (tip_y - ny) * w)?.0));
    }
    let li_y = head_mm(a, at(described, "lip.lower.outer.06")?).y;
    let (supramentale, sm_y) = profile_extreme(&f, a, li_y - 22.0, li_y - 3.0, false)?;
    found.push(("chin.supramentale", supramentale));
    let (pogonion, _) = profile_extreme(&f, a, sm_y - 30.0, sm_y - 3.0, true)?;
    found.push(("chin.mid", pogonion));
    let pog = f.position(&pogonion);
    let keep = |p: Vec3| p.x.abs() < 0.004 && p.y <= pog.y && p.y > pog.y - 0.035 && p.z > pog.z - 0.025;
    let menton = f.extreme("body", &[], &keep, Vec3::new(0.0, -0.866, 0.5))?;
    let me_y = f.position(&menton).y;
    found.push(("chin.menton", menton));
    let larynx_y = me_y + 0.4 * (SUPRASTERNALE_HEIGHT - me_y);
    found.push(("neck.front", f.ray("body", Vec3::new(0.0, larynx_y, a.neck[1].z), Vec3::Z)?));
    for (name, (vertices, weights)) in found {
        let d = described.iter_mut().find(|d| d.landmark.name == name).ok_or(ModelError::invalid(
            "landmark refit",
            0,
            "a profile landmark is missing",
        ))?;
        d.landmark.vertices = vertices;
        d.landmark.weights = weights;
    }
    Ok(())
}

/// The text of `data/template/landmarks.json`: every landmark with its group, description, face
/// corners, weights and rest position (metres).
pub fn json(landmarks: &[Described], positions: &[Vec3]) -> String {
    let mut s = String::with_capacity(landmarks.len() * 200);
    s.push_str("{\n  \"format\": \"humanart-landmarks\",\n  \"version\": 1,\n");
    s.push_str(
        "  \"generator\": \"humanart-model generate::landmarks (written by the generator's tests; do not edit)\",\n",
    );
    s.push_str("  \"naming\": \"Appendix C: <side>.<feature>.<index>, body <side>.<joint-or-point>\",\n");
    s.push_str("  \"units\": \"metres, Y up, the character faces +Z, left is +X\",\n");
    let count = |g: LandmarkGroup| landmarks.iter().filter(|d| d.landmark.group == g).count();
    let _ = writeln!(
        s,
        "  \"counts\": {{\"face\": {}, \"body\": {}}},",
        count(LandmarkGroup::Face),
        count(LandmarkGroup::Body)
    );
    s.push_str("  \"landmarks\": [\n");
    for (i, d) in landmarks.iter().enumerate() {
        let l = &d.landmark;
        let p = l.position(positions).unwrap_or(Vec3::ZERO);
        let group = match l.group {
            LandmarkGroup::Face => "face",
            LandmarkGroup::Body => "body",
        };
        let _ = write!(
            s,
            "    {{\"name\": \"{}\", \"group\": \"{group}\", \"vertices\": [{}, {}, {}], \"weights\": [{:.6}, {:.6}, {:.6}], \"position\": [{:.5}, {:.5}, {:.5}], \"description\": \"{}\"}}",
            l.name,
            l.vertices[0],
            l.vertices[1],
            l.vertices[2],
            l.weights[0],
            l.weights[1],
            l.weights[2],
            p.x,
            p.y,
            p.z,
            d.description.replace('"', "'"),
        );
        s.push_str(if i + 1 < landmarks.len() { ",\n" } else { "\n" });
    }
    s.push_str("  ]\n}\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mod_06_eye_slots_run_medial_to_lateral_on_both_lids() {
        // 16 slots, as on the left eye (increasing) and the right eye (decreasing parameters).
        for dir in [1.0_f32, -1.0] {
            let params: Vec<f32> = (0..16).map(|i| dir * i as f32 * TAU / 16.0).collect();
            let (upper, lower, medial, lateral) = eye_slots(&params);
            assert_eq!((upper.len(), lower.len()), (7, 7));
            let t = |i: usize| params[i].rem_euclid(TAU);
            assert_eq!(lateral.map(t), Some(0.0));
            assert!((medial.map_or(0.0, t) - PI).abs() < 1e-5);
            assert!(upper.windows(2).all(|w| t(w[0]) > t(w[1])), "upper lid not medial to lateral");
            assert!(lower.windows(2).all(|w| t(w[0]) < t(w[1])), "lower lid not medial to lateral");
        }
    }

    #[test]
    fn mod_06_json_lists_every_landmark() {
        let l = Described {
            landmark: Landmark {
                name: "nose.tip".into(),
                group: LandmarkGroup::Face,
                vertices: [0, 1, 2],
                weights: [0.2, 0.3, 0.5],
            },
            description: "Pronasale \"tip\".".into(),
        };
        let text = json(&[l], &[Vec3::ZERO, Vec3::X, Vec3::Y]);
        let v = humanart_io::json::parse_str(&text).unwrap();
        let first = v.get("landmarks").and_then(|a| a.at(0)).unwrap();
        assert_eq!(first.get("name").and_then(|n| n.as_str()), Some("nose.tip"));
        assert_eq!(v.get("counts").and_then(|c| c.get("face")).and_then(humanart_io::json::Value::as_u64), Some(1));
    }
}
