//! Finishing steps on the head after the scanned face (`[WP-25]`, todo 1.1, round 2): they run only
//! when [`super::build::Options::scan_face`] is on, so the fitting base
//! ([`super::build::BASE_VERSION`]) stays exactly what the scans were fitted to.
//!
//! - [`cranium`]: the procedural vault was a box (a flat top and a hard corner where the forehead
//!   meets the scalp, a wedge in three-quarter view). The skin above the brows and the ears moves
//!   radially onto a skull-shaped surface: an ellipse in the mid-sagittal profile (glabella, forehead
//!   slope, vertex, occiput) and a rounded rectangle across ([`VAULT`]); full on the vault, fading
//!   in over [`VAULT_BAND`] above a line that runs from the brows over the ears to the occiput.
//!   The form follows the adult skull (BodyParts3D's frontal, parietal and occipital bones as a
//!   reference for the profile; no data is taken from it); the sizes are ANSUR II head length,
//!   breadth and tragion-to-vertex height, checked by a test.
//! - [`fair_nose_root`]: the nose block's first ring juts out of the face at the nose root, a shelf
//!   with a fold of 88 degrees across it (the "nose-root seam"). The scanned faces have no shelf
//!   there, so the scan face squeezes the shelf's rows of faces to almost no area: slivers whose
//!   normals are noise (73 degrees from their neighbours). Smoothing along the normal cannot open
//!   them; the vertices inside [`NOSE_ROOT`] first spread out over the surface they lie on, then a
//!   local quadric fit rounds what is left.

use std::collections::HashMap;

use humanart_math::Vec3;

use super::anatomy::Anatomy;
use super::assemble::Assembled;
use crate::error::{ModelError, Result};
use crate::mesh::Mesh;
use crate::scanfit::nearest::TriangleTree;

/// The skull surface in the head frame (millimetres from the tragion midpoint): centre height, the
/// half breadth, the height of the vertex above the centre, the front and back half lengths, and
/// the exponents of the horizontal cross-section behind and in front of the ears (2 is an ellipse,
/// larger is squarer). In front it is nearly an ellipse (round 4): the frontal bone curves on from
/// the forehead into the temples, where a squarer section left a near-vertical edge in
/// three-quarter view.
pub const VAULT: Vault = Vault {
    centre_y: 30.0,
    half_breadth: 77.0,
    height: 101.1,
    front: 103.0,
    back: 96.0,
    exponent: 3.0,
    front_exponent: 2.3,
    side_exponent: 2.9,
};
/// Height of the vault's lower edge above the tragion (millimetres) in front (above the brows), at
/// the side (above the ears) and at the back (the occiput), and the width of the fade above it.
pub const VAULT_EDGE: (f32, f32, f32) = (55.0, 30.0, -15.0);
/// Width of the fade from the procedural head to the vault, millimetres.
pub const VAULT_BAND: f32 = 35.0;
/// The nose root zone: centre and radii (head frame, millimetres); full inside, gone at twice the
/// radii.
pub const NOSE_ROOT: ([f32; 3], [f32; 3]) = ([0.0, 2.0, 105.0], [11.0, 6.0, 9.0]);
/// Rounds of the relaxation over the surface at the nose root. Measured on the template (largest
/// fold across the nose root): 4 rounds 60.7, 8 rounds 23.5, 15 rounds 17.0, 20 rounds 9.0 degrees.
pub const NOSE_ROOT_ROUNDS: usize = 20;
/// Width of the Gaussian window of the quadric fits at the nose root, millimetres.
pub const NOSE_ROOT_WINDOW: f32 = 4.0;
/// Passes of the quadric fit at the nose root.
pub const NOSE_ROOT_PASSES: usize = 2;
/// How far from its relaxed position a vertex looks for the surface, millimetres.
pub const NOSE_ROOT_REACH: f32 = 5.0;

/// A skull-shaped surface (see [`VAULT`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vault {
    /// Height of the centre above the tragion, mm.
    pub centre_y: f32,
    /// Half breadth, mm.
    pub half_breadth: f32,
    /// Height of the vertex above the centre, mm.
    pub height: f32,
    /// Front half length (to the forehead), mm.
    pub front: f32,
    /// Back half length (to the occiput), mm.
    pub back: f32,
    /// Exponent of the horizontal cross-section behind the ears.
    pub exponent: f32,
    /// Exponent of the horizontal cross-section in front of the ears (the two halves meet with the
    /// same radius and tangent at the side, whatever the exponents).
    pub front_exponent: f32,
    /// Exponent of the vertical cross-section across the head (the mid-sagittal profile is an
    /// ellipse, exponent 2; across, the parietal bones keep the breadth up to a broad crown).
    pub side_exponent: f32,
}

impl Vault {
    /// The vault's radius from its centre in the direction `d` (head frame, any length).
    #[must_use]
    pub fn radius(&self, d: Vec3) -> f32 {
        let len = d.length();
        if len <= 0.0 {
            return 0.0;
        }
        let u = d / len;
        let (depth, p) = if u.z >= 0.0 { (self.front, self.front_exponent) } else { (self.back, self.exponent) };
        let horizontal = ((u.x / self.half_breadth).abs().powf(p) + (u.z / depth).abs().powf(p)).powf(2.0 / p);
        let across = u.x * u.x / (u.x * u.x + u.z * u.z).max(1.0e-12);
        let q = 2.0 + (self.side_exponent - 2.0) * across;
        let f = horizontal.powf(q / 2.0) + (u.y / self.height).abs().powf(q);
        if f > 0.0 { f.powf(-1.0 / q) } else { 0.0 }
    }

    /// The vault's centre in the head frame, mm.
    #[must_use]
    pub const fn centre(&self) -> Vec3 {
        Vec3::new(0.0, self.centre_y, 0.0)
    }
}

/// `0` at `x <= 0`, `1` at `x >= 1`, smooth between.
fn smoothstep(x: f32) -> f32 {
    let t = x.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// How much a head-frame point (mm) moves onto the vault: the lower edge runs from
/// [`VAULT_EDGE`]`.0` in front over `.1` at the side to `.2` at the back (smooth in the horizontal
/// angle), and the weight fades in over [`VAULT_BAND`] above it.
#[must_use]
pub fn vault_weight(h: Vec3) -> f32 {
    let r = (h.x * h.x + h.z * h.z).sqrt();
    let c = if r > 0.0 { h.z / r } else { 0.0 };
    // edge(c) = a + b c + e (2 c^2 - 1): front at c = 1, side at c = 0, back at c = -1.
    let (front, side, back) = VAULT_EDGE;
    let b = 0.5 * (front - back);
    let e = front.midpoint(back) - side;
    let a = side + e;
    let edge = a + b * c + e * (2.0 * c * c - 1.0);
    smoothstep((h.y - edge) / VAULT_BAND)
}

/// Moves the vault skin onto [`VAULT`] (module documentation). Returns the largest move, metres.
///
/// # Errors
/// When the moved mesh fails [`Mesh::new`]'s checks.
pub fn cranium(asm: &mut Assembled, anatomy: &Anatomy) -> Result<f32> {
    let vault = VAULT;
    let centre = vault.centre();
    let ears: Vec<u16> = ["ear-L", "ear-R"].iter().filter_map(|n| asm.regions.find(n)).collect();
    let mut data = asm.mesh.clone().into_data();
    let mut largest = 0.0_f32;
    for (v, p) in data.positions.iter_mut().enumerate() {
        if asm.source.get(v).is_none_or(|s| s.0 != 0) {
            continue;
        }
        let region = asm.regions.vertex_regions.get(v).copied().unwrap_or(u16::MAX);
        if ears.iter().any(|&e| asm.regions.is_within(region, e)) {
            continue;
        }
        let h = (*p - anatomy.head_origin) * 1000.0;
        let w = vault_weight(h);
        if w <= 0.0 {
            continue;
        }
        let d = h - centre;
        let target = centre + d.normalize_or_zero() * vault.radius(d);
        let moved = (target - h) * (w * 0.001);
        largest = largest.max(moved.length());
        *p += moved;
    }
    asm.mesh = Mesh::new(data)?;
    Ok(largest)
}

/// The welded skin around the nose root (UV seams duplicate vertices).
struct Patch {
    /// Head-frame positions of the nodes, mm.
    points: Vec<Vec3>,
    /// How much of the relaxation each node takes, `0..=1`.
    weights: Vec<f32>,
    /// The mesh vertices at each node.
    copies: Vec<Vec<u32>>,
    /// The neighbours of each node along face edges (`u32::MAX`: a vertex outside the patch).
    neighbours: Vec<Vec<u32>>,
    /// The body triangles with all three corners in the patch, three node indices each.
    triangles: Vec<u32>,
}

impl Patch {
    /// The body skin within three times `radii` of `centre` (head frame, mm): full weight inside
    /// the ellipsoid, none from twice its radii on, none in the regions of the eyes and none for
    /// a node at the patch's border.
    fn new(asm: &Assembled, anatomy: &Anatomy, centre: [f32; 3], radii: [f32; 3]) -> Self {
        let mesh = &asm.mesh;
        let eyes: Vec<u16> = ["eye-L", "eye-R"].iter().filter_map(|n| asm.regions.find(n)).collect();
        let mut index: HashMap<(i64, i64, i64), u32> = HashMap::new();
        let mut patch = Self {
            points: Vec::new(),
            weights: Vec::new(),
            copies: Vec::new(),
            neighbours: Vec::new(),
            triangles: Vec::new(),
        };
        let mut node_of = vec![u32::MAX; mesh.positions().len()];
        for (v, &p) in mesh.positions().iter().enumerate() {
            if asm.source.get(v).is_none_or(|s| s.0 != 0) {
                continue;
            }
            let h = (p - anatomy.head_origin) * 1000.0;
            let q = Vec3::new((h.x - centre[0]) / radii[0], (h.y - centre[1]) / radii[1], (h.z - centre[2]) / radii[2])
                .length();
            if q >= 3.0 {
                continue;
            }
            let key = ((p.x * 1.0e6).round() as i64, (p.y * 1.0e6).round() as i64, (p.z * 1.0e6).round() as i64);
            let node = *index.entry(key).or_insert_with(|| {
                patch.points.push(h);
                patch.weights.push(1.0 - smoothstep(q - 1.0));
                patch.copies.push(Vec::new());
                patch.neighbours.push(Vec::new());
                (patch.points.len() - 1) as u32
            });
            let region = asm.regions.vertex_regions.get(v).copied().unwrap_or(u16::MAX);
            if eyes.iter().any(|&e| asm.regions.is_within(region, e))
                && let Some(w) = patch.weights.get_mut(node as usize)
            {
                *w = 0.0;
            }
            if let (Some(list), Some(slot)) = (patch.copies.get_mut(node as usize), node_of.get_mut(v)) {
                list.push(v as u32);
                *slot = node;
            }
        }
        if let Some(body) = mesh.parts().iter().find(|p| p.name == "body") {
            for f in body.faces() {
                let nodes: Vec<u32> =
                    mesh.face(f).iter().map(|&v| node_of.get(v as usize).copied().unwrap_or(u32::MAX)).collect();
                for (i, &a) in nodes.iter().enumerate() {
                    let b = nodes.get((i + 1) % nodes.len()).copied().unwrap_or(a);
                    for (x, y) in [(a, b), (b, a)] {
                        if let Some(list) = patch.neighbours.get_mut(x as usize)
                            && x != y
                            && !list.contains(&y)
                        {
                            list.push(y);
                        }
                    }
                }
                if nodes.contains(&u32::MAX) {
                    continue;
                }
                // A fan from the first corner (the faces are triangles and quads).
                for pair in nodes.get(1..).unwrap_or(&[]).windows(2) {
                    if let (Some(&a), [b, c]) = (nodes.first(), pair) {
                        patch.triangles.extend([a, *b, *c]);
                    }
                }
            }
        }
        for (w, list) in patch.weights.iter_mut().zip(&patch.neighbours) {
            if list.is_empty() || list.contains(&u32::MAX) {
                *w = 0.0;
            }
        }
        patch
    }

    /// Per node the sum of the area vectors of its triangles (half the cross product, mm^2): its
    /// direction is the node's normal, its length three times the node's share of the area.
    fn area_normals(&self) -> Vec<Vec3> {
        let mut normals = vec![Vec3::ZERO; self.points.len()];
        for t in self.triangles.as_chunks::<3>().0 {
            let corner = |k: usize| self.points.get(t[k] as usize).copied().unwrap_or(Vec3::ZERO);
            let n = (corner(1) - corner(0)).cross(corner(2) - corner(0)) * 0.5;
            for &k in t {
                if let Some(slot) = normals.get_mut(k as usize) {
                    *slot += n;
                }
            }
        }
        normals
    }

    /// Where node `i` lies on the quadric fitted to the skin around it: along the Gaussian-weighted
    /// (width `window`, mm) mean `n` of the area normals there, at the height of the least-squares
    /// quadric `h = a + b u + c v + d u^2 + e u v + f v^2` over the plane across `n` (samples
    /// weighted by the same Gaussian and their area). `None` where the normal vanishes or the fit
    /// is singular.
    fn on_quadric(&self, i: usize, normals: &[Vec3], window: f32) -> Option<Vec3> {
        let at = self.points.get(i).copied()?;
        let reach = 2.5 * window;
        let gauss = |s: Vec3| {
            let d = s.distance(at);
            if d > reach { 0.0 } else { (-(d * d) / (window * window)).exp() }
        };
        let mean = self.points.iter().zip(normals).fold(Vec3::ZERO, |sum, (&s, &n)| sum + n * gauss(s));
        let n = mean.normalize_or_zero();
        if n == Vec3::ZERO {
            return None;
        }
        // Any two axes across the normal: the fitted height does not depend on their choice.
        let u = n.cross(if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y }).normalize_or_zero();
        let v = n.cross(u);
        let mut m = [[0.0_f64; 6]; 6];
        let mut r = [0.0_f64; 6];
        for (&s, &area) in self.points.iter().zip(normals) {
            let w = f64::from(gauss(s) * area.length());
            if w <= 0.0 {
                continue;
            }
            let d = s - at;
            let (x, y, z) = (f64::from(d.dot(u)), f64::from(d.dot(v)), f64::from(d.dot(n)));
            let basis = [1.0, x, y, x * x, x * y, y * y];
            for (row, &bi) in m.iter_mut().zip(&basis) {
                for (cell, &bj) in row.iter_mut().zip(&basis) {
                    *cell += w * bi * bj;
                }
            }
            for (cell, &bi) in r.iter_mut().zip(&basis) {
                *cell += w * bi * z;
            }
        }
        solve_constant(m, r).map(|a| at + n * a as f32)
    }

    /// Per node its mirror image: the node nearest the mirrored position within a micrometre
    /// (itself on the mid-line), `u32::MAX` where there is none.
    fn mirrors(&self) -> Vec<u32> {
        let key = |p: Vec3| ((p.x * 100.0).round() as i32, (p.y * 100.0).round() as i32, (p.z * 100.0).round() as i32);
        let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
        for (i, &p) in self.points.iter().enumerate() {
            grid.entry(key(p)).or_default().push(i as u32);
        }
        self.points
            .iter()
            .map(|&p| {
                let target = Vec3::new(-p.x, p.y, p.z);
                let (cx, cy, cz) = key(target);
                let mut best = (u32::MAX, 1.0e-3_f32);
                for (dx, dy, dz) in (-1..=1).flat_map(|x| (-1..=1).flat_map(move |y| (-1..=1).map(move |z| (x, y, z))))
                {
                    for &m in grid.get(&(cx + dx, cy + dy, cz + dz)).map_or(&[][..], Vec::as_slice) {
                        let d = self.points.get(m as usize).map_or(f32::MAX, |q| q.distance(target));
                        if d < best.1 {
                            best = (m, d);
                        }
                    }
                }
                best.0
            })
            .collect()
    }
}

/// Solves the symmetric 6 x 6 system `m x = r` by Gaussian elimination with partial pivoting and
/// returns `x[0]`; `None` when the system is singular (too few points, or points on a line).
fn solve_constant(mut m: [[f64; 6]; 6], mut r: [f64; 6]) -> Option<f64> {
    let scale = m.iter().flatten().fold(0.0_f64, |a, &b| a.max(b.abs()));
    if scale <= 0.0 {
        return None;
    }
    for col in 0..6 {
        let pivot = (col..6).max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))?;
        if m[pivot][col].abs() < scale * 1.0e-10 {
            return None;
        }
        m.swap(col, pivot);
        r.swap(col, pivot);
        let (lead, lead_r) = (m[col], r[col]);
        for (row, rhs) in m.iter_mut().zip(r.iter_mut()).skip(col + 1) {
            let f = row[col] / lead[col];
            for (cell, &above) in row.iter_mut().zip(&lead).skip(col) {
                *cell -= f * above;
            }
            *rhs -= f * lead_r;
        }
    }
    let mut x = [0.0_f64; 6];
    for row in (0..6).rev() {
        let tail: f64 = (row + 1..6).map(|k| m[row][k] * x[k]).sum();
        x[row] = (r[row] - tail) / m[row][row];
    }
    x[0].is_finite().then_some(x[0])
}

/// Takes the fold across the nose root out (module documentation), in two steps on the skin in
/// [`NOSE_ROOT`]; the regions of the eyes do not move, and the result is mirror-exact.
///
/// 1. The vertices spread out over the surface they lie on: each of [`NOSE_ROOT_ROUNDS`] rounds
///    moves a vertex half the way to the mean of its neighbours and back onto the surface as it
///    was (the closest point). This opens the collapsed faces without changing the shape.
/// 2. [`NOSE_ROOT_PASSES`] times, every vertex moves along the local mean normal onto the quadric
///    fitted to the skin within [`NOSE_ROOT_WINDOW`] of it: this rounds what is left of the edge.
///
/// Returns the largest move, metres.
///
/// # Errors
/// When the surface is not a valid triangle set or the moved mesh fails [`Mesh::new`]'s checks.
pub fn fair_nose_root(asm: &mut Assembled, anatomy: &Anatomy) -> Result<f32> {
    let mut patch = Patch::new(asm, anatomy, NOSE_ROOT.0, NOSE_ROOT.1);
    let rest = patch.points.clone();
    let mirrors = patch.mirrors();
    let surface = TriangleTree::build(&rest, &patch.triangles)
        .map_err(|_| ModelError::invalid("nose root", 0, "the skin at the nose root is not a triangle surface"))?;
    for _ in 0..NOSE_ROOT_ROUNDS {
        let next: Vec<Vec3> = patch
            .points
            .iter()
            .zip(&patch.weights)
            .zip(&patch.neighbours)
            .map(|((&p, &w), list)| {
                if w <= 0.0 {
                    return p;
                }
                let sum = list.iter().filter_map(|&n| patch.points.get(n as usize)).fold(Vec3::ZERO, |a, &b| a + b);
                let target = p + (sum / list.len().max(1) as f32 - p) * (0.5 * w);
                surface.nearest(target, NOSE_ROOT_REACH, |_| true).map_or(p, |hit| hit.point)
            })
            .collect();
        patch.points = next;
    }
    for _ in 0..NOSE_ROOT_PASSES {
        let normals = patch.area_normals();
        let next: Vec<Vec3> = patch
            .points
            .iter()
            .zip(&patch.weights)
            .enumerate()
            .map(|(i, (&p, &w))| {
                if w <= 0.0 {
                    return p;
                }
                patch.on_quadric(i, &normals, NOSE_ROOT_WINDOW).map_or(p, |q| p + (q - p) * w)
            })
            .collect();
        patch.points = next;
    }
    // Mirror-exact: a pair takes the mean of one and the other's mirror image.
    let relaxed = patch.points.clone();
    for (i, (&m, p)) in mirrors.iter().zip(patch.points.iter_mut()).enumerate() {
        let moves = |k: usize| patch.weights.get(k).is_some_and(|&w| w > 0.0);
        match relaxed.get(m as usize) {
            Some(&other) if moves(i) && moves(m as usize) => {
                *p = (*p + Vec3::new(-other.x, other.y, other.z)) * 0.5;
                if m as usize == i {
                    p.x = 0.0;
                }
            }
            // No mirror image that moves too: the vertex stays.
            _ => *p = rest.get(i).copied().unwrap_or(*p),
        }
    }
    let mut data = asm.mesh.clone().into_data();
    let mut largest = 0.0_f32;
    for ((&now, &was), list) in patch.points.iter().zip(&rest).zip(&patch.copies) {
        let moved = (now - was) * 0.001;
        if moved == Vec3::ZERO {
            continue;
        }
        largest = largest.max(moved.length());
        for &v in list {
            if let Some(p) = data.positions.get_mut(v as usize) {
                *p += moved;
            }
        }
    }
    asm.mesh = Mesh::new(data)?;
    Ok(largest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wp25_vault_meets_its_axes() {
        let v = VAULT;
        assert!((v.radius(Vec3::Y) - v.height).abs() < 1e-3);
        assert!((v.radius(Vec3::X) - v.half_breadth).abs() < 1e-3);
        assert!((v.radius(Vec3::Z) - v.front).abs() < 1e-3);
        assert!((v.radius(-Vec3::Z) - v.back).abs() < 1e-3);
        assert_eq!(v.radius(Vec3::ZERO), 0.0);
        // Round 4: in front the horizontal section is rounder than behind (the forehead's corner
        // into the temple), and the halves meet without a step at the side.
        let square = Vault { front_exponent: v.exponent, ..v };
        let diagonal = Vec3::new(1.0, 0.0, 1.0);
        assert!(
            v.radius(diagonal) < square.radius(diagonal) - 3.0,
            "{} vs {}",
            v.radius(diagonal),
            square.radius(diagonal)
        );
        assert!((v.radius(Vec3::new(1.0, 0.0, 1.0e-4)) - v.radius(Vec3::new(1.0, 0.0, -1.0e-4))).abs() < 1.0e-2);
        // The weight: none on the face and below the ears, full on the vault.
        assert_eq!(vault_weight(Vec3::new(0.0, 40.0, 100.0)), 0.0);
        assert_eq!(vault_weight(Vec3::new(75.0, 30.0, 0.0)), 0.0);
        assert_eq!(vault_weight(Vec3::new(0.0, 125.0, 0.0)), 1.0);
        assert!(vault_weight(Vec3::new(0.0, 70.0, 95.0)) > 0.0 && vault_weight(Vec3::new(0.0, 70.0, 95.0)) < 1.0);
    }

    /// `[WP-25]` todo 1.1 round 2: the finished head meets ANSUR II (sex midpoints): head length
    /// 194.7 mm, breadth 151.1 mm, circumference 567.8 mm (the band above the brows) within a
    /// third of a standard deviation (2.4 / 1.8 / 6 mm), tragion to top of head 131.1 mm within 1 mm;
    /// the vault adds no fold.
    #[test]
    fn wp25_finished_cranium_meets_ansur() {
        let g =
            super::super::build::template(&super::super::build::Options { lods: false, ..Default::default() }).unwrap();
        let t = g.template;
        let o = Anatomy::neutral().head_origin;
        let regions = t.regions();
        let ears: Vec<u16> = ["ear-L", "ear-R"].iter().map(|n| regions.find(n).unwrap()).collect();
        let body = t.mesh().parts().iter().find(|p| p.name == "body").unwrap();
        let mut head: Vec<Vec3> = Vec::new();
        for f in body.faces() {
            for &v in t.mesh().face(f) {
                let r = regions.vertex_regions[v as usize];
                let h = (t.mesh().positions()[v as usize] - o) * 1000.0;
                if h.y > -40.0 && !ears.iter().any(|&e| regions.is_within(r, e)) {
                    head.push(h);
                }
            }
        }
        let top = head.iter().map(|h| h.y).fold(f32::MIN, f32::max);
        let glabella = head
            .iter()
            .filter(|h| h.x.abs() < 3.0 && (30.0..60.0).contains(&h.y))
            .map(|h| h.z)
            .fold(f32::MIN, f32::max);
        let back = head.iter().filter(|h| h.x.abs() < 3.0 && h.y > 0.0).map(|h| h.z).fold(f32::MAX, f32::min);
        let breadth = 2.0 * head.iter().filter(|h| h.y > 20.0).map(|h| h.x.abs()).fold(0.0, f32::max);
        let band: Vec<(f32, f32)> = head.iter().filter(|h| (h.y - 50.0).abs() < 2.0).map(|h| (h.x, h.z)).collect();
        let girth = crate::identity::measure::hull_perimeter(&band);
        let rows = [
            ("tragiontopofhead", top, 131.1, 1.0),
            ("headlength", glabella - back, 194.7, 2.4),
            ("headbreadth", breadth, 151.1, 1.8),
            ("headcircumference", girth, 567.8, 6.0),
        ];
        let mut bad = Vec::new();
        for (name, got, want, tol) in rows {
            println!("[WP-25] {name}: {got:.1} mm (ANSUR II {want}, tolerance {tol})");
            if (got - want).abs() > tol {
                bad.push(name);
            }
        }
        assert!(bad.is_empty(), "{bad:?}");
    }

    /// A fold: the angle between two neighbouring body triangles (degrees), the middle of their
    /// edge (head frame, mm), the edge's direction and the distance between their centroids (mm).
    struct Fold {
        angle: f32,
        at: Vec3,
        along: Vec3,
        apart: f32,
    }

    impl Fold {
        /// The radius of the curve the two triangles sample, mm: a measure of sharpness that does
        /// not depend on how fine the mesh is.
        fn radius(&self) -> f32 {
            self.apart / self.angle.to_radians().max(1.0e-6)
        }

        /// Whether the fold line runs across the nose (as the seam did) rather than down it.
        fn across(&self) -> bool {
            self.along.x.abs() > self.along.y.abs()
        }
    }

    /// The folds of the body skin of `mesh` around the nose root (head origin `o`): everything
    /// [`fair_nose_root`] can reach and the skin around it, by welded edge.
    fn folds_around(mesh: &Mesh, o: Vec3) -> HashMap<(u32, u32), Fold> {
        let pos = mesh.positions();
        let tri = mesh.triangulate();
        let body = mesh.parts().iter().position(|p| p.name == "body").unwrap();
        let key = |p: Vec3| ((p.x * 1e6).round() as i64, (p.y * 1e6).round() as i64, (p.z * 1e6).round() as i64);
        let mut first: HashMap<(i64, i64, i64), u32> = HashMap::new();
        let weld: Vec<u32> = pos.iter().enumerate().map(|(i, &p)| *first.entry(key(p)).or_insert(i as u32)).collect();
        let mut edges: HashMap<(u32, u32), Vec<(Vec3, Vec3)>> = HashMap::new();
        for k in 0..tri.indices.len() / 3 {
            if mesh.part_of_face(tri.triangle_faces[k] as usize) != Some(body) {
                continue;
            }
            let c = [0, 1, 2].map(|e| tri.indices[3 * k + e] as usize);
            let n = (pos[c[1]] - pos[c[0]]).cross(pos[c[2]] - pos[c[0]]).normalize_or_zero();
            let centroid = (pos[c[0]] + pos[c[1]] + pos[c[2]]) / 3.0;
            for e in 0..3 {
                let (a, b) = (weld[c[e]], weld[c[(e + 1) % 3]]);
                edges.entry((a.min(b), a.max(b))).or_default().push((n, centroid));
            }
        }
        let mut folds = HashMap::new();
        for (&(a, b), sides) in &edges {
            let (pa, pb) = (pos[a as usize], pos[b as usize]);
            let at = ((pa + pb) * 0.5 - o) * 1000.0;
            if sides.len() == 2 && at.x.abs() < 45.0 && (-30.0..40.0).contains(&at.y) && at.z > 60.0 {
                let fold = Fold {
                    angle: sides[0].0.dot(sides[1].0).clamp(-1.0, 1.0).acos().to_degrees(),
                    at,
                    along: (pb - pa).normalize_or_zero(),
                    apart: sides[0].1.distance(sides[1].1) * 1000.0,
                };
                folds.insert((a, b), fold);
            }
        }
        folds
    }

    /// The folds at the nose root itself (where the seam was), largest angle first.
    fn nose_root_folds(mesh: &Mesh, o: Vec3) -> Vec<Fold> {
        let mut folds: Vec<Fold> = folds_around(mesh, o)
            .into_values()
            .filter(|f| f.at.x.abs() < 12.0 && (-6.0..10.0).contains(&f.at.y) && f.at.z > 95.0)
            .collect();
        folds.sort_by(|a, b| b.angle.total_cmp(&a.angle));
        folds
    }

    /// The largest fold across the nose, the largest down the nose (degrees) and the tightest
    /// radius (mm).
    fn summary(folds: &[Fold]) -> (f32, f32, f32) {
        let worst = |across: bool| folds.iter().filter(|f| f.across() == across).map(|f| f.angle).fold(0.0, f32::max);
        (worst(true), worst(false), folds.iter().map(Fold::radius).fold(f32::MAX, f32::min))
    }

    #[test]
    fn wp25_quadric_fit_solves_and_refuses() {
        // The constant of z = 2 + 3 x - y + 0.5 x^2 + 0.25 x y - 0.125 y^2 from its normal equations.
        let truth = [2.0, 3.0, -1.0, 0.5, 0.25, -0.125];
        let mut m = [[0.0_f64; 6]; 6];
        let mut r = [0.0_f64; 6];
        for i in -3..=3 {
            for j in -3..=3 {
                let (x, y) = (f64::from(i) * 0.7, f64::from(j) * 1.1);
                let basis = [1.0, x, y, x * x, x * y, y * y];
                let z: f64 = basis.iter().zip(&truth).map(|(b, t)| b * t).sum();
                for a in 0..6 {
                    for b in 0..6 {
                        m[a][b] += basis[a] * basis[b];
                    }
                    r[a] += basis[a] * z;
                }
            }
        }
        assert!((solve_constant(m, r).unwrap() - 2.0).abs() < 1e-9);
        // Points on one line do not determine a quadric; nothing at all does not either.
        let mut line = [[0.0_f64; 6]; 6];
        for i in -3..=3 {
            let x = f64::from(i);
            let basis = [1.0, x, 0.0, x * x, 0.0, 0.0];
            for a in 0..6 {
                for b in 0..6 {
                    line[a][b] += basis[a] * basis[b];
                }
            }
        }
        assert_eq!(solve_constant(line, [1.0; 6]), None);
        assert_eq!(solve_constant([[0.0; 6]; 6], [0.0; 6]), None);
    }

    /// A node lifted 0.8 mm off a paraboloid comes back to within 0.15 mm of it (the lifted node is
    /// one of the fit's own samples); the patch's mirror pairs are found.
    #[test]
    fn wp25_patch_fits_a_quadric_and_pairs_mirror_images() {
        let surface = |x: f32, y: f32| 0.04 * x * x + 0.02 * y * y;
        let side = 9_usize;
        let mut patch = Patch {
            points: Vec::new(),
            weights: Vec::new(),
            copies: Vec::new(),
            neighbours: Vec::new(),
            triangles: Vec::new(),
        };
        for j in 0..side {
            for i in 0..side {
                let (x, y) = (i as f32 - 4.0, j as f32 - 4.0);
                patch.points.push(Vec3::new(x, y, surface(x, y)));
                patch.weights.push(1.0);
            }
        }
        for j in 0..side - 1 {
            for i in 0..side - 1 {
                let at = |i: usize, j: usize| (j * side + i) as u32;
                patch.triangles.extend([at(i, j), at(i + 1, j), at(i + 1, j + 1)]);
                patch.triangles.extend([at(i, j), at(i + 1, j + 1), at(i, j + 1)]);
            }
        }
        let mirrors = patch.mirrors();
        let centre = 4 * side + 4;
        assert_eq!(mirrors[centre], centre as u32, "a mid-line node is its own mirror image");
        assert_eq!(mirrors[4 * side + 1], (4 * side + 7) as u32);
        assert!(mirrors.iter().all(|&m| m != u32::MAX));
        let on = patch.points[centre + 1];
        patch.points[centre + 1].z += 0.8;
        let normals = patch.area_normals();
        assert!(normals[centre].z > 0.9, "{:?}", normals[centre]);
        let fitted = patch.on_quadric(centre + 1, &normals, 3.0).unwrap();
        assert!(fitted.distance(on) < 0.15, "{fitted:?} is not back at {on:?}");
        // An empty neighbourhood has no normal.
        let lone = Patch {
            points: vec![Vec3::ZERO],
            weights: vec![1.0],
            copies: vec![Vec::new()],
            neighbours: vec![Vec::new()],
            triangles: Vec::new(),
        };
        assert_eq!(lone.on_quadric(0, &lone.area_normals(), 3.0), None);
    }

    /// `[WP-25]` The nose root has no fold left, measured on the finished template.
    ///
    /// - The seam ran ACROSS the nose root: 88.0 degrees on the base, 73.2 with the scanned face
    ///   before [`fair_nose_root`]; 9.0 after. Bound: 15.
    /// - No two neighbouring triangles turn on a radius under 3 mm (0.41 mm on the base, 0.43 with
    ///   the scanned face, 4.07 after).
    /// - The largest angle left (30.4 degrees) lies on the mid-line of the bridge, between
    ///   triangles 2.2 mm apart: the bridge's own cross-section (a radius of 4.1 mm sampled by
    ///   columns 3.3 mm apart; 32.3 degrees before the fairing, which does not flatten the
    ///   bridge). It is bounded by the radius above and printed, not by an angle.
    #[test]
    fn wp25_nose_root_has_no_fold() {
        let g =
            super::super::build::template(&super::super::build::Options { lods: false, ..Default::default() }).unwrap();
        let folds = nose_root_folds(g.template.mesh(), Anatomy::neutral().head_origin);
        let (across, down, radius) = summary(&folds);
        println!(
            "[WP-25] nose root: {} edges; largest fold across {across:.1} deg, down the bridge {down:.1} deg; tightest radius {radius:.2} mm",
            folds.len()
        );
        assert!(folds.len() > 200, "{} edges measured", folds.len());
        assert!(across < 15.0, "a fold of {across} deg across the nose root");
        assert!(radius > 3.0, "neighbouring triangles turn on a radius of {radius} mm");
    }

    /// `[WP-25]` What [`fair_nose_root`] does to the head after the scanned face and the cranium:
    /// it takes the seam out without changing the shape (the moved vertices lie within 1 mm of the
    /// surface as it was, and the old vertices within 1 mm of the new surface: measured 0.34 and
    /// 0.57 mm, while the vertices slide up to 5.2 mm along it), moves nothing outside [`NOSE_ROOT`]
    /// and nothing of the eyes, and leaves the head mirror-exact.
    #[test]
    fn wp25_nose_root_fairing_keeps_the_shape() {
        use super::super::{anatomy::Auricle, assemble::assemble, cage::Cage, landmarks, parts, scan_face, uv};
        let variant = super::super::genitals::Variant::default();
        let surface = super::super::body_surface_for(variant, Auricle::Finished).unwrap();
        let parts = parts::build(&surface.anatomy, &surface.body, &surface.targets).unwrap();
        let pieces: Vec<&Cage> = std::iter::once(surface.mesh()).chain(parts.iter().map(|p| &p.mesh)).collect();
        let layout = uv::layout(&pieces).unwrap();
        let mut asm = assemble(&surface, &parts, &layout).unwrap();
        let described = landmarks::build(&surface, &asm).unwrap();
        let o = surface.anatomy.head_origin;
        scan_face::apply(&mut asm, &described, &surface.anatomy).unwrap();
        cranium(&mut asm, &surface.anatomy).unwrap();
        let before = asm.mesh.positions().to_vec();
        let (across_before, down_before, radius_before) = summary(&nose_root_folds(&asm.mesh, o));
        let moved = fair_nose_root(&mut asm, &surface.anatomy).unwrap();
        let after = asm.mesh.positions();
        let (across, down, radius) = summary(&nose_root_folds(&asm.mesh, o));
        println!(
            "[WP-25] nose root fairing: across {across_before:.1} -> {across:.1} deg, down {down_before:.1} -> {down:.1} deg, radius {radius_before:.2} -> {radius:.2} mm; largest move {:.2} mm",
            moved * 1000.0
        );
        assert!(across_before > 60.0 && radius_before < 1.0, "the seam is gone before the fairing: remove it");
        assert!(across < 15.0 && radius > 3.0 && down <= down_before + 1.0);

        // The shape: the faired vertices lie on the surface as it was, and the vertices as they were
        // lie on the faired surface (no corner is cut, the bridge keeps its height).
        let tri = asm.mesh.triangulate();
        let body = asm.mesh.parts().iter().position(|p| p.name == "body").unwrap();
        let (c, r) = NOSE_ROOT;
        let zone = |p: Vec3| {
            let h = (p - o) * 1000.0;
            Vec3::new((h.x - c[0]) / r[0], (h.y - c[1]) / r[1], (h.z - c[2]) / r[2]).length()
        };
        let indices: Vec<u32> = (0..tri.indices.len() / 3)
            .filter(|&k| asm.mesh.part_of_face(tri.triangle_faces[k] as usize) == Some(body))
            .map(|k| [tri.indices[3 * k], tri.indices[3 * k + 1], tri.indices[3 * k + 2]])
            .filter(|t| t.iter().all(|&v| zone(before[v as usize]) < 3.0))
            .flatten()
            .collect();
        let was = TriangleTree::build(&before, &indices).unwrap();
        let now = TriangleTree::build(after, &indices).unwrap();
        let eyes: Vec<u16> = ["eye-L", "eye-R"].iter().map(|n| asm.regions.find(n).unwrap()).collect();
        let (mut off, mut cut, mut slid) = (0.0_f32, 0.0_f32, 0.0_f32);
        let mut moved_to: Vec<Vec3> = Vec::new();
        for (v, (&p, &q)) in before.iter().zip(after).enumerate() {
            if p == q {
                continue;
            }
            assert_eq!(asm.source[v].0, 0, "a vertex of a part moved");
            let region = asm.regions.vertex_regions[v];
            assert!(!eyes.iter().any(|&e| asm.regions.is_within(region, e)), "a vertex of an eye moved");
            assert!(zone(p) < 2.0, "a vertex outside the zone moved: {:?}", (p - o) * 1000.0);
            off = off.max(was.nearest(q, 0.01, |_| true).map_or(f32::MAX, |hit| hit.distance));
            cut = cut.max(now.nearest(p, 0.01, |_| true).map_or(f32::MAX, |hit| hit.distance));
            slid = slid.max(p.distance(q));
            moved_to.push(q);
        }
        // Mirror-exact: every moved vertex has its mirror image.
        let mirror = moved_to
            .iter()
            .map(|&q| moved_to.iter().map(|&m| m.distance(Vec3::new(-q.x, q.y, q.z))).fold(f32::MAX, f32::min))
            .fold(0.0, f32::max);
        println!(
            "[WP-25] nose root fairing: {} vertices moved, up to {:.2} mm along the surface; {:.3} mm off the old surface, old vertices {:.3} mm off the new one; mirror error {mirror:e} m",
            moved_to.len(),
            slid * 1000.0,
            off * 1000.0,
            cut * 1000.0
        );
        assert!(moved_to.len() > 100 && off < 1.0e-3 && cut < 1.0e-3, "{off} / {cut} m off the surface");
        assert!(mirror < 1.0e-7, "mirror error {mirror} m");
    }
}
