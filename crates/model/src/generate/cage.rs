//! The control cage `[MOD-01]`: a polygon mesh with a tag per face, the ring operations the body is
//! modelled with, Catmull-Clark subdivision and the fit that makes the subdivided surface pass
//! through the designed points.
//!
//! The generator models the body the way a modeller box-models a base mesh: closed rings of
//! vertices are bridged into tubes, blocks of a ring grid are cut out and refilled with concentric
//! ring sequences (eyes, mouth, nose, ears), tube ends are closed with grid caps. Any polygon is
//! allowed in the cage; one subdivision level turns every face into quads, so the subdivided
//! surface is all quads and keeps the cage's edge loops (a loop of cage edges stays a loop).
//!
//! Orientation rule: a ring is listed counter-clockwise around the direction its tube grows in (seen
//! from the tube's far end looking back). With that, [`Cage::bridge`] makes outward-facing quads
//! from an earlier ring to a later one, and [`Cage::cap`] closes the far end of a tube.

use std::collections::{HashMap, HashSet};

use humanart_math::Vec3;

use crate::error::{ModelError, Result};

/// What a face is: its region and its UV chart, both generator-side ids.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Tag {
    /// Region id (an index into the generator's region table).
    pub region: u16,
    /// Chart id (an index into the generator's chart table).
    pub chart: u16,
}

impl Tag {
    /// A tag from a region and a chart id.
    pub const fn new(region: u16, chart: u16) -> Self {
        Self { region, chart }
    }
}

/// The edge table of a cage: the undirected edges `(low, high)` in first-use order, the index of
/// each edge, and the (up to two) faces on each edge (`u32::MAX` when there is none).
pub type EdgeTable = (Vec<(u32, u32)>, HashMap<(u32, u32), u32>, Vec<[u32; 2]>);

/// A polygon mesh with one [`Tag`] per face. Faces are counter-clockwise seen from outside.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Cage {
    /// Vertex positions, metres.
    pub pos: Vec<Vec3>,
    /// `face_count + 1` offsets into `corners`.
    pub offsets: Vec<u32>,
    /// The corners of all faces.
    pub corners: Vec<u32>,
    /// The tag of each face.
    pub tags: Vec<Tag>,
}

impl Cage {
    /// An empty cage.
    pub fn new() -> Self {
        Self { offsets: vec![0], ..Self::default() }
    }

    /// Adds a vertex and returns its index.
    pub fn add(&mut self, p: Vec3) -> u32 {
        self.pos.push(p);
        (self.pos.len() - 1) as u32
    }

    /// Adds one vertex per point and returns their indices.
    pub fn add_all(&mut self, points: &[Vec3]) -> Vec<u32> {
        points.iter().map(|&p| self.add(p)).collect()
    }

    /// Adds a face (corners counter-clockwise from outside).
    pub fn face(&mut self, corners: &[u32], tag: Tag) {
        self.corners.extend_from_slice(corners);
        self.offsets.push(self.corners.len() as u32);
        self.tags.push(tag);
    }

    /// How many faces there are.
    pub fn face_count(&self) -> usize {
        self.offsets.len() - 1
    }

    /// The corners of face `f`.
    pub fn corners_of(&self, f: usize) -> &[u32] {
        &self.corners[self.offsets[f] as usize..self.offsets[f + 1] as usize]
    }

    /// The centroid of face `f`.
    pub fn centroid(&self, f: usize) -> Vec3 {
        let c = self.corners_of(f);
        c.iter().fold(Vec3::ZERO, |s, &v| s + self.pos[v as usize]) / c.len() as f32
    }

    /// Bridges two closed rings of equal length with quads, `a` before `b` along the tube.
    pub fn bridge(&mut self, a: &[u32], b: &[u32], tag: Tag) {
        let n = a.len().min(b.len());
        for i in 0..n {
            let j = (i + 1) % n;
            self.face(&[a[i], a[j], b[j], b[i]], tag);
        }
    }

    /// Bridges a closed ring `a` to a ring `b` with twice as many vertices (`b[2i]` next to `a[i]`)
    /// with one pentagon per edge of `a`: symmetric, unlike a loft.
    pub fn bridge_double(&mut self, a: &[u32], b: &[u32], tag: &dyn Fn(usize) -> Tag) {
        let n = a.len();
        if b.len() != 2 * n {
            return;
        }
        for i in 0..n {
            let j = (i + 1) % n;
            self.face(&[a[i], a[j], b[(2 * i + 2) % (2 * n)], b[2 * i + 1], b[2 * i]], tag(2 * i + 1));
        }
    }

    /// Bridges two open polylines of equal length (a strip), `a` before `b`.
    pub fn bridge_open(&mut self, a: &[u32], b: &[u32], tag: Tag) {
        for i in 0..a.len().min(b.len()).saturating_sub(1) {
            self.face(&[a[i], a[i + 1], b[i + 1], b[i]], tag);
        }
    }

    /// Bridges two closed rings of different lengths (a loft): vertex 0 of each ring correspond,
    /// and the rings are walked by their normalised index; where one ring advances alone a
    /// triangle is made, otherwise a quad.
    pub fn loft(&mut self, a: &[u32], b: &[u32], tag: Tag) {
        let (n, m) = (a.len(), b.len());
        let (mut i, mut j) = (0_usize, 0_usize);
        while i < n || j < m {
            let (ca, cb) = (i as f32 / n as f32, j as f32 / m as f32);
            let (ta, tb) = ((i + 1) as f32 / n as f32, (j + 1) as f32 / m as f32);
            let both = if i < n && j < m { (ta - tb).abs() } else { f32::INFINITY };
            let only_a = if i < n { (ta - cb).abs() } else { f32::INFINITY };
            let only_b = if j < m { (ca - tb).abs() } else { f32::INFINITY };
            let (ai, ai1) = (a[i % n], a[(i + 1) % n]);
            let (bj, bj1) = (b[j % m], b[(j + 1) % m]);
            if both <= only_a.min(only_b) + 1e-6 {
                self.face(&[ai, ai1, bj1, bj], tag);
                i += 1;
                j += 1;
            } else if only_a <= only_b {
                self.face(&[ai, ai1, bj], tag);
                i += 1;
            } else {
                self.face(&[ai, bj1, bj], tag);
                j += 1;
            }
        }
    }

    /// Closes the far end of a tube whose last ring is `ring` (`2 * (a + b)` vertices) with an
    /// `a` x `b` grid of quads. The grid's corners sit at ring vertices `start`, `start + a`,
    /// `start + a + b` and `start + 2a + b`; the interior vertices are placed by a Coons patch of the
    /// ring and then moved by `bulge` (called with the patch coordinates in `[0, 1]²` and the Coons
    /// position). Returns the interior vertices, row-major (`(a - 1) * (b - 1)` of them).
    pub fn cap(
        &mut self,
        ring: &[u32],
        a: usize,
        b: usize,
        start: usize,
        tag: Tag,
        bulge: &dyn Fn(f32, f32, Vec3) -> Vec3,
    ) -> Result<Vec<u32>> {
        let n = ring.len();
        if a == 0 || b == 0 || n != 2 * (a + b) {
            return Err(ModelError::invalid("cage cap", n, "ring length is not 2 * (a + b)"));
        }
        let at = |k: usize| ring[(start + k) % n];
        // Grid vertex (i, j), i along a (0..=a), j along b (0..=b); boundary from the ring:
        // bottom j = 0: i = 0..a -> ring[start + i]; right i = a: j -> ring[start + a + j];
        // top j = b: i -> ring[start + 2a + b - i]; left i = 0: j -> ring[start + 2a + 2b - j].
        let boundary = |i: usize, j: usize| -> Option<u32> {
            if j == 0 {
                Some(at(i))
            } else if i == a {
                Some(at(a + j))
            } else if j == b {
                Some(at(2 * a + b - i))
            } else if i == 0 {
                Some(at(2 * a + 2 * b - j))
            } else {
                None
            }
        };
        let mut grid = vec![0_u32; (a + 1) * (b + 1)];
        let mut interior = Vec::new();
        for j in 0..=b {
            for i in 0..=a {
                grid[j * (a + 1) + i] = if let Some(v) = boundary(i, j) {
                    v
                } else {
                    let (s, t) = (i as f32 / a as f32, j as f32 / b as f32);
                    let (b0, b1) = (boundary(i, 0), boundary(i, b));
                    let (l0, l1) = (boundary(0, j), boundary(a, j));
                    let (c00, c10, c01, c11) = (at(0), at(a), at(2 * a + b), at(a + b));
                    let (Some(b0), Some(b1), Some(l0), Some(l1)) = (b0, b1, l0, l1) else {
                        return Err(ModelError::invalid("cage cap", n, "grid boundary lookup failed"));
                    };
                    let p = |v: u32| self.pos[v as usize];
                    let ruled_s = p(l0) * (1.0 - s) + p(l1) * s;
                    let ruled_t = p(b0) * (1.0 - t) + p(b1) * t;
                    let bilinear = p(c00) * ((1.0 - s) * (1.0 - t))
                        + p(c10) * (s * (1.0 - t))
                        + p(c01) * ((1.0 - s) * t)
                        + p(c11) * (s * t);
                    let coons = ruled_s + ruled_t - bilinear;
                    let v = self.add(bulge(s, t, coons));
                    interior.push(v);
                    v
                };
            }
        }
        for j in 0..b {
            for i in 0..a {
                let g = |ii: usize, jj: usize| grid[jj * (a + 1) + ii];
                self.face(&[g(i, j), g(i + 1, j), g(i + 1, j + 1), g(i, j + 1)], tag);
            }
        }
        Ok(interior)
    }

    /// Appends another cage (its faces keep their tags); returns the vertex offset of `other`.
    pub fn append(&mut self, other: &Self) -> u32 {
        let base = self.pos.len() as u32;
        self.pos.extend_from_slice(&other.pos);
        for f in 0..other.face_count() {
            let c: Vec<u32> = other.corners_of(f).iter().map(|&v| v + base).collect();
            self.face(&c, other.tags[f]);
        }
        base
    }

    /// The undirected edges, each `(low, high)`, in first-use order, with the faces on each.
    pub fn edges(&self) -> EdgeTable {
        let mut list = Vec::new();
        let mut index: HashMap<(u32, u32), u32> = HashMap::with_capacity(self.corners.len());
        let mut faces: Vec<[u32; 2]> = Vec::new();
        for f in 0..self.face_count() {
            let c = self.corners_of(f);
            for k in 0..c.len() {
                let (a, b) = (c[k], c[(k + 1) % c.len()]);
                let key = (a.min(b), a.max(b));
                let e = *index.entry(key).or_insert_with(|| {
                    list.push(key);
                    faces.push([u32::MAX; 2]);
                    (list.len() - 1) as u32
                });
                let slot = &mut faces[e as usize];
                if slot[0] == u32::MAX {
                    slot[0] = f as u32;
                } else {
                    slot[1] = f as u32;
                }
            }
        }
        (list, index, faces)
    }
}

/// A subdivided cage and the map from each parent edge to its new edge vertex.
#[derive(Clone, Debug, Default)]
pub struct Subdivided {
    /// The subdivided mesh: parent vertex `i` keeps index `i`.
    pub cage: Cage,
    /// Parent edge `(low, high)` -> the index of its edge vertex.
    pub edge_points: HashMap<(u32, u32), u32>,
    /// Parent face -> the index of its face vertex.
    pub face_points: Vec<u32>,
}

impl Subdivided {
    /// Refines a polyline of parent vertices: the edge vertex is inserted between each pair.
    /// `closed` also refines the closing edge. Pairs that are not edges are kept as they are.
    pub fn refine(&self, line: &[u32], closed: bool) -> Vec<u32> {
        let mut out = Vec::with_capacity(2 * line.len());
        let n = line.len();
        for i in 0..n {
            out.push(line[i]);
            if i + 1 < n || closed {
                let (a, b) = (line[i], line[(i + 1) % n]);
                if let Some(&e) = self.edge_points.get(&(a.min(b), a.max(b))) {
                    out.push(e);
                }
            }
        }
        out
    }
}

/// One Catmull-Clark step. Edges in `sharp` (keys `(low, high)`) are creases: their edge vertex is
/// the midpoint and a vertex on exactly two sharp edges follows the crease rule (on three or more
/// it stays put). Boundary edges are treated as sharp.
pub fn subdivide(c: &Cage, sharp: &dyn Fn(u32, u32) -> bool) -> Subdivided {
    let v = c.pos.len();
    let f = c.face_count();
    let (edges, index, edge_faces) = c.edges();
    let e = edges.len();
    let face_pts: Vec<Vec3> = (0..f).map(|i| c.centroid(i)).collect();
    let is_sharp: Vec<bool> =
        edges.iter().zip(&edge_faces).map(|(&(a, b), ef)| ef[1] == u32::MAX || sharp(a, b)).collect();
    let mut out = Cage::new();
    out.pos = vec![Vec3::ZERO; v + e + f];
    // Edge points.
    for (k, (&(a, b), ef)) in edges.iter().zip(&edge_faces).enumerate() {
        let mid = (c.pos[a as usize] + c.pos[b as usize]) * 0.5;
        out.pos[v + k] =
            if is_sharp[k] { mid } else { (mid * 2.0 + face_pts[ef[0] as usize] + face_pts[ef[1] as usize]) * 0.25 };
    }
    // Vertex points: accumulate face-point and edge-midpoint sums per vertex.
    let mut face_sum = vec![Vec3::ZERO; v];
    let mut face_n = vec![0_u32; v];
    for (fi, &fp) in face_pts.iter().enumerate().take(f) {
        for &x in c.corners_of(fi) {
            face_sum[x as usize] += fp;
            face_n[x as usize] += 1;
        }
    }
    let mut mid_sum = vec![Vec3::ZERO; v];
    let mut valence = vec![0_u32; v];
    let mut sharp_n = vec![0_u32; v];
    let mut sharp_sum = vec![Vec3::ZERO; v];
    for (k, &(a, b)) in edges.iter().enumerate() {
        let (pa, pb) = (c.pos[a as usize], c.pos[b as usize]);
        let mid = (pa + pb) * 0.5;
        for (x, other) in [(a, pb), (b, pa)] {
            mid_sum[x as usize] += mid;
            valence[x as usize] += 1;
            if is_sharp[k] {
                sharp_n[x as usize] += 1;
                sharp_sum[x as usize] += other;
            }
        }
    }
    for i in 0..v {
        let p = c.pos[i];
        out.pos[i] = match sharp_n[i] {
            0 if valence[i] >= 3 && face_n[i] > 0 => {
                let n = valence[i] as f32;
                let q = face_sum[i] / face_n[i] as f32;
                let r = mid_sum[i] / n;
                (q + r * 2.0 + p * (n - 3.0)) / n
            }
            2 => p * 0.75 + sharp_sum[i] * 0.125,
            _ => p,
        };
    }
    out.pos[v + e..v + e + f].copy_from_slice(&face_pts[..f]);
    // Faces: one quad per corner.
    let edge_vertex = |a: u32, b: u32| -> u32 { v as u32 + index[&(a.min(b), a.max(b))] };
    for fi in 0..f {
        let cs = c.corners_of(fi);
        let n = cs.len();
        let fp = (v + e + fi) as u32;
        for k in 0..n {
            let (prev, cur, next) = (cs[(k + n - 1) % n], cs[k], cs[(k + 1) % n]);
            out.face(&[cur, edge_vertex(cur, next), fp, edge_vertex(prev, cur)], c.tags[fi]);
        }
    }
    let edge_points = index.iter().map(|(&key, &k)| (key, v as u32 + k)).collect();
    let face_points = (0..f).map(|i| (v + e + i) as u32).collect();
    Subdivided { cage: out, edge_points, face_points }
}

/// Subdivides `levels` times; the edges for which `sharp` is true are creases, and so are the
/// halves of a crease edge on later levels.
pub fn subdivide_levels(c: &Cage, levels: usize, sharp: &dyn Fn(u32, u32) -> bool) -> Vec<Subdivided> {
    let (edges, _, _) = c.edges();
    let mut creases: HashSet<(u32, u32)> = edges.into_iter().filter(|&(a, b)| sharp(a, b)).collect();
    let mut steps: Vec<Subdivided> = Vec::with_capacity(levels);
    for _ in 0..levels {
        let next = {
            let src = steps.last().map_or(c, |s| &s.cage);
            subdivide(src, &|a, b| creases.contains(&(a.min(b), a.max(b))))
        };
        let mut child = HashSet::with_capacity(2 * creases.len());
        for &(a, b) in &creases {
            if let Some(&m) = next.edge_points.get(&(a, b)) {
                child.insert((a.min(m), a.max(m)));
                child.insert((b.min(m), b.max(m)));
            }
        }
        creases = child;
        steps.push(next);
    }
    steps
}

/// Moves the cage vertices so that, after `levels` subdivision steps, the vertex that descends from
/// each cage vertex lands on `targets[i]` (inverse subdivision by fixed-point iteration).
pub fn fit(c: &mut Cage, targets: &[Vec3], levels: usize, iterations: usize, sharp: &dyn Fn(u32, u32) -> bool) {
    for _ in 0..iterations {
        let steps = subdivide_levels(c, levels, sharp);
        let Some(last) = steps.last() else { return };
        for (i, t) in targets.iter().enumerate().take(c.pos.len()) {
            let d = *t - last.cage.pos[i];
            c.pos[i] += d;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube() -> Cage {
        let mut c = Cage::new();
        for z in [0.0, 1.0] {
            for y in [0.0, 1.0] {
                for x in [0.0, 1.0] {
                    c.add(Vec3::new(x, y, z));
                }
            }
        }
        let t = Tag::default();
        for f in [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]] {
            c.face(&f, t);
        }
        c
    }

    #[test]
    fn subdivision_counts_and_closedness() {
        let c = cube();
        let s = subdivide(&c, &|_, _| false);
        assert_eq!(s.cage.pos.len(), 8 + 12 + 6);
        assert_eq!(s.cage.face_count(), 24);
        let (_, _, faces) = s.cage.edges();
        assert!(faces.iter().all(|f| f[1] != u32::MAX), "subdivided cube stays closed");
        // Parent vertex 0 keeps its index and moves inward.
        assert!(s.cage.pos[0].x > 0.0 && s.cage.pos[0].x < 0.5);
    }

    #[test]
    fn refine_inserts_edge_points() {
        let c = cube();
        let s = subdivide(&c, &|_, _| false);
        let loop_ = s.refine(&[0, 1, 5, 4], true);
        assert_eq!(loop_.len(), 8);
        assert_eq!(loop_[0], 0);
        assert!(loop_[1] >= 8);
    }

    #[test]
    fn fit_hits_targets() {
        let mut c = cube();
        let targets = c.pos.clone();
        fit(&mut c, &targets, 2, 30, &|_, _| false);
        let s = subdivide_levels(&c, 2, &|_, _| false);
        let last = &s[1].cage;
        let err = targets.iter().enumerate().map(|(i, t)| last.pos[i].distance(*t)).fold(0.0_f32, f32::max);
        assert!(err < 1e-3, "fit error {err}");
    }

    #[test]
    fn loft_and_cap_are_closed() {
        let mut c = Cage::new();
        let ring = |c: &mut Cage, n: usize, y: f32| -> Vec<u32> {
            (0..n)
                .map(|k| {
                    let a = std::f32::consts::TAU * k as f32 / n as f32;
                    c.add(Vec3::new(a.cos(), y, -a.sin()))
                })
                .collect()
        };
        let t = Tag::default();
        let a = ring(&mut c, 8, 0.0);
        let b = ring(&mut c, 12, 1.0);
        c.loft(&a, &b, t);
        let rev: Vec<u32> = a.iter().rev().copied().collect();
        c.cap(&rev, 2, 2, 0, t, &|_, _, p| p).unwrap();
        c.cap(&b, 3, 3, 0, t, &|_, _, p| p).unwrap();
        let (_, _, faces) = c.edges();
        assert!(faces.iter().all(|f| f[1] != u32::MAX));
        // Every edge is used once in each direction (consistent orientation).
        let mut dir: HashMap<(u32, u32), i32> = HashMap::new();
        for f in 0..c.face_count() {
            let cs = c.corners_of(f);
            for k in 0..cs.len() {
                *dir.entry((cs[k], cs[(k + 1) % cs.len()])).or_default() += 1;
            }
        }
        assert!(dir.keys().all(|&(a, b)| dir.get(&(b, a)) == Some(&1)));
    }
}
