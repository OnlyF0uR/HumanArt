//! The face context the expression recipes are written against (WP-19, `[EXP-01]`).
//!
//! A [`FaceField`] looks at a [`Template`] through **names only**: landmarks, joints, regions and
//! mesh parts are looked up by their stable names, never by vertex index or fixed coordinate, so the
//! recipes survive changes of the generator's face shape and topology.
//!
//! It works on the **welded surface**: vertices that share a position (UV seam copies) are one
//! *node*; a displacement computed per node is copied to every vertex of the node, so expression
//! deltas never split a seam. Distances are **geodesic** (shortest paths along mesh edges): the upper
//! and lower lip are far apart through the mouth corner even where they touch, and a separate part
//! (eyeball, tooth) is never reached from the skin.
//!
//! Helpers: [`falloff`] (a smooth 1-to-0 ramp), [`side_weight`] (a smooth left/right split across
//! the midline, so a left and a right target sum to the bilateral one).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use humanart_math::{Transform, Vec3};

use super::error::{ExprError, ExprResult};
use crate::names::Side;
use crate::template::Template;
use crate::topology::Csr;

/// The face context of one template: welded nodes with positions, normals, regions and parts, the
/// node adjacency, and name lookups.
#[derive(Clone, Debug)]
pub struct FaceField<'t> {
    template: &'t Template,
    node_of: Vec<u32>,
    rep: Vec<u32>,
    pos: Vec<Vec3>,
    normal: Vec<Vec3>,
    part: Vec<u16>,
    adj: Csr,
    rest_world: Vec<Transform>,
    triangles: Vec<[u32; 3]>,
}

impl<'t> FaceField<'t> {
    /// Builds the context: welds vertices by their position keys, derives node adjacency from the
    /// faces, area-weighted node normals and the part of every node.
    pub fn new(template: &'t Template) -> ExprResult<Self> {
        let mesh = template.mesh();
        let keys = mesh.position_keys();
        let n_vertices = mesh.vertex_count();
        let mut node_of = vec![u32::MAX; n_vertices];
        let mut rep = Vec::new();
        for v in 0..n_vertices {
            let k = keys.get(v).copied().unwrap_or(v as u32) as usize;
            let node = match node_of.get(k).copied() {
                Some(n) if n != u32::MAX && k < v => n,
                _ => {
                    rep.push(v as u32);
                    (rep.len() - 1) as u32
                }
            };
            node_of[v] = node;
        }
        let positions = mesh.positions();
        let pos: Vec<Vec3> = rep.iter().map(|&v| positions[v as usize]).collect();
        let mut normal = vec![Vec3::ZERO; rep.len()];
        let mut part = vec![u16::MAX; rep.len()];
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        for (pi, p) in mesh.parts().iter().enumerate() {
            for f in p.faces() {
                let face = mesh.face(f);
                let area = mesh.face_vector_area(f);
                for (i, &v) in face.iter().enumerate() {
                    let a = node_of[v as usize];
                    let b = node_of[face[(i + 1) % face.len()] as usize];
                    normal[a as usize] += area;
                    if part[a as usize] == u16::MAX {
                        part[a as usize] = pi as u16;
                    }
                    if a != b {
                        pairs.push((a, b));
                        pairs.push((b, a));
                    }
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        let adj = Csr::from_pairs(rep.len(), &pairs);
        for n in &mut normal {
            *n = n.normalize_or_zero();
        }
        let rest_world = template.skeleton().rest_world();
        let tri = mesh.triangulate();
        let triangles = tri
            .indices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|t| [node_of[t[0] as usize], node_of[t[1] as usize], node_of[t[2] as usize]])
            .filter(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2])
            .collect();
        Ok(Self { template, node_of, rep, pos, normal, part, adj, rest_world, triangles })
    }

    /// The template.
    #[must_use]
    pub const fn template(&self) -> &'t Template {
        self.template
    }

    /// How many welded nodes there are.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.pos.len()
    }

    /// The node of vertex `v` (`u32::MAX` when out of range).
    #[must_use]
    pub fn node_of(&self, v: usize) -> u32 {
        self.node_of.get(v).copied().unwrap_or(u32::MAX)
    }

    /// The node of every vertex.
    #[must_use]
    pub fn vertex_nodes(&self) -> &[u32] {
        &self.node_of
    }

    /// The lowest-indexed vertex of every node.
    #[must_use]
    pub fn representatives(&self) -> &[u32] {
        &self.rep
    }

    /// Rest positions per node.
    #[must_use]
    pub fn positions(&self) -> &[Vec3] {
        &self.pos
    }

    /// Unit area-weighted rest normals per node (zero for degenerate nodes).
    #[must_use]
    pub fn normals(&self) -> &[Vec3] {
        &self.normal
    }

    /// The mesh's triangles as node triples (triangles collapsed by the weld are left out).
    #[must_use]
    pub fn triangles(&self) -> &[[u32; 3]] {
        &self.triangles
    }

    /// The neighbours of node `n`.
    #[must_use]
    pub fn neighbours(&self, n: usize) -> &[u32] {
        self.adj.row(n)
    }

    /// The mesh part index of node `n` (`u16::MAX` for a node on no face).
    #[must_use]
    pub fn part_of(&self, n: usize) -> u16 {
        self.part.get(n).copied().unwrap_or(u16::MAX)
    }

    /// The region index of node `n` (its representative vertex's region).
    #[must_use]
    pub fn region_of(&self, n: usize) -> u16 {
        let v = self.rep.get(n).copied().unwrap_or(0) as usize;
        self.template.regions().vertex_regions.get(v).copied().unwrap_or(0)
    }

    /// The index of the mesh part called `name`.
    pub fn part(&self, name: &str) -> ExprResult<u16> {
        self.template
            .mesh()
            .parts()
            .iter()
            .position(|p| p.name == name)
            .map(|p| p as u16)
            .ok_or_else(|| ExprError::missing("mesh part", name))
    }

    /// The index of the region called `name`.
    pub fn region(&self, name: &str) -> ExprResult<u16> {
        self.template.regions().find(name).ok_or_else(|| ExprError::missing("region", name))
    }

    /// Whether node `n` lies in region `r` or one of its descendants.
    #[must_use]
    pub fn in_region(&self, n: usize, r: u16) -> bool {
        self.template.regions().is_within(self.region_of(n), r)
    }

    /// The rest position of the landmark called `name`.
    pub fn landmark(&self, name: &str) -> ExprResult<Vec3> {
        let l = self
            .template
            .landmarks()
            .iter()
            .find(|l| l.name == name)
            .ok_or_else(|| ExprError::missing("landmark", name))?;
        l.position(self.template.mesh().positions()).ok_or_else(|| ExprError::missing("landmark vertex", name))
    }

    /// The node carrying most of the landmark called `name`.
    pub fn landmark_node(&self, name: &str) -> ExprResult<u32> {
        let l = self
            .template
            .landmarks()
            .iter()
            .find(|l| l.name == name)
            .ok_or_else(|| ExprError::missing("landmark", name))?;
        let mut best = 0;
        for k in 1..3 {
            if l.weights[k] > l.weights[best] {
                best = k;
            }
        }
        let n = self.node_of(l.vertices[best] as usize);
        if n == u32::MAX {
            return Err(ExprError::missing("landmark vertex", name));
        }
        Ok(n)
    }

    /// The rest positions of the landmarks `prefix.01`, `prefix.02`, ... (until one is missing).
    #[must_use]
    pub fn landmark_series(&self, prefix: &str) -> Vec<Vec3> {
        let mut out = Vec::new();
        for k in 1..100 {
            match self.landmark(&format!("{prefix}.{k:02}")) {
                Ok(p) => out.push(p),
                Err(_) => break,
            }
        }
        out
    }

    /// The index of the joint called `name`.
    pub fn joint_index(&self, name: &str) -> ExprResult<u16> {
        self.template.skeleton().find(name).ok_or_else(|| ExprError::missing("joint", name))
    }

    /// The rest world position of the joint called `name`.
    pub fn joint(&self, name: &str) -> ExprResult<Vec3> {
        let j = self.joint_index(name)?;
        self.rest_world.get(j as usize).map(|t| t.translation).ok_or_else(|| ExprError::missing("joint", name))
    }

    /// The rest world transforms of all joints.
    #[must_use]
    pub fn rest_world(&self) -> &[Transform] {
        &self.rest_world
    }

    /// The skin weight joint `joint` has on node `n` (0 when it is not an influence).
    #[must_use]
    pub fn skin_weight(&self, n: usize, joint: u16) -> f32 {
        let v = self.rep.get(n).copied().unwrap_or(u32::MAX) as usize;
        let (joints, weights) = self.template.skin().vertex(v);
        joints.iter().zip(weights).filter(|(j, _)| **j == joint).map(|(_, w)| *w).sum()
    }

    /// Seeds along a polyline of landmarks: the nodes of the shortest edge paths between
    /// consecutive landmarks (so the seeds follow the surface, e.g. the right lip at the contact).
    pub fn line_seeds(&self, names: &[String]) -> ExprResult<Vec<u32>> {
        let mut nodes = Vec::with_capacity(names.len());
        for n in names {
            nodes.push(self.landmark_node(n)?);
        }
        let mut out: Vec<u32> = Vec::new();
        for w in nodes.windows(2) {
            out.extend(self.path(w[0], w[1], 0.2));
        }
        if let Some(&last) = nodes.last() {
            out.push(last);
        }
        out.sort_unstable();
        out.dedup();
        Ok(out)
    }

    /// The nodes of a shortest edge path from `a` to `b` (both included); just `[a]` when `b` is
    /// not reachable within `limit` metres.
    #[must_use]
    pub fn path(&self, a: u32, b: u32, limit: f32) -> Vec<u32> {
        let n = self.pos.len();
        if a as usize >= n || b as usize >= n {
            return Vec::new();
        }
        let mut dist: Vec<f32> = vec![f32::INFINITY; n];
        let mut prev: Vec<u32> = vec![u32::MAX; n];
        let mut heap = BinaryHeap::new();
        dist[a as usize] = 0.0;
        heap.push(Reverse((0_u32, a)));
        while let Some(Reverse((bits, u))) = heap.pop() {
            let du = f32::from_bits(bits);
            if du > dist[u as usize] {
                continue;
            }
            if u == b || du > limit {
                break;
            }
            for &w in self.adj.row(u as usize) {
                let d = du + (self.pos[w as usize] - self.pos[u as usize]).length();
                if d < dist[w as usize] {
                    dist[w as usize] = d;
                    prev[w as usize] = u;
                    heap.push(Reverse((d.to_bits(), w)));
                }
            }
        }
        if prev[b as usize] == u32::MAX && a != b {
            return vec![a];
        }
        let mut out = vec![b];
        let mut cur = b;
        while cur != a {
            cur = prev[cur as usize];
            if cur == u32::MAX || out.len() > n {
                break;
            }
            out.push(cur);
        }
        out.reverse();
        out
    }

    /// Geodesic (edge-path) distances from `seeds` to every node, `f32::INFINITY` beyond `limit`.
    /// Non-negative finite distances order like their bit patterns, which the heap relies on.
    #[must_use]
    pub fn distances(&self, seeds: &[u32], limit: f32) -> Vec<f32> {
        let n = self.pos.len();
        let mut dist = vec![f32::INFINITY; n];
        let mut heap = BinaryHeap::new();
        for &s in seeds {
            if let Some(d) = dist.get_mut(s as usize) {
                *d = 0.0;
                heap.push(Reverse((0_u32, s)));
            }
        }
        while let Some(Reverse((bits, u))) = heap.pop() {
            let du = f32::from_bits(bits);
            if du > dist[u as usize] {
                continue;
            }
            for &w in self.adj.row(u as usize) {
                let d = du + (self.pos[w as usize] - self.pos[u as usize]).length();
                if d < dist[w as usize] && d <= limit {
                    dist[w as usize] = d;
                    heap.push(Reverse((d.to_bits(), w)));
                }
            }
        }
        dist
    }

    /// The node of part `part` nearest to `p` (Euclidean), if the part has nodes.
    #[must_use]
    pub fn nearest(&self, p: Vec3, part: u16) -> Option<u32> {
        let mut best: Option<(f32, u32)> = None;
        for (i, &q) in self.pos.iter().enumerate() {
            if self.part[i] != part {
                continue;
            }
            let d = (q - p).length_squared();
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, i as u32));
            }
        }
        best.map(|(_, i)| i)
    }

    /// Expands per-node displacements to `(vertex, displacement)` entries, every vertex of a node
    /// getting the node's displacement; displacements shorter than `epsilon` are dropped.
    #[must_use]
    pub fn vertex_entries(&self, node_disp: &[Vec3], epsilon: f32) -> Vec<(u32, Vec3)> {
        let mut out = Vec::new();
        for (v, &n) in self.node_of.iter().enumerate() {
            if let Some(&d) = node_disp.get(n as usize)
                && d.length() > epsilon
            {
                out.push((v as u32, d));
            }
        }
        out
    }
}

/// A smooth ramp: 1 up to `inner`, 0 from `outer` on, a C1 smoothstep between.
#[must_use]
pub fn falloff(d: f32, inner: f32, outer: f32) -> f32 {
    if d.is_nan() || d >= outer {
        return 0.0;
    }
    if d <= inner || outer <= inner {
        return 1.0;
    }
    let t = (d - inner) / (outer - inner);
    1.0 - t * t * (3.0 - 2.0 * t)
}

/// The share of side `side` at lateral position `x` (metres, +X is the character's left): a
/// smoothstep from 0 to 1 over `[-width, width]` around the midline; the left and right shares of
/// one point sum to 1, the centre has 1 for [`Side::Center`].
#[must_use]
pub fn side_weight(x: f32, side: Side, width: f32) -> f32 {
    let s = match side {
        Side::Left => x,
        Side::Right => -x,
        Side::Center => return 1.0,
    };
    if width <= 0.0 {
        return if s > 0.0 {
            1.0
        } else if s < 0.0 {
            0.0
        } else {
            0.5
        };
    }
    let t = ((s + width) / (2.0 * width)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::template::fixtures;

    #[test]
    fn falloff_and_sides() {
        assert_eq!(falloff(0.0, 0.01, 0.02), 1.0);
        assert_eq!(falloff(0.02, 0.01, 0.02), 0.0);
        assert!((falloff(0.015, 0.01, 0.02) - 0.5).abs() < 1e-6);
        assert_eq!(falloff(f32::NAN, 0.0, 1.0), 0.0);
        for x in [-0.02, -0.003, 0.0, 0.001, 0.05] {
            let s = side_weight(x, Side::Left, 0.005) + side_weight(x, Side::Right, 0.005);
            assert!((s - 1.0).abs() < 1e-6);
        }
        assert_eq!(side_weight(0.01, Side::Left, 0.005), 1.0);
        assert_eq!(side_weight(0.01, Side::Right, 0.005), 0.0);
        assert_eq!(side_weight(0.0, Side::Left, 0.0), 0.5);
        assert_eq!(side_weight(0.3, Side::Center, 0.0), 1.0);
    }

    #[test]
    fn welds_seams_and_measures_geodesics() {
        let t = Template::new(fixtures::cube_template()).unwrap();
        let f = FaceField::new(&t).unwrap();
        // The cube's seam copies weld to one node each.
        let mesh = t.mesh();
        let distinct: std::collections::BTreeSet<u32> = mesh.position_keys().iter().copied().collect();
        assert_eq!(f.node_count(), distinct.len());
        for v in 0..mesh.vertex_count() {
            let n = f.node_of(v) as usize;
            assert_eq!(f.positions()[n], mesh.positions()[v]);
        }
        let d = f.distances(&[0], 10.0);
        assert_eq!(d[0], 0.0);
        assert!(d.iter().all(|x| x.is_finite()));
        let far = (0..f.node_count()).max_by(|&a, &b| d[a].total_cmp(&d[b])).unwrap();
        let path = f.path(0, far as u32, 10.0);
        assert_eq!(f.path(0, far as u32, 0.1), vec![0]);
        assert_eq!(path.first(), Some(&0));
        assert_eq!(path.last(), Some(&(far as u32)));
        let len: f32 =
            path.windows(2).map(|w| (f.positions()[w[1] as usize] - f.positions()[w[0] as usize]).length()).sum();
        assert!((len - d[far]).abs() < 1e-5);
        assert_eq!(f.distances(&[0], 0.0).iter().filter(|x| x.is_finite()).count(), 1);
        assert!(f.landmark("no-such").is_err());
        assert!(f.joint("no-such").is_err());
        assert!(f.region("no-such").is_err());
        let entries = f.vertex_entries(&vec![Vec3::X; f.node_count()], 0.0);
        assert_eq!(entries.len(), mesh.vertex_count());
    }
}
