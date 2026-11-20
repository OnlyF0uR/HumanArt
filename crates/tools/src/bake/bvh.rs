//! A bounding-volume hierarchy over triangles and the ray casts of the ray-traced bakers.
//!
//! **Build:** object-median splits along the axis of largest centroid extent, triangles ordered by
//! `(centroid on that axis, triangle index)` (a total order, so the tree is deterministic), at most
//! [`LEAF_SIZE`] triangles per leaf. Median splits bound the depth by `⌈log₂(n / LEAF_SIZE)⌉ + 1`
//! (24 for the largest mesh a template may have), well inside the traversal stack of
//! [`STACK_SIZE`]. The build is iterative: no recursion.
//!
//! **Layout:** nodes in one array, the two children of an interior node next to each other
//! (`first`, `first + 1`); a leaf holds `count > 0` triangles from `first` in the reordered
//! triangle array. [`Bvh::gpu_nodes`] and [`Bvh::gpu_triangles`] give the same arrays as words
//! for `shaders/bake/bvh.wgsl`, which traverses them exactly as [`Bvh::intersect`] does.
//!
//! **Intersection:** Möller-Trumbore with inclusive edge tests (rays through a shared edge hit
//! one of the two triangles rather than slipping between them), slab tests on the boxes with
//! zero direction components replaced by `1e-30`.

use humanart_math::Vec3;

use super::{BakeError, Result};

/// The most triangles in a leaf.
pub const LEAF_SIZE: usize = 4;
/// The traversal stack depth (CPU and GPU).
pub const STACK_SIZE: usize = 64;
/// The most triangles a hierarchy may hold.
pub const MAX_TRIANGLES: usize = 1 << 24;

/// One node of the hierarchy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Node {
    /// Lower corner of the box.
    pub min: Vec3,
    /// Upper corner of the box.
    pub max: Vec3,
    /// Interior: the left child (the right one follows). Leaf: the first triangle.
    pub first: u32,
    /// The number of triangles of a leaf; 0 for an interior node.
    pub count: u32,
}

/// A ray hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    /// Distance along the (unit) direction.
    pub t: f32,
    /// The triangle, as an index into the triangle list the hierarchy was built from.
    pub triangle: u32,
    /// Barycentric weight of the triangle's second corner.
    pub b1: f32,
    /// Barycentric weight of the triangle's third corner.
    pub b2: f32,
    /// The ray hit the triangle's front (counter-clockwise) side.
    pub front: bool,
}

/// The hierarchy with its reordered triangles.
#[derive(Clone, Debug, PartialEq)]
pub struct Bvh {
    nodes: Vec<Node>,
    triangles: Vec<[Vec3; 3]>,
    ids: Vec<u32>,
}

/// A ray with its precomputed inverse direction.
#[derive(Clone, Copy, Debug)]
struct Ray {
    origin: Vec3,
    inv: Vec3,
}

impl Ray {
    fn new(origin: Vec3, dir: Vec3) -> Self {
        let safe = |c: f32| if c == 0.0 { 1e-30 } else { c };
        Self { origin, inv: Vec3::new(1.0 / safe(dir.x), 1.0 / safe(dir.y), 1.0 / safe(dir.z)) }
    }

    /// The entry distance into `node`'s box if the ray meets it within `[t_min, t_max]`.
    fn enter(&self, node: &Node, t_min: f32, t_max: f32) -> Option<f32> {
        let a = (node.min - self.origin) * self.inv;
        let b = (node.max - self.origin) * self.inv;
        let near = a.min(b).max_element().max(t_min);
        let far = a.max(b).min_element().min(t_max);
        (near <= far).then_some(near)
    }
}

/// Möller-Trumbore: `(t, b1, b2, front)` if the ray meets the triangle with `t` in `(t_min, t_max)`.
pub fn intersect_triangle(origin: Vec3, dir: Vec3, tri: &[Vec3; 3], t_min: f32, t_max: f32) -> Option<Hit> {
    let e1 = tri[1] - tri[0];
    let e2 = tri[2] - tri[0];
    let p = dir.cross(e2);
    let det = e1.dot(p);
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    let inv = 1.0 / det;
    let s = origin - tri[0];
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = dir.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    (t > t_min && t < t_max).then_some(Hit { t, triangle: 0, b1: u, b2: v, front: det > 0.0 })
}

impl Bvh {
    /// Builds the hierarchy over the triangle list `indices` (three per triangle) into `positions`.
    pub fn build(positions: &[Vec3], indices: &[u32]) -> Result<Self> {
        if !indices.len().is_multiple_of(3) {
            return Err(BakeError::invalid("the triangle list length is not a multiple of 3"));
        }
        let count = indices.len() / 3;
        if count > MAX_TRIANGLES {
            return Err(BakeError::invalid("too many triangles for a hierarchy"));
        }
        let mut triangles = Vec::with_capacity(count);
        for tri in indices.as_chunks::<3>().0 {
            let mut corners = [Vec3::ZERO; 3];
            for (c, &i) in corners.iter_mut().zip(tri) {
                *c = *positions.get(i as usize).ok_or(BakeError::invalid("a triangle index is out of range"))?;
                if !c.is_finite() {
                    return Err(BakeError::invalid("a position is not finite"));
                }
            }
            triangles.push(corners);
        }
        let centroids: Vec<Vec3> = triangles.iter().map(|t| (t[0] + t[1] + t[2]) * (1.0 / 3.0)).collect();
        let mut order: Vec<u32> = (0..count as u32).collect();
        let bounds = |range: &[u32]| {
            range.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), &t| {
                let tri = &triangles[t as usize];
                (lo.min(tri[0]).min(tri[1]).min(tri[2]), hi.max(tri[0]).max(tri[1]).max(tri[2]))
            })
        };
        let (min, max) = bounds(&order);
        let mut nodes = vec![Node { min, max, first: 0, count: count as u32 }];
        // Work list of (node, start, end); iterative, depth bounded by the median split.
        let mut work = vec![(0_usize, 0_usize, count)];
        while let Some((node, start, end)) = work.pop() {
            let len = end - start;
            if len <= LEAF_SIZE {
                nodes[node].first = start as u32;
                nodes[node].count = len as u32;
                continue;
            }
            let range = &mut order[start..end];
            let (clo, chi) = range.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), &t| {
                (lo.min(centroids[t as usize]), hi.max(centroids[t as usize]))
            });
            let axis = (chi - clo).max_axis();
            range.sort_unstable_by(|&a, &b| {
                let ca = centroids[a as usize].to_array()[axis];
                let cb = centroids[b as usize].to_array()[axis];
                ca.total_cmp(&cb).then(a.cmp(&b))
            });
            let mid = start + len / 2;
            let left = nodes.len();
            for (s, e) in [(start, mid), (mid, end)] {
                let (min, max) = bounds(&order[s..e]);
                nodes.push(Node { min, max, first: 0, count: 0 });
            }
            nodes[node].first = left as u32;
            nodes[node].count = 0;
            work.push((left, start, mid));
            work.push((left + 1, mid, end));
        }
        let triangles = order.iter().map(|&t| triangles[t as usize]).collect();
        Ok(Self { nodes, triangles, ids: order })
    }

    /// The nodes (node 0 is the root).
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// The number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// The deepest leaf (root = 1).
    pub fn depth(&self) -> usize {
        let mut deepest = 0;
        let mut stack = vec![(0_usize, 1_usize)];
        while let Some((n, d)) = stack.pop() {
            let Some(node) = self.nodes.get(n) else { continue };
            deepest = deepest.max(d);
            if node.count == 0 && !self.triangles.is_empty() {
                stack.push((node.first as usize, d + 1));
                stack.push((node.first as usize + 1, d + 1));
            }
        }
        deepest
    }

    /// The closest hit of the ray `origin + t · dir` with `t` in `(t_min, t_max)` (`dir` unit).
    pub fn intersect(&self, origin: Vec3, dir: Vec3, t_min: f32, t_max: f32) -> Option<Hit> {
        self.traverse(origin, dir, t_min, t_max, false)
    }

    /// Whether anything is hit within `(t_min, t_max)`.
    pub fn occluded(&self, origin: Vec3, dir: Vec3, t_min: f32, t_max: f32) -> bool {
        self.traverse(origin, dir, t_min, t_max, true).is_some()
    }

    fn traverse(&self, origin: Vec3, dir: Vec3, t_min: f32, mut t_max: f32, any: bool) -> Option<Hit> {
        if self.triangles.is_empty() {
            return None;
        }
        let ray = Ray::new(origin, dir);
        let mut stack = [0_u32; STACK_SIZE];
        let mut sp = 1;
        let mut best: Option<Hit> = None;
        while sp > 0 {
            sp -= 1;
            let Some(node) = self.nodes.get(stack[sp] as usize) else { continue };
            if ray.enter(node, t_min, t_max).is_none() {
                continue;
            }
            if node.count > 0 {
                let first = node.first as usize;
                for k in first..first + node.count as usize {
                    let (Some(tri), Some(&id)) = (self.triangles.get(k), self.ids.get(k)) else { continue };
                    if let Some(mut hit) = intersect_triangle(origin, dir, tri, t_min, t_max) {
                        hit.triangle = id;
                        t_max = hit.t;
                        best = Some(hit);
                        if any {
                            return best;
                        }
                    }
                }
            } else if sp + 2 <= STACK_SIZE {
                // Visit the nearer child first: push it last.
                let (l, r) = (node.first, node.first + 1);
                let near_l = self.nodes.get(l as usize).and_then(|n| ray.enter(n, t_min, t_max));
                let near_r = self.nodes.get(r as usize).and_then(|n| ray.enter(n, t_min, t_max));
                match (near_l, near_r) {
                    (Some(a), Some(b)) => {
                        let (first, second) = if a <= b { (l, r) } else { (r, l) };
                        stack[sp] = second;
                        stack[sp + 1] = first;
                        sp += 2;
                    }
                    (Some(_), None) => {
                        stack[sp] = l;
                        sp += 1;
                    }
                    (None, Some(_)) => {
                        stack[sp] = r;
                        sp += 1;
                    }
                    (None, None) => {}
                }
            }
        }
        best
    }

    /// The nodes as words for the GPU: 8 per node, `min.xyz, first, max.xyz, count` (floats as
    /// their bits).
    pub fn gpu_nodes(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.nodes.len() * 8);
        for n in &self.nodes {
            out.extend([n.min.x.to_bits(), n.min.y.to_bits(), n.min.z.to_bits(), n.first]);
            out.extend([n.max.x.to_bits(), n.max.y.to_bits(), n.max.z.to_bits(), n.count]);
        }
        out
    }

    /// The reordered triangles as words for the GPU: 12 per triangle, three `vec4`s (corner
    /// `xyz`; the first `w` holds the original triangle index).
    pub fn gpu_triangles(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.triangles.len().max(1) * 12);
        for (tri, &id) in self.triangles.iter().zip(&self.ids) {
            for (k, c) in tri.iter().enumerate() {
                out.extend([c.x.to_bits(), c.y.to_bits(), c.z.to_bits(), if k == 0 { id } else { 0 }]);
            }
        }
        if out.is_empty() {
            out.resize(12, 0);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use humanart_core::rng::Rng;

    use super::*;

    fn random_soup(rng: &mut Rng, n: usize) -> (Vec<Vec3>, Vec<u32>) {
        let mut p = Vec::new();
        for _ in 0..n {
            let c = Vec3::new(rng.range_f32(-1.0, 1.0), rng.range_f32(-1.0, 1.0), rng.range_f32(-1.0, 1.0));
            for _ in 0..3 {
                p.push(c + Vec3::new(rng.range_f32(-0.1, 0.1), rng.range_f32(-0.1, 0.1), rng.range_f32(-0.1, 0.1)));
            }
        }
        let idx = (0..3 * n as u32).collect();
        (p, idx)
    }

    fn brute(p: &[Vec3], idx: &[u32], o: Vec3, d: Vec3) -> Option<(f32, u32)> {
        let mut best: Option<(f32, u32)> = None;
        for (t, tri) in idx.as_chunks::<3>().0.iter().enumerate() {
            let corners = tri.map(|i| p[i as usize]);
            if let Some(h) = intersect_triangle(o, d, &corners, 0.0, f32::MAX)
                && best.is_none_or(|b| h.t < b.0)
            {
                best = Some((h.t, t as u32));
            }
        }
        best
    }

    #[test]
    fn bake_08_bvh_matches_brute_force() {
        let mut rng = Rng::new(11);
        let (p, idx) = random_soup(&mut rng, 500);
        let bvh = Bvh::build(&p, &idx).unwrap();
        assert!(bvh.depth() <= 10, "{}", bvh.depth());
        let mut hits = 0;
        for _ in 0..2000 {
            let o = Vec3::new(rng.range_f32(-2.0, 2.0), rng.range_f32(-2.0, 2.0), rng.range_f32(-2.0, 2.0));
            let d = Vec3::new(rng.normal_f32(), rng.normal_f32(), rng.normal_f32()).normalize();
            let a = bvh.intersect(o, d, 0.0, f32::MAX).map(|h| (h.t, h.triangle));
            let b = brute(&p, &idx, o, d);
            assert_eq!(a, b);
            assert_eq!(bvh.occluded(o, d, 0.0, f32::MAX), b.is_some());
            hits += usize::from(b.is_some());
        }
        assert!(hits > 50, "{hits}");
    }

    #[test]
    fn front_and_back_faces() {
        let tri = [Vec3::ZERO, Vec3::X, Vec3::Y];
        let down = intersect_triangle(Vec3::new(0.2, 0.2, 1.0), -Vec3::Z, &tri, 0.0, 10.0).unwrap();
        assert!(down.front && (down.t - 1.0).abs() < 1e-6);
        let up = intersect_triangle(Vec3::new(0.2, 0.2, -1.0), Vec3::Z, &tri, 0.0, 10.0).unwrap();
        assert!(!up.front);
        assert!(intersect_triangle(Vec3::new(0.2, 0.2, 1.0), -Vec3::Z, &tri, 0.0, 0.5).is_none());
    }

    #[test]
    fn bad_input_is_an_error_and_empty_is_fine() {
        assert!(Bvh::build(&[Vec3::ZERO], &[0, 0]).is_err());
        assert!(Bvh::build(&[Vec3::ZERO], &[0, 0, 1]).is_err());
        assert!(Bvh::build(&[Vec3::splat(f32::NAN); 3], &[0, 1, 2]).is_err());
        let empty = Bvh::build(&[], &[]).unwrap();
        assert!(empty.intersect(Vec3::ZERO, Vec3::X, 0.0, 1.0).is_none());
        assert_eq!(empty.gpu_triangles().len(), 12);
    }
}
