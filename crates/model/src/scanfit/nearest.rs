//! Closest points on a triangle mesh `[WP-25]`: the correspondence search of the registration.
//!
//! [`TriangleTree`] is a bounding-volume hierarchy over the triangles of a mesh (object-median
//! splits on the axis of largest centroid extent, at most [`LEAF_SIZE`] triangles per leaf, built
//! iteratively; median splits bound the depth by `⌈log₂(n / LEAF_SIZE)⌉ + 1`, inside the query
//! stack of [`STACK_SIZE`]). [`TriangleTree::nearest`] returns the closest point on the triangles a
//! filter accepts, within a maximum distance, visiting the nearer child first and pruning boxes
//! farther than the best hit so far. Allocation-free per query.

use humanart_math::Vec3;

use super::{Result, ScanFitError};

/// The most triangles in a leaf.
pub const LEAF_SIZE: usize = 4;
/// The query stack depth.
pub const STACK_SIZE: usize = 64;
/// The most triangles a tree may hold.
pub const MAX_TRIANGLES: usize = 1 << 25;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Node {
    min: Vec3,
    max: Vec3,
    /// Interior: the left child (the right one follows). Leaf: the first triangle.
    first: u32,
    /// Triangles of a leaf; 0 for an interior node.
    count: u32,
}

/// The closest point found by [`TriangleTree::nearest`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nearest {
    /// The triangle (index into the triangle list the tree was built from).
    pub triangle: u32,
    /// The closest point.
    pub point: Vec3,
    /// Barycentric weights of the triangle's corners at the point.
    pub bary: [f32; 3],
    /// The distance from the query point.
    pub distance: f32,
}

/// A bounding-volume hierarchy for closest-point queries. See the module documentation.
#[derive(Clone, Debug, PartialEq)]
pub struct TriangleTree {
    nodes: Vec<Node>,
    triangles: Vec<[Vec3; 3]>,
    ids: Vec<u32>,
}

/// The closest point of triangle `t` to `p` and its barycentric weights (Ericson, Real-Time
/// Collision Detection, 5.1.5). Degenerate triangles return their nearest corner or edge point.
pub fn closest_on_triangle(p: Vec3, t: &[Vec3; 3]) -> (Vec3, [f32; 3]) {
    let [a, b, c] = *t;
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, [1.0, 0.0, 0.0]);
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return (b, [0.0, 1.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, [1.0 - v, v, 0.0]);
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return (c, [0.0, 0.0, 1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, [1.0 - w, 0.0, w]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w, [0.0, 1.0 - w, w]);
    }
    let sum = va + vb + vc;
    if sum.abs() < f32::MIN_POSITIVE || !sum.is_finite() {
        // Degenerate: the nearest of the corners.
        let best = [(a, [1.0, 0.0, 0.0]), (b, [0.0, 1.0, 0.0]), (c, [0.0, 0.0, 1.0])]
            .into_iter()
            .min_by(|x, y| x.0.distance_squared(p).total_cmp(&y.0.distance_squared(p)));
        return best.unwrap_or((a, [1.0, 0.0, 0.0]));
    }
    let denom = 1.0 / sum;
    let (v, w) = (vb * denom, vc * denom);
    (a + ab * v + ac * w, [1.0 - v - w, v, w])
}

/// The squared distance from `p` to the box `[min, max]` (0 inside).
fn box_distance_squared(p: Vec3, min: Vec3, max: Vec3) -> f32 {
    let d = (min - p).max(p - max).max(Vec3::ZERO);
    d.length_squared()
}

impl TriangleTree {
    /// Builds the tree over `indices` (three per triangle) into `positions`.
    ///
    /// # Errors
    /// [`ScanFitError::Invalid`] for an index count that is not a multiple of 3, an index out of
    /// range or a non-finite position; [`ScanFitError::Limit`] beyond [`MAX_TRIANGLES`].
    pub fn build(positions: &[Vec3], indices: &[u32]) -> Result<Self> {
        if !indices.len().is_multiple_of(3) {
            return Err(ScanFitError::invalid("the index count is not a multiple of 3"));
        }
        let count = indices.len() / 3;
        if count > MAX_TRIANGLES {
            return Err(ScanFitError::limit("tree triangle count", MAX_TRIANGLES, count));
        }
        let mut triangles = Vec::with_capacity(count);
        for t in indices.as_chunks::<3>().0 {
            let mut tri = [Vec3::ZERO; 3];
            for (corner, &i) in tri.iter_mut().zip(t) {
                *corner = *positions
                    .get(i as usize)
                    .ok_or(ScanFitError::invalid("a triangle index is past the last vertex"))?;
                if !corner.is_finite() {
                    return Err(ScanFitError::invalid("a vertex position is not finite"));
                }
            }
            triangles.push(tri);
        }
        let centroid = |t: &[Vec3; 3]| (t[0] + t[1] + t[2]) / 3.0;
        let mut order: Vec<u32> = (0..count as u32).collect();
        let mut nodes = Vec::with_capacity(2 * count.div_ceil(LEAF_SIZE).max(1));
        nodes.push(Node { min: Vec3::ZERO, max: Vec3::ZERO, first: 0, count: 0 });
        // Work list of (node, start, end) ranges of `order`.
        let mut work = vec![(0_usize, 0_usize, count)];
        while let Some((node, start, end)) = work.pop() {
            let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            let (mut clo, mut chi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for &t in &order[start..end] {
                let tri = &triangles[t as usize];
                for &p in tri {
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
                let c = centroid(tri);
                clo = clo.min(c);
                chi = chi.max(c);
            }
            if end - start <= LEAF_SIZE {
                nodes[node] = Node { min: lo, max: hi, first: start as u32, count: (end - start) as u32 };
                continue;
            }
            let axis = (chi - clo).max_axis();
            let key = |t: u32| centroid(&triangles[t as usize]).to_array()[axis];
            order[start..end].sort_unstable_by(|&a, &b| key(a).total_cmp(&key(b)).then(a.cmp(&b)));
            let mid = start + (end - start) / 2;
            let left = nodes.len();
            nodes.push(Node { min: Vec3::ZERO, max: Vec3::ZERO, first: 0, count: 0 });
            nodes.push(Node { min: Vec3::ZERO, max: Vec3::ZERO, first: 0, count: 0 });
            nodes[node] = Node { min: lo, max: hi, first: left as u32, count: 0 };
            work.push((left, start, mid));
            work.push((left + 1, mid, end));
        }
        let reordered = order.iter().map(|&t| triangles[t as usize]).collect();
        Ok(Self { nodes, triangles: reordered, ids: order })
    }

    /// The number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// The closest point to `p` on a triangle that `accept` (given the triangle index) lets
    /// through, no farther than `max_distance`. `None` when there is none.
    pub fn nearest(&self, p: Vec3, max_distance: f32, accept: impl Fn(u32) -> bool) -> Option<Nearest> {
        if self.triangles.is_empty() || !p.is_finite() {
            return None;
        }
        let mut best_d2 = max_distance * max_distance;
        let mut best: Option<Nearest> = None;
        let mut stack = [0_u32; STACK_SIZE];
        let mut sp = 1;
        while sp > 0 {
            sp -= 1;
            let Some(node) = self.nodes.get(stack[sp] as usize) else { continue };
            if box_distance_squared(p, node.min, node.max) > best_d2 {
                continue;
            }
            if node.count > 0 {
                let first = node.first as usize;
                for k in first..first + node.count as usize {
                    let (Some(tri), Some(&id)) = (self.triangles.get(k), self.ids.get(k)) else { continue };
                    if !accept(id) {
                        continue;
                    }
                    let (q, bary) = closest_on_triangle(p, tri);
                    let d2 = q.distance_squared(p);
                    if d2 <= best_d2 {
                        best_d2 = d2;
                        best = Some(Nearest { triangle: id, point: q, bary, distance: d2.sqrt() });
                    }
                }
            } else if sp + 2 <= STACK_SIZE {
                let (l, r) = (node.first, node.first + 1);
                let dist =
                    |i: u32| self.nodes.get(i as usize).map_or(f32::MAX, |n| box_distance_squared(p, n.min, n.max));
                let (dl, dr) = (dist(l), dist(r));
                // Visit the nearer child first: push it last.
                let (first, second) = if dl <= dr { (l, r) } else { (r, l) };
                stack[sp] = second;
                stack[sp + 1] = first;
                sp += 2;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    #[test]
    fn wp25_closest_point_regions_of_a_triangle() {
        let t = [Vec3::ZERO, Vec3::X, Vec3::Y];
        let cases = [
            (Vec3::new(0.2, 0.2, 1.0), Vec3::new(0.2, 0.2, 0.0)),
            (Vec3::new(-1.0, -1.0, 0.0), Vec3::ZERO),
            (Vec3::new(2.0, -0.5, 0.0), Vec3::X),
            (Vec3::new(0.5, -1.0, 0.3), Vec3::new(0.5, 0.0, 0.0)),
            (Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.5, 0.5, 0.0)),
            (Vec3::new(-1.0, 0.5, 0.0), Vec3::new(0.0, 0.5, 0.0)),
        ];
        for (p, want) in cases {
            let (q, b) = closest_on_triangle(p, &t);
            assert!(q.abs_diff_eq(want, 1e-6), "{p:?} -> {q:?}");
            let r = t[0] * b[0] + t[1] * b[1] + t[2] * b[2];
            assert!(r.abs_diff_eq(q, 1e-6) && (b.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
        let degenerate = [Vec3::ZERO, Vec3::ZERO, Vec3::ZERO];
        assert_eq!(closest_on_triangle(Vec3::ONE, &degenerate).0, Vec3::ZERO);
    }

    #[test]
    fn wp25_tree_matches_brute_force() {
        let mut rng = Rng::new(25);
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        for t in 0..500_u32 {
            let c = Vec3::new(rng.range_f32(-1.0, 1.0), rng.range_f32(-1.0, 1.0), rng.range_f32(-1.0, 1.0));
            for _ in 0..3 {
                positions
                    .push(c + Vec3::new(rng.range_f32(-0.1, 0.1), rng.range_f32(-0.1, 0.1), rng.range_f32(-0.1, 0.1)));
            }
            indices.extend([t * 3, t * 3 + 1, t * 3 + 2]);
        }
        let tree = TriangleTree::build(&positions, &indices).unwrap();
        assert_eq!(tree.triangle_count(), 500);
        for _ in 0..200 {
            let p = Vec3::new(rng.range_f32(-1.5, 1.5), rng.range_f32(-1.5, 1.5), rng.range_f32(-1.5, 1.5));
            let odd = |t: u32| t % 2 == 1;
            let got = tree.nearest(p, 10.0, odd).unwrap();
            let want = (0..500_u32)
                .filter(|&t| odd(t))
                .map(|t| {
                    let tri = [positions[t as usize * 3], positions[t as usize * 3 + 1], positions[t as usize * 3 + 2]];
                    closest_on_triangle(p, &tri).0.distance(p)
                })
                .fold(f32::MAX, f32::min);
            assert!((got.distance - want).abs() < 1e-5, "{} vs {want}", got.distance);
            assert!(odd(got.triangle));
        }
        assert!(tree.nearest(Vec3::splat(5.0), 0.1, |_| true).is_none());
        assert!(TriangleTree::build(&positions, &[0, 1]).is_err());
        assert!(TriangleTree::build(&positions, &[0, 1, 9999]).is_err());
    }
}
