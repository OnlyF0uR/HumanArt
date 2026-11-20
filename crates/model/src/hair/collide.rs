//! Closest points on a triangle mesh, the signed distance grid strands are pushed out of the head
//! with, and the penetration check of `[AC-WP17-1]` ("no strands inside the head mesh beyond
//! tolerance").
//!
//! - [`TriangleGrid`]: a uniform grid of triangle buckets with an exact closest-point query
//!   (expanding rings of cells, stopped once no unvisited cell can hold anything closer).
//! - [`SdfGrid`]: signed distances (negative inside) at the nodes of a regular grid around a
//!   closed mesh: exact in a band of two cells around the surface, chamfer-propagated outside the
//!   band, signed by a flood fill from the grid boundary. Sampled trilinearly with an analytic
//!   gradient; `groom_sdf` in `shaders/groom/common.wgsl` is the same interpolation.
//! - [`penetration`]: exact signed distances of strand points against the mesh.

use humanart_math::Vec3;

use super::error::{HairError, HairResult};

/// The most cells along one axis of a [`TriangleGrid`] or [`SdfGrid`].
pub const MAX_GRID_AXIS: u32 = 256;
/// The most nodes of an [`SdfGrid`].
pub const MAX_SDF_NODES: usize = 1 << 23;

/// The closest point of a mesh to a query point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosestPoint {
    /// The triangle it lies on.
    pub triangle: u32,
    /// Its barycentric coordinates `(u, v)` (weights of corners 1 and 2).
    pub bary: [f32; 2],
    /// The point.
    pub point: Vec3,
    /// The distance to the query point, signed by the triangle's face normal (negative behind).
    pub signed_distance: f32,
}

/// The closest point to `p` on triangle `a b c` and its barycentric `(u, v)` (Ericson, Real-Time
/// Collision Detection, 5.1.5).
pub fn closest_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Vec3, [f32; 2]) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, [0.0, 0.0]);
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, [1.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, [v, 0.0]);
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, [0.0, 1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, [0.0, w]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w, [1.0 - w, w]);
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (a + ab * v + ac * w, [v, w])
}

/// A uniform grid of triangle buckets over a mesh, for exact closest-point queries.
#[derive(Clone, Debug)]
pub struct TriangleGrid {
    positions: Vec<Vec3>,
    triangles: Vec<[u32; 3]>,
    origin: Vec3,
    cell: f32,
    dims: [u32; 3],
    starts: Vec<u32>,
    items: Vec<u32>,
}

impl TriangleGrid {
    /// Buckets `triangles` (indices into `positions`) in cells of about `cell` metres (made larger
    /// if an axis would exceed [`MAX_GRID_AXIS`] cells).
    pub fn new(positions: &[Vec3], triangles: &[[u32; 3]], cell: f32) -> HairResult<Self> {
        if triangles.is_empty() || !(cell.is_finite() && cell > 0.0) {
            return Err(HairError::invalid("triangle grid", 0, "no triangles or a bad cell size"));
        }
        if let Some(i) = triangles.iter().position(|t| t.iter().any(|&v| v as usize >= positions.len())) {
            return Err(HairError::invalid("triangle", i, "vertex index out of range"));
        }
        if let Some(i) = positions.iter().position(|p| !p.is_finite()) {
            return Err(HairError::invalid("vertex", i, "not finite"));
        }
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for t in triangles {
            for &v in t {
                lo = lo.min(positions[v as usize]);
                hi = hi.max(positions[v as usize]);
            }
        }
        let extent = (hi - lo).max_element().max(1e-6);
        let cell = cell.max(extent / (MAX_GRID_AXIS - 1) as f32);
        let origin = lo - Vec3::splat(cell * 0.5);
        let axis = |e: f32| ((e / cell).floor() as u32 + 2).min(MAX_GRID_AXIS);
        let size = hi - lo;
        let dims = [axis(size.x), axis(size.y), axis(size.z)];
        let mut grid = Self {
            positions: positions.to_vec(),
            triangles: triangles.to_vec(),
            origin,
            cell,
            dims,
            starts: Vec::new(),
            items: Vec::new(),
        };
        let count = dims.iter().map(|&d| d as usize).product::<usize>();
        let mut counts = vec![0_u32; count + 1];
        for pass in 0..2 {
            for (t, tri) in grid.triangles.iter().enumerate() {
                let (a, b, c) =
                    (grid.positions[tri[0] as usize], grid.positions[tri[1] as usize], grid.positions[tri[2] as usize]);
                let (l, h) = (grid.cell_of(a.min(b).min(c)), grid.cell_of(a.max(b).max(c)));
                for z in l[2]..=h[2] {
                    for y in l[1]..=h[1] {
                        for x in l[0]..=h[0] {
                            let i = grid.index([x, y, z]);
                            if pass == 0 {
                                counts[i + 1] += 1;
                            } else {
                                let slot = counts[i] as usize;
                                grid.items[slot] = t as u32;
                                counts[i] += 1;
                            }
                        }
                    }
                }
            }
            if pass == 0 {
                for i in 0..count {
                    counts[i + 1] += counts[i];
                }
                grid.starts.clone_from(&counts);
                grid.items = vec![0; counts[count] as usize];
            }
        }
        Ok(grid)
    }

    /// The cell containing `p`, clamped to the grid.
    fn cell_of(&self, p: Vec3) -> [u32; 3] {
        let q = (p - self.origin) / self.cell;
        let c = |v: f32, d: u32| (v.floor().max(0.0) as u32).min(d - 1);
        [c(q.x, self.dims[0]), c(q.y, self.dims[1]), c(q.z, self.dims[2])]
    }

    fn index(&self, c: [u32; 3]) -> usize {
        c[0] as usize + self.dims[0] as usize * (c[1] as usize + self.dims[1] as usize * c[2] as usize)
    }

    /// The mesh positions.
    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    /// The mesh triangles.
    pub fn triangles(&self) -> &[[u32; 3]] {
        &self.triangles
    }

    /// The face normal of triangle `t` (unit, or zero when degenerate or out of range).
    pub fn face_normal(&self, t: u32) -> Vec3 {
        self.triangles.get(t as usize).map_or(Vec3::ZERO, |tri| {
            let (a, b, c) =
                (self.positions[tri[0] as usize], self.positions[tri[1] as usize], self.positions[tri[2] as usize]);
            (b - a).cross(c - a).normalize_or_zero()
        })
    }

    /// The exact closest point of the mesh to `p`.
    pub fn closest(&self, p: Vec3) -> Option<ClosestPoint> {
        let centre = self.cell_of(p);
        let mut best: Option<(f32, u32, Vec3, [f32; 2])> = None;
        let max_ring = self.dims.iter().copied().max().unwrap_or(1);
        // Every point X of the grid box is at least max(outside, ring * cell) from p once rings
        // 0..=ring are done: projecting onto the (convex) box does not increase distances.
        let outside = {
            let q = p - self.origin;
            let hi = Vec3::new(self.dims[0] as f32, self.dims[1] as f32, self.dims[2] as f32) * self.cell;
            (q - q.clamp(Vec3::ZERO, hi)).length()
        };
        for ring in 0..=max_ring {
            let r = ring as i64;
            for dz in -r..=r {
                for dy in -r..=r {
                    for dx in -r..=r {
                        if dx.abs().max(dy.abs()).max(dz.abs()) != r {
                            continue;
                        }
                        let c = [centre[0] as i64 + dx, centre[1] as i64 + dy, centre[2] as i64 + dz];
                        if c.iter().zip(&self.dims).any(|(&v, &d)| v < 0 || v >= i64::from(d)) {
                            continue;
                        }
                        let i = self.index([c[0] as u32, c[1] as u32, c[2] as u32]);
                        for &t in &self.items[self.starts[i] as usize..self.starts[i + 1] as usize] {
                            let tri = self.triangles[t as usize];
                            let (q, bary) = closest_on_triangle(
                                p,
                                self.positions[tri[0] as usize],
                                self.positions[tri[1] as usize],
                                self.positions[tri[2] as usize],
                            );
                            let d = (p - q).length();
                            if best.is_none_or(|(bd, bt, ..)| d < bd || (d == bd && t < bt)) {
                                best = Some((d, t, q, bary));
                            }
                        }
                    }
                }
            }
            if let Some((d, ..)) = best
                && d <= outside.max(ring as f32 * self.cell)
            {
                break;
            }
        }
        best.map(|(d, t, q, bary)| {
            let sign = if (p - q).dot(self.face_normal(t)) < 0.0 { -1.0 } else { 1.0 };
            ClosestPoint { triangle: t, bary, point: q, signed_distance: sign * d }
        })
    }
}

/// Signed distances on a regular grid around a closed mesh (negative inside).
#[derive(Clone, Debug, PartialEq)]
pub struct SdfGrid {
    /// The world position of node `(0, 0, 0)`.
    pub origin: Vec3,
    /// The node spacing in metres.
    pub cell: f32,
    /// Nodes per axis (each at least 2).
    pub dims: [u32; 3],
    /// Distances, x fastest: node `(x, y, z)` is at `x + dims[0] (y + dims[1] z)`.
    pub values: Vec<f32>,
}

impl SdfGrid {
    /// The signed distance field of the closed mesh in `mesh` with nodes `cell` metres apart and a
    /// margin of `margin` metres (at least three cells) around the mesh bounds.
    pub fn build(mesh: &TriangleGrid, cell: f32, margin: f32) -> HairResult<Self> {
        if !(cell.is_finite() && cell > 0.0 && margin.is_finite()) {
            return Err(HairError::invalid("distance grid", 0, "bad cell size or margin"));
        }
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for t in mesh.triangles() {
            for &v in t {
                lo = lo.min(mesh.positions()[v as usize]);
                hi = hi.max(mesh.positions()[v as usize]);
            }
        }
        let margin = margin.max(3.0 * cell);
        let origin = lo - Vec3::splat(margin);
        let size = hi - lo + Vec3::splat(2.0 * margin);
        let axis = |e: f32| (e / cell).ceil() as u64 + 1;
        let d64 = [axis(size.x), axis(size.y), axis(size.z)];
        if d64.iter().any(|&d| d > u64::from(MAX_GRID_AXIS)) || d64.iter().product::<u64>() > MAX_SDF_NODES as u64 {
            return Err(HairError::limit("distance grid nodes", MAX_SDF_NODES as u64, d64.iter().product()));
        }
        let dims = [d64[0] as u32, d64[1] as u32, d64[2] as u32];
        let (nx, ny, nz) = (dims[0] as usize, dims[1] as usize, dims[2] as usize);
        let at = |x: usize, y: usize, z: usize| x + nx * (y + ny * z);
        let node = |x: usize, y: usize, z: usize| origin + Vec3::new(x as f32, y as f32, z as f32) * cell;
        let mut dist = vec![f32::INFINITY; nx * ny * nz];
        let mut sign = vec![1.0_f32; nx * ny * nz];
        // Exact distances in a band of two cells around every triangle.
        let band = 2.0 * cell;
        let to_node = |v: f32, n: usize| ((v / cell).max(0.0) as usize).min(n - 1);
        for (t, tri) in mesh.triangles().iter().enumerate() {
            let p = mesh.positions();
            let (a, b, c) = (p[tri[0] as usize], p[tri[1] as usize], p[tri[2] as usize]);
            let normal = mesh.face_normal(t as u32);
            let l = a.min(b).min(c) - Vec3::splat(band) - origin;
            let h = a.max(b).max(c) + Vec3::splat(band) - origin;
            for z in to_node(l.z, nz)..=(to_node(h.z, nz) + 1).min(nz - 1) {
                for y in to_node(l.y, ny)..=(to_node(h.y, ny) + 1).min(ny - 1) {
                    for x in to_node(l.x, nx)..=(to_node(h.x, nx) + 1).min(nx - 1) {
                        let q = node(x, y, z);
                        let (closest, _) = closest_on_triangle(q, a, b, c);
                        let d = (q - closest).length();
                        let i = at(x, y, z);
                        if d < dist[i] {
                            dist[i] = d;
                            sign[i] = if (q - closest).dot(normal) < 0.0 { -1.0 } else { 1.0 };
                        }
                    }
                }
            }
        }
        let banded: Vec<bool> = dist.iter().map(|&d| d <= band).collect();
        for (i, b) in banded.iter().enumerate() {
            if !b {
                dist[i] = f32::INFINITY;
            }
        }
        // Outside = reachable from the grid boundary without crossing the band.
        let mut outside = vec![false; dist.len()];
        let mut stack = Vec::new();
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let edge = x == 0 || y == 0 || z == 0 || x == nx - 1 || y == ny - 1 || z == nz - 1;
                    if edge && !banded[at(x, y, z)] {
                        outside[at(x, y, z)] = true;
                        stack.push((x, y, z));
                    }
                }
            }
        }
        while let Some((x, y, z)) = stack.pop() {
            let neighbours = [
                (x.wrapping_sub(1), y, z),
                (x + 1, y, z),
                (x, y.wrapping_sub(1), z),
                (x, y + 1, z),
                (x, y, z.wrapping_sub(1)),
                (x, y, z + 1),
            ];
            for (a, b, c) in neighbours {
                if a < nx && b < ny && c < nz {
                    let i = at(a, b, c);
                    if !banded[i] && !outside[i] {
                        outside[i] = true;
                        stack.push((a, b, c));
                    }
                }
            }
        }
        // Chamfer propagation of the unsigned distance from the band, forward then backward.
        let w = [cell, cell * std::f32::consts::SQRT_2, cell * 3.0_f32.sqrt()];
        for _ in 0..2 {
            for backward in [false, true] {
                for zi in 0..nz {
                    for yi in 0..ny {
                        for xi in 0..nx {
                            let (x, y, z) =
                                if backward { (nx - 1 - xi, ny - 1 - yi, nz - 1 - zi) } else { (xi, yi, zi) };
                            let i = at(x, y, z);
                            if banded[i] {
                                continue;
                            }
                            let mut best = dist[i];
                            for dz in -1_i64..=1 {
                                for dy in -1_i64..=1 {
                                    for dx in -1_i64..=1 {
                                        let (a, b, c) = (x as i64 + dx, y as i64 + dy, z as i64 + dz);
                                        if a < 0 || b < 0 || c < 0 || a >= nx as i64 || b >= ny as i64 || c >= nz as i64
                                        {
                                            continue;
                                        }
                                        let k = (dx.abs() + dy.abs() + dz.abs()) as usize;
                                        if k == 0 {
                                            continue;
                                        }
                                        best = best.min(dist[at(a as usize, b as usize, c as usize)] + w[k - 1]);
                                    }
                                }
                            }
                            dist[i] = best;
                        }
                    }
                }
            }
        }
        let values = (0..dist.len())
            .map(|i| {
                let s = if banded[i] {
                    sign[i]
                } else if outside[i] {
                    1.0
                } else {
                    -1.0
                };
                s * dist[i].min(1e3)
            })
            .collect();
        Ok(Self { origin, cell, dims, values })
    }

    /// The trilinear signed distance at `p` and its gradient (clamped to the grid). Mirrors
    /// `groom_sdf` in WGSL.
    pub fn sample(&self, p: Vec3) -> (f32, Vec3) {
        let q = (p - self.origin) * (1.0 / self.cell);
        let hi = Vec3::new((self.dims[0] - 1) as f32, (self.dims[1] - 1) as f32, (self.dims[2] - 1) as f32);
        let q = q.clamp(Vec3::ZERO, hi);
        let cx = (q.x.floor() as u32).min(self.dims[0] - 2);
        let cy = (q.y.floor() as u32).min(self.dims[1] - 2);
        let cz = (q.z.floor() as u32).min(self.dims[2] - 2);
        let f = q - Vec3::new(cx as f32, cy as f32, cz as f32);
        let (nx, ny) = (self.dims[0] as usize, self.dims[1] as usize);
        let v = |x: u32, y: u32, z: u32| {
            self.values.get(x as usize + nx * (y as usize + ny * z as usize)).copied().unwrap_or(0.0)
        };
        let c000 = v(cx, cy, cz);
        let c100 = v(cx + 1, cy, cz);
        let c010 = v(cx, cy + 1, cz);
        let c110 = v(cx + 1, cy + 1, cz);
        let c001 = v(cx, cy, cz + 1);
        let c101 = v(cx + 1, cy, cz + 1);
        let c011 = v(cx, cy + 1, cz + 1);
        let c111 = v(cx + 1, cy + 1, cz + 1);
        let x00 = c000 + (c100 - c000) * f.x;
        let x10 = c010 + (c110 - c010) * f.x;
        let x01 = c001 + (c101 - c001) * f.x;
        let x11 = c011 + (c111 - c011) * f.x;
        let y0 = x00 + (x10 - x00) * f.y;
        let y1 = x01 + (x11 - x01) * f.y;
        let d = y0 + (y1 - y0) * f.z;
        let gx0 = (c100 - c000) + ((c110 - c010) - (c100 - c000)) * f.y;
        let gx1 = (c101 - c001) + ((c111 - c011) - (c101 - c001)) * f.y;
        let gx = gx0 + (gx1 - gx0) * f.z;
        let gy = (x10 - x00) + ((x11 - x01) - (x10 - x00)) * f.z;
        let gz = y1 - y0;
        (d, Vec3::new(gx, gy, gz) * (1.0 / self.cell))
    }
}

/// The result of checking strand points against the head mesh `[AC-WP17-1]`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PenetrationReport {
    /// Points checked.
    pub points: u64,
    /// Points deeper inside the mesh than the tolerance.
    pub inside: u64,
    /// The deepest point's depth in metres (0 when none is inside).
    pub max_depth: f32,
}

/// Checks `points` against the closed mesh in `mesh`: a point is inside beyond `tolerance` when its
/// exact signed distance is below `-tolerance`. `sdf` (optional) skips the exact query for points
/// clearly outside.
pub fn penetration(mesh: &TriangleGrid, sdf: Option<&SdfGrid>, points: &[Vec3], tolerance: f32) -> PenetrationReport {
    let mut report = PenetrationReport { points: points.len() as u64, ..PenetrationReport::default() };
    for &p in points {
        if let Some(grid) = sdf
            && grid.sample(p).0 > 2.0 * grid.cell
        {
            continue;
        }
        if let Some(hit) = mesh.closest(p)
            && hit.signed_distance < 0.0
        {
            let depth = -hit.signed_distance;
            report.max_depth = report.max_depth.max(depth);
            if depth > tolerance {
                report.inside += 1;
            }
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::surface::ProxyHead;

    #[test]
    fn closest_point_on_triangle_regions() {
        let (a, b, c) = (Vec3::ZERO, Vec3::X, Vec3::Y);
        assert_eq!(closest_on_triangle(Vec3::new(-1.0, -1.0, 0.0), a, b, c).0, a);
        assert_eq!(closest_on_triangle(Vec3::new(2.0, -0.5, 0.0), a, b, c).0, b);
        let (q, bary) = closest_on_triangle(Vec3::new(0.25, 0.25, 1.0), a, b, c);
        assert!((q - Vec3::new(0.25, 0.25, 0.0)).length() < 1e-6 && (bary[0] - 0.25).abs() < 1e-6);
        let (q, _) = closest_on_triangle(Vec3::new(1.0, 1.0, 0.0), a, b, c);
        assert!((q - Vec3::new(0.5, 0.5, 0.0)).length() < 1e-6);
    }

    fn head() -> (ProxyHead, TriangleGrid) {
        let head = ProxyHead::new(24).unwrap();
        let grid = TriangleGrid::new(head.surface.positions(), head.surface.triangles(), 0.01).unwrap();
        (head, grid)
    }

    #[test]
    fn closest_matches_brute_force_and_signs() {
        let (head, grid) = head();
        let mut rng = humanart_core::rng::Rng::new(5);
        for _ in 0..300 {
            let p =
                head.centre + Vec3::new(rng.range_f32(-0.2, 0.2), rng.range_f32(-0.2, 0.2), rng.range_f32(-0.2, 0.2));
            let hit = grid.closest(p).unwrap();
            let brute = grid
                .triangles()
                .iter()
                .map(|t| {
                    let ps = grid.positions();
                    (p - closest_on_triangle(p, ps[t[0] as usize], ps[t[1] as usize], ps[t[2] as usize]).0).length()
                })
                .fold(f32::MAX, f32::min);
            assert!((hit.signed_distance.abs() - brute).abs() < 1e-6, "{} {brute}", hit.signed_distance);
            let inside = ((p - head.centre) / head.radii).length() < 1.0;
            if brute > 2e-3 {
                assert_eq!(hit.signed_distance < 0.0, inside);
            }
        }
    }

    #[test]
    fn sdf_grid_matches_the_exact_distance_and_sign() {
        let (head, grid) = head();
        let sdf = SdfGrid::build(&grid, 0.005, 0.02).unwrap();
        let mut rng = humanart_core::rng::Rng::new(9);
        let mut worst = 0.0_f32;
        for _ in 0..500 {
            let p = head.centre
                + Vec3::new(rng.range_f32(-0.1, 0.1), rng.range_f32(-0.13, 0.13), rng.range_f32(-0.12, 0.12));
            let exact = grid.closest(p).unwrap().signed_distance;
            let (d, g) = sdf.sample(p);
            if exact.abs() < 0.01 {
                worst = worst.max((d - exact).abs());
                assert!((g.length() - 1.0).abs() < 0.3, "gradient {g:?}");
            }
            if exact.abs() > 0.01 {
                assert_eq!(d < 0.0, exact < 0.0, "sign at {p:?}");
            }
        }
        assert!(worst < 1e-3, "band error {worst}");
        // Points pushed along the gradient by the distance land near the surface.
        let p = head.centre + Vec3::new(0.0, 0.0, 0.08);
        let (d, g) = sdf.sample(p);
        let q = p - g.normalize() * d;
        assert!(grid.closest(q).unwrap().signed_distance.abs() < 1e-3);
    }

    #[test]
    fn penetration_counts_points_inside() {
        let (head, grid) = head();
        let sdf = SdfGrid::build(&grid, 0.005, 0.02).unwrap();
        let points = [
            head.centre,
            head.centre + Vec3::new(0.0, 0.3, 0.0),
            head.surface_point(Vec3::Y) + Vec3::new(0.0, -0.002, 0.0),
        ];
        let r = penetration(&grid, Some(&sdf), &points, 0.001);
        assert_eq!(r.points, 3);
        assert_eq!(r.inside, 2);
        assert!(r.max_depth > 0.07);
    }

    #[test]
    fn grids_refuse_bad_input() {
        assert!(TriangleGrid::new(&[Vec3::ZERO], &[[0, 0, 1]], 0.1).is_err());
        assert!(TriangleGrid::new(&[Vec3::ZERO; 3], &[], 0.1).is_err());
        assert!(TriangleGrid::new(&[Vec3::ZERO; 3], &[[0, 1, 2]], 0.0).is_err());
        let (_, grid) = head();
        assert!(SdfGrid::build(&grid, 1e-5, 0.0).is_err());
    }
}
