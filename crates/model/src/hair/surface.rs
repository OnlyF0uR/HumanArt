//! The skin surface grooms grow from, the root frames strands are attached to, and an analytic
//! proxy head `[HAIR-01]`.
//!
//! A strand root is a triangle of the surface plus barycentric coordinates `(u, v)` (the point is
//! `p0 (1 - u - v) + p1 u + p2 v`). Its **root frame** is built from the surface at that point: the
//! origin, the interpolated vertex normal `n`, the tangent `t` (the surface derivative along the UV
//! `u` direction, made orthogonal to `n`) and `b = n x t`. Strand control points are stored in this
//! frame, so when the surface deforms (identity, expression, pose) the strands ride it: the frame is
//! recomputed from the deformed positions and normals and the stored offsets are carried along.
//!
//! [`root_frame`] is the CPU reference of the WGSL `root_frame` in `shaders/groom/common.wgsl`; the
//! two follow the same operations in the same order.

use humanart_math::{Vec2, Vec3};

use super::error::{HairError, HairResult};

/// The most vertices a hair surface may have.
pub const MAX_SURFACE_VERTICES: usize = 1 << 24;
/// The most triangles a hair surface may have.
pub const MAX_SURFACE_TRIANGLES: usize = 1 << 25;

/// An orthonormal frame at a strand root: `origin + t x + b y + n z` is the world position of the
/// local offset `(x, y, z)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootFrame {
    /// The root position on the surface.
    pub origin: Vec3,
    /// The tangent (UV `u` direction), unit length.
    pub t: Vec3,
    /// The bitangent `n x t`, unit length.
    pub b: Vec3,
    /// The surface normal, unit length.
    pub n: Vec3,
}

impl RootFrame {
    /// The world position of the local offset `local`.
    #[inline]
    pub fn to_world(&self, local: Vec3) -> Vec3 {
        self.origin + self.t * local.x + self.b * local.y + self.n * local.z
    }

    /// The local offset of the world position `world`.
    #[inline]
    pub fn to_local(&self, world: Vec3) -> Vec3 {
        let d = world - self.origin;
        Vec3::new(d.dot(self.t), d.dot(self.b), d.dot(self.n))
    }

    /// The world direction of the local direction `local` (no translation).
    #[inline]
    pub fn dir_to_world(&self, local: Vec3) -> Vec3 {
        self.t * local.x + self.b * local.y + self.n * local.z
    }

    /// The local direction of the world direction `world`.
    #[inline]
    pub fn dir_to_local(&self, world: Vec3) -> Vec3 {
        Vec3::new(world.dot(self.t), world.dot(self.b), world.dot(self.n))
    }
}

/// `v` normalised, or `fallback` when `v` is (nearly) zero. Mirrors `groom_normalize_or` in WGSL.
#[inline]
pub fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let l2 = v.dot(v);
    if l2 > 1e-30 { v * (1.0 / l2.sqrt()) } else { fallback }
}

/// A unit tangent orthogonal to the unit normal `n`, used when the UV tangent degenerates.
#[inline]
pub fn fallback_tangent(n: Vec3) -> Vec3 {
    let a = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    normalize_or(a - n * n.dot(a), Vec3::Z)
}

/// The tangent coefficients of a triangle: `t = e1 c0 + e2 c1` is the surface derivative along
/// UV `u`, with `e1 = p1 - p0`, `e2 = p2 - p0`. Constant for a triangle (UVs do not deform), so
/// they are computed once at load. `(1, 0)` (the first edge) for a degenerate UV triangle.
pub fn tangent_coefficients(uv: [Vec2; 3]) -> [f32; 2] {
    let d1 = uv[1] - uv[0];
    let d2 = uv[2] - uv[0];
    let det = d1.x * d2.y - d2.x * d1.y;
    if det.abs() < 1e-12 { [1.0, 0.0] } else { [d2.y / det, -d1.y / det] }
}

/// The root frame at barycentric `bary = (u, v)` of the triangle with corner positions `p` and
/// vertex normals `n`, with tangent coefficients `coef` ([`tangent_coefficients`]).
pub fn root_frame(p: [Vec3; 3], n: [Vec3; 3], bary: [f32; 2], coef: [f32; 2]) -> RootFrame {
    let (u, v) = (bary[0], bary[1]);
    let w = 1.0 - u - v;
    let origin = p[0] * w + p[1] * u + p[2] * v;
    let e1 = p[1] - p[0];
    let e2 = p[2] - p[0];
    let face = normalize_or(e1.cross(e2), Vec3::Z);
    let normal = normalize_or(n[0] * w + n[1] * u + n[2] * v, face);
    let t0 = e1 * coef[0] + e2 * coef[1];
    let t1 = t0 - normal * normal.dot(t0);
    let t = if t1.dot(t1) > 1e-20 { normalize_or(t1, Vec3::X) } else { fallback_tangent(normal) };
    RootFrame { origin, t, b: normal.cross(t), n: normal }
}

/// A triangle surface hair grows from: positions, vertex normals and UVs (SoA), and triangles.
#[derive(Clone, Debug)]
pub struct HairSurface {
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    uvs: Vec<Vec2>,
    triangles: Vec<[u32; 3]>,
}

impl HairSurface {
    /// A surface from its arrays. Lengths must agree, indices be in range and values finite.
    pub fn new(positions: Vec<Vec3>, normals: Vec<Vec3>, uvs: Vec<Vec2>, triangles: Vec<[u32; 3]>) -> HairResult<Self> {
        let n = positions.len();
        if n > MAX_SURFACE_VERTICES {
            return Err(HairError::limit("surface vertex count", MAX_SURFACE_VERTICES as u64, n as u64));
        }
        if triangles.len() > MAX_SURFACE_TRIANGLES {
            return Err(HairError::limit(
                "surface triangle count",
                MAX_SURFACE_TRIANGLES as u64,
                triangles.len() as u64,
            ));
        }
        if normals.len() != n || uvs.len() != n {
            return Err(HairError::invalid("surface", 0, "positions, normals and uvs differ in length"));
        }
        if let Some(i) = positions.iter().zip(&normals).position(|(p, q)| !p.is_finite() || !q.is_finite()) {
            return Err(HairError::invalid("surface vertex", i, "not finite"));
        }
        if let Some(i) = uvs.iter().position(|uv| !uv.is_finite()) {
            return Err(HairError::invalid("surface vertex", i, "uv not finite"));
        }
        if let Some(i) = triangles.iter().position(|t| t.iter().any(|&v| v as usize >= n)) {
            return Err(HairError::invalid("surface triangle", i, "vertex index out of range"));
        }
        Ok(Self { positions, normals, uvs, triangles })
    }

    /// A surface from a flat triangle index list (three per triangle).
    pub fn from_index_list(
        positions: Vec<Vec3>,
        normals: Vec<Vec3>,
        uvs: Vec<Vec2>,
        indices: &[u32],
    ) -> HairResult<Self> {
        if !indices.len().is_multiple_of(3) {
            return Err(HairError::invalid("surface", 0, "index count is not a multiple of 3"));
        }
        let triangles = indices.as_chunks::<3>().0.to_vec();
        Self::new(positions, normals, uvs, triangles)
    }

    /// Vertex positions.
    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    /// Vertex normals.
    pub fn normals(&self) -> &[Vec3] {
        &self.normals
    }

    /// Vertex UVs.
    pub fn uvs(&self) -> &[Vec2] {
        &self.uvs
    }

    /// Triangles.
    pub fn triangles(&self) -> &[[u32; 3]] {
        &self.triangles
    }

    /// The vertex indices of triangle `t`.
    #[inline]
    pub fn triangle(&self, t: u32) -> Option<[u32; 3]> {
        self.triangles.get(t as usize).copied()
    }

    /// The corner positions of triangle `t`.
    pub fn corners(&self, t: u32) -> Option<[Vec3; 3]> {
        let [a, b, c] = self.triangle(t)?;
        Some([self.positions[a as usize], self.positions[b as usize], self.positions[c as usize]])
    }

    /// The tangent coefficients of triangle `t` ([`tangent_coefficients`]).
    pub fn tangent_coefficients(&self, t: u32) -> Option<[f32; 2]> {
        let [a, b, c] = self.triangle(t)?;
        Some(tangent_coefficients([self.uvs[a as usize], self.uvs[b as usize], self.uvs[c as usize]]))
    }

    /// The area of triangle `t` in square metres (0 for an index out of range).
    pub fn area(&self, t: u32) -> f32 {
        self.corners(t).map_or(0.0, |[a, b, c]| 0.5 * (b - a).cross(c - a).length())
    }

    /// The UV at barycentric `bary` of triangle `t`.
    pub fn uv_at(&self, t: u32, bary: [f32; 2]) -> Option<Vec2> {
        let [a, b, c] = self.triangle(t)?;
        let w = 1.0 - bary[0] - bary[1];
        Some(self.uvs[a as usize] * w + self.uvs[b as usize] * bary[0] + self.uvs[c as usize] * bary[1])
    }

    /// The position at barycentric `bary` of triangle `t`.
    pub fn point_at(&self, t: u32, bary: [f32; 2]) -> Option<Vec3> {
        let [a, b, c] = self.corners(t)?;
        Some(a * (1.0 - bary[0] - bary[1]) + b * bary[0] + c * bary[1])
    }

    /// The root frame at barycentric `bary` of triangle `t` on this surface.
    pub fn frame(&self, t: u32, bary: [f32; 2]) -> Option<RootFrame> {
        self.frame_on(t, bary, &self.positions, &self.normals)
    }

    /// The root frame at `bary` of triangle `t` with the vertices moved to `positions` / `normals`
    /// (the deformed surface; same vertex count). `None` for an index out of range.
    pub fn frame_on(&self, t: u32, bary: [f32; 2], positions: &[Vec3], normals: &[Vec3]) -> Option<RootFrame> {
        let [a, b, c] = self.triangle(t)?;
        let p = [*positions.get(a as usize)?, *positions.get(b as usize)?, *positions.get(c as usize)?];
        let n = [*normals.get(a as usize)?, *normals.get(b as usize)?, *normals.get(c as usize)?];
        Some(root_frame(p, n, bary, self.tangent_coefficients(t)?))
    }

    /// Replaces the vertex positions and normals (a re-evaluated surface with the same topology).
    pub fn set_vertices(&mut self, positions: &[Vec3], normals: &[Vec3]) -> HairResult<()> {
        if positions.len() != self.positions.len() || normals.len() != self.normals.len() {
            return Err(HairError::invalid("surface", 0, "vertex count changed"));
        }
        if let Some(i) = positions.iter().zip(normals).position(|(p, q)| !p.is_finite() || !q.is_finite()) {
            return Err(HairError::invalid("surface vertex", i, "not finite"));
        }
        self.positions.copy_from_slice(positions);
        self.normals.copy_from_slice(normals);
        Ok(())
    }

    /// The triangles whose three vertices all satisfy `keep` (e.g. "is in the scalp region").
    pub fn triangles_where(&self, mut keep: impl FnMut(u32) -> bool) -> Vec<u32> {
        (0..self.triangles.len() as u32).filter(|&t| self.triangles[t as usize].iter().all(|&v| keep(v))).collect()
    }

    /// For each vertex, a canonical vertex with the same position: UV seams duplicate vertices,
    /// and walks over the surface must cross them.
    pub fn welded(&self) -> Vec<u32> {
        let mut map: std::collections::HashMap<[u32; 3], u32> = std::collections::HashMap::new();
        self.positions
            .iter()
            .enumerate()
            .map(|(i, p)| *map.entry([p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]).or_insert(i as u32))
            .collect()
    }

    /// Per vertex, the distance in metres along the surface edges to the boundary of the triangle
    /// set `mask` (the hairline of a scalp mask); `f32::INFINITY` for vertices outside the mask.
    /// Seams are welded. Multi-source Dijkstra over the welded edge graph.
    pub fn distance_to_mask_boundary(&self, mask: &[u32]) -> Vec<f32> {
        use std::collections::{BinaryHeap, HashMap};
        let weld = self.welded();
        let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
        let mut inside = vec![false; self.positions.len()];
        for &t in mask {
            let Some(tri) = self.triangle(t) else { continue };
            for k in 0..3 {
                let (a, b) = (weld[tri[k] as usize], weld[tri[(k + 1) % 3] as usize]);
                inside[a as usize] = true;
                *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        let mut adjacency: Vec<Vec<(u32, f32)>> = vec![Vec::new(); self.positions.len()];
        let mut dist = vec![f32::INFINITY; self.positions.len()];
        let mut heap = BinaryHeap::new();
        for (&(a, b), &count) in &edges {
            let l = self.positions[a as usize].distance(self.positions[b as usize]);
            adjacency[a as usize].push((b, l));
            adjacency[b as usize].push((a, l));
            if count == 1 {
                for v in [a, b] {
                    if dist[v as usize] > 0.0 {
                        dist[v as usize] = 0.0;
                        heap.push(std::cmp::Reverse((0_u32, v)));
                    }
                }
            }
        }
        // Distances are non-negative, so their bit patterns order like the values.
        while let Some(std::cmp::Reverse((bits, v))) = heap.pop() {
            let d = f32::from_bits(bits);
            if d > dist[v as usize] {
                continue;
            }
            for &(w, l) in &adjacency[v as usize] {
                let nd = d + l;
                if nd < dist[w as usize] {
                    dist[w as usize] = nd;
                    heap.push(std::cmp::Reverse((nd.to_bits(), w)));
                }
            }
        }
        // Copy the welded result to the duplicates; vertices not in the mask stay infinite.
        (0..self.positions.len())
            .map(|v| if inside[weld[v] as usize] { dist[weld[v] as usize] } else { f32::INFINITY })
            .collect()
    }
}

/// An analytic proxy head: an ellipsoid with a UV layout, a scalp mask with a hairline, and the
/// brow, lid-margin and beard areas of a face. Used by tests, benchmarks and the command-line tool
/// while the template has no scalp; it is a stand-in for geometry, not a model of a real head.
#[derive(Clone, Debug)]
pub struct ProxyHead {
    /// The closed head surface (seam vertices duplicated for UVs).
    pub surface: HairSurface,
    /// The scalp triangles (hairline: forehead 3.5 cm above the centre, nape 6 cm below).
    pub scalp: Vec<u32>,
    /// The lower-face triangles (beard and moustache area).
    pub lower_face: Vec<u32>,
    /// The ellipsoid centre.
    pub centre: Vec3,
    /// The ellipsoid radii (half width, half height, half depth).
    pub radii: Vec3,
}

impl ProxyHead {
    /// A head with `rings` latitude rings and `2 rings` longitude segments (`rings` in 4..=512),
    /// centred at `(0, 1.6, 0)` with radii 7.5 x 11 x 9.5 cm, facing +Z.
    pub fn new(rings: u32) -> HairResult<Self> {
        if !(4..=512).contains(&rings) {
            return Err(HairError::limit("proxy head rings", 512, u64::from(rings)));
        }
        let centre = Vec3::new(0.0, 1.6, 0.0);
        let radii = Vec3::new(0.075, 0.11, 0.095);
        let segments = rings * 2;
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        // Poles are single vertices; rings 1..rings-1 have segments + 1 vertices (seam duplicate).
        let pole = |sign: f32, positions: &mut Vec<Vec3>, normals: &mut Vec<Vec3>, uvs: &mut Vec<Vec2>| {
            positions.push(centre + Vec3::new(0.0, sign * radii.y, 0.0));
            normals.push(Vec3::new(0.0, sign, 0.0));
            uvs.push(Vec2::new(0.5, if sign > 0.0 { 0.0 } else { 1.0 }));
        };
        pole(1.0, &mut positions, &mut normals, &mut uvs);
        for i in 1..rings {
            let theta = std::f32::consts::PI * i as f32 / rings as f32;
            for j in 0..=segments {
                // The seam column repeats column 0 bit for bit, so welding by position finds it.
                let phi = std::f32::consts::TAU * (j % segments) as f32 / segments as f32;
                let d = Vec3::new(theta.sin() * phi.sin(), theta.cos(), theta.sin() * phi.cos());
                positions.push(centre + d * radii);
                normals.push(normalize_or(d / (radii * radii), Vec3::Y));
                uvs.push(Vec2::new(j as f32 / segments as f32, i as f32 / rings as f32));
            }
        }
        pole(-1.0, &mut positions, &mut normals, &mut uvs);
        let ring = |i: u32, j: u32| 1 + (i - 1) * (segments + 1) + j;
        let bottom = positions.len() as u32 - 1;
        let mut triangles = Vec::new();
        for j in 0..segments {
            triangles.push([0, ring(1, j), ring(1, j + 1)]);
        }
        for i in 1..rings - 1 {
            for j in 0..segments {
                let (a, b, c, d) = (ring(i, j), ring(i, j + 1), ring(i + 1, j + 1), ring(i + 1, j));
                triangles.push([a, d, c]);
                triangles.push([a, c, b]);
            }
        }
        for j in 0..segments {
            triangles.push([bottom, ring(rings - 1, j + 1), ring(rings - 1, j)]);
        }
        let surface = HairSurface::new(positions, normals, uvs, triangles)?;
        let local = |v: u32| (surface.positions[v as usize] - centre) / radii;
        let scalp = surface.triangles_where(|v| {
            let l = local(v);
            l.y > 0.32 * l.z.max(0.0) - 0.55 * (-l.z).max(0.0) + 0.05 * (1.0 - l.z.abs())
        });
        let lower_face = surface.triangles_where(|v| {
            let l = local(v);
            l.z > 0.35 && l.y < -0.45 && l.y > -0.97
        });
        Ok(Self { surface, scalp, lower_face, centre, radii })
    }

    /// The point on the ellipsoid in the direction `d` from the centre.
    pub fn surface_point(&self, d: Vec3) -> Vec3 {
        let dir = normalize_or(d, Vec3::Z);
        // centre + r dir is on the ellipsoid when |r dir / radii| = 1.
        self.centre + dir * (1.0 / (dir / self.radii).length())
    }

    /// The brow line of one side (`side` +1 for the character's left, at +X; -1 for the right),
    /// from the inner to the outer end, 9 points on the surface.
    pub fn brow_curve(&self, side: f32) -> Vec<Vec3> {
        (0..9)
            .map(|k| {
                let s = k as f32 / 8.0;
                let x = side * (0.12 + 0.55 * s);
                let y = 0.02 + 0.06 * (std::f32::consts::PI * s).sin() - 0.03 * s;
                self.surface_point(Vec3::new(x, y, 1.0))
            })
            .collect()
    }

    /// The upper lid margin of one side, from the inner to the outer canthus, 9 points.
    pub fn lid_curve(&self, side: f32) -> Vec<Vec3> {
        (0..9)
            .map(|k| {
                let s = k as f32 / 8.0;
                let x = side * (0.17 + 0.42 * s);
                let y = -0.17 + 0.05 * (std::f32::consts::PI * s).sin();
                self.surface_point(Vec3::new(x, y, 1.0))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_frame_is_orthonormal_and_right_handed() {
        let p = [Vec3::ZERO, Vec3::X, Vec3::Y];
        let n = [Vec3::Z; 3];
        let coef = tangent_coefficients([Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)]);
        let f = root_frame(p, n, [0.25, 0.25], coef);
        assert!((f.origin - Vec3::new(0.25, 0.25, 0.0)).length() < 1e-6);
        assert!((f.t - Vec3::X).length() < 1e-6 && (f.n - Vec3::Z).length() < 1e-6);
        assert!((f.b - Vec3::Y).length() < 1e-6);
        let local = Vec3::new(0.1, -0.2, 0.3);
        assert!((f.to_local(f.to_world(local)) - local).length() < 1e-6);
        // A degenerate UV triangle falls back to an orthonormal tangent.
        let f = root_frame(p, n, [0.2, 0.2], tangent_coefficients([Vec2::ZERO; 3]));
        assert!(f.t.dot(f.n).abs() < 1e-6 && (f.t.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn surface_checks_its_input() {
        let ok = || (vec![Vec3::ZERO, Vec3::X, Vec3::Y], vec![Vec3::Z; 3], vec![Vec2::ZERO; 3]);
        let (p, n, uv) = ok();
        assert!(HairSurface::new(p, n, uv, vec![[0, 1, 3]]).is_err());
        let (p, n, uv) = ok();
        assert!(HairSurface::new(p, n[..2].to_vec(), uv, vec![[0, 1, 2]]).is_err());
        let (mut p, n, uv) = ok();
        p[1].x = f32::NAN;
        assert!(HairSurface::new(p, n, uv, vec![[0, 1, 2]]).is_err());
        let (p, n, uv) = ok();
        assert!(HairSurface::from_index_list(p, n, uv, &[0, 1]).is_err());
        let (p, n, uv) = ok();
        let s = HairSurface::from_index_list(p, n, uv, &[0, 1, 2]).unwrap();
        assert!((s.area(0) - 0.5).abs() < 1e-6 && s.area(7) == 0.0);
        assert!(s.frame(1, [0.0, 0.0]).is_none());
    }

    #[test]
    fn proxy_head_is_closed_and_has_a_scalp_with_a_hairline() {
        let head = ProxyHead::new(32).unwrap();
        let s = &head.surface;
        // Closed: every welded edge is used by exactly two triangles.
        let weld = s.welded();
        let mut edges = std::collections::HashMap::new();
        for t in s.triangles() {
            for k in 0..3 {
                let (a, b) = (weld[t[k] as usize], weld[t[(k + 1) % 3] as usize]);
                *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        assert!(edges.values().all(|&c| c == 2));
        assert!(!head.scalp.is_empty() && !head.lower_face.is_empty());
        let scalp_area: f32 = head.scalp.iter().map(|&t| s.area(t)).sum();
        let total: f32 = (0..s.triangles().len() as u32).map(|t| s.area(t)).sum();
        assert!(scalp_area > 0.3 * total && scalp_area < 0.7 * total, "{scalp_area} of {total}");
        // The hairline distance is 0 on the boundary, grows inwards and is infinite outside.
        let d = s.distance_to_mask_boundary(&head.scalp);
        let top = 0;
        assert!(d[top] > 0.05 && d[top].is_finite());
        let low = s.positions().iter().position(|p| p.y < 1.5).unwrap();
        assert!(d[low].is_infinite());
        assert!(d.contains(&0.0));
        // Seam duplicates get the same distance.
        for (v, &w) in weld.iter().enumerate() {
            assert_eq!(d[v].to_bits(), d[w as usize].to_bits());
        }
        for side in [1.0, -1.0] {
            for p in head.brow_curve(side).into_iter().chain(head.lid_curve(side)) {
                let l = (p - head.centre) / head.radii;
                assert!((l.length() - 1.0).abs() < 1e-4);
                assert!(p.x * side > 0.0);
            }
        }
    }
}
