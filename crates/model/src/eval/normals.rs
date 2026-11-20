//! Step 5 of the evaluation: vertex normals and tangents `[EVAL-04]`.
//!
//! The two passes the GPU runs, written for the CPU. Corners are numbered `3·t + k` (corner `k` of
//! triangle `t` of the triangle list).
//!
//! 1. **Triangle pass**: for every triangle `(a, b, c)` its unit normal
//!    `normalize((b − a) × (c − a))`, its interior angle at each of its three corners, and its unit
//!    tangent and bitangent from the UV derivatives: with `e1 = b − a`, `e2 = c − a`, `(du1, dv1)`,
//!    `(du2, dv2)` the UV differences and `s = sign(du1·dv2 − du2·dv1)`:
//!    `T = normalize(s·(e1·dv2 − e2·dv1))`, `B = normalize(s·(e2·du1 − e1·du2))`. A degenerate
//!    triangle (zero area, or collapsed UVs for `T` and `B`) gets zero vectors.
//! 2. **Vertex gather** (angle-weighted, so the result does not depend on how a quad was split):
//!    - normal of vertex `v` = `normalize(Σ angle[c] · N[c / 3])` over the corners `c` of its
//!      [`normal_corner_gather`] row: the corners of every vertex in its normal class, so UV-seam
//!      duplicates share one smooth normal and hard-edge duplicates do not;
//!    - tangent of `v` = `Σ angle[c] · T[c / 3]` and `Σ angle[c] · B[c / 3]` over `v`'s own corners
//!      ([`own_corner_gather`]: UV seams keep their own tangents), made orthogonal to the normal
//!      (Gram-Schmidt) and normalised; `w = ±1` is the handedness: `w · (n × t)` points along
//!      increasing V (V runs down the tile).
//!
//! Degenerate cases are deterministic: a vertex whose normal sum is zero gets `+Z` (the direction
//! the character faces); a tangent that vanishes is replaced by `Vec3::any_orthonormal_vector` of
//! the normal.

use humanart_math::{Vec2, Vec3, Vec4};

use crate::mesh::Mesh;
use crate::topology::Csr;

/// Vertex → corners whose triangle normals it averages: every corner of a vertex sharing its normal
/// key (`indices` is the triangle list of `mesh`).
#[must_use]
pub fn normal_corner_gather(mesh: &Mesh, indices: &[u32]) -> Csr {
    let keys = mesh.normal_keys();
    let pairs: Vec<(u32, u32)> =
        indices.iter().enumerate().filter_map(|(c, &v)| keys.get(v as usize).map(|&k| (k, c as u32))).collect();
    let by_class = Csr::from_pairs(mesh.vertex_count(), &pairs);
    Csr::from_rows(keys.iter().map(|&k| by_class.row(k as usize)))
}

/// Vertex → its own corners in the triangle list `indices`.
#[must_use]
pub fn own_corner_gather(vertex_count: usize, indices: &[u32]) -> Csr {
    let pairs: Vec<(u32, u32)> = indices.iter().enumerate().map(|(c, &v)| (v, c as u32)).collect();
    Csr::from_pairs(vertex_count, &pairs)
}

/// Per-triangle output of the triangle pass, one entry per triangle (angles: three per triangle).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TriangleFrames {
    /// Unit normal (zero for a degenerate triangle).
    pub normals: Vec<Vec3>,
    /// Unit tangent along increasing U (zero when undefined).
    pub tangents: Vec<Vec3>,
    /// Unit bitangent along increasing V (zero when undefined).
    pub bitangents: Vec<Vec3>,
    /// Interior angle in radians at every corner (`3 · triangle_count`).
    pub angles: Vec<f32>,
}

impl TriangleFrames {
    /// Storage for `triangles` triangles.
    #[must_use]
    pub fn new(triangles: usize) -> Self {
        Self {
            normals: vec![Vec3::ZERO; triangles],
            tangents: vec![Vec3::ZERO; triangles],
            bitangents: vec![Vec3::ZERO; triangles],
            angles: vec![0.0; 3 * triangles],
        }
    }

    /// The triangle pass over the triangle list `indices` (see the module comment). Triangles with
    /// an index outside `positions` or `uvs` get zeros. Allocation-free.
    pub fn compute(&mut self, positions: &[Vec3], uvs: &[Vec2], indices: &[u32]) {
        let tris = indices.as_chunks::<3>().0;
        for (t, tri) in tris.iter().enumerate().take(self.normals.len()) {
            let [ia, ib, ic] = tri.map(|i| i as usize);
            let corners = (positions.get(ia), positions.get(ib), positions.get(ic));
            let (Some(&pa), Some(&pb), Some(&pc)) = corners else {
                self.clear(t);
                continue;
            };
            let (e1, e2, e3) = (pb - pa, pc - pa, pc - pb);
            self.normals[t] = e1.cross(e2).normalize_or_zero();
            let angle = |u: Vec3, v: Vec3| {
                let a = u.angle_between(v);
                if a.is_finite() { a } else { 0.0 }
            };
            if let Some(slot) = self.angles.get_mut(3 * t..3 * t + 3) {
                slot[0] = angle(e1, e2);
                slot[1] = angle(-e1, e3);
                slot[2] = angle(-e2, -e3);
            }
            let (Some(&ua), Some(&ub), Some(&uc)) = (uvs.get(ia), uvs.get(ib), uvs.get(ic)) else {
                self.tangents[t] = Vec3::ZERO;
                self.bitangents[t] = Vec3::ZERO;
                continue;
            };
            let (d1, d2) = (ub - ua, uc - ua);
            let det = d1.x * d2.y - d2.x * d1.y;
            let s = if det > 0.0 {
                1.0
            } else if det < 0.0 {
                -1.0
            } else {
                0.0
            };
            self.tangents[t] = ((e1 * d2.y - e2 * d1.y) * s).normalize_or_zero();
            self.bitangents[t] = ((e2 * d1.x - e1 * d2.x) * s).normalize_or_zero();
        }
    }

    fn clear(&mut self, t: usize) {
        self.normals[t] = Vec3::ZERO;
        self.tangents[t] = Vec3::ZERO;
        self.bitangents[t] = Vec3::ZERO;
        if let Some(slot) = self.angles.get_mut(3 * t..3 * t + 3) {
            slot.fill(0.0);
        }
    }

    /// Gathers vertex normals over the rows of `gather` ([`normal_corner_gather`]) into `out`.
    /// Allocation-free.
    pub fn gather_normals(&self, gather: &Csr, out: &mut [Vec3]) {
        for (v, n) in out.iter_mut().enumerate() {
            let mut sum = Vec3::ZERO;
            for &c in gather.row(v) {
                let c = c as usize;
                if let (Some(&a), Some(&tn)) = (self.angles.get(c), self.normals.get(c / 3)) {
                    sum += tn * a;
                }
            }
            *n = sum.normalize_or(Vec3::Z);
        }
    }

    /// Gathers vertex tangents over the rows of `gather` ([`own_corner_gather`]), orthogonal to
    /// `normals`, into `out` (`w` = handedness). Allocation-free.
    pub fn gather_tangents(&self, gather: &Csr, normals: &[Vec3], out: &mut [Vec4]) {
        for (v, (tan, &n)) in out.iter_mut().zip(normals).enumerate() {
            let mut t = Vec3::ZERO;
            let mut b = Vec3::ZERO;
            for &c in gather.row(v) {
                let c = c as usize;
                if let (Some(&a), Some(&tt), Some(&tb)) =
                    (self.angles.get(c), self.tangents.get(c / 3), self.bitangents.get(c / 3))
                {
                    t += tt * a;
                    b += tb * a;
                }
            }
            let ortho = t - n * n.dot(t);
            let dir = match ortho.try_normalize() {
                Some(d) if ortho.length() > 1e-6 * t.length() => d,
                _ => n.any_orthonormal_vector(),
            };
            let w = if n.cross(dir).dot(b) < 0.0 { -1.0 } else { 1.0 };
            *tan = dir.extend(w);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::fixtures::{cube_split, grid};

    fn normals_of(mesh: &Mesh, positions: &[Vec3]) -> (Vec<Vec3>, Vec<Vec4>) {
        let tri = mesh.triangulate();
        let mut f = TriangleFrames::new(tri.len());
        f.compute(positions, mesh.uvs(), &tri.indices);
        let mut n = vec![Vec3::ZERO; mesh.vertex_count()];
        f.gather_normals(&normal_corner_gather(mesh, &tri.indices), &mut n);
        let mut t = vec![Vec4::ZERO; mesh.vertex_count()];
        f.gather_tangents(&own_corner_gather(mesh.vertex_count(), &tri.indices), &n, &mut t);
        (n, t)
    }

    #[test]
    fn eval_04_flat_grid_faces_plus_z_with_u_tangent() {
        let mesh = grid(3);
        let (n, t) = normals_of(&mesh, mesh.positions());
        assert!(n.iter().all(|v| v.distance(Vec3::Z) < 1e-6));
        // The grid has u = x and v = 1 - y: the tangent is +X, increasing V runs along -Y, and
        // n x t = Z x X = +Y, so w = -1.
        for v in &t {
            assert!(v.xyz().distance(Vec3::X) < 1e-6, "{v:?}");
            assert_eq!(v.w, -1.0);
        }
    }

    #[test]
    fn eval_04_hard_cube_has_face_normals_and_smooth_cube_averages() {
        for hard in [true, false] {
            let mesh = cube_split(hard);
            let (n, t) = normals_of(&mesh, mesh.positions());
            let centre = Vec3::splat(0.5);
            for (v, &nv) in n.iter().enumerate() {
                let outward = (mesh.positions()[v] - centre).normalize();
                if hard {
                    // A face normal: one axis, pointing outward.
                    assert!((nv.abs().max_element() - 1.0).abs() < 1e-6);
                    assert!(nv.dot(outward) > 0.5);
                } else {
                    // Angle weighting: the corner normal is the diagonal whatever the quad split.
                    assert!(nv.distance(outward) < 1e-5, "{v}: {nv:?} vs {outward:?}");
                }
                assert!(t[v].xyz().dot(nv).abs() < 1e-5 && (t[v].xyz().length() - 1.0).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn degenerate_input_is_deterministic() {
        let p = vec![Vec3::ZERO; 3];
        let uv = vec![Vec2::ZERO; 3];
        let idx = [0, 1, 2, 0, 1, 9];
        let mut f = TriangleFrames::new(2);
        f.normals.fill(Vec3::ONE);
        f.compute(&p, &uv, &idx);
        assert_eq!(f.normals, vec![Vec3::ZERO; 2]);
        assert_eq!(f.tangents, vec![Vec3::ZERO; 2]);
        let gather = Csr::from_pairs(3, &[(0, 0), (1, 4)]);
        let mut n = vec![Vec3::ONE; 3];
        f.gather_normals(&gather, &mut n);
        assert_eq!(n, vec![Vec3::Z; 3]);
        let mut t = vec![Vec4::ZERO; 3];
        f.gather_tangents(&gather, &n, &mut t);
        for v in &t {
            assert!(v.xyz().dot(Vec3::Z).abs() < 1e-6 && (v.xyz().length() - 1.0).abs() < 1e-6);
            assert!(v.is_finite());
        }
    }
}
