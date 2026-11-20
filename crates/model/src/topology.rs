//! Edge and adjacency tables of a mesh, in compressed sparse row (CSR) layout `[EVAL-04]`.
//!
//! [`Topology`] is derived from a checked [`Mesh`] and its [`Triangulation`]; it is not stored in
//! the package. Edges are between **weld keys** (see [`crate::mesh`]), so a UV seam or hard edge does
//! not split an edge: watertightness and manifoldness are judged on the welded surface, and edge
//! strain `[EVAL-06]` is measured once per geometric edge.

use crate::mesh::{Mesh, Triangulation};

/// A compressed sparse row table: row `r` is `items[offsets[r]..offsets[r + 1]]`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Csr {
    offsets: Vec<u32>,
    items: Vec<u32>,
}

impl Csr {
    /// Builds a table of `rows` rows from `(row, item)` pairs; items keep their order within a row.
    /// Pairs whose row is out of range are dropped.
    #[must_use]
    pub fn from_pairs(rows: usize, pairs: &[(u32, u32)]) -> Self {
        let mut offsets = vec![0_u32; rows + 1];
        for &(r, _) in pairs {
            if let Some(o) = offsets.get_mut(r as usize + 1) {
                *o += 1;
            }
        }
        for r in 0..rows {
            offsets[r + 1] += offsets[r];
        }
        let mut cursor: Vec<u32> = offsets[..rows].to_vec();
        let mut items = vec![0_u32; offsets[rows] as usize];
        for &(r, item) in pairs {
            if let Some(c) = cursor.get_mut(r as usize) {
                items[*c as usize] = item;
                *c += 1;
            }
        }
        Self { offsets, items }
    }

    /// Builds a table from per-row lists.
    #[must_use]
    pub fn from_rows<'a>(rows: impl Iterator<Item = &'a [u32]>) -> Self {
        let mut offsets = vec![0_u32];
        let mut items = Vec::new();
        for row in rows {
            items.extend_from_slice(row);
            offsets.push(items.len() as u32);
        }
        Self { offsets, items }
    }

    /// How many rows the table has.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Row `r`; empty when `r` is out of range.
    #[must_use]
    pub fn row(&self, r: usize) -> &[u32] {
        match (self.offsets.get(r), self.offsets.get(r + 1)) {
            (Some(&a), Some(&b)) => &self.items[a as usize..b as usize],
            _ => &[],
        }
    }

    /// Iterates over the rows.
    pub fn iter(&self) -> impl Iterator<Item = &[u32]> + '_ {
        self.offsets.windows(2).map(|w| &self.items[w[0] as usize..w[1] as usize])
    }

    /// The `rows + 1` offsets (for upload to the GPU).
    #[must_use]
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// All items, row after row (for upload to the GPU).
    #[must_use]
    pub fn items(&self) -> &[u32] {
        &self.items
    }
}

/// The derived edge and adjacency tables of a mesh.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Topology {
    /// Unique undirected edges as weld-key pairs `[a, b]` with `a < b`, sorted.
    pub edges: Vec<[u32; 2]>,
    /// For every face corner (aligned with [`Mesh::face_vertices`]): the edge from that corner to
    /// the next one of its face.
    pub corner_edges: Vec<u32>,
    /// Edge to the faces that use it (2 on a closed manifold, 1 on a boundary).
    pub edge_faces: Csr,
    /// Render vertex to the faces it is a corner of.
    pub vertex_faces: Csr,
    /// Weld key to its neighbouring weld keys (the one-ring); rows of non-key vertices are empty.
    pub neighbors: Csr,
    /// Groups of two or more render vertices that share one position (UV seams and hard edges).
    pub duplicate_groups: Csr,
    /// Render vertex to the triangles whose normals it averages: the triangles of every vertex in
    /// its normal class. This is the gather list of the vertex-normal pass `[EVAL-04]`.
    pub normal_gather: Csr,
}

impl Topology {
    /// Computes the tables of `mesh`; `triangles` must be `mesh.triangulate()`.
    #[must_use]
    pub fn new(mesh: &Mesh, triangles: &Triangulation) -> Self {
        let v = mesh.vertex_count();
        let keys = mesh.position_keys();
        let offsets = mesh.face_offsets();
        let corners = mesh.face_vertices();

        // Directed corner edges as key pairs, then the unique undirected set.
        let mut corner_pairs: Vec<[u32; 2]> = Vec::with_capacity(corners.len());
        for w in offsets.windows(2) {
            let face = &corners[w[0] as usize..w[1] as usize];
            for (i, &a) in face.iter().enumerate() {
                let b = face[(i + 1) % face.len()];
                let (ka, kb) = (keys[a as usize], keys[b as usize]);
                corner_pairs.push(if ka < kb { [ka, kb] } else { [kb, ka] });
            }
        }
        let mut edges = corner_pairs.clone();
        edges.sort_unstable();
        edges.dedup();
        let corner_edges: Vec<u32> =
            corner_pairs.iter().map(|p| edges.binary_search(p).map_or(u32::MAX, |e| e as u32)).collect();

        let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(corners.len());
        for (f, w) in offsets.windows(2).enumerate() {
            for c in w[0]..w[1] {
                pairs.push((corner_edges[c as usize], f as u32));
            }
        }
        let edge_faces = Csr::from_pairs(edges.len(), &pairs);

        pairs.clear();
        for (f, w) in offsets.windows(2).enumerate() {
            for &c in &corners[w[0] as usize..w[1] as usize] {
                pairs.push((c, f as u32));
            }
        }
        let vertex_faces = Csr::from_pairs(v, &pairs);

        pairs.clear();
        for &[a, b] in &edges {
            pairs.push((a, b));
            pairs.push((b, a));
        }
        let neighbors = Csr::from_pairs(v, &pairs);

        pairs.clear();
        for (i, &k) in keys.iter().enumerate() {
            pairs.push((k, i as u32));
        }
        let by_key = Csr::from_pairs(v, &pairs);
        let duplicate_groups = Csr::from_rows(by_key.iter().filter(|g| g.len() >= 2));

        let normal_keys = mesh.normal_keys();
        pairs.clear();
        for (t, tri) in triangles.indices.as_chunks::<3>().0.iter().enumerate() {
            for &c in tri {
                pairs.push((normal_keys[c as usize], t as u32));
            }
        }
        let by_class = Csr::from_pairs(v, &pairs);
        let normal_gather = Csr::from_rows(normal_keys.iter().map(|&k| by_class.row(k as usize)));

        Self { edges, corner_edges, edge_faces, vertex_faces, neighbors, duplicate_groups, normal_gather }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::fixtures::{cube_shared, cube_split, grid};

    #[test]
    fn csr_builds_from_pairs_and_rows() {
        let t = Csr::from_pairs(3, &[(2, 7), (0, 1), (2, 8), (5, 9)]);
        assert_eq!(t.rows(), 3);
        assert_eq!(t.row(0), &[1]);
        assert!(t.row(1).is_empty());
        assert_eq!(t.row(2), &[7, 8]);
        assert!(t.row(3).is_empty());
        assert_eq!(t.offsets(), &[0, 1, 1, 3]);
        assert_eq!(t.items(), &[1, 7, 8]);
        let r = Csr::from_rows([&[1_u32, 2][..], &[], &[3]].into_iter());
        assert_eq!(r.iter().collect::<Vec<_>>(), vec![&[1, 2][..], &[], &[3]]);
        assert_eq!(Csr::default().rows(), 0);
    }

    #[test]
    fn closed_cube_has_12_edges_each_on_two_faces() {
        for mesh in [cube_shared(), cube_split(false), cube_split(true)] {
            let t = mesh.topology();
            assert_eq!(t.edges.len(), 12);
            assert!(t.edge_faces.iter().all(|f| f.len() == 2));
            assert_eq!(t.corner_edges.len(), 24);
            // Euler characteristic of a sphere: V - E + F = 2 on the welded surface.
            let welded = (0..mesh.vertex_count()).filter(|&i| mesh.position_keys()[i] as usize == i).count();
            assert_eq!(welded as i64 - 12 + 6, 2);
            // Every welded corner has three neighbours.
            for k in 0..mesh.vertex_count() {
                let expect = if mesh.position_keys()[k] as usize == k { 3 } else { 0 };
                assert_eq!(t.neighbors.row(k).len(), expect);
            }
        }
    }

    #[test]
    fn duplicate_groups_and_normal_gather() {
        let shared = cube_shared();
        let t = shared.topology();
        assert_eq!(t.duplicate_groups.rows(), 0);
        // Each corner touches 3 quads = 6 triangles, of which 3, 4 or 5 contain it.
        for v in 0..8 {
            assert_eq!(t.vertex_faces.row(v).len(), 3);
            assert!((3..=6).contains(&t.normal_gather.row(v).len()));
        }

        let smooth = cube_split(false);
        let t = smooth.topology();
        assert_eq!(t.duplicate_groups.rows(), 8);
        assert!(t.duplicate_groups.iter().all(|g| g.len() == 3));
        // Smooth seams: a duplicate gathers all triangles around its corner, like the shared cube.
        for v in 0..24 {
            assert!((3..=6).contains(&t.normal_gather.row(v).len()), "{v}");
            assert_eq!(t.vertex_faces.row(v).len(), 1);
        }

        let hard = cube_split(true);
        let t = hard.topology();
        // Hard edges: a vertex gathers only the triangles of its own face that contain it.
        for v in 0..24 {
            let g = t.normal_gather.row(v);
            assert!((1..=2).contains(&g.len()), "{v}: {g:?}");
            let tris = hard.triangulate();
            assert!(g.iter().all(|&tr| tris.triangle_faces[tr as usize] == t.vertex_faces.row(v)[0]));
        }
    }

    #[test]
    fn open_grid_has_boundary_edges() {
        let mesh = grid(3);
        let t = mesh.topology();
        // 3x3 quads: 2 * 3 * 4 = 24 edges, 12 on the boundary.
        assert_eq!(t.edges.len(), 24);
        assert_eq!(t.edge_faces.iter().filter(|f| f.len() == 1).count(), 12);
        assert!(t.corner_edges.iter().all(|&e| (e as usize) < t.edges.len()));
    }
}
