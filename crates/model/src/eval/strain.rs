//! Edge strain and its per-region reduction `[EVAL-06]`.
//!
//! The strain of an edge is `s = l / l0 − 1`, with `l0` its length in the character's rest shape
//! (identity, semantic and sculpt deltas applied: section 6.2 step 1) and `l` its length in the
//! evaluated surface (after expressions, correctives and skinning). Positive is stretch, negative
//! compression. Edges are the welded edges of the template ([`crate::Topology::edges`]), so a UV
//! seam is measured once.
//!
//! **Reduction.** Each edge counts half for the region of each of its two end vertices, weighted by
//! its rest length. For every region `r`:
//! `stretch_r = Σ w·max(s, 0) / Σ w` and `compression_r = Σ w·max(−s, 0) / Σ w`, summed over the
//! region and all its descendants (a parent region reports its whole subtree). A region with no
//! edges reports 0. These two numbers per region are the small buffer the skin shader reads for
//! wrinkle and tension blending ([`super::wrinkle`]). The per-vertex strain (mean signed strain of
//! the vertex's edges, the same on every duplicate) feeds the strain/tension viewport mode.
//!
//! Region-wise normalisation tables (`[BAKE-07]`) are not applied here: the wrinkle drivers carry
//! their own gains.

use humanart_math::Vec3;

use crate::mesh::Mesh;
use crate::region::RegionMap;
use crate::topology::Topology;

/// Edge strain, per-vertex strain and the per-region reduction, with preallocated storage.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StrainField {
    edges: Vec<[u32; 2]>,
    edge_regions: Vec<[u16; 2]>,
    parents: Vec<Option<u16>>,
    position_keys: Vec<u32>,
    rest_lengths: Vec<f32>,
    edge_strain: Vec<f32>,
    region_weight: Vec<f32>,
    region_stretch: Vec<f32>,
    region_compression: Vec<f32>,
    key_sum: Vec<f32>,
    key_count: Vec<u32>,
    vertex_strain: Vec<f32>,
}

impl StrainField {
    /// The field of `mesh` with its `topology` and region map; rest lengths are those of the
    /// template positions until [`set_rest`](Self::set_rest) is called.
    #[must_use]
    pub fn new(mesh: &Mesh, topology: &Topology, regions: &RegionMap) -> Self {
        let region_of = |v: u32| regions.vertex_regions.get(v as usize).copied().unwrap_or(0);
        let edges = topology.edges.clone();
        let edge_regions = edges.iter().map(|&[a, b]| [region_of(a), region_of(b)]).collect();
        let r = regions.regions.len();
        let v = mesh.vertex_count();
        let mut out = Self {
            edge_regions,
            parents: regions.regions.iter().map(|g| g.parent).collect(),
            position_keys: mesh.position_keys().to_vec(),
            rest_lengths: vec![0.0; edges.len()],
            edge_strain: vec![0.0; edges.len()],
            region_weight: vec![0.0; r],
            region_stretch: vec![0.0; r],
            region_compression: vec![0.0; r],
            key_sum: vec![0.0; v],
            key_count: vec![0; v],
            vertex_strain: vec![0.0; v],
            edges,
        };
        out.set_rest(mesh.positions());
        out
    }

    /// Measures the rest lengths in `rest` (the rest shape). Allocation-free.
    pub fn set_rest(&mut self, rest: &[Vec3]) {
        for (&[a, b], l) in self.edges.iter().zip(&mut self.rest_lengths) {
            *l = match (rest.get(a as usize), rest.get(b as usize)) {
                (Some(&pa), Some(&pb)) => pa.distance(pb),
                _ => 0.0,
            };
        }
    }

    /// Measures the strain of `deformed` against the rest lengths and reduces it per region and per
    /// vertex. Allocation-free.
    pub fn compute(&mut self, deformed: &[Vec3]) {
        self.region_weight.fill(0.0);
        self.region_stretch.fill(0.0);
        self.region_compression.fill(0.0);
        self.key_sum.fill(0.0);
        self.key_count.fill(0);
        for (e, &[a, b]) in self.edges.iter().enumerate() {
            let l0 = self.rest_lengths[e];
            let s = match (deformed.get(a as usize), deformed.get(b as usize)) {
                (Some(&pa), Some(&pb)) if l0 > 0.0 => pa.distance(pb) / l0 - 1.0,
                _ => 0.0,
            };
            let s = if s.is_finite() { s } else { 0.0 };
            self.edge_strain[e] = s;
            let w = 0.5 * l0;
            for r in self.edge_regions[e] {
                let r = r as usize;
                if r < self.region_weight.len() {
                    self.region_weight[r] += w;
                    self.region_stretch[r] += w * s.max(0.0);
                    self.region_compression[r] += w * (-s).max(0.0);
                }
            }
            for k in [a, b] {
                if let (Some(sum), Some(count)) = (self.key_sum.get_mut(k as usize), self.key_count.get_mut(k as usize))
                {
                    *sum += s;
                    *count += 1;
                }
            }
        }
        // Children come after their parents: walking backwards adds every subtree into its root.
        for r in (0..self.parents.len()).rev() {
            if let Some(p) = self.parents[r].map(usize::from).filter(|&p| p < r) {
                self.region_weight[p] += self.region_weight[r];
                self.region_stretch[p] += self.region_stretch[r];
                self.region_compression[p] += self.region_compression[r];
            }
        }
        for r in 0..self.region_weight.len() {
            let w = self.region_weight[r];
            let inv = if w > 0.0 { w.recip() } else { 0.0 };
            self.region_stretch[r] *= inv;
            self.region_compression[r] *= inv;
        }
        for (v, out) in self.vertex_strain.iter_mut().enumerate() {
            let k = self.position_keys.get(v).copied().unwrap_or(v as u32) as usize;
            *out = match (self.key_sum.get(k), self.key_count.get(k)) {
                (Some(&s), Some(&c)) if c > 0 => s / c as f32,
                _ => 0.0,
            };
        }
    }

    /// The welded edges (weld-key pairs).
    #[must_use]
    pub fn edges(&self) -> &[[u32; 2]] {
        &self.edges
    }

    /// The rest length of every edge.
    #[must_use]
    pub fn rest_lengths(&self) -> &[f32] {
        &self.rest_lengths
    }

    /// The strain of every edge from the last [`compute`](Self::compute).
    #[must_use]
    pub fn edge_strain(&self) -> &[f32] {
        &self.edge_strain
    }

    /// The mean stretch of every region (with its subtree).
    #[must_use]
    pub fn region_stretch(&self) -> &[f32] {
        &self.region_stretch
    }

    /// The mean compression of every region (with its subtree), as a positive number.
    #[must_use]
    pub fn region_compression(&self) -> &[f32] {
        &self.region_compression
    }

    /// The mean signed strain of every vertex's edges (equal on duplicates).
    #[must_use]
    pub fn vertex_strain(&self) -> &[f32] {
        &self.vertex_strain
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::fixtures::grid;
    use crate::names::Side;
    use crate::region::Region;

    fn two_region_grid() -> (Mesh, RegionMap) {
        let mesh = grid(4);
        let regions = RegionMap {
            regions: vec![
                Region { name: "all".into(), parent: None, side: Side::Center },
                Region { name: "left".into(), parent: Some(0), side: Side::Left },
                Region { name: "right".into(), parent: Some(0), side: Side::Right },
            ],
            vertex_regions: mesh.positions().iter().map(|p| if p.x < 0.5 { 1 } else { 2 }).collect(),
            soft: None,
        };
        (mesh, regions)
    }

    #[test]
    fn eval_06_uniform_scale_gives_uniform_strain() {
        let (mesh, regions) = two_region_grid();
        let mut s = StrainField::new(&mesh, &mesh.topology(), &regions);
        let stretched: Vec<Vec3> = mesh.positions().iter().map(|&p| p * 1.1).collect();
        s.compute(&stretched);
        assert!(s.edge_strain().iter().all(|&e| (e - 0.1).abs() < 1e-5));
        for r in 0..3 {
            assert!((s.region_stretch()[r] - 0.1).abs() < 1e-5);
            assert_eq!(s.region_compression()[r], 0.0);
        }
        assert!(s.vertex_strain().iter().all(|&e| (e - 0.1).abs() < 1e-5));
        let squashed: Vec<Vec3> = mesh.positions().iter().map(|&p| p * 0.8).collect();
        s.compute(&squashed);
        assert!((s.region_compression()[0] - 0.2).abs() < 1e-5);
        assert_eq!(s.region_stretch()[0], 0.0);
    }

    #[test]
    fn eval_06_local_stretch_stays_in_its_region_and_rolls_up() {
        let (mesh, regions) = two_region_grid();
        let mut s = StrainField::new(&mesh, &mesh.topology(), &regions);
        // Stretch only the far right column of the grid along X.
        let moved: Vec<Vec3> =
            mesh.positions().iter().map(|&p| if p.x > 0.9 { p + Vec3::X * 0.25 } else { p }).collect();
        s.compute(&moved);
        assert_eq!(s.region_stretch()[1], 0.0, "left region untouched");
        assert!(s.region_stretch()[2] > 0.0);
        assert!(s.region_stretch()[0] > 0.0 && s.region_stretch()[0] < s.region_stretch()[2]);
        assert_eq!(s.edges().len(), s.rest_lengths().len());
        // Rigid motion: no strain anywhere.
        let shifted: Vec<Vec3> = mesh.positions().iter().map(|&p| p + Vec3::ONE).collect();
        s.compute(&shifted);
        assert!(s.edge_strain().iter().all(|e| e.abs() < 1e-6));
    }

    #[test]
    fn short_or_broken_input_gives_zero_not_nan() {
        let (mesh, regions) = two_region_grid();
        let mut s = StrainField::new(&mesh, &mesh.topology(), &regions);
        s.set_rest(&[]);
        s.compute(mesh.positions());
        assert!(s.edge_strain().iter().all(|&e| e == 0.0));
        assert!(s.region_stretch().iter().all(|&e| e == 0.0));
        s.set_rest(mesh.positions());
        s.compute(&[]);
        assert!(s.edge_strain().iter().all(|&e| e == 0.0));
    }
}
