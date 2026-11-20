//! The GPU tables of a template: everything the deformation passes read that does not change after
//! load, built once on the CPU from a [`Template`] `[GPU-01]`.
//!
//! All tables are flat arrays of `u32` / `f32` words (structure of arrays `[PERF-R2]`), uploaded
//! as they are (little-endian on every target wgpu runs on):
//!
//! - **Delta entries, vertex-major** (`[EVAL-02]`): per vertex a row of `[u32; 2]` entries,
//!   word 0 = component index (low 16 bits) | z half (high 16 bits), word 1 = x half | y half
//!   << 16. Two tables: the *shape* components (rest shape, [`is_shape_kind`]) for the full
//!   re-accumulation, and the *dynamic* components (every component an expression, combination
//!   corrective or pose corrective names) for the per-frame expression pass. Rows list components
//!   in increasing order, the order in which the CPU reference re-accumulates. The rows are stored
//!   *sliced* ([`SlicedTable`], sliced ELLPACK with slices of 64 vertices, one per workgroup): entry
//!   `k` of the vertex in lane `l` of slice `s` is at `slice_offsets[s] + 64 k + l`, so the 64
//!   invocations of a workgroup read 64 consecutive entries at every step (coalesced), and rows
//!   shorter than the slice's longest are padded with `[0, 0]` (component 0, delta 0: adds nothing).
//! - **Delta entries, component-major**: the template's own layout (offsets, vertex indices) with
//!   `[u32; 2]` deltas, word 0 = x half | y half << 16, word 1 = z half, for the incremental
//!   per-component scatter `[EVAL-03]`.
//! - **Skin**: `influences` `[u32; 2]` per vertex, (joint, weight bits); bit 31 of the first
//!   influence's joint word is set when the vertex is in a dual-quaternion region `[SKIN-01]`.
//! - **Normals** (`[EVAL-04]`): the triangle list and the two gathers as one CSR: rows `0..=V`
//!   are the normal-class corners, rows `V + 1..=2V + 1` the vertex's own corners.
//! - **Strain** (`[EVAL-06]`): the welded edges, per region the list of edge halves that count for
//!   it cut into blocks of at most [`REDUCE_BLOCK`] (one workgroup each, never across regions), per
//!   region its subtree (itself and every descendant), the weld key of every vertex and per key its
//!   edges.
//!
//! Limits: counts and offsets are `u32` on the GPU, component and joint indices 16 bits; a template
//! beyond that is refused with [`DeformError::TooLarge`] (the model's own limits keep real templates
//! far below).

use humanart_math::F16;
use humanart_model::Template;
use humanart_model::delta::DeltaBases;
use humanart_model::eval::evaluator::is_shape_kind;
use humanart_model::eval::normals::{normal_corner_gather, own_corner_gather};
use humanart_model::lod::LodLevel;

use super::error::DeformError;

/// Vertices per slice of a [`SlicedTable`] (`WORKGROUP` in `deform/common.wgsl`).
pub const SLICE: usize = 64;

/// Edge halves one workgroup of the region reduction sums (`REDUCE_BLOCK` in `deform/strain.wgsl`).
pub const REDUCE_BLOCK: usize = 1024;

/// Bit 31 of the first influence's joint word: the vertex is skinned with dual quaternions under
/// the by-region mode.
pub const DQ_FLAG: u32 = 0x8000_0000;

/// A CSR table: `offsets` has one more entry than there are rows; row `r` is
/// `items[offsets[r]..offsets[r + 1]]`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CsrTable<T> {
    /// Row starts, then the total.
    pub offsets: Vec<u32>,
    /// The items of every row, row after row.
    pub items: Vec<T>,
}

impl<T: Copy + Default> CsrTable<T> {
    /// Builds the table of `rows` rows from `(row, item)` pairs; within a row, items keep the order
    /// of `pairs` (a stable counting sort). Pairs with a row out of range are dropped.
    fn from_pairs(rows: usize, pairs: impl Iterator<Item = (usize, T)> + Clone) -> Result<Self, DeformError> {
        let mut counts = vec![0_u32; rows + 1];
        let mut total = 0_u64;
        for (r, _) in pairs.clone() {
            if let Some(c) = counts.get_mut(r + 1).filter(|_| r < rows) {
                *c += 1;
                total += 1;
            }
        }
        check_u32("CSR items", total)?;
        for r in 0..rows {
            counts[r + 1] += counts[r];
        }
        let mut items = vec![T::default(); total as usize];
        let mut next = counts.clone();
        for (r, item) in pairs {
            if r < rows {
                let slot = next[r] as usize;
                if let Some(dst) = items.get_mut(slot) {
                    *dst = item;
                }
                next[r] += 1;
            }
        }
        Ok(Self { offsets: counts, items })
    }

    /// The number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Row `r` (empty when out of range).
    #[must_use]
    pub fn row(&self, r: usize) -> &[T] {
        match (self.offsets.get(r), self.offsets.get(r + 1)) {
            (Some(&a), Some(&b)) => self.items.get(a as usize..b as usize).unwrap_or(&[]),
            _ => &[],
        }
    }
}

/// Vertex rows of `[u32; 2]` entries in sliced ELLPACK layout (see the module documentation).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SlicedTable {
    /// Per slice of [`SLICE`] vertices: its first entry; then the total.
    pub slice_offsets: Vec<u32>,
    /// The entries, padded.
    pub entries: Vec<[u32; 2]>,
    /// Per vertex: its row length (CPU side, for checks; not uploaded).
    pub lengths: Vec<u32>,
}

impl SlicedTable {
    /// The sliced form of `csr`.
    fn from_csr(csr: &CsrTable<[u32; 2]>) -> Result<Self, DeformError> {
        let rows = csr.rows();
        let slices = rows.div_ceil(SLICE);
        let mut slice_offsets = Vec::with_capacity(slices + 1);
        let mut total = 0_u64;
        slice_offsets.push(0);
        for s in 0..slices {
            let width = (s * SLICE..((s + 1) * SLICE).min(rows)).map(|v| csr.row(v).len()).max().unwrap_or(0);
            total += (width * SLICE) as u64;
            slice_offsets.push(check_u32("sliced delta entries", total)?);
        }
        let mut entries = vec![[0_u32; 2]; total as usize];
        for (s, &base) in slice_offsets.iter().take(slices).enumerate() {
            let base = base as usize;
            for lane in 0..SLICE {
                for (k, &e) in csr.row(s * SLICE + lane).iter().enumerate() {
                    if let Some(slot) = entries.get_mut(base + k * SLICE + lane) {
                        *slot = e;
                    }
                }
            }
        }
        let lengths = (0..rows).map(|v| csr.row(v).len() as u32).collect();
        Ok(Self { slice_offsets, entries, lengths })
    }

    /// The entries of vertex `v`'s row, in order (empty when out of range).
    pub fn row(&self, v: usize) -> impl Iterator<Item = [u32; 2]> + '_ {
        let (s, lane) = (v / SLICE, v % SLICE);
        let base = self.slice_offsets.get(s).copied().unwrap_or(0) as usize;
        let len = self.lengths.get(v).copied().unwrap_or(0) as usize;
        (0..len).filter_map(move |k| self.entries.get(base + k * SLICE + lane).copied())
    }
}

/// Packs a delta for the vertex-major tables: `[component | z << 16, x | y << 16]`.
#[must_use]
pub fn pack_vertex_entry(component: u16, d: [F16; 3]) -> [u32; 2] {
    let [x, y, z] = d.map(|h| u32::from(h.to_bits()));
    [u32::from(component) | (z << 16), x | (y << 16)]
}

/// Packs a delta for the component-major table: `[x | y << 16, z]`.
#[must_use]
pub fn pack_component_delta(d: [F16; 3]) -> [u32; 2] {
    let [x, y, z] = d.map(|h| u32::from(h.to_bits()));
    [x | (y << 16), z]
}

/// Every load-time table of one template. See the module documentation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeformTables {
    /// Vertices of the mesh.
    pub vertex_count: u32,
    /// Delta components of the template.
    pub component_count: u32,
    /// Template positions, packed xyz.
    pub template_positions: Vec<f32>,
    /// UVs, packed uv.
    pub uvs: Vec<f32>,
    /// Per component: part of the rest shape.
    pub is_shape: Vec<bool>,
    /// Per component: named by an expression, combination or pose corrective.
    pub is_dynamic: Vec<bool>,
    /// Vertex-major shape entries.
    pub shape: SlicedTable,
    /// Vertex-major dynamic entries.
    pub dynamic: SlicedTable,
    /// Component-major entry offsets (the template's).
    pub component_offsets: Vec<u32>,
    /// Component-major vertex indices (the template's).
    pub component_indices: Vec<u32>,
    /// Component-major packed deltas.
    pub component_deltas: Vec<[u32; 2]>,
    /// Influences per vertex (0 for an unskinned template).
    pub influences: u32,
    /// `influences` (joint | flag, weight bits) per vertex.
    pub skin: Vec<[u32; 2]>,
    /// Joints of the skeleton.
    pub joint_count: u32,
    /// The triangle list (three vertex indices per triangle).
    pub triangles: Vec<u32>,
    /// Normal-class corner rows, then own-corner rows (`2 · (V + 1)` offsets).
    pub gather: CsrTable<u32>,
    /// Welded edges (weld-key vertex pairs).
    pub edges: Vec<[u32; 2]>,
    /// Regions of the region map.
    pub region_count: u32,
    /// Per region: the edges with a half counting for it (an edge twice when both ends are in it).
    pub region_edges: CsrTable<u32>,
    /// Reduction blocks: `[start, end)` ranges of `region_edges.items`, in region order.
    pub region_blocks: Vec<[u32; 2]>,
    /// Per region: its range of `region_blocks` (`region_count + 1` offsets).
    pub region_block_offsets: Vec<u32>,
    /// Per region: itself and its descendants.
    pub subtrees: CsrTable<u32>,
    /// Per vertex: its weld key.
    pub position_keys: Vec<u32>,
    /// Per weld key: the edges at it.
    pub key_edges: CsrTable<u32>,
}

fn check_u32(what: &'static str, value: u64) -> Result<u32, DeformError> {
    u32::try_from(value).map_err(|_| DeformError::TooLarge { what, value, limit: u64::from(u32::MAX) })
}

/// The packed xyz floats of `points`.
fn pack_xyz(points: &[humanart_math::Vec3]) -> Vec<f32> {
    points.iter().flat_map(|p| p.to_array()).collect()
}

/// Builds the vertex-major table of the components `selected` of `deltas`.
fn vertex_major(deltas: &DeltaBases, vertex_count: usize, selected: &[bool]) -> Result<SlicedTable, DeformError> {
    let pairs = (0..deltas.len()).filter(|&c| selected.get(c).copied().unwrap_or(false)).flat_map(move |c| {
        let component = u16::try_from(c).unwrap_or(u16::MAX);
        deltas
            .indices(c)
            .iter()
            .zip(deltas.deltas(c))
            .map(move |(&v, &d)| (v as usize, pack_vertex_entry(component, d)))
    });
    SlicedTable::from_csr(&CsrTable::from_pairs(vertex_count, pairs)?)
}

impl DeformTables {
    /// Builds the tables of `template`.
    pub fn new(template: &Template) -> Result<Self, DeformError> {
        let mesh = template.mesh();
        let v = mesh.vertex_count();
        let vertex_count = check_u32("vertex count", v as u64 * 3).map(|_| v as u32)?;
        let deltas = template.deltas();
        let components = deltas.len();
        if components > usize::from(u16::MAX) + 1 {
            return Err(DeformError::TooLarge {
                what: "delta components",
                value: components as u64,
                limit: u64::from(u16::MAX) + 1,
            });
        }
        let is_shape: Vec<bool> = deltas.components().iter().map(|c| is_shape_kind(c.kind)).collect();
        let mut is_dynamic = vec![false; components];
        let set = template.expressions();
        let named = set
            .expressions
            .iter()
            .map(|e| e.component)
            .chain(set.combinations.iter().map(|c| c.component))
            .chain(set.pose_correctives.iter().map(|p| p.component));
        for c in named {
            match is_dynamic.get_mut(c as usize) {
                Some(slot) => *slot = true,
                None => return Err(DeformError::Invalid { what: "an expression names a missing delta component" }),
            }
        }
        if deltas.all_indices().iter().any(|&i| i as usize >= v) {
            return Err(DeformError::Invalid { what: "a delta entry names a missing vertex" });
        }
        let shape = vertex_major(deltas, v, &is_shape)?;
        let dynamic = vertex_major(deltas, v, &is_dynamic)?;
        check_u32("delta entries", deltas.all_indices().len() as u64)?;

        // Skin.
        let skin = template.skin();
        let skeleton = template.skeleton();
        let joint_count = skeleton.joints.len();
        if joint_count > usize::from(u16::MAX) + 1 {
            return Err(DeformError::TooLarge {
                what: "joints",
                value: joint_count as u64,
                limit: u64::from(u16::MAX) + 1,
            });
        }
        let influences = if joint_count == 0 { 0 } else { usize::from(skin.influences) };
        let regions = template.regions();
        let dq_regions = &skin.dual_quaternion_regions;
        let mut packed_skin = Vec::with_capacity(influences * v);
        for vertex in 0..v {
            if influences == 0 {
                break;
            }
            let (joints, weights) = skin.vertex(vertex);
            let region = regions.vertex_regions.get(vertex).copied().unwrap_or(0);
            let dq = dq_regions.iter().any(|&d| regions.is_within(region, d));
            for k in 0..influences {
                let joint = joints.get(k).map_or(u32::from(u16::MAX), |&j| u32::from(j));
                let weight = weights.get(k).copied().unwrap_or(0.0);
                let flag = if k == 0 && dq { DQ_FLAG } else { 0 };
                packed_skin.push([joint | flag, weight.to_bits()]);
            }
        }

        // Normals.
        let tri = mesh.triangulate();
        check_u32("triangle indices", tri.indices.len() as u64)?;
        let normal = normal_corner_gather(mesh, &tri.indices);
        let own = own_corner_gather(v, &tri.indices);
        let mut gather = CsrTable { offsets: Vec::with_capacity(2 * v + 2), items: Vec::new() };
        gather.offsets.extend_from_slice(normal.offsets());
        gather.items.extend_from_slice(normal.items());
        let base = check_u32("normal gather corners", normal.items().len() as u64)?;
        check_u32("gather corners", (normal.items().len() + own.items().len()) as u64)?;
        gather.offsets.extend(own.offsets().iter().map(|&o| o + base));
        gather.items.extend_from_slice(own.items());
        if gather.offsets.len() != 2 * v + 2 {
            return Err(DeformError::Invalid { what: "a corner gather does not have one row per vertex" });
        }

        // Strain.
        let topology = mesh.topology();
        let edges = topology.edges.clone();
        check_u32("edges", edges.len() as u64)?;
        let region_count = regions.regions.len();
        let region_of = |vertex: u32| regions.vertex_regions.get(vertex as usize).copied().unwrap_or(0);
        let halves = edges
            .iter()
            .enumerate()
            .flat_map(|(e, &[a, b])| [(usize::from(region_of(a)), e as u32), (usize::from(region_of(b)), e as u32)]);
        let region_edges = CsrTable::from_pairs(region_count, halves)?;
        let mut region_blocks = Vec::new();
        let mut region_block_offsets = Vec::with_capacity(region_count + 1);
        region_block_offsets.push(0);
        for r in 0..region_count {
            let (start, end) = (region_edges.offsets[r], region_edges.offsets[r + 1]);
            let mut at = start;
            while at < end {
                let next = end.min(at + REDUCE_BLOCK as u32);
                region_blocks.push([at, next]);
                at = next;
            }
            region_block_offsets.push(check_u32("strain blocks", region_blocks.len() as u64)?);
        }
        // A region counts for itself and every ancestor reached through parents of lower index (the
        // rule of the CPU reduction, which walks regions backwards).
        let parents: Vec<Option<u16>> = regions.regions.iter().map(|r| r.parent).collect();
        let mut subtree_pairs: Vec<(usize, u32)> = Vec::with_capacity(region_count * 4);
        for s in 0..region_count {
            let mut at = s;
            subtree_pairs.push((at, s as u32));
            while let Some(p) = parents.get(at).copied().flatten().map(usize::from).filter(|&p| p < at) {
                at = p;
                subtree_pairs.push((at, s as u32));
            }
        }
        let subtrees = CsrTable::from_pairs(region_count, subtree_pairs.iter().copied())?;
        let position_keys: Vec<u32> =
            (0..v).map(|i| mesh.position_keys().get(i).copied().unwrap_or(i as u32)).collect();
        let key_pairs =
            edges.iter().enumerate().flat_map(|(e, &[a, b])| [(a as usize, e as u32), (b as usize, e as u32)]);
        let key_edges = CsrTable::from_pairs(v, key_pairs)?;

        let uvs: Vec<f32> = (0..v).flat_map(|i| mesh.uvs().get(i).map_or([0.0, 0.0], |u| u.to_array())).collect();
        Ok(Self {
            vertex_count,
            component_count: components as u32,
            template_positions: pack_xyz(mesh.positions()),
            uvs,
            is_shape,
            is_dynamic,
            shape,
            dynamic,
            component_offsets: deltas.offsets().to_vec(),
            component_indices: deltas.all_indices().to_vec(),
            component_deltas: deltas.all_deltas().iter().map(|&d| pack_component_delta(d)).collect(),
            influences: influences as u32,
            skin: packed_skin,
            joint_count: joint_count as u32,
            triangles: tri.indices,
            gather,
            edges,
            region_count: region_count as u32,
            region_edges,
            region_blocks,
            region_block_offsets,
            subtrees,
            position_keys,
            key_edges,
        })
    }

    /// The number of triangles.
    #[must_use]
    pub fn triangle_count(&self) -> u32 {
        (self.triangles.len() / 3) as u32
    }
}

/// The load-time tables of one reduced level of detail `[MOD-05]`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LodTables {
    /// Vertices of the level.
    pub vertex_count: u32,
    /// Per vertex: the three LOD0 sources and a zero.
    pub sources: Vec<[u32; 4]>,
    /// Per vertex: the three weights and a zero.
    pub weights: Vec<[f32; 4]>,
    /// UVs, packed uv.
    pub uvs: Vec<f32>,
    /// The level's triangle list.
    pub triangles: Vec<u32>,
    /// Normal-class corner rows, then own-corner rows.
    pub gather: CsrTable<u32>,
}

impl LodTables {
    /// Builds the tables of `level` of a template whose LOD0 has `base_vertex_count` vertices.
    pub fn new(level: &LodLevel, base_vertex_count: usize) -> Result<Self, DeformError> {
        let mesh = &level.mesh;
        let v = mesh.vertex_count();
        check_u32("level vertex count", v as u64 * 3)?;
        if level.sources.len() != v || level.source_weights.len() != v {
            return Err(DeformError::Invalid { what: "a level's sources do not match its vertices" });
        }
        if level.sources.iter().flatten().any(|&s| s as usize >= base_vertex_count) {
            return Err(DeformError::Invalid { what: "a level vertex names a missing LOD0 vertex" });
        }
        let tri = mesh.triangulate();
        check_u32("level triangle indices", tri.indices.len() as u64)?;
        let normal = normal_corner_gather(mesh, &tri.indices);
        let own = own_corner_gather(v, &tri.indices);
        let base = check_u32("level gather corners", (normal.items().len() + own.items().len()) as u64)
            .map(|_| normal.items().len() as u32)?;
        let mut gather = CsrTable { offsets: normal.offsets().to_vec(), items: normal.items().to_vec() };
        gather.offsets.extend(own.offsets().iter().map(|&o| o + base));
        gather.items.extend_from_slice(own.items());
        Ok(Self {
            vertex_count: v as u32,
            sources: level.sources.iter().map(|s| [s[0], s[1], s[2], 0]).collect(),
            weights: level.source_weights.iter().map(|w| [w[0], w[1], w[2], 0.0]).collect(),
            uvs: (0..v).flat_map(|i| mesh.uvs().get(i).map_or([0.0, 0.0], |u| u.to_array())).collect(),
            triangles: tri.indices,
            gather,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deform::tests::fixtures;

    #[test]
    fn csr_from_pairs_is_stable_and_drops_out_of_range_rows() {
        let pairs = [(2_usize, 10_u32), (0, 11), (2, 12), (5, 13), (0, 14)];
        let t = CsrTable::from_pairs(3, pairs.iter().copied()).unwrap();
        assert_eq!(t.offsets, vec![0, 2, 2, 4]);
        assert_eq!(t.row(0), &[11, 14]);
        assert_eq!(t.row(1), &[] as &[u32]);
        assert_eq!(t.row(2), &[10, 12]);
        assert_eq!(t.row(3), &[] as &[u32]);
        assert_eq!(t.rows(), 3);
    }

    #[test]
    fn entries_pack_the_half_bits() {
        let d = [F16::from_f32(0.5), F16::from_f32(-0.25), F16::from_f32(1.0)];
        let [w0, w1] = pack_vertex_entry(7, d);
        assert_eq!(w0 & 0xffff, 7);
        assert_eq!(w0 >> 16, u32::from(d[2].to_bits()));
        assert_eq!(w1 & 0xffff, u32::from(d[0].to_bits()));
        assert_eq!(w1 >> 16, u32::from(d[1].to_bits()));
        let [c0, c1] = pack_component_delta(d);
        assert_eq!(c0, w1);
        assert_eq!(c1, u32::from(d[2].to_bits()));
    }

    // [EVAL-02]: every delta entry is in exactly the right vertex-major table, rows in component order.
    #[test]
    fn eval_02_vertex_major_tables_hold_every_entry_once() {
        let t = fixtures::sheet(12, 4);
        let tables = DeformTables::new(&t).unwrap();
        let deltas = t.deltas();
        let mut shape_count = 0;
        let mut dynamic_count = 0;
        for c in 0..deltas.len() {
            for (&v, &d) in deltas.indices(c).iter().zip(deltas.deltas(c)) {
                let entry = pack_vertex_entry(c as u16, d);
                let in_shape = tables.shape.row(v as usize).filter(|&e| e == entry).count();
                let in_dynamic = tables.dynamic.row(v as usize).filter(|&e| e == entry).count();
                assert_eq!(in_shape, usize::from(tables.is_shape[c]));
                assert_eq!(in_dynamic, usize::from(tables.is_dynamic[c]));
                shape_count += in_shape;
                dynamic_count += in_dynamic;
            }
        }
        assert_eq!(shape_count, tables.shape.lengths.iter().sum::<u32>() as usize);
        assert_eq!(dynamic_count, tables.dynamic.lengths.iter().sum::<u32>() as usize);
        for v in 0..tables.vertex_count as usize {
            let row: Vec<[u32; 2]> = tables.shape.row(v).collect();
            assert!(row.windows(2).all(|w| (w[0][0] & 0xffff) < (w[1][0] & 0xffff)));
        }
        // Padding is [0, 0] and every slice is a whole number of 64-wide steps.
        let padded: usize = tables.shape.entries.iter().filter(|&&e| e == [0, 0]).count();
        assert_eq!(padded + shape_count, tables.shape.entries.len());
        assert!(tables.shape.slice_offsets.iter().all(|&o| (o as usize).is_multiple_of(SLICE)));
        assert!(tables.is_dynamic.iter().zip(&tables.is_shape).all(|(d, s)| !(d & s)));
        assert!(tables.is_dynamic.iter().any(|&d| d));
    }

    #[test]
    fn gather_rows_cover_every_corner() {
        let t = fixtures::sheet(6, 1);
        let tables = DeformTables::new(&t).unwrap();
        let v = tables.vertex_count as usize;
        assert_eq!(tables.gather.offsets.len(), 2 * v + 2);
        // Own-corner rows: every corner exactly once, at its vertex.
        let mut seen = vec![0_u32; tables.triangles.len()];
        for vertex in 0..v {
            for &c in tables.gather.row(v + 1 + vertex) {
                assert_eq!(tables.triangles[c as usize] as usize, vertex);
                seen[c as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&n| n == 1));
    }

    #[test]
    fn subtrees_and_region_halves_follow_the_region_tree() {
        let t = fixtures::sheet(8, 1);
        let tables = DeformTables::new(&t).unwrap();
        // Region 0 is the root of 1 and 2, region 3 a child of 1.
        let sorted = |r: usize| {
            let mut row = tables.subtrees.row(r).to_vec();
            row.sort_unstable();
            row
        };
        assert_eq!(sorted(0), vec![0, 1, 2, 3]);
        assert_eq!(sorted(1), vec![1, 3]);
        assert_eq!(sorted(2), vec![2]);
        let halves: usize = (0..tables.region_count as usize).map(|r| tables.region_edges.row(r).len()).sum();
        assert_eq!(halves, 2 * tables.edges.len());
        // Blocks tile every region's list without crossing into the next region.
        for r in 0..tables.region_count as usize {
            let blocks = &tables.region_blocks
                [tables.region_block_offsets[r] as usize..tables.region_block_offsets[r + 1] as usize];
            let covered: u32 = blocks.iter().map(|b| b[1] - b[0]).sum();
            assert_eq!(covered as usize, tables.region_edges.row(r).len());
            assert!(
                blocks
                    .iter()
                    .all(|b| b[0] >= tables.region_edges.offsets[r] && b[1] <= tables.region_edges.offsets[r + 1])
            );
            assert!(blocks.iter().all(|b| b[1] - b[0] <= REDUCE_BLOCK as u32 && b[1] > b[0]));
        }
        let key_halves: usize = (0..tables.vertex_count as usize).map(|k| tables.key_edges.row(k).len()).sum();
        assert_eq!(key_halves, 2 * tables.edges.len());
    }

    #[test]
    fn skin_flags_dual_quaternion_vertices() {
        let t = fixtures::sheet(8, 1);
        let tables = DeformTables::new(&t).unwrap();
        let k = tables.influences as usize;
        assert!(k > 0);
        let regions = t.regions();
        for v in 0..tables.vertex_count as usize {
            let flagged = tables.skin[v * k][0] & DQ_FLAG != 0;
            let region = regions.vertex_regions[v];
            let dq = t.skin().dual_quaternion_regions.iter().any(|&d| regions.is_within(region, d));
            assert_eq!(flagged, dq);
            assert!(tables.skin[v * k + 1..v * k + k].iter().all(|e| e[0] & DQ_FLAG == 0));
        }
    }
}
