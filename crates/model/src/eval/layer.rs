//! Sparse sculpt layers: the `ΔLAYER_l` terms of the rest shape (section 6.2 step 1, `[DOC-03]`).
//!
//! A layer is a sparse list of vertex displacements in metres, stored as two aligned arrays (SoA):
//! strictly increasing vertex indices and one f32 displacement each. Unlike the template's delta
//! bases a layer is user data (sculpt strokes, the landmark solver's residual layer `[LM-02]`), so
//! it is kept in f32.
//!
//! A layer must move every duplicate of a vertex (UV seam or hard edge copies) alike, or the surface
//! tears; [`SculptLayer::seam_consistent`] checks that and [`SculptLayer::spread_to_duplicates`]
//! makes it so.

use humanart_math::Vec3;

use super::error::{EvalError, EvalResult};
use crate::mesh::Mesh;

/// The most entries one layer may hold (`2^24`, far above any template's vertex count).
pub const MAX_LAYER_ENTRIES: usize = 1 << 24;

/// One sparse sculpt layer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SculptLayer {
    indices: Vec<u32>,
    deltas: Vec<Vec3>,
}

impl SculptLayer {
    /// A layer from aligned arrays: `indices` strictly increasing and below `vertex_count`, one finite
    /// displacement per index, at most [`MAX_LAYER_ENTRIES`] entries.
    pub fn new(indices: Vec<u32>, deltas: Vec<Vec3>, vertex_count: usize) -> EvalResult<Self> {
        if indices.len() > MAX_LAYER_ENTRIES {
            return Err(EvalError::limit("sculpt layer entry count", MAX_LAYER_ENTRIES, indices.len()));
        }
        if indices.len() != deltas.len() {
            return Err(EvalError::Length { what: "sculpt layer deltas", limit: indices.len(), got: deltas.len() });
        }
        if let Some(i) = indices.windows(2).position(|w| w[0] >= w[1]) {
            return Err(EvalError::Range { what: "sculpt layer index", index: i + 1, rule: "indices must increase" });
        }
        if let Some(i) = indices.iter().position(|&v| v as usize >= vertex_count) {
            return Err(EvalError::Missing { what: "vertex", index: i });
        }
        if let Some(i) = deltas.iter().position(|d| !d.is_finite()) {
            return Err(EvalError::NonFinite { what: "sculpt layer delta", index: i });
        }
        Ok(Self { indices, deltas })
    }

    /// A layer from unordered `(vertex, displacement)` pairs: sorted, repeated vertices summed, zero
    /// displacements dropped; the same checks as [`SculptLayer::new`].
    pub fn from_pairs(pairs: &[(u32, Vec3)], vertex_count: usize) -> EvalResult<Self> {
        let mut sorted = pairs.to_vec();
        sorted.sort_by_key(|p| p.0);
        let mut indices: Vec<u32> = Vec::with_capacity(sorted.len());
        let mut deltas: Vec<Vec3> = Vec::with_capacity(sorted.len());
        for (v, d) in sorted {
            match (indices.last(), deltas.last_mut()) {
                (Some(&last), Some(sum)) if last == v => *sum += d,
                _ => {
                    indices.push(v);
                    deltas.push(d);
                }
            }
        }
        let keep: Vec<bool> = deltas.iter().map(|d| *d != Vec3::ZERO).collect();
        let mut k = keep.iter();
        indices.retain(|_| k.next().copied().unwrap_or(false));
        let mut k = keep.iter();
        deltas.retain(|_| k.next().copied().unwrap_or(false));
        Self::new(indices, deltas, vertex_count)
    }

    /// The vertex indices, strictly increasing.
    #[must_use]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// The displacements, aligned with [`indices`](Self::indices).
    #[must_use]
    pub fn deltas(&self) -> &[Vec3] {
        &self.deltas
    }

    /// How many entries the layer has.
    #[must_use]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Whether the layer moves nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Adds `weight` times the layer to `positions` (entries outside `positions` are skipped; the
    /// evaluator checks the range when the layer is added).
    pub fn accumulate(&self, weight: f32, positions: &mut [Vec3]) {
        for (&v, &d) in self.indices.iter().zip(&self.deltas) {
            if let Some(p) = positions.get_mut(v as usize) {
                *p += d * weight;
            }
        }
    }

    /// Whether every vertex of `mesh` gets the same displacement as its weld key (so the duplicates
    /// of a seam or hard edge move together).
    #[must_use]
    pub fn seam_consistent(&self, mesh: &Mesh) -> bool {
        let keys = mesh.position_keys();
        let delta_of = |v: u32| self.indices.binary_search(&v).map_or(Vec3::ZERO, |i| self.deltas[i]);
        (0..keys.len() as u32).all(|v| delta_of(v) == delta_of(keys[v as usize]))
    }

    /// The layer with each weld key's displacement copied to every duplicate of that key (the
    /// displacement given for the key vertex wins; a displacement given only on a non-key duplicate
    /// is moved to its key first).
    pub fn spread_to_duplicates(&self, mesh: &Mesh) -> EvalResult<Self> {
        let keys = mesh.position_keys();
        let mut per_key = vec![None::<Vec3>; keys.len()];
        // Key vertices first, so that their displacement wins.
        for (&v, &d) in self.indices.iter().zip(&self.deltas) {
            if keys.get(v as usize) == Some(&v) {
                per_key[v as usize] = Some(d);
            }
        }
        for (&v, &d) in self.indices.iter().zip(&self.deltas) {
            if let Some(&k) = keys.get(v as usize)
                && let Some(slot) = per_key.get_mut(k as usize)
                && slot.is_none()
            {
                *slot = Some(d);
            }
        }
        let mut indices = Vec::new();
        let mut deltas = Vec::new();
        for (v, &k) in keys.iter().enumerate() {
            if let Some(Some(d)) = per_key.get(k as usize) {
                indices.push(v as u32);
                deltas.push(*d);
            }
        }
        Self::new(indices, deltas, keys.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::fixtures::cube_split;

    #[test]
    fn layers_are_checked() {
        assert!(SculptLayer::new(vec![0, 2], vec![Vec3::X, Vec3::Y], 3).is_ok());
        assert!(matches!(SculptLayer::new(vec![2, 2], vec![Vec3::X; 2], 3), Err(EvalError::Range { .. })));
        assert!(matches!(SculptLayer::new(vec![3], vec![Vec3::X], 3), Err(EvalError::Missing { .. })));
        assert!(matches!(SculptLayer::new(vec![0], vec![], 3), Err(EvalError::Length { .. })));
        assert!(matches!(SculptLayer::new(vec![0], vec![Vec3::splat(f32::NAN)], 3), Err(EvalError::NonFinite { .. })));
    }

    #[test]
    fn pairs_merge_and_accumulate() {
        let l = SculptLayer::from_pairs(&[(2, Vec3::X), (0, Vec3::Y), (2, Vec3::X), (1, Vec3::ZERO)], 3).unwrap();
        assert_eq!(l.indices(), &[0, 2]);
        assert_eq!(l.deltas(), &[Vec3::Y, Vec3::X * 2.0]);
        let mut p = vec![Vec3::ZERO; 3];
        l.accumulate(0.5, &mut p);
        assert_eq!(p, vec![Vec3::Y * 0.5, Vec3::ZERO, Vec3::X]);
        assert_eq!(l.len(), 2);
        assert!(!l.is_empty());
    }

    #[test]
    fn duplicates_follow_their_key() {
        let mesh = cube_split(false);
        // Vertex 0 is the key of corner (0,0,0); move only it.
        let l = SculptLayer::new(vec![0], vec![Vec3::X], mesh.vertex_count()).unwrap();
        assert!(!l.seam_consistent(&mesh));
        let s = l.spread_to_duplicates(&mesh).unwrap();
        assert!(s.seam_consistent(&mesh));
        assert_eq!(s.len(), 3, "a cube corner has three copies");
        let mut p = mesh.positions().to_vec();
        s.accumulate(1.0, &mut p);
        let moved: Vec<usize> = (0..p.len()).filter(|&i| p[i] != mesh.positions()[i]).collect();
        assert!(moved.iter().all(|&i| mesh.positions()[i] == Vec3::ZERO));
    }
}
