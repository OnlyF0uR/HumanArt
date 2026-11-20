//! Levels of detail `[MOD-05]`: LOD1 to LOD4 as their own meshes, each vertex tied back to the
//! hero mesh (LOD0) by a barycentric source, through which regions, skin weights and deltas are
//! transferred.
//!
//! The template's main mesh is LOD0. Each further level stores its mesh, a source triple per vertex
//! (three LOD0 vertices and weights summing to 1), the geometric error its generator declared, and
//! its switch threshold: the level is used once the character's projected height drops below
//! `switch_size` (a fraction of the screen height), with `hysteresis` as a relative dead band.

use humanart_math::Vec3;

use crate::delta::DeltaBases;
use crate::error::{ModelError, Result};
use crate::mesh::Mesh;
use crate::skin::SkinWeights;
use crate::topology::Csr;

/// The most extra levels (LOD1 to LOD4).
pub const MAX_LOD_LEVELS: usize = 4;

/// One reduced level.
#[derive(Clone, Debug, PartialEq)]
pub struct LodLevel {
    /// The level's mesh (its tile indices refer to the template's tile table).
    pub mesh: Mesh,
    /// Per vertex: three LOD0 vertices.
    pub sources: Vec<[u32; 3]>,
    /// Per vertex: the weights of the three sources, each in `[0, 1]`, summing to 1.
    pub source_weights: Vec<[f32; 3]>,
    /// The largest geometric deviation from LOD0 the generator states, metres.
    pub declared_error: f32,
    /// Projected height (fraction of screen height, in `(0, 1]`) below which this level is used;
    /// decreasing from level to level.
    pub switch_size: f32,
    /// Relative dead band around `switch_size`, in `[0, 0.5]`.
    pub hysteresis: f32,
}

impl LodLevel {
    /// Checks the level against LOD0's vertex count and the tile count.
    pub fn check(&self, base_vertex_count: usize, tile_count: usize) -> Result<()> {
        let n = self.mesh.vertex_count();
        if self.sources.len() != n || self.source_weights.len() != n {
            return Err(ModelError::invalid("LOD level", 0, "source arrays differ from the vertex count"));
        }
        if let Some(v) = self.sources.iter().position(|s| s.iter().any(|&x| x as usize >= base_vertex_count)) {
            return Err(ModelError::invalid("LOD vertex", v, "source vertex out of range"));
        }
        if let Some(v) = self
            .source_weights
            .iter()
            .position(|w| w.iter().any(|x| !(0.0..=1.0).contains(x)) || (w[0] + w[1] + w[2] - 1.0).abs() > 1e-3)
        {
            return Err(ModelError::invalid("LOD vertex", v, "source weights not in [0, 1] or not summing to 1"));
        }
        if self.mesh.face_tiles().iter().any(|&t| t as usize >= tile_count) {
            return Err(ModelError::invalid("LOD level", 0, "face tile out of range"));
        }
        if !(self.declared_error.is_finite() && self.declared_error >= 0.0) {
            return Err(ModelError::invalid("LOD level", 0, "declared error is not a finite non-negative length"));
        }
        if self.switch_size.is_nan()
            || !(self.switch_size > 0.0 && self.switch_size <= 1.0)
            || !(0.0..=0.5).contains(&self.hysteresis)
        {
            return Err(ModelError::invalid(
                "LOD level",
                0,
                "switch size outside (0, 1] or hysteresis outside [0, 0.5]",
            ));
        }
        Ok(())
    }

    /// Interpolates a per-vertex LOD0 attribute at this level's vertices; `None` when `base` is too
    /// short.
    #[must_use]
    pub fn transfer_positions(&self, base: &[Vec3]) -> Option<Vec<Vec3>> {
        self.sources
            .iter()
            .zip(&self.source_weights)
            .map(|(s, w)| {
                let mut p = Vec3::ZERO;
                for k in 0..3 {
                    p += *base.get(s[k] as usize)? * w[k];
                }
                Some(p)
            })
            .collect()
    }

    /// Transfers per-vertex region indices: each vertex takes the region of its heaviest source.
    #[must_use]
    pub fn transfer_regions(&self, base: &[u16]) -> Option<Vec<u16>> {
        self.sources
            .iter()
            .zip(&self.source_weights)
            .map(|(s, w)| {
                let k = if w[1] > w[0] && w[1] >= w[2] {
                    1
                } else if w[2] > w[0] && w[2] > w[1] {
                    2
                } else {
                    0
                };
                base.get(s[k] as usize).copied()
            })
            .collect()
    }

    /// Transfers skin weights: the sources' influences blended by the source weights, then the
    /// largest kept and normalised.
    pub fn transfer_skin(&self, skin: &SkinWeights) -> Result<SkinWeights> {
        if skin.influences == 0 {
            return Ok(SkinWeights {
                dual_quaternion_regions: skin.dual_quaternion_regions.clone(),
                ..SkinWeights::default()
            });
        }
        let mut lists: Vec<Vec<(u16, f32)>> = Vec::with_capacity(self.sources.len());
        for (v, (s, w)) in self.sources.iter().zip(&self.source_weights).enumerate() {
            let mut list = Vec::new();
            for k in 0..3 {
                let (joints, weights) = skin.vertex(s[k] as usize);
                if joints.is_empty() {
                    return Err(ModelError::invalid("LOD vertex", v, "source vertex has no skin weights"));
                }
                list.extend(joints.iter().zip(weights).map(|(&j, &x)| (j, x * w[k])));
            }
            lists.push(list);
        }
        let fallback = skin.joints.first().copied().unwrap_or(0);
        let mut out = SkinWeights::from_lists(skin.influences as usize, lists.iter().map(Vec::as_slice), fallback)?;
        out.dual_quaternion_regions.clone_from(&skin.dual_quaternion_regions);
        Ok(out)
    }

    /// Transfers delta bases: each LOD vertex's displacement is the weighted sum of its sources'.
    /// Components keep their names, kinds, regions and order. `bases` must have been checked
    /// against LOD0.
    pub fn transfer_deltas(&self, bases: &DeltaBases) -> Result<DeltaBases> {
        let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(3 * self.sources.len());
        let mut base_count = 0_usize;
        for (v, s) in self.sources.iter().enumerate() {
            for (k, &b) in s.iter().enumerate() {
                // A source listed twice in one triple is visited once (its weights are summed below).
                if s[..k].contains(&b) {
                    continue;
                }
                pairs.push((b, v as u32));
                base_count = base_count.max(b as usize + 1);
            }
        }
        let users = Csr::from_pairs(base_count, &pairs);
        let mut accum = vec![Vec3::ZERO; self.sources.len()];
        let mut touched: Vec<u32> = Vec::new();
        let mut builder = crate::delta::DeltaBasesBuilder::new();
        let mut entries: Vec<(u32, Vec3)> = Vec::new();
        for (c, comp) in bases.components().iter().enumerate() {
            touched.clear();
            for (&b, &d) in bases.indices(c).iter().zip(bases.deltas(c)) {
                let d = DeltaBases::delta_f32(d);
                for &v in users.row(b as usize) {
                    let (s, w) = (self.sources[v as usize], self.source_weights[v as usize]);
                    let weight: f32 = (0..3).filter(|&k| s[k] == b).map(|k| w[k]).sum();
                    if accum[v as usize] == Vec3::ZERO {
                        touched.push(v);
                    }
                    accum[v as usize] += d * weight;
                }
            }
            touched.sort_unstable();
            touched.dedup();
            entries.clear();
            for &v in &touched {
                entries.push((v, accum[v as usize]));
                accum[v as usize] = Vec3::ZERO;
            }
            builder.add(&comp.name, comp.kind, comp.region, &entries)?;
        }
        builder.finish()
    }
}

/// Checks the LOD levels: at most [`MAX_LOD_LEVELS`], each valid, switch sizes strictly decreasing.
pub fn check_lods(levels: &[LodLevel], base_vertex_count: usize, tile_count: usize) -> Result<()> {
    if levels.len() > MAX_LOD_LEVELS {
        return Err(ModelError::limit("LOD level count", MAX_LOD_LEVELS as u64, levels.len() as u64));
    }
    for (i, l) in levels.iter().enumerate() {
        l.check(base_vertex_count, tile_count).map_err(|e| match e {
            ModelError::Invalid { item: "LOD level", what, .. } => ModelError::invalid("LOD level", i + 1, what),
            other => other,
        })?;
    }
    if let Some(i) = levels.windows(2).position(|w| w[1].switch_size >= w[0].switch_size) {
        return Err(ModelError::invalid("LOD level", i + 2, "switch size does not decrease"));
    }
    Ok(())
}

/// Chooses the level for a character of projected height `screen_size` (fraction of the screen
/// height) when level `current` is shown: 0 is LOD0. A level switches coarser only below
/// `switch_size * (1 - hysteresis)` and back finer only above `switch_size * (1 + hysteresis)`.
#[must_use]
pub fn select_lod(levels: &[LodLevel], screen_size: f32, current: usize) -> usize {
    let mut lod = 0;
    for (i, l) in levels.iter().enumerate() {
        let coarser_now = current > i;
        let use_it = if coarser_now {
            screen_size <= l.switch_size * (1.0 + l.hysteresis)
        } else {
            screen_size < l.switch_size * (1.0 - l.hysteresis)
        };
        if use_it {
            lod = i + 1;
        } else {
            break;
        }
    }
    lod
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta::{DeltaBasesBuilder, DeltaKind};
    use crate::mesh::fixtures::grid;

    /// LOD1 of a 2x2 grid: the single quad over its corners.
    fn level() -> LodLevel {
        // grid(1) vertices are (0,0), (1,0), (0,1), (1,1); grid(2) has them as 0, 2, 6, 8.
        LodLevel {
            mesh: grid(1),
            sources: vec![[0, 1, 2], [2, 1, 0], [6, 7, 8], [8, 7, 6]],
            source_weights: vec![[1.0, 0.0, 0.0]; 4],
            declared_error: 0.0,
            switch_size: 0.5,
            hysteresis: 0.1,
        }
    }

    #[test]
    fn transfers_follow_the_sources() {
        let base = grid(2);
        let l = level();
        l.check(9, 1).unwrap();
        assert_eq!(l.transfer_positions(base.positions()).unwrap(), l.mesh.positions());
        assert!(l.transfer_positions(&base.positions()[..5]).is_none());
        let regions: Vec<u16> = (0..9).collect();
        assert_eq!(l.transfer_regions(&regions).unwrap(), vec![0, 2, 6, 8]);

        let lists: Vec<Vec<(u16, f32)>> = (0..9).map(|v| vec![((v % 2) as u16, 1.0)]).collect();
        let skin = SkinWeights::from_lists(2, lists.iter().map(Vec::as_slice), 0).unwrap();
        let t = l.transfer_skin(&skin).unwrap();
        assert_eq!(t.vertex(0).0[0], 0);
        assert_eq!(t.vertex_count(), 4);

        let mut b = DeltaBasesBuilder::new();
        b.add("push", DeltaKind::Semantic, 0, &[(8, Vec3::Z), (4, Vec3::Z)]).unwrap();
        let d = b.finish().unwrap();
        let td = l.transfer_deltas(&d).unwrap();
        assert_eq!(td.indices(0), &[3]);
        assert_eq!(DeltaBases::delta_f32(td.deltas(0)[0]), Vec3::Z);
    }

    #[test]
    fn blended_sources_mix() {
        let mut l = level();
        l.source_weights[1] = [0.5, 0.5, 0.0];
        let regions: Vec<u16> = (0..9).collect();
        assert_eq!(l.transfer_regions(&regions).unwrap()[1], 2);
        let lists: Vec<Vec<(u16, f32)>> = (0..9).map(|v| vec![(v as u16, 1.0)]).collect();
        let skin = SkinWeights::from_lists(2, lists.iter().map(Vec::as_slice), 0).unwrap();
        let t = l.transfer_skin(&skin).unwrap();
        let (j, w) = t.vertex(1);
        assert_eq!((j, w), (&[1_u16, 2][..], &[0.5_f32, 0.5][..]));
    }

    #[test]
    fn checks_and_selection() {
        let l = level();
        assert!(l.check(8, 1).is_err());
        assert!(l.check(9, 0).is_err());
        let mut bad = l.clone();
        bad.source_weights[0] = [0.5, 0.0, 0.0];
        assert!(bad.check(9, 1).is_err());
        let mut bad = l.clone();
        bad.switch_size = 0.0;
        assert!(bad.check(9, 1).is_err());
        let mut bad = l.clone();
        bad.sources.pop();
        assert!(bad.check(9, 1).is_err());

        let mut l2 = l.clone();
        l2.switch_size = 0.25;
        check_lods(&[l.clone(), l2.clone()], 9, 1).unwrap();
        assert!(check_lods(&[l2.clone(), l.clone()], 9, 1).is_err());
        assert!(check_lods(&vec![l.clone(); 5], 9, 1).is_err());

        let levels = [l, l2];
        assert_eq!(select_lod(&levels, 0.9, 0), 0);
        assert_eq!(select_lod(&levels, 0.47, 0), 0); // inside the dead band: stay
        assert_eq!(select_lod(&levels, 0.44, 0), 1);
        assert_eq!(select_lod(&levels, 0.53, 1), 1); // inside the dead band: stay
        assert_eq!(select_lod(&levels, 0.56, 1), 0);
        assert_eq!(select_lod(&levels, 0.1, 0), 2);
        assert_eq!(select_lod(&[], 0.1, 0), 0);
    }
}
