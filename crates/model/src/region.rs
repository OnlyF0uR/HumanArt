//! The region map `[MOD-03]`: a hierarchy of named regions, the region of every vertex, and
//! optional soft membership weights.
//!
//! Every vertex belongs to exactly one region (normally a leaf: `ear-L.helix` rather than `ear-L`).
//! Regions form a tree through [`Region::parent`]; a parent always comes before its children in the
//! table, so the hierarchy cannot have cycles. Delta components name a *bounding region* whose
//! subtree contains every vertex they touch `[EVAL-02]`, and region-local identity blending uses the
//! soft weights `[ID-02]`.

use crate::error::{ModelError, Result};
use crate::names::{Side, check_names};

/// The most regions a template may have (region ids are `u16`).
pub const MAX_REGIONS: usize = 4096;
/// The most soft membership entries one vertex may have.
pub const MAX_SOFT_PER_VERTEX: usize = 16;

/// One named region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    /// Stable name (`upper-lid-L`, `ear-L.helix`).
    pub name: String,
    /// The enclosing region; it has a lower index than this one.
    pub parent: Option<u16>,
    /// The side of the body the region is on.
    pub side: Side,
}

/// Soft membership weights per vertex, in CSR layout: vertex `v` has the entries
/// `offsets[v]..offsets[v + 1]` of `regions` and `weights`. A vertex's weights sum to 1
/// (validation reports deviations); vertices far from a boundary have one entry or none (none means
/// "entirely its own region").
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SoftRegionWeights {
    /// `vertex_count + 1` offsets, starting at 0, non-decreasing.
    pub offsets: Vec<u32>,
    /// Region index of each entry.
    pub regions: Vec<u16>,
    /// Weight of each entry, in `[0, 1]`.
    pub weights: Vec<f32>,
}

impl SoftRegionWeights {
    /// The entries of vertex `v` (empty when out of range).
    #[must_use]
    pub fn vertex(&self, v: usize) -> (&[u16], &[f32]) {
        match (self.offsets.get(v), self.offsets.get(v + 1)) {
            (Some(&a), Some(&b)) if a <= b && b as usize <= self.regions.len().min(self.weights.len()) => {
                (&self.regions[a as usize..b as usize], &self.weights[a as usize..b as usize])
            }
            _ => (&[], &[]),
        }
    }
}

/// The regions of a template and the region of each vertex.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RegionMap {
    /// The region table; parents come before children.
    pub regions: Vec<Region>,
    /// The region index of each vertex.
    pub vertex_regions: Vec<u16>,
    /// Optional soft membership weights.
    pub soft: Option<SoftRegionWeights>,
}

impl RegionMap {
    /// Checks the table and the per-vertex arrays against a mesh of `vertex_count` vertices.
    pub fn check(&self, vertex_count: usize) -> Result<()> {
        let n = self.regions.len();
        if n > MAX_REGIONS {
            return Err(ModelError::limit("region count", MAX_REGIONS as u64, n as u64));
        }
        check_names("region", self.regions.iter().map(|r| r.name.as_str()))?;
        for (i, r) in self.regions.iter().enumerate() {
            if r.parent.is_some_and(|p| p as usize >= i) {
                return Err(ModelError::invalid("region", i, "parent does not come before the region"));
            }
        }
        if self.vertex_regions.len() != vertex_count {
            return Err(ModelError::invalid("region map", 0, "vertex region count differs from the vertex count"));
        }
        if let Some(v) = self.vertex_regions.iter().position(|&r| r as usize >= n) {
            return Err(ModelError::invalid("vertex", v, "region index out of range"));
        }
        if let Some(soft) = &self.soft {
            check_soft(soft, vertex_count, n)?;
        }
        Ok(())
    }

    /// The index of the region called `name`.
    #[must_use]
    pub fn find(&self, name: &str) -> Option<u16> {
        self.regions.iter().position(|r| r.name == name).map(|i| i as u16)
    }

    /// Whether `region` is `ancestor` or lies in its subtree. Out-of-range indices are in nothing.
    #[must_use]
    pub fn is_within(&self, region: u16, ancestor: u16) -> bool {
        let mut at = Some(region);
        // Parents have lower indices, so the walk ends after at most `regions.len()` steps.
        for _ in 0..=self.regions.len() {
            match at {
                Some(r) if r == ancestor => return (r as usize) < self.regions.len(),
                Some(r) => at = self.regions.get(r as usize).and_then(|x| x.parent),
                None => return false,
            }
        }
        false
    }

    /// Whether region `r` has child regions.
    #[must_use]
    pub fn has_children(&self, r: u16) -> bool {
        self.regions.iter().any(|x| x.parent == Some(r))
    }

    /// How many vertices each region holds directly (not counting its subtree).
    #[must_use]
    pub fn vertex_counts(&self) -> Vec<u32> {
        let mut counts = vec![0_u32; self.regions.len()];
        for &r in &self.vertex_regions {
            if let Some(c) = counts.get_mut(r as usize) {
                *c += 1;
            }
        }
        counts
    }
}

fn check_soft(soft: &SoftRegionWeights, vertex_count: usize, regions: usize) -> Result<()> {
    if soft.offsets.len() != vertex_count + 1 || soft.offsets.first() != Some(&0) {
        return Err(ModelError::invalid(
            "soft region weights",
            0,
            "offsets do not have vertex_count + 1 entries from 0",
        ));
    }
    if soft.regions.len() != soft.weights.len() || soft.offsets.last().copied() != Some(soft.regions.len() as u32) {
        return Err(ModelError::invalid("soft region weights", 0, "entry arrays differ from the offsets"));
    }
    for (v, w) in soft.offsets.windows(2).enumerate() {
        let len = w[1].checked_sub(w[0]).ok_or(ModelError::invalid("soft region weights", v, "offsets decrease"))?;
        if len as usize > MAX_SOFT_PER_VERTEX {
            return Err(ModelError::invalid("soft region weights", v, "more than 16 entries for one vertex"));
        }
    }
    if let Some(i) = soft.regions.iter().position(|&r| r as usize >= regions) {
        return Err(ModelError::invalid("soft region entry", i, "region index out of range"));
    }
    if let Some(i) = soft.weights.iter().position(|w| !(0.0..=1.0).contains(w)) {
        return Err(ModelError::invalid("soft region entry", i, "weight outside [0, 1]"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(name: &str, parent: Option<u16>) -> Region {
        Region { name: name.into(), parent, side: Side::Center }
    }

    fn sample() -> RegionMap {
        RegionMap {
            regions: vec![
                region("body", None),
                region("head", Some(0)),
                region("ear-L", Some(1)),
                region("torso", Some(0)),
            ],
            vertex_regions: vec![2, 2, 3, 1],
            soft: Some(SoftRegionWeights {
                offsets: vec![0, 2, 2, 2, 2],
                regions: vec![2, 1],
                weights: vec![0.75, 0.25],
            }),
        }
    }

    #[test]
    fn hierarchy_queries() {
        let m = sample();
        m.check(4).unwrap();
        assert_eq!(m.find("ear-L"), Some(2));
        assert_eq!(m.find("nose"), None);
        assert!(m.is_within(2, 0) && m.is_within(2, 1) && m.is_within(2, 2));
        assert!(!m.is_within(2, 3) && !m.is_within(0, 1));
        assert!(!m.is_within(9, 9));
        assert!(m.has_children(1) && !m.has_children(2));
        assert_eq!(m.vertex_counts(), vec![0, 1, 2, 1]);
        let soft = m.soft.as_ref().unwrap();
        assert_eq!(soft.vertex(0), (&[2_u16, 1][..], &[0.75_f32, 0.25][..]));
        assert_eq!(soft.vertex(1).0.len(), 0);
        assert_eq!(soft.vertex(7).0.len(), 0);
    }

    #[test]
    fn checks_refuse_broken_maps() {
        type Change = fn(&mut RegionMap);
        let cases: [(&str, Change); 9] = [
            ("parent after", |m| m.regions[1].parent = Some(2)),
            ("self parent", |m| m.regions[0].parent = Some(0)),
            ("dup name", |m| m.regions[3].name = "head".into()),
            ("vertex count", |m| m.vertex_regions.truncate(3)),
            ("region index", |m| m.vertex_regions[0] = 4),
            ("soft offsets", |m| m.soft.as_mut().unwrap().offsets.truncate(4)),
            ("soft decreasing", |m| m.soft.as_mut().unwrap().offsets[1] = 3),
            ("soft weight", |m| m.soft.as_mut().unwrap().weights[0] = 1.5),
            ("soft region", |m| m.soft.as_mut().unwrap().regions[1] = 7),
        ];
        for (what, change) in cases {
            let mut m = sample();
            change(&mut m);
            assert!(m.check(4).is_err(), "{what} was accepted");
        }
        let mut m = sample();
        m.soft.as_mut().unwrap().weights[0] = f32::NAN;
        assert!(m.check(4).is_err());
    }
}
