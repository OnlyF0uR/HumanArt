//! Skin weights `[SKIN-01]`: a fixed number of joint influences per vertex, normalised, plus the
//! regions where dual-quaternion skinning replaces linear blend skinning.
//!
//! Influences are stored as two flat arrays of `influences * vertex_count` entries (vertex-major),
//! sorted by decreasing weight within a vertex; unused slots have weight 0 and joint 0. The fixed
//! count keeps the GPU skinning pass branch-free.

use crate::error::{ModelError, Result};

/// The most joint influences per vertex.
pub const MAX_INFLUENCES: usize = 8;

/// Per-vertex skin weights.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SkinWeights {
    /// Influences per vertex, 1 to [`MAX_INFLUENCES`]; 0 only for an unskinned template (no joints,
    /// empty arrays).
    pub influences: u8,
    /// `influences * vertex_count` joint indices.
    pub joints: Vec<u16>,
    /// `influences * vertex_count` weights, each in `[0, 1]`, per vertex sorted decreasing and
    /// summing to 1 (validation reports deviations).
    pub weights: Vec<f32>,
    /// Regions skinned with dual quaternions (neck, shoulders, forearms, wrists), sorted, unique.
    pub dual_quaternion_regions: Vec<u16>,
}

impl SkinWeights {
    /// Builds normalised weights from per-vertex `(joint, weight)` lists: keeps the `influences`
    /// largest positive weights of each vertex, merges repeated joints, sorts them decreasing and
    /// scales them to sum to 1. A vertex with no positive weight is bound fully to `fallback_joint`.
    pub fn from_lists<'a>(
        influences: usize,
        lists: impl Iterator<Item = &'a [(u16, f32)]>,
        fallback_joint: u16,
    ) -> Result<Self> {
        if !(1..=MAX_INFLUENCES).contains(&influences) {
            return Err(ModelError::limit("skin influences", MAX_INFLUENCES as u64, influences as u64));
        }
        let mut out = Self { influences: influences as u8, ..Self::default() };
        let mut merged: Vec<(u16, f32)> = Vec::new();
        for (v, list) in lists.enumerate() {
            merged.clear();
            for &(j, w) in list {
                if !w.is_finite() {
                    return Err(ModelError::invalid("skin vertex", v, "weight is not finite"));
                }
                if w <= 0.0 {
                    continue;
                }
                match merged.iter_mut().find(|e| e.0 == j) {
                    Some(e) => e.1 += w,
                    None => merged.push((j, w)),
                }
            }
            merged.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            merged.truncate(influences);
            if merged.is_empty() {
                merged.push((fallback_joint, 1.0));
            }
            let sum: f32 = merged.iter().map(|e| e.1).sum();
            for k in 0..influences {
                let (j, w) = merged.get(k).map_or((0, 0.0), |&(j, w)| (j, w / sum));
                out.joints.push(j);
                out.weights.push(w);
            }
        }
        Ok(out)
    }

    /// Checks the arrays against the vertex, joint and region counts.
    pub fn check(&self, vertex_count: usize, joint_count: usize, region_count: usize) -> Result<()> {
        let k = self.influences as usize;
        if k > MAX_INFLUENCES {
            return Err(ModelError::limit("skin influences", MAX_INFLUENCES as u64, k as u64));
        }
        if k == 0 && (joint_count != 0 || !self.joints.is_empty() || !self.weights.is_empty()) {
            return Err(ModelError::invalid("skin weights", 0, "no influences although the template has joints"));
        }
        let expected = k * vertex_count;
        if self.joints.len() != expected || self.weights.len() != expected {
            return Err(ModelError::invalid("skin weights", 0, "array length is not influences * vertex_count"));
        }
        if let Some(i) = self.joints.iter().position(|&j| j as usize >= joint_count) {
            return Err(ModelError::invalid("skin influence", i, "joint index out of range"));
        }
        if let Some(i) = self.weights.iter().position(|w| !(0.0..=1.0).contains(w)) {
            return Err(ModelError::invalid("skin influence", i, "weight outside [0, 1]"));
        }
        let regions = &self.dual_quaternion_regions;
        if regions.windows(2).any(|w| w[0] >= w[1]) || regions.last().is_some_and(|&r| r as usize >= region_count) {
            return Err(ModelError::invalid(
                "skin weights",
                0,
                "dual-quaternion regions not sorted, unique and in range",
            ));
        }
        Ok(())
    }

    /// The influences of vertex `v` (empty when out of range).
    #[must_use]
    pub fn vertex(&self, v: usize) -> (&[u16], &[f32]) {
        let k = self.influences as usize;
        match (self.joints.get(v * k..v * k + k), self.weights.get(v * k..v * k + k)) {
            (Some(j), Some(w)) => (j, w),
            _ => (&[], &[]),
        }
    }

    /// How many vertices the arrays hold.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        match self.influences {
            0 => 0,
            k => self.weights.len() / k as usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_lists_keeps_the_largest_and_normalises() {
        let lists: [&[(u16, f32)]; 3] = [&[(3, 0.1), (1, 0.6), (2, 0.3), (1, 0.2)], &[], &[(5, -1.0), (4, 2.0)]];
        let s = SkinWeights::from_lists(2, lists.into_iter(), 7).unwrap();
        assert_eq!(s.vertex_count(), 3);
        let (j, w) = s.vertex(0);
        assert_eq!(j, &[1, 2]);
        assert!((w[0] - 0.8 / 1.1).abs() < 1e-6 && (w[1] - 0.3 / 1.1).abs() < 1e-6);
        assert_eq!(s.vertex(1), (&[7_u16, 0][..], &[1.0_f32, 0.0][..]));
        assert_eq!(s.vertex(2), (&[4_u16, 0][..], &[1.0_f32, 0.0][..]));
        assert!(s.vertex(3).0.is_empty());
        s.check(3, 8, 0).unwrap();
        assert!(s.check(3, 7, 0).is_err());
        assert!(s.check(2, 8, 0).is_err());
        assert!(SkinWeights::from_lists(9, std::iter::empty(), 0).is_err());
        let nan: [&[(u16, f32)]; 1] = [&[(0, f32::NAN)]];
        assert!(SkinWeights::from_lists(1, nan.into_iter(), 0).is_err());
    }

    #[test]
    fn checks_refuse_broken_weights() {
        let lists: [&[(u16, f32)]; 2] = [&[(0, 1.0)], &[(1, 1.0)]];
        let good = SkinWeights {
            dual_quaternion_regions: vec![0, 2],
            ..SkinWeights::from_lists(1, lists.into_iter(), 0).unwrap()
        };
        good.check(2, 2, 3).unwrap();
        let mut s = good.clone();
        s.weights[0] = 1.5;
        assert!(s.check(2, 2, 3).is_err());
        let mut s = good.clone();
        s.dual_quaternion_regions = vec![2, 0];
        assert!(s.check(2, 2, 3).is_err());
        assert!(good.check(2, 2, 2).is_err());
        let unskinned = SkinWeights::default();
        unskinned.check(5, 0, 0).unwrap();
        assert!(unskinned.check(5, 1, 0).is_err());
        assert_eq!(unskinned.vertex_count(), 0);
    }
}
