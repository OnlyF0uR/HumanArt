//! Landmarks `[MOD-06]`: named surface points stored as barycentric coordinates, so they follow
//! every deformation of the mesh.
//!
//! A landmark names three vertices that are corners of one face and three weights that sum to 1.
//! Storing vertices rather than a triangle index keeps landmarks independent of how faces are
//! triangulated. Names follow Appendix C (`L.eye.upper.04`, `R.knee.center`). Validation
//! ([`crate::validate`]) checks that the corners share a face and that the weights are a proper
//! barycentric triple.

use humanart_math::Vec3;

use crate::error::{ModelError, Result};
use crate::names::check_names;

/// The most landmarks a template may have.
pub const MAX_LANDMARKS: usize = 4096;

/// Which landmark set a landmark belongs to (`[MOD-06]` counts them separately).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LandmarkGroup {
    /// Face, head and ear landmarks.
    #[default]
    Face,
    /// Body landmarks: joint centres and anatomical points.
    Body,
}

impl LandmarkGroup {
    /// The byte stored in packages.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::Face => 0,
            Self::Body => 1,
        }
    }

    /// The group for a stored byte, if valid.
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Face),
            1 => Some(Self::Body),
            _ => None,
        }
    }
}

/// One landmark.
#[derive(Clone, Debug, PartialEq)]
pub struct Landmark {
    /// Stable name.
    pub name: String,
    /// The set it belongs to.
    pub group: LandmarkGroup,
    /// Three corners of one face.
    pub vertices: [u32; 3],
    /// Barycentric weights of the corners; they sum to 1.
    pub weights: [f32; 3],
}

impl Landmark {
    /// The landmark's position on a mesh with the given vertex positions; `None` when an index is
    /// out of range.
    #[must_use]
    pub fn position(&self, positions: &[Vec3]) -> Option<Vec3> {
        let mut p = Vec3::ZERO;
        for (&v, &w) in self.vertices.iter().zip(&self.weights) {
            p += *positions.get(v as usize)? * w;
        }
        Some(p)
    }
}

/// Checks the landmark table against a mesh of `vertex_count` vertices: names, index ranges and
/// finite weights. (Whether the weights form a valid barycentric point on a face is a validation
/// item, see [`crate::validate`].)
pub fn check_landmarks(landmarks: &[Landmark], vertex_count: usize) -> Result<()> {
    if landmarks.len() > MAX_LANDMARKS {
        return Err(ModelError::limit("landmark count", MAX_LANDMARKS as u64, landmarks.len() as u64));
    }
    check_names("landmark", landmarks.iter().map(|l| l.name.as_str()))?;
    for (i, l) in landmarks.iter().enumerate() {
        if l.vertices.iter().any(|&v| v as usize >= vertex_count) {
            return Err(ModelError::invalid("landmark", i, "vertex index out of range"));
        }
        if l.weights.iter().any(|w| !w.is_finite()) {
            return Err(ModelError::invalid("landmark", i, "weight is not finite"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_interpolates_and_checks_ranges() {
        let positions = [Vec3::ZERO, Vec3::X, Vec3::Y];
        let l = Landmark {
            name: "nose.tip".into(),
            group: LandmarkGroup::Face,
            vertices: [0, 1, 2],
            weights: [0.5, 0.25, 0.25],
        };
        assert_eq!(l.position(&positions), Some(Vec3::new(0.25, 0.25, 0.0)));
        assert_eq!(l.position(&positions[..2]), None);
        check_landmarks(std::slice::from_ref(&l), 3).unwrap();
        assert!(check_landmarks(std::slice::from_ref(&l), 2).is_err());
        let mut bad = l.clone();
        bad.weights[1] = f32::NAN;
        assert!(check_landmarks(&[bad], 3).is_err());
        assert!(check_landmarks(&[l.clone(), l], 3).is_err());
        for g in [LandmarkGroup::Face, LandmarkGroup::Body] {
            assert_eq!(LandmarkGroup::from_u8(g.to_u8()), Some(g));
        }
        assert_eq!(LandmarkGroup::from_u8(2), None);
    }
}
