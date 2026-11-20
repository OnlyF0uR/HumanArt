//! Mouth interior shadowing `[MTH-04]`: the lip opening and the occlusion it implies inside the
//! cavity.
//!
//! The light that reaches a point inside the mouth comes through the lip opening. [`MouthOpening`]
//! takes that opening from the deformed lip contact landmarks (`lip.upper.inner.NN`,
//! `lip.lower.inner.NN`, `L.lip.commissure`, `R.lip.commissure`) as an ellipse in the plane facing
//! the head's forward axis. [`MouthOpening::occlusion`] is the fraction of the hemisphere's light
//! that still arrives: the form factor of a disc of the opening's area `A` seen from a point at
//! depth `d` behind it, `F = A / (A + π d²)` (the on-axis disc form factor `r² / (r² + d²)` with
//! `A = π r²`), with the lateral distance beyond the ellipse's rim added to the depth. A closed
//! mouth (`A = 0`) is dark inside; at the opening the factor is 1. The shader evaluates the same
//! function per pixel for teeth, gums, tongue and the skin of the mouth bag
//! (`shaders/mouth/mouth.wgsl`); baked AO (WP-15) multiplies on top. This is the dynamic part
//! that keeps the interior from looking lit and hollow.

use humanart_math::{Quat, Vec3};

use crate::error::{ModelError, Result};
use crate::landmark::Landmark;

/// The most landmarks per lip contact line.
pub const MAX_LIP_POINTS: usize = 32;

/// The lip opening: an ellipse facing `forward`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MouthOpening {
    /// Centre of the opening.
    pub centre: Vec3,
    /// Unit axis out of the mouth (the head's forward).
    pub forward: Vec3,
    /// Unit axis across the mouth (toward the character's left).
    pub across: Vec3,
    /// Half width (to the commissures), metres.
    pub half_width: f32,
    /// Half height (the lip separation at the midline over 2), metres; 0 when closed.
    pub half_height: f32,
}

impl MouthOpening {
    /// The area of the opening, square metres.
    #[must_use]
    pub fn area(&self) -> f32 {
        std::f32::consts::PI * self.half_width * self.half_height
    }

    /// The fraction of light reaching `p` through the opening (1 in front of it).
    #[must_use]
    pub fn occlusion(&self, p: Vec3) -> f32 {
        let q = p - self.centre;
        let depth = -q.dot(self.forward);
        if depth <= 0.0 {
            return 1.0;
        }
        let up = self.forward.cross(self.across);
        let (x, y) = (q.dot(self.across), q.dot(up));
        // Lateral distance outside the ellipse's rim (approximately: radial scaling).
        let (a, b) = (self.half_width.max(1e-5), self.half_height.max(1e-5));
        let e = ((x / a).powi(2) + (y / b).powi(2)).sqrt();
        let outside = if e > 1.0 { (x * x + y * y).sqrt() * (1.0 - 1.0 / e) } else { 0.0 };
        let area = self.area();
        let d2 = depth * depth + outside * outside;
        area / (area + std::f32::consts::PI * d2)
    }
}

/// The lip landmark indices, found once.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LipLandmarks {
    /// Upper lip contact line.
    pub upper: Vec<usize>,
    /// Lower lip contact line.
    pub lower: Vec<usize>,
    /// Left and right commissures.
    pub corners: (usize, usize),
}

impl LipLandmarks {
    /// Finds the contact-line and commissure landmarks.
    ///
    /// # Errors
    /// [`ModelError::Invalid`] when one is missing; [`ModelError::Limit`] beyond
    /// [`MAX_LIP_POINTS`].
    pub fn find(landmarks: &[Landmark]) -> Result<Self> {
        let (mut upper, mut lower) = (Vec::new(), Vec::new());
        let (mut left, mut right) = (None, None);
        for (i, l) in landmarks.iter().enumerate() {
            match l.name.as_str() {
                n if n.starts_with("lip.upper.inner.") => upper.push(i),
                n if n.starts_with("lip.lower.inner.") => lower.push(i),
                "L.lip.commissure" => left = Some(i),
                "R.lip.commissure" => right = Some(i),
                _ => {}
            }
        }
        if upper.len().max(lower.len()) > MAX_LIP_POINTS {
            return Err(ModelError::limit(
                "lip contact landmarks",
                MAX_LIP_POINTS as u64,
                upper.len().max(lower.len()) as u64,
            ));
        }
        let (Some(left), Some(right)) = (left, right) else {
            return Err(ModelError::invalid("lip landmarks", 0, "a commissure landmark is missing"));
        };
        if upper.is_empty() || lower.is_empty() {
            return Err(ModelError::invalid("lip landmarks", 1, "a lip has no contact landmark"));
        }
        Ok(Self { upper, lower, corners: (left, right) })
    }

    /// The opening on deformed `positions` for a head rotated by `head`. The midline separation
    /// is the smallest vertical gap between the two contact lines near the midline (the central
    /// third of the width). No allocation.
    ///
    /// # Errors
    /// [`ModelError::Invalid`] when a landmark's vertex is out of range or the corners coincide.
    pub fn opening(&self, landmarks: &[Landmark], positions: &[Vec3], head: Quat) -> Result<MouthOpening> {
        let at = |i: usize| -> Result<Vec3> {
            landmarks.get(i).and_then(|l| l.position(positions)).ok_or(ModelError::invalid(
                "lip landmarks",
                i,
                "landmark vertex out of range",
            ))
        };
        let (l, r) = (at(self.corners.0)?, at(self.corners.1)?);
        let forward = head.mul_vec3(Vec3::Z).normalize_or(Vec3::Z);
        let up = head.mul_vec3(Vec3::Y).normalize_or(Vec3::Y);
        let across = up.cross(forward).normalize_or(Vec3::X);
        let half_width = (l - r).dot(across) * 0.5;
        if !(half_width.is_finite() && half_width > 1e-4) {
            return Err(ModelError::invalid("lip landmarks", 2, "the commissures coincide"));
        }
        let mid = (l + r) * 0.5;
        // Mean height of each contact line near the midline.
        let band = half_width / 3.0;
        let mut mean = [(0.0_f32, 0.0_f32); 2];
        for (k, list) in [&self.upper, &self.lower].into_iter().enumerate() {
            for &i in list {
                let p = at(i)?;
                if (p - mid).dot(across).abs() <= band {
                    mean[k].0 += (p - mid).dot(up);
                    mean[k].1 += 1.0;
                }
            }
        }
        let height = |m: (f32, f32)| if m.1 > 0.0 { m.0 / m.1 } else { 0.0 };
        let (top, bottom) = (height(mean[0]), height(mean[1]));
        let half_height = ((top - bottom) * 0.5).max(0.0);
        // The centre: between the two contact lines at the midline, at the commissures' depth.
        let centre = mid + up * f32::midpoint(top, bottom);
        Ok(MouthOpening { centre, forward, across, half_width, half_height })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(h: f32) -> MouthOpening {
        MouthOpening { centre: Vec3::ZERO, forward: Vec3::Z, across: Vec3::X, half_width: 0.025, half_height: h }
    }

    #[test]
    fn mth_04_cavity_occlusion_is_the_opening_form_factor() {
        let m = open(0.008);
        assert_eq!(m.occlusion(Vec3::new(0.0, 0.0, 0.01)), 1.0);
        let at = |d: f32| m.occlusion(Vec3::new(0.0, 0.0, -d));
        // On axis: the disc form factor r²/(r²+d²) with the equal-area radius.
        let r2 = 0.025 * 0.008;
        for d in [0.002_f32, 0.01, 0.03] {
            assert!((at(d) - r2 / (r2 + d * d)).abs() < 1e-6);
        }
        assert!(at(0.001) > 0.95 && at(0.03) < 0.25 && at(0.03) > 0.0);
        // Deeper and further outside the rim: darker.
        assert!(at(0.02) < at(0.01));
        let side = m.occlusion(Vec3::new(0.04, 0.0, -0.01));
        assert!(side < at(0.01));
        // Closed: dark inside.
        assert!(open(0.0).occlusion(Vec3::new(0.0, 0.0, -0.01)) < 1e-3);
    }

    #[test]
    fn lip_landmarks_must_exist() {
        assert!(LipLandmarks::find(&[]).is_err());
    }
}
