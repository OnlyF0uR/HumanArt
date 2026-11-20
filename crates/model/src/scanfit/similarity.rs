//! A similarity transform (rotation, uniform scale, translation) `[WP-25]`: how a fit maps the
//! template into a scan's frame.

use humanart_math::Vec3;

/// A similarity transform `p ↦ scale · R · p + translation`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Similarity {
    /// The rotation as rows (proper: determinant +1).
    pub rotation: [Vec3; 3],
    /// The uniform scale.
    pub scale: f32,
    /// The translation, applied after rotation and scale.
    pub translation: Vec3,
}

impl Similarity {
    /// The identity.
    pub const IDENTITY: Self = Self { rotation: [Vec3::X, Vec3::Y, Vec3::Z], scale: 1.0, translation: Vec3::ZERO };

    /// Transforms a point.
    #[must_use]
    pub fn apply(&self, p: Vec3) -> Vec3 {
        self.rotate(p) * self.scale + self.translation
    }

    /// Rotates a direction (no scale, no translation).
    #[must_use]
    pub fn rotate(&self, v: Vec3) -> Vec3 {
        Vec3::new(self.rotation[0].dot(v), self.rotation[1].dot(v), self.rotation[2].dot(v))
    }

    /// The inverse transform (`scale` must not be 0).
    #[must_use]
    pub fn inverse(&self) -> Self {
        let [r0, r1, r2] = self.rotation;
        // The transpose of a rotation is its inverse.
        let rotation = [Vec3::new(r0.x, r1.x, r2.x), Vec3::new(r0.y, r1.y, r2.y), Vec3::new(r0.z, r1.z, r2.z)];
        let inv = Self { rotation, scale: 1.0 / self.scale, translation: Vec3::ZERO };
        Self { translation: -inv.rotate(self.translation) * inv.scale, ..inv }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wp25_similarity_inverse_undoes_apply() {
        let (c, s) = (0.6_f32.cos(), 0.6_f32.sin());
        let t = Similarity {
            rotation: [Vec3::new(c, 0.0, s), Vec3::Y, Vec3::new(-s, 0.0, c)],
            scale: 1.07,
            translation: Vec3::new(0.02, -0.3, 0.1),
        };
        for p in [Vec3::ZERO, Vec3::new(0.1, 0.2, -0.3), Vec3::ONE] {
            assert!(t.inverse().apply(t.apply(p)).abs_diff_eq(p, 1e-6));
        }
        assert_eq!(Similarity::IDENTITY.apply(Vec3::ONE), Vec3::ONE);
    }
}
