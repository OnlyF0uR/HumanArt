//! Gaze `[EYE-06]`: gaze angles, the look-at solve for a 3D target and the eye joint rotation.
//!
//! Angles are in the head frame (x to the character's left, y up, z forward): `yaw` turns the
//! gaze toward +X, `pitch` up. No microsaccade jitter is applied (the editor shows a steady gaze).

use humanart_math::{Quat, Vec3};

use super::params::EyeParams;

/// Gaze angles of one eye, radians.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Gaze {
    /// Horizontal angle, positive toward the character's left (+X).
    pub yaw: f32,
    /// Vertical angle, positive up.
    pub pitch: f32,
}

impl Gaze {
    /// The angles clamped to the eye's limits (non-finite angles become 0).
    #[must_use]
    pub fn clamped(self, params: &EyeParams) -> Self {
        let c = |v: f32, l: f32| if v.is_finite() { v.clamp(-l, l) } else { 0.0 };
        Self { yaw: c(self.yaw, params.gaze_limit_yaw), pitch: c(self.pitch, params.gaze_limit_pitch) }
    }

    /// The gaze that looks from `eye_centre` at `target` (world), for a head whose rotation is
    /// `head` (head frame to world), clamped to the limits. A target at the centre keeps the
    /// straight gaze.
    #[must_use]
    pub fn look_at(eye_centre: Vec3, head: Quat, target: Vec3, params: &EyeParams) -> Self {
        let d = head.conjugate().mul_vec3(target - eye_centre);
        if d.length_squared() < 1e-12 || !d.is_finite() {
            return Self::default();
        }
        let yaw = d.x.atan2(d.z);
        let pitch = d.y.atan2((d.x * d.x + d.z * d.z).sqrt());
        Self { yaw, pitch }.clamped(params)
    }

    /// The gaze direction in the head frame (unit).
    #[must_use]
    pub fn direction(self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cp * sy, sp, cp * cy)
    }

    /// The rotation in the head frame that turns the straight gaze (+Z) into [`Gaze::direction`]:
    /// yaw about Y after pitch about X, so the eye's horizontal axis stays level (no torsion).
    #[must_use]
    pub fn rotation(self) -> Quat {
        Quat::from_axis_angle(Vec3::Y, self.yaw) * Quat::from_axis_angle(Vec3::X, -self.pitch)
    }

    /// The local pose rotation of an eye joint whose rest world rotation is `rest_world` (the
    /// evaluator's `pose` entry), for a head frame equal to the world frame at rest.
    #[must_use]
    pub fn joint_rotation(self, rest_world: Quat) -> Quat {
        (rest_world.conjugate() * self.rotation() * rest_world).normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eye_06_look_at_points_the_gaze_at_the_target() {
        let p = EyeParams::default();
        let centre = Vec3::new(0.032, 1.6, 0.07);
        let target = centre + Vec3::new(0.1, -0.05, 0.6);
        let g = Gaze::look_at(centre, Quat::IDENTITY, target, &p);
        let want = (target - centre).normalize();
        assert!(g.direction().distance(want) < 1e-5);
        assert!(g.rotation().mul_vec3(Vec3::Z).distance(want) < 1e-5);
        // No torsion: the rotated X axis stays horizontal.
        assert!(g.rotation().mul_vec3(Vec3::X).y.abs() < 1e-6);
        // With a turned head the same world target gives the angles in the head frame.
        let head = Quat::from_axis_angle(Vec3::Y, 0.3);
        let g2 = Gaze::look_at(centre, head, target, &p);
        assert!(head.mul_vec3(g2.direction()).distance(want) < 1e-5);
    }

    #[test]
    fn eye_06_gaze_is_clamped_to_the_limits() {
        let p = EyeParams::default();
        let behind = Gaze::look_at(Vec3::ZERO, Quat::IDENTITY, Vec3::new(1.0, 5.0, -1.0), &p);
        assert!((behind.yaw - p.gaze_limit_yaw).abs() < 1e-6 && (behind.pitch - p.gaze_limit_pitch).abs() < 1e-6);
        assert_eq!(Gaze::look_at(Vec3::ONE, Quat::IDENTITY, Vec3::ONE, &p), Gaze::default());
        assert_eq!(Gaze { yaw: f32::NAN, pitch: 0.1 }.clamped(&p), Gaze { yaw: 0.0, pitch: 0.1 });
    }

    #[test]
    fn joint_rotation_matches_the_world_rotation_at_rest() {
        let rest = Quat::from_axis_angle(Vec3::new(1.0, 0.2, 0.0).normalize(), 1.1);
        let g = Gaze { yaw: 0.2, pitch: -0.1 };
        let local = g.joint_rotation(rest);
        // The posed world rotation is rest * local; applied relative to rest it equals g's rotation.
        let world = rest * local * rest.conjugate();
        assert!(world.angle_between(g.rotation()) < 1e-5);
    }
}
