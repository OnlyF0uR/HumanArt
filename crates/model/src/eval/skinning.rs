//! Step 4 of the evaluation: joint posing and skinning `[SKIN-01]`.
//!
//! Joint frames: the **bind frame** of joint `j` has the rotation `Rb_j` of the skeleton's world rest
//! transform and the position `bp_j` the joint regressor gives for the current shape `[MOD-07]`
//! (rest scales are not part of skinning). A pose gives every joint a rotation `q_j` relative to its
//! rest, in its own frame. The posed frames follow the hierarchy (parents come first):
//!
//! - skinning rotation `D_j = D_parent · Rb_j · q_j · Rb_j⁻¹` (`D_parent` = identity for a root),
//! - posed position `wp_j = wp_parent + D_parent · (bp_j − bp_parent)` (`bp_j` for a root),
//! - skinning transform `S_j(p) = D_j · p + t_j` with `t_j = wp_j − D_j · bp_j`.
//!
//! `S_j` maps the rest-pose surface to the posed one. Linear blend skinning averages the transformed
//! points `Σ w_k S_k(p) / Σ w_k`; dual-quaternion skinning blends the unit dual quaternions of `S_k`
//! (hemisphere-aligned to the first influence) and normalises. Both are written per vertex without
//! allocation; the GPU pass of WP-10 computes the same formulas.

use humanart_math::{DualQuat, Mat3, Quat, Vec3};

/// Which skinning a vertex gets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SkinningMode {
    /// Dual quaternions in the template's dual-quaternion regions, linear blending elsewhere
    /// `[SKIN-01]` (the default).
    #[default]
    ByRegion,
    /// Linear blend skinning everywhere.
    Linear,
    /// Dual-quaternion skinning everywhere.
    DualQuaternion,
}

/// Posed joint frames and skinning transforms, one entry per joint in each array (SoA).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct JointFrames {
    /// Bind position `bp_j` (world rest position from the regressor).
    pub bind_positions: Vec<Vec3>,
    /// Posed world position `wp_j`.
    pub world_positions: Vec<Vec3>,
    /// Posed world rotation `D_j · Rb_j`.
    pub world_rotations: Vec<Quat>,
    /// Skinning rotation `D_j`.
    pub skin_rotations: Vec<Quat>,
    /// `D_j` as a matrix, for linear blending.
    pub skin_matrices: Vec<Mat3>,
    /// Skinning translation `t_j`.
    pub skin_translations: Vec<Vec3>,
    /// `S_j` as a unit dual quaternion.
    pub skin_dual_quats: Vec<DualQuat>,
}

impl JointFrames {
    /// Frames for `joints` joints, all identity.
    #[must_use]
    pub fn new(joints: usize) -> Self {
        Self {
            bind_positions: vec![Vec3::ZERO; joints],
            world_positions: vec![Vec3::ZERO; joints],
            world_rotations: vec![Quat::IDENTITY; joints],
            skin_rotations: vec![Quat::IDENTITY; joints],
            skin_matrices: vec![Mat3::IDENTITY; joints],
            skin_translations: vec![Vec3::ZERO; joints],
            skin_dual_quats: vec![DualQuat::IDENTITY; joints],
        }
    }

    /// How many joints the frames hold.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bind_positions.len()
    }

    /// Whether there are no joints.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bind_positions.is_empty()
    }

    /// Poses the joints: `parents[j]` comes before `j` (checked by the skeleton), `bind_rotations`
    /// are the world rest rotations `Rb_j`, [`bind_positions`](Self::bind_positions) must already
    /// hold `bp_j`, `pose[j]` is the unit rotation relative to rest (missing entries: identity).
    /// Arrays shorter than the frames limit the joints posed. Allocation-free.
    pub fn pose(&mut self, parents: &[Option<u16>], bind_rotations: &[Quat], pose: &[Quat]) {
        let n = self.len().min(parents.len()).min(bind_rotations.len());
        for j in 0..n {
            let rb = bind_rotations[j];
            let q = pose.get(j).copied().unwrap_or(Quat::IDENTITY);
            let local = rb * q * rb.conjugate();
            let parent = parents[j].map(|p| p as usize).filter(|&p| p < j);
            let (d, wp) = match parent {
                Some(p) => {
                    let dp = self.skin_rotations[p];
                    let wp = self.world_positions[p] + dp * (self.bind_positions[j] - self.bind_positions[p]);
                    ((dp * local).normalize(), wp)
                }
                None => (local.normalize(), self.bind_positions[j]),
            };
            let t = wp - d * self.bind_positions[j];
            self.world_positions[j] = wp;
            self.world_rotations[j] = (d * rb).normalize();
            self.skin_rotations[j] = d;
            self.skin_matrices[j] = d.to_mat3();
            self.skin_translations[j] = t;
            self.skin_dual_quats[j] = DualQuat::from_rotation_translation(d, t);
        }
    }

    /// Linear blend skinning of point `p` with influences `joints` / `weights` (zero weights are
    /// skipped; the result is divided by the weight sum, `p` itself when the sum is not positive).
    #[must_use]
    pub fn skin_linear(&self, p: Vec3, joints: &[u16], weights: &[f32]) -> Vec3 {
        let mut sum = Vec3::ZERO;
        let mut total = 0.0_f32;
        for (&j, &w) in joints.iter().zip(weights) {
            if w == 0.0 {
                continue;
            }
            let (Some(m), Some(t)) = (self.skin_matrices.get(j as usize), self.skin_translations.get(j as usize))
            else {
                continue;
            };
            sum += (*m * p + *t) * w;
            total += w;
        }
        if total > 0.0 { sum * total.recip() } else { p }
    }

    /// Dual-quaternion skinning of point `p` (hemisphere-aligned to the first influence, normalised;
    /// `p` itself when no influence has weight).
    #[must_use]
    pub fn skin_dual_quat(&self, p: Vec3, joints: &[u16], weights: &[f32]) -> Vec3 {
        let mut pivot = None;
        let mut sum = DualQuat::ZERO;
        for (&j, &w) in joints.iter().zip(weights) {
            if w == 0.0 {
                continue;
            }
            let Some(&dq) = self.skin_dual_quats.get(j as usize) else {
                continue;
            };
            let pv = *pivot.get_or_insert(dq);
            sum = sum.accumulate(pv, dq, w);
        }
        if pivot.is_none() {
            return p;
        }
        sum.normalize().transform_point3(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain() -> (JointFrames, Vec<Option<u16>>, Vec<Quat>) {
        let mut f = JointFrames::new(3);
        f.bind_positions = vec![Vec3::ZERO, Vec3::Y, Vec3::Y * 2.0];
        (f, vec![None, Some(0), Some(1)], vec![Quat::IDENTITY; 3])
    }

    #[test]
    fn rest_pose_is_identity() {
        let (mut f, parents, rb) = chain();
        f.pose(&parents, &rb, &[]);
        for j in 0..3 {
            assert!(f.skin_rotations[j].abs_diff_eq(Quat::IDENTITY, 1e-6));
            assert!(f.skin_translations[j].length() < 1e-6);
            assert_eq!(f.world_positions[j], f.bind_positions[j]);
        }
        let p = Vec3::new(0.3, 1.5, 0.1);
        assert!(f.skin_linear(p, &[1, 2], &[0.5, 0.5]).distance(p) < 1e-6);
        assert!(f.skin_dual_quat(p, &[1, 2], &[0.5, 0.5]).distance(p) < 1e-6);
    }

    #[test]
    fn rotating_a_parent_carries_the_children() {
        let (mut f, parents, rb) = chain();
        let q = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        f.pose(&parents, &rb, &[Quat::IDENTITY, q]);
        // Joint 1 rotates about its own position (0,1,0): joint 2 swings from (0,2,0) to (-1,1,0).
        assert!(f.world_positions[2].distance(Vec3::new(-1.0, 1.0, 0.0)) < 1e-5);
        assert!(f.world_positions[1].distance(Vec3::Y) < 1e-6);
        let p = Vec3::new(0.0, 1.5, 0.0);
        assert!(f.skin_linear(p, &[1], &[1.0]).distance(Vec3::new(-0.5, 1.0, 0.0)) < 1e-5);
        assert!(f.skin_dual_quat(p, &[1], &[1.0]).distance(Vec3::new(-0.5, 1.0, 0.0)) < 1e-5);
        // A point bound to the root does not move.
        assert!(f.skin_linear(p, &[0], &[1.0]).distance(p) < 1e-6);
    }

    #[test]
    fn rotated_bind_frames_rotate_in_the_joint_frame() {
        // A joint whose rest frame is turned 90 degrees about Y: a local X rotation is a world -Z one
        // ... checked against the conjugation formula directly.
        let mut f = JointFrames::new(1);
        let rb = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let q = Quat::from_rotation_x(0.7);
        f.pose(&[None], &[rb], &[q]);
        let expect = rb * q * rb.conjugate();
        assert!(f.skin_rotations[0].abs_diff_eq(expect, 1e-6) || f.skin_rotations[0].abs_diff_eq(expect * -1.0, 1e-6));
        assert!(f.world_rotations[0].abs_diff_eq((expect * rb).normalize(), 1e-5));
    }

    #[test]
    fn dual_quaternions_keep_volume_where_linear_blending_collapses() {
        // Candy wrapper: a 180 degree twist between two joints at the same place. LBS collapses a point
        // halfway between them onto the axis; DQS keeps its distance.
        let mut f = JointFrames::new(2);
        f.bind_positions = vec![Vec3::ZERO, Vec3::ZERO];
        let twist = Quat::from_rotation_y(std::f32::consts::PI * 0.999);
        f.pose(&[None, None], &[Quat::IDENTITY; 2], &[Quat::IDENTITY, twist]);
        let p = Vec3::new(1.0, 0.0, 0.0);
        let lbs = f.skin_linear(p, &[0, 1], &[0.5, 0.5]);
        let dqs = f.skin_dual_quat(p, &[0, 1], &[0.5, 0.5]);
        assert!(lbs.length() < 0.01, "{lbs:?}");
        assert!((dqs.length() - 1.0).abs() < 1e-4, "{dqs:?}");
    }

    #[test]
    fn missing_influences_leave_the_point() {
        let f = JointFrames::new(1);
        let p = Vec3::ONE;
        assert_eq!(f.skin_linear(p, &[0], &[0.0]), p);
        assert_eq!(f.skin_dual_quat(p, &[5], &[1.0]), p);
        assert_eq!(f.skin_linear(p, &[5], &[1.0]), p);
        assert!(!f.is_empty());
    }
}
