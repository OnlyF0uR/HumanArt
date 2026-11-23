//! Twist distribution (todo 1.3): the roll of a limb spread along it by its twist joints, so a
//! turned forearm or upper arm twists its skin gradually instead of wringing it at one joint.
//!
//! The template's twist joints (`JointKind::Twist`) are driven, not posed: before skinning,
//! [`TwistRig::apply`] overwrites their rotations from the pose of the joints they follow
//! (what OpenRigLogic's twist/swing evaluator does, `src/riglogic/joints/cpu/twistswing`, MIT,
//! Epic Games):
//!
//! - **from the child** (`<side>.forearm.twist`): the twist about the bone of the hand's rotation
//!   (pronation and supination happen along the forearm, between the radius and the ulna, not at
//!   the wrist); the forearm's skin is weighted from the forearm at the elbow to the twist joint at
//!   the wrist, so the roll grows along the forearm;
//! - **against the parent** (`<side>.upper-arm.twist`): the inverse of the upper arm's own twist
//!   (the humerus rolls in the shoulder; the skin over the deltoid stays with the shoulder); the
//!   upper arm's skin is weighted to the twist joint at the shoulder and to the upper arm toward
//!   the elbow.
//!
//! The twist is the rotation's component about the joint frame's Y axis (the bone,
//! `generate::rig`), split off as in [`crate::expression::pose_space::swing_twist_y`]. Twist joints
//! share their parent's rest orientation, so a twist about the parent's Y is one about theirs.
//! A twist joint whose name fits neither rule, or whose partner is missing, is left as posed.

use humanart_math::Quat;

use crate::expression::pose_space::swing_twist_y;
use crate::skeleton::{JointKind, Skeleton};

/// How a twist joint is driven.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwistSource {
    /// The twist of this joint's rotation (a child of the twist joint's parent).
    Child(u16),
    /// The inverse twist of the twist joint's parent's rotation.
    AgainstParent(u16),
}

/// One driven twist joint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwistSetup {
    /// The twist joint.
    pub joint: u16,
    /// What drives it.
    pub source: TwistSource,
    /// The share of the source's twist it takes, 0..1.
    pub fraction: f32,
}

/// The driven twist joints of a skeleton.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TwistRig {
    setups: Vec<TwistSetup>,
}

impl TwistRig {
    /// The setups of `skeleton`'s twist joints by the naming rules of the module documentation.
    #[must_use]
    pub fn new(skeleton: &Skeleton) -> Self {
        let mut setups = Vec::new();
        for (i, joint) in skeleton.joints.iter().enumerate() {
            if joint.kind != JointKind::Twist {
                continue;
            }
            let Some(parent) = joint.parent else { continue };
            let side = joint.name.split('.').next().unwrap_or("");
            let source = if joint.name.ends_with(".forearm.twist") {
                skeleton.find(&format!("{side}.hand")).map(TwistSource::Child)
            } else if joint.name.ends_with(".upper-arm.twist") {
                Some(TwistSource::AgainstParent(parent))
            } else {
                None
            };
            if let Some(source) = source {
                setups.push(TwistSetup { joint: i as u16, source, fraction: 1.0 });
            }
        }
        Self { setups }
    }

    /// The setups.
    #[must_use]
    pub fn setups(&self) -> &[TwistSetup] {
        &self.setups
    }

    /// Whether no twist joint is driven.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.setups.is_empty()
    }

    /// Overwrites the driven twist joints' rotations in `pose` (relative to rest, one per joint;
    /// missing entries are skipped). No allocation.
    pub fn apply(&self, pose: &mut [Quat]) {
        for s in &self.setups {
            let source = match s.source {
                TwistSource::Child(j) | TwistSource::AgainstParent(j) => pose.get(j as usize).copied(),
            };
            let Some(q) = source else { continue };
            let (_, twist) = swing_twist_y(q.try_normalize().unwrap_or(Quat::IDENTITY));
            let twist = match s.source {
                TwistSource::Child(_) => twist,
                TwistSource::AgainstParent(_) => twist.conjugate(),
            };
            if let Some(slot) = pose.get_mut(s.joint as usize) {
                *slot = Quat::IDENTITY.slerp(twist, s.fraction.clamp(0.0, 1.0));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use humanart_math::Vec3;

    use super::*;
    use crate::generate::anatomy::Anatomy;
    use crate::generate::rig::skeleton;

    #[test]
    fn todo_1_3_twist_joints_take_the_roll_of_the_hand_and_undo_the_upper_arm_roll() {
        let rig = skeleton(&Anatomy::neutral(), 0).unwrap();
        let s = &rig.skeleton;
        let twist = TwistRig::new(s);
        assert_eq!(twist.setups().len(), 4, "{:?}", twist.setups());
        let find = |n: &str| s.find(n).unwrap() as usize;
        let mut pose = vec![Quat::IDENTITY; s.joints.len()];
        // The hand rolled 80 degrees and bent 30: the forearm twist takes the roll only.
        pose[find("L.hand")] = Quat::from_rotation_x(0.5) * Quat::from_rotation_y(1.4);
        // The upper arm raised and rolled: its twist joint undoes the roll.
        pose[find("L.upper-arm")] = Quat::from_rotation_z(0.6) * Quat::from_rotation_y(-0.7);
        pose[find("L.forearm.twist")] = Quat::from_rotation_x(1.0); // overwritten
        twist.apply(&mut pose);
        let fore = pose[find("L.forearm.twist")];
        assert!(fore.angle_between(Quat::from_rotation_y(1.4)) < 1e-4, "{fore:?}");
        let upper = pose[find("L.upper-arm.twist")];
        assert!(upper.angle_between(Quat::from_rotation_y(0.7)) < 1e-4, "{upper:?}");
        // Untouched: the right side at rest.
        assert!(pose[find("R.forearm.twist")].angle_between(Quat::IDENTITY) < 1e-6);
        // The twist axis is the bone: a pure bend gives no twist.
        pose.fill(Quat::IDENTITY);
        pose[find("L.hand")] = Quat::from_axis_angle(Vec3::new(1.0, 0.0, 1.0).normalize(), 0.8);
        twist.apply(&mut pose);
        assert!(pose[find("L.forearm.twist")].angle_between(Quat::IDENTITY) < 1e-4);
    }
}
