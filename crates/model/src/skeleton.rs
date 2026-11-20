//! The skeleton `[MOD-07]`: joint hierarchy, rest transforms, joint limits, and the joint regressor
//! that moves joints with the shape.
//!
//! Joints are stored parents-first (a parent has a lower index than its children), so one forward
//! pass computes world transforms. Rest transforms are local (relative to the parent's rest frame).
//!
//! The skeleton and the surface are **decoupled**: joint positions do not come from a fixed sum of
//! vertex positions but from a [`JointRegressor`], a sparse linear map from delta-component weights
//! to world-space joint position offsets (fitted from data, or authored with the identity axes).

use humanart_math::{Quat, Transform, Vec3};

use crate::error::{ModelError, Result};
use crate::names::check_names;

/// The most joints a skeleton may have (joint indices are `u16`).
pub const MAX_JOINTS: usize = 1024;

/// What a joint is for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum JointKind {
    /// An anatomical joint that deforms the skin (spine, limbs, fingers, jaw, tongue, eyes).
    #[default]
    Deform,
    /// A twist joint that distributes rotation along a limb segment.
    Twist,
    /// A helper joint driven by other joints or by correctives (eyelid helpers, face helpers).
    Helper,
    /// A non-deforming attachment point.
    Socket,
}

impl JointKind {
    /// The byte stored in packages.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::Deform => 0,
            Self::Twist => 1,
            Self::Helper => 2,
            Self::Socket => 3,
        }
    }

    /// The kind for a stored byte, if valid.
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Deform),
            1 => Some(Self::Twist),
            2 => Some(Self::Helper),
            3 => Some(Self::Socket),
            _ => None,
        }
    }
}

/// Rotation limits of a joint relative to its rest orientation, as Euler angles in radians in the
/// Y-X-Z order of `Quat::from_euler_yxz`. `min <= max` per axis, both within `[-2π, 2π]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointLimits {
    /// Lower limits (x, y, z).
    pub min: Vec3,
    /// Upper limits (x, y, z).
    pub max: Vec3,
}

impl JointLimits {
    /// No rotation allowed.
    pub const LOCKED: Self = Self { min: Vec3::ZERO, max: Vec3::ZERO };
    /// Any rotation allowed.
    pub const FREE: Self = Self { min: Vec3::splat(-std::f32::consts::PI), max: Vec3::splat(std::f32::consts::PI) };

    /// Clamps Euler angles into the limits.
    #[must_use]
    pub fn clamp(&self, angles: Vec3) -> Vec3 {
        angles.clamp(self.min, self.max)
    }
}

/// One joint.
#[derive(Clone, Debug, PartialEq)]
pub struct Joint {
    /// Stable name (`spine.03`, `L.index.02`, `jaw`).
    pub name: String,
    /// The parent joint; it has a lower index.
    pub parent: Option<u16>,
    /// What the joint is for.
    pub kind: JointKind,
    /// Rest transform relative to the parent's rest frame (unit rotation, positive scale).
    pub rest: Transform,
    /// Rotation limits.
    pub limits: JointLimits,
}

/// The joint regressor: for each delta component `c`, the joints it moves and by how much
/// (world-space position offset per unit weight), in CSR layout: component `c` has the entries
/// `offsets[c]..offsets[c + 1]` of `joints` and `deltas`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct JointRegressor {
    /// `component_count + 1` offsets, from 0, non-decreasing.
    pub offsets: Vec<u32>,
    /// Joint index of each entry; strictly increasing within a component.
    pub joints: Vec<u16>,
    /// Position offset of each entry, metres per unit weight.
    pub deltas: Vec<Vec3>,
}

impl JointRegressor {
    /// A regressor that moves no joint, for `components` delta components.
    #[must_use]
    pub fn empty(components: usize) -> Self {
        Self { offsets: vec![0; components + 1], joints: Vec::new(), deltas: Vec::new() }
    }

    /// The entries of component `c` (empty when out of range).
    #[must_use]
    pub fn component(&self, c: usize) -> (&[u16], &[Vec3]) {
        match (self.offsets.get(c), self.offsets.get(c + 1)) {
            (Some(&a), Some(&b)) if a <= b && b as usize <= self.joints.len().min(self.deltas.len()) => {
                (&self.joints[a as usize..b as usize], &self.deltas[a as usize..b as usize])
            }
            _ => (&[], &[]),
        }
    }
}

/// The skeleton of a template.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Skeleton {
    /// The joints, parents first.
    pub joints: Vec<Joint>,
    /// The shape-to-joint regressor.
    pub regressor: JointRegressor,
}

impl Skeleton {
    /// Checks the skeleton; `components` is the number of delta components of the template.
    pub fn check(&self, components: usize) -> Result<()> {
        let n = self.joints.len();
        if n > MAX_JOINTS {
            return Err(ModelError::limit("joint count", MAX_JOINTS as u64, n as u64));
        }
        check_names("joint", self.joints.iter().map(|j| j.name.as_str()))?;
        let two_pi = std::f32::consts::TAU;
        for (i, j) in self.joints.iter().enumerate() {
            if j.parent.is_some_and(|p| p as usize >= i) {
                return Err(ModelError::invalid("joint", i, "parent does not come before the joint"));
            }
            let r = j.rest;
            let q = [r.rotation.x, r.rotation.y, r.rotation.z, r.rotation.w];
            if !r.translation.is_finite() || !q.iter().all(|c| c.is_finite()) || !r.scale.is_finite() {
                return Err(ModelError::invalid("joint", i, "rest transform is not finite"));
            }
            if (r.rotation.length() - 1.0).abs() > 1e-3 {
                return Err(ModelError::invalid("joint", i, "rest rotation is not a unit quaternion"));
            }
            if r.scale.min_element() <= 0.0 {
                return Err(ModelError::invalid("joint", i, "rest scale is not positive"));
            }
            let (lo, hi) = (j.limits.min, j.limits.max);
            let in_range = |v: Vec3| v.is_finite() && v.abs().max_element() <= two_pi;
            if !in_range(lo) || !in_range(hi) || lo.x > hi.x || lo.y > hi.y || lo.z > hi.z {
                return Err(ModelError::invalid("joint", i, "limits are not finite, ordered and within 2 pi"));
            }
        }
        let reg = &self.regressor;
        if reg.offsets.len() != components + 1 || reg.offsets.first() != Some(&0) {
            return Err(ModelError::invalid(
                "joint regressor",
                0,
                "offsets do not have component_count + 1 entries from 0",
            ));
        }
        if reg.joints.len() != reg.deltas.len() || reg.offsets.last().copied() != Some(reg.joints.len() as u32) {
            return Err(ModelError::invalid("joint regressor", 0, "entry arrays differ from the offsets"));
        }
        for (c, w) in reg.offsets.windows(2).enumerate() {
            let Some(joints) = reg.joints.get(w[0] as usize..w[1] as usize) else {
                return Err(ModelError::invalid("joint regressor component", c, "offsets decrease or overrun"));
            };
            if joints.windows(2).any(|p| p[0] >= p[1]) {
                return Err(ModelError::invalid("joint regressor component", c, "joints not strictly increasing"));
            }
            if joints.last().is_some_and(|&j| j as usize >= n) {
                return Err(ModelError::invalid("joint regressor component", c, "joint index out of range"));
            }
        }
        if let Some(i) = reg.deltas.iter().position(|d| !d.is_finite()) {
            return Err(ModelError::invalid("joint regressor entry", i, "offset is not finite"));
        }
        Ok(())
    }

    /// The index of the joint called `name`.
    #[must_use]
    pub fn find(&self, name: &str) -> Option<u16> {
        self.joints.iter().position(|j| j.name == name).map(|i| i as u16)
    }

    /// The world rest transform of every joint (parent world * local rest).
    #[must_use]
    pub fn rest_world(&self) -> Vec<Transform> {
        let mut out: Vec<Transform> = Vec::with_capacity(self.joints.len());
        for j in &self.joints {
            let parent = j.parent.and_then(|p| out.get(p as usize).copied());
            out.push(parent.map_or(j.rest, |p| p * j.rest));
        }
        out
    }

    /// World joint positions for delta-component weights `weights` (missing weights count as 0):
    /// the rest positions plus the regressor's offsets.
    #[must_use]
    pub fn regressed_positions(&self, weights: &[f32]) -> Vec<Vec3> {
        let mut out: Vec<Vec3> = self.rest_world().iter().map(|t| t.translation).collect();
        for (c, &w) in weights.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            let (joints, deltas) = self.regressor.component(c);
            for (&j, &d) in joints.iter().zip(deltas) {
                if let Some(p) = out.get_mut(j as usize) {
                    *p += d * w;
                }
            }
        }
        out
    }
}

/// A joint with a rest translation and identity rotation, free limits: a convenience for tests and
/// generators.
#[must_use]
pub fn simple_joint(name: &str, parent: Option<u16>, translation: Vec3) -> Joint {
    Joint {
        name: name.to_owned(),
        parent,
        kind: JointKind::Deform,
        rest: Transform::from_rotation_translation(Quat::IDENTITY, translation),
        limits: JointLimits::FREE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain() -> Skeleton {
        Skeleton {
            joints: vec![simple_joint("root", None, Vec3::new(0.0, 1.0, 0.0)), simple_joint("tip", Some(0), Vec3::Y)],
            regressor: JointRegressor {
                offsets: vec![0, 1, 1],
                joints: vec![1],
                deltas: vec![Vec3::new(0.0, 0.5, 0.0)],
            },
        }
    }

    #[test]
    fn world_rest_and_regression() {
        let s = chain();
        s.check(2).unwrap();
        let w = s.rest_world();
        assert_eq!(w[1].translation, Vec3::new(0.0, 2.0, 0.0));
        assert_eq!(s.find("tip"), Some(1));
        let p = s.regressed_positions(&[2.0, 5.0]);
        assert_eq!(p, vec![Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 3.0, 0.0)]);
        assert_eq!(s.regressor.component(1).0.len(), 0);
        assert_eq!(s.regressor.component(9).0.len(), 0);
        assert_eq!(JointRegressor::empty(3).offsets, vec![0; 4]);
        assert_eq!(JointLimits::LOCKED.clamp(Vec3::ONE), Vec3::ZERO);
        for k in [JointKind::Deform, JointKind::Twist, JointKind::Helper, JointKind::Socket] {
            assert_eq!(JointKind::from_u8(k.to_u8()), Some(k));
        }
        assert_eq!(JointKind::from_u8(4), None);
    }

    #[test]
    fn checks_refuse_broken_skeletons() {
        type Change = fn(&mut Skeleton);
        let cases: [(&str, Change); 11] = [
            ("parent order", |s| s.joints[0].parent = Some(1)),
            ("name", |s| s.joints[1].name = "root".into()),
            ("nan", |s| s.joints[0].rest.translation.x = f32::NAN),
            ("rotation", |s| s.joints[0].rest.rotation.w = 2.0),
            ("scale", |s| s.joints[0].rest.scale.y = 0.0),
            ("limits order", |s| s.joints[0].limits.min.x = 4.0),
            ("limits range", |s| s.joints[0].limits.max.x = 7.0),
            ("reg offsets", |s| s.regressor.offsets.push(1)),
            ("reg joint", |s| s.regressor.joints[0] = 2),
            ("reg delta", |s| s.regressor.deltas[0].z = f32::INFINITY),
            ("reg arrays", |s| s.regressor.deltas.clear()),
        ];
        for (what, change) in cases {
            let mut s = chain();
            change(&mut s);
            assert!(s.check(2).is_err(), "{what} was accepted");
        }
        assert!(chain().check(3).is_err());
    }
}
