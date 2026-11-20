//! Expression targets, combination correctives and pose-space correctives `[EXP-01]` `[EXP-02]`
//! `[SKIN-02]`.
//!
//! Each definition points at a delta component of its own kind in [`crate::delta::DeltaBases`]
//! (by index). The definitions say *when* a component is active; the rig logic that maps named
//! controls to expression weights `[EXP-03]` is separate data.

use humanart_math::Quat;

use crate::delta::{DeltaBases, DeltaKind};
use crate::error::{ModelError, Result};
use crate::names::{Side, check_names};

pub mod action;
pub mod build;
pub mod controls;
pub mod error;
pub mod field;
pub mod presets;
pub mod recipes;
pub mod sweep;

/// The most entries of each table (expressions, combinations, pose correctives).
pub const MAX_DEFINITIONS: usize = 4096;
/// The most inputs of one combination corrective.
pub const MAX_COMBINATION_INPUTS: usize = 4;

/// What an expression target is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ExpressionKind {
    /// A FACS action unit (Appendix B), possibly one side of it.
    #[default]
    ActionUnit,
    /// A viseme-support shape (lip closure, funnel, pucker, stretch, jaw-open variants).
    Viseme,
    /// A tongue shape.
    Tongue,
    /// Soft-tissue support around a skeleton-driven motion (jaw, eyes).
    Support,
}

impl ExpressionKind {
    /// The byte stored in packages.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::ActionUnit => 0,
            Self::Viseme => 1,
            Self::Tongue => 2,
            Self::Support => 3,
        }
    }

    /// The kind for a stored byte, if valid.
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::ActionUnit),
            1 => Some(Self::Viseme),
            2 => Some(Self::Tongue),
            3 => Some(Self::Support),
            _ => None,
        }
    }
}

/// One expression target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expression {
    /// Stable name (`AU12-L`, `viseme.pucker`).
    pub name: String,
    /// What it is.
    pub kind: ExpressionKind,
    /// The side it acts on.
    pub side: Side,
    /// The FACS action unit number, 0 when not an action unit.
    pub facs: u16,
    /// Its delta component (kind [`DeltaKind::Expression`]).
    pub component: u32,
}

/// How a combination corrective's activation follows its inputs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CombineOp {
    /// The product of the input weights.
    #[default]
    Product,
    /// The smallest input weight.
    Min,
}

impl CombineOp {
    /// The byte stored in packages.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::Product => 0,
            Self::Min => 1,
        }
    }

    /// The operation for a stored byte, if valid.
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Product),
            1 => Some(Self::Min),
            _ => None,
        }
    }
}

/// A corrective shape for a combination of expressions `[EXP-02]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CombinationCorrective {
    /// Stable name.
    pub name: String,
    /// How the inputs combine.
    pub op: CombineOp,
    /// Expression indices, 2 to [`MAX_COMBINATION_INPUTS`], distinct.
    pub inputs: Vec<u32>,
    /// Its delta component (kind [`DeltaKind::CombinationCorrective`]).
    pub component: u32,
}

impl CombinationCorrective {
    /// The activation for expression weights `weights` (missing weights count as 0); inputs are
    /// clamped to `[0, 1]` first.
    #[must_use]
    pub fn activation(&self, weights: &[f32]) -> f32 {
        let w = |i: &u32| weights.get(*i as usize).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        match self.op {
            CombineOp::Product => self.inputs.iter().map(w).product(),
            CombineOp::Min => self.inputs.iter().map(w).fold(1.0, f32::min),
        }
    }
}

/// A pose-space corrective `[SKIN-02]`: a delta that fades in as a joint's rotation (relative to
/// its rest orientation) approaches a target rotation, with a compactly supported kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct PoseCorrective {
    /// Stable name.
    pub name: String,
    /// The driving joint.
    pub joint: u16,
    /// The rotation (relative to rest) at which the corrective is fully on; a unit quaternion.
    pub target: Quat,
    /// The angle in radians at which the activation reaches 0, in `(0, π]`.
    pub radius: f32,
    /// Its delta component (kind [`DeltaKind::PoseCorrective`]).
    pub component: u32,
}

impl PoseCorrective {
    /// The activation for the joint's rotation relative to rest: `(1 - (a / radius)^2)^2` for an
    /// angle `a` to the target below `radius`, else 0 (smooth, compact, 1 at the target).
    #[must_use]
    pub fn activation(&self, rotation: Quat) -> f32 {
        let a = rotation.angle_between(self.target);
        if a.is_nan() || a >= self.radius {
            return 0.0;
        }
        let t = a / self.radius;
        let s = 1.0 - t * t;
        s * s
    }
}

/// The expression, combination and pose-corrective definitions of a template.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExpressionSet {
    /// Expression targets.
    pub expressions: Vec<Expression>,
    /// Combination correctives.
    pub combinations: Vec<CombinationCorrective>,
    /// Pose-space correctives.
    pub pose_correctives: Vec<PoseCorrective>,
}

impl ExpressionSet {
    /// Checks the definitions against the delta bases and the joint count.
    pub fn check(&self, deltas: &DeltaBases, joint_count: usize) -> Result<()> {
        for (what, n) in [
            ("expression count", self.expressions.len()),
            ("combination corrective count", self.combinations.len()),
            ("pose corrective count", self.pose_correctives.len()),
        ] {
            if n > MAX_DEFINITIONS {
                return Err(ModelError::limit(what, MAX_DEFINITIONS as u64, n as u64));
            }
        }
        check_names("expression", self.expressions.iter().map(|e| e.name.as_str()))?;
        check_names("combination corrective", self.combinations.iter().map(|e| e.name.as_str()))?;
        check_names("pose corrective", self.pose_correctives.iter().map(|e| e.name.as_str()))?;
        let kind_ok = |c: u32, kind: DeltaKind| deltas.components().get(c as usize).is_some_and(|x| x.kind == kind);
        for (i, e) in self.expressions.iter().enumerate() {
            if !kind_ok(e.component, DeltaKind::Expression) {
                return Err(ModelError::invalid("expression", i, "component missing or not an expression delta"));
            }
        }
        for (i, c) in self.combinations.iter().enumerate() {
            if !kind_ok(c.component, DeltaKind::CombinationCorrective) {
                return Err(ModelError::invalid("combination corrective", i, "component missing or of another kind"));
            }
            if !(2..=MAX_COMBINATION_INPUTS).contains(&c.inputs.len()) {
                return Err(ModelError::invalid("combination corrective", i, "needs 2 to 4 inputs"));
            }
            if c.inputs.iter().any(|&x| x as usize >= self.expressions.len()) {
                return Err(ModelError::invalid("combination corrective", i, "input expression out of range"));
            }
            let mut sorted = c.inputs.clone();
            sorted.sort_unstable();
            if sorted.windows(2).any(|w| w[0] == w[1]) {
                return Err(ModelError::invalid("combination corrective", i, "input used twice"));
            }
        }
        for (i, p) in self.pose_correctives.iter().enumerate() {
            if !kind_ok(p.component, DeltaKind::PoseCorrective) {
                return Err(ModelError::invalid("pose corrective", i, "component missing or of another kind"));
            }
            if p.joint as usize >= joint_count {
                return Err(ModelError::invalid("pose corrective", i, "joint index out of range"));
            }
            let q = p.target;
            if ![q.x, q.y, q.z, q.w].iter().all(|c| c.is_finite()) || (q.length() - 1.0).abs() > 1e-3 {
                return Err(ModelError::invalid("pose corrective", i, "target is not a unit quaternion"));
            }
            if !(p.radius > 0.0 && p.radius <= std::f32::consts::PI) {
                return Err(ModelError::invalid("pose corrective", i, "radius outside (0, pi]"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod integration;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta::DeltaBasesBuilder;
    use humanart_math::Vec3;

    fn bases() -> DeltaBases {
        let mut b = DeltaBasesBuilder::new();
        b.add("AU12-L", DeltaKind::Expression, 0, &[(0, Vec3::X)]).unwrap();
        b.add("AU6-L", DeltaKind::Expression, 0, &[(1, Vec3::X)]).unwrap();
        b.add("AU12-L.AU6-L", DeltaKind::CombinationCorrective, 0, &[(0, Vec3::Y)]).unwrap();
        b.add("elbow-L.bend", DeltaKind::PoseCorrective, 0, &[(1, Vec3::Z)]).unwrap();
        b.finish().unwrap()
    }

    fn set() -> ExpressionSet {
        let e = |name: &str, c: u32| Expression {
            name: name.into(),
            kind: ExpressionKind::ActionUnit,
            side: Side::Left,
            facs: 12,
            component: c,
        };
        ExpressionSet {
            expressions: vec![e("AU12-L", 0), e("AU6-L", 1)],
            combinations: vec![CombinationCorrective {
                name: "AU12-L.AU6-L".into(),
                op: CombineOp::Product,
                inputs: vec![0, 1],
                component: 2,
            }],
            pose_correctives: vec![PoseCorrective {
                name: "elbow-L.bend".into(),
                joint: 1,
                target: Quat::from_axis_angle(Vec3::X, 1.5),
                radius: 1.5,
                component: 3,
            }],
        }
    }

    #[test]
    fn activations() {
        let s = set();
        let c = &s.combinations[0];
        assert!((c.activation(&[0.5, 0.8]) - 0.4).abs() < 1e-6);
        assert_eq!(c.activation(&[2.0]), 0.0);
        let m = CombinationCorrective { op: CombineOp::Min, ..c.clone() };
        assert_eq!(m.activation(&[0.5, 0.8]), 0.5);
        let p = &s.pose_correctives[0];
        assert!((p.activation(p.target) - 1.0).abs() < 1e-3);
        assert_eq!(p.activation(Quat::IDENTITY), 0.0);
        let half = p.activation(Quat::from_axis_angle(Vec3::X, 0.75));
        assert!((half - 0.5625).abs() < 1e-3, "{half}");
        for k in [ExpressionKind::ActionUnit, ExpressionKind::Viseme, ExpressionKind::Tongue, ExpressionKind::Support] {
            assert_eq!(ExpressionKind::from_u8(k.to_u8()), Some(k));
        }
        assert_eq!(ExpressionKind::from_u8(4), None);
        assert_eq!(CombineOp::from_u8(CombineOp::Min.to_u8()), Some(CombineOp::Min));
        assert_eq!(CombineOp::from_u8(2), None);
    }

    #[test]
    fn checks_refuse_broken_definitions() {
        type Change = fn(&mut ExpressionSet);
        let d = bases();
        set().check(&d, 2).unwrap();
        assert!(set().check(&d, 1).is_err());
        let cases: [(&str, Change); 9] = [
            ("expr kind", |s| s.expressions[0].component = 2),
            ("expr missing", |s| s.expressions[0].component = 9),
            ("combo kind", |s| s.combinations[0].component = 0),
            ("combo inputs", |s| s.combinations[0].inputs = vec![0]),
            ("combo range", |s| s.combinations[0].inputs = vec![0, 5]),
            ("combo twice", |s| s.combinations[0].inputs = vec![1, 1]),
            ("pose target", |s| s.pose_correctives[0].target = Quat::from_xyzw(0.0, 0.0, 0.0, 2.0)),
            ("pose radius", |s| s.pose_correctives[0].radius = 0.0),
            ("names", |s| s.expressions[1].name = "AU12-L".into()),
        ];
        for (what, change) in cases {
            let mut s = set();
            change(&mut s);
            assert!(s.check(&d, 2).is_err(), "{what} was accepted");
        }
    }
}
