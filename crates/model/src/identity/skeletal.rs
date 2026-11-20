//! Skeleton-driven identity deltas `[MOD-07]` `[ID-03]`: bone lengths and group scales that move
//! the joints and the skin together.
//!
//! Every joint has a **bone**: from its rest position (the anchor) to the end named by
//! [`bone_end`] (its main child joint, or a landmark at a tip). A [`Stretch`] gives every joint a
//! linear map `M_j` (per unit weight): lengthening along the bone axis, a shear, or a uniform scale
//! of a whole subtree (head, hands). Joint offsets follow the hierarchy parents first,
//! `o_child = o_parent + M_parent (J_child - J_parent)`, and a surface point `p` moves by the
//! skin-weighted blend `sum_k w_k (o_k + M_k (p - J_k))` of its influences: a linear-blend form of
//! the same map, so the surface and the skeleton stay together and the result is as smooth as the
//! skin weights. Those joint offsets are the component's joint-regressor row.
//!
//! The joint names used here are the stable names of the generated rig (`spine.01`, `L.thigh`,
//! `L.index.02`, ...); the tip landmarks are `head.vertex`, `chin.menton`, `L.hand.<finger>.tip`,
//! `L.foot.tip`.

use humanart_math::Vec3;

use super::error::{IdResult, IdentityError};
use super::space::Space;

/// A linear map of 3-vectors, stored as its three columns.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Lin {
    /// The image of +X.
    pub x: Vec3,
    /// The image of +Y.
    pub y: Vec3,
    /// The image of +Z.
    pub z: Vec3,
}

impl Lin {
    /// The zero map.
    pub const ZERO: Self = Self { x: Vec3::ZERO, y: Vec3::ZERO, z: Vec3::ZERO };

    /// Applies the map.
    #[must_use]
    pub fn apply(&self, v: Vec3) -> Vec3 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    /// `s` times the identity.
    #[must_use]
    pub fn scale(s: f32) -> Self {
        Self { x: Vec3::X * s, y: Vec3::Y * s, z: Vec3::Z * s }
    }

    /// `k a bᵀ`: maps `v` to `k a (b · v)`.
    #[must_use]
    pub fn outer(a: Vec3, b: Vec3, k: f32) -> Self {
        Self { x: a * (k * b.x), y: a * (k * b.y), z: a * (k * b.z) }
    }

    /// The sum of two maps.
    #[must_use]
    pub fn plus(self, o: Self) -> Self {
        Self { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z }
    }
}

/// Where a bone ends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum End {
    /// At another joint.
    Joint(String),
    /// At a landmark (a tip).
    Landmark(String),
}

/// The end of the bone of the joint called `name` (see the module documentation), `None` for
/// joints without a bone direction (eyes, lid helpers, the last tongue joint).
#[must_use]
pub fn bone_end(name: &str) -> Option<End> {
    let joint = |s: &str| Some(End::Joint(s.to_owned()));
    match name {
        "pelvis" => return joint("spine.01"),
        "spine.01" => return joint("spine.02"),
        "spine.02" => return joint("spine.03"),
        "spine.03" => return joint("spine.04"),
        "spine.04" => return joint("neck.01"),
        "neck.01" => return joint("neck.02"),
        "neck.02" => return joint("head"),
        "head" => return Some(End::Landmark("head.vertex".to_owned())),
        "jaw" => return Some(End::Landmark("chin.menton".to_owned())),
        "tongue.01" => return joint("tongue.02"),
        "tongue.02" => return joint("tongue.03"),
        "tongue.03" => return joint("tongue.04"),
        _ => {}
    }
    let (side, rest) = name.split_at_checked(2)?;
    if side != "L." && side != "R." {
        return None;
    }
    let sided = |s: &str| Some(End::Joint(format!("{side}{s}")));
    match rest {
        "clavicle" => return sided("upper-arm"),
        "upper-arm" | "upper-arm.twist" => return sided("forearm"),
        "forearm" | "forearm.twist" => return sided("hand"),
        "hand" => return sided("middle.01"),
        "thigh" => return sided("shin"),
        "shin" => return sided("foot"),
        "foot" => return sided("toes"),
        "toes" => return Some(End::Landmark(format!("{side}foot.tip"))),
        _ => {}
    }
    for finger in ["thumb", "index", "middle", "ring", "little"] {
        match rest.strip_prefix(finger).and_then(|r| r.strip_prefix('.')) {
            Some("01") => return sided(&format!("{finger}.02")),
            Some("02") => return sided(&format!("{finger}.03")),
            Some("03") => return Some(End::Landmark(format!("{side}hand.{finger}.tip"))),
            _ => {}
        }
    }
    None
}

/// The bones of a template's skeleton (see the module documentation).
#[derive(Clone, Debug)]
pub struct Bones {
    /// Rest world position of each joint.
    pub anchor: Vec<Vec3>,
    /// Unit bone direction of each joint (zero without a bone).
    pub axis: Vec<Vec3>,
    /// Bone length of each joint (0 without a bone).
    pub length: Vec<f32>,
    /// Parent of each joint.
    pub parent: Vec<Option<u16>>,
}

impl Bones {
    /// Reads the bones of the template of `space`. A named end that the template lacks leaves the
    /// joint without a bone; a joint whose end is missing is not an error (the rig may grow).
    pub fn new(space: &Space<'_>) -> IdResult<Self> {
        let skeleton = space.template().skeleton();
        let anchor: Vec<Vec3> = skeleton.rest_world().iter().map(|t| t.translation).collect();
        let mut axis = vec![Vec3::ZERO; anchor.len()];
        let mut length = vec![0.0; anchor.len()];
        for (j, joint) in skeleton.joints.iter().enumerate() {
            let end = match bone_end(&joint.name) {
                Some(End::Joint(n)) => skeleton.find(&n).and_then(|e| anchor.get(e as usize).copied()),
                Some(End::Landmark(n)) => space.landmark(&n).ok(),
                None => None,
            };
            if let (Some(e), Some(&a)) = (end, anchor.get(j)) {
                let d = e - a;
                if d.length() > 1.0e-5 {
                    axis[j] = d.normalize();
                    length[j] = d.length();
                }
            }
        }
        let parent = skeleton.joints.iter().map(|j| j.parent).collect();
        Ok(Self { anchor, axis, length, parent })
    }

    /// How many joints there are.
    #[must_use]
    pub fn len(&self) -> usize {
        self.anchor.len()
    }

    /// Whether there are no joints.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.anchor.is_empty()
    }

    /// Whether joint `j` is `ancestor` or lies below it.
    #[must_use]
    pub fn is_within(&self, j: u16, ancestor: u16) -> bool {
        let mut cur = Some(j);
        let mut steps = 0;
        while let Some(c) = cur {
            if c == ancestor {
                return true;
            }
            steps += 1;
            if steps > self.len() {
                return false;
            }
            cur = self.parent.get(c as usize).copied().flatten();
        }
        false
    }
}

/// Per-joint linear maps and extra offsets of one skeletal component (per unit weight).
#[derive(Clone, Debug)]
pub struct Stretch {
    /// The map of each joint.
    pub lin: Vec<Lin>,
    /// An extra offset of each joint, added to what the hierarchy gives (passed on to children).
    pub extra: Vec<Vec3>,
}

impl Stretch {
    /// No change for `joints` joints.
    #[must_use]
    pub fn new(joints: usize) -> Self {
        Self { lin: vec![Lin::ZERO; joints], extra: vec![Vec3::ZERO; joints] }
    }

    fn add(&mut self, j: u16, m: Lin) -> IdResult<()> {
        let slot =
            self.lin.get_mut(j as usize).ok_or(IdentityError::invalid("joint", j as usize, "index out of range"))?;
        *slot = slot.plus(m);
        Ok(())
    }

    /// Lengthens the bone of joint `j` by the fraction `k` along its axis.
    pub fn lengthen(&mut self, bones: &Bones, j: u16, k: f32) -> IdResult<()> {
        let a = bones.axis.get(j as usize).copied().unwrap_or(Vec3::ZERO);
        self.add(j, Lin::outer(a, a, k))
    }

    /// Lengthens the bone of joint `j` by the fraction `k` along the horizontal part of its axis
    /// (feet: the sole stays on the ground).
    pub fn lengthen_horizontal(&mut self, bones: &Bones, j: u16, k: f32) -> IdResult<()> {
        let a = bones.axis.get(j as usize).copied().unwrap_or(Vec3::ZERO);
        let h = Vec3::new(a.x, 0.0, a.z).normalize_or_zero();
        self.add(j, Lin::outer(h, h, k))
    }

    /// Adds `k a bᵀ` to joint `j`'s map, with `b` the bone axis: points along the bone move by `k`
    /// times their distance along it, in direction `a` (a shear when `a` is not the axis).
    pub fn shear(&mut self, bones: &Bones, j: u16, a: Vec3, k: f32) -> IdResult<()> {
        let b = bones.axis.get(j as usize).copied().unwrap_or(Vec3::ZERO);
        self.add(j, Lin::outer(a, b, k))
    }

    /// Scales joint `j` and its whole subtree uniformly by the fraction `s` about `j`.
    pub fn scale_subtree(&mut self, bones: &Bones, j: u16, s: f32) -> IdResult<()> {
        for k in 0..bones.len() as u16 {
            if bones.is_within(k, j) {
                self.add(k, Lin::scale(s))?;
            }
        }
        Ok(())
    }

    /// Adds an extra offset to joint `j` (its subtree follows).
    pub fn push(&mut self, j: u16, d: Vec3) -> IdResult<()> {
        let slot =
            self.extra.get_mut(j as usize).ok_or(IdentityError::invalid("joint", j as usize, "index out of range"))?;
        *slot += d;
        Ok(())
    }

    /// The joint offsets, parents first (joints are stored parents first).
    #[must_use]
    pub fn offsets(&self, bones: &Bones) -> Vec<Vec3> {
        let mut out = vec![Vec3::ZERO; bones.len()];
        for j in 0..bones.len() {
            let extra = self.extra.get(j).copied().unwrap_or(Vec3::ZERO);
            let from_parent = match bones.parent.get(j).copied().flatten() {
                Some(p) if (p as usize) < j => {
                    let lin = self.lin.get(p as usize).copied().unwrap_or(Lin::ZERO);
                    let (ap, aj) = (bones.anchor[p as usize], bones.anchor[j]);
                    out[p as usize] + lin.apply(aj - ap)
                }
                _ => Vec3::ZERO,
            };
            out[j] = from_parent + extra;
        }
        out
    }

    /// The displacement of a surface point `p` with skin influences `joints` / `weights`.
    #[must_use]
    pub fn displacement(&self, bones: &Bones, offsets: &[Vec3], joints: &[u16], weights: &[f32], p: Vec3) -> Vec3 {
        let mut d = Vec3::ZERO;
        let mut total = 0.0;
        for (&j, &w) in joints.iter().zip(weights) {
            let (Some(&o), Some(lin), Some(&a)) =
                (offsets.get(j as usize), self.lin.get(j as usize), bones.anchor.get(j as usize))
            else {
                continue;
            };
            d += (o + lin.apply(p - a)) * w;
            total += w;
        }
        if total > 0.0 { d * (1.0 / total) } else { Vec3::ZERO }
    }
}

/// Per-node displacements and `(joint, offset)` pairs of a skeletal component.
pub type SkeletalField = (Vec<Vec3>, Vec<(u16, Vec3)>);

/// Node displacements and the regressor row of a skeletal component. With `ground`, everything
/// is shifted vertically so that the ankles keep their height (the soles stay on the floor).
pub fn skeletal(space: &Space<'_>, bones: &Bones, stretch: &Stretch, ground: bool) -> IdResult<SkeletalField> {
    let mut offsets = stretch.offsets(bones);
    if ground {
        let mut feet = Vec::new();
        for name in ["L.foot", "R.foot"] {
            let j = space.joint_index(name)?;
            feet.push(offsets.get(j as usize).map_or(0.0, |o| o.y));
        }
        let shift = Vec3::Y * (feet.iter().sum::<f32>() / feet.len().max(1) as f32);
        for o in &mut offsets {
            *o -= shift;
        }
    }
    let template = space.template();
    let reps = space.field().representatives();
    let disp = space.eval(|pt| {
        let v = reps.get(pt.node).copied().unwrap_or(u32::MAX) as usize;
        let (joints, weights) = template.skin().vertex(v);
        stretch.displacement(bones, &offsets, joints, weights, pt.p)
    });
    let row = offsets.iter().enumerate().map(|(j, &o)| (j as u16, o)).collect();
    Ok((disp, row))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bone_ends_follow_the_rig_names() {
        assert_eq!(bone_end("pelvis"), Some(End::Joint("spine.01".into())));
        assert_eq!(bone_end("L.upper-arm.twist"), Some(End::Joint("L.forearm".into())));
        assert_eq!(bone_end("R.index.03"), Some(End::Landmark("R.hand.index.tip".into())));
        assert_eq!(bone_end("R.thumb.01"), Some(End::Joint("R.thumb.02".into())));
        assert_eq!(bone_end("L.toes"), Some(End::Landmark("L.foot.tip".into())));
        assert_eq!(bone_end("L.eye"), None);
        assert_eq!(bone_end("X.hand"), None);
        assert_eq!(bone_end(""), None);
        assert_eq!(bone_end("tongue.04"), None);
    }

    #[test]
    fn lin_maps() {
        let m = Lin::outer(Vec3::Y, Vec3::X, 2.0).plus(Lin::scale(1.0));
        assert_eq!(m.apply(Vec3::new(1.0, 0.0, 0.0)), Vec3::new(1.0, 2.0, 0.0));
        assert_eq!(Lin::ZERO.apply(Vec3::ONE), Vec3::ZERO);
    }

    #[test]
    fn chain_offsets_and_blend() {
        // A three-joint chain along +Y: lengthening the middle bone moves the last joint, and a
        // point halfway along the middle bone moves halfway.
        let bones = Bones {
            anchor: vec![Vec3::ZERO, Vec3::Y, Vec3::Y * 2.0],
            axis: vec![Vec3::Y, Vec3::Y, Vec3::ZERO],
            length: vec![1.0, 1.0, 0.0],
            parent: vec![None, Some(0), Some(1)],
        };
        let mut s = Stretch::new(3);
        s.lengthen(&bones, 1, 0.1).unwrap();
        let o = s.offsets(&bones);
        assert_eq!(o[0], Vec3::ZERO);
        assert_eq!(o[1], Vec3::ZERO);
        assert!((o[2].y - 0.1).abs() < 1e-6);
        let mid = s.displacement(&bones, &o, &[1], &[1.0], Vec3::Y * 1.5);
        assert!((mid.y - 0.05).abs() < 1e-6);
        // At the joint, both bones give the same displacement: the blend is continuous.
        let a = s.displacement(&bones, &o, &[1], &[1.0], Vec3::Y * 2.0);
        let b = s.displacement(&bones, &o, &[2], &[1.0], Vec3::Y * 2.0);
        assert!((a - b).length() < 1e-6);
        assert!(bones.is_within(2, 0) && !bones.is_within(0, 2));
        let mut g = Stretch::new(3);
        g.scale_subtree(&bones, 1, 0.5).unwrap();
        let o = g.offsets(&bones);
        assert!((o[2].y - 0.5).abs() < 1e-6);
        assert!(g.lengthen(&bones, 7, 1.0).is_err());
        assert!(g.push(9, Vec3::X).is_err());
        assert_eq!(g.displacement(&bones, &o, &[], &[], Vec3::ONE), Vec3::ZERO);
    }
}
