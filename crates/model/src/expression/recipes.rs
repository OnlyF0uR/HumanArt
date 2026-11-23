//! The authored expression targets: the FACS action units and additional controls of Appendix B,
//! viseme-support shapes and tongue shapes (WP-19, `[EXP-01]`, `[EXP-05]`, `[EXP-06]`).
//!
//! **Provenance.** Every displacement amount, angle and radius in this file is an *authored model
//! choice* made by eye on the generated template, in the spirit of the FACS descriptions of each
//! action's appearance; none is a measurement or an anatomical statistic. Captured data replaces
//! these targets through the statistical pipeline (`[STAT-05]`) when it exists.
//!
//! **Robustness.** Recipes find everything by name (landmarks, joints, regions, parts) and shape it
//! with geodesic falloffs ([`super::field`]), so they follow the generator's face when its shape or
//! topology changes. Bilateral actions are authored once with seeds on both sides; the left and
//! right targets are the bilateral displacement times a smooth midline split
//! ([`super::field::side_weight`]), so `left + right == bilateral` exactly and the pair is
//! mirror-symmetric on a symmetric template.
//!
//! **Skeleton-driven motion.** The jaw (AU26, AU27, AU30) and the tongue rotate through joint
//! drivers of the control rig (`data/template/expressions/drivers.json`); the targets of those
//! controls are *support shapes*: the residual between an authored soft-tissue result (rotation
//! blending instead of the linear blend's shrinkage, the corners following half way, the
//! submandibular skin following the chin) and what the skinning alone produces, computed in the
//! rest space so that skinning the target lands on the authored result ([`Recipes::rotation_support`]).
//! The angles they were computed for are the constants [`JAW_DROP`], [`MOUTH_STRETCH`],
//! [`JAW_SIDE`], [`VISEME_ROUND`] and [`VISEME_WIDE`]; the drivers must use the same values (tested).

use humanart_math::{Mat3, Quat, Vec3};

use super::ExpressionKind;
use super::action::{Action, AreaBounds, Mask, Shape};
use super::error::{ExprError, ExprResult};
use super::field::{FaceField, falloff, side_weight};
use super::sweep::{AREA_EXEMPT_PREFIXES, MIN_RATIO_AREA};
use crate::eval::skinning::JointFrames;
use crate::names::Side;

/// The jaw opening of AU26 at full weight, radians about the world X axis (authored).
pub const JAW_DROP: f32 = 0.20;
/// The jaw opening of AU27 at full weight, radians (authored).
pub const MOUTH_STRETCH: f32 = 0.28;
/// The jaw yaw of AU30 at full weight, radians about the world Y axis (authored).
pub const JAW_SIDE: f32 = 0.08;
/// The jaw opening of the rounded-open viseme, radians (authored).
pub const VISEME_ROUND: f32 = 0.12;
/// The jaw opening of the wide-open viseme, radians (authored).
pub const VISEME_WIDE: f32 = 0.24;
/// The width of the smooth left/right split across the midline, metres (authored).
pub const MIDLINE_BLEND: f32 = 0.006;
/// The clearance kept between a closing eyelid and the eyeball surface, metres (authored).
pub const LID_CLEARANCE: f32 = 0.0004;
/// Rounds of regularisation against folds and over-stretch applied to every target
/// ([`Shape::regularize`]).
pub const REGULARIZE_ROUNDS: usize = 80;
/// The smallest triangle area ratio a target may produce (a margin inside the sweep's 0.2).
pub const TARGET_MIN_AREA_RATIO: f32 = 0.25;
/// The largest triangle area ratio a target may produce (a margin inside the sweep's 5).
pub const TARGET_MAX_AREA_RATIO: f32 = 4.0;
/// The smallest cosine between a triangle's deformed and rest normals a target may produce: a
/// margin of about 6 degrees inside the sweep's fold test (90 degrees), which the rounding of the
/// evaluated positions can flip for a triangle stood on edge.
pub const TARGET_MIN_NORMAL_COS: f32 = 0.1;
/// [`TARGET_MIN_NORMAL_COS`] for triangles below the sweep's area floor or thinner than
/// [`SLIVER_QUALITY`] (the template's slivers at the lip corners): at most 60 degrees of turn, so a combination with another target does
/// not tip them over.
pub const TARGET_MIN_NORMAL_COS_SMALL: f32 = 0.5;
/// Rest height over longest edge below which a triangle is a sliver for
/// [`TARGET_MIN_NORMAL_COS_SMALL`] (the lip-corner slivers are about 0.04).
pub const SLIVER_QUALITY: f32 = 0.08;
/// The reach of the lip line's ride on the upper lip, metres (authored, [`Recipes::ride_lips`]).
pub const HELPER_REACH: f32 = 0.003;
/// The straight distance (mm) from a lip corner within which the lining of the mouth follows a
/// corner pull fully, and beyond which not at all (authored, [`Recipes::corner_mask`]).
pub const LINING_INNER: f32 = 4.0;
/// See [`LINING_INNER`].
pub const LINING_OUTER: f32 = 14.0;
/// Laplacian smoothing steps of the lid rotation field (authored).
pub const LID_RELAX: usize = 4;
/// The distance from a canthus over which the lid motion tapers in, metres (authored).
pub const CANTHUS_TAPER: f32 = 0.009;
/// The geodesic reach of the lower-lid rotation from its margin, millimetres (authored).
pub const LOWER_LID_REACH: f32 = 20.0;
/// The half width of the soft-tissue blend between the skull and the jaw for the jaw support
/// shapes, metres of geodesic distance on each side of the skin-weight boundary (authored).
pub const JAW_BLEND: f32 = 0.010;

/// One authored expression target.
#[derive(Clone, Debug, PartialEq)]
pub struct Target {
    /// Stable expression name (the registry's target name: `AU12-L`, `ctl.tongue-out`).
    pub name: String,
    /// What it is.
    pub kind: ExpressionKind,
    /// The side it acts on.
    pub side: Side,
    /// The FACS action unit number, 0 when not an action unit.
    pub facs: u16,
    /// The displacement per welded node.
    pub shape: Shape,
}

const fn letter(side: Side) -> &'static str {
    match side {
        Side::Left => "L",
        Side::Right => "R",
        Side::Center => "C",
    }
}

const fn sign(side: Side) -> f32 {
    match side {
        Side::Right => -1.0,
        Side::Left | Side::Center => 1.0,
    }
}

/// Millimetres to metres.
const fn mm(x: f32) -> f32 {
    x * 1.0e-3
}

/// A vector in millimetres, as metres.
const fn v_mm(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x * 1.0e-3, y * 1.0e-3, z * 1.0e-3)
}

/// `names("L.brow", 1, 5)` = `L.brow.01` .. `L.brow.05`.
fn names(prefix: &str, first: u32, last: u32) -> Vec<String> {
    (first..=last).map(|k| format!("{prefix}.{k:02}")).collect()
}

/// `d_b / (d_a + d_b)` as a smooth 0..1 share of "closer to a"; infinite distances count as far.
fn share(d_a: f32, d_b: f32) -> f32 {
    match (d_a.is_finite(), d_b.is_finite()) {
        (true, true) if d_a + d_b > 0.0 => {
            let s = d_b / (d_a + d_b);
            let t = ((s - 0.3) / 0.4).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        }
        (true, false) => 1.0,
        (false, true) => 0.0,
        _ => 0.5,
    }
}

/// Per-eye data.
#[derive(Clone, Debug)]
struct Eye {
    centre: Vec3,
    upper: Vec<Vec3>,
    lower: Vec<Vec3>,
    medial: Vec3,
    lateral: Vec3,
    /// Nodes of the eye's overlay parts (tear line, eye occlusion), with whether the node rides
    /// the upper lid's rotation (the occlusion lens) instead of keeping its place between the margins.
    overlay: Vec<(u32, bool)>,
    upper_seeds: Vec<u32>,
    lower_seeds: Vec<u32>,
    crease_seeds: Vec<u32>,
    /// 1 on the upper-lid side of the eye opening, 0 on the lower.
    upper_share: Vec<f32>,
    /// Unit directions and radii of the eyeball surface around the centre.
    ball: Vec<(Vec3, f32)>,
}

impl Eye {
    /// The eyeball surface radius in direction `dir` (the largest radius within about 12 degrees).
    fn radius(&self, dir: Vec3) -> f32 {
        let mut r = 0.0_f32;
        let mut fallback = f32::MAX;
        for &(d, rr) in &self.ball {
            fallback = fallback.min(rr);
            if d.dot(dir) > 0.978 {
                r = r.max(rr);
            }
        }
        if r > 0.0 {
            r
        } else if fallback < f32::MAX {
            fallback
        } else {
            0.0
        }
    }
}

/// The recipe context of one template: the face field plus precomputed lip, lid and jaw data.
#[derive(Clone, Debug)]
pub struct Recipes<'f, 't> {
    f: &'f FaceField<'t>,
    body: u16,
    lip_upper_share: Vec<f32>,
    lip_dist: Vec<f32>,
    corner_dist: Vec<f32>,
    margin_dist: Vec<f32>,
    eyes: [Eye; 2],
    jaw_family: Vec<f32>,
    helper: u16,
    /// Per mouth-helper node: the upper-lip nodes it rides and their weights (summing to 1,
    /// [`Recipes::ride_lips`]).
    helper_anchor: Vec<(u32, Vec<(u32, f32)>)>,
    /// Per node: whether it lines the mouth (the `mouth-interior` region of the skin).
    interior: Vec<bool>,
    bounds: AreaBounds,
}

impl<'f, 't> Recipes<'f, 't> {
    /// Precomputes the shared fields; fails when a required landmark, joint, region or part is
    /// missing from the template.
    pub fn new(f: &'f FaceField<'t>) -> ExprResult<Self> {
        let body = f.part("body")?;
        let upper_line = f.line_seeds(&names("lip.upper.inner", 1, 11))?;
        let lower_line = f.line_seeds(&names("lip.lower.inner", 1, 11))?;
        let du = f.distances(&upper_line, 0.06);
        let dl = f.distances(&lower_line, 0.06);
        let lip_upper_share: Vec<f32> = du.iter().zip(&dl).map(|(&a, &b)| share(a, b)).collect();
        let lip_dist: Vec<f32> = du.iter().zip(&dl).map(|(&a, &b)| a.min(b)).collect();
        // The corner is the chain from the cheilion (vermilion border) to the commissure (contact
        // line): both ends seed the corner distance, so the inner corner counts as corner too.
        let corners = [
            f.landmark_node("L.lip.corner")?,
            f.landmark_node("R.lip.corner")?,
            f.landmark_node("L.lip.commissure")?,
            f.landmark_node("R.lip.commissure")?,
        ];
        let corner_dist = f.distances(&corners, 0.06);
        let eyes = [Self::eye(f, Side::Left)?, Self::eye(f, Side::Right)?];
        let mut margins: Vec<u32> = Vec::new();
        for e in &eyes {
            margins.extend_from_slice(&e.upper_seeds);
            margins.extend_from_slice(&e.lower_seeds);
        }
        let margin_dist = f.distances(&margins, 0.04);
        let mut family = vec![f.joint_index("jaw")?];
        for k in 1..=4 {
            family.push(f.joint_index(&format!("tongue.{k:02}"))?);
        }
        let jaw_family: Vec<f32> =
            (0..f.node_count()).map(|n| family.iter().map(|&j| f.skin_weight(n, j)).sum::<f32>().min(1.0)).collect();
        let helper = f.part("mouth-helper")?;
        let exempt_parts = f
            .template()
            .mesh()
            .parts()
            .iter()
            .enumerate()
            .filter(|(_, p)| AREA_EXEMPT_PREFIXES.iter().any(|x| p.name.starts_with(x)))
            .filter_map(|(i, _)| u16::try_from(i).ok())
            .collect();
        let bounds = AreaBounds {
            min_ratio: TARGET_MIN_AREA_RATIO,
            max_ratio: TARGET_MAX_AREA_RATIO,
            min_area: MIN_RATIO_AREA,
            exempt_parts,
            min_cos: TARGET_MIN_NORMAL_COS,
            min_cos_small: TARGET_MIN_NORMAL_COS_SMALL,
            sliver_quality: SLIVER_QUALITY,
        };
        // The lip line (mouth helper) is a thin tube along the contact line, connected to nothing:
        // geodesic masks never reach it. Each of its nodes rides the nearest upper-lip node at
        // the contact.
        let upper_contact: Vec<usize> = (0..f.node_count())
            .filter(|&n| f.part_of(n) == body && lip_upper_share[n] > 0.5 && lip_dist[n] < mm(1.5))
            .collect();
        // A smooth (Gaussian) average over the contact nodes within HELPER_REACH, so the nodes of
        // one cross-section of the tube (at most 0.8 mm apart) ride nearly the same point.
        let helper_anchor = (0..f.node_count())
            .filter(|&n| f.part_of(n) == helper)
            .filter_map(|n| {
                let p = f.positions()[n];
                let mut near: Vec<(u32, f32)> = upper_contact
                    .iter()
                    .filter_map(|&a| {
                        let d = (f.positions()[a] - p).length();
                        let w = (-(d / (0.5 * HELPER_REACH)).powi(2)).exp();
                        (d < HELPER_REACH).then_some((u32::try_from(a).ok()?, w))
                    })
                    .collect();
                if near.is_empty() {
                    let best = upper_contact.iter().copied().min_by(|&a, &b| {
                        (f.positions()[a] - p).length_squared().total_cmp(&(f.positions()[b] - p).length_squared())
                    })?;
                    near.push((u32::try_from(best).ok()?, 1.0));
                }
                let sum: f32 = near.iter().map(|x| x.1).sum();
                for x in &mut near {
                    x.1 /= sum;
                }
                Some((u32::try_from(n).ok()?, near))
            })
            .collect();
        let lining = f.region("mouth-interior")?;
        let interior = (0..f.node_count()).map(|n| f.part_of(n) == body && f.in_region(n, lining)).collect();
        Ok(Self {
            f,
            body,
            lip_upper_share,
            lip_dist,
            corner_dist,
            margin_dist,
            eyes,
            jaw_family,
            helper,
            helper_anchor,
            interior,
            bounds,
        })
    }

    fn eye(f: &FaceField<'_>, side: Side) -> ExprResult<Eye> {
        let s = letter(side);
        let centre = f.joint(&format!("{s}.eye"))?;
        let upper_names = names(&format!("{s}.eye.upper"), 1, 7);
        let lower_names = names(&format!("{s}.eye.lower"), 1, 7);
        let mut upper = Vec::new();
        for n in &upper_names {
            upper.push(f.landmark(n)?);
        }
        let mut lower = Vec::new();
        for n in &lower_names {
            lower.push(f.landmark(n)?);
        }
        let medial = f.landmark(&format!("{s}.eye.canthus.medial"))?;
        let lateral = f.landmark(&format!("{s}.eye.canthus.lateral"))?;
        let mut up_line = vec![format!("{s}.eye.canthus.medial")];
        up_line.extend(upper_names);
        up_line.push(format!("{s}.eye.canthus.lateral"));
        let mut lo_line = vec![format!("{s}.eye.canthus.medial")];
        lo_line.extend(lower_names);
        lo_line.push(format!("{s}.eye.canthus.lateral"));
        // The canthi belong to both lids; seed each lid from its own margin points only.
        let upper_seeds = f.line_seeds(&up_line[1..up_line.len() - 1])?;
        let lower_seeds = f.line_seeds(&lo_line[1..lo_line.len() - 1])?;
        let crease_seeds = f.line_seeds(&names(&format!("{s}.eye.crease"), 1, 7))?;
        let du = f.distances(&upper_seeds, 0.03);
        let dl = f.distances(&lower_seeds, 0.03);
        let upper_share = du.iter().zip(&dl).map(|(&a, &b)| share(a, b)).collect();
        let (tear, lens) = (f.part(&format!("tear-line-{s}"))?, f.part(&format!("eye-occlusion-{s}"))?);
        let overlay = (0..f.node_count())
            .filter(|&n| f.part_of(n) == tear || f.part_of(n) == lens)
            .map(|n| (n as u32, f.part_of(n) == lens))
            .collect();
        let ball_part = f.part(&format!("eyeball-{s}"))?;
        let ball = (0..f.node_count())
            .filter(|&n| f.part_of(n) == ball_part)
            .filter_map(|n| {
                let d = f.positions()[n] - centre;
                let r = d.length();
                (r > 0.0).then(|| (d / r, r))
            })
            .collect();
        Ok(Eye {
            centre,
            upper,
            lower,
            medial,
            lateral,
            overlay,
            upper_seeds,
            lower_seeds,
            crease_seeds,
            upper_share,
            ball,
        })
    }

    fn eye_of(&self, side: Side) -> &Eye {
        if side == Side::Right { &self.eyes[1] } else { &self.eyes[0] }
    }

    fn shape(&self) -> Shape {
        Shape::new(self.f.node_count())
    }

    fn seeds(&self, names: &[String]) -> ExprResult<Vec<u32>> {
        self.f.line_seeds(names)
    }

    fn node(&self, name: &str) -> ExprResult<u32> {
        self.f.landmark_node(name)
    }

    /// Body nodes within `outer` (geodesic) of `seeds`, weight 1 up to `inner`.
    fn geo(&self, seeds: &[u32], inner: f32, outer: f32) -> Mask {
        Mask::geodesic(self.f, seeds, inner, outer)
    }

    fn x(&self, n: usize) -> f32 {
        self.f.positions()[n].x
    }

    /// Applies `action(side)` to `mask` split smoothly into its left and right halves.
    fn sided(&self, shape: &mut Shape, mask: &Mask, action: impl Fn(Side) -> Action) {
        for side in [Side::Left, Side::Right] {
            let m = mask.scale(|n| side_weight(self.x(n), side, MIDLINE_BLEND));
            shape.apply(self.f, &m, action(side));
        }
    }

    /// One side's part of a bilateral shape.
    fn split(&self, shape: &Shape, side: Side) -> Shape {
        let mut s = shape.clone();
        s.scale(|n| side_weight(self.x(n), side, MIDLINE_BLEND));
        s
    }

    /// The nearest body node to `p`.
    fn near(&self, p: Vec3) -> ExprResult<u32> {
        self.f.nearest(p, self.body).ok_or_else(|| ExprError::missing("mesh part nodes", "body"))
    }

    /// The nearest body node to the centroid of region `name`.
    fn region_centre(&self, name: &str) -> ExprResult<u32> {
        let r = self.f.region(name)?;
        let (mut sum, mut count) = (Vec3::ZERO, 0_u32);
        for n in 0..self.f.node_count() {
            if self.f.part_of(n) == self.body && self.f.in_region(n, r) {
                sum += self.f.positions()[n];
                count += 1;
            }
        }
        if count == 0 {
            return Err(ExprError::missing("region nodes", name));
        }
        self.near(sum / count as f32)
    }

    /// 0 at the lid margins, 1 from `outer` away: keeps brow and cheek actions off the lid edge.
    fn off_margin(&self, n: usize, inner: f32, outer: f32) -> f32 {
        1.0 - falloff(self.margin_dist.get(n).copied().unwrap_or(f32::INFINITY), inner, outer)
    }

    /// 0 at the lip corners, 1 from `outer` away.
    fn off_corner(&self, n: usize, outer: f32) -> f32 {
        1.0 - falloff(self.corner_dist.get(n).copied().unwrap_or(f32::INFINITY), 0.0, outer)
    }

    /// The upper-lip share (1 upper, 0 lower) of node `n`.
    fn upper(&self, n: usize) -> f32 {
        self.lip_upper_share.get(n).copied().unwrap_or(0.5)
    }

    /// Lip line seeds: `upper`/`lower` lip, `inner` (contact) or `outer` (vermilion border).
    fn lip_line(&self, lip: &str, edge: &str) -> ExprResult<Vec<u32>> {
        self.seeds(&names(&format!("lip.{lip}.{edge}"), 1, 11))
    }

    /// The mouth's contact point (between the middle inner lip landmarks).
    fn mouth_centre(&self) -> ExprResult<Vec3> {
        Ok((self.f.landmark("lip.upper.inner.06")? + self.f.landmark("lip.lower.inner.06")?) * 0.5)
    }

    // ---------------------------------------------------------------- brows and forehead

    fn brow_raise(&self, first: u32, last: u32, lift: f32, outer: f32) -> ExprResult<Shape> {
        let mut seeds = Vec::new();
        for side in [Side::Left, Side::Right] {
            seeds.extend(self.seeds(&names(&format!("{}.brow", letter(side)), first, last))?);
        }
        let m = self.geo(&seeds, mm(3.0), mm(outer)).scale(|n| self.off_margin(n, mm(2.0), mm(9.0)));
        let mut s = self.shape();
        s.apply(self.f, &m, Action::Translate { offset: v_mm(0.0, lift, 0.4) });
        Ok(s)
    }

    /// AU1: the inner brows rise (forehead skin bunches above them).
    fn au01(&self) -> ExprResult<Shape> {
        self.brow_raise(1, 2, 7.0, 28.0)
    }

    /// AU2: the outer brows rise.
    fn au02(&self) -> ExprResult<Shape> {
        self.brow_raise(3, 5, 7.5, 30.0)
    }

    /// AU4: the brows lower and draw together; the glabella bunches.
    fn au04(&self) -> ExprResult<Shape> {
        let mut seeds = Vec::new();
        for side in [Side::Left, Side::Right] {
            seeds.extend(self.seeds(&names(&format!("{}.brow", letter(side)), 1, 3))?);
        }
        seeds.push(self.node("forehead.glabella")?);
        let m = self.geo(&seeds, mm(3.0), mm(24.0)).scale(|n| self.off_margin(n, mm(1.5), mm(8.0)));
        let mut s = self.shape();
        s.apply(self.f, &m, Action::Translate { offset: v_mm(0.0, -4.5, 0.6) });
        self.sided(&mut s, &m, |side| Action::Translate { offset: v_mm(-3.2 * sign(side), 0.0, 0.0) });
        let root = self.geo(&[self.node("nose.root")?], mm(2.0), mm(12.0));
        s.apply(self.f, &root, Action::Translate { offset: v_mm(0.0, -1.2, 0.0) });
        Ok(s)
    }

    /// Additional control: the brows pull toward the glabella (corrugator), little lowering.
    fn brow_ridge_compress(&self) -> ExprResult<Shape> {
        let mut seeds = Vec::new();
        for side in [Side::Left, Side::Right] {
            seeds.extend(self.seeds(&names(&format!("{}.brow", letter(side)), 1, 3))?);
        }
        let m = self.geo(&seeds, mm(2.0), mm(18.0)).scale(|n| self.off_margin(n, mm(1.5), mm(8.0)));
        let mut s = self.shape();
        self.sided(&mut s, &m, |side| Action::Translate { offset: v_mm(-3.0 * sign(side), -0.8, 0.5) });
        Ok(s)
    }

    // ---------------------------------------------------------------- eyelids

    /// The rotation (about the world X axis through the eye centre) that takes margin point `from`
    /// to the height of `to`.
    fn lid_angle(c: Vec3, from: Vec3, to: Vec3) -> f32 {
        let a = (from.y - c.y).atan2(from.z - c.z);
        let b = (to.y - c.y).atan2(to.z - c.z);
        a - b
    }

    /// Piecewise-linear angle over x through `(x, angle)` samples sorted by x; 0 outside.
    fn angle_at(samples: &[(f32, f32)], x: f32) -> f32 {
        let (Some(first), Some(last)) = (samples.first(), samples.last()) else { return 0.0 };
        if x <= first.0 || x >= last.0 {
            return 0.0;
        }
        for w in samples.windows(2) {
            if x >= w[0].0 && x <= w[1].0 {
                let span = w[1].0 - w[0].0;
                let t = if span > 0.0 { (x - w[0].0) / span } else { 0.0 };
                return w[0].1 + (w[1].1 - w[0].1) * t;
            }
        }
        0.0
    }

    /// The rotations (about the world X axis through the eye centre) of the margin points when
    /// the upper (`upper_lid`) or lower margin moves a fraction `frac` of the way to the other:
    /// `(x, angle)` samples sorted by x, 0 at the canthi.
    fn margin_angles(e: &Eye, upper_lid: bool, frac: f32) -> Vec<(f32, f32)> {
        let mut samples: Vec<(f32, f32)> = vec![(e.medial.x, 0.0), (e.lateral.x, 0.0)];
        for (u, l) in e.upper.iter().zip(&e.lower) {
            let (from, to) = if upper_lid { (*u, *u + (*l - *u) * frac) } else { (*l, *l + (*u - *l) * frac) };
            // Taper in from the canthi, where the margins converge and the lid surface turns sideways.
            let d = ((from.x - e.medial.x).abs().min((from.x - e.lateral.x).abs()) / CANTHUS_TAPER).clamp(0.0, 1.0);
            samples.push((from.x, Self::lid_angle(e.centre, from, to) * d * d * (3.0 - 2.0 * d)));
        }
        samples.sort_by(|a, b| a.0.total_cmp(&b.0));
        samples
    }

    /// A margin (canthus, margin landmarks, canthus) as a densely sampled polyline.
    fn margin_line(e: &Eye, upper_lid: bool) -> Vec<Vec3> {
        let mut pts = vec![e.medial];
        pts.extend_from_slice(if upper_lid { &e.upper } else { &e.lower });
        pts.push(e.lateral);
        pts.sort_by(|a, b| a.x.total_cmp(&b.x));
        let mut out = Vec::with_capacity(pts.len() * 8);
        for w in pts.windows(2) {
            for k in 0..8 {
                out.push(w[0] + (w[1] - w[0]) * (k as f32 / 8.0));
            }
        }
        out.extend(pts.last());
        out
    }

    /// The distance from `p` to the nearest point of a sampled polyline.
    fn line_distance(line: &[Vec3], p: Vec3) -> f32 {
        line.iter().map(|q| (*q - p).length()).fold(f32::INFINITY, f32::min)
    }

    /// Moves one eye's lids: the upper margin a fraction `up` of the way down to the lower margin,
    /// the lower margin a fraction `low` of the way up (negative values open), by rotating the lid
    /// skin about the eye centre; then keeps the lids outside the eyeball (`[EXP-05]` contact). The
    /// overlay parts of the eye (tear line, eye occlusion) keep their relative place between the
    /// two margins, so they close with the lids without folding.
    fn lids(&self, side: Side, up: f32, low: f32) -> Shape {
        let e = self.eye_of(side);
        let mut s = self.shape();
        let upper = Self::margin_angles(e, true, up);
        let lower = Self::margin_angles(e, false, low);
        for (upper_lid, samples) in [(true, &upper), (false, &lower)] {
            let signed_max = samples.iter().map(|s| s.1).fold(0.0_f32, |m, a| if a.abs() > m.abs() { a } else { m });
            if signed_max == 0.0 {
                continue;
            }
            let seeds = if upper_lid { &e.upper_seeds } else { &e.lower_seeds };
            let reach = if upper_lid { 10.0 } else { LOWER_LID_REACH };
            let m = self.geo(seeds, mm(1.2), mm(reach)).scale(|n| {
                let lid = e.upper_share.get(n).copied().unwrap_or(0.5);
                let own = if upper_lid { lid } else { 1.0 - lid };
                own * Self::angle_at(samples, self.x(n)) / signed_max
            });
            s.apply(self.f, &m, Action::Rotate { pivot: e.centre, axis: Vec3::X, angle: signed_max });
        }
        self.contact(side, &mut s);
        // Overlays: interpolate the margin rotations by the node's place between the margins.
        let (up_line, low_line) = (Self::margin_line(e, true), Self::margin_line(e, false));
        for &(n, rides_upper) in &e.overlay {
            let p = self.f.positions()[n as usize];
            let t = if rides_upper {
                0.0
            } else {
                // The tear line rides the nearer lid, blending only where both are about as near.
                let (du, dl) = (Self::line_distance(&up_line, p), Self::line_distance(&low_line, p));
                let t = if du + dl > 0.0 { ((du / (du + dl) - 0.3) / 0.4).clamp(0.0, 1.0) } else { 0.5 };
                t * t * (3.0 - 2.0 * t)
            };
            let a = Self::angle_at(&upper, p.x) * (1.0 - t) + Self::angle_at(&lower, p.x) * t;
            if let Some(d) = s.displacements_mut().get_mut(n as usize) {
                *d = Action::Rotate { pivot: e.centre, axis: Vec3::X, angle: a }.apply(p, Vec3::ZERO, 1.0) - p;
            }
        }
        s
    }

    /// Smooths a shape's displacement over one eye's lid skin (folds start where the lid falloff
    /// meets the lid-cheek crease, or where cheek and lid motions stack), then restores the lids'
    /// contact with the eyeball `[EXP-05]`.
    fn contact(&self, side: Side, s: &mut Shape) {
        let e = self.eye_of(side);
        let both = self.geo(&e.upper_seeds, 0.0, mm(10.0)).union(&self.geo(&e.lower_seeds, 0.0, mm(LOWER_LID_REACH)));
        s.relax(self.f, LID_RELAX, 0.5, |n| self.f.part_of(n) == self.body && both.weight(n as u32) > 0.0);
        let outside = both.scale(|n| {
            let d = self.f.positions()[n] - e.centre;
            let r = d.length();
            if r > 0.0 && r >= e.radius(d / r) { 1.0 } else { 0.0 }
        });
        s.keep_outside(self.f, &outside, e.centre, |dir| e.radius(dir) + LID_CLEARANCE);
    }

    /// Deepens one eye's upper-lid crease by `depth` metres (inward along the normal).
    fn crease(&self, side: Side, depth: f32) -> Shape {
        let e = self.eye_of(side);
        let m = self.geo(&e.crease_seeds, mm(0.4), mm(3.0));
        let mut s = self.shape();
        s.apply(self.f, &m, Action::Normal { amount: -depth });
        s
    }

    /// AU5: the upper lid rises; the crease deepens.
    fn au05(&self, side: Side) -> Shape {
        let mut s = self.lids(side, -0.35, 0.0);
        s.add(&self.crease(side, mm(0.6)), 1.0);
        s
    }

    /// AU6: the cheeks rise and push the lower lid up; skin gathers at the outer eye corner.
    fn au06(&self) -> ExprResult<Shape> {
        let mut s = self.shape();
        for side in [Side::Left, Side::Right] {
            let l = letter(side);
            let cheek = self.geo(&[self.node(&format!("{l}.cheek.malar"))?], mm(5.0), mm(30.0));
            let cheek = cheek.scale(|n| 0.35 + 0.65 * self.off_margin(n, mm(1.0), mm(6.0)));
            s.apply(self.f, &cheek, Action::Translate { offset: v_mm(0.0, 3.5, 1.2) });
            let canthus = self.node(&format!("{l}.eye.canthus.lateral"))?;
            let corner = self.geo(&[canthus], mm(2.0), mm(14.0)).scale(|n| self.off_margin(n, mm(1.0), mm(4.0)));
            let target = self.f.landmark(&format!("{l}.eye.canthus.lateral"))?;
            s.apply(self.f, &corner, Action::Scale { centre: target, factor: Vec3::new(0.86, 0.86, 1.0) });
            s.add(&self.lids(side, 0.0, 0.12), 1.0);
        }
        for side in [Side::Left, Side::Right] {
            self.contact(side, &mut s);
        }
        Ok(s)
    }

    /// AU7: the lower lid rises and straightens; the upper lid lowers a little.
    fn au07(&self, side: Side) -> Shape {
        self.lids(side, 0.08, 0.32)
    }

    /// AU41: the upper lid droops.
    fn au41(&self, side: Side) -> Shape {
        self.lids(side, 0.3, 0.0)
    }

    /// AU42: the eye narrows to a slit.
    fn au42(&self, side: Side) -> Shape {
        self.lids(side, 0.6, 0.3)
    }

    /// AU43 / AU45: the eye closes (the margins stop 2% of the opening short of each other:
    /// closer settings fold slivers at the meeting line).
    fn closed(&self, side: Side) -> Shape {
        let mut s = self.lids(side, 0.84, 0.14);
        self.regularize(&mut s);
        s
    }

    /// The in-between of a closing eye at half weight: the lids at half closure minus half of the
    /// full closure. Linear blending would move the margin along the chord, into the eyeball; this
    /// shape, driven by a tent curve peaking at 0.5 in the rig logic, keeps it on the arc.
    fn closed_inbetween(&self, side: Side) -> Shape {
        let mut s = self.lids(side, 0.42, 0.07);
        self.regularize(&mut s);
        s.add(&self.closed(side), -0.5);
        s
    }

    /// AU44: squint, mostly the lower lid with a little cheek.
    fn au44(&self, side: Side) -> ExprResult<Shape> {
        let mut s = self.lids(side, 0.15, 0.45);
        let cheek = self.geo(&[self.node(&format!("{}.cheek.malar", letter(side)))?], mm(4.0), mm(22.0));
        let cheek = cheek.scale(|n| self.off_margin(n, mm(1.0), mm(6.0)));
        s.apply(self.f, &cheek, Action::Translate { offset: v_mm(0.0, 1.2, 0.4) });
        self.contact(side, &mut s);
        Ok(s)
    }

    /// AU46: a wink, closing with cheek raise and some brow lowering on that side.
    fn au46(&self, side: Side, au06: &Shape, au04: &Shape) -> Shape {
        let mut s = self.closed(side);
        s.add(&self.split(au06, side), 0.6);
        s.add(&self.split(au04, side), 0.3);
        self.contact(side, &mut s);
        s
    }

    // ---------------------------------------------------------------- nose and cheeks

    /// AU9: the nose wrinkles: bridge skin up, alae up, inner brows and upper lip slightly up.
    fn au09(&self) -> ExprResult<Shape> {
        let mut s = self.shape();
        let bridge = self.geo(&[self.node("nose.bridge.01")?, self.node("nose.bridge.02")?], mm(3.0), mm(14.0));
        s.apply(self.f, &bridge, Action::Translate { offset: v_mm(0.0, 2.5, -0.3) });
        let mut ala = Vec::new();
        for side in [Side::Left, Side::Right] {
            ala.push(self.node(&format!("{}.nose.alar-crease", letter(side)))?);
        }
        let ala = self.geo(&ala, mm(3.0), mm(15.0)).scale(|n| 0.3 + 0.7 * self.upper(n));
        s.apply(self.f, &ala, Action::Translate { offset: v_mm(0.0, 2.2, 0.3) });
        let root =
            self.geo(&[self.node("nose.root")?], mm(3.0), mm(16.0)).scale(|n| self.off_margin(n, mm(1.0), mm(6.0)));
        s.apply(self.f, &root, Action::Translate { offset: v_mm(0.0, -1.5, 0.0) });
        Ok(s)
    }

    /// AU38: the nostrils widen (alae outward).
    fn nostrils(&self, outward: f32, lift: f32, reach: f32) -> ExprResult<Shape> {
        let mut seeds = Vec::new();
        for side in [Side::Left, Side::Right] {
            let l = letter(side);
            seeds.push(self.node(&format!("{l}.nose.alare"))?);
            seeds.push(self.node(&format!("{l}.nose.alar-crease"))?);
        }
        let m = self.geo(&seeds, mm(1.5), mm(reach));
        let mut s = self.shape();
        self.sided(&mut s, &m, |side| Action::Translate { offset: v_mm(outward * sign(side), lift, 0.0) });
        Ok(s)
    }

    /// AU33/AU34/AU35: the cheeks blow out, puff or are sucked in (along the normal).
    fn cheeks(&self, amount: f32, inner: f32, outer: f32) -> ExprResult<Shape> {
        let seeds = [self.region_centre("cheek-L")?, self.region_centre("cheek-R")?];
        let m = self.geo(&seeds, mm(inner), mm(outer)).scale(|n| self.off_margin(n, mm(4.0), mm(14.0)));
        let mut s = self.shape();
        s.apply(self.f, &m, Action::Normal { amount: mm(amount) });
        s.relax(self.f, 4, 0.5, |n| self.f.part_of(n) == self.body);
        Ok(s)
    }

    /// AU31: the masseters bulge.
    fn au31(&self) -> ExprResult<Shape> {
        let mut seeds = Vec::new();
        for side in [Side::Left, Side::Right] {
            let l = letter(side);
            let p = (self.f.landmark(&format!("{l}.zygion"))? + self.f.landmark(&format!("jaw.{l}.01"))?) * 0.5;
            seeds.push(self.near(p)?);
        }
        let m = self.geo(&seeds, mm(3.0), mm(18.0));
        let mut s = self.shape();
        s.apply(self.f, &m, Action::Normal { amount: mm(1.8) });
        Ok(s)
    }

    // ---------------------------------------------------------------- lips and corners

    fn corner_seeds(&self) -> ExprResult<Vec<u32>> {
        let mut seeds = Vec::new();
        for side in [Side::Left, Side::Right] {
            let l = letter(side);
            seeds.push(self.node(&format!("{l}.lip.corner"))?);
            seeds.push(self.node(&format!("{l}.lip.commissure"))?);
        }
        Ok(seeds)
    }

    /// A corner action: `offset(side)` on the geodesic disc of radius `outer` around both corners.
    fn corners(&self, inner: f32, outer: f32, offset: impl Fn(Side) -> Vec3) -> ExprResult<Shape> {
        let m = self.geo(&self.corner_seeds()?, mm(inner), mm(outer));
        let mut s = self.shape();
        self.sided(&mut s, &m, |side| Action::Translate { offset: offset(side) });
        s.relax(self.f, 4, 0.5, |n| self.f.part_of(n) == self.body);
        Ok(s)
    }

    /// AU13: the corners pull up sharply (levator anguli oris); the lower lip follows less.
    fn au13(&self) -> ExprResult<Shape> {
        let m = self.corner_mask(0.0, mm(18.0))?.scale(|n| 0.35 + 0.65 * self.upper(n));
        let mut s = self.shape();
        self.sided(&mut s, &m, |side| Action::Translate { offset: v_mm(1.2 * sign(side), 4.8, -0.6) });
        s.relax(self.f, 4, 0.5, |n| self.f.part_of(n) == self.body);
        Ok(s)
    }

    /// AU10: the upper lip rises lateral to the philtrum; the nasolabial area lifts.
    fn au10(&self) -> ExprResult<Shape> {
        let mut seeds = self.seeds(&names("lip.upper.outer", 2, 5))?;
        seeds.extend(self.seeds(&names("lip.upper.outer", 7, 10))?);
        let m = self.geo(&seeds, mm(2.0), mm(15.0)).scale(|n| self.upper(n) * self.off_corner(n, mm(4.0)));
        let mut s = self.shape();
        s.apply(self.f, &m, Action::Translate { offset: v_mm(0.0, 3.2, 0.6) });
        let mut ala = Vec::new();
        for side in [Side::Left, Side::Right] {
            ala.push(self.node(&format!("{}.nose.alar-base", letter(side)))?);
        }
        let ala = self.geo(&ala, mm(2.0), mm(10.0));
        s.apply(self.f, &ala, Action::Translate { offset: v_mm(0.0, 1.0, 0.0) });
        Ok(s)
    }

    /// AU11: the skin lateral to the nasolabial furrow rises and moves over it.
    fn au11(&self) -> ExprResult<Shape> {
        let mut s = self.shape();
        for side in [Side::Left, Side::Right] {
            let l = letter(side);
            let a = self.node(&format!("{l}.nose.alar-crease"))?;
            let b = self.node(&format!("{l}.lip.corner"))?;
            let fold = self.f.path(a, b, 0.1);
            let m = self.geo(&fold, mm(1.0), mm(12.0)).scale(|n| {
                let lateral = (self.x(n) - self.f.positions()[b as usize].x * 0.6) * sign(side);
                falloff(-lateral, -mm(3.0), mm(3.0)) * self.upper(n)
            });
            s.apply(self.f, &m, Action::Translate { offset: v_mm(-0.6 * sign(side), 1.6, 0.6) });
        }
        Ok(s)
    }

    /// A corner mask: geodesic from the corner seeds over the skin, and for the lining of the
    /// mouth by straight distance to the nearer corner: the lining of the corner recess is far
    /// from the corner along the surface (round the inner lip) but right behind it, and when it
    /// lagged the pulled corner opened onto it (the red lining showed at both corners in the
    /// smile).
    fn corner_mask(&self, inner: f32, outer: f32) -> ExprResult<Mask> {
        let seeds = self.corner_seeds()?;
        let pts: Vec<Vec3> = seeds.iter().map(|&s| self.f.positions()[s as usize]).collect();
        let geo = self.geo(&seeds, inner, outer);
        let lining = Mask::from_fn(self.f, |n| {
            if !self.interior.get(n).copied().unwrap_or(false) {
                return 0.0;
            }
            let p = self.f.positions()[n];
            let d = pts.iter().map(|q| (*q - p).length()).fold(f32::INFINITY, f32::min);
            falloff(d, mm(LINING_INNER), mm(LINING_OUTER))
        });
        Ok(geo.union(&lining))
    }

    /// AU12: the lip corners pull up and back toward the cheekbones.
    fn au12(&self) -> ExprResult<Shape> {
        let m = self.corner_mask(mm(2.0), mm(30.0))?;
        let mut s = self.shape();
        for side in [Side::Left, Side::Right] {
            let l = letter(side);
            let corner = self.f.landmark(&format!("{l}.lip.corner"))?;
            let zyg = self.f.landmark(&format!("{l}.zygion"))?;
            let dir = (zyg - corner).normalize_or_zero();
            let half = m.scale(|n| side_weight(self.x(n), side, MIDLINE_BLEND));
            s.apply(self.f, &half, Action::Translate { offset: dir * mm(8.0) + v_mm(0.0, 1.5, 0.0) });
        }
        let cheek = self.geo(&[self.region_centre("cheek-L")?, self.region_centre("cheek-R")?], mm(4.0), mm(24.0));
        let cheek = cheek.scale(|n| self.off_margin(n, mm(2.0), mm(8.0)));
        s.apply(self.f, &cheek, Action::Normal { amount: mm(1.2) });
        s.relax(self.f, 3, 0.5, |n| self.f.part_of(n) == self.body);
        Ok(s)
    }

    /// AU14: the corners tighten and pull back into the cheek (buccinator), a little up, with a
    /// dimple lateral to them and the tightened tissue bulging beside the dimple. The one-sided
    /// form carries contempt (todo 1.8: 2.2 mm back and a 1 mm dimple read as a slight corner pull
    /// even at full weight).
    fn au14(&self) -> ExprResult<Shape> {
        let mut s = self.corners(0.0, 15.0, |side| v_mm(1.4 * sign(side), 0.9, -3.4))?;
        for side in [Side::Left, Side::Right] {
            let c = self.f.landmark(&format!("{}.lip.corner", letter(side)))?;
            let dimple = self.near(c + v_mm(7.0 * sign(side), -1.0, -2.0))?;
            let m = self.geo(&[dimple], mm(0.5), mm(5.5));
            s.apply(self.f, &m, Action::Normal { amount: -mm(1.6) });
            let bulge = self.near(c + v_mm(11.0 * sign(side), 2.5, -3.5))?;
            let m = self.geo(&[bulge], mm(2.0), mm(8.0)).scale(|n| self.off_corner(n, mm(6.0)));
            s.apply(self.f, &m, Action::Normal { amount: mm(0.7) });
        }
        s.relax(self.f, 2, 0.5, |n| self.f.part_of(n) == self.body);
        Ok(s)
    }

    /// AU16: the lower lip pulls down and everts; the corners stay.
    fn au16(&self) -> ExprResult<Shape> {
        let mut seeds = self.seeds(&names("lip.lower.outer", 3, 9))?;
        seeds.extend(self.seeds(&names("lip.lower.inner", 3, 9))?);
        let m = self.geo(&seeds, mm(3.0), mm(16.0)).scale(|n| (1.0 - self.upper(n)) * self.off_corner(n, mm(9.0)));
        let mut s = self.shape();
        s.apply(self.f, &m, Action::Translate { offset: v_mm(0.0, -3.8, 1.2) });
        Ok(s)
    }

    /// AU17: the chin boss pushes up and the lower lip pouts.
    fn au17(&self) -> ExprResult<Shape> {
        let chin = self.geo(&[self.node("chin.mid")?, self.node("chin.supramentale")?], mm(4.0), mm(22.0));
        let chin = chin.scale(|n| 1.0 - self.upper(n));
        let mut s = self.shape();
        s.apply(self.f, &chin, Action::Translate { offset: v_mm(0.0, 3.0, 1.0) });
        let lip = self
            .geo(&self.lip_line("lower", "outer")?, mm(2.0), mm(8.0))
            .scale(|n| (1.0 - self.upper(n)) * self.off_corner(n, mm(8.0)));
        s.apply(self.f, &lip, Action::Translate { offset: v_mm(0.0, 1.2, 1.5) });
        Ok(s)
    }

    /// Both lips' nodes (vermilion and skin around it) within `outer` of the lip lines.
    fn lips_mask(&self, inner: f32, outer: f32) -> ExprResult<Mask> {
        let mut seeds = self.lip_line("upper", "outer")?;
        seeds.extend(self.lip_line("lower", "outer")?);
        seeds.extend(self.lip_line("upper", "inner")?);
        seeds.extend(self.lip_line("lower", "inner")?);
        Ok(self.geo(&seeds, mm(inner), mm(outer)))
    }

    /// AU18: the lips push forward and gather toward the midline.
    fn au18(&self) -> ExprResult<Shape> {
        let c = self.mouth_centre()?;
        let m = self.lips_mask(2.0, 16.0)?;
        let mut s = self.shape();
        s.apply(self.f, &m, Action::Translate { offset: v_mm(0.0, 0.0, 5.5) });
        s.apply(self.f, &m, Action::Scale { centre: c, factor: Vec3::new(0.8, 1.0, 1.0) });
        s.relax(self.f, 3, 0.5, |n| self.f.part_of(n) == self.body);
        Ok(s)
    }

    /// AU22: the lips funnel: forward, flared and slightly apart, corners narrowed.
    fn au22(&self) -> ExprResult<Shape> {
        let c = self.mouth_centre()?;
        let m = self.lips_mask(2.0, 14.0)?;
        let mut s = self.shape();
        s.apply(self.f, &m.scale(|n| self.upper(n)), Action::Translate { offset: v_mm(0.0, 1.8, 4.5) });
        s.apply(self.f, &m.scale(|n| 1.0 - self.upper(n)), Action::Translate { offset: v_mm(0.0, -2.2, 4.5) });
        s.apply(self.f, &m, Action::Scale { centre: c, factor: Vec3::new(0.88, 1.0, 1.0) });
        // The vermilion rolls outward around the contact line.
        let near = self.geo(&self.lip_line("upper", "inner")?, 0.0, mm(9.0)).union(&self.geo(
            &self.lip_line("lower", "inner")?,
            0.0,
            mm(9.0),
        ));
        let pivot = c - v_mm(0.0, 0.0, 4.0);
        s.apply(self.f, &near.scale(|n| self.upper(n)), Action::Rotate { pivot, axis: Vec3::X, angle: -0.18 });
        s.apply(self.f, &near.scale(|n| 1.0 - self.upper(n)), Action::Rotate { pivot, axis: Vec3::X, angle: 0.18 });
        s.relax(self.f, 3, 0.5, |n| self.f.part_of(n) == self.body);
        Ok(s)
    }

    /// AU23 / AU24: the red of the lips narrows (`squeeze` vertically) and presses back.
    fn lip_press(&self, squeeze: f32, back: f32, narrow: f32) -> ExprResult<Shape> {
        let c = self.mouth_centre()?;
        let m = self.lips_mask(1.0, 7.0)?;
        let mut s = self.shape();
        s.apply(self.f, &m, Action::Scale { centre: c, factor: Vec3::new(narrow, squeeze, 1.0) });
        s.apply(self.f, &m, Action::Translate { offset: v_mm(0.0, 0.0, -back) });
        s.relax(self.f, 2, 0.5, |n| self.f.part_of(n) == self.body);
        Ok(s)
    }

    /// AU25: the lips part; the corners stay together.
    fn au25(&self) -> ExprResult<Shape> {
        let mut s = self.shape();
        for (lip, dy, side_share) in [("upper", 2.4_f32, true), ("lower", -3.0, false)] {
            let m = self.geo(&self.lip_line(lip, "inner")?, mm(0.5), mm(11.0)).scale(|n| {
                let u = self.upper(n);
                (if side_share { u } else { 1.0 - u }) * self.off_corner(n, mm(10.0))
            });
            s.apply(self.f, &m, Action::Translate { offset: v_mm(0.0, dy, 0.0) });
        }
        s.relax(self.f, 4, 0.5, |n| self.f.part_of(n) == self.body);
        // The corners join the lips: smooth the opening into them.
        s.relax(self.f, 6, 0.5, |n| self.f.part_of(n) == self.body && self.corner_dist[n] < mm(10.0));
        Ok(s)
    }

    /// Curls one lip's red inward over the teeth by `amount` (1 = a full lip suck): the contact
    /// edge moves back, staying against the other lip, and the vermilion border rolls back toward
    /// the contact line so the red disappears; the corners stay.
    fn lip_roll(&self, upper_lip: bool, amount: f32) -> ExprResult<Shape> {
        let (lip, dir) = if upper_lip { ("upper", 1.0) } else { ("lower", -1.0) };
        let own = |n: usize| {
            let u = self.upper(n);
            (if upper_lip { u } else { 1.0 - u }) * self.off_corner(n, mm(7.0))
        };
        let edge = self.geo(&self.lip_line(lip, "inner")?, mm(0.5), mm(6.0)).scale(own);
        let border = self.geo(&self.lip_line(lip, "outer")?, mm(0.5), mm(5.0)).scale(own);
        let mut s = self.shape();
        s.apply(self.f, &edge, Action::Translate { offset: v_mm(0.0, -0.2 * dir * amount, -3.8 * amount) });
        s.apply(self.f, &border, Action::Translate { offset: v_mm(0.0, -1.8 * dir * amount, -2.6 * amount) });
        s.relax(self.f, 3, 0.5, |n| self.f.part_of(n) == self.body);
        Ok(s)
    }

    /// AU32: the lower lip is drawn up and back under the upper teeth.
    fn au32(&self) -> ExprResult<Shape> {
        let mut s = self.lip_roll(false, 0.8)?;
        let m = self
            .geo(&self.lip_line("lower", "outer")?, mm(1.0), mm(9.0))
            .scale(|n| (1.0 - self.upper(n)) * self.off_corner(n, mm(7.0)));
        s.apply(self.f, &m, Action::Translate { offset: v_mm(0.0, 1.5, 0.0) });
        Ok(s)
    }

    /// Additional control: mouth shrug: lower lip and chin push up, the upper lip follows.
    fn mouth_shrug(&self) -> ExprResult<Shape> {
        let lips = self.lips_mask(2.0, 10.0)?.scale(|n| self.off_corner(n, mm(6.0)));
        let mut s = self.shape();
        s.apply(self.f, &lips.scale(|n| 1.0 - self.upper(n)), Action::Translate { offset: v_mm(0.0, 2.4, 1.0) });
        s.apply(self.f, &lips.scale(|n| self.upper(n)), Action::Translate { offset: v_mm(0.0, 1.2, 0.4) });
        let chin = self.geo(&[self.node("chin.mid")?], mm(3.0), mm(16.0)).scale(|n| 1.0 - self.upper(n));
        s.apply(self.f, &chin, Action::Translate { offset: v_mm(0.0, 1.5, 0.5) });
        Ok(s)
    }

    // ---------------------------------------------------------------- jaw

    /// The authored soft-tissue share of a jaw rotation per node: a smooth blend across the
    /// boundary of the jaw's (and tongue's) skin weight, by geodesic distance to either side over
    /// [`JAW_BLEND`] (metric, so the share is smooth on the fine lip crests and mirror-symmetric
    /// on a symmetric face), the lip corners following half way, the skin under the chin
    /// following partly, and the upper lip and the face above the mouth staying with the skull.
    fn jaw_follow(&self) -> ExprResult<Vec<f32>> {
        let f = self.f;
        let (head, neck) = (f.region("head")?, f.region("neck")?);
        let movable: Vec<bool> = (0..f.node_count())
            .map(|n| {
                (f.part_of(n) == self.body && (f.in_region(n, head) || f.in_region(n, neck)))
                    || f.part_of(n) == self.helper
            })
            .collect();
        let side = |jaw: bool| -> Vec<u32> {
            (0..f.node_count())
                .filter(|&n| movable[n] && (self.jaw_family[n] >= 0.5) == jaw)
                .filter_map(|n| u32::try_from(n).ok())
                .collect()
        };
        let reach = 4.0 * JAW_BLEND;
        let (dj, dh) = (f.distances(&side(true), reach), f.distances(&side(false), reach));
        let mut w: Vec<f32> = (0..f.node_count())
            .map(|n| {
                let own = self.jaw_family[n];
                if !movable[n] {
                    return own;
                }
                match (dj[n].is_finite(), dh[n].is_finite()) {
                    (true, true) => {
                        let t = (0.5 + 0.5 * (dh[n] - dj[n]) / JAW_BLEND).clamp(0.0, 1.0);
                        t * t * (3.0 - 2.0 * t)
                    }
                    (true, false) => 1.0,
                    (false, true) => 0.0,
                    (false, false) => own,
                }
            })
            .collect();
        let menton = f.distances(&[self.node("chin.menton")?], 0.06);
        for n in 0..f.node_count() {
            if !movable[n] {
                continue;
            }
            let c = falloff(self.corner_dist[n], mm(2.0), mm(14.0));
            w[n] += (0.5 - w[n]) * c;
            if f.in_region(n, neck) {
                w[n] = w[n].max(0.4 * falloff(menton[n], mm(6.0), mm(45.0)));
            }
            let above = self.upper(n) * falloff(self.lip_dist[n], mm(8.0), mm(14.0));
            w[n] *= 1.0 - above * (1.0 - c);
        }
        Ok(w)
    }

    /// The rest-space support shape of a rotation of `joint` by `angle` about the world axis `axis`:
    /// for every skin and mouth-helper node, the displacement that the skinning (LBS or DQS by region, as the
    /// evaluator does) carries onto the authored result, a rotation by `angle · follow[n]` about
    /// the joint's rest position.
    ///
    /// `extra` is authored soft-tissue motion added to the rotated surface (the lips' own
    /// behaviour when the jaw opens) before the surface is smoothed and regularised, so the folds
    /// it would make together with the rotation are repaired with it.
    pub fn rotation_support(
        &self,
        joint: &str,
        axis: Vec3,
        angle: f32,
        follow: &[f32],
        extra: Option<&Shape>,
    ) -> ExprResult<Shape> {
        let f = self.f;
        let t = f.template();
        let sk = t.skeleton();
        let j = f.joint_index(joint)? as usize;
        let rest = f.rest_world();
        let count = sk.joints.len();
        let mut frames = JointFrames::new(count);
        for (k, tr) in rest.iter().enumerate().take(count) {
            frames.bind_positions[k] = tr.translation;
        }
        let parents: Vec<Option<u16>> = sk.joints.iter().map(|j| j.parent).collect();
        let bind: Vec<Quat> = rest.iter().map(|t| t.rotation).collect();
        let mut pose = vec![Quat::IDENTITY; count];
        let world = Quat::from_axis_angle(axis.normalize_or_zero(), angle);
        if let (Some(p), Some(&rb)) = (pose.get_mut(j), bind.get(j)) {
            *p = (rb.conjugate() * world * rb).normalize();
        }
        frames.pose(&parents, &bind, &pose);
        let pivot = rest.get(j).map_or(Vec3::ZERO, |r| r.translation);
        let regions = t.regions();
        let dq_regions = &t.skin().dual_quaternion_regions;
        let h = 0.01;
        let axis = axis.normalize_or_zero();
        // Per skinned node: its skinning as an affine map (finite differences; exact for LBS and
        // DQS, which are affine per vertex for a fixed pose) and the authored target position.
        let mut maps: Vec<Option<(Vec3, Mat3)>> = vec![None; f.node_count()];
        let mut want = self.shape();
        for (n, map) in maps.iter_mut().enumerate() {
            let v = f.representatives()[n] as usize;
            let (js, ws) = t.skin().vertex(v);
            if js.is_empty() {
                continue;
            }
            let r = regions.vertex_regions.get(v).copied().unwrap_or(0);
            let dq = dq_regions.iter().any(|&d| regions.is_within(r, d));
            let skin = |p: Vec3| if dq { frames.skin_dual_quat(p, js, ws) } else { frames.skin_linear(p, js, ws) };
            let p = f.positions()[n];
            let base = skin(p);
            let m = Mat3::from_cols(
                (skin(p + Vec3::X * h) - base) / h,
                (skin(p + Vec3::Y * h) - base) / h,
                (skin(p + Vec3::Z * h) - base) / h,
            );
            let (target, added) = if f.part_of(n) == self.body || f.part_of(n) == self.helper {
                *map = Some((base, m));
                let w = follow.get(n).copied().unwrap_or(0.0);
                let added = extra.and_then(|e| e.displacements().get(n).copied()).unwrap_or(Vec3::ZERO);
                (pivot + Quat::from_axis_angle(axis, angle * w) * (p - pivot), added)
            } else {
                (base, Vec3::ZERO)
            };
            if let Some(x) = want.displacements_mut().get_mut(n) {
                *x = target + added - p;
            }
        }
        // The lip corners join a moving and a still lip: spread their motion evenly; then repair
        // any fold of the authored surface before deriving the residual.
        want.relax(self.f, 8, 0.5, |n| maps[n].is_some() && self.corner_dist[n] < mm(12.0));
        want.regularize(self.f, REGULARIZE_ROUNDS, &self.bounds, |n| maps[n].is_some());
        self.ride_lips(&mut want);
        let mut s = self.shape();
        for (n, map) in maps.iter().enumerate() {
            let Some((base, m)) = *map else { continue };
            let Some(inv) = m.inverse() else { continue };
            let p = f.positions()[n];
            let d = inv * (p + want.displacements()[n] - base);
            if d.is_finite()
                && let Some(x) = s.displacements_mut().get_mut(n)
            {
                *x = d;
            }
        }
        Ok(s)
    }

    /// AU26 / AU27 / visemes: the support shape of a jaw opening by `angle`, plus authored lip
    /// behaviour: the upper lip stretches down a little at its centre, the corners move in by
    /// `narrow` metres.
    fn jaw_open(&self, follow: &[f32], angle: f32, narrow: f32) -> ExprResult<Shape> {
        let mut lips = self.shape();
        let upper = self
            .geo(&self.lip_line("upper", "inner")?, mm(1.0), mm(10.0))
            .scale(|n| self.upper(n) * self.off_corner(n, mm(10.0)));
        lips.apply(self.f, &upper, Action::Translate { offset: v_mm(0.0, -0.8 * angle / JAW_DROP, 0.0) });
        if narrow != 0.0 {
            let m = self.geo(&self.corner_seeds()?, mm(2.0), mm(16.0));
            self.sided(&mut lips, &m, |side| Action::Translate { offset: Vec3::new(-narrow * sign(side), 0.0, 0.0) });
        }
        self.rotation_support("jaw", Vec3::X, angle, follow, Some(&lips))
    }

    /// AU29: the jaw pushes forward (no joint translation exists: a pure shape over the jaw's skin
    /// share, teeth, gums and tongue included).
    fn au29(&self) -> Shape {
        let m = Mask::from_fn(self.f, |n| self.jaw_family.get(n).copied().unwrap_or(0.0));
        let mut s = self.shape();
        s.apply(self.f, &m, Action::Translate { offset: v_mm(0.0, 0.4, 4.5) });
        s.relax(self.f, 4, 0.5, |n| self.f.part_of(n) == self.body);
        s
    }

    // ---------------------------------------------------------------- tongue

    /// The tongue nodes with their share along the tongue (0 at the root, 1 at the tip).
    fn tongue_mask(&self, from: f32) -> ExprResult<Mask> {
        let part = self.f.part("tongue")?;
        let zs: Vec<f32> =
            (0..self.f.node_count()).filter(|&n| self.f.part_of(n) == part).map(|n| self.f.positions()[n].z).collect();
        let (lo, hi) = zs.iter().fold((f32::MAX, f32::MIN), |(a, b), &z| (a.min(z), b.max(z)));
        if hi <= lo {
            return Err(ExprError::missing("mesh part nodes", "tongue"));
        }
        Ok(Mask::from_fn(self.f, |n| {
            if self.f.part_of(n) != part {
                return 0.0;
            }
            let t = (self.f.positions()[n].z - lo) / (hi - lo);
            ((t - from) / (1.0 - from)).clamp(0.0, 1.0)
        }))
    }

    /// The front of the tongue scaled across (`width`) and in height (`height`) about its own
    /// centre line.
    fn tongue_scale(&self, width: f32, height: f32) -> ExprResult<Shape> {
        let m = self.tongue_mask(0.3)?;
        let (mut sum, mut wsum) = (Vec3::ZERO, 0.0);
        for &(n, w) in m.entries() {
            sum += self.f.positions()[n as usize] * w;
            wsum += w;
        }
        let c = if wsum > 0.0 { sum / wsum } else { Vec3::ZERO };
        let mut s = self.shape();
        s.apply(self.f, &m, Action::Scale { centre: c, factor: Vec3::new(width, height, 1.0) });
        Ok(s)
    }

    /// Additional control: tongue out: the tongue extends forward, narrowing and thinning.
    fn tongue_out(&self) -> ExprResult<Shape> {
        let mut s = self.tongue_scale(0.85, 0.85)?;
        let m = self.tongue_mask(0.0)?;
        s.apply(self.f, &m, Action::Translate { offset: v_mm(0.0, -2.0, 26.0) });
        Ok(s)
    }

    // ---------------------------------------------------------------- the whole set

    /// The lip line (mouth helper) rides the upper lip: every helper node takes the displacement
    /// of its upper-lip anchor. Left to the masks (which never reach it) or to the jaw's skin
    /// share, it stayed between the parting lips and stretched into a sheet across the open
    /// mouth.
    fn ride_lips(&self, shape: &mut Shape) {
        for (n, anchors) in &self.helper_anchor {
            let d = anchors.iter().fold(Vec3::ZERO, |sum, &(a, w)| {
                sum + shape.displacements().get(a as usize).copied().unwrap_or(Vec3::ZERO) * w
            });
            if let Some(x) = shape.displacements_mut().get_mut(*n as usize) {
                *x = d;
            }
        }
    }

    /// Adds a plain target after regularisation ([`Shape::regularize`] over the parts it moves).
    fn push(&self, out: &mut Vec<Target>, name: &str, kind: ExpressionKind, side: Side, facs: u16, shape: Shape) {
        let mut shape = shape;
        self.regularize(&mut shape);
        out.push(Target { name: name.to_owned(), kind, side, facs, shape });
    }

    /// Regularises `shape` against folds and over-stretch, moving only nodes of the parts it
    /// already moves.
    fn regularize(&self, shape: &mut Shape) {
        let mut parts: Vec<u16> = Vec::new();
        for (n, d) in shape.displacements().iter().enumerate() {
            let p = self.f.part_of(n);
            if *d != Vec3::ZERO && !parts.contains(&p) {
                parts.push(p);
            }
        }
        shape.regularize(self.f, REGULARIZE_ROUNDS, &self.bounds, |n| parts.contains(&self.f.part_of(n)));
        self.ride_lips(shape);
    }

    /// Adds a target that is not a surface of its own (an in-between difference, a rest-space
    /// support residual): no fold repair, whose rest-space test would be meaningless.
    fn push_raw(out: &mut Vec<Target>, name: &str, kind: ExpressionKind, side: Side, facs: u16, shape: Shape) {
        out.push(Target { name: name.to_owned(), kind, side, facs, shape });
    }

    fn pair(&self, out: &mut Vec<Target>, name: &str, kind: ExpressionKind, facs: u16, bilateral: &Shape) {
        for side in [Side::Left, Side::Right] {
            let s = self.split(bilateral, side);
            self.push(out, &format!("{name}-{}", letter(side)), kind, side, facs, s);
        }
    }

    fn per_side(
        &self,
        out: &mut Vec<Target>,
        name: &str,
        kind: ExpressionKind,
        facs: u16,
        shape: impl Fn(Side) -> ExprResult<Shape>,
    ) -> ExprResult<()> {
        for side in [Side::Left, Side::Right] {
            let s = shape(side)?;
            self.push(out, &format!("{name}-{}", letter(side)), kind, side, facs, s);
        }
        Ok(())
    }

    /// Every target: the 82 registry targets (Appendix B) in registry order with the eye-closure
    /// in-betweens after AU46, then the viseme supports.
    pub fn all(&self) -> ExprResult<Vec<Target>> {
        use ExpressionKind::{ActionUnit as Au, Tongue, Viseme};
        let c = Side::Center;
        let mut out = Vec::with_capacity(96);
        let (au04, au06) = (self.au04()?, self.au06()?);
        self.pair(&mut out, "AU01", Au, 1, &self.au01()?);
        self.pair(&mut out, "AU02", Au, 2, &self.au02()?);
        self.pair(&mut out, "AU04", Au, 4, &au04);
        self.per_side(&mut out, "AU05", Au, 5, |s| Ok(self.au05(s)))?;
        self.pair(&mut out, "AU06", Au, 6, &au06);
        self.per_side(&mut out, "AU07", Au, 7, |s| Ok(self.au07(s)))?;
        self.pair(&mut out, "AU09", Au, 9, &self.au09()?);
        self.pair(&mut out, "AU10", Au, 10, &self.au10()?);
        self.pair(&mut out, "AU11", Au, 11, &self.au11()?);
        self.pair(&mut out, "AU12", Au, 12, &self.au12()?);
        self.pair(&mut out, "AU13", Au, 13, &self.au13()?);
        self.pair(&mut out, "AU14", Au, 14, &self.au14()?);
        self.pair(&mut out, "AU15", Au, 15, &self.corners(0.0, 22.0, |s| v_mm(0.8 * sign(s), -4.5, -0.4))?);
        self.push(&mut out, "AU16", Au, c, 16, self.au16()?);
        self.push(&mut out, "AU17", Au, c, 17, self.au17()?);
        self.push(&mut out, "AU18", Au, c, 18, self.au18()?);
        self.pair(&mut out, "AU20", Au, 20, &self.corners(0.0, 28.0, |s| v_mm(5.5 * sign(s), -0.8, -1.8))?);
        self.push(&mut out, "AU22", Au, c, 22, self.au22()?);
        self.push(&mut out, "AU23", Au, c, 23, self.lip_press(0.62, 0.6, 0.97)?);
        self.push(&mut out, "AU24", Au, c, 24, self.lip_press(0.72, 1.6, 1.0)?);
        self.push(&mut out, "AU25", Au, c, 25, self.au25()?);
        let follow = self.jaw_follow()?;
        Self::push_raw(&mut out, "AU26", Au, c, 26, self.jaw_open(&follow, JAW_DROP, 0.0)?);
        Self::push_raw(&mut out, "AU27", Au, c, 27, self.jaw_open(&follow, MOUTH_STRETCH, -mm(1.5))?);
        let mut au28 = self.lip_roll(true, 1.0)?;
        au28.add(&self.lip_roll(false, 1.0)?, 1.0);
        self.push(&mut out, "AU28", Au, c, 28, au28);
        self.push(&mut out, "AU29", Au, c, 29, self.au29());
        for side in [Side::Left, Side::Right] {
            let s = self.rotation_support("jaw", Vec3::Y, JAW_SIDE * sign(side), &follow, None)?;
            Self::push_raw(&mut out, &format!("AU30-{}", letter(side)), Au, side, 30, s);
        }
        self.push(&mut out, "AU31", Au, c, 31, self.au31()?);
        self.push(&mut out, "AU32", Au, c, 32, self.au32()?);
        self.pair(&mut out, "AU33", Au, 33, &self.cheeks(5.5, 6.0, 34.0)?);
        self.pair(&mut out, "AU34", Au, 34, &self.cheeks(3.8, 5.0, 30.0)?);
        self.pair(&mut out, "AU35", Au, 35, &self.cheeks(-3.5, 3.0, 22.0)?);
        self.pair(&mut out, "AU38", Au, 38, &self.nostrils(1.4, 0.0, 7.0)?);
        self.pair(&mut out, "AU39", Au, 39, &self.nostrils(-1.1, -0.3, 7.0)?);
        self.per_side(&mut out, "AU41", Au, 41, |s| Ok(self.au41(s)))?;
        self.per_side(&mut out, "AU42", Au, 42, |s| Ok(self.au42(s)))?;
        self.per_side(&mut out, "AU43", Au, 43, |s| Ok(self.closed(s)))?;
        self.per_side(&mut out, "AU44", Au, 44, |s| self.au44(s))?;
        let mut blink = self.closed(Side::Left);
        blink.add(&self.closed(Side::Right), 1.0);
        self.push(&mut out, "AU45", Au, c, 45, blink);
        self.per_side(&mut out, "AU46", Au, 46, |s| Ok(self.au46(s, &au06, &au04)))?;
        for side in [Side::Left, Side::Right] {
            let ib = self.closed_inbetween(side);
            Self::push_raw(&mut out, &format!("AU43-{}.inbetween-50", letter(side)), Au, side, 43, ib.clone());
            Self::push_raw(&mut out, &format!("AU46-{}.inbetween-50", letter(side)), Au, side, 46, ib);
        }
        let mut ib = self.closed_inbetween(Side::Left);
        ib.add(&self.closed_inbetween(Side::Right), 1.0);
        Self::push_raw(&mut out, "AU45.inbetween-50", Au, c, 45, ib);
        self.push(&mut out, "ctl.tongue-out", Tongue, c, 0, self.tongue_out()?);
        self.push(&mut out, "ctl.tongue-up", Tongue, c, 0, self.tongue_scale(0.9, 1.1)?);
        self.push(&mut out, "ctl.tongue-down", Tongue, c, 0, self.tongue_scale(1.08, 0.9)?);
        self.push(&mut out, "ctl.tongue-curl", Tongue, c, 0, self.tongue_scale(0.85, 1.0)?);
        self.per_side(&mut out, "ctl.tongue-side", Tongue, 0, |_| self.tongue_scale(0.94, 1.04))?;
        self.push(&mut out, "ctl.lip-roll-upper", Au, c, 0, self.lip_roll(true, 0.6)?);
        self.push(&mut out, "ctl.lip-roll-lower", Au, c, 0, self.lip_roll(false, 0.6)?);
        self.push(&mut out, "ctl.mouth-shrug", Au, c, 0, self.mouth_shrug()?);
        self.pair(&mut out, "ctl.corner-narrow", Au, 0, &self.corners(0.0, 18.0, |s| v_mm(-3.8 * sign(s), 0.0, 1.0))?);
        self.pair(&mut out, "ctl.corner-wide", Au, 0, &self.corners(0.0, 20.0, |s| v_mm(3.8 * sign(s), 0.0, -1.0))?);
        self.pair(&mut out, "ctl.nostril-flare", Au, 0, &self.nostrils(2.2, 0.8, 9.0)?);
        self.push(&mut out, "ctl.brow-ridge-compress", Au, c, 0, self.brow_ridge_compress()?);
        self.per_side(&mut out, "ctl.eyelid-crease", Au, 0, |s| Ok(self.crease(s, mm(1.0))))?;
        // Viseme supports `[EXP-01]`: lip closure, funnel, pucker, stretch, jaw-open variants.
        let mut closure = self.shape();
        let lips = self.lips_mask(1.0, 9.0)?.scale(|n| self.off_corner(n, mm(6.0)));
        closure.apply(self.f, &lips.scale(|n| 1.0 - self.upper(n)), Action::Translate { offset: v_mm(0.0, 2.2, 0.3) });
        closure.apply(self.f, &lips.scale(|n| self.upper(n)), Action::Translate { offset: v_mm(0.0, -0.8, 0.0) });
        self.push(&mut out, "viseme.lip-closure", Viseme, c, 0, closure);
        let mut funnel = self.au22()?;
        funnel.add(&self.au25()?, 0.4);
        self.push(&mut out, "viseme.funnel", Viseme, c, 0, funnel);
        let mut pucker = self.au18()?;
        pucker.add(&self.lip_press(0.72, 1.6, 1.0)?, 0.3);
        self.push(&mut out, "viseme.pucker", Viseme, c, 0, pucker);
        let mut stretch = self.corners(0.0, 28.0, |s| v_mm(4.4 * sign(s), -0.6, -1.4))?;
        stretch.add(&self.au25()?, 0.3);
        self.push(&mut out, "viseme.stretch", Viseme, c, 0, stretch);
        Self::push_raw(&mut out, "viseme.jaw-open-round", Viseme, c, 0, self.jaw_open(&follow, VISEME_ROUND, mm(3.0))?);
        Self::push_raw(&mut out, "viseme.jaw-open-wide", Viseme, c, 0, self.jaw_open(&follow, VISEME_WIDE, -mm(1.5))?);
        self.push(&mut out, "viseme.labiodental", Viseme, c, 0, self.lip_roll(false, 0.7)?);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::integration::tests::expressive;

    /// Diagnosis: a target's raw recipe shape and its regularised result: bad triangles (folds,
    /// area ratios outside the target bounds) and the displacement (mm) of the nodes of the
    /// vertices in `DIAG_V`. `DIAG_T` names the recipe (au25, au12, au27).
    #[test]
    #[ignore = "diagnostic: DIAG_T=au25 DIAG_V=1,2 cargo test diagnose_recipe -- --ignored --nocapture"]
    fn diagnose_recipe() {
        let t = expressive();
        let f = FaceField::new(&t).unwrap();
        let r = Recipes::new(&f).unwrap();
        let verts: Vec<usize> =
            std::env::var("DIAG_V").unwrap_or_default().split(',').filter_map(|s| s.trim().parse().ok()).collect();
        let report = |label: &str, s: &Shape| {
            let bad = s.bad_triangles(&f, &r.bounds);
            println!("{label}: {} bad, {} inverted", bad.len(), s.inverted(&f).len());
            let mc = r.mouth_centre().unwrap();
            for &k in bad.iter().take(30) {
                let tri = f.triangles()[k];
                let p = |n: u32| f.positions()[n as usize];
                let q = |n: u32| p(n) + s.displacements()[n as usize];
                let n0 = (p(tri[1]) - p(tri[0])).cross(p(tri[2]) - p(tri[0]));
                let n1 = (q(tri[1]) - q(tri[0])).cross(q(tri[2]) - q(tri[0]));
                let c = ((p(tri[0]) + p(tri[1]) + p(tri[2])) / 3.0 - mc) * 1e3;
                let reg = f.region_of(tri[0] as usize);
                let rname = t.regions().regions.get(reg as usize).map_or("?", |x| x.name.as_str());
                let d: Vec<String> = tri
                    .iter()
                    .map(|&n| {
                        let d = s.displacements()[n as usize] * 1e3;
                        format!("({:.2},{:.2},{:.2})", d.x, d.y, d.z)
                    })
                    .collect();
                println!(
                    "  {} {rname} at ({:.1},{:.1},{:.1}) a0 {:.4} ratio {:.3} n0 ({:.2},{:.2},{:.2}) nodes {tri:?} d {}",
                    if n0.dot(n1) < 0.0 { "INV" } else { "   " },
                    c.x,
                    c.y,
                    c.z,
                    0.5 * n0.length() * 1e6,
                    n1.length() / n0.length().max(1e-20),
                    n0.normalize_or_zero().x,
                    n0.normalize_or_zero().y,
                    n0.normalize_or_zero().z,
                    d.join(" ")
                );
            }
            for &v in &verts {
                let n = f.node_of(v) as usize;
                let d = s.displacements()[n] * 1e3;
                println!(
                    "  v{v} n{n} ({:.3},{:.3},{:.3}) lipdist {:.2} upper {:.2}",
                    d.x,
                    d.y,
                    d.z,
                    r.lip_dist[n] * 1e3,
                    r.upper(n)
                );
            }
        };
        let which = std::env::var("DIAG_T").unwrap_or_default();
        if let Some(name) = which.strip_prefix("all:") {
            let all = r.all().unwrap();
            for tg in all.iter().filter(|x| name.split('+').any(|n| n == x.name)) {
                report(&tg.name, &tg.shape);
                let d = t.deltas();
                let c = d.find(&tg.name).unwrap() as usize;
                for &v in &verts {
                    let stored = d
                        .indices(c)
                        .iter()
                        .position(|&x| x as usize == v)
                        .map_or(Vec3::ZERO, |i| crate::delta::DeltaBases::delta_f32(d.deltas(c)[i]));
                    let mine = tg.shape.displacements()[f.node_of(v) as usize];
                    println!("  v{v}: stored {:?} recipe {:?}", stored * 1e3, mine * 1e3);
                }
            }
            return;
        }
        let mut s = match which.as_str() {
            "au25" => r.au25().unwrap(),
            "au12" => r.au12().unwrap(),
            "au27" => {
                let follow = r.jaw_follow().unwrap();
                r.jaw_open(&follow, MOUTH_STRETCH, -mm(1.5)).unwrap()
            }
            "au44l" => r.au44(Side::Left).unwrap(),
            "au44r" => r.au44(Side::Right).unwrap(),
            "lids42" | "closed" => {
                let (up, low) = if which == "closed" { (0.84, 0.14) } else { (0.42, 0.07) };
                let mut s = r.lids(Side::Left, up, low);
                report("raw lids", &s);
                let left = s.repair_folds(&f, REGULARIZE_ROUNDS);
                report(&format!("repaired ({left} left)"), &s);
                return;
            }
            _ => return,
        };
        report("raw", &s);
        let mut parts: Vec<u16> = Vec::new();
        for (n, d) in s.displacements().iter().enumerate() {
            if *d != Vec3::ZERO && !parts.contains(&f.part_of(n)) {
                parts.push(f.part_of(n));
            }
        }
        let left = s.regularize(&f, REGULARIZE_ROUNDS, &r.bounds, |n| parts.contains(&f.part_of(n)));
        report(&format!("regularised ({left} left)"), &s);
    }
}
