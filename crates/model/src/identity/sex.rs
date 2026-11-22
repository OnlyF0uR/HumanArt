//! The Female / Male switch `identity.sex` (todo 1.3, ADR-039, ADR-046): one component,
//! [`SWITCH`], whose weight −1 is the female and +1 the male character. The registry default is
//! −1; the template's rest shape (weight 0) is the sex midpoint and no character of its own.
//!
//! **What the component holds**, per unit weight (half the male-minus-female difference):
//!
//! - the **body** toward the ANSUR II women's (−1) and men's (+1) means
//!   (`data/anthropometry/ansur2.json`): stature, biacromial breadth, chest breadth, waist girth,
//!   hip breadth, buttock depth, thigh girth, neck girth, upper-arm girth (flexed biceps,
//!   relative), head length and menton-sellion length, each by its own term, the amounts solved
//!   together ([`fit`]) so that the half difference of each measurement between −1 and +1 is half
//!   the difference of the means;
//! - what ANSUR II does not measure, **authored** (not statistics): the breasts (a mound of
//!   [`BREAST_HEIGHT`] with a slight ptosis, half of it baked into the rest shape by
//!   [`rest_shift`], so the female has all of it and the male none), the larynx (likewise, male),
//!   the limb and pectoral muscles, the brow ridge, the jaw and chin, the nose, the lips and the
//!   areolae (larger for the female);
//! - the **genital variant** (ADR-039: discrete, never blended): both variants sit in the rest
//!   shape halfway between their two states (`generate::genitals::male_rest`, `vulva_rest`); the
//!   component moves each by half the way, so at −1 the vulva shows and the male piece is kept
//!   shrunk inside the pelvis, at +1 the male piece shows and the vulva lies folded under it.
//!
//! The body part of the component moves the groin with the hips and thighs; the vulva moves with
//! it and the male piece follows the skin node nearest to where it belongs (a helper of
//! [`Space`]), so its border stays on the attachment loop. `identity.sex_shape`
//! (`global.sex`) stays the continuous variation within a sex.

use humanart_math::Vec3;

use super::body::{self, SOFT_TISSUE};
use super::error::{IdResult, IdentityError};
use super::face;
use super::measure::{
    BIACROMIAL, BIACROMIAL_GAUGE, CHEST_CIRC, CHEST_GIRTH, Gauge, HIP_BREADTH, HIP_BREADTH_GAUGE, HIPS_AND_THIGHS,
    Level, NECK_CIRC, NECK_GIRTH, STATURE, THIGH_CIRC, THIGH_GIRTH, Variable, WAIST_CIRC, WAIST_GIRTH,
};
use super::space::{IdentityComponent, Space, Symmetry, smoothstep};
use super::{Kit, Raw};
use crate::generate::genitals;

/// The component the switch binds to (`identity.sex`).
pub const SWITCH: &str = "global.sex-switch";

/// Height of the female breast mound (the `semantic.bust-size` shape), metres: authored so that
/// the female chest circumference at the bust meets the ANSUR II women's mean (test
/// `todo_1_3_sex_switch_meets_the_ansur_means`). No breast variable exists in ANSUR II.
pub const BREAST_HEIGHT: f32 = 0.030;
/// The breast's ptosis (the `semantic.breast-ptosis` shape) with the full mound, metres: a
/// moderate, slightly ptotic adult form, not a hemisphere (authored).
pub const BREAST_PTOSIS: f32 = 0.007;
/// The male larynx prominence, metres along the normal (authored).
pub const LARYNX: f32 = 0.003;

const fn var(id: &'static str, male_mean: f32, male_sd: f32, female_mean: f32, female_sd: f32) -> Variable {
    Variable { id, male_mean, male_sd, female_mean, female_sd }
}

/// `chestbreadth`: breadth of the chest at the nipples (women: the bust), mm.
pub const CHEST_BREADTH: Variable = var("chestbreadth", 289.4, 18.3, 269.3, 18.6);
/// `buttockdepth`: depth of the trunk at the buttocks, mm.
pub const BUTTOCK_DEPTH: Variable = var("buttockdepth", 245.8, 26.5, 232.9, 24.3);
/// `bicepscircumferenceflexed`: upper-arm circumference over the flexed biceps, mm.
pub const BICEPS_CIRC: Variable = var("bicepscircumferenceflexed", 358.1, 34.6, 305.6, 30.8);
/// `headlength`: glabella to opisthocranion, mm.
pub const HEAD_LENGTH: Variable = var("headlength", 199.5, 7.0, 189.8, 7.4);
/// `mentonsellionlength`: sellion (nasal root) to menton, mm.
pub const MENTON_SELLION: Variable = var("mentonsellionlength", 122.6, 6.7, 113.1, 6.1);
/// `bizygomaticbreadth`: breadth across the zygomatic arches, mm (reported, not fitted).
pub const BIZYGOMATIC: Variable = var("bizygomaticbreadth", 142.6, 6.2, 133.8, 5.6);
/// `chestdepth`: depth of the chest at the nipples (women: the bust), mm (reported).
pub const CHEST_DEPTH: Variable = var("chestdepth", 253.8, 26.2, 247.4, 27.3);

/// The variables of this module (checked against the data file by a test).
pub const VARIABLES: &[Variable] =
    &[CHEST_BREADTH, BUTTOCK_DEPTH, BICEPS_CIRC, HEAD_LENGTH, MENTON_SELLION, BIZYGOMATIC, CHEST_DEPTH];

/// The chest band at the nipples.
const NIPPLE_LO: Level = Level("L.chest.nipple", "torso.xiphoid", 0.15);
const NIPPLE_HI: Level = Level("L.chest.nipple", "torso.suprasternal", 0.1);
/// Chest breadth at the nipples (chest and back only: the A-pose arms stay out).
pub const CHEST_BREADTH_GAUGE: Gauge = Gauge::Breadth(&["chest", "back"], NIPPLE_LO, NIPPLE_HI);
/// Chest depth at the nipples.
pub const CHEST_DEPTH_GAUGE: Gauge = Gauge::Depth(&["chest", "back"], NIPPLE_LO, NIPPLE_HI);
/// Buttock depth (as `semantic.glute-size` is calibrated).
pub const BUTTOCK_DEPTH_GAUGE: Gauge = Gauge::Depth(
    HIPS_AND_THIGHS,
    Level("L.hip.trochanter", "L.knee.center", 0.1),
    Level("L.hip.trochanter", "L.hip.asis", 0.5),
);
/// Left upper-arm girth halfway down the upper arm (a slanted section of the A-pose arm: used
/// for relative change only).
pub const ARM_GIRTH: Gauge = Gauge::Girth(&["upper-arm"], Level("L.shoulder.acromion", "L.elbow.lateral", 0.5), true);
/// Head length.
pub const HEAD_LENGTH_GAUGE: Gauge = Gauge::Path(&["forehead.glabella", "head.opisthocranion"]);
/// Menton-sellion length.
pub const FACE_HEIGHT_GAUGE: Gauge = Gauge::Path(&["nose.root", "chin.menton"]);
/// Bizygomatic breadth.
pub const BIZYGOMATIC_GAUGE: Gauge = Gauge::Path(&["L.zygion", "R.zygion"]);

/// How a fitted measurement's target is set.
#[derive(Clone, Copy, Debug)]
pub enum Target {
    /// Female and male at the ANSUR II means: half the difference of the means.
    Absolute(Variable),
    /// Female and male at the template's value times the ratio of the means to their midpoint
    /// (for gauges that are not taken as the ANSUR II measurer takes them).
    Relative(Variable),
    /// Female and male at the template's value minus and plus half the difference of the means
    /// (for gauges whose rest value also holds the half breasts of the rest shape).
    Difference(Variable),
}

impl Target {
    /// The variable.
    #[must_use]
    pub const fn variable(self) -> Variable {
        match self {
            Self::Absolute(v) | Self::Relative(v) | Self::Difference(v) => v,
        }
    }
}

/// The fitted measurements, in the order of the terms of [`terms`].
pub const FITTED: [(Gauge, Target); 11] = [
    (Gauge::Stature, Target::Absolute(STATURE)),
    (BIACROMIAL_GAUGE, Target::Absolute(BIACROMIAL)),
    (CHEST_BREADTH_GAUGE, Target::Relative(CHEST_BREADTH)),
    (CHEST_DEPTH_GAUGE, Target::Difference(CHEST_DEPTH)),
    (WAIST_GIRTH, Target::Absolute(WAIST_CIRC)),
    (HIP_BREADTH_GAUGE, Target::Absolute(HIP_BREADTH)),
    (BUTTOCK_DEPTH_GAUGE, Target::Relative(BUTTOCK_DEPTH)),
    (THIGH_GIRTH, Target::Relative(THIGH_CIRC)),
    (NECK_GIRTH, Target::Relative(NECK_CIRC)),
    (HEAD_LENGTH_GAUGE, Target::Absolute(HEAD_LENGTH)),
    (FACE_HEIGHT_GAUGE, Target::Relative(MENTON_SELLION)),
];

/// The measurements reported beside the fitted ones (tests, STATUS).
pub const REPORTED: [(Gauge, Variable); 2] = [(CHEST_GIRTH, CHEST_CIRC), (BIZYGOMATIC_GAUGE, BIZYGOMATIC)];
/// Reported as relative changes (the gauge is not taken as the ANSUR II measurer takes it).
pub const REPORTED_RELATIVE: [(Gauge, Variable); 1] = [(ARM_GIRTH, BICEPS_CIRC)];

/// Ribcage scaling `k` (x and z about the chest's centre) in a band around the nipples, fading
/// toward the waist and the neck; the arms stay out.
fn chest_scale(kit: &Kit<'_, '_>, k: f32) -> IdResult<Raw> {
    let s = kit.space;
    let nipple = s.landmark("L.chest.nipple")?;
    let xiphoid = s.landmark("torso.xiphoid")?;
    let notch = s.landmark("torso.suprasternal")?;
    let t8 = s.landmark("spine.t8")?;
    let z_c = f32::midpoint(nipple.z, t8.z);
    let (lo, hi) = (xiphoid.y - 0.06, notch.y - 0.01);
    let band = move |y: f32| smoothstep(lo, nipple.y, y) * (1.0 - smoothstep(nipple.y + 0.04, hi, y));
    let disp = s.eval(|pt| Vec3::new(k * pt.p.x, 0.0, k * (pt.p.z - z_c)) * band(pt.p.y));
    Raw::confined(s, body::WAIST, Symmetry::Mirror, disp, Vec::new())
}

/// Ribcage depth `k` (z about the chest's centre, the front more than the back) in the band of
/// [`chest_scale`].
fn chest_depth(kit: &Kit<'_, '_>, k: f32) -> IdResult<Raw> {
    let s = kit.space;
    let nipple = s.landmark("L.chest.nipple")?;
    let xiphoid = s.landmark("torso.xiphoid")?;
    let notch = s.landmark("torso.suprasternal")?;
    let t8 = s.landmark("spine.t8")?;
    let z_c = 0.7 * t8.z + 0.3 * nipple.z;
    let (lo, hi) = (xiphoid.y - 0.06, notch.y - 0.01);
    let band = move |y: f32| smoothstep(lo, nipple.y, y) * (1.0 - smoothstep(nipple.y + 0.04, hi, y));
    let disp = s.eval(|pt| Vec3::Z * (k * (pt.p.z - z_c) * band(pt.p.y)));
    Raw::confined(s, body::WAIST, Symmetry::Mirror, disp, Vec::new())
}

/// The head (skeletal) scaled by `k`: head length and the face with it.
fn head_scale(kit: &Kit<'_, '_>, k: f32) -> IdResult<Raw> {
    let (s, b) = (kit.space, &kit.bones);
    body::skeleton_raw(kit, false, |st| st.scale_subtree(b, s.joint_index("head")?, k))
}

/// The fitted terms, in the order of [`FITTED`], each at an authored unit amount.
fn terms(kit: &Kit<'_, '_>) -> IdResult<Vec<Raw>> {
    Ok(vec![
        body::height(kit)?,
        body::shoulders(kit, 0.04, 0.01)?,
        chest_scale(kit, 0.04)?,
        chest_depth(kit, 0.04)?,
        body::waist(kit, 0.04)?,
        body::hips(kit, 0.03)?,
        body::glutes(kit, 0.008)?,
        body::thighs(kit, 0.006)?,
        face::neck_width(kit)?,
        head_scale(kit, 0.02)?,
        face::chin_height(kit)?,
    ])
}

/// The full female breast (mound and ptosis), per unit weight.
fn breast(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let mut raw = body::breasts(kit, BREAST_HEIGHT)?;
    raw.add(&body::breast_shape(kit, body::BreastShape::Ptosis, BREAST_PTOSIS)?, 1.0);
    Ok(raw)
}

/// The authored half differences (male minus female, halved) that are not fitted, except the
/// breasts and the larynx (one-sided, see [`rest_shift`]).
fn authored(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let mut raw = Raw::confined(s, SOFT_TISSUE, Symmetry::Mirror, body::muscle_field(kit, 0.3, 0.8)?, Vec::new())?;
    raw.add(&body::areolae(kit, -0.004)?, 1.0);
    raw.add(&face::brow_ridge(kit)?, 0.9);
    raw.add(&face::jaw_width(kit)?, 0.8);
    raw.add(&face::chin_width(kit)?, 0.5);
    raw.add(&face::nose_length(kit)?, 0.6);
    raw.add(&face::nostrils(kit)?, 0.5);
    raw.add(&face::lip_fullness(kit, true)?, -0.35);
    raw.add(&face::lip_fullness(kit, false)?, -0.35);
    Ok(raw)
}

/// What [`rest_shift`] bakes into the rest shape: half the breasts and half the larynx, and the
/// adult face of both sexes ([`adult_face`]).
fn one_sided(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let mut raw = breast(kit)?.scaled(0.5);
    raw.add(&face::larynx(kit, LARYNX)?, 0.5);
    raw.add(&adult_face(kit)?, 1.0);
    Ok(raw)
}

/// The adult face of both sexes (todo 1.3, authored, read off renders and the adult-proportion
/// test: the scanned mean face on the generated head read young): 3 mm less cheek fat,
/// cheekbones 1.5 mm and the chin 2 mm more prominent, the lower nose 1.5 mm longer and its tip
/// 2.25 mm farther forward (a juvenile nose is short and flat: Farkas sn-prn is about 17 to 20 mm
/// in adults), a slightly stronger brow.
fn adult_face(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let mut raw = face::cheek_fullness(kit)?.scaled(-1.5);
    raw.add(&face::cheekbones(kit, true)?, 1.0);
    raw.add(&face::chin_projection(kit)?, 1.0);
    raw.add(&face::nose_length(kit)?, 1.0);
    raw.add(&face::nose_tip(kit, Vec3::Z * 0.0015, &["nose.tip"])?, 1.5);
    raw.add(&face::brow_ridge(kit)?, 0.4);
    Ok(raw)
}

/// The rest-shape shift that makes the one-sided features exact at both settings (half the
/// breasts and half the larynx) and gives both sexes an adult face, as `(vertex, displacement)`. The generator adds it to the
/// template's rest positions before the identity components are built; the switch then removes
/// the breasts' half at +1 and adds it at −1 (and the larynx the other way).
pub fn rest_shift(template: &crate::template::Template) -> IdResult<Vec<(u32, Vec3)>> {
    let space = Space::new(template)?;
    let kit = Kit::new(&space)?;
    let raw = one_sided(&kit)?;
    Ok(raw.component(&space, "rest-shift", Symmetry::Mirror)?.entries)
}

/// The ANSUR target of `target` as `(female, male)`, metres, given the rest value `rest`.
#[must_use]
pub fn targets(target: Target, rest: f32) -> (f32, f32) {
    match target {
        Target::Absolute(v) => (v.female_mean * 1.0e-3, v.male_mean * 1.0e-3),
        Target::Relative(v) => {
            let mid = f32::midpoint(v.male_mean, v.female_mean);
            (rest * v.female_mean / mid, rest * v.male_mean / mid)
        }
        Target::Difference(v) => {
            let half = 0.5e-3 * (v.male_mean - v.female_mean);
            (rest - half, rest + half)
        }
    }
}

/// `(female, male)` of `gauge` with `raw` as the switch (weights −1 and +1).
pub fn states(s: &Space<'_>, gauge: &Gauge, raw: &Raw) -> IdResult<(f32, f32)> {
    Ok((gauge.measure(s, &raw.disp, -1.0)?, gauge.measure(s, &raw.disp, 1.0)?))
}

/// Largest change of one term's amount per sweep of [`fit`].
const MAX_STEP: f32 = 2.0;

/// The fitted switch: the authored and one-sided parts plus the fitted terms at the amounts that
/// put each [`FITTED`] measurement's half difference (male minus female, halved) at its target.
/// Each term moves mainly its own measurement, so the amounts are found term by term (Gauss-Seidel
/// on the measured central-difference slopes, eight sweeps, steps limited to [`MAX_STEP`]).
/// Returns the switch and the amounts.
pub fn fit(kit: &Kit<'_, '_>) -> IdResult<(Raw, Vec<f32>)> {
    let s = kit.space;
    let terms = terms(kit)?;
    let mut base = authored(kit)?;
    base.add(&breast(kit)?, -0.5);
    base.add(&face::larynx(kit, LARYNX)?, 0.5);
    let half = |raw: &Raw, g: &Gauge| -> IdResult<f32> {
        let (f, m) = states(s, g, raw)?;
        Ok(0.5 * (m - f))
    };
    let mut want = Vec::with_capacity(FITTED.len());
    let mut slopes = Vec::with_capacity(FITTED.len());
    for (i, ((g, t), term)) in FITTED.iter().zip(&terms).enumerate() {
        let (tf, tm) = targets(*t, g.measure(s, &[], 0.0)?);
        want.push(0.5 * (tm - tf));
        let slope = half(term, g)?;
        if slope.abs() < 1.0e-6 {
            return Err(IdentityError::invalid("sex switch", i, "a fitted term does not move its measurement"));
        }
        slopes.push(slope);
    }
    let mut amounts = vec![0.0_f32; terms.len()];
    let mut raw = base.clone();
    for _ in 0..8 {
        for (i, ((g, _), term)) in FITTED.iter().zip(&terms).enumerate() {
            let step = ((want[i] - half(&raw, g)?) / slopes[i]).clamp(-MAX_STEP, MAX_STEP);
            amounts[i] += step;
            raw.add(term, step);
        }
    }
    Ok((raw, amounts))
}

/// Moves the genitals with the groin: each vulva node by the inverse-square-distance mean of the
/// skin nodes around the attachment loop (outside the genital regions, within 4 cm of the site),
/// then each male-piece node like the skin node it follows. The body terms' region confinement
/// would otherwise hold the genitals still while the loop moves.
fn carry_genitals(s: &Space<'_>, female: &[u16], body: u16, disp: &mut [Vec3]) {
    let pos = s.field().positions();
    let vulva = |n: usize| s.field().part_of(n) == body && female.contains(&s.field().region_of(n));
    let ring: Vec<(Vec3, Vec3)> = (0..pos.len())
        .filter(|&n| s.field().part_of(n) == body && !vulva(n) && !s.is_stowed(n))
        .filter(|&n| pos[n].distance(genitals::STOW_ORIGIN) < 0.06)
        .filter(|&n| genitals::site_coords(pos[n]).2.abs() < 0.012)
        .map(|n| (pos[n], disp.get(n).copied().unwrap_or(Vec3::ZERO)))
        .collect();
    for n in 0..pos.len() {
        if !vulva(n) {
            continue;
        }
        let (mut sum, mut wsum) = (Vec3::ZERO, 0.0);
        for &(q, d) in &ring {
            let w = 1.0 / q.distance_squared(pos[n]).max(1.0e-8);
            sum += d * w;
            wsum += w;
        }
        if let Some(d) = disp.get_mut(n).filter(|_| wsum > 0.0) {
            *d = sum / wsum;
        }
    }
    for n in 0..pos.len() {
        if let (true, Some(m)) = (s.is_stowed(n), s.attached(n)) {
            let follow = disp.get(m).copied().unwrap_or(Vec3::ZERO);
            if let Some(d) = disp.get_mut(n) {
                *d = follow;
            }
        }
    }
}

/// The switch component ([`SWITCH`]): [`fit`] plus the genital variant.
pub fn switch(kit: &Kit<'_, '_>) -> IdResult<IdentityComponent> {
    let s = kit.space;
    let (mut raw, _) = fit(kit)?;
    let female: Vec<u16> = genitals::FEMALE_REGIONS.iter().map(|r| s.region(r.name())).collect::<IdResult<_>>()?;
    let body = s.part("body")?;
    let pos = s.field().positions();
    carry_genitals(s, &female, body, &mut raw.disp);
    let mut found = (false, false);
    for (n, d) in raw.disp.iter_mut().enumerate() {
        let Some(&p) = pos.get(n) else { continue };
        if s.is_stowed(n) {
            let real = genitals::male_real(p);
            *d += (real - genitals::stow(real)) * 0.5;
            found.0 = true;
        } else if s.field().part_of(n) == body && female.contains(&s.field().region_of(n)) {
            let (f, m) = genitals::vulva_states(p);
            *d += (m - f) * 0.5;
            found.1 = true;
        }
    }
    if found != (true, true) {
        return Err(IdentityError::missing("genital variant", "the template lacks the vulva or the male piece"));
    }
    raw.component(s, SWITCH, Symmetry::Mirror)
}
