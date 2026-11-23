//! Calibration of the identity axes that claim a real-world measurement `[ID-03]`, `[ID-07]`:
//! tape-measure style measurements on the template surface, and the ANSUR II summary statistics
//! (`data/anthropometry/ansur2.json`) the axes are scaled to.
//!
//! **Rule.** Where an axis claims a measurement, its displacement per unit weight is scaled so that
//! weight 1 changes that measurement, taken on the template like the ANSUR II measurer takes it, by
//! one standard deviation of the variable. A linear component cannot change its unit with the sex
//! axis, so the unit is the root mean square of the male and the female SD ([`Variable::sd`]); both
//! are listed in [`VARIABLES`] (they differ by less than 15 % for every variable used, except neck
//! circumference, 25.8 against 19.2 mm). Each recipe keeps its authored *shape* (where the change
//! happens); only its amount is measured. Axes the file has nothing to say about (breast volume and
//! shape, nipple and areola, hip flank, muscularity, age, the face except eye spacing and ear
//! length) stay authored amounts, documented as such at their recipes.
//!
//! **Population.** ANSUR II measured US Army soldiers (2010 to 2012; 4082 men, 1986 women, age 17
//! to 58): a working-age, screened, physically active population, not the general public (see the
//! `population_note` of the data file). The template is the sex midpoint; one SD is a population
//! spread, not a property of the template.
//!
//! | component | measurement on the template | ANSUR II variable(s) |
//! |---|---|---|
//! | `global.height` | top of the head above the soles | `stature` |
//! | `global.mass` | least-squares fit of waist, chest, buttock and thigh circumference to their expected change per SD of weight at fixed stature ([`mass_targets`]) | `weightkg`, `stature`, `waistcircumference`, `chestcircumference`, `buttockcircumference`, `thighcircumference` and their correlations |
//! | `global.sex` | the shoulder, hip, waist and neck terms: one within-sex SD per unit ([`Variable::sd`]) in the direction of [`Variable::sex_half_difference`]; the sexes themselves are the switch (`identity::sex`) | `biacromialbreadth`, `hipbreadth`, `waistbreadth`, `neckcircumference`, `stature` |
//! | the rest of [`CALIBRATIONS`] | see the table entries | one variable each (two for arm length) |

use humanart_math::Vec3;

use super::Raw;
use super::error::{IdResult, IdentityError};
use super::space::{PartKind, Reach, Space};

/// Summary of one ANSUR II variable (`data/anthropometry/ansur2.json`, `stats.male` and
/// `stats.female`): millimetres, kilograms for `weightkg`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Variable {
    /// The variable id (the CSV column of the dataset).
    pub id: &'static str,
    /// Male mean.
    pub male_mean: f32,
    /// Male sample standard deviation.
    pub male_sd: f32,
    /// Female mean.
    pub female_mean: f32,
    /// Female sample standard deviation.
    pub female_sd: f32,
}

impl Variable {
    /// The unit of a calibrated axis in the variable's own unit: the root mean square of the two
    /// within-sex standard deviations.
    #[must_use]
    pub fn sd(&self) -> f32 {
        f32::midpoint(self.male_sd * self.male_sd, self.female_sd * self.female_sd).sqrt()
    }

    /// [`sd`](Self::sd) in metres (for the length variables).
    #[must_use]
    pub fn sd_m(&self) -> f32 {
        self.sd() * 1.0e-3
    }

    /// Half the male-minus-female difference of the mean relative to mean stature, times
    /// `stature` metres: the change per unit of the sex axis that takes the template (the sex
    /// midpoint) to the male mean at +1 and the female mean at −1 at its own stature. The ratio of
    /// the means stands in for the mean of the ratios (the file has summaries only).
    #[must_use]
    pub fn sex_half_difference(&self, stature: f32) -> f32 {
        0.5 * (self.male_mean / STATURE.male_mean - self.female_mean / STATURE.female_mean) * stature
    }
}

const fn var(id: &'static str, male_mean: f32, male_sd: f32, female_mean: f32, female_sd: f32) -> Variable {
    Variable { id, male_mean, male_sd, female_mean, female_sd }
}

/// `stature`: standing height, mm.
pub const STATURE: Variable = var("stature", 1756.2, 68.6, 1628.5, 64.2);
/// `weightkg`: body weight, kg.
pub const WEIGHT: Variable = var("weightkg", 85.52, 14.22, 67.76, 10.98);
/// `biacromialbreadth`: distance between the acromion landmarks, mm.
pub const BIACROMIAL: Variable = var("biacromialbreadth", 415.7, 19.2, 365.3, 18.3);
/// `hipbreadth`: maximum horizontal breadth across the hips, standing, mm.
pub const HIP_BREADTH: Variable = var("hipbreadth", 345.7, 24.2, 353.8, 26.7);
/// `waistbreadth`: horizontal breadth of the waist at omphalion (the navel), mm.
pub const WAIST_BREADTH: Variable = var("waistbreadth", 326.4, 34.7, 299.9, 33.0);
/// `waistcircumference`: circumference of the waist at omphalion, mm.
pub const WAIST_CIRC: Variable = var("waistcircumference", 940.6, 111.7, 860.9, 99.9);
/// `chestcircumference`: circumference at the nipples (women: the fullest part of the breast), mm.
pub const CHEST_CIRC: Variable = var("chestcircumference", 1058.7, 87.4, 946.9, 82.7);
/// `buttockcircumference`: circumference at the maximum protrusion of the buttocks, mm.
pub const BUTTOCK_CIRC: Variable = var("buttockcircumference", 1019.5, 76.7, 1021.2, 75.9);
/// `buttockdepth`: horizontal depth of the trunk at the maximum protrusion of the buttock, mm.
pub const BUTTOCK_DEPTH: Variable = var("buttockdepth", 245.8, 26.5, 232.9, 24.3);
/// `thighcircumference`: thigh circumference at the lowest point of the gluteal furrow, mm.
pub const THIGH_CIRC: Variable = var("thighcircumference", 625.1, 58.5, 616.1, 55.8);
/// `crotchheight`: height of the crotch above the floor, mm.
pub const CROTCH_HEIGHT: Variable = var("crotchheight", 845.9, 46.5, 782.3, 44.6);
/// `acromionradialelength`: acromion to radiale (upper arm), mm.
pub const ACROMION_RADIALE: Variable = var("acromionradialelength", 335.2, 17.5, 311.2, 17.2);
/// `radialestylionlength`: radiale to stylion (forearm), mm.
pub const RADIALE_STYLION: Variable = var("radialestylionlength", 267.9, 15.4, 241.3, 15.2);
/// `handbreadth`: breadth of the hand between metacarpale II and V, mm.
pub const HAND_BREADTH: Variable = var("handbreadth", 88.3, 4.4, 78.2, 3.8);
/// `footlength`: heel (pternion) to the tip of the longest toe, mm.
pub const FOOT_LENGTH: Variable = var("footlength", 271.2, 13.1, 246.3, 12.4);
/// `footbreadthhorizontal`: maximum horizontal breadth of the foot, mm.
pub const FOOT_BREADTH: Variable = var("footbreadthhorizontal", 101.9, 5.2, 92.7, 4.8);
/// `interpupillarybreadth`: distance between the pupil centres, mm (stored in tenths of a
/// millimetre in the CSV; the data file has already converted it).
pub const INTERPUPILLARY: Variable = var("interpupillarybreadth", 64.0, 3.4, 61.7, 3.6);
/// `earlength`: superaurale to subaurale, mm.
pub const EAR_LENGTH: Variable = var("earlength", 64.2, 4.5, 59.4, 3.9);
/// `neckcircumference`: circumference of the neck just below the larynx, mm.
pub const NECK_CIRC: Variable = var("neckcircumference", 397.6, 25.8, 329.8, 19.2);

/// Every variable used here (checked against the data file by a test).
pub const VARIABLES: &[Variable] = &[
    STATURE,
    WEIGHT,
    BIACROMIAL,
    HIP_BREADTH,
    WAIST_BREADTH,
    WAIST_CIRC,
    CHEST_CIRC,
    BUTTOCK_CIRC,
    BUTTOCK_DEPTH,
    THIGH_CIRC,
    CROTCH_HEIGHT,
    ACROMION_RADIALE,
    RADIALE_STYLION,
    HAND_BREADTH,
    FOOT_LENGTH,
    FOOT_BREADTH,
    INTERPUPILLARY,
    EAR_LENGTH,
    NECK_CIRC,
];

/// A Pearson correlation of the data file (`correlation.male` / `.female`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Correlation {
    /// First variable id.
    pub a: &'static str,
    /// Second variable id.
    pub b: &'static str,
    /// Male r.
    pub male: f32,
    /// Female r.
    pub female: f32,
}

/// The correlations used here (checked against the data file by a test).
pub const CORRELATIONS: &[Correlation] = &[
    Correlation { a: "weightkg", b: "stature", male: 0.469, female: 0.533 },
    Correlation { a: "waistcircumference", b: "weightkg", male: 0.888, female: 0.836 },
    Correlation { a: "waistcircumference", b: "stature", male: 0.208, female: 0.229 },
    Correlation { a: "chestcircumference", b: "weightkg", male: 0.907, female: 0.829 },
    Correlation { a: "chestcircumference", b: "stature", male: 0.247, female: 0.260 },
    Correlation { a: "buttockcircumference", b: "weightkg", male: 0.933, female: 0.913 },
    Correlation { a: "buttockcircumference", b: "stature", male: 0.340, female: 0.379 },
    Correlation { a: "thighcircumference", b: "weightkg", male: 0.903, female: 0.893 },
    Correlation { a: "thighcircumference", b: "stature", male: 0.255, female: 0.317 },
    Correlation { a: "acromionradialelength", b: "radialestylionlength", male: 0.719, female: 0.687 },
];

/// The correlation of `a` and `b` (either order) as `(male, female)`.
fn correlation(a: &str, b: &str) -> IdResult<(f32, f32)> {
    CORRELATIONS
        .iter()
        .find(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a))
        .map(|c| (c.male, c.female))
        .ok_or_else(|| IdentityError::missing("ANSUR II correlation", a))
}

/// A height on the template: the rest height of landmark `.0` moved `.2` of the way to the rest
/// height of landmark `.1`.
#[derive(Clone, Copy, Debug)]
pub struct Level(pub &'static str, pub &'static str, pub f32);

impl Level {
    fn y(&self, s: &Space<'_>) -> IdResult<f32> {
        let (a, b) = (s.landmark(self.0)?.y, s.landmark(self.1)?.y);
        Ok(a + (b - a) * self.2)
    }
}

/// Half the height of the band of body nodes a girth is taken over, metres.
pub const GIRTH_BAND: f32 = 0.008;

/// How a measurement is taken on the (displaced) template. Landmarks move with their carrying
/// node; only nodes of the skin (`PartKind::Body`) are measured; region lists name subtrees.
#[derive(Clone, Copy, Debug)]
pub enum Gauge {
    /// The top of the head (`head.vertex`) above the lowest skin node.
    Stature,
    /// A landmark above the lowest skin node.
    Height(&'static str),
    /// The length of the polyline through the landmarks.
    Path(&'static [&'static str]),
    /// The horizontal (x-z) distance between two landmarks.
    Horizontal(&'static str, &'static str),
    /// The x extent of the skin nodes of the regions whose rest height lies between the levels.
    Breadth(&'static [&'static str], Level, Level),
    /// The z extent of the skin nodes of the regions whose rest height lies between the levels.
    Depth(&'static [&'static str], Level, Level),
    /// The perimeter of the convex hull (a tape measure) in the x-z plane of the skin nodes of the
    /// regions within [`GIRTH_BAND`] of the level; with `true` only the left side (x > 0, one
    /// thigh).
    Girth(&'static [&'static str], Level, bool),
    /// ANSUR II `mentonsellionlength` on the displaced midline profile of the outer face skin
    /// ([`midline_profile`], [`face_points`]): the straight distance from the sellion (the deepest
    /// point of the nasal root) to the menton (the lowest point of the chin). Found anew on each
    /// shape, so it follows the skin where fixed landmark vertices would not (the menton is
    /// whichever vertex is lowest on the reshaped chin).
    FaceHeight,
}

/// The regions under the face whose skin is inside the mouth (the profile skips them).
pub const MOUTH_INSIDE: [&str; 6] =
    ["mouth-interior", "teeth-upper", "teeth-lower", "gums-upper", "gums-lower", "tongue"];

/// The frontmost of `points` per millimetre of height between `lo` and `hi` among those near the
/// midline (|x| < 2 mm) and in front of the depth `front`, top down: the face's profile as a
/// side photograph shows it.
#[must_use]
pub fn midline_profile(points: impl Iterator<Item = Vec3>, lo: f32, hi: f32, front: f32) -> Vec<Vec3> {
    let span = hi - lo;
    if span.is_nan() || span <= 0.0 || span >= 10.0 {
        return Vec::new();
    }
    let mut bins: Vec<Option<Vec3>> = vec![None; ((hi - lo) * 1000.0) as usize + 1];
    for q in points.filter(|q| q.x.abs() < 0.002 && q.y > lo && q.y < hi && q.z > front) {
        if let Some(slot) = bins.get_mut(((hi - q.y) * 1000.0) as usize)
            && slot.is_none_or(|o: Vec3| q.z > o.z)
        {
            *slot = Some(q);
        }
    }
    bins.into_iter().flatten().collect()
}

/// The profile point between the heights `lo` and `hi` that lies farthest along `dir`, softly:
/// the mean of the points weighted by `exp((d - d_max) / 0.5 mm)` for their distance `d` along
/// `dir`. The hard extreme jumps from one vertex to another as a shape changes (2 mm jumps of the
/// face height made the sex switch's fit oscillate); the soft one moves continuously and lies
/// within a fraction of a millimetre of it.
#[must_use]
pub fn profile_support(profile: &[Vec3], lo: f32, hi: f32, dir: Vec3) -> Option<Vec3> {
    const SCALE: f32 = 0.0005;
    let window = || profile.iter().copied().filter(|q| q.y >= lo && q.y <= hi);
    let top = window().map(|q| q.dot(dir)).fold(f32::NEG_INFINITY, f32::max);
    if !top.is_finite() {
        return None;
    }
    let (sum, wsum) = window().fold((Vec3::ZERO, 0.0), |(s, w), q| {
        let k = ((q.dot(dir) - top) / SCALE).exp();
        (s + q * k, w + k)
    });
    (wsum > 0.0).then(|| sum / wsum)
}

/// The midline points `[sellion, supramentale, pogonion, menton]` of a profile, given the heights
/// of the nose tip, the brow and the lower lip.
/// `None` where a point has no profile sample.
#[must_use]
pub fn profile_points(profile: &[Vec3], nose_tip_y: f32, brow_y: f32, lower_lip_y: f32) -> Option<[Vec3; 4]> {
    let sellion = profile_support(profile, nose_tip_y + 0.012, brow_y + 0.006, -Vec3::Z)?;
    let supramentale = profile_support(profile, lower_lip_y - 0.022, lower_lip_y - 0.003, -Vec3::Z)?;
    let pogonion = profile_support(profile, supramentale.y - 0.03, supramentale.y - 0.003, Vec3::Z)?;
    // The menton: the lowest point of the chin, taken 60 degrees below the horizontal among the
    // points at most 25 mm behind the pogonion, so that the neck below does not count as chin.
    let chin: Vec<Vec3> = profile.iter().copied().filter(|q| q.z > pogonion.z - 0.025).collect();
    let menton = profile_support(&chin, pogonion.y - 0.035, pogonion.y, Vec3::new(0.0, -0.866, 0.5))?;
    Some([sellion, supramentale, pogonion, menton])
}

impl Gauge {
    /// The measurement, metres, on the rest positions displaced by `w` times `disp` (per node).
    pub fn measure(&self, s: &Space<'_>, disp: &[Vec3], w: f32) -> IdResult<f32> {
        let field = s.field();
        let rest = field.positions();
        let at = |n: usize| rest.get(n).copied().unwrap_or(Vec3::ZERO) + disp.get(n).copied().unwrap_or(Vec3::ZERO) * w;
        let landmark = |name: &str| -> IdResult<Vec3> {
            let node = s.landmark_nodes(&[name])?.first().copied().unwrap_or(u32::MAX) as usize;
            Ok(s.landmark(name)? + disp.get(node).copied().unwrap_or(Vec3::ZERO) * w)
        };
        let skin = |n: usize| s.kind_of(n) == PartKind::Body;
        let floor = || (0..s.node_count()).filter(|&n| skin(n)).map(|n| at(n).y).fold(f32::INFINITY, f32::min);
        let selected = |regions: &[&str], keep: &dyn Fn(Vec3) -> bool| -> IdResult<Vec<Vec3>> {
            let flags = s.reach_flags(&Reach { regions, except: &[], margin: 0.0 })?;
            Ok((0..s.node_count())
                .filter(|&n| {
                    skin(n)
                        && flags.get(field.region_of(n) as usize).copied().unwrap_or(false)
                        && rest.get(n).is_some_and(|&p| keep(p))
                })
                .map(at)
                .collect())
        };
        let value = match *self {
            Self::Stature => landmark("head.vertex")?.y - floor(),
            Self::Height(name) => landmark(name)?.y - floor(),
            Self::Path(names) => {
                let mut total = 0.0;
                let mut prev: Option<Vec3> = None;
                for name in names {
                    let p = landmark(name)?;
                    if let Some(q) = prev {
                        total += p.distance(q);
                    }
                    prev = Some(p);
                }
                total
            }
            Self::Horizontal(a, b) => {
                let d = landmark(a)? - landmark(b)?;
                d.x.hypot(d.z)
            }
            Self::Breadth(regions, lo, hi) | Self::Depth(regions, lo, hi) => {
                let (y0, y1) = (lo.y(s)?, hi.y(s)?);
                let (y0, y1) = (y0.min(y1), y0.max(y1));
                let pts = selected(regions, &|p| p.y >= y0 && p.y <= y1)?;
                let axis = |p: &Vec3| if matches!(self, Self::Breadth(..)) { p.x } else { p.z };
                let (lo, hi) =
                    pts.iter().map(axis).fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| (a.min(v), b.max(v)));
                hi - lo
            }
            Self::Girth(regions, level, left) => {
                let y = level.y(s)?;
                let pts = selected(regions, &|p| (p.y - y).abs() <= GIRTH_BAND && (!left || p.x > 0.0))?;
                let flat: Vec<(f32, f32)> = pts.iter().map(|p| (p.x, p.z)).collect();
                hull_perimeter(&flat)
            }
            Self::FaceHeight => {
                let flags = s.reach_flags(&Reach { regions: &["face"], except: &MOUTH_INSIDE, margin: 0.0 })?;
                let points = (0..s.node_count())
                    .filter(|&n| skin(n) && flags.get(field.region_of(n) as usize).copied().unwrap_or(false))
                    .map(at);
                let (prn, brow) = (landmark("nose.tip")?, landmark("L.brow.03")?);
                let (li, sto) = (landmark("lip.lower.outer.06")?, landmark("lip.upper.inner.06")?);
                let (vertex, tragion) = (landmark("head.vertex")?, landmark("ear.L.tragion")?);
                let profile = midline_profile(points, sto.y - 0.09, vertex.y, tragion.z);
                let [sellion, _, _, menton] = profile_points(&profile, prn.y, brow.y, li.y)
                    .ok_or(IdentityError::invalid("measurement", 0, "the face profile misses a point"))?;
                sellion.distance(menton)
            }
        };
        if value.is_finite() && value > 0.0 {
            Ok(value)
        } else {
            Err(IdentityError::invalid("measurement", 0, "no skin nodes to measure"))
        }
    }

    /// The change of the measurement from the rest shape to weight `w` of `disp`, metres.
    pub fn change(&self, s: &Space<'_>, disp: &[Vec3], w: f32) -> IdResult<f32> {
        Ok(self.measure(s, disp, w)? - self.measure(s, disp, 0.0)?)
    }

    /// The change of the measurement at weight 1 of `disp`, metres (the forward difference: the
    /// gauges that take an extreme point are not linear through the rest shape).
    pub fn per_unit(&self, s: &Space<'_>, disp: &[Vec3]) -> IdResult<f32> {
        self.change(s, disp, 1.0)
    }
}

/// The perimeter of the convex hull of `pts` (monotone chain); 0 for fewer than 3 points.
#[must_use]
pub fn hull_perimeter(pts: &[(f32, f32)]) -> f32 {
    let mut p: Vec<(f32, f32)> = pts.iter().copied().filter(|q| q.0.is_finite() && q.1.is_finite()).collect();
    if p.len() < 3 {
        return 0.0;
    }
    p.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let cross = |o: (f32, f32), a: (f32, f32), b: (f32, f32)| (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0);
    let mut hull: Vec<(f32, f32)> = Vec::with_capacity(p.len() * 2);
    for pass in 0..2 {
        let start = hull.len();
        let iter: Box<dyn Iterator<Item = &(f32, f32)>> =
            if pass == 0 { Box::new(p.iter()) } else { Box::new(p.iter().rev()) };
        for &q in iter {
            while hull.len() >= start + 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], q) <= 0.0 {
                hull.pop();
            }
            hull.push(q);
        }
        hull.pop();
    }
    let n = hull.len();
    (0..n)
        .map(|i| {
            let (a, b) = (hull[i], hull[(i + 1) % n]);
            (a.0 - b.0).hypot(a.1 - b.1)
        })
        .sum()
}

/// What one unit of a calibrated axis is.
#[derive(Clone, Copy, Debug)]
pub enum Unit {
    /// One SD of the variable ([`Variable::sd`]).
    Sd(Variable),
    /// One SD of the sum of two variables, per sex `sqrt(sa^2 + sb^2 + 2 r sa sb)` with their
    /// correlation, then the root mean square of the sexes.
    SumSd(Variable, Variable),
}

impl Unit {
    /// The target change per unit weight, metres.
    pub fn metres(&self) -> IdResult<f32> {
        match *self {
            Self::Sd(v) => Ok(v.sd_m()),
            Self::SumSd(a, b) => {
                let (rm, rf) = correlation(a.id, b.id)?;
                let sum = |sa: f32, sb: f32, r: f32| sa * sa + sb * sb + 2.0 * r * sa * sb;
                let m = sum(a.male_sd, b.male_sd, rm);
                let f = sum(a.female_sd, b.female_sd, rf);
                Ok(f32::midpoint(m, f).sqrt() * 1.0e-3)
            }
        }
    }
}

/// One calibrated axis: the component, how it is measured, and what one unit is.
#[derive(Clone, Copy, Debug)]
pub struct Calibration {
    /// The component name.
    pub component: &'static str,
    /// The measurement.
    pub gauge: Gauge,
    /// One unit of weight.
    pub unit: Unit,
}

/// Trunk regions (no arms: the A-pose arms hang beside the hips).
pub const TRUNK: &[&str] = &["chest", "abdomen", "back", "pelvis"];
/// Trunk below the chest and both thighs.
pub const HIPS_AND_THIGHS: &[&str] = &["pelvis", "abdomen", "back", "thigh"];

/// Waist level: omphalion.
const NAVEL: Level = Level("torso.navel", "torso.navel", 0.0);
/// The gluteal furrow, where ANSUR II takes thigh circumference: 12 % down from the pubis to the
/// knee centre (template geometry, not a statistic).
pub const THIGH_LEVEL: Level = Level("torso.pubis", "L.knee.center", 0.12);
/// Buttock level: the trochanter (the maximum protrusion of the template's buttocks lies there).
pub const BUTTOCK_LEVEL: Level = Level("L.hip.trochanter", "L.hip.trochanter", 0.0);
/// Neck circumference level: just below the larynx, 30 % of the way to the suprasternal notch.
pub const NECK_LEVEL: Level = Level("neck.front", "torso.suprasternal", 0.3);

/// Waist circumference at omphalion.
pub const WAIST_GIRTH: Gauge = Gauge::Girth(TRUNK, NAVEL, false);
/// Chest circumference at the nipples.
pub const CHEST_GIRTH: Gauge =
    Gauge::Girth(&["chest", "back", "abdomen"], Level("L.chest.nipple", "L.chest.nipple", 0.0), false);
/// Buttock circumference at the trochanter.
pub const BUTTOCK_GIRTH: Gauge = Gauge::Girth(HIPS_AND_THIGHS, BUTTOCK_LEVEL, false);
/// Left thigh circumference at the gluteal furrow.
pub const THIGH_GIRTH: Gauge = Gauge::Girth(&["thigh"], THIGH_LEVEL, true);
/// Hip breadth: the widest point between 15 % down the thigh and the iliac spine.
pub const HIP_BREADTH_GAUGE: Gauge = Gauge::Breadth(
    HIPS_AND_THIGHS,
    Level("torso.pubis", "L.knee.center", 0.15),
    Level("L.hip.asis", "L.hip.asis", 0.0),
);
/// Waist breadth at omphalion (a band of ±8 % of the navel-to-pubis distance).
pub const WAIST_BREADTH_GAUGE: Gauge =
    Gauge::Breadth(TRUNK, Level("torso.navel", "torso.pubis", 0.08), Level("torso.navel", "torso.pubis", -0.08));
/// Biacromial breadth.
pub const BIACROMIAL_GAUGE: Gauge = Gauge::Path(&["L.shoulder.acromion", "R.shoulder.acromion"]);
/// Neck circumference.
pub const NECK_GIRTH: Gauge = Gauge::Girth(&["neck"], NECK_LEVEL, false);

/// The axes calibrated to one SD of a single measurement (see the module table). `global.mass`
/// and `global.sex` are calibrated inside their recipes ([`mass_targets`],
/// [`Variable::sex_half_difference`]).
pub const CALIBRATIONS: &[Calibration] = &[
    Calibration { component: "global.height", gauge: Gauge::Stature, unit: Unit::Sd(STATURE) },
    Calibration {
        component: "semantic.leg-length",
        gauge: Gauge::Height("torso.pubis"),
        unit: Unit::Sd(CROTCH_HEIGHT),
    },
    Calibration {
        component: "semantic.arm-length",
        gauge: Gauge::Path(&["L.shoulder.acromion", "L.elbow.lateral", "L.wrist.radial"]),
        unit: Unit::SumSd(ACROMION_RADIALE, RADIALE_STYLION),
    },
    Calibration { component: "semantic.shoulder-width", gauge: BIACROMIAL_GAUGE, unit: Unit::Sd(BIACROMIAL) },
    Calibration { component: "semantic.hip-width", gauge: HIP_BREADTH_GAUGE, unit: Unit::Sd(HIP_BREADTH) },
    Calibration { component: "semantic.waist-width", gauge: WAIST_BREADTH_GAUGE, unit: Unit::Sd(WAIST_BREADTH) },
    Calibration {
        component: "semantic.glute-size",
        gauge: Gauge::Depth(
            HIPS_AND_THIGHS,
            Level("L.hip.trochanter", "L.knee.center", 0.1),
            Level("L.hip.trochanter", "L.hip.asis", 0.5),
        ),
        unit: Unit::Sd(BUTTOCK_DEPTH),
    },
    Calibration { component: "semantic.thigh-fullness", gauge: THIGH_GIRTH, unit: Unit::Sd(THIGH_CIRC) },
    Calibration {
        component: "semantic.palm-width",
        gauge: Gauge::Path(&["L.hand.knuckle.index", "L.hand.knuckle.little"]),
        unit: Unit::Sd(HAND_BREADTH),
    },
    Calibration {
        component: "semantic.foot-length",
        gauge: Gauge::Horizontal("L.heel", "L.foot.tip"),
        unit: Unit::Sd(FOOT_LENGTH),
    },
    Calibration {
        component: "semantic.foot-width",
        gauge: Gauge::Path(&["L.foot.mtp1", "L.foot.mtp5"]),
        unit: Unit::Sd(FOOT_BREADTH),
    },
    Calibration {
        component: "semantic.eye-spacing",
        gauge: Gauge::Path(&["L.iris.center", "R.iris.center"]),
        unit: Unit::Sd(INTERPUPILLARY),
    },
    Calibration {
        component: "semantic.ear-size",
        gauge: Gauge::Path(&["ear.L.superaurale", "ear.L.lobule"]),
        unit: Unit::Sd(EAR_LENGTH),
    },
    Calibration { component: "semantic.neck-width", gauge: NECK_GIRTH, unit: Unit::Sd(NECK_CIRC) },
];

/// The calibration of `component`, if it has one.
#[must_use]
pub fn calibration(component: &str) -> Option<&'static Calibration> {
    CALIBRATIONS.iter().find(|c| c.component == component)
}

/// The largest and smallest factor a calibration may apply to an authored amount: outside them the
/// measurement, not the amount, is wrong (a landmark that does not move, a gauge on the wrong
/// regions), so the build fails instead.
pub const SCALE_LIMITS: (f32, f32) = (0.02, 50.0);

/// `raw` scaled so that `gauge` changes by `target` metres per unit weight (`target` may be
/// negative: the field is reversed).
pub fn scaled_to(s: &Space<'_>, raw: Raw, gauge: &Gauge, target: f32) -> IdResult<Raw> {
    // A gauge that takes an extreme point is only piecewise linear in the weight: start from the
    // linear guess, bracket the target along its sign and bisect.
    let k0 = checked_scale(target / gauge.change(s, &raw.disp, 1.0)?)?;
    let sign = k0.signum();
    let reached = |a: f32| -> IdResult<f32> { Ok(gauge.change(s, &raw.disp, sign * a)? / target) };
    let (mut lo, mut hi) = (0.0, k0.abs());
    let mut grow = 0;
    while reached(hi)? < 1.0 {
        lo = hi;
        hi *= 2.0;
        grow += 1;
        if grow > 8 {
            return Err(IdentityError::invalid("calibration", 0, "the measurement does not reach its target"));
        }
    }
    for _ in 0..BISECTION_STEPS {
        let mid = f32::midpoint(lo, hi);
        if reached(mid)? < 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(raw.scaled(checked_scale(sign * f32::midpoint(lo, hi))?))
}

/// Bisection steps of a calibration (the bracket shrinks to 2^-24 of its width).
const BISECTION_STEPS: usize = 24;

/// Fixed-point steps of a calibration.
const CALIBRATION_STEPS: usize = 3;

/// `k` when it is a usable calibration factor (see [`SCALE_LIMITS`]).
fn checked_scale(k: f32) -> IdResult<f32> {
    if !k.is_finite() || k.abs() < SCALE_LIMITS.0 || k.abs() > SCALE_LIMITS.1 {
        return Err(IdentityError::invalid("calibration", 0, "the measurement does not follow the component"));
    }
    Ok(k)
}

/// `raw` calibrated when `component` is in [`CALIBRATIONS`], unchanged otherwise.
pub fn calibrate(s: &Space<'_>, component: &str, raw: Raw) -> IdResult<Raw> {
    match calibration(component) {
        Some(c) => scaled_to(s, raw, &c.gauge, c.unit.metres()?),
        None => Ok(raw),
    }
}

/// The circumferences that calibrate `global.mass`, each with its expected change per unit,
/// metres: the change of the circumference per SD of weight at fixed stature, from the
/// standardised partial regression slope `(r_cw - r_cs r_ws) / (1 - r_ws^2)` times the SD of the
/// circumference, per sex, then the mean of the sexes. The mass axis keeps the skeleton, so stature
/// is held fixed.
pub fn mass_targets() -> IdResult<Vec<(Gauge, f32)>> {
    let (wm, wf) = correlation(WEIGHT.id, STATURE.id)?;
    let mut out = Vec::with_capacity(4);
    for (gauge, v) in
        [(WAIST_GIRTH, WAIST_CIRC), (CHEST_GIRTH, CHEST_CIRC), (BUTTOCK_GIRTH, BUTTOCK_CIRC), (THIGH_GIRTH, THIGH_CIRC)]
    {
        let (cwm, cwf) = correlation(v.id, WEIGHT.id)?;
        let (csm, csf) = correlation(v.id, STATURE.id)?;
        let beta = |cw: f32, cs: f32, ws: f32| (cw - cs * ws) / (1.0 - ws * ws);
        let m = beta(cwm, csm, wm) * v.male_sd;
        let f = beta(cwf, csf, wf) * v.female_sd;
        out.push((gauge, f32::midpoint(m, f) * 1.0e-3));
    }
    Ok(out)
}

/// `raw` (the mass field) scaled by the least-squares factor that best matches [`mass_targets`].
pub fn calibrate_mass(s: &Space<'_>, raw: Raw) -> IdResult<Raw> {
    let targets = mass_targets()?;
    let mut k = 1.0;
    for _ in 0..CALIBRATION_STEPS {
        let (mut num, mut den) = (0.0, 0.0);
        for (gauge, target) in &targets {
            let m = gauge.change(s, &raw.disp, k)? / k;
            num += m * target;
            den += m * m;
        }
        k = checked_scale(num / den)?;
    }
    Ok(raw.scaled(k))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hull_perimeter_of_a_square_and_a_circle() {
        let square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5), (0.2, 0.7)];
        assert!((hull_perimeter(&square) - 4.0).abs() < 1.0e-5);
        let circle: Vec<(f32, f32)> = (0..256).map(|i| (i as f32 * std::f32::consts::TAU / 256.0).sin_cos()).collect();
        assert!((hull_perimeter(&circle) - std::f32::consts::TAU).abs() < 1.0e-3);
        assert_eq!(hull_perimeter(&[(0.0, 0.0), (1.0, 1.0)]), 0.0);
        assert_eq!(hull_perimeter(&[(f32::NAN, 0.0), (1.0, 1.0), (0.0, 1.0)]), 0.0);
    }

    #[test]
    fn units_follow_the_statistics() {
        // RMS of 68.6 and 64.2 mm.
        assert!((STATURE.sd_m() - 0.066_43).abs() < 1.0e-5);
        // Arm: male 17.5^2 + 15.4^2 + 2 0.719 17.5 15.4 = 930.9, female 886.1; RMS 30.14.
        let arm = Unit::SumSd(ACROMION_RADIALE, RADIALE_STYLION).metres().unwrap();
        assert!((arm - 0.030_14).abs() < 1.0e-4, "{arm}");
        // Women have wider hips relative to stature: negative toward male.
        assert!(HIP_BREADTH.sex_half_difference(1.7) < -0.015);
        assert!(BIACROMIAL.sex_half_difference(1.7) > 0.009);
        // Mass: about 1 SD of each circumference per SD of weight at fixed stature.
        let t = mass_targets().unwrap();
        assert!((t[0].1 - 0.1064).abs() < 0.002, "{}", t[0].1);
        assert!(t.iter().all(|&(_, x)| x > 0.04 && x < 0.12));
        assert!(correlation("stature", "nothing").is_err());
    }
}
