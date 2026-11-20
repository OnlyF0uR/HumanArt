//! Proportions of the neutral adult template and the anatomical points derived from them.
//!
//! # Sources
//!
//! The template is a *sex midpoint* adult: each body measurement that ANSUR II defines is the mean of
//! the male and female means of the 2012 US Army anthropometric survey (ANSUR II public working
//! databases, US Army NSRDEC, "cleared for unlimited public release"; definitions from Gordon et
//! al., 2014, NATICK/TR-15/007), rounded to the millimetre. The summary file
//! `data/anthropometry/ansur2.json` holds those statistics; every constant below that cites an
//! ANSUR II variable id agrees with that file's `sex_midpoint_mean` to 0.5 mm (test
//! `anat_01_constants_match_the_ansur_ii_file`). The midpoint is a design choice, not a
//! population statistic: ANSUR II measured US soldiers aged 17-58, and no real population has the
//! midpoint's mixture of male and female proportions.
//!
//! Joint centres are placed by the usual biomechanics conventions relative to those surface
//! landmarks (hip centre at the level of the top of the greater trochanter, knee centre between
//! the femoral epicondyles, ankle centre between the malleoli). Values that no document read for
//! this repository supports are marked **unsourced**: the head height, the eye's position and
//! radius in the head frame, the jaw pivot, the pelvis and spine joints, the shoulder and hip joint
//! offsets from the midline, the finger and thumb joint chains and the rest-pose angles. They are
//! design values; the identity axes move all of them.
//!
//! Units are metres, Y up, the character faces +Z, the left side is +X. Values are for the left
//! side; [`mirror`] gives the right.

use humanart_math::Vec3;

/// Stature, ANSUR II `stature` (male 1756.2, female 1628.5 mm).
pub const STATURE: f32 = 1.692;
/// Vertical distance from the tragion to the top of the head, ANSUR II `tragiontopofhead` (131.1 /
/// 126.5 mm). The head frame's origin is the midpoint between the tragions.
pub const TRAGION_TO_VERTEX: f32 = 0.129;
/// Head height, vertex to menton. **Unsourced** (Farkas' v-gn was quoted from memory and has not
/// been checked; ANSUR II has no vertex-menton measurement).
pub const HEAD_HEIGHT: f32 = 0.225;
/// Biacromial breadth over the skin, ANSUR II `biacromialbreadth` (415.7 / 365.3 mm).
pub const BIACROMIAL: f32 = 0.391;
/// Hip breadth: the maximum breadth over the buttocks (not at the trochanters), ANSUR II
/// `hipbreadth` (345.7 / 353.8 mm).
pub const HIP_BREADTH: f32 = 0.350;
/// Crotch height, ANSUR II `crotchheight` (845.9 / 782.3 mm).
pub const CROTCH_HEIGHT: f32 = 0.814;
/// Hand length, wrist crease (stylion) to the tip of the straight middle finger, ANSUR II
/// `handlength` (193.3 / 181.1 mm).
pub const HAND_LENGTH: f32 = 0.187;
/// Hand breadth across the metacarpals II-V, ANSUR II `handbreadth` (88.3 / 78.2 mm).
pub const HAND_BREADTH: f32 = 0.083;
/// Foot length, ANSUR II `footlength` (271.2 / 246.3 mm).
pub const FOOT_LENGTH: f32 = 0.259;
/// Horizontal foot breadth, ANSUR II `footbreadthhorizontal` (101.9 / 92.7 mm).
pub const FOOT_BREADTH: f32 = 0.097;
/// Interpupillary breadth, ANSUR II `interpupillarybreadth` (64.0 / 61.7 mm); the eyeball centres
/// are this far apart (eyes looking at infinity).
pub const INTERPUPILLARY: f32 = 0.063;
/// Suprasternale (jugular notch) height, ANSUR II `suprasternaleheight` (1438.6 / 1329.7 mm).
pub const SUPRASTERNALE_HEIGHT: f32 = 1.384;
/// Cervicale (spinous process of C7) height, ANSUR II `cervicaleheight` (1517.3 / 1395.7 mm).
pub const CERVICALE_HEIGHT: f32 = 1.456;
/// Acromion height, ANSUR II `acromialheight` (1440.7 / 1335.1 mm).
pub const ACROMIAL_HEIGHT: f32 = 1.388;
/// Acromion to radiale, ANSUR II `acromionradialelength` (335.2 / 311.2 mm).
pub const ACROMION_RADIALE: f32 = 0.323;
/// Radiale to stylion (the forearm), ANSUR II `radialestylionlength` (267.9 / 241.3 mm).
pub const RADIALE_STYLION: f32 = 0.255;
/// Trochanterion height, ANSUR II `trochanterionheight` (900.9 / 845.4 mm): the hip joint centre's
/// height by convention.
pub const TROCHANTERION_HEIGHT: f32 = 0.873;
/// Lateral femoral epicondyle height, ANSUR II `lateralfemoralepicondyleheight` (491.7 / 465.9 mm):
/// the knee centre's height.
pub const KNEE_HEIGHT: f32 = 0.479;
/// Lateral malleolus height, ANSUR II `lateralmalleolusheight` (72.9 / 62.7 mm): the ankle centre's
/// height.
pub const MALLEOLUS_HEIGHT: f32 = 0.068;
/// Glenohumeral joint centre below the acromion: a design value (**unsourced**); with it the
/// shoulder-to-elbow distance is `ACROMION_RADIALE` less about this offset.
pub const ACROMION_ABOVE_SHOULDER: f32 = 0.020;

/// The ANSUR II variable each cited constant equals (sex midpoint, metres), for the data test.
pub const ANSUR_CITED: [(&str, f32); 18] = [
    ("stature", STATURE),
    ("tragiontopofhead", TRAGION_TO_VERTEX),
    ("biacromialbreadth", BIACROMIAL),
    ("hipbreadth", HIP_BREADTH),
    ("crotchheight", CROTCH_HEIGHT),
    ("handlength", HAND_LENGTH),
    ("handbreadth", HAND_BREADTH),
    ("footlength", FOOT_LENGTH),
    ("footbreadthhorizontal", FOOT_BREADTH),
    ("interpupillarybreadth", INTERPUPILLARY),
    ("suprasternaleheight", SUPRASTERNALE_HEIGHT),
    ("cervicaleheight", CERVICALE_HEIGHT),
    ("acromialheight", ACROMIAL_HEIGHT),
    ("acromionradialelength", ACROMION_RADIALE),
    ("radialestylionlength", RADIALE_STYLION),
    ("trochanterionheight", TROCHANTERION_HEIGHT),
    ("lateralfemoralepicondyleheight", KNEE_HEIGHT),
    ("lateralmalleolusheight", MALLEOLUS_HEIGHT),
];

/// The rest pose: arm abduction from the vertical, radians (an A-pose, 45 degrees; a pose, not
/// anatomy).
pub const ARM_ABDUCTION: f32 = 0.785;
/// Elbow flexion in the rest pose, radians.
pub const ELBOW_FLEXION: f32 = 0.21;

/// Anatomical points of the template (left side where sided), metres.
#[derive(Clone, Debug, PartialEq)]
pub struct Anatomy {
    /// Head frame origin: the midpoint between the tragions (ear canals).
    pub head_origin: Vec3,
    /// Eyeball centre (left).
    pub eye_center: Vec3,
    /// Eyeball radius.
    pub eye_radius: f32,
    /// Jaw pivot (midpoint between the temporomandibular joints).
    pub jaw_pivot: Vec3,
    /// Pelvis (root) joint.
    pub pelvis: Vec3,
    /// Spine joints from the lumbar to the upper thoracic.
    pub spine: [Vec3; 4],
    /// Neck joints (base, middle) and the head joint (atlanto-occipital).
    pub neck: [Vec3; 3],
    /// Sternoclavicular joint (left).
    pub clavicle: Vec3,
    /// Glenohumeral joint centre (left).
    pub shoulder: Vec3,
    /// Elbow centre (left).
    pub elbow: Vec3,
    /// Wrist centre (left).
    pub wrist: Vec3,
    /// Upper-arm axis direction (shoulder to elbow).
    pub upper_arm_dir: Vec3,
    /// Forearm axis direction (elbow to wrist).
    pub forearm_dir: Vec3,
    /// Hand frame: distal axis, palmar normal and radial (thumb) direction.
    pub hand_axes: [Vec3; 3],
    /// Hip joint centre (left).
    pub hip: Vec3,
    /// Knee centre (left).
    pub knee: Vec3,
    /// Ankle centre (left).
    pub ankle: Vec3,
    /// Foot axis (heel to toes, horizontal).
    pub foot_dir: Vec3,
}

/// Finger data: name stem, metacarpophalangeal joint in hand coordinates (radial, distal, palmar),
/// and the three phalanx lengths (proximal, middle, distal), metres. **Unsourced** design values,
/// except that the straight middle finger reaches [`HAND_LENGTH`] from the wrist (tested).
pub const FINGERS: [(&str, [f32; 3], [f32; 3]); 4] = [
    ("index", [0.024, 0.094, 0.002], [0.041, 0.024, 0.019]),
    ("middle", [0.004, 0.098, 0.002], [0.043, 0.027, 0.019]),
    ("ring", [-0.015, 0.094, 0.002], [0.042, 0.026, 0.019]),
    ("little", [-0.031, 0.086, 0.003], [0.034, 0.019, 0.017]),
];

/// Thumb data: carpometacarpal joint in hand coordinates and the metacarpal, proximal and distal
/// lengths.
pub const THUMB: ([f32; 3], [f32; 3]) = ([0.020, 0.022, 0.014], [0.044, 0.032, 0.026]);

impl Anatomy {
    /// The anatomy of the neutral template.
    pub fn neutral() -> Self {
        let head_origin = Vec3::new(0.0, STATURE - TRAGION_TO_VERTEX, 0.0);
        // The joint 20 mm below the acromion; its offset from the midline is unsourced.
        let shoulder = Vec3::new(0.163, ACROMIAL_HEIGHT - ACROMION_ABOVE_SHOULDER, -0.012);
        let (s, c) = ARM_ABDUCTION.sin_cos();
        let upper_arm_dir = Vec3::new(s, -c, 0.06).normalize();
        // Flexion bends the forearm forward about the elbow hinge (perpendicular to the arm, in the
        // horizontal-ish plane).
        let hinge = upper_arm_dir.cross(Vec3::Z).normalize();
        let forearm_dir = rotate(upper_arm_dir, hinge, -ELBOW_FLEXION).normalize();
        // Acromion-radiale less the acromion's offset above the joint (rounded down 3 mm).
        let elbow = shoulder + upper_arm_dir * 0.300;
        let wrist = elbow + forearm_dir * RADIALE_STYLION;
        // Neutral forearm rotation: the palm faces the body, the thumb points forward.
        let distal = forearm_dir;
        let radial = (Vec3::Z - distal * Vec3::Z.dot(distal)).normalize();
        let palmar = distal.cross(radial).normalize();
        let hip = Vec3::new(0.085, TROCHANTERION_HEIGHT, -0.008);
        let knee = Vec3::new(0.097, KNEE_HEIGHT, 0.004);
        let ankle = Vec3::new(0.106, MALLEOLUS_HEIGHT, -0.024);
        let toe_out = 0.12_f32;
        let foot_dir = Vec3::new(toe_out.sin(), 0.0, toe_out.cos());
        Self {
            head_origin,
            // Eye centre height and depth in the head frame and the eye radius are unsourced.
            eye_center: head_origin + Vec3::new(INTERPUPILLARY * 0.5, 0.011, 0.074),
            eye_radius: 0.0120,
            jaw_pivot: head_origin + Vec3::new(0.0, -0.006, 0.012),
            pelvis: Vec3::new(0.0, 0.965, -0.020),
            spine: [
                Vec3::new(0.0, 1.060, -0.048),
                Vec3::new(0.0, 1.160, -0.060),
                Vec3::new(0.0, 1.260, -0.062),
                Vec3::new(0.0, 1.360, -0.055),
            ],
            neck: [
                Vec3::new(0.0, 1.430, -0.045),
                Vec3::new(0.0, 1.490, -0.030),
                head_origin + Vec3::new(0.0, -0.020, -0.012),
            ],
            clavicle: Vec3::new(0.020, 1.390, 0.040),
            shoulder,
            elbow,
            wrist,
            upper_arm_dir,
            forearm_dir,
            hand_axes: [distal, palmar, radial],
            hip,
            knee,
            ankle,
            foot_dir,
        }
    }

    /// A point in hand coordinates (radial, distal, palmar) to world (left hand).
    pub fn hand_point(&self, h: [f32; 3]) -> Vec3 {
        let [distal, palmar, radial] = self.hand_axes;
        self.wrist + radial * h[0] + distal * h[1] + palmar * h[2]
    }
}

/// Mirrors a point or direction to the other side (`x -> -x`).
pub fn mirror(p: Vec3) -> Vec3 {
    Vec3::new(-p.x, p.y, p.z)
}

/// Applies a side: `side > 0` keeps the left-side point, otherwise mirrors it.
pub fn sided(p: Vec3, side: f32) -> Vec3 {
    if side > 0.0 { p } else { mirror(p) }
}

/// Rotates `v` about the unit axis `k` by `angle` (Rodrigues).
pub fn rotate(v: Vec3, k: Vec3, angle: f32) -> Vec3 {
    let (s, c) = angle.sin_cos();
    v * c + k.cross(v) * s + k * (k.dot(v) * (1.0 - c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anat_01_constants_match_the_ansur_ii_file() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/anthropometry/ansur2.json");
        let bytes = std::fs::read(path).unwrap();
        let value = humanart_io::json::parse(&bytes).unwrap();
        let root = value.node();
        let mid = root.field("sex_midpoint_mean").unwrap();
        for (id, constant) in ANSUR_CITED {
            let mm = mid.field(id).unwrap().f64().unwrap();
            // Rounded to the millimetre: within 0.5 mm.
            let diff = (f64::from(constant) * 1000.0 - mm).abs();
            assert!(diff <= 0.5 + 1e-3, "{id}: constant {constant} m against midpoint {mm} mm");
        }
        // The straight middle finger chain is the hand length.
        let (_, mcp, ph) = FINGERS[1];
        let chain = mcp[1] + ph.iter().sum::<f32>();
        assert!((chain - HAND_LENGTH).abs() < 1e-4, "{chain}");
        // The joint centres sit at the landmarks they are defined by.
        let a = Anatomy::neutral();
        assert!((a.hip.y - TROCHANTERION_HEIGHT).abs() < 1e-6 && (a.knee.y - KNEE_HEIGHT).abs() < 1e-6);
        assert!((a.ankle.y - MALLEOLUS_HEIGHT).abs() < 1e-6);
        assert!((a.wrist.distance(a.elbow) - RADIALE_STYLION).abs() < 1e-5);
        assert!((a.eye_center.x * 2.0 - INTERPUPILLARY).abs() < 1e-6);
        assert!((a.head_origin.y + TRAGION_TO_VERTEX - STATURE).abs() < 1e-6);
    }

    #[test]
    fn proportions_are_plausible() {
        let a = Anatomy::neutral();
        // The head is about one eighth of the stature less a little (canon 7.5 heads).
        let heads = STATURE / HEAD_HEIGHT;
        assert!((7.2..7.8).contains(&heads), "{heads}");
        // Wrist near mid-thigh height in an A-pose, hand frame orthonormal.
        assert!(a.wrist.y > 0.85 && a.wrist.y < 1.05, "{:?}", a.wrist);
        let [d, p, r] = a.hand_axes;
        assert!(d.dot(p).abs() < 1e-5 && d.dot(r).abs() < 1e-5 && p.dot(r).abs() < 1e-5);
        // The palm faces the body (palmar normal points to -X for the left hand).
        assert!(p.x < 0.0);
        assert!((a.elbow - a.shoulder).length() > 0.29);
    }
}
