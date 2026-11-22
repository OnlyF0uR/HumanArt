//! The document's eye, mouth and hair-colour materials (`todo.md` 2.3): what the registry's
//! `eye.*`, `teeth.*`, `gums.*`, `tongue.*` and `hair.*` material targets set, gathered in one
//! plain value the views compare and turn into the renderer's inputs.
//!
//! [`MaterialTarget::from_target`] names the destination of a registry material target;
//! [`MaterialValues`] holds the document's values as the registry gives them (normalised 0..1,
//! standardised coefficients, linear Rec.709 colours). [`MaterialValues::eye_params`] and
//! [`MaterialValues::mouth_params`] turn them into the eye and mouth shading parameters
//! ([`EyeParams`], [`MouthParams`]), [`MaterialValues::hair_melanin`] into the fibre pigments.
//! Nothing allocates: a view compares two values and rebuilds its feature-table entries only
//! when they differ.
//!
//! The mappings, with their sources:
//! - Iris colour is the ciliary (outer) zone; the pupillary (inner) zone is the iris colour
//!   lightened by [`PUPILLARY_ZONE`] (the ratio of the default iris layers, `IrisParams::default`)
//!   unless central heterochromia is on, then it is the central colour.
//! - Collarette 0..1 places the ridge from 0.2 to 0.5 of the way from the pupil margin to the iris
//!   root (a farther ridge shows a wider inner zone; 0.5 gives about the default 0.32 + 0.03).
//! - Limbal ring width 0..1 is 0..0.3 of the iris radius (the range `EyeParams::validate` allows).
//! - Iris size is a coefficient (±3): the limbus radius 5.85 mm + 0.3 mm per unit, clamped to
//!   5.2..6.4 mm (visible iris diameter 11.71 ± 0.42 mm, Rüfer et al. 2005).
//! - Corneal bulge 0..1 is the anterior corneal radius 8.2..7.4 mm (default 7.8 mm, Le Grand;
//!   smaller bulges more).
//! - Iris depth 0..1 puts the iris plane 0.3..1.62 mm behind the plane of the limbus (default
//!   0.96 mm, which with the default cornea gives the 3.6 mm anterior chamber of the schematic
//!   eyes); the depth from the corneal apex is that plus the cornea's sagitta, clamped to the
//!   2.0..4.5 mm `EyeParams::validate` allows, so every combination stays a valid eye.
//! - Teeth colour is the mean crown colour: the cervical third is it times [`TEETH_CERVICAL`], the
//!   incisal edge it desaturated toward its luminance (85 %) and darkened to 86 % (the ratios of
//!   the default `TeethParams` to the registry default colour).
//! - Hair eumelanin `e` and pheomelanin `p` (0..1) are `10 e²` and `2 p²` on the pbrt scale, so
//!   the registry defaults (0.5, 0.2) give the dark brown of the default groom (2.5, 0.08), 0.17
//!   blonde (0.3) and 1 black (10). An artistic mapping, chosen so the slider is even to the eye.
//!
//! Not mapped (no field to carry them): the corneal and aqueous refractive indices and the tear
//! film (`EyeParams` has none and the eye shader uses constants), and the captured iris textures
//! (nothing reads the blob). [`MaterialTarget::from_target`] returns `None` for them.

use crate::eye::params::{EyeParams, IrisParams};
use crate::mouth::params::MouthParams;

/// An eye side.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    /// The character's left (`L`, `+X`).
    Left = 0,
    /// The character's right (`R`).
    Right = 1,
}

impl Side {
    /// Both sides, left first.
    pub const BOTH: [Self; 2] = [Self::Left, Self::Right];

    fn of_letter(letter: &str) -> Option<Self> {
        match letter {
            "L" => Some(Self::Left),
            "R" => Some(Self::Right),
            _ => None,
        }
    }
}

/// A scalar eye value of the registry (both eyes share it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EyeScalar {
    /// `eye.iris-fibres`, 0..1.
    Fibres = 0,
    /// `eye.iris-crypts`, 0..1.
    Crypts = 1,
    /// `eye.iris-collarette`, 0..1.
    Collarette = 2,
    /// `eye.limbal-width`, 0..1.
    LimbalWidth = 3,
    /// `eye.limbal-darkness`, 0..1.
    LimbalDarkness = 4,
    /// `eye.iris-radius`, a coefficient (±3).
    IrisSize = 5,
    /// `eye.iris-depth`, 0..1.
    IrisDepth = 6,
    /// `eye.pupil-dilation`, 0..1.
    Dilation = 7,
    /// `eye.sclera-veins`, 0..1.
    ScleraVeins = 8,
    /// `eye.sclera-redness`, 0..1.
    ScleraRedness = 9,
    /// `eye.sclera-yellowing`, 0..1.
    ScleraYellowing = 10,
    /// `eye.cornea-bulge`, 0..1.
    CorneaBulge = 11,
}

/// Number of [`EyeScalar`]s.
pub const EYE_SCALARS: usize = 12;

/// A scalar mouth value of the registry, 0..1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MouthScalar {
    /// `teeth.translucency`.
    TeethTranslucency = 0,
    /// `teeth.wear`.
    TeethWear = 1,
    /// `teeth.stain`.
    TeethStain = 2,
    /// `tongue.coating`.
    TongueCoating = 3,
}

/// Number of [`MouthScalar`]s.
pub const MOUTH_SCALARS: usize = 4;

/// A colour of the registry, linear Rec.709.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialColor {
    /// `eye.iris-color-L` / `-R`.
    Iris(Side),
    /// `eye.iris-central-color-L` / `-R`.
    IrisCentral(Side),
    /// `eye.sclera-tint`.
    Sclera,
    /// `teeth.color`.
    Teeth,
    /// `gums.color`.
    Gums,
    /// `tongue.color`.
    Tongue,
}

/// A hair colour value of the registry, 0..1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairPigment {
    /// `hair.eumelanin`.
    Eumelanin = 0,
    /// `hair.pheomelanin`.
    Pheomelanin = 1,
    /// `hair.grey-fraction`.
    Grey = 2,
}

/// Where a registry material target goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialTarget {
    /// A colour.
    Color(MaterialColor),
    /// An eye scalar.
    Eye(EyeScalar),
    /// A mouth scalar.
    Mouth(MouthScalar),
    /// A hair pigment.
    Hair(HairPigment),
    /// `eye.heterochromia`, a choice (`none`, `central`).
    Heterochromia,
}

impl MaterialTarget {
    /// The destination of a registry material target, `None` for targets these materials do not
    /// read (skin targets, and the unmapped eye targets of the module documentation).
    pub fn from_target(target: &str) -> Option<Self> {
        use EyeScalar as E;
        use MaterialColor as C;
        if let Some(side) = target.strip_prefix("eye.iris-color-").and_then(Side::of_letter) {
            return Some(Self::Color(C::Iris(side)));
        }
        if let Some(side) = target.strip_prefix("eye.iris-central-color-").and_then(Side::of_letter) {
            return Some(Self::Color(C::IrisCentral(side)));
        }
        Some(match target {
            "eye.sclera-tint" => Self::Color(C::Sclera),
            "teeth.color" => Self::Color(C::Teeth),
            "gums.color" => Self::Color(C::Gums),
            "tongue.color" => Self::Color(C::Tongue),
            "eye.heterochromia" => Self::Heterochromia,
            "eye.iris-fibres" => Self::Eye(E::Fibres),
            "eye.iris-crypts" => Self::Eye(E::Crypts),
            "eye.iris-collarette" => Self::Eye(E::Collarette),
            "eye.limbal-width" => Self::Eye(E::LimbalWidth),
            "eye.limbal-darkness" => Self::Eye(E::LimbalDarkness),
            "eye.iris-radius" => Self::Eye(E::IrisSize),
            "eye.iris-depth" => Self::Eye(E::IrisDepth),
            "eye.pupil-dilation" => Self::Eye(E::Dilation),
            "eye.sclera-veins" => Self::Eye(E::ScleraVeins),
            "eye.sclera-redness" => Self::Eye(E::ScleraRedness),
            "eye.sclera-yellowing" => Self::Eye(E::ScleraYellowing),
            "eye.cornea-bulge" => Self::Eye(E::CorneaBulge),
            "teeth.translucency" => Self::Mouth(MouthScalar::TeethTranslucency),
            "teeth.wear" => Self::Mouth(MouthScalar::TeethWear),
            "teeth.stain" => Self::Mouth(MouthScalar::TeethStain),
            "tongue.coating" => Self::Mouth(MouthScalar::TongueCoating),
            "hair.eumelanin" => Self::Hair(HairPigment::Eumelanin),
            "hair.pheomelanin" => Self::Hair(HairPigment::Pheomelanin),
            "hair.grey-fraction" => Self::Hair(HairPigment::Grey),
            _ => return None,
        })
    }
}

/// The pupillary zone's colour over the iris colour without central heterochromia (the default
/// iris layers' inner over outer colour).
pub const PUPILLARY_ZONE: [f32; 3] = [0.34 / 0.19, 0.20 / 0.115, 0.075 / 0.055];
/// The cervical third's colour over the mean crown colour.
pub const TEETH_CERVICAL: [f32; 3] = [0.64 / 0.78, 0.57 / 0.72, 0.45 / 0.6];
/// The eyeball radius the eye parameters assume, millimetres (`EyeParams::default`).
const EYEBALL_MM: f32 = 12.0;

/// The document's eye, mouth and hair-colour values. See the module documentation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialValues {
    /// Iris colour per side.
    pub iris: [[f32; 3]; 2],
    /// Central iris colour per side.
    pub iris_central: [[f32; 3]; 2],
    /// Whether central heterochromia is on.
    pub central_heterochromia: bool,
    /// Sclera tint.
    pub sclera: [f32; 3],
    /// Eye scalars, by [`EyeScalar`].
    pub eye: [f32; EYE_SCALARS],
    /// Teeth, gum and tongue colours.
    pub teeth: [f32; 3],
    /// Gum colour.
    pub gums: [f32; 3],
    /// Tongue colour.
    pub tongue: [f32; 3],
    /// Mouth scalars, by [`MouthScalar`].
    pub mouth: [f32; MOUTH_SCALARS],
    /// Hair pigments, by [`HairPigment`], 0..1.
    pub hair: [f32; 3],
}

impl Default for MaterialValues {
    /// The registry's defaults (`data/template/params/eyes.json`, `mouth.json`, `hair.json`). A
    /// view reading a registry overwrites every value the registry has a parameter for.
    fn default() -> Self {
        let mut eye = [0.5; EYE_SCALARS];
        eye[EyeScalar::Crypts as usize] = 0.3;
        eye[EyeScalar::IrisSize as usize] = 0.0;
        eye[EyeScalar::ScleraVeins as usize] = 0.2;
        eye[EyeScalar::ScleraRedness as usize] = 0.1;
        eye[EyeScalar::ScleraYellowing as usize] = 0.1;
        Self {
            iris: [[0.1, 0.05, 0.025]; 2],
            iris_central: [[0.1, 0.05, 0.025]; 2],
            central_heterochromia: false,
            sclera: [0.8, 0.77, 0.72],
            eye,
            teeth: [0.78, 0.72, 0.6],
            gums: [0.55, 0.2, 0.2],
            tongue: [0.55, 0.22, 0.22],
            mouth: [0.5, 0.2, 0.1, 0.2],
            hair: [0.5, 0.2, 0.0],
        }
    }
}

fn unit(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

fn unit3(c: [f32; 3]) -> [f32; 3] {
    c.map(unit)
}

impl MaterialValues {
    /// Sets a scalar value (non-finite values are ignored). Returns false when `target` is a
    /// colour or a choice.
    pub fn set_scalar(&mut self, target: MaterialTarget, value: f32) -> bool {
        if !value.is_finite() {
            return matches!(target, MaterialTarget::Eye(_) | MaterialTarget::Mouth(_) | MaterialTarget::Hair(_));
        }
        match target {
            MaterialTarget::Eye(EyeScalar::IrisSize) => self.eye[EyeScalar::IrisSize as usize] = value.clamp(-3.0, 3.0),
            MaterialTarget::Eye(e) => self.eye[e as usize] = unit(value),
            MaterialTarget::Mouth(m) => self.mouth[m as usize] = unit(value),
            MaterialTarget::Hair(h) => self.hair[h as usize] = unit(value),
            MaterialTarget::Color(_) | MaterialTarget::Heterochromia => return false,
        }
        true
    }

    /// Sets a colour (channels clamped to 0..1; a non-finite channel leaves the colour as it is).
    /// Returns false when `target` is not a colour.
    pub fn set_color(&mut self, target: MaterialTarget, rgb: [f32; 3]) -> bool {
        let MaterialTarget::Color(c) = target else { return false };
        if rgb.iter().all(|v| v.is_finite()) {
            let slot = match c {
                MaterialColor::Iris(s) => &mut self.iris[s as usize],
                MaterialColor::IrisCentral(s) => &mut self.iris_central[s as usize],
                MaterialColor::Sclera => &mut self.sclera,
                MaterialColor::Teeth => &mut self.teeth,
                MaterialColor::Gums => &mut self.gums,
                MaterialColor::Tongue => &mut self.tongue,
            };
            *slot = unit3(rgb);
        }
        true
    }

    /// Sets a choice. Returns false when `target` takes no choice or `choice` is not one of its
    /// choices.
    pub fn set_choice(&mut self, target: MaterialTarget, choice: &str) -> bool {
        match (target, choice) {
            (MaterialTarget::Heterochromia, "none") => self.central_heterochromia = false,
            (MaterialTarget::Heterochromia, "central") => self.central_heterochromia = true,
            _ => return false,
        }
        true
    }

    /// The eye parameters of `side` (see the module documentation). Always valid
    /// (`EyeParams::validate`); should a combination ever fail, the geometry falls back to the
    /// default eye's and the appearance is kept.
    #[must_use]
    pub fn eye_params(&self, side: Side) -> EyeParams {
        let e = |s: EyeScalar| self.eye[s as usize];
        let outer = self.iris[side as usize];
        let inner = if self.central_heterochromia {
            self.iris_central[side as usize]
        } else {
            [0, 1, 2].map(|i| unit(outer[i] * PUPILLARY_ZONE[i]))
        };
        let defaults = EyeParams::default();
        let iris = IrisParams {
            outer_color: outer,
            inner_color: inner,
            collarette: 0.2 + 0.3 * e(EyeScalar::Collarette),
            fibres: e(EyeScalar::Fibres),
            crypts: e(EyeScalar::Crypts),
            limbal_width: 0.3 * e(EyeScalar::LimbalWidth),
            limbal_darkness: e(EyeScalar::LimbalDarkness),
            seed: side as u32 + 1,
        };
        let sclera = crate::eye::params::ScleraParams {
            color: self.sclera,
            vein_density: e(EyeScalar::ScleraVeins),
            redness: e(EyeScalar::ScleraRedness),
            yellowing: e(EyeScalar::ScleraYellowing),
            ..defaults.sclera
        };
        let limbus = (5.85 + 0.3 * e(EyeScalar::IrisSize)).clamp(5.2, 6.4);
        let cornea = 8.2 - 0.8 * e(EyeScalar::CorneaBulge);
        // The cornea's sagitta over the limbus: the depth of the limbus plane behind the apex.
        let sagitta = cornea - (cornea * cornea - limbus * limbus).max(0.0).sqrt();
        let depth = (sagitta + 0.3 + 1.32 * e(EyeScalar::IrisDepth)).clamp(2.0, 4.5);
        let shaped = EyeParams {
            eyeball_radius_mm: EYEBALL_MM,
            limbus_radius_mm: limbus,
            cornea_radius_mm: cornea,
            iris_depth_mm: depth,
            dilation: e(EyeScalar::Dilation),
            iris,
            sclera,
            ..defaults
        };
        if shaped.validate().is_ok() {
            shaped
        } else {
            EyeParams { dilation: shaped.dilation, iris, sclera, ..defaults }
        }
    }

    /// The teeth, gum and tongue parameters (see the module documentation); the tooth
    /// proportions and the gum line are shape deltas, so they keep their defaults here.
    #[must_use]
    pub fn mouth_params(&self) -> MouthParams {
        let d = MouthParams::default();
        let m = |s: MouthScalar| self.mouth[s as usize];
        let c = self.teeth;
        let luma = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let teeth = crate::mouth::params::TeethParams {
            gingival_color: [0, 1, 2].map(|i| unit(c[i] * TEETH_CERVICAL[i])),
            incisal_color: c.map(|v| unit((v + (luma - v) * 0.85) * 0.86)),
            translucency: m(MouthScalar::TeethTranslucency),
            wear: m(MouthScalar::TeethWear),
            stain: m(MouthScalar::TeethStain),
            ..d.teeth
        };
        let gums = crate::mouth::params::GumParams { color: self.gums, ..d.gums };
        let tongue = crate::mouth::params::TongueParams {
            color: self.tongue,
            coating: m(MouthScalar::TongueCoating),
            ..d.tongue
        };
        MouthParams { teeth, gums, tongue }
    }

    /// The hair fibres' eumelanin and pheomelanin on the pbrt scale (see the module
    /// documentation).
    #[must_use]
    pub fn hair_melanin(&self) -> (f32, f32) {
        let (e, p) = (self.hair[HairPigment::Eumelanin as usize], self.hair[HairPigment::Pheomelanin as usize]);
        (10.0 * e * e, 2.0 * p * p)
    }

    /// The share of unpigmented (grey) hair, 0..1.
    #[must_use]
    pub fn hair_grey(&self) -> f32 {
        self.hair[HairPigment::Grey as usize]
    }

    /// Whether the eye parameters differ between `self` and `other`.
    #[must_use]
    pub fn eyes_differ(&self, other: &Self) -> bool {
        self.iris != other.iris
            || self.iris_central != other.iris_central
            || self.central_heterochromia != other.central_heterochromia
            || self.sclera != other.sclera
            || self.eye != other.eye
    }

    /// Whether the mouth parameters differ between `self` and `other`.
    #[must_use]
    pub fn mouth_differs(&self, other: &Self) -> bool {
        self.teeth != other.teeth || self.gums != other.gums || self.tongue != other.tongue || self.mouth != other.mouth
    }

    /// Whether the hair colour differs between `self` and `other`.
    #[must_use]
    pub fn hair_differs(&self, other: &Self) -> bool {
        self.hair != other.hair
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn todo_2_3_targets_map_to_their_values() {
        assert_eq!(
            MaterialTarget::from_target("eye.iris-color-R"),
            Some(MaterialTarget::Color(MaterialColor::Iris(Side::Right)))
        );
        assert_eq!(
            MaterialTarget::from_target("eye.iris-central-color-L"),
            Some(MaterialTarget::Color(MaterialColor::IrisCentral(Side::Left)))
        );
        assert_eq!(MaterialTarget::from_target("eye.iris-color-X"), None);
        assert_eq!(MaterialTarget::from_target("eye.cornea-ior"), None, "no field carries it");
        assert_eq!(MaterialTarget::from_target("skin.melanin"), None);
        assert_eq!(MaterialTarget::from_target("hair.grey-fraction"), Some(MaterialTarget::Hair(HairPigment::Grey)));
        let mut v = MaterialValues::default();
        assert!(v.set_color(MaterialTarget::Color(MaterialColor::Iris(Side::Right)), [0.1, 0.3, 2.0]));
        assert_eq!(v.iris[1], [0.1, 0.3, 1.0]);
        assert!(!v.set_color(MaterialTarget::Eye(EyeScalar::Crypts), [0.0; 3]));
        assert!(!v.set_scalar(MaterialTarget::Heterochromia, 1.0));
        assert!(v.set_choice(MaterialTarget::Heterochromia, "central") && v.central_heterochromia);
        assert!(!v.set_choice(MaterialTarget::Heterochromia, "sectoral"));
        assert!(v.set_scalar(MaterialTarget::Eye(EyeScalar::Dilation), f32::NAN));
        assert_eq!(v.eye[EyeScalar::Dilation as usize], 0.5, "non-finite values are ignored");
    }

    #[test]
    fn todo_2_3_defaults_give_the_default_eye_and_mouth() {
        let v = MaterialValues::default();
        let e = v.eye_params(Side::Left);
        let d = EyeParams::default();
        assert!((e.limbus_radius_mm - d.limbus_radius_mm).abs() < 1e-6);
        assert!((e.cornea_radius_mm - d.cornea_radius_mm).abs() < 1e-6);
        assert!((e.iris_depth_mm - d.iris_depth_mm).abs() < 0.01, "{}", e.iris_depth_mm);
        assert_ne!(v.eye_params(Side::Left).iris.seed, v.eye_params(Side::Right).iris.seed, "each eye its own pattern");
        let m = v.mouth_params();
        let t = MouthParams::default().teeth;
        for i in 0..3 {
            assert!((m.teeth.gingival_color[i] - t.gingival_color[i]).abs() < 0.01);
            assert!((m.teeth.incisal_color[i] - t.incisal_color[i]).abs() < 0.02, "{:?}", m.teeth.incisal_color);
        }
        m.validate().unwrap();
        let (eu, pheo) = v.hair_melanin();
        assert!((eu - 2.5).abs() < 1e-6 && (pheo - 0.08).abs() < 1e-6);
    }

    /// Every corner of the eye's shape controls gives a valid eye with the asked-for geometry
    /// (no fallback), and every value changes the parameters.
    #[test]
    fn todo_2_3_every_eye_value_moves_and_every_combination_is_valid() {
        for size in [-3.0, -1.0, 0.0, 1.5, 3.0] {
            for bulge in [0.0, 0.5, 1.0] {
                for depth in [0.0, 0.5, 1.0] {
                    for dilation in [0.0, 1.0] {
                        let mut v = MaterialValues::default();
                        v.set_scalar(MaterialTarget::Eye(EyeScalar::IrisSize), size);
                        v.set_scalar(MaterialTarget::Eye(EyeScalar::CorneaBulge), bulge);
                        v.set_scalar(MaterialTarget::Eye(EyeScalar::IrisDepth), depth);
                        v.set_scalar(MaterialTarget::Eye(EyeScalar::Dilation), dilation);
                        let e = v.eye_params(Side::Right);
                        e.validate().unwrap();
                        assert!((e.limbus_radius_mm - (5.85 + 0.3 * size).clamp(5.2, 6.4)).abs() < 1e-5);
                        assert!((e.cornea_radius_mm - (8.2 - 0.8 * bulge)).abs() < 1e-5, "no fallback");
                    }
                }
            }
        }
        let base = MaterialValues::default();
        let scalars = [
            EyeScalar::Fibres,
            EyeScalar::Crypts,
            EyeScalar::Collarette,
            EyeScalar::LimbalWidth,
            EyeScalar::LimbalDarkness,
            EyeScalar::IrisSize,
            EyeScalar::IrisDepth,
            EyeScalar::Dilation,
            EyeScalar::ScleraVeins,
            EyeScalar::ScleraRedness,
            EyeScalar::ScleraYellowing,
            EyeScalar::CorneaBulge,
        ];
        assert_eq!(scalars.len(), EYE_SCALARS);
        for s in scalars {
            let mut v = base;
            v.set_scalar(MaterialTarget::Eye(s), 1.0);
            assert!(v.eyes_differ(&base));
            assert_ne!(v.eye_params(Side::Left), base.eye_params(Side::Left), "{s:?}");
        }
        let mut v = base;
        v.set_color(MaterialTarget::Color(MaterialColor::IrisCentral(Side::Left)), [0.5, 0.4, 0.1]);
        assert_eq!(v.eye_params(Side::Left), base.eye_params(Side::Left), "the central colour waits for heterochromia");
        v.set_choice(MaterialTarget::Heterochromia, "central");
        assert_eq!(v.eye_params(Side::Left).iris.inner_color, [0.5, 0.4, 0.1]);
    }

    #[test]
    fn todo_2_3_mouth_and_hair_values_move() {
        let base = MaterialValues::default();
        for s in [
            MouthScalar::TeethTranslucency,
            MouthScalar::TeethWear,
            MouthScalar::TeethStain,
            MouthScalar::TongueCoating,
        ] {
            let mut v = base;
            v.set_scalar(MaterialTarget::Mouth(s), 0.9);
            assert!(v.mouth_differs(&base));
            assert_ne!(v.mouth_params(), base.mouth_params(), "{s:?}");
            v.mouth_params().validate().unwrap();
        }
        for c in [MaterialColor::Teeth, MaterialColor::Gums, MaterialColor::Tongue] {
            let mut v = base;
            v.set_color(MaterialTarget::Color(c), [1.0, 1.0, 1.0]);
            assert_ne!(v.mouth_params(), base.mouth_params(), "{c:?}");
            v.mouth_params().validate().unwrap();
        }
        let mut v = base;
        v.set_scalar(MaterialTarget::Hair(HairPigment::Eumelanin), 1.0);
        v.set_scalar(MaterialTarget::Hair(HairPigment::Grey), 0.4);
        assert!(v.hair_differs(&base) && !v.eyes_differ(&base) && !v.mouth_differs(&base));
        assert_eq!(v.hair_melanin().0, 10.0);
        assert_eq!(v.hair_grey(), 0.4);
    }
}
