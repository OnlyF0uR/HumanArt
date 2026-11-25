//! The document's global skin values (todo 1.4): what the registry's `skin.*` material targets
//! set, gathered in one plain value the views compare and apply.
//!
//! [`SkinTarget::from_target`] names the destination of a registry material target;
//! [`SkinValues`] holds the base attributes (pigment, tissue, oiliness, wetness), the micro-detail
//! inputs (pores, fine lines), the animated flush, the surface's moisture, the vellus density, the
//! lips' colour and moisture, the neck's resting lines and the makeup ([`super::makeup`]). [`SkinValues::resolved_globals`] turns
//! them into what the compositor reads: the base values and the zone table with the measured
//! regional surface ([`super::regional`]), the moisture's roughness and the lip colour's pigment
//! offsets; [`SkinValues::micro_detail`] into the micro-detail generator's settings;
//! [`SkinValues::fuzz_scale`] and [`SkinValues::lip_coat`] into the surface's vellus sheen and lip
//! coat (`humanart_render::surface::SurfaceSettings`, set by the view).
//!
//! The mappings (todo 2.3; artistic, each chosen so the registry default leaves the look as it
//! was):
//! - moisture `m` (stratum corneum hydration; dry skin scatters more at its surface): the primary
//!   lobe's roughness gains `(0.5 − m) ·` [`MOISTURE_ROUGHNESS`];
//! - vellus density `v`: the vellus sheen times `2 v`;
//! - lip moisture `m`: the lip coat times `2 m` (at most 1) and its roughness times `1.5 − m`;
//! - lip colour: the melanin, eumelanin ratio and haemoglobin that reproduce it under the pigment
//!   model (`PigmentModel::fit_lab`, from the lips' composited pigments) minus those that reproduce
//!   [`DEFAULT_LIPS`] are added to the lip zone's offsets, so the default colour keeps the measured
//!   zone table and another colour moves the lips' pigments toward it.

use humanart_math::color::lab::rgb_to_lab;

use super::channels::{ATTRIBUTE_COUNT, Attribute, BaseValues};
use super::makeup::{Makeup, MakeupColor, MakeupInput};
use super::microdetail::MicroDetailParams;
use super::pigment::{Mapping, PigmentModel};
use super::regional::apply_measured_surface;
use super::resolve::pigment_of;
use super::zones::{Zone, ZoneTable};

/// A micro-detail input of the registry (normalised 0..1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DetailInput {
    /// `skin.pore-density`.
    PoreDensity = 0,
    /// `skin.pore-depth`.
    PoreDepth = 1,
    /// `skin.pore-size`.
    PoreSize = 2,
    /// `skin.pore-elongation`.
    PoreElongation = 3,
    /// `skin.fine-line-depth`.
    LineDepth = 4,
    /// `skin.fine-line-scale`.
    LineScale = 5,
}

/// Number of [`DetailInput`]s.
pub const DETAIL_INPUTS: usize = 6;

/// A resting (age) wrinkle depth of the registry, 0..1: the share of the expression wrinkle that
/// stays when the face is at rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RestingWrinkle {
    /// `skin.wrinkle-forehead` (driver `forehead.raise`).
    Forehead = 0,
    /// `skin.wrinkle-glabella` (drivers `glabella.L`, `glabella.R`).
    Glabella = 1,
    /// `skin.wrinkle-crows-feet-L` (driver `crows-feet.L`).
    CrowsFeetLeft = 2,
    /// `skin.wrinkle-crows-feet-R` (driver `crows-feet.R`).
    CrowsFeetRight = 3,
    /// `skin.wrinkle-nasolabial-L` (driver `nasolabial.L`).
    NasolabialLeft = 4,
    /// `skin.wrinkle-nasolabial-R` (driver `nasolabial.R`).
    NasolabialRight = 5,
}

/// Number of [`RestingWrinkle`]s.
pub const RESTING_WRINKLES: usize = 6;

impl RestingWrinkle {
    /// The resting wrinkle a wrinkle driver (by its stable name) shows.
    pub fn of_driver(driver: &str) -> Option<Self> {
        Some(match driver {
            "forehead.raise" => Self::Forehead,
            "glabella.L" | "glabella.R" => Self::Glabella,
            "crows-feet.L" => Self::CrowsFeetLeft,
            "crows-feet.R" => Self::CrowsFeetRight,
            "nasolabial.L" => Self::NasolabialLeft,
            "nasolabial.R" => Self::NasolabialRight,
            _ => return None,
        })
    }
}

/// Where a registry material target goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SkinTarget {
    /// A base attribute of the compositor.
    Base(Attribute),
    /// A micro-detail input.
    Detail(DetailInput),
    /// The animated flush `skin.flush` (a shading-time value).
    Flush,
    /// A resting wrinkle depth (added to its drivers' fold every frame).
    Resting(RestingWrinkle),
    /// The amount of age spots `skin.age-spots` (solar lentigines, a generated layer).
    AgeSpots,
    /// The surface's moisture `skin.moisture`.
    Moisture,
    /// The vellus density `skin.vellus-density`.
    Vellus,
    /// The neck's resting lines `skin.wrinkle-neck` (a generated layer).
    NeckWrinkles,
    /// The lips' colour `lips.color` (linear Rec.709, [`SkinValues::set_color`]).
    LipColor,
    /// The lips' moisture `lips.moisture`.
    LipMoisture,
    /// A makeup amount or shape (`makeup.liner`, ...; a generated layer).
    Makeup(MakeupInput),
    /// A makeup colour (`makeup.lips-color`, ...; [`SkinValues::set_color`]).
    MakeupColor(MakeupColor),
}

impl SkinTarget {
    /// The destination of a registry material target, `None` for targets the skin does not read
    /// (wrinkle maps, generator layers and maps are applied elsewhere).
    pub fn from_target(target: &str) -> Option<Self> {
        if let Some(a) = BaseValues::attribute_of_target(target) {
            return Some(Self::Base(a));
        }
        if let Some(i) = MakeupInput::from_target(target) {
            return Some(Self::Makeup(i));
        }
        if let Some(c) = MakeupColor::from_target(target) {
            return Some(Self::MakeupColor(c));
        }
        Some(match target {
            "skin.pore-density" => Self::Detail(DetailInput::PoreDensity),
            "skin.pore-depth" => Self::Detail(DetailInput::PoreDepth),
            "skin.pore-size" => Self::Detail(DetailInput::PoreSize),
            "skin.pore-elongation" => Self::Detail(DetailInput::PoreElongation),
            "skin.fine-line-depth" => Self::Detail(DetailInput::LineDepth),
            "skin.fine-line-scale" => Self::Detail(DetailInput::LineScale),
            "skin.flush" => Self::Flush,
            "skin.age-spots" => Self::AgeSpots,
            "skin.moisture" => Self::Moisture,
            "skin.vellus-density" => Self::Vellus,
            "skin.wrinkle-neck" => Self::NeckWrinkles,
            "lips.color" => Self::LipColor,
            "lips.moisture" => Self::LipMoisture,
            "skin.wrinkle-forehead" => Self::Resting(RestingWrinkle::Forehead),
            "skin.wrinkle-glabella" => Self::Resting(RestingWrinkle::Glabella),
            "skin.wrinkle-crows-feet-L" => Self::Resting(RestingWrinkle::CrowsFeetLeft),
            "skin.wrinkle-crows-feet-R" => Self::Resting(RestingWrinkle::CrowsFeetRight),
            "skin.wrinkle-nasolabial-L" => Self::Resting(RestingWrinkle::NasolabialLeft),
            "skin.wrinkle-nasolabial-R" => Self::Resting(RestingWrinkle::NasolabialRight),
            _ => return None,
        })
    }
}

/// The document's global skin values. See the module documentation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkinValues {
    /// Base attributes (only the registry-driven ones change; the rest keep their defaults).
    pub base: BaseValues,
    /// Micro-detail inputs, by [`DetailInput`], 0..1.
    pub detail: [f32; DETAIL_INPUTS],
    /// Animated flush, 0..1.
    pub flush: f32,
    /// Resting wrinkle depths, by [`RestingWrinkle`], 0..1.
    pub resting: [f32; RESTING_WRINKLES],
    /// Amount of age spots, 0..1.
    pub age_spots: f32,
    /// Surface moisture, 0..1.
    pub moisture: f32,
    /// Vellus density, 0..1.
    pub vellus: f32,
    /// Depth of the neck's resting lines, 0..1.
    pub neck_wrinkles: f32,
    /// Lip colour, linear Rec.709.
    pub lips: [f32; 3],
    /// Lip moisture, 0..1.
    pub lip_moisture: f32,
    /// Makeup (bare by default).
    pub makeup: Makeup,
}

/// The registry's default lip colour (`mouth.lips.color`), which leaves the zone table as it is.
pub const DEFAULT_LIPS: [f32; 3] = [0.45, 0.18, 0.16];
/// Primary roughness added at moisture 0 and taken away at 1, per unit of `0.5 − moisture`.
pub const MOISTURE_ROUGHNESS: f32 = 0.12;

impl Default for SkinValues {
    /// [`BaseValues::default`], the micro-detail defaults of [`MicroDetailParams::default`] and no
    /// flush. A view reading a registry overwrites every value the registry has a parameter for.
    fn default() -> Self {
        Self {
            base: BaseValues::default(),
            detail: [0.5, 0.5, 0.5, 0.3, 0.4, 0.5],
            flush: 0.0,
            resting: [0.0; RESTING_WRINKLES],
            age_spots: 0.0,
            moisture: 0.5,
            vellus: 0.5,
            neck_wrinkles: 0.0,
            lips: DEFAULT_LIPS,
            lip_moisture: 0.5,
            makeup: Makeup::default(),
        }
    }
}

impl SkinValues {
    /// Sets one value (clamped to its range; a non-finite value is ignored).
    pub fn set(&mut self, target: SkinTarget, value: f32) {
        if !value.is_finite() {
            return;
        }
        match target {
            SkinTarget::Base(a) => self.base.set(a, value),
            SkinTarget::Detail(d) => self.detail[d as usize] = value.clamp(0.0, 1.0),
            SkinTarget::Flush => self.flush = value.clamp(0.0, 1.0),
            SkinTarget::Resting(r) => self.resting[r as usize] = value.clamp(0.0, 1.0),
            SkinTarget::AgeSpots => self.age_spots = value.clamp(0.0, 1.0),
            SkinTarget::Moisture => self.moisture = value.clamp(0.0, 1.0),
            SkinTarget::Vellus => self.vellus = value.clamp(0.0, 1.0),
            SkinTarget::NeckWrinkles => self.neck_wrinkles = value.clamp(0.0, 1.0),
            SkinTarget::LipMoisture => self.lip_moisture = value.clamp(0.0, 1.0),
            SkinTarget::Makeup(i) => self.makeup.set(i, value),
            SkinTarget::LipColor | SkinTarget::MakeupColor(_) => {}
        }
    }

    /// Sets a colour (channels clamped to 0..1; a non-finite channel leaves it as it is). Only
    /// [`SkinTarget::LipColor`] and [`SkinTarget::MakeupColor`] take one; false for every other
    /// target.
    pub fn set_color(&mut self, target: SkinTarget, rgb: [f32; 3]) -> bool {
        if let SkinTarget::MakeupColor(c) = target {
            self.makeup.set_color(c, rgb);
            return true;
        }
        if target != SkinTarget::LipColor {
            return false;
        }
        if rgb.iter().all(|v| v.is_finite()) {
            self.lips = rgb.map(|v| v.clamp(0.0, 1.0));
        }
        true
    }

    /// The factor on the surface's vellus sheen (`SurfaceSettings::fuzz`), 0..2.
    #[must_use]
    pub fn fuzz_scale(&self) -> f32 {
        2.0 * self.vellus
    }

    /// The lips' coat strength and roughness for the surface's defaults `coat` and `roughness`
    /// (`SurfaceSettings::lip_coat`, `lip_coat_roughness`), both 0..1.
    #[must_use]
    pub fn lip_coat(&self, coat: f32, roughness: f32) -> (f32, f32) {
        let m = self.lip_moisture;
        ((coat * 2.0 * m).clamp(0.0, 1.0), (roughness * (1.5 - m)).clamp(0.0, 1.0))
    }

    /// The compositor's globals: the base values and the zone table (`zones`' pigment and flush
    /// offsets) with the measured regional surface for the document's oiliness.
    pub fn resolved_globals(&self, zones: &ZoneTable) -> (BaseValues, ZoneTable) {
        let mut base = self.base;
        let mut table = *zones;
        apply_measured_surface(self.base.get(Attribute::Oiliness), &mut base, &mut table);
        let rough = base.get(Attribute::RoughnessPrimary);
        base.set(Attribute::RoughnessPrimary, rough + (0.5 - self.moisture) * MOISTURE_ROUGHNESS);
        if self.lips != DEFAULT_LIPS {
            let lips = &mut table.zones[Zone::Lips.slot()];
            let mut values = [0.0_f32; ATTRIBUTE_COUNT];
            for (i, v) in values.iter_mut().enumerate() {
                *v = base.0.get(i).copied().unwrap_or(0.0) + lips.offsets.get(i).copied().unwrap_or(0.0);
            }
            let start = pigment_of(&values);
            let model = PigmentModel::new(Mapping::DEFAULT);
            let fit = |rgb: [f32; 3]| model.fit_lab(rgb_to_lab(rgb.map(f64::from)), &start).params;
            let (to, from) = (fit(self.lips), fit(DEFAULT_LIPS));
            #[allow(clippy::cast_possible_truncation)] // Pigment parameters are 0..1.
            for (attribute, delta) in [
                (Attribute::Melanin, to.melanin - from.melanin),
                (Attribute::EumelaninRatio, to.eumelanin_ratio - from.eumelanin_ratio),
                (Attribute::Haemoglobin, to.haemoglobin - from.haemoglobin),
            ] {
                if let Some(o) = lips.offsets.get_mut(attribute.index()) {
                    *o += delta as f32;
                }
            }
        }
        (base, table)
    }

    /// The micro-detail generator's settings (age 0.3, seed 1, as [`MicroDetailParams::default`]).
    pub fn micro_detail(&self) -> MicroDetailParams {
        let d = self.detail;
        MicroDetailParams::from_registry(d[0], d[1], d[2], d[3], d[4], d[5], 0.3, 1)
    }

    /// Whether the compositor's inputs differ between `self` and `other` (a re-resolve is due).
    pub fn base_differs(&self, other: &Self) -> bool {
        self.base != other.base || self.moisture != other.moisture || self.lips != other.lips
    }

    /// Whether the surface's settings differ (flush, vellus sheen, lip coat).
    pub fn surface_differs(&self, other: &Self) -> bool {
        self.flush != other.flush || self.vellus != other.vellus || self.lip_moisture != other.lip_moisture
    }

    /// Whether the makeup differs (its layers must be generated anew).
    pub fn makeup_differs(&self, other: &Self) -> bool {
        self.makeup != other.makeup
    }

    /// Whether the micro-detail inputs differ (the exemplar must be regenerated).
    pub fn detail_differs(&self, other: &Self) -> bool {
        self.detail != other.detail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn todo_1_4_targets_map_to_their_values() {
        assert_eq!(SkinTarget::from_target("skin.melanin"), Some(SkinTarget::Base(Attribute::Melanin)));
        assert_eq!(SkinTarget::from_target("skin.pore-depth"), Some(SkinTarget::Detail(DetailInput::PoreDepth)));
        assert_eq!(SkinTarget::from_target("skin.flush"), Some(SkinTarget::Flush));
        assert_eq!(SkinTarget::from_target("skin.wrinkle-neck"), Some(SkinTarget::NeckWrinkles));
        assert_eq!(SkinTarget::from_target("skin.wrinkle-map"), None);
        assert_eq!(
            SkinTarget::from_target("skin.wrinkle-crows-feet-L"),
            Some(SkinTarget::Resting(RestingWrinkle::CrowsFeetLeft))
        );
        assert_eq!(RestingWrinkle::of_driver("glabella.R"), Some(RestingWrinkle::Glabella));
        assert_eq!(SkinTarget::from_target("skin.age-spots"), Some(SkinTarget::AgeSpots));
        let mut v = SkinValues::default();
        assert_eq!(v.micro_detail(), MicroDetailParams::default(), "defaults match the generator's");
        let before = v;
        v.set(SkinTarget::Base(Attribute::Melanin), 0.9);
        v.set(SkinTarget::Detail(DetailInput::PoreDepth), 3.0);
        v.set(SkinTarget::Flush, f32::NAN);
        assert_eq!(v.base.get(Attribute::Melanin), 0.9);
        assert_eq!(v.detail[DetailInput::PoreDepth as usize], 1.0);
        assert_eq!(v.flush, 0.0);
        assert!(v.base_differs(&before) && v.detail_differs(&before));
        assert!(v.micro_detail().pore_depth_mm > before.micro_detail().pore_depth_mm);
        let (base, _) = v.resolved_globals(&ZoneTable::default());
        assert_eq!(base.get(Attribute::Melanin), 0.9);
    }

    /// todo 2.3: moisture, vellus, lip colour and lip moisture reach what the compositor and the
    /// surface read; their defaults leave it as it was.
    #[test]
    fn todo_2_3_moisture_vellus_and_lips_reach_the_compositor_and_the_surface() {
        let zones = ZoneTable::default();
        let base = SkinValues::default();
        let (b0, z0) = base.resolved_globals(&zones);
        let mut dry = base;
        dry.set(SkinTarget::Moisture, 0.0);
        let (b1, _) = dry.resolved_globals(&zones);
        assert!(dry.base_differs(&base));
        assert!(
            (b1.get(Attribute::RoughnessPrimary) - b0.get(Attribute::RoughnessPrimary) - 0.5 * MOISTURE_ROUGHNESS)
                .abs()
                < 1e-6
        );
        let mut red = base;
        assert!(red.set_color(SkinTarget::LipColor, [0.6, 0.08, 0.08]));
        assert!(!red.set_color(SkinTarget::Moisture, [0.0; 3]));
        let (_, z1) = red.resolved_globals(&zones);
        let lips = |z: &ZoneTable, a: Attribute| z.zones[Zone::Lips.slot()].offsets[a.index()];
        println!(
            "[todo 2.3] red lips: haemoglobin offset {:.3} -> {:.3}, melanin {:.3} -> {:.3}",
            lips(&z0, Attribute::Haemoglobin),
            lips(&z1, Attribute::Haemoglobin),
            lips(&z0, Attribute::Melanin),
            lips(&z1, Attribute::Melanin)
        );
        assert!(lips(&z1, Attribute::Haemoglobin) > lips(&z0, Attribute::Haemoglobin), "redder lips, more blood");
        let mut pale = base;
        pale.set_color(SkinTarget::LipColor, [0.5, 0.35, 0.32]);
        let (_, z2) = pale.resolved_globals(&zones);
        assert!(lips(&z2, Attribute::Haemoglobin) < lips(&z0, Attribute::Haemoglobin), "paler lips, less blood");
        let mut wet = base;
        wet.set(SkinTarget::LipMoisture, 1.0);
        wet.set(SkinTarget::Vellus, 1.0);
        assert!(wet.surface_differs(&base) && !wet.base_differs(&base));
        assert_eq!(base.lip_coat(0.6, 0.15), (0.6, 0.15), "the default keeps the surface's lip coat");
        let (coat, rough) = wet.lip_coat(0.6, 0.15);
        assert!(coat > 0.6 && rough < 0.15);
        assert_eq!((base.fuzz_scale(), wet.fuzz_scale()), (1.0, 2.0));
        let mut neck = base;
        neck.set(SkinTarget::NeckWrinkles, 0.7);
        assert_eq!(neck.neck_wrinkles, 0.7);
    }
}
