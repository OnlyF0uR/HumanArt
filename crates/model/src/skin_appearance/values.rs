//! The document's global skin values (todo 1.4): what the registry's `skin.*` material targets
//! set, gathered in one plain value the views compare and apply.
//!
//! [`SkinTarget::from_target`] names the destination of a registry material target;
//! [`SkinValues`] holds the base attributes (pigment, tissue, oiliness, wetness), the micro-detail
//! inputs (pores, fine lines) and the animated flush. [`SkinValues::resolved_globals`] turns them
//! into what the compositor reads: the base values and the zone table with the measured regional
//! surface ([`super::regional`]); [`SkinValues::micro_detail`] into the micro-detail generator's
//! settings.

use super::channels::{Attribute, BaseValues};
use super::microdetail::MicroDetailParams;
use super::regional::apply_measured_surface;
use super::zones::ZoneTable;

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
}

impl SkinTarget {
    /// The destination of a registry material target, `None` for targets the skin does not read
    /// (wrinkle maps, generator layers and maps are applied elsewhere).
    pub fn from_target(target: &str) -> Option<Self> {
        if let Some(a) = BaseValues::attribute_of_target(target) {
            return Some(Self::Base(a));
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
}

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
        }
    }

    /// The compositor's globals: the base values and the zone table (`zones`' pigment and flush
    /// offsets) with the measured regional surface for the document's oiliness.
    pub fn resolved_globals(&self, zones: &ZoneTable) -> (BaseValues, ZoneTable) {
        let mut base = self.base;
        let mut table = *zones;
        apply_measured_surface(self.base.get(Attribute::Oiliness), &mut base, &mut table);
        (base, table)
    }

    /// The micro-detail generator's settings (age 0.3, seed 1, as [`MicroDetailParams::default`]).
    pub fn micro_detail(&self) -> MicroDetailParams {
        let d = self.detail;
        MicroDetailParams::from_registry(d[0], d[1], d[2], d[3], d[4], d[5], 0.3, 1)
    }

    /// Whether the compositor's inputs differ between `self` and `other` (a re-resolve is due).
    pub fn base_differs(&self, other: &Self) -> bool {
        self.base != other.base
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
        assert_eq!(SkinTarget::from_target("skin.wrinkle-neck"), None);
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
}
