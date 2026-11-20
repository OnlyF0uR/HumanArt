//! Region-modulated pigment `[SKN-11]`, blood-flow flush `[SKN-31]` and wetness `[SKN-32]`.
//!
//! The template's regions are turned (by the bakers) into soft **zone masks**, one per [`Zone`],
//! stored in the compositor's input array. Each zone adds an offset to the attributes under its
//! mask before the layers are applied, so the physically-inspired parameters stay the editing
//! surface and the face gets its natural variation: flushed cheeks and nose, darker under-eyes,
//! red lips and ears, darker knuckles, elbows and knees, bluish temples, an oily forehead.
//!
//! **Flush** is a cheap multiplier on the blood fraction (`Ch · (1 + gain · w)`), with `w` the
//! clamped sum of a whole-body floor and the zones' flush weights. The resolve step stores the
//! albedo change at full flush in the `flush` resolved channel, so animating the flush amount costs
//! one multiply-add in the shading pass and no re-resolve.
//!
//! **Wetness** lowers the primary lobe's roughness towards that of a water film and raises the
//! specular scale; the thin-film sheen itself is shaded from the wetness the detail channel carries.
//!
//! Every default offset here is an **artistic default** (a starting look to be tuned against
//! reference renders), not measured data.

use super::channels::{ATTRIBUTE_COUNT, Attribute};

/// Number of zone mask slots (three RGBA8 layers per tile).
pub const ZONE_SLOTS: usize = 12;

/// A soft-masked area of the skin with its own pigment modulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Zone {
    /// Cheeks (flush).
    Cheeks = 0,
    /// Nose (flush).
    Nose = 1,
    /// Under the eyes (shading).
    UnderEye = 2,
    /// Lips (colour).
    Lips = 3,
    /// Ears (redder).
    Ears = 4,
    /// Knuckles (darker).
    Knuckles = 5,
    /// Elbows (darker).
    Elbows = 6,
    /// Knees (darker).
    Knees = 7,
    /// Temples (bluish).
    Temples = 8,
    /// Forehead (sheen).
    Forehead = 9,
}

impl Zone {
    /// Every zone, in mask-slot order.
    pub const ALL: [Self; 10] = [
        Self::Cheeks,
        Self::Nose,
        Self::UnderEye,
        Self::Lips,
        Self::Ears,
        Self::Knuckles,
        Self::Elbows,
        Self::Knees,
        Self::Temples,
        Self::Forehead,
    ];

    /// The mask slot.
    pub const fn slot(self) -> usize {
        self as usize
    }

    /// The stable name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cheeks => "cheeks",
            Self::Nose => "nose",
            Self::UnderEye => "under-eye",
            Self::Lips => "lips",
            Self::Ears => "ears",
            Self::Knuckles => "knuckles",
            Self::Elbows => "elbows",
            Self::Knees => "knees",
            Self::Temples => "temples",
            Self::Forehead => "forehead",
        }
    }

    /// The zone with a stable name.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|z| z.name() == name)
    }

    /// The zone of a mask slot.
    pub fn from_slot(slot: usize) -> Option<Self> {
        Self::ALL.get(slot).copied()
    }
}

/// What one zone does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoneModulation {
    /// Offset added to each attribute under a full mask.
    pub offsets: [f32; ATTRIBUTE_COUNT],
    /// Share of the flush the zone receives (added to the body floor).
    pub flush_weight: f32,
}

impl ZoneModulation {
    /// No modulation.
    pub const NONE: Self = Self { offsets: [0.0; ATTRIBUTE_COUNT], flush_weight: 0.0 };

    const fn with(mut self, attribute: Attribute, offset: f32) -> Self {
        self.offsets[attribute.index()] = offset;
        self
    }

    const fn flush(mut self, weight: f32) -> Self {
        self.flush_weight = weight;
        self
    }
}

/// The modulation of every zone slot plus the flush and wetness constants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoneTable {
    /// Per mask slot ([`Zone::slot`]); unused slots are [`ZoneModulation::NONE`].
    pub zones: [ZoneModulation; ZONE_SLOTS],
    /// Flush weight everywhere (exertion reddens the whole body a little).
    pub flush_body: f32,
    /// Blood fraction multiplier at flush 1 and weight 1: `Ch · (1 + gain)`.
    pub flush_gain: f32,
    /// Roughness of a water film, the target of the primary lobe at wetness 1.
    pub wet_roughness: f32,
    /// Specular scale gain at wetness 1.
    pub wet_specular_gain: f32,
}

impl Default for ZoneTable {
    /// The artistic defaults (see the module documentation).
    fn default() -> Self {
        let mut zones = [ZoneModulation::NONE; ZONE_SLOTS];
        zones[Zone::Cheeks.slot()] = ZoneModulation::NONE.with(Attribute::Haemoglobin, 0.12).flush(1.0);
        zones[Zone::Nose.slot()] = ZoneModulation::NONE.with(Attribute::Haemoglobin, 0.10).flush(0.8);
        zones[Zone::UnderEye.slot()] = ZoneModulation::NONE
            .with(Attribute::Melanin, 0.05)
            .with(Attribute::Oxygenation, -0.2)
            .with(Attribute::EpidermalThickness, -0.3);
        zones[Zone::Lips.slot()] = ZoneModulation::NONE
            .with(Attribute::Haemoglobin, 0.3)
            .with(Attribute::EpidermalThickness, -0.4)
            .with(Attribute::Melanin, -0.03)
            .flush(0.4);
        zones[Zone::Ears.slot()] = ZoneModulation::NONE.with(Attribute::Haemoglobin, 0.12).flush(0.7);
        let dark = ZoneModulation::NONE.with(Attribute::Melanin, 0.06).with(Attribute::RoughnessPrimary, 0.08);
        zones[Zone::Knuckles.slot()] = dark;
        zones[Zone::Elbows.slot()] = dark;
        zones[Zone::Knees.slot()] = dark;
        zones[Zone::Temples.slot()] = ZoneModulation::NONE.with(Attribute::Oxygenation, -0.15);
        zones[Zone::Forehead.slot()] =
            ZoneModulation::NONE.with(Attribute::Oiliness, 0.25).with(Attribute::RoughnessPrimary, -0.05);
        Self { zones, flush_body: 0.25, flush_gain: 2.0, wet_roughness: 0.12, wet_specular_gain: 0.5 }
    }
}

impl ZoneTable {
    /// Adds the zone offsets under `masks` (one weight per slot, 0..1) to `values`.
    pub fn apply(&self, masks: &[f32; ZONE_SLOTS], values: &mut [f32; ATTRIBUTE_COUNT]) {
        for (zone, &m) in self.zones.iter().zip(masks) {
            if m > 0.0 {
                for (v, o) in values.iter_mut().zip(zone.offsets) {
                    *v += m * o;
                }
            }
        }
    }

    /// The flush weight under `masks`, 0..1.
    pub fn flush_weight(&self, masks: &[f32; ZONE_SLOTS]) -> f32 {
        let sum: f32 = self.zones.iter().zip(masks).map(|(z, &m)| z.flush_weight * m).sum();
        (self.flush_body + sum).clamp(0.0, 1.0)
    }

    /// The primary roughness and specular scale after wetness `[SKN-32]`.
    pub fn wet_surface(&self, roughness: f32, specular: f32, wetness: f32) -> (f32, f32) {
        let w = wetness.clamp(0.0, 1.0);
        (roughness + (self.wet_roughness - roughness) * w, specular * (1.0 + self.wet_specular_gain * w))
    }

    /// The table as GPU words: per slot 20 offsets then the flush weight, padded to 24 words;
    /// then `[flush_body, flush_gain, wet_roughness, wet_specular_gain]`.
    pub fn gpu_words(&self) -> [f32; ZONE_SLOTS * 24 + 4] {
        let mut out = [0.0; ZONE_SLOTS * 24 + 4];
        for (chunk, z) in out.as_chunks_mut::<24>().0.iter_mut().zip(self.zones.iter()) {
            chunk[..ATTRIBUTE_COUNT].copy_from_slice(&z.offsets);
            chunk[ATTRIBUTE_COUNT] = z.flush_weight;
        }
        let tail = ZONE_SLOTS * 24;
        out[tail..].copy_from_slice(&[self.flush_body, self.flush_gain, self.wet_roughness, self.wet_specular_gain]);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zone_names_and_slots() {
        for (i, z) in Zone::ALL.into_iter().enumerate() {
            assert_eq!(z.slot(), i);
            assert_eq!(Zone::from_name(z.name()), Some(z));
            assert_eq!(Zone::from_slot(i), Some(z));
        }
        assert!(Zone::ALL.len() <= ZONE_SLOTS);
        assert_eq!(Zone::from_slot(11), None);
    }

    #[test]
    fn skn11_zones_modulate_under_their_masks() {
        let table = ZoneTable::default();
        let mut values = [0.5; ATTRIBUTE_COUNT];
        let mut masks = [0.0; ZONE_SLOTS];
        table.apply(&masks, &mut values);
        assert_eq!(values, [0.5; ATTRIBUTE_COUNT], "no mask, no change");
        masks[Zone::Lips.slot()] = 1.0;
        masks[Zone::Knees.slot()] = 0.5;
        table.apply(&masks, &mut values);
        assert!((values[Attribute::Haemoglobin.index()] - 0.8).abs() < 1e-6, "lips are redder");
        assert!((values[Attribute::Melanin.index()] - (0.5 - 0.03 + 0.03)).abs() < 1e-6, "half a knee");
    }

    #[test]
    fn skn31_flush_weight_and_skn32_wetness() {
        let table = ZoneTable::default();
        let mut masks = [0.0; ZONE_SLOTS];
        assert_eq!(table.flush_weight(&masks), 0.25, "whole-body floor");
        masks[Zone::Cheeks.slot()] = 1.0;
        assert_eq!(table.flush_weight(&masks), 1.0, "clamped");
        let (r, s) = table.wet_surface(0.5, 1.0, 1.0);
        assert!((r - 0.12).abs() < 1e-6 && (s - 1.5).abs() < 1e-6);
        assert_eq!(table.wet_surface(0.5, 1.0, 0.0), (0.5, 1.0));
        let words = table.gpu_words();
        assert_eq!(words[Zone::Cheeks.slot() * 24 + Attribute::Haemoglobin.index()], 0.12);
        assert_eq!(words[Zone::Cheeks.slot() * 24 + 20], 1.0);
        assert_eq!(words[ZONE_SLOTS * 24 + 1], 2.0);
    }
}
