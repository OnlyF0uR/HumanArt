//! The document's character-wide groom settings (`todo.md` 2.3): what the registry's `groom`
//! targets (`brows.*`, `lashes.*`, `body.density`, `stubble.*`, `beard.density`,
//! `moustache.density`) set, gathered in one plain value, and how they scale the groom set
//! ([`GroomSettings::apply`] onto a [`GroomSetSpec`]).
//!
//! The registry values are normalised 0..1; the mappings (styling choices, chosen so the registry
//! default gives the generators' defaults):
//! - brow and lash density `v`: `2 v` times the default hair count (0 none, 0.5 default, 1 double);
//! - brow, lash and stubble length `v`: `0.2 + 1.6 v` times the default length (0.5 default);
//! - lash curl `v`: `2 v` times the default curl;
//! - brow thickness `v`: `0.5 + v` times the brow band's height, with the hair count in step (0.5
//!   default);
//! - lash thickness `v` (mascara): `1 + 1.5 v` times the default fibre diameter (0 is bare lashes);
//! - body hair density `v`: `v / 0.2` times the default (the registry default is 0.2);
//! - stubble, beard and moustache density: the share of the style's full density (0 grows none).
//!
//! Whether body hair grows at all is the view's choice (its options, or the document setting the
//! density itself); these settings only scale it.

use super::template::GroomSetSpec;

/// A groom setting of the registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GroomTarget {
    /// `body.density`.
    BodyDensity = 0,
    /// `brows.density`.
    BrowDensity = 1,
    /// `brows.length`.
    BrowLength = 2,
    /// `lashes.length`.
    LashLength = 3,
    /// `lashes.curl`.
    LashCurl = 4,
    /// `lashes.density`.
    LashDensity = 5,
    /// `stubble.density`.
    StubbleDensity = 6,
    /// `stubble.length`.
    StubbleLength = 7,
    /// `beard.density`.
    BeardDensity = 8,
    /// `moustache.density`.
    MoustacheDensity = 9,
    /// `lashes.thickness`.
    LashThickness = 10,
    /// `brows.thickness`.
    BrowThickness = 11,
}

/// Number of [`GroomTarget`]s.
pub const GROOM_TARGETS: usize = 12;

/// The registry default of `hair.body.density`, the value that gives the default body hair.
pub const DEFAULT_BODY_DENSITY: f32 = 0.2;

impl GroomTarget {
    /// The destination of a registry groom target, `None` for targets the groom set does not read.
    pub fn from_target(target: &str) -> Option<Self> {
        Some(match target {
            "body.density" => Self::BodyDensity,
            "brows.density" => Self::BrowDensity,
            "brows.length" => Self::BrowLength,
            "lashes.length" => Self::LashLength,
            "lashes.curl" => Self::LashCurl,
            "lashes.density" => Self::LashDensity,
            "stubble.density" => Self::StubbleDensity,
            "stubble.length" => Self::StubbleLength,
            "beard.density" => Self::BeardDensity,
            "moustache.density" => Self::MoustacheDensity,
            "lashes.thickness" => Self::LashThickness,
            "brows.thickness" => Self::BrowThickness,
            _ => return None,
        })
    }
}

/// The document's groom settings, by [`GroomTarget`], 0..1. See the module documentation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroomSettings {
    /// The values.
    pub values: [f32; GROOM_TARGETS],
}

impl Default for GroomSettings {
    /// The registry's defaults (`data/template/params/hair.json`).
    fn default() -> Self {
        Self { values: [DEFAULT_BODY_DENSITY, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5] }
    }
}

impl GroomSettings {
    /// Sets one value (clamped to 0..1; a non-finite value is ignored).
    pub fn set(&mut self, target: GroomTarget, value: f32) {
        if value.is_finite() {
            self.values[target as usize] = value.clamp(0.0, 1.0);
        }
    }

    /// One value.
    #[must_use]
    pub fn get(&self, target: GroomTarget) -> f32 {
        self.values[target as usize]
    }

    /// Writes the settings into `spec` (the scale factors and facial hair shares; the scalp, the
    /// colour and the on/off switches stay as they are).
    pub fn apply(&self, spec: &mut GroomSetSpec) {
        let v = |t: GroomTarget| self.get(t);
        let length = |t: GroomTarget| 0.2 + 1.6 * v(t);
        spec.brow_density = 2.0 * v(GroomTarget::BrowDensity);
        spec.brow_length = length(GroomTarget::BrowLength);
        spec.lash_density = 2.0 * v(GroomTarget::LashDensity);
        spec.lash_length = length(GroomTarget::LashLength);
        spec.lash_curl = 2.0 * v(GroomTarget::LashCurl);
        spec.lash_thickness = 1.0 + 1.5 * v(GroomTarget::LashThickness);
        spec.brow_thickness = 0.5 + v(GroomTarget::BrowThickness);
        spec.body_density = v(GroomTarget::BodyDensity) / DEFAULT_BODY_DENSITY;
        spec.stubble = v(GroomTarget::StubbleDensity);
        spec.stubble_length = length(GroomTarget::StubbleLength);
        spec.beard = v(GroomTarget::BeardDensity);
        spec.moustache = v(GroomTarget::MoustacheDensity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn todo_2_3_defaults_keep_the_default_groom_set_and_every_setting_moves_it() {
        let mut spec = GroomSetSpec::default();
        GroomSettings::default().apply(&mut spec);
        assert_eq!(spec, GroomSetSpec::default(), "the registry defaults give the default groom set");
        let targets = [
            ("body.density", GroomTarget::BodyDensity),
            ("brows.density", GroomTarget::BrowDensity),
            ("brows.length", GroomTarget::BrowLength),
            ("lashes.length", GroomTarget::LashLength),
            ("lashes.curl", GroomTarget::LashCurl),
            ("lashes.density", GroomTarget::LashDensity),
            ("stubble.density", GroomTarget::StubbleDensity),
            ("stubble.length", GroomTarget::StubbleLength),
            ("beard.density", GroomTarget::BeardDensity),
            ("moustache.density", GroomTarget::MoustacheDensity),
            ("lashes.thickness", GroomTarget::LashThickness),
            ("brows.thickness", GroomTarget::BrowThickness),
        ];
        assert_eq!(targets.len(), GROOM_TARGETS);
        for (name, t) in targets {
            assert_eq!(GroomTarget::from_target(name), Some(t));
            let mut s = GroomSettings::default();
            s.set(t, 1.0);
            let mut moved = GroomSetSpec::default();
            s.apply(&mut moved);
            assert_ne!(moved, GroomSetSpec::default(), "{name}");
        }
        assert_eq!(GroomTarget::from_target("scalp.density"), None);
        let mut s = GroomSettings::default();
        s.set(GroomTarget::BrowDensity, f32::NAN);
        s.set(GroomTarget::LashCurl, 7.0);
        assert_eq!(s.get(GroomTarget::BrowDensity), 0.5);
        assert_eq!(s.get(GroomTarget::LashCurl), 1.0);
    }
}
