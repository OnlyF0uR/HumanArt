//! The skin channel layout `[SKN-01]`: the attributes layers write, how they pack into the
//! composited parameter planes, and the resolved texture set the renderer samples, each with its
//! colour space.
//!
//! Layers write **attributes** (scalars). Four attributes make a **plane**, one RGBA16F texture
//! array per plane with one layer per UV tile. The compositor blends layers into the planes; the
//! resolve step turns the planes into the **resolved set**: albedo, scatter, specular and detail
//! (plus the planes themselves, which are the melanin / blood parameter maps of `[SKN-01]`).
//! Inputs that are baked rather than composited (macro normal, thickness, cavity bake, region
//! masks) come from the template bakers.

use humanart_io::ColorSpace;

/// Number of attributes.
pub const ATTRIBUTE_COUNT: usize = 20;
/// Number of composited planes.
pub const PLANE_COUNT: usize = 5;

/// One scalar a layer can write. The discriminant is the GPU index (`plane * 4 + component`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Attribute {
    /// Total melanin, 0..1 registry scale `[SKN-10]`.
    Melanin = 0,
    /// Eumelanin share.
    EumelaninRatio = 1,
    /// Blood in the dermis.
    Haemoglobin = 2,
    /// Blood oxygenation.
    Oxygenation = 3,
    /// β-carotene.
    Carotene = 4,
    /// Epidermis thickness.
    EpidermalThickness = 5,
    /// Sebum / oiliness (clear-coat strength, `[SHD-06]`).
    Oiliness = 6,
    /// Sweat / wetness `[SKN-32]`.
    Wetness = 7,
    /// Specular reflectance scale (1 = skin's F0 of 0.028).
    Specular = 8,
    /// Roughness of the primary specular lobe.
    RoughnessPrimary = 9,
    /// Roughness of the secondary lobe.
    RoughnessSecondary = 10,
    /// Weight of the secondary lobe.
    LobeMix = 11,
    /// Meso height in mm (fine wrinkles, scars, bumps).
    Height = 12,
    /// Meso detail normal, tangent x (-1..1).
    DetailNormalX = 13,
    /// Meso detail normal, tangent y (-1..1).
    DetailNormalY = 14,
    /// Cavity / occlusion (1 = open).
    Cavity = 15,
    /// Direct colour override, red (linear) `[SKN-12]`.
    OverrideRed = 16,
    /// Direct colour override, green.
    OverrideGreen = 17,
    /// Direct colour override, blue.
    OverrideBlue = 18,
    /// How much the override replaces the pigment albedo.
    OverrideCoverage = 19,
}

impl Attribute {
    /// Every attribute in GPU order.
    pub const ALL: [Self; ATTRIBUTE_COUNT] = [
        Self::Melanin,
        Self::EumelaninRatio,
        Self::Haemoglobin,
        Self::Oxygenation,
        Self::Carotene,
        Self::EpidermalThickness,
        Self::Oiliness,
        Self::Wetness,
        Self::Specular,
        Self::RoughnessPrimary,
        Self::RoughnessSecondary,
        Self::LobeMix,
        Self::Height,
        Self::DetailNormalX,
        Self::DetailNormalY,
        Self::Cavity,
        Self::OverrideRed,
        Self::OverrideGreen,
        Self::OverrideBlue,
        Self::OverrideCoverage,
    ];

    /// The GPU index.
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The plane holding the attribute.
    pub const fn plane(self) -> Plane {
        Plane::ALL[self.index() / 4]
    }

    /// The stable name (commands and files use it).
    pub const fn name(self) -> &'static str {
        match self {
            Self::Melanin => "melanin",
            Self::EumelaninRatio => "eumelanin-ratio",
            Self::Haemoglobin => "haemoglobin",
            Self::Oxygenation => "oxygenation",
            Self::Carotene => "carotene",
            Self::EpidermalThickness => "epidermis-thickness",
            Self::Oiliness => "oiliness",
            Self::Wetness => "wetness",
            Self::Specular => "specular",
            Self::RoughnessPrimary => "roughness-primary",
            Self::RoughnessSecondary => "roughness-secondary",
            Self::LobeMix => "lobe-mix",
            Self::Height => "height",
            Self::DetailNormalX => "detail-normal-x",
            Self::DetailNormalY => "detail-normal-y",
            Self::Cavity => "cavity",
            Self::OverrideRed => "override-red",
            Self::OverrideGreen => "override-green",
            Self::OverrideBlue => "override-blue",
            Self::OverrideCoverage => "override-coverage",
        }
    }

    /// The attribute with a stable name.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|a| a.name() == name)
    }

    /// The attribute with a GPU index.
    pub fn from_index(index: usize) -> Option<Self> {
        Self::ALL.get(index).copied()
    }

    /// The valid range of the attribute's values.
    pub const fn range(self) -> (f32, f32) {
        match self {
            Self::DetailNormalX | Self::DetailNormalY => (-1.0, 1.0),
            Self::Height => (-2.0, 2.0),
            Self::Specular => (0.0, 4.0),
            _ => (0.0, 1.0),
        }
    }
}

/// A set of attributes, as a bit mask over [`Attribute::index`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct AttributeSet(pub u32);

impl AttributeSet {
    /// No attribute.
    pub const EMPTY: Self = Self(0);
    /// Every attribute.
    pub const ALL: Self = Self((1 << ATTRIBUTE_COUNT) - 1);

    /// The set of `attributes`.
    pub const fn of(attributes: &[Attribute]) -> Self {
        let mut bits = 0;
        let mut i = 0;
        while i < attributes.len() {
            bits |= 1 << attributes[i].index();
            i += 1;
        }
        Self(bits)
    }

    /// Whether `attribute` is in the set.
    pub const fn contains(self, attribute: Attribute) -> bool {
        self.0 & (1 << attribute.index()) != 0
    }

    /// The set with `attribute` added.
    pub const fn with(self, attribute: Attribute) -> Self {
        Self(self.0 | (1 << attribute.index()))
    }

    /// The union.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether the set is empty.
    pub const fn is_empty(self) -> bool {
        self.0 & Self::ALL.0 == 0
    }

    /// The number of attributes in the set.
    pub const fn len(self) -> usize {
        (self.0 & Self::ALL.0).count_ones() as usize
    }

    /// The 4-bit component mask of `plane` (bit `c` = component `c` written).
    pub const fn plane_mask(self, plane: Plane) -> u32 {
        (self.0 >> (plane as u32 * 4)) & 0xf
    }

    /// The attributes in the set, in GPU order.
    pub fn iter(self) -> impl Iterator<Item = Attribute> {
        Attribute::ALL.into_iter().filter(move |a| self.contains(*a))
    }
}

/// A composited plane: four attributes in one RGBA16F texture array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Plane {
    /// Melanin, eumelanin ratio, haemoglobin, oxygenation: the pigment parameter map.
    Pigment = 0,
    /// Carotene, epidermis thickness, oiliness, wetness.
    Tissue = 1,
    /// Specular scale, two roughnesses, lobe mix.
    Surface = 2,
    /// Height, detail normal x / y, cavity.
    Relief = 3,
    /// Direct colour override RGB and coverage.
    Override = 4,
}

impl Plane {
    /// Every plane in GPU order.
    pub const ALL: [Self; PLANE_COUNT] = [Self::Pigment, Self::Tissue, Self::Surface, Self::Relief, Self::Override];

    /// The stable name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Pigment => "pigment",
            Self::Tissue => "tissue",
            Self::Surface => "surface",
            Self::Relief => "relief",
            Self::Override => "override",
        }
    }

    /// The plane's four attributes.
    pub const fn attributes(self) -> [Attribute; 4] {
        let base = self as usize * 4;
        [Attribute::ALL[base], Attribute::ALL[base + 1], Attribute::ALL[base + 2], Attribute::ALL[base + 3]]
    }

    /// How the plane's values are interpreted `[COL-01]`.
    pub const fn color_space(self) -> ColorSpace {
        match self {
            Self::Override => ColorSpace::Linear,
            _ => ColorSpace::Data,
        }
    }
}

/// One texture of the resolved set the render passes sample `[SKN-01]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResolvedChannel {
    /// RGB diffuse albedo (linear Rec.709), A cavity. RGBA16F.
    Albedo,
    /// RGB subsurface-scattering radius in mm, A thickness (baked). RGBA16F.
    Scatter,
    /// R specular reflectance scale / 4, G primary roughness, B secondary roughness, A lobe mix.
    /// RGBA8 unorm.
    Specular,
    /// RG meso detail normal (tangent x, y), B height in mm, A wetness (thin-film sheen). RGBA16F.
    Detail,
}

impl ResolvedChannel {
    /// Every resolved channel.
    pub const ALL: [Self; 4] = [Self::Albedo, Self::Scatter, Self::Specular, Self::Detail];

    /// The stable name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Albedo => "albedo",
            Self::Scatter => "scatter",
            Self::Specular => "specular",
            Self::Detail => "detail",
        }
    }

    /// How the values are interpreted `[COL-01]`.
    pub const fn color_space(self) -> ColorSpace {
        match self {
            Self::Albedo => ColorSpace::Linear,
            Self::Detail => ColorSpace::Normal,
            Self::Scatter | Self::Specular => ColorSpace::Data,
        }
    }

    /// Whether the channel is stored as 8-bit unorm (else 16-bit float).
    pub const fn is_unorm8(self) -> bool {
        matches!(self, Self::Specular)
    }
}

/// The base values of every attribute: what the planes hold before any layer, built from the
/// registry's global skin parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BaseValues(pub [f32; ATTRIBUTE_COUNT]);

impl Default for BaseValues {
    /// Registry defaults (`data/template/params/skin.json`) for the pigment and tissue attributes;
    /// the surface defaults are the dual-lobe skin specular starting point of `[SHD-01]` (lobe
    /// roughnesses and mix are artistic defaults to be calibrated with the shading work).
    fn default() -> Self {
        let mut v = [0.0; ATTRIBUTE_COUNT];
        v[Attribute::Melanin.index()] = 0.3;
        v[Attribute::EumelaninRatio.index()] = 0.7;
        v[Attribute::Haemoglobin.index()] = 0.3;
        v[Attribute::Oxygenation.index()] = 0.7;
        v[Attribute::Carotene.index()] = 0.1;
        v[Attribute::EpidermalThickness.index()] = 0.5;
        v[Attribute::Oiliness.index()] = 0.5;
        v[Attribute::Wetness.index()] = 0.0;
        v[Attribute::Specular.index()] = 1.0;
        v[Attribute::RoughnessPrimary.index()] = 0.45;
        v[Attribute::RoughnessSecondary.index()] = 0.75;
        v[Attribute::LobeMix.index()] = 0.15;
        v[Attribute::Cavity.index()] = 1.0;
        Self(v)
    }
}

impl BaseValues {
    /// Sets an attribute from a registry material target (`skin.melanin`, `skin.oiliness`, ...);
    /// false for a target that is not a base attribute. Values are clamped to the attribute's
    /// range; non-finite values are ignored.
    pub fn set_target(&mut self, target: &str, value: f32) -> bool {
        let Some(attribute) = Self::attribute_of_target(target) else { return false };
        self.set(attribute, value);
        true
    }

    /// The base attribute a registry material target sets (`skin.melanin` -> melanin), if any.
    pub fn attribute_of_target(target: &str) -> Option<Attribute> {
        Some(match target {
            "skin.melanin" => Attribute::Melanin,
            "skin.eumelanin-ratio" => Attribute::EumelaninRatio,
            "skin.haemoglobin" => Attribute::Haemoglobin,
            "skin.oxygenation" => Attribute::Oxygenation,
            "skin.carotene" => Attribute::Carotene,
            "skin.epidermis-thickness" => Attribute::EpidermalThickness,
            "skin.oiliness" => Attribute::Oiliness,
            "skin.wetness" => Attribute::Wetness,
            _ => return None,
        })
    }

    /// Sets one attribute, clamped to its range; a non-finite value is ignored.
    pub fn set(&mut self, attribute: Attribute, value: f32) {
        if value.is_finite() {
            let (lo, hi) = attribute.range();
            self.0[attribute.index()] = value.clamp(lo, hi);
        }
    }

    /// The value of one attribute.
    pub const fn get(&self, attribute: Attribute) -> f32 {
        self.0[attribute.index()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skn01_layout_is_consistent() {
        for (i, a) in Attribute::ALL.into_iter().enumerate() {
            assert_eq!(a.index(), i);
            assert_eq!(Attribute::from_name(a.name()), Some(a));
            assert_eq!(Attribute::from_index(i), Some(a));
            assert!(a.plane().attributes().contains(&a));
        }
        assert_eq!(Attribute::from_index(ATTRIBUTE_COUNT), None);
        // Names are unique.
        for a in Attribute::ALL {
            assert_eq!(Attribute::ALL.iter().filter(|b| b.name() == a.name()).count(), 1);
        }
        assert_eq!(Plane::Override.color_space(), ColorSpace::Linear);
        assert_eq!(ResolvedChannel::Albedo.color_space(), ColorSpace::Linear);
        assert!(ResolvedChannel::ALL.iter().all(|c| c.name().len() > 3));
    }

    #[test]
    fn attribute_sets() {
        let s = AttributeSet::of(&[Attribute::Melanin, Attribute::Haemoglobin, Attribute::Height]);
        assert_eq!(s.len(), 3);
        assert_eq!(s.plane_mask(Plane::Pigment), 0b0101);
        assert_eq!(s.plane_mask(Plane::Relief), 0b0001);
        assert_eq!(s.plane_mask(Plane::Surface), 0);
        assert_eq!(s.iter().collect::<Vec<_>>(), [Attribute::Melanin, Attribute::Haemoglobin, Attribute::Height]);
        assert!(AttributeSet::EMPTY.is_empty());
        assert_eq!(AttributeSet::ALL.len(), ATTRIBUTE_COUNT);
        assert!(s.with(Attribute::Cavity).contains(Attribute::Cavity));
    }

    #[test]
    fn base_values_from_targets() {
        let mut b = BaseValues::default();
        assert!(b.set_target("skin.melanin", 0.8));
        assert_eq!(b.get(Attribute::Melanin), 0.8);
        assert!(b.set_target("skin.wetness", 7.0));
        assert_eq!(b.get(Attribute::Wetness), 1.0);
        assert!(b.set_target("skin.oiliness", f32::NAN));
        assert_eq!(b.get(Attribute::Oiliness), 0.5);
        assert!(!b.set_target("skin.pore-depth", 0.5));
    }
}
