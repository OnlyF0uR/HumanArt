//! The generator's region table `[MOD-03]`, UV charts and UV tiles `[MOD-04]`.
//!
//! Region names are stable ids (never rename or reuse one). Every region of the `[MOD-03]` list is
//! here; groups (`eye-L`, `mouth`, `torso`, ...) are inner nodes, and the surface is assigned to
//! leaves. The separate parts of `[MOD-01]` have their own leaves under the group they belong to.

use crate::names::Side;
use crate::region::Region;

macro_rules! region_table {
    ($( $id:ident = $name:literal, $parent:ident, $side:ident; )*) => {
        /// A region of the generated template; the discriminant is its index in the region table.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[repr(u16)]
        pub enum R {
            $(
                #[doc = $name]
                $id,
            )*
        }

        /// The region table in index order: id, stable name, parent (the root is its own parent), side.
        pub const REGIONS: &[(R, &str, R, Side)] = &[$((R::$id, $name, R::$parent, Side::$side)),*];
    };
}

region_table! {
    Body = "body", Body, Center;
    Head = "head", Body, Center;
    Face = "face", Head, Center;
    Forehead = "forehead", Face, Center;
    BrowL = "brow-L", Face, Left;
    BrowR = "brow-R", Face, Right;
    EyeL = "eye-L", Face, Left;
    EyeR = "eye-R", Face, Right;
    UpperLidL = "upper-lid-L", EyeL, Left;
    UpperLidR = "upper-lid-R", EyeR, Right;
    LowerLidL = "lower-lid-L", EyeL, Left;
    LowerLidR = "lower-lid-R", EyeR, Right;
    InnerCanthusL = "inner-canthus-L", EyeL, Left;
    InnerCanthusR = "inner-canthus-R", EyeR, Right;
    OuterCanthusL = "outer-canthus-L", EyeL, Left;
    OuterCanthusR = "outer-canthus-R", EyeR, Right;
    EyeSocketL = "eye-socket-L", EyeL, Left;
    EyeSocketR = "eye-socket-R", EyeR, Right;
    Nose = "nose", Face, Center;
    NasalBridge = "nasal-bridge", Nose, Center;
    NasalTip = "nasal-tip", Nose, Center;
    NostrilL = "nostril-L", Nose, Left;
    NostrilR = "nostril-R", Nose, Right;
    AlaL = "ala-L", Nose, Left;
    AlaR = "ala-R", Nose, Right;
    Mouth = "mouth", Face, Center;
    Philtrum = "philtrum", Mouth, Center;
    UpperLip = "upper-lip", Mouth, Center;
    LowerLip = "lower-lip", Mouth, Center;
    LipCornerL = "lip-corner-L", Mouth, Left;
    LipCornerR = "lip-corner-R", Mouth, Right;
    MouthInterior = "mouth-interior", Mouth, Center;
    Chin = "chin", Face, Center;
    JawL = "jaw-L", Face, Left;
    JawR = "jaw-R", Face, Right;
    CheekL = "cheek-L", Face, Left;
    CheekR = "cheek-R", Face, Right;
    MalarL = "malar-L", Face, Left;
    MalarR = "malar-R", Face, Right;
    TempleL = "temple-L", Face, Left;
    TempleR = "temple-R", Face, Right;
    EarL = "ear-L", Head, Left;
    EarR = "ear-R", Head, Right;
    EarLHelix = "ear-L.helix", EarL, Left;
    EarRHelix = "ear-R.helix", EarR, Right;
    EarLAntihelix = "ear-L.antihelix", EarL, Left;
    EarRAntihelix = "ear-R.antihelix", EarR, Right;
    EarLConcha = "ear-L.concha", EarL, Left;
    EarRConcha = "ear-R.concha", EarR, Right;
    EarLLobe = "ear-L.lobe", EarL, Left;
    EarRLobe = "ear-R.lobe", EarR, Right;
    EarLBack = "ear-L.back", EarL, Left;
    EarRBack = "ear-R.back", EarR, Right;
    Scalp = "scalp", Head, Center;
    Hairline = "hairline", Head, Center;
    Neck = "neck", Body, Center;
    NeckFront = "neck-front", Neck, Center;
    NeckBack = "neck-back", Neck, Center;
    NeckSideL = "neck-side-L", Neck, Left;
    NeckSideR = "neck-side-R", Neck, Right;
    Torso = "torso", Body, Center;
    Shoulder = "shoulder", Torso, Center;
    ShoulderL = "shoulder-L", Shoulder, Left;
    ShoulderR = "shoulder-R", Shoulder, Right;
    Chest = "chest", Torso, Center;
    Abdomen = "abdomen", Torso, Center;
    Back = "back", Torso, Center;
    Pelvis = "pelvis", Torso, Center;
    UpperArm = "upper-arm", Body, Center;
    UpperArmL = "upper-arm-L", UpperArm, Left;
    UpperArmR = "upper-arm-R", UpperArm, Right;
    Forearm = "forearm", Body, Center;
    ForearmL = "forearm-L", Forearm, Left;
    ForearmR = "forearm-R", Forearm, Right;
    Hand = "hand", Body, Center;
    HandL = "hand-L", Hand, Left;
    HandR = "hand-R", Hand, Right;
    HandLPalm = "hand-L.palm", HandL, Left;
    HandRPalm = "hand-R.palm", HandR, Right;
    HandLThumb = "hand-L.thumb", HandL, Left;
    HandRThumb = "hand-R.thumb", HandR, Right;
    HandLIndex = "hand-L.index", HandL, Left;
    HandRIndex = "hand-R.index", HandR, Right;
    HandLMiddle = "hand-L.middle", HandL, Left;
    HandRMiddle = "hand-R.middle", HandR, Right;
    HandLRing = "hand-L.ring", HandL, Left;
    HandRRing = "hand-R.ring", HandR, Right;
    HandLLittle = "hand-L.little", HandL, Left;
    HandRLittle = "hand-R.little", HandR, Right;
    Thigh = "thigh", Body, Center;
    ThighL = "thigh-L", Thigh, Left;
    ThighR = "thigh-R", Thigh, Right;
    Shin = "shin", Body, Center;
    ShinL = "shin-L", Shin, Left;
    ShinR = "shin-R", Shin, Right;
    Foot = "foot", Body, Center;
    FootL = "foot-L", Foot, Left;
    FootR = "foot-R", Foot, Right;
    FootLDorsum = "foot-L.dorsum", FootL, Left;
    FootRDorsum = "foot-R.dorsum", FootR, Right;
    FootLSole = "foot-L.sole", FootL, Left;
    FootRSole = "foot-R.sole", FootR, Right;
    FootLToes = "foot-L.toes", FootL, Left;
    FootRToes = "foot-R.toes", FootR, Right;
    EyeballL = "eyeball-L", EyeL, Left;
    EyeballR = "eyeball-R", EyeR, Right;
    TearLineL = "tear-line-L", EyeL, Left;
    TearLineR = "tear-line-R", EyeR, Right;
    EyeOcclusionL = "eye-occlusion-L", EyeL, Left;
    EyeOcclusionR = "eye-occlusion-R", EyeR, Right;
    TeethUpper = "teeth-upper", Mouth, Center;
    TeethLower = "teeth-lower", Mouth, Center;
    GumsUpper = "gums-upper", Mouth, Center;
    GumsLower = "gums-lower", Mouth, Center;
    Tongue = "tongue", Mouth, Center;
    MouthHelper = "mouth-helper", Mouth, Center;
    Navel = "navel", Abdomen, Center;
    AreolaL = "areola-L", Chest, Left;
    AreolaR = "areola-R", Chest, Right;
    NippleL = "nipple-L", AreolaL, Left;
    NippleR = "nipple-R", AreolaR, Right;
    Perineum = "perineum", Pelvis, Center;
    Genitals = "genitals", Torso, Center;
    LabiaMajora = "labia-majora", Genitals, Center;
    LabiaMinora = "labia-minora", Genitals, Center;
    Clitoris = "clitoris", Genitals, Center;
    Vestibule = "vestibule", Genitals, Center;
    Penis = "penis", Genitals, Center;
    Glans = "glans", Penis, Center;
    Scrotum = "scrotum", Genitals, Center;
    HandLNail = "hand-L.nail", HandL, Left;
    HandRNail = "hand-R.nail", HandR, Right;
    HandLLunula = "hand-L.lunula", HandLNail, Left;
    HandRLunula = "hand-R.lunula", HandRNail, Right;
    FootLNail = "foot-L.nail", FootLToes, Left;
    FootRNail = "foot-R.nail", FootRToes, Right;
    FootLArch = "foot-L.arch", FootLSole, Left;
    FootRArch = "foot-R.arch", FootRSole, Right;
}

impl R {
    /// The region's index in the table.
    pub const fn id(self) -> u16 {
        self as u16
    }

    /// The region with index `i`, if there is one.
    pub fn from_id(i: u16) -> Option<Self> {
        REGIONS.get(i as usize).map(|r| r.0)
    }

    /// The stable name.
    pub fn name(self) -> &'static str {
        REGIONS[self as usize].1
    }

    /// The left or right variant of a sided pair (`side` > 0 picks the left one, `+X`).
    pub fn sided(left: Self, right: Self, side: f32) -> Self {
        if side > 0.0 { left } else { right }
    }

    /// Whether the region is part of the external genitalia (ADR-039): the faces inside the groin
    /// attachment loop, which go to the `genitals-*` mesh part.
    pub const fn is_genital(self) -> bool {
        matches!(
            self,
            Self::Genitals
                | Self::LabiaMajora
                | Self::LabiaMinora
                | Self::Clitoris
                | Self::Vestibule
                | Self::Penis
                | Self::Glans
                | Self::Scrotum
        )
    }
}

/// The model's region table built from [`REGIONS`].
pub fn region_table() -> Vec<Region> {
    REGIONS
        .iter()
        .map(|&(id, name, parent, side)| Region {
            name: name.to_owned(),
            parent: if parent == id { None } else { Some(parent.id()) },
            side,
        })
        .collect()
}

macro_rules! chart_table {
    ($( $id:ident = $name:literal, $tile:ident; )*) => {
        /// A UV chart (island) of the generated template.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[repr(u16)]
        pub enum C {
            $(
                #[doc = $name]
                $id,
            )*
        }

        /// The chart table in index order: id, name, tile.
        pub const CHARTS: &[(C, &str, T)] = &[$((C::$id, $name, T::$tile)),*];
    };
}

/// A UV tile `[MOD-04]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u16)]
pub enum T {
    /// Face and neck, the hero tile.
    Face,
    /// Torso.
    Torso,
    /// Arms.
    Arms,
    /// Legs.
    Legs,
    /// Hands and feet.
    HandsFeet,
    /// Eyes (eyeballs, sockets, tear lines, occlusion).
    Eyes,
    /// Mouth interior (mouth bag, teeth, gums, tongue).
    Mouth,
    /// Scalp.
    Scalp,
}

/// The tile table: tile, stable name, resolution, target texels per millimetre.
///
/// Face and neck aim at 10 texels/mm `[MOD-04]`; the body tiles at lower densities.
pub const TILES: &[(T, &str, u32, f32)] = &[
    (T::Face, "face", 8192, 10.0),
    (T::Torso, "torso", 4096, 3.0),
    (T::Arms, "arms", 4096, 3.0),
    (T::Legs, "legs", 4096, 2.5),
    (T::HandsFeet, "hands-feet", 4096, 5.0),
    (T::Eyes, "eyes", 2048, 10.0),
    (T::Mouth, "mouth", 2048, 5.0),
    (T::Scalp, "scalp", 4096, 4.0),
];

chart_table! {
    Face = "face", Face;
    Neck = "neck", Face;
    EarL = "ear-L", Face;
    EarR = "ear-R", Face;
    EarLBack = "ear-L-back", Face;
    EarRBack = "ear-R-back", Face;
    NostrilL = "nostril-L", Face;
    NostrilR = "nostril-R", Face;
    Scalp = "scalp", Scalp;
    TorsoFront = "torso-front", Torso;
    TorsoBack = "torso-back", Torso;
    ArmL = "arm-L", Arms;
    ArmR = "arm-R", Arms;
    LegL = "leg-L", Legs;
    LegR = "leg-R", Legs;
    HandLDorsal = "hand-L-dorsal", HandsFeet;
    HandRDorsal = "hand-R-dorsal", HandsFeet;
    HandLPalmar = "hand-L-palmar", HandsFeet;
    HandRPalmar = "hand-R-palmar", HandsFeet;
    FootLTop = "foot-L-top", HandsFeet;
    FootRTop = "foot-R-top", HandsFeet;
    FootLSole = "foot-L-sole", HandsFeet;
    FootRSole = "foot-R-sole", HandsFeet;
    SocketL = "socket-L", Eyes;
    SocketR = "socket-R", Eyes;
    MouthBag = "mouth-bag", Mouth;
    EyeballL = "eyeball-L", Eyes;
    EyeballR = "eyeball-R", Eyes;
    TearLineL = "tear-line-L", Eyes;
    TearLineR = "tear-line-R", Eyes;
    OcclusionL = "eye-occlusion-L", Eyes;
    OcclusionR = "eye-occlusion-R", Eyes;
    Teeth = "teeth", Mouth;
    Gums = "gums", Mouth;
    Tongue = "tongue", Mouth;
    MouthHelper = "mouth-helper", Mouth;
    Navel = "navel", Torso;
    NippleL = "nipple-L", Torso;
    NippleR = "nipple-R", Torso;
    Vulva = "vulva", Torso;
    Penis = "penis", Torso;
    Scrotum = "scrotum", Torso;
    GenitalBase = "genital-base", Torso;
}

impl C {
    /// The chart's index.
    pub const fn id(self) -> u16 {
        self as u16
    }

    /// The tile the chart is packed into.
    pub fn tile(self) -> T {
        CHARTS[self as usize].2
    }

    /// The left or right variant of a sided pair (`side` > 0 picks the left one).
    pub fn sided(left: Self, right: Self, side: f32) -> Self {
        if side > 0.0 { left } else { right }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region::RegionMap;

    #[test]
    fn mod_03_region_table_is_valid_and_complete() {
        let regions = region_table();
        let map = RegionMap { regions, vertex_regions: Vec::new(), soft: None };
        map.check(0).unwrap();
        // [MOD-03] required names.
        for name in [
            "forehead",
            "brow-L",
            "brow-R",
            "upper-lid-L",
            "upper-lid-R",
            "lower-lid-L",
            "lower-lid-R",
            "inner-canthus-L",
            "outer-canthus-R",
            "eye-socket-L",
            "nasal-bridge",
            "nasal-tip",
            "nostril-L",
            "ala-R",
            "philtrum",
            "upper-lip",
            "lower-lip",
            "lip-corner-L",
            "mouth-interior",
            "chin",
            "jaw-L",
            "cheek-R",
            "malar-L",
            "temple-R",
            "ear-L.helix",
            "ear-R.antihelix",
            "ear-L.lobe",
            "neck-front",
            "neck-back",
            "neck-side-L",
            "scalp",
            "hairline",
            "shoulder-L",
            "chest",
            "abdomen",
            "back",
            "upper-arm-R",
            "forearm-L",
            "hand-L.index",
            "pelvis",
            "thigh-L",
            "shin-R",
            "foot-L",
            "shoulder",
            "hand",
            "foot",
        ] {
            assert!(map.find(name).is_some(), "{name}");
        }
        for (i, r) in REGIONS.iter().enumerate() {
            assert_eq!(r.0 as usize, i);
        }
        for (i, c) in CHARTS.iter().enumerate() {
            assert_eq!(c.0 as usize, i);
        }
        for (i, t) in TILES.iter().enumerate() {
            assert_eq!(t.0 as usize, i);
        }
    }
}
