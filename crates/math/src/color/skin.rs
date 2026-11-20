//! Skin-colour helpers: Fitzpatrick reference swatches and nearest-swatch lookup, the individual
//! typology angle, a data-derived skin-colour region, YCbCr and the vectorscope skin-tone line
//! `[COL-03]` `[SKN-10]` `[SKN-13]` `[TOOL-07]`.
//!
//! **No built-in swatch data.** The Fitzpatrick I–VI reference swatches are data with provenance
//! (`data/test-scenes/skin-reference.json`, `[SKN-10]`); this module holds no colour values of its
//! own. Callers load the table and pass it as a slice of [`SkinSwatch`]; [`nearest_swatch`] finds the
//! closest one by CIEDE2000 (the colour probe of `[TOOL-07]` and the picker's reference swatches of
//! `[UI-07]`), and [`SkinRegion::enclosing`] derives the skin-tone sanity region of section 10.8 from
//! the same table.
//!
//! **Individual typology angle.** `ITA° = atan((L* − 50) / b*)` in degrees with the published
//! category limits of A. Chardon, I. Cretois and C. Hourseau, "Skin colour typology and suntanning
//! pathways", International Journal of Cosmetic Science 13(4), 1991, extended with the brown/dark
//! split of S. Del Bino et al., "Relationship between skin response to ultraviolet exposure and skin
//! color type", Pigment Cell Research 19(6), 2006. ITA describes a colour, not a person; it is not a
//! Fitzpatrick type.
//!
//! **Vectorscope.** [`rgb_to_ycbcr`] gives full-range Y′CbCr (Y in `[0, 1]`, Cb and Cr in
//! `[−0.5, 0.5]` for RGB in the unit cube) with the Rec.709 or Rec.601 luma weights; a vectorscope
//! plots `(Cb, Cr)`. The skin-tone line is the conventional "I-line": the +I axis of NTSC YIQ, 33°
//! past the R−Y axis, drawn at [`SKIN_TONE_LINE_DEGREES`] = 123° counter-clockwise from +Cb.
//! Scopes feed display-encoded (sRGB) values, as broadcast scopes do.

use super::delta_e::delta_e_2000;
use super::lab::{chroma, hue_degrees};

/// A Fitzpatrick skin phototype. The names `"I"`..`"VI"` are stable ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FitzpatrickType {
    /// Type I.
    I,
    /// Type II.
    II,
    /// Type III.
    III,
    /// Type IV.
    IV,
    /// Type V.
    V,
    /// Type VI.
    VI,
}

impl FitzpatrickType {
    /// All six types in order.
    pub const ALL: [Self; 6] = [Self::I, Self::II, Self::III, Self::IV, Self::V, Self::VI];

    /// The stable name, a Roman numeral.
    pub const fn name(self) -> &'static str {
        match self {
            Self::I => "I",
            Self::II => "II",
            Self::III => "III",
            Self::IV => "IV",
            Self::V => "V",
            Self::VI => "VI",
        }
    }

    /// The type with the stable name `name` (exact, upper case), if any.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.name() == name)
    }

    /// The type number 1..=6.
    pub const fn number(self) -> u8 {
        self as u8 + 1
    }

    /// The type with the number `n` in 1..=6, if any.
    pub const fn from_number(n: u8) -> Option<Self> {
        match n {
            1 => Some(Self::I),
            2 => Some(Self::II),
            3 => Some(Self::III),
            4 => Some(Self::IV),
            5 => Some(Self::V),
            6 => Some(Self::VI),
            _ => None,
        }
    }
}

/// One reference skin swatch, loaded from the reference table with its provenance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkinSwatch {
    /// The Fitzpatrick type the swatch represents.
    pub fitzpatrick: FitzpatrickType,
    /// The swatch colour in CIELAB under D65 ([`super::lab::rgb_to_lab`]).
    pub lab: [f64; 3],
}

/// The result of [`nearest_swatch`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NearestSwatch {
    /// The index of the swatch in the table.
    pub index: usize,
    /// The CIEDE2000 between the colour and the swatch.
    pub delta_e: f64,
}

/// The swatch closest to `lab` (CIELAB, D65) by CIEDE2000; the first of equally close ones.
/// Swatches with a non-finite component are skipped. `None` for a colour with a non-finite component
/// or when no swatch is usable.
pub fn nearest_swatch(swatches: &[SkinSwatch], lab: [f64; 3]) -> Option<NearestSwatch> {
    if !lab.iter().all(|v| v.is_finite()) {
        return None;
    }
    let mut best: Option<NearestSwatch> = None;
    for (index, swatch) in swatches.iter().enumerate() {
        if !swatch.lab.iter().all(|v| v.is_finite()) {
            continue;
        }
        let delta_e = delta_e_2000(lab, swatch.lab);
        if best.is_none_or(|b| delta_e < b.delta_e) {
            best = Some(NearestSwatch { index, delta_e });
        }
    }
    best
}

/// The individual typology angle of a CIELAB colour in degrees: `atan2(L* − 50, b*)`, which is
/// `atan((L* − 50) / b*)` for the `b* > 0` of every skin colour and stays defined elsewhere
/// (range `(−180°, 180°]`).
#[inline]
pub fn ita_degrees(lab: [f64; 3]) -> f64 {
    (lab[0] - 50.0).atan2(lab[2]).to_degrees()
}

/// The skin colour categories of the individual typology angle (Chardon et al. 1991; Del Bino et
/// al. 2006). The names are stable ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ItaCategory {
    /// ITA > 55°.
    VeryLight,
    /// 41° < ITA ≤ 55°.
    Light,
    /// 28° < ITA ≤ 41°.
    Intermediate,
    /// 10° < ITA ≤ 28°.
    Tan,
    /// −30° < ITA ≤ 10°.
    Brown,
    /// ITA ≤ −30°.
    Dark,
}

impl ItaCategory {
    /// The category of an ITA in degrees; NaN is [`Self::Dark`] (the last category).
    pub fn of_angle(ita: f64) -> Self {
        if ita > 55.0 {
            Self::VeryLight
        } else if ita > 41.0 {
            Self::Light
        } else if ita > 28.0 {
            Self::Intermediate
        } else if ita > 10.0 {
            Self::Tan
        } else if ita > -30.0 {
            Self::Brown
        } else {
            Self::Dark
        }
    }

    /// The category of a CIELAB colour.
    pub fn of_lab(lab: [f64; 3]) -> Self {
        Self::of_angle(ita_degrees(lab))
    }

    /// The stable name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::VeryLight => "very-light",
            Self::Light => "light",
            Self::Intermediate => "intermediate",
            Self::Tan => "tan",
            Self::Brown => "brown",
            Self::Dark => "dark",
        }
    }
}

/// A box in CIELAB lightness, LCh chroma and hue: the documented skin-colour region of the
/// skin-tone sanity metric (section 10.8). It holds no default values: it is derived from the
/// reference swatches with [`Self::enclosing`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkinRegion {
    /// The lightness range `[min, max]`.
    pub lightness: [f64; 2],
    /// The chroma range `[min, max]`.
    pub chroma: [f64; 2],
    /// The hue range `[min, max]` in degrees, measured counter-clockwise from `min`; `min > max`
    /// means the range wraps through 0°.
    pub hue: [f64; 2],
}

impl SkinRegion {
    /// The smallest region holding every swatch, widened by `margin_lightness`, `margin_chroma` and
    /// `margin_hue_degrees` on each side (negative or non-finite margins count as 0). The hue range
    /// is the shortest arc holding every swatch hue. `None` for an empty table or a non-finite
    /// swatch.
    pub fn enclosing(
        swatches: &[SkinSwatch],
        margin_lightness: f64,
        margin_chroma: f64,
        margin_hue_degrees: f64,
    ) -> Option<Self> {
        let margin = |m: f64| if m.is_finite() && m > 0.0 { m } else { 0.0 };
        let (ml, mc, mh) = (margin(margin_lightness), margin(margin_chroma), margin(margin_hue_degrees.min(180.0)));
        if swatches.len() > MAX_REGION_SWATCHES {
            return None;
        }
        let first = swatches.first()?;
        let mut lightness = [first.lab[0]; 2];
        let mut chroma_range = [chroma(first.lab[1], first.lab[2]); 2];
        let mut hues = [0.0_f64; MAX_REGION_SWATCHES];
        let count = swatches.len().min(MAX_REGION_SWATCHES);
        for (slot, swatch) in hues.iter_mut().zip(swatches) {
            if !swatch.lab.iter().all(|v| v.is_finite()) {
                return None;
            }
            let c = chroma(swatch.lab[1], swatch.lab[2]);
            lightness = [lightness[0].min(swatch.lab[0]), lightness[1].max(swatch.lab[0])];
            chroma_range = [chroma_range[0].min(c), chroma_range[1].max(c)];
            *slot = hue_degrees(swatch.lab[1], swatch.lab[2]);
        }
        let hues = &mut hues[..count];
        hues.sort_unstable_by(f64::total_cmp);
        // The shortest arc holding every hue starts just after the largest gap between neighbours.
        let mut gap_end = 0;
        let mut largest_gap = -1.0;
        for i in 0..count {
            let next = if i + 1 < count { hues[i + 1] } else { hues[0] + 360.0 };
            let gap = next - hues[i];
            if gap > largest_gap {
                largest_gap = gap;
                gap_end = (i + 1) % count;
            }
        }
        let start = hues[gap_end];
        let end = hues[(gap_end + count - 1) % count];
        let (hue_min, hue_max) = if 360.0 - largest_gap + 2.0 * mh >= 360.0 {
            (0.0, 360.0)
        } else {
            ((start - mh).rem_euclid(360.0), (end + mh).rem_euclid(360.0))
        };
        Some(Self {
            lightness: [lightness[0] - ml, lightness[1] + ml],
            chroma: [(chroma_range[0] - mc).max(0.0), chroma_range[1] + mc],
            hue: [hue_min, hue_max],
        })
    }

    /// Whether a CIELAB colour lies in the region (bounds included). NaN is outside.
    pub fn contains(&self, lab: [f64; 3]) -> bool {
        let c = chroma(lab[1], lab[2]);
        let in_l = lab[0] >= self.lightness[0] && lab[0] <= self.lightness[1];
        let in_c = c >= self.chroma[0] && c <= self.chroma[1];
        if !(in_l && in_c) {
            return false;
        }
        let [lo, hi] = self.hue;
        if hi - lo >= 360.0 {
            return true;
        }
        let h = hue_degrees(lab[1], lab[2]);
        if lo <= hi { h >= lo && h <= hi } else { h >= lo || h <= hi }
    }
}

/// The largest table [`SkinRegion::enclosing`] accepts; larger tables give `None`.
pub const MAX_REGION_SWATCHES: usize = 256;

/// The luma weights `(K_R, K_B)` of a Y′CbCr encoding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LumaWeights {
    /// The red weight `K_R`.
    pub kr: f64,
    /// The blue weight `K_B`.
    pub kb: f64,
}

impl LumaWeights {
    /// ITU-R BT.709: `K_R = 0.2126`, `K_B = 0.0722`.
    pub const REC709: Self = Self { kr: 0.2126, kb: 0.0722 };
    /// ITU-R BT.601: `K_R = 0.299`, `K_B = 0.114`.
    pub const REC601: Self = Self { kr: 0.299, kb: 0.114 };
}

/// RGB → full-range Y′CbCr: `Y = K_R R + K_G G + K_B B`, `Cb = (B − Y) / (2(1 − K_B))`,
/// `Cr = (R − Y) / (2(1 − K_R))`.
pub fn rgb_to_ycbcr(rgb: [f64; 3], weights: LumaWeights) -> [f64; 3] {
    let LumaWeights { kr, kb } = weights;
    let y = kr * rgb[0] + (1.0 - kr - kb) * rgb[1] + kb * rgb[2];
    [y, (rgb[2] - y) / (2.0 * (1.0 - kb)), (rgb[0] - y) / (2.0 * (1.0 - kr))]
}

/// Full-range Y′CbCr → RGB, the inverse of [`rgb_to_ycbcr`].
pub fn ycbcr_to_rgb(ycbcr: [f64; 3], weights: LumaWeights) -> [f64; 3] {
    let LumaWeights { kr, kb } = weights;
    let [y, cb, cr] = ycbcr;
    let r = y + 2.0 * (1.0 - kr) * cr;
    let b = y + 2.0 * (1.0 - kb) * cb;
    let g = (y - kr * r - kb * b) / (1.0 - kr - kb);
    [r, g, b]
}

/// The angle of the vectorscope skin-tone line, degrees counter-clockwise from +Cb.
pub const SKIN_TONE_LINE_DEGREES: f64 = 123.0;

/// The vectorscope position `(Cb, Cr)` of a (display-encoded) RGB colour with Rec.709 weights.
#[inline]
pub fn vectorscope_position(rgb: [f64; 3]) -> [f64; 2] {
    let [_, cb, cr] = rgb_to_ycbcr(rgb, LumaWeights::REC709);
    [cb, cr]
}

/// The unit direction of the skin-tone line in the `(Cb, Cr)` plane.
pub fn skin_tone_line_direction() -> [f64; 2] {
    let (s, c) = SKIN_TONE_LINE_DEGREES.to_radians().sin_cos();
    [c, s]
}

/// The signed angle in degrees from the skin-tone line to the vectorscope position of `rgb`,
/// counter-clockwise positive (towards yellow), clockwise negative (towards red and magenta), in
/// `(−180°, 180°]`; `None` for a colour without chroma (or NaN).
pub fn skin_tone_line_deviation_degrees(rgb: [f64; 3]) -> Option<f64> {
    let [cb, cr] = vectorscope_position(rgb);
    let radius = cb.hypot(cr);
    if radius.is_nan() || radius <= 1e-12 {
        return None;
    }
    let angle = cr.atan2(cb).to_degrees();
    let d = (angle - SKIN_TONE_LINE_DEGREES).rem_euclid(360.0);
    Some(if d > 180.0 { d - 360.0 } else { d })
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    /// Synthetic test swatches: arbitrary skin-like Lab values made up for the tests, NOT reference
    /// data.
    fn synthetic_swatches() -> [SkinSwatch; 3] {
        [
            SkinSwatch { fitzpatrick: FitzpatrickType::I, lab: [70.0, 10.0, 15.0] },
            SkinSwatch { fitzpatrick: FitzpatrickType::III, lab: [55.0, 14.0, 22.0] },
            SkinSwatch { fitzpatrick: FitzpatrickType::VI, lab: [30.0, 9.0, 12.0] },
        ]
    }

    #[test]
    fn fitzpatrick_names_and_numbers_round_trip() {
        for (i, t) in FitzpatrickType::ALL.into_iter().enumerate() {
            assert_eq!(FitzpatrickType::from_name(t.name()), Some(t));
            assert_eq!(t.number() as usize, i + 1);
            assert_eq!(FitzpatrickType::from_number(t.number()), Some(t));
        }
        assert_eq!(FitzpatrickType::from_name("iv"), None);
        assert_eq!(FitzpatrickType::from_name("VII"), None);
        assert_eq!(FitzpatrickType::from_number(0), None);
        assert_eq!(FitzpatrickType::from_number(7), None);
    }

    #[test]
    fn tool_07_nearest_swatch_by_ciede2000() {
        let table = synthetic_swatches();
        let hit = nearest_swatch(&table, [56.0, 13.0, 21.0]).unwrap();
        assert_eq!(hit.index, 1);
        assert!((hit.delta_e - delta_e_2000([56.0, 13.0, 21.0], table[1].lab)).abs() < 1e-15);
        assert_eq!(nearest_swatch(&table, table[2].lab), Some(NearestSwatch { index: 2, delta_e: 0.0 }));
        assert_eq!(nearest_swatch(&[], [50.0, 0.0, 0.0]), None);
        assert_eq!(nearest_swatch(&table, [f64::NAN, 0.0, 0.0]), None);
        let mut broken = table;
        broken[1].lab[2] = f64::INFINITY;
        assert_eq!(nearest_swatch(&broken, [56.0, 13.0, 21.0]).unwrap().index, 0);
        // Ties go to the first swatch.
        let twins = [table[0], table[0]];
        assert_eq!(nearest_swatch(&twins, [60.0, 0.0, 0.0]).unwrap().index, 0);
    }

    #[test]
    fn ita_angle_and_published_category_limits() {
        assert!(ita_degrees([50.0, 0.0, 20.0]).abs() < 1e-12);
        assert!((ita_degrees([70.0, 0.0, 20.0]) - 45.0).abs() < 1e-12);
        assert!((ita_degrees([30.0, 0.0, 20.0]) + 45.0).abs() < 1e-12);
        let cases = [
            (60.0, ItaCategory::VeryLight),
            (55.0, ItaCategory::Light),
            (41.5, ItaCategory::Light),
            (41.0, ItaCategory::Intermediate),
            (28.0, ItaCategory::Tan),
            (10.0, ItaCategory::Brown),
            (-29.9, ItaCategory::Brown),
            (-30.0, ItaCategory::Dark),
            (-80.0, ItaCategory::Dark),
            (f64::NAN, ItaCategory::Dark),
        ];
        for (angle, category) in cases {
            assert_eq!(ItaCategory::of_angle(angle), category, "{angle}");
        }
        assert_eq!(ItaCategory::of_lab([70.0, 0.0, 20.0]), ItaCategory::Light);
        assert_eq!(ItaCategory::Intermediate.name(), "intermediate");
    }

    #[test]
    fn skin_region_encloses_the_table() {
        let table = synthetic_swatches();
        let region = SkinRegion::enclosing(&table, 2.0, 1.0, 5.0).unwrap();
        for s in &table {
            assert!(region.contains(s.lab));
        }
        assert_eq!(region.lightness, [28.0, 72.0]);
        // Greys and greens are outside.
        assert!(!region.contains([50.0, 0.0, 0.0]));
        assert!(!region.contains([50.0, -20.0, 20.0]));
        assert!(!region.contains([90.0, 10.0, 15.0]));
        assert!(!region.contains([f64::NAN, 10.0, 15.0]));
        // A hue range that wraps through 0°.
        let wrap = [
            SkinSwatch { fitzpatrick: FitzpatrickType::II, lab: [50.0, 10.0, 1.0] },
            SkinSwatch { fitzpatrick: FitzpatrickType::II, lab: [50.0, 10.0, -1.0] },
        ];
        let region = SkinRegion::enclosing(&wrap, 0.0, 0.1, 1.0).unwrap();
        assert!(region.hue[0] > region.hue[1], "{region:?}");
        assert!(region.contains([50.0, 10.05, 0.0]));
        assert!(!region.contains([50.0, -10.05, 0.0]));
        // A huge hue margin covers the circle.
        let all = SkinRegion::enclosing(&table, 0.0, 100.0, 1000.0).unwrap();
        assert!(all.contains([50.0, 1.0, -10.0]));
        // Empty, non-finite, oversized tables.
        assert_eq!(SkinRegion::enclosing(&[], 1.0, 1.0, 1.0), None);
        let bad = [SkinSwatch { fitzpatrick: FitzpatrickType::I, lab: [f64::NAN, 0.0, 0.0] }];
        assert_eq!(SkinRegion::enclosing(&bad, 1.0, 1.0, 1.0), None);
        let big = vec![table[0]; MAX_REGION_SWATCHES + 1];
        assert_eq!(SkinRegion::enclosing(&big, 1.0, 1.0, 1.0), None);
        assert!(SkinRegion::enclosing(&big[1..], f64::NAN, -1.0, f64::INFINITY).is_some());
    }

    #[test]
    fn ycbcr_round_trips_and_has_the_textbook_corners() {
        for weights in [LumaWeights::REC709, LumaWeights::REC601] {
            let white = rgb_to_ycbcr([1.0; 3], weights);
            assert!((white[0] - 1.0).abs() < 1e-12 && white[1].abs() < 1e-12 && white[2].abs() < 1e-12);
            assert!((rgb_to_ycbcr([1.0, 0.0, 0.0], weights)[2] - 0.5).abs() < 1e-12);
            assert!((rgb_to_ycbcr([0.0, 0.0, 1.0], weights)[1] - 0.5).abs() < 1e-12);
            let mut rng = Rng::new(0x7CC);
            for _ in 0..1000 {
                let rgb = [rng.range_f64(0.0, 1.0), rng.range_f64(0.0, 1.0), rng.range_f64(0.0, 1.0)];
                let back = ycbcr_to_rgb(rgb_to_ycbcr(rgb, weights), weights);
                assert!((0..3).all(|c| (back[c] - rgb[c]).abs() < 1e-12));
            }
        }
        assert!((rgb_to_ycbcr([1.0, 0.0, 0.0], LumaWeights::REC709)[0] - 0.2126).abs() < 1e-15);
    }

    #[test]
    fn scopes_skin_tone_line() {
        let [dx, dy] = skin_tone_line_direction();
        assert!((dx.hypot(dy) - 1.0).abs() < 1e-15);
        assert!(dx < 0.0 && dy > 0.0, "between +Cr and −Cb");
        // A colour placed on the line has no deviation.
        let on_line = ycbcr_to_rgb([0.5, 0.1 * dx, 0.1 * dy], LumaWeights::REC709);
        assert!(skin_tone_line_deviation_degrees(on_line).unwrap().abs() < 1e-9);
        // Pure red is at atan2(0.5, −0.1146) ≈ 102.9°, clockwise of the line.
        assert!(skin_tone_line_deviation_degrees([1.0, 0.0, 0.0]).unwrap() < 0.0);
        // Greys have no hue.
        assert_eq!(skin_tone_line_deviation_degrees([0.4; 3]), None);
        assert_eq!(skin_tone_line_deviation_degrees([f64::NAN, 0.0, 0.0]), None);
        // An sRGB skin-like orange-tan (#e0ac69) lies about 21° counter-clockwise of the line (towards
        // yellow).
        let tan = [224.0 / 255.0, 172.0 / 255.0, 105.0 / 255.0];
        let deviation = skin_tone_line_deviation_degrees(tan).unwrap();
        assert!((15.0..25.0).contains(&deviation), "{deviation}");
        let [cb, cr] = vectorscope_position(tan);
        assert!(cb < 0.0 && cr > 0.0);
    }
}
