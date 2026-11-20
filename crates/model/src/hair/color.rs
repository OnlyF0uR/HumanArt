//! Hair colour from melanin, greying and a dye layer, as absorption coefficients for the fibre
//! shader `[HAIR-06]`.
//!
//! **Data and sources** (no other numbers are used):
//!
//! - The absorption of unit concentration of each pigment, linear Rec.709 RGB:
//!   eumelanin `(0.419, 0.697, 1.37)`, pheomelanin `(0.187, 0.4, 1.05)`. Source: pbrt-v3,
//!   `src/materials/hair.cpp`, `HairBSDF::SigmaAFromConcentration` (Pharr, Jakob, Humphreys), the
//!   same values as pbrt-v4, *Physically Based Rendering* 4th ed. (2023), section 9.9.8 "Hair
//!   Absorption Coefficients".
//! - Concentrations "of roughly 8, 1.3, and 0.3 give reasonable representations of black, brown,
//!   and blonde hair": pbrt-v4 section 9.9.8 ([`PRESETS`]); 1.3 eumelanin, 0 pheomelanin is
//!   pbrt-v3's default hair material ("brown-ish hair"), the default here.
//! - The colour-to-absorption fit used for the dye layer,
//!   `σa = (ln c / (5.969 − 0.215 βn + 2.532 βn² − 10.73 βn³ + 5.574 βn⁴ + 0.245 βn⁵))²`:
//!   Chiang, Bitterli, Tappan, Burley, "A Practical and Controllable Hair and Fur Model for
//!   Production Path Tracing", Computer Graphics Forum 35(2), 2016, as implemented in pbrt-v3
//!   `HairBSDF::SigmaAFromReflectance` (constants copied from there). The default azimuthal
//!   roughness βn = 0.3 is pbrt-v3's hair material default.
//!
//! **Units.** `σa` is in pbrt's convention: the transmittance of one pass through the fibre is
//! `exp(−σa · 2 cos γt / cos θt)`, a path length measured in fibre radii. The hair shader (WP-18)
//! must use that expression with these values.
//!
//! **Per strand.** A strand's pigment is the groom's concentrations scaled by its colour variation
//! (`color_id`, uniform over `u16`, and `color.variation`: a factor `1 + v (2 id / 65535 − 1)`),
//! then by `1 − grey / 255` (a grey strand has lost its melanin). The dye layer is applied last:
//! `σa = mix(σa(melanin), σa(dye colour), dye.amount)`, so a dye covers grey strands as well.
//! Unpigmented fibres have `σa = 0` here: no keratin absorption is modelled (no source used).

use super::params::GroomParams;

/// Absorption of unit eumelanin concentration, linear RGB (pbrt-v3/v4, see the module docs).
pub const EUMELANIN_SIGMA_A: [f32; 3] = [0.419, 0.697, 1.37];

/// Absorption of unit pheomelanin concentration, linear RGB (pbrt-v3/v4, see the module docs).
pub const PHEOMELANIN_SIGMA_A: [f32; 3] = [0.187, 0.4, 1.05];

/// A named melanin setting with a published source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MelaninPreset {
    /// The stable id.
    pub name: &'static str,
    /// Eumelanin concentration.
    pub eumelanin: f32,
    /// Pheomelanin concentration.
    pub pheomelanin: f32,
}

/// The melanin presets: the three concentrations pbrt-v4 section 9.9.8 gives.
pub const PRESETS: &[MelaninPreset] = &[
    MelaninPreset { name: "black", eumelanin: 8.0, pheomelanin: 0.0 },
    MelaninPreset { name: "brown", eumelanin: 1.3, pheomelanin: 0.0 },
    MelaninPreset { name: "blonde", eumelanin: 0.3, pheomelanin: 0.0 },
];

/// The preset called `name`.
#[must_use]
pub fn preset(name: &str) -> Option<MelaninPreset> {
    PRESETS.iter().copied().find(|p| p.name == name)
}

/// `σa` of a mix of pigment concentrations (negative values count as 0).
#[must_use]
pub fn sigma_a_from_concentration(eumelanin: f32, pheomelanin: f32) -> [f32; 3] {
    let (e, p) = (eumelanin.max(0.0), pheomelanin.max(0.0));
    std::array::from_fn(|i| e * EUMELANIN_SIGMA_A[i] + p * PHEOMELANIN_SIGMA_A[i])
}

/// `σa` that gives a fibre the multiply-scattered colour `color` (linear RGB) at azimuthal
/// roughness `beta_n` (Chiang et al. 2016). Channels are clamped to `[0.001, 1]`, `beta_n` to
/// `[0, 1]`, the domain of the fit.
#[must_use]
pub fn sigma_a_from_reflectance(color: [f32; 3], beta_n: f32) -> [f32; 3] {
    let b = if beta_n.is_finite() { beta_n.clamp(0.0, 1.0) } else { 0.3 };
    let (b2, b3) = (b * b, b * b * b);
    let denominator = 5.969 - 0.215 * b + 2.532 * b2 - 10.73 * b3 + 5.574 * b2 * b2 + 0.245 * b2 * b3;
    color.map(|c| {
        let c = if c.is_finite() { c.clamp(0.001, 1.0) } else { 1.0 };
        let x = c.ln() / denominator;
        x * x
    })
}

/// The colour settings of a groom, read from its [`GroomParams`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairColor {
    /// Eumelanin concentration.
    pub eumelanin: f32,
    /// Pheomelanin concentration.
    pub pheomelanin: f32,
    /// Strength of the per-strand variation, `0..=1`.
    pub variation: f32,
    /// The dye target colour, linear RGB.
    pub dye: [f32; 3],
    /// The dye opacity, `0..=1`.
    pub dye_amount: f32,
    /// Azimuthal roughness of the shader the dye colour is inverted for.
    pub beta_n: f32,
}

impl HairColor {
    /// The colour settings of `params`.
    #[must_use]
    pub fn from_params(params: &GroomParams) -> Self {
        Self {
            eumelanin: params.eumelanin,
            pheomelanin: params.pheomelanin,
            variation: params.color_variation,
            dye: [params.dye_r, params.dye_g, params.dye_b],
            dye_amount: params.dye_amount,
            beta_n: params.azimuthal_roughness,
        }
    }

    /// Writes these settings back into `params` (values clamped to the parameter ranges).
    pub fn apply(&self, params: &mut GroomParams) {
        let c = |v: f32, lo: f32, hi: f32| if v.is_finite() { v.clamp(lo, hi) } else { lo };
        params.eumelanin = c(self.eumelanin, 0.0, 16.0);
        params.pheomelanin = c(self.pheomelanin, 0.0, 16.0);
        params.color_variation = c(self.variation, 0.0, 1.0);
        params.dye_r = c(self.dye[0], 0.001, 1.0);
        params.dye_g = c(self.dye[1], 0.001, 1.0);
        params.dye_b = c(self.dye[2], 0.001, 1.0);
        params.dye_amount = c(self.dye_amount, 0.0, 1.0);
        params.azimuthal_roughness = c(self.beta_n, 0.0, 1.0);
    }

    /// The groom's natural `σa` (no variation, no grey, no dye).
    #[must_use]
    pub fn natural_sigma_a(&self) -> [f32; 3] {
        sigma_a_from_concentration(self.eumelanin, self.pheomelanin)
    }

    /// The pigment concentrations (eumelanin, pheomelanin) of a strand with `color_id` and
    /// `grey` (the attributes of [`super::Strands`]).
    #[must_use]
    pub fn strand_concentrations(&self, color_id: u16, grey: u8) -> (f32, f32) {
        let v = self.variation.clamp(0.0, 1.0);
        let scale = (1.0 + v * (2.0 * f32::from(color_id) / 65_535.0 - 1.0)) * (1.0 - f32::from(grey) / 255.0);
        (self.eumelanin.max(0.0) * scale, self.pheomelanin.max(0.0) * scale)
    }

    /// The `σa` of a strand with `color_id` and `grey`, dye included.
    #[must_use]
    pub fn strand_sigma_a(&self, color_id: u16, grey: u8) -> [f32; 3] {
        let (e, p) = self.strand_concentrations(color_id, grey);
        let natural = sigma_a_from_concentration(e, p);
        let a = if self.dye_amount.is_finite() { self.dye_amount.clamp(0.0, 1.0) } else { 0.0 };
        if a == 0.0 {
            return natural;
        }
        let dyed = sigma_a_from_reflectance(self.dye, self.beta_n);
        std::array::from_fn(|i| natural[i] + (dyed[i] - natural[i]) * a)
    }
}

impl Default for HairColor {
    fn default() -> Self {
        Self::from_params(&GroomParams::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hair_06_concentration_matches_pbrt() {
        // pbrt's default material: 1.3 eumelanin, 0 pheomelanin.
        let s = sigma_a_from_concentration(1.3, 0.0);
        let want = [1.3 * 0.419, 1.3 * 0.697, 1.3 * 1.37];
        assert!(s.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-6), "{s:?}");
        let mixed = sigma_a_from_concentration(1.0, 2.0);
        assert!((mixed[2] - (1.37 + 2.0 * 1.05)).abs() < 1e-6);
        assert_eq!(sigma_a_from_concentration(-1.0, -1.0), [0.0; 3]);
        // More eumelanin absorbs more in every channel, and blue most (hair is warm-toned).
        let black = preset("black").map(|p| sigma_a_from_concentration(p.eumelanin, p.pheomelanin));
        let blonde = preset("blonde").map(|p| sigma_a_from_concentration(p.eumelanin, p.pheomelanin));
        let (Some(black), Some(blonde)) = (black, blonde) else { panic!("presets") };
        assert!((0..3).all(|i| black[i] > blonde[i]));
        assert!(blonde[2] > blonde[1] && blonde[1] > blonde[0]);
        assert_eq!(HairColor::default().natural_sigma_a(), sigma_a_from_concentration(1.3, 0.0));
    }

    #[test]
    fn hair_06_reflectance_fit_is_chiang() {
        // White hair absorbs nothing; darker colours absorb more; c = e^-1 at beta_n = 0 gives
        // (1 / 5.969)^2.
        assert_eq!(sigma_a_from_reflectance([1.0; 3], 0.3), [0.0; 3]);
        let s = sigma_a_from_reflectance([(-1.0f32).exp(); 3], 0.0);
        assert!((s[0] - (1.0 / 5.969f32).powi(2)).abs() < 1e-6, "{s:?}");
        let dark = sigma_a_from_reflectance([0.05; 3], 0.3);
        let light = sigma_a_from_reflectance([0.5; 3], 0.3);
        assert!(dark[0] > light[0]);
        // Hostile input is clamped, never NaN.
        let odd = sigma_a_from_reflectance([f32::NAN, -1.0, 7.0], f32::INFINITY);
        assert!(odd.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn hair_06_grey_variation_and_dye() {
        let mut params = GroomParams::default();
        let mut color = HairColor::from_params(&params);
        color.variation = 0.5;
        // Grey strands lose their pigment.
        assert_eq!(color.strand_sigma_a(1234, 255), [0.0; 3]);
        // The variation spans +-variation around the groom value.
        let (lo, _) = color.strand_concentrations(0, 0);
        let (hi, _) = color.strand_concentrations(u16::MAX, 0);
        assert!((lo - 0.65).abs() < 1e-5 && (hi - 1.95).abs() < 1e-5, "{lo} {hi}");
        // A full dye replaces melanin, grey strands included; half a dye is halfway.
        color.dye = [0.6, 0.1, 0.05];
        color.dye_amount = 1.0;
        let dyed = sigma_a_from_reflectance(color.dye, color.beta_n);
        assert_eq!(color.strand_sigma_a(9, 255), dyed);
        color.dye_amount = 0.5;
        let half = color.strand_sigma_a(32_768, 0);
        let (e, p) = color.strand_concentrations(32_768, 0);
        let nat = sigma_a_from_concentration(e, p);
        assert!((0..3).all(|i| (half[i] - nat[i].midpoint(dyed[i])).abs() < 1e-5));
        // The settings round-trip through the document parameters.
        color.apply(&mut params);
        assert_eq!(HairColor::from_params(&params), color);
        assert!(params.check().is_ok());
        assert!(params.entries().iter().any(|(k, _)| *k == "dye.amount"));
    }
}
