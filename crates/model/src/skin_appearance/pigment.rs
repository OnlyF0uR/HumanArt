//! The physically-inspired skin colour model `[SKN-10]`: pigment parameters to diffuse albedo,
//! subsurface-scattering radius per colour channel and a reflectance spectrum.
//!
//! The model (derivation in `docs/shader-notes.md`, section "Skin pigment model") is spectral,
//! evaluated on the 81 bands of [`BANDS`] (380 to 780 nm, 5 nm):
//!
//! - **Epidermis** absorption (Donner & Jensen 2006, "A Spectral BSSRDF for Shading Human Skin",
//!   EGSR, eqs. 3 to 6), in mm⁻¹ with λ in nm: eumelanin `6.6e10 · λ^-3.33`, pheomelanin
//!   `2.9e14 · λ^-4.75`, baseline `0.0244 + 8.53 · exp(-(λ - 154) / 66.2)` (Jacques 1998, "Skin
//!   Optics", OMLC News), mixed as `Cm (βm σ_eu + (1 - βm) σ_pheo) + (1 - Cm) σ_base`, plus
//!   β-carotene `2.303 · ε_car · c` (PhotochemCAD molar extinction, c in mol/L).
//! - **Dermis** absorption (Donner & Jensen 2006, eq. 7): `Ch (γ σ_oxy + (1 - γ) σ_deoxy) +
//!   (1 - Ch) σ_base`, where the whole-blood coefficients come from Prahl's molar extinction table
//!   as `2.303 · ε · 150 g/L / 64 500 g/mol` (Prahl, OMLC, "Tabulated molar extinction coefficient
//!   for hemoglobin in water").
//! - **Scattering** (Donner & Jensen 2006, eq. 8): reduced scattering `14.74 · λ^-0.22 + 2.2e11 ·
//!   λ^-4` mm⁻¹ in the epidermis, half of it in the dermis.
//! - **Reflectance**: the dermis is a semi-infinite medium with the total diffuse reflectance of
//!   the dipole diffusion model (Jensen & Buhler 2002, `Rd = α'/2 (1 + e^{-4/3 A √(3(1-α'))})
//!   e^{-√(3(1-α'))}`, η = 1.4 as in Donner & Jensen); the epidermis is an absorbing filter
//!   crossed twice (the round-trip path of Jacques 1998): `R = exp(-2 d σa_epi) · Rd`. Specular
//!   reflection at the surface is not part of the albedo (the BRDF owns it, `[SHD-01]`).
//! - **Colour**: `XYZ = Σ R S x̄ / Σ S ȳ` with the CIE 1931 2° observer and illuminant D65, then
//!   linear Rec.709 (no adaptation: D65 is the Rec.709 white).
//! - **SSS radius**: the dermis diffusion length `1 / σtr`, `σtr = √(3 σa (σa + σs'))`, averaged
//!   per channel with the positive part of the channel's spectral weight times `R(λ)`.
//! - **Melanin packaging**: eumelanin sits in discrete packets (melanosomes and their clusters
//!   and caps), not in solution, so its mean absorption is flattened as Duysens (1956) derived
//!   for suspensions: [`packaged_absorption`], `(1 - e^{-σ s}) / s`, with an effective packet path
//!   `s` ([`Mapping::eumelanin_packet_mm`]). Without it the homogeneous model turns medium and
//!   dark skin far too yellow (b* 30 to 41 where ISSA measures 12 to 19; see below). Pheomelanin
//!   (small, sparsely pigmented pheomelanosomes) stays homogeneous.
//! - **Blood packaging**: blood is confined to dermal vessels, so light inside a vessel is
//!   attenuated before it reaches the vessel's centre and the deep Soret and Q bands of haemoglobin
//!   flatten the same way. Verkruysse et al. (1997, "Modelling light distributions of homogeneous
//!   versus discrete absorbers in light irradiated turbid media", Phys. Med. Biol. 42, 51-65)
//!   introduced the vessel correction factor and van Veen, Verkruysse & Sterenborg (2002,
//!   "Diffuse-reflectance spectroscopy from 500 to 1060 nm by correction for inhomogeneously
//!   distributed absorbers", Opt. Lett. 27, 246-248) gave it as `C = (1 - e^{-2 σ r}) / (2 σ r)`
//!   for vessels of radius `r`: the Duysens form with a path of `2r`. Each haemoglobin species is
//!   packaged on its own ([`Mapping::blood_vessel_radius_mm`], fitted: 0.013 mm), so the band
//!   table stays linear in the blood fraction and the oxygenation and the GPU compositor needs no
//!   change. Packaging the oxy/deoxy mix exactly would absorb up to 11 % more in single bands,
//!   which changes the colour by at most CIEDE2000 0.34
//!   (`blood_packaging_per_species_is_close_to_the_mix`).
//!
//! The registry stores every pigment parameter normalised to 0..1 (`data/template/params/skin.json`).
//! [`PigmentParams::physical`] maps them to the physical quantities with the constants of
//! [`Mapping`]. These constants are **model choices, not measurements**: the blood range is Donner &
//! Jensen's Table 1 range (0.001 to 0.1), the epidermis range is centred on the 60 µm Jacques
//! assumed when he derived those fractions, and the carotene scale is uncalibrated. Three constants
//! are **fitted to measured data**: the eumelanin packet path (0.035 mm) and the melanin range
//! (packet volume fraction up to 1), chosen on a grid (packet 0.02 to 0.06 mm, maximum 0.5 to 1)
//! to minimise the CIEDE2000 of the colour fits, and the blood vessel radius (0.013 mm), chosen
//! on a grid (0 to 0.02 mm, with the packet path 0.03 to 0.04 mm) to minimise the largest
//! spectral RMS of the six ITA bins with the colour tolerances kept, all against the ISSA
//! reference swatches
//! (`data/test-scenes/skin-reference.json`: Lu et al. 2025, Sci Data 12, 487, 15 256
//! specular-included spectrophotometer records, CC BY 4.0). The packet path is an effective
//! optical parameter, not a measured melanosome size. `[AC-WP13-1]`: every ISSA bin mean (ITA
//! category, L* band, body site) is reproduced within CIEDE2000 1.0 by
//! [`PigmentModel::fit_measured_lab`], with the specular component of the instrument
//! ([`SCI_SURFACE_REFLECTANCE`]) added.

use humanart_math::color::delta_e::delta_e_2000;
use humanart_math::color::lab::{rgb_to_lab, xyz_to_lab};
use humanart_math::color::matrix::mul_vec;
use humanart_math::color::spaces::XYZ_TO_REC709;

use super::spectra_table::BANDS;

/// Number of spectral bands.
pub const BAND_COUNT: usize = 81;
/// `f32` words per band in the GPU table ([`PigmentModel::write_gpu_table`]).
pub const GPU_WORDS_PER_BAND: usize = 16;
/// Bytes of the GPU band table.
pub const GPU_TABLE_BYTES: usize = BAND_COUNT * GPU_WORDS_PER_BAND * 4;
/// Refractive index of skin (Donner & Jensen 2006, section 3.5).
pub const SKIN_IOR: f64 = 1.4;
/// Reflectance of the skin surface at normal incidence, `((η - 1) / (η + 1))²` with η =
/// [`SKIN_IOR`] (Fresnel). A specular-included (SCI, d/8°) spectrophotometer adds this spectrally
/// flat specular component to the subsurface albedo the model computes: the comparison with
/// measured SCI data ([`PigmentModel::measured_lab`]) adds it back.
pub const SCI_SURFACE_REFLECTANCE: f64 = (SKIN_IOR - 1.0) * (SKIN_IOR - 1.0) / ((SKIN_IOR + 1.0) * (SKIN_IOR + 1.0));
/// Grams of haemoglobin per litre of whole blood (Prahl's conversion example).
pub const BLOOD_HB_G_PER_L: f64 = 150.0;
/// Gram molecular weight of haemoglobin (Prahl).
pub const HB_G_PER_MOL: f64 = 64_500.0;

/// Eumelanin absorption coefficient in mm⁻¹ (Donner & Jensen 2006, eq. 3).
pub fn eumelanin_absorption(nm: f64) -> f64 {
    6.6e10 * nm.powf(-3.33)
}

/// Pheomelanin absorption coefficient in mm⁻¹ (Donner & Jensen 2006, eq. 4).
pub fn pheomelanin_absorption(nm: f64) -> f64 {
    2.9e14 * nm.powf(-4.75)
}

/// Mean absorption coefficient of a medium whose absorber is packed in discrete packets: the
/// flattening of Duysens (1956, "The flattening of the absorption spectrum of suspensions, as
/// compared to that of solutions", Biochim. Biophys. Acta 19, 1-12) for randomly placed packets
/// of optical path `packet_mm`: `(1 - e^{-σ s}) / s` per unit packet volume fraction, which is
/// `σ` for small packets and saturates at `1 / s` where a packet is opaque. Linear in the packet
/// concentration, so the GPU band table can hold it per band.
pub fn packaged_absorption(sigma: f64, packet_mm: f64) -> f64 {
    if packet_mm > 0.0 { -(-sigma * packet_mm).exp_m1() / packet_mm } else { sigma }
}

/// Baseline (melanin- and blood-free) skin absorption in mm⁻¹ (Jacques 1998; Donner & Jensen
/// 2006, eq. 5).
pub fn baseline_absorption(nm: f64) -> f64 {
    0.0244 + 8.53 * (-(nm - 154.0) / 66.2).exp()
}

/// Reduced scattering coefficient of the epidermis in mm⁻¹ (Donner & Jensen 2006, eq. 8).
pub fn epidermis_scattering(nm: f64) -> f64 {
    14.74 * nm.powf(-0.22) + 2.2e11 * nm.powi(-4)
}

/// Absorption coefficient in mm⁻¹ of whole blood from a molar extinction coefficient in
/// cm⁻¹/M (Prahl's conversion with 150 g/L).
pub fn blood_absorption(molar_extinction: f64) -> f64 {
    2.303 * molar_extinction * BLOOD_HB_G_PER_L / HB_G_PER_MOL / 10.0
}

/// The diffuse Fresnel reflectance `Fdr(η)` (Egan & Hilgeman fit, as used by Jensen et al. 2001).
pub fn diffuse_fresnel(eta: f64) -> f64 {
    -1.440 / (eta * eta) + 0.710 / eta + 0.668 + 0.0636 * eta
}

/// Total diffuse reflectance of a semi-infinite medium with reduced albedo `alpha_prime`
/// (Jensen & Buhler 2002).
pub fn diffuse_reflectance(alpha_prime: f64, eta: f64) -> f64 {
    let fdr = diffuse_fresnel(eta);
    let a = (1.0 + fdr) / (1.0 - fdr);
    let s = (3.0 * (1.0 - alpha_prime.clamp(0.0, 1.0))).sqrt();
    0.5 * alpha_prime * (1.0 + (-4.0 / 3.0 * a * s).exp()) * (-s).exp()
}

/// The registry-to-physical mapping constants (model choices, two of them fitted to measured data;
/// see the module documentation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mapping {
    /// Melanin (packet) volume fraction of the epidermis at melanin = 1 (`Cm = max · m²`).
    pub melanin_max: f64,
    /// Blood volume fraction at haemoglobin = 0.
    pub blood_min: f64,
    /// Blood volume fraction at haemoglobin = 1 (log-linear in between).
    pub blood_max: f64,
    /// Epidermis thickness in mm at thickness = 0.
    pub epidermis_min_mm: f64,
    /// Epidermis thickness in mm at thickness = 1.
    pub epidermis_max_mm: f64,
    /// β-carotene molar concentration in the epidermis at carotene = 1 (uncalibrated).
    pub carotene_max_molar: f64,
    /// Optical path in mm through one eumelanin packet ([`packaged_absorption`]); 0 is the
    /// homogeneous model of Donner & Jensen 2006. Baked into the band table by
    /// [`PigmentModel::new`], not a shader input.
    pub eumelanin_packet_mm: f64,
    /// Radius in mm of the dermal vessels that hold the blood: each haemoglobin species is
    /// packaged with the vessel factor of van Veen et al. (2002), [`packaged_absorption`] with a
    /// path of twice the radius; 0 is homogeneous blood. Baked into the band table.
    pub blood_vessel_radius_mm: f64,
}

impl Mapping {
    /// The default mapping.
    pub const DEFAULT: Self = Self {
        melanin_max: 1.0,
        blood_min: 0.001,
        blood_max: 0.1,
        epidermis_min_mm: 0.03,
        epidermis_max_mm: 0.09,
        carotene_max_molar: 1.0e-5,
        eumelanin_packet_mm: 0.035,
        blood_vessel_radius_mm: 0.013,
    };
}

impl Default for Mapping {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// The pigment parameters as the registry and the parameter maps store them, each 0..1
/// (material targets `skin.melanin`, `skin.eumelanin-ratio`, `skin.haemoglobin`,
/// `skin.oxygenation`, `skin.carotene`, `skin.epidermis-thickness`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PigmentParams {
    /// Total melanin.
    pub melanin: f64,
    /// Share of eumelanin in the melanin.
    pub eumelanin_ratio: f64,
    /// Blood in the dermis.
    pub haemoglobin: f64,
    /// Oxygen saturation of the blood.
    pub oxygenation: f64,
    /// β-carotene.
    pub carotene: f64,
    /// Epidermis thickness.
    pub epidermal_thickness: f64,
}

impl Default for PigmentParams {
    /// The registry defaults of `data/template/params/skin.json`.
    fn default() -> Self {
        Self {
            melanin: 0.3,
            eumelanin_ratio: 0.7,
            haemoglobin: 0.3,
            oxygenation: 0.7,
            carotene: 0.1,
            epidermal_thickness: 0.5,
        }
    }
}

/// The material targets of [`PigmentParams`], in field order.
pub const PIGMENT_TARGETS: [&str; 6] = [
    "skin.melanin",
    "skin.eumelanin-ratio",
    "skin.haemoglobin",
    "skin.oxygenation",
    "skin.carotene",
    "skin.epidermis-thickness",
];

impl PigmentParams {
    /// The parameters as an array in [`PIGMENT_TARGETS`] order.
    pub const fn to_array(&self) -> [f64; 6] {
        [
            self.melanin,
            self.eumelanin_ratio,
            self.haemoglobin,
            self.oxygenation,
            self.carotene,
            self.epidermal_thickness,
        ]
    }

    /// The inverse of [`PigmentParams::to_array`].
    pub const fn from_array(a: [f64; 6]) -> Self {
        Self {
            melanin: a[0],
            eumelanin_ratio: a[1],
            haemoglobin: a[2],
            oxygenation: a[3],
            carotene: a[4],
            epidermal_thickness: a[5],
        }
    }

    /// Sets the parameter named by a material target; false for an unknown target. Values are
    /// clamped to 0..1 and a non-finite value leaves the parameter unchanged.
    pub fn set(&mut self, target: &str, value: f64) -> bool {
        let Some(index) = PIGMENT_TARGETS.iter().position(|t| *t == target) else { return false };
        if value.is_finite() {
            let mut a = self.to_array();
            a[index] = value.clamp(0.0, 1.0);
            *self = Self::from_array(a);
        }
        true
    }

    /// Every value clamped to 0..1 (non-finite values become the default).
    pub fn clamped(&self) -> Self {
        let defaults = Self::default().to_array();
        let mut a = self.to_array();
        for (v, d) in a.iter_mut().zip(defaults) {
            *v = if v.is_finite() { v.clamp(0.0, 1.0) } else { d };
        }
        Self::from_array(a)
    }

    /// The physical quantities under `mapping`.
    pub fn physical(&self, mapping: &Mapping) -> PhysicalPigment {
        let p = self.clamped();
        PhysicalPigment {
            melanin_fraction: mapping.melanin_max * p.melanin * p.melanin,
            eumelanin_blend: p.eumelanin_ratio,
            blood_fraction: mapping.blood_min * (mapping.blood_max / mapping.blood_min).powf(p.haemoglobin),
            oxygenation: p.oxygenation,
            carotene_molar: mapping.carotene_max_molar * p.carotene,
            epidermis_mm: mapping.epidermis_min_mm
                + (mapping.epidermis_max_mm - mapping.epidermis_min_mm) * p.epidermal_thickness,
        }
    }
}

/// The physical inputs of the spectral model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysicalPigment {
    /// Melanosome volume fraction of the epidermis `Cm`.
    pub melanin_fraction: f64,
    /// Eumelanin share `βm`.
    pub eumelanin_blend: f64,
    /// Blood volume fraction of the dermis `Ch`.
    pub blood_fraction: f64,
    /// Blood oxygenation `γ`.
    pub oxygenation: f64,
    /// β-carotene concentration in mol/L.
    pub carotene_molar: f64,
    /// Epidermis thickness in mm.
    pub epidermis_mm: f64,
}

/// What the model gives for one parameter set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PigmentResponse {
    /// Diffuse albedo, linear Rec.709.
    pub albedo: [f64; 3],
    /// Subsurface-scattering radius (dermis diffusion length) per channel, in mm.
    pub sss_radius_mm: [f64; 3],
    /// Diffuse reflectance per band.
    pub reflectance: [f64; BAND_COUNT],
}

/// The per-band coefficients, derived once from the band table.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
struct BandCoefficients {
    /// Contribution of a unit reflectance in this band to linear Rec.709 R, G, B.
    rgb: [f64; 3],
    /// Contribution of a unit reflectance in this band to CIE XYZ under D65 (white Y = 1).
    xyz: [f64; 3],
    /// The positive part of `rgb`, normalised per channel over all bands (SSS weights).
    sss: [f64; 3],
    eumelanin: f64,
    pheomelanin: f64,
    baseline: f64,
    oxy_blood: f64,
    deoxy_blood: f64,
    /// mm⁻¹ per mol/L of β-carotene.
    carotene: f64,
    /// Epidermis reduced scattering.
    scattering: f64,
}

/// The spectral pigment model with its derived band coefficients. Build once, evaluate often.
#[derive(Clone, Debug)]
pub struct PigmentModel {
    bands: [BandCoefficients; BAND_COUNT],
    mapping: Mapping,
}

impl Default for PigmentModel {
    fn default() -> Self {
        Self::new(Mapping::DEFAULT)
    }
}

/// A Lab fit's result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LabFit {
    /// The parameters found.
    pub params: PigmentParams,
    /// The CIEDE2000 difference left.
    pub delta_e: f64,
}

impl PigmentModel {
    /// Derives the band coefficients.
    pub fn new(mapping: Mapping) -> Self {
        let mut bands = [BandCoefficients::default(); BAND_COUNT];
        let white_y: f64 = BANDS.iter().map(|b| b[4] * b[2]).sum();
        let mut sss_sum = [0.0_f64; 3];
        for (out, band) in bands.iter_mut().zip(BANDS.iter()) {
            let [nm, x, y, z, d65, hbo2, hb, car] = *band;
            let xyz = [x * d65 / white_y, y * d65 / white_y, z * d65 / white_y];
            out.rgb = mul_vec(&XYZ_TO_REC709, xyz);
            out.xyz = xyz;
            out.sss = out.rgb.map(|v| v.max(0.0));
            for (sum, w) in sss_sum.iter_mut().zip(out.sss) {
                *sum += w;
            }
            out.eumelanin = packaged_absorption(eumelanin_absorption(nm), mapping.eumelanin_packet_mm);
            out.pheomelanin = pheomelanin_absorption(nm);
            out.baseline = baseline_absorption(nm);
            let vessel = 2.0 * mapping.blood_vessel_radius_mm;
            out.oxy_blood = packaged_absorption(blood_absorption(hbo2), vessel);
            out.deoxy_blood = packaged_absorption(blood_absorption(hb), vessel);
            out.carotene = 2.303 * car / 10.0;
            out.scattering = epidermis_scattering(nm);
        }
        for band in &mut bands {
            for (w, sum) in band.sss.iter_mut().zip(sss_sum) {
                if sum > 0.0 {
                    *w /= sum;
                }
            }
        }
        Self { bands, mapping }
    }

    /// The mapping in use.
    pub const fn mapping(&self) -> &Mapping {
        &self.mapping
    }

    /// The band wavelengths in nm.
    pub fn wavelengths() -> [f64; BAND_COUNT] {
        let mut out = [0.0; BAND_COUNT];
        for (o, b) in out.iter_mut().zip(BANDS.iter()) {
            *o = b[0];
        }
        out
    }

    /// Evaluates the model for physical inputs.
    pub fn evaluate_physical(&self, p: &PhysicalPigment) -> PigmentResponse {
        let mut reflectance = [0.0; BAND_COUNT];
        let mut albedo = [0.0; 3];
        let mut radius_num = [0.0; 3];
        let mut radius_den = [0.0; 3];
        let cm = p.melanin_fraction.clamp(0.0, 1.0);
        let bm = p.eumelanin_blend.clamp(0.0, 1.0);
        let ch = p.blood_fraction.clamp(0.0, 1.0);
        let g = p.oxygenation.clamp(0.0, 1.0);
        for (r, b) in reflectance.iter_mut().zip(self.bands.iter()) {
            let sigma_epi = cm * (bm * b.eumelanin + (1.0 - bm) * b.pheomelanin)
                + (1.0 - cm) * b.baseline
                + b.carotene * p.carotene_molar.max(0.0);
            let sigma_derm = ch * (g * b.oxy_blood + (1.0 - g) * b.deoxy_blood) + (1.0 - ch) * b.baseline;
            let scatter_derm = 0.5 * b.scattering;
            let alpha = scatter_derm / (scatter_derm + sigma_derm);
            let rd = diffuse_reflectance(alpha, SKIN_IOR);
            let value = (-2.0 * p.epidermis_mm.max(0.0) * sigma_epi).exp() * rd;
            *r = value;
            let sigma_tr = (3.0 * sigma_derm * (sigma_derm + scatter_derm)).sqrt();
            let length = 1.0 / sigma_tr.max(1e-9);
            for c in 0..3 {
                albedo[c] += b.rgb[c] * value;
                let w = b.sss[c] * value;
                radius_num[c] += w * length;
                radius_den[c] += w;
            }
        }
        let mut sss_radius_mm = [0.0; 3];
        for c in 0..3 {
            sss_radius_mm[c] = if radius_den[c] > 0.0 { radius_num[c] / radius_den[c] } else { 0.0 };
        }
        // Very dark parameter sets fall slightly outside the Rec.709 gamut (negative blue); the
        // stored albedo is clamped to the representable range.
        PigmentResponse { albedo: albedo.map(|v| v.clamp(0.0, 1.0)), sss_radius_mm, reflectance }
    }

    /// Evaluates the model for registry parameters.
    pub fn evaluate(&self, params: &PigmentParams) -> PigmentResponse {
        self.evaluate_physical(&params.physical(&self.mapping))
    }

    /// The diffuse albedo (linear Rec.709) for registry parameters.
    pub fn albedo(&self, params: &PigmentParams) -> [f64; 3] {
        self.evaluate(params).albedo
    }

    /// Writes the GPU band table: per band 16 little-endian `f32`s `[rgb.r, rgb.g, rgb.b, sss.r,
    /// sss.g, sss.b, eumelanin, pheomelanin, baseline, oxy blood, deoxy blood, carotene per mol/L,
    /// epidermis scattering, 0, 0, 0]` (mm⁻¹). `out` must hold [`GPU_TABLE_BYTES`]; returns false
    /// otherwise.
    pub fn write_gpu_table(&self, out: &mut [u8]) -> bool {
        if out.len() < GPU_TABLE_BYTES {
            return false;
        }
        for (chunk, b) in out.as_chunks_mut::<{ GPU_WORDS_PER_BAND * 4 }>().0.iter_mut().zip(self.bands.iter()) {
            let words = [
                b.rgb[0],
                b.rgb[1],
                b.rgb[2],
                b.sss[0],
                b.sss[1],
                b.sss[2],
                b.eumelanin,
                b.pheomelanin,
                b.baseline,
                b.oxy_blood,
                b.deoxy_blood,
                b.carotene,
                b.scattering,
                0.0,
                0.0,
                0.0,
            ];
            for (dst, w) in chunk.as_chunks_mut::<4>().0.iter_mut().zip(words) {
                dst.copy_from_slice(&(w as f32).to_le_bytes());
            }
        }
        true
    }

    /// The CIELAB (D65, CIE 1931 2°) colour a measuring instrument reports for registry
    /// parameters: the reflectance spectrum plus a spectrally flat `surface_reflectance` (the
    /// specular component of a specular-included instrument, [`SCI_SURFACE_REFLECTANCE`]; 0 for
    /// specular-excluded data), integrated over the bands. Unlike `rgb_to_lab(albedo)` this does
    /// not clip to the Rec.709 gamut.
    pub fn measured_lab(&self, params: &PigmentParams, surface_reflectance: f64) -> [f64; 3] {
        let response = self.evaluate(params);
        let mut xyz = [0.0; 3];
        let mut white = [0.0; 3];
        for (r, b) in response.reflectance.iter().zip(self.bands.iter()) {
            for c in 0..3 {
                xyz[c] += b.xyz[c] * (r + surface_reflectance);
                white[c] += b.xyz[c];
            }
        }
        xyz_to_lab(xyz, white)
    }

    /// Finds the melanin, eumelanin ratio and haemoglobin that reproduce a CIELAB (D65) colour,
    /// keeping the other parameters of `start`: a coarse grid then a shrinking pattern search on
    /// CIEDE2000. This is the inverse used to fit a photo's skin tone to the albedo.
    pub fn fit_lab(&self, target: [f64; 3], start: &PigmentParams) -> LabFit {
        fit(start, |p| delta_e_2000(rgb_to_lab(self.albedo(p)), target))
    }

    /// [`PigmentModel::fit_lab`] against a measured colour: the cost compares
    /// [`PigmentModel::measured_lab`] with `surface_reflectance` added. This is the inverse used to
    /// validate the mapping against measured swatches (`[AC-WP13-1]`,
    /// `data/test-scenes/skin-reference.json`).
    pub fn fit_measured_lab(&self, target: [f64; 3], start: &PigmentParams, surface_reflectance: f64) -> LabFit {
        fit(start, |p| delta_e_2000(self.measured_lab(p, surface_reflectance), target))
    }
}

/// Grid candidates the pattern search of [`fit`] starts from.
const FIT_STARTS: usize = 4;

/// Minimises `cost` over melanin, eumelanin ratio and haemoglobin from `start`: a coarse grid, then
/// from each of the best [`FIT_STARTS`] grid points (and `start`) a pattern search that tries ± step
/// on each axis and halves the step when nothing improves. Several starts keep the search out of
/// the shallow side valleys the eumelanin/pheomelanin trade-off makes.
fn fit(start: &PigmentParams, cost: impl Fn(&PigmentParams) -> f64) -> LabFit {
    let first = start.clamped();
    let mut starts = [(cost(&first), first); FIT_STARTS + 1];
    for i in 0..=16 {
        for j in 0..=4 {
            for k in 0..=4 {
                let mut p = first;
                p.melanin = f64::from(i) / 16.0;
                p.eumelanin_ratio = f64::from(j) / 4.0;
                p.haemoglobin = f64::from(k) / 4.0;
                let c = cost(&p);
                // Keep the best FIT_STARTS grid points in starts[1..], sorted by cost.
                if c < starts[FIT_STARTS].0 {
                    starts[FIT_STARTS] = (c, p);
                    let mut n = FIT_STARTS;
                    while n > 1 && starts[n].0 < starts[n - 1].0 {
                        starts.swap(n, n - 1);
                        n -= 1;
                    }
                }
            }
        }
    }
    let mut best = LabFit { params: first, delta_e: f64::INFINITY };
    for (c, p) in starts {
        if !c.is_finite() {
            continue;
        }
        let found = pattern_search(p, c, &cost);
        if found.delta_e < best.delta_e {
            best = found;
        }
    }
    best
}

/// The pattern search of [`fit`] from one start.
fn pattern_search(mut best: PigmentParams, mut best_cost: f64, cost: &impl Fn(&PigmentParams) -> f64) -> LabFit {
    let mut step = 0.0625;
    while step > 1e-4 {
        let mut improved = false;
        for axis in 0..3 {
            for sign in [-1.0, 1.0] {
                let mut p = best;
                let field = match axis {
                    0 => &mut p.melanin,
                    1 => &mut p.eumelanin_ratio,
                    _ => &mut p.haemoglobin,
                };
                *field = (*field + sign * step).clamp(0.0, 1.0);
                let c = cost(&p);
                if c < best_cost {
                    best = p;
                    best_cost = c;
                    improved = true;
                }
            }
        }
        if !improved {
            step *= 0.5;
        }
    }
    LabFit { params: best, delta_e: best_cost }
}

#[cfg(test)]
mod tests {
    use humanart_core::rng::Rng;
    use humanart_math::color::skin::{ItaCategory, ita_degrees};

    use super::*;

    #[test]
    fn published_formulas_reproduce_their_quoted_values() {
        // Jacques 1998 quotes mua.mel (cm^-1, 6.6e11 nm^-3.33) = 230 at 694 nm, 170 at 755 nm,
        // 55 at 1064 nm; Donner & Jensen's eq. 3 is the same law in mm^-1.
        for (nm, quoted_cm) in [(694.0, 230.0), (755.0, 170.0), (1064.0, 55.0)] {
            let mm = eumelanin_absorption(nm);
            assert!((mm * 10.0 - quoted_cm).abs() / quoted_cm < 0.03, "{nm}: {}", mm * 10.0);
        }
        // Jacques 1998 table: baseline 0.268 cm^-1 at 694 nm, 0.254 at 755, 0.244 at 1064.
        for (nm, quoted_cm) in [(694.0, 0.268), (755.0, 0.254), (1064.0, 0.244)] {
            assert!((baseline_absorption(nm) * 10.0 - quoted_cm).abs() < 0.002, "{nm}");
        }
        // Whole blood with Prahl's 150 g/L: at 550 nm the oxy coefficient is a few tens of mm^-1.
        let row = BANDS.iter().find(|b| b[0] == 550.0).map_or(0.0, |b| b[5]);
        assert_eq!(row, 43016.0, "Prahl's HbO2 extinction at 550 nm");
        assert!((blood_absorption(row) - 2.303 * 43016.0 * 150.0 / 64500.0 / 10.0).abs() < 1e-9);
    }

    #[test]
    fn band_table_is_the_cie_observer_under_d65() {
        let model = PigmentModel::default();
        // A perfect white reflector gives Rec.709 white (1, 1, 1) within the 5 nm sampling error.
        let mut white = [0.0; 3];
        for b in &model.bands {
            for (sum, v) in white.iter_mut().zip(b.rgb) {
                *sum += v;
            }
        }
        for c in white {
            assert!((c - 1.0).abs() < 0.01, "{white:?}");
        }
        // The wavelengths are 380..=780 in 5 nm steps.
        let w = PigmentModel::wavelengths();
        assert_eq!((w[0], w[80]), (380.0, 780.0));
        assert!(w.windows(2).all(|p| (p[1] - p[0] - 5.0).abs() < 1e-12));
    }

    #[test]
    fn diffusion_reflectance_limits() {
        assert!((diffuse_reflectance(1.0, 1.4) - 1.0).abs() < 1e-12);
        assert_eq!(diffuse_reflectance(0.0, 1.4), 0.0);
        let mut last = 0.0;
        for i in 1..=100 {
            let r = diffuse_reflectance(f64::from(i) / 100.0, 1.4);
            assert!(r > last, "monotonic in the albedo");
            last = r;
        }
        // Fdr(1.4) by hand: -1.44/1.96 + 0.71/1.4 + 0.668 + 0.0636 * 1.4 = 0.52949.
        let fdr = diffuse_fresnel(1.4);
        assert!((fdr - 0.52949).abs() < 1e-4, "{fdr}");
    }

    #[test]
    fn skn10_melanin_spans_the_ita_categories() {
        // [SKN-10] The melanin control alone sweeps the skin from "very light" (ITA > 55°) to
        // "dark" (ITA < -30°): the categories of Chardon et al. 1991 / Del Bino et al. 2006, the
        // only published reference this test uses (no swatch values are invented).
        let model = PigmentModel::default();
        let mut p = PigmentParams { melanin: 0.0, ..PigmentParams::default() };
        let light = ita_degrees(rgb_to_lab(model.albedo(&p)));
        p.melanin = 1.0;
        let dark = ita_degrees(rgb_to_lab(model.albedo(&p)));
        assert_eq!(ItaCategory::of_angle(light), ItaCategory::VeryLight, "light end ITA {light}");
        assert_eq!(ItaCategory::of_angle(dark), ItaCategory::Dark, "dark end ITA {dark}");
        // Monotonic: more melanin, lower ITA and lower luminance.
        let mut last = f64::INFINITY;
        for i in 0..=20 {
            p.melanin = f64::from(i) / 20.0;
            let ita = ita_degrees(rgb_to_lab(model.albedo(&p)));
            assert!(ita < last + 1e-9, "ITA must fall with melanin");
            last = ita;
        }
    }

    #[test]
    fn skn10_pigments_move_colour_the_documented_way() {
        let model = PigmentModel::default();
        let base = PigmentParams::default();
        let a0 = model.albedo(&base);
        // More blood: redder relative to green.
        let mut p = base;
        p.haemoglobin = 0.9;
        let a = model.albedo(&p);
        assert!(a[0] / a[1] > a0[0] / a0[1], "haemoglobin reddens");
        // Less oxygen: bluish-purple relative to oxygenated (deoxy absorbs more in the red).
        p = base;
        p.haemoglobin = 0.9;
        p.oxygenation = 0.0;
        let deoxy = model.albedo(&p);
        assert!(deoxy[0] / deoxy[2] < a[0] / a[2], "deoxygenated blood is bluer");
        // Carotene: yellower (blue falls more than red).
        p = base;
        p.carotene = 1.0;
        let c = model.albedo(&p);
        assert!(c[2] / c[0] < a0[2] / a0[0], "carotene yellows");
        // Pheomelanin-rich melanin is warmer (less blue relative to red) than eumelanin.
        p = base;
        p.melanin = 0.6;
        p.eumelanin_ratio = 0.0;
        let pheo = rgb_to_lab(model.albedo(&p));
        p.eumelanin_ratio = 1.0;
        let eu = rgb_to_lab(model.albedo(&p));
        // Pheomelanin absorbs more steeply towards blue (Donner & Jensen 2006, eqs. 3-4), so a low
        // eumelanin share gives a yellower tone. (Its lightness at equal volume fraction is no
        // longer ordered: eumelanin is packaged, which caps its absorption.)
        assert!(pheo[2].atan2(pheo[1]) > eu[2].atan2(eu[1]), "and its Lab hue is yellower");
        // Albedo stays in 0..1.
        let mut rng = Rng::new(7);
        for _ in 0..200 {
            let q = PigmentParams::from_array([0; 6].map(|_| rng.f64()));
            let r = model.evaluate(&q);
            assert!(r.albedo.iter().all(|v| (0.0..=1.0).contains(v)), "{:?}", r.albedo);
            assert!(r.reflectance.iter().all(|v| (0.0..=1.0).contains(v)));
        }
    }

    #[test]
    fn skn10_sss_radius_is_longest_in_red() {
        // Skin scatters red light farthest (blood and melanin absorb short wavelengths): the
        // per-channel ordering every SSS profile fit shows.
        let model = PigmentModel::default();
        let r = model.evaluate(&PigmentParams::default()).sss_radius_mm;
        assert!(r[0] > r[1] && r[1] > r[2], "{r:?}");
        assert!(r[0] > 0.1 && r[0] < 10.0, "a plausible millimetre scale: {r:?}");
    }

    #[test]
    fn ac_wp13_1_fit_recovers_synthetic_targets() {
        // [AC-WP13-1] machinery: targets are the model's own output for random parameters, so the
        // fit must reproduce them closely (the measured swatches are the test above).
        let model = PigmentModel::default();
        let mut rng = Rng::new(31);
        for _ in 0..12 {
            let truth = PigmentParams {
                melanin: rng.range_f64(0.05, 0.95),
                eumelanin_ratio: rng.range_f64(0.1, 0.9),
                haemoglobin: rng.range_f64(0.1, 0.9),
                ..PigmentParams::default()
            };
            let target = rgb_to_lab(model.albedo(&truth));
            let fit = model.fit_lab(target, &PigmentParams::default());
            assert!(fit.delta_e < 0.5, "ΔE2000 {} for {truth:?}", fit.delta_e);
        }
    }

    /// The measured skin reference (`data/test-scenes/skin-reference.json`, ISSA, CC BY 4.0).
    const SKIN_REFERENCE: &str = include_str!("../../../../data/test-scenes/skin-reference.json");

    /// The documented `[AC-WP13-1]` tolerance (the spec names none): CIEDE2000 1.0, about one
    /// just-noticeable difference, between a bin's mean colour and the model's best fit.
    const AC_WP13_1_MEAN_TOLERANCE: f64 = 1.0;
    /// CIEDE2000 for the edge swatches (the mean of a bin's lowest or highest decile of L*, a* or
    /// b*): a range check. The worst today is 1.86 (2.14 before blood packaging): blood raises b*
    /// with a*, so pale, red, low-yellow skin is the model's corner.
    const AC_WP13_1_EDGE_TOLERANCE: f64 = 2.5;

    /// One measured swatch: a label, its CIELAB and whether it is a bin mean (or an edge subset).
    struct Swatch {
        label: String,
        lab: [f64; 3],
        mean: bool,
    }

    fn lab_of(node: &humanart_io::json::Node<'_, '_>) -> [f64; 3] {
        let mut lab = [0.0; 3];
        node.field("lab_mean").and_then(|n| n.read_into(&mut lab)).unwrap_or_else(|e| panic!("{e}"));
        lab
    }

    fn reference_swatches() -> Vec<Swatch> {
        let doc = humanart_io::json::parse_str(SKIN_REFERENCE).unwrap_or_else(|e| panic!("{e}"));
        let root = doc.node();
        let text = |key: &str| root.get::<&str>(key).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(text("format"), "humanart.skin-reference");
        assert_eq!(text("licence"), "CC BY 4.0");
        let source = root.field("source").unwrap_or_else(|e| panic!("{e}"));
        let sha = source.get::<&str>("file_sha256").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(sha.len(), 64, "the source workbook's SHA-256 is recorded");
        let mut out = Vec::new();
        let bins = root.field("ita_bins").unwrap_or_else(|e| panic!("{e}"));
        for bin in bins.elements().unwrap_or_else(|e| panic!("{e}")) {
            let id = bin.get::<&str>("id").unwrap_or_else(|e| panic!("{e}"));
            out.push(Swatch { label: format!("ITA {id}"), lab: lab_of(&bin), mean: true });
            let subsets = bin.field("subsets").unwrap_or_else(|e| panic!("{e}"));
            for (name, subset) in subsets.members().unwrap_or_else(|e| panic!("{e}")) {
                out.push(Swatch { label: format!("ITA {id} {name}"), lab: lab_of(&subset), mean: false });
            }
        }
        let bins = root.field("lightness_bins").unwrap_or_else(|e| panic!("{e}"));
        for bin in bins.elements().unwrap_or_else(|e| panic!("{e}")) {
            let lo = bin.get::<f64>("l_min").unwrap_or_else(|e| panic!("{e}"));
            out.push(Swatch { label: format!("L* {lo}-{}", lo + 5.0), lab: lab_of(&bin), mean: true });
        }
        let sites = root.field("body_sites").unwrap_or_else(|e| panic!("{e}"));
        for site in sites.elements().unwrap_or_else(|e| panic!("{e}")) {
            let id = site.get::<&str>("id").unwrap_or_else(|e| panic!("{e}"));
            out.push(Swatch { label: format!("site {id}"), lab: lab_of(&site), mean: true });
        }
        out
    }

    #[test]
    fn ac_wp13_1_reproduces_the_issa_reference_swatches() {
        // [AC-WP13-1] [SKN-10] Every measured reference swatch (ISSA, Lu et al. 2025: 15 256
        // specular-included spectrophotometer records, binned by ITA category, by L* and by body
        // site) is reachable by the pigment model: the best fit over melanin, eumelanin ratio and
        // haemoglobin (the other parameters at their registry defaults), with the SCI specular
        // component added, is within AC_WP13_1_MEAN_TOLERANCE (bin means) or
        // AC_WP13_1_EDGE_TOLERANCE (decile edges). Run with --nocapture for the per-bin table.
        let model = PigmentModel::default();
        let swatches = reference_swatches();
        assert!(swatches.len() >= 6 + 36 + 8 + 12, "{} swatches", swatches.len());
        let mut failures = Vec::new();
        eprintln!("AC-WP13-1 swatch | target L a b | fit L a b | melanin eu-ratio haemoglobin | dE00");
        for s in &swatches {
            let fit = model.fit_measured_lab(s.lab, &PigmentParams::default(), SCI_SURFACE_REFLECTANCE);
            let got = model.measured_lab(&fit.params, SCI_SURFACE_REFLECTANCE);
            eprintln!(
                "{:28} | {:5.1} {:5.1} {:5.1} | {:5.1} {:5.1} {:5.1} | {:.3} {:.2} {:.2} | {:.2}",
                s.label,
                s.lab[0],
                s.lab[1],
                s.lab[2],
                got[0],
                got[1],
                got[2],
                fit.params.melanin,
                fit.params.eumelanin_ratio,
                fit.params.haemoglobin,
                fit.delta_e
            );
            let tolerance = if s.mean { AC_WP13_1_MEAN_TOLERANCE } else { AC_WP13_1_EDGE_TOLERANCE };
            if fit.delta_e > tolerance {
                failures.push(format!("{} dE00 {:.2} > {tolerance}", s.label, fit.delta_e));
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }

    /// Largest RMS difference (reflectance factor, 400-700 nm) allowed between an ITA bin's mean
    /// measured spectrum and the model's spectrum at the bin's colour fit. With homogeneous blood
    /// the light bins left 0.027 to 0.031 (the model's Soret and Q bands deeper than the measured
    /// ones); with vessel packaging the worst bin is 0.0082 (set after seeing the results).
    const AC_WP13_1_SPECTRUM_RMS_TOLERANCE: f64 = 0.012;

    #[test]
    fn ac_wp13_1_fitted_spectra_follow_the_measured_spectra() {
        // [AC-WP13-1] Beyond the colour: at the parameters that fit each ITA bin's mean colour, the
        // model's reflectance spectrum (plus the SCI surface term) follows the bin's mean measured
        // spectrum, so the match is not a metamer. Run with --nocapture for the per-bin RMS.
        let model = PigmentModel::default();
        let doc = humanart_io::json::parse_str(SKIN_REFERENCE).unwrap_or_else(|e| panic!("{e}"));
        let root = doc.node();
        let mut wavelengths = Vec::new();
        root.field("wavelengths_nm").and_then(|n| n.extend_into(&mut wavelengths)).unwrap_or_else(|e| panic!("{e}"));
        let wavelengths: Vec<f64> = wavelengths;
        let bands = PigmentModel::wavelengths();
        let index: Vec<usize> = wavelengths
            .iter()
            .map(|nm| bands.iter().position(|b| (b - nm).abs() < 1e-9).unwrap_or_else(|| panic!("{nm} nm")))
            .collect();
        let bins = root.field("ita_bins").unwrap_or_else(|e| panic!("{e}"));
        let mut count = 0;
        for bin in bins.elements().unwrap_or_else(|e| panic!("{e}")) {
            let id = bin.get::<&str>("id").unwrap_or_else(|e| panic!("{e}"));
            let mut measured = Vec::new();
            bin.field("reflectance_mean").and_then(|n| n.extend_into(&mut measured)).unwrap_or_else(|e| panic!("{e}"));
            let measured: Vec<f64> = measured;
            assert_eq!(measured.len(), index.len());
            let fit = model.fit_measured_lab(lab_of(&bin), &PigmentParams::default(), SCI_SURFACE_REFLECTANCE);
            let r = model.evaluate(&fit.params).reflectance;
            let sum: f64 =
                index.iter().zip(&measured).map(|(&i, m)| (r[i] + SCI_SURFACE_REFLECTANCE - m).powi(2)).sum();
            let rms = (sum / measured.len() as f64).sqrt();
            eprintln!("AC-WP13-1 spectrum ITA {id:12} RMS {rms:.4}");
            assert!(rms < AC_WP13_1_SPECTRUM_RMS_TOLERANCE, "ITA {id}: RMS {rms}");
            count += 1;
        }
        assert_eq!(count, 6, "the six ITA categories");
    }

    #[test]
    fn measured_lab_adds_the_surface_reflection() {
        let model = PigmentModel::default();
        let p = PigmentParams::default();
        // Without a surface term it is the albedo's colour (in gamut here).
        let from_albedo = rgb_to_lab(model.albedo(&p));
        let measured = model.measured_lab(&p, 0.0);
        for (a, b) in from_albedo.iter().zip(measured) {
            assert!((a - b).abs() < 0.02, "{from_albedo:?} {measured:?}");
        }
        // A flat surface reflection lightens and desaturates.
        let sci = model.measured_lab(&p, SCI_SURFACE_REFLECTANCE);
        assert!(sci[0] > measured[0]);
        assert!(sci[1].hypot(sci[2]) < measured[1].hypot(measured[2]));
        assert!((SCI_SURFACE_REFLECTANCE - 0.4 * 0.4 / (2.4 * 2.4)).abs() < 1e-15);
    }

    #[test]
    fn params_set_by_target_and_clamp() {
        let mut p = PigmentParams::default();
        assert!(p.set("skin.melanin", 2.0));
        assert_eq!(p.melanin, 1.0);
        assert!(p.set("skin.oxygenation", f64::NAN));
        assert_eq!(p.oxygenation, 0.7);
        assert!(!p.set("skin.unknown", 0.5));
        let wild = PigmentParams::from_array([f64::NAN, -1.0, 3.0, 0.5, f64::INFINITY, 0.2]).clamped();
        assert_eq!(wild.to_array(), [0.3, 0.0, 1.0, 0.5, 0.1, 0.2]);
        let phys = PigmentParams::default().physical(&Mapping::DEFAULT);
        assert!((phys.melanin_fraction - Mapping::DEFAULT.melanin_max * 0.09).abs() < 1e-12);
        assert!((phys.blood_fraction - 0.001 * 100_f64.powf(0.3)).abs() < 1e-12);
        assert!((phys.epidermis_mm - 0.06).abs() < 1e-12);
    }

    #[test]
    fn gpu_table_layout() {
        let model = PigmentModel::default();
        let mut bytes = vec![0_u8; GPU_TABLE_BYTES];
        assert!(!model.write_gpu_table(&mut bytes[..10]));
        assert!(model.write_gpu_table(&mut bytes));
        // Band 34 (550 nm), word 9 = oxy blood, packaged in the vessels.
        let at = (34 * GPU_WORDS_PER_BAND + 9) * 4;
        let word = f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
        let packaged = packaged_absorption(blood_absorption(43016.0), 2.0 * Mapping::DEFAULT.blood_vessel_radius_mm);
        assert!((f64::from(word) - packaged).abs() < 1e-3);
        assert!(packaged < blood_absorption(43016.0));
    }

    #[test]
    fn blood_packaging_per_species_is_close_to_the_mix() {
        // The band table packages oxy and deoxy blood separately (linear in the oxygenation for the
        // GPU); the vessel factor of van Veen et al. 2002 applies to the mixed coefficient. The
        // factor is concave, so the per-species form absorbs a little less. Bound the difference
        // per band and, what matters, in colour: the model's reflectance with the exact mixed
        // packaging against the band table's, as CIEDE2000.
        let model = PigmentModel::default();
        let path = 2.0 * Mapping::DEFAULT.blood_vessel_radius_mm;
        let mut worst_band = 0.0_f64;
        let mut worst_colour = 0.0_f64;
        for g in [0.25, 0.5, 0.7, 0.9] {
            for haemoglobin in [0.3, 0.6, 1.0] {
                let params = PigmentParams { haemoglobin, oxygenation: g, ..PigmentParams::default() };
                let p = params.physical(&Mapping::DEFAULT);
                let table = model.evaluate(&params).reflectance;
                let (mut xyz_exact, mut xyz_table, mut white) = ([0.0; 3], [0.0; 3], [0.0; 3]);
                for ((band, b), r_table) in BANDS.iter().zip(&model.bands).zip(table) {
                    let (oxy, deoxy) = (blood_absorption(band[5]), blood_absorption(band[6]));
                    let exact = packaged_absorption(g * oxy + (1.0 - g) * deoxy, path);
                    let split = g * b.oxy_blood + (1.0 - g) * b.deoxy_blood;
                    assert!(split <= exact * (1.0 + 1e-6));
                    worst_band = worst_band.max((exact - split) / exact);
                    // evaluate_physical with the exact blood term.
                    let sigma_epi = p.melanin_fraction
                        * (p.eumelanin_blend * b.eumelanin + (1.0 - p.eumelanin_blend) * b.pheomelanin)
                        + (1.0 - p.melanin_fraction) * b.baseline
                        + b.carotene * p.carotene_molar;
                    let sigma_derm = p.blood_fraction * exact + (1.0 - p.blood_fraction) * b.baseline;
                    let scatter = 0.5 * b.scattering;
                    let r_exact = (-2.0 * p.epidermis_mm * sigma_epi).exp()
                        * diffuse_reflectance(scatter / (scatter + sigma_derm), SKIN_IOR);
                    for c in 0..3 {
                        xyz_exact[c] += b.xyz[c] * r_exact;
                        xyz_table[c] += b.xyz[c] * r_table;
                        white[c] += b.xyz[c];
                    }
                }
                let de = delta_e_2000(xyz_to_lab(xyz_exact, white), xyz_to_lab(xyz_table, white));
                worst_colour = worst_colour.max(de);
            }
        }
        eprintln!(
            "blood packaging per species: worst band coefficient -{:.1} %, worst colour dE00 {worst_colour:.3}",
            100.0 * worst_band
        );
        assert!(worst_band < 0.15, "{worst_band}");
        assert!(worst_colour < 0.5, "{worst_colour}");
        // Thin vessels are homogeneous blood; thick ones saturate at 1 / path.
        assert!((packaged_absorption(10.0, 1e-9) - 10.0).abs() < 1e-6);
        assert!((packaged_absorption(1e6, 0.026) - 1.0 / 0.026).abs() < 1e-6);
    }

    #[test]
    fn ac_wp13_1_blood_packaging_flattens_the_light_skin_spectra() {
        // The gain that motivates the vessel packaging: the light ITA bins' spectral RMS with
        // homogeneous blood (radius 0) against the default. Run with --nocapture for the numbers.
        let doc = humanart_io::json::parse_str(SKIN_REFERENCE).unwrap_or_else(|e| panic!("{e}"));
        let root = doc.node();
        let mut wavelengths: Vec<f64> = Vec::new();
        root.field("wavelengths_nm").and_then(|n| n.extend_into(&mut wavelengths)).unwrap_or_else(|e| panic!("{e}"));
        let bands = PigmentModel::wavelengths();
        let index: Vec<usize> = wavelengths
            .iter()
            .map(|nm| bands.iter().position(|b| (b - nm).abs() < 1e-9).unwrap_or_else(|| panic!("{nm} nm")))
            .collect();
        let homogeneous = PigmentModel::new(Mapping { blood_vessel_radius_mm: 0.0, ..Mapping::DEFAULT });
        let packaged = PigmentModel::default();
        let bins = root.field("ita_bins").unwrap_or_else(|e| panic!("{e}"));
        for bin in bins.elements().unwrap_or_else(|e| panic!("{e}")).take(2) {
            let id = bin.get::<&str>("id").unwrap_or_else(|e| panic!("{e}"));
            let mut measured: Vec<f64> = Vec::new();
            bin.field("reflectance_mean").and_then(|n| n.extend_into(&mut measured)).unwrap_or_else(|e| panic!("{e}"));
            let rms = |model: &PigmentModel| {
                let fit = model.fit_measured_lab(lab_of(&bin), &PigmentParams::default(), SCI_SURFACE_REFLECTANCE);
                let r = model.evaluate(&fit.params).reflectance;
                let sum: f64 =
                    index.iter().zip(&measured).map(|(&i, m)| (r[i] + SCI_SURFACE_REFLECTANCE - m).powi(2)).sum();
                (sum / measured.len() as f64).sqrt()
            };
            let (before, after) = (rms(&homogeneous), rms(&packaged));
            eprintln!("AC-WP13-1 ITA {id}: spectral RMS homogeneous blood {before:.4}, packaged {after:.4}");
            assert!(after < 0.5 * before, "ITA {id}: {before} -> {after}");
        }
    }
}
