//! Measured regional skin surface (todo 1.4): how the specular lobe and the sebum film vary over
//! the face, from published measurements, written into the base values and the zone table the
//! compositor resolves.
//!
//! **Specular lobe per face region.** Weyrich et al. measured the faces of 149 subjects and fitted
//! a Torrance-Sparrow BRDF per point: Beckmann roughness `m` and specular scale `ρs` (Fresnel fixed
//! to a literature value). The per-region means across all subjects are only published as a plot:
//! T. Weyrich, *Acquisition of Human Faces Using a Measurement-Based Skin Reflectance Model*, PhD
//! thesis, ETH Zürich 2006, Figure 10.5 (the same data as Weyrich et al., "Analysis of Human Faces
//! using a Measurement-Based Skin Reflectance Model", ACM TOG 25(3) 2006, Figure 15). The values
//! below are the ellipse centres digitised from the figure's vector paths (reading error below
//! 0.001; `target/agents/skin-surface/scripts/fig105.py`). Their order matches the paper's text:
//! the nose has the smallest `m` and the largest `ρs`, then eyebrows, eyelids, lips and forehead.
//! The hair-covered regions (mustache, eyebrows, beard) are left out, as the authors warn their
//! fits are noisy.
//!
//! GGX with `α = m` approximates the Beckmann lobe of the same `m` (Walter et al. 2007, "Microfacet
//! Models for Refraction through Rough Surfaces", section 5.2), and the renderer's perceptual
//! roughness is `√α`, so the primary lobe's roughness is `√m`. `ρs` is used relative to the mean of
//! the six skin regions: the specular scale attribute (1 = the skin `f0` 0.028 of the material)
//! becomes `ρs / mean`, so the regional pattern is measured while the absolute level stays the
//! registry's.
//!
//! **Sebum film.** Algiert-Zielińska B., Batory M., Skubalski J., Rotsztejn H., "Evaluation of the
//! relation between lipid coat, transepidermal water loss, and skin pH", Int J Dermatol
//! 56(11):1192-1197, 2017 (doi:10.1111/ijd.13726; 50 healthy women 19-35 years, Sebumeter SM 815):
//! mean lipid coat 128 µg/cm² on the T-zone (forehead, nose, chin), 65.2 µg/cm² on the U-zone
//! (cheeks) and 5.6 µg/cm² on the thigh. The document's `skin.oiliness` is read as the T-zone's
//! value; the cheeks get 65.2/128 of it and the body 5.6/128. The vermilion of the lips is glabrous
//! (no follicles, Weyrich et al. 2006, section 7), so it gets no sebum beyond the body's.
//!
//! Numbers are facts quoted from the publications (cited above); no data file is copied.

use super::channels::{Attribute, BaseValues};
use super::zones::{Zone, ZoneTable};

/// One face region's Torrance-Sparrow fit: Beckmann roughness `m` and specular scale `ρs`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpecularFit {
    /// Beckmann roughness (RMS slope).
    pub m: f32,
    /// Specular scale of the fit.
    pub rho_s: f32,
}

impl SpecularFit {
    /// The renderer's perceptual roughness of the primary lobe (`√m`, see the module docs).
    pub fn roughness(self) -> f32 {
        self.m.max(0.0).sqrt()
    }
}

/// Forehead (Weyrich 2006, region 2).
pub const FOREHEAD: SpecularFit = SpecularFit { m: 0.2695, rho_s: 0.2721 };
/// Eyelids (region 4).
pub const EYELIDS: SpecularFit = SpecularFit { m: 0.2630, rho_s: 0.2920 };
/// Nose (region 6).
pub const NOSE: SpecularFit = SpecularFit { m: 0.2338, rho_s: 0.3043 };
/// Cheek (region 7).
pub const CHEEK: SpecularFit = SpecularFit { m: 0.2714, rho_s: 0.2342 };
/// Lips (region 8).
pub const LIPS: SpecularFit = SpecularFit { m: 0.2848, rho_s: 0.2884 };
/// Chin (region 9).
pub const CHIN: SpecularFit = SpecularFit { m: 0.2885, rho_s: 0.2281 };

/// The six skin regions the means are taken over.
pub const SKIN_REGIONS: [SpecularFit; 6] = [FOREHEAD, EYELIDS, NOSE, CHEEK, LIPS, CHIN];

/// Mean lipid coat of the T-zone, µg/cm² (Algiert-Zielińska et al. 2017).
pub const SEBUM_T_ZONE: f32 = 128.0;
/// Mean lipid coat of the U-zone (cheeks), µg/cm².
pub const SEBUM_U_ZONE: f32 = 65.2;
/// Mean lipid coat of the thigh (body skin), µg/cm².
pub const SEBUM_BODY: f32 = 5.6;

/// The mean `(m, ρs)` of [`SKIN_REGIONS`].
pub fn face_mean() -> SpecularFit {
    let n = SKIN_REGIONS.len() as f32;
    let (m, r) = SKIN_REGIONS.iter().fold((0.0, 0.0), |(m, r), f| (m + f.m, r + f.rho_s));
    SpecularFit { m: m / n, rho_s: r / n }
}

/// Writes the measured regional surface (see the module documentation) into `base` and `zones`,
/// for the document's T-zone `oiliness` (0..1): the primary lobe's roughness and the specular
/// scale per face zone, and the sebum film per zone. Other attributes and zone offsets (pigment,
/// flush) are left as they are.
pub fn apply_measured_surface(oiliness: f32, base: &mut BaseValues, zones: &mut ZoneTable) {
    let mean = face_mean();
    let rough = mean.roughness();
    base.set(Attribute::RoughnessPrimary, rough);
    let oil = if oiliness.is_finite() { oiliness.clamp(0.0, 1.0) } else { 0.0 };
    let body_oil = oil * SEBUM_BODY / SEBUM_T_ZONE;
    base.set(Attribute::Oiliness, body_oil);
    let t_zone = oil - body_oil;
    let u_zone = oil * (SEBUM_U_ZONE - SEBUM_BODY) / SEBUM_T_ZONE;
    let rows: [(Zone, SpecularFit, f32); 7] = [
        (Zone::Forehead, FOREHEAD, t_zone),
        (Zone::Nose, NOSE, t_zone),
        (Zone::Chin, CHIN, t_zone),
        (Zone::Cheeks, CHEEK, u_zone),
        (Zone::Lips, LIPS, 0.0),
        (Zone::Eyelids, EYELIDS, 0.0),
        (Zone::UnderEye, EYELIDS, 0.0),
    ];
    for (zone, fit, oil_offset) in rows {
        let Some(z) = zones.zones.get_mut(zone.slot()) else { continue };
        z.offsets[Attribute::RoughnessPrimary.index()] = fit.roughness() - rough;
        z.offsets[Attribute::Specular.index()] = fit.rho_s / mean.rho_s - 1.0;
        z.offsets[Attribute::Oiliness.index()] = oil_offset;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn todo_1_4_measured_order_matches_the_publication() {
        // Weyrich et al. 2006: the nose has the smallest m and the largest rho_s; rho_s falls
        // nose > eyelids > lips > forehead > cheek > chin.
        assert!(SKIN_REGIONS.iter().all(|f| f.m >= NOSE.m));
        let mut by_rho = SKIN_REGIONS;
        by_rho.sort_by(|a, b| b.rho_s.total_cmp(&a.rho_s));
        assert_eq!(by_rho, [NOSE, EYELIDS, LIPS, FOREHEAD, CHEEK, CHIN]);
        let mean = face_mean();
        assert!((mean.m - 0.26850).abs() < 1e-4 && (mean.rho_s - 0.26985).abs() < 1e-4, "{mean:?}");
    }

    #[test]
    fn todo_1_4_measured_surface_goes_into_the_zones() {
        let mut base = BaseValues::default();
        let mut zones = ZoneTable::default();
        apply_measured_surface(0.6, &mut base, &mut zones);
        let r = base.get(Attribute::RoughnessPrimary);
        assert!((r - 0.26850_f32.sqrt()).abs() < 1e-4);
        let oil =
            |zone: Zone| base.get(Attribute::Oiliness) + zones.zones[zone.slot()].offsets[Attribute::Oiliness.index()];
        assert!((oil(Zone::Forehead) - 0.6).abs() < 1e-6, "T-zone takes the document's value");
        assert!((oil(Zone::Cheeks) - 0.6 * 65.2 / 128.0).abs() < 1e-6);
        assert!((oil(Zone::Lips) - 0.6 * 5.6 / 128.0).abs() < 1e-6, "lips: body level");
        let nose = zones.zones[Zone::Nose.slot()].offsets;
        assert!(nose[Attribute::RoughnessPrimary.index()] < 0.0, "the nose is the smoothest");
        assert!(nose[Attribute::Specular.index()] > 0.1);
        let chin = zones.zones[Zone::Chin.slot()].offsets;
        assert!(chin[Attribute::Specular.index()] < -0.1);
        // Pigment offsets stay.
        assert_eq!(zones.zones[Zone::Lips.slot()].offsets[Attribute::Haemoglobin.index()], 0.3);
        // A non-finite oiliness is no oil.
        apply_measured_surface(f32::NAN, &mut base, &mut zones);
        assert_eq!(base.get(Attribute::Oiliness), 0.0);
    }
}
