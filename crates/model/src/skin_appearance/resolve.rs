//! The resolve step `[SKN-01]` `[SKN-10]` `[SKN-12]` `[SKN-31]` `[SKN-32]`: composited attributes
//! of one texel to the resolved channels the render passes sample. The CPU reference of the GPU
//! compositor's second pass (`shaders/skin/resolve.wgsl`).

use super::channels::{ATTRIBUTE_COUNT, Attribute};
use super::layers::TexelInputs;
use super::pigment::{PigmentModel, PigmentParams};
use super::zones::ZoneTable;

/// The resolved channels of one texel (see [`super::channels::ResolvedChannel`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedTexel {
    /// RGB albedo (linear), A cavity.
    pub albedo: [f32; 4],
    /// RGB SSS radius in mm, A thickness.
    pub scatter: [f32; 4],
    /// Specular scale / 4, primary roughness, secondary roughness, lobe mix.
    pub specular: [f32; 4],
    /// Detail normal x, y, height (mm), wetness.
    pub detail: [f32; 4],
    /// RGB albedo change at full flush, A oiliness.
    pub flush: [f32; 4],
}

/// The pigment parameters held by composited attributes.
pub fn pigment_of(v: &[f32; ATTRIBUTE_COUNT]) -> PigmentParams {
    PigmentParams {
        melanin: f64::from(v[Attribute::Melanin.index()]),
        eumelanin_ratio: f64::from(v[Attribute::EumelaninRatio.index()]),
        haemoglobin: f64::from(v[Attribute::Haemoglobin.index()]),
        oxygenation: f64::from(v[Attribute::Oxygenation.index()]),
        carotene: f64::from(v[Attribute::Carotene.index()]),
        epidermal_thickness: f64::from(v[Attribute::EpidermalThickness.index()]),
    }
}

/// Resolves one texel.
pub fn resolve_texel(
    v: &[f32; ATTRIBUTE_COUNT],
    inputs: &TexelInputs,
    model: &PigmentModel,
    zones: &ZoneTable,
) -> ResolvedTexel {
    let a = |x: Attribute| v[x.index()];
    let params = pigment_of(v);
    let mut physical = params.physical(model.mapping());
    let base = model.evaluate_physical(&physical);
    let flush_weight = zones.flush_weight(&inputs.zones);
    physical.blood_fraction = (physical.blood_fraction * (1.0 + f64::from(zones.flush_gain * flush_weight))).min(1.0);
    let flushed = model.evaluate_physical(&physical).albedo;

    let coverage = a(Attribute::OverrideCoverage).clamp(0.0, 1.0);
    let over = [a(Attribute::OverrideRed), a(Attribute::OverrideGreen), a(Attribute::OverrideBlue)];
    let mut albedo = [0.0; 4];
    let mut flush = [0.0; 4];
    for c in 0..3 {
        let pigment = base.albedo[c] as f32;
        albedo[c] = pigment + (over[c] - pigment) * coverage;
        flush[c] = (flushed[c] - base.albedo[c]) as f32 * (1.0 - coverage);
    }
    albedo[3] = (a(Attribute::Cavity) * inputs.cavity).clamp(0.0, 1.0);
    flush[3] = a(Attribute::Oiliness);

    let wetness = a(Attribute::Wetness).clamp(0.0, 1.0);
    let (rough, spec) = zones.wet_surface(a(Attribute::RoughnessPrimary), a(Attribute::Specular), wetness);
    let specular = [
        (spec / 4.0).clamp(0.0, 1.0),
        rough.clamp(0.0, 1.0),
        a(Attribute::RoughnessSecondary).clamp(0.0, 1.0),
        a(Attribute::LobeMix).clamp(0.0, 1.0),
    ];

    let (mut nx, mut ny) = (a(Attribute::DetailNormalX), a(Attribute::DetailNormalY));
    let len = (nx * nx + ny * ny).sqrt();
    if len > 1.0 {
        nx /= len;
        ny /= len;
    }
    let r = base.sss_radius_mm;
    ResolvedTexel {
        albedo,
        scatter: [r[0] as f32, r[1] as f32, r[2] as f32, inputs.thickness],
        specular,
        detail: [nx, ny, a(Attribute::Height), wetness],
        flush,
    }
}

#[cfg(test)]
mod tests {
    use super::super::channels::BaseValues;
    use super::super::zones::{ZONE_SLOTS, Zone};
    use super::*;

    #[test]
    fn skn12_override_replaces_pigment_albedo() {
        let model = PigmentModel::default();
        let zones = ZoneTable::default();
        let mut v = BaseValues::default().0;
        let r0 = resolve_texel(&v, &TexelInputs::default(), &model, &zones);
        let expect = model.albedo(&PigmentParams::default());
        for (got, want) in r0.albedo.iter().zip(expect) {
            assert!((f64::from(*got) - want).abs() < 1e-6);
        }
        v[Attribute::OverrideRed.index()] = 0.9;
        v[Attribute::OverrideCoverage.index()] = 1.0;
        let r = resolve_texel(&v, &TexelInputs::default(), &model, &zones);
        assert_eq!(r.albedo[0], 0.9);
        assert_eq!(r.flush[0], 0.0, "an override does not flush");
    }

    #[test]
    fn skn31_flush_reddens_more_on_cheeks() {
        let model = PigmentModel::default();
        let zones = ZoneTable::default();
        let v = BaseValues::default().0;
        let body = resolve_texel(&v, &TexelInputs::default(), &model, &zones);
        let mut masks = [0.0; ZONE_SLOTS];
        masks[Zone::Cheeks.slot()] = 1.0;
        let cheek_in = TexelInputs { zones: masks, ..TexelInputs::default() };
        let cheek = resolve_texel(&v, &cheek_in, &model, &zones);
        // More blood darkens green most: the flush delta is negative in green and strongest there.
        assert!(cheek.flush[1] < body.flush[1] && body.flush[1] < 0.0, "{:?} {:?}", cheek.flush, body.flush);
        assert!(cheek.flush[0] - cheek.flush[1] > 0.0, "relatively redder");
    }

    #[test]
    fn skn32_wetness_smooths_and_brightens_the_surface() {
        let model = PigmentModel::default();
        let zones = ZoneTable::default();
        let mut v = BaseValues::default().0;
        let dry = resolve_texel(&v, &TexelInputs::default(), &model, &zones);
        v[Attribute::Wetness.index()] = 1.0;
        let wet = resolve_texel(&v, &TexelInputs::default(), &model, &zones);
        assert!(wet.specular[1] < dry.specular[1]);
        assert!(wet.specular[0] > dry.specular[0]);
        assert_eq!(wet.detail[3], 1.0);
        // The detail normal is kept inside the unit disc.
        v[Attribute::DetailNormalX.index()] = 1.0;
        v[Attribute::DetailNormalY.index()] = 1.0;
        let r = resolve_texel(&v, &TexelInputs::default(), &model, &zones);
        assert!((r.detail[0].hypot(r.detail[1]) - 1.0).abs() < 1e-6);
    }
}
