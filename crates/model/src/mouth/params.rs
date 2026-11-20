//! Mouth interior parameters: teeth `[MTH-01]`, gums `[MTH-02]` and tongue `[MTH-03]`.
//!
//! Colours are linear Rec.709, lengths millimetres. Sources of the physical values:
//! - Enamel refractive index 1.63 (Meng et al., *J. Biomed. Opt.* 14(3) 034010, 2009, OCT
//!   measurement: enamel 1.631), so dry enamel reflects f0 ≈ 0.057; a saliva film (index ≈ 1.335,
//!   close to water) lowers the air interface to f0 ≈ 0.021 ([`TeethParams::f0`] blends by
//!   wetness).
//! - Tooth colour: the cervical third is more chromatic (yellower, the dentin shows through thin
//!   enamel), the incisal edge greyer and translucent (enamel only); the defaults sit in the
//!   lightness range of natural teeth (CIELAB L* ≈ 70–80; Joiner, *J. Dent.* 32 Suppl. 1, 2004).
//! - Tooth crown sizes are the means of the permanent dentition (Wheeler's dental anatomy), see
//!   `generate::parts`; the proportion controls scale them per tooth group.

use crate::error::{ModelError, Result};

/// Refractive index of enamel.
pub const ENAMEL_IOR: f32 = 1.63;
/// Refractive index of saliva (taken as water's).
pub const SALIVA_IOR: f32 = 1.335;

/// Teeth `[MTH-01]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TeethParams {
    /// Width scale of the incisors.
    pub incisor_width: f32,
    /// Crown height scale of the incisors.
    pub incisor_height: f32,
    /// Width scale of the canines.
    pub canine_width: f32,
    /// Crown height scale of the canines.
    pub canine_height: f32,
    /// Width scale of the premolars and molars.
    pub molar_width: f32,
    /// Crown height scale of the premolars and molars.
    pub molar_height: f32,
    /// Colour at the gingival (cervical) end.
    pub gingival_color: [f32; 3],
    /// Colour at the incisal edge.
    pub incisal_color: [f32; 3],
    /// Enamel translucency (light through the thin incisal edge), 0..1.
    pub translucency: f32,
    /// Incisal wear (flattened, darker edges), 0..1.
    pub wear: f32,
    /// Staining (yellow-brown at the gum line and between teeth), 0..1.
    pub stain: f32,
    /// Wetness of the saliva film, 0..1.
    pub wetness: f32,
}

impl Default for TeethParams {
    fn default() -> Self {
        Self {
            incisor_width: 1.0,
            incisor_height: 1.0,
            canine_width: 1.0,
            canine_height: 1.0,
            molar_width: 1.0,
            molar_height: 1.0,
            gingival_color: [0.64, 0.57, 0.45],
            incisal_color: [0.62, 0.62, 0.60],
            translucency: 0.5,
            wear: 0.2,
            stain: 0.15,
            wetness: 0.7,
        }
    }
}

impl TeethParams {
    /// Specular reflectance at normal incidence: dry enamel blended toward the saliva film.
    #[must_use]
    pub fn f0(&self) -> f32 {
        let dry = crate::eye::params::f0_from_ior(ENAMEL_IOR);
        let wet = crate::eye::params::f0_from_ior(SALIVA_IOR);
        dry + (wet - dry) * self.wetness.clamp(0.0, 1.0)
    }
}

/// Gums `[MTH-02]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GumParams {
    /// Base colour (coral pink).
    pub color: [f32; 3],
    /// Subsurface softness of the lighting (wrap), 0..1.
    pub scattering: f32,
    /// Wetness, 0..1.
    pub wetness: f32,
    /// Gum line height relative to the default, millimetres (positive covers more of the crown).
    pub line_offset_mm: f32,
}

impl Default for GumParams {
    fn default() -> Self {
        Self { color: [0.55, 0.20, 0.20], scattering: 0.6, wetness: 0.8, line_offset_mm: 0.0 }
    }
}

/// Tongue `[MTH-03]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TongueParams {
    /// Length scale.
    pub length: f32,
    /// Width scale.
    pub width: f32,
    /// Base colour.
    pub color: [f32; 3],
    /// Whitish coating of the dorsum, 0..1.
    pub coating: f32,
    /// Strength of the papillae micro-normal, 0..1.
    pub papillae: f32,
    /// Wetness, 0..1.
    pub wetness: f32,
}

impl Default for TongueParams {
    fn default() -> Self {
        Self { length: 1.0, width: 1.0, color: [0.50, 0.17, 0.18], coating: 0.25, papillae: 0.6, wetness: 0.8 }
    }
}

/// All mouth interior parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MouthParams {
    /// Teeth.
    pub teeth: TeethParams,
    /// Gums.
    pub gums: GumParams,
    /// Tongue.
    pub tongue: TongueParams,
}

fn check(v: f32, lo: f32, hi: f32, index: usize, what: &'static str) -> Result<()> {
    if v.is_finite() && (lo..=hi).contains(&v) {
        Ok(())
    } else {
        Err(ModelError::invalid("mouth parameter", index, what))
    }
}

fn check_color(c: [f32; 3], index: usize) -> Result<()> {
    c.iter().try_for_each(|&v| check(v, 0.0, 1.0, index, "colour channel outside 0..1"))
}

impl MouthParams {
    /// Checks the ranges: proportions 0.7..1.3, unit values 0..1, gum line ±2 mm, tongue scales
    /// 0.8..1.2.
    ///
    /// # Errors
    /// [`ModelError::Invalid`] naming the group (0 teeth, 1 gums, 2 tongue).
    pub fn validate(&self) -> Result<()> {
        let t = &self.teeth;
        for v in [t.incisor_width, t.incisor_height, t.canine_width, t.canine_height, t.molar_width, t.molar_height] {
            check(v, 0.7, 1.3, 0, "tooth proportion outside 0.7..1.3")?;
        }
        check_color(t.gingival_color, 0)?;
        check_color(t.incisal_color, 0)?;
        for v in [t.translucency, t.wear, t.stain, t.wetness] {
            check(v, 0.0, 1.0, 0, "teeth value outside 0..1")?;
        }
        let g = &self.gums;
        check_color(g.color, 1)?;
        check(g.scattering, 0.0, 1.0, 1, "gum scattering outside 0..1")?;
        check(g.wetness, 0.0, 1.0, 1, "gum wetness outside 0..1")?;
        check(g.line_offset_mm, -2.0, 2.0, 1, "gum line offset outside -2..2 mm")?;
        let u = &self.tongue;
        check(u.length, 0.8, 1.2, 2, "tongue length outside 0.8..1.2")?;
        check(u.width, 0.8, 1.2, 2, "tongue width outside 0.8..1.2")?;
        check_color(u.color, 2)?;
        for v in [u.coating, u.papillae, u.wetness] {
            check(v, 0.0, 1.0, 2, "tongue value outside 0..1")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid_and_enamel_reflectance_is_physical() {
        let p = MouthParams::default();
        p.validate().unwrap();
        let dry = TeethParams { wetness: 0.0, ..TeethParams::default() };
        let wet = TeethParams { wetness: 1.0, ..TeethParams::default() };
        assert!((dry.f0() - 0.0573).abs() < 5e-4, "{}", dry.f0());
        assert!((wet.f0() - 0.0206).abs() < 5e-4, "{}", wet.f0());
    }

    #[test]
    fn bad_values_are_refused() {
        let cases: [fn(&mut MouthParams); 4] = [
            |p| p.teeth.canine_width = 2.0,
            |p| p.gums.line_offset_mm = f32::INFINITY,
            |p| p.tongue.coating = -0.5,
            |p| p.teeth.incisal_color = [1.5, 0.0, 0.0],
        ];
        for (k, f) in cases.iter().enumerate() {
            let mut p = MouthParams::default();
            f(&mut p);
            assert!(p.validate().is_err(), "case {k}");
        }
    }
}
