//! Eye parameters: eyeball variation `[EYE-07]`, the iris layers `[EYE-02]` and the sclera
//! `[EYE-03]`.
//!
//! Lengths are millimetres (the `_mm` suffix), angles radians, colours linear Rec.709.
//!
//! Defaults and ranges, with their sources:
//! - Eyeball radius 12.0 mm (axial length about 24 mm; Atchison & Smith, *Optics of the Human Eye*,
//!   2000, ch. 1); range 10.5–13.5 mm.
//! - Limbus (visible iris) radius 5.85 mm: horizontal visible iris diameter 11.71 ± 0.42 mm
//!   (Rüfer, Schröder & Erb, *Cornea* 24(3), 2005); range 5.0–6.6 mm.
//! - Anterior corneal radius 7.8 mm, posterior 6.5 mm, central thickness 0.55 mm (Le Grand full
//!   theoretical eye, in Atchison & Smith 2000, table A3.1); the anterior radius is the "corneal
//!   bulge" control, range 7.0–8.6 mm (smaller is more bulge).
//! - Refractive indices: cornea 1.376, aqueous humour 1.336 (Gullstrand's exact schematic eye, as
//!   the spec `[EYE-02]` gives them).
//! - Anterior chamber depth (corneal vertex to the iris plane) 3.6 mm: the vertex-to-lens distance
//!   of the Gullstrand and Le Grand eyes (0.5 + 3.1 and 0.55 + 3.05 mm), where the pupil margin
//!   rests on the lens; range 2.0–4.5 mm.
//! - Pupil diameter limits 2–8 mm (the light-adapted range of Watson & Yellott, *J. Vision*
//!   12(10):12, 2012), as radii 1.0–4.0 mm.
//! - Gaze limits ±0.8 rad horizontally and ±0.6 rad vertically, the eye joints' limits in the
//!   template rig (`generate::rig`).
//! - Tear film reflectance: index 1.337 (tear fluid; Craig et al., *Optom. Vis. Sci.* 72(3), 1995),
//!   f0 = ((1.337 − 1) / 2.337)² ≈ 0.021.

use crate::error::{ModelError, Result};

/// Refractive index of the cornea (Gullstrand).
pub const CORNEA_IOR: f32 = 1.376;
/// Refractive index of the aqueous humour (Gullstrand).
pub const AQUEOUS_IOR: f32 = 1.336;
/// Refractive index of the tear film.
pub const TEAR_IOR: f32 = 1.337;

/// Fresnel reflectance at normal incidence from air into a medium of index `n`.
#[must_use]
pub fn f0_from_ior(n: f32) -> f32 {
    let r = (n - 1.0) / (n + 1.0);
    r * r
}

/// The procedural iris layers `[EYE-02]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IrisParams {
    /// Colour of the ciliary (outer) zone.
    pub outer_color: [f32; 3],
    /// Colour of the pupillary (inner) zone; differing from the outer colour gives central
    /// heterochromia.
    pub inner_color: [f32; 3],
    /// Radius of the collarette as a fraction of the way from the pupil margin to the iris root.
    pub collarette: f32,
    /// Contrast of the radial fibres, 0..1.
    pub fibres: f32,
    /// Amount of crypts (dark pits in the ciliary zone), 0..1.
    pub crypts: f32,
    /// Width of the limbal ring as a fraction of the iris radius, 0..0.3.
    pub limbal_width: f32,
    /// Darkness of the limbal ring, 0..1.
    pub limbal_darkness: f32,
    /// Seed of the procedural pattern (each eye differs).
    pub seed: u32,
}

impl Default for IrisParams {
    /// A medium hazel-brown iris.
    fn default() -> Self {
        Self {
            outer_color: [0.19, 0.115, 0.055],
            inner_color: [0.34, 0.20, 0.075],
            collarette: 0.32,
            fibres: 0.75,
            crypts: 0.4,
            limbal_width: 0.12,
            limbal_darkness: 0.7,
            seed: 1,
        }
    }
}

/// The sclera `[EYE-03]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScleraParams {
    /// Base colour.
    pub color: [f32; 3],
    /// Vein network density, 0..1.
    pub vein_density: f32,
    /// Vein redness (bloodshot), 0..1.
    pub redness: f32,
    /// Age yellowing / discoloration, 0..1.
    pub yellowing: f32,
    /// Darkening toward the limbus, 0..1.
    pub limbal_darkening: f32,
    /// Perceptual roughness of the wet surface, 0..1.
    pub roughness: f32,
    /// Strength of the fine specular modulation, 0..1.
    pub specular_variation: f32,
}

impl Default for ScleraParams {
    fn default() -> Self {
        Self {
            color: [0.78, 0.74, 0.70],
            vein_density: 0.35,
            redness: 0.2,
            yellowing: 0.1,
            limbal_darkening: 0.35,
            roughness: 0.12,
            specular_variation: 0.4,
        }
    }
}

/// Eyeball variation and appearance of one eye `[EYE-07]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeParams {
    /// Eyeball (scleral sphere) radius.
    pub eyeball_radius_mm: f32,
    /// Limbus radius: the radius of the visible iris and of the cornea's base.
    pub limbus_radius_mm: f32,
    /// Anterior corneal radius of curvature (smaller bulges more).
    pub cornea_radius_mm: f32,
    /// Posterior corneal radius of curvature.
    pub cornea_back_radius_mm: f32,
    /// Central corneal thickness.
    pub cornea_thickness_mm: f32,
    /// Distance from the corneal vertex to the iris plane.
    pub iris_depth_mm: f32,
    /// Smallest pupil radius (most constricted).
    pub pupil_min_mm: f32,
    /// Largest pupil radius (most dilated).
    pub pupil_max_mm: f32,
    /// Pupil dilation between the limits, 0..1.
    pub dilation: f32,
    /// Horizontal gaze limit (each way).
    pub gaze_limit_yaw: f32,
    /// Vertical gaze limit (each way).
    pub gaze_limit_pitch: f32,
    /// Iris layers.
    pub iris: IrisParams,
    /// Sclera.
    pub sclera: ScleraParams,
}

impl Default for EyeParams {
    fn default() -> Self {
        Self {
            eyeball_radius_mm: 12.0,
            limbus_radius_mm: 5.85,
            cornea_radius_mm: 7.8,
            cornea_back_radius_mm: 6.5,
            cornea_thickness_mm: 0.55,
            iris_depth_mm: 3.6,
            pupil_min_mm: 1.0,
            pupil_max_mm: 4.0,
            dilation: 0.35,
            gaze_limit_yaw: 0.8,
            gaze_limit_pitch: 0.6,
            iris: IrisParams::default(),
            sclera: ScleraParams::default(),
        }
    }
}

/// Checks that `v` is finite and inside `lo..=hi`; `index` names the field in the error.
fn check(v: f32, lo: f32, hi: f32, index: usize, what: &'static str) -> Result<()> {
    if v.is_finite() && (lo..=hi).contains(&v) {
        Ok(())
    } else {
        Err(ModelError::invalid("eye parameter", index, what))
    }
}

fn check_color(c: [f32; 3], index: usize) -> Result<()> {
    for v in c {
        check(v, 0.0, 1.0, index, "colour channel outside 0..1")?;
    }
    Ok(())
}

impl EyeParams {
    /// The pupil radius at the current dilation.
    #[must_use]
    pub fn pupil_radius_mm(&self) -> f32 {
        self.pupil_min_mm + (self.pupil_max_mm - self.pupil_min_mm) * self.dilation.clamp(0.0, 1.0)
    }

    /// Checks every field against its documented range and the geometric consistency (the cornea
    /// must close over the limbus, the iris plane must lie inside the eyeball and behind the
    /// cornea, the pupil inside the iris).
    ///
    /// # Errors
    /// [`ModelError::Invalid`] naming the first bad field (its index in declaration order).
    pub fn validate(&self) -> Result<()> {
        check(self.eyeball_radius_mm, 10.5, 13.5, 0, "eyeball radius outside 10.5..13.5 mm")?;
        check(self.limbus_radius_mm, 5.0, 6.6, 1, "limbus radius outside 5.0..6.6 mm")?;
        check(self.cornea_radius_mm, 7.0, 8.6, 2, "cornea radius outside 7.0..8.6 mm")?;
        check(self.cornea_back_radius_mm, 5.5, 8.0, 3, "posterior cornea radius outside 5.5..8.0 mm")?;
        check(self.cornea_thickness_mm, 0.4, 0.8, 4, "cornea thickness outside 0.4..0.8 mm")?;
        check(self.iris_depth_mm, 2.0, 4.5, 5, "iris depth outside 2.0..4.5 mm")?;
        check(self.pupil_min_mm, 0.5, 3.0, 6, "smallest pupil radius outside 0.5..3.0 mm")?;
        check(self.pupil_max_mm, self.pupil_min_mm, 4.5, 7, "largest pupil radius outside min..4.5 mm")?;
        check(self.dilation, 0.0, 1.0, 8, "dilation outside 0..1")?;
        check(self.gaze_limit_yaw, 0.0, 1.2, 9, "horizontal gaze limit outside 0..1.2 rad")?;
        check(self.gaze_limit_pitch, 0.0, 1.0, 10, "vertical gaze limit outside 0..1.0 rad")?;
        let i = &self.iris;
        check_color(i.outer_color, 11)?;
        check_color(i.inner_color, 11)?;
        check(i.collarette, 0.1, 0.7, 11, "collarette outside 0.1..0.7")?;
        check(i.fibres, 0.0, 1.0, 11, "fibre contrast outside 0..1")?;
        check(i.crypts, 0.0, 1.0, 11, "crypts outside 0..1")?;
        check(i.limbal_width, 0.0, 0.3, 11, "limbal ring width outside 0..0.3")?;
        check(i.limbal_darkness, 0.0, 1.0, 11, "limbal ring darkness outside 0..1")?;
        let s = &self.sclera;
        check_color(s.color, 12)?;
        for v in [s.vein_density, s.redness, s.yellowing, s.limbal_darkening, s.roughness, s.specular_variation] {
            check(v, 0.0, 1.0, 12, "sclera value outside 0..1")?;
        }
        // Geometry: the cornea sphere must reach over the limbus circle.
        if self.cornea_radius_mm <= self.limbus_radius_mm || self.cornea_back_radius_mm <= self.limbus_radius_mm * 0.8 {
            return Err(ModelError::invalid("eye parameter", 2, "cornea radius too small for the limbus"));
        }
        let g = super::optics::EyeGeometry::new(self);
        if self.pupil_max_mm >= self.limbus_radius_mm - 0.5 {
            return Err(ModelError::invalid("eye parameter", 7, "pupil does not fit inside the iris"));
        }
        // The iris plane lies behind the corneal back surface and inside the eyeball, wide enough
        // for the iris.
        let back_vertex = g.apex_z - f64::from(self.cornea_thickness_mm);
        if g.iris_z >= back_vertex - 0.5 {
            return Err(ModelError::invalid("eye parameter", 5, "iris plane too close to the cornea"));
        }
        let inside = (g.radius * g.radius - g.iris_z * g.iris_z).max(0.0).sqrt();
        if inside < g.limbus {
            return Err(ModelError::invalid("eye parameter", 5, "iris plane too deep for the iris to fit"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid_and_physical() {
        let p = EyeParams::default();
        p.validate().unwrap();
        // Tear film f0 about 0.021, cornea about 0.025.
        assert!((f0_from_ior(TEAR_IOR) - 0.0209).abs() < 5e-4);
        assert!((f0_from_ior(CORNEA_IOR) - 0.0249).abs() < 5e-4);
        assert!((p.pupil_radius_mm() - 2.05).abs() < 1e-5);
    }

    #[test]
    fn bad_values_are_refused() {
        let base = EyeParams::default();
        let cases: [fn(&mut EyeParams); 6] = [
            |p| p.eyeball_radius_mm = f32::NAN,
            |p| p.limbus_radius_mm = 9.0,
            |p| p.dilation = 1.5,
            |p| p.iris.outer_color[1] = -0.1,
            |p| p.sclera.redness = 2.0,
            |p| {
                p.pupil_max_mm = 4.5;
                p.limbus_radius_mm = 5.0;
            },
        ];
        for (k, f) in cases.iter().enumerate() {
            let mut p = base;
            f(&mut p);
            assert!(p.validate().is_err(), "case {k}");
        }
    }
}
