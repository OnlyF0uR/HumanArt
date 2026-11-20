//! Mouth interior shading inputs `[MTH-01]`..`[MTH-04]`: teeth, gums and tongue.
//!
//! The teeth (class [`crate::eye::TEETH_CLASS`]), gums ([`crate::eye::GUM_CLASS`]) and tongue
//! ([`crate::eye::TONGUE_CLASS`]) parts share one entry of the feature table
//! ([`crate::eye::FeatureTable`]) and are shaded by `shaders/mouth/mouth.wgsl`:
//!
//! - teeth: enamel with a gingival-to-incisal colour gradient over the crown height (up from the
//!   upper incisal edge, down from the lower one), a translucent
//!   (greyer, darker) incisal edge, staining near the gum line, wrapped diffuse for the
//!   SSS-lite look, and the specular of enamel wetted by saliva (`TeethParams::f0`);
//! - gums: wrapped (scattering) diffuse under a wet specular;
//! - tongue: a whitish coating on the dorsum, a procedural papillae micro-normal and a broad wet
//!   specular.
//!
//! Everything inside the mouth is darkened by the lip opening's form factor
//! (`humanart_model::mouth::MouthOpening::occlusion`, evaluated per pixel with the same formula)
//! on the ambient light and, more softly, on the direct light, so a barely open mouth is dark
//! inside and an open one is lit only near the opening `[MTH-04]`.

use humanart_math::Vec3;
use humanart_model::mouth::{MouthOpening, MouthParams};

use crate::error::RenderError;
use crate::eye::{FEATURE_MOUTH, FeatureEntry};

/// Everything the mouth shader needs, in world space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MouthShading {
    /// The lip opening (from the deformed lip landmarks).
    pub opening: MouthOpening,
    /// The head's up axis.
    pub up: Vec3,
    /// A point on the upper incisal edge (the gradient runs up from it).
    pub upper_edge: Vec3,
    /// A point on the lower incisal edge (the gradient runs down from it).
    pub lower_edge: Vec3,
    /// Crown height over which the gingival-to-incisal gradient runs, metres.
    pub crown_height: f32,
    /// Teeth, gums and tongue.
    pub params: MouthParams,
}

impl MouthShading {
    /// The table entry.
    ///
    /// # Errors
    /// [`RenderError::InvalidParameter`] when the parameters do not validate, the axes are
    /// degenerate or a value is not finite.
    pub fn entry(&self) -> Result<FeatureEntry, RenderError> {
        let p = &self.params;
        if p.validate().is_err() {
            return Err(RenderError::InvalidParameter { what: "mouth parameters out of range" });
        }
        let o = &self.opening;
        let up = self.up.normalize_or_zero();
        if o.forward.length() < 0.5 || o.across.length() < 0.5 || up.length() < 0.5 {
            return Err(RenderError::InvalidParameter { what: "degenerate mouth axes" });
        }
        if !(self.crown_height.is_finite() && self.crown_height > 1e-4) {
            return Err(RenderError::InvalidParameter { what: "crown height must be positive" });
        }
        let (t, g, n) = (&p.teeth, &p.gums, &p.tongue);
        let floats = [
            o.centre.x,
            o.centre.y,
            o.centre.z,
            o.half_width.max(0.0),
            o.forward.x,
            o.forward.y,
            o.forward.z,
            o.half_height.max(0.0),
            o.across.x,
            o.across.y,
            o.across.z,
            self.crown_height,
            up.x,
            up.y,
            up.z,
            up.dot(self.upper_edge),
            t.gingival_color[0],
            t.gingival_color[1],
            t.gingival_color[2],
            t.translucency,
            t.incisal_color[0],
            t.incisal_color[1],
            t.incisal_color[2],
            t.wear,
            t.stain,
            t.wetness,
            t.f0(),
            up.dot(self.lower_edge),
            g.color[0],
            g.color[1],
            g.color[2],
            g.scattering,
            g.wetness,
            0.0,
            n.coating,
            n.papillae,
            n.color[0],
            n.color[1],
            n.color[2],
            n.wetness,
        ];
        let entry = FeatureEntry::from_floats(FEATURE_MOUTH, &floats);
        if entry.0.iter().flatten().all(|v| v.is_finite()) {
            Ok(entry)
        } else {
            Err(RenderError::InvalidParameter { what: "mouth shading value not finite" })
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::eye::FeatureTable;

    fn mouth() -> MouthShading {
        MouthShading {
            opening: MouthOpening {
                centre: Vec3::new(0.0, 1.55, 0.09),
                forward: Vec3::Z,
                across: Vec3::X,
                half_width: 0.025,
                half_height: 0.008,
            },
            up: Vec3::Y,
            upper_edge: Vec3::new(0.0, 1.553, 0.085),
            lower_edge: Vec3::new(0.0, 1.549, 0.084),
            crown_height: 0.010,
            params: MouthParams::default(),
        }
    }

    #[test]
    fn mouth_entry_layout_matches_the_shader() {
        // shaders/mouth/mouth.wgsl reads these float positions.
        let e = mouth().entry().unwrap();
        assert_eq!(e.kind(), FEATURE_MOUTH);
        let f = |i: usize| e.0[i / 4][i % 4];
        assert_eq!(f(3), 0.025);
        assert_eq!(f(7), 0.008);
        assert_eq!(f(11), 0.010);
        assert!((f(15) - 1.553).abs() < 1e-6, "upper incisal edge along up");
        assert!((f(27) - 1.549).abs() < 1e-6, "lower incisal edge along up");
        assert_eq!(f(26), MouthParams::default().teeth.f0());
        assert_eq!(f(39), MouthParams::default().tongue.wetness);
        let mut table = FeatureTable::new();
        let index = table.add(&e).unwrap();
        for object in [3, 4, 5] {
            table.assign(object, Some(index)).unwrap();
        }
        assert_eq!(table.entry_of(5), Some(index));
    }

    #[test]
    fn bad_mouth_inputs_are_refused() {
        let mut m = mouth();
        m.crown_height = 0.0;
        assert!(m.entry().is_err());
        m = mouth();
        m.opening.forward = Vec3::ZERO;
        assert!(m.entry().is_err());
        m = mouth();
        m.params.teeth.wear = 2.0;
        assert!(m.entry().is_err());
        m = mouth();
        m.opening.centre.x = f32::INFINITY;
        assert!(m.entry().is_err());
    }
}
