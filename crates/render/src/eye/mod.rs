//! Eye shading inputs `[EYE-01]`..`[EYE-07]` and the feature table the geometry pass reads for the
//! eyes and the mouth.
//!
//! The eyeball part is drawn as an ordinary mesh of material class [`EYE_CLASS`]; what a pixel of
//! it shows is decided analytically in `shaders/eye/eye.wgsl`: the camera ray through the pixel is
//! traced against the analytic eye (`shaders/eye/optics.wgsl`, the f32 twin of
//! `humanart_model::eye::EyeGeometry`): the sclera, or the cornea, refracted through both corneal
//! surfaces onto the iris plane (true parallax, `[EYE-02]`, `[AC-WP16-2]`). The iris layers are
//! procedural (radial fibres, crypts, collarette, pupillary ruff, limbal ring, with the radial
//! coordinate normalised between the pupil and the limbus so the iris stretches with the pupil);
//! the sclera has a procedural vein network, limbal darkening and age yellowing under a wet tear
//! film whose roughness varies finely `[EYE-03]`; the cornea is a clear specular layer that
//! reflects the environment and the key lights as catchlights `[EYE-04]`. The lid aperture
//! (`humanart_model::eye::LidAperture`, recomputed from the deformed lid landmarks) darkens the
//! eyeball near the lid margins and under the upper lid and lashes `[EYE-05]`; the tear-line part
//! (class [`TEAR_CLASS`]) is shaded as a wet, smooth strip with the same occlusion.
//!
//! The table ([`FeatureTable`], `struct FeatureTable` in `shaders/eye/features.wgsl`) maps a scene
//! object index to one of [`MAX_FEATURE_ENTRIES`] entries of [`FEATURE_VEC4S`] vectors: an eye
//! ([`EyeShading`]) or a mouth (`crate::mouth::MouthShading`). Objects without an entry shade as
//! before. The table is one storage buffer of [`FEATURE_TABLE_BYTES`] bytes written in place
//! (`crate::renderer::Renderer::set_features`): changing the gaze, the pupil or the lids every
//! frame allocates nothing `[PERF-R1]`.
//!
//! [`guard::GuardMirror`] carries the CPU penetration guard's correction into the GPU
//! deformation's outputs `[EVAL-08]`.

pub mod guard;

use humanart_math::{Quat, Vec3};
use humanart_model::eye::{EyeFrame, EyeParams, LidAperture};

use crate::error::RenderError;
use crate::scene::MAX_OBJECTS;

/// Vec4s per table entry.
pub const FEATURE_VEC4S: usize = 16;
/// Entries in the table (16 eyes and mouths of 8 characters with room to spare).
pub const MAX_FEATURE_ENTRIES: usize = 32;
/// Objects the table maps (the scene's object limit).
pub const MAX_FEATURE_OBJECTS: usize = MAX_OBJECTS;
/// Size of the table buffer: the object slots (u32) and the entries.
pub const FEATURE_TABLE_BYTES: usize = MAX_FEATURE_OBJECTS * 4 + MAX_FEATURE_ENTRIES * FEATURE_VEC4S * 16;

/// Entry kind of an eye (`FEATURE_EYE` in the shader).
pub const FEATURE_EYE: u32 = 1;
/// Entry kind of a mouth (`FEATURE_MOUTH` in the shader).
pub const FEATURE_MOUTH: u32 = 2;

/// Material class of the eyeball part (`EYE_CLASS`).
pub const EYE_CLASS: u8 = 2;
/// Material class of the teeth (`TEETH_CLASS`).
pub const TEETH_CLASS: u8 = 4;
/// Material class of the tear-line (lacrimal meniscus) strip (`TEAR_CLASS`).
pub const TEAR_CLASS: u8 = 5;
/// Material class of the gums (`GUM_CLASS`).
pub const GUM_CLASS: u8 = 6;
/// Material class of the tongue (`TONGUE_CLASS`).
pub const TONGUE_CLASS: u8 = 7;
/// Material class of the eye-occlusion shell (`EYE_OCCLUSION_CLASS`): drawn as a stochastic
/// translucent shadow film over the eyeball.
pub const EYE_OCCLUSION_CLASS: u8 = 8;
/// Material class of the mouth-bag skin (`MOUTH_BAG_CLASS`): oral mucosa under the cavity's
/// occlusion.
pub const MOUTH_BAG_CLASS: u8 = 9;

/// One table entry as the shader reads it: [`FEATURE_VEC4S`] vectors, the kind in the last `w`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureEntry(pub [[f32; 4]; FEATURE_VEC4S]);

impl FeatureEntry {
    /// An entry of `kind` built from up to 63 floats (the 64th is the kind).
    pub(crate) fn from_floats(kind: u32, floats: &[f32]) -> Self {
        let mut out = [[0.0_f32; 4]; FEATURE_VEC4S];
        for (i, &v) in floats.iter().take(FEATURE_VEC4S * 4 - 1).enumerate() {
            out[i / 4][i % 4] = v;
        }
        out[FEATURE_VEC4S - 1][3] = kind as f32;
        Self(out)
    }

    /// The entry's kind ([`FEATURE_EYE`], [`FEATURE_MOUTH`]).
    #[must_use]
    pub fn kind(&self) -> u32 {
        let w = self.0[FEATURE_VEC4S - 1][3];
        if w.is_finite() && w >= 0.0 { (w + 0.5) as u32 } else { 0 }
    }

    /// Every value is finite.
    fn is_finite(&self) -> bool {
        self.0.iter().flatten().all(|v| v.is_finite())
    }
}

/// The eye and mouth table of the geometry pass (see the module documentation).
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureTable {
    /// The buffer image: slots, then entries, little-endian.
    bytes: Vec<u8>,
    /// Entries in use.
    entries: usize,
}

impl Default for FeatureTable {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureTable {
    /// An empty table: no object has an entry.
    #[must_use]
    pub fn new() -> Self {
        Self { bytes: vec![0; FEATURE_TABLE_BYTES], entries: 0 }
    }

    /// Entries in use.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries
    }

    /// No entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries == 0
    }

    /// Removes every entry and slot.
    pub fn clear(&mut self) {
        self.bytes.fill(0);
        self.entries = 0;
    }

    /// Adds `entry` and returns its index.
    ///
    /// # Errors
    /// [`RenderError::TooLarge`] beyond [`MAX_FEATURE_ENTRIES`]; [`RenderError::InvalidParameter`]
    /// for a non-finite value or an unknown kind.
    pub fn add(&mut self, entry: &FeatureEntry) -> Result<usize, RenderError> {
        if self.entries >= MAX_FEATURE_ENTRIES {
            return Err(RenderError::TooLarge {
                what: "eye and mouth entries",
                value: self.entries as u64 + 1,
                limit: MAX_FEATURE_ENTRIES as u64,
            });
        }
        let index = self.entries;
        self.entries += 1;
        if let Err(e) = self.set(index, entry) {
            self.entries -= 1;
            return Err(e);
        }
        Ok(index)
    }

    /// Replaces entry `index` (the per-frame update of gaze, pupil and lids).
    ///
    /// # Errors
    /// [`RenderError::InvalidParameter`] for an index not in use, a non-finite value or an
    /// unknown kind.
    pub fn set(&mut self, index: usize, entry: &FeatureEntry) -> Result<(), RenderError> {
        if index >= self.entries {
            return Err(RenderError::InvalidParameter { what: "feature entry index not in use" });
        }
        if !entry.is_finite() {
            return Err(RenderError::InvalidParameter { what: "feature entry value not finite" });
        }
        if !matches!(entry.kind(), FEATURE_EYE | FEATURE_MOUTH) {
            return Err(RenderError::InvalidParameter { what: "feature entry of unknown kind" });
        }
        let base = MAX_FEATURE_OBJECTS * 4 + index * FEATURE_VEC4S * 16;
        for (i, v) in entry.0.iter().flatten().enumerate() {
            let at = base + i * 4;
            self.bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
        }
        Ok(())
    }

    /// Makes object `object` (its index in the scene) shade with entry `entry`, or as an ordinary
    /// material for `None`.
    ///
    /// # Errors
    /// [`RenderError::InvalidParameter`] for an object beyond [`MAX_FEATURE_OBJECTS`] or an entry
    /// not in use.
    pub fn assign(&mut self, object: usize, entry: Option<usize>) -> Result<(), RenderError> {
        if object >= MAX_FEATURE_OBJECTS {
            return Err(RenderError::InvalidParameter { what: "feature object index beyond the object limit" });
        }
        let slot = match entry {
            Some(e) if e < self.entries => e as u32 + 1,
            Some(_) => return Err(RenderError::InvalidParameter { what: "feature entry index not in use" }),
            None => 0,
        };
        self.bytes[object * 4..object * 4 + 4].copy_from_slice(&slot.to_le_bytes());
        Ok(())
    }

    /// The entry object `object` shades with.
    #[must_use]
    pub fn entry_of(&self, object: usize) -> Option<usize> {
        let b = self.bytes.get(object * 4..object * 4 + 4)?;
        let slot = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
        (slot > 0).then(|| slot - 1)
    }

    /// The buffer image (`struct FeatureTable` in `shaders/eye/features.wgsl`).
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Floats 46..54 of an eye entry hold the upper lid curve, 54..62 the lower.
const LID_FLOATS: usize = 46;

/// Everything the eye shader needs about one eye, in world space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EyeShading {
    /// The eyeball fitted to the deformed eyeball part (centre, scleral radius, gaze axis).
    pub frame: EyeFrame,
    /// The head's rotation (its +Y is the iris plane's up, its frame the lid aperture's).
    pub head: Quat,
    /// Shape and appearance; lengths are scaled so the eyeball radius equals the fitted one.
    pub params: EyeParams,
    /// The lid opening over the eyeball.
    pub aperture: LidAperture,
    /// Density of the upper lashes' shadow on the eyeball, 0 (no lashes) to 1 (a full fringe):
    /// the hook through which the groom's lashes darken the band under the upper lid.
    pub lashes: f32,
    /// The table entry of the other eye drawn by the same object (both eyeballs, both tear lines
    /// and both occlusion shells are one object each): a pixel shades with the entry whose
    /// eyeball centre is nearer, and the partner's side is the medial side (caruncle).
    pub partner: Option<usize>,
}

/// Float of an eye entry holding the partner slot (entry + 1, 0 none).
const PARTNER_FLOAT: usize = 39;
/// Float of an eye entry holding the lash shadow density.
const LASH_FLOAT: usize = 62;

impl EyeShading {
    /// The table entry. Lengths in metres, the analytic eye scaled to the fitted radius.
    ///
    /// # Errors
    /// [`RenderError::InvalidParameter`] when the parameters do not validate or the frame is
    /// degenerate.
    pub fn entry(&self) -> Result<FeatureEntry, RenderError> {
        let p = &self.params;
        if p.validate().is_err() {
            return Err(RenderError::InvalidParameter { what: "eye parameters out of range" });
        }
        let f = &self.frame;
        if !(f.radius.is_finite() && f.radius > 1e-4 && f.centre.is_finite() && f.axis.length() > 0.5) {
            return Err(RenderError::InvalidParameter { what: "degenerate eye frame" });
        }
        // Millimetres of the parameters to metres of the fitted eyeball.
        let s = f.radius / p.eyeball_radius_mm;
        let up = self.head.mul_vec3(Vec3::Y);
        let q = self.head.to_array();
        let (iris, sclera, a) = (&p.iris, &p.sclera, &self.aperture);
        let mut floats = [0.0_f32; 63];
        let head: [f32; 46] = [
            f.centre.x,
            f.centre.y,
            f.centre.z,
            f.radius,
            f.axis.x,
            f.axis.y,
            f.axis.z,
            p.pupil_radius_mm() * s,
            up.x,
            up.y,
            up.z,
            p.limbus_radius_mm * s,
            p.cornea_radius_mm * s,
            p.cornea_back_radius_mm * s,
            p.cornea_thickness_mm * s,
            p.iris_depth_mm * s,
            humanart_model::eye::params::CORNEA_IOR,
            humanart_model::eye::params::AQUEOUS_IOR,
            sclera.roughness,
            sclera.specular_variation,
            iris.outer_color[0],
            iris.outer_color[1],
            iris.outer_color[2],
            iris.collarette,
            iris.inner_color[0],
            iris.inner_color[1],
            iris.inner_color[2],
            iris.fibres,
            iris.crypts,
            iris.limbal_width,
            iris.limbal_darkness,
            (iris.seed % 4096) as f32,
            sclera.color[0],
            sclera.color[1],
            sclera.color[2],
            sclera.vein_density,
            sclera.redness,
            sclera.yellowing,
            sclera.limbal_darkening,
            0.0, // PARTNER_FLOAT, written below.
            q[0],
            q[1],
            q[2],
            q[3],
            a.from,
            a.to,
        ];
        floats[..LID_FLOATS].copy_from_slice(&head);
        let n = humanart_model::eye::APERTURE_SAMPLES;
        floats[LID_FLOATS..LID_FLOATS + n].copy_from_slice(&a.upper);
        floats[LID_FLOATS + n..LID_FLOATS + 2 * n].copy_from_slice(&a.lower);
        #[allow(clippy::cast_precision_loss)] // Entry indices are below MAX_FEATURE_ENTRIES.
        let partner = self.partner.filter(|&e| e < MAX_FEATURE_ENTRIES).map_or(0.0, |e| (e + 1) as f32);
        floats[PARTNER_FLOAT] = partner;
        floats[LASH_FLOAT] = self.lashes.clamp(0.0, 1.0);
        let entry = FeatureEntry::from_floats(FEATURE_EYE, &floats);
        if entry.is_finite() {
            Ok(entry)
        } else {
            Err(RenderError::InvalidParameter { what: "eye shading value not finite" })
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn eye() -> EyeShading {
        EyeShading {
            frame: EyeFrame { centre: Vec3::new(0.03, 1.6, 0.05), radius: 0.0125, axis: Vec3::Z },
            head: Quat::IDENTITY,
            params: EyeParams::default(),
            aperture: LidAperture::default(),
            lashes: 0.6,
            partner: Some(3),
        }
    }

    fn float(entry: &FeatureEntry, i: usize) -> f32 {
        entry.0[i / 4][i % 4]
    }

    #[test]
    fn eye_entry_layout_matches_the_shader() {
        // shaders/eye/eye.wgsl reads these float positions.
        let e = eye().entry().unwrap();
        assert_eq!(e.kind(), FEATURE_EYE);
        let s = 0.0125 / 12.0;
        assert!((float(&e, 3) - 0.0125).abs() < 1e-7);
        assert!((float(&e, 11) - 5.85 * s).abs() < 1e-7, "limbus scaled to the fitted eyeball");
        assert!((float(&e, 15) - 3.6 * s).abs() < 1e-7, "iris depth");
        assert!((float(&e, 9) - 1.0).abs() < 1e-6, "head up");
        assert!((float(&e, 43) - 1.0).abs() < 1e-6, "head quaternion w");
        assert_eq!(float(&e, 44), LidAperture::default().from);
        assert_eq!(float(&e, LID_FLOATS + 3), LidAperture::default().upper[3]);
        assert_eq!(float(&e, LID_FLOATS + 8 + 7), LidAperture::default().lower[7]);
        assert_eq!(float(&e, PARTNER_FLOAT), 4.0, "partner slot = entry + 1");
        assert!((float(&e, LASH_FLOAT) - 0.6).abs() < 1e-7);
    }

    #[test]
    fn table_maps_objects_to_entries_in_place() {
        let mut table = FeatureTable::new();
        assert_eq!(table.as_bytes().len(), FEATURE_TABLE_BYTES);
        let e = table.add(&eye().entry().unwrap()).unwrap();
        table.assign(7, Some(e)).unwrap();
        assert_eq!(table.entry_of(7), Some(0));
        assert_eq!(table.entry_of(6), None);
        assert_eq!(&table.as_bytes()[28..32], &1_u32.to_le_bytes());
        let base = MAX_FEATURE_OBJECTS * 4;
        let w = f32::from_le_bytes(table.as_bytes()[base + 252..base + 256].try_into().unwrap());
        assert_eq!(w, FEATURE_EYE as f32);
        // An update rewrites the same bytes.
        let mut moved = eye();
        moved.frame.axis = Vec3::new(0.0, 0.1, 1.0).normalize();
        table.set(e, &moved.entry().unwrap()).unwrap();
        assert_eq!(table.len(), 1);
        table.assign(7, None).unwrap();
        assert_eq!(table.entry_of(7), None);
    }

    #[test]
    fn bad_entries_and_indices_are_refused() {
        let mut table = FeatureTable::new();
        let entry = eye().entry().unwrap();
        assert!(table.assign(0, Some(0)).is_err(), "no entry yet");
        assert!(table.set(0, &entry).is_err());
        assert!(table.assign(MAX_FEATURE_OBJECTS, None).is_err());
        let mut nan = entry;
        nan.0[2][1] = f32::NAN;
        assert!(table.add(&nan).is_err());
        assert!(table.is_empty(), "a refused entry is not kept");
        let mut unknown = entry;
        unknown.0[FEATURE_VEC4S - 1][3] = 9.0;
        assert!(table.add(&unknown).is_err());
        for _ in 0..MAX_FEATURE_ENTRIES {
            table.add(&entry).unwrap();
        }
        assert!(matches!(table.add(&entry), Err(RenderError::TooLarge { .. })));
        let mut bad = eye();
        bad.params.iris_depth_mm = -1.0;
        assert!(bad.entry().is_err());
        bad = eye();
        bad.frame.radius = 0.0;
        assert!(bad.entry().is_err());
        table.clear();
        assert!(table.is_empty());
        assert!(table.as_bytes().iter().all(|&b| b == 0));
    }
}
