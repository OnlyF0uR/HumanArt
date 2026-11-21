//! The face-shape components measured from scans `[WP-25]` (todo 1.2): the principal components of
//! the face model's subjects around their mean face (`generate::scan_face`), bound to the registry
//! as `face.scan.shape_01..07` ([`SCAN_BINDINGS`]).
//!
//! Unlike the authored components of this module's siblings they come from data, so their kind is
//! [`DeltaKind::IdentityGlobal`] (a PCA over head and neck, `[ID-01]` (a)) and the template counts
//! them as scan-derived. They are made mirror-symmetric like the mean face, move the eye joints
//! and lid helpers with the eyes, and keep the procedural ears, scalp and neck (the face mask of
//! `generate::scan_face`).
//!
//! **What they are not.** The model averages 8 HSRD-100 subjects; 7 components are all that 8
//! samples span. Each axis mixes ancestry, sex, age, body fat and the residual head pose of these
//! people, and its standard deviation is that of 8 samples: variety around the mean face, not a
//! statistical face model and not identity control. The registry marks them provisional and keeps
//! their range inside what folds no triangle (`face_scan_range_folds_no_triangle`).

use humanart_math::Vec3;

use super::error::{IdResult, IdentityError};
use super::space::{IdentityComponent, Reach, Space, Symmetry};
use crate::delta::DeltaKind;
use crate::template::Template;

/// Registry parameter id and the scan component it binds to, in registry order
/// (`data/template/params/face.json`, group `face/shape`).
pub const SCAN_BINDINGS: &[(&str, &str)] = &[
    ("face.scan.shape_01", "scan.face-pc1"),
    ("face.scan.shape_02", "scan.face-pc2"),
    ("face.scan.shape_03", "scan.face-pc3"),
    ("face.scan.shape_04", "scan.face-pc4"),
    ("face.scan.shape_05", "scan.face-pc5"),
    ("face.scan.shape_06", "scan.face-pc6"),
    ("face.scan.shape_07", "scan.face-pc7"),
];

/// What the scan components may move: the head (face, eyes, mouth) and the neck under the chin.
pub const SCAN_FACE: Reach<'static> = Reach { regions: &["head", "neck"], except: &[], margin: 0.0 };

/// The joints that move with the left eye (the right ones mirror them).
const EYE_JOINTS: [&str; 3] = ["L.eye", "L.eyelid.upper", "L.eyelid.lower"];

/// One scan component as `generate::scan_face` carries it to the template.
#[derive(Clone, Copy, Debug)]
pub struct ScanField<'a> {
    /// Per template vertex: the displacement of `+1` standard deviation, metres.
    pub offsets: &'a [Vec3],
    /// The left eye's translation, metres.
    pub eye: Vec3,
}

/// The identity components of `fields`, in order: `fields[k]` becomes [`SCAN_BINDINGS`]`[k].1`.
///
/// # Errors
/// [`IdentityError`] when there are more fields than bindings, a field's vertex count differs from
/// the template's, a displacement is not finite, or the template lacks the eye joints or a region of
/// [`SCAN_FACE`].
pub fn components(template: &Template, fields: &[ScanField<'_>]) -> IdResult<Vec<IdentityComponent>> {
    if fields.len() > SCAN_BINDINGS.len() {
        return Err(IdentityError::invalid("scan component", fields.len(), "more components than registry bindings"));
    }
    let space = Space::new(template)?;
    let reps = space.field().representatives();
    let eye_joints = EYE_JOINTS.iter().map(|j| space.joint_index(j)).collect::<IdResult<Vec<u16>>>()?;
    let mut out = Vec::with_capacity(fields.len());
    for (k, (field, &(_, name))) in fields.iter().zip(SCAN_BINDINGS).enumerate() {
        if field.offsets.len() != template.mesh().vertex_count() {
            return Err(IdentityError::invalid("scan component", k, "vertex count differs from the template's"));
        }
        let disp: Vec<Vec3> =
            reps.iter().map(|&v| field.offsets.get(v as usize).copied().unwrap_or(Vec3::ZERO)).collect();
        let joints = eye_joints.iter().map(|&j| (j, field.eye)).collect();
        let mut c = space.component(name, Symmetry::Mirror, &[SCAN_FACE], disp, joints)?;
        c.kind = DeltaKind::IdentityGlobal;
        out.push(c);
    }
    Ok(out)
}
