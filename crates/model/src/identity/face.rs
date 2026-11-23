//! Face, ears, neck, eyes and teeth `[ID-01]` (c), and the asymmetry components `[ID-05]`.
//!
//! Every amount is an **authored model choice per unit weight** (metres or a scale fraction),
//! not a measurement `[ID-07]`, except eye spacing, ear size and neck width, which are scaled to
//! one standard deviation of an ANSUR II variable ([`super::measure`]). Falloffs are geodesic (along the skin, so the upper and lower lip
//! are apart) around named landmarks, or Euclidean around joints where separate parts (eyeballs,
//! tear lines) must move with the skin.
//!
//! | component | per unit weight |
//! |---|---|
//! | nose bridge height / width | 1.2 mm forward / 10 % across, around the bridge landmarks |
//! | nose length | tip and base 1.5 mm down (0.3 mm forward) |
//! | nose tip projection / rotation | tip 1.5 mm forward / 1.2 mm up |
//! | nostril width | alae 6 % across |
//! | eye spacing / depth | eye region outward, scaled to one SD of ANSUR II `interpupillarybreadth` ([`super::measure`]) / 1.2 mm back, eyeballs and eye joints with it |
//! | eye size | lids turn open by 0.04 rad about the eye centre |
//! | canthal tilt | lids turn 0.035 rad in the face plane (outer corner up) |
//! | lid crease height / hooding | crease 0.8 mm up / skin above it 0.8 mm down |
//! | brow ridge / height | 1.5 mm forward / 1.5 mm up |
//! | forehead height / slope | hairline 2.5 mm up / 5 % backward lean |
//! | lip fullness, cupid's bow | 0.6 mm outward / 0.6 mm peaks, 0.4 mm dip |
//! | mouth width / protrusion / philtrum length | 6 % across / 1.5 mm forward / 1.2 mm down |
//! | jaw width / angle | 4 % across at the angle / 1.2 mm out and down at the angle |
//! | chin projection / height / width | 2 mm forward / 6 % longer / 10 % across |
//! | cheekbones / cheek fullness | 1.5 mm out / 2 mm out |
//! | ear size / protrusion / lobe | scaled about the ear, one SD of ANSUR II `earlength` / 0.08 rad / lobe 12 % |
//! | neck width / length | out all around, one SD of ANSUR II `neckcircumference` / neck bones 4.5 % longer |
//! | eyeball size | 4 % about the eye centre, lids pushed with it |
//! | teeth | incisors 5 % larger, canines 8 % longer, premolars and molars 6 % broader, gum margin 0.6 mm |
//! | asymmetry | nose 0.8 mm off centre, left eye 0.6 mm up and right 0.3 mm down, left mouth corner 0.7 mm up, chin 1 mm off centre, left ear 1.5 mm up and right 2 % smaller, left brow 1 mm up, left cheekbone 1 mm out |

use humanart_math::Vec3;

use super::error::IdResult;
use super::skeletal::{Stretch, skeletal};
use super::space::{
    EVERYWHERE, IdentityComponent, PartKind, Point, Reach, Space, Symmetry, bump, segment_distance, smoothstep,
};
use super::{Kit, Raw};
use crate::expression::field::falloff;

/// The skin of the face and neck, without the eyes and the inside of the mouth.
pub const FACE_SKIN: Reach<'static> = Reach {
    regions: &["face", "neck"],
    except: &["eye-L", "eye-R", "mouth-interior", "teeth-upper", "teeth-lower", "gums-upper", "gums-lower", "tongue"],
    margin: 0.003,
};
/// The nose and its surroundings.
pub const NOSE: Reach<'static> = Reach {
    regions: &["nose", "forehead", "cheek-L", "cheek-R", "malar-L", "malar-R", "philtrum", "upper-lip"],
    except: &[],
    margin: 0.003,
};
/// The nose alone.
pub const NOSE_ONLY: Reach<'static> = Reach { regions: &["nose", "philtrum"], except: &[], margin: 0.002 };
/// The upper face around the eyes, eyes included (eye position).
pub const EYES_WHOLE: Reach<'static> =
    Reach { regions: &["face"], except: &["mouth", "chin", "jaw-L", "jaw-R"], margin: 0.003 };
/// The lids and sockets, without the eyeballs.
pub const LIDS: Reach<'static> =
    Reach { regions: &["eye-L", "eye-R"], except: &["eyeball-L", "eyeball-R"], margin: 0.002 };
/// The lids, sockets and their surroundings, without the eyeballs.
pub const LIDS_AROUND: Reach<'static> = Reach {
    regions: &[
        "eye-L", "eye-R", "brow-L", "brow-R", "forehead", "temple-L", "temple-R", "malar-L", "malar-R", "cheek-L",
        "cheek-R",
    ],
    except: &["eyeball-L", "eyeball-R"],
    margin: 0.003,
};
/// Brows and forehead above the lids.
pub const BROWS: Reach<'static> = Reach {
    regions: &["forehead", "brow-L", "brow-R", "eye-L", "eye-R", "temple-L", "temple-R", "nasal-bridge", "hairline"],
    except: &[
        "eyeball-L",
        "eyeball-R",
        "lower-lid-L",
        "lower-lid-R",
        "tear-line-L",
        "tear-line-R",
        "eye-occlusion-L",
        "eye-occlusion-R",
    ],
    margin: 0.003,
};
/// Forehead, hairline and scalp.
pub const FOREHEAD: Reach<'static> = Reach {
    regions: &["forehead", "hairline", "scalp", "temple-L", "temple-R", "brow-L", "brow-R"],
    except: &[],
    margin: 0.003,
};
/// The upper lip.
pub const UPPER_LIP: Reach<'static> = Reach {
    regions: &["upper-lip", "lip-corner-L", "lip-corner-R", "philtrum", "mouth-helper"],
    except: &[],
    margin: 0.0015,
};
/// The lower lip.
pub const LOWER_LIP: Reach<'static> = Reach {
    regions: &["lower-lip", "lip-corner-L", "lip-corner-R", "chin", "mouth-helper"],
    except: &[],
    margin: 0.0015,
};
/// The mouth and its surroundings, without teeth, gums and tongue.
pub const MOUTH: Reach<'static> = Reach {
    regions: &["mouth", "cheek-L", "cheek-R", "chin", "jaw-L", "jaw-R"],
    except: &["teeth-upper", "teeth-lower", "gums-upper", "gums-lower", "tongue"],
    margin: 0.003,
};
/// Jaw, chin, cheeks and neck.
pub const JAW: Reach<'static> = Reach {
    regions: &["jaw-L", "jaw-R", "chin", "cheek-L", "cheek-R", "neck", "malar-L", "malar-R", "temple-L", "temple-R"],
    except: &[],
    margin: 0.004,
};
/// The chin and below it.
pub const CHIN: Reach<'static> = Reach {
    regions: &["chin", "jaw-L", "jaw-R", "neck-front", "neck-side-L", "neck-side-R"],
    except: &[],
    margin: 0.003,
};
/// Cheeks and cheekbones.
pub const CHEEKS: Reach<'static> = Reach {
    regions: &["malar-L", "malar-R", "cheek-L", "cheek-R", "temple-L", "temple-R", "jaw-L", "jaw-R"],
    except: &[],
    margin: 0.003,
};
/// The cheekbones: the cheeks' regions, ramping in over 8 mm. The scanned face (`generate::scan_face`)
/// sets the eyes 4 mm higher than the procedural head while the cheekbone stays, so the full
/// cheekbone lies a few millimetres from the lower lid: a 3 mm ramp there folds the lid border.
pub const CHEEKBONES: Reach<'static> = Reach { margin: 0.008, ..CHEEKS };
/// The ears, ramping in over 1 cm from where they join the head.
pub const EARS: Reach<'static> = Reach { regions: &["ear-L", "ear-R"], except: &[], margin: 0.01 };
/// The ears, ramping in over 1.5 cm (turning the ear bends its root).
pub const EARS_SOFT: Reach<'static> = Reach { regions: &["ear-L", "ear-R"], except: &[], margin: 0.015 };
/// The neck and where it meets head and trunk.
pub const NECK: Reach<'static> = Reach {
    regions: &["neck", "jaw-L", "jaw-R", "chin", "shoulder", "chest", "back", "scalp"],
    except: &[],
    margin: 0.01,
};
/// The eyes (eyeball scaling).
pub const EYES: Reach<'static> = Reach { regions: &["eye-L", "eye-R"], except: &[], margin: 0.003 };
/// The teeth.
pub const TEETH: Reach<'static> = Reach { regions: &["teeth-upper", "teeth-lower"], except: &[], margin: 0.0 };
/// The gums.
pub const GUMS: Reach<'static> = Reach { regions: &["gums-upper", "gums-lower"], except: &[], margin: 0.0 };

const LIP_SERIES: [&str; 4] = ["lip.upper.outer", "lip.upper.inner", "lip.lower.outer", "lip.lower.inner"];

/// The component bound to `name`, if it is one of this module's.
pub fn recipe(kit: &Kit<'_, '_>, name: &str) -> IdResult<Option<IdentityComponent>> {
    let s = kit.space;
    let (raw, symmetry) = match name {
        "semantic.nose-bridge-height" => (bridge(kit, false)?, Symmetry::Mirror),
        "semantic.nose-bridge-width" => (bridge(kit, true)?, Symmetry::Mirror),
        "semantic.nose-length" => (nose_length(kit)?, Symmetry::Mirror),
        "semantic.nose-tip-projection" => (nose_tip(kit, Vec3::Z * 0.0015, &["nose.tip"])?, Symmetry::Mirror),
        "semantic.nose-tip-rotation" => {
            (nose_tip(kit, Vec3::new(0.0, 0.0012, -0.0003), &["nose.tip", "nose.subnasale"])?, Symmetry::Mirror)
        }
        "semantic.nostril-width" => (nostrils(kit)?, Symmetry::Mirror),
        "semantic.eye-spacing" => (eye_shift(kit, Vec3::X * 0.001, 1.0, 0.0)?, Symmetry::Mirror),
        "semantic.eye-size" => (eye_size(kit)?, Symmetry::Mirror),
        "semantic.canthal-tilt" => (canthal_tilt(kit)?, Symmetry::Mirror),
        "semantic.eye-depth" => (eye_shift(kit, Vec3::Z * -0.0012, 1.0, 0.0)?, Symmetry::Mirror),
        "semantic.lid-crease-height" => (lid_crease(kit)?, Symmetry::Mirror),
        "semantic.lid-hooding" => (lid_hooding(kit)?, Symmetry::Mirror),
        "semantic.brow-ridge" => (brow_ridge(kit)?, Symmetry::Mirror),
        "semantic.brow-height" => (brow_height(kit)?, Symmetry::Mirror),
        "semantic.forehead-height" => (forehead(kit, false)?, Symmetry::Mirror),
        "semantic.forehead-slope" => (forehead(kit, true)?, Symmetry::Mirror),
        "semantic.upper-lip-fullness" => (lip_fullness(kit, true)?, Symmetry::Mirror),
        "semantic.lower-lip-fullness" => (lip_fullness(kit, false)?, Symmetry::Mirror),
        "semantic.cupids-bow" => (cupids_bow(kit)?, Symmetry::Mirror),
        "semantic.mouth-width" => (mouth_width(kit)?, Symmetry::Mirror),
        "semantic.mouth-protrusion" => (mouth_protrusion(kit)?, Symmetry::Mirror),
        "semantic.philtrum-length" => (philtrum(kit)?, Symmetry::Mirror),
        "semantic.jaw-width" => (jaw_width(kit)?, Symmetry::Mirror),
        "semantic.jaw-angle" => (jaw_angle(kit)?, Symmetry::Mirror),
        "semantic.chin-projection" => (chin_projection(kit)?, Symmetry::Mirror),
        "semantic.chin-height" => (chin_height(kit)?, Symmetry::Mirror),
        "semantic.chin-width" => (chin_width(kit)?, Symmetry::Mirror),
        "semantic.cheekbone-prominence" => (cheekbones(kit, true)?, Symmetry::Mirror),
        "semantic.cheek-fullness" => (cheek_fullness(kit)?, Symmetry::Mirror),
        "semantic.ear-size" => (ear_scale(kit, Vec3::splat(0.05))?, Symmetry::Mirror),
        "semantic.ear-protrusion" => (ear_protrusion(kit)?, Symmetry::Mirror),
        "semantic.earlobe-size" => (earlobes(kit)?, Symmetry::Mirror),
        "semantic.neck-width" => (neck_width(kit)?, Symmetry::Mirror),
        "semantic.neck-length" => (neck_length(kit)?, Symmetry::Mirror),
        "semantic.eyeball-size" => (eyeballs(kit)?, Symmetry::Mirror),
        "semantic.teeth-incisor" => (teeth(kit, Tooth::Incisor)?, Symmetry::Mirror),
        "semantic.teeth-canine" => (teeth(kit, Tooth::Canine)?, Symmetry::Mirror),
        "semantic.teeth-molar" => (teeth(kit, Tooth::Molar)?, Symmetry::Mirror),
        "semantic.gum-line" => (gums(kit)?, Symmetry::Mirror),
        "asymmetry.nose" => (asym_nose(kit)?, Symmetry::Free),
        "asymmetry.eyes" => (asym_eyes(kit)?, Symmetry::Free),
        "asymmetry.mouth" => (asym_mouth(kit)?, Symmetry::Free),
        "asymmetry.jaw" => (asym_jaw(kit)?, Symmetry::Free),
        "asymmetry.ears" => (asym_ears(kit)?, Symmetry::Free),
        "asymmetry.brow" => (asym_brow(kit)?, Symmetry::Free),
        "asymmetry.cheeks" => (cheekbones_left(kit, 0.001)?, Symmetry::Free),
        "asymmetry.global" => (asym_global(kit)?, Symmetry::Free),
        _ => return Ok(None),
    };
    let raw = super::measure::calibrate(s, name, raw)?;
    Ok(Some(raw.component(s, name, symmetry)?))
}

/// The landmark names of `prefix.01`, `prefix.02`, ... that exist.
fn series(s: &Space<'_>, prefix: &str) -> IdResult<Vec<String>> {
    let count = s.landmark_series(prefix)?.len();
    Ok((1..=count).map(|k| format!("{prefix}.{k:02}")).collect())
}

/// All lip contour landmarks plus the corners.
fn lip_names(s: &Space<'_>) -> IdResult<Vec<String>> {
    let mut out = Vec::new();
    for p in LIP_SERIES {
        out.extend(series(s, p)?);
    }
    out.push("L.lip.corner".into());
    out.push("R.lip.corner".into());
    Ok(out)
}

/// A geodesic falloff around owned landmark names.
fn around(s: &Space<'_>, names: &[String], inner: f32, outer: f32) -> IdResult<Vec<f32>> {
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    s.around(&refs, inner, outer)
}

fn at(w: &[f32], pt: &Point) -> f32 {
    w.get(pt.node).copied().unwrap_or(0.0)
}

/// The smallest distance from `p` to any of `points`.
fn nearest_distance(p: Vec3, points: &[Vec3]) -> f32 {
    points.iter().map(|q| q.distance(p)).fold(f32::INFINITY, f32::min)
}

/// Whether the region of `pt` lies in the subtree of `root`.
fn within(s: &Space<'_>, pt: &Point, root: u16) -> bool {
    s.template().regions().is_within(pt.region, root)
}

/// Nose bridge: forward (height) or across (width).
pub fn bridge(kit: &Kit<'_, '_>, width: bool) -> IdResult<Raw> {
    let s = kit.space;
    let w = s.around(&["nose.bridge.01", "nose.bridge.02"], 0.004, if width { 0.015 } else { 0.012 })?;
    let disp =
        s.eval(|pt| if width { Vec3::X * (0.10 * pt.p.x * at(&w, pt)) } else { Vec3::Z * (0.0012 * at(&w, pt)) });
    Raw::confined(s, NOSE, Symmetry::Mirror, disp, Vec::new())
}

/// `semantic.nose-length`: tip and base down.
pub fn nose_length(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let w = s.around_solid(
        &["nose.tip", "nose.subnasale", "L.nose.alare", "R.nose.alare", "L.nose.alar-crease", "R.nose.alar-crease"],
        0.008,
        0.024,
        "nose",
    )?;
    let lip = s.series_nodes("lip.upper.outer")?;
    let d_lip = s.geodesic(&lip, 0.012);
    let disp = s.eval(|pt| {
        let free = smoothstep(0.002, 0.012, d_lip.get(pt.node).copied().unwrap_or(f32::INFINITY));
        Vec3::new(0.0, -0.0015, 0.0003) * (at(&w, pt) * free)
    });
    Raw::confined(s, NOSE, Symmetry::Mirror, disp, Vec::new())
}

/// A translation `d` of the nose tip area around `seeds`.
pub(crate) fn nose_tip(kit: &Kit<'_, '_>, d: Vec3, seeds: &[&str]) -> IdResult<Raw> {
    let s = kit.space;
    let w = s.around_solid(seeds, 0.003, 0.013, "nose")?;
    let disp = s.eval(|pt| d * at(&w, pt));
    Raw::confined(s, NOSE_ONLY, Symmetry::Mirror, disp, Vec::new())
}

/// `semantic.nostril-width`: the alae across.
pub(crate) fn nostrils(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    // Euclidean: the alar crease folds, and the cheek beside the ala moves with it.
    let mut points = Vec::new();
    for n in [
        "L.nose.alare",
        "R.nose.alare",
        "L.nose.alar-crease",
        "R.nose.alar-crease",
        "L.nose.alar-base",
        "R.nose.alar-base",
    ] {
        points.push(s.landmark(n)?);
    }
    let disp = s.eval(|pt| Vec3::X * (0.06 * pt.p.x * falloff(nearest_distance(pt.p, &points), 0.004, 0.012)));
    Raw::confined(s, NOSE, Symmetry::Mirror, disp, Vec::new())
}

/// The whole left eye region (eyeball, lids, socket) moved by `d` with a Euclidean falloff around
/// the eye joint; `left` and `right` scale the left eye's and the mirrored right eye's share
/// (`1, 0` with mirror symmetry gives both eyes). The eye and lid joints follow.
fn eye_shift(kit: &Kit<'_, '_>, d: Vec3, left: f32, right: f32) -> IdResult<Raw> {
    let s = kit.space;
    let eye = s.joint("L.eye")?;
    let f = move |pt: &Point| d * falloff(pt.p.distance(eye), 0.018, 0.032);
    let disp = if right == 0.0 {
        s.eval(|pt| s.sym(pt, &f) * left)
    } else {
        s.eval(|pt| {
            let m = Point { p: super::space::mirror(pt.p), ..*pt };
            f(pt) * left + super::space::mirror(f(&m)) * right
        })
    };
    let mut joints = Vec::new();
    for n in ["eye", "eyelid.upper", "eyelid.lower"] {
        joints.push((s.joint_index(&format!("L.{n}"))?, d * left));
        if right != 0.0 {
            joints.push((s.joint_index(&format!("R.{n}"))?, super::space::mirror(d) * right));
        }
    }
    let symmetry = if right == 0.0 { Symmetry::Mirror } else { Symmetry::Free };
    Raw::confined(s, EYES_WHOLE, symmetry, disp, joints)
}

/// The rest positions of the left lid margin landmarks (upper and lower).
fn lid_margins(s: &Space<'_>) -> IdResult<(Vec<Vec3>, Vec<Vec3>)> {
    Ok((s.landmark_series("L.eye.upper")?, s.landmark_series("L.eye.lower")?))
}

/// `semantic.eye-size`: the lids turn open about the eye centre (they slide on the eyeball).
pub(crate) fn eye_size(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let eye = s.joint("L.eye")?;
    let (upper, lower) = lid_margins(s)?;
    let margin: Vec<Vec3> = upper.iter().chain(&lower).copied().collect();
    let theta = 0.04;
    let f = move |pt: &Point| {
        if pt.kind == PartKind::Rigid {
            return Vec3::ZERO;
        }
        let v = pt.p - eye;
        let u = 2.0 * smoothstep(-0.004, 0.004, v.y) - 1.0;
        let m = falloff(nearest_distance(pt.p, &margin), 0.002, 0.009);
        Vec3::X.cross(v) * (-theta * u * m)
    };
    let disp = s.eval(|pt| s.sym(pt, &f));
    Raw::confined(s, LIDS, Symmetry::Mirror, disp, Vec::new())
}

/// `semantic.canthal-tilt`: the lids turn in the face plane, outer corner up.
fn canthal_tilt(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let eye = s.joint("L.eye")?;
    let theta = 0.035;
    let f = move |pt: &Point| {
        if pt.kind == PartKind::Rigid {
            return Vec3::ZERO;
        }
        let v = pt.p - eye;
        Vec3::Z.cross(Vec3::new(v.x, v.y, 0.0)) * (theta * falloff(v.length(), 0.014, 0.024))
    };
    let disp = s.eval(|pt| s.sym(pt, &f));
    Raw::confined(s, LIDS_AROUND, Symmetry::Mirror, disp, Vec::new())
}

/// `semantic.lid-crease-height`: the crease up, the lash line fixed.
fn lid_crease(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let crease = s.landmark_series("L.eye.crease")?;
    let (upper, _) = lid_margins(s)?;
    let f = move |pt: &Point| {
        if pt.kind == PartKind::Rigid {
            return Vec3::ZERO;
        }
        let w = falloff(nearest_distance(pt.p, &crease), 0.0015, 0.006)
            * smoothstep(0.001, 0.004, nearest_distance(pt.p, &upper));
        Vec3::Y * (0.0008 * w)
    };
    let disp = s.eval(|pt| s.sym(pt, &f));
    Raw::confined(s, LIDS, Symmetry::Mirror, disp, Vec::new())
}

/// `semantic.lid-hooding`: the skin above the crease down over it.
fn lid_hooding(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let crease = s.landmark_series("L.eye.crease")?;
    let (upper, _) = lid_margins(s)?;
    let f = move |pt: &Point| {
        if pt.kind == PartKind::Rigid {
            return Vec3::ZERO;
        }
        let Some(&c) = crease.iter().min_by(|a, b| a.distance(pt.p).total_cmp(&b.distance(pt.p))) else {
            return Vec3::ZERO;
        };
        let w = falloff(c.distance(pt.p), 0.0, 0.008)
            * smoothstep(-0.001, 0.002, pt.p.y - c.y)
            * smoothstep(0.001, 0.004, nearest_distance(pt.p, &upper));
        Vec3::new(0.0, -0.0008, 0.0003) * w
    };
    let disp = s.eval(|pt| s.sym(pt, &f));
    Raw::confined(s, LIDS_AROUND, Symmetry::Mirror, disp, Vec::new())
}

/// The brow landmarks of both sides.
fn brow_names(s: &Space<'_>) -> IdResult<Vec<String>> {
    let mut out = series(s, "L.brow")?;
    out.extend(series(s, "R.brow")?);
    Ok(out)
}

/// A factor that keeps the lids still: 0 below the eye joints' height plus 6 mm and within 5 mm
/// of the upper lash line, 1 from 12 mm above the eyes and 11 mm from the lash line.
fn above_eyes(s: &Space<'_>) -> IdResult<impl Fn(Vec3) -> f32 + use<>> {
    let y = s.joint("L.eye")?.y;
    let mut lash = s.landmark_series("L.eye.upper")?;
    lash.extend(lash.clone().into_iter().map(super::space::mirror));
    Ok(move |p: Vec3| smoothstep(y + 0.006, y + 0.012, p.y) * smoothstep(0.005, 0.011, nearest_distance(p, &lash)))
}

/// `semantic.brow-ridge`: the brow forward (and a little down).
pub fn brow_ridge(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let mut names = brow_names(s)?;
    names.push("forehead.glabella".into());
    let w = around(s, &names, 0.003, 0.014)?;
    let up = above_eyes(s)?;
    let disp = s.eval(|pt| Vec3::new(0.0, -0.0003, 0.0015) * (at(&w, pt) * up(pt.p)));
    Raw::confined(s, BROWS, Symmetry::Mirror, disp, Vec::new())
}

/// `semantic.brow-height`: the brows up.
pub fn brow_height(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let w = around(s, &brow_names(s)?, 0.004, 0.014)?;
    let up = above_eyes(s)?;
    let disp = s.eval(|pt| Vec3::Y * (0.0015 * at(&w, pt) * up(pt.p)));
    Raw::confined(s, BROWS, Symmetry::Mirror, disp, Vec::new())
}

/// `semantic.forehead-height` (hairline up) or `semantic.forehead-slope` (forehead leaning back).
fn forehead(kit: &Kit<'_, '_>, slope: bool) -> IdResult<Raw> {
    let s = kit.space;
    let brows = s.landmark_series("L.brow")?;
    let y_b = brows.iter().map(|p| p.y).sum::<f32>() / brows.len() as f32;
    let hair = s.landmark_series("hairline")?;
    let y_h = hair.iter().map(|p| p.y).fold(f32::MIN, f32::max);
    let y_g = brows.iter().map(|p| p.y).fold(f32::MIN, f32::max) + 0.005;
    let z_t = s.landmark("ear.L.tragion")?.z;
    let disp = s.eval(|pt| {
        let front = smoothstep(z_t + 0.03, z_t + 0.07, pt.p.z);
        if slope {
            Vec3::Z * (-0.05 * (pt.p.y - y_g) * smoothstep(y_g, y_g + 0.02, pt.p.y) * front)
        } else {
            Vec3::Y * (0.0025 * smoothstep(y_b, y_h, pt.p.y) * front)
        }
    });
    Raw::confined(s, FOREHEAD, Symmetry::Mirror, disp, Vec::new())
}

/// Upper or lower lip fullness: the vermilion outward.
pub fn lip_fullness(kit: &Kit<'_, '_>, upper: bool) -> IdResult<Raw> {
    let s = kit.space;
    let (outer, inner, reach, lift) = if upper {
        ("lip.upper.outer", "lip.upper.inner", UPPER_LIP, 0.0002)
    } else {
        ("lip.lower.outer", "lip.lower.inner", LOWER_LIP, -0.0002)
    };
    let mut names = series(s, outer)?;
    names.extend(series(s, inner)?);
    let w = around(s, &names, 0.0005, 0.005)?;
    let disp = s.eval(|pt| (pt.n * 0.0006 + Vec3::Y * lift) * at(&w, pt));
    Raw::confined(s, reach, Symmetry::Mirror, disp, Vec::new())
}

/// `semantic.cupids-bow`: the peaks of the upper lip line up, the centre down.
fn cupids_bow(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let w = around(s, &series(s, "lip.upper.outer")?, 0.0005, 0.003)?;
    let peak = s.landmark("L.philtrum.column")?.x.abs().max(0.002);
    let disp = s.eval(|pt| {
        let x = pt.p.x;
        let g = 0.0006 * (-((x.abs() - peak) / 0.0025).powi(2)).exp() - 0.0004 * (-(x / 0.002).powi(2)).exp();
        Vec3::Y * (g * at(&w, pt))
    });
    Raw::confined(s, UPPER_LIP, Symmetry::Mirror, disp, Vec::new())
}

/// `semantic.mouth-width`: the mouth across.
fn mouth_width(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let w = around(s, &lip_names(s)?, 0.002, 0.012)?;
    let disp = s.eval(|pt| Vec3::X * (0.06 * pt.p.x * at(&w, pt)));
    Raw::confined(s, MOUTH, Symmetry::Mirror, disp, Vec::new())
}

/// `semantic.mouth-protrusion`: lips and their surroundings forward.
fn mouth_protrusion(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let w = around(s, &lip_names(s)?, 0.006, 0.02)?;
    let disp = s.eval(|pt| Vec3::Z * (0.0015 * at(&w, pt)));
    Raw::confined(s, MOUTH, Symmetry::Mirror, disp, Vec::new())
}

/// `semantic.philtrum-length`: the mouth down, the nose base fixed.
fn philtrum(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let w = around(s, &lip_names(s)?, 0.01, 0.025)?;
    let base = s.landmark_nodes(&["nose.subnasale", "L.nose.alar-base", "R.nose.alar-base"])?;
    let d_base = s.geodesic(&base, 0.012);
    let disp = s.eval(|pt| {
        let free = smoothstep(0.002, 0.010, d_base.get(pt.node).copied().unwrap_or(f32::INFINITY));
        Vec3::Y * (-0.0012 * at(&w, pt) * free)
    });
    Raw::confined(s, MOUTH, Symmetry::Mirror, disp, Vec::new())
}

/// `semantic.jaw-width`: the jaw angles apart.
pub fn jaw_width(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let g = s.landmark("jaw.L.01")?;
    let f = move |pt: &Point| Vec3::X * (0.04 * pt.p.x * falloff(pt.p.distance(g), 0.012, 0.04));
    let disp = s.eval(|pt| s.sym(pt, &f));
    Raw::confined(s, JAW, Symmetry::Mirror, disp, Vec::new())
}

/// `semantic.jaw-angle`: the jaw angle out and down.
fn jaw_angle(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let g = s.landmark("jaw.L.01")?;
    let dir = Vec3::new(1.0, -1.0, -0.3).normalize();
    let f = move |pt: &Point| dir * (0.0012 * falloff(pt.p.distance(g), 0.004, 0.018));
    let disp = s.eval(|pt| s.sym(pt, &f));
    Raw::confined(s, JAW, Symmetry::Mirror, disp, Vec::new())
}

const CHIN_SEEDS: [&str; 3] = ["chin.mid", "chin.supramentale", "chin.menton"];

/// `semantic.chin-projection`: the chin forward, the lower lip fixed.
pub(crate) fn chin_projection(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let w = s.around(&["chin.mid", "chin.menton"], 0.006, 0.025)?;
    let lip = s.series_nodes("lip.lower.outer")?;
    let d_lip = s.geodesic(&lip, 0.02);
    let disp = s.eval(|pt| {
        // Free of the lower lip: on the scanned face the skin below the vermilion (region
        // `lower-lip`) reaches 12 mm down, so the chin starts fading 18 mm from the border.
        let free = smoothstep(0.008, 0.018, d_lip.get(pt.node).copied().unwrap_or(f32::INFINITY));
        Vec3::Z * (0.002 * at(&w, pt) * free)
    });
    Raw::confined(s, CHIN, Symmetry::Mirror, disp, Vec::new())
}

/// `semantic.chin-height`: the chin longer below the lower lip.
pub fn chin_height(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let w = s.around(&CHIN_SEEDS, 0.01, 0.035)?;
    let y0 = s.landmark_series("lip.lower.outer")?.iter().map(|p| p.y).fold(f32::MAX, f32::min);
    let disp = s.eval(|pt| {
        let below = 1.0 - smoothstep(y0 - 0.004, y0, pt.p.y);
        Vec3::Y * (0.06 * (pt.p.y - y0) * at(&w, pt) * below)
    });
    Raw::confined(s, CHIN, Symmetry::Mirror, disp, Vec::new())
}

/// `semantic.chin-width`: the chin across.
pub fn chin_width(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let w = s.around(&CHIN_SEEDS, 0.006, 0.022)?;
    let disp = s.eval(|pt| Vec3::X * (0.10 * pt.p.x * at(&w, pt)));
    Raw::confined(s, JAW, Symmetry::Mirror, disp, Vec::new())
}

/// The skin under the chin lifted by `a` metres (per unit weight) toward the jaw: a more defined
/// angle between the chin's underside and the neck (the cervicomental angle), which is soft in
/// a child's face. Full from 1 cm to 3 cm behind the menton, between the jaw angles, on the
/// downward-facing skin only; nothing at the menton itself or on the neck's front.
pub(crate) fn submental_lift(kit: &Kit<'_, '_>, a: f32) -> IdResult<Raw> {
    let s = kit.space;
    let menton = s.landmark("chin.menton")?;
    let gonion = s.landmark("jaw.L.01")?;
    let f = move |pt: &Point| {
        let back = menton.z - pt.p.z;
        let along = smoothstep(0.0, 0.01, back) * smoothstep(0.05, 0.03, back);
        let across = smoothstep(gonion.x, gonion.x * 0.5, pt.p.x.abs());
        let below = smoothstep(menton.y + 0.004, menton.y - 0.002, pt.p.y)
            * smoothstep(menton.y - 0.04, menton.y - 0.02, pt.p.y);
        let down = smoothstep(-0.2, -0.6, pt.n.y);
        Vec3::Y * (a * along * across * below * down)
    };
    let mut disp = s.eval(|pt| s.sym(pt, &f));
    s.smooth_field(&mut disp, 10, &NECK)?;
    Raw::confined(s, NECK, Symmetry::Mirror, disp, Vec::new())
}

/// The cheekbone field of the left side: outward along the malar-zygion line.
fn cheekbone_field(s: &Space<'_>, a: f32) -> IdResult<impl Fn(&Point) -> Vec3 + use<>> {
    let malar = s.landmark("L.cheek.malar")?;
    // The arch toward the zygion, stopping short of the ear.
    let zygion = malar.lerp(s.landmark("L.zygion")?, 0.6);
    Ok(move |pt: &Point| {
        (pt.n * a + Vec3::Y * (a * 0.2)) * falloff(segment_distance(pt.p, malar, zygion), 0.006, 0.018)
    })
}

/// `semantic.cheekbone-prominence` (both sides).
pub(crate) fn cheekbones(kit: &Kit<'_, '_>, both: bool) -> IdResult<Raw> {
    let s = kit.space;
    let f = cheekbone_field(s, 0.0015)?;
    let disp = if both { s.eval(|pt| s.sym(pt, &f)) } else { s.eval(|pt| f(pt)) };
    Raw::confined(s, CHEEKBONES, if both { Symmetry::Mirror } else { Symmetry::Free }, disp, Vec::new())
}

/// The left cheekbone alone, `a` metres (asymmetry).
fn cheekbones_left(kit: &Kit<'_, '_>, a: f32) -> IdResult<Raw> {
    let s = kit.space;
    let f = cheekbone_field(s, a)?;
    let disp = s.eval(|pt| f(pt));
    Raw::confined(s, CHEEKBONES, Symmetry::Free, disp, Vec::new())
}

/// `semantic.cheek-fullness`: the soft cheek below the cheekbone outward.
pub(crate) fn cheek_fullness(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let c = s.landmark("L.cheek.malar")?.lerp(s.landmark("jaw.L.02")?, 0.5);
    let f = move |pt: &Point| pt.n * (0.002 * falloff(pt.p.distance(c), 0.008, 0.028));
    let disp = s.eval(|pt| s.sym(pt, &f));
    Raw::confined(s, CHEEKS, Symmetry::Mirror, disp, Vec::new())
}

/// Scales the ears per axis by `k` about the tragion (both sides).
pub fn ear_scale(kit: &Kit<'_, '_>, k: Vec3) -> IdResult<Raw> {
    let s = kit.space;
    let ear = s.region("ear-L")?;
    let t = s.landmark("ear.L.tragion")?;
    let f = move |pt: &Point| {
        if !within(s, pt, ear) {
            return Vec3::ZERO;
        }
        let v = pt.p - t;
        Vec3::new(k.x * v.x, k.y * v.y, k.z * v.z)
    };
    let disp = s.eval(|pt| s.sym(pt, &f));
    Raw::confined(s, EARS, Symmetry::Mirror, disp, Vec::new())
}

/// `semantic.ear-protrusion`: the ears turn away from the head about a vertical axis at their root.
fn ear_protrusion(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let ear = s.region("ear-L")?;
    let t = s.landmark("ear.L.tragion")?;
    let f = move |pt: &Point| {
        if !within(s, pt, ear) {
            return Vec3::ZERO;
        }
        (pt.p - t).cross(Vec3::Y) * 0.08
    };
    let disp = s.eval(|pt| s.sym(pt, &f));
    Raw::confined(s, EARS_SOFT, Symmetry::Mirror, disp, Vec::new())
}

/// `semantic.earlobe-size`: the lowest part of the ear (front and back) scaled about the lobe's
/// top, fading in over 8 mm of height.
fn earlobes(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let ear = s.region("ear-L")?;
    let c = s.landmark("ear.L.lobule")? + Vec3::Y * 0.006;
    let f = move |pt: &Point| {
        if !within(s, pt, ear) {
            return Vec3::ZERO;
        }
        (pt.p - c) * (0.12 * smoothstep(c.y + 0.004, c.y - 0.004, pt.p.y))
    };
    let disp = s.eval(|pt| s.sym(pt, &f));
    Raw::confined(s, EARS, Symmetry::Mirror, disp, Vec::new())
}

/// `semantic.neck-width`: the neck outward all around.
pub fn neck_width(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let mut t = s.region_amounts(&[("neck", 1.0)])?;
    // A long ramp into the shoulders and the jaw: the sex axis uses this field at several
    // millimetres.
    s.smooth(&mut t, 60);
    // Strongly smoothed normals (the raw normal turns sharply in the suprasternal notch and at
    // the neck-chin angle), horizontal part only: the crease under the jaw angle, inside the ramp
    // with its normals converging downward, would otherwise move down into the neck and fold
    // (192 turned faces with the sex axis at 2, todo 1.3).
    let normals = s.smooth_normals(40);
    let disp = s.eval(|pt| {
        let n = normals.get(pt.node).copied().unwrap_or(pt.n);
        Vec3::new(n.x, 0.0, n.z) * (0.0015 * at(&t, pt))
    });
    Raw::confined(s, NECK, Symmetry::Mirror, disp, Vec::new())
}

/// `semantic.neck-length`: the neck bones longer; the head and everything on it rises.
fn neck_length(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let (s, b) = (kit.space, &kit.bones);
    let mut st = Stretch::new(b.len());
    for n in ["neck.01", "neck.02"] {
        st.lengthen(b, s.joint_index(n)?, 0.045)?;
    }
    let (disp, joints) = skeletal(s, b, &st, false)?;
    Raw::confined(s, EVERYWHERE, Symmetry::Mirror, disp, joints)
}

/// A larynx prominence of `a` metres at the front of the neck.
pub fn larynx(kit: &Kit<'_, '_>, a: f32) -> IdResult<Raw> {
    let s = kit.space;
    let c = s.landmark("neck.front")?;
    let disp = s.eval(|pt| pt.n * (a * bump(pt.p.distance(c), 0.018)));
    Raw::confined(s, NECK, Symmetry::Mirror, disp, Vec::new())
}

/// `semantic.eyeball-size`: the eyeballs scaled about the eye centre; lids pushed with them.
fn eyeballs(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let eye = s.joint("L.eye")?;
    let ball = s.part_nodes("eyeball-L")?;
    let pos = s.field().positions();
    let r = ball.iter().filter_map(|&n| pos.get(n as usize)).map(|p| p.distance(eye)).fold(0.0, f32::max);
    let region = s.region("eye-L")?;
    let f = move |pt: &Point| {
        if !within(s, pt, region) {
            return Vec3::ZERO;
        }
        (pt.p - eye) * (0.04 * falloff(pt.p.distance(eye), r * 1.02, r + 0.008))
    };
    let disp = s.eval(|pt| s.sym(pt, &f));
    Raw::confined(s, EYES, Symmetry::Mirror, disp, Vec::new())
}

/// Which teeth a teeth component changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tooth {
    /// Central and lateral incisors: larger about their root end.
    Incisor,
    /// Canines: longer.
    Canine,
    /// Premolars and molars: broader.
    Molar,
}

/// One tooth: its nodes, centroid and root-end height.
struct ToothInfo {
    nodes: Vec<u32>,
    centre: Vec3,
    root_y: f32,
    rank: usize,
}

/// Splits a teeth part into teeth (connected node sets) and ranks them per side from the front
/// (0 = central incisor).
fn split_teeth(s: &Space<'_>, part: &str, upper: bool) -> IdResult<Vec<ToothInfo>> {
    let nodes = s.part_nodes(part)?;
    let pos = s.field().positions();
    let mut seen = vec![false; s.node_count()];
    let mut teeth = Vec::new();
    for &start in &nodes {
        if seen.get(start as usize).copied().unwrap_or(true) {
            continue;
        }
        let mut stack = vec![start];
        let mut group = Vec::new();
        if let Some(f) = seen.get_mut(start as usize) {
            *f = true;
        }
        while let Some(n) = stack.pop() {
            group.push(n);
            for &m in s.field().neighbours(n as usize) {
                if let Some(f) = seen.get_mut(m as usize)
                    && !*f
                {
                    *f = true;
                    stack.push(m);
                }
            }
        }
        let ps: Vec<Vec3> = group.iter().filter_map(|&n| pos.get(n as usize).copied()).collect();
        let centre = ps.iter().fold(Vec3::ZERO, |a, &p| a + p) * (1.0 / ps.len().max(1) as f32);
        let root_y = if upper {
            ps.iter().map(|p| p.y).fold(f32::MIN, f32::max)
        } else {
            ps.iter().map(|p| p.y).fold(f32::MAX, f32::min)
        };
        teeth.push(ToothInfo { nodes: group, centre, root_y, rank: 0 });
    }
    for left in [true, false] {
        let mut side: Vec<usize> = (0..teeth.len()).filter(|&i| (teeth[i].centre.x >= 0.0) == left).collect();
        side.sort_by(|&a, &b| teeth[b].centre.z.total_cmp(&teeth[a].centre.z));
        for (rank, i) in side.into_iter().enumerate() {
            teeth[i].rank = rank;
        }
    }
    Ok(teeth)
}

/// `semantic.teeth-*`: incisors larger, canines longer, or premolars and molars broader.
fn teeth(kit: &Kit<'_, '_>, which: Tooth) -> IdResult<Raw> {
    let s = kit.space;
    let pos = s.field().positions();
    let mut disp = vec![Vec3::ZERO; s.node_count()];
    for (part, upper) in [("teeth-upper", true), ("teeth-lower", false)] {
        for t in split_teeth(s, part, upper)? {
            let selected = match which {
                Tooth::Incisor => t.rank <= 1,
                Tooth::Canine => t.rank == 2,
                Tooth::Molar => t.rank >= 3,
            };
            if !selected {
                continue;
            }
            let root = Vec3::new(t.centre.x, t.root_y, t.centre.z);
            for &n in &t.nodes {
                let (Some(&p), Some(d)) = (pos.get(n as usize), disp.get_mut(n as usize)) else { continue };
                *d = match which {
                    Tooth::Incisor => (p - root) * 0.05,
                    Tooth::Canine => Vec3::Y * (0.08 * (p.y - t.root_y)),
                    Tooth::Molar => Vec3::new(p.x - t.centre.x, 0.0, p.z - t.centre.z) * 0.06,
                };
            }
        }
    }
    Raw::confined(s, TEETH, Symmetry::Mirror, disp, Vec::new())
}

/// `semantic.gum-line`: the gum margins toward the biting edges.
fn gums(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let pos = s.field().positions();
    let mut disp = vec![Vec3::ZERO; s.node_count()];
    for (part, upper) in [("gums-upper", true), ("gums-lower", false)] {
        let nodes = s.part_nodes(part)?;
        let ys: Vec<f32> = nodes.iter().filter_map(|&n| pos.get(n as usize).map(|p| p.y)).collect();
        let lo = ys.iter().copied().fold(f32::MAX, f32::min);
        let hi = ys.iter().copied().fold(f32::MIN, f32::max);
        for &n in &nodes {
            let (Some(&p), Some(d)) = (pos.get(n as usize), disp.get_mut(n as usize)) else { continue };
            *d = if upper {
                Vec3::Y * (-0.0006 * (1.0 - smoothstep(lo, hi, p.y)))
            } else {
                Vec3::Y * (0.0006 * smoothstep(lo, hi, p.y))
            };
        }
    }
    Raw::confined(s, GUMS, Symmetry::Mirror, disp, Vec::new())
}

/// `asymmetry.nose`: the lower nose off centre to the left, the left ala a little wider.
fn asym_nose(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    // Both weights are Euclidean everywhere, not gated by the nose region as `around_solid` is:
    // the gate made them jump at the alar crease, where the cheek next to the ala is near in space
    // but far along the surface (a 51-degree crease at weight 2, `id_02`). The cheek beside the
    // ala now moves a little with it.
    let mut seeds = Vec::new();
    for n in ["nose.tip", "nose.subnasale", "nose.bridge.02", "L.nose.alare", "R.nose.alare"] {
        seeds.push(s.landmark(n)?);
    }
    let ala = [s.landmark("L.nose.alare")?, s.landmark("L.nose.alar-crease")?];
    let root = s.landmark("nose.root")?.y;
    let tip = s.landmark("nose.tip")?.y;
    let nearest = |points: &[Vec3], p: Vec3| points.iter().map(|q| q.distance(p)).fold(f32::INFINITY, f32::min);
    let disp = s.eval(|pt| {
        let t = smoothstep(root, tip, pt.p.y);
        let w = falloff(nearest(&seeds, pt.p), 0.006, 0.02);
        let flare = if pt.p.x > 0.0 { 0.03 * pt.p.x * falloff(nearest(&ala, pt.p), 0.003, 0.01) } else { 0.0 };
        Vec3::X * (0.0008 * t * w + flare)
    });
    Raw::confined(s, NOSE, Symmetry::Free, disp, Vec::new())
}

/// `asymmetry.eyes`: the left eye up, the right eye down a little.
fn asym_eyes(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    eye_shift(kit, Vec3::Y * 0.0006, 1.0, -0.5)
}

/// `asymmetry.mouth`: the left mouth corner up.
fn asym_mouth(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    // Euclidean: the corner's outside and the fold into the mouth move together.
    let c = s.landmark("L.lip.corner")?;
    let disp = s.eval(|pt| Vec3::Y * (0.0007 * falloff(pt.p.distance(c), 0.008, 0.02)));
    Raw::confined(s, MOUTH, Symmetry::Free, disp, Vec::new())
}

/// `asymmetry.jaw`: the chin off centre to the left.
fn asym_jaw(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let w = s.around(&CHIN_SEEDS, 0.01, 0.04)?;
    let disp = s.eval(|pt| Vec3::X * (0.001 * at(&w, pt)));
    Raw::confined(s, JAW, Symmetry::Free, disp, Vec::new())
}

/// `asymmetry.ears`: the left ear up, the right ear smaller.
fn asym_ears(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let left = s.region("ear-L")?;
    let right = s.region("ear-R")?;
    let t = s.landmark("ear.R.tragion")?;
    let disp = s.eval(|pt| {
        if within(s, pt, left) {
            Vec3::Y * 0.0015
        } else if within(s, pt, right) {
            (pt.p - t) * -0.02
        } else {
            Vec3::ZERO
        }
    });
    Raw::confined(s, EARS, Symmetry::Free, disp, Vec::new())
}

/// `asymmetry.brow`: the left brow up.
fn asym_brow(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let w = around(s, &series(s, "L.brow")?, 0.004, 0.014)?;
    let up = above_eyes(s)?;
    let disp = s.eval(|pt| Vec3::Y * (0.001 * at(&w, pt) * up(pt.p)));
    Raw::confined(s, BROWS, Symmetry::Free, disp, Vec::new())
}

/// `asymmetry.global`: half of every regional asymmetry plus the left shoulder 4 mm higher
/// (clavicle end raised by 3 % of its length).
fn asym_global(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let (s, b) = (kit.space, &kit.bones);
    let mut raw = Raw::zero(s);
    for part in [asym_nose(kit)?, asym_eyes(kit)?, asym_mouth(kit)?, asym_jaw(kit)?, asym_ears(kit)?, asym_brow(kit)?] {
        raw.add(&part, 0.5);
    }
    raw.add(&cheekbones_left(kit, 0.001)?, 0.5);
    let mut st = Stretch::new(b.len());
    st.shear(b, s.joint_index("L.clavicle")?, Vec3::Y, 0.03)?;
    let (disp, joints) = skeletal(s, b, &st, false)?;
    raw.add(&Raw::confined(s, EVERYWHERE, Symmetry::Free, disp, joints)?, 1.0);
    Ok(raw)
}
