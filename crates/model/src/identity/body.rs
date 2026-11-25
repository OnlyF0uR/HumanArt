//! Global identity axes `[ID-03]`, body proportions, hands and feet `[ID-01]` (c).
//!
//! Each recipe has an authored **shape** (where and in which direction the surface moves). Its
//! **amount** per unit weight is either measured, one standard deviation of an ANSUR II variable
//! ([`super::measure`], "SD" below), or an authored model choice read off renders, not a statistic
//! `[ID-07]` ("authored" below):
//!
//! | component | shape | amount per unit weight |
//! |---|---|---|
//! | `global.height` | trunk, arm and leg bones longer, clavicles half as much, neck bones 0.4x as much, hands and feet 0.8x, head 0.32x as much; soles stay on the floor | SD of `stature` |
//! | `global.mass` | soft-tissue thickness along the normal by region (abdomen 10 : pelvis 9 : thigh 8 : chest, upper arm 6 : back 5 : ...), plus belly, buttocks and under the chin | expected waist, chest, buttock and thigh circumference change per SD of `weightkg` at fixed stature |
//! | `global.muscularity` | muscle-group bumps of 1 to 5 mm (deltoid, arms, chest, trapezius, back, thighs, buttocks, calves, neck, masseter) | authored |
//! | `global.sex` | within a sex, toward typically male shape: shoulders, hips, waist, neck; breasts −6 mm, chest muscle +2 mm, buttocks −4 mm, hip flanks −3 mm, brow ridge, jaw, chin, larynx, lips | shoulders, hips, waist and neck: one within-sex SD (RMS of the two sexes) in the male direction; the rest authored |
//! | `global.age` | toward older: jowls, cheek descent, longer nose and ears, thinner lips, lower brows, belly +3 mm, breast and buttock descent | authored |
//! | `semantic.arm-length` | upper arm and forearm longer | SD of `acromionradialelength` + `radialestylionlength` |
//! | `semantic.leg-length` | thigh and shin longer, everything above rises | SD of `crotchheight` |
//! | `semantic.shoulder-width` | clavicles longer, outer end raised by a quarter as much | SD of `biacromialbreadth` |
//! | `semantic.hip-width` | lateral scaling, full from the upper thigh to just above the trochanter, fading out toward the waist and the knee | SD of `hipbreadth` |
//! | `semantic.waist-width` | lateral (half as much front-back) scaling around the natural waist | SD of `waistbreadth` |
//! | `semantic.thigh-fullness` | soft tissue along the normal from the gluteal fold to two thirds down the thigh, fading at the knee | SD of `thighcircumference` |
//! | `semantic.glute-size` | a rounded buttock mound, back and a little out, its lower pole round into the gluteal fold | SD of `buttockdepth` |
//! | `semantic.hip-flank` | 8 mm over the upper outer thigh | authored |
//! | `semantic.bust-size` | a rounded breast mound of 12 mm on the pectoral, full lower pole over the inframammary fold (on top of the female breast of `identity.sex`; −2 leaves it 6 mm) | authored (no breast variable in ANSUR II) |
//! | `semantic.breast-projection` | the central part of the mound 6 mm forward | authored |
//! | `semantic.breast-ptosis` | the lower pole 8 mm down and the upper pole 3 mm flatter (negative: lifted) | authored |
//! | `semantic.breast-spacing` | the mounds 6 mm apart (negative: closer together) | authored |
//! | `semantic.nipple-size` | the nipple (region `nipple-L` / `-R`) 15 % larger about its base | authored |
//! | `semantic.areola-size` | the areola (region `areola-L` / `-R`) spread so its rim moves 2 mm outward, 0.4 mm puffier | authored |
//! | `semantic.palm-width` | across the knuckles, fading out toward the wrist | SD of `handbreadth` |
//! | `semantic.finger-length` | finger and thumb bones +5 % | authored |
//! | `semantic.finger-ratio` | ring finger middle and end bones +4 %, index finger ones −4 % (the knuckle webs stay) | authored |
//! | `semantic.knuckle-prominence` | a 1.2 mm bump centred on each knuckle and middle finger joint (inside the finger), back of the hand only, about 0.5 mm on the skin | authored |
//! | `semantic.foot-length` | foot and toe bones longer (horizontally) | SD of `footlength` |
//! | `semantic.foot-width` | across the ball and toes, fading out toward the heel | SD of `footbreadthhorizontal` (measured between the first and fifth metatarsal heads) |
//!
//! Nipple and areola size act on the template's nipple and areola regions (ADR-039); every other
//! component carries the nipples and the navel rigidly with the skin around them
//! ([`Space::carry_details`](super::space::Space::carry_details)). The sex axis never touches genital parts: ADR-039 makes them a
//! discrete variant; every body recipe here moves the `body` skin only.

use humanart_math::Vec3;

use super::error::{IdResult, IdentityError};
use super::face;
use super::measure::{
    self, BIACROMIAL, BIACROMIAL_GAUGE, Gauge, HIP_BREADTH, HIP_BREADTH_GAUGE, NECK_CIRC, NECK_GIRTH, WAIST_BREADTH,
    WAIST_BREADTH_GAUGE,
};
use super::skeletal::{Stretch, skeletal};
use super::space::{
    EVERYWHERE, IdentityComponent, PartKind, Point, Reach, Space, Symmetry, bump, bump_along, smoothstep,
};
use super::{Kit, Raw};

/// Height: fraction per unit weight of trunk, neck, arm and leg bone length (authored).
pub const HEIGHT_BONE: f32 = 0.025;
/// Height: uniform scale of the hands and horizontal scale of the feet (authored).
pub const HEIGHT_EXTREMITY: f32 = 0.02;
/// Height: uniform scale of the head (authored).
pub const HEIGHT_HEAD: f32 = 0.008;

/// Soft tissue everywhere except the eyes and the inside of the mouth.
pub const SOFT_TISSUE: Reach<'static> = Reach {
    regions: &["body"],
    except: &[
        "eye-L",
        "eye-R",
        "mouth-interior",
        "teeth-upper",
        "teeth-lower",
        "gums-upper",
        "gums-lower",
        "tongue",
        "foot-L.toes",
        "foot-R.toes",
        "hand-L.thumb",
        "hand-R.thumb",
        "hand-L.index",
        "hand-R.index",
        "hand-L.middle",
        "hand-R.middle",
        "hand-L.ring",
        "hand-R.ring",
        "hand-L.little",
        "hand-R.little",
    ],
    margin: 0.004,
};
/// Hips and thighs.
pub const HIPS: Reach<'static> = Reach { regions: &["pelvis", "abdomen", "back", "thigh"], except: &[], margin: 0.03 };
/// The trunk below the shoulders.
pub const WAIST: Reach<'static> = Reach { regions: &["chest", "abdomen", "back", "pelvis"], except: &[], margin: 0.03 };
/// The front of the chest.
pub const BUST: Reach<'static> = Reach { regions: &["chest", "abdomen"], except: &[], margin: 0.02 };
/// Buttocks and the backs of the thighs.
pub const GLUTES: Reach<'static> = Reach { regions: &["pelvis", "back", "thigh"], except: &[], margin: 0.03 };
/// Jacobi smoothing steps applied to the body-mass, sex, torso, breast, buttock and thigh fields
/// after they are evaluated, on [`TRUNK_AND_LIMBS`]: their steepest transitions (the inframammary
/// fold, the buttock cleft and fold, the armpit damping, the midline) then span several rings of
/// the template's 1 to 1.5 cm trunk mesh instead of a one-ring cliff that shades as a stair step
/// (measured by the `id_02_body_shapes_have_no_stepped_falloff` test).
pub const SHAPE_SMOOTHING: usize = 6;
/// The regions [`SHAPE_SMOOTHING`] works on (not the face, hands or feet, whose detail is finer).
pub const TRUNK_AND_LIMBS: Reach<'static> =
    Reach { regions: &["torso", "neck", "thigh", "upper-arm", "forearm", "shin"], except: &[], margin: 0.0 };
/// Thighs and the hips above them.
pub const THIGHS: Reach<'static> = Reach { regions: &["thigh", "pelvis"], except: &[], margin: 0.03 };
/// Hands.
pub const HANDS: Reach<'static> = Reach { regions: &["hand"], except: &[], margin: 0.01 };
/// Feet.
pub const FEET: Reach<'static> = Reach { regions: &["foot"], except: &[], margin: 0.01 };

const SIDES: [&str; 2] = ["L.", "R."];
const FINGERS: [&str; 5] = ["thumb", "index", "middle", "ring", "little"];

/// The component bound to `name`, if it is one of this module's.
pub fn recipe(kit: &Kit<'_, '_>, name: &str) -> IdResult<Option<IdentityComponent>> {
    let s = kit.space;
    let m = Symmetry::Mirror;
    let raw = match name {
        "global.height" => height(kit)?,
        "global.mass" => mass(kit)?,
        "global.muscularity" => muscularity(kit)?,
        "global.sex" => sex(kit)?,
        "global.age" => age(kit)?,
        "semantic.arm-length" => {
            limbs(kit, &["upper-arm", "upper-arm.twist", "forearm", "forearm.twist"], 0.04, false)?
        }
        "semantic.leg-length" => limbs(kit, &["thigh", "shin"], 0.035, true)?,
        "semantic.shoulder-width" => shoulders(kit, 0.08, 0.02)?,
        "semantic.hip-width" => hips(kit, 0.06)?,
        "semantic.palm-width" => palm_width(kit, 0.08)?,
        "semantic.finger-length" => fingers(kit, &[(&FINGERS, 0.05)], &["01", "02", "03"])?,
        "semantic.finger-ratio" => fingers(kit, &[(&["ring"], 0.04), (&["index"], -0.04)], &["02", "03"])?,
        "semantic.knuckle-prominence" => knuckles(kit, 0.0012)?,
        "semantic.foot-length" => foot_length(kit, 0.04)?,
        "semantic.foot-width" => foot_width(kit, 0.06)?,
        "semantic.waist-width" => waist(kit, 0.06)?,
        "semantic.thigh-fullness" => thighs(kit, 0.008)?,
        "semantic.glute-size" => glutes(kit, 0.012)?,
        "semantic.hip-flank" => flanks(kit, 0.008)?,
        "semantic.bust-size" => breasts(kit, 0.012)?,
        "semantic.breast-projection" => breast_shape(kit, BreastShape::Projection, 0.006)?,
        "semantic.breast-ptosis" => breast_shape(kit, BreastShape::Ptosis, 0.008)?,
        "semantic.breast-spacing" => breast_shape(kit, BreastShape::Spacing, 0.006)?,
        "semantic.nipple-size" => nipples(kit, 0.15)?,
        "semantic.areola-size" => areolae(kit, 0.002)?,
        _ => return Ok(None),
    };
    let raw = measure::calibrate(s, name, raw)?;
    Ok(Some(raw.component(s, name, m)?))
}

fn joint(s: &Space<'_>, name: &str) -> IdResult<u16> {
    s.joint_index(name)
}

/// A skeletal field from a stretch built by `f`, over the whole template.
pub(crate) fn skeleton_raw(
    kit: &Kit<'_, '_>,
    ground: bool,
    f: impl FnOnce(&mut Stretch) -> IdResult<()>,
) -> IdResult<Raw> {
    let mut st = Stretch::new(kit.bones.len());
    f(&mut st)?;
    let (disp, joints) = skeletal(kit.space, &kit.bones, &st, ground)?;
    Raw::confined(kit.space, EVERYWHERE, Symmetry::Mirror, disp, joints)
}

/// `global.height`: longer bones, a slightly larger head, hands and feet; the soles stay put.
pub fn height(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let (s, b) = (kit.space, &kit.bones);
    skeleton_raw(kit, true, |st| {
        for n in ["pelvis", "spine.01", "spine.02", "spine.03", "spine.04"] {
            st.lengthen(b, joint(s, n)?, HEIGHT_BONE)?;
        }
        // The neck takes a smaller share of stature variation (and its skin folds under
        // compression together with `semantic.neck-length`).
        for n in ["neck.01", "neck.02"] {
            st.lengthen(b, joint(s, n)?, HEIGHT_BONE * 0.4)?;
        }
        for side in SIDES {
            for n in ["upper-arm", "upper-arm.twist", "forearm", "forearm.twist", "thigh", "shin"] {
                st.lengthen(b, joint(s, &format!("{side}{n}"))?, HEIGHT_BONE)?;
            }
            st.lengthen(b, joint(s, &format!("{side}clavicle"))?, HEIGHT_BONE * 0.5)?;
            st.scale_subtree(b, joint(s, &format!("{side}hand"))?, HEIGHT_EXTREMITY)?;
            for n in ["foot", "toes"] {
                st.lengthen_horizontal(b, joint(s, &format!("{side}{n}"))?, HEIGHT_EXTREMITY)?;
            }
        }
        st.scale_subtree(b, joint(s, "head")?, HEIGHT_HEAD)
    })
}

/// Lengthens the bones `names` (both sides) by `k`; with `ground` the soles stay on the floor.
pub fn limbs(kit: &Kit<'_, '_>, names: &[&str], k: f32, ground: bool) -> IdResult<Raw> {
    let (s, b) = (kit.space, &kit.bones);
    skeleton_raw(kit, ground, |st| {
        for side in SIDES {
            for n in names {
                st.lengthen(b, joint(s, &format!("{side}{n}"))?, k)?;
            }
        }
        Ok(())
    })
}

/// Lengthens the clavicles by `k` and raises their outer end by `lift` times their length.
pub fn shoulders(kit: &Kit<'_, '_>, k: f32, lift: f32) -> IdResult<Raw> {
    let (s, b) = (kit.space, &kit.bones);
    skeleton_raw(kit, false, |st| {
        for side in SIDES {
            let c = joint(s, &format!("{side}clavicle"))?;
            st.lengthen(b, c, k)?;
            st.shear(b, c, Vec3::Y, lift)?;
        }
        Ok(())
    })
}

/// Lengthens the bones `segments` of the fingers of `groups` (finger names, both hands) by their
/// fractions.
pub fn fingers(kit: &Kit<'_, '_>, groups: &[(&[&str], f32)], segments: &[&str]) -> IdResult<Raw> {
    let (s, b) = (kit.space, &kit.bones);
    let mut st = Stretch::new(b.len());
    for &(names, k) in groups {
        for side in SIDES {
            for f in names {
                for seg in segments {
                    st.lengthen(b, joint(s, &format!("{side}{f}.{seg}"))?, k)?;
                }
            }
        }
    }
    let (disp, joints) = skeletal(s, b, &st, false)?;
    Raw::confined(s, HANDS, Symmetry::Mirror, disp, joints)
}

/// `semantic.foot-length`: foot and toe bones `k` longer, horizontally.
pub fn foot_length(kit: &Kit<'_, '_>, k: f32) -> IdResult<Raw> {
    let (s, b) = (kit.space, &kit.bones);
    let mut st = Stretch::new(b.len());
    for side in SIDES {
        for n in ["foot", "toes"] {
            st.lengthen_horizontal(b, joint(s, &format!("{side}{n}"))?, k)?;
        }
    }
    let (disp, joints) = skeletal(s, b, &st, false)?;
    Raw::confined(s, FEET, Symmetry::Mirror, disp, joints)
}

/// The lateral hip scaling `k x band(y)`: full from the upper thigh to just above the trochanter
/// (so the hips flow into the thighs instead of bulging at the iliac crest), fading out toward the
/// waist and toward the knee. Evaluated for both the skin and the hip joints.
pub fn hips(kit: &Kit<'_, '_>, k: f32) -> IdResult<Raw> {
    let s = kit.space;
    let y_t = s.landmark("L.hip.trochanter")?.y;
    let y_n = s.landmark("torso.navel")?.y;
    let y_k = s.landmark("L.knee.center")?.y;
    let thigh = y_t - y_k;
    let band = move |y: f32| {
        smoothstep(y_k + 0.1 * thigh, y_t - 0.3 * thigh, y) * (1.0 - smoothstep(y_t + 0.05 * thigh, y_n, y))
    };
    let f = move |p: Vec3| Vec3::new(k * p.x * band(p.y), 0.0, 0.0);
    let disp = s.eval(|pt| f(pt.p));
    let mut joints = Vec::new();
    for n in ["L.thigh", "R.thigh"] {
        let j = s.joint_index(n)?;
        joints.push((j, f(s.joint(n)?)));
    }
    Raw::confined(s, HIPS, Symmetry::Mirror, disp, joints)
}

/// Lateral (and some front-back) scaling `k` around the natural waist, between the lowest ribs
/// and the iliac crest.
pub fn waist(kit: &Kit<'_, '_>, k: f32) -> IdResult<Raw> {
    let s = kit.space;
    let navel = s.landmark("torso.navel")?;
    let xiphoid = s.landmark("torso.xiphoid")?;
    let l3 = s.landmark("spine.l3")?;
    let y_w = navel.y + (xiphoid.y - navel.y) * 0.4;
    let h = (xiphoid.y - navel.y) * 0.6 + 0.04;
    let z_c = f32::midpoint(navel.z, l3.z);
    let disp = s.eval(|pt| {
        let w = bump((pt.p.y - y_w).abs(), h);
        Vec3::new(k * pt.p.x * w, 0.0, 0.5 * k * (pt.p.z - z_c) * w)
    });
    Raw::confined(s, WAIST, Symmetry::Mirror, disp, Vec::new())
}

/// A soft mound on the left side of the trunk (mirrored through [`Space::sym`]): an elliptical
/// footprint around `c` in the x-y plane with separate lateral / medial and upper / lower radii, a
/// height profile `(1 - r^2)^e` whose exponent runs from `e_up` over the upper half (a long slope
/// into the trunk) to `e_down` over the lower half (a full, round lower pole that turns sharply into
/// its fold), and a depth window that keeps the far side of the trunk out.
#[derive(Clone, Copy, Debug)]
struct Mound {
    c: Vec3,
    rx_out: f32,
    rx_in: f32,
    ry_up: f32,
    ry_down: f32,
    e_up: f32,
    e_down: f32,
    /// +1 for a mound on the front (nodes in front of `c.z - depth` count), −1 on the back.
    facing: f32,
    depth: f32,
}

impl Mound {
    /// `(profile, v)`: the height profile in `[0, 1]` and the vertical footprint coordinate in
    /// `[-1, 1]` (negative below the centre).
    fn at(&self, p: Vec3) -> (f32, f32) {
        let d = p - self.c;
        let rx = if d.x >= 0.0 { self.rx_out } else { self.rx_in };
        let ry = if d.y >= 0.0 { self.ry_up } else { self.ry_down };
        let (u, v) = (d.x / rx.max(1.0e-3), d.y / ry.max(1.0e-3));
        let r2 = u * u + v * v;
        if r2 >= 1.0 {
            return (0.0, v.clamp(-1.0, 1.0));
        }
        let e = self.e_down + (self.e_up - self.e_down) * smoothstep(-0.4, 0.4, v);
        // The steep lower pole is rounded off over the outer 40 % of the radius, so the mound
        // meets the trunk over several rings of the mesh instead of a cliff one ring wide.
        let rim = smoothstep(1.0, 0.6, r2.sqrt());
        let window = smoothstep(-self.depth, -self.depth * 0.5, self.facing * d.z);
        ((1.0 - r2).powf(e) * rim * window, v)
    }

    /// The same mound with its radii scaled by `k` (a core) and new exponents.
    fn core(&self, k: f32, e_up: f32, e_down: f32) -> Self {
        Self {
            rx_out: self.rx_out * k,
            rx_in: self.rx_in * k,
            ry_up: self.ry_up * k,
            ry_down: self.ry_down * k,
            e_up,
            e_down,
            ..*self
        }
    }
}

/// The left breast mound: centred a little below the nipple landmark; the lower rim (the
/// inframammary fold) 30 % of the xiphoid-to-suprasternal height below the xiphoid level, the upper
/// rim 80 % of the way up to the notch; medially to near the sternum, laterally toward the
/// armpit.
fn breast_mound(s: &Space<'_>) -> IdResult<Mound> {
    let nipple = s.landmark("L.chest.nipple")?;
    let suprasternal = s.landmark("torso.suprasternal")?;
    let xiphoid = s.landmark("torso.xiphoid")?;
    let chest = (suprasternal.y - xiphoid.y).abs().max(0.05);
    let c = nipple - Vec3::Y * (chest * 0.08);
    let fold = xiphoid.y - chest * 0.3;
    Ok(Mound {
        c,
        rx_out: chest * 0.48,
        rx_in: (c.x - chest * 0.1).max(chest * 0.2),
        ry_up: (suprasternal.y - c.y) * 0.8,
        ry_down: (c.y - fold).max(chest * 0.2),
        // A fuller, rounder upper pole than 1.6, which sloped in concavely to a point (lower
        // exponents fold the lateral lower edge with the sex shape at +2).
        e_up: 1.2,
        e_down: 0.8,
        facing: 1.0,
        depth: chest * 0.5,
    })
}

/// Breast volume of `a` metres (per unit weight) on the pectoral: a rounded mound, forward and a
/// little outward, the lower pole hanging slightly over the inframammary fold. The armpit folds
/// keep their depth.
pub fn breasts(kit: &Kit<'_, '_>, a: f32) -> IdResult<Raw> {
    let s = kit.space;
    let m = breast_mound(s)?;
    let axilla = axilla_damping(s)?;
    let f = move |pt: &Point| {
        let (h, v) = m.at(pt.p);
        let lower = smoothstep(0.2, -0.8, v);
        (Vec3::Z + Vec3::X * 0.15 - Vec3::Y * (0.25 * lower)).normalize_or_zero() * (a * h * axilla(pt.p))
    };
    let mut disp = s.eval(|pt| s.sym(pt, &f));
    s.smooth_field(&mut disp, SHAPE_SMOOTHING, &TRUNK_AND_LIMBS)?;
    Raw::confined(s, BUST, Symmetry::Mirror, disp, Vec::new())
}

/// Which breast shape term [`breast_shape`] builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BreastShape {
    /// The central part of the mound forward.
    Projection,
    /// The lower pole down and the upper pole flatter (ptosis); negative weights lift.
    Ptosis,
    /// The mounds apart; negative weights bring them together.
    Spacing,
}

/// A breast shape term of `a` metres per unit weight on the footprint of [`breasts`].
pub fn breast_shape(kit: &Kit<'_, '_>, shape: BreastShape, a: f32) -> IdResult<Raw> {
    let s = kit.space;
    let m = breast_mound(s)?;
    let core = m.core(0.8, 2.0, 1.6);
    let axilla = axilla_damping(s)?;
    let f = move |pt: &Point| {
        let (h, v) = m.at(pt.p);
        let d = match shape {
            BreastShape::Projection => Vec3::Z * core.at(pt.p).0,
            BreastShape::Ptosis => {
                let upper = smoothstep(-0.2, 0.6, v);
                -Vec3::Y * h * (1.0 - 0.5 * upper) - Vec3::Z * (0.4 * h * upper)
            }
            BreastShape::Spacing => Vec3::X * h,
        };
        d * (a * axilla(pt.p))
    };
    let mut disp = s.eval(|pt| s.sym(pt, &f));
    s.smooth_field(&mut disp, SHAPE_SMOOTHING, &TRUNK_AND_LIMBS)?;
    Raw::confined(s, BUST, Symmetry::Mirror, disp, Vec::new())
}

/// The left nipple and areola of the template (ADR-039 regions `areola-L` and its child
/// `nipple-L`): areola centre, unit outward axis, areola radius, nipple base centre (the mean of the
/// nipple nodes next to the areola) and nipple radius about it, all measured on the rest mesh.
fn nipple_frame(s: &Space<'_>) -> IdResult<(Vec3, Vec3, f32, Vec3, f32)> {
    let field = s.field();
    let (areola, nipple) = (s.region("areola-L")?, s.region("nipple-L")?);
    let pos = field.positions();
    let normals = field.normals();
    let skin = |n: usize| s.kind_of(n) == PartKind::Body;
    let ring: Vec<usize> =
        (0..s.node_count()).filter(|&n| skin(n) && field.in_region(n, areola) && !field.in_region(n, nipple)).collect();
    let tip: Vec<usize> = (0..s.node_count()).filter(|&n| skin(n) && field.in_region(n, nipple)).collect();
    if ring.is_empty() || tip.is_empty() {
        return Err(IdentityError::missing("region nodes", "areola-L / nipple-L"));
    }
    let mean = |nodes: &[usize], v: &[Vec3]| {
        nodes.iter().filter_map(|&n| v.get(n)).fold(Vec3::ZERO, |a, &b| a + b) / nodes.len() as f32
    };
    let c = mean(&ring, pos);
    let axis = mean(&ring, normals).normalize_or_zero();
    let across = |p: Vec3| (p - c).reject_from(axis).length();
    let r_a = ring.iter().filter_map(|&n| pos.get(n)).map(|&p| across(p)).fold(0.0, f32::max).max(1.0e-3);
    let base_nodes: Vec<usize> = tip
        .iter()
        .copied()
        .filter(|&n| field.neighbours(n).iter().any(|&m| !field.in_region(m as usize, nipple)))
        .collect();
    let base = if base_nodes.is_empty() { c } else { mean(&base_nodes, pos) };
    let r_n = tip
        .iter()
        .filter_map(|&n| pos.get(n))
        .map(|&p| (p - base).reject_from(axis).length())
        .fold(0.0, f32::max)
        .max(1.0e-3);
    Ok((c, axis, r_a, base, r_n))
}

/// `semantic.nipple-size`: the nipple (region `nipple-L` / `-R`) scaled by `k` per unit weight
/// about its base, the areola skin around it following over one more nipple radius.
pub fn nipples(kit: &Kit<'_, '_>, k: f32) -> IdResult<Raw> {
    let s = kit.space;
    let (_, axis, _, base, r_n) = nipple_frame(s)?;
    let f = move |pt: &Point| {
        if pt.p.x <= 0.0 {
            return Vec3::ZERO;
        }
        let across = (pt.p - base).reject_from(axis).length();
        (pt.p - base) * (k * smoothstep(2.0 * r_n, r_n, across))
    };
    let disp = s.eval(|pt| s.sym(pt, &f));
    Raw::confined(s, BUST, Symmetry::Mirror, disp, Vec::new())
}

/// `semantic.areola-size`: the areola (region `areola-L` / `-R`) spread tangentially so its rim
/// moves `a` metres outward per unit weight, the skin around it following out to twice its radius,
/// and puffed by a fifth as much.
pub fn areolae(kit: &Kit<'_, '_>, a: f32) -> IdResult<Raw> {
    let s = kit.space;
    let (c, axis, r_a, _, _) = nipple_frame(s)?;
    let k = a / r_a;
    let f = move |pt: &Point| {
        if pt.p.x <= 0.0 {
            return Vec3::ZERO;
        }
        let t = (pt.p - c).reject_from(axis);
        let d = t.length();
        t * (k * smoothstep(2.0 * r_a, r_a, d)) + axis * (0.2 * a * smoothstep(1.2 * r_a, 0.6 * r_a, d))
    };
    let disp = s.eval(|pt| s.sym(pt, &f));
    Raw::confined(s, BUST, Symmetry::Mirror, disp, Vec::new())
}

/// The left buttock mound: centred a little above the trochanter, halfway between the midline and
/// the trochanter; medially to the cleft, laterally past the trochanter, up toward the iliac crest
/// and down to the gluteal fold.
fn glute_mound(s: &Space<'_>) -> IdResult<Mound> {
    let psis = s.landmark("L.hip.psis")?;
    let troch = s.landmark("L.hip.trochanter")?;
    let sacrum = s.landmark("spine.sacrum")?;
    let knee = s.landmark("L.knee.center")?;
    let thigh = (troch.y - knee.y).abs().max(0.1);
    let c = Vec3::new(troch.x * 0.5, troch.y + thigh * 0.02, sacrum.z);
    Ok(Mound {
        c,
        rx_out: troch.x * 0.75,
        rx_in: (c.x - 0.01).max(0.02),
        ry_up: (psis.y - c.y) + thigh * 0.12,
        ry_down: thigh * 0.34,
        e_up: 1.4,
        e_down: 0.9,
        facing: -1.0,
        depth: thigh * 0.3,
    })
}

/// Buttock volume of `a` metres (per unit weight): back and a little out, the lower pole round
/// into the gluteal fold.
pub fn glutes(kit: &Kit<'_, '_>, a: f32) -> IdResult<Raw> {
    let s = kit.space;
    let m = glute_mound(s)?;
    let f = move |pt: &Point| {
        let (h, v) = m.at(pt.p);
        let lower = smoothstep(0.2, -0.8, v);
        (-Vec3::Z + Vec3::X * 0.2 + Vec3::Y * (0.1 * lower)).normalize_or_zero() * (a * h)
    };
    let mut disp = s.eval(|pt| s.sym(pt, &f));
    s.smooth_field(&mut disp, SHAPE_SMOOTHING, &TRUNK_AND_LIMBS)?;
    Raw::confined(s, GLUTES, Symmetry::Mirror, disp, Vec::new())
}

/// Hip-flank fullness of `a` metres over the upper outer thigh, facing out.
pub fn flanks(kit: &Kit<'_, '_>, a: f32) -> IdResult<Raw> {
    let s = kit.space;
    let troch = s.landmark("L.hip.trochanter")?;
    let knee = s.landmark("L.knee.center")?;
    let axis = (troch - knee).normalize_or_zero();
    let len = troch.distance(knee);
    let c = troch - axis * (len * 0.25);
    let f = move |pt: &Point| {
        let w = bump_along(pt.p, c, axis, len * 0.45, troch.x * 0.6) * smoothstep(-0.2, 0.5, pt.n.x);
        pt.n * (a * w)
    };
    let mut disp = s.eval(|pt| s.sym(pt, &f));
    s.smooth_field(&mut disp, SHAPE_SMOOTHING, &TRUNK_AND_LIMBS)?;
    Raw::confined(s, HIPS, Symmetry::Mirror, disp, Vec::new())
}

/// Thigh fullness of `a` metres along the (smoothed) normal: full from the gluteal fold to half
/// way down the thigh, fading out at the knee and into the hip; the inner thigh takes 70 %, and
/// the crotch (within 2 to 7 cm of the midline below the pubis) and concave folds keep their shape.
pub fn thighs(kit: &Kit<'_, '_>, a: f32) -> IdResult<Raw> {
    let s = kit.space;
    let hip = s.joint("L.thigh")?;
    let knee = s.joint("L.shin")?;
    let axis = (knee - hip).normalize_or_zero();
    let len = hip.distance(knee).max(0.1);
    let pubis = s.landmark("torso.pubis")?;
    let normals = s.smooth_normals(40);
    let local = s.smooth_normals(8);
    let f = move |pt: &Point| {
        let t = (pt.p - hip).dot(axis) / len;
        let w = smoothstep(-0.05, 0.12, t) * (1.0 - smoothstep(0.5, 0.92, t));
        let n = normals.get(pt.node).copied().unwrap_or(pt.n);
        let inner = 0.7 + 0.3 * smoothstep(-0.6, 0.0, n.x);
        // Folds from lightly smoothed normals (the raw normal changes from one ring to the next).
        let fold = smoothstep(0.3, 0.8, local.get(pt.node).copied().unwrap_or(pt.n).dot(n));
        let crotch = smoothstep(0.02, 0.07, pt.p.x.abs()).max(smoothstep(pubis.y, pubis.y + 0.04, pt.p.y));
        n * (a * w * inner * fold * crotch)
    };
    let left = move |pt: &Point| if pt.p.x > 0.0 { f(pt) } else { Vec3::ZERO };
    let mut disp = s.eval(|pt| s.sym(pt, &left));
    s.smooth_field(&mut disp, SHAPE_SMOOTHING, &TRUNK_AND_LIMBS)?;
    Raw::confined(s, THIGHS, Symmetry::Mirror, disp, Vec::new())
}

/// A factor that keeps the armpit folds of the A-pose: 0.1 within 7 cm of a shoulder joint,
/// rising to 1 at 15 cm (the fold triangles turn over under a few millimetres of offset).
pub(crate) fn axilla_damping(s: &Space<'_>) -> IdResult<impl Fn(Vec3) -> f32 + use<>> {
    let joints = [s.joint("L.upper-arm")?, s.joint("R.upper-arm")?];
    Ok(move |p: Vec3| joints.iter().map(|&j| 0.1 + 0.9 * smoothstep(0.07, 0.15, p.distance(j))).fold(1.0, f32::min))
}

/// The smoothed per-region soft-tissue thickness of body mass (metres per unit weight).
fn mass_thickness(s: &Space<'_>) -> IdResult<Vec<f32>> {
    let mut t = s.region_amounts(&[
        ("body", 0.0),
        ("abdomen", 0.010),
        ("pelvis", 0.009),
        ("thigh", 0.008),
        ("chest", 0.006),
        ("back", 0.005),
        ("shoulder", 0.003),
        ("upper-arm", 0.006),
        ("forearm", 0.003),
        ("shin", 0.003),
        ("neck", 0.0025),
        ("hand", 0.0),
        ("foot", 0.0),
        ("mouth", 0.0),
        ("head", 0.0005),
        ("face", 0.0015),
        ("cheek-L", 0.003),
        ("cheek-R", 0.003),
        ("jaw-L", 0.003),
        ("jaw-R", 0.003),
        ("chin", 0.002),
        ("nose", 0.0),
        ("eye-L", 0.0),
        ("eye-R", 0.0),
        ("ear-L", 0.0),
        ("ear-R", 0.0),
    ])?;
    s.smooth(&mut t, 60);
    Ok(t)
}

/// `global.mass`: soft tissue along the normal, more on the trunk and thighs, scaled to one SD of
/// body weight through the circumferences ([`measure::calibrate_mass`]).
pub fn mass(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let t = mass_thickness(s)?;
    let normals = s.smooth_normals(40);
    let local = s.smooth_normals(8);
    let pubis = s.landmark("torso.pubis")?;
    let axilla = axilla_damping(s)?;
    let navel = s.landmark("torso.navel")?;
    let xiphoid = s.landmark("torso.xiphoid")?;
    let menton = s.landmark("chin.menton")?;
    let belly_c = navel + Vec3::Y * ((xiphoid.y - navel.y) * 0.15);
    let belly_r = (xiphoid.y - navel.y) * 0.9;
    let glute = glutes(kit, 0.004)?;
    // Keep the lip contact and the mouth corners free of normal offsets (they fold sharply).
    let mut lips = Vec::new();
    for prefix in ["lip.upper.inner", "lip.lower.inner"] {
        lips.extend(s.series_nodes(prefix)?);
    }
    lips.extend(s.landmark_nodes(&["L.lip.corner", "R.lip.corner"])?);
    let d_lips = s.geodesic(&lips, 0.012);
    // The ear's attachment crease turns over under a few millimetres of offset: keep the skin
    // within 1 cm of the lobule and tragion still, ramping to full at 3.5 cm.
    let ears = s.around(&["ear.L.lobule", "ear.R.lobule", "ear.L.tragion", "ear.R.tragion"], 0.01, 0.035)?;
    let disp = s.eval(|pt| {
        let free = smoothstep(0.003, 0.012, d_lips.get(pt.node).copied().unwrap_or(f32::INFINITY));
        let smooth_n = normals.get(pt.node).copied().unwrap_or(pt.n);
        // Folds (armpits, the crotch) keep their depth: where the surface turns away from the
        // smoothed normal, and between the thighs below the pubis.
        let fold = smoothstep(0.3, 0.8, local.get(pt.node).copied().unwrap_or(pt.n).dot(smooth_n));
        let crotch = 1.0
            - (1.0 - smoothstep(0.01, 0.06, pt.p.x.abs())) * (1.0 - smoothstep(pubis.y - 0.02, pubis.y + 0.04, pt.p.y));
        let ear = 1.0 - ears.get(pt.node).copied().unwrap_or(0.0);
        let base = t.get(pt.node).copied().unwrap_or(0.0) * free * fold * crotch * ear * axilla(pt.p);
        let belly = 0.006 * bump(pt.p.distance(belly_c), belly_r) * smoothstep(0.0, 0.6, pt.n.z);
        let chin =
            0.003 * bump(pt.p.distance(menton - Vec3::new(0.0, 0.012, 0.02)), 0.035) * smoothstep(-0.2, 0.4, -pt.n.y);
        smooth_n * (base + belly + chin)
    });
    let mut raw = Raw::confined(s, SOFT_TISSUE, Symmetry::Mirror, disp, Vec::new())?;
    raw.add(&glute, 1.0);
    s.smooth_field(&mut raw.disp, SHAPE_SMOOTHING, &TRUNK_AND_LIMBS)?;
    measure::calibrate_mass(s, raw)
}

/// `global.muscularity`: muscle-group bumps along the normal.
pub fn muscularity(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let mut raw = Raw::confined(s, SOFT_TISSUE, Symmetry::Mirror, muscle_field(kit, 1.0, 1.0)?, Vec::new())?;
    raw.add(&glutes(kit, 0.0025)?, 1.0);
    s.smooth_field(&mut raw.disp, SHAPE_SMOOTHING, &TRUNK_AND_LIMBS)?;
    Ok(raw)
}

/// The muscle bumps: `limbs` scales the arm, leg, back and neck groups, `chest` the pectorals.
pub(crate) fn muscle_field(kit: &Kit<'_, '_>, limbs: f32, chest: f32) -> IdResult<Vec<Vec3>> {
    let s = kit.space;
    let j = |n: &str| s.joint(n);
    let (shoulder, elbow, wrist) = (j("L.upper-arm")?, j("L.forearm")?, j("L.hand")?);
    let (hip, knee, ankle) = (j("L.thigh")?, j("L.shin")?, j("L.foot")?);
    let (neck, head) = (j("neck.01")?, j("head")?);
    let acromion = s.landmark("L.shoulder.acromion")?;
    let clav = s.landmark("L.clavicle.medial")?;
    let nipple = s.landmark("L.chest.nipple")?;
    let t8 = s.landmark("spine.t8")?;
    let gonion = s.landmark("jaw.L.01")?;
    let arm = (elbow - shoulder).normalize_or_zero();
    let fore = (wrist - elbow).normalize_or_zero();
    let thigh = (knee - hip).normalize_or_zero();
    let shin = (ankle - knee).normalize_or_zero();
    let (la, lf, lt, ls) = (shoulder.distance(elbow), elbow.distance(wrist), hip.distance(knee), knee.distance(ankle));
    let front = |n: Vec3| smoothstep(-0.1, 0.6, n.z);
    let back = |n: Vec3| smoothstep(-0.1, 0.6, -n.z);
    let pec_c = clav.lerp(nipple, 0.55);
    let axilla = axilla_damping(s)?;
    let trap_c = neck.lerp(acromion, 0.45) + Vec3::new(0.0, 0.02, -0.03);
    let lat_c = Vec3::new(nipple.x, t8.y - 0.04, f32::midpoint(t8.z, nipple.z) - 0.03);
    // Lightly smoothed normals: the raw normal changes from one ring to the next in folds.
    let normals = s.smooth_normals(8);
    let f = move |pt: &Point| {
        let (p, n) = (pt.p, normals.get(pt.node).copied().unwrap_or(pt.n));
        let mut a = 0.0;
        a += 0.004 * bump_along(p, shoulder.lerp(elbow, 0.12), arm, la * 0.45, 0.06);
        a += 0.0035 * bump_along(p, shoulder.lerp(elbow, 0.55), arm, la * 0.45, 0.055) * front(n);
        a += 0.003 * bump_along(p, shoulder.lerp(elbow, 0.5), arm, la * 0.5, 0.055) * back(n);
        a += 0.0025 * bump_along(p, elbow.lerp(wrist, 0.28), fore, lf * 0.4, 0.05);
        a += 0.0045 * bump_along(p, hip.lerp(knee, 0.5), thigh, lt * 0.45, 0.09) * front(n);
        a += 0.0025 * bump_along(p, hip.lerp(knee, 0.5), thigh, lt * 0.4, 0.09) * back(n);
        a += 0.0045 * bump_along(p, knee.lerp(ankle, 0.28) - Vec3::Z * 0.03, shin, ls * 0.3, 0.06) * back(n);
        a += 0.003 * bump(p.distance(trap_c), 0.08);
        a += 0.0025 * bump(p.distance(lat_c), 0.1) * smoothstep(-0.3, 0.3, -n.z);
        a += 0.0015 * bump_along(p, neck.lerp(head, 0.5), Vec3::Y, 0.06, 0.07);
        a += 0.001 * bump(p.distance(gonion + Vec3::new(0.0, 0.012, 0.012)), 0.025);
        a *= limbs;
        a += chest * 0.004 * bump_along(p, pec_c, Vec3::X, 0.07, 0.06) * smoothstep(0.2, 0.7, n.z);
        n * (a * axilla(p))
    };
    Ok(s.eval(|pt| s.sym(pt, &f)))
}

/// `global.sex` (`identity.sex_shape`): variation within a sex toward typically male shape for
/// positive weights (see the module table); the sex itself is `identity.sex`
/// ([`super::sex::SWITCH`]). The shoulder, hip, waist and neck terms are scaled so that one unit
/// moves `biacromialbreadth`, `hipbreadth`, `waistbreadth` and `neckcircumference` by one
/// within-sex SD (the RMS of the male and female SD, [`measure::Variable::sd`]: a linear axis
/// cannot take a different SD per sex), in the direction of the male-minus-female difference
/// relative to stature ([`measure::Variable::sex_half_difference`]); the other terms are authored.
/// Genital parts are a discrete variant (ADR-039) and never move here.
pub fn sex(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let stature = Gauge::Stature.measure(s, &[], 0.0)?;
    let term = |raw: Raw, gauge: &Gauge, v: &measure::Variable| {
        measure::scaled_to(s, raw, gauge, v.sd_m().copysign(v.sex_half_difference(stature)))
    };
    let mut raw = term(shoulders(kit, 0.04, 0.01)?, &BIACROMIAL_GAUGE, &BIACROMIAL)?;
    raw.add(&term(hips(kit, 0.03)?, &HIP_BREADTH_GAUGE, &HIP_BREADTH)?, 1.0);
    raw.add(&term(waist(kit, 0.04)?, &WAIST_BREADTH_GAUGE, &WAIST_BREADTH)?, 1.0);
    raw.add(&term(face::neck_width(kit)?, &NECK_GIRTH, &NECK_CIRC)?, 1.0);
    raw.add(&Raw::confined(s, SOFT_TISSUE, Symmetry::Mirror, muscle_field(kit, 0.0, 0.5)?, Vec::new())?, 1.0);
    raw.add(&glutes(kit, -0.004)?, 1.0);
    raw.add(&flanks(kit, -0.003)?, 1.0);
    s.smooth_field(&mut raw.disp, SHAPE_SMOOTHING, &TRUNK_AND_LIMBS)?;
    // The breast terms come after the smoothing (each is smoothed on its own already): smoothed
    // again with the rest, the reduction blurred across the inframammary fold and turned 8 faces
    // there over for a female at weight 2 (todo 1.3). The ptosis shrinks with the mound (as in
    // the switch's breast): a female breast reduced by this axis would otherwise keep a lower pole
    // hanging over a smaller mound and turn the faces at the fold over.
    raw.add(&breasts(kit, -0.006)?, 1.0);
    let ptosis = -0.006 * super::sex::BREAST_PTOSIS / super::sex::BREAST_HEIGHT;
    raw.add(&breast_shape(kit, BreastShape::Ptosis, ptosis)?, 1.0);
    raw.add(&face::brow_ridge(kit)?, 0.8);
    raw.add(&face::jaw_width(kit)?, 0.7);
    raw.add(&face::chin_height(kit)?, 0.5);
    raw.add(&face::chin_width(kit)?, 0.5);
    raw.add(&face::larynx(kit, 0.002)?, 1.0);
    raw.add(&face::lip_fullness(kit, true)?, -0.4);
    raw.add(&face::lip_fullness(kit, false)?, -0.4);
    // The groin stays still, so genital parts (attached to the groin loop, ADR-039) never move
    // with this axis: zero within an ellipse of 6 cm across and 9 cm up and down around the crotch,
    // ramping to full at 1.6 times that.
    let crotch = s.landmark("torso.pubis")? - Vec3::Y * 0.04;
    let pos = s.field().positions();
    for (n, d) in raw.disp.iter_mut().enumerate() {
        let p = pos.get(n).copied().unwrap_or(crotch);
        let e = (p.x / 0.06).hypot((p.y - crotch.y) / 0.09);
        *d *= smoothstep(1.0, 1.6, e);
    }
    Ok(raw)
}

/// `global.age`: toward an older shape for positive weights (see the module table).
pub fn age(kit: &Kit<'_, '_>) -> IdResult<Raw> {
    let s = kit.space;
    let jaw = s.landmark("jaw.L.03")?;
    let malar = s.landmark("L.cheek.malar")?;
    let navel = s.landmark("torso.navel")?;
    let xiphoid = s.landmark("torso.xiphoid")?;
    let nipple = s.landmark("L.chest.nipple")?;
    let face_f = move |pt: &Point| {
        let jowl = bump(pt.p.distance(jaw + Vec3::Y * 0.008), 0.022);
        let cheek = bump(pt.p.distance(malar), 0.025);
        (Vec3::Y * -0.0012 + pt.n * 0.0006) * jowl + Vec3::Y * (-0.0008 * cheek)
    };
    let face_disp = s.eval(|pt| s.sym(pt, &face_f));
    let mut raw = Raw::confined(s, face::FACE_SKIN, Symmetry::Mirror, face_disp, Vec::new())?;
    let belly_c = navel + Vec3::Y * ((xiphoid.y - navel.y) * 0.1);
    let belly_r = (xiphoid.y - navel.y) * 0.9;
    let breast_r = (nipple.y - xiphoid.y).abs() + 0.05;
    let body_f = move |pt: &Point| {
        let belly = 0.003 * bump(pt.p.distance(belly_c), belly_r) * smoothstep(0.0, 0.6, pt.n.z);
        let sag = bump(pt.p.distance(nipple - Vec3::Y * 0.02), breast_r) * smoothstep(0.0, 0.4, pt.n.z);
        pt.n * belly + Vec3::Y * (-0.003 * sag)
    };
    let body_disp = s.eval(|pt| s.sym(pt, &body_f));
    raw.add(&Raw::confined(s, SOFT_TISSUE, Symmetry::Mirror, body_disp, Vec::new())?, 1.0);
    raw.add(&glutes(kit, -0.002)?, 1.0);
    raw.add(&face::nose_length(kit)?, 0.35);
    raw.add(&face::ear_scale(kit, Vec3::new(0.005, 0.015, 0.005))?, 1.0);
    raw.add(&face::lip_fullness(kit, true)?, -0.4);
    raw.add(&face::lip_fullness(kit, false)?, -0.4);
    raw.add(&face::brow_height(kit)?, -0.3);
    Ok(raw)
}

/// The left hand's frame: wrist joint, knuckle centre, unit width axis (little to index), unit
/// hand axis (wrist to knuckles), unit back-of-hand direction.
fn hand_frame(s: &Space<'_>) -> IdResult<(Vec3, Vec3, Vec3, Vec3, Vec3)> {
    let wrist = s.joint("L.hand")?;
    let ki = s.landmark("L.hand.knuckle.index")?;
    let kl = s.landmark("L.hand.knuckle.little")?;
    let dorsal = s.landmark("L.wrist.dorsal")?;
    let c = ki.midpoint(kl);
    let w = (ki - kl).normalize_or_zero();
    let a = (c - wrist).reject_from(w).normalize_or_zero();
    let mut up = w.cross(a).normalize_or_zero();
    if up.dot(dorsal - wrist) < 0.0 {
        up = -up;
    }
    Ok((wrist, c, w, a, up))
}

/// `semantic.palm-width`: `k` across the knuckles, fading toward the wrist; finger joints follow.
pub fn palm_width(kit: &Kit<'_, '_>, k: f32) -> IdResult<Raw> {
    let s = kit.space;
    let (wrist, c, w, a, _) = hand_frame(s)?;
    let reach = (c - wrist).dot(a).max(1.0e-3);
    let f = move |p: Vec3| w * (k * (p - c).dot(w) * smoothstep(0.0, 0.8, (p - wrist).dot(a) / reach));
    let disp = s.eval(|pt| if pt.p.x > 0.0 { f(pt.p) } else { Vec3::ZERO });
    let mut joints = Vec::new();
    for fname in FINGERS {
        for seg in ["01", "02", "03"] {
            let n = format!("L.{fname}.{seg}");
            joints.push((s.joint_index(&n)?, f(s.joint(&n)?)));
        }
    }
    Raw::confined(s, HANDS, Symmetry::Mirror, disp, joints)
}

/// `semantic.knuckle-prominence`: `a` metres over the knuckles and middle finger joints, on the
/// back of the hand.
pub fn knuckles(kit: &Kit<'_, '_>, a: f32) -> IdResult<Raw> {
    let s = kit.space;
    let (_, _, _, _, up) = hand_frame(s)?;
    let mut centres = Vec::new();
    for fname in ["index", "middle", "ring", "little"] {
        centres.push((s.joint(&format!("L.{fname}.01"))?, 0.016));
        centres.push((s.joint(&format!("L.{fname}.02"))?, 0.011));
    }
    centres.push((s.joint("L.thumb.02")?, 0.011));
    let disp = s.eval(|pt| {
        if pt.p.x <= 0.0 {
            return Vec3::ZERO;
        }
        let w: f32 = centres.iter().map(|&(c, r)| bump(pt.p.distance(c), r)).fold(0.0, f32::max);
        pt.n * (a * w * smoothstep(0.1, 0.6, pt.n.dot(up)))
    });
    Raw::confined(s, HANDS, Symmetry::Mirror, disp, Vec::new())
}

/// `semantic.foot-width`: `k` across the ball and toes, fading toward the heel and the ankle.
pub fn foot_width(kit: &Kit<'_, '_>, k: f32) -> IdResult<Raw> {
    let s = kit.space;
    let m1 = s.landmark("L.foot.mtp1")?;
    let m5 = s.landmark("L.foot.mtp5")?;
    let heel = s.landmark("L.heel")?;
    let ankle = s.joint("L.foot")?;
    let toes = s.joint("L.toes")?;
    let c = m1.midpoint(m5);
    let a = Vec3::new(c.x - heel.x, 0.0, c.z - heel.z).normalize_or_zero();
    let l = Vec3::Y.cross(a).normalize_or_zero();
    let s_heel = (heel - c).dot(a);
    let f = move |p: Vec3| {
        let along = smoothstep(s_heel, 0.0, (p - c).dot(a));
        let low = 1.0 - smoothstep(ankle.y - 0.03, ankle.y + 0.02, p.y);
        l * (k * (p - c).dot(l) * along * low)
    };
    let disp = s.eval(|pt| if pt.p.x > 0.0 { f(pt.p) } else { Vec3::ZERO });
    let joints = vec![(s.joint_index("L.toes")?, f(toes))];
    Raw::confined(s, FEET, Symmetry::Mirror, disp, joints)
}
