//! The face features `[MOD-01]`: concentric ring fills of the eye, mouth, nose and ear blocks.
//!
//! - Eye: orbit ring (the block boundary, 8 slots) -> lid crease (16) -> lid margin at the lash line
//!   -> posterior lid margin -> conjunctival fornix -> a pocket behind the eyeball. The lids get
//!   four subdivided rows between crease and margin and four across the margin (`[MOD-01]` asks
//!   for at least four rows across each eyelid).
//! - Mouth: perioral ring (12) -> half way (24) -> white roll -> vermilion border -> red lip -> lip contact ->
//!   inner lip -> vestibule -> cheeks -> pharynx cap: the mouth interior is a closed bag of the body
//!   surface that holds the teeth, gums and tongue.
//! - Nose: block ring (8) -> nasal base -> alar ring -> tip cap whose two lower quads open into the
//!   nostril pockets.
//! - Ear: block ring (8) -> root -> helix back -> helix rim -> scapha -> antihelix -> concha cap.
//!
//! Feature points follow Farkas' facial norms (sexes averaged): palpebral fissure 30 mm wide and
//! 10 mm high, intercanthal width 30 mm, mouth width 50 mm, upper vermilion 9 mm, lower 10 mm,
//! nose width 33 mm, nose protrusion 22 mm, ear length 62 mm and width 33 mm.

use std::f32::consts::{FRAC_PI_2, PI, TAU};

use humanart_math::Vec3;

use super::anatomy::{Anatomy, Auricle, EAR_BREADTH, EAR_LENGTH, sided};
use super::body::head_point;
use super::cage::{Cage, Tag};
use super::regions::{C, R};
use super::tube::{mix, unwrap};
use crate::error::{ModelError, Result};

/// The cage loops of one eye.
#[derive(Clone, Debug, Default)]
pub struct EyeLoops {
    /// Rings from the orbit (block boundary) inward; index 2 is the lid margin at the lash line,
    /// 3 the posterior lid margin, 4 the fornix.
    pub rings: Vec<Vec<u32>>,
    /// The slot parameters of the 16-slot rings (0 = lateral canthus, PI = medial canthus,
    /// upper lid in between).
    pub params: Vec<f32>,
    /// The pocket's back vertex.
    pub back: u32,
}

/// The cage loops of the mouth.
#[derive(Clone, Debug, Default)]
pub struct MouthLoops {
    /// Rings from the perioral block boundary inward: 0 block (12 slots), 1 perioral (24 slots
    /// from here on), 2 vermilion border, 3 red lip, 4 lip contact, 5 inner lip, 6 vestibule,
    /// 7 cheeks, 8 pharynx. The white roll between 1 and 2 is [`MouthLoops::roll`].
    pub rings: Vec<Vec<u32>>,
    /// The x position (mm, signed) and upper/lower flag of each slot of the 24-slot rings.
    pub slots: Vec<(f32, bool)>,
    /// The white roll: the skin crest just outside the vermilion border (24 slots).
    pub roll: Vec<u32>,
}

/// The cage loops of the nose.
#[derive(Clone, Debug, Default)]
pub struct NoseLoops {
    /// Rings from the block boundary to the alar ring.
    pub rings: Vec<Vec<u32>>,
    /// The tip (pronasale) vertex.
    pub tip: u32,
    /// Nostril rings (rim, pocket), left then right.
    pub nostrils: [Vec<Vec<u32>>; 2],
}

/// The cage loops of one ear.
#[derive(Clone, Debug, Default)]
pub struct EarLoops {
    /// Rings from the block boundary outward: 0 block, 1 root, 2 helix back, 3 helix rim,
    /// 4 scapha, 5 antihelix (and 6 the concha wall in the finished auricle, whose rings 2 to 6
    /// have twice the block's slots).
    pub rings: Vec<Vec<u32>>,
    /// Slot angles (0 = back, 90 degrees = up).
    pub params: Vec<f32>,
    /// The concha centre vertex.
    pub concha: u32,
}

fn snap(params: &mut [f32], targets: &[f32]) {
    for &t in targets {
        let best =
            params.iter().enumerate().min_by(|a, b| angle_dist(*a.1, t).total_cmp(&angle_dist(*b.1, t))).map(|x| x.0);
        if let Some(i) = best {
            let d = (t - params[i] + PI).rem_euclid(TAU) - PI;
            params[i] += d;
        }
    }
}

fn angle_dist(a: f32, b: f32) -> f32 {
    ((a - b + PI).rem_euclid(TAU) - PI).abs()
}

/// Doubles a ring's slot parameters: each slot, then the midpoint to the next.
fn double(params: &[f32]) -> Vec<f32> {
    let mut p = params.to_vec();
    p.push(params[0] + if params[params.len() - 1] > params[0] { TAU } else { -TAU });
    unwrap(&mut p);
    let n = params.len();
    let mut out = Vec::with_capacity(2 * n);
    for i in 0..n {
        out.push(p[i]);
        out.push(f32::midpoint(p[i], p[i + 1]));
    }
    out
}

/// Ring faces between `a` and `b` with a tag chosen per face from the slot index of `a` (for
/// equal lengths) or a loft (unequal lengths, one tag).
fn bridge_tagged(cage: &mut Cage, a: &[u32], b: &[u32], tag: &dyn Fn(usize) -> Tag) {
    if a.len() == b.len() {
        let n = a.len();
        for i in 0..n {
            let j = (i + 1) % n;
            cage.face(&[a[i], a[j], b[j], b[i]], tag(i));
        }
    } else if b.len() == 2 * a.len() {
        cage.bridge_double(a, b, tag);
    } else {
        let start = cage.face_count();
        cage.loft(a, b, tag(0));
        // Re-tag loft faces by their position along the ring.
        for f in start..cage.face_count() {
            let first = cage.corners_of(f)[0];
            let i = a.iter().position(|&v| v == first).unwrap_or(0);
            cage.tags[f] = tag(i * b.len().max(1) / a.len().max(1));
        }
    }
}

// --- Eye -------------------------------------------------------------------------------------

/// The visible palpebral fissure at slot parameter `t` (0 lateral canthus, PI medial, upper lid
/// between): eye-local millimetres (lateral, up). Seen from the front the upper lid crosses the
/// pupil line 4.5 mm above the eye centre and the lower lid 5.4 mm below it (margin-reflex
/// distances MRD1 / MRD2), so the upper lid covers the top of the iris and the lower lid meets
/// its bottom edge.
pub(crate) fn fissure(t: f32) -> (f32, f32) {
    let t = t.rem_euclid(TAU);
    let (lat, med) = ((14.0_f32, 2.0_f32), (-15.0_f32, -0.5_f32));
    if t <= PI {
        let u = t / PI;
        let x = mix(lat.0, med.0, u);
        let y = mix(lat.1, med.1, u) + 3.9 * (PI * u.powf(1.09)).sin();
        (x, y)
    } else {
        let u = (TAU - t) / PI;
        let x = mix(lat.0, med.0, u);
        let y = mix(lat.1, med.1, u) - 6.2 * (PI * u.powf(0.87)).sin();
        (x, y)
    }
}

/// Depth (mm, forward from the eyeball centre) of a point at lateral offset `x`, height `y` on a
/// sphere of radius `r`, continued backward past the sphere's outline, plus the medial forward lean
/// of the lids.
pub(crate) fn on_sphere(x: f32, y: f32, r: f32) -> f32 {
    let d2 = x * x + y * y;
    let z = if d2 < r * r { (r * r - d2).sqrt() } else { -(d2.sqrt() - r) * 0.9 };
    z + 2.2 * (-x / 15.0).max(0.0).powi(2)
}

/// How far (mm) the subdivided lid margin rolls into the opening past its cage rings.
const MARGIN_ROLL: f32 = 1.4;
/// How far (mm) the fornix ring turns back at the lateral canthus of the finished head (its
/// forward depth there goes from 4 mm to 4 mm less this).
const LATERAL_FORNIX_TUCK: f32 = 4.0;

/// Fills an eye block (8-vertex ring, counter-clockwise from outside).
pub fn eye(cage: &mut Cage, a: &Anatomy, hole: &[u32], side: f32) -> Result<EyeLoops> {
    if hole.len() != 8 {
        return Err(ModelError::invalid("eye block", hole.len(), "eye block ring must have 8 vertices"));
    }
    let e = sided(a.eye_center, side);
    let r = a.eye_radius * 1000.0;
    let local = |x: f32, y: f32, z: f32| -> Vec3 { e + Vec3::new(side * x, y, z) * 0.001 };
    let mut hp: Vec<f32> = hole
        .iter()
        .map(|&v| {
            let d = (cage.pos[v as usize] - e) * 1000.0;
            d.y.atan2(side * d.x)
        })
        .collect();
    unwrap(&mut hp);
    snap(&mut hp, &[0.0, PI]);
    let params = double(&hp);
    let lid = R::sided(R::UpperLidL, R::UpperLidR, side);
    let lower = R::sided(R::LowerLidL, R::LowerLidR, side);
    let socket = R::sided(R::EyeSocketL, R::EyeSocketR, side);
    let face_chart = C::Face.id();
    let socket_chart = C::sided(C::SocketL, C::SocketR, side).id();
    // 1 on the upper lid, 0 on the lower, 1/2 exactly at the canthi (so both sides agree there
    // whatever the rounding of the snapped canthus parameters).
    let upper_w = |t: f32| -> f32 {
        if angle_dist(t, 0.0) < 1e-3 || angle_dist(t, PI) < 1e-3 {
            0.5
        } else if t.rem_euclid(TAU) < PI {
            1.0
        } else {
            0.0
        }
    };
    let is_upper = |t: f32| upper_w(t) > 0.75;
    let lid_tag = |t: f32| -> Tag { Tag::new(if is_upper(t) { lid.id() } else { lower.id() }, face_chart) };
    // Ring 1: lid crease (upper) and lid-cheek line (lower).
    let crease = |t: f32| -> Vec3 {
        let (x, y) = fissure(t);
        let w = (PI * (t.rem_euclid(TAU) / PI).fract()).sin().max(0.0);
        let u = upper_w(t);
        let (y, depth) = (mix(y - 1.5 - 3.5 * w, y + 2.5 + 5.5 * w.powf(0.8), u), mix(r + 3.2, r + 4.2 - 0.4 * w, u));
        let x = x * 1.08;
        local(x, y, on_sphere(x, y, depth))
    };
    let margin = |t: f32, inset: f32, radius: f32| -> Vec3 {
        let (x, y) = fissure(t);
        let (cx, cy) = (-0.5, 0.5);
        // The subdivided lid margin rolls about 1.4 mm into the opening between the lash line and
        // the posterior margin, so both rings sit outside the visible fissure by that much (less
        // at the canthi, where the fold runs sideways).
        let w = (PI * (t.rem_euclid(TAU) / PI).fract()).sin().max(0.0);
        let (dx, dy) = (x - cx, y - cy);
        let len = (dx * dx + dy * dy).sqrt().max(1e-3);
        let grow = MARGIN_ROLL * (0.3 + 0.7 * w);
        let (x, y) = (x + dx / len * grow, y + dy / len * grow);
        let (x, y) = (cx + (x - cx) * (1.0 - inset), cy + (y - cy) * (1.0 - inset));
        local(x, y, on_sphere(x, y, radius))
    };
    let fornix = |t: f32, z: f32, rad: f32| -> Vec3 {
        let (x, y) = fissure(t);
        let d = Vec3::new(x * 0.92, y * 1.7 + mix(-1.0, 2.0, upper_w(t)), z).normalize() * rad;
        local(d.x, d.y, d.z)
    };
    let ring1: Vec<Vec3> = params.iter().map(|&t| crease(t)).collect();
    let ring2: Vec<Vec3> = params.iter().map(|&t| margin(t, 0.0, r + 1.7)).collect();
    let ring3: Vec<Vec3> = params.iter().map(|&t| margin(t, 0.04, r + 0.35)).collect();
    // Finished head (round 4): at the lateral canthus the fornix turns back, so the strip of
    // socket lining between the eyeball's outline and the meeting lids lies in the corner's shadow
    // instead of facing the viewer as a pale column ([`LATERAL_FORNIX_TUCK`]).
    let lateral_tuck = |t: f32| -> f32 {
        if a.auricle != Auricle::Finished || t.cos() <= 0.0 {
            return 0.0;
        }
        let w = (PI * (t.rem_euclid(TAU) / PI).fract()).sin().max(0.0);
        LATERAL_FORNIX_TUCK * (1.0 - w).powi(3)
    };
    let ring4: Vec<Vec3> = params.iter().map(|&t| fornix(t, 4.0 - lateral_tuck(t), r + 0.7)).collect();
    let ring5: Vec<Vec3> = params.iter().map(|&t| fornix(t, -6.0, r + 1.2)).collect();
    // The parameter in the middle of slot `i` (between ring vertices `i` and `i + 1`).
    let n = params.len();
    let step = if n > 1 && params[1] < params[0] { -TAU } else { TAU };
    let slot_mid = |i: usize| -> f32 {
        let next = if i + 1 < n { params[i + 1] } else { params[0] + step };
        f32::midpoint(params[i % n], next)
    };
    let mut rings = vec![hole.to_vec()];
    let r1 = cage.add_all(&ring1);
    bridge_tagged(cage, hole, &r1, &|i| lid_tag(slot_mid(i.min(n - 1))));
    rings.push(r1);
    for (k, pts) in [ring2, ring3, ring4, ring5].iter().enumerate() {
        let ring = cage.add_all(pts);
        let prev = rings.last().cloned().unwrap_or_default();
        let tag = |i: usize| -> Tag {
            // The lid margin (lash line to the posterior edge) belongs to the socket chart: the seam
            // runs along the lash line.
            let at_canthus = |t: f32| angle_dist(t, 0.0) < 1e-3 || angle_dist(t, PI) < 1e-3;
            if k >= 2 {
                Tag::new(socket.id(), socket_chart)
            } else if k == 1 || (at_canthus(params[i % n]) || at_canthus(params[(i + 1) % n])) {
                // The canthal corners of the lids also go to the socket chart: the face chart's
                // eye hole stays round, so its conformal map has no cone at the sharp corners.
                Tag::new(lid_tag(slot_mid(i % n)).region, socket_chart)
            } else {
                lid_tag(slot_mid(i % n))
            }
        };
        bridge_tagged(cage, &prev, &ring, &tag);
        rings.push(ring);
    }
    let last = rings.last().cloned().unwrap_or_default();
    let back = local(0.0, 0.5, -(r + 2.0));
    // Cap corners half way between the canthi and the lid middles, on both sides alike.
    let lateral = (0..n).find(|&i| angle_dist(params[i], 0.0) < 1e-3).unwrap_or(0);
    let interior =
        cage.cap(&last, 4, 4, (lateral + 2) % n.max(1), Tag::new(socket.id(), socket_chart), &|_, _, p| {
            let d = (p - e).normalize_or_zero();
            let d = Vec3::new(d.x, d.y, d.z.min(-0.4)).normalize_or_zero();
            e + d * ((r + 1.6) * 0.001)
        })?;
    let back_v = interior.get(interior.len() / 2).copied().unwrap_or(0);
    if let Some(p) = cage.pos.get_mut(back_v as usize) {
        *p = back;
    }
    Ok(EyeLoops { rings, params, back: back_v })
}

// --- Mouth -----------------------------------------------------------------------------------

/// How far (mm) the white-roll ring lies outside the vermilion border on the finished head
/// ([`Auricle::Finished`], the head the scan face is applied to). The white roll is a band of 1
/// to 2 mm; at [`BASE_ROLL_OFFSET`] the cage fit cannot reach the crest and folds the band
/// between roll and border: inverted faces on the upper vermilion 10 to 23 mm from the midline,
/// which renders show as holes (`diag_mouth_folds`).
const ROLL_OFFSET: f32 = 1.6;
/// The roll offset of the fitting base ([`Auricle::Base`], `build::BASE_VERSION`): the scans were
/// fitted to this head, so it keeps the offset it was measured with, folds included, until the
/// face model is rebuilt.
const BASE_ROLL_OFFSET: f32 = 0.9;

/// Half width of the mouth (cheilion), mm.
const MOUTH_HALF: f32 = 25.0;

/// A smooth bump: 1 at `s = 0`, falling to 0 at `|s| = 1` with zero slope.
fn bump(s: f32) -> f32 {
    let q = (1.0 - s * s).max(0.0);
    q * q
}

/// The x position of a lip slot as a fraction of the half width at `u` (0 at the midline, 1 at
/// the corner, uniform in the ring angle): denser near the midline, so slots fall on the Cupid's
/// bow peaks (`u` = 2/6 gives 0.289) and on the philtrum columns.
fn slot_x(u: f32) -> f32 {
    0.8 * u + 0.2 * u * u
}

/// Lip shape constants (mm). **Authored**: the proportions are in the ranges facial
/// anthropometry reports for adults (mouth width ch-ch ~50 mm, upper vermilion height ls-sto
/// ~8-9 mm, lower vermilion height sto-li ~10 mm, sn-sto ~21 mm; Farkas' landmark definitions),
/// not fitted to one published table.
mod lip {
    /// Cupid's bow peak, as a fraction of the half width (x = 7.2 mm).
    pub const PEAK: f32 = 0.289;
    /// Upper vermilion height at the midline (the bow's dip).
    pub const UPPER_MID: f32 = 8.3;
    /// Upper vermilion height at the bow's peaks.
    pub const UPPER_PEAK: f32 = 9.9;
    /// Lower vermilion height at the midline.
    pub const LOWER_MID: f32 = 10.2;
}

/// The vermilion border at `t` = |x| / half width: (y relative to the stomion, z in the head
/// frame). The upper border has the Cupid's bow (a dip at the midline between two peaks); both
/// borders fall to the stomion at the corners and recede along the dental arch.
fn vermilion(t: f32, upper: bool) -> (f32, f32) {
    let t = t.min(1.0);
    if upper {
        let y = if t < lip::PEAK {
            lip::UPPER_MID + (lip::UPPER_PEAK - lip::UPPER_MID) * (FRAC_PI_2 * t / lip::PEAK).sin().powi(2)
        } else {
            let u = (t - lip::PEAK) / (1.0 - lip::PEAK);
            lip::UPPER_PEAK * (1.0 - u.powf(1.5)).max(0.0).powf(0.8)
        };
        (y.max(0.4), 86.5 + 18.5 * (1.0 - t.powf(1.8)))
    } else {
        let s = (1.0 - t.powf(2.1)).max(0.0).powf(0.75);
        ((-lip::LOWER_MID * s).min(-0.4), 86.5 + 17.0 * (1.0 - t.powf(1.6)))
    }
}

/// The lip contact line at `t`: (y, z). The upper lip's tubercle pushes the middle of the line
/// down a little.
fn contact(t: f32, upper: bool) -> (f32, f32) {
    let s = 1.0 - t * t;
    let tubercle = -0.6 * bump(t / 0.3);
    let y = if upper { 0.3 * s + tubercle } else { -0.3 * s + tubercle };
    (y, 86.0 + 16.0 * (1.0 - t.powf(1.7)))
}

/// Fills the mouth block (12-vertex ring, counter-clockwise from the front).
///
/// Rings: the block, the perioral skin (24 slots, carrying the philtrum columns and groove), the
/// white roll (a fine skin crest [`ROLL_OFFSET`] outside the vermilion border, [`MouthLoops::roll`]), the
/// vermilion border, the red lip (the upper tubercle and the everted lower lip), the contact
/// line, the inner lip and the mouth bag. Each ring meets the corner at its own point, stepping
/// inward and back, so the commissure tucks in under the modiolus.
pub fn mouth(cage: &mut Cage, a: &Anatomy, hole: &[u32]) -> Result<MouthLoops> {
    if hole.len() != 12 {
        return Err(ModelError::invalid("mouth block", hole.len(), "mouth block ring must have 12 vertices"));
    }
    let sto_y = -48.0_f32;
    let hpt = |x: f32, y: f32, z: f32| head_point(a, x, sto_y + y, z);
    let centre = hpt(0.0, 0.0, 92.0);
    let mut hp: Vec<f32> = hole
        .iter()
        .map(|&v| {
            let d = cage.pos[v as usize] - centre;
            d.y.atan2(d.x)
        })
        .collect();
    unwrap(&mut hp);
    snap(&mut hp, &[0.0, FRAC_PI_2, PI, 3.0 * FRAC_PI_2]);
    let params = double(&hp);
    // Slot x positions: corners at +-MOUTH_HALF, denser near the midline (`slot_x`).
    let slots: Vec<(f32, bool)> = params
        .iter()
        .map(|&t| {
            let t = t.rem_euclid(TAU);
            let (s, up) = if t <= PI { (1.0 - 2.0 * t / PI, true) } else { (2.0 * (t - PI) / PI - 1.0, false) };
            (MOUTH_HALF * s.signum() * slot_x(s.abs()), up)
        })
        .collect();
    let is_corner = |x: f32| x.abs() > MOUTH_HALF - 0.01;
    // Each lip ring has its own half width, so the rings meet the corner (commissure) at distinct
    // points that step inward and back, instead of fanning out of one point.
    let ring = |half: f32, f: &dyn Fn(bool, f32) -> (f32, f32)| -> Vec<Vec3> {
        slots
            .iter()
            .map(|&(x, up)| {
                let t = (x.abs() / MOUTH_HALF).min(1.0);
                // Both corners count as upper (their slot flags depend on angle rounding).
                let (y, z) = f(up || is_corner(x), t);
                hpt(x / MOUTH_HALF * half, if is_corner(x) { 0.0 } else { y }, z)
            })
            .collect()
    };
    // The corner recess: how far (mm) the lips sink back at the commissure.
    let tuck = |t: f32| 1.6 * t.powi(6);
    let roll_offset = if a.auricle == Auricle::Finished { ROLL_OFFSET } else { BASE_ROLL_OFFSET };
    let roll = ring(26.4, &|up, t| {
        let (y, z) = vermilion(t, up);
        let w = 1.0 - t.powi(4);
        // A little higher over the bow's peaks, where the philtrum columns end.
        let peak = if up { 0.25 * bump((t - lip::PEAK) / 0.12) } else { 0.0 };
        // The skin meets the border at a steeper angle than the vermilion leaves it: the
        // border is a crest, not a groove.
        (y + if up { roll_offset } else { -roll_offset } * w, z - if up { 0.3 } else { 0.6 } * w + peak - tuck(t))
    });
    let border = ring(25.5, &|up, t| {
        let (y, z) = vermilion(t, up);
        (y, z - tuck(t))
    });
    let red = ring(24.5, &|up, t| {
        let (yb, zb) = vermilion(t, up);
        let (yc, _) = contact(t, up);
        if up {
            // The upper vermilion faces forward and down; the tubercle hangs at the midline.
            (mix(yb, yc, 0.52) - 0.5 * bump(t / 0.3), zb + 0.3 + 0.6 * bump(t / 0.35) - 1.0 * t.powi(4) - tuck(t))
        } else {
            // The lower lip is everted: its middle stands forward of the border and faces up.
            (mix(yb, yc, 0.5), zb + 2.0 * (1.0 - t * t) - 0.8 * t.powi(4) - tuck(t))
        }
    });
    let lip_contact = ring(23.5, &|up, t| {
        let (y, z) = contact(t, up);
        (y, z - 1.6 * t.powi(4) - tuck(t))
    });
    let inner = ring(21.0, &|up, t| {
        let (y, z) = contact(t, up);
        let open = 1.6 * (1.0 - t * t) + 0.4;
        (y + if up { open } else { -open }, z - 5.0 - 1.0 * (1.0 - t) - 2.0 * t.powi(4))
    });
    let bag = |x_half: f32, y_up: f32, y_down: f32, z_front: f32, z_side: f32| -> Vec<Vec3> {
        params
            .iter()
            .map(|&t| {
                let (s, c) = t.sin_cos();
                let x = x_half * c;
                let y = if s >= 0.0 { y_up * s } else { y_down * -s };
                let z = mix(z_side, z_front, s.abs().powf(0.7));
                hpt(x, y, z)
            })
            .collect()
    };
    let vestibule = bag(27.0, 15.0, -19.0, 94.0, 76.0);
    let cheeks = bag(35.0, 14.0, -22.0, 70.0, 64.0);
    let pharynx = bag(31.0, 11.0, -22.0, 46.0, 44.0);
    // Perioral ring (24 slots): half way between the block ring (its edge midpoints at the odd
    // slots) and the white roll. On the upper lip it carries the philtrum: a groove at the
    // midline between two raised columns; at the corners the modiolus stands a little forward.
    let n_hole = hole.len();
    let perioral: Vec<Vec3> = (0..2 * n_hole)
        .map(|k| {
            let p = if k % 2 == 0 {
                cage.pos[hole[k / 2] as usize]
            } else {
                (cage.pos[hole[k / 2] as usize] + cage.pos[hole[(k / 2 + 1) % n_hole] as usize]) * 0.5
            };
            let m = p * 0.5 + roll[k] * 0.5;
            let (x, up) = slots[k];
            let t = (x.abs() / MOUTH_HALF).min(1.0);
            // Columns at the first slot off the midline (~5.5 mm out), the groove between.
            let philtrum = match (up, x.abs() < 1e-3, t < 0.2) {
                (true, true, _) => -1.1,
                (true, false, true) => 0.9,
                _ => 0.0,
            };
            let modiolus = 1.0 * bump((t - 1.0) / 0.2);
            m + Vec3::Z * ((philtrum + modiolus) * 0.001)
        })
        .collect();
    let face = C::Face.id();
    let bag_chart = C::MouthBag.id();
    let lip_tag = |up: bool| Tag::new(if up { R::UpperLip.id() } else { R::LowerLip.id() }, face);
    // A face between slots i and i + 1 is on the upper lip when both slots are (the corner slots
    // count as upper), so the two corners are tagged as mirror images.
    let n = slots.len();
    let up_or_corner = |k: usize| slots[k % n].1 || is_corner(slots[k % n].0);
    let face_up = |i: usize| up_or_corner(i) && up_or_corner(i + 1);
    let mut rings = vec![hole.to_vec()];
    let r1 = cage.add_all(&perioral);
    // Block to perioral: one pentagon per block edge, tagged by its middle (odd) slot, never a
    // corner, so the corners are mirror images.
    bridge_tagged(cage, hole, &r1, &|i| lip_tag(slots[i % n].1));
    rings.push(r1.clone());
    let roll_v = cage.add_all(&roll);
    bridge_tagged(cage, &r1, &roll_v, &|i| lip_tag(face_up(i)));
    let interior = Tag::new(R::MouthInterior.id(), bag_chart);
    let mut prev = roll_v.clone();
    for (k, pts) in [&border, &red, &lip_contact, &inner, &vestibule, &cheeks, &pharynx].iter().enumerate() {
        let ring = cage.add_all(pts);
        let tag = |i: usize| -> Tag {
            if k >= 4 {
                interior
            } else if k == 3 {
                // The inner lips belong to the mouth-bag chart: the seam is the lip contact line.
                Tag::new(lip_tag(face_up(i)).region, bag_chart)
            } else {
                lip_tag(face_up(i))
            }
        };
        bridge_tagged(cage, &prev, &ring, &tag);
        rings.push(ring.clone());
        prev = ring;
    }
    let last = rings.last().cloned().unwrap_or_default();
    let back = hpt(0.0, -4.0, 36.0);
    // The cap's corners must be mirror images of each other: start three slots before the top
    // centre slot (mirroring maps slot `top + k` to `top - k`).
    let top = slots.iter().position(|&(x, up)| up && x.abs() < 1e-3).unwrap_or(n / 4);
    let start = (top + n - 3) % n.max(1);
    cage.cap(&last, 6, 6, start, interior, &|s, t, p| {
        let w = ((PI * s).sin() * (PI * t).sin()).powf(0.6);
        Vec3::new(p.x, p.y, p.z + (back.z - p.z) * 0.6 * w)
    })?;
    Ok(MouthLoops { rings, slots, roll: roll_v })
}

// --- Nose ------------------------------------------------------------------------------------

/// Fills the nose block (8-vertex ring, counter-clockwise from the front).
pub fn nose(cage: &mut Cage, a: &Anatomy, hole: &[u32]) -> Result<NoseLoops> {
    if hole.len() != 8 {
        return Err(ModelError::invalid("nose block", hole.len(), "nose block ring must have 8 vertices"));
    }
    let hpt = |x: f32, y: f32, z: f32| head_point(a, x, y, z);
    let centre = hpt(0.0, -12.0, 104.0);
    // Slots by angle: 270 = subnasale, 315/225 = alar bases, 0/180 = alar creases, 45/135 = side
    // walls, 90 = dorsum.
    let angle = |v: u32| -> f32 {
        let d = cage.pos[v as usize] - centre;
        d.y.atan2(d.x).rem_euclid(TAU)
    };
    let slot = |deg: f32| -> Option<usize> {
        let t = deg.to_radians();
        (0..8).min_by(|&i, &j| angle_dist(angle(hole[i]), t).total_cmp(&angle_dist(angle(hole[j]), t)))
    };
    // Base ring (nasal base, alar-facial groove) and alar ring, per slot direction.
    let base_pt = |deg: i32| -> Vec3 {
        let s = if deg % 360 < 90 || deg % 360 > 270 { 1.0 } else { -1.0 };
        match deg.rem_euclid(360) {
            270 => hpt(0.0, -25.5, 104.0),
            315 | 225 => hpt(s * 11.5, -26.0, 99.0),
            0 | 180 => hpt(s * 17.5, -15.5, 96.5),
            45 | 135 => hpt(s * 7.5, 1.5, 105.5),
            _ => hpt(0.0, 0.5, 113.0),
        }
    };
    let alar_pt = |deg: i32| -> Vec3 {
        let s = if deg % 360 < 90 || deg % 360 > 270 { 1.0 } else { -1.0 };
        match deg.rem_euclid(360) {
            270 => hpt(0.0, -24.8, 110.0),
            315 | 225 => hpt(s * 11.0, -25.2, 102.0),
            0 | 180 => hpt(s * 16.0, -18.5, 101.0),
            45 | 135 => hpt(s * 6.0, -6.0, 113.0),
            _ => hpt(0.0, -5.5, 117.5),
        }
    };
    let tip_pt = |deg: i32| -> Vec3 {
        let s = if deg % 360 < 90 || deg % 360 > 270 { 1.0 } else { -1.0 };
        match deg.rem_euclid(360) {
            270 => hpt(0.0, -23.0, 119.0),
            315 | 225 => hpt(s * 5.5, -23.0, 117.0),
            0 | 180 => hpt(s * 7.5, -18.5, 118.5),
            45 | 135 => hpt(s * 4.5, -13.0, 120.5),
            _ => hpt(0.0, -11.5, 121.5),
        }
    };
    let degs: Vec<i32> = hole
        .iter()
        .map(|&v| {
            let d = angle(v).to_degrees();
            ((d / 45.0).round() as i32 * 45).rem_euclid(360)
        })
        .collect();
    let mut sorted = degs.clone();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.len() != 8 || slot(270.0).is_none() {
        return Err(ModelError::invalid("nose block", 0, "nose block slots do not cover the eight directions"));
    }
    // A face between slots j and j + 1 is tagged by its first slot. This splits the nose into
    // tip / ala / bridge slightly differently on the two sides (the regions all belong to `nose`,
    // which is what skinning and the identity axes use); a mirror-exact split by the face's middle
    // direction made identity's `id_02` border-crease check fail for `semantic.nose-length`, so the
    // original split stays until identity revisits it.
    let tag_of = |j: usize| -> Tag {
        let region = match degs[j] {
            0 | 315 => R::AlaL,
            180 | 225 => R::AlaR,
            270 => R::NasalTip,
            _ => R::NasalBridge,
        };
        Tag::new(region.id(), C::Face.id())
    };
    let base: Vec<Vec3> = degs.iter().map(|&d| base_pt(d)).collect();
    let alar: Vec<Vec3> = degs.iter().map(|&d| alar_pt(d)).collect();
    let b = cage.add_all(&base);
    bridge_tagged(cage, hole, &b, &tag_of);
    let al = cage.add_all(&alar);
    bridge_tagged(cage, &b, &al, &tag_of);
    let tips: Vec<Vec3> = degs.iter().map(|&d| tip_pt(d)).collect();
    let tr = cage.add_all(&tips);
    // Alar ring to tip ring: the quads from the columella to the alar base open into the nostrils.
    let mut nostrils: [Vec<Vec<u32>>; 2] = [Vec::new(), Vec::new()];
    for j in 0..8 {
        let j1 = (j + 1) % 8;
        let quad = [al[j], al[j1], tr[j1], tr[j]];
        let nostril_side = match (degs[j], degs[j1]) {
            (270, 315) | (315, 270) => Some(1.0),
            (270, 225) | (225, 270) => Some(-1.0),
            _ => None,
        };
        match nostril_side {
            Some(side) => nostrils[usize::from(side < 0.0)] = nostril(cage, a, &quad, side),
            None => cage.face(&quad, tag_of(j)),
        }
    }
    let tip_tag = Tag::new(R::NasalTip.id(), C::Face.id());
    let tip_point = hpt(0.0, -18.5, 123.0);
    let interior = cage.cap(&tr, 2, 2, 0, tip_tag, &|_, _, _| tip_point)?;
    let tip = interior.first().copied().unwrap_or(0);
    Ok(NoseLoops { rings: vec![hole.to_vec(), b, al, tr], tip, nostrils })
}

/// Opens a nostril in a lower nose quad (columella, alar base, ala, tip): rim ring, pocket ring,
/// closed top. Returns the rim, middle and pocket rings.
fn nostril(cage: &mut Cage, a: &Anatomy, quad: &[u32; 4], side: f32) -> Vec<Vec<u32>> {
    let hpt = |x: f32, y: f32, z: f32| head_point(a, side * x, y, z);
    // Rim points in the quad's corner order: medial (columella side), posterior (alar base side),
    // lateral (ala side), anterior (tip side).
    let rim = [hpt(3.0, -24.6, 111.5), hpt(9.0, -25.4, 103.5), hpt(9.5, -24.0, 108.0), hpt(4.5, -23.6, 116.0)];
    // The pocket is the vestibule, ~10 mm deep above the rim (it carries the nostril region).
    let pocket = [hpt(3.6, -14.5, 109.0), hpt(7.6, -15.0, 104.5), hpt(8.4, -14.0, 107.5), hpt(4.8, -13.5, 111.5)];
    let region = R::sided(R::NostrilL, R::NostrilR, side);
    let chart = C::sided(C::NostrilL, C::NostrilR, side);
    // Order the corner points like the quad: the cyclic assignment (either direction, as the
    // right side runs the other way round) with the least total squared distance.
    let corners = quad.map(|v| cage.pos[v as usize]);
    let mut order = [0_usize, 1, 2, 3];
    let mut best = f32::INFINITY;
    for shift in 0..4 {
        for reverse in [false, true] {
            let cand: [usize; 4] = std::array::from_fn(|i| if reverse { (shift + 4 - i) % 4 } else { (shift + i) % 4 });
            let cost: f32 = (0..4).map(|i| corners[i].distance_squared(rim[cand[i]])).sum();
            if cost < best {
                best = cost;
                order = cand;
            }
        }
    }
    let rim_v: Vec<u32> = order.iter().map(|&i| cage.add(rim[i])).collect();
    let pocket_v: Vec<u32> = order.iter().map(|&i| cage.add(pocket[i])).collect();
    // The vestibule widens a little above the rim: a middle ring 1.2 mm out from the nostril axis.
    let axis_rim = rim.iter().fold(Vec3::ZERO, |a, &p| a + p) / 4.0;
    let axis_pocket = pocket.iter().fold(Vec3::ZERO, |a, &p| a + p) / 4.0;
    let mid: Vec<Vec3> = (0..4)
        .map(|i| {
            let p = rim[i] + (pocket[i] - rim[i]) * 0.45;
            let c = axis_rim + (axis_pocket - axis_rim) * 0.45;
            p + (p - c).normalize_or_zero() * 0.0012
        })
        .collect();
    let mid_v: Vec<u32> = order.iter().map(|&i| cage.add(mid[i])).collect();
    let ala = Tag::new(R::sided(R::AlaL, R::AlaR, side).id(), C::Face.id());
    cage.bridge(quad, &rim_v, Tag::new(ala.region, chart.id()));
    cage.bridge(&rim_v, &mid_v, Tag::new(region.id(), chart.id()));
    cage.bridge(&mid_v, &pocket_v, Tag::new(region.id(), chart.id()));
    cage.face(&pocket_v, Tag::new(region.id(), chart.id()));
    vec![rim_v, mid_v, pocket_v]
}

// --- Ear -------------------------------------------------------------------------------------

/// The auricle outline at angle `t` (0 = back, 90 degrees = up): (back, up) in mm from the ear
/// centre, and the lateral protrusion of the helix there.
fn auricle(t: f32) -> (f32, f32, f32) {
    let (s, c) = t.sin_cos();
    // An egg: wider at the top, the lobe narrower and lower.
    let back = if c >= 0.0 { 17.5 * c } else { 12.0 * c };
    let up = if s >= 0.0 { 31.0 * s } else { 30.0 * s };
    let lobe = (-s).max(0.0).powi(3);
    let back = back - 4.0 * lobe * c.abs();
    let protrude = 7.0 + 9.0 * c.max(0.0) + 3.0 * s.max(0.0) - 2.0 * lobe;
    (back + 3.0, up, protrude)
}

/// Fills an ear block (8-vertex ring, counter-clockwise from outside).
/// `design` picks the auricle ([`Auricle`]).
pub fn ear(cage: &mut Cage, hole: &[u32], side: f32, design: Auricle) -> Result<EarLoops> {
    if hole.len() != 8 {
        return Err(ModelError::invalid("ear block", hole.len(), "ear block ring must have 8 vertices"));
    }
    if design == Auricle::Finished {
        return ear_finished(cage, hole, side);
    }
    let centre = hole.iter().fold(Vec3::ZERO, |s, &v| s + cage.pos[v as usize]) / 8.0;
    let tilt = 0.30_f32;
    // The base design faces 16 degrees back (it reads as a blade from the front).
    let out = Vec3::new(side * 0.96, 0.0, -0.28).normalize();
    let up = Vec3::new(0.0, tilt.cos(), -tilt.sin());
    let back = up.cross(out) * side;
    let local = |b: f32, u: f32, o: f32| -> Vec3 { centre + (back * b + up * u + out * o) * 0.001 };
    let mut params: Vec<f32> = hole
        .iter()
        .map(|&v| {
            let d = cage.pos[v as usize] - centre;
            d.dot(up).atan2(d.dot(back))
        })
        .collect();
    unwrap(&mut params);
    let helix = R::sided(R::EarLHelix, R::EarRHelix, side);
    let antihelix = R::sided(R::EarLAntihelix, R::EarRAntihelix, side);
    let concha = R::sided(R::EarLConcha, R::EarRConcha, side);
    let lobe = R::sided(R::EarLLobe, R::EarRLobe, side);
    let back_r = R::sided(R::EarLBack, R::EarRBack, side);
    let chart = C::sided(C::EarL, C::EarR, side).id();
    let back_chart = C::sided(C::EarLBack, C::EarRBack, side).id();
    let is_lobe = |t: f32| t.sin() < -0.55;
    let ring_pts = |scale: f32, lift: f32, depth: f32| -> Vec<Vec3> {
        params
            .iter()
            .map(|&t| {
                let (b, u, o) = auricle(t);
                local(b * scale, u * scale + lift, o * depth)
            })
            .collect()
    };
    let root: Vec<Vec3> = params
        .iter()
        .map(|&t| {
            let (b, u, _) = auricle(t);
            local(b * 0.55, u * 0.32, 4.0 + 2.5 * t.cos().max(0.0))
        })
        .collect();
    let rings_pts = [
        root,
        ring_pts(1.0, 0.0, 0.88),
        ring_pts(0.97, 0.0, 1.06),
        ring_pts(0.74, -0.5, 0.92),
        ring_pts(0.52, -1.0, 0.97),
    ];
    let tags: [&dyn Fn(f32) -> R; 5] = [
        &|_| back_r,
        &|t| if is_lobe(t) { lobe } else { back_r },
        &|t| if is_lobe(t) { lobe } else { helix },
        &|t| if is_lobe(t) { lobe } else { helix },
        &|t| if is_lobe(t) { lobe } else { antihelix },
    ];
    let mut rings = vec![hole.to_vec()];
    for (k, pts) in rings_pts.iter().enumerate() {
        let ring = cage.add_all(pts);
        let prev = rings.last().cloned().unwrap_or_default();
        let tagf = tags[k];
        // The medial side of the auricle (up to the back of the helix rim) is its own chart.
        let ring_chart = if k <= 1 { back_chart } else { chart };
        // Tagged by the face's middle parameter (mirror-consistent on both ears).
        let n_par = params.len();
        bridge_tagged(cage, &prev, &ring, &|i| {
            let (a0, a1) = (params[i], params[(i + 1) % n_par]);
            let mid = (a0.sin() + a1.sin()).atan2(a0.cos() + a1.cos());
            Tag::new(tagf(mid).id(), ring_chart)
        });
        rings.push(ring);
    }
    let last = rings.last().cloned().unwrap_or_default();
    let concha_pt = local(-1.0, -3.0, 1.5);
    let interior = cage.cap(&last, 2, 2, 1, Tag::new(concha.id(), chart), &|_, _, _| concha_pt)?;
    let concha_v = interior.first().copied().unwrap_or(0);
    Ok(EarLoops { rings, params, concha: concha_v })
}

/// How far the back of the finished auricle's rolled rim lies outside its outline on the helix, mm.
const EAR_RIM_OUT: f32 = 1.2;
/// How much shorter and narrower the finished auricle's outline is than the ANSUR length and
/// breadth, mm: on the surface the rolled rim and the round lobe add this much (measured by the
/// test `wp25_finished_ear_is_fold_free_and_ansur_sized`).
const EAR_OUTLINE_LESS: (f32, f32) = (5.5, 1.2);
/// Auriculocephalic angle of the finished auricle, degrees: the helix stands off the side of the
/// head by this angle, hinged at the front edge of the auricle (normal ears 15 to 30 degrees); it
/// sets the protrusion (ANSUR II `earprotrusion`, checked by a test).
const EAR_ANGLE_DEG: f32 = 26.0;
/// The concha's middle in the finished auricle's frame (back, up), mm: in front of and below the
/// auricle's centre, where the ear canal opens.
const CONCHA_CENTRE: (f32, f32) = (1.0, -4.0);
/// The concha wall's rim around [`CONCHA_CENTRE`]: radii back, front, up, down, mm.
const CONCHA_RADII: [f32; 4] = [8.0, 5.0, 12.0, 8.2];
/// How high the concha wall's rim stands, as a share of the antihelix's height off the plane.
const CONCHA_RIM: f32 = 0.8;
/// Depth of the concha floor below the lowest point of the concha wall's rim, mm.
const CONCHA_DEPTH: f32 = 5.0;
/// Height of the finished auricle's root off the auricle's plane behind, at the front edge and at
/// the lobe, mm.
const ROOT_LIFT: (f32, f32, f32) = (7.0, 3.9, 3.1);
/// The finished auricle's root behind, as a share of the back of the rim's distance from the ear
/// block's centre.
const ROOT_SHARE: f32 = 0.7;
/// How far the finished auricle's root lies in front of the back of the rim at the front edge, mm.
const ROOT_FRONT: f32 = 0.85;
/// The finished auricle's root stays this far outside the concha's wall, mm.
const ROOT_MARGIN: f32 = 1.8;
/// The finished auricle's root stays within this share of the ear block's outline where a groove
/// runs around it, and where it lies in front of the rim (the skin rises from the face into the
/// helix root and the tragus).
const ROOT_REACH: (f32, f32) = (0.91, 0.9);

/// The finished auricle at slot angle `t` (0 = back, 90 degrees = up), in mm in the ear's frame:
/// the outline point (back, up), the helix rim's protrusion off the auricle's plane, and how
/// much of the lobe is there (0 above, 1 at the bottom).
fn auricle_finished(t: f32) -> (f32, f32, f32, f32) {
    let (s, c) = t.sin_cos();
    let length = EAR_LENGTH * 1000.0 - EAR_OUTLINE_LESS.0;
    let breadth = EAR_BREADTH * 1000.0 - EAR_OUTLINE_LESS.1;
    let half_up = length * 0.52;
    let half_down = length * 0.48;
    // Broad, rounded top (a superellipse), a straighter back edge, the lobe narrower and round.
    let back = if c >= 0.0 { 0.55 * breadth * c.powf(0.8) } else { -0.45 * breadth * (-c).powf(0.9) };
    let up = if s >= 0.0 { half_up * s.powf(0.85) } else { -half_down * (-s) };
    let lobe = (-s).max(0.0).powi(2);
    let back = back * (1.0 - 0.3 * lobe) + 3.0;
    // Hinged at the front edge: the farther back, the farther out; the top of the helix flares,
    // the lobe lies closer to the head.
    let hinge = EAR_ANGLE_DEG.to_radians().tan();
    let front = -0.45 * breadth + 3.0;
    let protrude = (3.5 + hinge * (back - front) + 2.5 * s.max(0.0) + 3.0 * (-c).max(0.0)) * (1.0 - 0.15 * lobe);
    (back, up, protrude, lobe)
}

/// The finished auricle ([`Auricle::Finished`]): 16 slots from the root on (twice the block's 8),
/// a helix rim rolled into a tube about 5 mm thick, the scapha groove inside it, the antihelix
/// ridge, the concha's wall around [`CONCHA_CENTRE`] and its floor, a thick round lobe, hinged off
/// the head at [`EAR_ANGLE_DEG`]; ANSUR II length and breadth ([`EAR_LENGTH`], [`EAR_BREADTH`]).
fn ear_finished(cage: &mut Cage, hole: &[u32], side: f32) -> Result<EarLoops> {
    let centre = hole.iter().fold(Vec3::ZERO, |s, &v| s + cage.pos[v as usize]) / 8.0;
    let tilt = 0.30_f32;
    // The auricle's plane faces out and a little forward (6 degrees); the hinge in
    // `auricle_finished` turns the helix off the head.
    let out = Vec3::new(side * 0.9945, 0.0, 0.105).normalize();
    let up = Vec3::new(0.0, tilt.cos(), -tilt.sin());
    let back = up.cross(out) * side;
    let local = |b: f32, u: f32, o: f32| -> Vec3 { centre + (back * b + up * u + out * o) * 0.001 };
    let mut params: Vec<f32> = hole
        .iter()
        .map(|&v| {
            let d = cage.pos[v as usize] - centre;
            d.dot(up).atan2(d.dot(back))
        })
        .collect();
    unwrap(&mut params);
    let fine = double(&params);
    let helix = R::sided(R::EarLHelix, R::EarRHelix, side);
    let antihelix = R::sided(R::EarLAntihelix, R::EarRAntihelix, side);
    let concha = R::sided(R::EarLConcha, R::EarRConcha, side);
    let lobe = R::sided(R::EarLLobe, R::EarRLobe, side);
    let back_r = R::sided(R::EarLBack, R::EarRBack, side);
    let chart = C::sided(C::EarL, C::EarR, side).id();
    let back_chart = C::sided(C::EarLBack, C::EarRBack, side).id();
    let is_lobe = |t: f32| t.sin() < -0.55;
    let (cb, cu) = CONCHA_CENTRE;
    let bump = |t: f32, at_deg: f32, width_deg: f32| {
        let d = (t - at_deg.to_radians()).sin().atan2((t - at_deg.to_radians()).cos());
        (-(d / width_deg.to_radians()).powi(2)).exp()
    };
    // How far toward the front edge a slot is (0 on the helix, 1 at the front).
    let frontness = |t: f32| {
        let f = ((-t.cos() - 0.3) / 0.5).clamp(0.0, 1.0);
        f * f * (3.0 - 2.0 * f)
    };
    // A ring `inset` mm inside the outline (negative: outside), shrunk toward the concha, and
    // `drop` mm below the rim's protrusion, as (inset, drop) on the helix, at the lobe (no rim, no
    // groove there) and at the front edge (where the helix root and the tragus close the concha:
    // a plain slope); `raise` lifts the ring at the tragus (front, below the middle) and the
    // antitragus (behind and below the concha, above the lobe). Returns (back, up, out) per slot.
    let ring = |helix_at: (f32, f32), lobe_at: (f32, f32), front_at: (f32, f32), raise: (f32, f32)| -> Vec<[f32; 3]> {
        fine.iter()
            .map(|&t| {
                let (b, u, o, l) = auricle_finished(t);
                let f = frontness(t);
                let inset = mix(mix(helix_at.0, front_at.0, f), lobe_at.0, l);
                let drop = mix(mix(helix_at.1, front_at.1, f), lobe_at.1, l)
                    - raise.0 * bump(t, 200.0, 22.0)
                    - raise.1 * bump(t, 310.0, 20.0);
                let (db, du) = (b - cb, u - cu);
                let k = (1.0 - inset / db.hypot(du).max(1.0)).max(0.2);
                [cb + db * k, cu + du * k, o - drop]
            })
            .collect()
    };
    let to_cage = |pts: &[[f32; 3]]| -> Vec<Vec3> { pts.iter().map(|p| local(p[0], p[1], p[2])).collect() };
    // From the back of the rim over its crest into the scapha, up the antihelix (rings 2..5; ring 3
    // is the helix rim, whose slots name the helix landmarks).
    let rim_back = ring((-EAR_RIM_OUT, 5.0), (-0.8, 2.5), (0.5, 0.6), (0.0, 0.0));
    let antihelix_ring = ring((8.5, 1.2), (7.5, 1.6), (6.5, 1.5), (2.5, 2.0));
    // The concha wall's rim (ring 6): an ellipse around the concha, in the direction of each
    // slot's antihelix point (so the rings cannot cross), part way down from the antihelix.
    let wall: Vec<[f32; 3]> = antihelix_ring
        .iter()
        .map(|p| {
            let (db, du) = (p[0] - cb, p[1] - cu);
            let len = db.hypot(du).max(1.0e-3);
            let (ub, uu) = (db / len, du / len);
            let [back_r, front_r, up_r, down_r] = CONCHA_RADII;
            let rb = if ub >= 0.0 { back_r } else { front_r };
            let ru = if uu >= 0.0 { up_r } else { down_r };
            let r = (1.0 / ((ub / rb).powi(2) + (uu / ru).powi(2)).sqrt()).min(len - 1.5);
            [cb + ub * r, cu + uu * r, p[2] * CONCHA_RIM]
        })
        .collect();
    let floor = wall.iter().map(|p| p[2]).fold(f32::MAX, f32::min) - CONCHA_DEPTH;
    // The root, 16 slots like the rings outside it, each out from the block's centre toward its
    // slot of the back of the rim (ring 2): behind, [`ROOT_SHARE`] of the way out to it (the free
    // auricle stands on the root, the groove behind the ear runs around it); at the front,
    // [`ROOT_FRONT`] in front of it, where the helix root and the tragus grow out of the face.
    // Never closer to the concha's wall than [`ROOT_MARGIN`] (the auricle's back would pass under
    // the concha) and never farther out than [`ROOT_REACH`] of the block's outline in that
    // direction (the outline is irregular: a shallow cone to it would fold). Lifted off the head
    // by [`ROOT_LIFT`].
    let flat_hole: Vec<[f32; 3]> = hole
        .iter()
        .map(|&v| {
            let d = (cage.pos[v as usize] - centre) * 1000.0;
            [d.dot(back), d.dot(up), d.dot(out)]
        })
        .collect();
    // Where the ray from the centre in the unit direction (db, du) leaves the block's outline:
    // the distance and the outline's height off the auricle's plane there.
    let leave = |db: f32, du: f32| -> (f32, f32) {
        let n = flat_hole.len();
        let mut best = (f32::MAX, 0.0);
        for e in 0..n {
            let (p, q) = (flat_hole[e], flat_hole[(e + 1) % n]);
            let (ex, ey) = (q[0] - p[0], q[1] - p[1]);
            let den = db * ey - du * ex;
            if den.abs() < 1.0e-6 {
                continue;
            }
            // p + s e = r d  =>  r = (p x e) / (d x e), s = (p x d) / (d x e).
            let r = (p[0] * ey - p[1] * ex) / den;
            let s = (p[0] * du - p[1] * db) / den;
            if r > 0.0 && (0.0..=1.0).contains(&s) && r < best.0 {
                best = (r, mix(p[2], q[2], s));
            }
        }
        best
    };
    let radius = |p: &[f32; 3]| p[0].hypot(p[1]);
    let n_fine = fine.len();
    let root: Vec<Vec3> = fine
        .iter()
        .enumerate()
        .map(|(k, &t)| {
            let f = frontness(t);
            let (_, _, _, l) = auricle_finished(t);
            let r_wall = [k + n_fine - 1, k, k + 1].iter().map(|&j| radius(&wall[j % n_fine])).fold(0.0, f32::max);
            let rim = rim_back[k];
            let r_rim = radius(&rim).max(1.0e-3);
            let (db, du) = (rim[0] / r_rim, rim[1] / r_rim);
            let (r_hole, o_hole) = leave(db, du);
            // In front of the rim where the block's outline leaves room for it; elsewhere (behind,
            // and where the outline dents in at the top front) a groove around the root.
            let wanted = mix(ROOT_SHARE * r_rim, r_rim + ROOT_FRONT, f);
            let r = if f > 0.5 && wanted >= r_rim + 0.5 * ROOT_FRONT && wanted <= ROOT_REACH.1 * r_hole {
                wanted
            } else {
                wanted.min(ROOT_REACH.0 * r_hole).max(r_wall + ROOT_MARGIN)
            };
            // Lower toward the front over a wider arc than the rim's front edge: the block's
            // outline comes close at the top front, and a tall root there twists the skin to it.
            let wide = ((-t.cos() + 0.3) / 0.8).clamp(0.0, 1.0);
            let lift = mix(mix(ROOT_LIFT.0, ROOT_LIFT.1, wide), ROOT_LIFT.2, l);
            local(db * r, du * r, o_hole * (r / r_hole).min(1.0) + lift)
        })
        .collect();
    let rings_pts = [
        root,
        to_cage(&rim_back),
        to_cage(&ring((1.0, -0.3), (1.5, 0.0), (1.0, 0.0), (3.0, 0.0))),
        to_cage(&ring((4.5, 3.2), (4.0, 0.8), (3.5, 0.6), (3.5, 0.5))),
        to_cage(&antihelix_ring),
        to_cage(&wall),
    ];
    let tags: [&dyn Fn(f32) -> R; 6] = [
        &|_| back_r,
        &|t| if is_lobe(t) { lobe } else { back_r },
        &|t| if is_lobe(t) { lobe } else { helix },
        &|t| if is_lobe(t) { lobe } else { helix },
        &|t| if is_lobe(t) { lobe } else { antihelix },
        &|t| if is_lobe(t) { lobe } else { concha },
    ];
    let mut rings = vec![hole.to_vec()];
    for (k, pts) in rings_pts.iter().enumerate() {
        let ring = cage.add_all(pts);
        let prev = rings.last().cloned().unwrap_or_default();
        let tagf = tags[k];
        // The medial side of the auricle (up to the back of the helix rim) is its own chart.
        let ring_chart = if k <= 1 { back_chart } else { chart };
        // Tagged by the face's middle slot angle (mirror-consistent on both ears). Between rings
        // of 8 slots `i` indexes `params`; a doubling bridge and the rings of 16 index `fine`.
        let slots = if prev.len() == ring.len() && ring.len() == params.len() { &params } else { &fine };
        let n = slots.len();
        bridge_tagged(cage, &prev, &ring, &|i| {
            let (a0, a1) = (slots[i % n], slots[(i + 1) % n]);
            let mid = (a0.sin() + a1.sin()).atan2(a0.cos() + a1.cos());
            Tag::new(tagf(mid).id(), ring_chart)
        });
        rings.push(ring);
    }
    // The concha floor: a 4 x 4 grid, its middle pressed down to `floor` off the auricle's plane.
    let last = rings.last().cloned().unwrap_or_default();
    let interior = cage.cap(&last, 4, 4, 2, Tag::new(concha.id(), chart), &|s, t, p| {
        let o = (p - centre).dot(out) * 1e3;
        let w = ((s * PI).sin() * (t * PI).sin()).sqrt();
        p + out * ((mix(o, floor, w) - o) * 1e-3)
    })?;
    let concha_v = interior.get(4).copied().unwrap_or(0);
    Ok(EarLoops { rings, params, concha: concha_v })
}

#[cfg(test)]
mod tests {
    use humanart_math::Vec3;

    use super::super::cage::Cage;

    /// The faces of `cage` within `box_mm` of `centre` (mm per axis) whose normal points against
    /// one of their vertices' mean normals (folds), as (centroid - centre in mm, worst dot).
    fn folds(cage: &Cage, centre: Vec3, box_mm: Vec3) -> Vec<(Vec3, f32, usize)> {
        let face_normal = |f: usize| {
            let c = cage.corners_of(f);
            let o = cage.pos[c[0] as usize];
            (1..c.len() - 1)
                .map(|k| (cage.pos[c[k] as usize] - o).cross(cage.pos[c[k + 1] as usize] - o))
                .fold(Vec3::ZERO, |a, b| a + b)
        };
        let mut normals = vec![Vec3::ZERO; cage.pos.len()];
        for f in 0..cage.face_count() {
            let n = face_normal(f);
            for &v in cage.corners_of(f) {
                normals[v as usize] += n;
            }
        }
        let mut out = Vec::new();
        for f in 0..cage.face_count() {
            let d = (cage.centroid(f) - centre) * 1e3;
            if d.x.abs() > box_mm.x || d.y.abs() > box_mm.y || d.z.abs() > box_mm.z {
                continue;
            }
            let n = face_normal(f).normalize_or_zero();
            let worst = cage
                .corners_of(f)
                .iter()
                .map(|&v| n.dot(normals[v as usize].normalize_or_zero()))
                .fold(1.0_f32, f32::min);
            if worst < 0.0 {
                out.push((d, worst, f));
            }
        }
        out
    }

    /// Diagnostic: folded faces around the mouth in the designed cage, the fitted cage and every
    /// subdivision level of the body surface.
    #[test]
    #[ignore = "diagnostic: cargo test diag_mouth_folds -- --ignored --nocapture"]
    fn diag_mouth_folds() {
        let auricle = if std::env::var_os("DIAG_FINISHED").is_some() {
            super::super::anatomy::Auricle::Finished
        } else {
            super::super::anatomy::Auricle::Base
        };
        let s = super::super::body_surface_for(super::super::genitals::Variant::default(), auricle).unwrap();
        let rings = &s.body.loops.mouth.rings;
        let contact = &rings[5];
        let centre = contact.iter().fold(Vec3::ZERO, |a, &v| a + s.body.cage.pos[v as usize]) / contact.len() as f32;
        let mut design = s.body.cage.clone();
        design.pos.clone_from(&s.targets);
        let ring_of = |v: u32| rings.iter().position(|r| r.contains(&v));
        let bx = Vec3::new(40.0, 30.0, 40.0);
        for (label, cage) in [("design", &design), ("fitted", &s.body.cage)].into_iter().chain(
            s.levels.iter().enumerate().map(|(i, l)| (if i + 1 == s.levels.len() { "last" } else { "level" }, &l.cage)),
        ) {
            let f = folds(cage, centre, bx);
            let upper = f.iter().filter(|x| x.0.y > 2.0 && x.0.y < 15.0).count();
            let lower = f.iter().filter(|x| x.0.y < -2.0 && x.0.y > -15.0).count();
            println!("{label}: {} folded faces near the mouth ({upper} upper lip, {lower} lower lip)", f.len());
            if label == "last" {
                for (d, worst, face) in f.iter().filter(|x| x.0.y > 2.0 && x.0.x > 0.0) {
                    let n = cage.corners_of(*face).len();
                    println!("  last ({:.2},{:.2},{:.2}) mm worst {worst:.2} {n}-gon", d.x, d.y, d.z);
                }
                // The designed and fitted ring points on the left upper lip.
                let roll = &s.body.loops.mouth.roll;
                for (name, r) in [("perioral", &rings[1]), ("roll", roll), ("border", &rings[2]), ("red", &rings[3])] {
                    for &v in r {
                        let p = (s.targets[v as usize] - centre) * 1e3;
                        let q = (s.body.cage.pos[v as usize] - centre) * 1e3;
                        if p.x > 8.0 && p.x < 24.0 && p.y > 0.5 {
                            println!(
                                "  {name} design ({:.2},{:.2},{:.2}) fitted ({:.2},{:.2},{:.2})",
                                p.x, p.y, p.z, q.x, q.y, q.z
                            );
                        }
                    }
                }
            }
            for (d, worst, face) in f.iter().take(0) {
                let rs: Vec<Option<usize>> = cage.corners_of(*face).iter().map(|&v| ring_of(v)).collect();
                println!("  ({:.1},{:.1},{:.1}) mm worst {worst:.2} rings {rs:?}", d.x, d.y, d.z);
            }
        }
    }

    /// Diagnostic: folded faces around the mouth of the finished template (scan face on), with
    /// vertex normals welded across UV seams.
    #[test]
    #[ignore = "diagnostic: cargo test diag_finished_mouth_folds -- --ignored --nocapture"]
    fn diag_finished_mouth_folds() {
        let options = super::super::build::Options { lods: false, ..super::super::build::Options::default() };
        let t = super::super::build::template(&options).unwrap().template;
        let mesh = t.mesh();
        let pos = mesh.positions();
        let keys = mesh.position_keys();
        let at = |name: &str| t.landmarks().iter().find(|l| l.name == name).unwrap().position(pos).unwrap();
        let centre = (at("lip.upper.inner.06") + at("lip.lower.inner.06")) * 0.5;
        let faces: Vec<Vec<u32>> = mesh.parts()[0].faces().map(|f| mesh.face(f).to_vec()).collect();
        let normal = |c: &[u32]| {
            let o = pos[c[0] as usize];
            (1..c.len() - 1)
                .map(|k| (pos[c[k] as usize] - o).cross(pos[c[k + 1] as usize] - o))
                .fold(Vec3::ZERO, |a, b| a + b)
        };
        let mut normals = vec![Vec3::ZERO; pos.len()];
        for c in &faces {
            let n = normal(c);
            for &v in c {
                normals[keys[v as usize] as usize] += n;
            }
        }
        let (mut upper, mut lower, mut other) = (0, 0, 0);
        for c in &faces {
            let d = (c.iter().fold(Vec3::ZERO, |s, &v| s + pos[v as usize]) / c.len() as f32 - centre) * 1e3;
            if d.x.abs() > 40.0 || d.y.abs() > 30.0 || d.z.abs() > 40.0 {
                continue;
            }
            let n = normal(c).normalize_or_zero();
            if c.iter().any(|&v| n.dot(normals[keys[v as usize] as usize].normalize_or_zero()) < 0.0) {
                match d.y {
                    y if y > 2.0 && y < 15.0 => upper += 1,
                    y if y < -2.0 && y > -15.0 => lower += 1,
                    _ => other += 1,
                }
                if d.y.abs() < 15.0 {
                    println!("  folded at ({:.1},{:.1},{:.1}) mm", d.x, d.y, d.z);
                }
            }
        }
        println!("finished template: {upper} upper lip, {lower} lower lip, {other} other folded faces near the mouth");
    }

    /// What the finished left auricle measures on the body surface (last subdivision level).
    struct EarMeasures {
        /// Ear faces whose normal points against one of their corners' mean normals.
        folds: Vec<(Vec3, f32)>,
        /// Number of ear faces measured.
        faces: usize,
        /// The smallest cosine between an ear face's normal and its corners' mean normals.
        worst: f32,
        /// Length along the long axis, breadth across it (side view), protrusion off the head, mm.
        length: f32,
        breadth: f32,
        protrusion: f32,
        /// Auriculocephalic angle (the helix rim's plane against the side of the head), degrees.
        angle: f32,
        /// The rolled rim at the top of the helix, seen from the front (lateral extent), mm.
        rim: f32,
    }

    fn measure_left_ear(s: &super::super::Surface) -> EarMeasures {
        use super::super::regions::R;
        let cage = s.mesh();
        let ear_ids = [R::EarLHelix, R::EarLAntihelix, R::EarLConcha, R::EarLLobe, R::EarLBack].map(R::id);
        let is_ear = |f: usize| ear_ids.contains(&cage.tags[f].region);
        let face_normal = |f: usize| {
            let c = cage.corners_of(f);
            let o = cage.pos[c[0] as usize];
            (1..c.len() - 1)
                .map(|k| (cage.pos[c[k] as usize] - o).cross(cage.pos[c[k + 1] as usize] - o))
                .fold(Vec3::ZERO, |a, b| a + b)
        };
        let mut normals = vec![Vec3::ZERO; cage.pos.len()];
        for f in 0..cage.face_count() {
            let n = face_normal(f);
            for &v in cage.corners_of(f) {
                normals[v as usize] += n;
            }
        }
        let loops = &s.body.loops.ears[0];
        let hole = &loops.rings[0];
        let centre = hole.iter().fold(Vec3::ZERO, |a, &v| a + cage.pos[v as usize]) / hole.len() as f32;
        let tilt = 0.30_f32;
        let up = Vec3::new(0.0, tilt.cos(), -tilt.sin());
        let across = Vec3::new(0.0, tilt.sin(), tilt.cos());
        let (mut folds, mut faces, mut least) = (Vec::new(), 0, 1.0_f32);
        let mut ear_v = std::collections::HashSet::new();
        for f in 0..cage.face_count() {
            if is_ear(f) {
                faces += 1;
                let n = face_normal(f).normalize_or_zero();
                let worst = cage
                    .corners_of(f)
                    .iter()
                    .map(|&v| n.dot(normals[v as usize].normalize_or_zero()))
                    .fold(1.0_f32, f32::min);
                least = least.min(worst);
                if worst < 0.0 {
                    folds.push(((cage.centroid(f) - centre) * 1e3, worst));
                }
                // The free auricle (not the band from the head to its root) is what is measured.
                if cage.tags[f].region != R::EarLBack.id() {
                    ear_v.extend(cage.corners_of(f).iter().copied());
                }
            }
        }
        let ear_p: Vec<Vec3> = ear_v.iter().map(|&v| cage.pos[v as usize]).collect();
        let extent = |axis: Vec3| {
            let (lo, hi) =
                ear_p.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p.dot(axis)), hi.max(p.dot(axis))));
            (hi - lo) * 1e3
        };
        // The side of the head under the ear: the plane of the ear block (Newell normal).
        // The plane of a closed ring (Newell normal), facing out of the head (+x on the left).
        let plane = |ring: &[u32]| {
            let n = ring
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let (a, b) =
                        (cage.pos[v as usize] - centre, cage.pos[ring[(i + 1) % ring.len()] as usize] - centre);
                    a.cross(b)
                })
                .fold(Vec3::ZERO, |a, b| a + b)
                .normalize_or_zero();
            if n.x < 0.0 { -n } else { n }
        };
        // Protrusion: the farthest point of the auricle off the side of the head under it (the
        // ear block's plane); the angle: between that plane and the helix rim's.
        let side = plane(hole);
        let protrusion = ear_p.iter().map(|&p| (p - centre).dot(side)).fold(0.0_f32, f32::max);
        let angle = plane(&loops.rings[3]).dot(side).clamp(-1.0, 1.0).acos().to_degrees();
        // The rim's lateral extent at the top slot of rings 2 (back of the rim) to 4 (scapha).
        let top = (0..loops.rings[3].len())
            .max_by(|&a, &b| cage.pos[loops.rings[3][a] as usize].y.total_cmp(&cage.pos[loops.rings[3][b] as usize].y))
            .unwrap_or(0);
        let xs: Vec<f32> = (2..=4).map(|r| cage.pos[loops.rings[r][top] as usize].x).collect();
        let rim = (xs.iter().copied().fold(f32::MIN, f32::max) - xs.iter().copied().fold(f32::MAX, f32::min)) * 1e3;
        EarMeasures {
            folds,
            faces,
            worst: least,
            length: extent(up),
            breadth: extent(across),
            protrusion: protrusion * 1e3,
            angle,
            rim,
        }
    }

    /// `[WP-25]` todo 1.1 round 4: the finished auricle has no folded face on the body surface,
    /// meets ANSUR II ear length, breadth and protrusion (sex midpoints 61.8, 34.7, 21.8 mm; within
    /// half a standard deviation: 2.1, 1.4, 1.4 mm), stands off the head by 15 to 30 degrees and has
    /// a rolled rim at least 4 mm thick seen from the front.
    #[test]
    fn wp25_finished_ear_is_fold_free_and_ansur_sized() {
        use super::super::anatomy::{Auricle, EAR_BREADTH, EAR_LENGTH, EAR_PROTRUSION};
        let s = super::super::body_surface_for(super::super::genitals::Variant::default(), Auricle::Finished).unwrap();
        let m = measure_left_ear(&s);
        println!(
            "[WP-25] finished ear: {} faces, {} folded, least cosine {:.2}; length {:.1} mm, breadth {:.1} mm, protrusion {:.1} mm, angle {:.1} deg, rim {:.1} mm",
            m.faces,
            m.folds.len(),
            m.worst,
            m.length,
            m.breadth,
            m.protrusion,
            m.angle,
            m.rim
        );
        let loops = &s.body.loops.ears[0];
        let hole = &loops.rings[0];
        let centre = hole.iter().fold(Vec3::ZERO, |a, &v| a + s.mesh().pos[v as usize]) / hole.len() as f32;
        for (at, worst) in m.folds.iter().take(40) {
            let p = centre + *at * 1e-3;
            let mut near: Vec<(f32, usize, usize)> = loops
                .rings
                .iter()
                .enumerate()
                .flat_map(|(k, r)| r.iter().enumerate().map(move |(i, &v)| (k, i, v)))
                .map(|(k, i, v)| (s.mesh().pos[v as usize].distance(p) * 1e3, k, i))
                .collect();
            near.sort_by(|a, b| a.0.total_cmp(&b.0));
            println!(
                "  fold at ({:.1}, {:.1}, {:.1}) mm from the ear block's centre, dot {worst:.2}; near {:?}",
                at.x,
                at.y,
                at.z,
                &near[..3]
            );
        }
        assert!(m.faces > 500, "{} ear faces", m.faces);
        assert!(m.folds.is_empty(), "{} folded ear faces", m.folds.len());
        assert!((m.length - EAR_LENGTH * 1e3).abs() < 2.1, "length {}", m.length);
        assert!((m.breadth - EAR_BREADTH * 1e3).abs() < 1.4, "breadth {}", m.breadth);
        assert!((m.protrusion - EAR_PROTRUSION * 1e3).abs() < 1.4, "protrusion {}", m.protrusion);
        assert!((15.0..=30.0).contains(&m.angle), "angle {}", m.angle);
        assert!(m.rim >= 4.0, "rim {}", m.rim);
    }
}
