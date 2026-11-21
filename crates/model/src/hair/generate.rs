//! The CPU reference of the generation modifiers `[HAIR-02]`, mirrored operation for operation by
//! `shaders/groom/generate.wgsl` (`humanart-render::groom` runs it on the GPU and its tests compare
//! the two `[AC-WP17-2]`).
//!
//! Three stages, each a pure function of its inputs:
//!
//! 1. **Shape** ([`shape_strand`], one strand at a time, rest pose): blend the strand's guides
//!    (resampled to the groom's point count) in the root frame, or in world directions turned into
//!    the root frame (`GenerateInput::world_curves`, what [`generate_groom`] and the template's
//!    grooms use); set the length (guide length x
//!    `length.scale` x length map x random variation) by resampling along the curve, extrapolating
//!    past its end; turn the strand about the root normal by the direction map, the parting, the
//!    cowlicks and the per-strand noise; add the curl helix, the planar wave, frizz and flyaways;
//!    move to world space with the rest root frame.
//! 2. **Finish** ([`finish_strand`], needs every strand's shape): pull towards the clump centre
//!    strand (`clump.strength` x clump map, keeping `clump.spread` of the distance at the root and
//!    `clump.tip_spread` at the tip); push points out of the head along the distance gradient to
//!    `collision.offset` above the skin; store the offsets in the rest root frame as `f16`, and the
//!    per-strand attributes (length, root and tip diameter, colour id, grey, stiffness, curl phase).
//! 3. **Attach** ([`attach`], every frame the skin moves): the stored offsets in the deformed root
//!    frames, so the strands ride the skin.
//!
//! Random values come from [`hash01`] of the strand key with fixed salts ([`salt`]).

use std::f32::consts::{PI, TAU};

use humanart_math::{F16, Vec3, Vec4};

use super::collide::SdfGrid;
use super::error::{HairError, HairResult};
use super::params::GroomParams;
use super::placement::GroomLayout;
use super::strands::{Guides, MAX_STRAND_POINTS, Strands};
use super::surface::{HairSurface, RootFrame, normalize_or};
use super::{hash_u32, hash01};

/// The salts of the per-strand random values (shared with WGSL).
pub mod salt {
    /// Length variation.
    pub const LENGTH: u32 = 1;
    /// Turn noise.
    pub const NOISE: u32 = 2;
    /// Curl radius variation.
    pub const CURL_RADIUS: u32 = 3;
    /// Curl frequency variation.
    pub const CURL_FREQUENCY: u32 = 4;
    /// Curl and wave phase.
    pub const PHASE: u32 = 5;
    /// Flyaway choice.
    pub const FLYAWAY: u32 = 6;
    /// Flyaway direction x.
    pub const FLYAWAY_X: u32 = 7;
    /// Flyaway direction y.
    pub const FLYAWAY_Y: u32 = 8;
    /// Diameter variation.
    pub const THICKNESS: u32 = 9;
    /// Colour id.
    pub const COLOR: u32 = 10;
    /// Grey choice.
    pub const GREY: u32 = 11;
    /// Stiffness variation.
    pub const STIFFNESS: u32 = 12;
    /// Frizz of point `k` uses salts `FRIZZ + 3k .. FRIZZ + 3k + 2`.
    pub const FRIZZ: u32 = 16;
}

/// Wraps an angle to `[-pi, pi)`.
#[inline]
pub fn wrap_angle(a: f32) -> f32 {
    a - TAU * ((a + PI) / TAU).floor()
}

/// `fract(x)` as WGSL defines it: `x - floor(x)`.
#[inline]
fn fract(x: f32) -> f32 {
    x - x.floor()
}

/// The world points of every strand after stage 1 and 2 plus the stored strands.
#[derive(Clone, Debug, PartialEq)]
pub struct Generated {
    /// Stage 1 output: world points, `points` per strand.
    pub shape: Vec<Vec3>,
    /// Stage 2 output before `f16` storage: world points.
    pub world: Vec<Vec3>,
    /// The stored strands.
    pub strands: Strands,
}

/// Everything the modifiers read: layout, guides resampled to the point count, parameters, the
/// rest root frames and the optional distance field.
#[derive(Clone, Copy, Debug)]
pub struct GenerateInput<'a> {
    /// The strand layout.
    pub layout: &'a GroomLayout,
    /// The guides resampled to `params.point_count()` points: offsets in each guide's root frame
    /// ([`Guides::resampled`]), or world-space offsets from each guide's root with
    /// `world_curves` ([`Guides::resampled_world`]).
    pub curves: &'a [Vec3],
    /// The curves are world-space offsets: the strand blends its guides in world directions and
    /// turns the blend into its own root frame, so guides whose root frames' tangents disagree
    /// (across a UV seam) blend into the shape they have in the world instead of a bent one.
    pub world_curves: bool,
    /// The parameters.
    pub params: &'a GroomParams,
    /// The rest root frame of every strand ([`root_frames`]).
    pub frames: &'a [RootFrame],
    /// The head's signed distance field (no collision without).
    pub sdf: Option<&'a SdfGrid>,
}

/// The rest root frame of every strand of `layout` on `surface`.
pub fn root_frames(layout: &GroomLayout, surface: &HairSurface) -> HairResult<Vec<RootFrame>> {
    (0..layout.len())
        .map(|i| {
            surface
                .frame(layout.triangle[i], layout.bary[i])
                .ok_or_else(|| HairError::invalid("strand", i, "root triangle out of range"))
        })
        .collect()
}

/// Stage 1 for strand `i`: writes its `P` world points to `out` (`out.len()` is `P`).
pub fn shape_strand(input: &GenerateInput<'_>, i: usize, out: &mut [Vec3]) {
    let p = input.params;
    let l = input.layout;
    let n = out.len();
    if n < 2 || i >= l.len() {
        return;
    }
    let frame = input.frames[i];
    let key = l.key[i];
    let factors = l.factors[i];
    let mut base = [Vec3::ZERO; MAX_STRAND_POINTS as usize];
    let mut res = [Vec3::ZERO; MAX_STRAND_POINTS as usize];
    // 1. Blend the guides (in the root frame, or in world directions turned into it).
    for (k, b) in base.iter_mut().enumerate().take(n) {
        let mut acc = Vec3::ZERO;
        for j in 0..4 {
            let g = l.guides[i][j] as usize;
            acc += input.curves.get(g * n + k).copied().unwrap_or(Vec3::ZERO) * l.weights[i][j];
        }
        *b = if input.world_curves { frame.dir_to_local(acc) } else { acc };
    }
    // 2. Length: resample along the curve to the target length, extrapolating the last segment.
    let mut lb = 0.0_f32;
    for k in 1..n {
        lb += (base[k] - base[k - 1]).length();
    }
    let variation = 1.0 + p.length_variation * (2.0 * hash01(key, salt::LENGTH) - 1.0);
    let length = (lb * p.length_scale * factors[0] * variation).max(0.0);
    let (mut seg, mut acc) = (0_usize, 0.0_f32);
    for (k, r) in res.iter_mut().enumerate().take(n) {
        let s = length * (k as f32 / (n - 1) as f32);
        while seg < n - 2 && acc + (base[seg + 1] - base[seg]).length() < s {
            acc += (base[seg + 1] - base[seg]).length();
            seg += 1;
        }
        let d = base[seg + 1] - base[seg];
        let len = d.length();
        let t = if len > 1e-12 { (s - acc) / len } else { 0.0 };
        *r = base[seg] + d * t;
    }
    // 3. Turn about the root normal.
    let tip = res[n - 1];
    let heading = tip.y.atan2(tip.x);
    let horizontal = (tip.x * tip.x + tip.y * tip.y).sqrt();
    let mut delta = 0.0_f32;
    if horizontal > 1e-6 {
        let [fx, fy] = l.direction[i];
        let m = (fx * fx + fy * fy).sqrt();
        if m > 1e-6 {
            delta += wrap_angle(fy.atan2(fx) - heading) * m.min(1.0) * p.direction_strength;
        }
        if p.parting_strength > 0.0 && p.parting_width > 0.0 {
            let part = p.parting();
            let d = (frame.origin - part.point).dot(part.normal);
            let ad = d.abs();
            if ad < part.width {
                let side = if d >= 0.0 { 1.0 } else { -1.0 };
                let lx = part.normal.dot(frame.t) * side;
                let ly = part.normal.dot(frame.b) * side;
                if lx * lx + ly * ly > 1e-8 {
                    let f = 1.0 - ad / part.width;
                    delta += wrap_angle(ly.atan2(lx) - heading) * (part.strength * f * f);
                }
            }
        }
        for c in 0..4 {
            let Some(cw) = p.cowlick(c) else { continue };
            if cw.radius <= 0.0 {
                continue;
            }
            let rel = frame.origin - cw.centre;
            let dist = rel.length();
            if dist < cw.radius {
                let sense = if cw.strength < 0.0 { -1.0 } else { 1.0 };
                let swirl = frame.n.cross(rel) * sense;
                let lx = swirl.dot(frame.t);
                let ly = swirl.dot(frame.b);
                if lx * lx + ly * ly > 1e-12 {
                    let f = 1.0 - dist / cw.radius;
                    delta += wrap_angle(ly.atan2(lx) - heading) * (cw.strength.abs() * f * f);
                }
            }
        }
    }
    delta += p.noise * (2.0 * hash01(key, salt::NOISE) - 1.0);
    let (sn, cs) = (delta.sin(), delta.cos());
    // 4. Curl, wave, frizz, flyaways about the strand axis.
    let a = normalize_or(tip, Vec3::Z);
    let b1 = normalize_or(a.cross(Vec3::Z), Vec3::X);
    let b2 = a.cross(b1);
    let radius = p.curl_radius * factors[2] * (1.0 + p.curl_variation * (2.0 * hash01(key, salt::CURL_RADIUS) - 1.0));
    let frequency = p.curl_frequency * (1.0 + p.curl_variation * (2.0 * hash01(key, salt::CURL_FREQUENCY) - 1.0));
    let phase = hash01(key, salt::PHASE);
    let flyaway = hash01(key, salt::FLYAWAY) < p.flyaway_fraction;
    let fly = Vec3::new(2.0 * hash01(key, salt::FLYAWAY_X) - 1.0, 2.0 * hash01(key, salt::FLYAWAY_Y) - 1.0, 1.0);
    let frizz = p.frizz * factors[3];
    for k in 0..n {
        let t = k as f32 / (n - 1) as f32;
        let s = length * t;
        let ramp = (t * 4.0).min(1.0);
        let mut q = res[k];
        let x = fract(frequency * s + phase) * TAU;
        q += (b1 * x.cos() + b2 * x.sin()) * (radius * ramp);
        let w = fract(p.wave_frequency * s + phase) * TAU;
        q += b1 * (p.wave_amplitude * ramp * w.sin());
        if k > 0 {
            let base_salt = salt::FRIZZ + 3 * k as u32;
            let r = Vec3::new(
                2.0 * hash01(key, base_salt) - 1.0,
                2.0 * hash01(key, base_salt + 1) - 1.0,
                2.0 * hash01(key, base_salt + 2) - 1.0,
            );
            q += r * (frizz * t * t);
            if flyaway {
                q += fly * (p.flyaway_strength * t * t);
            }
        }
        let rotated = Vec3::new(q.x * cs - q.y * sn, q.x * sn + q.y * cs, q.z);
        out[k] = frame.to_world(rotated);
    }
}

/// The per-strand attributes of stage 2.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrandAttributes {
    /// Polyline length after clumping and collision.
    pub length: f32,
    /// Root diameter.
    pub width_root: f32,
    /// Tip diameter.
    pub width_tip: f32,
    /// Colour variation id.
    pub color_id: u16,
    /// 0 pigmented, 255 white.
    pub grey: u8,
    /// Stiffness hint.
    pub stiffness: f32,
    /// Curl phase.
    pub curl_phase: f32,
}

/// Stage 2 for strand `i`: reads every strand's stage-1 points in `shape` (`P` per strand),
/// writes strand `i`'s final world points to `out` and returns its attributes.
pub fn finish_strand(input: &GenerateInput<'_>, shape: &[Vec3], i: usize, out: &mut [Vec3]) -> StrandAttributes {
    let p = input.params;
    let l = input.layout;
    let n = out.len();
    let frame = input.frames[i];
    let key = l.key[i];
    let centre = l.clump[i] as usize;
    let strength = p.clump_strength * l.factors[i][1];
    for k in 0..n {
        let mut q = shape[i * n + k];
        if centre != i && strength > 0.0 {
            let t = k as f32 / (n - 1) as f32;
            let smooth = t * t * (3.0 - 2.0 * t);
            let keep = p.clump_spread + (p.clump_tip_spread - p.clump_spread) * t;
            let pull = strength * smooth * (1.0 - keep);
            let c = shape[centre * n + k];
            q += (c - q) * pull;
        }
        if k > 0
            && let Some(sdf) = input.sdf
        {
            let (d, g) = sdf.sample(q);
            if d < p.collision_offset {
                q += normalize_or(g, frame.n) * (p.collision_offset - d);
            }
        }
        out[k] = q;
    }
    let mut length = 0.0_f32;
    for k in 1..n {
        length += (out[k] - out[k - 1]).length();
    }
    let width_root = p.thickness
        * input.layout.width_factor(i)
        * (1.0 + p.thickness_variation * (2.0 * hash01(key, salt::THICKNESS) - 1.0));
    StrandAttributes {
        length,
        width_root,
        width_tip: width_root * p.tip_ratio,
        color_id: (hash_u32(key ^ hash_u32(salt::COLOR)) & 0xFFFF) as u16,
        grey: if hash01(key, salt::GREY) < p.grey_fraction { 255 } else { 0 },
        stiffness: (p.stiffness * (1.0 + 0.2 * (2.0 * hash01(key, salt::STIFFNESS) - 1.0))).clamp(0.0, 1.0),
        curl_phase: hash01(key, salt::PHASE),
    }
}

/// Runs stages 1 and 2 for every strand.
pub fn generate(input: &GenerateInput<'_>) -> HairResult<Generated> {
    let n = input.params.point_count() as usize;
    let count = input.layout.len();
    if input.frames.len() != count {
        return Err(HairError::invalid("groom", 0, "one root frame per strand needed"));
    }
    let guide_count = input.curves.len() / n;
    if input.curves.len() != guide_count * n || guide_count == 0 {
        return Err(HairError::invalid("groom", 0, "guide curves are not resampled to the point count"));
    }
    if let Some(i) = input.layout.guides.iter().position(|g| g.iter().any(|&x| x as usize >= guide_count)) {
        return Err(HairError::invalid("strand", i, "guide index out of range"));
    }
    if let Some(i) = input.layout.clump.iter().position(|&c| c as usize >= count) {
        return Err(HairError::invalid("strand", i, "clump centre out of range"));
    }
    input.params.check()?;
    let mut shape = vec![Vec3::ZERO; count * n];
    for i in 0..count {
        shape_strand(input, i, &mut shape[i * n..(i + 1) * n]);
    }
    let mut world = vec![Vec3::ZERO; count * n];
    let mut strands = Strands::new(n as u32)?;
    strands.offsets.reserve(count * n * 3);
    for i in 0..count {
        let a = finish_strand(input, &shape, i, &mut world[i * n..(i + 1) * n]);
        let frame = input.frames[i];
        for &q in &world[i * n..(i + 1) * n] {
            let local = frame.to_local(q);
            strands.offsets.extend([F16::from_f32(local.x), F16::from_f32(local.y), F16::from_f32(local.z)]);
        }
        strands.triangle.push(input.layout.triangle[i]);
        strands.bary.push(input.layout.bary[i]);
        strands.length.push(a.length);
        strands.width_root.push(a.width_root);
        strands.width_tip.push(a.width_tip);
        strands.color_id.push(a.color_id);
        strands.grey.push(a.grey);
        strands.stiffness.push(a.stiffness);
        strands.curl_phase.push(a.curl_phase);
        strands.clump.push(input.layout.clump[i]);
    }
    Ok(Generated { shape, world, strands })
}

/// Convenience: resamples `guides` in world directions ([`Guides::resampled_world`]), builds the
/// rest frames and runs [`generate`] with the world blend.
pub fn generate_groom(
    surface: &HairSurface,
    layout: &GroomLayout,
    guides: &Guides,
    params: &GroomParams,
    sdf: Option<&SdfGrid>,
) -> HairResult<Generated> {
    let curves = guides.resampled_world(params.point_count(), surface)?;
    let frames = root_frames(layout, surface)?;
    generate(&GenerateInput { layout, curves: &curves, params, frames: &frames, sdf, world_curves: true })
}

/// Stage 3: the world points of `strands` on the deformed surface (`positions`, `normals` replace
/// the surface's vertices), `w` the diameter interpolated from root to tip. `out` holds
/// `points` entries per strand. Allocation-free.
pub fn attach(
    strands: &Strands,
    surface: &HairSurface,
    positions: &[Vec3],
    normals: &[Vec3],
    out: &mut [Vec4],
) -> HairResult<()> {
    let n = strands.points as usize;
    if out.len() != strands.len() * n {
        return Err(HairError::invalid("attach", 0, "output length is not strands x points"));
    }
    for i in 0..strands.len() {
        let frame = surface
            .frame_on(strands.triangle[i], strands.bary[i], positions, normals)
            .ok_or_else(|| HairError::invalid("strand", i, "root not on the surface"))?;
        for k in 0..n {
            let t = k as f32 / (n - 1) as f32;
            let w = strands.width_root[i] + (strands.width_tip[i] - strands.width_root[i]) * t;
            out[i * n + k] = frame.to_world(strands.offset(i, k)).extend(w);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::collide::{TriangleGrid, penetration};
    use crate::hair::params::GroomMaps;
    use crate::hair::placement::{ScalpMask, place};
    use crate::hair::surface::ProxyHead;

    /// Guides growing along the normal then bending down, on a coarse scalp.
    pub(crate) fn guides_on(head: &ProxyHead, mask: &ScalpMask, count: usize, length: f32) -> Guides {
        let mut g = Guides::new(12).unwrap();
        let mut rng = humanart_core::rng::Rng::new(11);
        let tris = mask.triangles();
        while g.len() < count {
            let t = tris[rng.index(tris.len())];
            let bary = [0.33, 0.33];
            let f = head.surface.frame(t, bary).unwrap();
            let mut world = vec![f.origin];
            let mut dir = f.n;
            let mut p = f.origin;
            for _ in 0..11 {
                dir = (dir + Vec3::new(0.0, -0.5, 0.0)).normalize();
                p += dir * (length / 11.0);
                world.push(p);
            }
            let local: Vec<Vec3> = world.iter().map(|&w| f.to_local(w)).collect();
            g.push(t, bary, &local).unwrap();
        }
        g
    }

    #[test]
    fn wrap_angle_range() {
        for a in [-10.0_f32, -PI, 0.0, PI, 7.0] {
            let w = wrap_angle(a);
            assert!((-PI..PI).contains(&w) || (w - PI).abs() < 1e-6);
            assert!(((w - a) / TAU - ((w - a) / TAU).round()).abs() < 1e-5);
        }
    }

    #[test]
    fn ac_wp17_1_generated_groom_is_deterministic_rooted_and_outside_the_head() {
        let head = ProxyHead::new(48).unwrap();
        let mask = ScalpMask::new(&head.surface, head.scalp.clone()).unwrap();
        let guides = guides_on(&head, &mask, 150, 0.15);
        let params = GroomParams {
            strand_count: 4000.0,
            curl_radius: 0.004,
            wave_amplitude: 0.002,
            parting_strength: 0.8,
            parting_y: 1.6,
            cowlick0_radius: 0.03,
            cowlick0_strength: 0.7,
            cowlick0_y: 1.71,
            ..GroomParams::default()
        };
        let grid = TriangleGrid::new(head.surface.positions(), head.surface.triangles(), 0.01).unwrap();
        let sdf = SdfGrid::build(&grid, 0.004, 0.02).unwrap();
        let layout = place(&head.surface, &mask, &guides, &params, &GroomMaps::default(), 7).unwrap();
        let a = generate_groom(&head.surface, &layout, &guides, &params, Some(&sdf)).unwrap();
        let b = generate_groom(&head.surface, &layout, &guides, &params, Some(&sdf)).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.strands.len(), 4000);
        a.strands.check(Some(&head.surface)).unwrap();
        // Roots are exactly on the scalp: point 0 is the root.
        let n = 16;
        for i in 0..a.strands.len() {
            let root = head.surface.point_at(layout.triangle[i], layout.bary[i]).unwrap();
            assert!((a.world[i * n] - root).length() < 1e-5);
        }
        // No strand point inside the head beyond 0.5 mm.
        let report = penetration(&grid, Some(&sdf), &a.world, 5e-4);
        assert_eq!(report.inside, 0, "{report:?}");
        // Without collision some guide-following points would be inside: the check has teeth.
        let c = generate_groom(&head.surface, &layout, &guides, &params, None).unwrap();
        assert!(penetration(&grid, Some(&sdf), &c.world, 5e-4).inside > 0);
        // Lengths follow the guides (0.15 m +- 10 %, curl and wave lengthen the polyline).
        let mean = a.strands.length.iter().sum::<f32>() / 4000.0;
        assert!(mean > 0.12 && mean < 0.25, "{mean}");
    }

    #[test]
    fn modifiers_do_what_they_say() {
        let head = ProxyHead::new(32).unwrap();
        let mask = ScalpMask::new(&head.surface, head.scalp.clone()).unwrap();
        let guides = guides_on(&head, &mask, 60, 0.1);
        let base = GroomParams {
            strand_count: 500.0,
            noise: 0.0,
            frizz: 0.0,
            flyaway_fraction: 0.0,
            clump_strength: 0.0,
            length_variation: 0.0,
            ..GroomParams::default()
        };
        let layout = place(&head.surface, &mask, &guides, &base, &GroomMaps::default(), 3).unwrap();
        let run = |p: &GroomParams| generate_groom(&head.surface, &layout, &guides, p, None).unwrap();
        let plain = run(&base);
        // length.scale halves the length.
        let half = run(&GroomParams { length_scale: 0.5, ..base });
        let ratio = half.strands.length[5] / plain.strands.length[5];
        assert!((ratio - 0.5).abs() < 0.01, "{ratio}");
        // Clumping pulls tips together.
        let spread = |g: &Generated| {
            let tips: Vec<Vec3> = (0..500).map(|i| g.world[i * 16 + 15]).collect();
            let mut d = 0.0;
            for i in 0..500 {
                d += (tips[i] - tips[layout.clump[i] as usize]).length();
            }
            d / 500.0
        };
        let clumped = run(&GroomParams { clump_strength: 1.0, clump_tip_spread: 0.0, ..base });
        assert!(spread(&clumped) < 0.2 * spread(&plain));
        // Curl moves mid-strand points off the uncurled curve by about the radius.
        let curled = run(&GroomParams { curl_radius: 0.005, curl_variation: 0.0, ..base });
        let d = (curled.world[8] - plain.world[8]).length();
        assert!(d > 0.002 && d < 0.0101, "{d}");
        // Grey fraction 1 makes every strand white, 0 none.
        assert!(run(&GroomParams { grey_fraction: 1.0, ..base }).strands.grey.iter().all(|&g| g == 255));
        assert!(plain.strands.grey.iter().all(|&g| g == 0));
        // A cowlick turns strands near it, not far away.
        let centre = head.surface_point(Vec3::Y);
        let cow = run(&GroomParams {
            cowlick0_x: centre.x,
            cowlick0_y: centre.y,
            cowlick0_z: centre.z,
            cowlick0_radius: 0.04,
            cowlick0_strength: 1.0,
            ..base
        });
        let mut near_moved = 0;
        for i in 0..500 {
            let moved = (cow.world[i * 16 + 15] - plain.world[i * 16 + 15]).length() > 1e-4;
            let near = (plain.world[i * 16] - centre).length() < 0.04;
            if moved {
                assert!(near);
                near_moved += 1;
            }
        }
        assert!(near_moved > 0);
    }

    #[test]
    fn attach_rides_the_deformed_surface() {
        let head = ProxyHead::new(24).unwrap();
        let mask = ScalpMask::new(&head.surface, head.scalp.clone()).unwrap();
        let guides = guides_on(&head, &mask, 30, 0.08);
        let params = GroomParams { strand_count: 200.0, ..GroomParams::default() };
        let layout = place(&head.surface, &mask, &guides, &params, &GroomMaps::default(), 3).unwrap();
        let g = generate_groom(&head.surface, &layout, &guides, &params, None).unwrap();
        let mut out = vec![Vec4::ZERO; 200 * 16];
        attach(&g.strands, &head.surface, head.surface.positions(), head.surface.normals(), &mut out).unwrap();
        // At rest, attach reproduces the generated points within f16 precision.
        for (a, b) in out.iter().zip(&g.world) {
            assert!((a.truncate() - *b).length() < 3e-4);
        }
        // A translated surface carries the strands rigidly.
        let moved: Vec<Vec3> = head.surface.positions().iter().map(|&p| p + Vec3::new(0.1, 0.0, 0.0)).collect();
        let mut out2 = vec![Vec4::ZERO; 200 * 16];
        attach(&g.strands, &head.surface, &moved, head.surface.normals(), &mut out2).unwrap();
        for (a, b) in out.iter().zip(&out2) {
            assert!((b.truncate() - a.truncate() - Vec3::new(0.1, 0.0, 0.0)).length() < 1e-5);
        }
        assert!(attach(&g.strands, &head.surface, &moved, head.surface.normals(), &mut out2[1..]).is_err());
    }
}
