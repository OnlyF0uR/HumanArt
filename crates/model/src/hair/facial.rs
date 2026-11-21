//! Facial and body hair generators `[HAIR-05]`: beard, moustache, goatee, stubble, and body hair
//! (chest, arms, legs).
//!
//! These are ordinary grooms (guides interpolated into strands) of short hair: the generators
//! grow guides on an area of the skin with a direction field, and set the parameters (length,
//! density, curl, thickness). The parameters are plain physical settings (lengths, counts per
//! area, angles); none is derived from sex, age or ancestry. Density varies over the area through
//! the groom's density map ([`super::GroomMaps::density`], painted or computed), which [`place`]
//! reads at every root.
//!
//! Facial styles take [`FaceAnchors`] (the mouth centre, the chin and the face's up direction)
//! to cut the beard area into a moustache band or a goatee and to comb the hair: down the face,
//! and outward from the mouth for the moustache. Body hair follows one flow direction per call
//! (down the chest, toward the hand along an arm, down a leg).
//!
//! Lengths and counts are styling choices for these generators, not measurements.
//!
//! [`place`]: super::placement::place

use humanart_core::rng::Rng;
use humanart_math::Vec3;

use super::collide::SdfGrid;
use super::error::{HairError, HairResult};
use super::params::GroomParams;
use super::placement::ScalpMask;
use super::presets::{GrowSpec, PresetGroom, grow_guide};
use super::strands::{Guides, MAX_GUIDES, MAX_STRANDS};
use super::surface::{HairSurface, normalize_or};

/// A facial hair style's id and description.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FacialStyle {
    /// The stable id.
    pub name: &'static str,
    /// What it looks like.
    pub doc: &'static str,
}

/// Every facial hair style (stable ids).
pub const FACIAL_STYLES: &[FacialStyle] = &[
    FacialStyle { name: "full_beard", doc: "A full beard about 4 cm long over the whole beard area." },
    FacialStyle { name: "short_beard", doc: "A trimmed beard about 8 mm long over the whole beard area." },
    FacialStyle { name: "stubble", doc: "About 1 mm of stubble over the whole beard area." },
    FacialStyle { name: "moustache", doc: "A moustache about 12 mm long above the mouth." },
    FacialStyle { name: "goatee", doc: "A moustache and a chin beard about 2.5 cm long." },
];

/// Where the mouth and chin are, for cutting and combing facial hair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceAnchors {
    /// The centre of the mouth (between the lips), world space.
    pub mouth: Vec3,
    /// The chin point, world space.
    pub chin: Vec3,
    /// The face's up direction (`+Y` for an upright head).
    pub up: Vec3,
}

/// Whether a triangle centroid `c` is in the moustache band above the mouth.
fn in_moustache(c: Vec3, a: &FaceAnchors) -> bool {
    let d = c - a.mouth;
    let height = d.dot(a.up);
    (0.003..0.022).contains(&height) && d.reject_from(a.up).length() < 0.035
}

/// The triangles of `area` a style covers.
fn style_triangles(style: &str, surface: &HairSurface, area: &ScalpMask, anchors: &FaceAnchors) -> Vec<u32> {
    let centroid = |t: u32| surface.corners(t).map_or(Vec3::splat(f32::NAN), |[a, b, c]| (a + b + c) * (1.0 / 3.0));
    area.triangles()
        .iter()
        .copied()
        .filter(|&t| {
            let c = centroid(t);
            match style {
                "moustache" => in_moustache(c, anchors),
                "goatee" => in_moustache(c, anchors) || (c - anchors.chin).length() < 0.035,
                _ => true,
            }
        })
        .collect()
}

/// Grows facial hair `style` on the beard `area` with `guide_count` guides. `sdf` keeps long
/// beards outside the face.
pub fn facial_groom(
    style: &str,
    surface: &HairSurface,
    area: &ScalpMask,
    anchors: &FaceAnchors,
    guide_count: usize,
    sdf: Option<&SdfGrid>,
    seed: u64,
) -> HairResult<PresetGroom> {
    let spec = FACIAL_STYLES.iter().find(|s| s.name == style).ok_or_else(|| HairError::Param {
        name: style.chars().take(64).collect(),
        what: "unknown facial hair style",
    })?;
    if guide_count == 0 || guide_count > MAX_GUIDES {
        return Err(HairError::limit("facial guide count", MAX_GUIDES as u64, guide_count as u64));
    }
    if !(anchors.mouth.is_finite() && anchors.chin.is_finite() && anchors.up.is_finite()) {
        return Err(HairError::invalid("face anchors", 0, "not finite"));
    }
    let anchors = FaceAnchors { up: normalize_or(anchors.up, Vec3::Y), ..*anchors };
    let mask = ScalpMask::new(surface, style_triangles(spec.name, surface, area, &anchors))?;
    if mask.triangles().is_empty() || mask.area() <= 0.0 {
        return Err(HairError::invalid("facial hair", 0, "the style's area is empty"));
    }
    let (length, points, per_cm2) = match spec.name {
        "full_beard" => (0.04, 16_u32, 60.0_f32),
        "short_beard" => (0.008, 8, 60.0),
        "stubble" => (0.001, 4, 60.0),
        "moustache" => (0.012, 8, 70.0),
        _ => (0.025, 12, 60.0),
    };
    let mut params = GroomParams {
        strand_count: (mask.area() * 10_000.0 * per_cm2).min(MAX_STRANDS as f32).floor(),
        points: points as f32,
        thickness: 0.000_1,
        tip_ratio: if spec.name == "stubble" { 0.9 } else { 0.4 },
        noise: 0.25,
        length_variation: 0.25,
        clump_count: 0.0,
        clump_strength: 0.0,
        hairline_softness: 0.004,
        collision_offset: 0.001,
        ..GroomParams::default()
    };
    if matches!(spec.name, "full_beard" | "goatee") {
        params.curl_radius = 0.002;
        params.curl_frequency = 80.0;
        params.frizz = 0.001;
        params.clump_count = 150.0;
        params.clump_strength = 0.3;
    }
    let mut rng = Rng::new(seed);
    let mut guides = Guides::new(points)?;
    let mut attempts = 0usize;
    while guides.len() < guide_count && attempts < guide_count.saturating_mul(8) {
        attempts += 1;
        let Some((t, bary)) = mask.sample_root(&mut rng) else { break };
        let Some(frame) = surface.frame(t, bary) else { continue };
        let d = frame.origin - anchors.mouth;
        let side = d.reject_from(anchors.up).reject_from(frame.n);
        let comb = if spec.name == "moustache" || (spec.name == "goatee" && in_moustache(frame.origin, &anchors)) {
            -anchors.up + normalize_or(side, Vec3::ZERO) * 0.6
        } else {
            -anchors.up
        };
        let grow = GrowSpec {
            length: length * rng.range_f32(0.85, 1.15),
            comb,
            comb_rate: 4.0 / length,
            gravity: if length > 0.02 { 10.0 } else { 0.0 },
            offset: 0.001,
        };
        // Stubble is shorter than the push-out distance: it grows free.
        let world = grow_guide(&frame, &grow, points, if length > 0.005 { sdf } else { None });
        let local: Vec<Vec3> = world.iter().map(|&w| frame.to_local(w)).collect();
        guides.push(t, bary, &local)?;
    }
    if guides.is_empty() {
        return Err(HairError::invalid("facial hair", 0, "no guide could be placed"));
    }
    Ok(PresetGroom { name: spec.name, guides, params, area: mask.triangles().to_vec(), part: None })
}

/// Settings of [`body_hair`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyHairSpec {
    /// Mean hair length, metres.
    pub length: f32,
    /// Strands per square centimetre where the density map is 1.
    pub density: f32,
    /// The world direction the hair lies toward.
    pub flow: Vec3,
    /// Angle between a hair and the skin at the root, radians.
    pub lift: f32,
    /// Curl helix radius, metres.
    pub curl_radius: f32,
    /// Fibre diameter at the root, metres.
    pub thickness: f32,
}

/// The body hair presets by region (stable ids). `flow` is `−Y`: the caller turns it toward the
/// hand for an arm that is not hanging.
pub const BODY_REGIONS: &[(&str, BodyHairSpec)] = &[
    (
        "chest",
        BodyHairSpec {
            length: 0.02,
            density: 8.0,
            flow: Vec3::new(0.0, -1.0, 0.0),
            lift: 0.35,
            curl_radius: 0.002,
            thickness: 0.000_06,
        },
    ),
    (
        "arm",
        BodyHairSpec {
            length: 0.01,
            density: 10.0,
            flow: Vec3::new(0.0, -1.0, 0.0),
            lift: 0.2,
            curl_radius: 0.0,
            thickness: 0.000_04,
        },
    ),
    (
        "leg",
        BodyHairSpec {
            length: 0.015,
            density: 10.0,
            flow: Vec3::new(0.0, -1.0, 0.0),
            lift: 0.25,
            curl_radius: 0.001,
            thickness: 0.000_05,
        },
    ),
];

/// The body hair preset of `region`.
#[must_use]
pub fn body_region(region: &str) -> Option<BodyHairSpec> {
    BODY_REGIONS.iter().find(|(name, _)| *name == region).map(|&(_, spec)| spec)
}

/// Grows body hair on `area` with `guide_count` guides.
pub fn body_hair(
    surface: &HairSurface,
    area: &ScalpMask,
    spec: &BodyHairSpec,
    guide_count: usize,
    seed: u64,
) -> HairResult<PresetGroom> {
    if guide_count == 0 || guide_count > MAX_GUIDES {
        return Err(HairError::limit("body hair guide count", MAX_GUIDES as u64, guide_count as u64));
    }
    let finite = [spec.length, spec.density, spec.lift, spec.curl_radius, spec.thickness].iter().all(|v| v.is_finite());
    if !finite || !spec.flow.is_finite() || spec.length <= 0.0 || spec.density < 0.0 {
        return Err(HairError::Param { name: "body hair".into(), what: "length, density or flow not usable" });
    }
    if area.triangles().is_empty() || area.area() <= 0.0 {
        return Err(HairError::invalid("body hair", 0, "the area is empty"));
    }
    let params = GroomParams {
        strand_count: (area.area() * 10_000.0 * spec.density).min(MAX_STRANDS as f32).floor(),
        points: 6.0,
        thickness: spec.thickness.clamp(0.000_001, 0.000_5),
        tip_ratio: 0.3,
        noise: 0.3,
        length_variation: 0.3,
        clump_count: 0.0,
        clump_strength: 0.0,
        curl_radius: spec.curl_radius.clamp(0.0, 0.05),
        curl_frequency: 120.0,
        frizz: 0.000_3,
        flyaway_fraction: 0.0,
        hairline_softness: 0.01,
        collision_offset: 0.000_5,
        ..GroomParams::default()
    };
    params.check()?;
    let flow = normalize_or(spec.flow, -Vec3::Y);
    let mut rng = Rng::new(seed);
    let mut guides = Guides::new(6)?;
    let mut attempts = 0usize;
    while guides.len() < guide_count && attempts < guide_count.saturating_mul(8) {
        attempts += 1;
        let Some((t, bary)) = area.sample_root(&mut rng) else { break };
        let Some(frame) = surface.frame(t, bary) else { continue };
        let length = spec.length * rng.range_f32(0.8, 1.2);
        let flat = normalize_or(flow.reject_from(frame.n), frame.b);
        let lift = spec.lift.clamp(0.0, 1.5);
        let world: Vec<Vec3> = (0..6)
            .map(|k| {
                let s = k as f32 / 5.0;
                frame.origin + flat * (length * s * lift.cos()) + frame.n * (length * lift.sin() * s * (1.0 - 0.5 * s))
            })
            .collect();
        let local: Vec<Vec3> = world.iter().map(|&w| frame.to_local(w)).collect();
        guides.push(t, bary, &local)?;
    }
    if guides.is_empty() {
        return Err(HairError::invalid("body hair", 0, "no guide could be placed"));
    }
    Ok(PresetGroom { name: "body", guides, params, area: area.triangles().to_vec(), part: None })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::collide::{TriangleGrid, penetration};
    use crate::hair::placement::place;
    use crate::hair::surface::ProxyHead;
    use crate::hair::{GroomMap, GroomMaps, generate_groom};
    use humanart_math::Vec2;

    fn anchors(head: &ProxyHead) -> FaceAnchors {
        FaceAnchors {
            mouth: head.surface_point(Vec3::new(0.0, -1.0, 1.0)),
            chin: head.surface_point(Vec3::new(0.0, -2.2, 1.0)),
            up: Vec3::Y,
        }
    }

    #[test]
    fn hair_05_facial_styles_cover_their_areas_and_point_down() {
        let head = ProxyHead::new(96).unwrap();
        let s = &head.surface;
        let area = ScalpMask::new(s, head.lower_face.clone()).unwrap();
        let grid = TriangleGrid::new(s.positions(), s.triangles(), 0.01).unwrap();
        let sdf = SdfGrid::build(&grid, 0.004, 0.06).unwrap();
        let a = anchors(&head);
        for style in FACIAL_STYLES {
            let g = facial_groom(style.name, s, &area, &a, 80, Some(&sdf), 4).unwrap();
            assert_eq!(g.guides.len(), 80, "{}", style.name);
            assert!(g.params.strand_count > 100.0, "{}", style.name);
            let mut pts = Vec::new();
            let mut down = 0;
            for i in 0..g.guides.len() {
                let w = g.guides.world_points(i, &g.guides.frame(i, s).unwrap());
                if style.name == "moustache" {
                    let h = (w[0] - a.mouth).dot(a.up);
                    assert!((0.0..0.03).contains(&h), "{h}");
                }
                down += usize::from(w[w.len() - 1].y < w[0].y);
                pts.extend(w);
            }
            assert!(down * 10 >= g.guides.len() * 8, "{}: {down}", style.name);
            assert_eq!(penetration(&grid, Some(&sdf), &pts, 0.001).inside, 0, "{}", style.name);
            let params = GroomParams { strand_count: 400.0, ..g.params };
            let mask = ScalpMask::new(s, g.area.clone()).unwrap();
            let layout = place(s, &mask, &g.guides, &params, &GroomMaps::default(), 1).unwrap();
            generate_groom(s, &layout, &g.guides, &params, Some(&sdf)).unwrap();
        }
        let lengths = |name: &str| {
            let g = facial_groom(name, s, &area, &a, 50, Some(&sdf), 4).unwrap();
            (0..50).map(|i| g.guides.length(i)).sum::<f32>() / 50.0
        };
        let (stubble, full) = (lengths("stubble"), lengths("full_beard"));
        assert!(stubble < 0.0015 && full > 0.03, "{stubble} {full}");
        assert!(facial_groom("sideburns", s, &area, &a, 5, None, 1).is_err());
        let far = FaceAnchors { mouth: Vec3::splat(9.0), ..a };
        assert!(facial_groom("moustache", s, &area, &far, 5, None, 1).is_err());
    }

    /// A vertical cylinder of radius 4 cm and height 40 cm: a stand-in limb.
    fn limb() -> HairSurface {
        let (rings, segs) = (20_u32, 24_u32);
        let mut p = Vec::new();
        let mut n = Vec::new();
        let mut uv = Vec::new();
        for i in 0..=rings {
            for j in 0..=segs {
                let a = std::f32::consts::TAU * (j % segs) as f32 / segs as f32;
                let d = Vec3::new(a.cos(), 0.0, a.sin());
                p.push(d * 0.04 + Vec3::new(0.0, 0.4 * i as f32 / rings as f32, 0.0));
                n.push(d);
                uv.push(Vec2::new(j as f32 / segs as f32, i as f32 / rings as f32));
            }
        }
        let mut t = Vec::new();
        let v = |i: u32, j: u32| i * (segs + 1) + j;
        for i in 0..rings {
            for j in 0..segs {
                t.push([v(i, j), v(i + 1, j), v(i + 1, j + 1)]);
                t.push([v(i, j), v(i + 1, j + 1), v(i, j + 1)]);
            }
        }
        HairSurface::new(p, n, uv, t).unwrap()
    }

    #[test]
    fn hair_05_body_hair_follows_the_flow_and_the_density_map() {
        let s = limb();
        let all: Vec<u32> = (0..s.triangles().len() as u32).collect();
        let area = ScalpMask::new(&s, all).unwrap();
        let spec = body_region("arm").unwrap();
        let g = body_hair(&s, &area, &spec, 200, 8).unwrap();
        let expected = area.area() * 10_000.0 * spec.density;
        assert!((g.params.strand_count - expected.floor()).abs() < 1.0);
        for i in 0..g.guides.len() {
            let w = g.guides.world_points(i, &g.guides.frame(i, &s).unwrap());
            let d = w[w.len() - 1] - w[0];
            assert!(d.y < -0.5 * d.length(), "hair {i} does not follow the flow: {d:?}");
            assert!(g.guides.length(i) < 0.013);
        }
        // A density map that is 0 on the lower half of the UV square keeps hair off it.
        let mut density = GroomMap::new(2, 2, 1, &[1.0]).unwrap();
        density.texel_mut(0, 0).unwrap()[0] = 0.0;
        density.texel_mut(1, 0).unwrap()[0] = 0.0;
        let maps = GroomMaps { density: Some(density), ..GroomMaps::default() };
        let params = GroomParams { strand_count: 2000.0, ..g.params };
        let layout = place(&s, &area, &g.guides, &params, &maps, 3).unwrap();
        let low = (0..layout.len()).filter(|&i| s.uv_at(layout.triangle[i], layout.bary[i]).unwrap().y < 0.2).count();
        assert!(low * 50 < layout.len(), "{low} of {}", layout.len());
        let bad = BodyHairSpec { length: f32::NAN, ..spec };
        assert!(body_hair(&s, &area, &bad, 10, 1).is_err());
        assert!(body_region("back").is_none());
    }

    // Contact sheet: brows and lashes, then each facial style, seen from the front, written to
    // target/agents/hair/facial.png.
    #[test]
    fn hair_04_05_facial_contact_sheet() {
        use crate::hair::features::{BrowSpec, LashSpec, brow_groom, lash_groom};
        let head = ProxyHead::new(96).unwrap();
        let s = &head.surface;
        let area = ScalpMask::new(s, head.lower_face.clone()).unwrap();
        let grid = TriangleGrid::new(s.positions(), s.triangles(), 0.01).unwrap();
        let sdf = SdfGrid::build(&grid, 0.004, 0.06).unwrap();
        let a = anchors(&head);
        let world_of = |g: &Guides| -> Vec<Vec3> {
            (0..g.len()).flat_map(|i| g.world_points(i, &g.frame(i, s).unwrap())).collect()
        };
        let mut eye = Vec::new();
        for side in [-1.0, 1.0] {
            eye.extend(world_of(
                &brow_groom(s, &grid, &head.brow_curve(side), Vec3::Y, &BrowSpec::default(), 1).unwrap().guides,
            ));
            eye.extend(world_of(
                &lash_groom(s, &grid, &head.lid_curve(side), -Vec3::Y, None, &LashSpec::upper(), 2).unwrap().guides,
            ));
        }
        let mut sets = vec![(eye, crate::hair::features::FEATURE_POINTS as usize)];
        for style in ["full_beard", "short_beard", "moustache", "goatee"] {
            let g = facial_groom(style, s, &area, &a, 300, Some(&sdf), 4).unwrap();
            let params = GroomParams { strand_count: g.params.strand_count.min(6000.0), ..g.params };
            let mask = ScalpMask::new(s, g.area.clone()).unwrap();
            let layout = place(s, &mask, &g.guides, &params, &GroomMaps::default(), 1).unwrap();
            let out = generate_groom(s, &layout, &g.guides, &params, Some(&sdf)).unwrap();
            sets.push((out.world, out.strands.points as usize));
        }
        let panels: Vec<crate::hair::sheet::Panel<'_>> = sets
            .iter()
            .map(|(w, p)| crate::hair::sheet::Panel {
                dots: s.positions(),
                strands: w,
                points: *p,
                centre: head.centre - Vec3::new(0.0, 0.03, 0.0),
                scale: 1200.0,
                front: true,
            })
            .collect();
        crate::hair::sheet::write("facial.png", &panels, 5, 220, 300);
    }
}
