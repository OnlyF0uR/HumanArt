//! Normal and height detail baking `[BAKE-04]`: the difference between a detailed source surface
//! (a high-resolution scan registered to the template, or the template with sculpt layers
//! applied) and the template, stored in the template's UVs.
//!
//! **Method:** for every texel of the template (the *low* surface), two rays leave the surface
//! point along the shading normal, one outward and one inward, each up to
//! [`DetailSettings::max_distance`] (the cage distance) and each starting a ray offset behind the
//! point so that coincident surfaces are found. The nearer hit on the source wins:
//!
//! - **height**: the signed distance from the low surface to the source along the low normal,
//!   metres, positive outward (0 where neither ray hits);
//! - **normal**: the source's interpolated shading normal at the hit, expressed in the low
//!   surface's tangent frame `(x along +U, y along +V, z along the normal)`; `(0, 0, 1)` where
//!   nothing is hit.
//!
//! A sculpt layer on the template is baked by passing the sculpted shape as the source: build a
//! [`BakeMesh`] of the template mesh (or of its subdivision) with the sculpted positions. The
//! source needs no UVs and no particular topology; scans come in through `humanart_io::obj` or
//! `humanart_io::ply` and the registration of the capture pipeline.
//!
//! The tangent frames are those of `[EVAL-04]` ([`super::surface::bake_frames`]), so the
//! renderer's per-pixel frame reproduces the source normal exactly from this map.

use super::raster::SurfaceMap;
use super::{BakeError, BakeMap, BakeMesh, Coverage, Result};

/// What [`bake_detail`] does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetailSettings {
    /// How far from the low surface the source is searched, metres (> 0).
    pub max_distance: f32,
}

impl Default for DetailSettings {
    /// A 5 mm cage.
    fn default() -> Self {
        Self { max_distance: 0.005 }
    }
}

/// The maps of [`bake_detail`].
#[derive(Clone, Debug, PartialEq)]
pub struct DetailMaps {
    /// Signed height, metres (1 channel).
    pub height: BakeMap,
    /// Tangent-space normal (3 channels).
    pub normals: BakeMap,
    /// The number of covered texels whose rays found the source.
    pub hits: usize,
}

/// Bakes the detail of `source` onto the surface `surface` of `low`.
pub fn bake_detail(
    low: &BakeMesh,
    surface: &SurfaceMap,
    source: &BakeMesh,
    settings: &DetailSettings,
) -> Result<DetailMaps> {
    if !(settings.max_distance > 0.0 && settings.max_distance.is_finite()) {
        return Err(BakeError::invalid("detail: the search distance must be positive"));
    }
    let bvh = source.bvh()?;
    let mut normals = BakeMap::new(&surface.target, 3, 0.0)?;
    let mut hits = 0;
    let eps = low.ray_offset();
    let width = surface.target.width as usize;
    let height = surface.bake(low, 1, 0.0, |x, y, p, out| {
        let i = y as usize * width + x as usize;
        // Each ray starts a hair behind the point, so a source lying exactly on the low surface
        // is found at distance 0 instead of slipping past both rays.
        let reach = settings.max_distance + eps;
        let up = bvh.intersect(p.position - p.normal * eps, p.normal, 0.0, reach).map(|mut h| {
            h.t -= eps;
            h
        });
        let down = bvh.intersect(p.position + p.normal * eps, -p.normal, 0.0, reach).map(|mut h| {
            h.t -= eps;
            h
        });
        let hit = match (up, down) {
            (Some(u), Some(d)) => Some(if u.t.abs() <= d.t.abs() { (u, 1.0) } else { (d, -1.0) }),
            (Some(u), None) => Some((u, 1.0)),
            (None, Some(d)) => Some((d, -1.0)),
            (None, None) => None,
        };
        let mut n = [0.0, 0.0, 1.0];
        if let Some((h, sign)) = hit
            && let Some(sp) = source.point(h.triangle as usize, h.b1, h.b2)
        {
            out[0] = sign * h.t;
            n = p.to_tangent(sp.normal).normalize_or(humanart_math::Vec3::Z).to_array();
            hits += 1;
        }
        normals.set(i, &n, Coverage::Surface);
    })?;
    Ok(DetailMaps { height, normals, hits })
}

#[cfg(test)]
mod tests {
    use humanart_math::Vec3;

    use super::*;
    use crate::bake::BakeTarget;
    use crate::bake::raster::rasterize;
    use crate::bake::tests::box_mesh;

    #[test]
    fn bake_04_recovers_a_sine_relief() {
        let (low, _) = box_mesh(0.06, 0.06, 0.06, 4);
        let (_, fine) = box_mesh(0.06, 0.06, 0.06, 96);
        // Three waves across the +Z face, 0.4 mm deep, applied by position (duplicates agree).
        let (a, k) = (0.0004_f32, 3.0 * std::f32::consts::TAU / 0.06);
        let displaced: Vec<Vec3> = fine
            .positions()
            .iter()
            .map(|p| if p.z > 0.0299 { Vec3::new(p.x, p.y, p.z + a * (k * p.x).sin()) } else { *p })
            .collect();
        let source = BakeMesh::new(&fine, &displaced, None).unwrap();
        let surface = rasterize(&low, &BakeTarget::new(0, 192, 128)).unwrap();
        let maps = bake_detail(&low, &surface, &source, &DetailSettings::default()).unwrap();
        assert!(maps.hits >= surface.covered_count() * 99 / 100, "{} of {}", maps.hits, surface.covered_count());
        let mut checked = 0;
        for (i, t, b1, b2) in surface.covered() {
            let p = low.point(t, b1, b2).unwrap();
            if !(p.normal.z > 0.99 && p.position.x.abs() < 0.028 && p.position.y.abs() < 0.028) {
                continue;
            }
            let want_h = a * (k * p.position.x).sin();
            let got_h = maps.height.texel(i).unwrap()[0];
            assert!((got_h - want_h).abs() < 0.02 * a, "{got_h} vs {want_h}");
            // Analytic normal (-a k cos(kx), 0, 1) in world = tangent frame (t = +x, b = -y here).
            let world = Vec3::new(-a * k * (k * p.position.x).cos(), 0.0, 1.0).normalize();
            let n = maps.normals.texel(i).unwrap();
            let got = p.tangent * n[0] + p.bitangent * n[1] + p.normal * n[2];
            assert!(got.dot(world) > 0.999, "{got:?} vs {world:?}");
            checked += 1;
        }
        assert!(checked > 500, "{checked}");
        // Baking the low mesh onto itself: zero height, flat normals.
        let same = bake_detail(&low, &surface, &low, &DetailSettings::default()).unwrap();
        for (i, ..) in surface.covered() {
            assert!(same.height.texel(i).unwrap()[0].abs() < 1e-6);
        }
        assert!(bake_detail(&low, &surface, &low, &DetailSettings { max_distance: 0.0 }).is_err());
    }
}
