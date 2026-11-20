//! Surface maps `[BAKE-03]`: mean curvature, tangent frames and region ids in UV space.
//!
//! **Mean curvature** ([`vertex_mean_curvature`], [`bake_curvature`]): per welded vertex, the
//! cotangent discretisation of the Laplace-Beltrami operator with Meyer et al.'s mixed Voronoi
//! areas, `Δx_i = (1 / 2A_i) Σ_j (cot α_ij + cot β_ij)(x_j − x_i)`, and `H = −½ Δx · n` (positive
//! where the surface is convex: `1/r` on a sphere of radius `r`, `1/(2r)` on a cylinder). UV-seam
//! duplicates share one value (the sum runs over weld keys); a vertex on an open boundary, where
//! the stencil is incomplete, takes the mean of its interior neighbours. The map holds `H` in
//! 1/m, interpolated barycentrically (1 channel, `f32`; for an 8-bit view use
//! `0.5 + H · scale`). The pre-integrated skin variant reads it `[SHD-04]`.
//!
//! **Tangent frames** ([`bake_frames`]): the interpolated shading normal (3 channels, object
//! space, unit) and tangent (4 channels: unit object-space tangent along +U orthogonal to the
//! normal, `w = ±1` the handedness, `w · (n × t)` along +V), the frames of `[EVAL-04]` that the
//! renderer reconstructs per pixel; tangent-space maps of the other bakers are expressed in them.
//!
//! **Region ids** ([`bake_region_ids`]): the region of the triangle corner nearest to the texel
//! centre, stored as a float (exact for `u16`), [`super::NO_REGION`] outside the surface. Pad it
//! with [`super::dilate::DilateMode::Nearest`]; [`region_id_image`] turns it into a 16-bit data
//! image.

use std::collections::HashMap;

use humanart_io::{ColorSpace, Format, Image, Samples};
use humanart_math::Vec3;

use super::raster::SurfaceMap;
use super::{BakeMap, BakeMesh, Coverage, NO_REGION, Result};

/// The mean curvature of every vertex of `mesh` (1/m; duplicates share their weld key's value).
pub fn vertex_mean_curvature(mesh: &BakeMesh) -> Vec<f32> {
    let n = mesh.positions().len();
    let keys = mesh.position_keys();
    let key = |v: usize| keys.get(v).map_or(v, |&k| k as usize);
    let pos = mesh.positions();
    let mut lap = vec![Vec3::ZERO; n];
    let mut area = vec![0.0_f32; n];
    let mut edge_uses: HashMap<(usize, usize), u32> = HashMap::new();
    for t in 0..mesh.triangle_count() {
        let Some(tri) = mesh.triangle(t) else { continue };
        let k = tri.map(key);
        if k[0] == k[1] || k[1] == k[2] || k[0] == k[2] {
            continue;
        }
        let p = k.map(|i| pos[i]);
        let double_area = (p[1] - p[0]).cross(p[2] - p[0]).length();
        if !(double_area > 0.0 && double_area.is_finite()) {
            continue;
        }
        // Corner c: the cotangent of its angle, and the opposite edge (c+1, c+2).
        let mut cot = [0.0_f32; 3];
        for c in 0..3 {
            let (a, b) = (p[(c + 1) % 3] - p[c], p[(c + 2) % 3] - p[c]);
            cot[c] = a.dot(b) / double_area;
        }
        for c in 0..3 {
            let (i, j) = (k[(c + 1) % 3], k[(c + 2) % 3]);
            let d = p[(c + 2) % 3] - p[(c + 1) % 3];
            lap[i] += d * cot[c];
            lap[j] -= d * cot[c];
            *edge_uses.entry((i.min(j), i.max(j))).or_insert(0) += 1;
        }
        // Mixed areas.
        let tri_area = 0.5 * double_area;
        let obtuse = (0..3).find(|&c| cot[c] < 0.0);
        for c in 0..3 {
            area[k[c]] += match obtuse {
                Some(o) if o == c => tri_area * 0.5,
                Some(_) => tri_area * 0.25,
                None => {
                    // Voronoi area of corner c: (|e_ab|² cot_opposite + |e_ac|² cot_opposite) / 8.
                    let (b, d) = ((c + 1) % 3, (c + 2) % 3);
                    ((p[b] - p[c]).length_squared() * cot[d] + (p[d] - p[c]).length_squared() * cot[b]) / 8.0
                }
            };
        }
    }
    let normals = mesh.normals();
    let mut h = vec![0.0_f32; n];
    let mut boundary = vec![false; n];
    for (&(i, j), &uses) in &edge_uses {
        if uses == 1 {
            boundary[i] = true;
            boundary[j] = true;
        }
    }
    for k in 0..n {
        if area[k] > 0.0 && !boundary[k] {
            h[k] = -0.5 * (lap[k] / (2.0 * area[k])).dot(normals[k]);
        }
    }
    // Boundary keys: the mean of their interior neighbours (edges visited in a sorted order).
    let mut edges: Vec<(usize, usize)> = edge_uses.keys().copied().collect();
    edges.sort_unstable();
    let mut sum = vec![0.0_f32; n];
    let mut count = vec![0_u32; n];
    for &(i, j) in &edges {
        for (a, b) in [(i, j), (j, i)] {
            if boundary[a] && !boundary[b] {
                sum[a] += h[b];
                count[a] += 1;
            }
        }
    }
    for k in 0..n {
        if boundary[k] && count[k] > 0 {
            h[k] = sum[k] / count[k] as f32;
        }
    }
    (0..n).map(|v| h[key(v)]).collect()
}

/// Bakes the mean-curvature map (1 channel, 1/m).
pub fn bake_curvature(mesh: &BakeMesh, surface: &SurfaceMap) -> Result<BakeMap> {
    let h = vertex_mean_curvature(mesh);
    surface.bake(mesh, 1, 0.0, |_, _, p, out| out[0] = mesh.interpolate(p, &h))
}

/// The maps of [`bake_frames`].
#[derive(Clone, Debug, PartialEq)]
pub struct FrameMaps {
    /// Unit object-space shading normal (3 channels).
    pub normal: BakeMap,
    /// Unit object-space tangent and handedness (4 channels).
    pub tangent: BakeMap,
}

/// Bakes the tangent frames.
pub fn bake_frames(mesh: &BakeMesh, surface: &SurfaceMap) -> Result<FrameMaps> {
    let normal = surface.bake(mesh, 3, 0.0, |_, _, p, out| out.copy_from_slice(&p.normal.to_array()))?;
    let tangent =
        surface.bake(mesh, 4, 0.0, |_, _, p, out| out.copy_from_slice(&p.tangent.extend(p.sign).to_array()))?;
    Ok(FrameMaps { normal, tangent })
}

/// Bakes the region-id map (1 channel, the id as a float, [`NO_REGION`] outside).
pub fn bake_region_ids(mesh: &BakeMesh, surface: &SurfaceMap) -> Result<BakeMap> {
    surface.bake(mesh, 1, f32::from(NO_REGION), |_, _, p, out| out[0] = f32::from(p.region))
}

/// A region-id map as a 16-bit data image; empty texels are [`NO_REGION`].
pub fn region_id_image(map: &BakeMap) -> Result<Image> {
    if map.channels != 1 {
        return Err(super::BakeError::invalid("a region-id map has one channel"));
    }
    let samples = map
        .data
        .iter()
        .zip(&map.coverage)
        .map(|(&v, &c)| if c == Coverage::Empty || !(0.0..=65535.0).contains(&v) { NO_REGION } else { v as u16 })
        .collect();
    Ok(Image::new(Format::Png, map.width, map.height, 1, ColorSpace::Data, Samples::U16(samples))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bake::dilate::{DilateMode, dilate};
    use crate::bake::raster::rasterize;
    use crate::bake::tests::{box_mesh, sphere_mesh, tube_mesh};
    use crate::bake::{BakeTarget, NO_REGION};

    #[test]
    fn bake_08_sphere_and_cylinder_curvature() {
        let r = 0.03;
        let sphere = sphere_mesh(r, 64, 32);
        let h = vertex_mean_curvature(&sphere);
        for (v, &p) in sphere.positions().iter().enumerate() {
            // Away from the poles (their fans are irregular).
            if p.y.abs() < 0.6 * r {
                assert!((h[v] * r - 1.0).abs() < 0.025, "{}", h[v] * r);
            }
        }
        let tube = tube_mesh(0.008, 0.06, 48, 96);
        let tube = BakeMesh::new(&tube, tube.positions(), None).unwrap();
        let h = vertex_mean_curvature(&tube);
        for (v, &p) in tube.positions().iter().enumerate() {
            if p.y.abs() < 0.02 {
                assert!((h[v] * 0.016 - 1.0).abs() < 0.03, "{}", h[v] * 0.016);
            }
        }
        // A box face is flat.
        let (b, _) = box_mesh(0.1, 0.1, 0.1, 6);
        let h = vertex_mean_curvature(&b);
        for (v, &p) in b.positions().iter().enumerate() {
            if p.x.abs() < 0.03 && p.y.abs() < 0.03 {
                assert!(h[v].abs() < 1e-3, "{}", h[v]);
            }
        }
    }

    #[test]
    fn bake_08_curvature_map_and_frames() {
        let r = 0.03;
        let sphere = sphere_mesh(r, 64, 32);
        let surface = rasterize(&sphere, &BakeTarget::new(0, 64, 32)).unwrap();
        let map = bake_curvature(&sphere, &surface).unwrap();
        let frames = bake_frames(&sphere, &surface).unwrap();
        for (i, t, b1, b2) in surface.covered() {
            let p = sphere.point(t, b1, b2).unwrap();
            if p.position.y.abs() < 0.6 * r {
                assert!((map.texel(i).unwrap()[0] * r - 1.0).abs() < 0.03);
            }
            let n = Vec3::from_slice(frames.normal.texel(i).unwrap()).unwrap();
            assert!(n.dot(p.position.normalize()) > 0.99);
            let tw = frames.tangent.texel(i).unwrap();
            let tv = Vec3::new(tw[0], tw[1], tw[2]);
            assert!((tv.length() - 1.0).abs() < 1e-4 && tv.dot(n).abs() < 1e-4 && tw[3].abs() == 1.0);
        }
    }

    #[test]
    fn bake_08_open_boundary_takes_interior_values() {
        // One face of a box alone is an open flat patch: curvature 0 everywhere, boundary too.
        let (_, m) = box_mesh(0.1, 0.1, 0.1, 4);
        let mut data = m.data().clone();
        let face_count = 16;
        let keep = data.face_offsets[face_count] as usize;
        data.face_vertices.truncate(keep);
        data.face_offsets.truncate(face_count + 1);
        data.face_tiles.truncate(face_count);
        data.parts[0].face_count = face_count as u32;
        data.parts[0].closed = false;
        let mesh = humanart_model::Mesh::new(data).unwrap();
        let bake = BakeMesh::new(&mesh, mesh.positions(), None).unwrap();
        assert!(vertex_mean_curvature(&bake).iter().all(|h| h.abs() < 1e-3));
    }

    #[test]
    fn bake_03_region_ids_pad_with_nearest() {
        let (_, m) = box_mesh(0.1, 0.1, 0.1, 2);
        // Region = which face the vertex's first triangle is on: x > 0 vertices 3, others 7.
        let regions: Vec<u16> = m.positions().iter().map(|p| if p.x > 0.0 { 3 } else { 7 }).collect();
        let mesh = BakeMesh::new(&m, m.positions(), Some(&regions)).unwrap();
        let surface = rasterize(&mesh, &BakeTarget::new(0, 60, 40)).unwrap();
        let mut map = bake_region_ids(&mesh, &surface).unwrap();
        dilate(&mut map, 1, DilateMode::Nearest).unwrap();
        let image = region_id_image(&map).unwrap();
        let ids = image.as_u16().unwrap();
        assert!(ids.contains(&3) && ids.contains(&7) && ids.contains(&NO_REGION));
        assert!(ids.iter().all(|&i| i == 3 || i == 7 || i == NO_REGION));
        assert!(region_id_image(&map.channel(0).unwrap()).is_ok());
    }
}
