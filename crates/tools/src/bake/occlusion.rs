//! Ambient occlusion, cavity and bent normals `[BAKE-02]`, from the rest or a deformed mesh.
//!
//! **Method:** from the texel's surface point, offset outward along the triangle normal,
//! [`OcclusionSettings::rays`] cosine-weighted rays are cast around the shading normal (directions
//! from [`super::sampling`], mirrored back above the triangle's plane when the shading normal
//! tilts them under it). One cast per ray, up to [`OcclusionSettings::max_distance`], answers all
//! three questions:
//!
//! - **ambient occlusion** = the fraction of rays that reach `max_distance` unblocked: the
//!   cosine-weighted visibility of the hemisphere, 1 = open, 0 = fully enclosed;
//! - **cavity** = the fraction of rays not blocked within [`OcclusionSettings::cavity_distance`]
//!   (a few millimetres): only the small-scale concavities (skin folds, the alar crease, the
//!   corners of the eyes and mouth) darken it, which drives specular and micro occlusion
//!   `[SHD-08]` `[SHD-09]`;
//! - **bent normal** = the normalised mean direction of the unblocked rays (the shading normal when
//!   all are blocked): the mean direction light arrives from, for the diffuse environment lookup
//!   `[SHD-08]`.
//!
//! **Output** ([`bake_occlusion`]):
//!
//! - `occlusion`: 2 channels `[ao, cavity]`, each in `[0, 1]`;
//! - `bent_normals`: 3 channels, the unit bent normal in the texel's tangent frame
//!   `(x along +U, y along +V, z along the normal)`, components in `[−1, 1]` (encode as
//!   `0.5 · v + 0.5` for an 8-bit texture).
//!
//! The GPU baker (`shaders/bake/cavity_ao.wgsl`) computes the same values.

use humanart_math::Vec3;

use super::bvh::Bvh;
use super::raster::SurfaceMap;
use super::sampling::{cosine_hemisphere, keep_side, point, rotation, texel_hash, to_world};
use super::{BakeError, BakeMap, BakeMesh, Result, SurfacePoint};

/// The most rays per texel.
pub const MAX_RAYS: u32 = 4096;

/// What [`bake_occlusion`] does.
#[derive(Clone, Debug, PartialEq)]
pub struct OcclusionSettings {
    /// Rays per texel, 1 to [`MAX_RAYS`].
    pub rays: u32,
    /// The ambient-occlusion reach, metres (> 0).
    pub max_distance: f32,
    /// The cavity reach, metres (> 0, at most `max_distance`).
    pub cavity_distance: f32,
    /// The seed of the ray rotation.
    pub seed: u32,
}

impl Default for OcclusionSettings {
    /// 64 rays, 5 cm occlusion reach, 3 mm cavity reach.
    fn default() -> Self {
        Self { rays: 64, max_distance: 0.05, cavity_distance: 0.003, seed: 0 }
    }
}

impl OcclusionSettings {
    /// Checks the settings.
    pub fn check(&self) -> Result<()> {
        if self.rays == 0 || self.rays > MAX_RAYS {
            return Err(BakeError::invalid("occlusion: rays must be 1 to 4096"));
        }
        let ok = |d: f32| d > 0.0 && d.is_finite();
        if !ok(self.max_distance) || !ok(self.cavity_distance) || self.cavity_distance > self.max_distance {
            return Err(BakeError::invalid("occlusion: distances must be positive, cavity within the reach"));
        }
        Ok(())
    }
}

/// The occlusion of one surface point: `(ao, cavity, bent normal in object space)`.
pub fn occlusion_at(
    bvh: &Bvh,
    p: &SurfacePoint,
    offset: f32,
    x: u32,
    y: u32,
    settings: &OcclusionSettings,
) -> (f32, f32, Vec3) {
    let origin = p.position + p.geometric * offset;
    let r = rotation(texel_hash(x, y, settings.seed));
    let (mut open, mut open_cavity) = (0_u32, 0_u32);
    let mut bent = Vec3::ZERO;
    for i in 0..settings.rays {
        let d = keep_side(to_world(cosine_hemisphere(point(i, settings.rays, r)), p.normal), p.geometric, 1.0)
            .normalize_or(p.geometric);
        match bvh.intersect(origin, d, 0.0, settings.max_distance) {
            None => {
                open += 1;
                open_cavity += 1;
                bent += d;
            }
            Some(h) if h.t >= settings.cavity_distance => open_cavity += 1,
            Some(_) => {}
        }
    }
    let n = settings.rays as f32;
    (open as f32 / n, open_cavity as f32 / n, bent.normalize_or(p.normal))
}

/// The maps of [`bake_occlusion`].
#[derive(Clone, Debug, PartialEq)]
pub struct OcclusionMaps {
    /// `[ao, cavity]`.
    pub occlusion: BakeMap,
    /// The tangent-space bent normal.
    pub bent_normals: BakeMap,
}

/// Bakes ambient occlusion, cavity and bent normals of the surface `surface` of `mesh`.
pub fn bake_occlusion(mesh: &BakeMesh, surface: &SurfaceMap, settings: &OcclusionSettings) -> Result<OcclusionMaps> {
    settings.check()?;
    let bvh = mesh.bvh()?;
    let offset = mesh.ray_offset();
    let mut bent_values: Vec<(usize, Vec3)> = Vec::with_capacity(surface.covered_count());
    let width = surface.target.width as usize;
    let occlusion = surface.bake(mesh, 2, 1.0, |x, y, p, out| {
        let (ao, cavity, bent) = occlusion_at(&bvh, p, offset, x, y, settings);
        out[0] = ao;
        out[1] = cavity;
        bent_values.push((y as usize * width + x as usize, p.to_tangent(bent)));
    })?;
    let mut bent_normals = super::BakeMap::new(&surface.target, 3, 0.0)?;
    for (i, b) in bent_values {
        bent_normals.set(i, &b.to_array(), super::Coverage::Surface);
    }
    Ok(OcclusionMaps { occlusion, bent_normals })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bake::BakeTarget;
    use crate::bake::raster::rasterize;
    use crate::bake::tests::{box_mesh, sphere_mesh};

    #[test]
    fn bake_08_convex_shapes_are_unoccluded() {
        let mesh = sphere_mesh(0.05, 32, 16);
        let surface = rasterize(&mesh, &BakeTarget::new(0, 48, 24)).unwrap();
        let maps = bake_occlusion(&mesh, &surface, &OcclusionSettings::default()).unwrap();
        for (i, ..) in surface.covered() {
            assert_eq!(maps.occlusion.texel(i).unwrap(), &[1.0, 1.0]);
            let b = maps.bent_normals.texel(i).unwrap();
            // Unoccluded: the bent normal is the mean cosine-lobe direction, close to the normal.
            assert!(b[2] > 0.95, "{b:?}");
        }
    }

    /// Two parallel plates 1 cm apart (two flat boxes): the inner faces see the other plate.
    #[test]
    fn bake_08_parallel_plates_match_the_analytic_occlusion() {
        let (_, ma) = box_mesh(0.4, 0.4, 0.002, 6);
        // Combine two plates at z = ±(5 mm + 1 mm) into one mesh.
        let mut data = ma.data().clone();
        let n = data.positions.len() as u32;
        let shift = Vec3::new(0.0, 0.0, 0.006);
        let mut second = ma.data().clone();
        for p in &mut data.positions {
            *p += shift;
        }
        for p in &mut second.positions {
            *p -= shift;
        }
        data.positions.extend(second.positions);
        data.uvs.extend(second.uvs);
        let corners = data.face_vertices.len() as u32;
        data.face_vertices.extend(second.face_vertices.iter().map(|v| v + n));
        data.face_offsets.extend(second.face_offsets.iter().skip(1).map(|o| o + corners));
        data.face_tiles.extend(second.face_tiles.iter().map(|_| 1));
        data.position_keys.extend(second.position_keys.iter().map(|v| v + n));
        data.normal_keys.extend(second.normal_keys.iter().map(|v| v + n));
        data.parts[0].face_count *= 2;
        let mesh = humanart_model::Mesh::new(data).unwrap();
        let regions = vec![0; mesh.vertex_count()];
        let both = BakeMesh::new(&mesh, mesh.positions(), Some(&regions)).unwrap();
        // Tile 1 is the lower plate; its +Z face looks at the upper plate 1 cm away.
        let surface = rasterize(&both, &BakeTarget::new(1, 96, 64)).unwrap();
        let settings = OcclusionSettings { rays: 256, max_distance: 0.02, cavity_distance: 0.005, seed: 3 };
        let maps = bake_occlusion(&both, &surface, &settings).unwrap();
        // Facing an infinite plane at distance h, a cosine-weighted ray escapes the reach R when
        // h / cos θ > R: visibility = 1 − sin²θc with cos θc = h / R, i.e. (h / R)².
        let want_ao = (0.01_f32 / 0.02).powi(2);
        let mut checked = 0;
        for (i, t, b1, b2) in surface.covered() {
            let p = both.point(t, b1, b2).unwrap();
            if p.normal.z > 0.99 && p.position.x.abs() < 0.05 && p.position.y.abs() < 0.05 {
                let v = maps.occlusion.texel(i).unwrap();
                assert!((v[0] - want_ao).abs() < 0.04, "ao {} vs {want_ao}", v[0]);
                assert_eq!(v[1], 1.0, "the plate is beyond the cavity reach");
                let b = maps.bent_normals.texel(i).unwrap();
                // The open rays are the grazing ones, spread evenly around the normal.
                assert!(b[2] > 0.8, "{b:?}");
                checked += 1;
            }
        }
        assert!(checked > 10, "{checked}");
        assert!(OcclusionSettings { cavity_distance: 1.0, ..OcclusionSettings::default() }.check().is_err());
    }
}
