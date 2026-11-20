//! Thickness and translucency `[BAKE-01]`: how much tissue lies behind each point of the surface.
//!
//! **Method** (distance to the opposite surface by ray casting): from the texel's surface point,
//! offset just inside along the triangle normal, [`ThicknessSettings::rays`] rays are cast into a
//! cone of half-angle [`ThicknessSettings::cone_angle`] around the inward shading normal
//! (uniform in solid angle, directions from [`super::sampling`]). Each ray's length is the
//! distance to the first surface it meets, or [`ThicknessSettings::max_distance`] when it meets
//! none (an open edge or a direction that leaves the body). The raw thickness is the mean length;
//! a ray tilted by `θ` crosses a slab of thickness `d` over `d / cos θ`, so on a slab the raw value
//! is `d · ln(1/cos θₘ) / (1 − cos θₘ)` (1.074 `d` for the default 30° cone).
//!
//! **Per-region scale:** the raw thickness is multiplied by [`ThicknessSettings::region_scale`]
//! of the texel's region (the region of the nearest triangle corner; 1 for a region without an
//! entry), so an artist can say that cartilage (ear) or a fingernail transmits less than its
//! geometric thickness suggests.
//!
//! **Output** ([`bake_thickness`]): a 2-channel map, `[thickness, translucency]`:
//!
//! - channel 0: the scaled thickness in metres (linear, `f32`; at most `max_distance · scale`);
//! - channel 1: the translucency `exp(−thickness / falloff)` in `[0, 1]`, the fraction of light
//!   a thin feature passes for the transmittance term of the skin shader `[SHD-03]`.
//!
//! The GPU baker (`super::gpu`, `shaders/bake/thickness.wgsl`) computes the same values.

use super::bvh::Bvh;
use super::raster::SurfaceMap;
use super::sampling::{keep_side, point, rotation, texel_hash, to_world, uniform_cone};
use super::{BakeError, BakeMap, BakeMesh, Result};

/// The most rays per texel.
pub const MAX_RAYS: u32 = 4096;

/// What [`bake_thickness`] does.
#[derive(Clone, Debug, PartialEq)]
pub struct ThicknessSettings {
    /// Rays per texel, 1 to [`MAX_RAYS`].
    pub rays: u32,
    /// Half-angle of the ray cone around the inward normal, radians, in `[0, π/2)`.
    pub cone_angle: f32,
    /// The length of a ray that meets nothing, metres (> 0).
    pub max_distance: f32,
    /// The distance over which translucency falls to `1/e`, metres (> 0).
    pub falloff: f32,
    /// Thickness multiplier per region id (finite, > 0); regions past the end use 1.
    pub region_scale: Vec<f32>,
    /// The seed of the ray rotation.
    pub seed: u32,
}

impl Default for ThicknessSettings {
    /// 64 rays in a 30° cone, 5 cm reach, 4 mm falloff, no region scales.
    fn default() -> Self {
        Self {
            rays: 64,
            cone_angle: 30_f32.to_radians(),
            max_distance: 0.05,
            falloff: 0.004,
            region_scale: Vec::new(),
            seed: 0,
        }
    }
}

impl ThicknessSettings {
    /// Checks the settings.
    pub fn check(&self) -> Result<()> {
        if self.rays == 0 || self.rays > MAX_RAYS {
            return Err(BakeError::invalid("thickness: rays must be 1 to 4096"));
        }
        if !(0.0..std::f32::consts::FRAC_PI_2).contains(&self.cone_angle) {
            return Err(BakeError::invalid("thickness: the cone angle must be in [0, pi/2)"));
        }
        if !(self.max_distance > 0.0 && self.max_distance.is_finite() && self.falloff > 0.0 && self.falloff.is_finite())
        {
            return Err(BakeError::invalid("thickness: distances must be positive and finite"));
        }
        if self.region_scale.len() > usize::from(u16::MAX)
            || self.region_scale.iter().any(|s| !(*s > 0.0 && s.is_finite()))
        {
            return Err(BakeError::invalid("thickness: region scales must be positive and finite"));
        }
        Ok(())
    }

    /// The scale of region `r`.
    pub fn scale(&self, r: u16) -> f32 {
        self.region_scale.get(usize::from(r)).copied().unwrap_or(1.0)
    }
}

/// The raw (unscaled) mean ray length at a surface point; `x, y` choose the ray rotation.
pub fn raw_thickness(
    bvh: &Bvh,
    p: &super::SurfacePoint,
    offset: f32,
    x: u32,
    y: u32,
    settings: &ThicknessSettings,
) -> f32 {
    let origin = p.position - p.geometric * offset;
    let r = rotation(texel_hash(x, y, settings.seed));
    let cos_max = settings.cone_angle.cos();
    let mut sum = 0.0_f32;
    for i in 0..settings.rays {
        let d = keep_side(to_world(uniform_cone(point(i, settings.rays, r), cos_max), -p.normal), p.geometric, -1.0)
            .normalize_or(-p.geometric);
        sum += bvh.intersect(origin, d, 0.0, settings.max_distance).map_or(settings.max_distance, |h| h.t);
    }
    sum / settings.rays as f32
}

/// Bakes the thickness map of the surface `surface` of `mesh` (see the module documentation for
/// the two channels).
pub fn bake_thickness(mesh: &BakeMesh, surface: &SurfaceMap, settings: &ThicknessSettings) -> Result<BakeMap> {
    settings.check()?;
    let bvh = mesh.bvh()?;
    let offset = mesh.ray_offset();
    surface.bake(mesh, 2, 0.0, |x, y, p, out| {
        let t = raw_thickness(&bvh, p, offset, x, y, settings) * settings.scale(p.region);
        out[0] = t;
        out[1] = (-t / settings.falloff).exp();
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bake::BakeTarget;
    use crate::bake::raster::rasterize;
    use crate::bake::tests::{box_mesh, sphere_mesh};

    #[test]
    fn bake_08_slab_thickness_matches_the_cone_formula() {
        // A 10 x 10 x 1 cm slab: the middle of the big faces sees the 1 cm thickness.
        let (mesh, _) = box_mesh(0.1, 0.1, 0.01, 8);
        let surface = rasterize(&mesh, &BakeTarget::new(0, 96, 64)).unwrap();
        let settings = ThicknessSettings { rays: 128, ..ThicknessSettings::default() };
        let map = bake_thickness(&mesh, &surface, &settings).unwrap();
        let c = settings.cone_angle.cos();
        let want = 0.01 * (1.0 / c).ln() / (1.0 - c);
        // Texels of the +Z face near its middle (the box layout puts +Z at column 1, row 0).
        let mut checked = 0;
        for (i, t, b1, b2) in surface.covered() {
            let p = mesh.point(t, b1, b2).unwrap();
            if p.normal.z > 0.99 && p.position.x.abs() < 0.03 && p.position.y.abs() < 0.03 {
                let v = map.texel(i).unwrap();
                assert!((v[0] - want).abs() < 0.03 * want, "{} vs {want}", v[0]);
                assert!((v[1] - (-v[0] / settings.falloff).exp()).abs() < 1e-6);
                checked += 1;
            }
        }
        assert!(checked > 20, "{checked}");
    }

    #[test]
    fn bake_08_sphere_thickness_is_the_mean_chord() {
        // A ray at angle θ from the inward normal crosses a sphere of radius R over 2R cos θ.
        let r = 0.02;
        let mesh = sphere_mesh(r, 48, 24);
        let surface = rasterize(&mesh, &BakeTarget::new(0, 64, 32)).unwrap();
        let settings = ThicknessSettings { rays: 256, ..ThicknessSettings::default() };
        let map = bake_thickness(&mesh, &surface, &settings).unwrap();
        let c = settings.cone_angle.cos();
        let want = 2.0 * r * (1.0 + c) / 2.0; // E[cos θ] uniform in the cone = (1 + cos θm) / 2
        let (mut sum, mut n) = (0.0, 0);
        for (i, ..) in surface.covered() {
            sum += map.texel(i).unwrap()[0];
            n += 1;
        }
        let mean = sum / n as f32;
        assert!((mean - want).abs() < 0.02 * want, "{mean} vs {want}");
    }

    #[test]
    fn bake_01_region_scale_multiplies_thickness() {
        let (mesh, _) = box_mesh(0.1, 0.1, 0.01, 4);
        let surface = rasterize(&mesh, &BakeTarget::new(0, 48, 32)).unwrap();
        let base = bake_thickness(&mesh, &surface, &ThicknessSettings::default()).unwrap();
        // The box's vertices are all in region 0.
        let scaled = ThicknessSettings { region_scale: vec![2.5], ..ThicknessSettings::default() };
        let map = bake_thickness(&mesh, &surface, &scaled).unwrap();
        for (i, ..) in surface.covered() {
            assert!((map.texel(i).unwrap()[0] - 2.5 * base.texel(i).unwrap()[0]).abs() < 1e-7);
        }
        let bad = ThicknessSettings { region_scale: vec![0.0], ..ThicknessSettings::default() };
        assert!(bake_thickness(&mesh, &surface, &bad).is_err());
        let bad = ThicknessSettings { rays: 0, ..ThicknessSettings::default() };
        assert!(bad.check().is_err());
    }
}
