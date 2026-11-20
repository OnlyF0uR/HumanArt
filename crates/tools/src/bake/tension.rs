//! The wrinkle and tension map generator `[BAKE-05]` for the expression wrinkle maps `[EXP-04]`.
//!
//! Given the rest surface and one deformed pose (an expression target at full weight), the
//! generator measures how the skin is squeezed and stretched and grows wrinkles across the
//! squeeze.
//!
//! **Tension.** Per triangle, the Green strain `E = ½(FᵀF − I)` of the deformation gradient `F`
//! from the rest to the deformed triangle, lifted to a symmetric 3 × 3 tensor in the rest
//! triangle's plane (independent of the frame chosen in it), area-averaged onto the welded
//! vertices and interpolated per texel. At the texel it is projected onto the rest tangent frame
//! and split into principal stretches `√(1 + 2e)`: **compression** `max(0, 1 − λ_min)`,
//! **stretch** `max(0, λ_max − 1)` and the unit **direction of compression** in tangent space.
//!
//! **Wrinkles.** Folds run across the compression direction. Both bands are sparse-convolution
//! Gabor noise (Lagae et al.) evaluated at the rest position: impulses on a jittered 3D grid of
//! one wavelength (three per cell, random sign and phase from a PCG hash of the cell and the seed),
//! each a cosine along the local compression direction under a Gaussian envelope of one
//! wavelength, so the lines are a few wavelengths long and follow the field. The noise is turned
//! into creases (narrow valleys, flat between) by `c = max(0, g / 1.5σ)²`:
//!
//! - **medium wrinkles**: depth `medium_depth · a`, with the activation
//!   `a = smoothstep(threshold, saturation, compression)`;
//! - **fine-line field**: depth `fine_depth · max(fine_rest, a)`, the fine lines present faintly at
//!   rest and deepened by compression.
//!
//! **Output** ([`TensionMaps`]):
//!
//! - `tension`: 4 channels `[compression, stretch, dir_x, dir_y]` (`dir` the unit compression
//!   direction in the texel's tangent frame, sign chosen with `dir_x ≥ 0`; zero where there is no
//!   compression);
//! - `wrinkles`: 3 channels `[medium height, fine height, cavity delta]`: heights in metres
//!   (≤ 0, folds go into the skin), the cavity/albedo delta in `[−1, 0]` (the medium crease
//!   times its activation) `[EXP-04]`;
//! - `normals`: 3 channels, the tangent-space normal of the summed height (`x` along +U, `y`
//!   along +V, `z` out), by central differences scaled to metres per texel; `(0, 0, 1)` where
//!   flat.
//!
//! Stretch reduces micro-detail contrast at shading time from the `stretch` channel; nothing here
//! bakes it in.

use humanart_core::rng::Rng;
use humanart_math::{Vec2, Vec3};

use super::raster::SurfaceMap;
use super::sampling::pcg_hash;
use super::{BakeError, BakeMap, BakeMesh, Coverage, Result};

/// What [`bake_tension`] does.
#[derive(Clone, Debug, PartialEq)]
pub struct WrinkleSettings {
    /// Medium wrinkle wavelength, metres (> 0).
    pub medium_wavelength: f32,
    /// Medium wrinkle depth at full activation, metres (≥ 0).
    pub medium_depth: f32,
    /// Fine-line wavelength, metres (> 0).
    pub fine_wavelength: f32,
    /// Fine-line depth at full activation, metres (≥ 0).
    pub fine_depth: f32,
    /// The fraction of the fine depth present without compression, `[0, 1]`.
    pub fine_rest: f32,
    /// Compression below which no medium wrinkle forms, `[0, 1)`.
    pub threshold: f32,
    /// Compression at which wrinkles reach full depth, above `threshold`.
    pub saturation: f32,
    /// Regions that grow wrinkles (empty: all).
    pub regions: Vec<u16>,
    /// The seed of the noise.
    pub seed: u32,
}

impl Default for WrinkleSettings {
    /// 6 mm / 0.5 mm medium wrinkles, 1.5 mm / 60 µm fine lines at 20 % at rest, activation
    /// from 2 % to 20 % compression.
    fn default() -> Self {
        Self {
            medium_wavelength: 0.006,
            medium_depth: 0.0005,
            fine_wavelength: 0.0015,
            fine_depth: 0.00006,
            fine_rest: 0.2,
            threshold: 0.02,
            saturation: 0.2,
            regions: Vec::new(),
            seed: 0,
        }
    }
}

impl WrinkleSettings {
    /// Checks the settings.
    pub fn check(&self) -> Result<()> {
        let pos = |v: f32| v > 0.0 && v.is_finite();
        let non_neg = |v: f32| v >= 0.0 && v.is_finite();
        if !pos(self.medium_wavelength) || !pos(self.fine_wavelength) {
            return Err(BakeError::invalid("wrinkles: wavelengths must be positive"));
        }
        if !non_neg(self.medium_depth) || !non_neg(self.fine_depth) || !(0.0..=1.0).contains(&self.fine_rest) {
            return Err(BakeError::invalid("wrinkles: depths must be non-negative, fine_rest in [0, 1]"));
        }
        if !(0.0..1.0).contains(&self.threshold) || self.saturation <= self.threshold || !self.saturation.is_finite() {
            return Err(BakeError::invalid("wrinkles: need 0 <= threshold < saturation"));
        }
        Ok(())
    }
}

/// The maps of [`bake_tension`].
#[derive(Clone, Debug, PartialEq)]
pub struct TensionMaps {
    /// `[compression, stretch, dir_x, dir_y]`.
    pub tension: BakeMap,
    /// `[medium height, fine height, cavity delta]`.
    pub wrinkles: BakeMap,
    /// The tangent-space normal of the wrinkle height.
    pub normals: BakeMap,
}

/// A symmetric 3 × 3 tensor `[xx, yy, zz, xy, xz, yz]`.
type Sym3 = [f32; 6];

fn sym_add(a: &mut Sym3, b: &Sym3, w: f32) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y * w;
    }
}

/// `uᵀ S v`.
fn sym_form(s: &Sym3, u: Vec3, v: Vec3) -> f32 {
    let sv = Vec3::new(
        s[0] * v.x + s[3] * v.y + s[4] * v.z,
        s[3] * v.x + s[1] * v.y + s[5] * v.z,
        s[4] * v.x + s[5] * v.y + s[2] * v.z,
    );
    u.dot(sv)
}

/// The Green strain of every welded vertex as a 3D tensor in the rest tangent plane.
fn vertex_strain(rest: &BakeMesh, deformed: &[Vec3]) -> Vec<Sym3> {
    let n = rest.positions().len();
    let keys = rest.position_keys();
    let key = |v: usize| keys.get(v).map_or(v, |&k| k as usize);
    let mut sum = vec![[0.0_f32; 6]; n];
    let mut weight = vec![0.0_f32; n];
    for t in 0..rest.triangle_count() {
        let Some([a, b, c]) = rest.triangle(t) else { continue };
        let p = rest.positions();
        let (e1, e2) = (p[b] - p[a], p[c] - p[a]);
        let normal = e1.cross(e2);
        let area = 0.5 * normal.length();
        let (Some(a0), Some(nn)) = (e1.try_normalize(), normal.try_normalize()) else { continue };
        let a1 = nn.cross(a0);
        // Rest edges in the 2D frame (a0, a1); deformed edges in 3D.
        let (r1, r2) = (Vec2::new(e1.dot(a0), e1.dot(a1)), Vec2::new(e2.dot(a0), e2.dot(a1)));
        let det = r1.x * r2.y - r2.x * r1.y;
        if det.abs() < f32::MIN_POSITIVE || !det.is_finite() {
            continue;
        }
        let (d1, d2) = (deformed[b] - deformed[a], deformed[c] - deformed[a]);
        // F = [d1 d2] · [r1 r2]⁻¹: the columns of F (images of the frame axes).
        let inv = [[r2.y / det, -r2.x / det], [-r1.y / det, r1.x / det]];
        let f0 = d1 * inv[0][0] + d2 * inv[1][0];
        let f1 = d1 * inv[0][1] + d2 * inv[1][1];
        // E = ½(FᵀF − I) in (a0, a1), lifted: S = Σ E_ij a_i a_jᵀ.
        let e00 = 0.5 * (f0.dot(f0) - 1.0);
        let e11 = 0.5 * (f1.dot(f1) - 1.0);
        let e01 = 0.5 * f0.dot(f1);
        let outer = |u: Vec3, v: Vec3| -> Sym3 {
            [
                u.x * v.x,
                u.y * v.y,
                u.z * v.z,
                f32::midpoint(u.x * v.y, u.y * v.x),
                f32::midpoint(u.x * v.z, u.z * v.x),
                f32::midpoint(u.y * v.z, u.z * v.y),
            ]
        };
        let mut s = [0.0_f32; 6];
        sym_add(&mut s, &outer(a0, a0), e00);
        sym_add(&mut s, &outer(a1, a1), e11);
        sym_add(&mut s, &outer(a0, a1), 2.0 * e01);
        if s.iter().any(|v| !v.is_finite()) {
            continue;
        }
        for v in [a, b, c] {
            sym_add(&mut sum[key(v)], &s, area);
            weight[key(v)] += area;
        }
    }
    (0..n)
        .map(|v| {
            let k = key(v);
            let w = weight[k];
            if w > 0.0 { sum[k].map(|x| x / w) } else { [0.0; 6] }
        })
        .collect()
}

/// Principal values `(e_min, e_max)` and the unit eigenvector of `e_min` of the symmetric 2 × 2
/// `[[a, b], [b, c]]`.
fn eigen2(a: f32, b: f32, c: f32) -> (f32, f32, Vec2) {
    let mean = f32::midpoint(a, c);
    let r = (0.25 * (a - c) * (a - c) + b * b).sqrt();
    let (lo, hi) = (mean - r, mean + r);
    // (A − hi·I) has the e_min eigenvector in its row space's orthogonal complement.
    let v = if b.abs() > 1e-12 {
        Vec2::new(b, lo - a)
    } else if a <= c {
        Vec2::new(1.0, 0.0)
    } else {
        Vec2::new(0.0, 1.0)
    };
    let v = v.normalize_or(Vec2::new(1.0, 0.0));
    let v = if v.x < 0.0 || (v.x == 0.0 && v.y < 0.0) { -v } else { v };
    (lo, hi, v)
}

/// Sparse-convolution Gabor noise at `p` with wavelength `lambda`, oriented along `dir`.
pub fn gabor(p: Vec3, dir: Vec3, lambda: f32, seed: u32) -> f32 {
    let cell = (p / lambda).floor();
    let mut sum = 0.0;
    let a2 = std::f32::consts::PI / (lambda * lambda);
    let k = std::f32::consts::TAU / lambda;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let c = cell + Vec3::new(dx as f32, dy as f32, dz as f32);
                let h = pcg_hash(
                    (c.x as i32) as u32 ^ pcg_hash((c.y as i32) as u32 ^ pcg_hash((c.z as i32) as u32 ^ seed)),
                );
                let mut rng = Rng::new(u64::from(h));
                for _ in 0..3 {
                    let at = (c + Vec3::new(rng.f32(), rng.f32(), rng.f32())) * lambda;
                    let w = if rng.f32() < 0.5 { -1.0 } else { 1.0 };
                    let phase = rng.f32() * std::f32::consts::TAU;
                    let r = p - at;
                    sum += w * (-a2 * r.length_squared()).exp() * (k * r.dot(dir) + phase).cos();
                }
            }
        }
    }
    sum
}

/// The standard deviation of [`gabor`]: three unit impulses per cubic wavelength, each with
/// `∫K² ≈ ½ (λ/√2)³`, give `σ² ≈ 0.53` (the test measures it).
pub const GABOR_SIGMA: f32 = 0.73;

fn crease(g: f32) -> f32 {
    (g / (1.5 * GABOR_SIGMA)).clamp(0.0, 1.0).powi(2)
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Bakes the tension and wrinkle maps of the pose `deformed` (one position per vertex of `rest`)
/// on the surface `surface` of the rest mesh.
pub fn bake_tension(
    rest: &BakeMesh,
    deformed: &[Vec3],
    surface: &SurfaceMap,
    settings: &WrinkleSettings,
) -> Result<TensionMaps> {
    settings.check()?;
    if deformed.len() != rest.positions().len() || deformed.iter().any(|p| !p.is_finite()) {
        return Err(BakeError::invalid("wrinkles: the pose needs one finite position per vertex"));
    }
    let strain = vertex_strain(rest, deformed);
    // Metres per texel along U and V, per triangle (from the rest UV Jacobian).
    let (w, h) = (surface.target.width as f32, surface.target.height as f32);
    let mut scale = vec![(0.0_f32, 0.0_f32); rest.triangle_count()];
    for (t, s) in scale.iter_mut().enumerate() {
        let Some([a, b, c]) = rest.triangle(t) else { continue };
        let (p, uv) = (rest.positions(), rest.uvs());
        let (e1, e2) = (p[b] - p[a], p[c] - p[a]);
        let (d1, d2) = (uv[b] - uv[a], uv[c] - uv[a]);
        let det = d1.x * d2.y - d2.x * d1.y;
        if det != 0.0 && det.is_finite() {
            let dpdu = (e1 * d2.y - e2 * d1.y) / det;
            let dpdv = (e2 * d1.x - e1 * d2.x) / det;
            *s = (dpdu.length() / w, dpdv.length() / h);
        }
    }
    let wanted = |r: u16| settings.regions.is_empty() || settings.regions.contains(&r);
    let mut wrinkles = BakeMap::new(&surface.target, 3, 0.0)?;
    let mut texel_scale = vec![(0.0_f32, 0.0_f32); surface.target.texel_count()];
    let width = surface.target.width as usize;
    let tension = surface.bake(rest, 4, 0.0, |x, y, p, out| {
        let i = y as usize * width + x as usize;
        let mut s = [0.0_f32; 6];
        if let Some(corners) = rest.triangle(p.triangle as usize) {
            for (&v, &wgt) in corners.iter().zip(&p.weights) {
                sym_add(&mut s, &strain[v], wgt);
            }
        }
        let (tt, bb) = (p.tangent, p.normal.cross(p.tangent));
        let (lo, hi, dir) = eigen2(sym_form(&s, tt, tt), sym_form(&s, tt, bb), sym_form(&s, bb, bb));
        let compression = (1.0 - (1.0 + 2.0 * lo).max(0.0).sqrt()).max(0.0);
        let stretch = ((1.0 + 2.0 * hi).max(0.0).sqrt() - 1.0).max(0.0);
        // The bitangent of the frame is sign · (n × t): express the direction in (t, sign·b).
        let dir = if compression > 0.0 { Vec2::new(dir.x, dir.y * p.sign) } else { Vec2::ZERO };
        out.copy_from_slice(&[compression, stretch, dir.x, dir.y]);
        let active =
            if wanted(p.region) { smoothstep(settings.threshold, settings.saturation, compression) } else { 0.0 };
        let fine_active = if wanted(p.region) { settings.fine_rest.max(active) } else { 0.0 };
        // Lines run across the compression; with no compression the fine lines follow U.
        let d3 = if compression > 0.0 { (tt * dir.x + p.bitangent * dir.y).normalize_or(tt) } else { tt };
        let medium = crease(gabor(p.position, d3, settings.medium_wavelength, settings.seed));
        let fine = crease(gabor(p.position, d3, settings.fine_wavelength, settings.seed ^ 0x9E37_79B9));
        wrinkles.set(
            i,
            &[-settings.medium_depth * active * medium, -settings.fine_depth * fine_active * fine, -active * medium],
            Coverage::Surface,
        );
        if let Some(sc) = scale.get(p.triangle as usize) {
            texel_scale[i] = *sc;
        }
    })?;
    let mut normals = BakeMap::new(&surface.target, 3, 0.0)?;
    let (wi, hi) = (surface.target.width as i64, surface.target.height as i64);
    let height_at = |x: i64, y: i64| -> Option<f32> {
        if x < 0 || y < 0 || x >= wi || y >= hi {
            return None;
        }
        let i = (y * wi + x) as usize;
        (wrinkles.coverage[i] == Coverage::Surface).then(|| wrinkles.data[i * 3] + wrinkles.data[i * 3 + 1])
    };
    for y in 0..hi {
        for x in 0..wi {
            let i = (y * wi + x) as usize;
            if wrinkles.coverage[i] != Coverage::Surface {
                continue;
            }
            let (su, sv) = texel_scale[i];
            let slope = |a: Option<f32>, b: Option<f32>, c: f32, step: f32| -> f32 {
                if step <= 0.0 {
                    return 0.0;
                }
                match (a, b) {
                    (Some(a), Some(b)) => (b - a) / (2.0 * step),
                    (None, Some(b)) => (b - c) / step,
                    (Some(a), None) => (c - a) / step,
                    (None, None) => 0.0,
                }
            };
            let c = wrinkles.data[i * 3] + wrinkles.data[i * 3 + 1];
            let dx = slope(height_at(x - 1, y), height_at(x + 1, y), c, su);
            let dy = slope(height_at(x, y - 1), height_at(x, y + 1), c, sv);
            let n = Vec3::new(-dx, -dy, 1.0).normalize();
            normals.set(i, &n.to_array(), Coverage::Surface);
        }
    }
    Ok(TensionMaps { tension, wrinkles, normals })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bake::BakeTarget;
    use crate::bake::raster::rasterize;
    use crate::bake::tests::box_mesh;

    /// The +Z face of a flat 6 x 6 cm box squeezed along x by 25 %.
    fn squeezed() -> (BakeMesh, Vec<Vec3>, SurfaceMap) {
        let (mesh, _) = box_mesh(0.06, 0.06, 0.06, 24);
        let deformed: Vec<Vec3> = mesh.positions().iter().map(|p| Vec3::new(p.x * 0.75, p.y, p.z)).collect();
        let surface = rasterize(&mesh, &BakeTarget::new(0, 384, 256)).unwrap();
        (mesh, deformed, surface)
    }

    #[test]
    fn bake_05_tension_measures_uniform_compression() {
        let (mesh, deformed, surface) = squeezed();
        let maps = bake_tension(&mesh, &deformed, &surface, &WrinkleSettings::default()).unwrap();
        let mut checked = 0;
        for (i, t, b1, b2) in surface.covered() {
            let p = mesh.point(t, b1, b2).unwrap();
            if p.normal.z > 0.99 && p.position.x.abs() < 0.025 && p.position.y.abs() < 0.025 {
                let v = maps.tension.texel(i).unwrap();
                assert!((v[0] - 0.25).abs() < 1e-3 && v[1].abs() < 1e-3, "{v:?}");
                // The compression direction is world x, expressed in the tangent frame.
                let world = p.tangent * v[2] + p.bitangent * v[3];
                assert!(world.x.abs() > 0.999, "{world:?}");
                checked += 1;
            }
        }
        assert!(checked > 100);
        // The identity pose has no tension and no medium wrinkles.
        let maps = bake_tension(&mesh, mesh.positions(), &surface, &WrinkleSettings::default()).unwrap();
        for (i, ..) in surface.covered() {
            assert!(maps.tension.texel(i).unwrap()[0] < 1e-5);
            assert_eq!(maps.wrinkles.texel(i).unwrap()[0], 0.0);
        }
    }

    #[test]
    fn bake_05_wrinkles_run_across_the_compression() {
        let (mesh, deformed, surface) = squeezed();
        let settings = WrinkleSettings { fine_depth: 0.0, ..WrinkleSettings::default() };
        let maps = bake_tension(&mesh, &deformed, &surface, &settings).unwrap();
        // On the +Z face, x is along U and -y along V (box layout): height varies fast along x.
        let (mut along_x, mut along_y, mut deepest) = (0.0_f64, 0.0_f64, 0.0_f32);
        let w = 384;
        for (i, t, b1, b2) in surface.covered() {
            let p = mesh.point(t, b1, b2).unwrap();
            if !(p.normal.z > 0.99 && p.position.x.abs() < 0.025 && p.position.y.abs() < 0.025) {
                continue;
            }
            let h = |j: usize| f64::from(maps.wrinkles.texel(j).unwrap()[0]);
            along_x += (h(i + 1) - h(i)).abs();
            along_y += (h(i + w) - h(i)).abs();
            deepest = deepest.min(maps.wrinkles.texel(i).unwrap()[0]);
        }
        assert!(along_x > 3.0 * along_y, "{along_x} vs {along_y}");
        assert!(deepest < -0.5 * settings.medium_depth && deepest >= -settings.medium_depth, "{deepest}");
        // Deterministic.
        let again = bake_tension(&mesh, &deformed, &surface, &settings).unwrap();
        assert_eq!(again, maps);
        // Normals tilt where the height changes.
        let tilted = surface.covered().filter(|&(i, ..)| maps.normals.texel(i).unwrap()[2] < 0.99).count();
        assert!(tilted > 100, "{tilted}");
    }

    #[test]
    fn eigen_and_settings() {
        let (lo, hi, v) = eigen2(2.0, 0.0, 1.0);
        assert_eq!((lo, hi), (1.0, 2.0));
        assert_eq!(v, Vec2::new(0.0, 1.0));
        let (lo, hi, v) = eigen2(1.0, 1.0, 1.0);
        assert!((lo - 0.0).abs() < 1e-6 && (hi - 2.0).abs() < 1e-6);
        assert!((v.x - v.y.abs()).abs() < 1e-6 && v.y < 0.0);
        assert!(WrinkleSettings { saturation: 0.0, ..WrinkleSettings::default() }.check().is_err());
        // The noise's spread is what GABOR_SIGMA says.
        let mut rng = Rng::new(5);
        let n = 4000;
        let var: f32 = (0..n)
            .map(|_| {
                let p = Vec3::new(rng.f32(), rng.f32(), rng.f32()) * 0.1;
                gabor(p, Vec3::X, 0.01, 1).powi(2)
            })
            .sum::<f32>()
            / n as f32;
        assert!((var.sqrt() - GABOR_SIGMA).abs() < 0.04, "{}", var.sqrt());
    }
}
