//! The micro-detail generator's CPU reference `[SKN-20]` `[SKN-21]` `[SKN-22]`.
//!
//! The generator makes a periodic **detail exemplar**: a square height tile (millimetres) a few
//! millimetres of skin wide, at a resolution far above the stored textures. The render passes
//! sample it with histogram-preserving tiling and blending ([`super::antitile`]), rotated to the
//! local skin-line direction and scaled by the region's density/depth maps, so no repetition is
//! visible. The GPU kernel (`shaders/skin/microdetail.wgsl`) evaluates exactly this function from
//! the same inputs; this module is its reference and the subject of the statistics tests.
//!
//! The height is the sum of
//!
//! - **pores**: anisotropic Gaussian pits at blue-noise centres (Poisson-disk sampling on the torus,
//!   Bridson 2007, minimum distance = the pore spacing), with per-pore depth and size jitter, an
//!   elongation along the skin-line direction;
//! - **fine lines**: two families of grooves crossing at an angle (the crosshatch of Langer-line
//!   aligned skin lines), each family periodic on the tile (integer normal vectors), wobbled by
//!   periodic noise, with per-line depth jitter and depth fading in and out along each line;
//! - **micro-relief**: low-amplitude periodic fractal noise.
//!
//! **Age** deepens and widens the lines (lost elasticity), enlarges pores and roughens the relief.
//! Crow's feet, nasolabial and forehead lines are macro wrinkles placed by region: they are
//! procedural skin features ([`super::features`]) whose depth also follows age.
//!
//! The mapping from the registry's normalised parameters to millimetres ([`MicroDetailParams::from_registry`])
//! is an **artistic default** to be fitted to captured detail statistics when the exemplar
//! library of `[APR-04]` exists; nothing here is measured.

use humanart_core::rng::Rng;

use super::error::SkinError;
use super::noise::{fbm, hash3, unit};

/// Largest exemplar side in texels.
pub const MAX_EXEMPLAR_SIZE: u32 = 2048;
/// Most pores in one exemplar.
pub const MAX_PORES: usize = 1 << 16;
/// Horizon directions of the micro-shadowing map `[SKN-21]`.
pub const HORIZON_DIRECTIONS: usize = 8;

/// Generator settings in physical units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MicroDetailParams {
    /// Exemplar side in texels (power of two, up to [`MAX_EXEMPLAR_SIZE`]).
    pub size: u32,
    /// Texel size in mm.
    pub texel_mm: f32,
    /// Minimum distance between pore centres, mm.
    pub pore_spacing_mm: f32,
    /// Pore radius (Gaussian σ), mm.
    pub pore_radius_mm: f32,
    /// Pore depth, mm.
    pub pore_depth_mm: f32,
    /// Pore elongation 0..1 (1: three times longer than wide).
    pub pore_elongation: f32,
    /// Fine-line depth, mm.
    pub line_depth_mm: f32,
    /// Fine-line spacing, mm.
    pub line_spacing_mm: f32,
    /// Fine-line half width (Gaussian σ), mm.
    pub line_width_mm: f32,
    /// Micro-relief amplitude, mm.
    pub relief_mm: f32,
    /// Age 0..1.
    pub age: f32,
    /// Random seed.
    pub seed: u32,
}

impl Default for MicroDetailParams {
    fn default() -> Self {
        Self::from_registry(0.5, 0.5, 0.5, 0.3, 0.4, 0.5, 0.3, 1)
    }
}

impl MicroDetailParams {
    /// Settings from the registry's normalised values (`skin.pore-density`, `skin.pore-depth`,
    /// `skin.pore-size`, `skin.pore-elongation`, `skin.fine-line-depth`, `skin.fine-line-scale`,
    /// and the identity's age 0..1). A 512² exemplar of 0.02 mm texels (10.24 mm of skin).
    pub fn from_registry(
        density: f32,
        depth: f32,
        size: f32,
        elongation: f32,
        line_depth: f32,
        line_scale: f32,
        age: f32,
        seed: u32,
    ) -> Self {
        let c = |v: f32| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.5 };
        Self {
            size: 512,
            texel_mm: 0.02,
            pore_spacing_mm: 0.6 - 0.35 * c(density),
            pore_radius_mm: 0.025 + 0.035 * c(size),
            pore_depth_mm: 0.01 + 0.04 * c(depth),
            pore_elongation: c(elongation),
            line_depth_mm: 0.003 + 0.02 * c(line_depth),
            line_spacing_mm: 0.25 + 0.5 * c(line_scale),
            line_width_mm: 0.02 + 0.02 * c(line_scale),
            relief_mm: 0.004,
            age: c(age),
            seed,
        }
    }

    /// Checks sizes and values.
    pub fn check(&self) -> Result<(), SkinError> {
        if !self.size.is_power_of_two() || self.size < 16 || self.size > MAX_EXEMPLAR_SIZE {
            return Err(SkinError::Invalid { what: "exemplar size must be a power of two in 16..=2048" });
        }
        let values = [
            self.texel_mm,
            self.pore_spacing_mm,
            self.pore_radius_mm,
            self.pore_depth_mm,
            self.pore_elongation,
            self.line_depth_mm,
            self.line_spacing_mm,
            self.line_width_mm,
            self.relief_mm,
            self.age,
        ];
        if values.iter().any(|v| !v.is_finite() || *v < 0.0) || self.texel_mm <= 0.0 || self.pore_spacing_mm <= 0.0 {
            return Err(SkinError::Invalid {
                what: "micro-detail values must be finite, non-negative, with positive texel and spacing",
            });
        }
        if self.line_spacing_mm < self.texel_mm * 2.0 {
            return Err(SkinError::Invalid { what: "fine-line spacing below two texels" });
        }
        Ok(())
    }

    /// The tile side in mm.
    pub fn tile_mm(&self) -> f32 {
        self.size as f32 * self.texel_mm
    }
}

/// The derived, GPU-ready description of one exemplar: everything the kernel reads.
#[derive(Clone, Debug, PartialEq)]
pub struct ExemplarPlan {
    /// The settings.
    pub params: MicroDetailParams,
    /// Pores: `[x, y, depth, sigma]` in texels / mm (depth in mm, x/y/sigma in texels).
    pub pores: Vec<[f32; 4]>,
    /// Pore grid cells per side.
    pub grid: u32,
    /// CSR start of every grid cell into `pores` (`grid² + 1` entries).
    pub cell_start: Vec<u32>,
    /// The two line families: integer normals `(p, q)` and line counts per period.
    pub lines: [[i32; 3]; 2],
    /// Effective (age-adjusted) line depth, line σ in texels, pore σ scale, relief amplitude.
    pub age_terms: [f32; 4],
}

/// Poisson-disk points on a `side × side` torus with minimum distance `radius` (Bridson 2007,
/// 30 candidates per active point), deterministic from `seed`.
pub fn poisson_torus(side: f32, radius: f32, seed: u64) -> Result<Vec<[f32; 2]>, SkinError> {
    if !(side.is_finite() && radius.is_finite()) || radius <= 0.0 || side <= radius {
        return Err(SkinError::Invalid { what: "Poisson sampling needs side > radius > 0" });
    }
    // Equal cells no wider than r/√2 (at most one point each), tiling the torus exactly.
    let cells = (side / (radius / std::f32::consts::SQRT_2)).ceil() as usize;
    let cell = side / cells as f32;
    let reach = (radius / cell).ceil() as isize;
    if cells * cells > MAX_PORES * 4 {
        return Err(SkinError::limit("Poisson grid cells", (MAX_PORES * 4) as u64, (cells * cells) as u64));
    }
    let mut grid = vec![u32::MAX; cells * cells];
    let mut points: Vec<[f32; 2]> = Vec::new();
    let mut active: Vec<u32> = Vec::new();
    let mut rng = Rng::new(seed);
    let wrap = |v: f32| v.rem_euclid(side);
    let cell_of =
        |p: [f32; 2]| ((p[0] / cell) as usize).min(cells - 1) + ((p[1] / cell) as usize).min(cells - 1) * cells;
    let first = [rng.f32() * side, rng.f32() * side];
    grid[cell_of(first)] = 0;
    points.push(first);
    active.push(0);
    while let Some(&current) = active.last() {
        if points.len() >= MAX_PORES {
            break;
        }
        let origin = points[current as usize];
        let mut placed = false;
        for _ in 0..30 {
            let angle = rng.f32() * std::f32::consts::TAU;
            let dist = radius * (1.0 + rng.f32());
            let p = [wrap(origin[0] + dist * angle.cos()), wrap(origin[1] + dist * angle.sin())];
            let (cx, cy) = ((p[0] / cell) as isize, (p[1] / cell) as isize);
            let mut ok = true;
            'search: for dy in -reach..=reach {
                for dx in -reach..=reach {
                    let gx = (cx + dx).rem_euclid(cells as isize) as usize;
                    let gy = (cy + dy).rem_euclid(cells as isize) as usize;
                    let index = grid[gx + gy * cells];
                    if index == u32::MAX {
                        continue;
                    }
                    let q = points[index as usize];
                    let mut ddx = (p[0] - q[0]).abs();
                    let mut ddy = (p[1] - q[1]).abs();
                    ddx = ddx.min(side - ddx);
                    ddy = ddy.min(side - ddy);
                    if ddx * ddx + ddy * ddy < radius * radius {
                        ok = false;
                        break 'search;
                    }
                }
            }
            if ok && grid[cell_of(p)] == u32::MAX {
                grid[cell_of(p)] = points.len() as u32;
                active.push(points.len() as u32);
                points.push(p);
                placed = true;
                break;
            }
        }
        if !placed {
            active.pop();
        }
    }
    Ok(points)
}

/// The integer normal `(p, q)` closest in direction to `angle` with `|p|, |q| ≤ 4`, `p ≥ 1`.
fn integer_normal(angle: f32) -> (i32, i32) {
    let mut best = (1, 0);
    let mut best_err = f32::MAX;
    for p in 1..=4_i32 {
        for q in -4..=4_i32 {
            let a = (q as f32).atan2(p as f32);
            let err = (a - angle).abs();
            if err < best_err - 1e-6 {
                best = (p, q);
                best_err = err;
            }
        }
    }
    best
}

impl ExemplarPlan {
    /// Plans an exemplar: pore centres, the pore grid and the line families.
    pub fn new(params: &MicroDetailParams) -> Result<Self, SkinError> {
        params.check()?;
        let side = params.size as f32;
        let age = params.age;
        let pore_scale = 1.0 + 0.3 * age;
        let radius_texels = params.pore_spacing_mm / params.texel_mm;
        let sigma = params.pore_radius_mm / params.texel_mm * pore_scale;
        let centres = poisson_torus(side, radius_texels, u64::from(params.seed))?;
        let e = 1.0 + 2.0 * params.pore_elongation;
        // Influence radius: 3 sigma along the long axis (with the 1.3 size jitter).
        let reach = 3.0 * sigma * e.sqrt() * 1.3;
        let grid = ((side / reach.max(1.0)).floor() as u32).clamp(1, 256);
        let mut pores: Vec<[f32; 4]> = centres
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let h = hash3(i as u32, 17, params.seed);
                let depth = params.pore_depth_mm * (0.6 + 0.8 * unit(h));
                let s = sigma * (0.7 + 0.6 * unit(hash3(i as u32, 29, params.seed)));
                [c[0], c[1], depth, s]
            })
            .collect();
        let cell_size = side / grid as f32;
        let cell_of = |p: &[f32; 4]| {
            let cx = ((p[0] / cell_size) as u32).min(grid - 1);
            let cy = ((p[1] / cell_size) as u32).min(grid - 1);
            cy * grid + cx
        };
        pores.sort_by_key(|p| cell_of(p));
        let mut cell_start = vec![0_u32; (grid * grid) as usize + 1];
        for p in &pores {
            cell_start[cell_of(p) as usize + 1] += 1;
        }
        for i in 1..cell_start.len() {
            cell_start[i] += cell_start[i - 1];
        }
        // Crosshatch: normals at ±35° around the x axis (lines run roughly along y, the skin-line
        // direction the sampler rotates into place).
        let mut lines = [[0_i32; 3]; 2];
        for (family, angle) in lines.iter_mut().zip([0.61_f32, -0.61]) {
            let (p, q) = integer_normal(angle);
            let len = ((p * p + q * q) as f32).sqrt();
            let count = (params.tile_mm() / (params.line_spacing_mm * len)).round().max(1.0) as i32;
            *family = [p, q, count];
        }
        let line_depth = params.line_depth_mm * (0.4 + 1.6 * age);
        let line_sigma = params.line_width_mm / params.texel_mm * (1.0 + 0.5 * age);
        let relief = params.relief_mm * (1.0 + age);
        Ok(Self { params: *params, pores, grid, cell_start, lines, age_terms: [line_depth, line_sigma, e, relief] })
    }

    /// The height in mm at texel centre `(x, y)` (wrapping).
    pub fn height(&self, x: u32, y: u32) -> f32 {
        let side = self.params.size as f32;
        let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
        let mut h = 0.0;
        // Pores.
        let e = self.age_terms[2];
        let (ex, ey) = (e.sqrt(), 1.0 / e.sqrt());
        let cell = side / self.grid as f32;
        let (cx, cy) = ((px / cell) as i32, (py / cell) as i32);
        let g = self.grid as i32;
        let mut visited = [u32::MAX; 9];
        let mut n_visited = 0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let gx = (cx + dx).rem_euclid(g) as u32;
                let gy = (cy + dy).rem_euclid(g) as u32;
                let c = gy * self.grid + gx;
                // A grid of fewer than 3 cells wraps onto itself: visit each cell once.
                if visited[..n_visited].contains(&c) {
                    continue;
                }
                visited[n_visited] = c;
                n_visited += 1;
                let (Some(&s), Some(&t)) = (self.cell_start.get(c as usize), self.cell_start.get(c as usize + 1))
                else {
                    continue;
                };
                for pore in self.pores.get(s as usize..t as usize).unwrap_or(&[]) {
                    let mut ddx = px - pore[0];
                    let mut ddy = py - pore[1];
                    ddx -= side * (ddx / side).round();
                    ddy -= side * (ddy / side).round();
                    // Elongated along y (the skin-line direction).
                    let u = ddx / (pore[3] * ey);
                    let v = ddy / (pore[3] * ex);
                    h -= pore[2] * (-0.5 * (u * u + v * v)).exp();
                }
            }
        }
        // Fine lines.
        let seed = self.params.seed;
        for (k, family) in self.lines.iter().enumerate() {
            let [p, q, count] = *family;
            let warp = 0.35 * fbm(px / side * 4.0, py / side * 4.0, 4, 3, 0.5, seed.wrapping_add(101 + k as u32));
            let u = ((p as f32 * px + q as f32 * py) / side) * count as f32 + warp;
            let line = u.floor();
            let f = u - line;
            let d = f.min(1.0 - f);
            let nearest = if f < 0.5 { line } else { line + 1.0 };
            let index = (nearest as i32).rem_euclid(count.max(1)) as u32;
            let along = fbm(px / side * 8.0, py / side * 8.0, 8, 2, 0.5, seed.wrapping_add(301 + k as u32));
            // Depth varies along each line (lines fade in and out) and between lines.
            let jitter = (0.5 + unit(hash3(index, k as u32 + 7, seed))) * (0.3 + 2.8 * (along + 0.5)).clamp(0.0, 1.7);
            let spacing = side / (count as f32 * ((p * p + q * q) as f32).sqrt());
            let dist = d * spacing;
            let sigma = self.age_terms[1];
            let weight = if k == 0 { 1.0 } else { 0.7 };
            h -= weight * self.age_terms[0] * jitter * (-0.5 * (dist / sigma) * (dist / sigma)).exp();
        }
        // Micro-relief.
        h += self.age_terms[3] * fbm(px / side * 16.0, py / side * 16.0, 16, 4, 0.5, seed.wrapping_add(211));
        h
    }

    /// The whole exemplar, row-major, in mm.
    pub fn heights(&self) -> Vec<f32> {
        let n = self.params.size;
        let mut out = Vec::with_capacity((n * n) as usize);
        for y in 0..n {
            for x in 0..n {
                out.push(self.height(x, y));
            }
        }
        out
    }

    /// The GPU word streams: `pores` as 4 f32 each, `cell_start` as u32.
    pub fn gpu_pore_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.pores.len() * 16);
        for p in &self.pores {
            for v in p {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    /// The CSR cell starts as little-endian bytes.
    pub fn gpu_cell_bytes(&self) -> Vec<u8> {
        self.cell_start.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

/// Tangent-space normal `(x, y)` (z = √(1 - x² - y²)) of a periodic height field in mm with texels
/// of `texel_mm`, by central differences.
pub fn normals_from_height(height: &[f32], size: usize, texel_mm: f32) -> Vec<[f32; 2]> {
    let mut out = vec![[0.0; 2]; height.len()];
    if height.len() != size * size || size == 0 {
        return out;
    }
    for y in 0..size {
        for x in 0..size {
            let h = |xx: usize, yy: usize| height[(yy % size) * size + xx % size];
            let dx = (h(x + 1, y) - h(x + size - 1, y)) / (2.0 * texel_mm);
            let dy = (h(x, y + 1) - h(x, y + size - 1)) / (2.0 * texel_mm);
            let len = (dx * dx + dy * dy + 1.0).sqrt();
            out[y * size + x] = [-dx / len, -dy / len];
        }
    }
    out
}

/// The horizon map `[SKN-21]`: for every texel and each of [`HORIZON_DIRECTIONS`] azimuths, the sine
/// of the highest elevation angle of the height field seen within `radius` texels. Light below
/// the horizon at its azimuth is shadowed by the micro-relief.
pub fn horizon_map(height: &[f32], size: usize, texel_mm: f32, radius: u32) -> Vec<[f32; HORIZON_DIRECTIONS]> {
    let mut out = vec![[0.0; HORIZON_DIRECTIONS]; height.len()];
    if height.len() != size * size || size == 0 {
        return out;
    }
    for y in 0..size {
        for x in 0..size {
            let h0 = height[y * size + x];
            for (d, slot) in out[y * size + x].iter_mut().enumerate() {
                let angle = d as f32 * std::f32::consts::TAU / HORIZON_DIRECTIONS as f32;
                let (dx, dy) = (angle.cos(), angle.sin());
                let mut best: f32 = 0.0;
                for step in 1..=radius {
                    let sx = (x as f32 + dx * step as f32).round() as isize;
                    let sy = (y as f32 + dy * step as f32).round() as isize;
                    let sx = sx.rem_euclid(size as isize) as usize;
                    let sy = sy.rem_euclid(size as isize) as usize;
                    let rise = height[sy * size + sx] - h0;
                    let run = step as f32 * texel_mm;
                    best = best.max(rise / (rise * rise + run * run).sqrt());
                }
                *slot = best;
            }
        }
    }
    out
}

/// Horizon-based self-shadowing `[SKN-21]`: the visibility of a light at elevation sine
/// `light_sin` given the horizon sine at its azimuth, with a soft transition of `softness`.
pub fn horizon_visibility(horizon_sin: f32, light_sin: f32, softness: f32) -> f32 {
    let s = softness.max(1e-4);
    ((light_sin - horizon_sin) / s + 0.5).clamp(0.0, 1.0)
}

/// Cavity-driven micro-shadowing `[SKN-21]` `[SHD-09]`: `saturate(|N·L| + 2·ao² - 1)` (D. Chan,
/// "Material Advances in Call of Duty: WWII", SIGGRAPH 2018 Advances course).
pub fn micro_shadow(ao: f32, n_dot_l: f32) -> f32 {
    (n_dot_l.abs() + 2.0 * ao * ao - 1.0).clamp(0.0, 1.0)
}

/// Vellus hair (peach fuzz) settings `[SKN-22]` for the fuzz lobe of `[SHD-07]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VellusLobe {
    /// Strength of the retro-reflective rim lobe, 0..1.
    pub strength: f32,
    /// Width of the lobe (sheen roughness), 0..1.
    pub width: f32,
    /// Whether sparse strands should be drawn at close range.
    pub strands: bool,
}

/// The fuzz lobe for a texel: `density` is the registry's `skin.vellus-density`, `zone_mask` the
/// summed masks of the vellus zones (cheeks, upper lip, forehead, ears) clamped to 1, `age` 0..1
/// and `masculinity` 0..1 (the sex axis: more terminal hair, less fuzz). Artistic defaults.
pub fn vellus_lobe(density: f32, zone_mask: f32, age: f32, masculinity: f32) -> VellusLobe {
    let c = |v: f32| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
    let strength = c(density) * (0.35 + 0.65 * c(zone_mask)) * (1.0 - 0.4 * c(age)) * (1.0 - 0.3 * c(masculinity));
    VellusLobe { strength, width: 0.3 + 0.4 * c(age), strands: strength > 0.4 }
}

#[cfg(test)]
mod tests {
    use super::super::spectrum::{angular_power, power_spectrum, radial_power};
    use super::*;
    use humanart_core::rng::Rng;

    fn plan(params: &MicroDetailParams) -> ExemplarPlan {
        ExemplarPlan::new(params).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn poisson_points_keep_their_distance() {
        let side = 128.0;
        let radius = 6.0;
        let points = poisson_torus(side, radius, 4).unwrap_or_else(|e| panic!("{e}"));
        for (i, a) in points.iter().enumerate() {
            for b in &points[i + 1..] {
                let mut dx = (a[0] - b[0]).abs();
                let mut dy = (a[1] - b[1]).abs();
                dx = dx.min(side - dx);
                dy = dy.min(side - dy);
                assert!(dx * dx + dy * dy >= radius * radius * 0.999);
            }
        }
        // Maximal: Bridson fills to roughly 0.6-0.7 of the hexagonal packing density.
        let hex = side * side / (radius * radius * 3_f32.sqrt() / 2.0);
        let fill = points.len() as f32 / hex;
        assert!(fill > 0.5 && fill < 0.85, "{fill}");
        assert!(poisson_torus(4.0, 5.0, 1).is_err());
    }

    #[test]
    fn exemplar_is_periodic_and_signed() {
        let p = MicroDetailParams { size: 128, ..MicroDetailParams::default() };
        let plan = plan(&p);
        assert_eq!(plan.cell_start.last().copied(), Some(plan.pores.len() as u32));
        let h = plan.heights();
        assert!(h.iter().all(|v| v.is_finite()));
        let min = h.iter().copied().fold(f32::MAX, f32::min);
        assert!(min < -0.005, "pits exist: {min}");
        // Seamless: the step across the wrap edge is like a step inside the tile.
        let n = 128;
        let inner: f32 = (0..n).map(|y| (h[y * n + 64] - h[y * n + 63]).abs()).sum::<f32>() / n as f32;
        let edge: f32 = (0..n).map(|y| (h[y * n] - h[y * n + n - 1]).abs()).sum::<f32>() / n as f32;
        assert!(edge < inner * 2.5 + 1e-4, "edge {edge} inner {inner}");
        assert!(MicroDetailParams { size: 100, ..p }.check().is_err());
        assert!(MicroDetailParams { texel_mm: f32::NAN, ..p }.check().is_err());
    }

    #[test]
    fn ac_wp13_2_pore_field_has_a_blue_noise_spectrum() {
        // [AC-WP13-2] Pores only (no lines, no relief): power concentrates around the pore spacing
        // frequency and is suppressed at low frequency, the signature of blue-noise centres.
        let p = MicroDetailParams {
            size: 256,
            line_depth_mm: 0.0,
            relief_mm: 0.0,
            pore_elongation: 0.0,
            ..MicroDetailParams::default()
        };
        let plan = plan(&p);
        let h = plan.heights();
        let size = 256;
        let spectrum = power_spectrum(&h, size).unwrap_or_else(|e| panic!("{e}"));
        let radial = radial_power(&spectrum, size);
        // Expected peak near side / spacing cycles per tile.
        let spacing_texels = p.pore_spacing_mm / p.texel_mm;
        let f0 = size as f32 / spacing_texels;
        let peak = radial.iter().enumerate().skip(1).max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
        assert!((peak as f32 / f0 - 1.0).abs() < 0.35, "peak {peak} expected near {f0}");
        // Per-ring power density (power / ring circumference): low frequencies are suppressed
        // relative to the peak, unlike the same pits at white-noise (uniform random) centres.
        let low_ratio = |radial: &[f64]| {
            let density = |r: usize| radial[r] / (r as f64 * std::f64::consts::TAU);
            let top = (f0 * 0.3) as usize;
            let low: f64 = (1..top).map(density).sum::<f64>() / (top - 1).max(1) as f64;
            let peak = (1..radial.len()).map(density).fold(0.0, f64::max);
            low / peak
        };
        let blue = low_ratio(&radial);
        let mut white = vec![0.0_f32; size * size];
        let mut rng = Rng::new(99);
        for pore in &plan.pores {
            let (cx, cy) = (rng.f32() * size as f32, rng.f32() * size as f32);
            let reach = (pore[3] * 4.0) as i32 + 1;
            for dy in -reach..=reach {
                for dx in -reach..=reach {
                    let x = ((cx as i32 + dx).rem_euclid(size as i32)) as usize;
                    let y = ((cy as i32 + dy).rem_euclid(size as i32)) as usize;
                    let (ddx, ddy) = (cx.floor() + dx as f32 + 0.5 - cx, cy.floor() + dy as f32 + 0.5 - cy);
                    let r2 = (ddx * ddx + ddy * ddy) / (pore[3] * pore[3]);
                    white[y * size + x] -= pore[2] * (-0.5 * r2).exp();
                }
            }
        }
        let white_spectrum = power_spectrum(&white, size).unwrap_or_else(|e| panic!("{e}"));
        let white_ratio = low_ratio(&radial_power(&white_spectrum, size));
        assert!(blue < 0.4 * white_ratio, "blue {blue} vs white {white_ratio}");
    }

    #[test]
    fn ac_wp13_2_fine_lines_are_a_crosshatch_and_age_deepens_them() {
        let p = MicroDetailParams { size: 256, pore_depth_mm: 0.0, relief_mm: 0.0, ..MicroDetailParams::default() };
        let size = 256;
        let young = plan(&MicroDetailParams { age: 0.0, ..p }).heights();
        let old = plan(&MicroDetailParams { age: 1.0, ..p }).heights();
        let s_young = power_spectrum(&young, size).unwrap_or_else(|e| panic!("{e}"));
        let s_old = power_spectrum(&old, size).unwrap_or_else(|e| panic!("{e}"));
        let total = |s: &[f64]| s.iter().sum::<f64>();
        assert!(total(&s_old) > total(&s_young) * 4.0, "age deepens the lines");
        // The power sits at the two line normals (about ±35° from x), not spread evenly.
        let angular = angular_power(&s_young, size, 12, 4.0, 128.0);
        let sum: f64 = angular.iter().sum();
        let mut sorted = angular.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        assert!((sorted[0] + sorted[1] + sorted[2] + sorted[3]) / sum > 0.7, "{angular:?}");
        // The vertical-frequency bins (90°) hold almost nothing.
        assert!(angular[6] / sum < 0.03, "{angular:?}");
    }

    #[test]
    fn skn21_horizon_and_micro_shadow() {
        // A single wall: texels just in front of it see a high horizon towards it.
        let size = 32;
        let mut h = vec![0.0_f32; size * size];
        for y in 0..size {
            h[y * size + 20] = 0.1;
        }
        let map = horizon_map(&h, size, 0.02, 8);
        let at = map[16 * size + 18];
        assert!(at[0] > 0.9, "towards +x: {at:?}");
        assert!(at[4] < 1e-6, "towards -x is open");
        assert_eq!(horizon_visibility(0.5, 0.1, 0.05), 0.0);
        assert_eq!(horizon_visibility(0.1, 0.5, 0.05), 1.0);
        assert_eq!(micro_shadow(1.0, 0.0), 1.0);
        assert_eq!(micro_shadow(0.5, 0.2), 0.0);
        assert!((micro_shadow(0.8, 0.5) - 0.78).abs() < 1e-6);
        let n = normals_from_height(&h, size, 0.02);
        assert!(n[16 * size + 19][0] < -0.5, "slope towards the wall tilts the normal away");
    }

    #[test]
    fn skn22_vellus() {
        let cheeks = vellus_lobe(1.0, 1.0, 0.0, 0.0);
        let body = vellus_lobe(1.0, 0.0, 0.0, 0.0);
        let old = vellus_lobe(1.0, 1.0, 1.0, 0.0);
        assert!(cheeks.strength > body.strength && cheeks.strength > old.strength);
        assert!(cheeks.strands && !body.strands);
        assert_eq!(vellus_lobe(f32::NAN, 1.0, 0.0, 0.0).strength, 0.0);
    }
}
