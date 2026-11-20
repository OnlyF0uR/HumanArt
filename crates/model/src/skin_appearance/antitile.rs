//! Histogram-preserving tiling and blending `[SKN-20]`, after E. Heitz and F. Neyret, "High-Performance
//! By-Example Noise using a Histogram-Preserving Blending Operator", HPG 2018.
//!
//! The exemplar is **Gaussianised** once: every texel is replaced by the Gaussian quantile of its
//! rank (mean 0.5, σ = 1/6, as in the paper), and a lookup table maps Gaussian values back to the
//! exemplar's histogram. Sampling picks the three vertices of the triangle grid around the point
//! (the paper's skewed grid at 2√3 the exemplar frequency), reads the Gaussianised exemplar at the
//! point shifted by a random offset per vertex, blends with the barycentric weights, restores the
//! variance (`(G - 0.5) / √Σw² + 0.5`) and maps back through the table. The result has the
//! exemplar's histogram and local look with no visible repetition.
//!
//! Offsets come from the PCG hash of [`super::noise`] so the WGSL sampler (`skin/antitile.wgsl`)
//! reproduces the CPU reference. The normal CDF uses the Abramowitz & Stegun 7.1.26 erf
//! approximation (|error| < 1.5e-7); its inverse is tabulated by bisection.

use super::error::SkinError;
use super::noise::{hash3, unit};

/// Entries of the inverse lookup table.
pub const LUT_SIZE: usize = 1024;
/// σ of the Gaussianised exemplar (mean 0.5).
pub const GAUSS_SIGMA: f32 = 1.0 / 6.0;

/// `erf(x)` by Abramowitz & Stegun 7.1.26.
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly =
        t * (0.254_829_592 + t * (-0.284_496_736 + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// The standard normal CDF.
pub fn normal_cdf(x: f64) -> f64 {
    f64::midpoint(1.0, erf(x / std::f64::consts::SQRT_2))
}

/// The inverse of [`normal_cdf`] by bisection on `-8..8`.
pub fn normal_quantile(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    let (mut lo, mut hi) = (-8.0_f64, 8.0_f64);
    for _ in 0..60 {
        let mid = f64::midpoint(lo, hi);
        if normal_cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    f64::midpoint(lo, hi)
}

/// An exemplar prepared for histogram-preserving sampling.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianExemplar {
    /// Side in texels.
    pub size: usize,
    /// The original values (row-major).
    pub values: Vec<f32>,
    /// The Gaussianised values (mean 0.5, σ 1/6).
    pub gauss: Vec<f32>,
    /// Inverse table: entry `i` is the exemplar value at Gaussian value `(i + 0.5) / LUT_SIZE`.
    pub lut: Vec<f32>,
}

impl GaussianExemplar {
    /// Gaussianises a `size × size` exemplar.
    pub fn new(values: &[f32], size: usize) -> Result<Self, SkinError> {
        if size == 0 || values.len() != size * size || values.iter().any(|v| !v.is_finite()) {
            return Err(SkinError::Invalid { what: "exemplar must be size x size finite values" });
        }
        let n = values.len();
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by(|&a, &b| values[a as usize].total_cmp(&values[b as usize]));
        let mut gauss = vec![0.0; n];
        for (rank, &i) in order.iter().enumerate() {
            let u = (rank as f64 + 0.5) / n as f64;
            gauss[i as usize] = (0.5 + f64::from(GAUSS_SIGMA) * normal_quantile(u)) as f32;
        }
        let mut lut = vec![0.0; LUT_SIZE];
        for (i, slot) in lut.iter_mut().enumerate() {
            let g = (i as f64 + 0.5) / LUT_SIZE as f64;
            let u = normal_cdf((g - 0.5) / f64::from(GAUSS_SIGMA));
            let rank = ((u * n as f64) as usize).min(n - 1);
            *slot = values[order[rank] as usize];
        }
        Ok(Self { size, values: values.to_vec(), gauss, lut })
    }

    /// Nearest-texel fetch of `data` at exemplar coordinates (texels, wrapping).
    fn fetch(&self, data: &[f32], u: f32, v: f32) -> f32 {
        let s = self.size as f32;
        let x = (u.rem_euclid(s) as usize).min(self.size - 1);
        let y = (v.rem_euclid(s) as usize).min(self.size - 1);
        data[y * self.size + x]
    }

    /// Plain repetition: the exemplar at `(u, v)` in exemplar tiles.
    pub fn sample_repeat(&self, u: f32, v: f32) -> f32 {
        let s = self.size as f32;
        self.fetch(&self.values, u * s, v * s)
    }

    /// The histogram-preserving blend at `(u, v)` in exemplar tiles.
    pub fn sample_tiled(&self, u: f32, v: f32, seed: u32) -> f32 {
        let s = self.size as f32;
        let (weights, vertices) = triangle_grid(u, v);
        let mut g = 0.0;
        let mut w2 = 0.0;
        for (w, vertex) in weights.iter().zip(vertices) {
            let (ox, oy) = vertex_offset(vertex, seed);
            g += w * self.fetch(&self.gauss, (u + ox) * s, (v + oy) * s);
            w2 += w * w;
        }
        let g = (g - 0.5) / w2.sqrt() + 0.5;
        let index = ((g * LUT_SIZE as f32) as isize).clamp(0, LUT_SIZE as isize - 1) as usize;
        self.lut[index]
    }
}

/// The barycentric weights and the three vertex ids of the triangle-grid cell containing `(u, v)`
/// (Heitz & Neyret 2018, the skewed grid at `2√3` times the input frequency).
pub fn triangle_grid(u: f32, v: f32) -> ([f32; 3], [[i32; 2]; 3]) {
    let (x, y) = (u * 2.0 * 3_f32.sqrt(), v * 2.0 * 3_f32.sqrt());
    // gridToSkewedGrid = [[1, 0], [-0.57735027, 1.15470054]] applied as in the paper's GLSL
    // (column-major mat2 times vector): skewed = (x - 0.57735027 y, 1.15470054 y).
    let sx = x - 0.577_350_27 * y;
    let sy = 1.154_700_5 * y;
    let (bx, by) = (sx.floor(), sy.floor());
    let (fx, fy) = (sx - bx, sy - by);
    let fz = 1.0 - fx - fy;
    let (bx, by) = (bx as i32, by as i32);
    if fz > 0.0 {
        ([fz, fy, fx], [[bx, by], [bx, by + 1], [bx + 1, by]])
    } else {
        ([-fz, 1.0 - fy, 1.0 - fx], [[bx + 1, by + 1], [bx + 1, by], [bx, by + 1]])
    }
}

/// The random offset of a triangle-grid vertex, in exemplar tiles `[0, 1)²`.
pub fn vertex_offset(vertex: [i32; 2], seed: u32) -> (f32, f32) {
    let (x, y) = (vertex[0] as u32, vertex[1] as u32);
    (unit(hash3(x, y, seed)), unit(hash3(x, y, seed ^ 0x9e37_79b9)))
}

#[cfg(test)]
mod tests {
    use humanart_core::rng::Rng;

    use super::super::microdetail::{ExemplarPlan, MicroDetailParams};
    use super::super::spectrum::tiling_score;
    use super::*;

    #[test]
    fn normal_functions() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-9);
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-4);
        for p in [0.01, 0.2, 0.5, 0.8, 0.99] {
            assert!((normal_cdf(normal_quantile(p)) - p).abs() < 1e-6);
        }
        assert!((erf(1.0) - 0.842_700_79).abs() < 2e-7);
    }

    #[test]
    fn triangle_weights_sum_to_one() {
        let mut rng = Rng::new(2);
        for _ in 0..1000 {
            let (w, _) = triangle_grid(rng.range_f32(-5.0, 5.0), rng.range_f32(-5.0, 5.0));
            assert!(((w[0] + w[1] + w[2]) - 1.0).abs() < 1e-5);
            assert!(w.iter().all(|&x| x >= -1e-6));
        }
    }

    fn synthesize(ex: &GaussianExemplar, out: usize, tiles: f32, tiled: bool) -> Vec<f32> {
        let mut img = Vec::with_capacity(out * out);
        for y in 0..out {
            for x in 0..out {
                let u = (x as f32 + 0.5) / out as f32 * tiles;
                let v = (y as f32 + 0.5) / out as f32 * tiles;
                img.push(if tiled { ex.sample_tiled(u, v, 77) } else { ex.sample_repeat(u, v) });
            }
        }
        img
    }

    fn mean_std(v: &[f32]) -> (f64, f64) {
        let n = v.len() as f64;
        let m = v.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
        let var = v.iter().map(|&x| (f64::from(x) - m).powi(2)).sum::<f64>() / n;
        (m, var.sqrt())
    }

    #[test]
    fn ac_wp13_2_tiling_detector_below_threshold_at_close_up_scales() {
        // [AC-WP13-2] The generated exemplar (128², pores + lines + relief) repeated plainly scores
        // ~1 on the tiling detector; with histogram-preserving blending the score stays below 0.2
        // at every close-up scale (2 and 4 exemplar periods across a 256² view), and the
        // histogram (mean, σ) is preserved within 10 %.
        let params = MicroDetailParams { size: 128, ..MicroDetailParams::default() };
        let plan = ExemplarPlan::new(&params).unwrap_or_else(|e| panic!("{e}"));
        let ex = GaussianExemplar::new(&plan.heights(), 128).unwrap_or_else(|e| panic!("{e}"));
        let (m0, s0) = mean_std(&ex.values);
        for tiles in [2.0_f32, 4.0] {
            let out = 256;
            let period = (out as f32 / tiles) as usize;
            let plain = synthesize(&ex, out, tiles, false);
            let blended = synthesize(&ex, out, tiles, true);
            let plain_score = tiling_score(&plain, out, period).unwrap_or_else(|e| panic!("{e}"));
            let score = tiling_score(&blended, out, period).unwrap_or_else(|e| panic!("{e}"));
            assert!(plain_score > 0.95, "plain repetition {plain_score}");
            assert!(score < 0.2, "tiling score {score} at {tiles} periods");
            let (m, s) = mean_std(&blended);
            assert!((m - m0).abs() < 0.1 * s0, "mean {m} vs {m0}");
            assert!((s / s0 - 1.0).abs() < 0.1, "σ {s} vs {s0}");
        }
    }

    #[test]
    fn rejects_bad_exemplars() {
        assert!(GaussianExemplar::new(&[0.0; 3], 2).is_err());
        assert!(GaussianExemplar::new(&[f32::NAN; 4], 2).is_err());
        let ex = GaussianExemplar::new(&[0.0, 1.0, 2.0, 3.0], 2).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(ex.lut[0], 0.0);
        assert_eq!(ex.lut[LUT_SIZE - 1], 3.0);
    }
}
