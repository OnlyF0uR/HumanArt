//! Subsurface scattering and translucency of skin `[SHD-02]`..`[SHD-05]`: the diffusion profile,
//! the separable screen-space kernel, the transmittance through thin geometry and the
//! pre-integrated (curvature) table, on the CPU. The GPU passes (`render/geometry.wgsl`,
//! `render/sss.wgsl`, `render/subsurface.wgsl`) read the data made here, and the tests use these
//! functions as the reference of what the shaders compute. Derivations: `docs/shader-notes.md`,
//! section "Skin subsurface scattering" (proposed text in the skin-shading STATUS until merged).
//!
//! **Profile.** The radial diffusion profile is the sum of six Gaussians that d'Eon, Luebke and
//! Enderton fitted to the three-layer skin model (d'Eon & Luebke 2007, "Advanced Techniques for
//! Realistic Real-Time Skin Rendering", GPU Gems 3 ch. 14, table 14-1): per channel
//! `R(r) = Σ w_i G(v_i, r)` with `G(v, r) = exp(-r² / 2v) / (2π v)`, variances in mm². The weights
//! of every channel sum to 1, so the profile only moves light: the albedo stays the albedo. A
//! per-object width `scale` stretches the profile (`v_i → v_i scale²`); 1 is the measured skin.
//!
//! **Separable kernel.** The 1D kernel is the profile's marginal along one axis,
//! `k(x) = Σ w_i exp(-x² / 2v_i) / √(2π v_i)`; applied along x and then along y it keeps the
//! energy and the per-axis variance of the profile (the separable approximation of Jimenez et al.
//! 2015, "Separable Subsurface Scattering"). [`KERNEL_TAPS`] taps at offsets that grow with the
//! square of the tap index cover ±[`KERNEL_RANGE_MM`]; each tap weighs the integral of `k` over its
//! cell (the outer cells reach infinity), so every channel's weights sum to 1.
//!
//! **Transmittance.** Light entering the back of a slab `d` mm thick and leaving at the point
//! opposite is the profile integrated over the back plane:
//! `T(d) = ∫∫ R(√(r² + d²)) dA = Σ w_i exp(-d² / 2v_i)` (Jimenez et al. 2010, "Real-Time
//! Realistic Skin Translucency"), 1 at `d = 0`; red reaches farthest.
//!
//! **Pre-integrated.** For the Interactive tier the diffuse falloff of a light is integrated over
//! a ring of radius `ρ` (Penner 2011, "Pre-Integrated Skin Shading"):
//! `D(θ, ρ) = ∫ max(cos(θ + x), 0) R(2ρ sin(x/2)) dx / ∫ R(2ρ sin(x/2)) dx`, x over (-π, π],
//! tabulated over `cos θ` and `scale / ρ` ([`PREINTEGRATED_SIZE`]²).

use crate::error::RenderError;

/// Variances of the six Gaussians of the skin profile, mm² (d'Eon & Luebke 2007, table 14-1).
pub const PROFILE_VARIANCES_MM2: [f32; 6] = [0.0064, 0.0484, 0.187, 0.567, 1.99, 7.41];

/// Red, green, blue weights of the six Gaussians (d'Eon & Luebke 2007, table 14-1); each channel
/// sums to 1.
pub const PROFILE_WEIGHTS: [[f32; 3]; 6] = [
    [0.233, 0.455, 0.649],
    [0.100, 0.336, 0.344],
    [0.118, 0.198, 0.0],
    [0.113, 0.007, 0.007],
    [0.358, 0.004, 0.0],
    [0.078, 0.0, 0.0],
];

/// Taps of the separable kernel (odd: one centre tap).
pub const KERNEL_TAPS: usize = 25;

/// Half the extent of the kernel in mm at scale 1: three standard deviations of the widest
/// Gaussian.
pub const KERNEL_RANGE_MM: f32 = 8.2;

/// Side of the pre-integrated table (texels per axis).
pub const PREINTEGRATED_SIZE: usize = 32;

/// The largest `scale / ρ` (1/mm at scale 1) the pre-integrated table covers: curvature radii
/// down to 2 mm. Beyond it the table's last row applies.
pub const PREINTEGRATED_MAX_INV_RADIUS: f32 = 0.5;

/// Material class of skin in the G-buffer (`Material::class`).
pub const SKIN_CLASS: u8 = 1;

/// The skin shading parameters of an object of class [`SKIN_CLASS`] `[SHD-01]` `[SHD-02]`
/// `[SHD-03]`. The material's roughness is the primary specular lobe, its `f0` the dielectric
/// reflectance (0.028 for skin).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkinShading {
    /// Perceptual roughness of the secondary specular lobe, 0..1.
    pub secondary_roughness: f32,
    /// Weight of the secondary lobe, 0..1.
    pub lobe_mix: f32,
    /// Width of the diffusion profile relative to the measured skin (1), 0..[`MAX_SCATTER_SCALE`].
    pub scatter_scale: f32,
    /// Strength of the transmission through thin geometry, 0..1.
    pub translucency: f32,
    /// Whether the object reads the renderer's skin surface (`crate::surface`: resolved
    /// textures, micro-detail, wrinkles, coat and sheen); its texture coordinates must carry the UV
    /// tile in the integer part of u.
    pub surface: bool,
}

/// The widest profile scale accepted.
pub const MAX_SCATTER_SCALE: f32 = 4.0;

impl Default for SkinShading {
    /// The skin defaults of the skin appearance model (`humanart_model::skin_appearance`):
    /// secondary roughness 0.75, lobe mix 0.15; the measured profile; full translucency.
    fn default() -> Self {
        Self { secondary_roughness: 0.75, lobe_mix: 0.15, scatter_scale: 1.0, translucency: 1.0, surface: false }
    }
}

impl SkinShading {
    /// Checks the ranges.
    pub fn validate(&self) -> Result<(), RenderError> {
        let in01 = |v: f32| (0.0..=1.0).contains(&v);
        if !in01(self.secondary_roughness)
            || !in01(self.lobe_mix)
            || !in01(self.translucency)
            || !(0.0..=MAX_SCATTER_SCALE).contains(&self.scatter_scale)
        {
            return Err(RenderError::InvalidParameter { what: "skin shading values outside their ranges" });
        }
        Ok(())
    }

    /// The four floats the shaders read (`Object.skin` in `render/scene.wgsl`).
    pub fn to_gpu(&self) -> [f32; 4] {
        [self.secondary_roughness, self.lobe_mix, self.scatter_scale, self.translucency]
    }

    /// The profile scale that gives the mean of the per-channel scattering radii `radius_mm` of
    /// the pigment model (`PigmentResponse::sss_radius_mm`, the dermis diffusion length) relative
    /// to the mean RMS radius of the measured profile, clamped to the accepted range. The
    /// per-channel shape stays the measured one (one width per pixel, see the module docs).
    pub fn scale_for_radius_mm(radius_mm: [f32; 3]) -> f32 {
        let reference = reference_radius_mm();
        let want = (radius_mm[0] + radius_mm[1] + radius_mm[2]) / 3.0;
        let have = (reference[0] + reference[1] + reference[2]) / 3.0;
        if want.is_finite() && want > 0.0 { (want / have).clamp(0.0, MAX_SCATTER_SCALE) } else { 0.0 }
    }
}

/// The RMS radius `√(Σ w_i 2 v_i)` of each channel's profile at scale 1, mm.
pub fn reference_radius_mm() -> [f32; 3] {
    let mut out = [0.0; 3];
    for (v, w) in PROFILE_VARIANCES_MM2.iter().zip(&PROFILE_WEIGHTS) {
        for c in 0..3 {
            out[c] += w[c] * 2.0 * v;
        }
    }
    out.map(f32::sqrt)
}

/// The radial profile `R(r)` per channel at `r_mm` from the entry point (1/mm²) for a profile
/// `scale`.
pub fn profile(r_mm: f32, scale: f32) -> [f32; 3] {
    let mut out = [0.0; 3];
    let s2 = (scale * scale).max(1e-12);
    for (&v, w) in PROFILE_VARIANCES_MM2.iter().zip(&PROFILE_WEIGHTS) {
        let v = v * s2;
        let g = (-r_mm * r_mm / (2.0 * v)).exp() / (std::f32::consts::TAU * v);
        for c in 0..3 {
            out[c] += w[c] * g;
        }
    }
    out
}

/// The 1D kernel `k(x)` per channel (1/mm): the profile's marginal along one axis.
pub fn profile_1d(x_mm: f32, scale: f32) -> [f32; 3] {
    let mut out = [0.0; 3];
    let s2 = (scale * scale).max(1e-12);
    for (&v, w) in PROFILE_VARIANCES_MM2.iter().zip(&PROFILE_WEIGHTS) {
        let v = v * s2;
        let g = (-x_mm * x_mm / (2.0 * v)).exp() / (std::f32::consts::TAU * v).sqrt();
        for c in 0..3 {
            out[c] += w[c] * g;
        }
    }
    out
}

/// The transmittance `T(d)` per channel through `d_mm` of skin for a profile `scale`.
pub fn transmittance(d_mm: f32, scale: f32) -> [f32; 3] {
    let mut out = [0.0; 3];
    let s2 = (scale * scale).max(1e-12);
    for (&v, w) in PROFILE_VARIANCES_MM2.iter().zip(&PROFILE_WEIGHTS) {
        let t = (-d_mm * d_mm / (2.0 * v * s2)).exp();
        for c in 0..3 {
            out[c] += w[c] * t;
        }
    }
    out
}

/// The error function (Abramowitz & Stegun 7.1.26, absolute error below 1.5e-7), in f64.
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly =
        t * (0.254_829_592 + t * (-0.284_496_736 + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let y = 1.0 - poly * (-x * x).exp();
    if x < 0.0 { -y } else { y }
}

/// The integral of the 1D kernel at scale 1 from -∞ to `x_mm`, per channel.
fn kernel_cdf(x_mm: f64) -> [f64; 3] {
    let mut out = [0.0; 3];
    for (&v, w) in PROFILE_VARIANCES_MM2.iter().zip(&PROFILE_WEIGHTS) {
        let c = f64::midpoint(1.0, erf(x_mm / (2.0 * f64::from(v)).sqrt()));
        for k in 0..3 {
            out[k] += f64::from(w[k]) * c;
        }
    }
    out
}

/// The tap offsets of the separable kernel at scale 1, mm, ascending: `±range (i / half)²`.
pub fn kernel_offsets() -> [f32; KERNEL_TAPS] {
    let half = (KERNEL_TAPS / 2) as f32;
    std::array::from_fn(|i| {
        let t = (i as f32 - half) / half;
        KERNEL_RANGE_MM * t * t.abs()
    })
}

/// The separable kernel at scale 1: per tap the offset in mm and the red, green and blue weights.
/// Each channel's weights sum to 1 (the outer cells extend to infinity, then the sums are
/// normalised against rounding).
pub fn separable_kernel() -> [[f32; 4]; KERNEL_TAPS] {
    let offsets = kernel_offsets();
    let mut out = [[0.0_f32; 4]; KERNEL_TAPS];
    let mut sums = [0.0_f64; 3];
    for (i, tap) in out.iter_mut().enumerate() {
        let lo = if i == 0 { f64::NEG_INFINITY } else { 0.5 * f64::from(offsets[i - 1] + offsets[i]) };
        let hi = if i + 1 == KERNEL_TAPS { f64::INFINITY } else { 0.5 * f64::from(offsets[i] + offsets[i + 1]) };
        let a = if lo.is_finite() { kernel_cdf(lo) } else { [0.0; 3] };
        let b = if hi.is_finite() { kernel_cdf(hi) } else { [1.0; 3] };
        tap[0] = offsets[i];
        for c in 0..3 {
            let w = (b[c] - a[c]).max(0.0);
            sums[c] += w;
            tap[c + 1] = w as f32;
        }
    }
    for tap in &mut out {
        for c in 0..3 {
            if sums[c] > 0.0 {
                tap[c + 1] = (f64::from(tap[c + 1]) / sums[c]) as f32;
            }
        }
    }
    out
}

/// The kernel as the bytes of the shader's `array<vec4<f32>, KERNEL_TAPS>` (little-endian).
pub fn kernel_bytes() -> Vec<u8> {
    separable_kernel().iter().flatten().flat_map(|v| v.to_le_bytes()).collect()
}

/// One separable pass of the CPU reference over an RGBA image (`w` x `h`, rows top to bottom):
/// what `render/sss.wgsl` computes for a skin surface facing the camera at constant depth
/// (`px_per_mm` pixels per millimetre, profile `scale`), every pixel skin. Samples between pixel
/// centres are linearly interpolated, outside the image the edge pixel repeats (a clamped
/// sampler).
pub fn blur_pass_reference(
    src: &[[f32; 4]],
    w: usize,
    h: usize,
    horizontal: bool,
    px_per_mm: f32,
    scale: f32,
) -> Vec<[f32; 4]> {
    let kernel = separable_kernel();
    let fetch = |x: i64, y: i64| -> [f32; 4] {
        let xc = x.clamp(0, w as i64 - 1) as usize;
        let yc = y.clamp(0, h as i64 - 1) as usize;
        src.get(yc * w + xc).copied().unwrap_or([0.0; 4])
    };
    let mut out = vec![[0.0_f32; 4]; src.len()];
    for y in 0..h {
        for x in 0..w {
            let mut sum = [0.0_f32; 3];
            for tap in &kernel {
                let o = tap[0] * scale * px_per_mm;
                let (fx, fy) = if horizontal { (x as f32 + o, y as f32) } else { (x as f32, y as f32 + o) };
                let (x0, y0) = (fx.floor(), fy.floor());
                let (tx, ty) = (fx - x0, fy - y0);
                let (x0, y0) = (x0 as i64, y0 as i64);
                let a = fetch(x0, y0);
                let b = if horizontal { fetch(x0 + 1, y0) } else { fetch(x0, y0 + 1) };
                let t = if horizontal { tx } else { ty };
                for c in 0..3 {
                    sum[c] += tap[c + 1] * (a[c] + (b[c] - a[c]) * t);
                }
            }
            if let Some(o) = out.get_mut(y * w + x) {
                *o = [sum[0], sum[1], sum[2], src.get(y * w + x).map_or(0.0, |p| p[3])];
            }
        }
    }
    out
}

/// The pre-integrated diffuse falloff `D(θ, ρ)` per channel for `cos θ` and `inv_radius`
/// (`scale / ρ` in 1/mm), integrated with `steps` samples.
pub fn preintegrated(cos_theta: f32, inv_radius: f32, steps: usize) -> [f32; 3] {
    let theta = cos_theta.clamp(-1.0, 1.0).acos();
    if inv_radius <= 1e-6 {
        let d = cos_theta.max(0.0);
        return [d; 3];
    }
    let rho = 1.0 / inv_radius;
    let n = steps.max(8);
    let mut num = [0.0_f64; 3];
    let mut den = [0.0_f64; 3];
    let dx = std::f32::consts::TAU / n as f32;
    for i in 0..n {
        let x = -std::f32::consts::PI + (i as f32 + 0.5) * dx;
        let r = profile(2.0 * rho * (0.5 * x).sin().abs(), 1.0);
        let lit = (theta + x).cos().max(0.0);
        for c in 0..3 {
            num[c] += f64::from(lit * r[c]);
            den[c] += f64::from(r[c]);
        }
    }
    std::array::from_fn(|c| if den[c] > 0.0 { (num[c] / den[c]) as f32 } else { cos_theta.max(0.0) })
}

/// The pre-integrated table, [`PREINTEGRATED_SIZE`]² RGBA texels, row-major: column `u` is
/// `cos θ = 2u / (size - 1) - 1`, row `v` is `scale / ρ = v / (size - 1) ·`
/// [`PREINTEGRATED_MAX_INV_RADIUS`]; alpha 1.
pub fn preintegrated_table() -> Vec<[f32; 4]> {
    let n = PREINTEGRATED_SIZE;
    let last = (n - 1) as f32;
    let mut out = Vec::with_capacity(n * n);
    for v in 0..n {
        let inv_radius = v as f32 / last * PREINTEGRATED_MAX_INV_RADIUS;
        for u in 0..n {
            let d = preintegrated(2.0 * u as f32 / last - 1.0, inv_radius, 256);
            out.push([d[0], d[1], d[2], 1.0]);
        }
    }
    out
}

/// The pre-integrated table as `rgba16float` texel bytes (little-endian).
pub fn preintegrated_bytes() -> Vec<u8> {
    preintegrated_table().iter().flatten().flat_map(|&v| humanart_math::f32_to_f16_bits(v).to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Numerical integral of `f` over (-limit, limit) with `n` midpoint steps.
    fn integrate(f: impl Fn(f32) -> [f32; 3], limit: f32, n: usize) -> [f64; 3] {
        let dx = 2.0 * limit / n as f32;
        let mut sum = [0.0_f64; 3];
        for i in 0..n {
            let v = f(-limit + (i as f32 + 0.5) * dx);
            for c in 0..3 {
                sum[c] += f64::from(v[c] * dx);
            }
        }
        sum
    }

    // [AC-WP14-1]: the profile's weights sum to 1 per channel, and so does its integral over the
    // plane (the profile only moves light).
    #[test]
    fn ac_wp14_1_profile_weights_sum_to_one_per_channel() {
        for c in 0..3 {
            let sum: f32 = PROFILE_WEIGHTS.iter().map(|w| w[c]).sum();
            assert!((sum - 1.0).abs() < 1e-3, "channel {c}: {sum}");
        }
        for scale in [0.5, 1.0, 2.0] {
            // ∫ R(r) 2πr dr over r in [0, 40 scale].
            let limit = 40.0 * scale;
            let total =
                integrate(|r| profile(r.abs(), scale).map(|v| v * std::f32::consts::PI * r.abs()), limit, 400_000);
            for (c, t) in total.iter().enumerate() {
                assert!((t - 1.0).abs() < 2e-3, "scale {scale}, channel {c}: {t}");
            }
        }
    }

    // The 1D kernel is the marginal of the radial profile: ∫ R(√(x² + y²)) dy = k(x).
    #[test]
    fn the_1d_kernel_is_the_profile_marginal() {
        for x in [0.0_f32, 0.1, 0.5, 1.5, 4.0] {
            let marginal = integrate(|y| profile((x * x + y * y).sqrt(), 1.0), 30.0, 200_000);
            let k = profile_1d(x, 1.0);
            for c in 0..3 {
                let tol = 2e-3 * f64::from(k[c]).max(1e-3);
                assert!((marginal[c] - f64::from(k[c])).abs() < tol, "x {x} channel {c}: {} vs {}", marginal[c], k[c]);
            }
        }
    }

    // [AC-WP14-1]: the separable kernel's weights sum to 1 per channel, are symmetric and match the
    // 1D kernel integrated over each tap's cell; the offsets grow from the centre.
    #[test]
    fn ac_wp14_1_kernel_weights_sum_to_one_and_follow_the_profile() {
        let kernel = separable_kernel();
        for c in 1..4 {
            let sum: f32 = kernel.iter().map(|t| t[c]).sum();
            assert!((sum - 1.0).abs() < 1e-5, "channel {c}: {sum}");
        }
        for i in 0..KERNEL_TAPS {
            let j = KERNEL_TAPS - 1 - i;
            assert!((kernel[i][0] + kernel[j][0]).abs() < 1e-6, "offsets symmetric");
            assert!(kernel[i][1..].iter().zip(&kernel[j][1..]).all(|(a, b)| (a - b).abs() < 1e-6), "weights symmetric");
            if i > 0 {
                assert!(kernel[i][0] > kernel[i - 1][0], "offsets ascend");
            }
        }
        assert_eq!(kernel[KERNEL_TAPS / 2][0], 0.0);
        // An inner tap against a midpoint integration of k over its cell.
        let offsets = kernel_offsets();
        let i = KERNEL_TAPS / 2 + 3;
        let (lo, hi) = (f32::midpoint(offsets[i - 1], offsets[i]), f32::midpoint(offsets[i], offsets[i + 1]));
        let n = 20_000;
        let dx = (hi - lo) / n as f32;
        let mut cell = [0.0_f32; 3];
        for s in 0..n {
            let k = profile_1d(lo + (s as f32 + 0.5) * dx, 1.0);
            for c in 0..3 {
                cell[c] += k[c] * dx;
            }
        }
        for c in 0..3 {
            assert!((cell[c] - kernel[i][c + 1]).abs() < 1e-4, "channel {c}: {} vs {}", cell[c], kernel[i][c + 1]);
        }
        // Red scatters farthest: its centre weight is the smallest.
        let centre = kernel[KERNEL_TAPS / 2];
        assert!(centre[1] < centre[2] && centre[2] < centre[3], "{centre:?}");
        assert_eq!(kernel_bytes().len(), KERNEL_TAPS * 16);
    }

    // [SHD-03]: transmittance is 1 through nothing, falls with thickness, red reaches farthest, and
    // equals the profile integrated over the back plane.
    #[test]
    fn shd_03_transmittance_follows_the_profile() {
        assert!(transmittance(0.0, 1.0).iter().all(|&t| (t - 1.0).abs() < 1e-3));
        let mut last = transmittance(0.0, 1.0);
        for d in 1..20 {
            let t = transmittance(d as f32 * 0.5, 1.0);
            for c in 0..3 {
                assert!(t[c] <= last[c] + 1e-7);
            }
            last = t;
        }
        let t = transmittance(2.0, 1.0);
        assert!(t[0] > 5.0 * t[1] && t[1] >= t[2], "red glows: {t:?}");
        assert_eq!(transmittance(2.0, 2.0), transmittance(1.0, 1.0), "the scale stretches the thickness");
        // ∫∫ R(√(r² + d²)) dA over the back plane = T(d).
        let d = 1.5;
        let plane = integrate(
            |r| profile((r * r + d * d).sqrt(), 1.0).map(|v| v * std::f32::consts::PI * r.abs()),
            40.0,
            400_000,
        );
        for (c, (p, t)) in plane.iter().zip(t_at(d)).enumerate() {
            assert!((p - f64::from(t)).abs() < 2e-3, "channel {c}");
        }
    }

    fn t_at(d: f32) -> [f32; 3] {
        transmittance(d, 1.0)
    }

    // The CPU blur keeps a uniform image and the energy of an impulse, and spreads red widest.
    #[test]
    fn the_reference_blur_keeps_energy_and_spreads_red_widest() {
        let (w, h) = (129, 3);
        let uniform = vec![[0.7, 0.4, 0.2, 1.0]; w * h];
        let out = blur_pass_reference(&uniform, w, h, true, 4.0, 1.0);
        assert!(out.iter().all(|p| (p[0] - 0.7).abs() < 1e-5 && (p[1] - 0.4).abs() < 1e-5));
        let mut impulse = vec![[0.0; 4]; w * h];
        impulse[w + 64] = [1.0, 1.0, 1.0, 1.0];
        let out = blur_pass_reference(&impulse, w, h, true, 4.0, 1.0);
        let row = &out[w..2 * w];
        for c in 0..3 {
            let sum: f32 = row.iter().map(|p| p[c]).sum();
            assert!((sum - 1.0).abs() < 1e-4, "channel {c} keeps its energy: {sum}");
        }
        let spread = |c: usize| row.iter().enumerate().map(|(x, p)| p[c] * (x as f32 - 64.0).powi(2)).sum::<f32>();
        assert!(spread(0) > spread(1) && spread(1) > spread(2));
    }

    // [SHD-04]: the pre-integrated table is the clamped cosine on a flat surface, wraps light past
    // the terminator on a curved one (red the most), and never exceeds the lit side's cosine by
    // much.
    #[test]
    fn shd_04_preintegrated_table_wraps_light_past_the_terminator() {
        for c in [-0.5_f32, 0.0, 0.3, 1.0] {
            let d = preintegrated(c, 0.0, 64);
            assert!(d.iter().all(|&v| (v - c.max(0.0)).abs() < 1e-6));
        }
        let terminator = preintegrated(-0.05, PREINTEGRATED_MAX_INV_RADIUS, 512);
        assert!(
            terminator[0] > terminator[1] && terminator[1] >= terminator[2] && terminator[0] > 0.02,
            "{terminator:?}"
        );
        let lit = preintegrated(1.0, PREINTEGRATED_MAX_INV_RADIUS, 512);
        assert!(lit.iter().all(|&v| v <= 1.0 && v > 0.8), "{lit:?}");
        let table = preintegrated_table();
        assert_eq!(table.len(), PREINTEGRATED_SIZE * PREINTEGRATED_SIZE);
        assert!(table.iter().all(|t| t.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v))));
        assert_eq!(preintegrated_bytes().len(), PREINTEGRATED_SIZE * PREINTEGRATED_SIZE * 8);
    }

    #[test]
    fn skin_parameters_are_checked_and_scaled_from_pigment_radii() {
        assert!(SkinShading::default().validate().is_ok());
        assert!(SkinShading { lobe_mix: 1.5, ..SkinShading::default() }.validate().is_err());
        assert!(SkinShading { scatter_scale: f32::NAN, ..SkinShading::default() }.validate().is_err());
        assert!(SkinShading { scatter_scale: 5.0, ..SkinShading::default() }.validate().is_err());
        let reference = reference_radius_mm();
        assert!(reference[0] > reference[1] && reference[1] > reference[2]);
        assert!((SkinShading::scale_for_radius_mm(reference) - 1.0).abs() < 1e-5);
        assert!((SkinShading::scale_for_radius_mm(reference.map(|r| r * 2.0)) - 2.0).abs() < 1e-5);
        assert_eq!(SkinShading::scale_for_radius_mm([f32::NAN; 3]), 0.0);
        assert_eq!(SkinShading::scale_for_radius_mm([100.0; 3]), MAX_SCATTER_SCALE);
        assert!((erf(0.5) - 0.520_499_877_8).abs() < 2e-7 && erf(-3.0) < -0.9999);
    }
}
