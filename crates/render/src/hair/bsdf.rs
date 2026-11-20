//! The hair fibre scattering model `[HR-02]`: the CPU reference of `shaders/hair/bsdf.wgsl` and the
//! tables the shader reads.
//!
//! **Single scattering.** The model of Chiang, Bitterli, Tappan and Burley, "A Practical and
//! Controllable Hair and Fur Model for Production Path Tracing" (CGF 35(2), 2016), as written in
//! pbrt-v3 (`src/materials/hair.cpp`, BSD-2) and the pbrt-v4 book (Pharr, Jakob, Humphreys 2023,
//! sec. 9.9): a fibre of refractive index η = 1.55 scatters light into the lobes R (p = 0,
//! surface reflection), TT (p = 1, through the fibre), TRT (p = 2, one internal reflection) and a
//! residual p ≥ 3. Each lobe is `M_p(θi, θo) · A_p(h) · N_p(φ, h)`:
//! - `M_p`, the longitudinal lobe of d'Eon, Francois, Hill, Letteri, "An Energy-Conserving Hair
//!   Reflectance Model" (EGSR 2011), with variances `v0 = (0.726 βm + 0.812 βm² + 3.7 βm²⁰)²`,
//!   `v1 = v0 / 4`, `v2 = v3 = 4 v0`, evaluated at the outgoing angle tilted by the cuticle scale
//!   angle α: R by −2α, TT by +α, TRT by +4α. The opposite shifts of R and TRT are the two
//!   highlights (the dual-highlight shift of Marschner, Jensen, Cammarano, Worley, Hanrahan,
//!   "Light Scattering from Human Hair Fibers", SIGGRAPH 2003).
//! - `A_p`, the attenuation: Fresnel `f` at the fibre surface and the absorption
//!   `T = exp(−σa · 2 cos γt / cos θt)` (σa per unit fibre radius, the convention of
//!   `humanart_model::hair::color`): `A0 = f`, `A1 = (1 − f)² T`, `A2 = A1 T f`,
//!   `A3 = A2 f T / (1 − f T)`.
//! - `N_p`, the azimuthal lobe: a trimmed logistic of scale `s = √(π/8) (0.265 βn + 1.194 βn² +
//!   5.372 βn²²)` around `Φ(p, h) = 2p γt − 2γo + pπ`; the residual is uniform, `1 / 2π`.
//!
//! The sum `f = Σ M_p A_p N_p` is radiance per unit irradiance arriving perpendicular to the
//! light (pbrt's `HairBSDF::f` with its `1 / |cos θ|` removed by the integrator): a fibre lit by a
//! directional light of illuminance `E` has radiance `f · E`, and `∫ f dω = Σ A_p` (1 without
//! absorption, the white-furnace test).
//!
//! **Far field.** A rendered strand is narrower than a pixel, so the shader needs the fibre's
//! scattering averaged over its width: `N̄_p(φ, θo, σa) = ½ ∫ A_p(h) N_p(φ, h) dh`. It is
//! tabulated ([`AzimuthalTable`]) over φ ∈ [0, π] (it is even in φ), θo ∈ [0, π/2] and σa ∈
//! [0, [`SIGMA_MAX`]] on a square-root axis, per scalar absorption, so each colour channel looks
//! up its own σa. The integral over h uses the substitution `h = sin γ` (midpoint rule in γ).
//!
//! **Multiple scattering.** The dual-scattering approximation of Zinke, Yuksel, Weber, Keyser,
//! "Dual Scattering Approximation for Fast Multiple Scattering in Hair" (SIGGRAPH 2008): the
//! average forward and backward attenuations `āf(θ)`, `āb(θ)` of one fibre (the far-field `f`
//! integrated over the forward / backward half of the azimuths), their lobe-weighted shifts and
//! variances (`αf`, `βf²`, `αb`, `βb²`), and from them the local back-scattering of the hair
//! around a point (Zinke's eqs. for `Ab = A1 + A3`, `Δb`, `σb`), tabulated per (θ, σa) in
//! [`DualTable`]. Global forward scattering through `τ` expected fibre crossings towards the
//! light (`τ` from the deep opacity map, [`crate::hair`]) is `exp(−τ (1 − āf))` in total, of which
//! `exp(−τ)` arrives unscattered; the scattered part is shaded with the longitudinal variances
//! widened by `τ βf²` ([`dual_scattering`], the shader's `hair_dual`). Deviations from Zinke: the
//! number of crossings is Poisson with mean `τ` instead of a hard count (no `n > 0` switch), the
//! back-scattered lobe is spread uniformly over all azimuths and normalised to carry exactly
//! `Ab`, and the azimuthal spread of the forward-scattered light is not widened.

use std::f64::consts::PI;

/// The refractive index of a hair fibre (keratin), Marschner et al. 2003, pbrt-v3's default.
pub const ETA: f32 = 1.55;
/// The largest absorption the tables cover, per fibre radius (black hair is about 11 in blue).
pub const SIGMA_MAX: f32 = 16.0;
/// Azimuth samples of [`AzimuthalTable`] over [0, π].
pub const PHI_SIZE: usize = 64;
/// θ samples of the tables over [0, π/2].
pub const THETA_SIZE: usize = 16;
/// σa samples of the tables over [0, [`SIGMA_MAX`]] (square-root axis).
pub const SIGMA_SIZE: usize = 32;
/// Samples of γ (`h = sin γ`) per far-field integral.
const H_SAMPLES: usize = 64;
/// Zinke's density factor of forward scattering.
pub const DENSITY_FORWARD: f32 = 0.7;
/// Zinke's density factor of backward scattering.
pub const DENSITY_BACKWARD: f32 = 0.7;

/// The fibre's roughness and cuticle settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FibreParams {
    /// Longitudinal roughness βm, `0.05..=1` (pbrt default 0.3).
    pub beta_m: f32,
    /// Azimuthal roughness βn, `0.05..=1` (pbrt default 0.3).
    pub beta_n: f32,
    /// Cuticle scale angle α in radians, `0..=0.2` (pbrt default 2 degrees).
    pub alpha: f32,
}

impl Default for FibreParams {
    fn default() -> Self {
        Self { beta_m: 0.3, beta_n: 0.3, alpha: 2.0_f32.to_radians() }
    }
}

/// A fibre setting outside its range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BadFibre {
    /// Which setting.
    pub what: &'static str,
}

impl std::fmt::Display for BadFibre {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "fibre setting out of range: {}", self.what)
    }
}

impl std::error::Error for BadFibre {}

impl FibreParams {
    /// Checks the ranges (finite, βm and βn in `0.05..=1`, α in `0..=0.2`).
    pub fn check(&self) -> Result<(), BadFibre> {
        let ok = |v: f32, lo: f32, hi: f32| v.is_finite() && (lo..=hi).contains(&v);
        if !ok(self.beta_m, 0.05, 1.0) {
            return Err(BadFibre { what: "longitudinal roughness (0.05..=1)" });
        }
        if !ok(self.beta_n, 0.05, 1.0) {
            return Err(BadFibre { what: "azimuthal roughness (0.05..=1)" });
        }
        if !ok(self.alpha, 0.0, 0.2) {
            return Err(BadFibre { what: "cuticle angle (0..=0.2 rad)" });
        }
        Ok(())
    }

    /// The longitudinal variances `v0..v3`.
    #[must_use]
    pub fn variances(&self) -> [f32; 4] {
        let b = f64::from(self.beta_m);
        let v0 = (0.726 * b + 0.812 * b * b + 3.7 * b.powi(20)).powi(2);
        [v0 as f32, (0.25 * v0) as f32, (4.0 * v0) as f32, (4.0 * v0) as f32]
    }

    /// The logistic scale `s` of the azimuthal lobes.
    #[must_use]
    pub fn logistic_scale(&self) -> f32 {
        let b = f64::from(self.beta_n);
        ((PI / 8.0).sqrt() * (0.265 * b + 1.194 * b * b + 5.372 * b.powi(22))) as f32
    }

    /// The longitudinal shift of lobe `p` (radians added to θo): −2α, α, 4α, 0.
    #[must_use]
    pub fn shift(&self, p: usize) -> f32 {
        match p {
            0 => -2.0 * self.alpha,
            1 => self.alpha,
            2 => 4.0 * self.alpha,
            _ => 0.0,
        }
    }
}

/// `ln I0(x)`, the log of the modified Bessel function of the first kind (pbrt-v3 `LogI0`).
fn log_i0(x: f64) -> f64 {
    if x > 12.0 { x + 0.5 * (-(2.0 * PI).ln() + (1.0 / x).ln() + 1.0 / (8.0 * x)) } else { i0(x).ln() }
}

/// `I0(x)`, ten terms of its series (pbrt-v3 `I0`).
fn i0(x: f64) -> f64 {
    let mut val = 0.0;
    let mut x2i = 1.0;
    let mut ifact = 1.0_f64;
    let mut i4 = 1.0;
    for i in 0..10 {
        if i > 1 {
            ifact *= f64::from(i);
        }
        val += x2i / (i4 * ifact * ifact);
        x2i *= x * x;
        i4 *= 4.0;
    }
    val
}

/// The longitudinal lobe `M_p` (d'Eon et al. 2011) for variance `v`.
#[must_use]
pub fn longitudinal(cos_i: f64, cos_o: f64, sin_i: f64, sin_o: f64, v: f64) -> f64 {
    let a = cos_i * cos_o / v;
    let b = sin_i * sin_o / v;
    if v <= 0.1 {
        (log_i0(a) - b - 1.0 / v + std::f64::consts::LN_2 + (1.0 / (2.0 * v)).ln()).exp()
    } else {
        ((-b).exp() * i0(a)) / ((1.0 / v).sinh() * 2.0 * v)
    }
}

/// Fresnel reflectance of a dielectric from outside (index 1) into index `eta`.
#[must_use]
pub fn fresnel(cos_i: f64, eta: f64) -> f64 {
    let cos_i = cos_i.clamp(-1.0, 1.0).abs();
    let sin_t = (1.0 - cos_i * cos_i).max(0.0).sqrt() / eta;
    if sin_t >= 1.0 {
        return 1.0;
    }
    let cos_t = (1.0 - sin_t * sin_t).max(0.0).sqrt();
    let parallel = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    let perpendicular = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    f64::midpoint(parallel * parallel, perpendicular * perpendicular)
}

/// The attenuations `A0..A3` at offset `h` for outgoing `cos θo`, one channel of absorption
/// `sigma_a`.
#[must_use]
pub fn attenuation(cos_o: f64, h: f64, sigma_a: f64) -> [f64; 4] {
    let eta = f64::from(ETA);
    let sin_o = (1.0 - cos_o * cos_o).max(0.0).sqrt();
    let sin_t = sin_o / eta;
    let cos_t = (1.0 - sin_t * sin_t).max(0.0).sqrt();
    let etap = (eta * eta - sin_o * sin_o).max(0.0).sqrt() / cos_o.max(1e-9);
    let sin_gt = h / etap;
    let cos_gt = (1.0 - sin_gt * sin_gt).max(0.0).sqrt();
    let t = (-sigma_a * (2.0 * cos_gt / cos_t.max(1e-9))).exp();
    let cos_go = (1.0 - h * h).max(0.0).sqrt();
    let f = fresnel(cos_o * cos_go, eta);
    let a0 = f;
    let a1 = (1.0 - f) * (1.0 - f) * t;
    let a2 = a1 * t * f;
    let a3 = a2 * f * t / (1.0 - t * f).max(1e-9);
    [a0, a1, a2, a3]
}

/// The trimmed logistic on [−π, π] with scale `s`.
fn trimmed_logistic(x: f64, s: f64) -> f64 {
    let logistic = |x: f64| {
        let e = (-x.abs() / s).exp();
        e / (s * (1.0 + e) * (1.0 + e))
    };
    let cdf = |x: f64| 1.0 / (1.0 + (-x / s).exp());
    logistic(x) / (cdf(PI) - cdf(-PI))
}

/// `x` wrapped to [−π, π].
fn wrap(mut x: f64) -> f64 {
    while x > PI {
        x -= 2.0 * PI;
    }
    while x < -PI {
        x += 2.0 * PI;
    }
    x
}

/// The azimuthal lobe `N_p` at relative azimuth `phi` for offset `h` (lobes 0..=2).
#[must_use]
pub fn azimuthal(phi: f64, p: usize, s: f64, cos_o: f64, h: f64) -> f64 {
    let eta = f64::from(ETA);
    let sin_o = (1.0 - cos_o * cos_o).max(0.0).sqrt();
    let etap = (eta * eta - sin_o * sin_o).max(0.0).sqrt() / cos_o.max(1e-9);
    let gamma_o = h.clamp(-1.0, 1.0).asin();
    let gamma_t = (h / etap).clamp(-1.0, 1.0).asin();
    let pf = p as f64;
    let big_phi = 2.0 * pf * gamma_t - 2.0 * gamma_o + pf * PI;
    trimmed_logistic(wrap(phi - big_phi), s)
}

/// A direction in the fibre frame: x along the fibre, (y, z) across it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FibreDir {
    /// sin θ (the component along the fibre).
    pub sin_theta: f64,
    /// The azimuth about the fibre.
    pub phi: f64,
}

impl FibreDir {
    /// The fibre-frame angles of the unit vector `(x, y, z)`.
    #[must_use]
    pub fn from_vector(v: [f64; 3]) -> Self {
        Self { sin_theta: v[0].clamp(-1.0, 1.0), phi: v[2].atan2(v[1]) }
    }

    fn cos_theta(self) -> f64 {
        (1.0 - self.sin_theta * self.sin_theta).max(0.0).sqrt()
    }
}

/// `sin` and `cos` of θ + `shift`, the cosine kept non-negative.
fn tilted(d: FibreDir, shift: f64) -> (f64, f64) {
    let (s, c) = (d.sin_theta, d.cos_theta());
    let (ss, cs) = shift.sin_cos();
    (s * cs + c * ss, (c * cs - s * ss).abs())
}

/// The single-scattering fibre function at offset `h` for one channel of absorption `sigma_a`
/// (radiance per unit perpendicular irradiance; `wo` towards the viewer, `wi` towards the light).
#[must_use]
pub fn eval_near(params: &FibreParams, sigma_a: f64, h: f64, wo: FibreDir, wi: FibreDir) -> f64 {
    let v = params.variances().map(f64::from);
    let s = f64::from(params.logistic_scale());
    let cos_o = wo.cos_theta();
    let a = attenuation(cos_o, h, sigma_a);
    let phi = wi.phi - wo.phi;
    let (sin_i, cos_i) = (wi.sin_theta, wi.cos_theta());
    let mut sum = 0.0;
    for p in 0..3 {
        let (sin_o, cos_ot) = tilted(wo, f64::from(params.shift(p)));
        sum += longitudinal(cos_i, cos_ot, sin_i, sin_o, v[p]) * a[p] * azimuthal(phi, p, s, cos_o, h);
    }
    sum + longitudinal(cos_i, cos_o, sin_i, wo.sin_theta, v[3]) * a[3] / (2.0 * PI)
}

/// The far-field azimuthal terms `[N̄0, N̄1, N̄2, Ā3 / 2π]` at relative azimuth `phi`, outgoing
/// `cos θo`, absorption `sigma_a` (integrated over h, the reference of [`AzimuthalTable`]).
#[must_use]
pub fn far_azimuthal(params: &FibreParams, phi: f64, cos_o: f64, sigma_a: f64) -> [f64; 4] {
    let s = f64::from(params.logistic_scale());
    let mut out = [0.0; 4];
    for k in 0..H_SAMPLES {
        // h = sin γ, dh = cos γ dγ, midpoint rule over γ in (−π/2, π/2).
        let gamma = -0.5 * PI + PI * (k as f64 + 0.5) / H_SAMPLES as f64;
        let h = gamma.sin();
        let w = gamma.cos() * PI / H_SAMPLES as f64 * 0.5;
        let a = attenuation(cos_o, h, sigma_a);
        for p in 0..3 {
            out[p] += w * a[p] * azimuthal(phi, p, s, cos_o, h);
        }
        out[3] += w * a[3] / (2.0 * PI);
    }
    out
}

/// The far-field fibre function (integrated over h), one channel.
#[must_use]
pub fn eval_far(params: &FibreParams, sigma_a: f64, wo: FibreDir, wi: FibreDir) -> f64 {
    let n = far_azimuthal(params, wi.phi - wo.phi, wo.cos_theta(), sigma_a);
    combine(params, &n, wo, wi, 0.0)
}

/// `Σ M_p N̄_p` for far-field azimuthal terms `n`, with `extra_variance` added to every lobe.
fn combine(params: &FibreParams, n: &[f64; 4], wo: FibreDir, wi: FibreDir, extra_variance: f64) -> f64 {
    let v = params.variances().map(f64::from);
    let (sin_i, cos_i) = (wi.sin_theta, wi.cos_theta());
    let mut sum = 0.0;
    for p in 0..3 {
        let (sin_o, cos_o) = tilted(wo, f64::from(params.shift(p)));
        sum += longitudinal(cos_i, cos_o, sin_i, sin_o, v[p] + extra_variance) * n[p];
    }
    sum + longitudinal(cos_i, wo.cos_theta(), sin_i, wo.sin_theta, v[3] + extra_variance) * n[3]
}

/// The σa of table row `j` (square-root axis).
#[must_use]
pub fn sigma_of_row(j: usize) -> f32 {
    let u = j as f32 / (SIGMA_SIZE - 1) as f32;
    SIGMA_MAX * u * u
}

/// The θ of table column `i` over [0, π/2] (cell centres).
#[must_use]
pub fn theta_of_column(i: usize) -> f32 {
    (i as f32 + 0.5) / THETA_SIZE as f32 * std::f32::consts::FRAC_PI_2
}

/// The φ of table column `i` over [0, π] (cell centres).
#[must_use]
pub fn phi_of_column(i: usize) -> f32 {
    (i as f32 + 0.5) / PHI_SIZE as f32 * std::f32::consts::PI
}

/// The far-field azimuthal terms tabulated for one [`FibreParams`]: `[N̄0, N̄1, N̄2, Ā3 / 2π]`
/// per (φ, θo, σa), φ fastest. The shader reads it as a 3D texture (x φ, y θo, z σa) with
/// linear filtering.
#[derive(Clone, Debug, PartialEq)]
pub struct AzimuthalTable {
    /// The settings it was built for.
    pub params: FibreParams,
    /// `PHI_SIZE * THETA_SIZE * SIGMA_SIZE` entries.
    pub values: Vec<[f32; 4]>,
}

impl AzimuthalTable {
    /// Integrates the table (about 6 million lobe evaluations; built when the roughness changes,
    /// never per frame).
    pub fn build(params: &FibreParams) -> Result<Self, BadFibre> {
        params.check()?;
        let mut values = Vec::with_capacity(PHI_SIZE * THETA_SIZE * SIGMA_SIZE);
        for k in 0..SIGMA_SIZE {
            let sigma = f64::from(sigma_of_row(k));
            for j in 0..THETA_SIZE {
                let cos_o = f64::from(theta_of_column(j)).cos();
                for i in 0..PHI_SIZE {
                    let n = far_azimuthal(params, f64::from(phi_of_column(i)), cos_o, sigma);
                    values.push(n.map(|x| x as f32));
                }
            }
        }
        Ok(Self { params: *params, values })
    }

    /// Trilinear lookup at azimuth `phi` (any sign), `cos θo` and absorption `sigma_a`, with the
    /// texture's clamp-to-edge addressing: the CPU mirror of the shader's sampling.
    #[must_use]
    pub fn lookup(&self, phi: f32, cos_o: f32, sigma_a: f32) -> [f32; 4] {
        let phi = wrap(f64::from(phi)).abs() as f32;
        let theta = cos_o.clamp(0.0, 1.0).acos();
        let x = phi / std::f32::consts::PI * PHI_SIZE as f32 - 0.5;
        let y = theta / std::f32::consts::FRAC_PI_2 * THETA_SIZE as f32 - 0.5;
        let z = (sigma_a.clamp(0.0, SIGMA_MAX) / SIGMA_MAX).sqrt() * (SIGMA_SIZE - 1) as f32;
        let axis = |c: f32, n: usize| {
            let c = c.clamp(0.0, (n - 1) as f32);
            let i0 = (c.floor() as usize).min(n - 1);
            let i1 = (i0 + 1).min(n - 1);
            (i0, i1, c - i0 as f32)
        };
        let (x0, x1, fx) = axis(x, PHI_SIZE);
        let (y0, y1, fy) = axis(y, THETA_SIZE);
        let (z0, z1, fz) = axis(z, SIGMA_SIZE);
        let at = |i: usize, j: usize, k: usize| self.values[(k * THETA_SIZE + j) * PHI_SIZE + i];
        let mut out = [0.0; 4];
        for (k, wz) in [(z0, 1.0 - fz), (z1, fz)] {
            for (j, wy) in [(y0, 1.0 - fy), (y1, fy)] {
                for (i, wx) in [(x0, 1.0 - fx), (x1, fx)] {
                    let v = at(i, j, k);
                    for c in 0..4 {
                        out[c] += wx * wy * wz * v[c];
                    }
                }
            }
        }
        out
    }

    /// The far-field fibre function from the table, one channel (CPU mirror of the shader).
    #[must_use]
    pub fn eval(&self, sigma_a: f32, wo: FibreDir, wi: FibreDir, extra_variance: f32) -> f32 {
        let n = self.lookup((wi.phi - wo.phi) as f32, wo.cos_theta() as f32, sigma_a).map(f64::from);
        combine(&self.params, &n, wo, wi, f64::from(extra_variance)) as f32
    }
}

/// Dual-scattering quantities of one (θ, σa) cell.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct DualCell {
    /// Average forward attenuation `āf`.
    pub a_f: f32,
    /// Average backward attenuation `āb`.
    pub a_b: f32,
    /// Lobe-weighted forward shift `αf` (radians).
    pub alpha_f: f32,
    /// Lobe-weighted forward variance `βf²`.
    pub beta_f2: f32,
    /// Lobe-weighted backward shift `αb`.
    pub alpha_b: f32,
    /// Lobe-weighted backward variance `βb²`.
    pub beta_b2: f32,
    /// Local back-scattering attenuation `Ab = A1 + A3`.
    pub big_a_b: f32,
    /// Back-scattering shift `Δb`.
    pub delta_b: f32,
    /// Back-scattering variance `σb²`.
    pub sigma_b2: f32,
}

/// The dual-scattering table: one [`DualCell`] per (θ, σa), θ fastest. The shader reads two
/// layers of a 2D array texture: `(āf, Ab, Δb, σb²)` and `(βf², αf, āb, 0)`.
#[derive(Clone, Debug, PartialEq)]
pub struct DualTable {
    /// `THETA_SIZE * SIGMA_SIZE` cells.
    pub cells: Vec<DualCell>,
}

/// Outgoing directions per dimension when integrating the dual-scattering averages.
const DUAL_THETA: usize = 48;
/// Outgoing azimuths over [0, π] (the far-field function is even in φ).
const DUAL_PHI: usize = 64;

impl DualTable {
    /// Integrates the averages from `table` (the far-field fibre function) and derives Zinke's
    /// back-scattering terms.
    #[must_use]
    pub fn build(table: &AzimuthalTable) -> Self {
        let params = table.params;
        let v = params.variances();
        let mut cells = Vec::with_capacity(THETA_SIZE * SIGMA_SIZE);
        for k in 0..SIGMA_SIZE {
            let sigma = sigma_of_row(k);
            for j in 0..THETA_SIZE {
                let wi = FibreDir { sin_theta: f64::from(theta_of_column(j)).sin(), phi: 0.0 };
                // Per lobe: forward and backward integrals of M_p N̄_p over the outgoing sphere.
                let mut fwd = [0.0_f64; 4];
                let mut bwd = [0.0_f64; 4];
                for ti in 0..DUAL_THETA {
                    let theta_o = -0.5 * PI + PI * (ti as f64 + 0.5) / DUAL_THETA as f64;
                    let (sin_o, cos_o) = theta_o.sin_cos();
                    let d_theta = PI / DUAL_THETA as f64;
                    for pi in 0..DUAL_PHI {
                        let phi = PI * (pi as f64 + 0.5) / DUAL_PHI as f64;
                        // Both signs of φ: twice the half range.
                        let d_omega = cos_o * d_theta * (PI / DUAL_PHI as f64) * 2.0;
                        let wo = FibreDir { sin_theta: sin_o, phi };
                        let n = table.lookup(phi as f32, cos_o as f32, sigma).map(f64::from);
                        let (sin_i, cos_i) = (wi.sin_theta, wi.cos_theta());
                        for p in 0..4 {
                            let (s_o, c_o) = tilted(wo, f64::from(params.shift(p)));
                            let m = longitudinal(cos_i, c_o, sin_i, s_o, f64::from(v[p]));
                            let value = m * n[p] * d_omega;
                            // Forward: the light keeps going (relative azimuth beyond π/2).
                            if phi > 0.5 * PI {
                                fwd[p] += value;
                            } else {
                                bwd[p] += value;
                            }
                        }
                    }
                }
                cells.push(dual_cell(&params, &v, fwd, bwd));
            }
        }
        Self { cells }
    }

    /// The cell at (|θ|, σa), bilinear between cell centres with clamp-to-edge addressing: the
    /// CPU mirror of the shader's filtered lookup.
    #[must_use]
    pub fn lookup(&self, theta: f32, sigma_a: f32) -> DualCell {
        let x =
            (theta.abs() / std::f32::consts::FRAC_PI_2 * THETA_SIZE as f32 - 0.5).clamp(0.0, (THETA_SIZE - 1) as f32);
        let y = (sigma_a.clamp(0.0, SIGMA_MAX) / SIGMA_MAX).sqrt() * (SIGMA_SIZE - 1) as f32;
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(THETA_SIZE - 1), (y0 + 1).min(SIGMA_SIZE - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let at = |i: usize, j: usize| self.cells.get(j * THETA_SIZE + i).copied().unwrap_or_default();
        let mut out = DualCell::default();
        for (i, j, w) in
            [(x0, y0, (1.0 - fx) * (1.0 - fy)), (x1, y0, fx * (1.0 - fy)), (x0, y1, (1.0 - fx) * fy), (x1, y1, fx * fy)]
        {
            let c = at(i, j);
            out.a_f += w * c.a_f;
            out.a_b += w * c.a_b;
            out.alpha_f += w * c.alpha_f;
            out.beta_f2 += w * c.beta_f2;
            out.alpha_b += w * c.alpha_b;
            out.beta_b2 += w * c.beta_b2;
            out.big_a_b += w * c.big_a_b;
            out.delta_b += w * c.delta_b;
            out.sigma_b2 += w * c.sigma_b2;
        }
        out
    }
}

/// Zinke's back-scattering terms from the per-lobe forward and backward integrals.
fn dual_cell(params: &FibreParams, v: &[f32; 4], fwd: [f64; 4], bwd: [f64; 4]) -> DualCell {
    let a_f: f64 = fwd.iter().sum();
    let a_b: f64 = bwd.iter().sum();
    let weighted = |w: &[f64; 4], f: &dyn Fn(usize) -> f64| {
        let total: f64 = w.iter().sum();
        if total <= 1e-12 { 0.0 } else { (0..4).map(|p| w[p] * f(p)).sum::<f64>() / total }
    };
    let alpha_f = weighted(&fwd, &|p| f64::from(params.shift(p)));
    let alpha_b = weighted(&bwd, &|p| f64::from(params.shift(p)));
    let beta_f2 = weighted(&fwd, &|p| f64::from(v[p]));
    let beta_b2 = weighted(&bwd, &|p| f64::from(v[p]));
    let (beta_f, beta_b) = (beta_f2.sqrt(), beta_b2.sqrt());
    let af2 = (a_f * a_f).min(0.999_9);
    let one = 1.0 - af2;
    let a1 = a_b * af2 / one;
    let a3 = a_b.powi(3) * af2 / one.powi(3);
    let delta_b = alpha_b * (1.0 - 2.0 * a_b * a_b / (one * one))
        + alpha_f * (2.0 * one * one + 4.0 * af2 * a_b * a_b) / one.powi(3);
    let num = (1.0 + f64::from(DENSITY_BACKWARD) * af2)
        * (a_b * (2.0 * beta_f2 + beta_b2).sqrt() + a_b.powi(3) * (2.0 * beta_f2 + 3.0 * beta_b2).sqrt());
    let den = a_b + a_b.powi(3) * (2.0 * beta_f + 3.0 * beta_b);
    let sigma_b = if den > 1e-12 { num / den } else { beta_b };
    DualCell {
        a_f: a_f as f32,
        a_b: a_b as f32,
        alpha_f: alpha_f as f32,
        beta_f2: beta_f2 as f32,
        alpha_b: alpha_b as f32,
        beta_b2: beta_b2 as f32,
        big_a_b: (a1 + a3).min(1.0) as f32,
        delta_b: delta_b.clamp(-0.5, 0.5) as f32,
        sigma_b2: (sigma_b * sigma_b).min(1.0) as f32,
    }
}

/// The local back-scattering lobe: `Ab M(θi, θo + Δb; σb² + extra) / 2π`, the longitudinal lobe
/// of d'Eon (normalised, so the lobe carries exactly `Ab` over the sphere) shifted like the
/// single-scattering lobes, spread uniformly over the azimuth.
#[must_use]
pub fn back_scatter(cell: &DualCell, wo: FibreDir, wi: FibreDir, extra_variance: f64) -> f64 {
    let (sin_o, cos_o) = tilted(wo, f64::from(cell.delta_b));
    let var = (f64::from(cell.sigma_b2) + extra_variance).max(1e-3);
    f64::from(cell.big_a_b) * longitudinal(wi.cos_theta(), cos_o, wi.sin_theta, sin_o, var) / (2.0 * PI)
}

/// One channel of dual-scattered radiance per unit light irradiance at a point `tau` expected
/// fibre crossings deep towards the light: unscattered `exp(−τ)` shaded with the single-scattering
/// function plus local back-scattering, and the forward-scattered rest `exp(−τ(1 − āf)) −
/// exp(−τ)` (times Zinke's `df`) shaded with lobes widened by `τ βf²`.
#[must_use]
pub fn dual_scattering(
    table: &AzimuthalTable,
    dual: &DualTable,
    sigma_a: f32,
    tau: f32,
    wo: FibreDir,
    wi: FibreDir,
) -> f32 {
    let cell = dual.lookup(wi.sin_theta.asin() as f32, sigma_a);
    let tau = f64::from(tau.max(0.0));
    let direct = (-tau).exp();
    let scattered = f64::from(DENSITY_FORWARD) * ((-tau * (1.0 - f64::from(cell.a_f))).exp() - direct).max(0.0);
    let spread = tau * f64::from(cell.beta_f2);
    let db = f64::from(DENSITY_BACKWARD);
    let single = f64::from(table.eval(sigma_a, wo, wi, 0.0));
    let widened = f64::from(table.eval(sigma_a, wo, wi, spread as f32));
    let out = direct * (single + db * back_scatter(&cell, wo, wi, 0.0))
        + scattered * (widened + db * back_scatter(&cell, wo, wi, spread));
    out as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    fn uniform_sphere(rng: &mut Rng) -> FibreDir {
        let z = 1.0 - 2.0 * f64::from(rng.f32());
        let r = (1.0 - z * z).max(0.0).sqrt();
        let a = 2.0 * PI * f64::from(rng.f32());
        FibreDir::from_vector([z, r * a.cos(), r * a.sin()])
    }

    /// pbrt-v3's white furnace: without absorption `∫ f dω = 1` for every outgoing direction and
    /// roughness `[HR-02]`.
    #[test]
    fn hr_02_white_furnace_near_field() {
        let mut rng = Rng::new(7);
        for beta in [0.2_f32, 0.3, 0.5, 0.8] {
            let params = FibreParams { beta_m: beta, beta_n: beta, ..FibreParams::default() };
            let wo = uniform_sphere(&mut rng);
            let count = 60_000;
            let mut sum = 0.0;
            for _ in 0..count {
                let h = -1.0 + 2.0 * f64::from(rng.f32());
                let wi = uniform_sphere(&mut rng);
                sum += eval_near(&params, 0.0, h, wo, wi);
            }
            let avg = sum * 4.0 * PI / f64::from(count);
            assert!((avg - 1.0).abs() < 0.05, "beta {beta}: furnace {avg}");
        }
    }

    /// The far-field table reproduces the h-integrated function, and the furnace still holds.
    #[test]
    fn hr_02_far_field_table_matches_the_integral() {
        let params = FibreParams::default();
        let table = AzimuthalTable::build(&params).unwrap();
        let mut rng = Rng::new(11);
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for _ in 0..400 {
            let wo = uniform_sphere(&mut rng);
            let wi = uniform_sphere(&mut rng);
            let sigma = rng.range_f32(0.0, 6.0);
            let reference = eval_far(&params, f64::from(sigma), wo, wi);
            let from_table = f64::from(table.eval(sigma, wo, wi, 0.0));
            worst = worst.max((reference - from_table).abs());
            scale = scale.max(reference);
        }
        // Interpolation error relative to the largest value seen.
        assert!(worst < 0.03 * scale, "worst {worst} of {scale}");
        // Furnace with the table, σa = 0.
        let wo = FibreDir { sin_theta: 0.3, phi: 0.4 };
        let mut sum = 0.0;
        let count = 80_000;
        for _ in 0..count {
            sum += f64::from(table.eval(0.0, wo, uniform_sphere(&mut rng), 0.0));
        }
        let avg = sum * 4.0 * PI / f64::from(count);
        assert!((avg - 1.0).abs() < 0.05, "table furnace {avg}");
    }

    /// The R highlight sits towards the root and the TRT highlight towards the tip: the two
    /// lobes' longitudinal peaks are separated by about 6α (Marschner's dual highlight).
    #[test]
    fn hr_02_dual_highlight_shift() {
        let params = FibreParams { beta_m: 0.1, beta_n: 0.3, alpha: 3.0_f32.to_radians() };
        let v = params.variances().map(f64::from);
        let peak = |p: usize| {
            let mut best = (f64::MIN, 0.0);
            for k in 0..2001 {
                let theta_o = -0.6 + 1.2 * f64::from(k) / 2000.0;
                let wo = FibreDir { sin_theta: theta_o.sin(), phi: 0.0 };
                let (s, c) = tilted(wo, f64::from(params.shift(p)));
                let m = longitudinal(1.0, c, 0.0, s, v[p]);
                if m > best.0 {
                    best = (m, theta_o);
                }
            }
            best.1
        };
        let (r, trt) = (peak(0), peak(2));
        let alpha = f64::from(params.alpha);
        assert!((r - 2.0 * alpha).abs() < 0.01, "R peak {r}");
        assert!((trt + 4.0 * alpha).abs() < 0.01, "TRT peak {trt}");
    }

    /// Melanin darkens: more absorption, less scattered energy, monotonically; blue is absorbed
    /// first (brown hair is red-brown).
    #[test]
    fn hr_02_absorption_is_monotonic() {
        let table = AzimuthalTable::build(&FibreParams::default()).unwrap();
        let dual = DualTable::build(&table);
        let mut last = f32::MAX;
        for k in 0..SIGMA_SIZE {
            let c = dual.cells[k * THETA_SIZE + 4];
            let total = c.a_f + c.a_b;
            assert!(total <= last + 1e-3, "row {k}: {total} after {last}");
            assert!(total <= 1.02, "albedo {total}");
            last = total;
        }
        // σa = 0: forward and backward add to the furnace's 1.
        let c = dual.cells[4];
        assert!((c.a_f + c.a_b - 1.0).abs() < 0.04, "{c:?}");
        assert!(c.a_f > c.a_b, "a clear fibre mostly scatters forward: {c:?}");
        let brown = humanart_model::hair::color::sigma_a_from_concentration(1.3, 0.0);
        let albedo = |s: f32| {
            let c = dual.lookup(0.3, s);
            c.a_f + c.a_b
        };
        assert!(albedo(brown[0]) > albedo(brown[1]) && albedo(brown[1]) > albedo(brown[2]));
    }

    /// The back-scattering lobe carries about `Ab` over the sphere, and the dual-scattered
    /// response falls with depth into the hair without going negative.
    #[test]
    fn hr_02_dual_scattering_terms() {
        let table = AzimuthalTable::build(&FibreParams::default()).unwrap();
        let dual = DualTable::build(&table);
        let cell = dual.lookup(0.2, 0.3);
        assert!(cell.big_a_b > 0.0 && cell.big_a_b < 1.0, "{cell:?}");
        let wi = FibreDir { sin_theta: 0.2_f64.sin(), phi: 0.0 };
        let mut rng = Rng::new(3);
        let count = 60_000;
        let mut sum = 0.0;
        for _ in 0..count {
            sum += back_scatter(&cell, uniform_sphere(&mut rng), wi, 0.0);
        }
        let total = sum * 4.0 * PI / f64::from(count);
        assert!((total - f64::from(cell.big_a_b)).abs() < 0.05 * f64::from(cell.big_a_b), "{total} vs {cell:?}");
        let wo = FibreDir { sin_theta: 0.1, phi: 2.0 };
        let mut last = f32::MAX;
        for tau in [0.0_f32, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0] {
            let v = dual_scattering(&table, &dual, 0.3, tau, wo, wi);
            assert!(v >= 0.0 && v <= last + 1e-6, "tau {tau}: {v} after {last}");
            last = v;
        }
        // Blonde hair keeps light deep inside (forward scattering), dark hair does not.
        let deep =
            |s: f32| dual_scattering(&table, &dual, s, 4.0, wo, wi) / dual_scattering(&table, &dual, s, 0.0, wo, wi);
        assert!(deep(0.1) > 3.0 * deep(8.0), "blonde {} dark {}", deep(0.1), deep(8.0));
    }

    #[test]
    fn bad_fibre_settings_are_rejected() {
        let bad = FibreParams { beta_m: f32::NAN, ..FibreParams::default() };
        assert!(AzimuthalTable::build(&bad).is_err());
        assert!(FibreParams { alpha: 0.5, ..FibreParams::default() }.check().is_err());
        assert!(FibreParams { beta_n: 0.0, ..FibreParams::default() }.check().is_err());
        assert!(FibreParams::default().check().is_ok());
    }
}
