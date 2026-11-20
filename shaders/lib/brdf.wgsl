// lib/brdf.wgsl: the microfacet building blocks of the skin model [SHD-01]: the GGX
// (Trowbridge-Reitz) distribution, height-correlated Smith visibility, Schlick and exact dielectric
// Fresnel, the dual specular lobe, the split-sum DFG integral and multi-scatter energy compensation.
//
// `alpha` is the GGX width, the perceptual roughness squared. Directions are unit vectors; the dot
// products are clamped by the caller (n.v and n.l > 0 for a lit, visible point).
// Derivations: docs/shader-notes.md, "BRDF".

#include "lib/math.wgsl"
#include "lib/sampling.wgsl"

// The reflectance of skin at normal incidence ((1.4 - 1) / (1.4 + 1))^2 ~ 0.028 [SHD-01].
const F0_SKIN: f32 = 0.028;
// The narrowest lobe used; below it D overflows f32 at n.h = 1.
const MIN_ALPHA: f32 = 0.002;

// The GGX width of a perceptual roughness, never below MIN_ALPHA.
fn alpha_from_roughness(roughness: f32) -> f32 {
    return max(roughness * roughness, MIN_ALPHA);
}

// The GGX normal distribution D(h) = alpha^2 / (pi ((n.h)^2 (alpha^2 - 1) + 1)^2).
fn d_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = (n_dot_h * a2 - n_dot_h) * n_dot_h + 1.0;
    return a2 / (PI * d * d);
}

// The Smith masking function G1 of GGX: 2 n.v / (n.v + sqrt(alpha^2 + (1 - alpha^2) (n.v)^2)).
fn g1_smith_ggx(n_dot_v: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    return 2.0 * n_dot_v / (n_dot_v + sqrt(a2 + (1.0 - a2) * n_dot_v * n_dot_v));
}

// The height-correlated Smith visibility V = G2 / (4 n.l n.v) of GGX (Heitz 2014, exact form):
// 0.5 / (n.l sqrt((n.v)^2 (1 - alpha^2) + alpha^2) + n.v sqrt((n.l)^2 (1 - alpha^2) + alpha^2)).
fn v_smith_ggx_correlated(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let gv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let gl = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-20);
}

// (1 - x)^5, the Schlick weight.
fn schlick_weight(cos_theta: f32) -> f32 {
    let m = saturate(1.0 - cos_theta);
    let m2 = m * m;
    return m2 * m2 * m;
}

// Schlick's Fresnel approximation from f0 (at normal incidence) to f90 (at grazing).
fn f_schlick(f0: vec3<f32>, f90: f32, v_dot_h: f32) -> vec3<f32> {
    return f0 + (vec3<f32>(f90) - f0) * schlick_weight(v_dot_h);
}

// Schlick's Fresnel for one channel.
fn f_schlick_scalar(f0: f32, f90: f32, v_dot_h: f32) -> f32 {
    return f0 + (f90 - f0) * schlick_weight(v_dot_h);
}

// The reflectance at normal incidence of a dielectric of relative index `ior`.
fn f0_from_ior(ior: f32) -> f32 {
    return sq((ior - 1.0) / (ior + 1.0));
}

// The exact unpolarised Fresnel reflectance of a dielectric interface with relative index `eta`
// (transmitted over incident side) at incidence cosine `cos_i` (Walter et al. 2007, eq. 22).
// Total internal reflection gives 1.
fn f_dielectric(cos_i: f32, eta: f32) -> f32 {
    let c = abs(cos_i);
    let g2 = eta * eta - 1.0 + c * c;
    if (g2 < 0.0) {
        return 1.0;
    }
    let g = sqrt(g2);
    let a = (g - c) / (g + c);
    let b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    return 0.5 * a * a * (1.0 + b * b);
}

// One GGX specular lobe without the cosine: D * V * F.
fn specular_ggx(n_dot_v: f32, n_dot_l: f32, n_dot_h: f32, v_dot_h: f32, alpha: f32, f0: vec3<f32>) -> vec3<f32> {
    let dv = d_ggx(n_dot_h, alpha) * v_smith_ggx_correlated(n_dot_v, n_dot_l, alpha);
    return dv * f_schlick(f0, 1.0, v_dot_h);
}

// The dual-lobe distribution-visibility term of skin [SHD-01]: the two GGX lobes `alpha0` and
// `alpha1` mixed by `blend` (0 = only the first). Multiply by Fresnel and the cosine.
fn dual_lobe_dv(n_dot_v: f32, n_dot_l: f32, n_dot_h: f32, alpha0: f32, alpha1: f32, blend: f32) -> f32 {
    let lobe0 = d_ggx(n_dot_h, alpha0) * v_smith_ggx_correlated(n_dot_v, n_dot_l, alpha0);
    let lobe1 = d_ggx(n_dot_h, alpha1) * v_smith_ggx_correlated(n_dot_v, n_dot_l, alpha1);
    return mix(lobe0, lobe1, blend);
}

// The Lambert diffuse BRDF for unit albedo.
fn diffuse_lambert() -> f32 {
    return INV_PI;
}

// The density of sample_ggx_vndf: G1(v) max(v.h, 0) D(h) / v.z, for local unit v and h.
fn pdf_ggx_vndf(v: vec3<f32>, h: vec3<f32>, alpha: f32) -> f32 {
    return g1_smith_ggx(v.z, alpha) * max(dot(v, h), 0.0) * d_ggx(h.z, alpha) / v.z;
}

// The split-sum DFG terms (Karis 2013) for view cosine `n_dot_v`: (A, B) with the directional
// albedo of one lobe equal to f0 * A + f90 * B. Integrated with `sample_count` Hammersley points
// warped by sample_ggx_vndf, where each sample weighs G2 / G1(v). For baking the DFG table, not
// for per-pixel use.
fn integrate_dfg(n_dot_v: f32, alpha: f32, sample_count: u32) -> vec2<f32> {
    let nv = clamp(n_dot_v, 1e-4, 1.0);
    let v = vec3<f32>(sqrt(1.0 - nv * nv), 0.0, nv);
    let g1v = g1_smith_ggx(nv, alpha);
    var sum = vec2<f32>(0.0);
    let n = max(sample_count, 1u);
    for (var i = 0u; i < n; i++) {
        let h = sample_ggx_vndf(hammersley(i, n), v, alpha);
        let v_dot_h = dot(v, h);
        let l = 2.0 * v_dot_h * h - v;
        if (l.z > 0.0 && v_dot_h > 0.0) {
            // f cos / pdf = D V F n.l / (G1 D (v.h) / n.v / (4 v.h)) = 4 V n.l n.v F / G1.
            let w = 4.0 * v_smith_ggx_correlated(nv, l.z, alpha) * l.z * nv / g1v;
            let fc = schlick_weight(v_dot_h);
            sum += vec2<f32>((1.0 - fc) * w, fc * w);
        }
    }
    return sum / f32(n);
}

// The multi-scatter energy compensation factor of one lobe (Fdez-Aguera 2019, as in Filament):
// 1 + f0 (1 / E - 1) with E = A + B the single-scatter albedo for f0 = f90 = 1. Multiply the
// single-scatter specular by it.
fn specular_energy_compensation(f0: vec3<f32>, dfg: vec2<f32>) -> vec3<f32> {
    let e = max(dfg.x + dfg.y, 1e-4);
    return vec3<f32>(1.0) + f0 * (1.0 / e - 1.0);
}
