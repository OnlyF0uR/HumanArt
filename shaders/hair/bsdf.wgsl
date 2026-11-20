// hair/bsdf.wgsl: the far-field hair fibre scattering of crates/render/src/hair/bsdf.rs [HR-02]:
// the R, TT, TRT and residual lobes of Chiang et al. 2016 (pbrt-v3 HairBSDF) with d'Eon's
// longitudinal lobes tilted by the cuticle angle (the dual highlight), the azimuthal lobes
// integrated over the fibre width from a table, melanin absorption per colour channel, and the
// dual-scattering approximation of Zinke et al. 2008 (forward scattering through `tau` expected
// fibre crossings towards the light, local back-scattering).
//
// Needs lib/math.wgsl. The including shader declares:
//   hair_azimuthal: texture_3d<f32>          (x phi over [0, pi], y theta_o over [0, pi/2],
//                                             z sqrt(sigma_a / HAIR_SIGMA_MAX)), rgba = N0, N1, N2, A3/2pi
//   hair_dual: texture_2d_array<f32>         (x theta over [0, pi/2], y as z above), layer 0 =
//                                             (a_f, A_b, delta_b, sigma_b^2), layer 1 = (beta_f^2, alpha_f, a_b, 0)
//   hair_lut_sampler: sampler                (linear, clamp to edge)

const HAIR_SIGMA_MAX: f32 = 16.0;
const HAIR_SIGMA_SIZE: f32 = 32.0;
// Zinke's density factors of forward and backward scattering.
const HAIR_DENSITY_FORWARD: f32 = 0.7;
const HAIR_DENSITY_BACKWARD: f32 = 0.7;

// The fibre's longitudinal variances v0..v3 and lobe shifts (R, TT, TRT, back-scatter unused).
struct HairFibre {
    variances: vec4<f32>,
    shifts: vec4<f32>,
}

// I0(x), ten terms of its series (pbrt-v3).
fn hair_i0(x: f32) -> f32 {
    var val = 0.0;
    var x2i = 1.0;
    var ifact = 1.0;
    var i4 = 1.0;
    for (var i = 0; i < 10; i++) {
        if (i > 1) {
            ifact *= f32(i);
        }
        val += x2i / (i4 * ifact * ifact);
        x2i *= x * x;
        i4 *= 4.0;
    }
    return val;
}

// ln I0(x) (pbrt-v3 LogI0).
fn hair_log_i0(x: f32) -> f32 {
    if (x > 12.0) {
        return x + 0.5 * (-log(TAU) + log(1.0 / x) + 1.0 / (8.0 * x));
    }
    return log(hair_i0(x));
}

// d'Eon's longitudinal lobe for variance v.
fn hair_longitudinal(cos_i: f32, cos_o: f32, sin_i: f32, sin_o: f32, v: f32) -> f32 {
    let a = cos_i * cos_o / v;
    let b = sin_i * sin_o / v;
    if (v <= 0.1) {
        return exp(hair_log_i0(a) - b - 1.0 / v + 0.6931472 + log(1.0 / (2.0 * v)));
    }
    // sinh(1/v) for v > 0.1 stays below about 1.1e4.
    let s = 0.5 * (exp(1.0 / v) - exp(-1.0 / v));
    return exp(-b) * hair_i0(a) / (s * 2.0 * v);
}

// sin and |cos| of theta + shift for (sin theta, cos theta) = sc.
fn hair_tilt(sc: vec2<f32>, shift: f32) -> vec2<f32> {
    let ss = sin(shift);
    let cs = cos(shift);
    return vec2<f32>(sc.x * cs + sc.y * ss, abs(sc.y * cs - sc.x * ss));
}

// The table coordinate of absorption sigma (square-root axis, row centres).
fn hair_sigma_coord(sigma: f32) -> f32 {
    let u = sqrt(clamp(sigma, 0.0, HAIR_SIGMA_MAX) / HAIR_SIGMA_MAX);
    return (u * (HAIR_SIGMA_SIZE - 1.0) + 0.5) / HAIR_SIGMA_SIZE;
}

// The h-integrated azimuthal terms (N0, N1, N2, A3/2pi) at |phi|, |theta_o| and absorption sigma.
fn hair_azimuthal_terms(phi: f32, theta_o: f32, sigma: f32) -> vec4<f32> {
    let uvw = vec3<f32>(phi * INV_PI, theta_o / HALF_PI, hair_sigma_coord(sigma));
    return textureSampleLevel(hair_azimuthal, hair_lut_sampler, uvw, 0.0);
}

// The four longitudinal lobes at (sin, cos) of theta_o and theta_i, every variance plus `extra`.
fn hair_lobes(o: vec2<f32>, i: vec2<f32>, fibre: HairFibre, extra: f32) -> vec4<f32> {
    let r = hair_tilt(o, fibre.shifts.x);
    let tt = hair_tilt(o, fibre.shifts.y);
    let trt = hair_tilt(o, fibre.shifts.z);
    let v = fibre.variances + vec4<f32>(extra);
    return vec4<f32>(
        hair_longitudinal(i.y, r.y, i.x, r.x, v.x),
        hair_longitudinal(i.y, tt.y, i.x, tt.x, v.y),
        hair_longitudinal(i.y, trt.y, i.x, trt.x, v.z),
        hair_longitudinal(i.y, o.y, i.x, o.x, v.w),
    );
}

// The far-field single-scattering function per channel of absorption `sigma`: radiance per unit
// irradiance perpendicular to the light. `o`, `i` = (sin, cos) of theta_o, theta_i; `phi` the
// relative azimuth in [0, pi]; `extra` widens every longitudinal lobe.
fn hair_single(sigma: vec3<f32>, o: vec2<f32>, i: vec2<f32>, phi: f32, fibre: HairFibre, extra: f32) -> vec3<f32> {
    let m = hair_lobes(o, i, fibre, extra);
    let theta_o = asin(min(abs(o.x), 1.0));
    return vec3<f32>(
        dot(m, hair_azimuthal_terms(phi, theta_o, sigma.r)),
        dot(m, hair_azimuthal_terms(phi, theta_o, sigma.g)),
        dot(m, hair_azimuthal_terms(phi, theta_o, sigma.b)),
    );
}

// The dual-scattering cell of one channel: layer 0 and layer 1 at |theta_i| and sigma.
fn hair_dual_cell(theta_i: f32, sigma: f32, layer: i32) -> vec4<f32> {
    let uv = vec2<f32>(theta_i / HALF_PI, hair_sigma_coord(sigma));
    return textureSampleLevel(hair_dual, hair_lut_sampler, uv, layer, 0.0);
}

// Zinke's local back-scattering of one channel: A_b M(theta_i, theta_o + delta_b; sigma_b^2 +
// extra) / 2pi (d'Eon's normalised lobe, uniform over the azimuth).
fn hair_back(cell: vec4<f32>, o: vec2<f32>, i: vec2<f32>, extra: f32) -> f32 {
    let t = hair_tilt(o, cell.z);
    return cell.y * hair_longitudinal(i.y, t.y, i.x, t.x, max(cell.w + extra, 1e-3)) * INV_TAU;
}

// The dual-scattered radiance per unit irradiance of a fibre with tangent `t` seen along `wo`
// (towards the viewer) and lit along `wi` (towards the light), `tau` expected fibre crossings
// deep towards the light (0: directly lit): exp(-tau) arrives unscattered and is shaded with the
// single-scattering function plus local back-scattering; the forward-scattered rest
// exp(-tau (1 - a_f)) - exp(-tau) (times the density factor) with lobes widened by tau beta_f^2.
// `extra` widens every longitudinal lobe by the light's own angular variance (an area light seen
// from the fibre is a spread of directions, not the point at its centre; the lobes stay
// normalised, so a glint spreads instead of growing with the panel's solid angle).
fn hair_dual_scattering(sigma: vec3<f32>, t: vec3<f32>, wo: vec3<f32>, wi: vec3<f32>, tau: f32, fibre: HairFibre, extra: f32) -> vec3<f32> {
    let sin_o = clamp(dot(wo, t), -1.0, 1.0);
    let sin_i = clamp(dot(wi, t), -1.0, 1.0);
    let o = vec2<f32>(sin_o, sqrt(max(1.0 - sin_o * sin_o, 0.0)));
    let i = vec2<f32>(sin_i, sqrt(max(1.0 - sin_i * sin_i, 0.0)));
    let po = wo - t * sin_o;
    let pi = wi - t * sin_i;
    let denom = sqrt(max(dot(po, po) * dot(pi, pi), 1e-12));
    let phi = acos(clamp(dot(po, pi) / denom, -1.0, 1.0));
    let theta_i = asin(min(abs(sin_i), 1.0));
    let c0r = hair_dual_cell(theta_i, sigma.r, 0);
    let c0g = hair_dual_cell(theta_i, sigma.g, 0);
    let c0b = hair_dual_cell(theta_i, sigma.b, 0);
    let direct = exp(-tau);
    let single = hair_single(sigma, o, i, phi, fibre, extra);
    let back = vec3<f32>(hair_back(c0r, o, i, extra), hair_back(c0g, o, i, extra), hair_back(c0b, o, i, extra));
    var out = direct * (single + HAIR_DENSITY_BACKWARD * back);
    if (tau > 1e-3) {
        let a_f = vec3<f32>(c0r.x, c0g.x, c0b.x);
        let scattered = HAIR_DENSITY_FORWARD * max(exp(-tau * (vec3<f32>(1.0) - a_f)) - vec3<f32>(direct), vec3<f32>(0.0));
        if (max3(scattered) > 1e-5) {
            let beta_f2 = (hair_dual_cell(theta_i, sigma.r, 1).x + hair_dual_cell(theta_i, sigma.g, 1).x
                + hair_dual_cell(theta_i, sigma.b, 1).x) / 3.0;
            let spread = tau * beta_f2 + extra;
            let widened = hair_single(sigma, o, i, phi, fibre, spread);
            let wback = vec3<f32>(hair_back(c0r, o, i, spread), hair_back(c0g, o, i, spread), hair_back(c0b, o, i, spread));
            out += scattered * (widened + HAIR_DENSITY_BACKWARD * wback);
        }
    }
    return max(out, vec3<f32>(0.0));
}

// Average forward and backward attenuation of one fibre per channel at |theta|: (a_f, a_b).
fn hair_albedo(theta: f32, sigma: vec3<f32>) -> array<vec3<f32>, 2> {
    let r = hair_dual_cell(theta, sigma.r, 1);
    let g = hair_dual_cell(theta, sigma.g, 1);
    let b = hair_dual_cell(theta, sigma.b, 1);
    let fr = hair_dual_cell(theta, sigma.r, 0).x;
    let fg = hair_dual_cell(theta, sigma.g, 0).x;
    let fb = hair_dual_cell(theta, sigma.b, 0).x;
    return array<vec3<f32>, 2>(vec3<f32>(fr, fg, fb), vec3<f32>(r.z, g.z, b.z));
}
