// test/brdf.wgsl: runs lib/brdf.wgsl for the white-furnace and CPU comparisons in
// crates/shaders/tests/library.rs [AC-WP05-2]. Its overrides are specialised for every value listed
// in the manifest [PERF-R5].

#include "test/harness.wgsl"
#include "lib/brdf.wgsl"

// Threads per workgroup.
override WORKGROUP_SIZE: u32 = 64u;
// Samples per integrate_dfg call.
override SAMPLE_COUNT: u32 = 1024u;

// The quadrature grid of furnace_rows over the hemisphere: theta rows, phi columns per row.
const THETA_STEPS: u32 = 1024u;
const PHI_STEPS: u32 = 512u;

// Thread t integrates row t % THETA_STEPS of cell t / THETA_STEPS over phi (midpoint rule) and
// writes the row's share of the hemisphere integral to output t.x. Cell input: (n.v, alpha0,
// alpha1, blend). Integrand by mode: 0 specular_ggx with f0 = 1 times n.l (the white furnace),
// 1 dual_lobe_dv times n.l, 2 d_ggx(cos theta) cos theta over half vectors (D's normalisation),
// 3 diffuse_lambert times n.l.
@compute @workgroup_size(WORKGROUP_SIZE)
fn furnace_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    let t = id.x;
    if (t >= params.count) {
        return;
    }
    let cell = input_f32(t / THETA_STEPS);
    let row = t % THETA_STEPS;
    let n_dot_v = cell.x;
    let v = vec3<f32>(sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v);
    let d_theta = HALF_PI / f32(THETA_STEPS);
    let d_phi = TAU / f32(PHI_STEPS);
    let theta = (f32(row) + 0.5) * d_theta;
    let sin_theta = sin(theta);
    let cos_theta = cos(theta);
    var sum = 0.0;
    for (var j = 0u; j < PHI_STEPS; j++) {
        let phi = (f32(j) + 0.5) * d_phi;
        let l = vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
        let h = normalize(v + l);
        var f = 0.0;
        switch params.mode {
            case 0u: {
                f = specular_ggx(n_dot_v, l.z, h.z, dot(v, h), cell.y, vec3<f32>(1.0)).x * l.z;
            }
            case 1u: {
                f = dual_lobe_dv(n_dot_v, l.z, h.z, cell.y, cell.z, cell.w) * l.z;
            }
            case 2u: {
                f = d_ggx(l.z, cell.y) * l.z;
            }
            default: {
                f = diffuse_lambert() * l.z;
            }
        }
        sum += f;
    }
    store_f32(t, vec4<f32>(sum * sin_theta * d_theta * d_phi, 0.0, 0.0, 0.0));
}

// Cell i: (n.v, alpha) -> (A, B, A + B, specular_energy_compensation for f0 = F0_SKIN).
@compute @workgroup_size(WORKGROUP_SIZE)
fn dfg(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let cell = input_f32(i);
    let ab = integrate_dfg(cell.x, cell.y, SAMPLE_COUNT);
    store_f32(i, vec4<f32>(ab, ab.x + ab.y, specular_energy_compensation(vec3<f32>(F0_SKIN), ab).x));
}

// Case i: (cos_i, eta, f0, n.v) -> (f_dielectric(cos_i, eta), f_schlick_scalar(f0, 1, cos_i),
// f0_from_ior(eta), g1_smith_ggx(n.v, alpha = params.a)).
@compute @workgroup_size(WORKGROUP_SIZE)
fn fresnel(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let c = input_f32(i);
    store_f32(i, vec4<f32>(f_dielectric(c.x, c.y), f_schlick_scalar(c.z, 1.0, c.x), f0_from_ior(c.y), g1_smith_ggx(c.w, params.a)));
}
