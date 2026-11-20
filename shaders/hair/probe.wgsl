// hair/probe.wgsl: test kernel of the hair fibre scattering [HR-02]: evaluates
// hair_dual_scattering for a list of cases so the tests compare it with the CPU reference
// (crates/render/src/hair/bsdf.rs).

#include "lib/math.wgsl"
#include "hair/bsdf.wgsl"

@group(0) @binding(0) var hair_azimuthal: texture_3d<f32>;
@group(0) @binding(1) var hair_dual: texture_2d_array<f32>;
@group(0) @binding(2) var hair_lut_sampler: sampler;

// One case: sigma.xyz + tau, tangent, wo, wi (xyz each, w unused).
struct ProbeCase {
    sigma_tau: vec4<f32>,
    tangent: vec4<f32>,
    wo: vec4<f32>,
    wi: vec4<f32>,
}

@group(0) @binding(3) var<uniform> probe_fibre: HairFibre;
@group(0) @binding(4) var<storage, read> probe_cases: array<ProbeCase>;
@group(0) @binding(5) var<storage, read_write> probe_out: array<vec4<f32>>;

@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&probe_cases)) {
        return;
    }
    let c = probe_cases[i];
    let v = hair_dual_scattering(c.sigma_tau.xyz, c.tangent.xyz, c.wo.xyz, c.wi.xyz, c.sigma_tau.w, probe_fibre, 0.0);
    probe_out[i] = vec4<f32>(v, 0.0);
}
