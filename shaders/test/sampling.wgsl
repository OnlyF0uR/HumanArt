// test/sampling.wgsl: runs lib/sampling.wgsl for the CPU comparison and the distribution moments in
// crates/shaders/tests/library.rs [AC-WP05-2].

#include "test/harness.wgsl"
#include "lib/sampling.wgsl"
#include "lib/brdf.wgsl"

// Point i: (hammersley(i, count), r2_sequence(i)). No input is read.
@compute @workgroup_size(64)
fn sequences(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    store_f32(i, vec4<f32>(hammersley(i, params.count), r2_sequence(i)));
}

// Point i of the Hammersley set of `count` points warped by mode: 0 uniform sphere, 1 uniform
// hemisphere, 2 cosine hemisphere (w = its density), 3 GGX NDF with alpha = a (w = D(h) h.z),
// 4 GGX VNDF with alpha = a for the view of cosine b in the xz plane (w = its density),
// 5 concentric disk (zw = 0). No input is read.
@compute @workgroup_size(64)
fn warps(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let u = hammersley(i, params.count);
    var r = vec4<f32>(0.0);
    switch params.mode {
        case 0u: {
            r = vec4<f32>(sample_uniform_sphere(u), 0.25 * INV_PI);
        }
        case 1u: {
            r = vec4<f32>(sample_uniform_hemisphere(u), INV_TAU);
        }
        case 2u: {
            let d = sample_cosine_hemisphere(u);
            r = vec4<f32>(d, pdf_cosine_hemisphere(d.z));
        }
        case 3u: {
            let h = sample_ggx_ndf(u, params.a);
            r = vec4<f32>(h, d_ggx(h.z, params.a) * h.z);
        }
        case 4u: {
            let v = vec3<f32>(sqrt(1.0 - params.b * params.b), 0.0, params.b);
            let h = sample_ggx_vndf(u, v, params.a);
            r = vec4<f32>(h, pdf_ggx_vndf(v, h, params.a));
        }
        default: {
            r = vec4<f32>(sample_disk_concentric(u), 0.0, 0.0);
        }
    }
    store_f32(i, r);
}
