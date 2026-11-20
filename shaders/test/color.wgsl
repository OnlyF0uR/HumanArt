// test/color.wgsl: runs lib/color.wgsl for the CPU comparison in crates/shaders/tests/library.rs
// [AC-WP05-2].

#include "test/harness.wgsl"
#include "lib/color.wgsl"

// mode 0: (linear_to_srgb of xyz, luminance of xyz); mode 1: (srgb_to_linear of xyz,
// exposure_from_ev100 of w).
@compute @workgroup_size(64)
fn transfer(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let v = input_f32(i);
    if (params.mode == 0u) {
        store_f32(i, vec4<f32>(linear_to_srgb_rgb(v.xyz), luminance(v.xyz)));
    } else {
        store_f32(i, vec4<f32>(srgb_to_linear_rgb(v.xyz), exposure_from_ev100(v.w)));
    }
}

// tone_map(mode, xyz, exposure = params.a).
@compute @workgroup_size(64)
fn tone_mappers(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    store_f32(i, vec4<f32>(tone_map(params.mode, input_f32(i).xyz, params.a), 0.0));
}
