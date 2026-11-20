// test/packing.wgsl: runs lib/packing.wgsl for the CPU comparison in crates/shaders/tests/library.rs
// [AC-WP05-2].

#include "test/harness.wgsl"
#include "lib/packing.wgsl"

// mode 0: input xyz direction, w bitangent sign -> (oct16, oct8, tangent, snorm 10-10-10-2).
// mode 1: input values -> (unorm 10-10-10-2, pack4x8snorm, pack4x8unorm, pack2x16snorm of xy).
// mode 2: input values -> (pack4x16float, pack2x16unorm of xy, pack2x16unorm of zw).
@compute @workgroup_size(64)
fn encode(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let v = input_f32(i);
    switch params.mode {
        case 0u: {
            output[i] = vec4<u32>(pack_oct16(v.xyz), pack_oct8(v.xyz), pack_tangent(v.xyz, v.w), pack_snorm_10_10_10_2(v));
        }
        case 1u: {
            output[i] = vec4<u32>(pack_unorm_10_10_10_2(v), pack4x8snorm(v), pack4x8unorm(v), pack2x16snorm(v.xy));
        }
        default: {
            output[i] = vec4<u32>(pack4x16float(v), pack2x16unorm(v.xy), pack2x16unorm(v.zw));
        }
    }
}

// Input word x decoded by mode: 0 unpack_oct16, 1 unpack_oct8, 2 unpack_tangent,
// 3 unpack_snorm_10_10_10_2, 4 unpack_unorm_10_10_10_2, 5 unpack4x16float of (x, y).
@compute @workgroup_size(64)
fn decode(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let p = input[i];
    var r = vec4<f32>(0.0);
    switch params.mode {
        case 0u: {
            r = vec4<f32>(unpack_oct16(p.x), 0.0);
        }
        case 1u: {
            r = vec4<f32>(unpack_oct8(p.x), 0.0);
        }
        case 2u: {
            r = unpack_tangent(p.x);
        }
        case 3u: {
            r = unpack_snorm_10_10_10_2(p.x);
        }
        case 4u: {
            r = unpack_unorm_10_10_10_2(p.x);
        }
        default: {
            r = unpack4x16float(p.xy);
        }
    }
    store_f32(i, r);
}
