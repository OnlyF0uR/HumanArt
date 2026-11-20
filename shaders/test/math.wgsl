// test/math.wgsl: runs lib/math.wgsl for the CPU comparison in crates/shaders/tests/library.rs
// [AC-WP05-2].

#include "test/harness.wgsl"
#include "lib/math.wgsl"

// Case i reads (q, r, v) from input 3i..3i+3 and writes (quat_rotate(q, v), quat_mul(q, r),
// safe_normalize(v, +Z)) to output 3i..3i+3.
@compute @workgroup_size(64)
fn quaternions(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let q = input_f32(3u * i);
    let r = input_f32(3u * i + 1u);
    let v = input_f32(3u * i + 2u).xyz;
    store_f32(3u * i, vec4<f32>(quat_rotate(q, v), 0.0));
    store_f32(3u * i + 1u, quat_mul(q, r));
    store_f32(3u * i + 2u, vec4<f32>(safe_normalize(v, vec3<f32>(0.0, 0.0, 1.0)), 0.0));
}

// Case i reads a unit normal and writes the columns of orthonormal_basis to output 3i..3i+3, the
// last one's w being to_local(basis, n).z (1 for a correct frame).
@compute @workgroup_size(64)
fn basis(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let n = input_f32(i).xyz;
    let m = orthonormal_basis(n);
    store_f32(3u * i, vec4<f32>(m[0], 0.0));
    store_f32(3u * i + 1u, vec4<f32>(m[1], 0.0));
    store_f32(3u * i + 2u, vec4<f32>(m[2], to_local(m, from_local(m, vec3<f32>(0.0, 0.0, 1.0))).z));
}
