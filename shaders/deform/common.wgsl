// deform/common.wgsl: helpers shared by the GPU deformation passes (humanart_render::deform).
//
// Every deformation pass runs 64 invocations per workgroup on a 2D grid of workgroups (x at most
// 65535, as the default limits allow), so a stream of more than 4 million items is still one
// dispatch. `thread_index` flattens the grid; passes compare it with their item count.
//
// Streams are structure of arrays [PERF-R2]: positions, rest positions and normals are packed f32
// triples (`array<f32>`, three per vertex: the layout of the scene's position stream), so the
// renderer pulls them directly [EVAL-05].

const WORKGROUP: u32 = 64u;

// The flat index of an invocation of a 2D dispatch of WORKGROUP-wide groups.
fn thread_index(group: vec3<u32>, groups: vec3<u32>, local: u32) -> u32 {
    return (group.y * groups.x + group.x) * WORKGROUP + local;
}

// The value of a finite IEEE half in the low 16 bits of `bits`, exactly, without f32 subnormals
// (some GPUs flush them): a subnormal half is its mantissa times 2^-24, a normal half is rebuilt
// with the f32 exponent bias. The CPU twin is humanart_model::eval::evaluator::decode_finite_f16.
fn decode_f16(bits: u32) -> f32 {
    let exponent = (bits >> 10u) & 31u;
    let mantissa = bits & 1023u;
    var magnitude: f32;
    if (exponent == 0u) {
        magnitude = f32(mantissa) * 5.9604645e-8;
    } else {
        magnitude = bitcast<f32>(((exponent + 112u) << 23u) | (mantissa << 13u));
    }
    return select(magnitude, -magnitude, (bits & 0x8000u) != 0u);
}

// The delta of a packed entry word pair: `xy` holds the x half (low) and the y half (high), the
// low 16 bits of `z` the z half.
fn decode_delta(xy: u32, z: u32) -> vec3<f32> {
    return vec3<f32>(decode_f16(xy & 0xffffu), decode_f16(xy >> 16u), decode_f16(z & 0xffffu));
}
