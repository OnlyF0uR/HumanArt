// deform/skin.wgsl: expressions, correctives and skinning (spec 6.2 steps 2 to 4) on the GPU
// [EVAL-02] [SKIN-01] [SKIN-02].
//
// One invocation per vertex:
//
// 1. P = P_rest + sum_c w_c * D_c over the vertex's row of the vertex-major dynamic table: the
//    components of expressions, combination correctives and pose correctives, with the weights
//    and activations the CPU side resolved for this frame (one f32 per component). The table is
//    sliced per workgroup like the shape table of deform/accumulate.wgsl (coalesced reads).
// 2. Skinning with the posed joint transforms S_j(p) = D_j p + t_j, five vec4 per joint:
//    (D_j column 0, t_j.x), (column 1, t_j.y), (column 2, t_j.z), the dual quaternion's real part,
//    its dual part. Linear blending: sum w_k S_k(p) / sum w_k. Dual quaternions: the blend of the
//    influences' dual quaternions, each flipped onto the hemisphere of the first, normalised, then
//    applied. Influences of weight 0 or with an unknown joint are skipped; no weight leaves P.
//    The first influence's joint word carries the vertex's dual-quaternion flag in bit 31 (the
//    template's dual-quaternion regions).
//
// The CPU reference is humanart_model::eval::Evaluator (`apply_expressions`,
// `apply_pose_correctives`, `skin`) with humanart_model::eval::JointFrames.

#include "deform/common.wgsl"
#include "lib/math.wgsl"

struct Params {
    vertex_count: u32,
    influences: u32,
    joint_count: u32,
    // 0: dual quaternions where the vertex is flagged, 1: linear everywhere, 2: dual quaternions
    // everywhere.
    mode: u32,
}

const MODE_BY_REGION: u32 = 0u;
const MODE_LINEAR: u32 = 1u;
const DQ_FLAG: u32 = 0x80000000u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> rest: array<f32>;
@group(0) @binding(2) var<storage, read> dynamic_slices: array<u32>;
@group(0) @binding(3) var<storage, read> dynamic_entries: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read> dynamic_weights: array<f32>;
@group(0) @binding(5) var<storage, read> joints: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> influences: array<vec2<u32>>;
@group(0) @binding(7) var<storage, read_write> positions: array<f32>;

fn skin_linear(p: vec3<f32>, first: u32) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    var total = 0.0;
    for (var k = 0u; k < params.influences; k += 1u) {
        let inf = influences[first + k];
        let j = inf.x & 0xffffu;
        let w = bitcast<f32>(inf.y);
        if (w == 0.0 || j >= params.joint_count) {
            continue;
        }
        let c0 = joints[5u * j];
        let c1 = joints[5u * j + 1u];
        let c2 = joints[5u * j + 2u];
        let moved = c0.xyz * p.x + c1.xyz * p.y + c2.xyz * p.z + vec3<f32>(c0.w, c1.w, c2.w);
        sum += moved * w;
        total += w;
    }
    if (total > 0.0) {
        return sum * (1.0 / total);
    }
    return p;
}

fn skin_dual_quat(p: vec3<f32>, first: u32) -> vec3<f32> {
    var real = vec4<f32>(0.0);
    var dual = vec4<f32>(0.0);
    var pivot = vec4<f32>(0.0);
    var found = false;
    for (var k = 0u; k < params.influences; k += 1u) {
        let inf = influences[first + k];
        let j = inf.x & 0xffffu;
        var w = bitcast<f32>(inf.y);
        if (w == 0.0 || j >= params.joint_count) {
            continue;
        }
        let r = joints[5u * j + 3u];
        let d = joints[5u * j + 4u];
        if (!found) {
            pivot = r;
            found = true;
        }
        if (dot(pivot, r) < 0.0) {
            w = -w;
        }
        real += r * w;
        dual += d * w;
    }
    if (!found) {
        return p;
    }
    // Normalise: unit real part, dual part orthogonal to it (humanart_math::DualQuat::normalize).
    let len = length(real);
    if (!(len > 0.0) || len > F32_MAX) {
        return p;
    }
    let inv = 1.0 / len;
    let r = real * inv;
    var d = dual * inv;
    d = d - r * dot(r, d);
    // translation = 2 (dual * conjugate(real)).xyz
    let t = quat_mul(d, vec4<f32>(-r.xyz, r.w)).xyz * 2.0;
    return quat_rotate(r, p) + t;
}

@compute @workgroup_size(64)
fn skin(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let v = thread_index(group, groups, local);
    if (v >= params.vertex_count) {
        return;
    }
    let base = 3u * v;
    var p = vec3<f32>(rest[base], rest[base + 1u], rest[base + 2u]);
    let slice = v / WORKGROUP;
    let end = dynamic_slices[slice + 1u];
    for (var i = dynamic_slices[slice] + v % WORKGROUP; i < end; i += WORKGROUP) {
        let e = dynamic_entries[i];
        let w = dynamic_weights[e.x & 0xffffu];
        if (w != 0.0) {
            p += decode_delta(e.y, e.x >> 16u) * w;
        }
    }
    if (params.joint_count > 0u && params.influences > 0u) {
        let first = v * params.influences;
        let flagged = (influences[first].x & DQ_FLAG) != 0u;
        let dq = params.mode == 2u || (params.mode == MODE_BY_REGION && flagged);
        if (dq) {
            p = skin_dual_quat(p, first);
        } else {
            p = skin_linear(p, first);
        }
    }
    positions[base] = p.x;
    positions[base + 1u] = p.y;
    positions[base + 2u] = p.z;
}
