// deform/normals.wgsl: vertex normals and tangents of the deformed surface (spec 6.2 step 5)
// [EVAL-04] [EVAL-05].
//
// Two passes:
//
// 1. `triangles`, one invocation per triangle (a, b, c): with e1 = b - a, e2 = c - a, the unit
//    normal normalize(e1 x e2), the interior angle at each corner, and the unit tangent and
//    bitangent from the UV derivatives (s = sign of the UV determinant):
//    T = normalize(s (e1 dv2 - e2 dv1)), B = normalize(s (e2 du1 - e1 du2)). Degenerate
//    triangles get zero vectors. The frame of triangle t is three vec4: (N, angle at corner 0),
//    (T, angle at corner 1), (B, angle at corner 2), so the angle of corner c = 3t + k is
//    `frames[c].w`.
// 2. `vertices`, one invocation per vertex, angle-weighted gathers over precomputed CSR rows:
//    the normal over the corners of every vertex of its normal class (UV-seam duplicates share it,
//    hard-edge duplicates do not), normalised (+Z when the sum vanishes); the tangent and
//    bitangent over the vertex's own corners, the tangent made orthogonal to the normal
//    (Gram-Schmidt) and normalised (any perpendicular of the normal when it vanishes), w = -1 when
//    (n x t) . b < 0 else 1. The CSR rows live in one offsets array (normal rows 0..=V, own-corner
//    rows V+1..=2V+1) and one corner array.
//
// The CPU reference is humanart_model::eval::normals.

#include "deform/common.wgsl"
#include "lib/math.wgsl"

struct Params {
    vertex_count: u32,
    triangle_count: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> uvs: array<f32>;
@group(0) @binding(3) var<storage, read> triangles: array<u32>;
@group(0) @binding(4) var<storage, read_write> frames: array<vec4<f32>>;
// vertices
@group(0) @binding(5) var<storage, read> frames_in: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> gather_offsets: array<u32>;
@group(0) @binding(7) var<storage, read> gather_corners: array<u32>;
@group(0) @binding(8) var<storage, read_write> normals: array<f32>;
@group(0) @binding(9) var<storage, read_write> tangents: array<vec4<f32>>;

fn position(v: u32) -> vec3<f32> {
    return vec3<f32>(positions[3u * v], positions[3u * v + 1u], positions[3u * v + 2u]);
}

fn uv(v: u32) -> vec2<f32> {
    return vec2<f32>(uvs[2u * v], uvs[2u * v + 1u]);
}

// normalize(v), or `fallback` when the length is zero or not finite (humanart_math's
// `normalize_or`).
fn normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    if (len > 0.0 && len <= F32_MAX) {
        return v / len;
    }
    return fallback;
}

// The angle between u and v (0 when either vanishes), as humanart_math::Vec3::angle_between.
fn angle_between(u: vec3<f32>, v: vec3<f32>) -> f32 {
    let d = sqrt(dot(u, u) * dot(v, v));
    if (d > 0.0) {
        let a = acos(clamp(dot(u, v) / d, -1.0, 1.0));
        if (a <= F32_MAX) {
            return a;
        }
    }
    return 0.0;
}

// A unit vector perpendicular to the unit vector n (humanart_math::Vec3::any_orthonormal_vector;
// the sign follows the sign bit of n.z, so -0 counts as negative there too).
fn any_orthonormal(n: vec3<f32>) -> vec3<f32> {
    let s = select(1.0, -1.0, (bitcast<u32>(n.z) & 0x80000000u) != 0u);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
}

@compute @workgroup_size(64)
fn triangle_frames(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let t = thread_index(group, groups, local);
    if (t >= params.triangle_count) {
        return;
    }
    let ia = triangles[3u * t];
    let ib = triangles[3u * t + 1u];
    let ic = triangles[3u * t + 2u];
    if (ia >= params.vertex_count || ib >= params.vertex_count || ic >= params.vertex_count) {
        frames[3u * t] = vec4<f32>(0.0);
        frames[3u * t + 1u] = vec4<f32>(0.0);
        frames[3u * t + 2u] = vec4<f32>(0.0);
        return;
    }
    let pa = position(ia);
    let pb = position(ib);
    let pc = position(ic);
    let e1 = pb - pa;
    let e2 = pc - pa;
    let e3 = pc - pb;
    let n = normalize_or(cross(e1, e2), vec3<f32>(0.0));
    let a0 = angle_between(e1, e2);
    let a1 = angle_between(-e1, e3);
    let a2 = angle_between(-e2, -e3);
    let ua = uv(ia);
    let d1 = uv(ib) - ua;
    let d2 = uv(ic) - ua;
    let s = sign(d1.x * d2.y - d2.x * d1.y);
    let tangent = normalize_or((e1 * d2.y - e2 * d1.y) * s, vec3<f32>(0.0));
    let bitangent = normalize_or((e2 * d1.x - e1 * d2.x) * s, vec3<f32>(0.0));
    frames[3u * t] = vec4<f32>(n, a0);
    frames[3u * t + 1u] = vec4<f32>(tangent, a1);
    frames[3u * t + 2u] = vec4<f32>(bitangent, a2);
}

@compute @workgroup_size(64)
fn vertex_frames(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let v = thread_index(group, groups, local);
    if (v >= params.vertex_count) {
        return;
    }
    var sum = vec3<f32>(0.0);
    let end = gather_offsets[v + 1u];
    for (var i = gather_offsets[v]; i < end; i += 1u) {
        let c = gather_corners[i];
        sum += frames_in[c - c % 3u].xyz * frames_in[c].w;
    }
    let n = normalize_or(sum, vec3<f32>(0.0, 0.0, 1.0));
    var t = vec3<f32>(0.0);
    var b = vec3<f32>(0.0);
    let own = params.vertex_count + 1u;
    let own_end = gather_offsets[own + v + 1u];
    for (var i = gather_offsets[own + v]; i < own_end; i += 1u) {
        let c = gather_corners[i];
        let a = frames_in[c].w;
        let first = c - c % 3u;
        t += frames_in[first + 1u].xyz * a;
        b += frames_in[first + 2u].xyz * a;
    }
    let ortho = t - n * dot(n, t);
    let len = length(ortho);
    var dir: vec3<f32>;
    if (len > 0.0 && len <= F32_MAX && len > 1e-6 * length(t)) {
        dir = ortho / len;
    } else {
        dir = any_orthonormal(n);
    }
    let w = select(1.0, -1.0, dot(cross(n, dir), b) < 0.0);
    normals[3u * v] = n.x;
    normals[3u * v + 1u] = n.y;
    normals[3u * v + 2u] = n.z;
    tangents[v] = vec4<f32>(dir, w);
}
