// lib/math.wgsl: constants and small helpers every shader shares.
//
// Conventions (CLAUDE.md): right-handed, Y up, column-major matrices (`M * p`), quaternions stored
// as vec4 (x, y, z, w) with the Hamilton product, matching humanart_math::Quat.
// Derivations: docs/shader-notes.md, "Shared math".

const PI: f32 = 3.14159265358979323846;
const TAU: f32 = 6.28318530717958647692;
const HALF_PI: f32 = 1.57079632679489661923;
const INV_PI: f32 = 0.31830988618379067154;
const INV_TAU: f32 = 0.15915494309189533577;
const F32_MAX: f32 = 0x1.fffffep+127f;

// x * x.
fn sq(x: f32) -> f32 {
    return x * x;
}

// The largest component.
fn max3(v: vec3<f32>) -> f32 {
    return max(v.x, max(v.y, v.z));
}

// The smallest component.
fn min3(v: vec3<f32>) -> f32 {
    return min(v.x, min(v.y, v.z));
}

// `(x - lo) / (hi - lo)` clamped to [0, 1]; 0 when the range is empty.
fn remap01(x: f32, lo: f32, hi: f32) -> f32 {
    let range = hi - lo;
    if (abs(range) <= 0.0) {
        return 0.0;
    }
    return saturate((x - lo) / range);
}

// `v` normalised, or `fallback` when `v` has no usable length.
fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len2 = dot(v, v);
    if (len2 > 1e-30 && len2 <= F32_MAX) {
        return v * inverseSqrt(len2);
    }
    return fallback;
}

// An orthonormal basis around the unit vector `n`: the columns are (tangent, bitangent, n), right-
// handed (`cross(t, b) = n`). Duff et al., "Building an Orthonormal Basis, Revisited" (JCGT 2017):
// branch-free and continuous except across the n.z = 0 sign flip.
fn orthonormal_basis(n: vec3<f32>) -> mat3x3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bt = vec3<f32>(b, s + n.y * n.y * a, -n.y);
    return mat3x3<f32>(t, bt, n);
}

// `v` (world) in the frame of `basis` (columns t, b, n).
fn to_local(basis: mat3x3<f32>, v: vec3<f32>) -> vec3<f32> {
    return v * basis;
}

// `v` (in the frame of `basis`) in world space.
fn from_local(basis: mat3x3<f32>, v: vec3<f32>) -> vec3<f32> {
    return basis * v;
}

// The Hamilton product `a * b` (apply b first, then a).
fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
        a.w * b.w - dot(a.xyz, b.xyz),
    );
}

// `v` rotated by the unit quaternion `q`: `v + 2w (u x v) + 2 u x (u x v)`, the form of
// humanart_math::Quat::mul_vec3.
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}
