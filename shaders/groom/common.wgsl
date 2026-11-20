// groom/common.wgsl: shared pieces of the hair generation modifiers [HAIR-01] [HAIR-02].
//
// Every function mirrors a CPU reference in humanart_model::hair operation for operation:
// groom_hash = hash_u32 (the PCG output hash, Jarzynski and Olano, JCGT 2020), groom_rand = hash01,
// groom_normalize_or = surface::normalize_or, groom_fallback_tangent = surface::fallback_tangent,
// groom_root_frame = surface::root_frame, groom_wrap_angle = generate::wrap_angle.

const GROOM_PI: f32 = 3.14159265358979;
const GROOM_TAU: f32 = 6.28318530717959;

struct RootFrame {
    origin: vec3<f32>,
    t: vec3<f32>,
    b: vec3<f32>,
    n: vec3<f32>,
}

fn groom_hash(x: u32) -> u32 {
    let state = x * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// A uniform value in [0, 1) from a strand key and a salt, exact in f32.
fn groom_rand(key: u32, salt: u32) -> f32 {
    return f32(groom_hash(key ^ groom_hash(salt)) >> 8u) * (1.0 / 16777216.0);
}

fn groom_normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    if (l2 > 1e-30) {
        return v * (1.0 / sqrt(l2));
    }
    return fallback;
}

fn groom_fallback_tangent(n: vec3<f32>) -> vec3<f32> {
    var a = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(n.x) < 0.9) {
        a = vec3<f32>(1.0, 0.0, 0.0);
    }
    return groom_normalize_or(a - n * dot(n, a), vec3<f32>(0.0, 0.0, 1.0));
}

// The root frame at barycentric (u, v) of a triangle with corners p0..p2, vertex normals n0..n2
// and tangent coefficients coef (t = e1 coef.x + e2 coef.y).
fn groom_root_frame(
    p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>,
    n0: vec3<f32>, n1: vec3<f32>, n2: vec3<f32>,
    bary: vec2<f32>, coef: vec2<f32>,
) -> RootFrame {
    let u = bary.x;
    let v = bary.y;
    let w = 1.0 - u - v;
    let origin = p0 * w + p1 * u + p2 * v;
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let face = groom_normalize_or(cross(e1, e2), vec3<f32>(0.0, 0.0, 1.0));
    let normal = groom_normalize_or(n0 * w + n1 * u + n2 * v, face);
    let t0 = e1 * coef.x + e2 * coef.y;
    let t1 = t0 - normal * dot(normal, t0);
    var t = groom_fallback_tangent(normal);
    if (dot(t1, t1) > 1e-20) {
        t = groom_normalize_or(t1, vec3<f32>(1.0, 0.0, 0.0));
    }
    return RootFrame(origin, t, cross(normal, t), normal);
}

fn groom_to_world(f: RootFrame, local: vec3<f32>) -> vec3<f32> {
    return f.origin + f.t * local.x + f.b * local.y + f.n * local.z;
}

fn groom_to_local(f: RootFrame, world: vec3<f32>) -> vec3<f32> {
    let d = world - f.origin;
    return vec3<f32>(dot(d, f.t), dot(d, f.b), dot(d, f.n));
}

// Wraps an angle to [-pi, pi).
fn groom_wrap_angle(a: f32) -> f32 {
    return a - GROOM_TAU * floor((a + GROOM_PI) / GROOM_TAU);
}

// The linear invocation index of a 2D dispatch of 64-wide workgroups, `groups_x` along x.
fn groom_thread(gid: vec3<u32>, groups_x: u32) -> u32 {
    return gid.x + gid.y * groups_x * 64u;
}
