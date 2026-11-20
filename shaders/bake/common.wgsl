// bake/common.wgsl: shared by the ray-traced bakers of humanart_tools::bake::gpu [BAKE-01] [BAKE-02].
//
// - Ray directions: the twin of humanart_tools::bake::sampling (the same PCG hash of the texel and
//   seed, rotated Hammersley points, cosine-hemisphere and uniform-cone warps, the Duff et al. frame).
// - The surface point of a texel from its G-buffer triangle and barycentrics: the twin of
//   humanart_tools::bake::BakeMesh::point.
// - The bounding-volume hierarchy in the word layout of Bvh::gpu_nodes and Bvh::gpu_triangles,
//   traversed exactly as Bvh::intersect does: nearer child first, a 64-entry stack,
//   Moller-Trumbore with inclusive edges, zero direction components replaced by 1e-30.
//
// The including kernel declares the mesh and hierarchy bindings these functions read (a library
// declares no bindings): `positions` (array<vec4<f32>>, xyz + the region id bits in w),
// `normals` (array<vec4<f32>>), `tangents` (array<vec4<f32>>, w = handedness), `indices`
// (array<u32>, three per triangle), `nodes` (array<vec4<u32>>, two per node: min.xyz bits and
// first, max.xyz bits and count) and `tris` (array<vec4<f32>>, three corners per triangle, the
// first w the original triangle index bits).

const BAKE_TAU: f32 = 6.28318530717958647692;
const BAKE_STACK: u32 = 64u;

// O'Neill's pcg_hash.
fn bake_pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn bake_texel_hash(x: u32, y: u32, seed: u32) -> u32 {
    return bake_pcg(x ^ bake_pcg(y ^ bake_pcg(seed)));
}

fn bake_rotation(h: u32) -> vec2<f32> {
    return vec2<f32>(f32(h & 0xFFFFu) / 65536.0, f32(h >> 16u) / 65536.0);
}

// The 2D point of ray i of n with rotation r.
fn bake_point(i: u32, n: u32, r: vec2<f32>) -> vec2<f32> {
    let u1 = fract((f32(i) + 0.5) / f32(max(n, 1u)) + r.x);
    let u2 = fract(f32(reverseBits(i) >> 8u) * (1.0 / 16777216.0) + r.y);
    return vec2<f32>(u1, u2);
}

fn bake_cosine_hemisphere(u: vec2<f32>) -> vec3<f32> {
    let r = sqrt(u.x);
    let phi = BAKE_TAU * u.y;
    return vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(1.0 - u.x, 0.0)));
}

fn bake_uniform_cone(u: vec2<f32>, cos_max: f32) -> vec3<f32> {
    let cos_theta = 1.0 - u.x * (1.0 - cos_max);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let phi = BAKE_TAU * u.y;
    return vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

// The first vector of the Duff et al. frame around the unit vector n (Vec3::any_orthonormal_pair).
fn bake_frame_t(n: vec3<f32>) -> vec3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
}

fn bake_to_world(l: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bt = vec3<f32>(b, s + n.y * n.y * a, -n.y);
    return t * l.x + bt * l.y + n * l.z;
}

// d mirrored to the side of the plane with normal g that `toward` (its sign) names.
fn bake_keep_side(d: vec3<f32>, g: vec3<f32>, toward: f32) -> vec3<f32> {
    let s = dot(d, g);
    return select(d, d - g * (2.0 * s), s * toward < 0.0);
}

// normalize(v), or `fallback` when v has no direction.
fn bake_normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    return select(fallback, v / len, len > 0.0 && len < 3.0e38);
}

struct BakeSurface {
    position: vec3<f32>,
    normal: vec3<f32>,
    geometric: vec3<f32>,
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
    region: u32,
}

// The surface point of triangle t at barycentrics b = (b1, b2).
fn bake_surface(t: u32, b: vec2<f32>) -> BakeSurface {
    let ia = indices[3u * t];
    let ib = indices[3u * t + 1u];
    let ic = indices[3u * t + 2u];
    let b0 = 1.0 - b.x - b.y;
    let pa = positions[ia];
    let pb = positions[ib];
    let pc = positions[ic];
    var s: BakeSurface;
    s.position = pa.xyz * b0 + pb.xyz * b.x + pc.xyz * b.y;
    let face = bake_normalize_or(cross(pb.xyz - pa.xyz, pc.xyz - pa.xyz), vec3<f32>(0.0));
    let n = normals[ia].xyz * b0 + normals[ib].xyz * b.x + normals[ic].xyz * b.y;
    s.normal = bake_normalize_or(n, bake_normalize_or(face, vec3<f32>(0.0, 0.0, 1.0)));
    s.geometric = select(s.normal, face, any(face != vec3<f32>(0.0)));
    let t4 = tangents[ia] * b0 + tangents[ib] * b.x + tangents[ic] * b.y;
    let ortho = t4.xyz - s.normal * dot(s.normal, t4.xyz);
    s.tangent = bake_normalize_or(ortho, bake_frame_t(s.normal));
    let sign = select(1.0, -1.0, t4.w < 0.0);
    s.bitangent = cross(s.normal, s.tangent) * sign;
    var corner = ic;
    if (b0 >= b.x && b0 >= b.y) {
        corner = ia;
    } else if (b.x >= b.y) {
        corner = ib;
    }
    s.region = bitcast<u32>(positions[corner].w);
    return s;
}

// (v . t, v . b, v . n) in the frame of s.
fn bake_to_tangent(s: BakeSurface, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(v, s.tangent), dot(v, s.bitangent), dot(v, s.normal));
}

struct BakeHit {
    t: f32,
    triangle: u32,
    hit: bool,
}

// The entry distance into the box [lo, hi] if the ray meets it within [t_min, t_max], else -1.
fn bake_enter(lo: vec3<f32>, hi: vec3<f32>, o: vec3<f32>, inv: vec3<f32>, t_min: f32, t_max: f32) -> f32 {
    let a = (lo - o) * inv;
    let b = (hi - o) * inv;
    let lo_t = min(a, b);
    let hi_t = max(a, b);
    let near = max(max(max(lo_t.x, lo_t.y), lo_t.z), t_min);
    let far = min(min(min(hi_t.x, hi_t.y), hi_t.z), t_max);
    return select(-1.0, near, near <= far);
}

// The closest hit of o + t d with t in (t_min, t_max).
fn bake_intersect(o: vec3<f32>, d: vec3<f32>, t_min: f32, t_max_in: f32) -> BakeHit {
    var best: BakeHit;
    best.hit = false;
    best.t = t_max_in;
    best.triangle = 0u;
    var t_max = t_max_in;
    let safe = vec3<f32>(select(d.x, 1e-30, d.x == 0.0), select(d.y, 1e-30, d.y == 0.0), select(d.z, 1e-30, d.z == 0.0));
    let inv = 1.0 / safe;
    let node_count = arrayLength(&nodes) / 2u;
    let tri_count = arrayLength(&tris) / 3u;
    var stack: array<u32, 64>;
    stack[0] = 0u;
    var sp = 1u;
    // A tree visits each node at most once; the cap only guards against corrupt data.
    var steps = 0u;
    loop {
        if (sp == 0u || steps > 2u * node_count + 2u) {
            break;
        }
        steps += 1u;
        sp -= 1u;
        let ni = stack[sp];
        if (ni >= node_count) {
            continue;
        }
        let n0 = nodes[2u * ni];
        let n1 = nodes[2u * ni + 1u];
        if (bake_enter(bitcast<vec3<f32>>(n0.xyz), bitcast<vec3<f32>>(n1.xyz), o, inv, t_min, t_max) < 0.0) {
            continue;
        }
        let first = n0.w;
        let count = n1.w;
        if (count > 0u) {
            for (var k = first; k < first + count && k < tri_count; k += 1u) {
                let v0 = tris[3u * k];
                let v1 = tris[3u * k + 1u].xyz;
                let v2 = tris[3u * k + 2u].xyz;
                let e1 = v1 - v0.xyz;
                let e2 = v2 - v0.xyz;
                let p = cross(d, e2);
                let det = dot(e1, p);
                if (det == 0.0 || abs(det) > 3.0e38) {
                    continue;
                }
                let inv_det = 1.0 / det;
                let s = o - v0.xyz;
                let u = dot(s, p) * inv_det;
                if (u < 0.0 || u > 1.0) {
                    continue;
                }
                let q = cross(s, e1);
                let v = dot(d, q) * inv_det;
                if (v < 0.0 || u + v > 1.0) {
                    continue;
                }
                let t = dot(e2, q) * inv_det;
                if (t > t_min && t < t_max) {
                    t_max = t;
                    best.t = t;
                    best.triangle = bitcast<u32>(v0.w);
                    best.hit = true;
                }
            }
        } else if (sp + 2u <= BAKE_STACK) {
            let l = first;
            let r = first + 1u;
            var near_l = -1.0;
            var near_r = -1.0;
            if (l < node_count) {
                let a0 = nodes[2u * l];
                let a1 = nodes[2u * l + 1u];
                near_l = bake_enter(bitcast<vec3<f32>>(a0.xyz), bitcast<vec3<f32>>(a1.xyz), o, inv, t_min, t_max);
            }
            if (r < node_count) {
                let c0 = nodes[2u * r];
                let c1 = nodes[2u * r + 1u];
                near_r = bake_enter(bitcast<vec3<f32>>(c0.xyz), bitcast<vec3<f32>>(c1.xyz), o, inv, t_min, t_max);
            }
            if (near_l >= 0.0 && near_r >= 0.0) {
                if (near_l <= near_r) {
                    stack[sp] = r;
                    stack[sp + 1u] = l;
                } else {
                    stack[sp] = l;
                    stack[sp + 1u] = r;
                }
                sp += 2u;
            } else if (near_l >= 0.0) {
                stack[sp] = l;
                sp += 1u;
            } else if (near_r >= 0.0) {
                stack[sp] = r;
                sp += 1u;
            }
        }
    }
    return best;
}
