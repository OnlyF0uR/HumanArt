// groom/generate.wgsl: the hair generation modifiers on the GPU [HAIR-02].
//
// Three entry points, the three stages of humanart_model::hair::generate (the CPU reference; the
// code follows it operation for operation, see that module for what each step means):
//
// - `shape`, one invocation per strand: blend the strand's guides in its root frame, set the
//   length by resampling along the curve, turn it about the root normal (direction map, parting,
//   cowlicks, noise), add curl, wave, frizz and flyaways, write world points to `shape`.
// - `finish`, one invocation per strand, after `shape` for every strand: clump towards the clump
//   centre's shape, push points out of the head through the signed distance grid, store the
//   offsets in the rest root frame as f16 (`stored`) and the per-strand attributes (`attrs`).
// - `attach`, one invocation per control point, every frame the skin moves: the stored offsets in
//   the deformed root frames, written to `world` as (position, diameter).
//
// Strand records (`strands`, six vec4<u32> per strand): (v0, v1, v2, clump centre),
// (bary.u, bary.v, tangent coef 0, tangent coef 1) as f32 bits, (guide 0..3), (weight 0..3) as f32
// bits, (length, clump, curl, frizz map factors) as f32 bits, (direction x, direction y, colour
// offset as f32 bits, key). Positions and normals are packed f32 xyz (the deformation outputs).

#include "groom/common.wgsl"

const MAX_POINTS: u32 = 32u;

struct Params {
    // strand count, points per strand, guide count, workgroups along x of the per-strand dispatch.
    counts: vec4<u32>,
    // workgroups along x of the per-point dispatch, has a distance grid, unused, unused.
    flags: vec4<u32>,
    // length scale, length variation, direction strength, noise.
    shape: vec4<f32>,
    // curl radius, curl frequency, curl variation, wave amplitude.
    curl: vec4<f32>,
    // wave frequency, frizz, flyaway fraction, flyaway strength.
    wave: vec4<f32>,
    // clump strength, clump spread, clump tip spread, collision offset.
    clump: vec4<f32>,
    // thickness, tip ratio, thickness variation, grey fraction.
    thickness: vec4<f32>,
    // stiffness, unused x3.
    misc: vec4<f32>,
    // parting point xyz, width.
    parting_point: vec4<f32>,
    // parting unit normal xyz, strength.
    parting_normal: vec4<f32>,
    // per cowlick: (centre xyz, radius), (strength, 0, 0, 0).
    cowlicks: array<vec4<f32>, 8>,
    // distance grid origin xyz, cell size.
    sdf_origin: vec4<f32>,
    // distance grid nodes per axis.
    sdf_dims: vec4<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> strands: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> curves: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> rest_positions: array<f32>;
@group(0) @binding(4) var<storage, read> rest_normals: array<f32>;
@group(0) @binding(5) var<storage, read_write> shape_points: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> sdf: array<f32>;
@group(0) @binding(7) var<storage, read_write> stored: array<vec2<u32>>;
@group(0) @binding(8) var<storage, read_write> attrs: array<vec4<u32>>;
@group(0) @binding(9) var<storage, read> positions: array<f32>;
@group(0) @binding(10) var<storage, read> normals: array<f32>;
@group(0) @binding(11) var<storage, read_write> world: array<vec4<f32>>;

fn rest_position(v: u32) -> vec3<f32> {
    return vec3<f32>(rest_positions[3u * v], rest_positions[3u * v + 1u], rest_positions[3u * v + 2u]);
}

fn rest_normal(v: u32) -> vec3<f32> {
    return vec3<f32>(rest_normals[3u * v], rest_normals[3u * v + 1u], rest_normals[3u * v + 2u]);
}

fn rest_frame(i: u32) -> RootFrame {
    let r0 = strands[6u * i];
    let r1 = bitcast<vec4<f32>>(strands[6u * i + 1u]);
    return groom_root_frame(
        rest_position(r0.x), rest_position(r0.y), rest_position(r0.z),
        rest_normal(r0.x), rest_normal(r0.y), rest_normal(r0.z),
        r1.xy, r1.zw,
    );
}

fn deformed_frame(i: u32) -> RootFrame {
    let r0 = strands[6u * i];
    let r1 = bitcast<vec4<f32>>(strands[6u * i + 1u]);
    let p0 = vec3<f32>(positions[3u * r0.x], positions[3u * r0.x + 1u], positions[3u * r0.x + 2u]);
    let p1 = vec3<f32>(positions[3u * r0.y], positions[3u * r0.y + 1u], positions[3u * r0.y + 2u]);
    let p2 = vec3<f32>(positions[3u * r0.z], positions[3u * r0.z + 1u], positions[3u * r0.z + 2u]);
    let n0 = vec3<f32>(normals[3u * r0.x], normals[3u * r0.x + 1u], normals[3u * r0.x + 2u]);
    let n1 = vec3<f32>(normals[3u * r0.y], normals[3u * r0.y + 1u], normals[3u * r0.y + 2u]);
    let n2 = vec3<f32>(normals[3u * r0.z], normals[3u * r0.z + 1u], normals[3u * r0.z + 2u]);
    return groom_root_frame(p0, p1, p2, n0, n1, n2, r1.xy, r1.zw);
}

// The turn towards the local direction (lx, ly) from `heading`, weighted.
fn turn(lx: f32, ly: f32, heading: f32, weight: f32) -> f32 {
    return groom_wrap_angle(atan2(ly, lx) - heading) * weight;
}

@compute @workgroup_size(64)
fn shape(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = groom_thread(gid, params.counts.w);
    let count = params.counts.x;
    let n = params.counts.y;
    if (i >= count || n < 2u || n > MAX_POINTS) {
        return;
    }
    let guide_count = max(params.counts.z, 1u);
    let frame = rest_frame(i);
    let guides = strands[6u * i + 2u];
    let weights = bitcast<vec4<f32>>(strands[6u * i + 3u]);
    let factors = bitcast<vec4<f32>>(strands[6u * i + 4u]);
    let r5 = strands[6u * i + 5u];
    let direction = bitcast<vec2<f32>>(r5.xy);
    let key = r5.w;
    var base: array<vec3<f32>, 32>;
    var res: array<vec3<f32>, 32>;
    // 1. Blend the guides.
    for (var k = 0u; k < n; k++) {
        var acc = vec3<f32>(0.0);
        for (var j = 0u; j < 4u; j++) {
            let g = min(guides[j], guide_count - 1u);
            acc += curves[g * n + k].xyz * weights[j];
        }
        base[k] = acc;
    }
    // 2. Length.
    var lb = 0.0;
    for (var k = 1u; k < n; k++) {
        lb += length(base[k] - base[k - 1u]);
    }
    let variation = 1.0 + params.shape.y * (2.0 * groom_rand(key, 1u) - 1.0);
    let total = max(lb * params.shape.x * factors.x * variation, 0.0);
    var seg = 0u;
    var acc_len = 0.0;
    for (var k = 0u; k < n; k++) {
        let s = total * (f32(k) / f32(n - 1u));
        while (seg < n - 2u && acc_len + length(base[seg + 1u] - base[seg]) < s) {
            acc_len += length(base[seg + 1u] - base[seg]);
            seg += 1u;
        }
        let d = base[seg + 1u] - base[seg];
        let len = length(d);
        var t = 0.0;
        if (len > 1e-12) {
            t = (s - acc_len) / len;
        }
        res[k] = base[seg] + d * t;
    }
    // 3. Turn about the root normal.
    let tip = res[n - 1u];
    let heading = atan2(tip.y, tip.x);
    let horizontal = sqrt(tip.x * tip.x + tip.y * tip.y);
    var delta = 0.0;
    if (horizontal > 1e-6) {
        let m = sqrt(direction.x * direction.x + direction.y * direction.y);
        if (m > 1e-6) {
            delta += turn(direction.x, direction.y, heading, min(m, 1.0) * params.shape.z);
        }
        let part_width = params.parting_point.w;
        let part_strength = params.parting_normal.w;
        if (part_strength > 0.0 && part_width > 0.0) {
            let pn = params.parting_normal.xyz;
            let d = dot(frame.origin - params.parting_point.xyz, pn);
            let ad = abs(d);
            if (ad < part_width) {
                var side = -1.0;
                if (d >= 0.0) {
                    side = 1.0;
                }
                let lx = dot(pn, frame.t) * side;
                let ly = dot(pn, frame.b) * side;
                if (lx * lx + ly * ly > 1e-8) {
                    let f = 1.0 - ad / part_width;
                    delta += turn(lx, ly, heading, part_strength * f * f);
                }
            }
        }
        for (var c = 0u; c < 4u; c++) {
            let cw = params.cowlicks[2u * c];
            let strength = params.cowlicks[2u * c + 1u].x;
            if (cw.w <= 0.0) {
                continue;
            }
            let rel = frame.origin - cw.xyz;
            let dist = length(rel);
            if (dist < cw.w) {
                var sense = 1.0;
                if (strength < 0.0) {
                    sense = -1.0;
                }
                let swirl = cross(frame.n, rel) * sense;
                let lx = dot(swirl, frame.t);
                let ly = dot(swirl, frame.b);
                if (lx * lx + ly * ly > 1e-12) {
                    let f = 1.0 - dist / cw.w;
                    delta += turn(lx, ly, heading, abs(strength) * f * f);
                }
            }
        }
    }
    delta += params.shape.w * (2.0 * groom_rand(key, 2u) - 1.0);
    let sn = sin(delta);
    let cs = cos(delta);
    // 4. Curl, wave, frizz, flyaways.
    let a = groom_normalize_or(tip, vec3<f32>(0.0, 0.0, 1.0));
    let b1 = groom_normalize_or(cross(a, vec3<f32>(0.0, 0.0, 1.0)), vec3<f32>(1.0, 0.0, 0.0));
    let b2 = cross(a, b1);
    let radius = params.curl.x * factors.z * (1.0 + params.curl.z * (2.0 * groom_rand(key, 3u) - 1.0));
    let frequency = params.curl.y * (1.0 + params.curl.z * (2.0 * groom_rand(key, 4u) - 1.0));
    let phase = groom_rand(key, 5u);
    let flyaway = groom_rand(key, 6u) < params.wave.z;
    let fly = vec3<f32>(2.0 * groom_rand(key, 7u) - 1.0, 2.0 * groom_rand(key, 8u) - 1.0, 1.0);
    let frizz = params.wave.y * factors.w;
    for (var k = 0u; k < n; k++) {
        let t = f32(k) / f32(n - 1u);
        let s = total * t;
        let ramp = min(t * 4.0, 1.0);
        var q = res[k];
        let x = fract(frequency * s + phase) * GROOM_TAU;
        q += (b1 * cos(x) + b2 * sin(x)) * (radius * ramp);
        let w = fract(params.wave.x * s + phase) * GROOM_TAU;
        q += b1 * (params.curl.w * ramp * sin(w));
        if (k > 0u) {
            let salt = 16u + 3u * k;
            let r = vec3<f32>(
                2.0 * groom_rand(key, salt) - 1.0,
                2.0 * groom_rand(key, salt + 1u) - 1.0,
                2.0 * groom_rand(key, salt + 2u) - 1.0,
            );
            q += r * (frizz * t * t);
            if (flyaway) {
                q += fly * (params.wave.w * t * t);
            }
        }
        let rotated = vec3<f32>(q.x * cs - q.y * sn, q.x * sn + q.y * cs, q.z);
        shape_points[i * n + k] = vec4<f32>(groom_to_world(frame, rotated), 0.0);
    }
}

fn sdf_node(x: u32, y: u32, z: u32) -> f32 {
    let d = params.sdf_dims;
    return sdf[x + d.x * (y + d.y * z)];
}

// The trilinear signed distance at p and its gradient (humanart_model::hair::SdfGrid::sample).
fn sdf_sample(p: vec3<f32>) -> vec4<f32> {
    let dims = params.sdf_dims;
    let cell = params.sdf_origin.w;
    let hi = vec3<f32>(f32(dims.x - 1u), f32(dims.y - 1u), f32(dims.z - 1u));
    let q = clamp((p - params.sdf_origin.xyz) * (1.0 / cell), vec3<f32>(0.0), hi);
    let cx = min(u32(floor(q.x)), dims.x - 2u);
    let cy = min(u32(floor(q.y)), dims.y - 2u);
    let cz = min(u32(floor(q.z)), dims.z - 2u);
    let f = q - vec3<f32>(f32(cx), f32(cy), f32(cz));
    let c000 = sdf_node(cx, cy, cz);
    let c100 = sdf_node(cx + 1u, cy, cz);
    let c010 = sdf_node(cx, cy + 1u, cz);
    let c110 = sdf_node(cx + 1u, cy + 1u, cz);
    let c001 = sdf_node(cx, cy, cz + 1u);
    let c101 = sdf_node(cx + 1u, cy, cz + 1u);
    let c011 = sdf_node(cx, cy + 1u, cz + 1u);
    let c111 = sdf_node(cx + 1u, cy + 1u, cz + 1u);
    let x00 = c000 + (c100 - c000) * f.x;
    let x10 = c010 + (c110 - c010) * f.x;
    let x01 = c001 + (c101 - c001) * f.x;
    let x11 = c011 + (c111 - c011) * f.x;
    let y0 = x00 + (x10 - x00) * f.y;
    let y1 = x01 + (x11 - x01) * f.y;
    let d = y0 + (y1 - y0) * f.z;
    let gx0 = (c100 - c000) + ((c110 - c010) - (c100 - c000)) * f.y;
    let gx1 = (c101 - c001) + ((c111 - c011) - (c101 - c001)) * f.y;
    let gx = gx0 + (gx1 - gx0) * f.z;
    let gy = (x10 - x00) + ((x11 - x01) - (x10 - x00)) * f.z;
    let gz = y1 - y0;
    return vec4<f32>(vec3<f32>(gx, gy, gz) * (1.0 / cell), d);
}

@compute @workgroup_size(64)
fn finish(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = groom_thread(gid, params.counts.w);
    let count = params.counts.x;
    let n = params.counts.y;
    if (i >= count || n < 2u || n > MAX_POINTS) {
        return;
    }
    let frame = rest_frame(i);
    let centre = min(strands[6u * i].w, count - 1u);
    let factors = bitcast<vec4<f32>>(strands[6u * i + 4u]);
    let key = strands[6u * i + 5u].w;
    let strength = params.clump.x * factors.y;
    let offset = params.clump.w;
    var prev = vec3<f32>(0.0);
    var total = 0.0;
    for (var k = 0u; k < n; k++) {
        var q = shape_points[i * n + k].xyz;
        if (centre != i && strength > 0.0) {
            let t = f32(k) / f32(n - 1u);
            let smooth_t = t * t * (3.0 - 2.0 * t);
            let keep = params.clump.y + (params.clump.z - params.clump.y) * t;
            let pull = strength * smooth_t * (1.0 - keep);
            let c = shape_points[centre * n + k].xyz;
            q += (c - q) * pull;
        }
        if (k > 0u && params.flags.y != 0u) {
            let s = sdf_sample(q);
            if (s.w < offset) {
                q += groom_normalize_or(s.xyz, frame.n) * (offset - s.w);
            }
        }
        if (k > 0u) {
            total += length(q - prev);
        }
        prev = q;
        let l = groom_to_local(frame, q);
        stored[i * n + k] = vec2<u32>(pack2x16float(l.xy), pack2x16float(vec2<f32>(l.z, 0.0)));
    }
    let width_root = params.thickness.x * (1.0 + params.thickness.z * (2.0 * groom_rand(key, 9u) - 1.0));
    let color_id = groom_hash(key ^ groom_hash(10u)) & 0xFFFFu;
    var grey = 0u;
    if (groom_rand(key, 11u) < params.thickness.w) {
        grey = 255u;
    }
    let stiffness = clamp(params.misc.x * (1.0 + 0.2 * (2.0 * groom_rand(key, 12u) - 1.0)), 0.0, 1.0);
    attrs[2u * i] = bitcast<vec4<u32>>(vec4<f32>(total, width_root, width_root * params.thickness.y, groom_rand(key, 5u)));
    attrs[2u * i + 1u] = vec4<u32>(color_id | (grey << 16u), bitcast<u32>(stiffness), centre, 0u);
}

@compute @workgroup_size(64)
fn attach(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = groom_thread(gid, params.flags.x);
    let n = params.counts.y;
    if (n < 2u || index >= params.counts.x * n) {
        return;
    }
    let i = index / n;
    let k = index - i * n;
    let frame = deformed_frame(i);
    let packed = stored[index];
    let xy = unpack2x16float(packed.x);
    let z = unpack2x16float(packed.y).x;
    let a = bitcast<vec4<f32>>(attrs[2u * i]);
    let t = f32(k) / f32(n - 1u);
    let w = a.y + (a.z - a.y) * t;
    world[index] = vec4<f32>(groom_to_world(frame, vec3<f32>(xy, z)), w);
}
