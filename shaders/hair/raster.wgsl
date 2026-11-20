// hair/raster.wgsl: the compute software rasteriser of sub-pixel strands [HR-01] [HR-05]: one
// thread per strand segment projects it with the jittered camera, clips it to the near plane and
// the viewport and walks it one pixel per step along its major axis (at most HAIR_MAX_STEPS).
// Across the segment the strand's footprint (its projected width times the step length, at least
// one pixel wide) is box-filtered over the pixels it overlaps, which gives each fragment its
// coverage (coverage-based anti-aliasing). Fragments behind the opaque scene depth are dropped.
//
// Two passes over the same fragments make an order-independent visibility buffer (no sorting):
// `raster_depth` adds every fragment's coverage to the pixel's coverage sum (fixed point, stops
// adding once the sum saturates) and keeps the nearest fragment depth with atomicMax (reversed-Z);
// `raster_records` recomputes the same fragments (bit-identical depths) and the one whose depth
// equals the pixel's nearest writes the pixel's record: tangent, groom, colour id, grey and its
// motion for TAA. `voxelize` adds every segment's fibre side area to a coarse density grid (the
// ambient occlusion and hair-on-skin occlusion of the resolve pass).
//
// Far field [HR-04]: strands whose hash is above `draw.dispatch.z` are skipped (uniform random
// pruning keeps every clump populated in proportion) and the survivors' coverage is scaled by
// `draw.scale.x`; `draw.counts.z` > 1 joins that many segments into one (sub-pixel segments).
//
// Close range [HR-04]: a segment's walk is split over `draw.scale.w` threads (adjacent lanes of
// one wave), each walking its share of the major-axis pixels, so a few 100-pixel segments do not
// keep the rest of their wave idle. Every fragment is still produced once, by one thread, with
// the same depth and coverage, so the image does not change.

#include "lib/math.wgsl"
#include "lib/packing.wgsl"
#include "render/frame.wgsl"
#include "hair/common.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> view: HairView;

@group(1) @binding(0) var<uniform> draw: HairDraw;
@group(1) @binding(1) var<storage, read> points: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read> prev_points: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read> attrs: array<vec4<u32>>;

@group(2) @binding(0) var scene_depth: texture_depth_2d;
@group(2) @binding(1) var<storage, read_write> hair_depth: array<atomic<u32>>;
@group(2) @binding(2) var<storage, read_write> hair_coverage: array<atomic<u32>>;
@group(2) @binding(3) var<storage, read_write> hair_records: array<vec4<u32>>;
@group(2) @binding(4) var<storage, read_write> hair_voxels: array<atomic<u32>>;

// A segment this thread draws: strand, first and last point index, and which of the segment's
// `parts` walk shares it takes.
struct SegmentRef {
    strand: u32,
    first: u32,
    last: u32,
    part: u32,
    parts: u32,
    ok: bool,
}

fn segment_of(gid: vec3<u32>) -> SegmentRef {
    var out: SegmentRef;
    out.ok = false;
    let per = draw.dispatch.x;
    let n = draw.counts.y;
    if (per == 0u || n < 2u) {
        return out;
    }
    let parts = clamp(u32(draw.scale.w), 1u, 8u);
    let thread = hair_thread(gid, draw.dispatch.y);
    let strand = thread / (per * parts);
    if (strand >= draw.counts.x || !hair_kept(strand, draw.dispatch.z)) {
        return out;
    }
    let lane = thread - strand * per * parts;
    let k = lane / parts;
    out.part = lane - k * parts;
    out.parts = parts;
    let stride = max(draw.counts.z, 1u);
    let first = k * stride;
    if (first >= n - 1u) {
        return out;
    }
    out.strand = strand;
    out.first = strand * n + first;
    out.last = strand * n + min(first + stride, n - 1u);
    out.ok = true;
    return out;
}

// The projected segment after clipping: screen ends (pixels), reversed-Z depths, widths
// (pixels) and the segment parameters of the ends.
struct Span {
    a: vec2<f32>,
    b: vec2<f32>,
    z: vec2<f32>,
    w: vec2<f32>,
    t: vec2<f32>,
    ok: bool,
}

fn to_screen(c: vec4<f32>) -> vec2<f32> {
    let ndc = c.xy / c.w;
    return vec2<f32>((ndc.x * 0.5 + 0.5) * view.screen.x, (0.5 - ndc.y * 0.5) * view.screen.y);
}

fn project_span(p0: vec4<f32>, p1: vec4<f32>) -> Span {
    var s: Span;
    s.ok = false;
    let c0 = frame.view_proj * vec4<f32>(p0.xyz, 1.0);
    let c1 = frame.view_proj * vec4<f32>(p1.xyz, 1.0);
    // Near plane: reversed-Z puts it at z = w; inside is w - z >= 0.
    let d0 = c0.w - c0.z;
    let d1 = c1.w - c1.z;
    if ((d0 < 0.0 && d1 < 0.0) || (c0.w <= 1e-6 && c1.w <= 1e-6)) {
        return s;
    }
    var t = vec2<f32>(0.0, 1.0);
    if (d0 < 0.0) {
        t.x = d0 / (d0 - d1);
    }
    if (d1 < 0.0) {
        t.y = d0 / (d0 - d1);
    }
    let ca = mix(c0, c1, t.x);
    let cb = mix(c0, c1, t.y);
    if (ca.w <= 1e-6 || cb.w <= 1e-6) {
        return s;
    }
    var a = to_screen(ca);
    var b = to_screen(cb);
    var z = vec2<f32>(ca.z / ca.w, cb.z / cb.w);
    let focal = view.screen.z * draw.scale.z;
    var w = vec2<f32>(mix(p0.w, p1.w, t.x) * focal / ca.w, mix(p0.w, p1.w, t.y) * focal / cb.w);
    // Clip to the viewport grown by the widest footprint (Liang-Barsky).
    let lo = vec2<f32>(-HAIR_MAX_WIDTH);
    let hi = view.screen.xy + vec2<f32>(HAIR_MAX_WIDTH);
    let d = b - a;
    var u0 = 0.0;
    var u1 = 1.0;
    for (var axis = 0; axis < 2; axis++) {
        let p = d[axis];
        let q0 = a[axis] - lo[axis];
        let q1 = hi[axis] - a[axis];
        if (abs(p) < 1e-9) {
            if (q0 < 0.0 || q1 < 0.0) {
                return s;
            }
        } else {
            let r0 = -q0 / p;
            let r1 = q1 / p;
            u0 = max(u0, min(r0, r1));
            u1 = min(u1, max(r0, r1));
        }
    }
    if (u0 > u1) {
        return s;
    }
    s.a = a + d * u0;
    s.b = a + d * u1;
    s.z = vec2<f32>(mix(z.x, z.y, u0), mix(z.x, z.y, u1));
    s.w = vec2<f32>(mix(w.x, w.y, u0), mix(w.x, w.y, u1));
    s.t = vec2<f32>(mix(t.x, t.y, u0), mix(t.x, t.y, u1));
    s.ok = true;
    return s;
}

// Motion (pixels, current minus previous) of the world point mix(p, q, t) between frames.
fn motion_of(cur: vec3<f32>, prev: vec3<f32>) -> vec2<f32> {
    let c = frame.unjittered_view_proj * vec4<f32>(cur, 1.0);
    let p = frame.prev_view_proj * vec4<f32>(prev, 1.0);
    if (c.w <= 1e-6 || p.w <= 1e-6) {
        return vec2<f32>(0.0);
    }
    return (c.xy / c.w - p.xy / p.w) * vec2<f32>(0.5, -0.5) * view.screen.xy;
}

// Walks share `part` of `parts` of the span; `records` false adds coverage and depth, true writes
// the nearest's record.
fn walk(s: Span, part: u32, parts: u32, records: bool, record: vec4<u32>, motion: array<vec2<f32>, 2>) {
    let d = s.b - s.a;
    let major = select(1, 0, abs(d.x) >= abs(d.y));
    let minor = 1 - major;
    let length_major = d[major];
    if (abs(length_major) < 1e-6) {
        return;
    }
    let slope = d[minor] / length_major;
    let stretch = sqrt(1.0 + slope * slope);
    let width = view.screen.xy;
    let saturation = u32(view.screen.w);
    let lo = min(s.a[major], s.b[major]);
    let hi = max(s.a[major], s.b[major]);
    // Pixel centres c + 0.5 in [lo, hi) along the major axis.
    let first = i32(ceil(lo - 0.5));
    let last = min(i32(ceil(hi - 0.5)) - 1, first + HAIR_MAX_STEPS - 1);
    let size_major = i32(width[major]);
    let size_minor = i32(width[minor]);
    // This thread's share of the on-screen pixels.
    let start = max(first, 0);
    let end = min(last, size_major - 1);
    let share = (end - start + i32(parts)) / i32(parts);
    let mine = start + i32(part) * share;
    for (var i = mine; i <= min(end, mine + share - 1); i++) {
        let u = (f32(i) + 0.5 - s.a[major]) / length_major;
        let centre = s.a[minor] + u * d[minor];
        let z = mix(s.z.x, s.z.y, u);
        let footprint = min(mix(s.w.x, s.w.y, u) * stretch, HAIR_MAX_WIDTH);
        let half_extent = 0.5 * max(footprint, 1.0);
        let j0 = max(i32(floor(centre - half_extent)), 0);
        let j1 = min(i32(floor(centre + half_extent)), size_minor - 1);
        for (var j = j0; j <= j1; j++) {
            let overlap = max(min(f32(j + 1), centre + half_extent) - max(f32(j), centre - half_extent), 0.0);
            let coverage = min(footprint * overlap / (2.0 * half_extent), 1.0);
            if (coverage <= 0.0) {
                continue;
            }
            var pixel = vec2<i32>(i, j);
            if (major == 1) {
                pixel = vec2<i32>(j, i);
            }
            let index = u32(pixel.y) * u32(width.x) + u32(pixel.x);
            let bits = bitcast<u32>(z);
            if (records) {
                if (bits == atomicLoad(&hair_depth[index])) {
                    var r = hair_record_with_coverage(record, coverage * draw.scale.x);
                    r.z = pack2x16float(mix(motion[0], motion[1], u));
                    hair_records[index] = r;
                }
                continue;
            }
            // The scene depth test first: it culls the fragments behind the skin through the
            // texture cache, before any atomic traffic (testing saturation first was 4.7 ms
            // slower at 1440p close range).
            if (z <= textureLoad(scene_depth, pixel, 0)) {
                continue;
            }
            let q = u32(coverage * draw.scale.x * HAIR_COVERAGE_ONE + 0.5);
            if (q > 0u && atomicLoad(&hair_coverage[index]) < saturation) {
                atomicAdd(&hair_coverage[index], q);
            }
            if (bits > atomicLoad(&hair_depth[index])) {
                atomicMax(&hair_depth[index], bits);
            }
        }
    }
}

@compute @workgroup_size(64)
fn raster_depth(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seg = segment_of(gid);
    if (!seg.ok) {
        return;
    }
    let s = project_span(points[seg.first], points[seg.last]);
    if (!s.ok) {
        return;
    }
    walk(s, seg.part, seg.parts, false, vec4<u32>(0u), array<vec2<f32>, 2>(vec2<f32>(0.0), vec2<f32>(0.0)));
}

@compute @workgroup_size(64)
fn raster_records(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seg = segment_of(gid);
    if (!seg.ok) {
        return;
    }
    let p0 = points[seg.first];
    let p1 = points[seg.last];
    let s = project_span(p0, p1);
    if (!s.ok) {
        return;
    }
    let q0 = prev_points[seg.first].xyz;
    let q1 = prev_points[seg.last].xyz;
    let motion = array<vec2<f32>, 2>(
        motion_of(mix(p0.xyz, p1.xyz, s.t.x), mix(q0, q1, s.t.x)),
        motion_of(mix(p0.xyz, p1.xyz, s.t.y), mix(q0, q1, s.t.y)),
    );
    let tangent = safe_normalize(p1.xyz - p0.xyz, vec3<f32>(0.0, -1.0, 0.0));
    let record = hair_pack_record(tangent, draw.counts.w, attrs[2u * seg.strand + 1u].x, vec2<f32>(0.0), seg.strand);
    walk(s, seg.part, seg.parts, true, record, motion);
}

@compute @workgroup_size(64)
fn voxelize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seg = segment_of(gid);
    if (!seg.ok || seg.part != 0u) {
        return;
    }
    let p0 = points[seg.first];
    let p1 = points[seg.last];
    let n = view.counts.x;
    let cell = (0.5 * (p0.xyz + p1.xyz) - view.voxel.xyz) / view.voxel.w;
    if (n == 0u || any(cell < vec3<f32>(0.0)) || any(cell >= vec3<f32>(f32(n)))) {
        return;
    }
    let c = vec3<u32>(cell);
    let area = length(p1.xyz - p0.xyz) * 0.5 * (p0.w + p1.w) * draw.scale.x;
    let q = u32(area / HAIR_VOXEL_UNIT + 0.5);
    if (q > 0u) {
        atomicAdd(&hair_voxels[(c.z * n + c.y) * n + c.x], q);
    }
}
