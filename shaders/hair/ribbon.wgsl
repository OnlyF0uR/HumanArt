// hair/ribbon.wgsl: the hardware ribbon rasteriser of the hair [HR-01], the fallback of the compute
// rasteriser (hair/raster.wgsl) that needs neither its atomics-heavy compute walk nor mesh
// shaders. Each segment becomes a screen-aligned quad (two triangles, six vertices) one pixel
// either side of the strand (wider strands: their half width plus one pixel); the fragment's
// coverage is the strand's projected width times a one-pixel tent across it (box with a one-pixel
// ramp when wider than a pixel), so a column of fragments sums to the strand's footprint like
// the compute walk. The opaque scene depth is the read-only depth attachment (early rejection of
// fibres behind the skin). The fragments feed the same visibility buffer as the compute path:
// `fs_ribbon_depth` adds coverage and keeps the nearest depth (atomics), `fs_ribbon_records`
// writes the record of the fragment whose depth is the pixel's nearest (the position is
// @invariant, so both pipelines produce bit-identical depths). Location 0 is a dummy output the
// pipelines mask off (a fragment stage must write its target).

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

@group(2) @binding(0) var<storage, read_write> hair_depth: array<atomic<u32>>;
@group(2) @binding(1) var<storage, read_write> hair_coverage: array<atomic<u32>>;
@group(2) @binding(2) var<storage, read_write> hair_records: array<vec4<u32>>;
// The opaque scene's depth (reversed-Z). The fragments test it themselves: they write storage
// buffers, and a pipeline depth test may run after the fragment shader (late Z), too late to stop
// the writes of hidden fragments.
@group(2) @binding(3) var scene_depth: texture_depth_2d;

struct RibbonVertex {
    @builtin(position) @invariant position: vec4<f32>,
    // Signed distance from the strand's centre line, pixels.
    @location(0) across: f32,
    // Projected strand width, pixels.
    @location(1) width: f32,
    @location(2) motion: vec2<f32>,
    @location(3) @interpolate(flat) record: vec4<u32>,
}

fn screen_of(c: vec4<f32>) -> vec2<f32> {
    let ndc = c.xy / c.w;
    return vec2<f32>((ndc.x * 0.5 + 0.5) * view.screen.x, (0.5 - ndc.y * 0.5) * view.screen.y);
}

fn motion_at(cur: vec3<f32>, prev: vec3<f32>) -> vec2<f32> {
    let c = frame.unjittered_view_proj * vec4<f32>(cur, 1.0);
    let p = frame.prev_view_proj * vec4<f32>(prev, 1.0);
    if (c.w <= 1e-6 || p.w <= 1e-6) {
        return vec2<f32>(0.0);
    }
    return (c.xy / c.w - p.xy / p.w) * vec2<f32>(0.5, -0.5) * view.screen.xy;
}

@vertex
fn vs_ribbon(@builtin(vertex_index) index: u32) -> RibbonVertex {
    var out: RibbonVertex;
    out.position = vec4<f32>(2.0, 2.0, -1.0, 1.0);
    out.across = 0.0;
    out.width = 0.0;
    out.motion = vec2<f32>(0.0);
    out.record = vec4<u32>(0u);
    let n = draw.counts.y;
    let per = draw.dispatch.x;
    if (n < 2u || per == 0u) {
        return out;
    }
    let segment = index / 6u;
    let corner = index - segment * 6u;
    let strand = segment / per;
    if (strand >= draw.counts.x || !hair_kept(strand, draw.dispatch.z)) {
        return out;
    }
    let stride = max(draw.counts.z, 1u);
    let k = (segment - strand * per) * stride;
    if (k >= n - 1u) {
        return out;
    }
    let first = strand * n + k;
    let last = strand * n + min(k + stride, n - 1u);
    let p0 = points[first];
    let p1 = points[last];
    let c0 = frame.view_proj * vec4<f32>(p0.xyz, 1.0);
    let c1 = frame.view_proj * vec4<f32>(p1.xyz, 1.0);
    // Segments crossing the near plane are dropped (the compute path clips them).
    if (c0.w - c0.z < 0.0 || c1.w - c1.z < 0.0 || c0.w <= 1e-6 || c1.w <= 1e-6) {
        return out;
    }
    // Corner order of the two triangles: (end, side).
    let ends = array<u32, 6>(0u, 1u, 0u, 0u, 1u, 1u);
    let sides = array<f32, 6>(-1.0, -1.0, 1.0, 1.0, -1.0, 1.0);
    let end = ends[corner];
    let side = sides[corner];
    let s0 = screen_of(c0);
    let s1 = screen_of(c1);
    let dir = s1 - s0;
    let len = length(dir);
    var normal = vec2<f32>(0.0, 1.0);
    if (len > 1e-6) {
        normal = vec2<f32>(-dir.y, dir.x) / len;
    }
    let c = select(c0, c1, end == 1u);
    let p = select(p0, p1, end == 1u);
    let focal = view.screen.z * draw.scale.z;
    let width = min(p.w * focal / c.w, HAIR_MAX_WIDTH);
    let half_extent = max(0.5 * width + 1.0, 1.0);
    // Offset in pixels, back to clip space (y flips between pixels and NDC).
    let offset = normal * (side * half_extent);
    let clip_offset = vec2<f32>(offset.x * 2.0 / view.screen.x, -offset.y * 2.0 / view.screen.y) * c.w;
    out.position = vec4<f32>(c.xy + clip_offset, c.z, c.w);
    out.across = side * half_extent;
    out.width = width;
    let q = select(prev_points[first].xyz, prev_points[last].xyz, end == 1u);
    out.motion = motion_at(p.xyz, q);
    let tangent = safe_normalize(p1.xyz - p0.xyz, vec3<f32>(0.0, -1.0, 0.0));
    out.record = hair_pack_record(tangent, draw.counts.w, attrs[2u * strand + 1u].x, vec2<f32>(0.0), strand);
    return out;
}

// The fragment's coverage: width times a one-pixel tent, or a box with a one-pixel ramp.
fn ribbon_coverage(in: RibbonVertex) -> f32 {
    let d = abs(in.across);
    if (in.width < 1.0) {
        return in.width * saturate(1.0 - d);
    }
    return saturate(0.5 * in.width + 0.5 - d);
}

// Whether the fragment at `position` is behind the opaque scene.
fn ribbon_hidden(position: vec4<f32>) -> bool {
    return position.z <= textureLoad(scene_depth, vec2<i32>(position.xy), 0);
}

fn ribbon_index(position: vec4<f32>) -> u32 {
    let pixel = vec2<u32>(position.xy);
    return pixel.y * u32(view.screen.x) + pixel.x;
}

@fragment
fn fs_ribbon_depth(in: RibbonVertex) -> @location(0) vec4<f32> {
    let coverage = ribbon_coverage(in);
    if (coverage <= 0.0 || ribbon_hidden(in.position)) {
        discard;
    }
    let index = ribbon_index(in.position);
    let q = u32(coverage * draw.scale.x * HAIR_COVERAGE_ONE + 0.5);
    if (q > 0u && atomicLoad(&hair_coverage[index]) < u32(view.screen.w)) {
        atomicAdd(&hair_coverage[index], q);
    }
    let bits = bitcast<u32>(in.position.z);
    if (bits > atomicLoad(&hair_depth[index])) {
        atomicMax(&hair_depth[index], bits);
    }
    return vec4<f32>(coverage);
}

@fragment
fn fs_ribbon_records(in: RibbonVertex) -> @location(0) vec4<f32> {
    let coverage = ribbon_coverage(in);
    if (coverage <= 0.0 || ribbon_hidden(in.position)) {
        discard;
    }
    let index = ribbon_index(in.position);
    if (bitcast<u32>(in.position.z) == atomicLoad(&hair_depth[index])) {
        var r = hair_record_with_coverage(in.record, coverage * draw.scale.x);
        r.z = pack2x16float(in.motion);
        hair_records[index] = r;
    }
    return vec4<f32>(coverage);
}
