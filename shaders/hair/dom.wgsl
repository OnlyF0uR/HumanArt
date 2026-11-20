// hair/dom.wgsl: deep opacity maps of the hair for up to two lights [HR-03] (Yuksel and Keyser,
// "Deep Opacity Maps", Eurographics 2008): every kept strand's segments drawn as hardware lines in
// an orthographic frame along the light. `vs_dom` alone (depth-only pipeline, reversed-Z,
// GreaterEqual) finds the first fibre per texel; `fs_dom_opacity` then adds each fragment's
// coverage (fibre diameter over the texel size, times the strand step of the subsampling) to one
// of four layers by its distance behind that first fibre (additive blending). The resolve pass
// turns the layers into the expected number of fibres crossed towards the light.

#include "lib/math.wgsl"
#include "lib/packing.wgsl"
#include "render/frame.wgsl"
#include "hair/common.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> view: HairView;

@group(1) @binding(0) var<uniform> draw: HairDraw;
@group(1) @binding(1) var<storage, read> points: array<vec4<f32>>;

@group(2) @binding(0) var dom_depth: texture_depth_2d_array;

struct DomVertex {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) coverage: f32,
    @location(1) @interpolate(flat) map: u32,
}

@vertex
fn vs_dom(@builtin(vertex_index) index: u32, @builtin(instance_index) map: u32) -> DomVertex {
    var out: DomVertex;
    out.map = map;
    out.coverage = 0.0;
    // Outside the clip volume unless the vertex is valid.
    out.position = vec4<f32>(2.0, 2.0, -1.0, 1.0);
    let n = draw.counts.y;
    let step = max(draw.dispatch.w, 1u);
    if (n < 2u || map >= HAIR_MAX_DOMS) {
        return out;
    }
    let segment = index / 2u;
    let kept = segment / (n - 1u);
    let k = segment - kept * (n - 1u);
    let strand = kept * step;
    if (strand >= draw.counts.x) {
        return out;
    }
    let p = points[strand * n + k + (index & 1u)];
    let params = view.dom_params[map];
    let local = hair_dom_local(p.xyz, view.dom_right[map], view.dom_up[map], view.dom_forward[map], params.x);
    out.position = vec4<f32>(local.xy, local.z, 1.0);
    out.coverage = min(p.w / max(params.z, 1e-9) * draw.scale.y, 1.0);
    return out;
}

@fragment
fn fs_dom_opacity(in: DomVertex) -> @location(0) vec4<f32> {
    let first = textureLoad(dom_depth, vec2<i32>(in.position.xy), i32(in.map), 0);
    let radius = view.dom_params[in.map].x;
    let behind = max(first - in.position.z, 0.0) * 2.0 * radius;
    let edges = view.dom_layers;
    var layer = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if (behind < edges.x) {
        layer = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    } else if (behind < edges.y) {
        layer = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    } else if (behind < edges.z) {
        layer = vec4<f32>(0.0, 0.0, 1.0, 0.0);
    }
    return layer * in.coverage;
}
