// deform/accumulate.wgsl: the rest shape P_rest = P_template + sum w_c * D_c + sum a_l * L_l
// (spec 6.2 step 1) on the GPU [EVAL-02] [EVAL-03].
//
// Three entry points share the rest buffer (packed f32 xyz per vertex):
//
// - `gather_rest`, the full re-accumulation: one invocation per vertex sums its row of the
//   vertex-major shape table (component, f16 delta) with the per-component weights, on top of the
//   template position. No two invocations write one vertex, so there are no races. The table is
//   sliced per workgroup (entry k of lane l at slice_offsets[slice] + 64 k + l, rows padded with
//   zero deltas), so the workgroup's reads are coalesced.
// - `scatter_component`, the incremental update: one invocation per entry of one delta component
//   adds `weight` (the difference between the new and the applied weight) times its f16 delta.
//   A component lists every vertex at most once, so one dispatch per changed component is
//   race-free; consecutive dispatches in a pass are ordered by WebGPU.
// - `scatter_layer`: the same for a sculpt layer, whose deltas are packed f32 triples.
//
// The CPU reference is humanart_model::eval::Evaluator (`apply_rest`, `refresh_rest`).

#include "deform/common.wgsl"

struct Params {
    // Vertices (gather) or entries (scatter).
    count: u32,
    // The component's first entry in the shared entry arrays (scatter_component).
    first: u32,
    // The weight difference (scatter).
    weight: f32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> rest: array<f32>;
// gather_rest
@group(0) @binding(2) var<storage, read> template_positions: array<f32>;
@group(0) @binding(3) var<storage, read> shape_slices: array<u32>;
@group(0) @binding(4) var<storage, read> shape_entries: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read> shape_weights: array<f32>;
// scatter_component
@group(0) @binding(6) var<storage, read> component_indices: array<u32>;
@group(0) @binding(7) var<storage, read> component_deltas: array<vec2<u32>>;
// scatter_layer
@group(0) @binding(8) var<storage, read> layer_indices: array<u32>;
@group(0) @binding(9) var<storage, read> layer_deltas: array<f32>;

fn add_rest(v: u32, d: vec3<f32>) {
    let base = 3u * v;
    if (base + 2u >= arrayLength(&rest)) {
        return;
    }
    rest[base] += d.x;
    rest[base + 1u] += d.y;
    rest[base + 2u] += d.z;
}

@compute @workgroup_size(64)
fn gather_rest(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let v = thread_index(group, groups, local);
    if (v >= params.count) {
        return;
    }
    let base = 3u * v;
    var p = vec3<f32>(template_positions[base], template_positions[base + 1u], template_positions[base + 2u]);
    let slice = v / WORKGROUP;
    let end = shape_slices[slice + 1u];
    for (var i = shape_slices[slice] + v % WORKGROUP; i < end; i += WORKGROUP) {
        let e = shape_entries[i];
        let w = shape_weights[e.x & 0xffffu];
        if (w != 0.0) {
            p += decode_delta(e.y, e.x >> 16u) * w;
        }
    }
    rest[base] = p.x;
    rest[base + 1u] = p.y;
    rest[base + 2u] = p.z;
}

@compute @workgroup_size(64)
fn scatter_component(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let i = thread_index(group, groups, local);
    if (i >= params.count) {
        return;
    }
    let k = params.first + i;
    let d = component_deltas[k];
    add_rest(component_indices[k], decode_delta(d.x, d.y) * params.weight);
}

@compute @workgroup_size(64)
fn scatter_layer(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let i = thread_index(group, groups, local);
    if (i >= params.count) {
        return;
    }
    let d = vec3<f32>(layer_deltas[3u * i], layer_deltas[3u * i + 1u], layer_deltas[3u * i + 2u]);
    add_rest(layer_indices[i], d * params.weight);
}
