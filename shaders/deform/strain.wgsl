// deform/strain.wgsl: edge strain against the rest shape and its per-region reduction (spec 6.2
// step 6) [EVAL-06].
//
// Four passes over the welded edges (pairs of weld-key vertices) of the template:
//
// 1. `edge_strain`, per edge: l0 = |rest[a] - rest[b]| and s = |p[a] - p[b]| / l0 - 1 (0 when
//    l0 is 0 or the result is not finite); writes (s, l0).
// 2. `region_sums`, one workgroup per block: every edge counts half for the region of each end
//    vertex, weighted w = l0 / 2. A region's list of edge halves is cut into blocks of at most
//    REDUCE_BLOCK (never across regions, so large regions spread over many workgroups); each block
//    sums (sum w, sum w max(s, 0), sum w max(-s, 0)), reduced in workgroup memory in a fixed order.
// 3. `region_strain`, per region: the block sums of the region and all its descendants (a CSR list
//    that includes the region itself), normalised: (stretch, compression), 0 for a region without
//    edges. This vec2 per region is the small buffer the skin shader reads for wrinkle and
//    tension blending.
// 4. `vertex_strain`, per vertex: the mean signed strain of the edges at its weld key (the same on
//    every duplicate), for the strain viewport mode.
//
// The CPU reference is humanart_model::eval::StrainField.

#include "deform/common.wgsl"
#include "lib/math.wgsl"

struct Params {
    edge_count: u32,
    region_count: u32,
    vertex_count: u32,
    block_count: u32,
}

const REDUCE: u32 = 256u;
const REDUCE_BLOCK: u32 = 1024u;

@group(0) @binding(0) var<uniform> params: Params;
// edge_strain
@group(0) @binding(1) var<storage, read> rest: array<f32>;
@group(0) @binding(2) var<storage, read> positions: array<f32>;
@group(0) @binding(3) var<storage, read> edges: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> edge_out: array<vec2<f32>>;
// region_sums
@group(0) @binding(5) var<storage, read> edge_in: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read> blocks: array<vec2<u32>>;
@group(0) @binding(7) var<storage, read> region_edges: array<u32>;
@group(0) @binding(8) var<storage, read_write> block_sums: array<vec4<f32>>;
// region_strain
@group(0) @binding(9) var<storage, read> block_in: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read> subtree_offsets: array<u32>;
@group(0) @binding(11) var<storage, read> subtree_regions: array<u32>;
@group(0) @binding(12) var<storage, read_write> strain_out: array<vec2<f32>>;
// vertex_strain (also reads edge_in)
@group(0) @binding(13) var<storage, read> position_keys: array<u32>;
@group(0) @binding(14) var<storage, read> key_offsets: array<u32>;
@group(0) @binding(15) var<storage, read> key_edges: array<u32>;
@group(0) @binding(16) var<storage, read_write> vertex_out: array<f32>;
@group(0) @binding(17) var<storage, read> region_blocks: array<u32>;

var<workgroup> partial: array<vec3<f32>, REDUCE>;

fn load(buffer_index: u32, v: u32) -> vec3<f32> {
    if (buffer_index == 0u) {
        return vec3<f32>(rest[3u * v], rest[3u * v + 1u], rest[3u * v + 2u]);
    }
    return vec3<f32>(positions[3u * v], positions[3u * v + 1u], positions[3u * v + 2u]);
}

@compute @workgroup_size(64)
fn edge_strain(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let e = thread_index(group, groups, local);
    if (e >= params.edge_count) {
        return;
    }
    let ab = edges[e];
    if (ab.x >= params.vertex_count || ab.y >= params.vertex_count) {
        edge_out[e] = vec2<f32>(0.0, 0.0);
        return;
    }
    let l0 = distance(load(0u, ab.x), load(0u, ab.y));
    var s = 0.0;
    if (l0 > 0.0) {
        s = distance(load(1u, ab.x), load(1u, ab.y)) / l0 - 1.0;
        if (!(abs(s) <= F32_MAX)) {
            s = 0.0;
        }
    }
    edge_out[e] = vec2<f32>(s, l0);
}

@compute @workgroup_size(256)
fn region_sums(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let b = group.y * groups.x + group.x;
    var acc = vec3<f32>(0.0);
    if (b < params.block_count) {
        let range = blocks[b];
        for (var i = range.x + local; i < range.y; i += REDUCE) {
            let sl = edge_in[region_edges[i]];
            let w = 0.5 * sl.y;
            acc += vec3<f32>(w, w * max(sl.x, 0.0), w * max(-sl.x, 0.0));
        }
    }
    partial[local] = acc;
    workgroupBarrier();
    for (var stride = REDUCE / 2u; stride > 0u; stride /= 2u) {
        if (local < stride) {
            partial[local] += partial[local + stride];
        }
        workgroupBarrier();
    }
    if (local == 0u && b < params.block_count) {
        block_sums[b] = vec4<f32>(partial[0], 0.0);
    }
}

@compute @workgroup_size(64)
fn region_strain(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let r = thread_index(group, groups, local);
    if (r >= params.region_count) {
        return;
    }
    var sum = vec3<f32>(0.0);
    let end = subtree_offsets[r + 1u];
    for (var i = subtree_offsets[r]; i < end; i += 1u) {
        let s = subtree_regions[i];
        let last = region_blocks[s + 1u];
        for (var k = region_blocks[s]; k < last; k += 1u) {
            sum += block_in[k].xyz;
        }
    }
    var inv = 0.0;
    if (sum.x > 0.0) {
        inv = 1.0 / sum.x;
    }
    strain_out[r] = vec2<f32>(sum.y * inv, sum.z * inv);
}

@compute @workgroup_size(64)
fn vertex_strain(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let v = thread_index(group, groups, local);
    if (v >= params.vertex_count) {
        return;
    }
    let k = position_keys[v];
    var sum = 0.0;
    var count = 0u;
    if (k < params.vertex_count) {
        let end = key_offsets[k + 1u];
        for (var i = key_offsets[k]; i < end; i += 1u) {
            sum += edge_in[key_edges[i]].x;
            count += 1u;
        }
    }
    var mean = 0.0;
    if (count > 0u) {
        mean = sum / f32(count);
    }
    vertex_out[v] = mean;
}
