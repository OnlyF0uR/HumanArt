// deform/lod.wgsl: the deformed surface of a reduced level of detail [MOD-05], from the deformed
// hero mesh (LOD0).
//
// `transfer`, one invocation per vertex of the level: p = sum_k w_k * P0[s_k] over the vertex's
// three LOD0 sources and their barycentric weights (humanart_model::lod::LodLevel), so a level
// follows every identity, expression, corrective and pose of LOD0 without its own deltas or skin.
// The level's normals and tangents then come from the passes of deform/normals.wgsl on its own
// triangles. The CPU reference is humanart_model::lod::LodLevel::transfer_positions.

#include "deform/common.wgsl"

struct Params {
    vertex_count: u32,
    source_count: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> source_positions: array<f32>;
// Per level vertex: three LOD0 vertex indices (x, y, z; w unused).
@group(0) @binding(2) var<storage, read> sources: array<vec4<u32>>;
// Per level vertex: the three weights (x, y, z; w unused).
@group(0) @binding(3) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> positions: array<f32>;

fn source(v: u32) -> vec3<f32> {
    if (v >= params.source_count) {
        return vec3<f32>(0.0);
    }
    return vec3<f32>(source_positions[3u * v], source_positions[3u * v + 1u], source_positions[3u * v + 2u]);
}

@compute @workgroup_size(64)
fn transfer(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let v = thread_index(group, groups, local);
    if (v >= params.vertex_count) {
        return;
    }
    let s = sources[v];
    let w = weights[v];
    let p = source(s.x) * w.x + source(s.y) * w.y + source(s.z) * w.z;
    positions[3u * v] = p.x;
    positions[3u * v + 1u] = p.y;
    positions[3u * v + 2u] = p.z;
}
