// bake/cavity_ao.wgsl: ambient occlusion, cavity and bent normals per texel [BAKE-02], the GPU
// twin of humanart_tools::bake::occlusion::bake_occlusion.
//
// One invocation per texel of a band of rows. A surface texel casts `rays` cosine-weighted rays
// around the shading normal from just outside the surface, each up to max_distance: ao is the
// fraction that escapes, cavity the fraction not blocked within cavity_distance, the bent normal
// the normalised mean of the escaping directions (the shading normal when none escapes), in the
// tangent frame of the texel. out0 = (ao, cavity, 0, 1) and out1 = (bent.xyz, 1) on the surface;
// out0 = (1, 1, 0, 0) and out1 = 0 elsewhere.

#include "bake/common.wgsl"

struct Params {
    size: vec2<u32>,
    row0: u32,
    seed: u32,
    rays: u32,
    pad0: u32,
    max_distance: f32,
    cavity_distance: f32,
    offset: f32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// The mesh and its hierarchy, read by bake/common.wgsl.
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> normals: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> tangents: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> indices: array<u32>;
@group(0) @binding(5) var<storage, read> nodes: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> tris: array<vec4<f32>>;
@group(0) @binding(8) var gbuffer_ids: texture_2d<u32>;
@group(0) @binding(9) var gbuffer_bary: texture_2d<f32>;
@group(0) @binding(10) var out0: texture_storage_2d<rgba32float, write>;
@group(0) @binding(11) var out1: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn cavity_ao(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y + params.row0;
    if (x >= params.size.x || y >= params.size.y) {
        return;
    }
    let texel = vec2<i32>(i32(x), i32(y));
    let id = textureLoad(gbuffer_ids, texel, 0).x;
    if (id == 0u) {
        textureStore(out0, texel, vec4<f32>(1.0, 1.0, 0.0, 0.0));
        textureStore(out1, texel, vec4<f32>(0.0));
        return;
    }
    let s = bake_surface(id - 1u, textureLoad(gbuffer_bary, texel, 0).xy);
    let origin = s.position + s.geometric * params.offset;
    let r = bake_rotation(bake_texel_hash(x, y, params.seed));
    var open = 0u;
    var open_cavity = 0u;
    var bent = vec3<f32>(0.0);
    for (var i = 0u; i < params.rays; i += 1u) {
        let local = bake_cosine_hemisphere(bake_point(i, params.rays, r));
        let d = bake_normalize_or(bake_keep_side(bake_to_world(local, s.normal), s.geometric, 1.0), s.geometric);
        let hit = bake_intersect(origin, d, 0.0, params.max_distance);
        if (!hit.hit) {
            open += 1u;
            open_cavity += 1u;
            bent += d;
        } else if (hit.t >= params.cavity_distance) {
            open_cavity += 1u;
        }
    }
    let n = f32(params.rays);
    let b = bake_to_tangent(s, bake_normalize_or(bent, s.normal));
    textureStore(out0, texel, vec4<f32>(f32(open) / n, f32(open_cavity) / n, 0.0, 1.0));
    textureStore(out1, texel, vec4<f32>(b, 1.0));
}
