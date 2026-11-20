// bake/thickness.wgsl: thickness and translucency per texel [BAKE-01], the GPU twin of
// humanart_tools::bake::thickness::bake_thickness.
//
// One invocation per texel of a band of rows (`row0` onward; the host splits large maps into
// bands so no single dispatch runs long). A texel the G-buffer marks as surface casts `rays` rays
// into the cone of half-angle acos(cos_max) around the inward normal from just inside the surface;
// a ray's length is the distance to the first surface it meets, or max_distance. The mean length
// times the region's scale is the thickness; translucency = exp(-thickness / falloff).
// out0 = (thickness, translucency, 0, 1) on the surface, 0 elsewhere.

#include "bake/common.wgsl"

struct Params {
    size: vec2<u32>,
    row0: u32,
    seed: u32,
    rays: u32,
    region_count: u32,
    cos_max: f32,
    max_distance: f32,
    falloff: f32,
    offset: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// The mesh and its hierarchy, read by bake/common.wgsl.
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> normals: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> tangents: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> indices: array<u32>;
@group(0) @binding(5) var<storage, read> nodes: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> tris: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> region_scale: array<f32>;
@group(0) @binding(8) var gbuffer_ids: texture_2d<u32>;
@group(0) @binding(9) var gbuffer_bary: texture_2d<f32>;
@group(0) @binding(10) var out0: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn thickness(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y + params.row0;
    if (x >= params.size.x || y >= params.size.y) {
        return;
    }
    let texel = vec2<i32>(i32(x), i32(y));
    let id = textureLoad(gbuffer_ids, texel, 0).x;
    if (id == 0u) {
        textureStore(out0, texel, vec4<f32>(0.0));
        return;
    }
    let s = bake_surface(id - 1u, textureLoad(gbuffer_bary, texel, 0).xy);
    let origin = s.position - s.geometric * params.offset;
    let r = bake_rotation(bake_texel_hash(x, y, params.seed));
    var sum = 0.0;
    for (var i = 0u; i < params.rays; i += 1u) {
        let local = bake_uniform_cone(bake_point(i, params.rays, r), params.cos_max);
        let d = bake_normalize_or(bake_keep_side(bake_to_world(local, -s.normal), s.geometric, -1.0), -s.geometric);
        let hit = bake_intersect(origin, d, 0.0, params.max_distance);
        sum += select(params.max_distance, hit.t, hit.hit);
    }
    var scale = 1.0;
    if (s.region < params.region_count && s.region < arrayLength(&region_scale)) {
        scale = region_scale[s.region];
    }
    let t = sum / f32(params.rays) * scale;
    textureStore(out0, texel, vec4<f32>(t, exp(-t / params.falloff), 0.0, 1.0));
}
