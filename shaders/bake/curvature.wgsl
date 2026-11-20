// bake/curvature.wgsl: the mean-curvature map [BAKE-03], the GPU half of
// humanart_tools::bake::surface::bake_curvature.
//
// The per-vertex mean curvature (cotangent Laplace-Beltrami with mixed Voronoi areas, an O(V) pass
// over the welded mesh) comes from the CPU (surface::vertex_mean_curvature) in `values`; this kernel
// interpolates it barycentrically into every texel the G-buffer covers, one invocation per texel
// of a band of rows. out0 = (H, 0, 0, 1) on the surface, 0 elsewhere.

struct Params {
    size: vec2<u32>,
    row0: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> indices: array<u32>;
@group(0) @binding(7) var<storage, read> values: array<f32>;
@group(0) @binding(8) var gbuffer_ids: texture_2d<u32>;
@group(0) @binding(9) var gbuffer_bary: texture_2d<f32>;
@group(0) @binding(10) var out0: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn curvature(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y + params.row0;
    if (x >= params.size.x || y >= params.size.y) {
        return;
    }
    let texel = vec2<i32>(i32(x), i32(y));
    let id = textureLoad(gbuffer_ids, texel, 0).x;
    let corner = 3u * (id - 1u);
    if (id == 0u || corner + 2u >= arrayLength(&indices)) {
        textureStore(out0, texel, vec4<f32>(0.0));
        return;
    }
    let b = textureLoad(gbuffer_bary, texel, 0).xy;
    let n = arrayLength(&values);
    let ia = min(indices[corner], n - 1u);
    let ib = min(indices[corner + 1u], n - 1u);
    let ic = min(indices[corner + 2u], n - 1u);
    let h = values[ia] * (1.0 - b.x - b.y) + values[ib] * b.x + values[ic] * b.y;
    textureStore(out0, texel, vec4<f32>(h, 0.0, 0.0, 1.0));
}
