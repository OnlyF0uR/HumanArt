// bake/gbuffer.wgsl: the UV-space G-buffer of the GPU bakers, the twin of
// humanart_tools::bake::raster::rasterize.
//
// Draws every triangle (non-indexed, 3 vertices per triangle, pulled from storage buffers) at its
// UV position: clip = (2u - 1, 1 - 2v), so texel (x, y) of the target is UV ((x + 1/2) / w,
// (y + 1/2) / h) with V down, and the top-left rule of the hardware decides texels on shared
// edges as the CPU rasteriser does. Triangles of other tiles collapse outside the clip volume.
// Target 0 (r32uint) receives triangle + 1 (0 = empty), target 1 (rg32float) the barycentric
// weights (b1, b2) of the second and third corners of the triangle.

struct Params {
    tile: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> uvs: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> tiles: array<u32>;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) triangle: u32,
    @location(1) bary: vec2<f32>,
}

@vertex
fn vs_gbuffer(@builtin(vertex_index) vi: u32) -> VsOut {
    let t = vi / 3u;
    let k = vi % 3u;
    var out: VsOut;
    out.triangle = t + 1u;
    out.bary = vec2<f32>(select(0.0, 1.0, k == 1u), select(0.0, 1.0, k == 2u));
    out.position = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
    if (tiles[t] == params.tile) {
        let uv = uvs[indices[vi]];
        out.position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y, 0.0, 1.0);
    }
    return out;
}

struct FsOut {
    @location(0) id: vec4<u32>,
    @location(1) bary: vec4<f32>,
}

@fragment
fn fs_gbuffer(in: VsOut) -> FsOut {
    var out: FsOut;
    out.id = vec4<u32>(in.triangle, 0u, 0u, 0u);
    out.bary = vec4<f32>(in.bary, 0.0, 0.0);
    return out;
}
