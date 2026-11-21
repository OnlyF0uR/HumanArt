// render/pick.wgsl: the picking buffer [RND-03] [VP-01]: the unjittered scene rasterised (usually
// scissored to the pixels under the cursor), read back asynchronously.
// x = pick id of the object (0 is nothing), y = bitcast of the reversed-Z depth of the hit,
// z = the triangle of the object's mesh (its index range starts at the object's first index),
// w = the hit's barycentric weights of the triangle's second and third corners as two unorm16
//     (pack2x16unorm; the first corner's weight is 1 minus both).
// Triangles are pulled through the index buffer (a non-indexed draw of the object's index count),
// so the vertex index is the corner: triangle = corner / 3, and the barycentric weights are the
// corners' unit vectors interpolated perspective-correctly.

#include "render/frame.wgsl"
#include "render/scene.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<storage, read> positions: array<f32>;
@group(1) @binding(1) var<storage, read> prev_positions: array<f32>;
@group(1) @binding(2) var<storage, read> normals: array<f32>;
@group(1) @binding(3) var<storage, read> uvs: array<f32>;
@group(1) @binding(4) var<storage, read> objects: array<Object>;
@group(1) @binding(5) var<storage, read> indices: array<u32>;

struct PickOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) id: u32,
    @location(1) @interpolate(flat) triangle: u32,
    @location(2) bary: vec3<f32>,
}

@vertex
fn vs_pick(@builtin(vertex_index) corner: u32, @builtin(instance_index) object_index: u32) -> PickOut {
    let object = objects[object_index];
    let vertex = object.ids.w + indices[object.ids.z + corner];
    var out: PickOut;
    out.clip = frame.unjittered_view_proj * vec4<f32>(world_position(object, vertex), 1.0);
    out.id = object.ids.x;
    out.triangle = corner / 3u;
    let k = corner % 3u;
    out.bary = vec3<f32>(f32(k == 0u), f32(k == 1u), f32(k == 2u));
    return out;
}

@fragment
fn fs_pick(in: PickOut) -> @location(0) vec4<u32> {
    let b = clamp(in.bary, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<u32>(in.id, bitcast<u32>(in.clip.z), in.triangle, pack2x16unorm(b.yz));
}
