// render/pick.wgsl: the picking buffer [RND-03]: the unjittered scene rasterised (usually scissored
// to the pixels under the cursor) with the pick id of the object and the depth of the hit, read back
// asynchronously. x = pick id (0 is nothing), y = bitcast of the reversed-Z depth.

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
}

@vertex
fn vs_pick(@builtin(vertex_index) vertex: u32, @builtin(instance_index) object_index: u32) -> PickOut {
    let object = objects[object_index];
    var out: PickOut;
    out.clip = frame.unjittered_view_proj * vec4<f32>(world_position(object, vertex), 1.0);
    out.id = object.ids.x;
    return out;
}

@fragment
fn fs_pick(in: PickOut) -> @location(0) vec2<u32> {
    return vec2<u32>(in.id, bitcast<u32>(in.clip.z));
}
