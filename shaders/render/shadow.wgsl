// render/shadow.wgsl: the shadow-map pass [RND-03] item 4: depth only, reversed-Z, one draw per
// object into one layer of the shadow atlas. The light's world-to-clip matrix comes from the
// uniform ring (group 0), fitted to the character by crates/render/src/light.rs.

#include "render/scene.wgsl"

@group(1) @binding(0) var<storage, read> positions: array<f32>;
@group(1) @binding(1) var<storage, read> prev_positions: array<f32>;
@group(1) @binding(2) var<storage, read> normals: array<f32>;
@group(1) @binding(3) var<storage, read> uvs: array<f32>;
@group(1) @binding(4) var<storage, read> objects: array<Object>;
@group(1) @binding(5) var<storage, read> indices: array<u32>;

struct ShadowView {
    view_proj: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> shadow_view: ShadowView;

@vertex
fn vs_shadow(@builtin(vertex_index) vertex: u32, @builtin(instance_index) object_index: u32) -> @builtin(position) vec4<f32> {
    let object = objects[object_index];
    return shadow_view.view_proj * vec4<f32>(world_position(object, vertex), 1.0);
}
