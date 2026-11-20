// render/scene.wgsl: the scene's vertex data at group 1, read by vertex pulling [RND-03] [RND-04].
//
// Attributes are separate streams (structure of arrays, [PERF-R2]): positions, the previous frame's
// positions (kept by the renderer, later written by the deformation pass, so motion vectors follow
// expressions and poses), normals, texture coordinates, the object table and the index buffer (for
// passes that pull triangles themselves, like the wireframe overlay). Mirrors `GpuObject` in
// crates/render/src/scene.rs. The including shader declares the bindings
// (group 1, bindings 0-5: positions, prev_positions, normals, uvs, objects, indices; see
// render/geometry.wgsl).

struct Object {
    // Object to world.
    model: mat4x4<f32>,
    // The previous frame's object to world.
    prev_model: mat4x4<f32>,
    // Inverse transpose of `model` (upper 3x3 used).
    normal_matrix: mat4x4<f32>,
    // Linear base colour; w = perceptual roughness.
    base_color_roughness: vec4<f32>,
    // Metalness, dielectric f0, thickness (0..1), material class.
    material: vec4<f32>,
    // Skin (class 1): secondary lobe roughness, lobe mix, profile scale, translucency
    // (crates/render/src/sss.rs, `SkinShading`).
    skin: vec4<f32>,
    // Pick id, flags (bit 0 selected), first index, base vertex (added to every pulled index).
    ids: vec4<u32>,
}

const OBJECT_SELECTED: u32 = 1u;

fn fetch_position(vertex: u32) -> vec3<f32> {
    let i = 3u * vertex;
    return vec3<f32>(positions[i], positions[i + 1u], positions[i + 2u]);
}

fn fetch_prev_position(vertex: u32) -> vec3<f32> {
    let i = 3u * vertex;
    return vec3<f32>(prev_positions[i], prev_positions[i + 1u], prev_positions[i + 2u]);
}

fn fetch_normal(vertex: u32) -> vec3<f32> {
    let i = 3u * vertex;
    return vec3<f32>(normals[i], normals[i + 1u], normals[i + 2u]);
}

fn fetch_uv(vertex: u32) -> vec2<f32> {
    let i = 2u * vertex;
    return vec2<f32>(uvs[i], uvs[i + 1u]);
}

// The world position of a vertex of an object.
fn world_position(object: Object, vertex: u32) -> vec3<f32> {
    let p = object.model * vec4<f32>(fetch_position(vertex), 1.0);
    return p.xyz;
}

// The world position of a vertex in the previous frame.
fn prev_world_position(object: Object, vertex: u32) -> vec3<f32> {
    let p = object.prev_model * vec4<f32>(fetch_prev_position(vertex), 1.0);
    return p.xyz;
}

// The world normal of a vertex (not normalised).
fn world_normal(object: Object, vertex: u32) -> vec3<f32> {
    let n = object.normal_matrix * vec4<f32>(fetch_normal(vertex), 0.0);
    return n.xyz;
}
