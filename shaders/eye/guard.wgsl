// eye/guard.wgsl: mirrors the CPU penetration guard's correction [EVAL-08] into the GPU
// deformation's outputs. The CPU reference evaluator runs the eyelid and lip guard after skinning
// and recomputes the normals (humanart_model::eval::penetration); the GPU deformation does not,
// so after its pass this kernel writes the corrected positions and normals of every guarded
// vertex over the deformer's (crates/render/src/eye/guard.rs, `GuardMirror`).
//
// `fixes` holds two vec4 per guarded vertex: (position, vertex index as bits) and (normal, 0).
// Positions and normals are packed f32 triples, as the deformer writes them.

struct GuardParams {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: GuardParams;
@group(0) @binding(1) var<storage, read> fixes: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> positions: array<f32>;
@group(0) @binding(3) var<storage, read_write> normals: array<f32>;

@compute @workgroup_size(64)
fn apply_fixes(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count || 2u * i + 1u >= arrayLength(&fixes)) {
        return;
    }
    let p = fixes[2u * i];
    let n = fixes[2u * i + 1u];
    let base = bitcast<u32>(p.w) * 3u;
    if (base + 2u >= arrayLength(&positions) || base + 2u >= arrayLength(&normals)) {
        return;
    }
    positions[base] = p.x;
    positions[base + 1u] = p.y;
    positions[base + 2u] = p.z;
    normals[base] = n.x;
    normals[base + 1u] = n.y;
    normals[base + 2u] = n.z;
}
