// skin/page_copy.wgsl: one streaming page from the editor's resolved skin set [SKN-03].
//
// The source is the compositor's resolved array (rgba16float, layer tile * 4 + channel) or its
// specular array (rgba8unorm, layer tile). A page (x, y) of mip m covers the source square of
// side page << m from (x, y) * (page << m); every page texel is the box average of its
// (1 << m)^2 source texels (texels past the source edge are clamped to it). The result goes to a
// page-sized staging texture of the pool's format, which the caller copies into the pool slot.

struct Copy {
    origin: vec2<u32>,
    layer: u32,
    scale_log2: u32,
    page: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> copy: Copy;
@group(0) @binding(1) var source: texture_2d_array<f32>;
@group(0) @binding(2) var page_f16: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var page_unorm8: texture_storage_2d<rgba8unorm, write>;

fn box_average(p: vec2<u32>) -> vec4<f32> {
    let size = textureDimensions(source);
    let s = 1u << copy.scale_log2;
    let base = copy.origin + p * s;
    var sum = vec4<f32>(0.0);
    for (var j = 0u; j < s; j += 1u) {
        for (var i = 0u; i < s; i += 1u) {
            let q = min(base + vec2<u32>(i, j), size - vec2<u32>(1u));
            sum += textureLoad(source, vec2<i32>(q), i32(copy.layer), 0);
        }
    }
    return sum / f32(s * s);
}

@compute @workgroup_size(8, 8, 1)
fn copy_f16(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= copy.page || gid.y >= copy.page) {
        return;
    }
    textureStore(page_f16, vec2<i32>(gid.xy), box_average(gid.xy));
}

@compute @workgroup_size(8, 8, 1)
fn copy_unorm8(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= copy.page || gid.y >= copy.page) {
        return;
    }
    textureStore(page_unorm8, vec2<i32>(gid.xy), box_average(gid.xy));
}
