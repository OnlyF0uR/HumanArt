// texture/bc_fetch.wgsl: test kernel [SKN-04]. Reads every texel of level 0 of a texture with
// textureLoad and stores it as four floats, so the GPU's hardware BC6H / BC7 decoder can be compared
// with the CPU reference decoders (humanart_io::bc) and encoder output can be checked by sampling.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> texels: array<vec4<f32>>;

@compute @workgroup_size(8, 8)
fn fetch(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(src, 0);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    texels[id.y * size.x + id.x] = textureLoad(src, vec2<i32>(id.xy), 0);
}
