// render/detail_mips.wgsl: the skin micro-detail exemplar as a mip-mapped slope texture for the
// geometry pass [SKN-20] [SKN-21] (crates/render/src/surface.rs).
//
// `convert` turns the generator's detail texture (skin/microdetail.wgsl `derive`: tangent normal
// x, y, height mm, cavity) into (s_x, s_y, s_x^2 + s_y^2, cavity) with s the height slope
// (mm per mm along +x and +y of the exemplar). `downsample` averages 2 x 2 texels into the next mip.
// Every channel is linear in the filter, so at any mip the slope variance the filter removed is
// z - (x^2 + y^2) (an isotropic LEAN-style moment, Olano & Baker 2010), which the geometry pass
// adds to the GGX alpha^2 so the pores' effect survives as roughness when they are smaller than a
// pixel.

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var destination: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn convert(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(destination);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }
    let d = textureLoad(source, gid.xy, 0);
    let nz = sqrt(max(1.0 - dot(d.xy, d.xy), 1e-4));
    let s = -d.xy / nz;
    textureStore(destination, gid.xy, vec4<f32>(s, dot(s, s), d.w));
}

@compute @workgroup_size(8, 8, 1)
fn downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(destination);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }
    let limit = textureDimensions(source) - vec2<u32>(1u);
    let p = gid.xy * 2u;
    let a = textureLoad(source, min(p, limit), 0);
    let b = textureLoad(source, min(p + vec2<u32>(1u, 0u), limit), 0);
    let c = textureLoad(source, min(p + vec2<u32>(0u, 1u), limit), 0);
    let e = textureLoad(source, min(p + vec2<u32>(1u, 1u), limit), 0);
    textureStore(destination, gid.xy, 0.25 * (a + b + c + e));
}
