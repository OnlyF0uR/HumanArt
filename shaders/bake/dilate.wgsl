// bake/dilate.wgsl: one pass of seam-padding dilation [BAKE-03], the GPU twin of
// humanart_tools::bake::dilate::dilate.
//
// A filled texel (mask != 0) is copied. An empty one with filled 8-neighbours becomes padding
// (mask 2): MODE 0 takes their mean (summed left, right, up, down, up-left, up-right, down-left,
// down-right and multiplied by the reciprocal of the count, the same literals as the CPU, so the
// result is bit-identical), MODE 1 copies the first filled one in that order. The host ping-pongs
// two value textures and two masks, one dispatch per pass.

override MODE: u32 = 0u;

struct Params {
    size: vec2<u32>,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var src_mask: texture_2d<u32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var dst_mask: texture_storage_2d<r32uint, write>;

const OFFSETS = array<vec2<i32>, 8>(
    vec2<i32>(-1, 0),
    vec2<i32>(1, 0),
    vec2<i32>(0, -1),
    vec2<i32>(0, 1),
    vec2<i32>(-1, -1),
    vec2<i32>(1, -1),
    vec2<i32>(-1, 1),
    vec2<i32>(1, 1),
);
const RECIPROCALS = array<f32, 8>(1.0, 0.5, 0.33333334, 0.25, 0.2, 0.16666667, 0.14285715, 0.125);

@compute @workgroup_size(8, 8)
fn dilate(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) {
        return;
    }
    let p = vec2<i32>(gid.xy);
    let size = vec2<i32>(params.size);
    let mask = textureLoad(src_mask, p, 0).x;
    let value = textureLoad(src, p, 0);
    if (mask != 0u) {
        textureStore(dst, p, value);
        textureStore(dst_mask, p, vec4<u32>(mask, 0u, 0u, 0u));
        return;
    }
    var sum = vec4<f32>(0.0);
    var count = 0u;
    for (var k = 0u; k < 8u; k += 1u) {
        let q = p + OFFSETS[k];
        if (any(q < vec2<i32>(0)) || any(q >= size)) {
            continue;
        }
        if (textureLoad(src_mask, q, 0).x == 0u) {
            continue;
        }
        let v = textureLoad(src, q, 0);
        if (MODE == 1u) {
            sum = v;
            count = 1u;
            break;
        }
        sum += v;
        count += 1u;
    }
    if (count > 0u) {
        textureStore(dst, p, sum * RECIPROCALS[count - 1u]);
        textureStore(dst_mask, p, vec4<u32>(2u, 0u, 0u, 0u));
    } else {
        textureStore(dst, p, value);
        textureStore(dst_mask, p, vec4<u32>(0u));
    }
}
