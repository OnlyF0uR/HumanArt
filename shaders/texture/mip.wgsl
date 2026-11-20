// texture/mip.wgsl: one level of mip generation [IO-05], the GPU twin of humanart_io::mip.
//
// Each dispatch makes level k + 1 from level k with the exact area filter (up to 3 x 3 taps when a
// size is odd; the weights are the same integer ratios as humanart_io::mip::taps). The filter runs on
// working values kept in an rgba32float chain (`work_out`), never on the quantised output, so the
// rounding of one level does not feed the next; every level is also resolved into the output
// texture (`out_unorm` or `out_float`, one entry point each).
//
// Variants are override constants [PERF-R5]:
// - COLOR_SPACE 0: linear colour, 1: sRGB colour (decoded to linear light, encoded after), with the
//   colour premultiplied by alpha while filtering; 2: data (every channel as stored); 3: normal map
//   packed as RGB = n * 0.5 + 0.5 and A = perceptual roughness. Normal working values are the
//   unnormalised mean normal and the mean GGX alpha^2; the output is the renormalised mean and the
//   roughness widened by the variance the shortened mean reveals (alpha'^2 = mean alpha^2 + 2 / kappa,
//   kappa = (3r - r^3) / (1 - r^2), the von Mises-Fisher fit of Neubelt and Pettineo 2013).
// - FROM_SOURCE true: `src` is the stored level 0 (decoded here); false: `src` is the previous level
//   of the working chain.

#include "lib/color.wgsl"

override COLOR_SPACE: u32 = 1u;
override FROM_SOURCE: bool = true;

struct Params {
    src_size: vec2<u32>,
    dst_size: vec2<u32>,
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var work_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var out_unorm: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var out_float: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params: Params;

const SPACE_LINEAR: u32 = 0u;
const SPACE_SRGB: u32 = 1u;
const SPACE_DATA: u32 = 2u;
const SPACE_NORMAL: u32 = 3u;

// A stored texel as a working value.
fn decode(c: vec4<f32>) -> vec4<f32> {
    if (!FROM_SOURCE) {
        return c;
    }
    if (COLOR_SPACE == SPACE_LINEAR) {
        return vec4<f32>(c.rgb * c.a, c.a);
    }
    if (COLOR_SPACE == SPACE_SRGB) {
        return vec4<f32>(srgb_to_linear_rgb(c.rgb) * c.a, c.a);
    }
    if (COLOR_SPACE == SPACE_NORMAL) {
        let n = c.xyz * 2.0 - 1.0;
        let len = length(n);
        let unit = select(vec3<f32>(0.0, 0.0, 1.0), n / len, len > 0.0);
        let r = clamp(c.a, 0.0, 1.0);
        return vec4<f32>(unit, r * r * r * r);
    }
    return c;
}

// The roughness alpha^2 a mean normal of length r adds (humanart_io::mip::variance_alpha_sq).
fn variance_alpha_sq(r: f32) -> f32 {
    if (r >= 1.0) {
        return 0.0;
    }
    let rr = max(r, 0.0);
    let kappa = (3.0 * rr - rr * rr * rr) / (1.0 - rr * rr);
    return select(min(2.0 / kappa, 1.0), 1.0, kappa <= 0.0);
}

// A working value as the texel to store.
fn resolve(v: vec4<f32>) -> vec4<f32> {
    if (COLOR_SPACE == SPACE_LINEAR || COLOR_SPACE == SPACE_SRGB) {
        let rgb = select(vec3<f32>(0.0), v.rgb / v.a, v.a > 0.0);
        if (COLOR_SPACE == SPACE_SRGB) {
            return vec4<f32>(linear_to_srgb_rgb(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0))), v.a);
        }
        return vec4<f32>(rgb, v.a);
    }
    if (COLOR_SPACE == SPACE_NORMAL) {
        let r = length(v.xyz);
        let n = select(vec3<f32>(0.0, 0.0, 1.0), v.xyz / r, r > 0.0);
        let rough = sqrt(sqrt(clamp(v.w + variance_alpha_sq(r), 0.0, 1.0)));
        return vec4<f32>(n * 0.5 + 0.5, rough);
    }
    return v;
}

// The source index and weight of tap k of destination texel x (humanart_io::mip::taps).
fn tap(x: u32, k: u32, src_n: u32, dst_n: u32) -> vec2<f32> {
    let lo = x * src_n;
    let hi = lo + src_n;
    let i = lo / dst_n + k;
    let a = max(i * dst_n, lo);
    let b = min((i + 1u) * dst_n, hi);
    if (b > a && i < src_n) {
        return vec2<f32>(f32(i), f32(b - a) / f32(src_n));
    }
    return vec2<f32>(0.0, 0.0);
}

// The area-filtered working value of destination texel `id`.
fn filtered(id: vec2<u32>) -> vec4<f32> {
    var sum = vec4<f32>(0.0);
    for (var ky = 0u; ky < 3u; ky++) {
        let ty = tap(id.y, ky, params.src_size.y, params.dst_size.y);
        if (ty.y == 0.0) {
            continue;
        }
        var row = vec4<f32>(0.0);
        for (var kx = 0u; kx < 3u; kx++) {
            let tx = tap(id.x, kx, params.src_size.x, params.dst_size.x);
            if (tx.y == 0.0) {
                continue;
            }
            let c = textureLoad(src, vec2<i32>(i32(tx.x), i32(ty.x)), 0);
            row += tx.y * decode(c);
        }
        sum += ty.y * row;
    }
    return sum;
}

@compute @workgroup_size(8, 8)
fn reduce_unorm(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.dst_size.x || id.y >= params.dst_size.y) {
        return;
    }
    let v = filtered(id.xy);
    textureStore(work_out, vec2<i32>(id.xy), v);
    textureStore(out_unorm, vec2<i32>(id.xy), resolve(v));
}

@compute @workgroup_size(8, 8)
fn reduce_float(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.dst_size.x || id.y >= params.dst_size.y) {
        return;
    }
    let v = filtered(id.xy);
    textureStore(work_out, vec2<i32>(id.xy), v);
    textureStore(out_float, vec2<i32>(id.xy), resolve(v));
}
