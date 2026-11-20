// render/taa.wgsl: temporal anti-aliasing and still accumulation [RND-03] item 10 [QUAL-01]
// [QUAL-02]. Reprojects the history with the motion vectors of the geometry pass (dilated to the
// nearest depth of the 3x3 neighbourhood) or, for the backdrop, with the previous camera; samples it
// with a 9-tap Catmull-Rom filter; clips it to the YCoCg box of the current 3x3 neighbourhood
// (TAA mode) and blends with weight max(minimum, 1 / (n + 1)). In accumulation mode (stills,
// converged review) the history is not clipped and the weight is exactly 1 / (n + 1), so n frames
// average to their mean.

#include "lib/math.wgsl"
#include "lib/packing.wgsl"
#include "render/frame.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;

@group(1) @binding(0) var current_tex: texture_2d<f32>;
@group(1) @binding(1) var history_tex: texture_2d<f32>;
@group(1) @binding(2) var surface_tex: texture_2d<u32>;
@group(1) @binding(3) var depth_tex: texture_depth_2d;
@group(1) @binding(4) var linear_sampler: sampler;

@vertex
fn vs_taa(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_position(index);
}

fn rgb_to_ycocg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}

fn ycocg_to_rgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// The history at `uv` through a Catmull-Rom filter built from 9 bilinear taps (the corner taps of
// the 16-tap kernel dropped), clamped to be non-negative.
fn sample_history(uv: vec2<f32>) -> vec3<f32> {
    let size = frame.viewport.xy;
    let position = uv * size;
    let center = floor(position - 0.5) + 0.5;
    let f = position - center;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let offset12 = w2 / w12;
    let t0 = (center - 1.0) * frame.viewport.zw;
    let t3 = (center + 2.0) * frame.viewport.zw;
    let t12 = (center + offset12) * frame.viewport.zw;
    var result = vec3<f32>(0.0);
    var weight = 0.0;
    var taps = array<vec2<f32>, 3>(t0, t12, t3);
    var wx = array<f32, 3>(w0.x, w12.x, w3.x);
    var wy = array<f32, 3>(w0.y, w12.y, w3.y);
    for (var j = 0u; j < 3u; j++) {
        for (var i = 0u; i < 3u; i++) {
            if ((i == 1u) || (j == 1u)) {
                let w = wx[i] * wy[j];
                result += textureSampleLevel(history_tex, linear_sampler, vec2<f32>(taps[i].x, taps[j].y), 0.0).rgb * w;
                weight += w;
            }
        }
    }
    return max(result / max(weight, 1e-6), vec3<f32>(0.0));
}

// The pixel motion (current minus previous position, pixels) at `p`, dilated to the nearest
// surface in the 3x3 neighbourhood; the backdrop is reprojected with the previous camera.
fn motion_at(position: vec2<f32>) -> vec2<f32> {
    let p = vec2<i32>(position);
    let last = vec2<i32>(frame.viewport.xy) - 1;
    var best = p;
    var best_depth = -1.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let q = clamp(p + vec2<i32>(x, y), vec2<i32>(0), last);
            let d = textureLoad(depth_tex, q, 0);
            if (d > best_depth) {
                best_depth = d;
                best = q;
            }
        }
    }
    if (best_depth > 0.0) {
        return unpack2x16float(textureLoad(surface_tex, best, 0).w);
    }
    // A point at infinity: reproject homogeneously through the previous camera.
    let ndc = pixel_to_ndc(position);
    let world = frame.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let prev = frame.prev_view_proj * world;
    if (abs(prev.w) < 1e-12) {
        return vec2<f32>(0.0);
    }
    let prev_ndc = prev.xy / prev.w;
    return (ndc - prev_ndc) * vec2<f32>(0.5, -0.5) * frame.viewport.xy;
}

@fragment
fn fs_taa(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(position.xy);
    let current = max(textureLoad(current_tex, p, 0).rgb, vec3<f32>(0.0));
    let n = frame.taa.z;
    if (frame.taa.w > 0.5 || n < 0.5) {
        return vec4<f32>(current, 1.0);
    }
    let motion = motion_at(position.xy);
    let prev_uv = (position.xy - motion) * frame.viewport.zw;
    if (any(prev_uv < vec2<f32>(0.0)) || any(prev_uv > vec2<f32>(1.0))) {
        return vec4<f32>(current, 1.0);
    }
    if (frame.taa.y > 0.5) {
        // Accumulation: the exact running mean of the jittered frames.
        let history = textureSampleLevel(history_tex, linear_sampler, prev_uv, 0.0).rgb;
        return vec4<f32>(mix(history, current, 1.0 / (n + 1.0)), 1.0);
    }
    var history = sample_history(prev_uv);
    // Neighbourhood box in YCoCg.
    let last = vec2<i32>(frame.viewport.xy) - 1;
    var lo = vec3<f32>(F32_MAX);
    var hi = vec3<f32>(-F32_MAX);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let q = clamp(p + vec2<i32>(x, y), vec2<i32>(0), last);
            let c = rgb_to_ycocg(max(textureLoad(current_tex, q, 0).rgb, vec3<f32>(0.0)));
            lo = min(lo, c);
            hi = max(hi, c);
        }
    }
    // Clip the history towards the box centre.
    let h = rgb_to_ycocg(history);
    let mid = 0.5 * (hi + lo);
    let extent = max(0.5 * (hi - lo), vec3<f32>(1e-6));
    let offset = h - mid;
    let units = abs(offset / extent);
    let m = max3(units);
    var clipped = h;
    if (m > 1.0) {
        clipped = mid + offset / m;
    }
    history = max(ycocg_to_rgb(clipped), vec3<f32>(0.0));
    let alpha = max(frame.taa.x, 1.0 / (n + 1.0));
    return vec4<f32>(mix(history, current, alpha), 1.0);
}
