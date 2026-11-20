// post/tonemap.wgsl: the display transform [COL-02] [LIT-03]: a full-screen triangle that reads the
// scene-referred HDR colour, applies the exposure and the tone mapper, and encodes for the target.
//
// Variants are override constants [PERF-R5]: TONE_MAPPER picks the mapper (lib/color.wgsl
// TONE_MAPPER_* numbering), ENCODE_SRGB applies the sRGB curve in the shader (true for a UNORM
// target, false for an *Srgb target, whose hardware encodes on write).

#include "lib/color.wgsl"

override TONE_MAPPER: u32 = 0u;
override ENCODE_SRGB: bool = true;

struct TonemapParams {
    // The exposure scale (exposure_from_ev100 of the camera).
    exposure: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var<uniform> tonemap: TonemapParams;

// One triangle covering the viewport: vertices (-1, -1), (3, -1), (-1, 3).
@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index & 1u) << 2u) - 1.0;
    let y = f32((index & 2u) << 1u) - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

// The HDR texel under the pixel, exposed, tone mapped and encoded.
@fragment
fn fs_tonemap(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let hdr_color = textureLoad(hdr, vec2<i32>(position.xy), 0).rgb;
    var mapped = tone_map(TONE_MAPPER, hdr_color, tonemap.exposure);
    if (ENCODE_SRGB) {
        mapped = linear_to_srgb_rgb(mapped);
    }
    return vec4<f32>(mapped, 1.0);
}
