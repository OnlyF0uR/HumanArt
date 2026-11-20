// hair/motion.wgsl: hair in the TAA inputs [HR-05]. Where the hair covers at least half of a
// pixel, its nearest fibre's motion replaces the G-buffer's (the surface target's w, the only
// channel the pipeline writes) and its depth the scene depth, so the TAA reprojects and dilates
// hair pixels with the strands' own motion (strand-aware history rejection: a hair pixel's history
// comes from where that strand was, and is clipped to the current neighbourhood like any pixel).

#include "lib/math.wgsl"
#include "lib/packing.wgsl"
#include "render/frame.wgsl"
#include "hair/common.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> view: HairView;

@group(1) @binding(0) var<storage, read> hair_depth: array<u32>;
@group(1) @binding(1) var<storage, read> hair_coverage: array<u32>;
@group(1) @binding(2) var<storage, read> hair_records: array<vec4<u32>>;

struct MotionOut {
    @location(0) surface: vec4<u32>,
    @builtin(frag_depth) depth: f32,
}

@vertex
fn vs_motion(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_position(index);
}

@fragment
fn fs_motion(@builtin(position) position: vec4<f32>) -> MotionOut {
    let pixel = vec2<u32>(position.xy);
    let index = pixel.y * u32(view.screen.x) + pixel.x;
    let bits = hair_depth[index];
    let alpha = hair_alpha(hair_coverage[index], hair_records[index]);
    if (bits == 0u || alpha < 0.5) {
        discard;
    }
    var out: MotionOut;
    out.surface = vec4<u32>(0u, 0u, 0u, hair_records[index].z);
    out.depth = bitcast<f32>(bits);
    return out;
}
