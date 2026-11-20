// render/composite.wgsl: the lighting composite [RND-03] item 9: diffuse lighting times albedo plus
// specular where there is geometry; the backdrop elsewhere [LIT-04] (a flat grey, already divided
// by the exposure, or the environment). The subsurface, eye and hair passes add their inputs here
// when they exist.

#include "lib/brdf.wgsl"
#include "lib/color.wgsl"
#include "lib/packing.wgsl"
#include "render/frame.wgsl"
#include "render/lighting.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(2) @binding(0) var env_specular: texture_cube<f32>;
@group(2) @binding(1) var env_sampler: sampler;
@group(2) @binding(2) var dfg_lut: texture_2d<f32>;
@group(2) @binding(3) var<storage, read> irradiance_sh: array<vec4<f32>, 9>;
@group(2) @binding(4) var shadow_atlas: texture_depth_2d_array;
@group(2) @binding(5) var shadow_sampler: sampler_comparison;

@group(1) @binding(0) var diffuse_tex: texture_2d<f32>;
@group(1) @binding(1) var specular_tex: texture_2d<f32>;
@group(1) @binding(2) var surface_tex: texture_2d<u32>;
@group(1) @binding(3) var depth_tex: texture_depth_2d;

@vertex
fn vs_composite(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_position(index);
}

// The backdrop radiance behind a pixel.
fn background(position: vec2<f32>) -> vec3<f32> {
    if (frame.background.w < 0.5) {
        return frame.background.rgb;
    }
    let d = rotate_y(pixel_direction(position), -frame.environment.y);
    return textureSampleLevel(env_specular, env_sampler, d, 0.0).rgb * frame.environment.x;
}

@fragment
fn fs_composite(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(position.xy);
    let depth = textureLoad(depth_tex, p, 0);
    if (depth <= 0.0) {
        return vec4<f32>(background(position.xy), 1.0);
    }
    let surface = textureLoad(surface_tex, p, 0);
    let albedo = srgb_to_linear_rgb(unpack4x8unorm(surface.y).rgb);
    let diffuse = textureLoad(diffuse_tex, p, 0).rgb;
    let specular = textureLoad(specular_tex, p, 0).rgb;
    return vec4<f32>(diffuse * albedo + specular, 1.0);
}
