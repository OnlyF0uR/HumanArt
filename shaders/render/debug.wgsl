// render/debug.wgsl: the debug and analysis views of section 8.5 that the core passes can show,
// written straight to the display target (data views are not tone mapped; lighting views are, with
// the frame's exposure and the PBR Neutral mapper). The view is frame.counters.y (numbering of
// `DebugView` in crates/render/src/renderer.rs). Views that need later work packages (SSS, strain,
// wrinkles, UV layout, ...) are added with them.

#include "lib/color.wgsl"
#include "lib/brdf.wgsl"
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

override ENCODE_SRGB: bool = true;

@group(1) @binding(0) var diffuse_tex: texture_2d<f32>;
@group(1) @binding(1) var specular_tex: texture_2d<f32>;
@group(1) @binding(2) var surface_tex: texture_2d<u32>;
@group(1) @binding(3) var depth_tex: texture_depth_2d;
@group(1) @binding(4) var scene_tex: texture_2d<f32>;
// The geometry pass's diffuse before the skin diffusion (diffuse_tex is after it).
@group(1) @binding(5) var irradiance_tex: texture_2d<f32>;

const VIEW_ALBEDO: u32 = 1u;
const VIEW_DIFFUSE: u32 = 2u;
const VIEW_SPECULAR: u32 = 3u;
const VIEW_IRRADIANCE: u32 = 4u;
const VIEW_NORMALS: u32 = 5u;
const VIEW_ROUGHNESS: u32 = 6u;
const VIEW_METALNESS: u32 = 7u;
const VIEW_THICKNESS: u32 = 8u;
const VIEW_OCCLUSION: u32 = 9u;
const VIEW_MATERIAL: u32 = 10u;
const VIEW_DEPTH: u32 = 11u;
const VIEW_MOTION: u32 = 12u;
const VIEW_SHADOW_MAP: u32 = 13u;
const VIEW_FALSE_COLOR: u32 = 14u;

@vertex
fn vs_debug(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_position(index);
}

// A display colour for a lighting quantity: exposed and tone mapped.
fn exposed(c: vec3<f32>) -> vec3<f32> {
    return tone_map(TONE_MAPPER_PBR_NEUTRAL, c, frame.eye_exposure.w);
}

// A data value in [0, 1] shown as is on the display (undoes the display encoding).
fn raw(v: vec3<f32>) -> vec3<f32> {
    return srgb_to_linear_rgb(saturate(v));
}

// A colour per small integer (material classes).
fn id_color(id: u32) -> vec3<f32> {
    let h = (id * 2654435761u) ^ (id >> 3u);
    return vec3<f32>(f32(h & 255u), f32((h >> 8u) & 255u), f32((h >> 16u) & 255u)) / 255.0 * 0.8 + 0.2;
}

// False colour of exposure: stops relative to middle grey (0.18) after exposure, in bands of one
// stop: below -5 black, then blue, cyan, green (within +-0.5 stop of middle grey), yellow, orange,
// red, above +5 white.
fn false_color(c: vec3<f32>) -> vec3<f32> {
    let l = luminance(c) * frame.eye_exposure.w;
    let stops = log2(max(l, 1e-9) / 0.18);
    if (stops < -5.0) { return vec3<f32>(0.0); }
    if (stops < -2.5) { return vec3<f32>(0.1, 0.1, 0.7); }
    if (stops < -0.5) { return vec3<f32>(0.1, 0.6, 0.8); }
    if (stops < 0.5) { return vec3<f32>(0.2, 0.75, 0.2); }
    if (stops < 2.5) { return vec3<f32>(0.9, 0.85, 0.1); }
    if (stops < 4.0) { return vec3<f32>(0.95, 0.5, 0.1); }
    if (stops < 5.0) { return vec3<f32>(0.9, 0.1, 0.1); }
    return vec3<f32>(1.0);
}

fn debug_color(position: vec2<f32>) -> vec3<f32> {
    let p = vec2<i32>(position);
    let view = frame.counters.y;
    let depth = textureLoad(depth_tex, p, 0);
    if (view == VIEW_FALSE_COLOR) {
        return false_color(textureLoad(scene_tex, p, 0).rgb);
    }
    if (view == VIEW_SHADOW_MAP) {
        let size = vec2<f32>(textureDimensions(shadow_atlas));
        let uv = position * frame.viewport.zw;
        let d = textureLoad(shadow_atlas, vec2<i32>(uv * size), 0, 0);
        return raw(vec3<f32>(saturate(d)));
    }
    if (view == VIEW_DEPTH) {
        // Reversed-Z: near is 1. Shown as a log distance ramp so far and near both read.
        if (depth <= 0.0) {
            return vec3<f32>(0.0);
        }
        return raw(vec3<f32>(saturate(log2(depth * 1024.0) / 10.0)));
    }
    if (depth <= 0.0) {
        return vec3<f32>(0.05);
    }
    let surface = textureLoad(surface_tex, p, 0);
    let albedo = unpack4x8unorm(surface.y);
    let params = unpack4x8unorm(surface.z);
    switch (view) {
        case VIEW_ALBEDO: { return srgb_to_linear_rgb(albedo.rgb); }
        case VIEW_DIFFUSE: {
            return exposed(textureLoad(diffuse_tex, p, 0).rgb * srgb_to_linear_rgb(albedo.rgb));
        }
        case VIEW_SPECULAR: { return exposed(textureLoad(specular_tex, p, 0).rgb); }
        case VIEW_IRRADIANCE: { return exposed(textureLoad(irradiance_tex, p, 0).rgb); }
        case VIEW_NORMALS: { return raw(unpack_oct16(surface.x) * 0.5 + 0.5); }
        case VIEW_ROUGHNESS: { return raw(vec3<f32>(params.x)); }
        case VIEW_METALNESS: { return raw(vec3<f32>(params.y)); }
        case VIEW_THICKNESS: { return raw(vec3<f32>(params.z)); }
        case VIEW_OCCLUSION: { return raw(vec3<f32>(params.w)); }
        case VIEW_MATERIAL: { return id_color(u32(albedo.w * 255.0 + 0.5)); }
        case VIEW_MOTION: {
            let m = unpack2x16float(surface.w);
            return raw(saturate(vec3<f32>(0.5 + 0.05 * m.x, 0.5 - 0.05 * m.y, 0.5)));
        }
        default: { return vec3<f32>(1.0, 0.0, 1.0); }
    }
}

@fragment
fn fs_debug(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    var c = saturate(debug_color(position.xy));
    if (ENCODE_SRGB) {
        c = linear_to_srgb_rgb(c);
    }
    return vec4<f32>(c, 1.0);
}
