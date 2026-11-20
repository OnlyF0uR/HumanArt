// render/frame.wgsl: the per-frame constants of the renderer [RND-03]: camera matrices (jittered for
// rasterisation, unjittered for motion and reprojection), shadow matrices, exposure, viewport and
// tile, environment, backdrop, counters and the light array. Mirrors `FrameUniforms` in
// crates/render/src/uniforms.rs (1488 bytes); bound at group 0 from the uniform ring with a dynamic
// offset. Also the full-screen triangle every post pass draws.
//
// The including shader declares the binding:
//   @group(0) @binding(0) var<uniform> frame: Frame;

struct Light {
    // Linear colour times intensity (lux, candela or nits); w = kind (0 directional, 1 spot, 2 rect).
    color_kind: vec4<f32>,
    // Position; w = range (spot).
    position_range: vec4<f32>,
    // Unit direction the light travels; w = cos of the outer angle (spot).
    direction_cos_outer: vec4<f32>,
    // Right axis times half width (rect); w = cos of the inner angle (spot).
    right_cos_inner: vec4<f32>,
    // Up axis times half height (rect); w = source size (angular diameter, or radius).
    up_size: vec4<f32>,
    // x = shadow layer or -1, y = soft (1/0), z = normal bias in texels.
    shadow: vec4<f32>,
}

const MAX_LIGHTS: u32 = 8u;
const MAX_SHADOWED: u32 = 4u;
const LIGHT_DIRECTIONAL: u32 = 0u;
const LIGHT_SPOT: u32 = 1u;
const LIGHT_RECT: u32 = 2u;

struct Frame {
    // World to clip with the TAA jitter.
    view_proj: mat4x4<f32>,
    // World to clip without jitter.
    unjittered_view_proj: mat4x4<f32>,
    // The previous frame's unjittered world to clip.
    prev_view_proj: mat4x4<f32>,
    // Clip to world of `unjittered_view_proj`.
    inv_view_proj: mat4x4<f32>,
    // World to shadow clip, per atlas layer.
    shadow_view_proj: array<mat4x4<f32>, 4>,
    // Per layer: near, far, half extent (orthographic: radius; perspective: tan of the half angle),
    // penumbra (orthographic: width per metre of blocker distance; perspective: emitter size).
    shadow_params: array<vec4<f32>, 4>,
    // Per layer: 0 orthographic, 1 perspective.
    shadow_kind: vec4<f32>,
    // Eye position; w = exposure scale.
    eye_exposure: vec4<f32>,
    // Target width, height, 1 / width, 1 / height.
    viewport: vec4<f32>,
    // Tile origin in the full image (pixels), full width, full height.
    tile: vec4<f32>,
    // Current jitter (pixels), previous jitter.
    jitter: vec4<f32>,
    // Environment scale (nits), rotation about +Y (radians), specular mip count, light count.
    environment: vec4<f32>,
    // Backdrop radiance; w = 0 flat colour, 1 environment.
    background: vec4<f32>,
    // TAA: minimum blend weight, mode (0 TAA, 1 accumulate), frames accumulated, reset (1/0).
    taa: vec4<f32>,
    // Frame index, debug view, shadow map size, flags.
    counters: vec4<u32>,
    lights: array<Light, 8>,
}

// The pixel position in the full image (tiles add their origin): what per-pixel noise is seeded
// with, so a tiled still has the same noise as the untiled one [QUAL-04].
fn global_pixel(position: vec2<f32>) -> vec2<f32> {
    return floor(position) + frame.tile.xy;
}

// One triangle covering the viewport: vertices (-1, -1), (3, -1), (-1, 3).
fn fullscreen_position(index: u32) -> vec4<f32> {
    let x = f32((index & 1u) << 2u) - 1.0;
    let y = f32((index & 2u) << 1u) - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

// Normalised device coordinates of a pixel centre of the target (Y up).
fn pixel_to_ndc(position: vec2<f32>) -> vec2<f32> {
    let uv = position * frame.viewport.zw;
    return vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}

// The world-space direction through a pixel (unjittered), for the backdrop and reprojection.
fn pixel_direction(position: vec2<f32>) -> vec3<f32> {
    let ndc = pixel_to_ndc(position);
    let near = frame.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let far = frame.inv_view_proj * vec4<f32>(ndc, 1e-7, 1.0);
    return normalize(far.xyz / far.w - near.xyz / near.w);
}

// Rotates a direction about +Y by `angle` radians (the environment rotation).
fn rotate_y(d: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(c * d.x + s * d.z, d.y, -s * d.x + c * d.z);
}
