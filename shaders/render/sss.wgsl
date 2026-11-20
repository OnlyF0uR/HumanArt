// render/sss.wgsl: the separable screen-space subsurface scattering of skin [SHD-02] [SHD-05]
// (Jimenez et al. 2015, "Separable Subsurface Scattering"). Two passes over the diffuse target
// (lighting for unit albedo, shadows included, so the shadow terminator softens by the profile and
// not by a shadow blur): `fs_sss_h` blurs along x into a temporary, `fs_sss_v` along y into the
// diffuse the composite reads. The geometry pass marks skin: the diffuse alpha holds the view
// depth (metres) and the specular alpha the profile scale; the other pixels pass through. One
// filtered fetch per tap brings both the colour and the depth of the tap's surface.
//
// The kernel (group 1, binding 3) is sss::separable_kernel: per tap the offset in mm at profile
// scale 1 and the red, green, blue weights, each channel summing to 1. Offsets become pixels
// through the surface distance one pixel covers along the blur axis at the centre (from the depth
// slope across the two neighbours, so the kernel shrinks on surfaces seen at a grazing angle: the
// local depth gradient). A tap that lands on another surface takes the centre's value, which keeps
// the weights' sum and stops light bleeding across the edge: not skin, or farther than
// DEPTH_TOLERANCE_MM (times the profile scale, fading over as much again) from the centre's
// tangent plane along the view. The tangent-plane test is what makes the weights normal-aware
// (jaw over neck, ear over head, nose over cheek: the other surface leaves the plane) without a
// normal fetch per tap. The CPU reference is sss::blur_pass_reference.

#include "lib/math.wgsl"
#include "render/frame.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var source_tex: texture_2d<f32>;
@group(1) @binding(1) var scale_tex: texture_2d<f32>;
@group(1) @binding(2) var source_sampler: sampler;
@group(1) @binding(3) var<uniform> kernel: array<vec4<f32>, 25>;

// Mirrors sss::KERNEL_TAPS and sss::KERNEL_RANGE_MM.
const KERNEL_TAPS: u32 = 25u;
const KERNEL_RANGE_MM: f32 = 8.2;
// Distance from the centre's tangent plane (mm per unit of profile scale) up to which a tap counts
// as the same surface; it fades to foreign over as much again. 3 mm keeps the sagitta of the
// kernel on a nose-sized curvature (about 3 mm at 8 mm from the centre on a 10 mm radius).
const DEPTH_TOLERANCE_MM: f32 = 3.0;

@vertex
fn vs_sss(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_position(index);
}

// The world position of a pixel centre at a reversed-Z depth.
fn world_at(position: vec2<f32>, depth: f32) -> vec3<f32> {
    let p = frame.inv_view_proj * vec4<f32>(pixel_to_ndc(position), depth, 1.0);
    return p.xyz / p.w;
}

// The view depth (metres along the view axis) of a world position: clip w of a perspective.
fn view_depth(world: vec3<f32>) -> f32 {
    return (frame.unjittered_view_proj * vec4<f32>(world, 1.0)).w;
}

fn blur(position: vec4<f32>, axis: vec2<i32>) -> vec4<f32> {
    let p = vec2<i32>(position.xy);
    let centre = textureLoad(source_tex, p, 0);
    let scale = textureLoad(scale_tex, p, 0).w;
    // Alpha is the view depth on skin, 0 elsewhere.
    let z = centre.w;
    if (scale <= 0.0 || z <= 0.0) {
        return centre;
    }
    let size = vec2<i32>(textureDimensions(source_tex));
    // The view-depth slope along the axis (metres per pixel) from the two neighbours when both are
    // skin, else none (an edge: the taps then test against the plane facing the camera).
    let z_minus = textureLoad(source_tex, clamp(p - axis, vec2<i32>(0), size - 1), 0).w;
    let z_plus = textureLoad(source_tex, clamp(p + axis, vec2<i32>(0), size - 1), 0).w;
    var slope = 0.0;
    if (z_minus > 0.0 && z_plus > 0.0) {
        // The flatter side: next to an edge the other side's step is not the surface's slope.
        let before = z - z_minus;
        let after = z_plus - z;
        slope = select(after, before, abs(before) < abs(after));
    }
    // Surface distance per pixel: the step across the view at this depth (the step on the near
    // plane scaled by depth) and the depth slope.
    let near_point = world_at(position.xy, 1.0);
    let near_step = length(world_at(position.xy + vec2<f32>(axis), 1.0) - near_point);
    let across = near_step * z / max(view_depth(near_point), 1e-9);
    let metres_per_pixel = sqrt(across * across + slope * slope);
    let px_per_unit = scale * 0.001 / max(metres_per_pixel, 1e-9);
    // A kernel narrower than half a pixel changes nothing.
    if (KERNEL_RANGE_MM * px_per_unit < 0.5) {
        return centre;
    }
    let step = vec2<f32>(axis);
    let tolerance = DEPTH_TOLERANCE_MM * scale;
    var sum = vec3<f32>(0.0);
    for (var i = 0u; i < KERNEL_TAPS; i++) {
        let tap = kernel[i];
        let offset = tap.x * px_per_unit;
        let uv = (position.xy + step * offset) * frame.viewport.zw;
        let colour = textureSampleLevel(source_tex, source_sampler, uv, 0.0);
        // How far the tap's surface lies from the centre's tangent plane along the view, mm (a
        // tap on another surface or off the skin, whose alpha is 0, lies far from it).
        let off_plane = abs(colour.w - (z + slope * offset)) * 1000.0;
        let foreign = saturate((off_plane - tolerance) / tolerance);
        sum += mix(colour.rgb, centre.rgb, foreign) * tap.yzw;
    }
    return vec4<f32>(sum, z);
}

@fragment
fn fs_sss_h(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return blur(position, vec2<i32>(1, 0));
}

@fragment
fn fs_sss_v(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return blur(position, vec2<i32>(0, 1));
}
