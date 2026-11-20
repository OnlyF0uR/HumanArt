// ui/quad.wgsl: the UI's single pass [UI-02]. Every primitive is one instanced quad (a triangle
// strip of 4 vertices); the instance comes from a storage buffer indexed by the instance index and
// is evaluated as a signed distance field in the fragment shader.
//
// Instances are written by humanart_ui::paint::DrawList, in physical pixels, colours linear with
// straight alpha. params.z selects the kind (humanart_ui::paint::Kind):
//   0 shape     rounded rectangle, filled, bordered inside its edge when params.x (width) > 0
//   1 shadow    rounded rectangle blurred by params.x: alpha 1 - smoothstep(-blur, blur, d)
//   2 glyph     multi-channel distance field from the atlas layer params.w; params.x is the
//               atlas distance range in screen pixels, params.y a small outward shift of the
//               edge in screen pixels (stem darkening of small text)
//   3 image     the group-1 texture over rect, tinted by color, clipped to the rounded rectangle
//   4 segment   a line from rect.xy to rect.zw, half width params.x, round caps
//   5 gradient  rounded rectangle from color to color2, horizontal (params.x 0) or vertical (1)
// Every kind is clipped to the instance's clip rectangle, so a clip change never breaks a batch.
//
// ENCODE_SRGB (override, [PERF-R5]): true for a UNORM target. Colours are then encoded to sRGB
// before blending, so the UI blends in display space like the reference application; false for an
// *Srgb target, whose hardware encodes on write. The output is premultiplied alpha.

#include "lib/color.wgsl"

override ENCODE_SRGB: bool = true;

const KIND_SHADOW: u32 = 1u;
const KIND_GLYPH: u32 = 2u;
const KIND_IMAGE: u32 = 3u;
const KIND_SEGMENT: u32 = 4u;
const KIND_GRADIENT: u32 = 5u;

struct Globals {
    // The target size in physical pixels.
    viewport: vec2<f32>,
    _pad0: f32,
    _pad1: f32,
}

// Mirrors humanart_ui::paint::Instance (112 bytes).
struct Instance {
    rect: vec4<f32>,
    radii: vec4<f32>,
    color: vec4<f32>,
    color2: vec4<f32>,
    uv: vec4<f32>,
    clip: vec4<f32>,
    params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var atlas: texture_2d_array<f32>;
@group(0) @binding(3) var linear_sampler: sampler;
@group(1) @binding(0) var image: texture_2d<f32>;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) rect: vec4<f32>,
    @location(1) @interpolate(flat) radii: vec4<f32>,
    @location(2) @interpolate(flat) color: vec4<f32>,
    @location(3) @interpolate(flat) color2: vec4<f32>,
    @location(4) @interpolate(flat) uv: vec4<f32>,
    @location(5) @interpolate(flat) clip: vec4<f32>,
    @location(6) @interpolate(flat) params: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32, @builtin(instance_index) index: u32) -> VertexOut {
    let inst = instances[index];
    let kind = u32(inst.params.z);
    // The area the primitive can touch: its rectangle plus the anti-aliasing fringe, the blur or the
    // segment's width; glyph quads already include the distance range.
    var lo = inst.rect.xy - 1.0;
    var hi = inst.rect.zw + 1.0;
    if kind == KIND_SEGMENT {
        let m = inst.params.x + 1.0;
        lo = min(inst.rect.xy, inst.rect.zw) - m;
        hi = max(inst.rect.xy, inst.rect.zw) + m;
    } else if kind == KIND_SHADOW {
        lo = inst.rect.xy - (inst.params.x + 1.0);
        hi = inst.rect.zw + (inst.params.x + 1.0);
    } else if kind == KIND_GLYPH {
        lo = inst.rect.xy;
        hi = inst.rect.zw;
    }
    // Never rasterise outside the clip: no overdraw for clipped content.
    lo = max(lo, inst.clip.xy);
    hi = max(min(hi, inst.clip.zw), lo);
    let corner = vec2<f32>(f32(vertex & 1u), f32(vertex >> 1u));
    let p = mix(lo, hi, corner);

    var out: VertexOut;
    out.position = vec4<f32>(p / globals.viewport * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    out.rect = inst.rect;
    out.radii = inst.radii;
    out.color = inst.color;
    out.color2 = inst.color2;
    out.uv = inst.uv;
    out.clip = inst.clip;
    out.params = inst.params;
    return out;
}

// Signed distance from p to the rounded box rect (x0, y0, x1, y1) with per-corner radii
// (top-left, top-right, bottom-right, bottom-left; Y down). Negative inside.
fn rounded_box(p: vec2<f32>, rect: vec4<f32>, radii: vec4<f32>) -> f32 {
    let center = (rect.xy + rect.zw) * 0.5;
    let half_size = (rect.zw - rect.xy) * 0.5;
    let q0 = p - center;
    var r = select(radii.z, radii.y, q0.y < 0.0);
    if q0.x < 0.0 {
        r = select(radii.w, radii.x, q0.y < 0.0);
    }
    r = min(r, min(half_size.x, half_size.y));
    let q = abs(q0) - half_size + r;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

// Coverage of a pixel by the inside of a distance field with a one-pixel anti-aliasing ramp.
fn coverage(d: f32) -> f32 {
    return clamp(0.5 - d, 0.0, 1.0);
}

// The median of three values: the distance a multi-channel field encodes.
fn median3(a: f32, b: f32, c: f32) -> f32 {
    return max(min(a, b), min(max(a, b), c));
}

// A linear colour in the space blending happens in.
fn to_output(rgb: vec3<f32>) -> vec3<f32> {
    if ENCODE_SRGB {
        return linear_to_srgb_rgb(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)));
    }
    return rgb;
}

// A straight colour in output space, premultiplied and scaled by a coverage.
fn premultiplied(c: vec4<f32>, cover: f32) -> vec4<f32> {
    let a = c.a * cover;
    return vec4<f32>(to_output(c.rgb) * a, a);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let p = in.position.xy;
    if p.x < in.clip.x || p.y < in.clip.y || p.x >= in.clip.z || p.y >= in.clip.w {
        return vec4<f32>(0.0);
    }
    let kind = u32(in.params.z);
    let size = max(in.rect.zw - in.rect.xy, vec2<f32>(1e-3));
    let t = (p - in.rect.xy) / size;
    if kind == KIND_SHADOW {
        let d = rounded_box(p, in.rect, in.radii);
        let blur = max(in.params.x, 1e-3);
        return premultiplied(in.color, 1.0 - smoothstep(-blur, blur, d));
    }
    if kind == KIND_GLYPH {
        let uv = mix(in.uv.xy, in.uv.zw, t);
        let s = textureSampleLevel(atlas, linear_sampler, uv, i32(in.params.w), 0.0).rgb;
        let d = median3(s.r, s.g, s.b) - 0.5;
        return premultiplied(in.color, clamp(d * in.params.x + in.params.y + 0.5, 0.0, 1.0));
    }
    if kind == KIND_IMAGE {
        let uv = mix(in.uv.xy, in.uv.zw, t);
        let texel = textureSampleLevel(image, linear_sampler, uv, 0.0);
        let cover = coverage(rounded_box(p, in.rect, in.radii));
        return premultiplied(vec4<f32>(texel.rgb * in.color.rgb, texel.a * in.color.a), cover);
    }
    if kind == KIND_SEGMENT {
        let a = in.rect.xy;
        let ab = in.rect.zw - a;
        let h = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
        let d = length(p - a - ab * h) - in.params.x;
        return premultiplied(in.color, coverage(d));
    }
    let d = rounded_box(p, in.rect, in.radii);
    if kind == KIND_GRADIENT {
        let along = select(t.x, t.y, in.params.x > 0.5);
        let start = premultiplied(in.color, 1.0);
        let end = premultiplied(in.color2, 1.0);
        return mix(start, end, clamp(along, 0.0, 1.0)) * coverage(d);
    }
    // A shape: the fill, with the border band (params.x wide, inside the edge) replacing it.
    let fill = premultiplied(in.color, 1.0);
    var c = fill;
    if in.params.x > 0.0 {
        let border = premultiplied(in.color2, 1.0);
        c = mix(border, fill, coverage(d + in.params.x));
    }
    return c * coverage(d);
}
