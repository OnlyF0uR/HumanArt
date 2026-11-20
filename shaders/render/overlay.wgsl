// render/overlay.wgsl: the overlay pass [RND-03] item 11, drawn on the display target after tone
// mapping with the scene depth as a read-only test (reversed-Z, GreaterEqual) and alpha blending:
//   vs_lines: screen-space lines and round points (gizmos, landmarks, brush cursors) from the
//             primitive buffer, 6 vertices per primitive;
//   vs_wire:  the wireframe of an object, triangles pulled from the index buffer, edges from
//             barycentrics;
//   vs_selection: the selected object again, shaded as a rim highlight.
// Colours are display colours (sRGB-encoded, straight alpha). The unjittered camera is used: the
// image under the overlay is the TAA-resolved one.

#include "lib/math.wgsl"
#include "render/frame.wgsl"
#include "render/scene.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<storage, read> positions: array<f32>;
@group(1) @binding(1) var<storage, read> prev_positions: array<f32>;
@group(1) @binding(2) var<storage, read> normals: array<f32>;
@group(1) @binding(3) var<storage, read> uvs: array<f32>;
@group(1) @binding(4) var<storage, read> objects: array<Object>;
@group(1) @binding(5) var<storage, read> indices: array<u32>;

struct Primitive {
    // Start (world); w = width in pixels (diameter of a point).
    a: vec4<f32>,
    // End (world, = a for a point); w = kind (bits 0-3: 0 line, 1 point) + 16 if drawn on top.
    b: vec4<f32>,
    // Display colour, straight alpha.
    color: vec4<f32>,
}

@group(2) @binding(0) var<storage, read> primitives: array<Primitive>;

const KIND_LINE: u32 = 0u;
const KIND_POINT: u32 = 1u;
const KIND_WIRE: u32 = 2u;
const KIND_SELECTION: u32 = 3u;
const ON_TOP: u32 = 16u;
const WIRE_COLOR: vec4<f32> = vec4<f32>(0.85, 0.9, 1.0, 0.6);
const SELECTION_COLOR: vec4<f32> = vec4<f32>(1.0, 0.62, 0.15, 0.8);

struct OverlayOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) bary: vec3<f32>,
    @location(3) @interpolate(flat) kind: u32,
    @location(4) normal: vec3<f32>,
    @location(5) world: vec3<f32>,
}

fn empty_out() -> OverlayOut {
    var out: OverlayOut;
    out.clip = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.color = vec4<f32>(0.0);
    out.uv = vec2<f32>(0.0);
    out.bary = vec3<f32>(1.0);
    out.kind = KIND_LINE;
    out.normal = vec3<f32>(0.0, 1.0, 0.0);
    out.world = vec3<f32>(0.0);
    return out;
}

@vertex
fn vs_lines(@builtin(vertex_index) index: u32) -> OverlayOut {
    var out = empty_out();
    let prim = primitives[index / 6u];
    let corner = index % 6u;
    let kind = u32(prim.b.w);
    let a = frame.unjittered_view_proj * vec4<f32>(prim.a.xyz, 1.0);
    let b = frame.unjittered_view_proj * vec4<f32>(prim.b.xyz, 1.0);
    // Corners of the quad: (along, across) in {0, 1} x {-1, 1}.
    var along = array<f32, 6>(0.0, 1.0, 1.0, 0.0, 1.0, 0.0);
    var across = array<f32, 6>(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
    let t = along[corner];
    let s = across[corner];
    let half_width = 0.5 * max(prim.a.w, 1.0);
    let pixels = frame.viewport.xy;
    var clip = mix(a, b, t);
    if ((kind & 15u) == KIND_POINT) {
        let offset = vec2<f32>(2.0 * t - 1.0, s) * half_width;
        clip = vec4<f32>(a.xy + offset * 2.0 / pixels * a.w, a.z, a.w);
        out.uv = vec2<f32>(2.0 * t - 1.0, s);
    } else {
        let sa = a.xy / max(a.w, 1e-6) * pixels;
        let sb = b.xy / max(b.w, 1e-6) * pixels;
        let dir = safe_normalize(vec3<f32>(sb - sa, 0.0), vec3<f32>(1.0, 0.0, 0.0)).xy;
        let normal = vec2<f32>(-dir.y, dir.x);
        clip = vec4<f32>(clip.xy + normal * s * half_width * 2.0 / pixels * clip.w, clip.z, clip.w);
        out.uv = vec2<f32>(t, s);
    }
    if ((kind & ON_TOP) != 0u) {
        clip.z = clip.w;
    }
    out.clip = clip;
    out.color = prim.color;
    out.kind = kind & 15u;
    return out;
}

// One corner of the triangle `index / 3` of the object `object_index`, pulled through the index
// buffer.
fn pulled_corner(index: u32, object_index: u32) -> OverlayOut {
    var out = empty_out();
    let object = objects[object_index];
    let vertex = object.ids.w + indices[object.ids.z + index];
    let world = world_position(object, vertex);
    var clip = frame.unjittered_view_proj * vec4<f32>(world, 1.0);
    // Pull towards the camera a little so the surface under the line does not hide it.
    clip.z += 2e-5 * clip.w;
    out.clip = clip;
    let corner = index % 3u;
    out.bary = vec3<f32>(f32(corner == 0u), f32(corner == 1u), f32(corner == 2u));
    out.normal = world_normal(object, vertex);
    out.world = world;
    return out;
}

@vertex
fn vs_wire(@builtin(vertex_index) index: u32, @builtin(instance_index) object_index: u32) -> OverlayOut {
    var out = pulled_corner(index, object_index);
    out.kind = KIND_WIRE;
    out.color = WIRE_COLOR;
    return out;
}

@vertex
fn vs_selection(@builtin(vertex_index) index: u32, @builtin(instance_index) object_index: u32) -> OverlayOut {
    var out = pulled_corner(index, object_index);
    out.kind = KIND_SELECTION;
    out.color = SELECTION_COLOR;
    return out;
}

@fragment
fn fs_overlay(in: OverlayOut) -> @location(0) vec4<f32> {
    // Derivatives first, in uniform control flow.
    let width = fwidth(in.bary);
    let uv_width = fwidth(in.uv);
    var color = in.color;
    if (in.kind == KIND_POINT) {
        let r = length(in.uv);
        let edge = max(uv_width.x, 1e-4);
        color.a *= 1.0 - smoothstep(1.0 - edge, 1.0 + edge, r);
    } else if (in.kind == KIND_LINE) {
        let edge = max(uv_width.y, 1e-4);
        color.a *= 1.0 - smoothstep(1.0 - edge, 1.0 + edge, abs(in.uv.y));
    } else if (in.kind == KIND_WIRE) {
        let d = in.bary / max(width, vec3<f32>(1e-6));
        let nearest = min3(d);
        color.a *= 1.0 - smoothstep(0.5, 1.5, nearest);
    } else {
        let n = safe_normalize(in.normal, vec3<f32>(0.0, 1.0, 0.0));
        let v = safe_normalize(frame.eye_exposure.xyz - in.world, n);
        let rim = 1.0 - abs(dot(n, v));
        color.a *= 0.15 + 0.85 * rim * rim * rim;
    }
    if (color.a <= 0.002) {
        discard;
    }
    return color;
}
