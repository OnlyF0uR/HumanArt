// eye/probe.wgsl: GPU test kernel for eye/optics.wgsl [AC-WP16-2]: traces one ray per case through
// the analytic eye and writes what it sees, for comparison with the f64 reference
// (humanart_model::eye::EyeGeometry) in crates/render/tests/eyes_mouth.rs.
//
// shape: (centre, radius), (axis, pupil), (up, limbus), (cornea radius, back radius, thickness,
// iris depth), (cornea ior, aqueous ior, -, -). cases: per case (origin, -), (direction, -).
// out: per case (kind, iris point), (surface point, distance from the axis on the iris plane).

#include "lib/math.wgsl"
#include "eye/optics.wgsl"

struct ProbeShape {
    centre_radius: vec4<f32>,
    axis_pupil: vec4<f32>,
    up_limbus: vec4<f32>,
    cornea: vec4<f32>,
    iors: vec4<f32>,
}

@group(0) @binding(0) var<uniform> shape: ProbeShape;
@group(0) @binding(1) var<storage, read> cases: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;

@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (2u * i + 1u >= arrayLength(&cases)) {
        return;
    }
    let s = eye_shape(shape.centre_radius, shape.axis_pupil, shape.up_limbus, shape.cornea, shape.iors.xy);
    let t = eye_trace(s, cases[2u * i].xyz, normalize(cases[2u * i + 1u].xyz));
    out[2u * i] = vec4<f32>(f32(t.kind), t.iris);
    out[2u * i + 1u] = vec4<f32>(t.point, t.radius);
}
