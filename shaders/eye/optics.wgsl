// eye/optics.wgsl: the analytic eyeball and the refraction through the cornea onto the iris
// plane [EYE-01] [EYE-02]. The f32 port of `humanart_model::eye::optics::EyeGeometry` (the f64
// reference); eye/probe.wgsl runs it against that reference [AC-WP16-2].
//
// World space, metres. The eyeball centre is `centre`, the gaze `axis`; heights (`_z`) are along
// the axis from the centre. The anterior corneal sphere (radius `cornea_radius`, centre at
// `cornea_centre_z`) caps the scleral sphere above the limbus circle (`limbus_z`); the posterior
// corneal sphere (`back_radius`, `back_centre_z`) bounds the aqueous; the iris is the plane at
// `iris_z`, the pupil a hole of radius `pupil` in it.

struct EyeShape {
    centre: vec3<f32>,
    radius: f32,
    axis: vec3<f32>,
    pupil: f32,
    // Unit, perpendicular to the axis: the iris plane's up.
    up: vec3<f32>,
    limbus: f32,
    cornea_radius: f32,
    back_radius: f32,
    limbus_z: f32,
    cornea_centre_z: f32,
    apex_z: f32,
    back_centre_z: f32,
    iris_z: f32,
    cornea_ior: f32,
    aqueous_ior: f32,
}

// Builds the shape: centre and eyeball radius, gaze axis and pupil radius, head up and limbus
// radius, (anterior and posterior corneal radii, central thickness, iris depth below the apex),
// (cornea, aqueous) indices. Lengths in metres.
fn eye_shape(centre_radius: vec4<f32>, axis_pupil: vec4<f32>, up_limbus: vec4<f32>, cornea: vec4<f32>, iors: vec2<f32>) -> EyeShape {
    var s: EyeShape;
    s.centre = centre_radius.xyz;
    s.radius = centre_radius.w;
    let axis_len = length(axis_pupil.xyz);
    s.axis = select(vec3<f32>(0.0, 0.0, 1.0), axis_pupil.xyz / axis_len, axis_len > 1e-6);
    s.pupil = axis_pupil.w;
    var up = up_limbus.xyz - s.axis * dot(up_limbus.xyz, s.axis);
    let up_len = length(up);
    if (up_len < 1e-6) {
        up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(s.axis.y) > 0.9);
        up = up - s.axis * dot(up, s.axis);
    }
    s.up = normalize(up);
    s.limbus = up_limbus.w;
    s.cornea_radius = cornea.x;
    s.back_radius = cornea.y;
    s.limbus_z = sqrt(max(s.radius * s.radius - s.limbus * s.limbus, 0.0));
    s.cornea_centre_z = s.limbus_z - sqrt(max(s.cornea_radius * s.cornea_radius - s.limbus * s.limbus, 0.0));
    s.apex_z = s.cornea_centre_z + s.cornea_radius;
    s.back_centre_z = s.apex_z - cornea.z - s.back_radius;
    s.iris_z = s.apex_z - cornea.w;
    s.cornea_ior = iors.x;
    s.aqueous_ior = iors.y;
    return s;
}

const EYE_MISS: u32 = 0u;
const EYE_SCLERA: u32 = 1u;
const EYE_CORNEA: u32 = 2u;

// What a ray sees on the eye. For the cornea: the iris plane hit, its coordinates in the plane
// (along up x axis, along up) and distance from the axis.
struct EyeTrace {
    kind: u32,
    point: vec3<f32>,
    normal: vec3<f32>,
    iris: vec3<f32>,
    iris_local: vec2<f32>,
    radius: f32,
}

// Ray parameters (nearer, farther) where `o + t d` (unit d) meets the sphere; z = 1 when it does.
fn eye_sphere_hits(o: vec3<f32>, d: vec3<f32>, c: vec3<f32>, r: f32) -> vec3<f32> {
    let oc = o - c;
    let b = dot(oc, d);
    let cc = dot(oc, oc) - r * r;
    let disc = b * b - cc;
    if (disc < 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let s = sqrt(disc);
    return vec3<f32>(-b - s, -b + s, 1.0);
}

// Height of a point along the axis.
fn eye_height(s: EyeShape, p: vec3<f32>) -> f32 {
    return dot(p - s.centre, s.axis);
}

// From a point on the anterior cornea (outward unit `normal`) and the unit incoming direction:
// fills `iris`, `iris_local` and `radius` of `t` (kind stays as given); kind becomes EYE_MISS
// when the ray cannot reach the iris plane.
fn eye_through_cornea(s: EyeShape, t_in: EyeTrace, dir: vec3<f32>) -> EyeTrace {
    var t = t_in;
    let inside = refract(dir, t.normal, 1.0 / s.cornea_ior);
    if (dot(inside, inside) < 0.5) {
        t.kind = EYE_MISS;
        return t;
    }
    let cp = s.centre + s.axis * s.back_centre_z;
    var o = t.point;
    var d = inside;
    let dz = dot(d, s.axis);
    let to_plane = select(-1.0, (s.iris_z - eye_height(s, o)) / dz, dz < -1e-7);
    let back = eye_sphere_hits(o, d, cp, s.back_radius);
    if (back.z > 0.5 && back.x > 0.0 && (to_plane < 0.0 || back.x < to_plane)) {
        let q = o + d * back.x;
        let n = normalize(q - cp);
        let d2 = refract(d, n, s.cornea_ior / s.aqueous_ior);
        if (dot(d2, d2) < 0.5) {
            t.kind = EYE_MISS;
            return t;
        }
        o = q;
        d = d2;
    }
    let dz2 = dot(d, s.axis);
    if (dz2 > -1e-7) {
        t.kind = EYE_MISS;
        return t;
    }
    t.iris = o + d * ((s.iris_z - eye_height(s, o)) / dz2);
    let rel = t.iris - s.centre;
    let side = cross(s.up, s.axis);
    t.iris_local = vec2<f32>(dot(rel, side), dot(rel, s.up));
    t.radius = length(t.iris_local);
    return t;
}

// Follows the ray `origin + t dir` (unit dir) into the eye: the sclera, or the cornea and the
// iris plane behind it.
fn eye_trace(s: EyeShape, origin: vec3<f32>, dir: vec3<f32>) -> EyeTrace {
    var t: EyeTrace;
    t.kind = EYE_MISS;
    t.point = origin;
    t.normal = -dir;
    t.iris = origin;
    t.iris_local = vec2<f32>(0.0);
    t.radius = 0.0;
    let ca = s.centre + s.axis * s.cornea_centre_z;
    var best = F32_MAX;
    let hc = eye_sphere_hits(origin, dir, ca, s.cornea_radius);
    if (hc.z > 0.5) {
        if (hc.x > 0.0 && eye_height(s, origin + dir * hc.x) >= s.limbus_z) {
            best = hc.x;
            t.kind = EYE_CORNEA;
        } else if (hc.y > 0.0 && eye_height(s, origin + dir * hc.y) >= s.limbus_z) {
            best = hc.y;
            t.kind = EYE_CORNEA;
        }
    }
    let hs = eye_sphere_hits(origin, dir, s.centre, s.radius);
    if (hs.z > 0.5) {
        if (hs.x > 0.0 && eye_height(s, origin + dir * hs.x) <= s.limbus_z && hs.x < best) {
            best = hs.x;
            t.kind = EYE_SCLERA;
        } else if (hs.x <= 0.0 && hs.y > 0.0 && eye_height(s, origin + dir * hs.y) <= s.limbus_z && hs.y < best) {
            best = hs.y;
            t.kind = EYE_SCLERA;
        }
    }
    if (t.kind == EYE_MISS) {
        return t;
    }
    t.point = origin + dir * best;
    if (t.kind == EYE_SCLERA) {
        t.normal = normalize(t.point - s.centre);
        return t;
    }
    t.normal = normalize(t.point - ca);
    return eye_through_cornea(s, t, dir);
}
