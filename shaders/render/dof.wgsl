// render/dof.wgsl: depth of field of a thin lens for stills (todo 1.9), one gather pass over the
// composited HDR image before the temporal accumulation.
//
// A point at view depth z, with the lens focused at s, spreads over a circle of confusion whose
// radius in pixels is K |1 - s / z| (K: the radius at infinity, from the focal length, the
// f-number, the focus distance and the sensor, crates/render/src/camera.rs `Lens`); the radius is
// capped at `frame.lens.z`.
//
// The gather (after Gustafsson 2018, "Bokeh depth of field in a single pass"): DOF_TAPS samples on a
// golden-angle spiral over the largest radius. A sample counts when its own circle reaches this
// pixel; one that lies behind this pixel counts no further than this pixel's own circle (twice
// it), since the pixel hides what is behind it. A sample that does not count is replaced by the
// mean so far. So a sharp pixel takes nothing from its neighbours and gives nothing to them: hair
// in focus keeps its edge, and the blurred background next to it is made of background only, with
// no halo of the subject's colour and no dark rim.
//
// The spiral turns per pixel and per frame; the temporal accumulation that follows makes a smooth
// disc of it. The Interactive tier's variant (fewer taps) turns it per 4 x 4 block of pixels, so
// neighbouring pixels read neighbouring texels (at 2560 x 1440: 1.0 ms instead of 2.2); the blocks'
// correlated noise would stay in an accumulated still, so the still tiers turn per pixel. Samples reach at most `frame.lens.z` pixels: a tiled still needs that much overlap.

#include "lib/math.wgsl"
#include "lib/sampling.wgsl"
#include "render/frame.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var color_tex: texture_2d<f32>;
@group(1) @binding(1) var depth_tex: texture_depth_2d;

override DOF_TAPS: u32 = 48u;

// The cosine and sine of the golden angle (pi (3 - sqrt 5)): the turn from one tap of the spiral to
// the next.
const GOLDEN_TURN: vec2<f32> = vec2<f32>(-0.7373688780783197, 0.6754902942615238);
// Depth probes that decide whether a sharp pixel needs the gather at all.
const PROBES: u32 = 6u;

@vertex
fn vs_dof(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_position(index);
}

// The radius of the circle of confusion, pixels, of a reversed-Z depth (0: infinitely far).
// `frame.lens.x` is the focus distance over the near distance, so s / z = lens.x * depth.
fn confusion(depth: f32) -> f32 {
    return min(frame.lens.y * abs(1.0 - frame.lens.x * depth), frame.lens.z);
}

@fragment
fn fs_dof(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let centre = textureLoad(color_tex, pixel, 0);
    if (frame.lens.w < 0.5) {
        return centre;
    }
    let size = vec2<i32>(textureDimensions(color_tex));
    let centre_depth = textureLoad(depth_tex, pixel, 0);
    let centre_radius = confusion(centre_depth);
    let reach = frame.lens.z;
    let seed = select(global_pixel(position.xy), floor(global_pixel(position.xy) * 0.25), DOF_TAPS < 48u);
    let turn = TAU * interleaved_gradient_noise(seed + 5.588238 * f32(frame.counters.x % 64u));
    // A sharp pixel with nothing blurred in front of it nearby (six probes on two rings) stays as
    // it is: most of the subject, for the price of seven depth fetches.
    if (centre_radius < 0.5) {
        var reached = false;
        for (var i = 0u; i < PROBES; i++) {
            let radius = reach * select(0.4, 0.8, i % 2u == 0u);
            let angle = f32(i) * (TAU / f32(PROBES)) + turn;
            let q = clamp(vec2<i32>(floor(position.xy + vec2<f32>(cos(angle), sin(angle)) * radius)), vec2<i32>(0), size - 1);
            let depth = textureLoad(depth_tex, q, 0);
            // Only what lies in front of this pixel can spread over it.
            reached = reached || (depth > centre_depth && confusion(depth) > 0.5);
        }
        if (!reached) {
            return centre;
        }
    }
    var sum = centre.rgb;
    var total = 1.0;
    let n = max(DOF_TAPS, 1u);
    let step = reach * inverseSqrt(f32(n));
    // The spiral's direction, turned by the golden angle from tap to tap (no sine and cosine per tap).
    var direction = vec2<f32>(cos(turn), sin(turn));
    for (var i = 0u; i < n; i++) {
        let radius = sqrt(f32(i) + 0.5) * step;
        let q = clamp(vec2<i32>(floor(position.xy + direction * radius)), vec2<i32>(0), size - 1);
        direction = vec2<f32>(
            direction.x * GOLDEN_TURN.x - direction.y * GOLDEN_TURN.y,
            direction.x * GOLDEN_TURN.y + direction.y * GOLDEN_TURN.x,
        );
        let depth = textureLoad(depth_tex, q, 0);
        var spread = confusion(depth);
        // Reversed-Z: a smaller depth is farther away.
        if (depth < centre_depth) {
            spread = min(spread, 2.0 * centre_radius);
        }
        let counts = smoothstep(radius - 0.5, radius + 0.5, spread);
        // A tap that does not count stands for the mean so far and needs no colour fetch.
        var tap = sum / total;
        if (counts > 0.0) {
            tap = mix(tap, textureLoad(color_tex, q, 0).rgb, counts);
        }
        sum += tap;
        total += 1.0;
    }
    return vec4<f32>(sum / total, centre.a);
}
