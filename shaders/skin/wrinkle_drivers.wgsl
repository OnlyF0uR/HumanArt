// skin/wrinkle_drivers.wgsl: wrinkle driver activations and wrinkle blending over a rectangle
// [SKN-30] [EVAL-06] [EXP-04].
//
// `activate`, one invocation per driver: fold = clamp(term + g_c * compression_r, 0, 1) and
// stretch = clamp(g_s * stretch_r, 0, 1), with `term` the expression part summed on the CPU, r the
// driver's region and (stretch_r, compression_r) the per-region strain the deformer's
// `deform/strain.wgsl` wrote (it never leaves the GPU). Non-finite inputs count as 0, as in
// humanart_model::skin_appearance::wrinkle::DriverTable::activate.
//
// `blend`, one invocation per texel of a rectangle of one wrinkle tile: the blended state
// (height, slope u, slope v, cavity) and (detail scale, 0, 0, 0) into two vec4 per texel of a
// storage buffer, for the wrinkle/strain inspection readback and the CPU twin test.

#include "skin/wrinkle.wgsl"

struct Params {
    driver_count: u32,
    region_count: u32,
    tile_slot: u32,
    _pad: u32,
    rect: vec4<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;

@group(1) @binding(0) var<storage, read> drivers: array<vec4<u32>>;
@group(1) @binding(1) var<storage, read> terms: array<f32>;
@group(1) @binding(2) var<storage, read> region_strain: array<vec2<f32>>;
@group(1) @binding(3) var<storage, read_write> activations: array<vec2<f32>>;

@group(1) @binding(4) var maps: texture_2d_array<f32>;
@group(1) @binding(5) var<storage, read> activations_in: array<vec2<f32>>;
@group(1) @binding(6) var<storage, read_write> blended: array<vec4<f32>>;

fn finite_or_zero(x: f32) -> f32 {
    return select(0.0, x, abs(x) <= 3.40282347e38);
}

@compute @workgroup_size(64, 1, 1)
fn activate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= params.driver_count) {
        return;
    }
    let record = drivers[d];
    var strain = vec2<f32>(0.0);
    if (record.x < params.region_count) {
        strain = region_strain[record.x];
    }
    let fold = finite_or_zero(terms[d]) + bitcast<f32>(record.y) * finite_or_zero(strain.y);
    let stretch = bitcast<f32>(record.z) * finite_or_zero(strain.x);
    let ok_fold = abs(fold) <= 3.40282347e38;
    let ok_stretch = abs(stretch) <= 3.40282347e38;
    activations[d] = vec2<f32>(
        select(0.0, clamp(fold, 0.0, 1.0), ok_fold),
        select(0.0, clamp(stretch, 0.0, 1.0), ok_stretch),
    );
}

fn driver_activation(id: u32) -> vec2<f32> {
    if (id == 0u || id > params.driver_count) {
        return vec2<f32>(0.0);
    }
    return activations_in[id - 1u];
}

@compute @workgroup_size(8, 8, 1)
fn blend(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = params.rect;
    if (gid.x >= r.z || gid.y >= r.w) {
        return;
    }
    let p = vec2<i32>(vec2<u32>(r.x + gid.x, r.y + gid.y));
    let t = wrinkle_fetch(maps, i32(params.tile_slot * 3u), p);
    let ids = wrinkle_driver(t);
    let s = wrinkle_blend(t, driver_activation(ids.x), driver_activation(ids.y));
    let i = gid.y * r.z + gid.x;
    blended[2u * i] = vec4<f32>(s.height, s.slope, s.cavity);
    blended[2u * i + 1u] = vec4<f32>(s.detail_scale, 0.0, 0.0, 0.0);
}
