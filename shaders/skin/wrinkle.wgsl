// skin/wrinkle.wgsl: wrinkle and tension map blending [SKN-30] driven by the edge strain of
// [EVAL-06] through the wrinkle drivers of [EXP-04]; the twin of
// humanart_model::skin_appearance::wrinkle::blend_texel.
//
// A wrinkle tile is three rgba16float array layers from `base = 3 * k` (k: the tile's wrinkle slot):
// (h0, su0, sv0, c0), (h1, su1, sv1, c1), (id0, id1, flatten0, flatten1), with h the height delta
// in mm at full fold, (su, sv) its slope along +U and +V, c the cavity delta and id = driver + 1
// (0: empty slot). The includer binds the maps and the activations buffer (one vec2 per driver:
// fold, stretch), looks the activations up with `wrinkle_driver`, and calls `wrinkle_blend`:
// height, slope and cavity scale with the fold; the micro-detail contrast is multiplied by
// clamp(1 - sum stretch * flatten, 0, 1).

struct WrinkleTexel {
    slot0: vec4<f32>,
    slot1: vec4<f32>,
    info: vec4<f32>,
}

struct WrinkleSample {
    height: f32,
    slope: vec2<f32>,
    cavity: f32,
    detail_scale: f32,
}

// The three layers of texel `p` of the wrinkle tile starting at array layer `base`.
fn wrinkle_fetch(maps: texture_2d_array<f32>, base: i32, p: vec2<i32>) -> WrinkleTexel {
    return WrinkleTexel(
        textureLoad(maps, p, base, 0),
        textureLoad(maps, p, base + 1, 0),
        textureLoad(maps, p, base + 2, 0),
    );
}

// The driver + 1 of both slots (0: empty).
fn wrinkle_driver(t: WrinkleTexel) -> vec2<u32> {
    return vec2<u32>(round(max(t.info.xy, vec2<f32>(0.0))));
}

// Blends a texel with the (fold, stretch) activations of its two slots' drivers (zero for an empty
// slot).
fn wrinkle_blend(t: WrinkleTexel, a0: vec2<f32>, a1: vec2<f32>) -> WrinkleSample {
    var s: WrinkleSample;
    s.height = a0.x * t.slot0.x + a1.x * t.slot1.x;
    s.slope = a0.x * t.slot0.yz + a1.x * t.slot1.yz;
    s.cavity = a0.x * t.slot0.w + a1.x * t.slot1.w;
    s.detail_scale = clamp(1.0 - (a0.y * t.info.z + a1.y * t.info.w), 0.0, 1.0);
    return s;
}
