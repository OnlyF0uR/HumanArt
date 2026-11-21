// skin/composite.wgsl: the first compositor pass [SKN-02]: for every texel of one dirty
// rectangle of one tile, the base values plus the zone offsets [SKN-11], then every layer bottom to
// top (fill or content, blend mode, opacity, zone mask, coverage) [SKN-12], clamped, written to the
// five planes. The twin of humanart_model::skin_appearance::layers::composite_texel.

#include "skin/common.wgsl"

struct Layer {
    // blend, opacity (f32 bits), zone (NO_ZONE: none), kind (0 fill, 1 content, 2 skipped).
    head: vec4<u32>,
    // Content: the attribute of each component (UNUSED: none).
    attrs: vec4<u32>,
    // x: fill attribute bit mask.
    fill_mask: vec4<u32>,
    fill: array<vec4<f32>, 5>,
}

@group(0) @binding(0) var planes: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(1) var atlas_values: texture_2d<f32>;
@group(0) @binding(2) var atlas_coverage: texture_2d<f32>;
@group(0) @binding(3) var inputs: texture_2d_array<f32>;
@group(0) @binding(4) var<storage, read> layers: array<Layer>;
// Per layer and tile: atlas x | y << 16, rect x | y << 16, rect w | h << 16, valid.
@group(0) @binding(5) var<storage, read> layer_tiles: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> globals: Globals;
@group(0) @binding(7) var<uniform> params: Params;

fn blend(mode: u32, v: f32, value: f32, w: f32) -> f32 {
    switch mode {
        case 1u: { return v + value * w; }
        case 2u: { return v * (1.0 + (value - 1.0) * w); }
        case 3u: { return v + (max(v, value) - v) * w; }
        case 4u: { return v + (min(v, value) - v) * w; }
        default: { return v + (value - v) * w; }
    }
}

fn unpack16(v: u32) -> vec2<u32> {
    return vec2<u32>(v & 0xffffu, v >> 16u);
}

@compute @workgroup_size(8, 8, 1)
fn composite(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) {
        return;
    }
    let p = params.origin + gid.xy;
    let tile = params.tile;
    var v: array<f32, 20>;
    for (var a = 0u; a < ATTRIBUTES; a += 1u) {
        v[a] = base_value(globals, a);
    }
    var masks: array<f32, 12>;
    for (var k = 0u; k < 3u; k += 1u) {
        let m = textureLoad(inputs, p, i32(tile * 4u + 1u + k), 0);
        masks[k * 4u] = m.x;
        masks[k * 4u + 1u] = m.y;
        masks[k * 4u + 2u] = m.z;
        masks[k * 4u + 3u] = m.w;
    }
    for (var z = 0u; z < ZONE_SLOTS; z += 1u) {
        let m = masks[z];
        if (m > 0.0) {
            for (var a = 0u; a < ATTRIBUTES; a += 1u) {
                v[a] += m * globals.zones[z].offsets[a / 4u][a % 4u];
            }
        }
    }
    // Low-frequency tone variation (humanart_model::skin_appearance::variation): the baked value in
    // the first input layer's blue channel, 0.5 neutral, scaled into melanin (attribute 0) and
    // haemoglobin (attribute 2).
    let tone = 2.0 * (textureLoad(inputs, p, i32(tile * 4u), 0).z - 0.5);
    v[0] += tone * globals.mapping1.z;
    v[2] += tone * globals.mapping1.w;
    for (var l = 0u; l < params.layer_count; l += 1u) {
        let layer = layers[l];
        let mode = layer.head.x;
        let opacity = bitcast<f32>(layer.head.y);
        let zone = layer.head.z;
        let kind = layer.head.w;
        if (kind == 2u) {
            continue;
        }
        var w = opacity;
        if (zone != NO_ZONE) {
            w *= masks[min(zone, ZONE_SLOTS - 1u)];
        }
        if (kind == 0u) {
            let bits = layer.fill_mask.x;
            for (var a = 0u; a < ATTRIBUTES; a += 1u) {
                if ((bits & (1u << a)) != 0u) {
                    v[a] = blend(mode, v[a], layer.fill[a / 4u][a % 4u], w);
                }
            }
            continue;
        }
        let lt = layer_tiles[l * MAX_TILES + tile];
        if (lt.w == 0u) {
            continue;
        }
        let rect_origin = unpack16(lt.y);
        let rect_size = unpack16(lt.z);
        if (p.x < rect_origin.x || p.y < rect_origin.y || p.x >= rect_origin.x + rect_size.x || p.y >= rect_origin.y + rect_size.y) {
            continue;
        }
        let q = unpack16(lt.x) + (p - rect_origin);
        let value = textureLoad(atlas_values, q, 0);
        let coverage = clamp(textureLoad(atlas_coverage, q, 0).r, 0.0, 1.0);
        let wc = w * coverage;
        for (var k = 0u; k < 4u; k += 1u) {
            let a = layer.attrs[k];
            if (a < ATTRIBUTES) {
                v[a] = blend(mode, v[a], value[k], wc);
            }
        }
    }
    for (var plane = 0u; plane < PLANES; plane += 1u) {
        var out: vec4<f32>;
        for (var c = 0u; c < 4u; c += 1u) {
            let a = plane * 4u + c;
            let r = attribute_range(a);
            let x = v[a];
            // NaN fails both comparisons and takes the floor value, as the CPU reference does.
            out[c] = select(max(r.x, 0.0), clamp(x, r.x, r.y), x == x && abs(x) < 3.0e38);
        }
        textureStore(planes, p, i32(tile * PLANES + plane), out);
    }
}
