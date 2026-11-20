// skin/common.wgsl: the data layout shared by the skin compositor passes [SKN-01] [SKN-02], the
// GPU side of humanart_model::skin_appearance::{channels, zones, layers} and
// humanart_render::skin (which writes these buffers).
//
// Planes: one rgba16float array, layer = tile * 5 + plane (pigment, tissue, surface, relief,
// override); attribute a lives in plane a / 4, component a % 4.
// Inputs: one rgba8unorm array, layer = tile * 4 + k: k = 0 (thickness, cavity, -, -), k = 1..3
// zone masks 0..11.

const ATTRIBUTES: u32 = 20u;
const PLANES: u32 = 5u;
const ZONE_SLOTS: u32 = 12u;
const MAX_TILES: u32 = 64u;
const NO_ZONE: u32 = 0xffffffffu;
const UNUSED: u32 = 0xffu;

// Attribute indices used by the resolve.
const A_MELANIN: u32 = 0u;
const A_EUMELANIN: u32 = 1u;
const A_HAEMOGLOBIN: u32 = 2u;
const A_OXYGENATION: u32 = 3u;
const A_CAROTENE: u32 = 4u;
const A_EPIDERMIS: u32 = 5u;
const A_OILINESS: u32 = 6u;
const A_WETNESS: u32 = 7u;
const A_SPECULAR: u32 = 8u;
const A_ROUGH_PRIMARY: u32 = 9u;
const A_ROUGH_SECONDARY: u32 = 10u;
const A_LOBE_MIX: u32 = 11u;
const A_HEIGHT: u32 = 12u;
const A_NORMAL_X: u32 = 13u;
const A_NORMAL_Y: u32 = 14u;
const A_CAVITY: u32 = 15u;
const A_OVERRIDE_R: u32 = 16u;
const A_OVERRIDE_COVERAGE: u32 = 19u;

struct ZoneMod {
    offsets: array<vec4<f32>, 5>,
    // x: flush weight.
    flush: vec4<f32>,
}

struct Globals {
    base: array<vec4<f32>, 5>,
    zones: array<ZoneMod, 12>,
    // flush body, flush gain, wet roughness, wet specular gain.
    surface: vec4<f32>,
    // melanin max, blood min, blood max, epidermis min mm.
    mapping0: vec4<f32>,
    // epidermis max mm, carotene max mol/L, -, -.
    mapping1: vec4<f32>,
}

// One dispatch: the tile, the rectangle, the layer count.
struct Params {
    tile: u32,
    layer_count: u32,
    origin: vec2<u32>,
    size: vec2<u32>,
    _pad: vec2<u32>,
}

// The range of attribute `a` (humanart_model Attribute::range).
fn attribute_range(a: u32) -> vec2<f32> {
    if (a == A_NORMAL_X || a == A_NORMAL_Y) {
        return vec2<f32>(-1.0, 1.0);
    }
    if (a == A_HEIGHT) {
        return vec2<f32>(-2.0, 2.0);
    }
    if (a == A_SPECULAR) {
        return vec2<f32>(0.0, 4.0);
    }
    return vec2<f32>(0.0, 1.0);
}

fn base_value(g: Globals, a: u32) -> f32 {
    return g.base[a / 4u][a % 4u];
}
