// lib/color.wgsl: the sRGB transfer curve, Rec.709 luminance, exposure and the display tone mappers.
//
// Colours are linear Rec.709 unless a name says srgb. Every function mirrors a CPU reference in
// humanart_math::color (transfer, spaces, tonemap) with the same constants; the GPU tests compare
// them. Derivations: docs/shader-notes.md, "Colour".

#include "lib/math.wgsl"

// The Rec.709 / sRGB luminance weights, row Y of the RGB-to-XYZ matrix derived from the primaries
// and the D65 white (0.3127, 0.3290): humanart_math::color::LUMINANCE_REC709_F32.
const LUMINANCE_REC709: vec3<f32> = vec3<f32>(0.21263901, 0.71516868, 0.07219232);

// The relative luminance Y of a linear Rec.709 colour.
fn luminance(rgb: vec3<f32>) -> f32 {
    return dot(rgb, LUMINANCE_REC709);
}

// The sRGB EOTF (IEC 61966-2-1) on one channel, extended as an odd function for negative input.
fn srgb_to_linear(v: f32) -> f32 {
    let a = abs(v);
    let l = select(pow((a + 0.055) / 1.055, 2.4), a / 12.92, a <= 0.04045);
    return select(l, -l, v < 0.0);
}

// The inverse sRGB EOTF on one channel, extended as an odd function for negative input.
fn linear_to_srgb(v: f32) -> f32 {
    let a = abs(v);
    let e = select(1.055 * pow(a, 1.0 / 2.4) - 0.055, a * 12.92, a <= 0.0031308);
    return select(e, -e, v < 0.0);
}

// srgb_to_linear per channel.
fn srgb_to_linear_rgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(srgb_to_linear(c.x), srgb_to_linear(c.y), srgb_to_linear(c.z));
}

// linear_to_srgb per channel.
fn linear_to_srgb_rgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(linear_to_srgb(c.x), linear_to_srgb(c.y), linear_to_srgb(c.z));
}

// The tone mappers, numbered in the order of humanart_math::color::ToneMapper::ALL. The stable
// names are "pbr-neutral", "aces-fitted" and "clamp".
const TONE_MAPPER_PBR_NEUTRAL: u32 = 0u;
const TONE_MAPPER_ACES_FITTED: u32 = 1u;
const TONE_MAPPER_CLAMP: u32 = 2u;

// The range EV100 is clamped to (humanart_math::color::EV100_RANGE).
const EV100_MIN: f32 = -32.0;
const EV100_MAX: f32 = 64.0;

// The scale from scene luminance to the display range: 1 / (1.2 * 2^EV100), EV100 clamped
// (Lagarde and de Rousiers 2014, section 4.5).
fn exposure_from_ev100(ev100: f32) -> f32 {
    return 1.0 / (1.2 * exp2(clamp(ev100, EV100_MIN, EV100_MAX)));
}

// Each component clamped to [0, 1]; NaN gives 0. The "clamp" tone mapper.
fn tone_map_clamp(c: vec3<f32>) -> vec3<f32> {
    return select(vec3<f32>(0.0), min(c, vec3<f32>(1.0)), c > vec3<f32>(0.0));
}

// Where PBR Neutral starts compressing (0.8 - 0.04) and how much it desaturates.
const NEUTRAL_START_COMPRESSION: f32 = 0.76;
const NEUTRAL_DESATURATION: f32 = 0.15;

// Khronos PBR Neutral (Khronos 3D Formats Working Group, 2024), the default tone mapper: colours
// below 0.76 pass through apart from a small black offset, highlights compress towards white.
fn pbr_neutral(rgb: vec3<f32>) -> vec3<f32> {
    let x = min3(rgb);
    let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
    let color = rgb - vec3<f32>(offset);
    let peak = max3(color);
    if (!(peak >= NEUTRAL_START_COMPRESSION) || peak > F32_MAX) {
        return tone_map_clamp(color);
    }
    let d = 1.0 - NEUTRAL_START_COMPRESSION;
    let new_peak = 1.0 - d * d / (peak + d - NEUTRAL_START_COMPRESSION);
    let scaled = color * (new_peak / peak);
    let g = 1.0 - 1.0 / (NEUTRAL_DESATURATION * (peak - new_peak) + 1.0);
    return tone_map_clamp(scaled + (vec3<f32>(new_peak) - scaled) * g);
}

// Hill's ACES fitted input matrix rows (linear sRGB -> AP1 with the RRT saturation) and output
// matrix rows (ODT saturation, AP1 -> linear sRGB), as published in BakingLab's ACES.hlsl.
const ACES_FITTED_INPUT_0: vec3<f32> = vec3<f32>(0.59719, 0.35458, 0.04823);
const ACES_FITTED_INPUT_1: vec3<f32> = vec3<f32>(0.07600, 0.90834, 0.01566);
const ACES_FITTED_INPUT_2: vec3<f32> = vec3<f32>(0.02840, 0.13383, 0.83777);
const ACES_FITTED_OUTPUT_0: vec3<f32> = vec3<f32>(1.60475, -0.53108, -0.07367);
const ACES_FITTED_OUTPUT_1: vec3<f32> = vec3<f32>(-0.10208, 1.10813, -0.00605);
const ACES_FITTED_OUTPUT_2: vec3<f32> = vec3<f32>(-0.00327, -0.07276, 1.07602);

// Where aces_rrt_odt_fit clamps its input: beyond it v * v would overflow f32 (from about 1.8e19,
// making inf / inf = NaN and so black), while the fit is already within 5e-7 of its asymptote
// 1 / 0.983729. The f64 CPU version needs no clamp; with it the two agree for every input.
const ACES_FIT_INPUT_LIMIT: f32 = 1e6;

// Hill's rational fit of the RRT + ODT curve, per component.
fn aces_rrt_odt_fit(v_in: vec3<f32>) -> vec3<f32> {
    let v = clamp(v_in, vec3<f32>(-ACES_FIT_INPUT_LIMIT), vec3<f32>(ACES_FIT_INPUT_LIMIT));
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

// Hill's ACES fitted filmic tone mapper.
fn aces_fitted(rgb: vec3<f32>) -> vec3<f32> {
    let ap1 = vec3<f32>(dot(ACES_FITTED_INPUT_0, rgb), dot(ACES_FITTED_INPUT_1, rgb), dot(ACES_FITTED_INPUT_2, rgb));
    let c = aces_rrt_odt_fit(ap1);
    return tone_map_clamp(vec3<f32>(dot(ACES_FITTED_OUTPUT_0, c), dot(ACES_FITTED_OUTPUT_1, c), dot(ACES_FITTED_OUTPUT_2, c)));
}

// `rgb * exposure` through the tone mapper `mapper` (a TONE_MAPPER_* value; unknown values clamp).
// Linear [0, 1] out; the sRGB encoding is a separate step.
fn tone_map(mapper: u32, rgb: vec3<f32>, exposure: f32) -> vec3<f32> {
    let exposed = rgb * exposure;
    switch mapper {
        case 0u: {
            return pbr_neutral(exposed);
        }
        case 1u: {
            return aces_fitted(exposed);
        }
        default: {
            return tone_map_clamp(exposed);
        }
    }
}
