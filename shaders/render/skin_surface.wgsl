// render/skin_surface.wgsl: the skin surface of the geometry pass (crates/render/src/surface.rs):
// the compositor's resolved set and zone masks [SKN-01] [SKN-11], micro-detail with
// histogram-preserving tiling oriented along the skin lines [SKN-20] [SKN-21], wrinkle and tension
// maps [SKN-30], and the layer terms of [SHD-06] (oil coat, lips), [SHD-07] (vellus fuzz),
// [SHD-08] (cavity occlusion) and [SHD-09] (micro-shadowing).
//
// The includer declares the group-3 bindings:
//   skin_resolved (texture_2d_array, rgba16float, layer 4 t + k: albedo + cavity, scatter radius +
//   baked thickness, meso normal x y + height + wetness, flush delta + oiliness),
//   skin_specular (texture_2d_array, rgba8unorm, layer t: specular / 4, roughness 1, roughness 2,
//   lobe mix), skin_inputs (texture_2d_array, rgba8unorm, layer 4 t + k: k = 0 baked thickness +
//   cavity, k = 1..3 zone mask slots), skin_detail (texture_2d, mip-mapped slopes from
//   render/detail_mips.wgsl), wrinkle_maps / wrinkle_activations / wrinkle_tile_slots
//   (skin/wrinkle.wgsl), clamp_sampler, repeat_sampler and `surface: SurfaceParams`.
//
// Texture coordinates of a surfaced mesh are u + tile, v (the tile of humanart_model's UV tile
// table in the integer part of u). Tangent frames come from screen-space derivatives (Schüler 2013,
// "Normal Mapping without Precomputed Tangents"), so no tangent stream is needed.
//
// Skin-line (Langer) orientation [SKN-20]: no authored field exists on the template yet; the
// per-region default used here runs the lines around the body's long axis (the object's +Y), i.e.
// horizontal on the forehead, neck and trunk and around the limbs, falling back to around +Z where
// the surface faces up or down. The exemplar's line direction (+y, see
// humanart_model::skin_appearance::microdetail) is rotated onto it per anti-tiling vertex, about
// that vertex, so a varying direction never shears the texture by more than one cell.

#include "skin/antitile.wgsl"
#include "skin/wrinkle.wgsl"
#include "skin/surface.wgsl"

const MAX_UV_TILES: u32 = 16u;
// GpuObject ids.y bit: the object reads the skin surface.
const OBJECT_SKIN_SURFACE: u32 = 2u;
const NO_WRINKLE_SLOT: u32 = 0xffffffffu;
// Reflectance at normal incidence of the sebum / sweat film (n about 1.42).
const COAT_F0: f32 = 0.03;
// A baked thickness byte of 1.0 means "no data"; below it, millimetres = value * this.
const BAKED_THICKNESS_RANGE_MM: f32 = 25.5;

struct SurfaceParams {
    // Per UV tile: x mm of skin per UV unit (0: not surfaced), y the resolved-set tile (< 0: none).
    tiles: array<vec4<f32>, 16>,
    // Exemplar side mm (0: no micro-detail), detail strength, micro-shadowing strength, flush 0..1.
    detail: vec4<f32>,
    // Oil coat per unit oiliness, coat roughness, lip coat, lip coat roughness.
    coat: vec4<f32>,
    // Vellus strength at a full fuzz zone, fuzz width (sheen roughness), lip roughness scale,
    // 1 / the measured profile's mean RMS radius in mm (crate::sss::reference_radius_mm).
    fuzz: vec4<f32>,
    // Anti-tiling seed, wrinkle maps bound (0 / 1), -, -.
    misc: vec4<u32>,
}

// The tangent frame of a pixel from screen-space derivatives (computed in uniform control flow).
struct SurfaceFrame {
    // Unit tangent along +U and bitangent along +V, in the tangent plane of `n`.
    t: vec3<f32>,
    b: vec3<f32>,
    // Whether the derivatives define a frame.
    valid: bool,
}

// Everything the shading reads from the surface at one pixel.
struct SkinSurfaceSample {
    albedo: vec3<f32>,
    // Macro occlusion (baked cavity x wrinkle cavity) and the pore-scale cavity.
    cavity: f32,
    micro_cavity: f32,
    // Multiplier of the material's f0.
    f0_scale: f32,
    roughness0: f32,
    roughness1: f32,
    lobe_mix: f32,
    // Slope variance the filtering removed (added to GGX alpha^2).
    alpha_widen: f32,
    normal: vec3<f32>,
    oil: f32,
    wet: f32,
    lips: f32,
    fuzz: f32,
    // Baked thickness in mm (negative: none).
    thickness_mm: f32,
    // Profile scale from the resolved scattering radii (negative: the object's).
    sss_scale: f32,
}

struct DetailSample {
    slope: vec2<f32>,
    variance: f32,
    cavity: f32,
}

fn surface_frame(n: vec3<f32>, dpx: vec3<f32>, dpy: vec3<f32>, duvx: vec2<f32>, duvy: vec2<f32>) -> SurfaceFrame {
    let dp2perp = cross(dpy, n);
    let dp1perp = cross(n, dpx);
    let t = dp2perp * duvx.x + dp1perp * duvy.x;
    let b = dp2perp * duvx.y + dp1perp * duvy.y;
    var frame: SurfaceFrame;
    let tt = t - n * dot(n, t);
    frame.valid = dot(tt, tt) > 1e-20 && dot(b, b) > 1e-20;
    frame.t = select(vec3<f32>(1.0, 0.0, 0.0), tt * inverseSqrt(max(dot(tt, tt), 1e-30)), frame.valid);
    let c = cross(n, frame.t);
    frame.b = select(c, -c, dot(c, b) < 0.0);
    return frame;
}

// The skin-line direction (unit, in the tangent plane) for world normal `n`, the object's up
// axis and front axis.
fn langer_direction(n: vec3<f32>, up: vec3<f32>, front: vec3<f32>) -> vec3<f32> {
    let around = cross(up, n);
    var other = cross(front, n);
    other = select(other, -other, dot(other, around) < 0.0);
    let k = smoothstep(0.75, 0.95, abs(dot(n, up)));
    let d = mix(around, other, k);
    let len = length(d);
    return select(vec3<f32>(1.0, 0.0, 0.0), d / len, len > 1e-6);
}

// The centre of an anti-tiling vertex in input periods (inverse of antitile_triangle's skew).
fn antitile_vertex_position(v: vec2<i32>) -> vec2<f32> {
    let s = vec2<f32>(v);
    return vec2<f32>(s.x + 0.5 * s.y, 0.8660254 * s.y) / 3.4641016;
}

// The mip level for screen derivatives `dx`, `dy` of a coordinate in texels (isotropic, as the
// hardware does without anisotropy). Explicit levels let every fetch of a set share one level
// computation and take the fast sampling path (textureSampleGrad is several times slower here).
fn mip_level(dx: vec2<f32>, dy: vec2<f32>) -> f32 {
    return max(0.5 * log2(max(max(dot(dx, dx), dot(dy, dy)), 1e-12)), 0.0);
}

// The micro-detail at `p_mm` (skin millimetres in the UV frame) with its screen derivatives, the
// skin-line direction `line` (unit, UV frame). Slopes are blended with the variance-preserving
// operator of Heitz & Neyret 2018 (their Gaussian case: the slopes have zero mean), the removed
// variance and cavity linearly.
fn micro_detail(p_mm: vec2<f32>, dpx_mm: vec2<f32>, dpy_mm: vec2<f32>, line: vec2<f32>, seed: u32) -> DetailSample {
    let side = max(surface.detail.x, 1e-3);
    let u = p_mm / side;
    let cell = antitile_triangle(u);
    let ey = line;
    let ex = vec2<f32>(line.y, -line.x);
    let size = f32(textureDimensions(skin_detail).x);
    let lod = mip_level(dpx_mm * (size / side), dpy_mm * (size / side));
    var vertices = array<vec2<i32>, 3>(cell.v1, cell.v2, cell.v3);
    let weights = cell.weights;
    var slope = vec2<f32>(0.0);
    var variance = 0.0;
    var cavity = 0.0;
    for (var i = 0u; i < 3u; i++) {
        let d = u - antitile_vertex_position(vertices[i]);
        let q = vec2<f32>(dot(d, ex), dot(d, ey)) + antitile_offset(vertices[i], seed);
        let s = textureSampleLevel(skin_detail, repeat_sampler, q, lod);
        let w = weights[i];
        slope += w * s.xy;
        variance += w * max(s.z - dot(s.xy, s.xy), 0.0);
        cavity += w * s.w;
    }
    slope *= inverseSqrt(max(dot(weights, weights), 1e-6));
    var out: DetailSample;
    out.slope = slope.x * ex + slope.y * ey;
    out.variance = variance;
    out.cavity = cavity;
    return out;
}

// The wrinkle activations of a driver + 1 (0: empty slot).
fn wrinkle_activation(id: u32) -> vec2<f32> {
    if (id == 0u || id > arrayLength(&wrinkle_activations)) {
        return vec2<f32>(0.0);
    }
    return wrinkle_activations[id - 1u];
}

// Bilinear wrinkle blending at tile-local `uv` of UV tile `tile` [SKN-30]: each of the four
// texels is blended with its own drivers (skin/wrinkle.wgsl), then the results are interpolated.
fn wrinkle_at(tile: u32, uv: vec2<f32>) -> WrinkleSample {
    var out = WrinkleSample(0.0, vec2<f32>(0.0), 0.0, 1.0);
    if (surface.misc.y == 0u || tile >= arrayLength(&wrinkle_tile_slots)) {
        return out;
    }
    let slot = wrinkle_tile_slots[tile];
    let base = i32(slot) * 3;
    if (slot == NO_WRINKLE_SLOT || u32(base) + 3u > textureNumLayers(wrinkle_maps)) {
        return out;
    }
    let size = vec2<i32>(textureDimensions(wrinkle_maps).xy);
    let f = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * vec2<f32>(size) - 0.5;
    let origin = vec2<i32>(floor(f));
    let fr = f - floor(f);
    out.detail_scale = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let o = vec2<i32>(i32(k & 1u), i32(k >> 1u));
        let p = clamp(origin + o, vec2<i32>(0), size - 1);
        let w = select(1.0 - fr.x, fr.x, o.x == 1) * select(1.0 - fr.y, fr.y, o.y == 1);
        let t = wrinkle_fetch(wrinkle_maps, base, p);
        let ids = wrinkle_driver(t);
        let s = wrinkle_blend(t, wrinkle_activation(ids.x), wrinkle_activation(ids.y));
        out.height += w * s.height;
        out.slope += w * s.slope;
        out.cavity += w * s.cavity;
        out.detail_scale += w * s.detail_scale;
    }
    return out;
}

// The surface of a skin pixel without surface data: the material's values.
fn plain_surface(n: vec3<f32>, base: vec3<f32>, roughness: vec3<f32>) -> SkinSurfaceSample {
    var s: SkinSurfaceSample;
    s.albedo = base;
    s.cavity = 1.0;
    s.micro_cavity = 1.0;
    s.f0_scale = 1.0;
    s.roughness0 = roughness.x;
    s.roughness1 = roughness.y;
    s.lobe_mix = roughness.z;
    s.alpha_widen = 0.0;
    s.normal = n;
    s.oil = 0.0;
    s.wet = 0.0;
    s.lips = 0.0;
    s.fuzz = 0.0;
    s.thickness_mm = -1.0;
    s.sss_scale = -1.0;
    return s;
}

// The surface at a pixel of a surfaced skin object. `uv` is the mesh's (u + tile, v); `n` the
// geometric normal; `dpx`/`dpy` and `duvx`/`duvy` the screen derivatives of the world position
// and of `uv`; `up` and `front` the object's axes in world space; `base` the material's values
// (used where a tile has no resolved data).
fn skin_surface(
    uv: vec2<f32>,
    n: vec3<f32>,
    frame: SurfaceFrame,
    duvx: vec2<f32>,
    duvy: vec2<f32>,
    up: vec3<f32>,
    front: vec3<f32>,
    base: vec3<f32>,
    roughness: vec3<f32>,
) -> SkinSurfaceSample {
    var s = plain_surface(n, base, roughness);
    let tile_f = floor(uv.x);
    if (tile_f < 0.0 || tile_f >= f32(MAX_UV_TILES)) {
        return s;
    }
    let tile = u32(tile_f);
    let info = surface.tiles[tile];
    if (info.x <= 0.0) {
        return s;
    }
    let local = vec2<f32>(uv.x - tile_f, uv.y);
    var slope = vec2<f32>(0.0);
    var fuzz_zone = 0.0;
    if (info.y >= 0.0) {
        let layer = i32(info.y) * 4;
        let texels = f32(textureDimensions(skin_resolved).x);
        let lod = mip_level(duvx * texels, duvy * texels);
        let albedo = textureSampleLevel(skin_resolved, clamp_sampler, local, layer, lod);
        let scatter = textureSampleLevel(skin_resolved, clamp_sampler, local, layer + 1, lod);
        let meso = textureSampleLevel(skin_resolved, clamp_sampler, local, layer + 2, lod);
        let flush = textureSampleLevel(skin_resolved, clamp_sampler, local, layer + 3, lod);
        let spec = textureSampleLevel(skin_specular, clamp_sampler, local, i32(info.y), lod);
        let zones0 = textureSampleLevel(skin_inputs, clamp_sampler, local, layer + 1, lod);
        let zones1 = textureSampleLevel(skin_inputs, clamp_sampler, local, layer + 2, lod);
        s.albedo = skin_flushed_albedo(albedo.rgb, flush.rgb, surface.detail.w);
        s.cavity = clamp(albedo.a, 0.0, 1.0);
        s.thickness_mm = select(-1.0, scatter.a * BAKED_THICKNESS_RANGE_MM, scatter.a < 0.999);
        // [SHD-02]: the profile width per pixel from the pigment model's radii.
        s.sss_scale = clamp((scatter.r + scatter.g + scatter.b) / 3.0 * surface.fuzz.w, 0.0, 4.0);
        // Meso normal (tangent x, y of a unit normal, skin/microdetail.wgsl convention) to slopes.
        let nz = sqrt(max(1.0 - dot(meso.xy, meso.xy), 1e-4));
        slope = -meso.xy / nz;
        s.wet = clamp(meso.a, 0.0, 1.0);
        s.oil = clamp(flush.a, 0.0, 1.0);
        s.f0_scale = spec.x * 4.0;
        s.roughness0 = spec.y;
        s.roughness1 = spec.z;
        s.lobe_mix = spec.w;
        // Zone slots: 0 cheeks, 1 nose, 3 lips, 4 ears (humanart_model::skin_appearance::zones). The
        // vellus zones are the cheeks, nose and ears; the forehead (slot 9, a third fetch) is left
        // at the baseline to keep the pass in its budget.
        s.lips = zones0.w;
        fuzz_zone = saturate(zones0.x + zones0.y + zones1.x);
    }
    let wrinkle = wrinkle_at(tile, local);
    slope += wrinkle.slope;
    s.cavity = clamp(s.cavity + wrinkle.cavity, 0.0, 1.0);
    // Micro-detail [SKN-20] [SKN-21].
    let strength = surface.detail.y * wrinkle.detail_scale;
    if (surface.detail.x > 0.0 && strength > 0.0 && frame.valid) {
        let d = langer_direction(n, up, front);
        let line = vec2<f32>(dot(d, frame.t), dot(d, frame.b));
        let len = length(line);
        let line_uv = select(vec2<f32>(0.0, 1.0), line / len, len > 1e-4);
        let m = micro_detail(local * info.x, duvx * info.x, duvy * info.x, line_uv, surface.misc.x);
        slope += strength * m.slope;
        s.alpha_widen = strength * strength * m.variance;
        s.micro_cavity = clamp(1.0 + surface.detail.z * strength * (m.cavity - 1.0), 0.0, 1.0);
    }
    if (frame.valid) {
        let bent = n - slope.x * frame.t - slope.y * frame.b;
        s.normal = bent * inverseSqrt(max(dot(bent, bent), 1e-12));
    }
    // Lips [SHD-06]: smoother, no vellus.
    s.roughness0 = s.roughness0 * mix(1.0, surface.fuzz.z, s.lips);
    s.roughness1 = s.roughness1 * mix(1.0, surface.fuzz.z, s.lips);
    s.fuzz = surface.fuzz.x * (0.35 + 0.65 * fuzz_zone) * (1.0 - s.lips);
    return s;
}

// The oil / sweat coat [SHD-06]: strength 0..1 from oiliness, wetness and the lip mask.
fn coat_strength(s: SkinSurfaceSample) -> f32 {
    return saturate(surface.coat.x * s.oil + s.wet + surface.coat.z * s.lips);
}

// The coat's perceptual roughness (the lips' where the lip mask is set).
fn coat_roughness(s: SkinSurfaceSample) -> f32 {
    return clamp(mix(surface.coat.y, surface.coat.w, s.lips), 0.02, 1.0);
}

// Charlie sheen distribution (Estevez & Kulla 2017, "Production Friendly Microfacet Sheen BRDF").
fn d_charlie(roughness: f32, n_dot_h: f32) -> f32 {
    let inv_r = 1.0 / clamp(roughness, 0.05, 1.0);
    let sin2 = max(1.0 - n_dot_h * n_dot_h, 0.0078125);
    return (2.0 + inv_r) * pow(sin2, inv_r * 0.5) / (2.0 * 3.14159265);
}

// Neubelt & Pettineo 2013 visibility for cloth / sheen.
fn v_neubelt(n_dot_v: f32, n_dot_l: f32) -> f32 {
    return 1.0 / max(4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v), 1e-4);
}

// The coat's GGX lobe (distribution x correlated visibility, lib/brdf.wgsl), Fresnel excluded.
fn coat_lobe(n_dot_h: f32, n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    return d_ggx(n_dot_h, alpha) * v_smith_ggx_correlated(n_dot_v, n_dot_l, alpha);
}
