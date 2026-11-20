// skin/resolve.wgsl: the second compositor pass [SKN-01] [SKN-10] [SKN-12] [SKN-31] [SKN-32]: the
// planes of one dirty rectangle to the resolved channels. The twin of
// humanart_model::skin_appearance::resolve::resolve_texel; the spectral pigment model and its
// sources are documented in humanart_model::skin_appearance::pigment (docs/shader-notes.md,
// "Skin pigment model").
//
// Resolved: one rgba16float array, layer = tile * 4 + k: k = 0 albedo (rgb linear, a cavity),
// 1 scatter (rgb SSS radius mm, a thickness), 2 detail (normal x, y, height mm, wetness),
// 3 flush (rgb albedo delta at full flush, a oiliness); specular: rgba8unorm array, layer = tile
// (specular / 4, primary roughness, secondary roughness, lobe mix).

#include "skin/common.wgsl"

const BANDS: u32 = 81u;
const SKIN_IOR: f32 = 1.4;

// One spectral band: rgb weight, sss weight, absorptions (mm^-1) and scattering; see
// PigmentModel::write_gpu_table.
struct Band {
    // rgb weight r, g, b; sss weight r.
    a: vec4<f32>,
    // sss weight g, b; eumelanin; pheomelanin.
    b: vec4<f32>,
    // baseline; oxy blood; deoxy blood; carotene per mol/L.
    c: vec4<f32>,
    // epidermis reduced scattering; -, -, -.
    d: vec4<f32>,
}

@group(0) @binding(0) var planes: texture_2d_array<f32>;
@group(0) @binding(1) var inputs: texture_2d_array<f32>;
@group(0) @binding(2) var resolved: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var specular_out: texture_storage_2d_array<rgba8unorm, write>;
@group(0) @binding(4) var<storage, read> globals: Globals;
@group(0) @binding(5) var<storage, read> bands: array<Band, 81>;
@group(0) @binding(6) var<uniform> params: Params;

// Total diffuse reflectance of a semi-infinite medium (Jensen & Buhler 2002).
fn diffuse_reflectance(alpha_prime: f32) -> f32 {
    let eta = SKIN_IOR;
    let fdr = -1.440 / (eta * eta) + 0.710 / eta + 0.668 + 0.0636 * eta;
    let a = (1.0 + fdr) / (1.0 - fdr);
    let s = sqrt(3.0 * (1.0 - clamp(alpha_prime, 0.0, 1.0)));
    return 0.5 * alpha_prime * (1.0 + exp(-4.0 / 3.0 * a * s)) * exp(-s);
}

@compute @workgroup_size(8, 8, 1)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) {
        return;
    }
    let p = params.origin + gid.xy;
    let tile = params.tile;
    let pigment = textureLoad(planes, p, i32(tile * PLANES + 0u), 0);
    let tissue = textureLoad(planes, p, i32(tile * PLANES + 1u), 0);
    let surface = textureLoad(planes, p, i32(tile * PLANES + 2u), 0);
    let relief = textureLoad(planes, p, i32(tile * PLANES + 3u), 0);
    let over = textureLoad(planes, p, i32(tile * PLANES + 4u), 0);
    let baked = textureLoad(inputs, p, i32(tile * 4u), 0);

    // Registry values to physical quantities (pigment::PigmentParams::physical).
    let m0 = globals.mapping0;
    let m1 = globals.mapping1;
    let melanin = clamp(pigment.x, 0.0, 1.0);
    let cm = m0.x * melanin * melanin;
    let bm = clamp(pigment.y, 0.0, 1.0);
    let ch = m0.y * pow(m0.z / m0.y, clamp(pigment.z, 0.0, 1.0));
    let gamma = clamp(pigment.w, 0.0, 1.0);
    let carotene = m1.y * clamp(tissue.x, 0.0, 1.0);
    let epidermis = m0.w + (m1.x - m0.w) * clamp(tissue.y, 0.0, 1.0);

    // Flush weight from the zone masks [SKN-31].
    var flush_weight = globals.surface.x;
    for (var k = 0u; k < 3u; k += 1u) {
        let m = textureLoad(inputs, p, i32(tile * 4u + 1u + k), 0);
        for (var c = 0u; c < 4u; c += 1u) {
            flush_weight += globals.zones[k * 4u + c].flush.x * m[c];
        }
    }
    flush_weight = clamp(flush_weight, 0.0, 1.0);
    let ch_flush = min(ch * (1.0 + globals.surface.y * flush_weight), 1.0);

    var albedo = vec3<f32>(0.0);
    var flushed = vec3<f32>(0.0);
    var radius_num = vec3<f32>(0.0);
    var radius_den = vec3<f32>(0.0);
    for (var i = 0u; i < BANDS; i += 1u) {
        let band = bands[i];
        let rgb = band.a.xyz;
        let sss = vec3<f32>(band.a.w, band.b.x, band.b.y);
        let baseline = band.c.x;
        let sigma_epi = cm * (bm * band.b.z + (1.0 - bm) * band.b.w) + (1.0 - cm) * baseline + band.c.w * carotene;
        let transmit = exp(-2.0 * epidermis * sigma_epi);
        let blood = gamma * band.c.y + (1.0 - gamma) * band.c.z;
        let scatter = 0.5 * band.d.x;
        let sigma_derm = ch * blood + (1.0 - ch) * baseline;
        let r = transmit * diffuse_reflectance(scatter / (scatter + sigma_derm));
        let sigma_flush = ch_flush * blood + (1.0 - ch_flush) * baseline;
        let rf = transmit * diffuse_reflectance(scatter / (scatter + sigma_flush));
        albedo += rgb * r;
        flushed += rgb * rf;
        let length_mm = 1.0 / max(sqrt(3.0 * sigma_derm * (sigma_derm + scatter)), 1e-9);
        radius_num += sss * r * length_mm;
        radius_den += sss * r;
    }
    let pigment_albedo = clamp(albedo, vec3<f32>(0.0), vec3<f32>(1.0));
    let flushed_albedo = clamp(flushed, vec3<f32>(0.0), vec3<f32>(1.0));
    let radius = select(vec3<f32>(0.0), radius_num / radius_den, radius_den > vec3<f32>(0.0));

    // Direct colour override [SKN-12].
    let coverage = clamp(over.w, 0.0, 1.0);
    let final_albedo = pigment_albedo + (over.xyz - pigment_albedo) * coverage;
    let flush_delta = (flushed_albedo - pigment_albedo) * (1.0 - coverage);
    let cavity = clamp(relief.w * baked.y, 0.0, 1.0);

    // Wetness [SKN-32].
    let wet = clamp(tissue.w, 0.0, 1.0);
    let rough = surface.y + (globals.surface.z - surface.y) * wet;
    let spec = surface.x * (1.0 + globals.surface.w * wet);

    var n = relief.yz;
    let len = length(n);
    if (len > 1.0) {
        n = n / len;
    }
    textureStore(resolved, p, i32(tile * 4u + 0u), vec4<f32>(final_albedo, cavity));
    textureStore(resolved, p, i32(tile * 4u + 1u), vec4<f32>(radius, baked.x));
    textureStore(resolved, p, i32(tile * 4u + 2u), vec4<f32>(n, relief.x, wet));
    textureStore(resolved, p, i32(tile * 4u + 3u), vec4<f32>(flush_delta, tissue.z));
    textureStore(specular_out, p, i32(tile), clamp(vec4<f32>(spec / 4.0, rough, surface.z, surface.w), vec4<f32>(0.0), vec4<f32>(1.0)));
}
