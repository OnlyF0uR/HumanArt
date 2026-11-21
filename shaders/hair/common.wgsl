// hair/common.wgsl: the constants of the hair passes [HR-01] [HR-03]: the per-view block
// (`HairView`, mirrors `ViewBlock` in crates/render/src/hair/view.rs), the per-groom draw block
// (`HairDraw`), the per-strand absorption from the melanin settings [HAIR-06], the visibility
// record of a pixel and the deep-opacity-map frames.
//
// Needs lib/packing.wgsl (pack_oct16, unpack_oct16).

const HAIR_MAX_GROOMS: u32 = 16u;
const HAIR_MAX_DOMS: u32 = 2u;
// Fixed-point unit of the per-pixel coverage sum (1.0 coverage = 256).
const HAIR_COVERAGE_ONE: f32 = 256.0;
// Most pixel steps along one segment after clipping to the viewport.
const HAIR_MAX_STEPS: i32 = 4096;
// Widest strand footprint across a segment, pixels.
const HAIR_MAX_WIDTH: f32 = 4.0;
// Side of the scene depth tiles of the hidden-segment test, pixels.
const HAIR_TILE: u32 = 16u;
// Fixed-point unit of the voxel fibre density (square metres of fibre side area per unit).
const HAIR_VOXEL_UNIT: f32 = 1e-9;

// The melanin settings of one groom: sigma_a of the natural pigment (before the per-strand
// scale) and the variation; the dye's sigma_a and its opacity.
struct HairGroomColor {
    sigma_variation: vec4<f32>,
    dye: vec4<f32>,
}

struct HairView {
    // Target width, height, focal length in pixels, coverage saturation (fixed point).
    screen: vec4<f32>,
    // Per deep opacity map: the light frame (right, up, forward = the light's travel direction)
    // with the centre in w, and (radius, light slot, texel size in metres, self-shadow bias).
    dom_right: array<vec4<f32>, 2>,
    dom_up: array<vec4<f32>, 2>,
    dom_forward: array<vec4<f32>, 2>,
    dom_params: array<vec4<f32>, 2>,
    // Far edges of the four opacity layers behind the first fibre, metres.
    dom_layers: vec4<f32>,
    // Voxel grid origin (metres) and cell size.
    voxel: vec4<f32>,
    // Longitudinal variances v0..v3 of the fibre.
    variances: vec4<f32>,
    // Lobe shifts R, TT, TRT; the voxel density to optical depth factor.
    shifts: vec4<f32>,
    // Voxels per side, deep opacity maps, grooms, flags (HAIR_FLAG_*).
    counts: vec4<u32>,
    grooms: array<HairGroomColor, 16>,
}

// Flags of HairView.counts.w.
const HAIR_FLAG_DUAL: u32 = 1u;
const HAIR_FLAG_OCCLUSION: u32 = 2u;
const HAIR_FLAG_SKIN_SHADOW: u32 = 4u;
const HAIR_FLAG_CULL: u32 = 8u;

struct HairDraw {
    // Strands, points per strand, point stride (far field), groom index.
    counts: vec4<u32>,
    // Segments per strand after the stride, threads per dispatch row, keep threshold (strands
    // whose hash is above it are pruned), strand step of the deep opacity maps.
    dispatch: vec4<u32>,
    // Coverage scale (1 / kept fraction), deep-opacity coverage scale, width scale, raster
    // threads per segment (the walk split, 1..=8).
    scale: vec4<f32>,
}

// PCG hash.
fn hair_hash(x: u32) -> u32 {
    let state = x * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Whether strand `strand` survives the far-field pruning threshold `keep`.
fn hair_kept(strand: u32, keep: u32) -> bool {
    return hair_hash(strand ^ 0x5bd1e995u) <= keep;
}

// sigma_a of a strand: the groom's natural pigment scaled by the strand's colour variation and
// greyness (humanart_model::hair::HairColor::strand_sigma_a), mixed towards the dye.
fn hair_sigma_a(color: HairGroomColor, color_id: u32, grey: u32) -> vec3<f32> {
    let id = (f32(color_id) + 0.5) / 4096.0;
    let scale = (1.0 + color.sigma_variation.w * (2.0 * id - 1.0)) * (1.0 - f32(grey) / 255.0);
    let natural = color.sigma_variation.xyz * max(scale, 0.0);
    return mix(natural, color.dye.xyz, clamp(color.dye.w, 0.0, 1.0));
}

// The visibility record of a pixel: x tangent (octahedral 16:16), y groom << 28 | grey << 20 |
// colour id (12 bits) << 8 | the nearest fragment's own coverage (8 bits, hair_record_with_coverage),
// z motion (pixels, two f16), w strand + 1.
fn hair_pack_record(tangent: vec3<f32>, groom: u32, attributes: u32, motion: vec2<f32>, strand: u32) -> vec4<u32> {
    let color_id = (attributes & 0xFFFFu) >> 4u;
    let grey = (attributes >> 16u) & 0xFFu;
    return vec4<u32>(pack_oct16(tangent), (groom << 28u) | (grey << 20u) | (color_id << 8u), pack2x16float(motion), strand + 1u);
}

// `r` with the coverage of its own fragment (0..1, after the far-field scale) in the low 8 bits of y.
fn hair_record_with_coverage(r: vec4<u32>, coverage: f32) -> vec4<u32> {
    var out = r;
    out.y = (r.y & 0xFFFFFF00u) | u32(round(saturate(coverage) * 255.0));
    return out;
}

// The coverage of the pixel's nearest fragment.
fn hair_record_coverage(r: vec4<u32>) -> f32 {
    return f32(r.y & 0xFFu) / 255.0;
}

// A pixel's hair opacity from its coverage sum (fixed point) and record: the nearest fibre's own
// coverage c0 exactly, the rest as independent fibres, 1 - (1 - c0) exp(-(sum - c0)). (The
// exponential alone leaves a fibre covering the whole pixel 63 % opaque.)
fn hair_alpha(coverage_sum: u32, r: vec4<u32>) -> f32 {
    let total = f32(coverage_sum) / HAIR_COVERAGE_ONE;
    let front = min(hair_record_coverage(r), total);
    return 1.0 - (1.0 - front) * exp(-(total - front));
}

fn hair_record_groom(r: vec4<u32>) -> u32 {
    return r.y >> 28u;
}

fn hair_record_grey(r: vec4<u32>) -> u32 {
    return (r.y >> 20u) & 0xFFu;
}

fn hair_record_color(r: vec4<u32>) -> u32 {
    return (r.y >> 8u) & 0xFFFu;
}

// A world point in deep opacity map `frame_right`/... space: (x, y) in [-1, 1] across the map,
// z the reversed light depth in [0, 1] (1 nearest the light).
fn hair_dom_local(p: vec3<f32>, right: vec4<f32>, up: vec4<f32>, forward: vec4<f32>, radius: f32) -> vec3<f32> {
    let d = p - vec3<f32>(right.w, up.w, forward.w);
    return vec3<f32>(dot(d, right.xyz) / radius, dot(d, up.xyz) / radius, 0.5 - dot(d, forward.xyz) / (2.0 * radius));
}

// The opacity accumulated in front of a point `behind` metres behind the first fibre, from the
// four layers `layers` with far edges `edges` (the last layer is open-ended): the expected number
// of fibres crossed towards the light (Yuksel and Keyser 2008, deep opacity maps).
fn hair_dom_tau(layers: vec4<f32>, edges: vec4<f32>, behind: f32) -> f32 {
    if (behind <= 0.0) {
        return 0.0;
    }
    var tau = 0.0;
    var near = 0.0;
    for (var k = 0; k < 4; k++) {
        let far = edges[k];
        if (behind >= far && k < 3) {
            tau += layers[k];
        } else {
            tau += layers[k] * saturate((behind - near) / max(far - near, 1e-6));
            break;
        }
        near = far;
    }
    return tau;
}

// Thread index of a 2D dispatch whose rows hold `row` threads.
fn hair_thread(gid: vec3<u32>, row: u32) -> u32 {
    return gid.x + gid.y * row;
}
