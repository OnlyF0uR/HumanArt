// skin/microdetail.wgsl: the GPU micro-detail generator [SKN-20] [SKN-21], the twin of
// humanart_model::skin_appearance::microdetail::ExemplarPlan::height.
//
// - `generate`: the periodic detail exemplar's height (mm) into an r32float texture: blue-noise
//   pores (centres from the CPU Poisson-disk plan, bucketed in a CSR grid), two crosshatched
//   families of fine lines with per-line depth jitter and periodic wobble, and micro-relief noise,
//   all scaled by the age terms.
// - `derive`: tangent normal (x, y) by central differences, height, and a cavity term from the
//   height relative to its 5 x 5 neighbourhood, into rgba16float.
// - `synthesize`: histogram-preserving tiling and blending of the Gaussianised exemplar over an
//   output texture (the close-up surface, or a bake into a tile).

#include "skin/antitile.wgsl"

struct Plan {
    // size, grid, seed, pore count.
    dims: vec4<u32>,
    // line family 0: p, q, count; family 1 p.
    lines0: vec4<i32>,
    // line family 1: q, count; -, -.
    lines1: vec4<i32>,
    // line depth mm, line sigma texels, elongation e, relief mm.
    age: vec4<f32>,
    // texel mm, output tiles across (synthesize), output seed bits (synthesize), -.
    misc: vec4<f32>,
}

@group(0) @binding(0) var<uniform> plan: Plan;
@group(0) @binding(1) var<storage, read> pores: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> cell_start: array<u32>;
@group(0) @binding(3) var height_out: texture_storage_2d<r32float, write>;

@group(1) @binding(0) var height_in: texture_2d<f32>;
@group(1) @binding(1) var detail_out: texture_storage_2d<rgba16float, write>;

@group(1) @binding(2) var gauss: texture_2d<f32>;
@group(1) @binding(3) var lut: texture_2d<f32>;
@group(1) @binding(4) var synth_out: texture_storage_2d<r32float, write>;

fn line_height(px: f32, py: f32, side: f32, p: i32, q: i32, count: i32, k: u32) -> f32 {
    let seed = plan.dims.z;
    let warp = 0.35 * fbm(px / side * 4.0, py / side * 4.0, 4u, 3u, 0.5, seed + 101u + k);
    let u = ((f32(p) * px + f32(q) * py) / side) * f32(count) + warp;
    let line = floor(u);
    let f = u - line;
    let d = min(f, 1.0 - f);
    let nearest = select(line + 1.0, line, f < 0.5);
    let c = max(count, 1);
    let index = u32(((i32(nearest) % c) + c) % c);
    let along = fbm(px / side * 8.0, py / side * 8.0, 8u, 2u, 0.5, seed + 301u + k);
    let jitter = (0.5 + hash_unit(hash3(index, k + 7u, seed))) * clamp(0.3 + 2.8 * (along + 0.5), 0.0, 1.7);
    let spacing = side / (f32(count) * sqrt(f32(p * p + q * q)));
    let dist = d * spacing;
    let sigma = plan.age.y;
    let weight = select(0.7, 1.0, k == 0u);
    return -weight * plan.age.x * jitter * exp(-0.5 * (dist / sigma) * (dist / sigma));
}

@compute @workgroup_size(8, 8, 1)
fn generate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = plan.dims.x;
    if (gid.x >= size || gid.y >= size) {
        return;
    }
    let side = f32(size);
    let px = f32(gid.x) + 0.5;
    let py = f32(gid.y) + 0.5;
    var h = 0.0;
    let e = plan.age.z;
    let ex = sqrt(e);
    let ey = 1.0 / sqrt(e);
    let grid = i32(plan.dims.y);
    let cell = side / f32(grid);
    let cx = i32(px / cell);
    let cy = i32(py / cell);
    var visited: array<u32, 9>;
    var n_visited = 0u;
    for (var dy = -1; dy <= 1; dy += 1) {
        for (var dx = -1; dx <= 1; dx += 1) {
            let gx = u32(((cx + dx) % grid + grid) % grid);
            let gy = u32(((cy + dy) % grid + grid) % grid);
            let c = gy * u32(grid) + gx;
            var seen = false;
            for (var i = 0u; i < n_visited; i += 1u) {
                seen = seen || visited[i] == c;
            }
            if (seen) {
                continue;
            }
            visited[n_visited] = c;
            n_visited += 1u;
            let start = cell_start[c];
            let end = cell_start[c + 1u];
            for (var i = start; i < end; i += 1u) {
                let pore = pores[i];
                var ddx = px - pore.x;
                var ddy = py - pore.y;
                ddx -= side * round(ddx / side);
                ddy -= side * round(ddy / side);
                let u = ddx / (pore.w * ey);
                let v = ddy / (pore.w * ex);
                h -= pore.z * exp(-0.5 * (u * u + v * v));
            }
        }
    }
    h += line_height(px, py, side, plan.lines0.x, plan.lines0.y, plan.lines0.z, 0u);
    h += line_height(px, py, side, plan.lines0.w, plan.lines1.x, plan.lines1.y, 1u);
    h += plan.age.w * fbm(px / side * 16.0, py / side * 16.0, 16u, 4u, 0.5, plan.dims.z + 211u);
    textureStore(height_out, gid.xy, vec4<f32>(h, 0.0, 0.0, 0.0));
}

fn height_at(x: i32, y: i32) -> f32 {
    let size = i32(textureDimensions(height_in).x);
    let wx = ((x % size) + size) % size;
    let wy = ((y % size) + size) % size;
    return textureLoad(height_in, vec2<i32>(wx, wy), 0).r;
}

@compute @workgroup_size(8, 8, 1)
fn derive(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(height_in);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let texel_mm = plan.misc.x;
    let dx = (height_at(x + 1, y) - height_at(x - 1, y)) / (2.0 * texel_mm);
    let dy = (height_at(x, y + 1) - height_at(x, y - 1)) / (2.0 * texel_mm);
    let len = sqrt(dx * dx + dy * dy + 1.0);
    let h = height_at(x, y);
    var mean = 0.0;
    for (var j = -2; j <= 2; j += 1) {
        for (var i = -2; i <= 2; i += 1) {
            mean += height_at(x + i, y + j);
        }
    }
    mean /= 25.0;
    // Cavity: 1 on open skin, falling towards 0 in pits deeper than 0.05 mm below the neighbourhood.
    let cavity = clamp(1.0 + (h - mean) / 0.05, 0.0, 1.0);
    textureStore(detail_out, gid.xy, vec4<f32>(-dx / len, -dy / len, h, cavity));
}

@compute @workgroup_size(8, 8, 1)
fn synthesize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(synth_out);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }
    let tiles = plan.misc.y;
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(size) * tiles;
    let value = antitile_sample(gauss, lut, uv, bitcast<u32>(plan.misc.z));
    textureStore(synth_out, gid.xy, vec4<f32>(value, 0.0, 0.0, 0.0));
}
