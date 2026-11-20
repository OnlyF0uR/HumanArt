// skin/antitile.wgsl: histogram-preserving tiling and blending [SKN-20] (Heitz & Neyret 2018,
// "High-Performance By-Example Noise using a Histogram-Preserving Blending Operator", HPG), the
// twin of humanart_model::skin_appearance::antitile. `gauss` is the Gaussianised exemplar (mean
// 0.5, sigma 1/6), `lut` the 1024 x 1 inverse table back to the exemplar's histogram; both are
// read with textureLoad (nearest texel), as the CPU reference does.

#include "skin/noise.wgsl"

const ANTITILE_LUT_SIZE: f32 = 1024.0;

struct TriangleCell {
    weights: vec3<f32>,
    v1: vec2<i32>,
    v2: vec2<i32>,
    v3: vec2<i32>,
}

// The barycentric weights and vertex ids of the skewed triangle grid at 2 sqrt(3) times the input
// frequency.
fn antitile_triangle(uv: vec2<f32>) -> TriangleCell {
    let p = uv * 3.4641016;
    let skewed = vec2<f32>(p.x - 0.57735027 * p.y, 1.1547005 * p.y);
    let base = floor(skewed);
    let f = skewed - base;
    let fz = 1.0 - f.x - f.y;
    let b = vec2<i32>(base);
    var cell: TriangleCell;
    if (fz > 0.0) {
        cell.weights = vec3<f32>(fz, f.y, f.x);
        cell.v1 = b;
        cell.v2 = b + vec2<i32>(0, 1);
        cell.v3 = b + vec2<i32>(1, 0);
    } else {
        cell.weights = vec3<f32>(-fz, 1.0 - f.y, 1.0 - f.x);
        cell.v1 = b + vec2<i32>(1, 1);
        cell.v2 = b + vec2<i32>(1, 0);
        cell.v3 = b + vec2<i32>(0, 1);
    }
    return cell;
}

// The random offset of a vertex in exemplar tiles.
fn antitile_offset(vertex: vec2<i32>, seed: u32) -> vec2<f32> {
    let x = bitcast<u32>(vertex.x);
    let y = bitcast<u32>(vertex.y);
    return vec2<f32>(hash_unit(hash3(x, y, seed)), hash_unit(hash3(x, y, seed ^ 0x9e3779b9u)));
}

fn antitile_fetch(tex: texture_2d<f32>, uv: vec2<f32>) -> f32 {
    let size = vec2<f32>(textureDimensions(tex));
    let p = uv * size;
    let wrapped = p - floor(p / size) * size;
    let texel = min(vec2<u32>(wrapped), vec2<u32>(size) - 1u);
    return textureLoad(tex, texel, 0).r;
}

// The histogram-preserving blend at `uv` (exemplar tiles).
fn antitile_sample(gauss: texture_2d<f32>, lut: texture_2d<f32>, uv: vec2<f32>, seed: u32) -> f32 {
    let cell = antitile_triangle(uv);
    let w = cell.weights;
    let g1 = antitile_fetch(gauss, uv + antitile_offset(cell.v1, seed));
    let g2 = antitile_fetch(gauss, uv + antitile_offset(cell.v2, seed));
    let g3 = antitile_fetch(gauss, uv + antitile_offset(cell.v3, seed));
    let g = (w.x * g1 + w.y * g2 + w.z * g3 - 0.5) / sqrt(dot(w, w)) + 0.5;
    let index = u32(clamp(i32(g * ANTITILE_LUT_SIZE), 0, i32(ANTITILE_LUT_SIZE) - 1));
    return textureLoad(lut, vec2<u32>(index, 0u), 0).r;
}
