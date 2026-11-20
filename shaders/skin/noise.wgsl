// skin/noise.wgsl: the PCG hash and periodic value noise of humanart_model::skin_appearance::noise,
// bit for bit (u32 arithmetic) and to float rounding (f32), so GPU and CPU generators agree.
// PCG: Jarzynski & Olano 2020, "Hash Functions for GPU Rendering", JCGT 9(3).

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash3(x: u32, y: u32, seed: u32) -> u32 {
    return pcg(x ^ pcg(y ^ pcg(seed)));
}

// A hash mapped to [0, 1) with 24 bits.
fn hash_unit(h: u32) -> f32 {
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Value noise in [0, 1), periodic with `period` lattice cells.
fn value_noise(x: f32, y: f32, period: u32, seed: u32) -> f32 {
    let p = max(period, 1u);
    let fx = floor(x);
    let fy = floor(y);
    let tx = fade(x - fx);
    let ty = fade(y - fy);
    let pf = f32(p);
    let ix = u32(fx - floor(fx / pf) * pf) % p;
    let iy = u32(fy - floor(fy / pf) * pf) % p;
    let jx = (ix + 1u) % p;
    let jy = (iy + 1u) % p;
    let a = hash_unit(hash3(ix, iy, seed));
    let b = hash_unit(hash3(jx, iy, seed));
    let c = hash_unit(hash3(ix, jy, seed));
    let d = hash_unit(hash3(jx, jy, seed));
    let top = a + (b - a) * tx;
    let bottom = c + (d - c) * tx;
    return top + (bottom - top) * ty;
}

// Periodic fractal value noise centred on 0.
fn fbm(x: f32, y: f32, period: u32, octaves: u32, gain: f32, seed: u32) -> f32 {
    var sum = 0.0;
    var amplitude = 1.0;
    var norm = 0.0;
    var scale = 1.0;
    for (var o = 0u; o < min(octaves, 12u); o += 1u) {
        sum += amplitude * (value_noise(x * scale, y * scale, period << o, seed + o) - 0.5);
        norm += amplitude;
        amplitude *= gain;
        scale *= 2.0;
    }
    return select(0.0, sum / norm, norm > 0.0);
}
