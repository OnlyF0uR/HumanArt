//! Hashes and periodic noise shared bit for bit with the WGSL of `shaders/skin/` (`skin/noise.wgsl`).
//!
//! The hash is PCG (Jarzynski & Olano 2020, "Hash Functions for GPU Rendering", JCGT 9(3)): one
//! 32-bit multiply-xorshift round, chosen there for its quality per cost on GPUs. Everything is
//! `u32` and `f32` arithmetic the GPU reproduces exactly; value noise is periodic on an integer
//! lattice so a generated exemplar tile wraps seamlessly.

/// The PCG hash of one word.
pub const fn pcg(v: u32) -> u32 {
    let state = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// Hash of a lattice point and a seed.
pub const fn hash3(x: u32, y: u32, seed: u32) -> u32 {
    pcg(x ^ pcg(y ^ pcg(seed)))
}

/// A hash mapped to `[0, 1)` with 24 bits.
pub fn unit(h: u32) -> f32 {
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Quintic fade `6t⁵ - 15t⁴ + 10t³`.
pub fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Value noise in `[0, 1)` at `(x, y)` (lattice units), periodic with `period` lattice cells on both
/// axes (`period ≥ 1`).
pub fn value_noise(x: f32, y: f32, period: u32, seed: u32) -> f32 {
    let p = period.max(1);
    let (fx, fy) = (x.floor(), y.floor());
    let (tx, ty) = (fade(x - fx), fade(y - fy));
    let pf = p as f32;
    let ix = (fx - (fx / pf).floor() * pf) as u32 % p;
    let iy = (fy - (fy / pf).floor() * pf) as u32 % p;
    let jx = (ix + 1) % p;
    let jy = (iy + 1) % p;
    let a = unit(hash3(ix, iy, seed));
    let b = unit(hash3(jx, iy, seed));
    let c = unit(hash3(ix, jy, seed));
    let d = unit(hash3(jx, jy, seed));
    let top = a + (b - a) * tx;
    let bottom = c + (d - c) * tx;
    top + (bottom - top) * ty
}

/// Periodic fractal value noise centred on 0 (about -0.5..0.5): `octaves` octaves, each twice the
/// frequency and `gain` times the amplitude of the previous; `period` is the period of the first
/// octave in lattice cells.
pub fn fbm(x: f32, y: f32, period: u32, octaves: u32, gain: f32, seed: u32) -> f32 {
    let mut sum = 0.0;
    let mut amplitude = 1.0;
    let mut norm = 0.0;
    let mut scale = 1.0;
    for o in 0..octaves.min(12) {
        sum += amplitude * (value_noise(x * scale, y * scale, period << o, seed.wrapping_add(o)) - 0.5);
        norm += amplitude;
        amplitude *= gain;
        scale *= 2.0;
    }
    if norm > 0.0 { sum / norm } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcg_reference_values() {
        // Fixed outputs, also asserted by the WGSL parity test in humanart-render.
        assert_eq!(pcg(0), 129_708_002);
        assert_ne!(pcg(1), pcg(0));
        assert_ne!(hash3(1, 2, 3), hash3(2, 1, 3));
        let u = unit(u32::MAX);
        assert!(u < 1.0 && u > 0.999);
    }

    #[test]
    fn value_noise_is_periodic_and_bounded() {
        for i in 0..200 {
            let x = i as f32 * 0.37;
            let y = i as f32 * 0.53;
            let a = value_noise(x, y, 8, 3);
            let b = value_noise(x + 8.0, y - 16.0, 8, 3);
            assert!((a - b).abs() < 1e-5, "periodic");
            assert!((0.0..1.0).contains(&a));
            let f = fbm(x, y, 4, 4, 0.5, 9);
            assert!((fbm(x + 4.0, y + 4.0, 4, 4, 0.5, 9) - f).abs() < 1e-5);
            assert!(f.abs() <= 0.5);
        }
        // Continuous across lattice lines.
        let l = value_noise(2.999_99, 1.5, 8, 1);
        let r = value_noise(3.000_01, 1.5, 8, 1);
        assert!((l - r).abs() < 1e-3);
    }
}
