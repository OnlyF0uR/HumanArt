//! Photographed skin colour detail (`todo.md` 1, item 4): a small periodic tile of the colour
//! variation of real skin (pores, mottling, redness, faint freckles) that the skin shader
//! multiplies into the albedo, anti-tiled like the micro-detail exemplar, at its own fine scale,
//! independent of the compositor's resolution.
//!
//! **Why.** The pigment model and the compositor give the skin its colour, zone by zone, at
//! about a millimetre per texel on the face; real skin varies at every scale below that, and a
//! skin without that variation reads as wax or plastic however good its shape and light. A
//! photogrammetry texture of a real face has it.
//!
//! **What is kept.** A square patch of a scan's albedo (cheek skin, away from hair and features)
//! is high-passed: each channel divided by its Gaussian blur ([`high_pass`]), so the capture's
//! lighting and the person's base colour (both low-frequency) drop out and only the relative
//! variation below a few millimetres stays. The ratio is stored as its natural logarithm,
//! `l = ln(ratio)`, encoded `v = 0.5 + l / LOG_RANGE` in 8 bits ([`encode`], [`decode`]): blending
//! log ratios keeps the mean at 1 and suits the shader's variance-preserving blend. The patch is
//! made periodic ([`make_periodic`]) by blending it with its half-shifted copy under a window, so
//! it repeats without a seam; the shader's anti-tiling hides the repetition.
//!
//! The patch is a small piece of one consenting person's cheek (HSRD-100, CC BY 4.0, ADR-056):
//! texture, not a likeness.

use super::error::SkinError;

/// Range of the stored log ratio: `l` in `[-LOG_RANGE / 2, LOG_RANGE / 2]` (ratios 0.61 to 1.65).
pub const LOG_RANGE: f32 = 1.0;
/// Largest tile side.
pub const MAX_SIDE: u32 = 1024;

/// The 8-bit code of log ratio `l`.
#[must_use]
pub fn encode(l: f32) -> u8 {
    let v = (0.5 + l / LOG_RANGE).clamp(0.0, 1.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Clamped to 0..=255.
    let b = (v * 255.0).round() as u8;
    b
}

/// The log ratio of an 8-bit code.
#[must_use]
pub fn decode(b: u8) -> f32 {
    (f32::from(b) / 255.0 - 0.5) * LOG_RANGE
}

/// A separable Gaussian blur of `side`² channels with standard deviation `sigma` texels, edges
/// clamped (the patch is not periodic yet).
fn blur(values: &[f32], side: usize, sigma: f32) -> Vec<f32> {
    let radius = (3.0 * sigma).ceil().max(1.0) as usize;
    let kernel: Vec<f32> = (0..=radius).map(|i| (-0.5 * (i as f32 / sigma).powi(2)).exp()).collect();
    let norm: f32 = kernel[0] + 2.0 * kernel[1..].iter().sum::<f32>();
    let pass = |src: &[f32], horizontal: bool| -> Vec<f32> {
        let mut out = vec![0.0; src.len()];
        for y in 0..side {
            for x in 0..side {
                let mut sum = 0.0;
                for (k, &w) in kernel.iter().enumerate() {
                    for sign in [-1_i64, 1] {
                        if k == 0 && sign == 1 {
                            continue;
                        }
                        let d = sign * k as i64;
                        let (sx, sy) = if horizontal { (x as i64 + d, y as i64) } else { (x as i64, y as i64 + d) };
                        let sx = sx.clamp(0, side as i64 - 1) as usize;
                        let sy = sy.clamp(0, side as i64 - 1) as usize;
                        sum += w * src[sy * side + sx];
                    }
                }
                out[y * side + x] = sum / norm;
            }
        }
        out
    };
    pass(&pass(values, true), false)
}

/// The log of each channel over its Gaussian blur (`sigma` texels): the relative colour variation
/// of `rgb` (`side`² linear colours) below the blur's scale.
///
/// # Errors
/// [`SkinError::Invalid`] when the sizes disagree or `sigma` is not positive.
pub fn high_pass(rgb: &[[f32; 3]], side: usize, sigma: f32) -> Result<Vec<[f32; 3]>, SkinError> {
    if rgb.len() != side * side || side == 0 || side > MAX_SIDE as usize || !(sigma > 0.0 && sigma.is_finite()) {
        return Err(SkinError::Invalid { what: "albedo detail: patch size or blur out of range" });
    }
    let mut out = vec![[0.0; 3]; rgb.len()];
    for c in 0..3 {
        let channel: Vec<f32> = rgb.iter().map(|p| p[c].max(1e-4)).collect();
        let low = blur(&channel, side, sigma);
        for ((o, &v), &m) in out.iter_mut().zip(&channel).zip(&low) {
            o[c] = (v / m.max(1e-4)).ln();
        }
    }
    Ok(out)
}

/// `values` (`side`² log ratios) made periodic: blended with its copy shifted by half a side under
/// the window `sin²` (1 at the centre, 0 at the edges) and its shifted complement, then re-centred
/// on a mean of zero and re-scaled to the original standard deviation (the blend averages two
/// independent textures where both weigh in, which would lower the contrast).
#[must_use]
pub fn make_periodic(values: &[[f32; 3]], side: usize) -> Vec<[f32; 3]> {
    if values.len() != side * side || side == 0 {
        return values.to_vec();
    }
    let window = |i: usize| (std::f32::consts::PI * (i as f32 + 0.5) / side as f32).sin().powi(2);
    let half = side / 2;
    let mut out = vec![[0.0; 3]; values.len()];
    for y in 0..side {
        for x in 0..side {
            let (sx, sy) = ((x + half) % side, (y + half) % side);
            let wa = window(x) * window(y);
            let wb = window(sx) * window(sy);
            let a = values[y * side + x];
            let b = values[sy * side + sx];
            let norm = (wa * wa + wb * wb).sqrt().max(1e-6);
            for c in 0..3 {
                out[y * side + x][c] = (wa * a[c] + wb * b[c]) / norm;
            }
        }
    }
    for c in 0..3 {
        let stats = |v: &[[f32; 3]]| {
            let n = v.len() as f32;
            let mean = v.iter().map(|p| p[c]).sum::<f32>() / n;
            let var = v.iter().map(|p| (p[c] - mean).powi(2)).sum::<f32>() / n;
            (mean, var.sqrt())
        };
        let (_, sd0) = stats(values);
        let (mean, sd) = stats(&out);
        let k = if sd > 1e-6 { sd0 / sd } else { 1.0 };
        for p in &mut out {
            p[c] = (p[c] - mean) * k;
        }
    }
    out
}

/// The tile as RGBA8 (log ratios encoded, alpha 255), row-major.
#[must_use]
pub fn to_rgba8(values: &[[f32; 3]]) -> Vec<u8> {
    values.iter().flat_map(|p| [encode(p[0]), encode(p[1]), encode(p[2]), 255]).collect()
}

/// The mip chain of an RGBA8 tile of `side` (a power of two): each level the 2 × 2 box mean of
/// the previous in the log domain (the codes are linear in the log ratio), down to 1 × 1.
///
/// # Errors
/// [`SkinError::Invalid`] when the side is not a power of two up to [`MAX_SIDE`] or the bytes do
/// not match it.
pub fn mips(rgba: &[u8], side: u32) -> Result<Vec<Vec<u8>>, SkinError> {
    if !side.is_power_of_two() || side > MAX_SIDE || rgba.len() != (side * side * 4) as usize {
        return Err(SkinError::Invalid { what: "albedo detail: tile side must be a power of two matching the bytes" });
    }
    let mut levels = vec![rgba.to_vec()];
    let mut s = side as usize;
    while s > 1 {
        let prev = levels.last().map_or(&[][..], Vec::as_slice);
        let n = s / 2;
        let mut next = vec![0_u8; n * n * 4];
        for y in 0..n {
            for x in 0..n {
                for c in 0..4 {
                    let at = |xx: usize, yy: usize| u32::from(prev[(yy * s + xx) * 4 + c]);
                    let sum = at(2 * x, 2 * y) + at(2 * x + 1, 2 * y) + at(2 * x, 2 * y + 1) + at(2 * x + 1, 2 * y + 1);
                    #[allow(clippy::cast_possible_truncation)] // A mean of four bytes.
                    let mean = ((sum + 2) / 4) as u8;
                    next[(y * n + x) * 4 + c] = mean;
                }
            }
        }
        levels.push(next);
        s = n;
    }
    Ok(levels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_round_trip_the_log_ratio() {
        for l in [-0.5, -0.2, 0.0, 0.13, 0.5] {
            assert!((decode(encode(l)) - l).abs() <= LOG_RANGE / 255.0, "{l}");
        }
        assert_eq!(encode(9.0), 255);
        assert_eq!(encode(-9.0), 0);
    }

    #[test]
    fn todo_1_high_pass_drops_the_light_and_keeps_the_grain() {
        // A gradient (the capture's light) times a fine checker (the grain).
        let side = 64;
        let rgb: Vec<[f32; 3]> = (0..side * side)
            .map(|i| {
                let (x, y) = (i % side, i / side);
                let light = 0.2 + 0.6 * x as f32 / side as f32;
                let grain = if (x + y) % 2 == 0 { 1.1 } else { 0.9 };
                [light * grain, light * grain * 0.8, light * grain * 0.6]
            })
            .collect();
        let hp = high_pass(&rgb, side, 4.0).unwrap();
        // The grain survives at about ±ln(1.1), the gradient does not: the left and right
        // halves have the same mean.
        let mean = |x0: usize, x1: usize| {
            let v: Vec<f32> = (0..side * side).filter(|i| (x0..x1).contains(&(i % side))).map(|i| hp[i][0]).collect();
            v.iter().sum::<f32>() / v.len() as f32
        };
        assert!((mean(8, 24) - mean(40, 56)).abs() < 0.01, "{} {}", mean(8, 24), mean(40, 56));
        let centre = hp[32 * side + 32][0];
        assert!((centre.abs() - 0.1).abs() < 0.03, "{centre}");
        assert!(high_pass(&rgb, side + 1, 4.0).is_err());
        assert!(high_pass(&rgb, side, 0.0).is_err());
    }

    #[test]
    fn todo_1_periodic_tile_has_no_seam_and_keeps_its_contrast() {
        let side = 64;
        let mut rng = humanart_core::rng::Rng::new(7);
        let v: Vec<[f32; 3]> = (0..side * side).map(|_| [rng.range_f32(-0.2, 0.2); 3]).collect();
        let p = make_periodic(&v, side);
        // Across the wrap the step is like a step inside.
        let step = |a: usize, b: usize| (0..side).map(|y| (p[y * side + a][0] - p[y * side + b][0]).abs()).sum::<f32>();
        let inside = step(31, 32);
        let wrap = step(side - 1, 0);
        assert!(wrap < inside * 2.0, "wrap {wrap} inside {inside}");
        let sd = |v: &[[f32; 3]]| (v.iter().map(|x| x[0] * x[0]).sum::<f32>() / v.len() as f32).sqrt();
        assert!((sd(&p) - sd(&v)).abs() < 0.01);
    }

    #[test]
    fn mips_halve_down_to_one_texel() {
        let rgba: Vec<u8> = (0..16 * 16).flat_map(|i| [if i % 2 == 0 { 100 } else { 200 }, 128, 128, 255]).collect();
        let levels = mips(&rgba, 16).unwrap();
        assert_eq!(levels.len(), 5);
        assert_eq!(levels[4].len(), 4);
        assert_eq!(levels[1][0], 150);
        assert!(mips(&rgba, 15).is_err());
    }
}
