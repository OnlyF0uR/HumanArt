//! The forward DCT and quantisation of the encoder: the Arai-Agui-Nakajima factorisation in 32-bit
//! floats (as libjpeg's `jfdctflt`), with the AAN scale factors folded into the divisors.

/// `cos(k * pi / 16) * sqrt(2)` for k = 1..7, and 1 for k = 0: the AAN scale factors.
const AAN_SCALE: [f64; 8] =
    [1.0, 1.387_039_845, 1.306_562_965, 1.175_875_602, 1.0, 0.785_694_958, 0.541_196_100, 0.275_899_379];

/// Reciprocal divisors for [`fdct_quantize`]: `1 / (q * s[row] * s[col] * 8)`, natural order.
pub(super) type FdctTable = [f32; 64];

/// The divisors of a quantisation table (natural order).
pub(super) fn divisors(quant: &[u16; 64]) -> FdctTable {
    let mut out = [0.0_f32; 64];
    for (i, o) in out.iter_mut().enumerate() {
        *o = (1.0 / (f64::from(quant[i]) * AAN_SCALE[i / 8] * AAN_SCALE[i % 8] * 8.0)) as f32;
    }
    out
}

/// One 8-point AAN forward DCT (outputs scaled by the AAN factors).
#[inline(always)]
fn fdct_1d(d: [f32; 8]) -> [f32; 8] {
    let tmp0 = d[0] + d[7];
    let tmp7 = d[0] - d[7];
    let tmp1 = d[1] + d[6];
    let tmp6 = d[1] - d[6];
    let tmp2 = d[2] + d[5];
    let tmp5 = d[2] - d[5];
    let tmp3 = d[3] + d[4];
    let tmp4 = d[3] - d[4];
    // Even part.
    let tmp10 = tmp0 + tmp3;
    let tmp13 = tmp0 - tmp3;
    let tmp11 = tmp1 + tmp2;
    let tmp12 = tmp1 - tmp2;
    let z1 = (tmp12 + tmp13) * core::f32::consts::FRAC_1_SQRT_2;
    // Odd part.
    let t10 = tmp4 + tmp5;
    let t11 = tmp5 + tmp6;
    let t12 = tmp6 + tmp7;
    let z5 = (t10 - t12) * 0.382_683_43;
    let z2 = 0.541_196_1 * t10 + z5;
    let z4 = 1.306_563 * t12 + z5;
    let z3 = t11 * core::f32::consts::FRAC_1_SQRT_2;
    let z11 = tmp7 + z3;
    let z13 = tmp7 - z3;
    [tmp10 + tmp11, z11 + z4, tmp13 + z1, z13 - z2, tmp10 - tmp11, z13 + z2, tmp13 - z1, z11 - z4]
}

/// The quantised DCT of 64 level-shifted samples (row-major), in natural order. DC is clamped to
/// ±2047 and AC to ±1023, the ranges the baseline Huffman tables code.
#[inline]
pub(super) fn fdct_quantize(samples: &[f32; 64], table: &FdctTable, out: &mut [i16; 64]) {
    let mut ws = [0.0_f32; 64];
    for (src, dst) in samples.as_chunks::<8>().0.iter().zip(ws.as_chunks_mut::<8>().0) {
        *dst = fdct_1d(*src);
    }
    for col in 0..8 {
        let c = |row: usize| ws[row * 8 + col];
        let v = fdct_1d([c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7)]);
        for (row, value) in v.into_iter().enumerate() {
            let i = row * 8 + col;
            let limit = if i == 0 { 2047.0 } else { 1023.0 };
            out[i] = (value * table[i]).round().clamp(-limit, limit) as i16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    #[test]
    fn matches_the_exact_dct() {
        let mut rng = Rng::new(0xFDC7);
        let quant = [1_u16; 64];
        let table = divisors(&quant);
        let mut worst = 0_i32;
        for _ in 0..500 {
            let mut samples = [0.0_f32; 64];
            for s in &mut samples {
                *s = rng.index(256) as f32 - 128.0;
            }
            let mut ours = [0_i16; 64];
            fdct_quantize(&samples, &table, &mut ours);
            for v in 0..8 {
                for u in 0..8 {
                    let cu = if u == 0 { core::f64::consts::FRAC_1_SQRT_2 } else { 1.0 };
                    let cv = if v == 0 { core::f64::consts::FRAC_1_SQRT_2 } else { 1.0 };
                    let mut sum = 0.0;
                    for y in 0..8 {
                        for x in 0..8 {
                            sum += f64::from(samples[y * 8 + x])
                                * ((2 * x + 1) as f64 * u as f64 * core::f64::consts::PI / 16.0).cos()
                                * ((2 * y + 1) as f64 * v as f64 * core::f64::consts::PI / 16.0).cos();
                        }
                    }
                    let exact = (cu * cv * sum / 4.0).round() as i32;
                    worst = worst.max((i32::from(ours[v * 8 + u]) - exact).abs());
                }
            }
        }
        assert!(worst <= 1, "worst {worst}");
    }

    #[test]
    fn extreme_blocks_stay_in_the_baseline_ranges() {
        let table = divisors(&[1; 64]);
        let mut out = [0_i16; 64];
        fdct_quantize(&[-128.0; 64], &table, &mut out);
        assert_eq!(out[0], -1024);
        assert!(out[1..].iter().all(|&c| c == 0));
        let checker: [f32; 64] = core::array::from_fn(|i| if (i / 8 + i % 8) % 2 == 0 { 127.0 } else { -128.0 });
        fdct_quantize(&checker, &table, &mut out);
        assert!(out.iter().all(|&c| c.abs() <= 1023));
        assert!(out[63].abs() > 800);
    }
}
