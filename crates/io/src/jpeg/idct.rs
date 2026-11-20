//! The inverse DCT: the Arai-Agui-Nakajima factorisation in 32-bit floats (as libjpeg's
//! `jidctflt`), with dequantisation and the AAN scale factors folded into one prescaled table.
//!
//! Accuracy: against an exact double-precision IDCT rounded to the nearest integer, every output
//! sample is within 1 (most are equal); the test `matches_the_exact_idct_within_one` checks this on
//! random blocks of every magnitude a legal 8-bit file can hold. Floats also mean a hostile
//! coefficient can never overflow anything: the worst it does is saturate the output.

/// `cos(k * pi / 16) * sqrt(2)` for k = 1..7, and 1 for k = 0: the AAN scale factors.
const AAN_SCALE: [f64; 8] =
    [1.0, 1.387_039_845, 1.306_562_965, 1.175_875_602, 1.0, 0.785_694_958, 0.541_196_100, 0.275_899_379];

/// A dequantisation table prescaled for [`idct_block`]: `q[i] * s[row] * s[col] / 8`, natural order.
pub(super) type IdctTable = [f32; 64];

/// Prescales a quantisation table (natural order) for [`idct_block`].
pub(super) fn prescale(quant: &[u16; 64]) -> IdctTable {
    let mut out = [0.0_f32; 64];
    for (i, o) in out.iter_mut().enumerate() {
        *o = (f64::from(quant[i]) * AAN_SCALE[i / 8] * AAN_SCALE[i % 8] / 8.0) as f32;
    }
    out
}

/// 2^23: adding it to a float in [0, 2^23) rounds it to an integer held in the low mantissa bits.
const MAGIC: f32 = 8_388_608.0;

/// Turns a level-shifted float sample into a byte: `v + 128.5` saturated to [0, 255] (NaN to 0)
/// and truncated, which rounds `v + 128` half up. Branch-free and without a float-to-integer cast
/// (whose saturation the compiler does not vectorise): the floor comes from rounding through
/// [`MAGIC`] and stepping down where that rounded up, and the byte from the mantissa bits.
#[inline(always)]
fn to_byte(v: f32) -> u8 {
    // `max` returns the other operand for NaN, so NaN becomes 0 (`clamp` would keep it).
    #[allow(clippy::manual_clamp)]
    let y = (v + 128.5).max(0.0).min(255.0);
    let r = (y + MAGIC) - MAGIC;
    let floor = if r > y { r - 1.0 } else { r };
    // An integer in [0, 255] plus 2^23 is exact, with the integer in the low byte.
    (floor + MAGIC).to_bits() as u8
}

/// The inverse DCT of one block of coefficients (natural order) into an 8×8 area of `out` whose
/// rows are `stride` bytes apart. `out` must hold `7 * stride + 8` bytes.
///
/// Written so the compiler vectorises it: the column pass is a loop over the eight columns whose
/// body reads one coefficient of every row (eight lanes of contiguous rows), the row pass a loop
/// over the rows; every column and row goes through exactly the arithmetic of [`idct_1d`].
#[inline]
pub(super) fn idct_block(coef: &[i16; 64], table: &IdctTable, out: &mut [u8], stride: usize) {
    let (rows, _) = coef.as_chunks::<8>();
    // An OR over the AC coefficients (the DC masked out) in eight lanes, so it vectorises.
    let mut any = rows[0];
    any[0] = 0;
    for row in &rows[1..] {
        for (a, &c) in any.iter_mut().zip(row) {
            *a |= c;
        }
    }
    if any == [0; 8] {
        let v = to_byte(f32::from(coef[0]) * table[0]);
        for row in out.chunks_mut(stride).take(8) {
            row[..8].fill(v);
        }
        return;
    }
    let mut d = [[0.0_f32; 8]; 8];
    for ((lanes, c), q) in d.iter_mut().zip(rows).zip(table.as_chunks::<8>().0) {
        for i in 0..8 {
            lanes[i] = f32::from(c[i]) * q[i];
        }
    }
    columns_then_rows(&d, out, stride);
}

/// The two passes of [`idct_block`] on dequantised coefficients. Kept out of line: inlined into
/// its caller (rustc 1.99) the column loop is no longer vectorised.
#[inline(never)]
fn columns_then_rows(d: &[[f32; 8]; 8], out: &mut [u8], stride: usize) {
    // Pass 1: the columns.
    let mut ws = [[0.0_f32; 8]; 8];
    for col in 0..8 {
        let v = idct_1d([d[0][col], d[1][col], d[2][col], d[3][col], d[4][col], d[5][col], d[6][col], d[7][col]]);
        for (row, value) in v.into_iter().enumerate() {
            ws[row][col] = value;
        }
    }
    // Pass 2: the rows, into bytes.
    let mut bytes = [[0_u8; 8]; 8];
    for (b, w) in bytes.iter_mut().zip(&ws) {
        for (o, value) in b.iter_mut().zip(idct_1d(*w)) {
            *o = to_byte(value);
        }
    }
    for (line, b) in out.chunks_mut(stride).zip(&bytes) {
        line[..8].copy_from_slice(b);
    }
}

/// One 8-point AAN inverse DCT of prescaled inputs.
#[inline(always)]
fn idct_1d(d: [f32; 8]) -> [f32; 8] {
    // Even part.
    let tmp10 = d[0] + d[4];
    let tmp11 = d[0] - d[4];
    let tmp13 = d[2] + d[6];
    let tmp12 = (d[2] - d[6]) * core::f32::consts::SQRT_2 - tmp13;
    let e0 = tmp10 + tmp13;
    let e3 = tmp10 - tmp13;
    let e1 = tmp11 + tmp12;
    let e2 = tmp11 - tmp12;
    // Odd part.
    let z13 = d[5] + d[3];
    let z10 = d[5] - d[3];
    let z11 = d[1] + d[7];
    let z12 = d[1] - d[7];
    let o7 = z11 + z13;
    let tmp11 = (z11 - z13) * core::f32::consts::SQRT_2;
    let z5 = (z10 + z12) * 1.847_759;
    let tmp10 = 1.082_392_2 * z12 - z5;
    let tmp12 = -2.613_126 * z10 + z5;
    let o6 = tmp12 - o7;
    let o5 = tmp11 - o6;
    let o4 = tmp10 + o5;
    [e0 + o7, e1 + o6, e2 + o5, e3 - o4, e3 + o4, e2 - o5, e1 - o6, e0 - o7]
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    /// The conversion this module had before the branch-free one.
    fn saturating_cast(v: f32) -> u8 {
        (v + 128.5).clamp(0.0, 255.0) as u8
    }

    #[test]
    fn branch_free_bytes_equal_the_saturating_cast() {
        let mut rng = Rng::new(0xB17E);
        let mut values = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX, f32::MIN, 0.0, -0.0, -128.5, 126.5];
        // Every half step and its neighbours over the range and past both ends.
        for k in -1000..=1000 {
            let v = k as f32 * 0.5 - 128.0;
            values.extend([v, v.next_up(), v.next_down()]);
        }
        values.extend((0..100_000).map(|_| rng.range_f32(-1000.0, 1000.0)));
        for v in values {
            assert_eq!(to_byte(v), saturating_cast(v), "{v}");
        }
    }

    // The scalar column-by-column, row-by-row form this module had before the eight-lane one:
    // the reference the eight-lane form must equal exactly.
    /// The inverse DCT of one block of coefficients (natural order) into an 8×8 area of `out` whose
    /// rows are `stride` bytes apart. `out` must hold `7 * stride + 8` bytes.
    fn scalar_idct_block(coef: &[i16; 64], table: &IdctTable, out: &mut [u8], stride: usize) {
        if coef[1..].iter().all(|&c| c == 0) {
            let v = saturating_cast(f32::from(coef[0]) * table[0]);
            for row in out.chunks_mut(stride).take(8) {
                row[..8].fill(v);
            }
            return;
        }
        let mut ws = [0.0_f32; 64];
        // Pass 1: columns.
        for col in 0..8 {
            let c = |row: usize| f32::from(coef[row * 8 + col]) * table[row * 8 + col];
            if (1..8).all(|row| coef[row * 8 + col] == 0) {
                let dc = c(0);
                for row in 0..8 {
                    ws[row * 8 + col] = dc;
                }
                continue;
            }
            let v = idct_1d([c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7)]);
            for (row, value) in v.into_iter().enumerate() {
                ws[row * 8 + col] = value;
            }
        }
        // Pass 2: rows, into bytes.
        for (row, line) in ws.as_chunks::<8>().0.iter().zip(out.chunks_mut(stride)) {
            let v = idct_1d(*row);
            for (o, value) in line[..8].iter_mut().zip(v) {
                *o = saturating_cast(value);
            }
        }
    }

    /// One 8-point AAN inverse DCT of prescaled inputs.
    fn idct_1d(d: [f32; 8]) -> [f32; 8] {
        // Even part.
        let tmp10 = d[0] + d[4];
        let tmp11 = d[0] - d[4];
        let tmp13 = d[2] + d[6];
        let tmp12 = (d[2] - d[6]) * core::f32::consts::SQRT_2 - tmp13;
        let e0 = tmp10 + tmp13;
        let e3 = tmp10 - tmp13;
        let e1 = tmp11 + tmp12;
        let e2 = tmp11 - tmp12;
        // Odd part.
        let z13 = d[5] + d[3];
        let z10 = d[5] - d[3];
        let z11 = d[1] + d[7];
        let z12 = d[1] - d[7];
        let o7 = z11 + z13;
        let tmp11 = (z11 - z13) * core::f32::consts::SQRT_2;
        let z5 = (z10 + z12) * 1.847_759;
        let tmp10 = 1.082_392_2 * z12 - z5;
        let tmp12 = -2.613_126 * z10 + z5;
        let o6 = tmp12 - o7;
        let o5 = tmp11 - o6;
        let o4 = tmp10 + o5;
        [e0 + o7, e1 + o6, e2 + o5, e3 - o4, e3 + o4, e2 - o5, e1 - o6, e0 - o7]
    }

    /// The textbook IDCT of T.81 A.3.3 in doubles, rounded, level-shifted and clamped.
    fn exact(coef: &[i16; 64], quant: &[u16; 64]) -> [u8; 64] {
        let mut out = [0_u8; 64];
        for y in 0..8 {
            for x in 0..8 {
                let mut sum = 0.0_f64;
                for v in 0..8 {
                    for u in 0..8 {
                        let cu = if u == 0 { core::f64::consts::FRAC_1_SQRT_2 } else { 1.0 };
                        let cv = if v == 0 { core::f64::consts::FRAC_1_SQRT_2 } else { 1.0 };
                        let f = f64::from(coef[v * 8 + u]) * f64::from(quant[v * 8 + u]);
                        sum += cu
                            * cv
                            * f
                            * ((2 * x + 1) as f64 * u as f64 * core::f64::consts::PI / 16.0).cos()
                            * ((2 * y + 1) as f64 * v as f64 * core::f64::consts::PI / 16.0).cos();
                    }
                }
                out[y * 8 + x] = (sum / 4.0 + 128.0).round().clamp(0.0, 255.0) as u8;
            }
        }
        out
    }

    #[test]
    fn matches_the_exact_idct_within_one() {
        let mut rng = Rng::new(0x1DC7);
        let mut worst = 0;
        let mut differing = 0;
        let blocks = 3000;
        for case in 0..blocks {
            let mut quant = [0_u16; 64];
            for q in &mut quant {
                *q = 1 + rng.index(if case % 3 == 0 { 4 } else { 60 }) as u16;
            }
            // Sparse, dense, and high-amplitude blocks.
            let mut coef = [0_i16; 64];
            let density = [4, 16, 64][case % 3];
            for _ in 0..density {
                let i = rng.index(64);
                let amplitude = [30, 300, 1000][rng.index(3)];
                coef[i] = (rng.index(2 * amplitude + 1) as i32 - amplitude as i32) as i16 / quant[i] as i16;
            }
            let mut ours = [0_u8; 64];
            idct_block(&coef, &prescale(&quant), &mut ours, 8);
            let reference = exact(&coef, &quant);
            for (a, b) in ours.iter().zip(&reference) {
                let d = a.abs_diff(*b);
                worst = worst.max(d);
                differing += usize::from(d != 0);
            }
        }
        println!("idct: {differing} of {} samples differ by 1 from the exact IDCT", blocks * 64);
        assert!(worst <= 1, "worst error {worst}");
        assert!(differing * 100 < blocks * 64, "{differing} samples off by one");
    }

    #[test]
    fn flat_blocks_and_saturation() {
        let mut quant = [1_u16; 64];
        quant[0] = 8;
        let mut coef = [0_i16; 64];
        coef[0] = 10; // 10 * 8 / 8 = +10
        let mut out = [0_u8; 8 * 10];
        idct_block(&coef, &prescale(&quant), &mut out, 10);
        assert!(out.chunks(10).all(|row| row[..8].iter().all(|&v| v == 138) && row[8..] == [0, 0]));
        // Absurd coefficients saturate instead of overflowing.
        let coef = [i16::MAX; 64];
        idct_block(&coef, &prescale(&[u16::MAX; 64]), &mut out, 8);
        let coef = [i16::MIN; 64];
        idct_block(&coef, &prescale(&[u16::MAX; 64]), &mut out, 8);
        assert!(out[..64].contains(&0));
    }

    #[test]
    fn eight_lane_form_equals_the_scalar_transform_exactly() {
        let mut rng = Rng::new(0x1A7E);
        for case in 0..5000 {
            let mut quant = [0_u16; 64];
            for q in &mut quant {
                *q = 1 + rng.index(if case % 3 == 0 { 4 } else { 255 }) as u16;
            }
            let mut coef = [0_i16; 64];
            for _ in 0..[1, 3, 8, 64][case % 4] {
                let i = rng.index(64);
                coef[i] = (rng.index(4001) as i32 - 2000) as i16;
            }
            let table = prescale(&quant);
            let (mut a, mut b) = ([0_u8; 9 * 8], [0_u8; 9 * 8]);
            idct_block(&coef, &table, &mut a, 9);
            scalar_idct_block(&coef, &table, &mut b, 9);
            assert_eq!(a, b, "case {case}");
        }
    }

    #[test]
    #[ignore = "timing: cargo test --release -p humanart-io --features jpeg idct_speed -- --ignored --nocapture"]
    fn idct_speed() {
        use std::time::Instant;
        let mut rng = Rng::new(5);
        let table = prescale(&[3; 64]);
        let blocks: Vec<[i16; 64]> = (0..4096)
            .map(|_| {
                let mut c = [0_i16; 64];
                for _ in 0..10 {
                    c[rng.index(20)] = (rng.index(201) as i32 - 100) as i16;
                }
                c
            })
            .collect();
        let mut out = vec![0_u8; 64];
        for round in 0..3 {
            let t = Instant::now();
            for _ in 0..50 {
                for b in &blocks {
                    idct_block(b, &table, &mut out, 8);
                }
            }
            let lanes = t.elapsed().as_secs_f64();
            let t = Instant::now();
            for _ in 0..50 {
                for b in &blocks {
                    scalar_idct_block(b, &table, &mut out, 8);
                }
            }
            let scalar = t.elapsed().as_secs_f64();
            let n = 50.0 * blocks.len() as f64;
            println!(
                "round {round}: vectorised {:.1} ns/block, scalar {:.1} ns/block",
                lanes / n * 1e9,
                scalar / n * 1e9
            );
        }
    }
}
