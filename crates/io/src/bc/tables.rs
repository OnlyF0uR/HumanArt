//! The constant tables of BC6H and BC7: partitions, anchor (fix-up) indices and interpolation
//! weights, as given by the D3D11 functional specification and the Khronos Data Format
//! Specification (sections "BC6H" and "BC7"). They are shared with the GPU encoders, which receive
//! them through [`gpu_words`] so the shaders and the CPU decoders cannot disagree.

/// The 64 two-subset partitions of BC7 (BC6H uses the first 32): bit `i` set means texel `i`
/// (row-major in the 4×4 block) belongs to subset 1.
pub const PARTITIONS_2: [u16; 64] = [
    0xCCCC, 0x8888, 0xEEEE, 0xECC8, 0xC880, 0xFEEC, 0xFEC8, 0xEC80, 0xC800, 0xFFEC, 0xFE80, 0xE800, 0xFFE8, 0xFF00,
    0xFFF0, 0xF000, 0xF710, 0x008E, 0x7100, 0x08CE, 0x008C, 0x7310, 0x3100, 0x8CCE, 0x088C, 0x3110, 0x6666, 0x366C,
    0x17E8, 0x0FF0, 0x718E, 0x399C, 0xAAAA, 0xF0F0, 0x5A5A, 0x33CC, 0x3C3C, 0x55AA, 0x9696, 0xA55A, 0x73CE, 0x13C8,
    0x324C, 0x3BDC, 0x6996, 0xC33C, 0x9966, 0x0660, 0x0272, 0x04E4, 0x4E40, 0x2720, 0xC936, 0x936C, 0x39C6, 0x639C,
    0x9336, 0x9CC6, 0x817E, 0xE718, 0xCCF0, 0x0FCC, 0x7744, 0xEE22,
];

/// The 64 three-subset partitions of BC7: bits `2i..2i+2` hold the subset (0 to 2) of texel `i`.
pub const PARTITIONS_3: [u32; 64] = [
    0xAA685050, 0x6A5A5040, 0x5A5A4200, 0x5450A0A8, 0xA5A50000, 0xA0A05050, 0x5555A0A0, 0x5A5A5050, 0xAA550000,
    0xAA555500, 0xAAAA5500, 0x90909090, 0x94949494, 0xA4A4A4A4, 0xA9A59450, 0x2A0A4250, 0xA5945040, 0x0A425054,
    0xA5A5A500, 0x55A0A0A0, 0xA8A85454, 0x6A6A4040, 0xA4A45000, 0x1A1A0500, 0x0050A4A4, 0xAAA59090, 0x14696914,
    0x69691400, 0xA08585A0, 0xAA821414, 0x50A4A450, 0x6A5A0200, 0xA9A58000, 0x5090A0A8, 0xA8A09050, 0x24242424,
    0x00AA5500, 0x24924924, 0x24499224, 0x50A50A50, 0x500AA550, 0xAAAA4444, 0x66660000, 0xA5A0A5A0, 0x50A050A0,
    0x69286928, 0x44AAAA44, 0x66666600, 0xAA444444, 0x54A854A8, 0x95809580, 0x96969600, 0xA85454A8, 0x80959580,
    0xAA141414, 0x96960000, 0xAAAA1414, 0xA05050A0, 0xA0A5A5A0, 0x96000000, 0x40804080, 0xA9A8A9A8, 0xAAAAAA44,
    0x2A4A5254,
];

/// The anchor texel of subset 1 for each two-subset partition (subset 0's anchor is texel 0).
pub const ANCHOR_2: [u8; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, //
    15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2, //
    15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, //
    6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15,
];

/// The anchor texel of subset 1 for each three-subset partition.
pub const ANCHOR_3_1: [u8; 64] = [
    3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, //
    3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5, 15, 15, //
    8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, //
    3, 15, 5, 5, 5, 8, 5, 10, 5, 10, 8, 13, 15, 12, 3, 3,
];

/// The anchor texel of subset 2 for each three-subset partition.
pub const ANCHOR_3_2: [u8; 64] = [
    15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, //
    15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6, 10, 15, 15, 10, 8, //
    15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, //
    15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8,
];

/// Interpolation weights (out of 64) for 2-bit indices.
pub const WEIGHTS_2: [u8; 4] = [0, 21, 43, 64];
/// Interpolation weights (out of 64) for 3-bit indices.
pub const WEIGHTS_3: [u8; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
/// Interpolation weights (out of 64) for 4-bit indices.
pub const WEIGHTS_4: [u8; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

/// The weights for `bits`-bit indices (2, 3 or 4); an empty slice for any other count.
#[must_use]
pub const fn weights(bits: u32) -> &'static [u8] {
    match bits {
        2 => &WEIGHTS_2,
        3 => &WEIGHTS_3,
        4 => &WEIGHTS_4,
        _ => &[],
    }
}

/// The subset (0, 1 or 2) of texel `texel` (0..16) in partition `partition` of a block with
/// `subsets` subsets (1, 2 or 3). Out-of-range arguments give subset 0.
#[must_use]
pub const fn subset(subsets: u32, partition: usize, texel: usize) -> usize {
    if texel >= 16 || partition >= 64 {
        return 0;
    }
    match subsets {
        2 => ((PARTITIONS_2[partition] >> texel) & 1) as usize,
        3 => ((PARTITIONS_3[partition] >> (2 * texel)) & 3) as usize,
        _ => 0,
    }
}

/// Whether texel `texel` is an anchor of partition `partition` with `subsets` subsets: its index is
/// stored with one bit less (its most significant bit is implicitly zero).
#[must_use]
pub const fn is_anchor(subsets: u32, partition: usize, texel: usize) -> bool {
    if texel == 0 {
        return true;
    }
    if partition >= 64 {
        return false;
    }
    match subsets {
        2 => texel == ANCHOR_2[partition] as usize,
        3 => texel == ANCHOR_3_1[partition] as usize || texel == ANCHOR_3_2[partition] as usize,
        _ => false,
    }
}

/// The index weight of BC7's optimal single-colour encoding: index 1 of a 2-bit index set.
pub const SINGLE_COLOR_WEIGHT: u32 = WEIGHTS_2[1] as u32;

/// The 8-bit value a BC7 endpoint of `bits` bits (p-bit included) decodes to.
const fn expand_bits(value: u32, bits: u32) -> u32 {
    let x = value << (8 - bits);
    x | (x >> bits)
}

/// Optimal single-colour endpoints for BC7 mode 5 colour (7-bit endpoints, no p-bits, 2-bit
/// indices): for every 8-bit value `v`, the endpoints `(a, b)` (packed `a | b << 8`) whose palette
/// entry at index 1 (weight [`SINGLE_COLOR_WEIGHT`], 21/64) decodes to exactly `v`. Every value has
/// such a pair (checked by the tests against an exhaustive search), so a block of one colour is
/// encoded without error: the colour channels from this table, alpha (8-bit endpoints) as itself.
///
/// The parity of an 8-bit value does not matter here, unlike the p-bit modes, where one p-bit
/// shared by the channels of an endpoint makes colours whose channels disagree in parity one step
/// off. Of the exact pairs the search meets, the one with the closest endpoints is kept, then the
/// smallest `a`.
pub const BC7_MODE5_SINGLE: [u16; 256] = single_color_table();

/// Builds [`BC7_MODE5_SINGLE`]: for each `a`, only the `b` values next to the exact solution of the
/// interpolation are tried, so the search is 256 x 128 x 3 steps.
const fn single_color_table() -> [u16; 256] {
    let w = SINGLE_COLOR_WEIGHT;
    let mut table = [0_u16; 256];
    let mut v = 0;
    while v < 256 {
        let mut best = 0_u16;
        let mut best_gap = u32::MAX;
        let mut a = 0;
        while a < 128 {
            let ea = expand_bits(a, 7);
            // (ea * (64 - w) + eb * w + 32) >> 6 == v  =>  eb is near (64 v - (64 - w) ea) / w.
            let target = (64 * v as i32 - (64 - w as i32) * ea as i32) / w as i32;
            let centre = target / 2;
            let mut d = -1;
            while d <= 1 {
                let b = centre + d;
                if b >= 0 && b < 128 {
                    let eb = expand_bits(b as u32, 7);
                    let value = (ea * (64 - w) + eb * w + 32) >> 6;
                    let gap = ea.abs_diff(eb);
                    if value == v && gap < best_gap {
                        best_gap = gap;
                        best = (a | ((b as u32) << 8)) as u16;
                    }
                }
                d += 1;
            }
            a += 1;
        }
        table[v as usize] = best;
        v += 1;
    }
    table
}

/// Number of `u32` words in [`single_color_gpu_words`].
pub const SINGLE_COLOR_GPU_WORDS: usize = 128;

/// [`BC7_MODE5_SINGLE`] as the GPU encoder reads it: two entries per word, value `2 i` in the low
/// half of word `i`, `2 i + 1` in the high half.
#[must_use]
pub fn single_color_gpu_words() -> [u32; SINGLE_COLOR_GPU_WORDS] {
    let mut words = [0_u32; SINGLE_COLOR_GPU_WORDS];
    for (word, pair) in words.iter_mut().zip(BC7_MODE5_SINGLE.as_chunks::<2>().0) {
        *word = u32::from(pair[0]) | (u32::from(pair[1]) << 16);
    }
    words
}

/// Number of `u32` words in [`gpu_words`].
pub const GPU_WORDS: usize = 64 + 64 + 64;

/// The partition and anchor tables as the GPU encoders read them from a storage buffer: words
/// `0..64` the two-subset masks, `64..128` the three-subset words, `128..192` the anchors of each
/// partition packed as `ANCHOR_2 | ANCHOR_3_1 << 8 | ANCHOR_3_2 << 16`.
#[must_use]
pub fn gpu_words() -> [u32; GPU_WORDS] {
    let mut words = [0_u32; GPU_WORDS];
    for p in 0..64 {
        words[p] = u32::from(PARTITIONS_2[p]);
        words[64 + p] = PARTITIONS_3[p];
        words[128 + p] = u32::from(ANCHOR_2[p]) | (u32::from(ANCHOR_3_1[p]) << 8) | (u32::from(ANCHOR_3_2[p]) << 16);
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_anchor_lies_in_its_subset_and_partitions_use_every_subset() {
        for p in 0..64 {
            assert_eq!(subset(2, p, ANCHOR_2[p] as usize), 1, "two-subset partition {p}");
            assert_eq!(subset(3, p, ANCHOR_3_1[p] as usize), 1, "three-subset partition {p}, subset 1");
            assert_eq!(subset(3, p, ANCHOR_3_2[p] as usize), 2, "three-subset partition {p}, subset 2");
            assert_eq!(subset(2, p, 0), 0, "texel 0 is always in subset 0");
            assert_eq!(subset(3, p, 0), 0);
            let mut seen = [false; 3];
            for t in 0..16 {
                seen[subset(3, p, t)] = true;
                assert!(subset(3, p, t) < 3, "partition {p} texel {t} has subset 3");
            }
            assert_eq!(seen, [true; 3], "three-subset partition {p}");
        }
    }

    #[test]
    fn weights_are_symmetric() {
        for bits in 2..=4 {
            let w = weights(bits);
            for (i, &v) in w.iter().enumerate() {
                assert_eq!(u32::from(v) + u32::from(w[w.len() - 1 - i]), 64, "{bits}-bit weight {i}");
            }
        }
    }

    #[test]
    fn gpu_words_pack_the_tables() {
        let words = gpu_words();
        assert_eq!(words[0], 0xCCCC);
        assert_eq!(words[64], 0xAA685050);
        assert_eq!(words[128 + 17], 2 | (3 << 8) | (8 << 16));
    }

    #[test]
    fn skn_04_mode5_single_color_table_is_exact_for_every_value() {
        // Exhaustive: every 8-bit value has an exact pair, and the table holds one.
        for v in 0..256_u32 {
            let entry = BC7_MODE5_SINGLE[v as usize];
            let (a, b) = (u32::from(entry & 0x7F), u32::from(entry >> 8));
            assert_eq!(u32::from(entry) & !0x7F7F, 0, "value {v}: entry {entry:#x} has stray bits");
            let (ea, eb) = (expand_bits(a, 7), expand_bits(b, 7));
            assert_eq!((ea * 43 + eb * 21 + 32) >> 6, v, "value {v}: endpoints {a}, {b}");
            let any =
                (0..128).any(|x| (0..128).any(|y| (expand_bits(x, 7) * 43 + expand_bits(y, 7) * 21 + 32) >> 6 == v));
            assert!(any);
        }
        let words = single_color_gpu_words();
        assert_eq!(words[0] & 0xFFFF, u32::from(BC7_MODE5_SINGLE[0]));
        assert_eq!(words[127] >> 16, u32::from(BC7_MODE5_SINGLE[255]));
    }
}
