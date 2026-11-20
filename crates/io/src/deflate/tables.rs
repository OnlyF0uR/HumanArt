//! The constant tables of RFC 1951 that the decoder and the encoder share.

/// Number of literal/length symbols a dynamic block may declare (0..=285).
pub(super) const NUM_LITLEN: usize = 286;
/// Number of distance symbols a dynamic block may declare (0..=29).
pub(super) const NUM_DIST: usize = 30;
/// Number of symbols of the code-length code.
pub(super) const NUM_PRECODE: usize = 19;
/// The end-of-block symbol.
pub(super) const END_OF_BLOCK: usize = 256;
/// Shortest match DEFLATE can express.
pub(super) const MIN_MATCH: usize = 3;
/// Longest match DEFLATE can express.
pub(super) const MAX_MATCH: usize = 258;
/// Largest match distance.
pub(super) const WINDOW: usize = 32_768;

/// Base match length of length symbols 257..=285.
pub(super) const LEN_BASE: [u16; 29] =
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
/// Extra bits of length symbols 257..=285.
pub(super) const LEN_EXTRA: [u8; 29] =
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
/// Base distance of distance symbols 0..=29.
pub(super) const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
    8193, 12289, 16385, 24577,
];
/// Extra bits of distance symbols 0..=29.
pub(super) const DIST_EXTRA: [u8; 30] =
    [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
/// The order in which a dynamic block stores the code-length code's own lengths.
pub(super) const PRECODE_ORDER: [u8; NUM_PRECODE] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/// The code length of literal/length symbol `sym` in a fixed-Huffman block (288 symbols).
pub(super) const fn fixed_litlen_len(sym: usize) -> u8 {
    match sym {
        0..=143 | 280..=287 => 8,
        144..=255 => 9,
        _ => 7,
    }
}

/// The code length of every distance symbol in a fixed-Huffman block (32 symbols).
pub(super) const FIXED_DIST_LEN: u8 = 5;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_and_distance_tables_are_contiguous() {
        // Every base is the previous base plus the previous range, so no length or distance is
        // unreachable or covered twice (length 258 has its own symbol and ends the list).
        for i in 1..28 {
            assert_eq!(u32::from(LEN_BASE[i]), u32::from(LEN_BASE[i - 1]) + (1 << LEN_EXTRA[i - 1]), "length {i}");
        }
        assert_eq!(usize::from(LEN_BASE[27]) + (1 << LEN_EXTRA[27]) - 1, MAX_MATCH);
        assert_eq!(usize::from(LEN_BASE[28]), MAX_MATCH);
        for i in 1..30 {
            assert_eq!(u32::from(DIST_BASE[i]), u32::from(DIST_BASE[i - 1]) + (1 << DIST_EXTRA[i - 1]), "distance {i}");
        }
        assert_eq!(usize::from(DIST_BASE[29]) + (1 << DIST_EXTRA[29]) - 1, WINDOW);
        let mut seen = [false; NUM_PRECODE];
        for &s in &PRECODE_ORDER {
            seen[usize::from(s)] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn fixed_code_is_complete() {
        let kraft: u32 = (0..288).map(|s| 1_u32 << (15 - fixed_litlen_len(s))).sum();
        assert_eq!(kraft, 1 << 15);
        assert_eq!(32 * (1_u32 << (15 - FIXED_DIST_LEN)), 1 << 15);
    }
}
