//! Constant tables of ITU T.81: the zigzag order and the example tables of Annex K.

/// The natural (row-major) index of the coefficient at each zigzag position (T.81 figure A.6).
pub(super) const ZIGZAG: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
    28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54,
    47, 55, 62, 63,
];

/// The luminance quantisation table of Annex K.1, in natural order.
pub(super) const STD_LUMA_QUANT: [u8; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51,
    87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
];

/// The chrominance quantisation table of Annex K.1, in natural order.
pub(super) const STD_CHROMA_QUANT: [u8; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99,
];

/// A Huffman table specification as a DHT segment carries it: how many codes there are of each
/// length 1 to 16, and the symbols in order of increasing code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct HuffSpec {
    /// The number of codes of each length 1 to 16.
    pub counts: [u8; 16],
    /// The symbols, shortest code first.
    pub symbols: Vec<u8>,
}

/// The DC symbols of both example tables: the twelve difference categories.
const STD_DC_SYMBOLS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// Table K.3: luminance DC.
const STD_DC_LUMA_COUNTS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
/// Table K.4: chrominance DC.
const STD_DC_CHROMA_COUNTS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
/// Table K.5: luminance AC.
const STD_AC_LUMA_COUNTS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D];
const STD_AC_LUMA_SYMBOLS: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14,
    0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09,
    0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A,
    0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65,
    0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88,
    0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9,
    0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA,
    0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
    0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
];
/// Table K.6: chrominance AC.
const STD_AC_CHROMA_COUNTS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
const STD_AC_CHROMA_SYMBOLS: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32,
    0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0, 0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16,
    0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64,
    0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86,
    0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8,
    0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9,
    0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
];

/// One of the four example Huffman tables of Annex K.3: `ac` selects the class, `chroma` the
/// destination (0 = luminance, 1 = chrominance).
pub(super) fn std_huffman(ac: bool, chroma: bool) -> HuffSpec {
    let (counts, symbols): ([u8; 16], &[u8]) = match (ac, chroma) {
        (false, false) => (STD_DC_LUMA_COUNTS, &STD_DC_SYMBOLS),
        (false, true) => (STD_DC_CHROMA_COUNTS, &STD_DC_SYMBOLS),
        (true, false) => (STD_AC_LUMA_COUNTS, &STD_AC_LUMA_SYMBOLS),
        (true, true) => (STD_AC_CHROMA_COUNTS, &STD_AC_CHROMA_SYMBOLS),
    };
    HuffSpec { counts, symbols: symbols.to_vec() }
}

/// The quantisation table for an IJG quality of 1 to 100 (clamped), in natural order: the Annex K
/// table scaled the way libjpeg's `jpeg_set_quality` does it (50 = the table as printed, 100 = all
/// ones), each entry clamped to 1..=255 so the table stays legal for baseline.
pub(super) fn scaled_quant(base: &[u8; 64], quality: u8) -> [u16; 64] {
    let quality = u32::from(quality.clamp(1, 100));
    let scale = if quality < 50 { 5000 / quality } else { 200 - quality * 2 };
    let mut out = [0_u16; 64];
    for (o, &b) in out.iter_mut().zip(base) {
        *o = ((u32::from(b) * scale + 50) / 100).clamp(1, 255) as u16;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_is_a_permutation_that_walks_the_antidiagonals() {
        let mut seen = [false; 64];
        let mut previous_diagonal = 0;
        for &n in &ZIGZAG {
            assert!(!seen[n as usize]);
            seen[n as usize] = true;
            let diagonal = n / 8 + n % 8;
            assert!(diagonal == previous_diagonal || diagonal == previous_diagonal + 1);
            previous_diagonal = diagonal;
        }
        assert!(seen.iter().all(|&s| s));
        assert_eq!(&ZIGZAG[..6], &[0, 1, 8, 16, 9, 2]);
        assert_eq!(&ZIGZAG[58..], &[61, 54, 47, 55, 62, 63]);
    }

    #[test]
    fn standard_huffman_tables_hold_exactly_the_baseline_symbols() {
        for chroma in [false, true] {
            let dc = std_huffman(false, chroma);
            assert_eq!(dc.counts.iter().map(|&c| usize::from(c)).sum::<usize>(), 12);
            let ac = std_huffman(true, chroma);
            assert_eq!(ac.counts.iter().map(|&c| usize::from(c)).sum::<usize>(), 162);
            // Every run/size pair with size 1..=10, plus end-of-block and the 16-zero run.
            let mut expected: Vec<u8> = vec![0x00, 0xF0];
            for run in 0..16_u8 {
                for size in 1..=10_u8 {
                    expected.push(run << 4 | size);
                }
            }
            expected.sort_unstable();
            let mut got = ac.symbols.clone();
            got.sort_unstable();
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn quality_scaling_follows_the_ijg_formula() {
        assert_eq!(scaled_quant(&STD_LUMA_QUANT, 50).map(|v| v as u8), STD_LUMA_QUANT);
        assert!(scaled_quant(&STD_LUMA_QUANT, 100).iter().all(|&v| v == 1));
        // Quality 75 halves the table (scale 50), rounding half up.
        assert_eq!(&scaled_quant(&STD_LUMA_QUANT, 75)[..4], &[8, 6, 5, 8]);
        // Quality 1 saturates at 255; quality 0 is treated as 1 and 200 as 100.
        assert!(scaled_quant(&STD_CHROMA_QUANT, 1).iter().all(|&v| v == 255));
        assert_eq!(scaled_quant(&STD_LUMA_QUANT, 0), scaled_quant(&STD_LUMA_QUANT, 1));
        assert_eq!(scaled_quant(&STD_LUMA_QUANT, 200), scaled_quant(&STD_LUMA_QUANT, 100));
        assert_eq!(scaled_quant(&STD_LUMA_QUANT, 25)[0], 32);
    }
}
