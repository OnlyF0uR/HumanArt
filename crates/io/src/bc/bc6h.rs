//! The BC6H block decoder: all fourteen modes, unsigned (`BC6H_UF16`) and signed (`BC6H_SF16`),
//! bit-exact with the D3D11 / Vulkan block format.
//!
//! A block is 128 bits, least significant bit first. The first two bits select modes 0 and 1;
//! otherwise five bits select the mode, and the four reserved five-bit codes decode to zero, as the
//! specification requires. The endpoint bits of each mode are scattered over the block; [`MODES`]
//! lists, for every stored bit in order, which endpoint component and which of its bits it is.
//! Output texels are IEEE half floats (as bit patterns), RGB.

use super::bits::Bits;
use super::tables;

/// Field numbers of the scattered layout: `R0 G0 B0` is endpoint 0 (the base), `R1..B1` endpoint
/// 1, `R2..B3` the endpoints of region 1, `D` the partition.
const R0: u8 = 0;
const G0: u8 = 1;
const B0: u8 = 2;
const R1: u8 = 3;
const G1: u8 = 4;
const B1: u8 = 5;
const R2: u8 = 6;
const G2: u8 = 7;
const B2: u8 = 8;
const R3: u8 = 9;
const G3: u8 = 10;
const B3: u8 = 11;
const D: u8 = 12;

/// Number of layout fields (12 endpoint components and the partition).
pub const FIELDS: usize = 13;

/// The most stored bits of one layout (two-region modes with the 2-bit mode code: 80).
pub const MAX_LAYOUT_BITS: usize = 80;

/// A run of consecutive bits of one field: `(field, first bit, count, descending)`.
#[derive(Clone, Copy)]
struct Run(u8, u8, u8, bool);

const fn up(field: u8, lo: u8, count: u8) -> Run {
    Run(field, lo, count, false)
}
const fn one(field: u8, bit: u8) -> Run {
    Run(field, bit, 1, false)
}
const fn down(field: u8, hi: u8, count: u8) -> Run {
    Run(field, hi, count, true)
}

/// Expands runs into one byte per stored bit: `field << 4 | bit`.
const fn layout(runs: &[Run]) -> ([u8; MAX_LAYOUT_BITS], u8) {
    let mut out = [0xFF_u8; MAX_LAYOUT_BITS];
    let mut n = 0;
    let mut r = 0;
    while r < runs.len() {
        let Run(field, start, count, descending) = runs[r];
        let mut i = 0;
        while i < count {
            let bit = if descending { start - i } else { start + i };
            if n < MAX_LAYOUT_BITS {
                out[n] = (field << 4) | bit;
            }
            n += 1;
            i += 1;
        }
        r += 1;
    }
    (out, n as u8)
}

/// One BC6H mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModeInfo {
    /// The mode code as stored in the low bits of the block.
    pub code: u8,
    /// How many bits the code takes (2 or 5).
    pub code_bits: u8,
    /// Number of regions (1 or 2).
    pub regions: u8,
    /// Whether endpoints after the first are stored as signed deltas from it.
    pub transformed: bool,
    /// Bits of the base endpoint (and of every endpoint after the transform).
    pub endpoint_bits: u8,
    /// Bits of the stored deltas (or of the other endpoints when not transformed), per channel.
    pub delta_bits: [u8; 3],
    /// The stored field bits after the code, in block order: `field << 4 | bit`.
    pub layout: [u8; MAX_LAYOUT_BITS],
    /// How many entries of `layout` are used.
    pub layout_len: u8,
}

const fn mode(
    code: u8,
    code_bits: u8,
    regions: u8,
    transformed: bool,
    endpoint_bits: u8,
    delta_bits: [u8; 3],
    runs: &[Run],
) -> ModeInfo {
    let (layout, layout_len) = layout(runs);
    ModeInfo { code, code_bits, regions, transformed, endpoint_bits, delta_bits, layout, layout_len }
}

/// The fourteen modes, in the order of the D3D11 specification (its modes 1 to 14).
#[rustfmt::skip]
pub const MODES: [ModeInfo; 14] = [
    mode(0b00, 2, 2, true, 10, [5, 5, 5], &[
        one(G2, 4), one(B2, 4), one(B3, 4), up(R0, 0, 10), up(G0, 0, 10), up(B0, 0, 10), up(R1, 0, 5), one(G3, 4),
        up(G2, 0, 4), up(G1, 0, 5), one(B3, 0), up(G3, 0, 4), up(B1, 0, 5), one(B3, 1), up(B2, 0, 4), up(R2, 0, 5),
        one(B3, 2), up(R3, 0, 5), one(B3, 3), up(D, 0, 5),
    ]),
    mode(0b01, 2, 2, true, 7, [6, 6, 6], &[
        one(G2, 5), one(G3, 4), one(G3, 5), up(R0, 0, 7), one(B3, 0), one(B3, 1), one(B2, 4), up(G0, 0, 7),
        one(B2, 5), one(B3, 2), one(G2, 4), up(B0, 0, 7), one(B3, 3), one(B3, 5), one(B3, 4), up(R1, 0, 6),
        up(G2, 0, 4), up(G1, 0, 6), up(G3, 0, 4), up(B1, 0, 6), up(B2, 0, 4), up(R2, 0, 6), up(R3, 0, 6), up(D, 0, 5),
    ]),
    mode(0b00010, 5, 2, true, 11, [5, 4, 4], &[
        up(R0, 0, 10), up(G0, 0, 10), up(B0, 0, 10), up(R1, 0, 5), one(R0, 10), up(G2, 0, 4), up(G1, 0, 4),
        one(G0, 10), one(B3, 0), up(G3, 0, 4), up(B1, 0, 4), one(B0, 10), one(B3, 1), up(B2, 0, 4), up(R2, 0, 5),
        one(B3, 2), up(R3, 0, 5), one(B3, 3), up(D, 0, 5),
    ]),
    mode(0b00110, 5, 2, true, 11, [4, 5, 4], &[
        up(R0, 0, 10), up(G0, 0, 10), up(B0, 0, 10), up(R1, 0, 4), one(R0, 10), one(G3, 4), up(G2, 0, 4),
        up(G1, 0, 5), one(G0, 10), up(G3, 0, 4), up(B1, 0, 4), one(B0, 10), one(B3, 1), up(B2, 0, 4), up(R2, 0, 4),
        one(B3, 0), one(B3, 2), up(R3, 0, 4), one(G2, 4), one(B3, 3), up(D, 0, 5),
    ]),
    mode(0b01010, 5, 2, true, 11, [4, 4, 5], &[
        up(R0, 0, 10), up(G0, 0, 10), up(B0, 0, 10), up(R1, 0, 4), one(R0, 10), one(B2, 4), up(G2, 0, 4),
        up(G1, 0, 4), one(G0, 10), one(B3, 0), up(G3, 0, 4), up(B1, 0, 5), one(B0, 10), up(B2, 0, 4), up(R2, 0, 4),
        one(B3, 1), one(B3, 2), up(R3, 0, 4), one(B3, 4), one(B3, 3), up(D, 0, 5),
    ]),
    mode(0b01110, 5, 2, true, 9, [5, 5, 5], &[
        up(R0, 0, 9), one(B2, 4), up(G0, 0, 9), one(G2, 4), up(B0, 0, 9), one(B3, 4), up(R1, 0, 5), one(G3, 4),
        up(G2, 0, 4), up(G1, 0, 5), one(B3, 0), up(G3, 0, 4), up(B1, 0, 5), one(B3, 1), up(B2, 0, 4), up(R2, 0, 5),
        one(B3, 2), up(R3, 0, 5), one(B3, 3), up(D, 0, 5),
    ]),
    mode(0b10010, 5, 2, true, 8, [6, 5, 5], &[
        up(R0, 0, 8), one(G3, 4), one(B2, 4), up(G0, 0, 8), one(B3, 2), one(G2, 4), up(B0, 0, 8), one(B3, 3),
        one(B3, 4), up(R1, 0, 6), up(G2, 0, 4), up(G1, 0, 5), one(B3, 0), up(G3, 0, 4), up(B1, 0, 5), one(B3, 1),
        up(B2, 0, 4), up(R2, 0, 6), up(R3, 0, 6), up(D, 0, 5),
    ]),
    mode(0b10110, 5, 2, true, 8, [5, 6, 5], &[
        up(R0, 0, 8), one(B3, 0), one(B2, 4), up(G0, 0, 8), one(G2, 5), one(G2, 4), up(B0, 0, 8), one(G3, 5),
        one(B3, 4), up(R1, 0, 5), one(G3, 4), up(G2, 0, 4), up(G1, 0, 6), up(G3, 0, 4), up(B1, 0, 5), one(B3, 1),
        up(B2, 0, 4), up(R2, 0, 5), one(B3, 2), up(R3, 0, 5), one(B3, 3), up(D, 0, 5),
    ]),
    mode(0b11010, 5, 2, true, 8, [5, 5, 6], &[
        up(R0, 0, 8), one(B3, 1), one(B2, 4), up(G0, 0, 8), one(B2, 5), one(G2, 4), up(B0, 0, 8), one(B3, 5),
        one(B3, 4), up(R1, 0, 5), one(G3, 4), up(G2, 0, 4), up(G1, 0, 5), one(B3, 0), up(G3, 0, 4), up(B1, 0, 6),
        up(B2, 0, 4), up(R2, 0, 5), one(B3, 2), up(R3, 0, 5), one(B3, 3), up(D, 0, 5),
    ]),
    mode(0b11110, 5, 2, false, 6, [6, 6, 6], &[
        up(R0, 0, 6), one(G3, 4), one(B3, 0), one(B3, 1), one(B2, 4), up(G0, 0, 6), one(G2, 5), one(B2, 5),
        one(B3, 2), one(G2, 4), up(B0, 0, 6), one(G3, 5), one(B3, 3), one(B3, 5), one(B3, 4), up(R1, 0, 6),
        up(G2, 0, 4), up(G1, 0, 6), up(G3, 0, 4), up(B1, 0, 6), up(B2, 0, 4), up(R2, 0, 6), up(R3, 0, 6), up(D, 0, 5),
    ]),
    mode(0b00011, 5, 1, false, 10, [10, 10, 10], &[
        up(R0, 0, 10), up(G0, 0, 10), up(B0, 0, 10), up(R1, 0, 10), up(G1, 0, 10), up(B1, 0, 10),
    ]),
    mode(0b00111, 5, 1, true, 11, [9, 9, 9], &[
        up(R0, 0, 10), up(G0, 0, 10), up(B0, 0, 10), up(R1, 0, 9), one(R0, 10), up(G1, 0, 9), one(G0, 10),
        up(B1, 0, 9), one(B0, 10),
    ]),
    mode(0b01011, 5, 1, true, 12, [8, 8, 8], &[
        up(R0, 0, 10), up(G0, 0, 10), up(B0, 0, 10), up(R1, 0, 8), down(R0, 11, 2), up(G1, 0, 8), down(G0, 11, 2),
        up(B1, 0, 8), down(B0, 11, 2),
    ]),
    mode(0b01111, 5, 1, true, 16, [4, 4, 4], &[
        up(R0, 0, 10), up(G0, 0, 10), up(B0, 0, 10), up(R1, 0, 4), down(R0, 15, 6), up(G1, 0, 4), down(G0, 15, 6),
        up(B1, 0, 4), down(B0, 15, 6),
    ]),
];

/// `u32` words per mode in [`gpu_words`].
pub const GPU_WORDS_PER_MODE: usize = 24;

/// The mode table as the GPU encoder reads it from a storage buffer, [`GPU_WORDS_PER_MODE`] words
/// per mode in [`MODES`] order: word 0 `code | code_bits << 8 | regions << 16 | transformed << 24`,
/// word 1 `endpoint_bits | delta_bits[0] << 8 | delta_bits[1] << 16 | delta_bits[2] << 24`, word 2
/// the layout length, word 3 zero, words 4..24 the layout bytes, four per word, first byte lowest.
#[must_use]
pub fn gpu_words() -> [u32; 14 * GPU_WORDS_PER_MODE] {
    let mut words = [0_u32; 14 * GPU_WORDS_PER_MODE];
    for (m, info) in MODES.iter().enumerate() {
        let w = &mut words[m * GPU_WORDS_PER_MODE..(m + 1) * GPU_WORDS_PER_MODE];
        w[0] = u32::from(info.code)
            | (u32::from(info.code_bits) << 8)
            | (u32::from(info.regions) << 16)
            | (u32::from(info.transformed) << 24);
        w[1] = u32::from(info.endpoint_bits)
            | (u32::from(info.delta_bits[0]) << 8)
            | (u32::from(info.delta_bits[1]) << 16)
            | (u32::from(info.delta_bits[2]) << 24);
        w[2] = u32::from(info.layout_len);
        for (b, &entry) in info.layout.iter().enumerate() {
            w[4 + b / 4] |= u32::from(entry) << (8 * (b % 4));
        }
    }
    words
}

/// The mode index (0 to 13, into [`MODES`]) of a block, or `None` for a reserved code.
#[must_use]
pub fn block_mode(block: &[u8; 16]) -> Option<usize> {
    let low = block[0] & 0b11;
    if low < 2 {
        return Some(low as usize);
    }
    let code = block[0] & 0b1_1111;
    MODES.iter().position(|m| m.code_bits == 5 && m.code == code)
}

/// Sign-extends the low `bits` bits of `v`.
const fn sign_extend(v: i32, bits: u32) -> i32 {
    if bits == 0 || bits >= 32 {
        return v;
    }
    let shift = 32 - bits;
    (v << shift) >> shift
}

/// Expands an endpoint component of `bits` bits to the 16-bit (unsigned) or 15-bit plus sign
/// (signed) interpolation domain.
const fn unquantize(comp: i32, bits: u32, signed: bool) -> i32 {
    if signed {
        if bits >= 16 {
            return comp;
        }
        let negative = comp < 0;
        let c = if negative { -comp } else { comp };
        let u = if c == 0 {
            0
        } else if c >= (1 << (bits - 1)) - 1 {
            0x7FFF
        } else {
            ((c << 15) + 0x4000) >> (bits - 1)
        };
        if negative { -u } else { u }
    } else if bits >= 15 {
        comp
    } else if comp == 0 {
        0
    } else if comp == (1 << bits) - 1 {
        0xFFFF
    } else {
        ((comp << 16) + 0x8000) >> bits
    }
}

/// Scales an interpolated value to a half float bit pattern. Signed values are scaled first and
/// take their sign afterwards, so a negative value that scales to zero is `+0` (as the D3D
/// reference decoder and the hardware produce).
const fn finish_unquantize(comp: i32, signed: bool) -> u16 {
    if signed {
        let scaled = if comp < 0 { -(((-comp) * 31) >> 5) } else { (comp * 31) >> 5 };
        if scaled < 0 { (-scaled) as u16 | 0x8000 } else { scaled as u16 }
    } else {
        ((comp * 31) >> 6) as u16
    }
}

/// Decodes one block into 16 RGB texels of half-float bit patterns (row-major). `signed` selects
/// `BC6H_SF16`.
#[must_use]
pub fn decode_block(block: &[u8; 16], signed: bool) -> [[u16; 3]; 16] {
    let Some(index) = block_mode(block) else {
        return [[0; 3]; 16];
    };
    let info = &MODES[index];
    let mut bits = Bits::new(block);
    bits.skip(u32::from(info.code_bits));
    let mut fields = [0_i32; FIELDS];
    for &entry in info.layout.iter().take(info.layout_len as usize) {
        let field = (entry >> 4) as usize;
        if let Some(slot) = fields.get_mut(field) {
            *slot |= (bits.read(1) as i32) << (entry & 15);
        }
    }
    let partition = fields[D as usize] as usize;
    let ends = 2 * info.regions as usize;
    let epb = u32::from(info.endpoint_bits);
    // endpoints[i][channel]: i = 0, 1 region 0; 2, 3 region 1.
    let mut e = [[0_i32; 3]; 4];
    for (i, end) in e.iter_mut().enumerate().take(ends) {
        for (c, v) in end.iter_mut().enumerate() {
            *v = fields[3 * i + c];
        }
    }
    let mask = (1_i32 << epb) - 1;
    if signed {
        for v in &mut e[0] {
            *v = sign_extend(*v, epb);
        }
    }
    let base = e[0];
    for end in e.iter_mut().take(ends).skip(1) {
        for c in 0..3 {
            if info.transformed {
                let delta = sign_extend(end[c], u32::from(info.delta_bits[c]));
                end[c] = (base[c] + delta) & mask;
                if signed {
                    end[c] = sign_extend(end[c], epb);
                }
            } else if signed {
                end[c] = sign_extend(end[c], epb);
            }
        }
    }
    for end in e.iter_mut().take(ends) {
        for v in end.iter_mut() {
            *v = unquantize(*v, epb, signed);
        }
    }

    let (index_bits, weights) =
        if info.regions == 2 { (3, &tables::WEIGHTS_3[..]) } else { (4, &tables::WEIGHTS_4[..]) };
    let regions = u32::from(info.regions);
    let mut out = [[0_u16; 3]; 16];
    for (texel, px) in out.iter_mut().enumerate() {
        let anchor = tables::is_anchor(regions, partition, texel);
        let idx = bits.read(index_bits - u32::from(anchor)) as usize;
        let w = i32::from(weights[idx]);
        let region = tables::subset(regions, partition, texel);
        let (a, b) = (e[2 * region], e[2 * region + 1]);
        for c in 0..3 {
            let v = (a[c] * (64 - w) + b[c] * w + 32) >> 6;
            px[c] = finish_unquantize(v, signed);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::bits::tests::Writer;
    use super::*;

    #[test]
    fn every_layout_stores_each_endpoint_bit_once() {
        for (m, info) in MODES.iter().enumerate() {
            let expected = if info.regions == 2 { 82 } else { 65 } - usize::from(info.code_bits);
            assert_eq!(info.layout_len as usize, expected, "mode {m} bit count");
            let mut seen = [0_u32; FIELDS];
            for &entry in &info.layout[..info.layout_len as usize] {
                let (f, b) = ((entry >> 4) as usize, entry & 15);
                assert_eq!(seen[f] & (1 << b), 0, "mode {m}: field {f} bit {b} twice");
                seen[f] |= 1 << b;
            }
            let ends = 2 * info.regions as usize;
            for i in 0..ends {
                for c in 0..3 {
                    let width = if i == 0 { info.endpoint_bits } else { info.delta_bits[c] };
                    assert_eq!(seen[3 * i + c], (1_u32 << width) - 1, "mode {m}: endpoint {i} channel {c}");
                }
            }
            let d = if info.regions == 2 { 0b11111 } else { 0 };
            assert_eq!(seen[D as usize], d, "mode {m}: partition bits");
        }
    }

    #[test]
    fn gpu_words_hold_the_layouts() {
        let words = gpu_words();
        let m13 = &words[13 * GPU_WORDS_PER_MODE..];
        assert_eq!(m13[0], 0b01111 | (5 << 8) | (1 << 16) | (1 << 24));
        assert_eq!(m13[1], 16 | (4 << 8) | (4 << 16) | (4 << 24));
        assert_eq!(m13[2], 60);
        // Layout byte 34 of mode 13 is the first reversed bit: R0 bit 15.
        assert_eq!((m13[4 + 34 / 4] >> (8 * (34 % 4))) & 0xFF, 15);
    }

    #[test]
    fn mode_codes_are_unique_and_reserved_codes_decode_to_zero() {
        for (m, info) in MODES.iter().enumerate() {
            let block = [info.code, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
            assert_eq!(block_mode(&block), Some(m));
        }
        for code in [0b10011_u8, 0b10111, 0b11011, 0b11111] {
            let block =
                [code, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
            assert_eq!(block_mode(&block), None);
            assert_eq!(decode_block(&block, false), [[0; 3]; 16]);
            assert_eq!(decode_block(&block, true), [[0; 3]; 16]);
        }
    }

    /// Mode 10 (code 00011, 10-bit endpoints): endpoint 0 at the maximum 1023 unquantises to
    /// 0xFFFF and finishes to 0x7BFF (65504, the largest half); endpoint 1 at zero gives 0.
    #[test]
    fn skn_04_mode10_extremes() {
        let mut w = Writer::default();
        w.put(0b00011, 5);
        for _ in 0..3 {
            w.put(1023, 10);
        }
        for _ in 0..3 {
            w.put(0, 10);
        }
        // Indices: texel 0 (anchor, 3 bits) = 0, texel 1 = 15, the rest 0.
        w.put(0, 3);
        w.put(15, 4);
        assert_eq!(w.pos(), 72);
        let out = decode_block(&w.finish(), false);
        assert_eq!(out[0], [0x7BFF; 3]);
        assert_eq!(out[1], [0; 3]);
        assert_eq!(out[2], [0x7BFF; 3]);
    }

    /// Mode 13 (16-bit base, 4-bit deltas): the base is stored with its top six bits reversed. A
    /// base of ceil(0x3C00·64/31) is the half 1.0 exactly after the finishing scale.
    #[test]
    fn skn_04_mode13_reversed_high_bits() {
        // finish: (v * 31) >> 6 == 0x3C00 -> v = ceil(0x3C00 * 64 / 31) = 31711.
        let v: u32 = 31711;
        assert_eq!((v * 31) >> 6, 0x3C00);
        let mut w = Writer::default();
        w.put(0b01111, 5);
        for _ in 0..3 {
            w.put(v & 0x3FF, 10);
        }
        for _ in 0..3 {
            w.put(0, 4); // delta 0
            for bit in (10..16).rev() {
                w.put((v >> bit) & 1, 1);
            }
        }
        assert_eq!(w.pos(), 65);
        let out = decode_block(&w.finish(), false);
        assert_eq!(out[7], [0x3C00; 3]);
    }

    /// Signed mode 10: endpoint 0 = -1 (all ones, 10 bits) unquantises to
    /// -((1 << 15) + 0x4000) >> 9 = -96 and finishes to -(96·31 >> 5) = -93 as sign-magnitude bits.
    #[test]
    fn skn_04_signed_negative_endpoint() {
        let mut w = Writer::default();
        w.put(0b00011, 5);
        for _ in 0..3 {
            w.put(1023, 10);
        }
        for _ in 0..3 {
            w.put(1023, 10);
        }
        let out = decode_block(&w.finish(), true);
        let u = unquantize(-1, 10, true);
        assert_eq!(u, -96);
        assert_eq!(out[3], [0x8000 | 0x5D; 3]);
    }

    /// Mode 0 (two regions): partition 13 (0xFF00: the bottom half is region 1). Region 0 has the
    /// base endpoints, region 1 endpoints are base + delta with the deltas sign-extended.
    #[test]
    fn skn_04_mode0_two_regions_deltas() {
        let info = &MODES[0];
        let mut fields = [0_u32; FIELDS];
        // base 512 on every channel, endpoint 1 delta 0, region 1 endpoints delta -16 (5 bits: 0b10000).
        fields[R0 as usize] = 512;
        fields[G0 as usize] = 512;
        fields[B0 as usize] = 512;
        for f in [R2, G2, B2, R3, G3, B3] {
            fields[f as usize] = 0b10000;
        }
        fields[D as usize] = 13;
        let mut w = Writer::default();
        w.put(0b00, 2);
        for &entry in &info.layout[..info.layout_len as usize] {
            w.put((fields[(entry >> 4) as usize] >> (entry & 15)) & 1, 1);
        }
        assert_eq!(w.pos(), 82);
        let out = decode_block(&w.finish(), false);
        let top = finish_unquantize(unquantize(512, 10, false), false);
        let bottom = finish_unquantize(unquantize(496, 10, false), false);
        assert_eq!(out[0], [top; 3]);
        assert_eq!(out[15], [bottom; 3]);
    }
}
