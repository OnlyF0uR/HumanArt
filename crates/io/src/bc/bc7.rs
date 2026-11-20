//! The BC7 block decoder: all eight modes, bit-exact with the D3D11 / Vulkan block format
//! (`BC7_UNORM`, `VK_FORMAT_BC7_UNORM_BLOCK`).
//!
//! A block is 128 bits, read least significant bit first from 16 little-endian bytes. The mode is
//! the number of zero bits before the first one bit; a block whose first byte is zero (mode 8) is
//! reserved and decodes to transparent black, as the specification requires.

use super::bits::Bits;
use super::tables;

/// The fixed layout parameters of one BC7 mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModeInfo {
    /// Number of subsets (1 to 3).
    pub subsets: u32,
    /// Bits of the partition number.
    pub partition_bits: u32,
    /// Bits of the channel rotation (modes 4 and 5).
    pub rotation_bits: u32,
    /// Bits of the index-selection flag (mode 4).
    pub index_selection_bits: u32,
    /// Bits per colour endpoint channel, without the p-bit.
    pub color_bits: u32,
    /// Bits per alpha endpoint, without the p-bit; 0 when the mode has no alpha (alpha is 255).
    pub alpha_bits: u32,
    /// One p-bit per endpoint.
    pub endpoint_pbits: bool,
    /// One p-bit per subset, shared by its two endpoints.
    pub shared_pbits: bool,
    /// Bits of the primary indices.
    pub index_bits: u32,
    /// Bits of the secondary indices (modes 4 and 5), 0 otherwise.
    pub index2_bits: u32,
}

/// The eight BC7 modes.
pub const MODES: [ModeInfo; 8] = [
    mode(3, 4, 0, 0, 4, 0, true, false, 3, 0),
    mode(2, 6, 0, 0, 6, 0, false, true, 3, 0),
    mode(3, 6, 0, 0, 5, 0, false, false, 2, 0),
    mode(2, 6, 0, 0, 7, 0, true, false, 2, 0),
    mode(1, 0, 2, 1, 5, 6, false, false, 2, 3),
    mode(1, 0, 2, 0, 7, 8, false, false, 2, 2),
    mode(1, 0, 0, 0, 7, 7, true, false, 4, 0),
    mode(2, 6, 0, 0, 5, 5, true, false, 2, 0),
];

const fn mode(
    subsets: u32,
    partition_bits: u32,
    rotation_bits: u32,
    index_selection_bits: u32,
    color_bits: u32,
    alpha_bits: u32,
    endpoint_pbits: bool,
    shared_pbits: bool,
    index_bits: u32,
    index2_bits: u32,
) -> ModeInfo {
    ModeInfo {
        subsets,
        partition_bits,
        rotation_bits,
        index_selection_bits,
        color_bits,
        alpha_bits,
        endpoint_pbits,
        shared_pbits,
        index_bits,
        index2_bits,
    }
}

/// The mode of a block (0 to 7), or `None` for the reserved mode 8.
#[must_use]
pub const fn block_mode(block: &[u8; 16]) -> Option<u32> {
    if block[0] == 0 { None } else { Some(block[0].trailing_zeros()) }
}

/// Expands an `n`-bit endpoint value (5 to 8 bits, p-bit included) to 8 bits by bit replication.
const fn expand(value: u32, n: u32) -> u32 {
    let v = value << (8 - n);
    v | (v >> n)
}

/// `(a·(64−w) + b·w + 32) >> 6`, the BC7 interpolation.
const fn interpolate(a: u32, b: u32, w: u32) -> u32 {
    (a * (64 - w) + b * w + 32) >> 6
}

/// Decodes one block into 16 RGBA texels (row-major).
#[must_use]
pub fn decode_block(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let Some(mode) = block_mode(block) else {
        return [[0; 4]; 16];
    };
    let info = MODES[mode as usize];
    let mut bits = Bits::new(block);
    bits.skip(mode + 1);
    let partition = bits.read(info.partition_bits) as usize;
    let rotation = bits.read(info.rotation_bits);
    let index_selection = bits.read(info.index_selection_bits);
    let subsets = info.subsets as usize;

    // endpoints[subset][end][channel], raw values without p-bits.
    let mut endpoints = [[[0_u32; 4]; 2]; 3];
    let channels = if info.alpha_bits > 0 { 4 } else { 3 };
    for channel in 0..channels {
        let n = if channel < 3 { info.color_bits } else { info.alpha_bits };
        for subset in endpoints.iter_mut().take(subsets) {
            for end in subset.iter_mut() {
                end[channel] = bits.read(n);
            }
        }
    }
    let mut color_n = info.color_bits;
    let mut alpha_n = info.alpha_bits;
    if info.endpoint_pbits || info.shared_pbits {
        color_n += 1;
        if info.alpha_bits > 0 {
            alpha_n += 1;
        }
        for subset in endpoints.iter_mut().take(subsets) {
            let shared = if info.shared_pbits { bits.read(1) } else { 0 };
            for end in subset.iter_mut() {
                let p = if info.endpoint_pbits { bits.read(1) } else { shared };
                for c in end.iter_mut().take(channels) {
                    *c = (*c << 1) | p;
                }
            }
        }
    }
    for subset in endpoints.iter_mut().take(subsets) {
        for end in subset.iter_mut() {
            for (channel, c) in end.iter_mut().enumerate() {
                *c = if channel < 3 {
                    expand(*c, color_n)
                } else if channels == 4 {
                    expand(*c, alpha_n)
                } else {
                    255
                };
            }
        }
    }

    let mut index = [0_u32; 16];
    for (texel, slot) in index.iter_mut().enumerate() {
        let anchor = tables::is_anchor(info.subsets, partition, texel);
        *slot = bits.read(info.index_bits - u32::from(anchor));
    }
    let mut index2 = [0_u32; 16];
    if info.index2_bits > 0 {
        for (texel, slot) in index2.iter_mut().enumerate() {
            *slot = bits.read(info.index2_bits - u32::from(texel == 0));
        }
    }

    let primary = tables::weights(info.index_bits);
    let secondary = tables::weights(info.index2_bits);
    let mut out = [[0_u8; 4]; 16];
    for (texel, pixel) in out.iter_mut().enumerate() {
        let subset = tables::subset(info.subsets, partition, texel);
        let [e0, e1] = endpoints[subset];
        let (wc, wa) = if info.index2_bits == 0 {
            let w = u32::from(primary[index[texel] as usize]);
            (w, w)
        } else if index_selection == 0 {
            (u32::from(primary[index[texel] as usize]), u32::from(secondary[index2[texel] as usize]))
        } else {
            (u32::from(secondary[index2[texel] as usize]), u32::from(primary[index[texel] as usize]))
        };
        let mut rgba = [0_u32; 4];
        for channel in 0..4 {
            let w = if channel < 3 { wc } else { wa };
            rgba[channel] = interpolate(e0[channel], e1[channel], w);
        }
        match rotation {
            1 => rgba.swap(0, 3),
            2 => rgba.swap(1, 3),
            3 => rgba.swap(2, 3),
            _ => {}
        }
        *pixel = rgba.map(|v| v.min(255) as u8);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::bits::tests::Writer;
    use super::*;

    /// Mode 6 with both endpoints white: every texel is opaque white whatever the indices.
    #[test]
    fn skn_04_mode6_constant_white() {
        let mut w = Writer::default();
        w.put(1 << 6, 7); // mode 6
        for _ in 0..8 {
            w.put(0x7F, 7); // R0 R1 G0 G1 B0 B1 A0 A1
        }
        w.put(1, 1);
        w.put(1, 1); // p-bits
        let block = w.finish();
        assert_eq!(block_mode(&block), Some(6));
        assert_eq!(decode_block(&block), [[255; 4]; 16]);
    }

    /// Mode 6, black to white, indices 0..15 in texel order (the anchor stores 0 with 3 bits): the
    /// texels are the 4-bit weights scaled to 255 with the specification's rounding.
    #[test]
    fn skn_04_mode6_ramp_uses_the_four_bit_weights() {
        let mut w = Writer::default();
        w.put(1 << 6, 7);
        for _ in 0..4 {
            w.put(0, 7);
            w.put(0x7F, 7);
        }
        w.put(0, 1);
        w.put(1, 1);
        w.put(0, 3);
        for i in 1..16 {
            w.put(i, 4);
        }
        let out = decode_block(&w.finish());
        for (i, px) in out.iter().enumerate() {
            let wgt = u32::from(tables::WEIGHTS_4[i]);
            let v = (wgt * 255 + 32) >> 6;
            assert_eq!(*px, [v as u8; 4], "texel {i}");
        }
    }

    /// Mode 1, partition 0 (left two columns subset 0, right two subset 1): red on the left,
    /// blue on the right, all indices zero.
    #[test]
    fn skn_04_mode1_partition0_splits_columns() {
        let mut w = Writer::default();
        w.put(0b10, 2); // mode 1
        w.put(0, 6); // partition 0
        // R: s0e0 s0e1 s1e0 s1e1, then G, then B; 6 bits each.
        for v in [63, 63, 0, 0] {
            w.put(v, 6);
        }
        for _ in 0..4 {
            w.put(0, 6);
        }
        for v in [0, 0, 63, 63] {
            w.put(v, 6);
        }
        w.put(1, 1);
        w.put(1, 1); // shared p-bits
        // 16 indices of 3 bits, anchors (0 and 15) with 2.
        let block = w.finish();
        let out = decode_block(&block);
        for (i, px) in out.iter().enumerate() {
            let expected = if i % 4 < 2 { [255, 0, 0, 255] } else { [0, 0, 255, 255] };
            // 6-bit 0 + p-bit 1 = 0b0000001 expands to 2 for the zero channels.
            let expected = expected.map(|c| if c == 0 { 2 } else { c });
            let expected = [expected[0], expected[1], expected[2], 255];
            assert_eq!(*px, expected, "texel {i}");
        }
    }

    /// Mode 5 with rotation 1 swaps the decoded alpha into red.
    #[test]
    fn skn_04_mode5_rotation_swaps_alpha_and_red() {
        let mut w = Writer::default();
        w.put(1 << 5, 6);
        w.put(1, 2); // rotation 1
        for _ in 0..6 {
            w.put(0, 7); // RGB endpoints black
        }
        w.put(200, 8);
        w.put(200, 8); // alpha endpoints
        let out = decode_block(&w.finish());
        assert_eq!(out[5], [200, 0, 0, 0]);
    }

    /// Mode 4 with index selection 1: colour takes the 3-bit index set, alpha the 2-bit one.
    #[test]
    fn skn_04_mode4_index_selection() {
        let mut w = Writer::default();
        w.put(1 << 4, 5);
        w.put(0, 2); // no rotation
        w.put(1, 1); // index selection
        for _ in 0..3 {
            w.put(0, 5);
            w.put(31, 5);
        }
        w.put(0, 6);
        w.put(63, 6);
        // primary 2-bit indices (texel 0 anchor: 1 bit), all 3 -> anchor gets 1.
        w.put(1, 1);
        for _ in 1..16 {
            w.put(3, 2);
        }
        // secondary 3-bit indices: texel 0 anchor 2 bits = 0, others 7.
        w.put(0, 2);
        for _ in 1..16 {
            w.put(7, 3);
        }
        let out = decode_block(&w.finish());
        // Texel 0: colour index 0 (3-bit set) -> 0; alpha 2-bit index 1 -> weight 21.
        let a0 = ((255 * 21 + 32) >> 6) as u8;
        assert_eq!(out[0], [0, 0, 0, a0]);
        assert_eq!(out[1], [255, 255, 255, 255]);
    }

    #[test]
    fn skn_04_reserved_mode_is_transparent_black() {
        let mut block = [0xFF_u8; 16];
        block[0] = 0;
        assert_eq!(block_mode(&block), None);
        assert_eq!(decode_block(&block), [[0; 4]; 16]);
    }
}
