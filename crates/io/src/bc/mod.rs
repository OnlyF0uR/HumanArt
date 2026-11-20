//! BC7 and BC6H block compression: the CPU reference decoders and the shared tables `[SKN-04]`.
//!
//! The runtime package stores colour and normal maps as BC7 and HDR or signed data as BC6H. The
//! encoders run on the GPU (`humanart-render::compress`); this module is their reference: the
//! decoders here are bit-exact with the D3D11 / Vulkan block formats (verified against hand-built
//! blocks from the specification and against the GPU's own hardware decoder on random blocks), and
//! the partition, anchor and weight tables in [`tables`] are the ones the encoders are given.
//!
//! - [`bc7`]: `decode_block` for all eight modes; the reserved mode decodes to transparent black.
//! - [`bc6h`]: `decode_block` for all fourteen modes, unsigned and signed; reserved codes give 0.
//! - [`decode`]: a whole block stream to an [`Image`], with [`Limits`] checked before allocation.
//! - [`psnr`] and [`psnr_f16`]: the quality measures the encoder tests and benchmarks report.
//!
//! Block streams are rows of 4×4 blocks, 16 bytes each, top to bottom; images whose size is not a
//! multiple of 4 are padded to whole blocks and the padding is cropped on decode.

pub mod bc6h;
pub mod bc7;
mod bits;
pub mod tables;

use crate::error::{Error, Format, Result};
use crate::image::{ColorSpace, Image, Limits, Samples};

/// Bytes per 4×4 block.
pub const BLOCK_BYTES: usize = 16;

/// Which block format a stream holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BcFormat {
    /// BC7: RGBA, 8 bits per channel after decoding (linear or sRGB, by the image's colour space).
    Bc7,
    /// BC6H unsigned float (`BC6H_UF16`): RGB half floats, non-negative.
    Bc6hUfloat,
    /// BC6H signed float (`BC6H_SF16`): RGB half floats.
    Bc6hSfloat,
}

/// The number of blocks across and down for an image of `width × height` texels.
#[must_use]
pub const fn block_dims(width: u32, height: u32) -> (u32, u32) {
    (width.div_ceil(4), height.div_ceil(4))
}

/// The byte size of the block stream of a `width × height` image, `None` on overflow.
#[must_use]
pub fn stream_size(width: u32, height: u32) -> Option<usize> {
    let (bx, by) = block_dims(width, height);
    (bx as usize).checked_mul(by as usize)?.checked_mul(BLOCK_BYTES)
}

/// Decodes a block stream of a `width × height` image.
///
/// BC7 gives 4-channel 8-bit samples labelled `color_space`; BC6H gives 3-channel half floats
/// (`Samples::F16`) labelled `color_space`. The stream must be exactly [`stream_size`] bytes.
pub fn decode(
    format: BcFormat,
    data: &[u8],
    width: u32,
    height: u32,
    color_space: ColorSpace,
    limits: &Limits,
) -> Result<Image> {
    let pixels = limits.check_image(Format::Bc, u64::from(width), u64::from(height))?;
    let expected = stream_size(width, height).ok_or(Error::Limit {
        format: Format::Bc,
        what: "block stream size",
        limit: usize::MAX as u64,
        got: u64::MAX,
    })?;
    if data.len() < expected {
        return Err(Error::Truncated { format: Format::Bc, offset: data.len() });
    }
    if data.len() > expected {
        return Err(Error::Invalid { format: Format::Bc, offset: expected, what: "bytes after the last block" });
    }
    let (bx, _) = block_dims(width, height);
    let (w, h) = (width as usize, height as usize);
    match format {
        BcFormat::Bc7 => {
            limits.check_bytes(Format::Bc, "decoded image", pixels as u64 * 4)?;
            let mut out = vec![0_u8; pixels * 4];
            for (b, block) in data.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
                let texels = bc7::decode_block(block);
                let (x0, y0) = ((b % bx as usize) * 4, (b / bx as usize) * 4);
                for (t, px) in texels.iter().enumerate() {
                    let (x, y) = (x0 + t % 4, y0 + t / 4);
                    if x < w && y < h {
                        let at = (y * w + x) * 4;
                        out[at..at + 4].copy_from_slice(px);
                    }
                }
            }
            Image::new(Format::Bc, width, height, 4, color_space, Samples::U8(out))
        }
        BcFormat::Bc6hUfloat | BcFormat::Bc6hSfloat => {
            let signed = format == BcFormat::Bc6hSfloat;
            limits.check_bytes(Format::Bc, "decoded image", pixels as u64 * 6)?;
            let mut out = vec![0_u16; pixels * 3];
            for (b, block) in data.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
                let texels = bc6h::decode_block(block, signed);
                let (x0, y0) = ((b % bx as usize) * 4, (b / bx as usize) * 4);
                for (t, px) in texels.iter().enumerate() {
                    let (x, y) = (x0 + t % 4, y0 + t / 4);
                    if x < w && y < h {
                        let at = (y * w + x) * 3;
                        out[at..at + 3].copy_from_slice(px);
                    }
                }
            }
            Image::new(Format::Bc, width, height, 3, color_space, Samples::F16(out))
        }
    }
}

/// Peak signal-to-noise ratio in dB between two equally long 8-bit sample arrays, over the channels
/// selected by `mask` (bit `c` = channel `c`) of `channels`-channel pixels. Identical inputs give
/// `f64::INFINITY`; mismatched lengths or no selected channel give `None`.
#[must_use]
pub fn psnr(a: &[u8], b: &[u8], channels: usize, mask: u32) -> Option<f64> {
    if a.len() != b.len() || channels == 0 || channels > 4 || mask & ((1 << channels) - 1) == 0 {
        return None;
    }
    let mut sum = 0.0_f64;
    let mut n = 0_u64;
    for (pa, pb) in a.chunks_exact(channels).zip(b.chunks_exact(channels)) {
        for c in 0..channels {
            if mask & (1 << c) != 0 {
                let d = f64::from(pa[c]) - f64::from(pb[c]);
                sum += d * d;
                n += 1;
            }
        }
    }
    if n == 0 {
        return None;
    }
    let mse = sum / n as f64;
    Some(if mse == 0.0 { f64::INFINITY } else { 10.0 * (255.0 * 255.0 / mse).log10() })
}

/// PSNR in dB of HDR data, measured as the industry does for BC6H: both images are mapped by
/// `log2(1 + x)` per channel (negative values by `-log2(1 − x)`), and the peak is the largest mapped
/// magnitude of the reference `a`. Inputs are equally long `f32` arrays.
#[must_use]
pub fn psnr_f16(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let map = |v: f32| -> f64 {
        let v = f64::from(v);
        if v >= 0.0 { (1.0 + v).log2() } else { -(1.0 - v).log2() }
    };
    let mut sum = 0.0_f64;
    let mut peak = 0.0_f64;
    for (&x, &y) in a.iter().zip(b) {
        let (mx, my) = (map(x), map(y));
        peak = peak.max(mx.abs());
        sum += (mx - my) * (mx - my);
    }
    let mse = sum / a.len() as f64;
    if peak == 0.0 {
        return None;
    }
    Some(if mse == 0.0 { f64::INFINITY } else { 10.0 * (peak * peak / mse).log10() })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use humanart_core::rng::Rng;

    use super::*;

    #[test]
    fn decode_crops_partial_blocks_and_checks_sizes() {
        // A 5x3 image needs 2x1 blocks.
        assert_eq!(block_dims(5, 3), (2, 1));
        assert_eq!(stream_size(5, 3), Some(32));
        let mut data = vec![0_u8; 32];
        data[0] = 1 << 6; // mode 6, all-zero endpoints: black with alpha 0
        data[16] = 1 << 6;
        let image = decode(BcFormat::Bc7, &data, 5, 3, ColorSpace::Srgb, &Limits::DEFAULT).expect("decodes");
        assert_eq!((image.width, image.height, image.channels), (5, 3, 4));
        assert_eq!(image.as_u8().map(<[u8]>::len), Some(5 * 3 * 4));
        assert!(matches!(
            decode(BcFormat::Bc7, &data[..31], 5, 3, ColorSpace::Srgb, &Limits::DEFAULT),
            Err(Error::Truncated { .. })
        ));
        data.push(0);
        assert!(matches!(
            decode(BcFormat::Bc7, &data, 5, 3, ColorSpace::Srgb, &Limits::DEFAULT),
            Err(Error::Invalid { .. })
        ));
        assert!(decode(BcFormat::Bc6hUfloat, &[], 0, 4, ColorSpace::Linear, &Limits::DEFAULT).is_err());
        let huge = Limits { max_dimension: 8, ..Limits::DEFAULT };
        assert!(matches!(
            decode(BcFormat::Bc7, &[0; 16 * 9], 12, 12, ColorSpace::Srgb, &huge),
            Err(Error::Limit { .. })
        ));
    }

    #[test]
    fn psnr_measures() {
        assert_eq!(psnr(&[1, 2, 3, 4], &[1, 2, 3, 4], 4, 0xF), Some(f64::INFINITY));
        let p = psnr(&[0, 0, 0, 0], &[1, 1, 1, 1], 4, 0xF).expect("valid");
        assert!((p - 48.13).abs() < 0.01, "{p}");
        // Only alpha differs and alpha is masked out.
        assert_eq!(psnr(&[0, 0, 0, 0], &[0, 0, 0, 9], 4, 0x7), Some(f64::INFINITY));
        assert_eq!(psnr(&[0], &[0, 0], 1, 1), None);
        assert!(psnr_f16(&[1.0, 2.0], &[1.0, 2.1]).expect("valid") > 30.0);
    }

    #[test]
    fn rob_01_fuzz_bc_decoders() {
        let mut rng = Rng::new(0xBC67);
        let mut corpus = Vec::new();
        for _ in 0..8 {
            let (w, h) = (1 + rng.below(12) as u32, 1 + rng.below(12) as u32);
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&w.to_le_bytes());
            bytes.extend_from_slice(&h.to_le_bytes());
            let mut blocks = vec![0_u8; stream_size(w, h).unwrap_or(0)];
            rng.fill(&mut blocks);
            bytes.extend_from_slice(&blocks);
            corpus.push(bytes);
        }
        humanart_core::fuzz::run("bc", &corpus, 600, 0xBC7B_C6E1, Duration::from_secs(120), |bytes| {
            let word = |i: usize| bytes.get(i..i + 4).map_or(0, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]));
            let (w, h) = (word(0), word(4));
            let data = bytes.get(8..).unwrap_or(&[]);
            for format in [BcFormat::Bc7, BcFormat::Bc6hUfloat, BcFormat::Bc6hSfloat] {
                let _ = decode(format, data, w, h, ColorSpace::Linear, &Limits::FUZZ);
            }
        });
    }
}
