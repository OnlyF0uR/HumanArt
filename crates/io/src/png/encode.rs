//! The PNG encoder: 8- and 16-bit grey, grey + alpha, RGB and RGBA, adaptive filtering.

use super::filter::{self, AVERAGE, NONE, PAETH, SUB, UP};
use super::{COLOR_SPACE_KEYWORD, EncodeOptions, FORMAT, FilterChoice, SIGNATURE, color_space_name};
use crate::deflate::zlib_compress;
use crate::error::{Error, Result};
use crate::image::{ColorSpace, Image, Samples};
use humanart_core::hash::{crc32, crc32_update};

/// Largest IDAT payload written (decoders handle any size; this keeps chunks a few per image).
const IDAT_MAX: usize = 1 << 20;

fn encode_error(what: &'static str) -> Error {
    Error::Encode { format: FORMAT, what }
}

/// Appends one chunk with its length and CRC.
fn write_chunk(out: &mut Vec<u8>, kind: [u8; 4], data: &[u8]) {
    // The caller keeps every chunk under 2^31 bytes (IDAT is split at `IDAT_MAX`).
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(&kind);
    out.extend_from_slice(data);
    out.extend_from_slice(&crc32_update(crc32(&kind), data).to_be_bytes());
}

/// Encodes `image` with the default options (adaptive filtering, [`crate::deflate::Level::Default`]).
pub fn encode(image: &Image) -> Result<Vec<u8>> {
    encode_with(image, &EncodeOptions::default())
}

/// Encodes `image`, which must hold 1 to 4 channels of [`Samples::U8`] or [`Samples::U16`].
pub fn encode_with(image: &Image, options: &EncodeOptions) -> Result<Vec<u8>> {
    let (width, height, channels) = (image.width, image.height, usize::from(image.channels));
    if width == 0 || height == 0 || width > (1 << 31) - 1 || height > (1 << 31) - 1 {
        return Err(encode_error("image width or height is 0 or over 2^31 - 1"));
    }
    if !(1..=4).contains(&channels) {
        return Err(encode_error("PNG holds 1 to 4 channels"));
    }
    let (depth, sample_size) = match image.samples {
        Samples::U8(_) => (8_u8, 1_usize),
        Samples::U16(_) => (16, 2),
        Samples::F16(_) | Samples::F32(_) => return Err(encode_error("PNG holds 8- or 16-bit integer samples only")),
    };
    let row_samples = (width as usize).checked_mul(channels).ok_or(encode_error("image is too large"))?;
    if row_samples.checked_mul(height as usize) != Some(image.samples.len()) {
        return Err(encode_error("sample count does not match the dimensions"));
    }
    let stride = row_samples.checked_mul(sample_size).ok_or(encode_error("image is too large"))?;
    let total = (stride + 1).checked_mul(height as usize).ok_or(encode_error("image is too large"))?;
    let bpp = channels * sample_size;

    let mut filtered = Vec::with_capacity(total);
    let mut prev = vec![0_u8; stride];
    let mut cur = vec![0_u8; stride];
    let mut candidates = if options.filter == FilterChoice::Adaptive { vec![0_u8; 5 * stride] } else { Vec::new() };
    let mut single = vec![0_u8; if options.filter == FilterChoice::Adaptive { 0 } else { stride }];
    for y in 0..height as usize {
        let range = y * row_samples..(y + 1) * row_samples;
        match &image.samples {
            Samples::U8(v) => cur.copy_from_slice(&v[range]),
            Samples::U16(v) => {
                for (d, &s) in cur.as_chunks_mut::<2>().0.iter_mut().zip(&v[range]) {
                    *d = s.to_be_bytes();
                }
            }
            Samples::F16(_) | Samples::F32(_) => {}
        }
        let fixed = match options.filter {
            FilterChoice::Adaptive => None,
            FilterChoice::None => Some(NONE),
            FilterChoice::Sub => Some(SUB),
            FilterChoice::Up => Some(UP),
            FilterChoice::Average => Some(AVERAGE),
            FilterChoice::Paeth => Some(PAETH),
        };
        if let Some(kind) = fixed {
            filter::filter(kind, &cur, &prev, bpp, &mut single);
            filtered.push(kind);
            filtered.extend_from_slice(&single);
        } else {
            let mut best = (u64::MAX, 0_usize);
            for (kind, out) in candidates.chunks_exact_mut(stride.max(1)).enumerate() {
                filter::filter(kind as u8, &cur, &prev, bpp, out);
                let cost = filter::cost(out);
                if cost < best.0 {
                    best = (cost, kind);
                }
            }
            filtered.push(best.1 as u8);
            filtered.extend_from_slice(&candidates[best.1 * stride..(best.1 + 1) * stride]);
        }
        core::mem::swap(&mut prev, &mut cur);
    }
    let stream = zlib_compress(&filtered, options.level);
    drop(filtered);

    let mut out = Vec::with_capacity(stream.len() + 128);
    out.extend_from_slice(&SIGNATURE);
    let color_type: u8 = match channels {
        1 => 0,
        2 => 4,
        3 => 2,
        _ => 6,
    };
    let mut ihdr = [0_u8; 13];
    ihdr[..4].copy_from_slice(&width.to_be_bytes());
    ihdr[4..8].copy_from_slice(&height.to_be_bytes());
    ihdr[8] = depth;
    ihdr[9] = color_type;
    write_chunk(&mut out, *b"IHDR", &ihdr);
    match image.color_space {
        ColorSpace::Srgb => {
            // sRGB with the gAMA the specification recommends beside it for older decoders.
            write_chunk(&mut out, *b"sRGB", &[0]);
            write_chunk(&mut out, *b"gAMA", &45_455_u32.to_be_bytes());
        }
        space => {
            write_chunk(&mut out, *b"gAMA", &100_000_u32.to_be_bytes());
            if space != ColorSpace::Linear {
                let mut text = COLOR_SPACE_KEYWORD.as_bytes().to_vec();
                text.push(0);
                text.extend_from_slice(color_space_name(space).as_bytes());
                write_chunk(&mut out, *b"tEXt", &text);
            }
        }
    }
    for piece in stream.chunks(IDAT_MAX) {
        write_chunk(&mut out, *b"IDAT", piece);
    }
    write_chunk(&mut out, *b"IEND", &[]);
    Ok(out)
}
