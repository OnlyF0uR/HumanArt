//! The baseline encoder `[IO-02]`: JFIF, the Annex K quantisation tables scaled to an IJG quality,
//! the Annex K Huffman tables, 4:4:4 or 4:2:0, optional restart intervals.

use super::fdct::{FdctTable, divisors, fdct_quantize};
use super::huffman::{BitWriter, EncodeTable, magnitude_bits, magnitude_code};
use super::tables::{STD_CHROMA_QUANT, STD_LUMA_QUANT, ZIGZAG, scaled_quant, std_huffman};
use super::{EncodeOptions, FORMAT, Subsampling};
use crate::error::{Error, Result};
use crate::image::{ColorSpace, Image, Samples};

const fn encode_error(what: &'static str) -> Error {
    Error::Encode { format: FORMAT, what }
}

/// Appends a marker segment.
fn segment(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    out.extend_from_slice(&[0xFF, marker]);
    out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(payload);
}

/// One component plane, padded to whole MCUs by edge replication.
struct Plane {
    samples: Vec<u8>,
    stride: usize,
}

impl Plane {
    /// The 8×8 block at block coordinates `(bx, by)`, level-shifted.
    fn block(&self, bx: usize, by: usize) -> [f32; 64] {
        let mut out = [0.0_f32; 64];
        for (r, row) in out.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let start = (by * 8 + r) * self.stride + bx * 8;
            for (o, &s) in row.iter_mut().zip(&self.samples[start..start + 8]) {
                *o = f32::from(s) - 128.0;
            }
        }
        out
    }
}

/// The Huffman tables and quantisation of one component.
struct Coding {
    divisors: FdctTable,
    dc: EncodeTable,
    ac: EncodeTable,
}

/// Encodes one block: DC difference and run/size coded AC coefficients (T.81 F.1.2).
fn encode_block(w: &mut BitWriter<'_>, coef: &[i16; 64], pred: &mut i32, coding: &Coding) {
    let diff = i32::from(coef[0]) - *pred;
    *pred = i32::from(coef[0]);
    let n = magnitude_bits(diff);
    w.symbol(&coding.dc, n as u8);
    w.put(magnitude_code(diff), n);
    let mut run = 0_u32;
    for &z in &ZIGZAG[1..] {
        let v = i32::from(coef[usize::from(z)]);
        if v == 0 {
            run += 1;
            continue;
        }
        while run > 15 {
            w.symbol(&coding.ac, 0xF0);
            run -= 16;
        }
        let n = magnitude_bits(v);
        w.symbol(&coding.ac, (run << 4 | n) as u8);
        w.put(magnitude_code(v), n);
        run = 0;
    }
    if run > 0 {
        w.symbol(&coding.ac, 0x00);
    }
}

/// JFIF RGB to YCbCr with libjpeg's fixed-point constants (`jccolor.c`).
#[inline]
fn rgb_to_ycc(r: u8, g: u8, b: u8) -> [u8; 3] {
    let (r, g, b) = (i32::from(r), i32::from(g), i32::from(b));
    [
        ((19_595 * r + 38_470 * g + 7_471 * b + 32_768) >> 16) as u8,
        ((-11_059 * r - 21_709 * g + 32_768 * b + (128 << 16) + 32_767) >> 16) as u8,
        ((32_768 * r - 27_439 * g - 5_329 * b + (128 << 16) + 32_767) >> 16) as u8,
    ]
}

/// Encodes an 8-bit sRGB image as baseline JPEG with the default options at `quality` (1 to 100).
pub fn encode(image: &Image, quality: u8) -> Result<Vec<u8>> {
    encode_with(image, &EncodeOptions { quality, ..EncodeOptions::default() })
}

/// Encodes an 8-bit sRGB image as baseline JPEG.
///
/// Grey (1 channel) is written as one component, RGB (3) as YCbCr; an alpha channel (2 or 4
/// channels) is dropped, since JPEG has none. Refuses other sample types, colour spaces other than
/// sRGB (the decoder could not give them back), and sides over 65 535 pixels.
pub fn encode_with(image: &Image, options: &EncodeOptions) -> Result<Vec<u8>> {
    let Samples::U8(src) = &image.samples else {
        return Err(encode_error("JPEG holds 8-bit samples only"));
    };
    if image.color_space != ColorSpace::Srgb {
        return Err(encode_error("JPEG holds sRGB colour only"));
    }
    let (width, height) = (image.width as usize, image.height as usize);
    if width == 0 || height == 0 || width > 65_535 || height > 65_535 {
        return Err(encode_error("image sides must be 1 to 65 535 pixels"));
    }
    let channels = usize::from(image.channels);
    if !(1..=4).contains(&channels) || src.len() != width * height * channels {
        return Err(encode_error("sample count does not match the dimensions"));
    }
    let grey = channels <= 2;
    let quality = options.quality.clamp(1, 100);
    let luma_q = scaled_quant(&STD_LUMA_QUANT, quality);
    let chroma_q = scaled_quant(&STD_CHROMA_QUANT, quality);
    let (h, v) = if grey || options.subsampling == Subsampling::S444 { (1, 1) } else { (2, 2) };
    let (mcu_w, mcu_h) = (8 * h, 8 * v);
    let (mcux, mcuy) = (width.div_ceil(mcu_w), height.div_ceil(mcu_h));
    let (pw, ph) = (mcux * mcu_w, mcuy * mcu_h);

    // Full-resolution component planes padded to whole MCUs by edge replication.
    let ncomp = if grey { 1 } else { 3 };
    let mut full: Vec<Vec<u8>> = (0..ncomp).map(|_| vec![0_u8; pw * ph]).collect();
    for y in 0..ph {
        let sy = y.min(height - 1);
        for x in 0..pw {
            let sx = x.min(width - 1);
            let px = &src[(sy * width + sx) * channels..][..channels];
            if grey {
                full[0][y * pw + x] = px[0];
            } else {
                for (plane, value) in full.iter_mut().zip(rgb_to_ycc(px[0], px[1], px[2])) {
                    plane[y * pw + x] = value;
                }
            }
        }
    }
    let mut planes: Vec<Plane> = Vec::with_capacity(ncomp);
    for (i, samples) in full.into_iter().enumerate() {
        if i == 0 || (h, v) == (1, 1) {
            planes.push(Plane { samples, stride: pw });
        } else {
            // 2×2 box average for 4:2:0.
            let (cw, ch) = (pw / 2, ph / 2);
            let mut half = vec![0_u8; cw * ch];
            for y in 0..ch {
                for x in 0..cw {
                    let at = 2 * y * pw + 2 * x;
                    let sum = u32::from(samples[at])
                        + u32::from(samples[at + 1])
                        + u32::from(samples[at + pw])
                        + u32::from(samples[at + pw + 1]);
                    half[y * cw + x] = ((sum + 2) >> 2) as u8;
                }
            }
            planes.push(Plane { samples: half, stride: cw });
        }
    }

    let mut out = Vec::with_capacity(1024 + width * height / 2);
    out.extend_from_slice(&[0xFF, 0xD8]);
    segment(&mut out, 0xE0, b"JFIF\0\x01\x01\0\0\x01\0\x01\0\0");
    let mut dqt = Vec::with_capacity(130);
    for (slot, table) in [luma_q, chroma_q].iter().enumerate().take(if grey { 1 } else { 2 }) {
        dqt.push(slot as u8);
        dqt.extend(ZIGZAG.iter().map(|&z| table[usize::from(z)] as u8));
    }
    segment(&mut out, 0xDB, &dqt);
    let mut sof = vec![8];
    sof.extend_from_slice(&(height as u16).to_be_bytes());
    sof.extend_from_slice(&(width as u16).to_be_bytes());
    sof.push(ncomp as u8);
    for i in 0..ncomp {
        let sampling = if i == 0 { (h << 4 | v) as u8 } else { 0x11 };
        sof.extend_from_slice(&[i as u8 + 1, sampling, u8::from(i > 0)]);
    }
    segment(&mut out, 0xC0, &sof);
    let mut dht = Vec::new();
    for (class, ac) in [(0x00_u8, false), (0x10, true)] {
        for chroma in [false, true].iter().take(if grey { 1 } else { 2 }) {
            let spec = std_huffman(ac, *chroma);
            dht.push(class | u8::from(*chroma));
            dht.extend_from_slice(&spec.counts);
            dht.extend_from_slice(&spec.symbols);
        }
    }
    segment(&mut out, 0xC4, &dht);
    if options.restart_interval > 0 {
        segment(&mut out, 0xDD, &options.restart_interval.to_be_bytes());
    }
    let mut sos = vec![ncomp as u8];
    for i in 0..ncomp {
        sos.extend_from_slice(&[i as u8 + 1, if i == 0 { 0x00 } else { 0x11 }]);
    }
    sos.extend_from_slice(&[0, 63, 0]);
    segment(&mut out, 0xDA, &sos);

    let coding = |chroma: bool| -> Result<Coding> {
        Ok(Coding {
            divisors: divisors(if chroma { &chroma_q } else { &luma_q }),
            dc: EncodeTable::new(&std_huffman(false, chroma))?,
            ac: EncodeTable::new(&std_huffman(true, chroma))?,
        })
    };
    let codings = [coding(false)?, coding(true)?];
    let restart = usize::from(options.restart_interval);
    let mut preds = [0_i32; 3];
    let mut coef = [0_i16; 64];
    let mut w = BitWriter::new(&mut out);
    let mut mcu = 0_usize;
    for my in 0..mcuy {
        for mx in 0..mcux {
            if restart > 0 && mcu > 0 && mcu.is_multiple_of(restart) {
                w.flush();
                let n = ((mcu / restart - 1) % 8) as u8;
                w.out().extend_from_slice(&[0xFF, 0xD0 + n]);
                preds = [0; 3];
            }
            mcu += 1;
            for (i, plane) in planes.iter().enumerate() {
                let (ch, cv) = if i == 0 { (h, v) } else { (1, 1) };
                let coding = &codings[usize::from(i > 0)];
                for by in 0..cv {
                    for bx in 0..ch {
                        fdct_quantize(&plane.block(mx * ch + bx, my * cv + by), &coding.divisors, &mut coef);
                        encode_block(&mut w, &coef, &mut preds[i], coding);
                    }
                }
            }
        }
    }
    w.flush();
    out.extend_from_slice(&[0xFF, 0xD9]);
    Ok(out)
}
