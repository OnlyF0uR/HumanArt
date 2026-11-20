//! Radiance HDR pictures (RGBE, `.hdr` / `.pic`): reader and writer `[IO-03]` `[DEP-05]`.
//!
//! The format is Greg Ward's: a text header, a resolution line and scanlines of four-byte pixels
//! (three 8-bit mantissas sharing one 8-bit exponent). It is what most HDRI panoramas for image-based
//! lighting come in `[LIT-01]`.
//!
//! # What is read
//!
//! - The magic line `#?RADIANCE` or `#?RGBE`, header lines up to the blank line (`FORMAT=`,
//!   `EXPOSURE=`, comments; every other variable is ignored), LF or CRLF line ends.
//! - `FORMAT=32-bit_rle_rgbe`, or no `FORMAT` line at all (Radiance's default).
//! - All eight resolution strings (`-Y h +X w` and its flips and transposes). The decoded [`Image`]
//!   always has its first row at the top and its first column on the left.
//! - All three scanline encodings: the adaptive run-length encoding of each component ("new" RLE),
//!   the old run-length encoding (a pixel with mantissas 1, 1, 1 repeats the previous pixel) and flat
//!   scanlines.
//!
//! # What is not
//!
//! - `FORMAT=32-bit_rle_xyze` (CIE XYZ pictures) is [`Error::Unsupported`].
//! - `EXPOSURE` is reported in [`Info::exposure`] and not applied: samples are the stored values.
//!   Dividing by it gives radiance in W/sr/m².
//! - `COLORCORR`, `PRIMARIES`, `PIXASPECT` and `VIEW` are ignored; samples are taken as linear
//!   Rec.709.
//!
//! # RGBE and floats
//!
//! [`rgbe_to_f32`] follows Radiance's `colr_color`: a component is `(mantissa + 0.5) · 2^(e − 136)`,
//! and an exponent byte of 0 is black. The half is Ward's convention: a mantissa stands for the middle
//! of the interval of values that truncate to it, so [`f32_to_rgbe`] truncates (as `setcolr` does) and
//! the round trip errs by at most half a step instead of a whole one. A consequence is that a zero
//! component of a non-black pixel reads back as half a step, not as 0.
//!
//! # Limits `[ROB-02]`
//!
//! The header may be at most [`MAX_HEADER_BYTES`] long. Width and height are checked with
//! [`Limits::check_image`], the decoded size (12 bytes a pixel) with [`Limits::check_bytes`], and the
//! input must hold at least the smallest possible encoding of every scanline, all before the samples
//! are allocated.

use crate::bytes::Reader;
use crate::error::{Error, Format, Result};
use crate::image::{ColorSpace, Image, Limits, Samples};

/// The format every error of this module names.
const FORMAT: Format = Format::Hdr;

/// The longest header (magic line through resolution line) the reader accepts, in bytes.
pub const MAX_HEADER_BYTES: usize = 64 * 1024;

/// The shortest scanline that may use the adaptive run-length encoding (Radiance `MINELEN`).
const MIN_RLE_LEN: usize = 8;
/// The longest scanline that may use the adaptive run-length encoding (Radiance `MAXELEN`).
const MAX_RLE_LEN: usize = 0x7FFF;
/// The shortest run the writer encodes as a run (Radiance `MINRUN`).
const MIN_RUN: usize = 4;
/// The exponent bias of RGBE: the exponent byte is the binary exponent plus 128.
const EXPONENT_BIAS: i32 = 128;

/// `2^(e − 136)` for every exponent byte `e`: the weight of one mantissa step.
const SCALE: [f32; 256] = scale_table();

/// Builds [`SCALE`] from bit patterns, so every entry (the subnormal ones included) is exact.
const fn scale_table() -> [f32; 256] {
    let mut table = [0.0_f32; 256];
    let mut e = 0;
    while e < 256 {
        let exponent = e as i32 - (EXPONENT_BIAS + 8);
        table[e] = if exponent >= -126 {
            f32::from_bits(((exponent + 127) as u32) << 23)
        } else {
            // Subnormal: the smallest is 2^-149, and the smallest exponent here is -136.
            f32::from_bits(1 << (exponent + 149))
        };
        e += 1;
    }
    table
}

/// An [`Error::Invalid`] of this module.
const fn invalid(offset: usize, what: &'static str) -> Error {
    Error::Invalid { format: FORMAT, offset, what }
}

/// How the scanlines of a file are laid out, as named by its resolution string.
///
/// Radiance's Y axis points up, so the usual `-Y h +X w` (rows from the top, pixels from the left)
/// has `y_decreasing` set and nothing else.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Orientation {
    /// The X axis is named first: scanlines are columns of `height` pixels (`+X w +Y h`).
    pub x_major: bool,
    /// X runs from right to left in file order (`-X`).
    pub x_decreasing: bool,
    /// Y runs from top to bottom in file order (`-Y`).
    pub y_decreasing: bool,
}

impl Orientation {
    /// `-Y h +X w`: rows from the top, pixels from the left. What every writer uses.
    pub const STANDARD: Self = Self { x_major: false, x_decreasing: false, y_decreasing: true };

    /// Where the pixel at position `pos` of scanline `scan` lands in a top-left image of
    /// `width × height`, as `(column, row)`.
    const fn place(self, scan: usize, pos: usize, width: usize, height: usize) -> (usize, usize) {
        let (x, y) = if self.x_major { (scan, pos) } else { (pos, scan) };
        let column = if self.x_decreasing { width - 1 - x } else { x };
        let row = if self.y_decreasing { y } else { height - 1 - y };
        (column, row)
    }
}

/// What the header of a Radiance picture says.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Info {
    /// Width in pixels of the decoded image.
    pub width: u32,
    /// Height in pixels of the decoded image.
    pub height: u32,
    /// The product of every `EXPOSURE=` line (1 if there is none). The stored samples are radiance
    /// multiplied by this.
    pub exposure: f32,
    /// The scanline layout of the file. The decoded image is always top-left.
    pub orientation: Orientation,
}

/// Converts an RGBE pixel to linear floats: `(mantissa + 0.5) · 2^(e − 136)`, black if `e` is 0.
///
/// Exact for every input (the smallest results are subnormal floats).
#[must_use]
#[inline]
pub fn rgbe_to_f32(rgbe: [u8; 4]) -> [f32; 3] {
    if rgbe[3] == 0 {
        return [0.0; 3];
    }
    let scale = SCALE[rgbe[3] as usize];
    [(f32::from(rgbe[0]) + 0.5) * scale, (f32::from(rgbe[1]) + 0.5) * scale, (f32::from(rgbe[2]) + 0.5) * scale]
}

/// Converts linear floats to an RGBE pixel, truncating the mantissas (Radiance `setcolr`).
///
/// The largest component gets a mantissa of 128 to 255. Negative components and NaN become 0;
/// values over the largest RGBE value (about 1.7e38) are clamped to it, keeping the ratio of the
/// components; a pixel whose largest component is under `2^-128` (about 2.9e-39) becomes black.
/// `rgbe_to_f32(f32_to_rgbe(c))` differs from `c` by at most half a mantissa step, that is by less
/// than 1/256 of the largest component, and `f32_to_rgbe(rgbe_to_f32(p)) == p` for every pixel a
/// writer produces (largest mantissa at least 128).
#[must_use]
pub fn f32_to_rgbe(rgb: [f32; 3]) -> [u8; 4] {
    let clean = |c: f32| if c.is_nan() || c <= 0.0 { 0.0_f64 } else { f64::from(c.min(f32::MAX)) };
    let (r, g, b) = (clean(rgb[0]), clean(rgb[1]), clean(rgb[2]));
    let max = r.max(g).max(b);
    if max <= 0.0 {
        return [0; 4];
    }
    // frexp: max = f · 2^exponent with f in [0.5, 1). A positive f32 is a normal f64, so the
    // exponent field can be read directly.
    let exponent = ((max.to_bits() >> 52) & 0x7FF) as i32 - 1022;
    if exponent + EXPONENT_BIAS < 1 {
        return [0; 4];
    }
    let exponent = exponent.min(255 - EXPONENT_BIAS);
    // 2^(8 − exponent), between 2^-119 and 2^135: a normal f64.
    let scale = f64::from_bits(((8 - exponent + 1023) as u64) << 52);
    let mantissa = |c: f64| (c * scale).min(255.0) as u8;
    [mantissa(r), mantissa(g), mantissa(b), (exponent + EXPONENT_BIAS) as u8]
}

/// Reads the next header line (without its line end) and consumes it.
fn header_line<'a>(r: &mut Reader<'a>) -> Result<&'a [u8]> {
    let rest = r.rest();
    let budget = MAX_HEADER_BYTES.saturating_sub(r.pos());
    let window = &rest[..budget.min(rest.len())];
    let Some(end) = window.iter().position(|&b| b == b'\n') else {
        if rest.len() > window.len() || r.pos() >= MAX_HEADER_BYTES {
            // The header's real size is unknown (it has no end in sight): report the input's.
            let got = r.input().len() as u64;
            return Err(Error::Limit { format: FORMAT, what: "header size", limit: MAX_HEADER_BYTES as u64, got });
        }
        return Err(Error::Truncated { format: FORMAT, offset: r.input().len() });
    };
    let line = r.bytes(end)?;
    r.skip(1)?;
    Ok(line.strip_suffix(b"\r").unwrap_or(line))
}

/// Parses one `<sign><axis> <count>` half of a resolution line.
fn resolution_part(sign_axis: &[u8], count: &[u8], at: usize) -> Result<(bool, u8, u32)> {
    let bad = invalid(at, "malformed resolution line");
    let (&[sign, axis], false) = (sign_axis, count.is_empty()) else {
        return Err(bad);
    };
    if !(sign == b'+' || sign == b'-') || !(axis == b'X' || axis == b'Y') {
        return Err(bad);
    }
    let mut value = 0_u32;
    for &digit in count {
        if !digit.is_ascii_digit() {
            return Err(bad);
        }
        // An overflowing size is far beyond any limit: report it as the largest value.
        value = value.checked_mul(10).and_then(|v| v.checked_add(u32::from(digit - b'0'))).unwrap_or(u32::MAX);
    }
    Ok((sign == b'-', axis, value))
}

/// Parses the header and the resolution line, leaving `r` at the first scanline.
fn read_header(r: &mut Reader<'_>, limits: &Limits) -> Result<Info> {
    let magic = header_line(r)?;
    if magic != b"#?RADIANCE" && magic != b"#?RGBE" {
        return Err(invalid(0, "not a Radiance picture (no #?RADIANCE or #?RGBE line)"));
    }
    let mut exposure = 1.0_f32;
    loop {
        let at = r.pos();
        let line = header_line(r)?;
        if line.is_empty() {
            break;
        }
        if let Some(value) = line.strip_prefix(b"FORMAT=") {
            match value.trim_ascii() {
                b"32-bit_rle_rgbe" => {}
                b"32-bit_rle_xyze" => {
                    return Err(Error::Unsupported { format: FORMAT, what: "the XYZE pixel format" });
                }
                _ => return Err(invalid(at, "unknown FORMAT")),
            }
        } else if let Some(value) = line.strip_prefix(b"EXPOSURE=") {
            let factor = std::str::from_utf8(value).ok().and_then(|text| text.trim().parse::<f32>().ok());
            match factor {
                Some(factor) if factor.is_finite() && factor > 0.0 => exposure *= factor,
                _ => return Err(invalid(at, "EXPOSURE is not a positive number")),
            }
        }
    }
    if !exposure.is_finite() || exposure <= 0.0 {
        return Err(invalid(r.pos(), "the combined EXPOSURE is out of range"));
    }

    let at = r.pos();
    let line = header_line(r)?;
    let mut parts = line.split(|&b| b == b' ').filter(|part| !part.is_empty());
    let mut next = || parts.next().unwrap_or(&[]);
    let first = resolution_part(next(), next(), at)?;
    let second = resolution_part(next(), next(), at)?;
    if !next().is_empty() || first.1 == second.1 {
        return Err(invalid(at, "malformed resolution line"));
    }
    let x_major = first.1 == b'X';
    let (x, y) = if x_major { (first, second) } else { (second, first) };
    // Radiance's Y points up: `-Y` is top to bottom.
    let orientation = Orientation { x_major, x_decreasing: x.0, y_decreasing: y.0 };
    limits.check_image(FORMAT, u64::from(x.2), u64::from(y.2)).map_err(|e| match e {
        Error::Invalid { what, .. } => invalid(at, what),
        other => other,
    })?;
    Ok(Info { width: x.2, height: y.2, exposure, orientation })
}

/// Reads one old-style scanline: flat pixels, where a pixel with mantissas 1, 1, 1 repeats the
/// previous pixel `exponent << shift` times and consecutive repeats shift by eight more bits.
fn read_old_scanline(r: &mut Reader<'_>, line: &mut [[u8; 4]]) -> Result<()> {
    let mut filled = 0;
    let mut shift = 0_u32;
    // Progress: every turn consumes four input bytes or fails.
    while filled < line.len() {
        let at = r.pos();
        let pixel = r.array::<4>()?;
        if pixel[0] == 1 && pixel[1] == 1 && pixel[2] == 1 {
            if filled == 0 {
                return Err(invalid(at, "repeat count before the first pixel of a scanline"));
            }
            if shift > 24 {
                return Err(invalid(at, "repeat count is too large"));
            }
            let count = (pixel[3] as usize) << shift;
            if count > line.len() - filled {
                return Err(invalid(at, "repeat runs past the end of the scanline"));
            }
            let previous = line[filled - 1];
            line[filled..filled + count].fill(previous);
            filled += count;
            shift += 8;
        } else {
            line[filled] = pixel;
            filled += 1;
            shift = 0;
        }
    }
    Ok(())
}

/// Reads one scanline in whichever of the three encodings it uses.
fn read_scanline(r: &mut Reader<'_>, line: &mut [[u8; 4]]) -> Result<()> {
    let len = line.len();
    let adaptive = (MIN_RLE_LEN..=MAX_RLE_LEN).contains(&len)
        && matches!(r.rest(), [2, 2, high, ..] if high & 0x80 == 0)
        && r.rest().len() >= 4;
    if !adaptive {
        return read_old_scanline(r, line);
    }
    let at = r.pos();
    let marker = r.array::<4>()?;
    if (marker[2] as usize) << 8 | marker[3] as usize != len {
        return Err(invalid(at, "scanline length does not match the picture"));
    }
    for component in 0..4 {
        let mut filled = 0;
        // Progress: every turn consumes input and adds at least one sample, or fails.
        while filled < len {
            let at = r.pos();
            let code = r.u8()?;
            let (run, count) = if code > 128 { (true, (code & 127) as usize) } else { (false, code as usize) };
            if count == 0 || count > len - filled {
                return Err(invalid(at, "run does not fit the scanline"));
            }
            let target = &mut line[filled..filled + count];
            if run {
                let value = r.u8()?;
                for pixel in target {
                    pixel[component] = value;
                }
            } else {
                let values = r.bytes(count)?;
                target.iter_mut().zip(values).for_each(|(pixel, &value)| pixel[component] = value);
            }
            filled += count;
        }
    }
    Ok(())
}

/// Reads the header of a Radiance picture without decoding its pixels.
///
/// The dimensions are checked against `limits`.
pub fn read_info(bytes: &[u8], limits: &Limits) -> Result<Info> {
    read_header(&mut Reader::new(bytes, FORMAT), limits)
}

/// Decodes a Radiance picture to a three-channel linear `f32` image, first row at the top.
///
/// See the module documentation for what is read and for the limits.
pub fn decode(bytes: &[u8], limits: &Limits) -> Result<Image> {
    decode_with_info(bytes, limits).map(|(_, image)| image)
}

/// [`decode`], also returning what the header says (exposure, the file's orientation).
pub fn decode_with_info(bytes: &[u8], limits: &Limits) -> Result<(Info, Image)> {
    let mut r = Reader::new(bytes, FORMAT);
    let info = read_header(&mut r, limits)?;
    let (width, height) = (info.width as usize, info.height as usize);
    let pixels = limits.check_image(FORMAT, u64::from(info.width), u64::from(info.height))?;
    limits.check_bytes(FORMAT, "decoded size", pixels as u64 * 12)?;
    let (scan_len, scan_count) = if info.orientation.x_major { (height, width) } else { (width, height) };
    // The smallest scanline is one pixel, plus one repeat pixel if it is longer than that.
    r.check_count(scan_count, 4 * scan_len.min(2))?;

    let mut samples = vec![0.0_f32; pixels * 3];
    let mut line = vec![[0_u8; 4]; scan_len];
    for scan in 0..scan_count {
        read_scanline(&mut r, &mut line)?;
        if info.orientation == Orientation::STANDARD {
            let row = &mut samples[scan * width * 3..(scan + 1) * width * 3];
            for (out, &pixel) in row.as_chunks_mut::<3>().0.iter_mut().zip(&line) {
                out.copy_from_slice(&rgbe_to_f32(pixel));
            }
        } else {
            for (pos, &pixel) in line.iter().enumerate() {
                let (column, row) = info.orientation.place(scan, pos, width, height);
                let at = (row * width + column) * 3;
                samples[at..at + 3].copy_from_slice(&rgbe_to_f32(pixel));
            }
        }
    }
    let image = Image::new(FORMAT, info.width, info.height, 3, ColorSpace::Linear, Samples::F32(samples))?;
    Ok((info, image))
}

/// Appends one component of a scanline in the adaptive run-length encoding: a count byte over 128
/// is a run of `count − 128` copies of the next byte, any other is that many literal bytes.
fn write_rle_component(out: &mut Vec<u8>, data: &[u8]) {
    let mut done = 0;
    while done < data.len() {
        // Find the next run of at least MIN_RUN equal bytes at or after `done`.
        let mut run_start = done;
        let mut run_len = 0;
        while run_start < data.len() {
            run_len = 1;
            while run_len < 127 && run_start + run_len < data.len() && data[run_start + run_len] == data[run_start] {
                run_len += 1;
            }
            if run_len >= MIN_RUN {
                break;
            }
            run_start += run_len;
        }
        while done < run_start {
            let count = (run_start - done).min(128);
            out.push(count as u8);
            out.extend_from_slice(&data[done..done + count]);
            done += count;
        }
        if run_start < data.len() {
            out.push(128 + run_len as u8);
            out.push(data[run_start]);
            done = run_start + run_len;
        }
    }
}

/// Appends one scanline: run-length encoded when its length allows it, flat otherwise.
fn write_scanline(out: &mut Vec<u8>, line: &[[u8; 4]], component: &mut Vec<u8>) {
    if !(MIN_RLE_LEN..=MAX_RLE_LEN).contains(&line.len()) {
        for pixel in line {
            out.extend_from_slice(pixel);
        }
        return;
    }
    out.extend_from_slice(&[2, 2, (line.len() >> 8) as u8, line.len() as u8]);
    for index in 0..4 {
        component.clear();
        component.extend(line.iter().map(|pixel| pixel[index]));
        write_rle_component(out, component);
    }
}

/// Encodes an image as a Radiance picture (`#?RADIANCE`, `-Y h +X w`, run-length encoded scanlines).
///
/// The image must hold `f32` samples with 1 (grey), 3 (RGB) or 4 (RGBA, alpha is dropped) channels,
/// taken as linear. Each pixel goes through [`f32_to_rgbe`], which says what happens to negative,
/// non-finite and out-of-range values. Scanlines shorter than 8 or longer than 32767 pixels cannot
/// be run-length encoded and are written flat. The output is deterministic.
pub fn encode(image: &Image) -> Result<Vec<u8>> {
    let Samples::F32(samples) = &image.samples else {
        return Err(Error::Encode { format: FORMAT, what: "samples must be f32" });
    };
    let channels = image.channels as usize;
    if !matches!(channels, 1 | 3 | 4) {
        return Err(Error::Encode { format: FORMAT, what: "channel count must be 1, 3 or 4" });
    }
    let width = image.width as usize;
    if width == 0 || image.height == 0 || Some(samples.len()) != image.pixel_count().checked_mul(channels) {
        return Err(Error::Encode { format: FORMAT, what: "sample count does not match the dimensions" });
    }
    let mut out = Vec::with_capacity(64 + samples.len());
    out.extend_from_slice(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n");
    out.extend_from_slice(format!("-Y {} +X {}\n", image.height, image.width).as_bytes());
    let mut line = vec![[0_u8; 4]; width];
    let mut component = Vec::with_capacity(width);
    for row in samples.chunks_exact(width * channels) {
        for (pixel, source) in line.iter_mut().zip(row.chunks_exact(channels)) {
            *pixel = match *source {
                [grey] => f32_to_rgbe([grey; 3]),
                [r, g, b, ..] => f32_to_rgbe([r, g, b]),
                _ => [0; 4],
            };
        }
        write_scanline(&mut out, &line, &mut component);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;
    use std::time::Duration;

    /// A 24×17 fixture written by ffmpeg's `hdr` encoder (provenance in the agent STATUS file).
    const FFMPEG_HDR: &[u8] = include_bytes!("../tests/fixtures/hdr/ffmpeg_24x17.hdr");
    /// The same picture decoded by ffmpeg to planar G, B, R `f32` little-endian.
    const FFMPEG_RAW: &[u8] = include_bytes!("../tests/fixtures/hdr/ffmpeg_24x17.gbrpf32le.bin");

    /// A deterministic test picture: smooth areas (runs), noise, black, and a wide range of values.
    fn picture(width: u32, height: u32, seed: u64) -> Image {
        let mut rng = Rng::new(seed);
        let mut samples = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let band = (x / 9 + y / 3) % 4;
                let pixel = match band {
                    0 => [0.25, 0.5, 1.0],
                    1 => [rng.f32() * 4.0, rng.f32(), rng.f32() * 0.01],
                    2 => [0.0, 0.0, 0.0],
                    _ => {
                        let scale = 2.0_f32.powi(rng.range_i64(-30, 30) as i32);
                        [scale * (1.0 + x as f32), scale * 3.0, scale * rng.f32()]
                    }
                };
                samples.extend_from_slice(&pixel);
            }
        }
        Image::new(FORMAT, width, height, 3, ColorSpace::Linear, Samples::F32(samples)).unwrap()
    }

    /// What a picture reads back as: every pixel through RGBE and back.
    fn quantised(image: &Image) -> Vec<f32> {
        let samples = image.as_f32().unwrap();
        samples.as_chunks::<3>().0.iter().flat_map(|&p| rgbe_to_f32(f32_to_rgbe(p))).collect()
    }

    /// How a hand-built file stores its scanlines.
    #[derive(Clone, Copy, PartialEq)]
    enum Mode {
        Flat,
        Adaptive,
        Old,
    }

    /// Old-style run-length encoding of one scanline, with multi-pixel repeat counts.
    fn write_old_scanline(out: &mut Vec<u8>, line: &[[u8; 4]]) {
        let mut i = 0;
        while i < line.len() {
            out.extend_from_slice(&line[i]);
            let mut run = 0_usize;
            while i + 1 + run < line.len() && line[i + 1 + run] == line[i] {
                run += 1;
            }
            i += 1 + run;
            // The count is sent in base 256, least significant digit first; zero digits must be
            // sent too (they advance the shift) unless they are the leading ones.
            let mut digits = Vec::new();
            while run > 0 {
                digits.push((run & 0xFF) as u8);
                run >>= 8;
            }
            for digit in digits {
                out.extend_from_slice(&[1, 1, 1, digit]);
            }
        }
    }

    /// Builds a file with any orientation and scanline encoding from a top-left image.
    fn build(image: &Image, orientation: Orientation, mode: Mode, header: &str) -> Vec<u8> {
        let (width, height) = (image.width as usize, image.height as usize);
        let samples = image.as_f32().unwrap();
        let mut out = header.as_bytes().to_vec();
        let x = format!("{}X {width}", if orientation.x_decreasing { '-' } else { '+' });
        let y = format!("{}Y {height}", if orientation.y_decreasing { '-' } else { '+' });
        let line = if orientation.x_major { format!("{x} {y}\n") } else { format!("{y} {x}\n") };
        out.extend_from_slice(line.as_bytes());
        let (scan_len, scan_count) = if orientation.x_major { (height, width) } else { (width, height) };
        let mut component = Vec::new();
        for scan in 0..scan_count {
            let pixels: Vec<[u8; 4]> = (0..scan_len)
                .map(|pos| {
                    let (column, row) = orientation.place(scan, pos, width, height);
                    let at = (row * width + column) * 3;
                    f32_to_rgbe([samples[at], samples[at + 1], samples[at + 2]])
                })
                .collect();
            match mode {
                Mode::Flat => pixels.iter().for_each(|p| out.extend_from_slice(p)),
                Mode::Adaptive => write_scanline(&mut out, &pixels, &mut component),
                Mode::Old => write_old_scanline(&mut out, &pixels),
            }
        }
        out
    }

    const HEADER: &str = "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n";

    #[test]
    fn rgbe_follows_wards_half_step_convention() {
        assert_eq!(rgbe_to_f32([0, 0, 0, 0]), [0.0; 3]);
        assert_eq!(rgbe_to_f32([200, 100, 50, 0]), [0.0; 3], "exponent 0 is black whatever the mantissas");
        // 1.0 = 0.5 · 2^1: exponent byte 129, mantissa 128; it reads back as (128 + 0.5) / 256 · 2.
        assert_eq!(f32_to_rgbe([1.0, 1.0, 1.0]), [128, 128, 128, 129]);
        let step = 1.0 / 256.0;
        assert_eq!(rgbe_to_f32([128, 128, 128, 129]), [1.0 + step; 3]);
        // A zero component of a lit pixel reads back as half a step.
        assert_eq!(f32_to_rgbe([1.0, 0.0, 0.5]), [128, 0, 64, 129]);
        assert_eq!(rgbe_to_f32([128, 0, 64, 129]), [1.0 + step, step, 0.5 + step]);
        // The ends of the range are exact, the low one in subnormal floats.
        assert_eq!(f64::from(rgbe_to_f32([0, 0, 0, 1])[0]), 0.5 * 2.0_f64.powi(-135));
        assert_eq!(f64::from(rgbe_to_f32([255, 0, 0, 255])[0]), 255.5 * 2.0_f64.powi(119));
        for e in 1..=255_i32 {
            assert_eq!(f64::from(SCALE[e as usize]), 2.0_f64.powi(e - 136), "{e}");
        }
    }

    #[test]
    fn ac_wp03_1_normalised_rgbe_survives_the_float_round_trip_exactly() {
        let mut rng = Rng::new(11);
        for e in 1..=255_u8 {
            for _ in 0..200 {
                let mut pixel = [rng.next_u32() as u8, rng.next_u32() as u8, rng.next_u32() as u8, e];
                pixel[rng.index(3)] |= 0x80;
                assert_eq!(f32_to_rgbe(rgbe_to_f32(pixel)), pixel);
            }
            for top in [128, 255] {
                for pixel in [[top, 0, 0, e], [0, top, 0, e], [0, 0, top, e], [top, top, top, e]] {
                    assert_eq!(f32_to_rgbe(rgbe_to_f32(pixel)), pixel);
                }
            }
        }
    }

    #[test]
    fn ac_wp03_1_floats_round_trip_within_half_a_step_across_the_exponent_range() {
        let mut rng = Rng::new(12);
        // 2^-127 … 2^126: everything RGBE holds with a normalised mantissa.
        for exponent in -127..=126 {
            for _ in 0..50 {
                let scale = 2.0_f64.powi(exponent);
                let rgb = [1.0 + rng.f64(), rng.f64() * 2.0, rng.f64() * 0.01].map(|c| (c * scale) as f32);
                let back = rgbe_to_f32(f32_to_rgbe(rgb));
                let max = f64::from(rgb[0].max(rgb[1]).max(rgb[2]));
                for (a, b) in rgb.iter().zip(back) {
                    let error = (f64::from(*a) - f64::from(b)).abs();
                    assert!(error <= max / 256.0, "{rgb:?} -> {back:?}");
                }
            }
        }
        // Out-of-range input: never a panic, always a defined pixel.
        assert_eq!(f32_to_rgbe([-1.0, f32::NAN, 0.0]), [0; 4]);
        assert_eq!(f32_to_rgbe([-1.0, f32::NAN, 2.0]), [0, 0, 128, 130]);
        assert_eq!(f32_to_rgbe([f32::MAX, f32::MAX / 2.0, 0.0]), [255, 255, 0, 255]);
        assert_eq!(f32_to_rgbe([f32::INFINITY, 1.0, 0.0]), [255, 0, 0, 255]);
        assert_eq!(f32_to_rgbe([1.0e38, 0.5e38, 0.0]), [150, 75, 0, 255]);
        assert_eq!(f32_to_rgbe([f32::MIN_POSITIVE / 8.0; 3]), [0; 4], "below 2^-128 is black");
        assert_eq!(f32_to_rgbe([2.0_f32.powi(-128); 3]), [128, 128, 128, 1]);
    }

    #[test]
    fn ac_wp03_1_pictures_round_trip_through_the_run_length_writer() {
        for (width, height, seed) in [(8, 1, 1), (9, 7, 2), (64, 5, 3), (300, 4, 4), (1, 1, 5), (7, 3, 6), (1, 9, 7)] {
            let image = picture(width, height, seed);
            let bytes = encode(&image).unwrap();
            assert_eq!(bytes, encode(&image).unwrap(), "deterministic");
            let (info, back) = decode_with_info(&bytes, &Limits::DEFAULT).unwrap();
            assert_eq!(
                info,
                Info { width, height, exposure: 1.0, orientation: Orientation::STANDARD },
                "{width}x{height}"
            );
            assert_eq!((back.width, back.height, back.channels), (width, height, 3));
            assert_eq!(back.color_space, ColorSpace::Linear);
            assert_eq!(back.as_f32().unwrap(), quantised(&image), "{width}x{height}");
            // Decoding what was decoded and encoding again reproduces the file: RGBE is stable.
            assert_eq!(encode(&back).unwrap(), bytes);
        }
        // A flat picture of 300 pixels a row compresses to runs: far below four bytes a pixel.
        let flat = Image::new(FORMAT, 300, 2, 3, ColorSpace::Linear, Samples::F32(vec![0.5; 1800])).unwrap();
        assert!(encode(&flat).unwrap().len() < 150);
    }

    #[test]
    fn the_writer_takes_grey_and_rgba_and_refuses_the_rest() {
        let grey = Image::new(FORMAT, 9, 1, 1, ColorSpace::Linear, Samples::F32(vec![0.75; 9])).unwrap();
        let back = decode(&encode(&grey).unwrap(), &Limits::DEFAULT).unwrap();
        assert_eq!(back.as_f32().unwrap(), vec![rgbe_to_f32(f32_to_rgbe([0.75; 3]))[0]; 27]);
        let rgba: Vec<f32> = (0..9).flat_map(|_| [1.0, 2.0, 3.0, 0.5]).collect();
        let rgba = Image::new(FORMAT, 9, 1, 4, ColorSpace::Linear, Samples::F32(rgba)).unwrap();
        let back = decode(&encode(&rgba).unwrap(), &Limits::DEFAULT).unwrap();
        assert_eq!(&back.as_f32().unwrap()[..3], rgbe_to_f32(f32_to_rgbe([1.0, 2.0, 3.0])));
        let two = Image::new(FORMAT, 1, 1, 2, ColorSpace::Linear, Samples::F32(vec![0.0; 2])).unwrap();
        assert!(matches!(encode(&two), Err(Error::Encode { .. })));
        let bytes = Image::new(FORMAT, 1, 1, 3, ColorSpace::Srgb, Samples::U8(vec![0; 3])).unwrap();
        assert!(matches!(encode(&bytes), Err(Error::Encode { .. })));
        let mut broken = picture(4, 4, 1);
        broken.height = 5;
        assert!(matches!(encode(&broken), Err(Error::Encode { .. })));
    }

    #[test]
    fn ac_wp03_1_all_eight_orientations_decode_to_the_same_top_left_image() {
        for (width, height) in [(3, 2), (11, 9), (5, 12)] {
            let image = picture(width, height, 21);
            let expected = quantised(&image);
            let mut seen = 0;
            for bits in 0..8 {
                let orientation =
                    Orientation { x_major: bits & 1 != 0, x_decreasing: bits & 2 != 0, y_decreasing: bits & 4 != 0 };
                for mode in [Mode::Flat, Mode::Adaptive, Mode::Old] {
                    let bytes = build(&image, orientation, mode, HEADER);
                    let (info, back) = decode_with_info(&bytes, &Limits::DEFAULT).unwrap();
                    assert_eq!(info.orientation, orientation);
                    assert_eq!((back.width, back.height), (width, height));
                    assert_eq!(back.as_f32().unwrap(), expected, "{orientation:?} {width}x{height}");
                    seen += 1;
                }
            }
            assert_eq!(seen, 24);
        }
        // The orientation is read from the text, spelled out here once per string.
        let image = picture(2, 2, 3);
        let p = |x: usize, y: usize| {
            let at = (y * 2 + x) * 3;
            let s = image.as_f32().unwrap();
            f32_to_rgbe([s[at], s[at + 1], s[at + 2]])
        };
        // File order of the four pixels (as top-left coordinates) for each string.
        let cases: [(&str, [(usize, usize); 4]); 8] = [
            ("-Y 2 +X 2", [(0, 0), (1, 0), (0, 1), (1, 1)]),
            ("-Y 2 -X 2", [(1, 0), (0, 0), (1, 1), (0, 1)]),
            ("+Y 2 +X 2", [(0, 1), (1, 1), (0, 0), (1, 0)]),
            ("+Y 2 -X 2", [(1, 1), (0, 1), (1, 0), (0, 0)]),
            ("+X 2 -Y 2", [(0, 0), (0, 1), (1, 0), (1, 1)]),
            ("-X 2 -Y 2", [(1, 0), (1, 1), (0, 0), (0, 1)]),
            ("+X 2 +Y 2", [(0, 1), (0, 0), (1, 1), (1, 0)]),
            ("-X 2 +Y 2", [(1, 1), (1, 0), (0, 1), (0, 0)]),
        ];
        for (text, order) in cases {
            let mut bytes = format!("{HEADER}{text}\n").into_bytes();
            for (x, y) in order {
                bytes.extend_from_slice(&p(x, y));
            }
            let back = decode(&bytes, &Limits::DEFAULT).unwrap();
            assert_eq!(back.as_f32().unwrap(), quantised(&image), "{text}");
        }
    }

    #[test]
    fn old_run_length_scanlines_with_long_repeats_decode() {
        // 700 equal pixels after the first: the count 699 needs two repeat pixels (187, then 2 << 8).
        let samples: Vec<f32> = (0..700).flat_map(|_| [0.3, 0.6, 0.9]).collect();
        let image = Image::new(FORMAT, 700, 1, 3, ColorSpace::Linear, Samples::F32(samples)).unwrap();
        let bytes = build(&image, Orientation::STANDARD, Mode::Old, HEADER);
        assert_eq!(bytes.len(), HEADER.len() + "-Y 1 +X 700\n".len() + 12);
        assert_eq!(decode(&bytes, &Limits::DEFAULT).unwrap().as_f32().unwrap(), quantised(&image));
        // A zero digit in the middle of a count: 513 pixels = 1 + 512 = 1 + (0, then 2 << 8).
        let mut bytes = format!("{HEADER}-Y 1 +X 513\n").into_bytes();
        bytes.extend_from_slice(&[10, 20, 30, 128, 1, 1, 1, 0, 1, 1, 1, 2]);
        let back = decode(&bytes, &Limits::DEFAULT).unwrap();
        assert_eq!(back.as_f32().unwrap().len(), 513 * 3);
        assert_eq!(&back.as_f32().unwrap()[512 * 3..], rgbe_to_f32([10, 20, 30, 128]));

        let file = |pixels: &[u8]| {
            let mut bytes = format!("{HEADER}-Y 1 +X 9\n").into_bytes();
            bytes.extend_from_slice(pixels);
            bytes
        };
        let what = |pixels: &[u8]| match decode(&file(pixels), &Limits::DEFAULT) {
            Err(Error::Invalid { what, .. }) => what,
            other => panic!("{other:?}"),
        };
        assert_eq!(what(&[1, 1, 1, 5, 0, 0, 0, 0]), "repeat count before the first pixel of a scanline");
        assert_eq!(what(&[9, 9, 9, 9, 1, 1, 1, 9]), "repeat runs past the end of the scanline");
        assert_eq!(
            what(&[9, 9, 9, 9, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0]),
            "repeat count is too large"
        );
    }

    #[test]
    fn header_variants_are_read() {
        let image = picture(9, 2, 8);
        let expected = quantised(&image);
        let headers = [
            "#?RGBE\nFORMAT=32-bit_rle_rgbe\n\n",
            "#?RADIANCE\n\n",
            "#?RADIANCE\r\n# a comment\r\nFORMAT=32-bit_rle_rgbe\r\nSOFTWARE=x\r\n\r\n",
            "#?RADIANCE\nVIEW= -vtv -vp 0 0 0\nPIXASPECT=1\nFORMAT=32-bit_rle_rgbe \nGAMMA=1\n\n",
        ];
        for header in headers {
            let bytes = build(&image, Orientation::STANDARD, Mode::Adaptive, header);
            let (info, back) = decode_with_info(&bytes, &Limits::DEFAULT).unwrap();
            assert_eq!(info.exposure, 1.0);
            assert_eq!(back.as_f32().unwrap(), expected, "{header:?}");
        }
        // EXPOSURE lines multiply; the samples stay as stored.
        let header = "#?RADIANCE\nEXPOSURE=2.5\nFORMAT=32-bit_rle_rgbe\nEXPOSURE= 4\n\n";
        let bytes = build(&image, Orientation::STANDARD, Mode::Adaptive, header);
        let (info, back) = decode_with_info(&bytes, &Limits::DEFAULT).unwrap();
        assert_eq!(info.exposure, 10.0);
        assert_eq!(read_info(&bytes, &Limits::DEFAULT).unwrap(), info);
        assert_eq!(back.as_f32().unwrap(), expected);
    }

    #[test]
    fn bad_headers_are_refused_with_the_reason() {
        let body = "-Y 1 +X 1\n\u{1}\u{2}\u{3}\u{4}";
        let decode_text = |text: String| decode(text.as_bytes(), &Limits::DEFAULT);
        assert!(decode_text(format!("{HEADER}{body}")).is_ok());
        assert!(matches!(
            decode_text(format!("#?RADIANCE\nFORMAT=32-bit_rle_xyze\n\n{body}")),
            Err(Error::Unsupported { .. })
        ));
        for (text, expected) in [
            (format!("#?PNG\n\n{body}"), "not a Radiance picture (no #?RADIANCE or #?RGBE line)"),
            (format!("#?RADIANCE\nFORMAT=jpeg\n\n{body}"), "unknown FORMAT"),
            (format!("#?RADIANCE\nEXPOSURE=abc\n\n{body}"), "EXPOSURE is not a positive number"),
            (format!("#?RADIANCE\nEXPOSURE=-1\n\n{body}"), "EXPOSURE is not a positive number"),
            (format!("#?RADIANCE\nEXPOSURE=inf\n\n{body}"), "EXPOSURE is not a positive number"),
            (format!("#?RADIANCE\nEXPOSURE=1e30\nEXPOSURE=1e30\n\n{body}"), "the combined EXPOSURE is out of range"),
            (format!("{HEADER}-Y 1 +Y 1\n"), "malformed resolution line"),
            (format!("{HEADER}-Y 1 +X\n"), "malformed resolution line"),
            (format!("{HEADER}-Y 1 +X 1 2\n"), "malformed resolution line"),
            (format!("{HEADER}Y 1 X 1\n"), "malformed resolution line"),
            (format!("{HEADER}-Y 1 +Z 1\n"), "malformed resolution line"),
            (format!("{HEADER}-Y -1 +X 1\n"), "malformed resolution line"),
            (format!("{HEADER}\n"), "malformed resolution line"),
            (format!("{HEADER}-Y 0 +X 4\n"), "image has a zero dimension"),
        ] {
            match decode_text(text.clone()) {
                Err(Error::Invalid { what, .. }) => assert_eq!(what, expected, "{text:?}"),
                other => panic!("{text:?}: {other:?}"),
            }
        }
        // The adaptive marker must name the scanline length.
        let mut wrong = format!("{HEADER}-Y 1 +X 9\n").into_bytes();
        wrong.extend_from_slice(&[2, 2, 0, 8, 0x89, 0, 0x89, 0, 0x89, 0, 0x89, 0]);
        assert!(matches!(
            decode(&wrong, &Limits::DEFAULT),
            Err(Error::Invalid { what: "scanline length does not match the picture", .. })
        ));
        // Runs and literals must fit the scanline, and a zero count is refused.
        for codes in [[0x8A_u8, 0], [10, 0], [0, 0], [0x80, 0]] {
            let mut bytes = format!("{HEADER}-Y 1 +X 9\n").into_bytes();
            bytes.extend_from_slice(&[2, 2, 0, 9]);
            bytes.extend_from_slice(&codes);
            bytes.extend_from_slice(&[0; 64]);
            let result = decode(&bytes, &Limits::DEFAULT);
            assert!(matches!(result, Err(Error::Invalid { what: "run does not fit the scanline", .. })), "{codes:?}");
        }
    }

    #[test]
    fn rob_02_sizes_are_limited_before_allocation() {
        let huge = |text: &str| decode(format!("{HEADER}{text}\n").as_bytes(), &Limits::DEFAULT);
        assert!(matches!(huge("-Y 40000 +X 2"), Err(Error::Limit { what: "image dimension", .. })));
        assert!(matches!(huge("-Y 99999999999999999999 +X 2"), Err(Error::Limit { what: "image dimension", .. })));
        assert!(matches!(huge("-Y 32768 +X 32768"), Err(Error::Limit { what: "pixel count", .. })));
        // Within the limits but with no data behind it: refused before the samples are allocated.
        assert!(matches!(huge("-Y 16384 +X 16384"), Err(Error::Truncated { .. })));
        let small = Limits { max_bytes: 1000, ..Limits::DEFAULT };
        let bytes = encode(&picture(10, 10, 1)).unwrap();
        assert!(matches!(decode(&bytes, &small), Err(Error::Limit { what: "decoded size", .. })));
        assert!(matches!(read_info(&bytes, &Limits { max_dimension: 9, ..small }), Err(Error::Limit { .. })));
        // A header that never ends.
        let mut endless = b"#?RADIANCE\n".to_vec();
        endless.resize(MAX_HEADER_BYTES + 100, b'x');
        assert!(matches!(decode(&endless, &Limits::DEFAULT), Err(Error::Limit { what: "header size", .. })));
        let mut many = b"#?RADIANCE\n".to_vec();
        while many.len() < MAX_HEADER_BYTES + 100 {
            many.extend_from_slice(b"A=1\n");
        }
        assert!(matches!(decode(&many, &Limits::DEFAULT), Err(Error::Limit { what: "header size", .. })));
    }

    #[test]
    fn ac_wp03_2_every_truncation_is_an_error() {
        for (image, mode) in [(picture(12, 5, 1), Mode::Adaptive), (picture(5, 4, 2), Mode::Flat)] {
            let bytes = build(&image, Orientation::STANDARD, mode, HEADER);
            assert!(decode(&bytes, &Limits::DEFAULT).is_ok());
            for len in 0..bytes.len() {
                let result = decode(&bytes[..len], &Limits::DEFAULT);
                assert!(matches!(result, Err(Error::Truncated { .. })), "{len}: {result:?}");
            }
        }
    }

    #[test]
    fn ac_wp03_1_decodes_the_ffmpeg_fixture_like_ffmpeg() {
        let (info, image) = decode_with_info(FFMPEG_HDR, &Limits::DEFAULT).unwrap();
        assert_eq!((info.width, info.height), (24, 17));
        let ours = image.as_f32().unwrap();
        let plane = 24 * 17;
        assert_eq!(FFMPEG_RAW.len(), plane * 3 * 4);
        let theirs = |channel: usize, pixel: usize| {
            // Planar G, B, R.
            let at = ([2, 0, 1][channel] * plane + pixel) * 4;
            f32::from_le_bytes([FFMPEG_RAW[at], FFMPEG_RAW[at + 1], FFMPEG_RAW[at + 2], FFMPEG_RAW[at + 3]])
        };
        let mut lit = 0;
        for pixel in 0..plane {
            let rgb = [ours[pixel * 3], ours[pixel * 3 + 1], ours[pixel * 3 + 2]];
            // ffmpeg reads a mantissa as m · 2^(e − 136), without Ward's half step: the two decoders
            // must agree on the RGBE bytes exactly.
            let rgbe = f32_to_rgbe(rgb);
            for channel in 0..3 {
                let without_half = f32::from(rgbe[channel]) * SCALE[rgbe[3] as usize];
                assert_eq!(theirs(channel, pixel), if rgbe[3] == 0 { 0.0 } else { without_half }, "pixel {pixel}");
            }
            lit += usize::from(rgbe[3] != 0);
        }
        assert!(lit > plane / 2, "the fixture is not mostly black");
        let (low, high) = ours.iter().fold((f32::MAX, 0.0_f32), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        assert!(low < 0.01 && high > 100.0, "the fixture spans a high dynamic range: {low} … {high}");
    }

    #[test]
    fn ac_wp03_2_rob_01_mutation_fuzz() {
        let image = picture(40, 9, 31);
        let small = picture(5, 6, 32);
        let column = Orientation { x_major: true, x_decreasing: true, y_decreasing: false };
        let corpus = vec![
            FFMPEG_HDR.to_vec(),
            encode(&image).unwrap(),
            build(&image, Orientation::STANDARD, Mode::Old, HEADER),
            build(&small, Orientation::STANDARD, Mode::Flat, "#?RGBE\nEXPOSURE=2\n\n"),
            build(&image, column, Mode::Adaptive, HEADER),
        ];
        humanart_core::fuzz::run("hdr", &corpus, 4000, 0x4844_5221, Duration::from_secs(120), |bytes| {
            if let Ok((info, image)) = decode_with_info(bytes, &Limits::FUZZ) {
                assert_eq!(image.samples.len(), info.width as usize * info.height as usize * 3);
            }
            let _ = read_info(bytes, &Limits::FUZZ);
        });
    }
}
