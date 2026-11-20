//! The PNG decoder: chunk walk, IHDR and colour chunks, inflate, unfilter, Adam7, sample conversion.

use super::filter::unfilter;
use super::{COLOR_SPACE_KEYWORD, ColorType, FORMAT, Info, MAX_CHUNK_LEN, MAX_CHUNKS, SIGNATURE, color_space_name};
use crate::deflate::zlib_decompress_into;
use crate::error::{Error, Result};
use crate::image::{ColorSpace, Image, Limits, Samples};
use humanart_core::hash::{crc32, crc32_update};

/// The Adam7 passes: first column, first row, column step, row step.
const ADAM7: [(usize, usize, usize, usize); 7] =
    [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)];

/// A gAMA value within this range of 100 000 (gamma 1.0) marks linear data.
const LINEAR_GAMMA: core::ops::RangeInclusive<u32> = 95_000..=105_000;

fn invalid(offset: usize, what: &'static str) -> Error {
    Error::Invalid { format: FORMAT, offset, what }
}

/// One chunk of the file.
struct Chunk<'a> {
    kind: [u8; 4],
    data: &'a [u8],
    /// Offset of the chunk's length field.
    offset: usize,
}

/// Walks the chunks after the signature, checking lengths, types and CRCs.
struct Chunks<'a> {
    input: &'a [u8],
    pos: usize,
    count: usize,
}

impl<'a> Chunks<'a> {
    /// The next chunk with a good CRC. Ancillary chunks with a bad CRC are skipped; a critical one
    /// is an error.
    fn next(&mut self) -> Result<Chunk<'a>> {
        loop {
            if self.count >= MAX_CHUNKS {
                return Err(Error::Limit {
                    format: FORMAT,
                    what: "chunk count",
                    limit: MAX_CHUNKS as u64,
                    got: self.count as u64 + 1,
                });
            }
            self.count += 1;
            let at = self.pos;
            let truncated = Error::Truncated { format: FORMAT, offset: self.input.len() };
            let head = self.input.get(at..).and_then(<[u8]>::first_chunk::<8>).ok_or(truncated.clone())?;
            let len = u32::from_be_bytes([head[0], head[1], head[2], head[3]]) as usize;
            if len > MAX_CHUNK_LEN {
                return Err(invalid(at, "chunk length over 2^31 - 1"));
            }
            let kind = [head[4], head[5], head[6], head[7]];
            if !kind.iter().all(u8::is_ascii_alphabetic) {
                return Err(invalid(at + 4, "chunk type is not four letters"));
            }
            let data_start = at + 8;
            let data_end = data_start.checked_add(len).ok_or(truncated.clone())?;
            let data = self.input.get(data_start..data_end).ok_or(truncated.clone())?;
            let crc = self.input.get(data_end..).and_then(<[u8]>::first_chunk::<4>).ok_or(truncated)?;
            self.pos = data_end + 4;
            if crc32_update(crc32(&kind), data) == u32::from_be_bytes(*crc) {
                return Ok(Chunk { kind, data, offset: at });
            }
            if kind[0].is_ascii_uppercase() {
                return Err(invalid(data_end, "bad chunk CRC"));
            }
        }
    }
}

/// The IHDR fields.
#[derive(Clone, Copy)]
struct Header {
    width: u32,
    height: u32,
    depth: u8,
    color_type: ColorType,
    interlaced: bool,
}

impl Header {
    fn parse(chunk: &Chunk<'_>) -> Result<Self> {
        let at = chunk.offset + 8;
        let Some(d) = chunk.data.first_chunk::<13>().filter(|_| chunk.data.len() == 13) else {
            return Err(invalid(chunk.offset, "IHDR is not 13 bytes long"));
        };
        let width = u32::from_be_bytes([d[0], d[1], d[2], d[3]]);
        let height = u32::from_be_bytes([d[4], d[5], d[6], d[7]]);
        if width == 0 || height == 0 || width > (1 << 31) - 1 || height > (1 << 31) - 1 {
            return Err(invalid(at, "image width or height is 0 or over 2^31 - 1"));
        }
        let depth = d[8];
        let color_type = ColorType::from_code(d[9])
            .filter(|t| t.allows_depth(depth))
            .ok_or_else(|| invalid(at + 8, "illegal colour type and bit depth combination"))?;
        if d[10] != 0 {
            return Err(invalid(at + 10, "unknown compression method"));
        }
        if d[11] != 0 {
            return Err(invalid(at + 11, "unknown filter method"));
        }
        if d[12] > 1 {
            return Err(invalid(at + 12, "unknown interlace method"));
        }
        Ok(Self { width, height, depth, color_type, interlaced: d[12] == 1 })
    }

    /// Bytes of one filtered row of `width` pixels, without its filter byte.
    fn stride(&self, width: usize) -> usize {
        (width * usize::from(self.color_type.stored_channels()) * usize::from(self.depth)).div_ceil(8)
    }

    /// Bytes per complete pixel for the filters (at least 1).
    fn filter_bpp(&self) -> usize {
        (usize::from(self.color_type.stored_channels()) * usize::from(self.depth) / 8).max(1)
    }

    /// The size of the decompressed image data: every row of every pass with its filter byte.
    fn raw_size(&self) -> u64 {
        let (w, h) = (u64::from(self.width), u64::from(self.height));
        let bits = u64::from(self.color_type.stored_channels()) * u64::from(self.depth);
        let pass_size = |pw: u64, ph: u64| if pw == 0 || ph == 0 { 0 } else { ph * (1 + (pw * bits).div_ceil(8)) };
        if self.interlaced {
            ADAM7
                .iter()
                .map(|&(x0, y0, dx, dy)| {
                    let (x0, y0, dx, dy) = (x0 as u64, y0 as u64, dx as u64, dy as u64);
                    pass_size(w.saturating_sub(x0).div_ceil(dx), h.saturating_sub(y0).div_ceil(dy))
                })
                .sum()
        } else {
            pass_size(w, h)
        }
    }
}

/// Everything read from the chunks.
struct Parsed<'a> {
    header: Header,
    /// PLTE entries, three bytes each (empty when there is none).
    palette: &'a [u8],
    trns: Option<&'a [u8]>,
    srgb_intent: Option<u8>,
    gamma: Option<u32>,
    chromaticities: Option<[u32; 8]>,
    has_icc_profile: bool,
    tagged: Option<ColorSpace>,
    /// The IDAT payloads in order.
    idat: Vec<&'a [u8]>,
    /// Offset of the first IDAT chunk.
    idat_offset: usize,
}

/// Reads the chunks: up to the first IDAT when `header_only`, else up to IEND.
fn parse(bytes: &[u8], header_only: bool) -> Result<Parsed<'_>> {
    if !bytes.starts_with(&SIGNATURE) {
        return Err(if SIGNATURE.starts_with(bytes) {
            Error::Truncated { format: FORMAT, offset: bytes.len() }
        } else {
            invalid(0, "not a PNG signature")
        });
    }
    let mut chunks = Chunks { input: bytes, pos: SIGNATURE.len(), count: 0 };
    let first = chunks.next()?;
    if &first.kind != b"IHDR" {
        return Err(invalid(first.offset, "the first chunk is not IHDR"));
    }
    let header = Header::parse(&first)?;
    let mut p = Parsed {
        header,
        palette: &[],
        trns: None,
        srgb_intent: None,
        gamma: None,
        chromaticities: None,
        has_icc_profile: false,
        tagged: None,
        idat: Vec::new(),
        idat_offset: 0,
    };
    // 0: before the image data, 1: inside the IDAT run, 2: after it.
    let mut stage = 0_u8;
    loop {
        let chunk = chunks.next()?;
        let (data, at) = (chunk.data, chunk.offset);
        if &chunk.kind == b"IDAT" {
            match stage {
                0 => {
                    if header.color_type == ColorType::Indexed && p.palette.is_empty() {
                        return Err(invalid(at, "indexed image without a PLTE chunk"));
                    }
                    p.idat_offset = at;
                    stage = 1;
                }
                1 => {}
                _ => return Err(invalid(at, "IDAT chunks are not consecutive")),
            }
            p.idat.push(data);
            if header_only {
                return Ok(p);
            }
            continue;
        }
        if stage == 1 {
            stage = 2;
        }
        match &chunk.kind {
            b"IEND" => break,
            b"IHDR" => return Err(invalid(at, "second IHDR chunk")),
            b"PLTE" => {
                if stage != 0 || !p.palette.is_empty() {
                    return Err(invalid(at, "PLTE chunk after the image data or repeated"));
                }
                if matches!(header.color_type, ColorType::Grey | ColorType::GreyAlpha) {
                    return Err(invalid(at, "PLTE chunk in a grey image"));
                }
                if data.is_empty() || data.len() % 3 != 0 || data.len() > 3 * 256 {
                    return Err(invalid(at, "PLTE length is not 3 to 768 bytes in whole entries"));
                }
                p.palette = data;
            }
            b"tRNS" if stage == 0 && p.trns.is_none() => match header.color_type {
                ColorType::Indexed => {
                    if p.palette.is_empty() {
                        return Err(invalid(at, "tRNS chunk before PLTE"));
                    }
                    // Entries beyond the palette's are ignored, as libpng does.
                    p.trns = data.get(..data.len().min(p.palette.len() / 3));
                }
                ColorType::Grey if data.len() == 2 => p.trns = Some(data),
                ColorType::Rgb if data.len() == 6 => p.trns = Some(data),
                _ => {}
            },
            b"sRGB" if stage == 0 => p.srgb_intent = data.first().copied().filter(|_| data.len() == 1),
            b"gAMA" if stage == 0 => {
                p.gamma = data
                    .first_chunk::<4>()
                    .filter(|_| data.len() == 4)
                    .map(|g| u32::from_be_bytes(*g))
                    .filter(|&g| g != 0);
            }
            b"cHRM" if stage == 0 && data.len() == 32 => {
                let mut values = [0_u32; 8];
                for (v, b) in values.iter_mut().zip(data.as_chunks::<4>().0) {
                    *v = u32::from_be_bytes(*b);
                }
                p.chromaticities = Some(values);
            }
            b"iCCP" if stage == 0 => p.has_icc_profile = true,
            b"tEXt" if stage == 0 => {
                if let Some(nul) = data.iter().position(|&b| b == 0)
                    && &data[..nul] == COLOR_SPACE_KEYWORD.as_bytes()
                {
                    let value = &data[nul + 1..];
                    p.tagged = [ColorSpace::Srgb, ColorSpace::Linear, ColorSpace::Normal, ColorSpace::Data]
                        .into_iter()
                        .find(|&s| color_space_name(s).as_bytes() == value);
                }
            }
            kind if kind[0].is_ascii_uppercase() => {
                return Err(Error::Unsupported { format: FORMAT, what: "unknown critical chunk" });
            }
            _ => {}
        }
    }
    if p.idat.is_empty() {
        return Err(invalid(chunks.pos, "no IDAT chunk"));
    }
    Ok(p)
}

/// The [`Info`] of parsed chunks.
fn info(p: &Parsed<'_>) -> Info {
    let h = &p.header;
    let has_transparency = p.trns.is_some();
    let channels = match h.color_type {
        ColorType::Grey => 1 + u8::from(has_transparency),
        ColorType::GreyAlpha => 2,
        ColorType::Rgb | ColorType::Indexed => 3 + u8::from(has_transparency),
        ColorType::Rgba => 4,
    };
    let color_space = p.tagged.unwrap_or(if p.srgb_intent.is_some() || p.has_icc_profile {
        ColorSpace::Srgb
    } else if p.gamma.is_some_and(|g| LINEAR_GAMMA.contains(&g)) {
        ColorSpace::Linear
    } else {
        ColorSpace::Srgb
    });
    Info {
        width: h.width,
        height: h.height,
        bit_depth: h.depth,
        color_type: h.color_type,
        interlaced: h.interlaced,
        has_transparency,
        channels,
        color_space,
        srgb_intent: p.srgb_intent,
        gamma: p.gamma,
        chromaticities: p.chromaticities,
        has_icc_profile: p.has_icc_profile,
    }
}

/// Reads a PNG's header and colour chunks (everything before the image data) without decoding it.
pub fn read_info(bytes: &[u8], limits: &Limits) -> Result<Info> {
    let p = parse(bytes, true)?;
    limits.check_image(FORMAT, u64::from(p.header.width), u64::from(p.header.height))?;
    Ok(info(&p))
}

/// Decodes a PNG (see the module documentation for the layout of the result).
pub fn decode(bytes: &[u8], limits: &Limits) -> Result<Image> {
    decode_with_info(bytes, limits).map(|(_, image)| image)
}

/// Decodes a PNG and reports its header and colour chunks.
pub fn decode_with_info(bytes: &[u8], limits: &Limits) -> Result<(Info, Image)> {
    let p = parse(bytes, false)?;
    let info = info(&p);
    let h = p.header;
    let pixels = limits.check_image(FORMAT, u64::from(h.width), u64::from(h.height))?;
    let channels = usize::from(info.channels);
    let sample_count = pixels.checked_mul(channels).ok_or(Error::Limit {
        format: FORMAT,
        what: "decoded image size",
        limit: limits.max_bytes,
        got: u64::MAX,
    })?;
    let sample_size: u64 = if h.depth == 16 { 2 } else { 1 };
    limits.check_bytes(FORMAT, "decoded image size", sample_count as u64 * sample_size)?;
    let raw_len = limits.check_bytes(FORMAT, "decompressed image data", h.raw_size())?;

    // The zlib stream, contiguous: borrowed when it sits in one IDAT, else joined.
    let joined;
    let stream: &[u8] = if let [single] = p.idat.as_slice() {
        single
    } else {
        joined = p.idat.concat();
        &joined
    };
    let mut raw = vec![0_u8; raw_len];
    match zlib_decompress_into(stream, &mut raw) {
        Ok(d) if d.written == raw_len => {}
        Ok(_) => return Err(invalid(p.idat_offset, "image data is shorter than the image")),
        Err(Error::Limit { .. }) => return Err(invalid(p.idat_offset, "image data is longer than the image")),
        Err(e) => return Err(e),
    }

    let conv = Converter::new(&p);
    let samples = if h.depth == 16 {
        let mut out = vec![0_u16; sample_count];
        reconstruct(&h, &mut raw, &mut out, channels, p.idat_offset, |src, width, dst| conv.row_u16(src, width, dst))?;
        Samples::U16(out)
    } else {
        let mut out = vec![0_u8; sample_count];
        reconstruct(&h, &mut raw, &mut out, channels, p.idat_offset, |src, width, dst| conv.row_u8(src, width, dst))?;
        Samples::U8(out)
    };
    let image = Image::new(FORMAT, h.width, h.height, info.channels, info.color_space, samples)?;
    Ok((info, image))
}

/// Unfilters the decompressed rows in place and converts them into `out` (`channels` samples per
/// pixel), pass by pass for an interlaced image. `convert(src, width, dst)` turns one unfiltered row
/// of `width` pixels into `dst`, or returns `false` for a bad palette index.
fn reconstruct<T: Copy + Default>(
    h: &Header,
    raw: &mut [u8],
    out: &mut [T],
    channels: usize,
    offset: usize,
    convert: impl Fn(&[u8], usize, &mut [T]) -> bool,
) -> Result<()> {
    let (width, height) = (h.width as usize, h.height as usize);
    let bpp = h.filter_bpp();
    let out_row = width * channels;
    let zero = vec![0_u8; h.stride(width)];
    let passes: &[(usize, usize, usize, usize)] = if h.interlaced { &ADAM7 } else { &[(0, 0, 1, 1)] };
    // One converted row of a pass, scattered from here into the image.
    let mut pass_row = if h.interlaced { vec![T::default(); out_row] } else { Vec::new() };
    let mut start = 0;
    for &(x0, y0, dx, dy) in passes {
        let pw = width.saturating_sub(x0).div_ceil(dx);
        let ph = height.saturating_sub(y0).div_ceil(dy);
        if pw == 0 || ph == 0 {
            continue;
        }
        let stride = h.stride(pw);
        let end = start + ph * (stride + 1);
        let pass = raw.get_mut(start..end).ok_or_else(|| invalid(offset, "image data is shorter than the image"))?;
        for py in 0..ph {
            let (done, rest) = pass.split_at_mut(py * (stride + 1));
            let prev = if py == 0 { &zero[..stride] } else { &done[done.len() - stride..] };
            let Some((&mut kind, row)) = rest.get_mut(..=stride).and_then(<[u8]>::split_first_mut) else {
                return Err(invalid(offset, "image data is shorter than the image"));
            };
            if !unfilter(kind, row, prev, bpp) {
                return Err(invalid(offset, "unknown filter type"));
            }
            let y = y0 + py * dy;
            let ok = if h.interlaced {
                let dst = &mut pass_row[..pw * channels];
                let ok = convert(row, pw, dst);
                let image_row = &mut out[y * out_row..(y + 1) * out_row];
                for (px, src) in dst.chunks_exact(channels).enumerate() {
                    let x = x0 + px * dx;
                    image_row[x * channels..(x + 1) * channels].copy_from_slice(src);
                }
                ok
            } else {
                convert(row, width, &mut out[y * out_row..(y + 1) * out_row])
            };
            if !ok {
                return Err(invalid(offset, "palette index out of range"));
            }
        }
        start = end;
    }
    Ok(())
}

/// The unpacked samples of a row of 1-, 2-, 4- or 8-bit samples, high bits first.
fn unpack(src: &[u8], depth: u8) -> impl Iterator<Item = u8> + '_ {
    let per = 8 / depth;
    let mask = ((1_u16 << depth) - 1) as u8;
    src.iter().flat_map(move |&byte| (0..per).map(move |j| (byte >> (8 - depth * (j + 1))) & mask))
}

/// Turns unfiltered rows into output samples.
struct Converter {
    color_type: ColorType,
    depth: u8,
    /// The grey or RGB key colour of tRNS.
    key: Option<[u16; 3]>,
    /// The palette as RGBA, alpha from tRNS (255 where it has no entry).
    palette: [[u8; 4]; 256],
    palette_len: usize,
}

impl Converter {
    fn new(p: &Parsed<'_>) -> Self {
        let mut palette = [[0, 0, 0, 255]; 256];
        let entries = p.palette.as_chunks::<3>().0;
        for (slot, rgb) in palette.iter_mut().zip(entries) {
            *slot = [rgb[0], rgb[1], rgb[2], 255];
        }
        let mut key = None;
        if let Some(trns) = p.trns {
            if p.header.color_type == ColorType::Indexed {
                for (slot, &alpha) in palette.iter_mut().zip(trns) {
                    slot[3] = alpha;
                }
            } else {
                let mut k = [0_u16; 3];
                for (v, b) in k.iter_mut().zip(trns.as_chunks::<2>().0) {
                    *v = u16::from_be_bytes(*b);
                }
                key = Some(k);
            }
        }
        Self { color_type: p.header.color_type, depth: p.header.depth, key, palette, palette_len: entries.len() }
    }

    /// Converts a row of a 1- to 8-bit image.
    fn row_u8(&self, src: &[u8], width: usize, dst: &mut [u8]) -> bool {
        let depth = self.depth;
        match (self.color_type, self.key) {
            (ColorType::Grey, None) => {
                if depth == 8 {
                    dst.copy_from_slice(&src[..width]);
                } else {
                    let scale = 255 / ((1_u8 << depth) - 1);
                    for (d, s) in dst.iter_mut().zip(unpack(src, depth)) {
                        *d = s * scale;
                    }
                }
            }
            (ColorType::Grey, Some([k, ..])) => {
                let scale = if depth == 8 { 1 } else { 255 / ((1_u8 << depth) - 1) };
                for (d, s) in dst.as_chunks_mut::<2>().0.iter_mut().zip(unpack(src, depth)) {
                    *d = [s * scale, if u16::from(s) == k { 0 } else { 255 }];
                }
            }
            (ColorType::Rgb, Some(k)) => {
                for (d, s) in dst.as_chunks_mut::<4>().0.iter_mut().zip(src.as_chunks::<3>().0) {
                    let opaque = [u16::from(s[0]), u16::from(s[1]), u16::from(s[2])] != k;
                    *d = [s[0], s[1], s[2], if opaque { 255 } else { 0 }];
                }
            }
            (ColorType::Indexed, _) => {
                // With tRNS the output is RGBA, else RGB.
                let len = self.palette_len;
                let mut ok = true;
                if dst.len() == width * 4 {
                    for (d, i) in dst.as_chunks_mut::<4>().0.iter_mut().zip(unpack(src, depth)) {
                        ok &= usize::from(i) < len;
                        *d = self.palette[usize::from(i)];
                    }
                } else {
                    for (d, i) in dst.as_chunks_mut::<3>().0.iter_mut().zip(unpack(src, depth)) {
                        ok &= usize::from(i) < len;
                        let [r, g, b, _] = self.palette[usize::from(i)];
                        *d = [r, g, b];
                    }
                }
                return ok;
            }
            // RGB without tRNS, grey + alpha and RGBA are stored as they are decoded.
            _ => dst.copy_from_slice(&src[..dst.len()]),
        }
        true
    }

    /// Converts a row of a 16-bit image.
    fn row_u16(&self, src: &[u8], _width: usize, dst: &mut [u16]) -> bool {
        let words = src.as_chunks::<2>().0;
        match (self.color_type, self.key) {
            (ColorType::Grey, Some([k, ..])) => {
                for (d, s) in dst.as_chunks_mut::<2>().0.iter_mut().zip(words) {
                    let v = u16::from_be_bytes(*s);
                    *d = [v, if v == k { 0 } else { u16::MAX }];
                }
            }
            (ColorType::Rgb, Some(k)) => {
                for (d, s) in dst.as_chunks_mut::<4>().0.iter_mut().zip(src.as_chunks::<6>().0) {
                    let rgb = [
                        u16::from_be_bytes([s[0], s[1]]),
                        u16::from_be_bytes([s[2], s[3]]),
                        u16::from_be_bytes([s[4], s[5]]),
                    ];
                    *d = [rgb[0], rgb[1], rgb[2], if rgb == k { 0 } else { u16::MAX }];
                }
            }
            _ => {
                for (d, s) in dst.iter_mut().zip(words) {
                    *d = u16::from_be_bytes(*s);
                }
            }
        }
        true
    }
}
