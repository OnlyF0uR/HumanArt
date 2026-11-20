//! OpenEXR images: reader and writer `[IO-03]` `[DEP-05]`, written from the published file layout
//! (no OpenEXR code is used).
//!
//! EXR holds HDRI panoramas for image-based lighting `[LIT-01]` and the multi-channel AOV and
//! displacement exports of the studio.
//!
//! # What is read
//!
//! - Single-part scan-line files (version 2, short or long names).
//! - NONE, RLE, ZIPS, ZIP and PIZ compression, and chunks stored uncompressed because compression did
//!   not pay.
//! - `uint`, `half` and `float` channels, any number of them (up to [`MAX_CHANNELS`]), with any
//!   names: [`decode_channels`] returns them all; [`decode`] maps `R`, `G`, `B` (+ `A`), or `Y`
//!   (+ `A`), or a lone channel, to an [`Image`].
//! - Any data window (the decoded image covers it; its origin is in [`Info::data_window`]), every
//!   line order (chunks are placed by their line number).
//! - Unknown attributes are kept as bytes in [`Info::attributes`]; `channels`, `compression` and
//!   `dataWindow` are required, the other standard ones get their default when missing.
//!
//! # What is not
//!
//! Tiled, multi-part and deep files, subsampled channels (sampling other than 1) and the PXR24, B44,
//! B44A, DWAA and DWAB compressions are [`Error::Unsupported`]. The spec asks for scan-line
//! reading and writing only.
//!
//! # What is written
//!
//! Single-part scan-line files with the data and display window at the origin, increasing line
//! order, any of the five compressions (ZIP by default), `half`, `float` and `uint` channels sorted by
//! name as the format requires. A chunk is stored uncompressed when compression does not make it
//! smaller. Output is deterministic.
//!
//! # Limits `[ROB-01]` `[ROB-02]`
//!
//! The data window is checked with [`Limits::check_image`], the decoded size and one chunk's
//! uncompressed size with [`Limits::check_bytes`], the attribute count with
//! [`Limits::check_elements`], names are at most [`MAX_NAME_LEN`] bytes. The offset table and every
//! chunk header and size are checked against the input before anything is read from them, and every
//! decompressor must fill its chunk exactly.

mod codec;
mod header;
mod huf;
mod piz;

pub use header::{MAX_CHANNELS, MAX_NAME_LEN};

use crate::bytes::{Reader, Writer};
use crate::deflate::Level;
use crate::error::{Error, Format, Result};
use crate::image::{ColorSpace, Image, Limits, Samples};
use humanart_math::{f16_bits_to_f32, f32_to_f16_bits};

/// The format every error of this module names.
const FORMAT: Format = Format::Exr;

/// How the chunks of a file are compressed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Compression {
    /// Uncompressed, one line per chunk.
    None,
    /// Run-length coding of predicted bytes, one line per chunk.
    Rle,
    /// zlib over predicted bytes, one line per chunk.
    Zips,
    /// zlib over predicted bytes, 16 lines per chunk.
    #[default]
    Zip,
    /// Wavelet and Huffman coding, 32 lines per chunk. Lossless.
    Piz,
}

impl Compression {
    /// The number of lines in one chunk.
    #[must_use]
    pub const fn lines_per_chunk(self) -> usize {
        match self {
            Self::None | Self::Rle | Self::Zips => 1,
            Self::Zip => 16,
            Self::Piz => 32,
        }
    }

    /// The code of the `compression` attribute.
    const fn byte(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Rle => 1,
            Self::Zips => 2,
            Self::Zip => 3,
            Self::Piz => 4,
        }
    }

    /// Reads the code of the `compression` attribute found at `at`.
    fn from_byte(code: u8, at: usize) -> Result<Self> {
        let unsupported = |what| Err(Error::Unsupported { format: FORMAT, what });
        match code {
            0 => Ok(Self::None),
            1 => Ok(Self::Rle),
            2 => Ok(Self::Zips),
            3 => Ok(Self::Zip),
            4 => Ok(Self::Piz),
            5 => unsupported("PXR24 compression"),
            6 => unsupported("B44 compression"),
            7 => unsupported("B44A compression"),
            8 => unsupported("DWAA compression"),
            9 => unsupported("DWAB compression"),
            _ => Err(Error::Invalid { format: FORMAT, offset: at, what: "unknown compression" }),
        }
    }
}

/// The sample type of a channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PixelType {
    /// 32-bit unsigned integer (ids, masks).
    Uint,
    /// 16-bit IEEE half float.
    Half,
    /// 32-bit IEEE float.
    Float,
}

impl PixelType {
    /// Bytes per sample.
    #[must_use]
    pub const fn size(self) -> usize {
        match self {
            Self::Half => 2,
            Self::Uint | Self::Float => 4,
        }
    }
}

/// The order in which a file stores its chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LineOrder {
    /// Top to bottom.
    IncreasingY,
    /// Bottom to top.
    DecreasingY,
    /// Any order.
    RandomY,
}

/// An inclusive pixel rectangle (`box2i`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Window {
    /// Left column.
    pub x_min: i32,
    /// Top line.
    pub y_min: i32,
    /// Right column (inclusive).
    pub x_max: i32,
    /// Bottom line (inclusive).
    pub y_max: i32,
}

impl Window {
    /// The width in pixels, `None` if the window is empty.
    #[must_use]
    pub fn width(self) -> Option<u64> {
        let w = i64::from(self.x_max) - i64::from(self.x_min) + 1;
        u64::try_from(w).ok().filter(|&w| w > 0)
    }

    /// The height in pixels, `None` if the window is empty.
    #[must_use]
    pub fn height(self) -> Option<u64> {
        let h = i64::from(self.y_max) - i64::from(self.y_min) + 1;
        u64::try_from(h).ok().filter(|&h| h > 0)
    }
}

/// One entry of a file's channel list.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ChannelInfo {
    /// The channel name (`R`, `diffuse.G`, `Z`, ...).
    pub name: String,
    /// The sample type.
    pub pixel_type: PixelType,
    /// The `pLinear` hint: the values are perceptually linear.
    pub linear: bool,
}

/// An attribute this module does not interpret, kept as it was stored.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Attribute {
    /// The attribute name.
    pub name: String,
    /// The attribute type (`string`, `float`, `m44f`, ...).
    pub type_name: String,
    /// The value bytes (little-endian, as the format defines them per type).
    pub value: Vec<u8>,
}

/// What the header of a scan-line OpenEXR file says.
#[derive(Clone, Debug, PartialEq)]
pub struct Info {
    /// The pixels the file stores.
    pub data_window: Window,
    /// The full picture (defaults to the data window).
    pub display_window: Window,
    /// The chunk compression.
    pub compression: Compression,
    /// The order the chunks were written in.
    pub line_order: LineOrder,
    /// Pixel width over height (default 1).
    pub pixel_aspect_ratio: f32,
    /// `screenWindowCenter` (default 0, 0).
    pub screen_window_center: [f32; 2],
    /// `screenWindowWidth` (default 1).
    pub screen_window_width: f32,
    /// The channels, in file order.
    pub channels: Vec<ChannelInfo>,
    /// Every other attribute, in file order.
    pub attributes: Vec<Attribute>,
}

impl Info {
    /// Width of the data window in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        // The reader checked the window against the image limits, which fit u32.
        self.data_window.width().map_or(0, |w| w as u32)
    }

    /// Height of the data window in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.data_window.height().map_or(0, |h| h as u32)
    }
}

/// The samples of one channel, one per pixel, rows top to bottom.
#[derive(Clone, Debug, PartialEq)]
pub enum ChannelData {
    /// 32-bit unsigned integers.
    Uint(Vec<u32>),
    /// Half floats, as their bit patterns.
    Half(Vec<u16>),
    /// 32-bit floats.
    Float(Vec<f32>),
}

impl ChannelData {
    /// The number of samples.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Uint(v) => v.len(),
            Self::Half(v) => v.len(),
            Self::Float(v) => v.len(),
        }
    }

    /// Whether there are no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The sample type.
    #[must_use]
    pub const fn pixel_type(&self) -> PixelType {
        match self {
            Self::Uint(_) => PixelType::Uint,
            Self::Half(_) => PixelType::Half,
            Self::Float(_) => PixelType::Float,
        }
    }

    /// Sample `i` as `f32` (a `uint` above 2^24 loses precision). `i` must be in range.
    fn f32_at(&self, i: usize) -> f32 {
        match self {
            Self::Uint(v) => v[i] as f32,
            Self::Half(v) => f16_bits_to_f32(v[i]),
            Self::Float(v) => v[i],
        }
    }
}

/// One named channel of a [`ChannelImage`].
#[derive(Clone, Debug, PartialEq)]
pub struct Channel {
    /// The channel name.
    pub name: String,
    /// The `pLinear` hint.
    pub linear: bool,
    /// The samples.
    pub data: ChannelData,
}

/// An image with any number of named channels of any sample type (AOV layers, ids, depth).
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The channels, each `width × height` samples.
    pub channels: Vec<Channel>,
}

impl ChannelImage {
    /// The channel called `name`.
    #[must_use]
    pub fn channel(&self, name: &str) -> Option<&Channel> {
        self.channels.iter().find(|c| c.name == name)
    }

    /// Splits an [`Image`] into channels: 1 = `Y`, 2 = `Y`, `A`, 3 = `R`, `G`, `B`, 4 = `R`, `G`,
    /// `B`, `A`. `F16` samples become `half` channels and `F32` ones `float` channels.
    ///
    /// 8- and 16-bit integer images and sRGB-encoded images are refused: EXR holds linear values.
    pub fn from_image(image: &Image) -> Result<Self> {
        let refuse = |what| Err(Error::Encode { format: FORMAT, what });
        if image.color_space == ColorSpace::Srgb {
            return refuse("sRGB-encoded samples must be linearised before they are written to EXR");
        }
        let names: &[&str] = match image.channels {
            1 => &["Y"],
            2 => &["Y", "A"],
            3 => &["R", "G", "B"],
            4 => &["R", "G", "B", "A"],
            _ => return refuse("channel count must be 1 to 4"),
        };
        let n = names.len();
        if Some(image.samples.len()) != image.pixel_count().checked_mul(n) {
            return refuse("sample count does not match the dimensions");
        }
        let channels = names
            .iter()
            .enumerate()
            .map(|(k, name)| {
                let data = match &image.samples {
                    Samples::F16(v) => Ok(ChannelData::Half(v.iter().skip(k).step_by(n).copied().collect())),
                    Samples::F32(v) => Ok(ChannelData::Float(v.iter().skip(k).step_by(n).copied().collect())),
                    Samples::U8(_) | Samples::U16(_) => {
                        Err(Error::Encode { format: FORMAT, what: "samples must be f16 or f32" })
                    }
                }?;
                Ok(Channel { name: (*name).to_owned(), linear: false, data })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { width: image.width, height: image.height, channels })
    }

    /// Joins channels into an [`Image`]: `R`, `G`, `B` (+ `A`) if present, else `Y` (+ `A`), else
    /// the only channel (as [`ColorSpace::Data`]). Colour is [`ColorSpace::Linear`].
    ///
    /// The samples are `F16` if every chosen channel is `half`, `F32` otherwise (`uint` values above
    /// 2^24 lose precision). Other layouts are [`Error::Unsupported`]: read them channel by channel.
    pub fn to_image(&self) -> Result<Image> {
        let find = |name: &str| self.channels.iter().position(|c| c.name == name);
        let alpha = find("A");
        let (mut picks, color_space) = if let (Some(r), Some(g), Some(b)) = (find("R"), find("G"), find("B")) {
            (vec![r, g, b], ColorSpace::Linear)
        } else if let Some(y) = find("Y") {
            (vec![y], ColorSpace::Linear)
        } else if self.channels.len() == 1 {
            (vec![0], ColorSpace::Data)
        } else {
            return Err(Error::Unsupported {
                format: FORMAT,
                what: "channel layouts without R, G, B or Y as an image (use decode_channels)",
            });
        };
        if color_space == ColorSpace::Linear {
            picks.extend(alpha);
        }
        let pixels = (self.width as usize).saturating_mul(self.height as usize);
        let chosen: Vec<&ChannelData> = picks.iter().map(|&i| &self.channels[i].data).collect();
        if chosen.iter().any(|d| d.len() != pixels) {
            return Err(Error::Encode { format: FORMAT, what: "channel length does not match the dimensions" });
        }
        let samples = if chosen.iter().all(|d| d.pixel_type() == PixelType::Half) {
            let halves: Vec<&[u16]> = chosen
                .iter()
                .filter_map(|d| if let ChannelData::Half(v) = d { Some(v.as_slice()) } else { None })
                .collect();
            Samples::F16((0..pixels).flat_map(|p| halves.iter().map(move |v| v[p])).collect())
        } else {
            Samples::F32((0..pixels).flat_map(|p| chosen.iter().map(move |d| d.f32_at(p))).collect())
        };
        Image::new(FORMAT, self.width, self.height, picks.len() as u8, color_space, samples)
    }
}

/// How [`encode_with`] and [`encode_channels`] write a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EncodeOptions {
    /// The chunk compression.
    pub compression: Compression,
    /// The type `half` and `float` channels are written as (`None`: as they are). Floats become
    /// halves with round-to-nearest-even; `uint` channels stay `uint`; `Some(PixelType::Uint)` is
    /// refused.
    pub pixel_type: Option<PixelType>,
    /// The deflate effort for ZIP and ZIPS.
    pub level: Level,
}

impl Default for EncodeOptions {
    /// ZIP, types as they are, the default deflate level.
    fn default() -> Self {
        Self { compression: Compression::Zip, pixel_type: None, level: Level::Default }
    }
}

/// Reads the header of an OpenEXR file without decoding its pixels; the data window is checked
/// against `limits`.
pub fn read_info(bytes: &[u8], limits: &Limits) -> Result<Info> {
    header::read(&mut Reader::new(bytes, FORMAT), limits)
}

/// Decodes a scan-line OpenEXR file to an [`Image`] (see [`ChannelImage::to_image`] for the channel
/// mapping), first row at the top of the data window.
pub fn decode(bytes: &[u8], limits: &Limits) -> Result<Image> {
    decode_with_info(bytes, limits).map(|(_, image)| image)
}

/// [`decode`], also returning the header.
pub fn decode_with_info(bytes: &[u8], limits: &Limits) -> Result<(Info, Image)> {
    let (info, channels) = decode_channels(bytes, limits)?;
    let image = channels.to_image()?;
    Ok((info, image))
}

/// Decodes every channel of a scan-line OpenEXR file, in file order.
pub fn decode_channels(bytes: &[u8], limits: &Limits) -> Result<(Info, ChannelImage)> {
    let mut r = Reader::new(bytes, FORMAT);
    let info = header::read(&mut r, limits)?;
    let (width, height) = (info.width() as usize, info.height() as usize);
    let pixels = limits.check_image(FORMAT, width as u64, height as u64)?;
    let pixel_bytes: usize = info.channels.iter().map(|c| c.pixel_type.size()).sum();
    limits.check_bytes(FORMAT, "decoded size", pixels as u64 * pixel_bytes as u64)?;
    let lines_per_chunk = info.compression.lines_per_chunk();
    let line_bytes = width * pixel_bytes;
    let chunk_bytes = limits.check_bytes(FORMAT, "chunk size", (line_bytes * lines_per_chunk.min(height)) as u64)?;
    let chunk_count = height.div_ceil(lines_per_chunk);
    let table_len = r.check_count(chunk_count, 8)?;
    let table = r.bytes(table_len)?;
    // Every chunk has at least its 8-byte header after the table.
    r.check_count(chunk_count, 8)?;

    let words: Vec<usize> = info.channels.iter().map(|c| c.pixel_type.size() / 2).collect();
    let mut data: Vec<ChannelData> = info
        .channels
        .iter()
        .map(|c| match c.pixel_type {
            PixelType::Uint => ChannelData::Uint(vec![0; pixels]),
            PixelType::Half => ChannelData::Half(vec![0; pixels]),
            PixelType::Float => ChannelData::Float(vec![0.0; pixels]),
        })
        .collect();
    let mut raw = vec![0_u8; chunk_bytes];
    let mut scratch = codec::Scratch::new(chunk_bytes);
    for (index, entry) in table.as_chunks::<8>().0.iter().enumerate() {
        let entry_at = 8 + index * 8;
        let offset = usize::try_from(u64::from_le_bytes(*entry)).unwrap_or(usize::MAX);
        if offset > bytes.len() {
            return Err(Error::Invalid {
                format: FORMAT,
                offset: entry_at,
                what: "chunk offset is past the end of the file",
            });
        }
        let first_line = index * lines_per_chunk;
        let lines = lines_per_chunk.min(height - first_line);
        let mut c = Reader::new(bytes, FORMAT);
        c.seek(offset)?;
        let y = c.i32_le()?;
        if i64::from(y) != i64::from(info.data_window.y_min) + first_line as i64 {
            return Err(c.invalid("chunk has the wrong line number"));
        }
        let Ok(size) = usize::try_from(c.i32_le()?) else {
            return Err(c.invalid("negative chunk size"));
        };
        let data_at = c.pos();
        let chunk = c.bytes(size)?;
        let out = &mut raw[..lines * line_bytes];
        let layout = piz::Layout { width, lines, words: &words };
        codec::decompress(info.compression, chunk, data_at, &layout, out, &mut scratch)?;

        // Lines of channels of little-endian samples.
        let mut at = 0;
        for line in 0..lines {
            let row = (first_line + line) * width;
            for channel in &mut data {
                match channel {
                    ChannelData::Half(v) => {
                        for (dst, src) in v[row..row + width].iter_mut().zip(out[at..].as_chunks::<2>().0) {
                            *dst = u16::from_le_bytes(*src);
                        }
                        at += width * 2;
                    }
                    ChannelData::Float(v) => {
                        for (dst, src) in v[row..row + width].iter_mut().zip(out[at..].as_chunks::<4>().0) {
                            *dst = f32::from_le_bytes(*src);
                        }
                        at += width * 4;
                    }
                    ChannelData::Uint(v) => {
                        for (dst, src) in v[row..row + width].iter_mut().zip(out[at..].as_chunks::<4>().0) {
                            *dst = u32::from_le_bytes(*src);
                        }
                        at += width * 4;
                    }
                }
            }
        }
    }
    let channels = info
        .channels
        .iter()
        .zip(data)
        .map(|(c, data)| Channel { name: c.name.clone(), linear: c.linear, data })
        .collect();
    let image = ChannelImage { width: width as u32, height: height as u32, channels };
    Ok((info, image))
}

/// Encodes an [`Image`] with the default options (ZIP, sample types as stored); see
/// [`ChannelImage::from_image`] for what is accepted.
pub fn encode(image: &Image) -> Result<Vec<u8>> {
    encode_with(image, &EncodeOptions::default())
}

/// Encodes an [`Image`] with the given options.
pub fn encode_with(image: &Image, options: &EncodeOptions) -> Result<Vec<u8>> {
    encode_channels(&ChannelImage::from_image(image)?, options)
}

/// Encodes named channels as a single-part scan-line file; the channels are stored sorted by name.
///
/// Every channel must have `width × height` samples and a unique name of 1 to [`MAX_NAME_LEN`]
/// bytes without NUL; there must be 1 to [`MAX_CHANNELS`] of them, the image at least 1 × 1 and
/// smaller than 2^31 a side, and one uncompressed chunk at most 2 GiB.
pub fn encode_channels(image: &ChannelImage, options: &EncodeOptions) -> Result<Vec<u8>> {
    let refuse = |what| Err(Error::Encode { format: FORMAT, what });
    let (width, height) = (image.width as usize, image.height as usize);
    if width == 0 || height == 0 || image.width > i32::MAX as u32 || image.height > i32::MAX as u32 {
        return refuse("image dimensions are out of range");
    }
    if image.channels.is_empty() || image.channels.len() > MAX_CHANNELS {
        return refuse("channel count must be 1 to 1024");
    }
    if options.pixel_type == Some(PixelType::Uint) {
        return refuse("floating-point channels cannot be converted to uint");
    }
    let pixels = width.saturating_mul(height);
    let mut order: Vec<&Channel> = image.channels.iter().collect();
    order.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
    for (i, channel) in order.iter().enumerate() {
        let name = channel.name.as_bytes();
        if name.is_empty() || name.len() > MAX_NAME_LEN || name.contains(&0) {
            return refuse("channel names must be 1 to 255 bytes without NUL");
        }
        if i > 0 && order[i - 1].name == channel.name {
            return refuse("duplicate channel name");
        }
        if channel.data.len() != pixels {
            return refuse("channel length does not match the dimensions");
        }
    }
    let infos: Vec<ChannelInfo> = order
        .iter()
        .map(|c| {
            let stored = c.data.pixel_type();
            let pixel_type = match (stored, options.pixel_type) {
                (PixelType::Uint, _) | (_, None) => stored,
                (_, Some(target)) => target,
            };
            ChannelInfo { name: c.name.clone(), pixel_type, linear: c.linear }
        })
        .collect();
    let pixel_bytes: usize = infos.iter().map(|c| c.pixel_type.size()).sum();
    let line_bytes = width.saturating_mul(pixel_bytes);
    let lines_per_chunk = options.compression.lines_per_chunk();
    let chunk_bytes = line_bytes.saturating_mul(lines_per_chunk.min(height));
    if chunk_bytes > i32::MAX as usize {
        return refuse("one chunk would be larger than 2 GiB");
    }
    let chunk_count = height.div_ceil(lines_per_chunk);

    let mut w = Writer::with_capacity(pixels.saturating_mul(pixel_bytes) / 2 + 1024);
    header::write(&mut w, image.width, image.height, &infos, options.compression);
    let table_at = w.len();
    for _ in 0..chunk_count {
        w.u64_le(0);
    }
    let words: Vec<usize> = infos.iter().map(|c| c.pixel_type.size() / 2).collect();
    let (mut raw, mut packed, mut tmp) = (Vec::with_capacity(chunk_bytes), Vec::new(), Vec::new());
    for index in 0..chunk_count {
        let first_line = index * lines_per_chunk;
        let lines = lines_per_chunk.min(height - first_line);
        raw.clear();
        for line in 0..lines {
            let row = (first_line + line) * width;
            for (channel, info) in order.iter().zip(&infos) {
                let range = row..row + width;
                match (&channel.data, info.pixel_type) {
                    (ChannelData::Half(v), PixelType::Half) => {
                        for s in &v[range] {
                            raw.extend_from_slice(&s.to_le_bytes());
                        }
                    }
                    (ChannelData::Half(v), _) => {
                        for &s in &v[range] {
                            raw.extend_from_slice(&f16_bits_to_f32(s).to_le_bytes());
                        }
                    }
                    (ChannelData::Float(v), PixelType::Half) => {
                        for &s in &v[range] {
                            raw.extend_from_slice(&f32_to_f16_bits(s).to_le_bytes());
                        }
                    }
                    (ChannelData::Float(v), _) => {
                        for s in &v[range] {
                            raw.extend_from_slice(&s.to_le_bytes());
                        }
                    }
                    (ChannelData::Uint(v), _) => {
                        for s in &v[range] {
                            raw.extend_from_slice(&s.to_le_bytes());
                        }
                    }
                }
            }
        }
        let layout = piz::Layout { width, lines, words: &words };
        codec::compress(options.compression, options.level, &raw, &layout, &mut packed, &mut tmp)?;
        w.patch_u64_le(table_at + index * 8, w.len() as u64);
        w.i32_le(first_line as i32);
        w.i32_le(packed.len() as i32);
        w.bytes(&packed);
    }
    Ok(w.into_vec())
}

#[cfg(test)]
mod tests;
