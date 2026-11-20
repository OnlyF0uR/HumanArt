//! The magic number, version field and attribute header of an OpenEXR file.

use super::{Attribute, ChannelInfo, Compression, Info, LineOrder, PixelType, Window};
use crate::bytes::{Reader, Writer};
use crate::error::{Error, Format, Result};
use crate::image::Limits;

const FORMAT: Format = Format::Exr;

/// The first four bytes of every OpenEXR file, as a little-endian `u32`.
pub(super) const MAGIC: u32 = 20_000_630;
/// The file format version this module reads and writes.
const VERSION: u32 = 2;
/// Version flag: the file is tiled.
const FLAG_TILED: u32 = 0x200;
/// Version flag: names may be up to 255 bytes long.
const FLAG_LONG_NAMES: u32 = 0x400;
/// Version flag: the file holds deep data.
const FLAG_DEEP: u32 = 0x800;
/// Version flag: the file has several parts.
const FLAG_MULTIPART: u32 = 0x1000;
/// The longest attribute, type or channel name, in bytes (with the long-names flag; files without
/// it are held to the same bound, not to 31).
pub const MAX_NAME_LEN: usize = 255;
/// The largest number of channels a file may have.
pub const MAX_CHANNELS: usize = 1024;
/// Names without the long-names flag are at most this long.
const SHORT_NAME_LEN: usize = 31;

/// An [`Error::Invalid`] of this module.
const fn invalid(offset: usize, what: &'static str) -> Error {
    Error::Invalid { format: FORMAT, offset, what }
}

/// Reads a NUL-terminated name of at most [`MAX_NAME_LEN`] bytes and consumes it with its NUL.
fn read_name<'a>(r: &mut Reader<'a>) -> Result<&'a [u8]> {
    let rest = r.rest();
    let window = &rest[..rest.len().min(MAX_NAME_LEN + 1)];
    let Some(end) = window.iter().position(|&b| b == 0) else {
        if rest.len() > MAX_NAME_LEN {
            return Err(r.invalid("name is longer than 255 bytes"));
        }
        return Err(Error::Truncated { format: FORMAT, offset: r.input().len() });
    };
    let name = r.bytes(end)?;
    r.skip(1)?;
    Ok(name)
}

/// A name as text.
fn name_text(bytes: &[u8], at: usize) -> Result<String> {
    std::str::from_utf8(bytes).map(str::to_owned).map_err(|_| invalid(at, "name is not UTF-8"))
}

/// Parses a `chlist` attribute value.
fn read_channels(value: &[u8], at: usize) -> Result<Vec<ChannelInfo>> {
    let mut r = Reader::new(value, FORMAT);
    let shift = |e: Error| match e {
        Error::Truncated { .. } => Error::Truncated { format: FORMAT, offset: at + value.len() },
        Error::Invalid { what, offset, .. } => invalid(at + offset, what),
        other => other,
    };
    let mut channels: Vec<ChannelInfo> = Vec::new();
    loop {
        let name_at = at + r.pos();
        let name = read_name(&mut r).map_err(shift)?;
        if name.is_empty() {
            break;
        }
        if channels.len() == MAX_CHANNELS {
            return Err(Error::Limit {
                format: FORMAT,
                what: "channel count",
                limit: MAX_CHANNELS as u64,
                got: MAX_CHANNELS as u64 + 1,
            });
        }
        let name = name_text(name, name_at)?;
        let pixel_type = match r.i32_le().map_err(shift)? {
            0 => PixelType::Uint,
            1 => PixelType::Half,
            2 => PixelType::Float,
            _ => return Err(invalid(name_at, "unknown channel pixel type")),
        };
        let linear = r.u8().map_err(shift)? != 0;
        r.skip(3).map_err(shift)?;
        let x_sampling = r.i32_le().map_err(shift)?;
        let y_sampling = r.i32_le().map_err(shift)?;
        if x_sampling < 1 || y_sampling < 1 {
            return Err(invalid(name_at, "channel sampling is not positive"));
        }
        if x_sampling != 1 || y_sampling != 1 {
            return Err(Error::Unsupported { format: FORMAT, what: "subsampled channels" });
        }
        if channels.iter().any(|c| c.name == name) {
            return Err(invalid(name_at, "duplicate channel name"));
        }
        channels.push(ChannelInfo { name, pixel_type, linear });
    }
    if channels.is_empty() {
        return Err(invalid(at, "the channel list is empty"));
    }
    Ok(channels)
}

/// Parses a `box2i` value.
fn read_window(value: &[u8]) -> Window {
    let word = |i: usize| i32::from_le_bytes([value[i], value[i + 1], value[i + 2], value[i + 3]]);
    Window { x_min: word(0), y_min: word(4), x_max: word(8), y_max: word(12) }
}

/// A little-endian `f32` at `i` of an attribute value whose size was checked.
fn f32_at(value: &[u8], i: usize) -> f32 {
    f32::from_le_bytes([value[i], value[i + 1], value[i + 2], value[i + 3]])
}

/// Reads the magic number, the version field and the header, leaving `r` at the offset table.
///
/// The data window is checked against `limits`.
pub(super) fn read(r: &mut Reader<'_>, limits: &Limits) -> Result<Info> {
    if r.u32_le()? != MAGIC {
        return Err(invalid(0, "not an OpenEXR file (bad magic number)"));
    }
    let version = r.u32_le()?;
    if version & 0xFF != VERSION {
        return Err(Error::Unsupported { format: FORMAT, what: "file format versions other than 2" });
    }
    if version & FLAG_TILED != 0 {
        return Err(Error::Unsupported { format: FORMAT, what: "tiled images" });
    }
    if version & FLAG_DEEP != 0 {
        return Err(Error::Unsupported { format: FORMAT, what: "deep data" });
    }
    if version & FLAG_MULTIPART != 0 {
        return Err(Error::Unsupported { format: FORMAT, what: "multi-part files" });
    }
    if version & !(0xFF | FLAG_LONG_NAMES) != 0 {
        return Err(Error::Unsupported { format: FORMAT, what: "unknown version flags" });
    }

    let mut channels = None;
    let mut compression = None;
    let mut data_window = None;
    let mut display_window = None;
    let mut line_order = LineOrder::IncreasingY;
    let mut pixel_aspect_ratio = 1.0;
    let mut screen_window_center = [0.0; 2];
    let mut screen_window_width = 1.0;
    let mut attributes = Vec::new();
    let mut count = 0_u64;
    loop {
        let at = r.pos();
        let name = read_name(r)?;
        if name.is_empty() {
            break;
        }
        count += 1;
        limits.check_elements(FORMAT, "attribute count", count)?;
        let type_name = read_name(r)?;
        let size_at = r.pos();
        let Ok(size) = usize::try_from(r.i32_le()?) else {
            return Err(invalid(size_at, "negative attribute size"));
        };
        let value_at = r.pos();
        let value = r.bytes(size)?;
        let wrong = || invalid(at, "attribute has the wrong type or size");
        let expect = |ty: &[u8], len: usize| if type_name == ty && size == len { Ok(()) } else { Err(wrong()) };
        match name {
            b"channels" => {
                if type_name != b"chlist" {
                    return Err(wrong());
                }
                channels = Some(read_channels(value, value_at)?);
            }
            b"compression" => {
                expect(b"compression", 1)?;
                compression = Some(Compression::from_byte(value[0], value_at)?);
            }
            b"dataWindow" => {
                expect(b"box2i", 16)?;
                data_window = Some(read_window(value));
            }
            b"displayWindow" => {
                expect(b"box2i", 16)?;
                display_window = Some(read_window(value));
            }
            b"lineOrder" => {
                expect(b"lineOrder", 1)?;
                line_order = match value[0] {
                    0 => LineOrder::IncreasingY,
                    1 => LineOrder::DecreasingY,
                    2 => LineOrder::RandomY,
                    _ => return Err(invalid(value_at, "unknown line order")),
                };
            }
            b"pixelAspectRatio" => {
                expect(b"float", 4)?;
                pixel_aspect_ratio = f32_at(value, 0);
            }
            b"screenWindowCenter" => {
                expect(b"v2f", 8)?;
                screen_window_center = [f32_at(value, 0), f32_at(value, 4)];
            }
            b"screenWindowWidth" => {
                expect(b"float", 4)?;
                screen_window_width = f32_at(value, 0);
            }
            b"type" if value != b"scanlineimage" => {
                return Err(Error::Unsupported { format: FORMAT, what: "parts other than scan-line images" });
            }
            _ => attributes.push(Attribute {
                name: name_text(name, at)?,
                type_name: name_text(type_name, at)?,
                value: value.to_vec(),
            }),
        }
    }

    let Some(channels) = channels else {
        return Err(invalid(r.pos(), "the channels attribute is missing"));
    };
    let Some(compression) = compression else {
        return Err(invalid(r.pos(), "the compression attribute is missing"));
    };
    let Some(data_window) = data_window else {
        return Err(invalid(r.pos(), "the dataWindow attribute is missing"));
    };
    let (Some(width), Some(height)) = (data_window.width(), data_window.height()) else {
        return Err(invalid(r.pos(), "the data window is empty"));
    };
    limits.check_image(FORMAT, width, height)?;
    Ok(Info {
        data_window,
        display_window: display_window.unwrap_or(data_window),
        compression,
        line_order,
        pixel_aspect_ratio,
        screen_window_center,
        screen_window_width,
        channels,
        attributes,
    })
}

/// Appends one attribute.
fn attribute(w: &mut Writer, name: &str, type_name: &str, value: &[u8]) {
    w.bytes(name.as_bytes());
    w.u8(0);
    w.bytes(type_name.as_bytes());
    w.u8(0);
    w.i32_le(value.len() as i32);
    w.bytes(value);
}

/// Appends the magic number, the version field and a single-part scan-line header for a
/// `width × height` image with its origin at (0, 0). `channels` must be sorted by name, unique,
/// and have names of 1 to [`MAX_NAME_LEN`] bytes without NUL (the caller checks).
pub(super) fn write(w: &mut Writer, width: u32, height: u32, channels: &[ChannelInfo], compression: Compression) {
    let long_names = channels.iter().any(|c| c.name.len() > SHORT_NAME_LEN);
    w.u32_le(MAGIC);
    w.u32_le(VERSION | if long_names { FLAG_LONG_NAMES } else { 0 });

    let mut list = Writer::new();
    for channel in channels {
        list.bytes(channel.name.as_bytes());
        list.u8(0);
        list.i32_le(match channel.pixel_type {
            PixelType::Uint => 0,
            PixelType::Half => 1,
            PixelType::Float => 2,
        });
        list.bytes(&[u8::from(channel.linear), 0, 0, 0]);
        list.i32_le(1);
        list.i32_le(1);
    }
    list.u8(0);
    attribute(w, "channels", "chlist", list.as_slice());
    attribute(w, "compression", "compression", &[compression.byte()]);
    let mut window = Writer::new();
    for v in [0, 0, width as i32 - 1, height as i32 - 1] {
        window.i32_le(v);
    }
    attribute(w, "dataWindow", "box2i", window.as_slice());
    attribute(w, "displayWindow", "box2i", window.as_slice());
    attribute(w, "lineOrder", "lineOrder", &[0]);
    attribute(w, "pixelAspectRatio", "float", &1.0_f32.to_le_bytes());
    attribute(w, "screenWindowCenter", "v2f", &[0; 8]);
    attribute(w, "screenWindowWidth", "float", &1.0_f32.to_le_bytes());
    w.u8(0);
}
