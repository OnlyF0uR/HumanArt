//! DEFLATE (RFC 1951) and its zlib wrapper (RFC 1950), written in-house `[IO-01]` `[DEP-05]`.
//!
//! Decoding is table-driven ([`inflate`] and friends): the PNG and OpenEXR readers sit on it, so it is
//! the hot path of texture loading. Every decoder entry point takes an output bound: a stream that
//! would grow past it fails with [`Error::Limit`] before the extra byte is written, so a
//! decompression bomb costs at most the bound `[ROB-02]`. Hostile streams `[ROB-01]` fail with
//! [`Error::Invalid`] or [`Error::Truncated`]; see `inflate.rs` for what is checked.
//!
//! Encoding ([`deflate`], [`zlib_compress`]) is LZ77 over hash chains with lazy matching, with each
//! block written as whichever of a dynamic Huffman code, the fixed code or stored bytes is smallest.
//! [`Level`] trades speed for size.
//!
//! The zlib functions check the header (method 8, window at most 32 KiB, header checksum) and the
//! Adler-32 of the output. A preset dictionary is [`Error::Unsupported`] (neither PNG nor OpenEXR uses
//! one). Bytes after a stream's end (or after the zlib trailer) are ignored; [`inflate_into`] and
//! friends report how many input bytes the stream took.

mod encode;
mod inflate;
mod tables;

pub use encode::Level;

use crate::error::{Error, Format, Result};
use humanart_core::hash::adler32;
use inflate::{Inflater, SliceSink, VecSink};

const FORMAT: Format = Format::Deflate;

/// What a decoder wrote into a caller's buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decoded {
    /// Bytes written to the start of the output buffer.
    pub written: usize,
    /// Bytes of input the stream occupied (header and trailer included for zlib).
    pub consumed: usize,
}

/// Decompresses one raw DEFLATE stream into a new vector of at most `max_output` bytes.
pub fn inflate(input: &[u8], max_output: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    inflate_append(input, &mut out, max_output)?;
    Ok(out)
}

/// Decompresses one raw DEFLATE stream onto the end of `out`, adding at most `max_output` bytes;
/// returns how many input bytes the stream took. On error, `out` is cut back to its earlier length.
pub fn inflate_append(input: &[u8], out: &mut Vec<u8>, max_output: usize) -> Result<usize> {
    let start = out.len();
    // A first guess at the size (most data compresses 2-4x); the sink grows it as needed.
    let guess = input.len().saturating_mul(4).max(1024);
    let mut inflater = Inflater::new(input);
    let result = inflater.run(&mut VecSink::new(out, max_output, guess));
    match result {
        Ok(written) => {
            out.truncate(start + written);
            Ok(inflater.consumed())
        }
        Err(e) => {
            out.truncate(start);
            Err(e)
        }
    }
}

/// Decompresses one raw DEFLATE stream into `out`, which bounds the output size.
pub fn inflate_into(input: &[u8], out: &mut [u8]) -> Result<Decoded> {
    let mut inflater = Inflater::new(input);
    let written = inflater.run(&mut SliceSink::new(out))?;
    Ok(Decoded { written, consumed: inflater.consumed() })
}

/// Compresses `input` as one raw DEFLATE stream.
pub fn deflate(input: &[u8], level: Level) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() / 2 + 64);
    encode::compress(input, level, &mut out);
    out
}

/// Checks a zlib header and returns its length (always 2: dictionaries are refused).
fn zlib_header(input: &[u8]) -> Result<usize> {
    let [cmf, flg] = *input.first_chunk::<2>().ok_or(Error::Truncated { format: FORMAT, offset: input.len() })?;
    if cmf & 0x0F != 8 {
        return Err(Error::Invalid { format: FORMAT, offset: 0, what: "zlib compression method is not DEFLATE" });
    }
    if cmf >> 4 > 7 {
        return Err(Error::Invalid { format: FORMAT, offset: 0, what: "zlib window size over 32 KiB" });
    }
    if (u16::from(cmf) << 8 | u16::from(flg)) % 31 != 0 {
        return Err(Error::Invalid { format: FORMAT, offset: 1, what: "zlib header checksum" });
    }
    if flg & 0x20 != 0 {
        return Err(Error::Unsupported { format: FORMAT, what: "zlib preset dictionary" });
    }
    Ok(2)
}

/// Checks the Adler-32 trailer at `at` against `data`; returns the offset after it.
fn zlib_trailer(input: &[u8], at: usize, data: &[u8]) -> Result<usize> {
    let trailer = input
        .get(at..)
        .and_then(<[u8]>::first_chunk::<4>)
        .ok_or(Error::Truncated { format: FORMAT, offset: input.len() })?;
    if u32::from_be_bytes(*trailer) != adler32(data) {
        return Err(Error::Invalid { format: FORMAT, offset: at, what: "zlib Adler-32 checksum mismatch" });
    }
    Ok(at + 4)
}

/// Adds `base` to the offset of an error from the inner raw stream.
fn shift(e: Error, base: usize) -> Error {
    match e {
        Error::Truncated { format, offset } => Error::Truncated { format, offset: offset + base },
        Error::Invalid { format, offset, what } => Error::Invalid { format, offset: offset + base, what },
        other => other,
    }
}

/// Decompresses one zlib stream into a new vector of at most `max_output` bytes.
pub fn zlib_decompress(input: &[u8], max_output: usize) -> Result<Vec<u8>> {
    let header = zlib_header(input)?;
    let body = input.get(header..).unwrap_or_default();
    let mut out = Vec::new();
    let used = inflate_append(body, &mut out, max_output).map_err(|e| shift(e, header))?;
    zlib_trailer(input, header + used, &out)?;
    Ok(out)
}

/// Decompresses one zlib stream into `out`, which bounds the output size.
pub fn zlib_decompress_into(input: &[u8], out: &mut [u8]) -> Result<Decoded> {
    let header = zlib_header(input)?;
    let body = input.get(header..).unwrap_or_default();
    let inner = inflate_into(body, out).map_err(|e| shift(e, header))?;
    let data = out.get(..inner.written).unwrap_or_default();
    let consumed = zlib_trailer(input, header + inner.consumed, data)?;
    Ok(Decoded { written: inner.written, consumed })
}

/// Compresses `input` as one zlib stream (32 KiB window, no dictionary).
pub fn zlib_compress(input: &[u8], level: Level) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() / 2 + 64);
    // CMF: method 8, window 2^15. FLG: the level hint, then the check bits that make the pair a
    // multiple of 31.
    let cmf = 0x78_u8;
    let hint: u8 = match level {
        Level::Store => 0,
        Level::Fast => 1,
        Level::Default => 2,
        Level::Best => 3,
    };
    let flg_base = hint << 6;
    let check = 31 - ((u16::from(cmf) << 8 | u16::from(flg_base)) % 31);
    out.push(cmf);
    out.push(flg_base | (check % 31) as u8);
    encode::compress(input, level, &mut out);
    out.extend_from_slice(&adler32(input).to_be_bytes());
    out
}

#[cfg(test)]
mod tests;
