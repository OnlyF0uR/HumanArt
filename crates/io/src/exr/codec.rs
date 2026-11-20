//! Chunk compression of scanline OpenEXR files: NONE, RLE, ZIPS, ZIP and PIZ.
//!
//! RLE and ZIP share a preprocessing step (the reference's `ImfZip.cpp` / `ImfRle.cpp`): the bytes
//! of a chunk are split into the even-indexed and the odd-indexed ones, and each byte is replaced by
//! its difference to the previous one plus 128. RLE then codes runs (a signed count byte: `-n`
//! introduces `n` literal bytes, `n ≥ 0` repeats the next byte `n + 1` times); ZIP and ZIPS deflate
//! the result as one zlib stream. A chunk whose compressed form would not be smaller is stored
//! uncompressed, and a reader recognises it by its size.

use super::Compression;
use super::piz;
use crate::deflate::{self, Level};
use crate::error::{Error, Format, Result};

const FORMAT: Format = Format::Exr;
/// The shortest run the RLE writer codes as a run.
const MIN_RUN: usize = 3;
/// The longest literal stretch one RLE count byte can introduce.
const MAX_LITERAL: usize = 127;
/// The longest run one RLE count byte can describe.
const MAX_RUN: usize = 128;

/// An [`Error::Invalid`] of this module.
const fn invalid(offset: usize, what: &'static str) -> Error {
    Error::Invalid { format: FORMAT, offset, what }
}

/// Undoes the byte predictor and the even/odd split: `tmp` (the decompressed bytes) becomes the
/// chunk's bytes in `out`.
fn unpredict_and_merge(tmp: &mut [u8], out: &mut [u8]) {
    for i in 1..tmp.len() {
        tmp[i] = tmp[i - 1].wrapping_add(tmp[i]).wrapping_sub(128);
    }
    let half = tmp.len().div_ceil(2);
    let (even, odd) = tmp.split_at(half);
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = if i % 2 == 0 { even[i / 2] } else { odd[i / 2] };
    }
}

/// The even/odd split and the byte predictor: the chunk's bytes in `raw` become `tmp`.
fn split_and_predict(raw: &[u8], tmp: &mut Vec<u8>) {
    tmp.clear();
    tmp.extend(raw.iter().step_by(2));
    tmp.extend(raw.iter().skip(1).step_by(2));
    let mut previous = tmp.first().copied().unwrap_or(0);
    for byte in tmp.iter_mut().skip(1) {
        let current = *byte;
        *byte = current.wrapping_sub(previous).wrapping_add(128);
        previous = current;
    }
}

/// Decodes RLE data, which must fill `out` exactly. `base` is the file offset of `input`.
fn rle_decode(input: &[u8], base: usize, out: &mut [u8]) -> Result<()> {
    let (mut i, mut o) = (0, 0);
    // Progress: every turn consumes at least one input byte.
    while i < input.len() {
        let count = input[i] as i8;
        i += 1;
        if count < 0 {
            let n = usize::from(count.unsigned_abs());
            let Some(literal) = input.get(i..i + n) else {
                return Err(Error::Truncated { format: FORMAT, offset: base + input.len() });
            };
            let Some(target) = out.get_mut(o..o + n) else {
                return Err(invalid(base + i - 1, "RLE data is longer than the chunk"));
            };
            target.copy_from_slice(literal);
            i += n;
            o += n;
        } else {
            let n = count as usize + 1;
            let Some(&value) = input.get(i) else {
                return Err(Error::Truncated { format: FORMAT, offset: base + input.len() });
            };
            let Some(target) = out.get_mut(o..o + n) else {
                return Err(invalid(base + i - 1, "RLE data is longer than the chunk"));
            };
            target.fill(value);
            i += 1;
            o += n;
        }
    }
    if o != out.len() {
        return Err(invalid(base + input.len(), "RLE data is shorter than the chunk"));
    }
    Ok(())
}

/// Appends the RLE coding of `data` (the reference's `rleCompress`).
fn rle_encode(data: &[u8], out: &mut Vec<u8>) {
    let mut start = 0;
    while start < data.len() {
        let mut end = start + 1;
        while end < data.len() && data[end] == data[start] && end - start < MAX_RUN {
            end += 1;
        }
        if end - start >= MIN_RUN {
            out.push((end - start - 1) as u8);
            out.push(data[start]);
        } else {
            // A literal stretch ends before the next run of three equal bytes.
            while end < data.len()
                && end - start < MAX_LITERAL
                && !(end + 2 < data.len() && data[end] == data[end + 1] && data[end + 1] == data[end + 2])
            {
                end += 1;
            }
            out.push((-((end - start) as i32)) as u8);
            out.extend_from_slice(&data[start..end]);
        }
        start = end;
    }
}

/// Buffers reused for every chunk of a file.
pub(super) struct Scratch {
    tmp: Vec<u8>,
    piz: piz::Scratch,
}

impl Scratch {
    /// Buffers for chunks of up to `chunk_bytes` uncompressed bytes.
    pub(super) fn new(chunk_bytes: usize) -> Self {
        Self { tmp: Vec::with_capacity(chunk_bytes), piz: piz::Scratch::new(chunk_bytes / 2) }
    }
}

/// Decompresses one chunk's data into `out`, its uncompressed bytes. `base` is the file offset of
/// `data`.
pub(super) fn decompress(
    compression: Compression,
    data: &[u8],
    base: usize,
    layout: &piz::Layout<'_>,
    out: &mut [u8],
    scratch: &mut Scratch,
) -> Result<()> {
    if data.len() == out.len() {
        // Stored uncompressed because compression did not pay.
        out.copy_from_slice(data);
        return Ok(());
    }
    if data.len() > out.len() {
        return Err(invalid(base, "chunk is larger than its uncompressed size"));
    }
    match compression {
        Compression::None => Err(invalid(base, "uncompressed chunk has the wrong size")),
        Compression::Rle => {
            scratch.tmp.clear();
            scratch.tmp.resize(out.len(), 0);
            rle_decode(data, base, &mut scratch.tmp)?;
            unpredict_and_merge(&mut scratch.tmp, out);
            Ok(())
        }
        Compression::Zips | Compression::Zip => {
            scratch.tmp.clear();
            scratch.tmp.resize(out.len(), 0);
            let decoded = deflate::zlib_decompress_into(data, &mut scratch.tmp)?;
            if decoded.written != out.len() {
                return Err(invalid(base, "zlib data is shorter than the chunk"));
            }
            unpredict_and_merge(&mut scratch.tmp, out);
            Ok(())
        }
        Compression::Piz => piz::decompress(data, base, layout, out, &mut scratch.piz),
    }
}

/// Compresses one chunk's uncompressed bytes into `out` (cleared first). If the result would not
/// be smaller, `out` holds `raw` itself, as the format requires.
pub(super) fn compress(
    compression: Compression,
    level: Level,
    raw: &[u8],
    layout: &piz::Layout<'_>,
    out: &mut Vec<u8>,
    tmp: &mut Vec<u8>,
) -> Result<()> {
    out.clear();
    match compression {
        Compression::None => {}
        Compression::Rle => {
            split_and_predict(raw, tmp);
            rle_encode(tmp, out);
        }
        Compression::Zips | Compression::Zip => {
            split_and_predict(raw, tmp);
            *out = deflate::zlib_compress(tmp, level);
        }
        Compression::Piz => piz::compress(raw, layout, out)?,
    }
    if compression == Compression::None || out.len() >= raw.len() {
        out.clear();
        out.extend_from_slice(raw);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    #[test]
    fn predictor_and_split_invert() {
        let mut rng = Rng::new(1);
        for len in [0, 1, 2, 3, 10, 101] {
            let raw: Vec<u8> = (0..len).map(|_| rng.next_u32() as u8).collect();
            let mut tmp = Vec::new();
            split_and_predict(&raw, &mut tmp);
            let mut back = vec![0; len];
            unpredict_and_merge(&mut tmp, &mut back);
            assert_eq!(back, raw);
        }
        // The reference's layout: [even bytes][odd bytes], differences + 128.
        let mut tmp = Vec::new();
        split_and_predict(&[10, 20, 11, 21, 12], &mut tmp);
        assert_eq!(tmp, [10, 129, 129, 136, 129]);
    }

    #[test]
    fn rle_round_trips_and_matches_the_reference_coding() {
        let mut rng = Rng::new(4);
        let mut out = Vec::new();
        rle_encode(&[7, 7, 7, 7, 1, 2], &mut out);
        assert_eq!(out, [3, 7, 0xFE, 1, 2]);
        for len in [0, 1, 2, 3, 127, 128, 129, 300, 1000] {
            for mode in 0..3 {
                let data: Vec<u8> = (0..len)
                    .map(|i| match mode {
                        0 => rng.next_u32() as u8,
                        1 => (i / 150) as u8,
                        _ => {
                            if rng.chance(0.3) {
                                rng.next_u32() as u8
                            } else {
                                0
                            }
                        }
                    })
                    .collect();
                let mut packed = Vec::new();
                rle_encode(&data, &mut packed);
                let mut back = vec![0; len];
                rle_decode(&packed, 0, &mut back).unwrap();
                assert_eq!(back, data, "{len} {mode}");
            }
        }
        let mut out = [0; 4];
        assert!(matches!(rle_decode(&[0xFB, 1, 2], 0, &mut out), Err(Error::Truncated { .. })));
        assert!(matches!(rle_decode(&[5, 1], 0, &mut out), Err(Error::Invalid { .. })));
        assert!(matches!(rle_decode(&[2], 0, &mut out), Err(Error::Truncated { .. })));
        assert!(matches!(rle_decode(&[1, 9], 0, &mut out), Err(Error::Invalid { .. })));
    }
}
