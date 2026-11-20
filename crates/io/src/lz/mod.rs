//! The in-house LZ compression codec and its frame `[DEP-05]` `[DOC-22]` `[IO-30]` `[ROB-01]`.
//!
//! A fast byte-oriented LZ77 codec for blobs, KTX2 supercompression and chunked containers: it
//! trades ratio for speed (several hundred MB/s to compress, GB/s to decompress) and has no entropy
//! stage. [`block`] holds the block codec; this module holds the *frame* that wraps blocks with a
//! magic number, the content size, the block size and a checksum. The byte layout of both is
//! documented in `docs/format-lz.md`.
//!
//! The frame decoder reads the declared content size first and refuses it when it is over
//! [`Limits::max_bytes`] or more than [`MAX_EXPANSION`] times the input, before anything is
//! allocated (decompression bombs) `[ROB-02]`; then it decodes block by block, each against its
//! exact expected length, and verifies the checksum of the whole content.
//!
//! ```
//! use humanart_io::{Limits, lz};
//! let data = b"blob blob blob blob blob blob blob blob".repeat(100);
//! let frame = lz::compress(&data);
//! assert!(frame.len() < data.len() / 10);
//! assert_eq!(lz::decompress(&frame, &Limits::DEFAULT)?, data);
//! # Ok::<(), humanart_io::Error>(())
//! ```

pub mod block;

use crate::bytes::Reader;
use crate::error::{Error, Format, Result};
use crate::image::Limits;
use humanart_core::hash::Hasher128;

pub use block::{Encoder, max_compressed_len};

/// The four bytes every frame starts with: `HALZ`.
pub const MAGIC: [u8; 4] = *b"HALZ";
/// The frame version this code writes and reads.
pub const VERSION: u8 = 1;
/// The size of the frame header in bytes.
pub const HEADER_LEN: usize = 16;
/// The size of the checksum after the last block.
pub const CHECKSUM_LEN: usize = 8;
/// The smallest block size, as log2 (64 KiB).
pub const MIN_BLOCK_LOG2: u8 = 16;
/// The largest block size, as log2 (4 MiB).
pub const MAX_BLOCK_LOG2: u8 = 22;
/// The block size [`compress`] uses, as log2 (256 KiB).
pub const DEFAULT_BLOCK_LOG2: u8 = 18;
/// The decoder refuses a content size above this many times the input length: no valid frame
/// expands further (one extra length byte stands for at most 255 output bytes).
pub const MAX_EXPANSION: u64 = 256;
/// The high bit of a block word: the block is stored uncompressed.
const STORED: u32 = 1 << 31;

const fn invalid(offset: usize, what: &'static str) -> Error {
    Error::Invalid { format: Format::Lz, offset, what }
}

/// Compresses `input` into a frame with 256 KiB blocks.
#[must_use]
pub fn compress(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    compress_into(input, DEFAULT_BLOCK_LOG2, &mut Encoder::new(), &mut out);
    out
}

/// Compresses `input` into a frame with blocks of `1 << block_log2` bytes (16 to 22). Larger blocks
/// cost nothing in ratio (matches reach 64 KiB back either way) but decode in fewer, larger steps.
pub fn compress_with(input: &[u8], block_log2: u8) -> Result<Vec<u8>> {
    if !(MIN_BLOCK_LOG2..=MAX_BLOCK_LOG2).contains(&block_log2) {
        return Err(Error::Encode { format: Format::Lz, what: "block size must be 2^16 to 2^22 bytes" });
    }
    let mut out = Vec::new();
    compress_into(input, block_log2, &mut Encoder::new(), &mut out);
    Ok(out)
}

/// Appends the frame of `input` to `out`, reusing `encoder`; `block_log2` is clamped to 16..=22.
/// Returns the number of bytes appended. A block that does not shrink is stored as it is, so the
/// frame is never more than 24 bytes plus 4 bytes per block larger than the input.
pub fn compress_into(input: &[u8], block_log2: u8, encoder: &mut Encoder, out: &mut Vec<u8>) -> usize {
    let start = out.len();
    let block_log2 = block_log2.clamp(MIN_BLOCK_LOG2, MAX_BLOCK_LOG2);
    let block_size = 1_usize << block_log2;
    out.reserve(HEADER_LEN + input.len() + input.len() / block_size * 4 + 4 + CHECKSUM_LEN);
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    out.push(block_log2);
    out.extend_from_slice(&0_u16.to_le_bytes());
    out.extend_from_slice(&(input.len() as u64).to_le_bytes());
    let mut hasher = Hasher128::new();
    for chunk in input.chunks(block_size) {
        hasher.update(chunk);
        let word_at = out.len();
        out.extend_from_slice(&0_u32.to_le_bytes());
        let packed = encoder.compress_block(chunk, out);
        let word = if packed < chunk.len() {
            // A block is at most 4 MiB, so its size fits the 31 bits.
            packed as u32
        } else {
            out.truncate(word_at + 4);
            out.extend_from_slice(chunk);
            chunk.len() as u32 | STORED
        };
        if let Some(slot) = out.get_mut(word_at..word_at + 4) {
            slot.copy_from_slice(&word.to_le_bytes());
        }
    }
    out.extend_from_slice(&hasher.finish().fold64().to_le_bytes());
    out.len() - start
}

/// The header of a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameInfo {
    /// The size of the decompressed content in bytes.
    pub content_size: u64,
    /// The block size as log2.
    pub block_log2: u8,
}

/// Reads and checks the header of a frame without decoding it.
pub fn frame_info(frame: &[u8]) -> Result<FrameInfo> {
    let mut reader = Reader::new(frame, Format::Lz);
    if reader.array::<4>()? != MAGIC {
        return Err(invalid(0, "not an LZ frame (bad magic)"));
    }
    let version = reader.u8()?;
    if version != VERSION {
        return Err(Error::Unsupported { format: Format::Lz, what: "frame version other than 1" });
    }
    let block_log2 = reader.u8()?;
    if !(MIN_BLOCK_LOG2..=MAX_BLOCK_LOG2).contains(&block_log2) {
        return Err(invalid(5, "block size out of range"));
    }
    if reader.u16_le()? != 0 {
        return Err(invalid(6, "reserved header bytes are not zero"));
    }
    let content_size = reader.u64_le()?;
    Ok(FrameInfo { content_size, block_log2 })
}

/// Decompresses a frame that holds exactly the input (nothing may follow it).
pub fn decompress(frame: &[u8], limits: &Limits) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    decompress_into(frame, limits, &mut out)?;
    Ok(out)
}

/// Decompresses a frame and appends its content to `out`; returns the content size. On an error
/// `out` is left as it was.
pub fn decompress_into(frame: &[u8], limits: &Limits, out: &mut Vec<u8>) -> Result<usize> {
    let start = out.len();
    let result = decode_frame(frame, limits, out);
    if result.is_err() {
        out.truncate(start);
    }
    result
}

fn decode_frame(frame: &[u8], limits: &Limits, out: &mut Vec<u8>) -> Result<usize> {
    let info = frame_info(frame)?;
    if info.content_size > limits.max_bytes {
        return Err(Error::Limit {
            format: Format::Lz,
            what: "content size",
            limit: limits.max_bytes,
            got: info.content_size,
        });
    }
    let input_len = frame.len() as u64;
    if info.content_size > input_len.saturating_mul(MAX_EXPANSION) {
        return Err(invalid(8, "content size is impossible for the frame length"));
    }
    let content_size =
        usize::try_from(info.content_size).map_err(|_| invalid(8, "content size does not fit in memory"))?;
    let block_size = 1_usize << info.block_log2;
    let mut reader = Reader::new(frame, Format::Lz);
    reader.skip(HEADER_LEN)?;
    out.reserve(content_size);
    let start = out.len();
    let mut hasher = Hasher128::new();
    let mut produced = 0;
    while produced < content_size {
        let block_len = block_size.min(content_size - produced);
        let word_at = reader.pos();
        let word = reader.u32_le()?;
        let size = (word & !STORED) as usize;
        let payload_at = reader.pos();
        let payload = reader.bytes(size)?;
        let block_start = out.len();
        if word & STORED != 0 {
            if size != block_len {
                return Err(invalid(word_at, "stored block size differs from the block length"));
            }
            out.extend_from_slice(payload);
        } else {
            if size > max_compressed_len(block_len) {
                return Err(invalid(word_at, "compressed block is larger than any encoder makes"));
            }
            block::decompress_block(payload, block_len, out, payload_at)?;
        }
        hasher.update(out.get(block_start..).unwrap_or_default());
        produced += block_len;
    }
    let checksum_at = reader.pos();
    let checksum = reader.u64_le()?;
    if checksum != hasher.finish().fold64() {
        return Err(invalid(checksum_at, "checksum mismatch"));
    }
    if !reader.is_empty() {
        return Err(invalid(reader.pos(), "unexpected data after the frame"));
    }
    Ok(out.len() - start)
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::fuzz;
    use humanart_core::rng::Rng;
    use std::time::{Duration, Instant};

    /// Test data that looks like what the codec carries: mesh-like floats, text, random bytes.
    fn sample(rng: &mut Rng, len: usize, kind: usize) -> Vec<u8> {
        let mut data = vec![0_u8; len];
        match kind % 4 {
            0 => rng.fill(&mut data),
            1 => {
                let text = include_bytes!("../../../../docs/reference-conventions.md");
                for (i, b) in data.iter_mut().enumerate() {
                    *b = text[i % text.len()];
                }
            }
            2 => {
                for (i, chunk) in data.chunks_mut(4).enumerate() {
                    let v = ((i / 3) as f32 * 0.002).sin() * 0.5 + (i % 3) as f32;
                    let bytes = v.to_le_bytes();
                    chunk.copy_from_slice(&bytes[..chunk.len()]);
                }
            }
            _ => {
                for (i, b) in data.iter_mut().enumerate() {
                    *b = if i % 97 < 60 { 0 } else { rng.index(8) as u8 };
                }
            }
        }
        data
    }

    #[test]
    fn frames_round_trip_at_every_block_size() {
        let mut rng = Rng::new(0x1A_2B);
        for case in 0..24 {
            let len = [0, 1, 13, 65_535, 65_536, 65_537, 300_000, 1 << 20][case % 8];
            let data = sample(&mut rng, len, case);
            for log2 in [16, 18, 22] {
                let frame = compress_with(&data, log2).unwrap();
                assert_eq!(&frame[..4], b"HALZ");
                assert_eq!(frame_info(&frame).unwrap(), FrameInfo { content_size: len as u64, block_log2: log2 });
                let blocks = len.div_ceil(1 << log2);
                assert!(frame.len() <= len + HEADER_LEN + CHECKSUM_LEN + blocks * 4, "case {case}");
                let mut out = vec![1, 2];
                assert_eq!(decompress_into(&frame, &Limits::DEFAULT, &mut out).unwrap(), len);
                assert_eq!(&out[2..], &data[..], "case {case} log2 {log2}");
            }
        }
        assert!(compress_with(b"x", 15).is_err() && compress_with(b"x", 23).is_err());
    }

    #[test]
    fn the_empty_frame_is_24_bytes() {
        let frame = compress(&[]);
        assert_eq!(frame.len(), HEADER_LEN + CHECKSUM_LEN);
        assert_eq!(&frame[..8], [b'H', b'A', b'L', b'Z', 1, 18, 0, 0]);
        assert_eq!(decompress(&frame, &Limits::DEFAULT).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn incompressible_blocks_are_stored() {
        let mut rng = Rng::new(5);
        let data = sample(&mut rng, 100_000, 0);
        let frame = compress_with(&data, 16).unwrap();
        let word = u32::from_le_bytes(frame[16..20].try_into().unwrap());
        assert_eq!(word, STORED | 65_536);
        assert_eq!(frame.len(), data.len() + 24 + 8);
    }

    #[test]
    fn rob_02_corrupt_and_hostile_frames_are_refused() {
        let data = b"some content some content some content, then more content".repeat(40);
        let frame = compress(&data);
        let limits = Limits::DEFAULT;
        let refuse = |bytes: &[u8]| {
            let mut out = vec![9_u8];
            let result = decompress_into(bytes, &limits, &mut out);
            assert_eq!(out, [9], "output restored on error");
            result.unwrap_err()
        };
        // Every truncation is an error.
        for cut in 0..frame.len() {
            refuse(&frame[..cut]);
        }
        let mut bad = frame.clone();
        bad[0] = b'X';
        assert!(matches!(refuse(&bad), Error::Invalid { offset: 0, .. }));
        let mut bad = frame.clone();
        bad[4] = 2;
        assert!(matches!(refuse(&bad), Error::Unsupported { .. }));
        let mut bad = frame.clone();
        bad[5] = 23;
        assert!(matches!(refuse(&bad), Error::Invalid { offset: 5, .. }));
        let mut bad = frame.clone();
        bad[6] = 1;
        assert!(matches!(refuse(&bad), Error::Invalid { offset: 6, .. }));
        // A flipped content bit fails the checksum (or the block structure).
        let last = frame.len() - CHECKSUM_LEN - 3;
        let mut bad = frame.clone();
        bad[last] ^= 0x40;
        assert!(matches!(refuse(&bad), Error::Invalid { .. }));
        let mut bad = frame.clone();
        bad.push(0);
        assert!(matches!(refuse(&bad), Error::Invalid { what: "unexpected data after the frame", .. }));
        // A decompression bomb is refused from the header alone.
        let mut bomb = frame.clone();
        bomb[8..16].copy_from_slice(&(1_u64 << 40).to_le_bytes());
        assert!(matches!(refuse(&bomb), Error::Limit { what: "content size", .. }));
        bomb[8..16].copy_from_slice(&(1_u64 << 30).to_le_bytes());
        assert!(matches!(refuse(&bomb), Error::Invalid { offset: 8, .. }));
        let small = Limits { max_bytes: 100, ..Limits::DEFAULT };
        assert!(matches!(decompress(&frame, &small), Err(Error::Limit { .. })));
    }

    #[test]
    fn rob_01_mutation_fuzz() {
        let mut rng = Rng::new(0x6C_7A);
        let mut corpus = vec![compress(&[]), compress(b"a"), compress(&b"abcabcabcabcabcabcabcabc".repeat(50))];
        for kind in 0..4 {
            let data = sample(&mut rng, 3000 + kind * 1000, kind);
            corpus.push(compress_with(&data, 16).unwrap_or_default());
        }
        // Raw blocks too, through the block decoder.
        fuzz::run("lz-frame", &corpus, 1500, 0x6C_7A, Duration::from_secs(120), |bytes| {
            let limits = Limits::FUZZ;
            if let Ok(out) = decompress(bytes, &limits) {
                assert_eq!(frame_info(bytes).map(|i| i.content_size).ok(), Some(out.len() as u64));
            }
            for len in [0, 7, 64, 5000] {
                let mut out = Vec::new();
                if block::decompress_block(bytes, len, &mut out, 0).is_ok() {
                    assert_eq!(out.len(), len);
                } else {
                    assert!(out.len() <= len);
                }
            }
        });
    }

    /// Prints the measured speed; run with `--ignored --nocapture` on an optimised build.
    #[test]
    #[ignore = "measurement, not a check"]
    fn measure_speed() {
        let mut rng = Rng::new(77);
        for (kind, name) in [(1, "text"), (2, "mesh floats"), (3, "sparse"), (0, "random")] {
            let data = sample(&mut rng, 32 << 20, kind);
            let mut encoder = Encoder::new();
            let mut frame = Vec::new();
            let begin = Instant::now();
            let rounds = 4;
            for _ in 0..rounds {
                frame.clear();
                compress_into(&data, DEFAULT_BLOCK_LOG2, &mut encoder, &mut frame);
            }
            let compress_s = begin.elapsed().as_secs_f64() / f64::from(rounds);
            let mut out = Vec::with_capacity(data.len());
            let begin = Instant::now();
            for _ in 0..rounds {
                out.clear();
                decompress_into(&frame, &Limits::DEFAULT, &mut out).unwrap();
            }
            let decompress_s = begin.elapsed().as_secs_f64() / f64::from(rounds);
            assert_eq!(out, data);
            let mb = data.len() as f64 / 1e6;
            println!(
                "{name}: ratio {:.3}, compress {:.0} MB/s, decompress {:.0} MB/s",
                frame.len() as f64 / data.len() as f64,
                mb / compress_s,
                mb / decompress_s
            );
        }
    }
}
