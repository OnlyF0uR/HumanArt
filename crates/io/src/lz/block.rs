//! The LZ block codec: one block of bytes as a sequence of literal runs and back-references
//! `[DEP-05]` `[ROB-01]`.
//!
//! The layout of a block is documented byte by byte in `docs/format-lz.md`. In short, a block is a
//! list of *sequences*; each is a token byte (high nibble: literal count, low nibble: match length
//! minus 4, the value 15 meaning "more length bytes follow"), the extra literal-length bytes, the
//! literals, a 2-byte little-endian match offset (1 to 65535) and the extra match-length bytes. The
//! last sequence has literals only and ends the block. This is the sequence layout of the LZ4 block
//! format; interoperability with other LZ4 implementations is not claimed and not tested.
//!
//! The encoder is a greedy single-pass matcher with a 16 384-entry hash table of 4-byte prefixes
//! (the speed-first design of LZ4's fast mode). The decoder checks every length, offset and output
//! position against the input and the declared block length, so hostile input gives an error,
//! never a panic or an overrun; the output never grows beyond the declared length.

use crate::error::{Error, Format, Result};

/// The shortest match the format can express.
pub const MIN_MATCH: usize = 4;
/// The largest match offset (a 2-byte field).
pub const MAX_OFFSET: usize = 65_535;
/// The encoder leaves at least this many bytes at the end of a block as literals.
const LAST_LITERALS: usize = 5;
/// The encoder starts no match within this many bytes of the end of a block.
const MATCH_FIND_LIMIT: usize = 12;
/// log2 of the number of hash table entries.
const HASH_LOG: u32 = 14;

/// The largest size the compressed form of `len` input bytes can have: one length byte per 255
/// literals plus the token, which is what an incompressible block costs.
#[must_use]
pub const fn max_compressed_len(len: usize) -> usize {
    len + len / 255 + 16
}

fn read_u32(src: &[u8], at: usize) -> u32 {
    match src.get(at..at + 4) {
        Some(&[a, b, c, d]) => u32::from_le_bytes([a, b, c, d]),
        _ => 0,
    }
}

fn read_u64(src: &[u8], at: usize) -> u64 {
    match src.get(at..at + 8) {
        Some(&[a, b, c, d, e, f, g, h]) => u64::from_le_bytes([a, b, c, d, e, f, g, h]),
        _ => 0,
    }
}

const fn hash(sequence: u32) -> usize {
    (sequence.wrapping_mul(2_654_435_761) >> (32 - HASH_LOG)) as usize
}

/// Appends the extra bytes of a length that did not fit its nibble: 255 while more than 254 remain,
/// then the rest.
fn push_length(out: &mut Vec<u8>, mut rest: usize) {
    while rest >= 255 {
        out.push(255);
        rest -= 255;
    }
    // `rest` is below 255 here.
    out.push(rest as u8);
}

/// Appends one sequence: `literals`, then (unless `matched` is `None`) a match of the given offset
/// and length.
fn push_sequence(out: &mut Vec<u8>, literals: &[u8], matched: Option<(usize, usize)>) {
    let lit = literals.len();
    let lit_nibble = lit.min(15) as u8;
    let match_nibble = matched.map_or(0, |(_, len)| (len - MIN_MATCH).min(15) as u8);
    out.push(lit_nibble << 4 | match_nibble);
    if lit >= 15 {
        push_length(out, lit - 15);
    }
    out.extend_from_slice(literals);
    if let Some((offset, len)) = matched {
        // The matcher only produces offsets up to MAX_OFFSET.
        out.extend_from_slice(&(offset as u16).to_le_bytes());
        if len - MIN_MATCH >= 15 {
            push_length(out, len - MIN_MATCH - 15);
        }
    }
}

/// A block encoder. It owns its hash table, so one encoder compresses any number of blocks without
/// allocating again.
#[derive(Clone)]
pub struct Encoder {
    table: Box<[u32]>,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Encoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Encoder").field("table_entries", &self.table.len()).finish()
    }
}

impl Encoder {
    /// An encoder with an empty hash table (64 KiB).
    #[must_use]
    pub fn new() -> Self {
        Self { table: vec![0; 1 << HASH_LOG].into_boxed_slice() }
    }

    /// Appends the compressed form of `src` to `out`; returns the number of bytes appended (at
    /// most [`max_compressed_len`]). The block decodes on its own: matches never reach outside it.
    /// Blocks longer than 4 GiB are valid but only find matches in their first 4 GiB.
    pub fn compress_block(&mut self, src: &[u8], out: &mut Vec<u8>) -> usize {
        let start_len = out.len();
        out.reserve(max_compressed_len(src.len()));
        let len = src.len();
        let mut anchor = 0;
        if len > MATCH_FIND_LIMIT {
            self.table.fill(0);
            let table = &mut self.table;
            // A match starts before `ip_limit` and ends at or before `match_limit`. Every read below
            // stays inside `src`: positions are below `ip_limit`, which is 12 bytes before the end.
            let ip_limit = len - MATCH_FIND_LIMIT;
            let match_limit = len - LAST_LITERALS;
            let mut ip = 0;
            'blocks: loop {
                // Find the next match; the step grows while nothing is found, so incompressible data
                // is skipped quickly.
                let mut misses = 0_usize;
                let (mut start, mut candidate) = loop {
                    if ip >= ip_limit {
                        break 'blocks;
                    }
                    let sequence = read_u32(src, ip);
                    let slot = hash(sequence);
                    let candidate = table.get(slot).map_or(0, |&c| c as usize);
                    if let Some(entry) = table.get_mut(slot) {
                        *entry = ip as u32;
                    }
                    if candidate < ip && ip - candidate <= MAX_OFFSET && read_u32(src, candidate) == sequence {
                        break (ip, candidate);
                    }
                    ip += 1 + (misses >> 6);
                    misses += 1;
                };
                // Extend backwards over equal bytes not yet emitted.
                while start > anchor && candidate > 0 && src.get(start - 1) == src.get(candidate - 1) {
                    start -= 1;
                    candidate -= 1;
                }
                // Extend forwards, eight bytes at a time.
                let mut matched = MIN_MATCH;
                loop {
                    if start + matched + 8 <= match_limit {
                        let diff = read_u64(src, start + matched) ^ read_u64(src, candidate + matched);
                        if diff == 0 {
                            matched += 8;
                            continue;
                        }
                        matched += (diff.trailing_zeros() / 8) as usize;
                        break;
                    }
                    while start + matched < match_limit && src.get(start + matched) == src.get(candidate + matched) {
                        matched += 1;
                    }
                    break;
                }
                push_sequence(out, src.get(anchor..start).unwrap_or_default(), Some((start - candidate, matched)));
                ip = start + matched;
                anchor = ip;
                if ip >= ip_limit {
                    break;
                }
                // Remember a position inside the match, which helps the next search.
                let inside = ip - 2;
                if let Some(entry) = table.get_mut(hash(read_u32(src, inside))) {
                    *entry = inside as u32;
                }
            }
        }
        push_sequence(out, src.get(anchor..).unwrap_or_default(), None);
        out.len() - start_len
    }
}

/// Compresses one block with a fresh [`Encoder`] (which allocates its 64 KiB table); see
/// [`Encoder::compress_block`].
pub fn compress_block(src: &[u8], out: &mut Vec<u8>) -> usize {
    Encoder::new().compress_block(src, out)
}

/// Reads the extra bytes of a length at `*at` and returns their sum; fails when the input ends or
/// the sum exceeds `cap`.
fn read_length(src: &[u8], at: &mut usize, cap: usize, base: usize) -> Result<usize> {
    let mut total = 0_usize;
    loop {
        let byte = *src.get(*at).ok_or(Error::Truncated { format: Format::Lz, offset: base + *at })?;
        *at += 1;
        total += usize::from(byte);
        if total > cap {
            return Err(Error::Invalid {
                format: Format::Lz,
                offset: base + *at - 1,
                what: "length runs past the block",
            });
        }
        if byte != 255 {
            return Ok(total);
        }
    }
}

/// Bytes copied at once by the fast paths of the decoder. A fast copy may write up to this many
/// bytes past the end of a literal run or match, inside the output buffer; the next sequence
/// overwrites them.
const WILD: usize = 16;

/// Copies a match of `len` bytes from `offset` bytes before `op` to `op` inside `dst`, where the
/// copy may overlap what it produces (offset 1 repeats one byte). The caller has checked
/// `1 <= offset <= op` and `op + len <= dst.len()`.
fn copy_match(dst: &mut [u8], op: usize, offset: usize, len: usize) {
    let from = op - offset;
    if offset >= WILD && op + len + WILD <= dst.len() {
        // Whole 16-byte chunks, each read from bytes that are already final.
        let mut k = 0;
        while k < len {
            dst.copy_within(from + k..from + k + WILD, op + k);
            k += WILD;
        }
    } else if offset >= len {
        dst.copy_within(from..from + len, op);
    } else if offset == 1 {
        let byte = dst.get(from).copied().unwrap_or_default();
        if let Some(run) = dst.get_mut(op..op + len) {
            run.fill(byte);
        }
    } else {
        // The bytes from `from` on repeat with period `offset`; copy whole periods, doubling.
        let mut done = 0;
        while done < len {
            let period = op + done - from;
            let n = (len - done).min(period - period % offset);
            dst.copy_within(from..from + n, op + done);
            done += n;
        }
    }
}

/// Decodes one block from `src` into `dst`, which is exactly as long as the block; returns how many
/// bytes the block produced. `base` is the offset of `src` in the enclosing input.
fn decode(src: &[u8], dst: &mut [u8], base: usize) -> Result<usize> {
    let (src_len, dst_len) = (src.len(), dst.len());
    let invalid = |at: usize, what: &'static str| Error::Invalid { format: Format::Lz, offset: base + at, what };
    let truncated = |at: usize| Error::Truncated { format: Format::Lz, offset: base + at };
    let mut ip = 0;
    let mut op = 0;
    loop {
        let token = *src.get(ip).ok_or_else(|| truncated(ip))?;
        ip += 1;
        let mut literals = usize::from(token >> 4);
        if literals == 15 {
            literals += read_length(src, &mut ip, dst_len, base)?;
        }
        if literals > dst_len - op {
            return Err(invalid(ip, "literals run past the block"));
        }
        if literals > src_len - ip {
            return Err(truncated(src_len));
        }
        if literals <= WILD && ip + WILD <= src_len && op + WILD <= dst_len {
            dst[op..op + WILD].copy_from_slice(&src[ip..ip + WILD]);
        } else {
            dst[op..op + literals].copy_from_slice(&src[ip..ip + literals]);
        }
        ip += literals;
        op += literals;
        if ip == src_len {
            return Ok(op);
        }
        let offset = match src.get(ip..ip + 2) {
            Some(&[lo, hi]) => usize::from(u16::from_le_bytes([lo, hi])),
            _ => return Err(truncated(src_len)),
        };
        if offset == 0 || offset > op {
            return Err(invalid(ip, "match offset outside the block"));
        }
        ip += 2;
        let mut matched = usize::from(token & 15) + MIN_MATCH;
        if token & 15 == 15 {
            matched += read_length(src, &mut ip, dst_len, base)?;
        }
        if matched > dst_len - op {
            return Err(invalid(ip, "match runs past the block"));
        }
        copy_match(dst, op, offset, matched);
        op += matched;
    }
}

/// Decodes one block from `src`, which must hold exactly one block, appending exactly `block_len`
/// bytes to `out`. `base` is the offset of `src` in the enclosing input, for error offsets. On an
/// error `out` is left as it was.
///
/// Refused: a match offset of 0 or reaching before the start of the block, literals or a match that
/// run past `block_len`, input that ends inside a sequence, a block that ends after a match, and a
/// block whose output is shorter than `block_len`.
pub fn decompress_block(src: &[u8], block_len: usize, out: &mut Vec<u8>, base: usize) -> Result<()> {
    let start = out.len();
    let end = start.checked_add(block_len).ok_or(Error::Limit {
        format: Format::Lz,
        what: "block length",
        limit: usize::MAX as u64,
        got: block_len as u64,
    })?;
    out.resize(end, 0);
    let result = match decode(src, out.get_mut(start..).unwrap_or_default(), base) {
        Ok(produced) if produced == block_len => Ok(()),
        Ok(_) => {
            Err(Error::Invalid { format: Format::Lz, offset: base + src.len(), what: "block is shorter than declared" })
        }
        Err(error) => Err(error),
    };
    if result.is_err() {
        out.truncate(start);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    fn round_trip(data: &[u8]) -> usize {
        let mut packed = Vec::new();
        let n = compress_block(data, &mut packed);
        assert_eq!(n, packed.len());
        assert!(n <= max_compressed_len(data.len()), "{n} > bound for {}", data.len());
        let mut out = vec![7_u8; 3];
        decompress_block(&packed, data.len(), &mut out, 0).unwrap();
        assert_eq!(&out[..3], [7, 7, 7], "existing output is kept");
        assert_eq!(&out[3..], data);
        n
    }

    #[test]
    fn edge_lengths_round_trip() {
        for len in 0..80 {
            let zeros = vec![0_u8; len];
            round_trip(&zeros);
            let ramp: Vec<u8> = (0..len).map(|i| i as u8).collect();
            round_trip(&ramp);
        }
        // An empty block is a single zero token.
        let mut packed = Vec::new();
        compress_block(&[], &mut packed);
        assert_eq!(packed, [0]);
    }

    #[test]
    fn repetitive_data_compresses_and_overlapping_matches_decode() {
        assert!(round_trip(&vec![0xAB; 1 << 20]) < 5000);
        for period in 1..20 {
            let data: Vec<u8> = (0..10_000).map(|i| (i % period) as u8 * 13).collect();
            assert!(round_trip(&data) < 200, "period {period}");
        }
        let text = b"the quick brown fox jumps over the lazy dog; ".repeat(500);
        assert!(round_trip(&text) < text.len() / 20);
    }

    #[test]
    fn random_and_mixed_data_round_trip() {
        let mut rng = Rng::new(0x1_2_3);
        for case in 0..200 {
            let len = rng.index(70_000);
            let mut data = vec![0_u8; len];
            match case % 4 {
                0 => rng.fill(&mut data),
                1 => data.iter_mut().for_each(|b| *b = rng.index(4) as u8),
                2 => {
                    // Copies from far and near, including beyond the 64 KiB window.
                    rng.fill(&mut data);
                    for _ in 0..50 {
                        if len < 2 {
                            break;
                        }
                        let from = rng.index(len);
                        let to = rng.index(len);
                        let n = rng.index(300).min(len - from).min(len - to);
                        data.copy_within(from..from + n, to);
                    }
                }
                _ => {
                    for (i, b) in data.iter_mut().enumerate() {
                        let x = (i as f32 * 0.001).sin();
                        *b = x.to_bits().to_le_bytes()[i % 4];
                    }
                }
            }
            let n = round_trip(&data);
            if case % 4 == 0 {
                assert!(n <= max_compressed_len(len));
            }
        }
    }

    #[test]
    fn long_lengths_use_extra_bytes() {
        // 300 literals then a 600-byte match: both lengths need extra bytes.
        let mut rng = Rng::new(9);
        let mut data = vec![0_u8; 300];
        rng.fill(&mut data);
        let copy = data.clone();
        data.extend_from_slice(&copy);
        data.extend_from_slice(&copy);
        data.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7]);
        let mut packed = Vec::new();
        compress_block(&data, &mut packed);
        assert_eq!(packed[0], 0xFF);
        assert!(packed.len() < 330, "{}", packed.len());
        let mut out = Vec::new();
        decompress_block(&packed, data.len(), &mut out, 0).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn hand_built_blocks_decode_and_bad_ones_are_refused() {
        let decode = |src: &[u8], len: usize| {
            let mut out = Vec::new();
            decompress_block(src, len, &mut out, 100).map(|()| out)
        };
        // "ab" then a match of 6 at offset 2 then literals "c".
        assert_eq!(decode(&[0x22, b'a', b'b', 2, 0, 0x10, b'c'], 9).unwrap(), b"ababababc");
        // RLE: one byte then a 4-byte match at offset 1.
        assert_eq!(decode(&[0x10, b'z', 1, 0, 0x00], 5).unwrap(), b"zzzzz");
        let invalid = |src: &[u8], len: usize, what: &str| match decode(src, len) {
            Err(Error::Invalid { what: got, offset, .. }) => {
                assert_eq!(got, what);
                assert!(offset >= 100);
            }
            other => panic!("{other:?}"),
        };
        invalid(&[0x10, b'z', 0, 0, 0x00], 5, "match offset outside the block");
        invalid(&[0x10, b'z', 2, 0, 0x00], 5, "match offset outside the block");
        invalid(&[0x10, b'z', 1, 0, 0x00], 4, "match runs past the block");
        invalid(&[0x30, b'a', b'b', b'c'], 2, "literals run past the block");
        invalid(&[0x20, b'a', b'b'], 3, "block is shorter than declared");
        invalid(&[0xF0, 255, 255, 255], 100, "length runs past the block");
        let truncated =
            |src: &[u8], len: usize| assert!(matches!(decode(src, len), Err(Error::Truncated { .. })), "{src:?}");
        truncated(&[], 0);
        truncated(&[0x30, b'a'], 3);
        truncated(&[0xF0, 255], 1000);
        truncated(&[0x10, b'z', 1], 5);
        // A block may not end with a match.
        truncated(&[0x10, b'z', 1, 0], 5);
    }
}
