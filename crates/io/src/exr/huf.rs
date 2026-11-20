//! The Huffman coder of OpenEXR's PIZ compression (`ImfHuf.cpp` of the reference library), written
//! from the format's description `[DEP-05]`.
//!
//! A block is a 20-byte header (five little-endian `u32`: the lowest coded symbol `im`, the highest
//! `iM`, the table length, the number of data bits, a reserved zero), the code-length table and the
//! bit stream. Symbols are the 65 536 values of a `u16` plus `iM`, a run-length pseudo-symbol:
//! "repeat the previous value `n` more times", `n` being the next 8 bits.
//!
//! The table holds a 6-bit code length per symbol from `im` to `iM`, where the values 59 to 62
//! stand for 2 to 5 unused symbols and 63 for 6 to 261 (the count follows in 8 bits). Codes are
//! canonical with OpenEXR's ordering: a shorter code, padded with zeros, is numerically higher than
//! a longer one, and codes of one length increase with the symbol. Bits are packed from the most
//! significant end of each byte.
//!
//! The decoder checks every count against the input and the output, refuses tables that are not a
//! prefix code, and requires the bit stream to fill the output exactly `[ROB-01]`.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::error::{Error, Format, Result};

const FORMAT: Format = Format::Exr;

/// The number of symbols: every 16-bit value plus the run-length pseudo-symbol.
const ENC_SIZE: usize = (1 << 16) + 1;
/// The bits resolved by the decoder's direct lookup table.
const DEC_BITS: u32 = 14;
/// The longest code the format can describe.
const MAX_CODE_LEN: usize = 58;
/// The 6-bit table value of the shortest run of unused symbols (2 of them).
const SHORT_ZEROCODE_RUN: u64 = 59;
/// The 6-bit table value that announces a long run of unused symbols.
const LONG_ZEROCODE_RUN: u64 = 63;
/// The shortest run of unused symbols written as a long run.
const SHORTEST_LONG_RUN: usize = 2 + (LONG_ZEROCODE_RUN - SHORT_ZEROCODE_RUN) as usize;
/// The longest run of unused symbols one table entry can describe.
const LONGEST_LONG_RUN: usize = 255 + SHORTEST_LONG_RUN;
/// The size of the block header.
const HEADER_LEN: usize = 20;

/// An [`Error::Invalid`] of this module.
const fn invalid(offset: usize, what: &'static str) -> Error {
    Error::Invalid { format: FORMAT, offset, what }
}

/// For every code length, the numerically lowest code of that length (the reference's
/// `hufCanonicalCodeTable`), or `None` if lengths with these counts cannot form a prefix code.
fn first_codes(count: &[u64; MAX_CODE_LEN + 1]) -> Option<[u64; MAX_CODE_LEN + 1]> {
    let mut first = [0_u64; MAX_CODE_LEN + 1];
    let mut shorter: u64 = count[1..].iter().sum();
    let mut c = 0_u64;
    for l in (1..=MAX_CODE_LEN).rev() {
        let total = c + count[l];
        if total > 1_u64 << l {
            return None;
        }
        first[l] = c;
        shorter -= count[l];
        // An odd total leaves the last code of this length a prefix of the first shorter one.
        if total & 1 == 1 && shorter > 0 {
            return None;
        }
        c = total >> 1;
    }
    Some(first)
}

/// Writes bits from the most significant end of each byte.
struct BitWriter<'a> {
    out: &'a mut Vec<u8>,
    acc: u64,
    held: u32,
    total: u64,
}

impl BitWriter<'_> {
    /// Appends the low `n` bits of `value` (`n` at most 58).
    fn put(&mut self, n: u32, value: u64) {
        if n > 32 {
            self.put(n - 32, value >> 32);
            self.put(32, value & 0xFFFF_FFFF);
            return;
        }
        self.acc = (self.acc << n) | value;
        self.held += n;
        self.total += u64::from(n);
        while self.held >= 8 {
            self.held -= 8;
            self.out.push((self.acc >> self.held) as u8);
        }
    }

    /// Flushes the last partial byte, padded with zeros.
    fn finish(self) -> u64 {
        if self.held > 0 {
            self.out.push((self.acc << (8 - self.held)) as u8);
        }
        self.total
    }
}

/// Computes Huffman code lengths for the symbols `lo..=hi` with non-zero `freq` (the reference's
/// `hufBuildEncTable`, which merges linked lists instead of building a tree).
fn code_lengths(freq: &mut [u64], lo: usize, hi: usize, lengths: &mut [u8]) -> Result<()> {
    let mut link: Vec<usize> = (0..=hi).collect();
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> =
        (lo..=hi).filter(|&i| freq[i] > 0).map(|i| Reverse((freq[i], i))).collect();
    let too_long = Error::Encode { format: FORMAT, what: "a Huffman code would exceed 58 bits" };
    while heap.len() > 1 {
        let (Some(Reverse((_, mm))), Some(Reverse((_, m)))) = (heap.pop(), heap.pop()) else {
            break;
        };
        freq[m] += freq[mm];
        heap.push(Reverse((freq[m], m)));
        // Every symbol below the merged node gets one bit longer; the lists are joined at m.
        let mut j = m;
        loop {
            lengths[j] += 1;
            if usize::from(lengths[j]) > MAX_CODE_LEN {
                return Err(too_long);
            }
            if link[j] == j {
                link[j] = mm;
                break;
            }
            j = link[j];
        }
        let mut j = mm;
        loop {
            lengths[j] += 1;
            if usize::from(lengths[j]) > MAX_CODE_LEN {
                return Err(too_long);
            }
            if link[j] == j {
                break;
            }
            j = link[j];
        }
    }
    Ok(())
}

/// Appends the code-length table for the symbols `lo..=hi` (the reference's `hufPackEncTable`).
fn pack_table(lengths: &[u8], lo: usize, hi: usize, w: &mut BitWriter<'_>) {
    let mut i = lo;
    while i <= hi {
        let l = lengths[i];
        if l == 0 {
            let mut run = 1;
            while i < hi && run < LONGEST_LONG_RUN && lengths[i + 1] == 0 {
                i += 1;
                run += 1;
            }
            if run >= SHORTEST_LONG_RUN {
                w.put(6, LONG_ZEROCODE_RUN);
                w.put(8, (run - SHORTEST_LONG_RUN) as u64);
                i += 1;
                continue;
            }
            if run >= 2 {
                w.put(6, SHORT_ZEROCODE_RUN + run as u64 - 2);
                i += 1;
                continue;
            }
        }
        w.put(6, u64::from(l));
        i += 1;
    }
}

/// Compresses `raw` as one Huffman block appended to `out`; nothing is appended for no input.
pub(super) fn compress(raw: &[u16], out: &mut Vec<u8>) -> Result<()> {
    if raw.is_empty() {
        return Ok(());
    }
    let mut freq = vec![0_u64; ENC_SIZE];
    for &v in raw {
        freq[usize::from(v)] += 1;
    }
    let lo = freq.iter().position(|&f| f > 0).unwrap_or(0);
    let last = freq.iter().rposition(|&f| f > 0).unwrap_or(0);
    // The run-length pseudo-symbol follows the highest value.
    let rlc = last + 1;
    freq[rlc] = 1;
    let mut lengths = vec![0_u8; ENC_SIZE];
    code_lengths(&mut freq, lo, rlc, &mut lengths)?;

    let mut count = [0_u64; MAX_CODE_LEN + 1];
    for &l in &lengths[lo..=rlc] {
        count[usize::from(l)] += 1;
    }
    count[0] = 0;
    let Some(mut next) = first_codes(&count) else {
        return Err(Error::Encode { format: FORMAT, what: "the Huffman code lengths do not form a prefix code" });
    };
    // codes[i]: the code of symbol i in the high bits, its length in the low 6.
    let mut codes = vec![0_u64; ENC_SIZE];
    for i in lo..=rlc {
        let l = usize::from(lengths[i]);
        if l > 0 {
            codes[i] = (next[l] << 6) | l as u64;
            next[l] += 1;
        }
    }

    let start = out.len();
    out.extend_from_slice(&[0; HEADER_LEN]);
    let mut w = BitWriter { out, acc: 0, held: 0, total: 0 };
    pack_table(&lengths, lo, rlc, &mut w);
    w.finish();
    let table_len = out.len() - start - HEADER_LEN;

    let mut w = BitWriter { out, acc: 0, held: 0, total: 0 };
    let send = |w: &mut BitWriter<'_>, code: u64, run: usize| {
        let (len, rl) = (code & 63, codes[rlc] & 63);
        if len + rl + 8 < len * run as u64 {
            w.put(len as u32, code >> 6);
            w.put(rl as u32, codes[rlc] >> 6);
            w.put(8, run as u64);
        } else {
            for _ in 0..=run {
                w.put(len as u32, code >> 6);
            }
        }
    };
    let mut symbol = raw[0];
    let mut run = 0;
    for &v in &raw[1..] {
        if v == symbol && run < 255 {
            run += 1;
        } else {
            send(&mut w, codes[usize::from(symbol)], run);
            run = 0;
        }
        symbol = v;
    }
    send(&mut w, codes[usize::from(symbol)], run);
    let n_bits = w.finish();
    let Ok(n_bits) = i32::try_from(n_bits) else {
        return Err(Error::Encode { format: FORMAT, what: "a PIZ chunk needs more than 2^31 Huffman bits" });
    };

    let header = [lo as u32, rlc as u32, table_len as u32, n_bits as u32, 0];
    for (slot, value) in out[start..start + HEADER_LEN].as_chunks_mut::<4>().0.iter_mut().zip(header) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

/// Reads bits from the most significant end of each byte, zeros past the end of the data.
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    /// Bits not yet consumed, aligned to the top.
    acc: u128,
    held: u32,
}

impl BitReader<'_> {
    /// Loads whole bytes until more than 120 bits are held or the data ends.
    #[inline]
    fn refill(&mut self) {
        while self.held <= 120 {
            let Some(&byte) = self.data.get(self.pos) else { break };
            self.acc |= u128::from(byte) << (120 - self.held);
            self.held += 8;
            self.pos += 1;
        }
    }

    /// The next `n` bits (1 to 64) without consuming them.
    #[inline]
    const fn peek(&self, n: u32) -> u64 {
        (self.acc >> (128 - n)) as u64
    }

    /// Drops `n` bits that a [`Self::peek`] has seen.
    #[inline]
    const fn consume(&mut self, n: u32) {
        self.acc <<= n;
        self.held = self.held.saturating_sub(n);
    }
}

/// The decoder's tables, reused for every block of a file.
pub(super) struct Decoder {
    /// The code length of every symbol of the current block.
    lengths: Vec<u8>,
    /// The coded symbols, sorted by code length, then by value.
    symbols: Vec<u32>,
    /// For every 14-bit prefix: `(symbol << 6) | length` of the short code it starts with, or 0.
    table: Vec<u32>,
}

impl Decoder {
    /// A decoder with its tables allocated once.
    pub(super) fn new() -> Self {
        Self { lengths: vec![0; ENC_SIZE], symbols: Vec::new(), table: vec![0; 1 << DEC_BITS] }
    }

    /// Decompresses one Huffman block into `out`, which it must fill exactly. `base` is the file
    /// offset of `input`, for error messages.
    pub(super) fn decompress(&mut self, input: &[u8], base: usize, out: &mut [u16]) -> Result<()> {
        if input.is_empty() {
            return if out.is_empty() { Ok(()) } else { Err(invalid(base, "empty Huffman block")) };
        }
        let Some(header) = input.get(..HEADER_LEN) else {
            return Err(Error::Truncated { format: FORMAT, offset: base + input.len() });
        };
        let word = |i: usize| {
            u32::from_le_bytes([header[i * 4], header[i * 4 + 1], header[i * 4 + 2], header[i * 4 + 3]]) as usize
        };
        let (lo, rlc, n_bits) = (word(0), word(1), word(3));
        if lo >= ENC_SIZE || rlc >= ENC_SIZE || lo > rlc {
            return Err(invalid(base, "Huffman symbol range is out of bounds"));
        }
        let table_bytes = &input[HEADER_LEN..];
        let used = self.read_lengths(table_bytes, base + HEADER_LEN, lo, rlc)?;
        let data = &table_bytes[used..];
        let data_at = base + HEADER_LEN + used;
        if n_bits.div_ceil(8) > data.len() {
            return Err(Error::Truncated { format: FORMAT, offset: data_at + data.len() });
        }

        // The canonical code: per length, the count, the first code and where its symbols start.
        let mut count = [0_u64; MAX_CODE_LEN + 1];
        for &l in &self.lengths[lo..=rlc] {
            count[usize::from(l)] += 1;
        }
        count[0] = 0;
        let Some(first) = first_codes(&count) else {
            return Err(invalid(base + HEADER_LEN, "Huffman code lengths do not form a prefix code"));
        };
        let mut start = [0_usize; MAX_CODE_LEN + 2];
        for l in 1..=MAX_CODE_LEN {
            start[l + 1] = start[l] + count[l] as usize;
        }
        let max_len = (1..=MAX_CODE_LEN).rev().find(|&l| count[l] > 0).unwrap_or(0);
        self.symbols.clear();
        self.symbols.resize(start[MAX_CODE_LEN + 1], 0);
        let mut cursor = start;
        for symbol in lo..=rlc {
            let l = usize::from(self.lengths[symbol]);
            if l > 0 {
                self.symbols[cursor[l]] = symbol as u32;
                cursor[l] += 1;
            }
        }
        self.table.fill(0);
        for l in 1..=(DEC_BITS as usize).min(max_len) {
            for k in 0..count[l] {
                let code = (first[l] + k) as usize;
                let symbol = self.symbols[start[l] + k as usize];
                let shift = DEC_BITS as usize - l;
                self.table[code << shift..(code + 1) << shift].fill((symbol << 6) | l as u32);
            }
        }

        let mut bits = BitReader { data: &data[..n_bits.div_ceil(8)], pos: 0, acc: 0, held: 0 };
        let mut left = n_bits as u64;
        let mut o = 0;
        let error_at = |bits: &BitReader<'_>, what| invalid(data_at + bits.pos, what);
        while o < out.len() {
            if left == 0 {
                return Err(error_at(&bits, "Huffman data ends before the chunk is full"));
            }
            bits.refill();
            let entry = self.table[bits.peek(DEC_BITS) as usize];
            let (symbol, len) = if entry != 0 {
                ((entry >> 6) as usize, entry & 63)
            } else {
                let mut found = None;
                for l in DEC_BITS as usize + 1..=max_len {
                    if l as u64 > left {
                        break;
                    }
                    let v = bits.peek(l as u32);
                    if v >= first[l] && v - first[l] < count[l] {
                        found = Some((self.symbols[start[l] + (v - first[l]) as usize] as usize, l as u32));
                        break;
                    }
                }
                found.ok_or_else(|| error_at(&bits, "invalid Huffman code"))?
            };
            if u64::from(len) > left {
                return Err(error_at(&bits, "Huffman code runs past the end of the data"));
            }
            bits.consume(len);
            left -= u64::from(len);
            if symbol == rlc {
                if left < 8 {
                    return Err(error_at(&bits, "Huffman run count runs past the end of the data"));
                }
                let run = bits.peek(8) as usize;
                bits.consume(8);
                left -= 8;
                let Some(&previous) = out[..o].last() else {
                    return Err(error_at(&bits, "Huffman run before the first value"));
                };
                if run > out.len() - o {
                    return Err(error_at(&bits, "Huffman run past the end of the chunk"));
                }
                out[o..o + run].fill(previous);
                o += run;
            } else {
                // Every symbol other than the pseudo-symbol is below it, so it fits 16 bits.
                out[o] = symbol as u16;
                o += 1;
            }
        }
        if left != 0 {
            return Err(error_at(&bits, "Huffman data is longer than the chunk"));
        }
        Ok(())
    }

    /// Reads the code-length table for the symbols `lo..=hi`; returns the bytes it occupied.
    fn read_lengths(&mut self, table: &[u8], base: usize, lo: usize, hi: usize) -> Result<usize> {
        self.lengths.fill(0);
        let (mut acc, mut held, mut pos) = (0_u64, 0_u32, 0_usize);
        let mut get = |n: u32| -> Result<u64> {
            while held < n {
                let Some(&byte) = table.get(pos) else {
                    return Err(Error::Truncated { format: FORMAT, offset: base + pos });
                };
                acc = (acc << 8) | u64::from(byte);
                held += 8;
                pos += 1;
            }
            held -= n;
            Ok((acc >> held) & ((1 << n) - 1))
        };
        let mut i = lo;
        // Progress: every turn advances i by at least one.
        while i <= hi {
            let l = get(6)?;
            let run = if l == LONG_ZEROCODE_RUN {
                get(8)? as usize + SHORTEST_LONG_RUN
            } else if l >= SHORT_ZEROCODE_RUN {
                (l - SHORT_ZEROCODE_RUN) as usize + 2
            } else {
                self.lengths[i] = l as u8;
                1
            };
            if i + run > hi + 1 {
                return Err(invalid(base, "Huffman table describes too many symbols"));
            }
            i += run;
        }
        Ok(pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    fn round_trip(raw: &[u16]) -> Vec<u8> {
        let mut block = Vec::new();
        compress(raw, &mut block).unwrap();
        let mut back = vec![0_u16; raw.len()];
        Decoder::new().decompress(&block, 0, &mut back).unwrap();
        assert_eq!(back, raw);
        block
    }

    #[test]
    fn blocks_round_trip() {
        let mut rng = Rng::new(7);
        round_trip(&[]);
        round_trip(&[0]);
        round_trip(&[65535]);
        round_trip(&[5; 1000]);
        round_trip(&[0, 65535, 0, 65535]);
        let skewed: Vec<u16> = (0..20_000).map(|_| (rng.normal() * 30.0) as i16 as u16).collect();
        let block = round_trip(&skewed);
        assert!(block.len() < skewed.len(), "a peaked distribution compresses: {}", block.len());
        let uniform: Vec<u16> = (0..5000).map(|_| rng.next_u32() as u16).collect();
        round_trip(&uniform);
        // Runs of every length around the 255 limit, sparse symbols (long zero runs in the table).
        let mut runs = Vec::new();
        for len in [1, 2, 3, 200, 255, 256, 257, 600] {
            runs.extend(std::iter::repeat_n((len * 97) as u16, len));
        }
        round_trip(&runs);
        // Fibonacci frequencies give the longest codes a block of this size can have.
        let mut fib = Vec::new();
        let (mut a, mut b) = (1_usize, 1_usize);
        for symbol in 0..24_u16 {
            fib.extend(std::iter::repeat_n(symbol * 2000, a));
            (a, b) = (b, a + b);
        }
        rng.shuffle(&mut fib);
        round_trip(&fib);
    }

    #[test]
    fn canonical_codes_follow_the_reference_ordering() {
        // Lengths 1, 2, 3, 3: OpenEXR gives the longest codes the lowest values.
        let mut count = [0_u64; MAX_CODE_LEN + 1];
        count[1] = 1;
        count[2] = 1;
        count[3] = 2;
        let first = first_codes(&count).unwrap();
        assert_eq!((first[3], first[2], first[1]), (0, 1, 1));
        // Oversubscribed and non-prefix length sets are refused.
        count[1] = 3;
        assert!(first_codes(&count).is_none());
        let mut odd = [0_u64; MAX_CODE_LEN + 1];
        odd[1] = 1;
        odd[3] = 1;
        assert!(first_codes(&odd).is_none());
    }

    #[test]
    fn broken_blocks_are_refused() {
        let mut rng = Rng::new(3);
        let raw: Vec<u16> = (0..3000).map(|_| (rng.normal() * 9.0) as i16 as u16).collect();
        let mut block = Vec::new();
        compress(&raw, &mut block).unwrap();
        let mut decoder = Decoder::new();
        let mut out = vec![0_u16; raw.len()];
        for len in 0..block.len() {
            assert!(decoder.decompress(&block[..len], 0, &mut out).is_err(), "{len}");
        }
        // One value too many or too few to fill.
        assert!(decoder.decompress(&block, 0, &mut vec![0; raw.len() + 1]).is_err());
        assert!(decoder.decompress(&block, 0, &mut vec![0; raw.len() - 1]).is_err());
        // Symbol range out of bounds.
        let mut bad = block.clone();
        bad[4..8].copy_from_slice(&70_000_u32.to_le_bytes());
        assert!(decoder.decompress(&bad, 0, &mut out).is_err());
        // A bit count beyond the data.
        let mut bad = block.clone();
        bad[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(decoder.decompress(&bad, 0, &mut out), Err(Error::Truncated { .. })));
        for seed in 0..300 {
            let mut bad = block.clone();
            let mut rng = Rng::new(seed);
            let at = HEADER_LEN + rng.index(bad.len() - HEADER_LEN);
            bad[at] ^= 1 << rng.index(8);
            // Any outcome but a panic; a decoded result has the right size by construction.
            let _ = decoder.decompress(&bad, 0, &mut out);
        }
    }
}
