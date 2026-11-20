//! The DEFLATE decoder (RFC 1951): table-driven, with a 64-bit bit buffer `[IO-01]`.
//!
//! Each Huffman code is decoded through a two-level lookup table: a main table indexed by the next
//! [`LITLEN_BITS`] (or [`DIST_BITS`]) bits of the input, whose entries either decode a symbol outright or
//! point at a subtable for the longer codewords. One entry carries the symbol's base value, the bits
//! of the codeword and the extra bits, so a symbol costs one load and two shifts.
//!
//! Hostile input `[ROB-01]`: every code length set is validated before its table is built
//! (over-subscribed and incomplete sets are refused, with the two exceptions zlib also makes: a single
//! one-bit code, and a block without distance codes), every distance is checked against the output
//! written so far, the input is never read past its end, and the output never grows past the
//! caller's bound: room is asked from the [`Sink`] before a byte is written.

use super::tables::{
    DIST_BASE, DIST_EXTRA, END_OF_BLOCK, FIXED_DIST_LEN, LEN_BASE, LEN_EXTRA, NUM_DIST, NUM_LITLEN, NUM_PRECODE,
    PRECODE_ORDER, fixed_litlen_len,
};
use crate::error::{Error, Format, Result};

const FORMAT: Format = Format::Deflate;

/// Index bits of the literal/length main table.
const LITLEN_BITS: u32 = 11;
/// Index bits of the distance main table.
const DIST_BITS: u32 = 8;
/// Index bits of the code-length code's table (its codewords are at most 7 bits: no subtables).
const PRECODE_BITS: u32 = 7;
/// Main table plus the largest possible set of subtables for 288 symbols of at most 15 bits (the
/// bound zlib's `enough` tool computes for an 11-bit root).
const LITLEN_TABLE: usize = 2342;
/// The same bound for 32 symbols and an 8-bit root.
const DIST_TABLE: usize = 402;
const PRECODE_TABLE: usize = 1 << PRECODE_BITS;
/// Longest codeword of the literal/length and distance codes.
const MAX_CODE_LEN: usize = 15;
/// Longest codeword of the code-length code.
const MAX_PRECODE_LEN: usize = 7;
/// Symbols of the fixed literal/length code (two of them never occur in valid data).
const FIXED_LITLEN: usize = 288;
/// Symbols of the fixed distance code (two of them never occur in valid data).
const FIXED_DIST: usize = 32;

// Table entry layout (u32):
//   bits 0..=7    how many bits the entry consumes (rest of the codeword + extra bits)
//   bits 8..=11   how many of those belong to the codeword (for a subtable pointer: the subtable's
//                 index bits)
//   bits 16..=30  the literal, the length base, the distance base, or the subtable's start index
//   bit 31        LIT: the entry is a literal
//   bit 15        EXC: the entry is not a plain symbol; then
//   bit 14        SUB: it points at a subtable
//   bit 13        EOB: it is the end-of-block symbol
//                 (neither: a codeword that is unused or decodes to a symbol that may not occur)
const LIT: u32 = 1 << 31;
const EXC: u32 = 1 << 15;
const SUB: u32 = 1 << 14;
const EOB: u32 = 1 << 13;

/// The fast loop runs only while this much output room is left: three literals, the longest match
/// and the overrun of its 16-byte copy steps.
const FAST_OUT_MARGIN: usize = 3 + 258 + 16;
/// The fast loop runs only while this much input is left: two unconditional 8-byte refills.
const FAST_IN_MARGIN: usize = 16;

/// For a match distance below 8: the smallest multiple of it that is at least 8.
const WIDE_DISTANCE: [usize; 8] = [8, 8, 8, 9, 8, 10, 12, 14];

/// Where the decoder writes. Position 0 of [`Sink::buf`] is the first byte of the stream's output,
/// which is what match distances are checked against.
pub(super) trait Sink {
    /// The output written so far followed by the room that is left.
    fn buf(&mut self) -> &mut [u8];
    /// Makes `buf().len() >= written + need`, or reports that the output bound is reached.
    fn grow(&mut self, written: usize, need: usize) -> Result<()>;
}

/// A sink that appends to a vector, never beyond `max` bytes of output.
pub(super) struct VecSink<'a> {
    vec: &'a mut Vec<u8>,
    start: usize,
    max: usize,
}

impl<'a> VecSink<'a> {
    /// Appends after the current contents of `vec`; `guess` bytes of room are made at once (clamped
    /// to `max`).
    pub(super) fn new(vec: &'a mut Vec<u8>, max: usize, guess: usize) -> Self {
        let start = vec.len();
        vec.resize(start.saturating_add(guess.min(max)), 0);
        Self { vec, start, max }
    }
}

impl Sink for VecSink<'_> {
    fn buf(&mut self) -> &mut [u8] {
        self.vec.get_mut(self.start..).unwrap_or_default()
    }

    fn grow(&mut self, written: usize, need: usize) -> Result<()> {
        let target = written.saturating_add(need);
        if target > self.max {
            return Err(Error::Limit {
                format: FORMAT,
                what: "decompressed size",
                limit: self.max as u64,
                got: target as u64,
            });
        }
        let room = self.vec.len() - self.start;
        if target > room {
            // Double, so a long stream costs a logarithmic number of moves; never past the bound.
            let new_room = room.saturating_mul(2).max(target.saturating_add(4096)).min(self.max);
            self.vec.resize(self.start + new_room, 0);
        }
        Ok(())
    }
}

/// A sink over a caller's slice: the bound is the slice's length.
pub(super) struct SliceSink<'a> {
    out: &'a mut [u8],
}

impl<'a> SliceSink<'a> {
    pub(super) fn new(out: &'a mut [u8]) -> Self {
        Self { out }
    }
}

impl Sink for SliceSink<'_> {
    fn buf(&mut self) -> &mut [u8] {
        self.out
    }

    fn grow(&mut self, written: usize, need: usize) -> Result<()> {
        let target = written.saturating_add(need);
        if target > self.out.len() {
            return Err(Error::Limit {
                format: FORMAT,
                what: "decompressed size",
                limit: self.out.len() as u64,
                got: target as u64,
            });
        }
        Ok(())
    }
}

/// Which code a table is built for: it decides what a symbol's entry looks like and which incomplete
/// length sets are tolerated.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Precode,
    Litlen,
    Dist,
}

/// The entry of `sym` without its codeword bits, and the number of extra bits it reads.
fn symbol_entry(kind: Kind, sym: usize) -> (u32, u32) {
    match kind {
        Kind::Precode => ((sym as u32) << 16, 0),
        Kind::Litlen => match sym.cmp(&END_OF_BLOCK) {
            core::cmp::Ordering::Less => (LIT | ((sym as u32) << 16), 0),
            core::cmp::Ordering::Equal => (EXC | EOB, 0),
            core::cmp::Ordering::Greater => match (LEN_BASE.get(sym - 257), LEN_EXTRA.get(sym - 257)) {
                (Some(&base), Some(&extra)) => (u32::from(base) << 16, u32::from(extra)),
                // Symbols 286 and 287 exist in the fixed code but may not occur.
                _ => (EXC, 0),
            },
        },
        Kind::Dist => match (DIST_BASE.get(sym), DIST_EXTRA.get(sym)) {
            (Some(&base), Some(&extra)) => (u32::from(base) << 16, u32::from(extra)),
            // Symbols 30 and 31 exist in the fixed code but may not occur.
            _ => (EXC, 0),
        },
    }
}

/// Builds the decoding table of the canonical Huffman code with codeword lengths `lens` (0 = symbol
/// unused, at most `max_len`; at most 288 symbols).
///
/// The length set must be complete: exactly filling the code space. Over-subscribed sets are always
/// refused. An incomplete set is accepted in the cases zlib accepts: a literal/length or distance
/// code with a single one-bit codeword, and a distance code with no codeword at all; the unused
/// codewords then decode to an error.
fn build_table(
    table: &mut [u32],
    lens: &[u8],
    table_bits: u32,
    max_len: usize,
    kind: Kind,
) -> core::result::Result<(), &'static str> {
    let main_size = 1_usize << table_bits;
    if lens.len() > FIXED_LITLEN || max_len > MAX_CODE_LEN || table.len() < main_size {
        return Err("code table parameters out of range");
    }
    let mut count = [0_u32; MAX_CODE_LEN + 1];
    for &len in lens {
        let len = usize::from(len);
        if len > max_len {
            return Err("code length out of range");
        }
        count[len] += 1;
    }
    let used = lens.len() as u32 - count[0];

    // Kraft sum: `left` is the unclaimed code space after each length, in codewords of that length.
    let mut left: i64 = 1;
    for &n in &count[1..=max_len] {
        left = (left << 1) - i64::from(n);
        if left < 0 {
            return Err("over-subscribed code length set");
        }
    }
    if left > 0 {
        let single = used == 1 && count[1] == 1;
        let allowed = match kind {
            Kind::Precode => false,
            Kind::Litlen => single,
            Kind::Dist => single || used == 0,
        };
        if !allowed {
            return Err("incomplete code length set");
        }
        // Every codeword that is not assigned below is an error to meet.
        table[..main_size].fill(EXC);
    }

    // Symbols ordered by codeword length, then by value: the canonical order.
    let mut offsets = [0_usize; MAX_CODE_LEN + 2];
    for len in 1..=max_len {
        offsets[len + 1] = offsets[len] + count[len] as usize;
    }
    let mut sorted = [0_u16; FIXED_LITLEN];
    for (sym, &len) in lens.iter().enumerate() {
        if len != 0 {
            let slot = &mut offsets[usize::from(len)];
            sorted[*slot] = sym as u16;
            *slot += 1;
        }
    }

    let mut code: u32 = 0; // the canonical codeword, first bit most significant
    let mut next_symbol = 0_usize;
    let mut next_free = main_size;
    let mut current_prefix = usize::MAX;
    let mut sub_start = 0_usize;
    let mut sub_bits = 0_u32;
    for len in 1..=max_len {
        let len_bits = len as u32;
        let n = count[len];
        for k in 0..n {
            let sym = usize::from(sorted[next_symbol]);
            next_symbol += 1;
            // The input delivers a codeword's first bit in the lowest position.
            let reversed = (code.reverse_bits() >> (32 - len_bits)) as usize;
            let (payload, extra) = symbol_entry(kind, sym);
            if len_bits <= table_bits {
                let entry = payload | (len_bits << 8) | (len_bits + extra);
                let mut i = reversed;
                while i < main_size {
                    table[i] = entry;
                    i += 1 << len_bits;
                }
            } else {
                let prefix = reversed & (main_size - 1);
                if prefix != current_prefix {
                    // A new subtable: wide enough for the longest codeword that shares this prefix.
                    current_prefix = prefix;
                    sub_bits = len_bits - table_bits;
                    let mut space = n - k;
                    let mut longer = len;
                    while space < (1 << sub_bits) && longer < max_len {
                        sub_bits += 1;
                        longer += 1;
                        space = (space << 1) + count[longer];
                    }
                    sub_start = next_free;
                    next_free += 1 << sub_bits;
                    if next_free > table.len() {
                        return Err("code table overflow");
                    }
                    table[prefix] = EXC | SUB | ((sub_start as u32) << 16) | (sub_bits << 8) | table_bits;
                }
                let rest = len_bits - table_bits;
                let entry = payload | (rest << 8) | (rest + extra);
                let mut i = reversed >> table_bits;
                while i < (1 << sub_bits) {
                    table[sub_start + i] = entry;
                    i += 1 << rest;
                }
            }
            code += 1;
        }
        code <<= 1;
    }
    Ok(())
}

/// Reads eight bytes at `at` as a little-endian word, if they are there.
#[inline(always)]
fn read_u64(input: &[u8], at: usize) -> Option<u64> {
    let chunk = input.get(at..)?.first_chunk::<8>()?;
    Some(u64::from_le_bytes(*chunk))
}

/// Copies a match of `len` bytes from `distance` back, when at least `len + 16` bytes of room follow
/// `op` (the copy runs in steps of 8 or 16 bytes and may write past the match's end; what it writes
/// there is overwritten by the output that follows).
#[inline(always)]
fn copy_match_fast(out: &mut [u8], op: usize, distance: usize, len: usize) {
    let src = op - distance;
    if distance >= 16 {
        let mut i = 0;
        while i < len {
            out.copy_within(src + i..src + i + 16, op + i);
            i += 16;
        }
    } else if distance == 1 {
        let byte = out[src];
        out[op..op + len].fill(byte);
    } else {
        // The output repeats with period `distance`. Write the first bytes one by one until a source
        // a multiple of the period back, and at least 8 bytes away, exists; then go in 8-byte steps.
        let wide = if distance < 8 { WIDE_DISTANCE[distance] } else { distance };
        let lead = (wide - distance).min(len);
        for i in 0..lead {
            out[op + i] = out[src + i];
        }
        let mut i = lead;
        while i < len {
            let from = op + i - wide;
            out.copy_within(from..from + 8, op + i);
            i += 8;
        }
    }
}

/// Copies a match exactly, with no room to spare.
fn copy_match_exact(out: &mut [u8], op: usize, distance: usize, len: usize) {
    let src = op - distance;
    if distance >= len {
        out.copy_within(src..src + len, op);
    } else {
        for i in 0..len {
            out[op + i] = out[src + i];
        }
    }
}

/// How a run of the block decoder ended.
enum Status {
    /// The end-of-block symbol was read.
    Done,
    /// The next symbol needs this many bytes of room; nothing of it was consumed.
    NeedOutput(usize),
}

/// The decoder's state over one input.
pub(super) struct Inflater<'a> {
    input: &'a [u8],
    /// The next input byte to load into the bit buffer.
    pos: usize,
    /// The unread bits, the next one lowest. Bits at and above `bitcnt` are zero or already equal to
    /// the input bits that will be loaded there.
    bitbuf: u64,
    /// How many bits of `bitbuf` are valid.
    bitcnt: u32,
    litlen: [u32; LITLEN_TABLE],
    dist: [u32; DIST_TABLE],
    /// Whether the tables currently hold the fixed codes.
    fixed_loaded: bool,
}

impl<'a> Inflater<'a> {
    pub(super) fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            pos: 0,
            bitbuf: 0,
            bitcnt: 0,
            litlen: [EXC; LITLEN_TABLE],
            dist: [EXC; DIST_TABLE],
            fixed_loaded: false,
        }
    }

    /// The number of input bytes the stream occupied (after [`Inflater::run`] returned).
    pub(super) fn consumed(&self) -> usize {
        self.pos - (self.bitcnt / 8) as usize
    }

    fn truncated(&self) -> Error {
        Error::Truncated { format: FORMAT, offset: self.input.len() }
    }

    fn invalid(&self, what: &'static str) -> Error {
        Error::Invalid { format: FORMAT, offset: self.consumed(), what }
    }

    /// Loads input until at least 56 bits are buffered or the input is used up.
    #[inline]
    fn refill(&mut self) {
        if let Some(word) = read_u64(self.input, self.pos) {
            self.bitbuf |= word << self.bitcnt;
            self.pos += ((63 - self.bitcnt) >> 3) as usize;
            self.bitcnt |= 0b11_1000;
        } else {
            // Stops at 56..=63 bits, so the count never reaches the buffer's width.
            while self.bitcnt < 56 {
                let Some(&byte) = self.input.get(self.pos) else { break };
                self.bitbuf |= u64::from(byte) << self.bitcnt;
                self.pos += 1;
                self.bitcnt += 8;
            }
        }
    }

    /// Drops `n` buffered bits, which must be there.
    #[inline]
    fn take(&mut self, n: u32) -> Result<()> {
        if n > self.bitcnt {
            return Err(self.truncated());
        }
        self.bitbuf >>= n;
        self.bitcnt -= n;
        Ok(())
    }

    /// Reads `n` bits (at most 32) as a number.
    fn bits(&mut self, n: u32) -> Result<u32> {
        if self.bitcnt < n {
            self.refill();
        }
        let value = (self.bitbuf & ((1_u64 << n) - 1)) as u32;
        self.take(n)?;
        Ok(value)
    }

    /// Decodes the whole stream into `sink`; returns the number of bytes written.
    pub(super) fn run<S: Sink>(&mut self, sink: &mut S) -> Result<usize> {
        let mut written = 0_usize;
        loop {
            let header = self.bits(3)?;
            match header >> 1 {
                0 => self.stored(sink, &mut written)?,
                1 => {
                    self.load_fixed()?;
                    self.huffman(sink, &mut written)?;
                }
                2 => {
                    self.read_dynamic()?;
                    self.huffman(sink, &mut written)?;
                }
                _ => return Err(self.invalid("reserved block type")),
            }
            if header & 1 != 0 {
                return Ok(written);
            }
        }
    }

    /// A stored block: the rest of the current byte is skipped, then LEN, its complement and LEN
    /// raw bytes follow.
    fn stored<S: Sink>(&mut self, sink: &mut S, written: &mut usize) -> Result<()> {
        // Give the whole bytes still in the bit buffer back to the input.
        self.pos -= (self.bitcnt / 8) as usize;
        self.bitbuf = 0;
        self.bitcnt = 0;
        let header =
            self.input.get(self.pos..).and_then(|rest| rest.first_chunk::<4>()).ok_or_else(|| self.truncated())?;
        let len = u16::from_le_bytes([header[0], header[1]]);
        let check = u16::from_le_bytes([header[2], header[3]]);
        if len != !check {
            return Err(self.invalid("stored block length does not match its complement"));
        }
        let len = usize::from(len);
        let start = self.pos + 4;
        let data = self.input.get(start..start + len).ok_or_else(|| self.truncated())?;
        if sink.buf().len() - *written < len {
            sink.grow(*written, len)?;
        }
        let room = sink.buf().get_mut(*written..*written + len).ok_or(Error::Limit {
            format: FORMAT,
            what: "decompressed size",
            limit: *written as u64,
            got: (*written + len) as u64,
        })?;
        room.copy_from_slice(data);
        *written += len;
        self.pos = start + len;
        Ok(())
    }

    /// Loads the fixed codes of block type 1.
    fn load_fixed(&mut self) -> Result<()> {
        if self.fixed_loaded {
            return Ok(());
        }
        let mut lens = [0_u8; FIXED_LITLEN];
        for (sym, len) in lens.iter_mut().enumerate() {
            *len = fixed_litlen_len(sym);
        }
        let built = build_table(&mut self.litlen, &lens, LITLEN_BITS, MAX_CODE_LEN, Kind::Litlen).and_then(|()| {
            build_table(&mut self.dist, &[FIXED_DIST_LEN; FIXED_DIST], DIST_BITS, MAX_CODE_LEN, Kind::Dist)
        });
        built.map_err(|what| self.invalid(what))?;
        self.fixed_loaded = true;
        Ok(())
    }

    /// Reads the header of a dynamic block (type 2) and builds its two tables.
    fn read_dynamic(&mut self) -> Result<()> {
        self.fixed_loaded = false;
        let num_litlen = self.bits(5)? as usize + 257;
        let num_dist = self.bits(5)? as usize + 1;
        let num_precode = self.bits(4)? as usize + 4;
        if num_litlen > NUM_LITLEN || num_dist > NUM_DIST {
            return Err(self.invalid("too many length or distance symbols"));
        }
        let mut precode_lens = [0_u8; NUM_PRECODE];
        for &sym in PRECODE_ORDER.iter().take(num_precode) {
            precode_lens[usize::from(sym)] = self.bits(3)? as u8;
        }
        let mut precode = [0_u32; PRECODE_TABLE];
        build_table(&mut precode, &precode_lens, PRECODE_BITS, MAX_PRECODE_LEN, Kind::Precode)
            .map_err(|what| self.invalid(what))?;

        let total = num_litlen + num_dist;
        let mut lens = [0_u8; NUM_LITLEN + NUM_DIST];
        let mut i = 0;
        while i < total {
            // A codeword of at most 7 bits and at most 7 extra bits.
            if self.bitcnt < 14 {
                self.refill();
            }
            let entry = precode[(self.bitbuf & (PRECODE_TABLE as u64 - 1)) as usize];
            self.take(entry & 0xFF)?;
            let sym = entry >> 16;
            let (value, repeat) = match sym {
                0..=15 => {
                    lens[i] = sym as u8;
                    i += 1;
                    continue;
                }
                16 => {
                    if i == 0 {
                        return Err(self.invalid("code length repeat with nothing to repeat"));
                    }
                    (lens[i - 1], 3 + self.small_bits(2)?)
                }
                17 => (0, 3 + self.small_bits(3)?),
                _ => (0, 11 + self.small_bits(7)?),
            };
            if repeat > total - i {
                return Err(self.invalid("code length repeat runs past the last symbol"));
            }
            lens[i..i + repeat].fill(value);
            i += repeat;
        }
        if lens[END_OF_BLOCK] == 0 {
            return Err(self.invalid("block has no end-of-block code"));
        }
        let built = build_table(&mut self.litlen, &lens[..num_litlen], LITLEN_BITS, MAX_CODE_LEN, Kind::Litlen)
            .and_then(|()| build_table(&mut self.dist, &lens[num_litlen..total], DIST_BITS, MAX_CODE_LEN, Kind::Dist));
        built.map_err(|what| self.invalid(what))
    }

    /// Reads `n` already-buffered bits (at most 7) as a count.
    fn small_bits(&mut self, n: u32) -> Result<usize> {
        let value = (self.bitbuf & ((1_u64 << n) - 1)) as usize;
        self.take(n)?;
        Ok(value)
    }

    /// Decodes the symbols of a Huffman block until its end-of-block symbol.
    fn huffman<S: Sink>(&mut self, sink: &mut S, written: &mut usize) -> Result<()> {
        loop {
            let out = sink.buf();
            if self.fast_loop(out, written)? {
                return Ok(());
            }
            match self.careful_loop(out, written)? {
                Status::Done => return Ok(()),
                Status::NeedOutput(need) => sink.grow(*written, need)?,
            }
        }
    }

    /// Decodes symbols while plenty of input and output room is left, without per-symbol checks.
    /// Returns whether the end of the block was reached.
    fn fast_loop(&mut self, out: &mut [u8], written: &mut usize) -> Result<bool> {
        const LITLEN_MASK: u64 = (1 << LITLEN_BITS) - 1;
        const DIST_MASK: u64 = (1 << DIST_BITS) - 1;
        let input = self.input;
        let litlen = &self.litlen;
        let dist = &self.dist;
        let mut bitbuf = self.bitbuf;
        let mut bitcnt = self.bitcnt;
        let mut ip = self.pos;
        let mut op = *written;
        let mut done = false;
        let mut failure = None;

        while input.len() - ip >= FAST_IN_MARGIN && out.len() - op >= FAST_OUT_MARGIN {
            let Some(word) = read_u64(input, ip) else { break };
            bitbuf |= word << bitcnt;
            ip += ((63 - bitcnt) >> 3) as usize;
            bitcnt |= 0b11_1000;

            // At least 56 bits: room for three literals (45 bits), or two and a length (50 bits).
            let mut entry = litlen[(bitbuf & LITLEN_MASK) as usize];
            if entry & LIT != 0 {
                bitbuf >>= entry & 0xFF;
                bitcnt -= entry & 0xFF;
                out[op] = (entry >> 16) as u8;
                op += 1;
                entry = litlen[(bitbuf & LITLEN_MASK) as usize];
                if entry & LIT != 0 {
                    bitbuf >>= entry & 0xFF;
                    bitcnt -= entry & 0xFF;
                    out[op] = (entry >> 16) as u8;
                    op += 1;
                    entry = litlen[(bitbuf & LITLEN_MASK) as usize];
                    if entry & LIT != 0 {
                        bitbuf >>= entry & 0xFF;
                        bitcnt -= entry & 0xFF;
                        out[op] = (entry >> 16) as u8;
                        op += 1;
                        continue;
                    }
                }
            }
            if entry & EXC != 0 {
                if entry & SUB == 0 {
                    if entry & EOB != 0 {
                        bitbuf >>= entry & 0xFF;
                        bitcnt -= entry & 0xFF;
                        done = true;
                    } else {
                        failure = Some("invalid literal/length code");
                    }
                    break;
                }
                bitbuf >>= entry & 0xFF;
                bitcnt -= entry & 0xFF;
                let index = (entry >> 16) as usize + (bitbuf & ((1_u64 << ((entry >> 8) & 0xF)) - 1)) as usize;
                entry = litlen.get(index).copied().unwrap_or(EXC);
                if entry & LIT != 0 {
                    bitbuf >>= entry & 0xFF;
                    bitcnt -= entry & 0xFF;
                    out[op] = (entry >> 16) as u8;
                    op += 1;
                    continue;
                }
                if entry & EXC != 0 {
                    if entry & EOB != 0 {
                        bitbuf >>= entry & 0xFF;
                        bitcnt -= entry & 0xFF;
                        done = true;
                    } else {
                        failure = Some("invalid literal/length code");
                    }
                    break;
                }
            }
            // A length: its base plus the extra bits that follow the codeword.
            let total = entry & 0xFF;
            let len = (entry >> 16) as usize + ((bitbuf & ((1_u64 << total) - 1)) >> ((entry >> 8) & 0xF)) as usize;
            bitbuf >>= total;
            bitcnt -= total;

            // The distance needs up to 28 bits: load again (the input margin covers this read).
            let Some(word) = read_u64(input, ip) else {
                failure = Some("input ends inside a match");
                break;
            };
            bitbuf |= word << bitcnt;
            ip += ((63 - bitcnt) >> 3) as usize;
            bitcnt |= 0b11_1000;

            let mut entry = dist[(bitbuf & DIST_MASK) as usize];
            if entry & EXC != 0 {
                if entry & SUB == 0 {
                    failure = Some("invalid distance code");
                    break;
                }
                bitbuf >>= entry & 0xFF;
                bitcnt -= entry & 0xFF;
                let index = (entry >> 16) as usize + (bitbuf & ((1_u64 << ((entry >> 8) & 0xF)) - 1)) as usize;
                entry = dist.get(index).copied().unwrap_or(EXC);
                if entry & EXC != 0 {
                    failure = Some("invalid distance code");
                    break;
                }
            }
            let total = entry & 0xFF;
            let distance =
                (entry >> 16) as usize + ((bitbuf & ((1_u64 << total) - 1)) >> ((entry >> 8) & 0xF)) as usize;
            bitbuf >>= total;
            bitcnt -= total;
            if distance > op {
                failure = Some("match distance reaches before the start of the output");
                break;
            }
            copy_match_fast(out, op, distance, len);
            op += len;
        }

        self.bitbuf = bitbuf;
        self.bitcnt = bitcnt;
        self.pos = ip;
        *written = op;
        match failure {
            Some(what) => Err(self.invalid(what)),
            None => Ok(done),
        }
    }

    /// Looks up the literal/length or distance symbol at the head of the bit buffer and consumes its
    /// subtable prefix, if it has one. The returned entry is a symbol, the end of block, or invalid.
    #[inline]
    fn lookup(&mut self, litlen: bool) -> Result<u32> {
        let (table, bits): (&[u32], u32) = if litlen { (&self.litlen, LITLEN_BITS) } else { (&self.dist, DIST_BITS) };
        let entry = table.get((self.bitbuf & ((1_u64 << bits) - 1)) as usize).copied().unwrap_or(EXC);
        if entry & (EXC | SUB) != (EXC | SUB) {
            return Ok(entry);
        }
        let index_bits = (entry >> 8) & 0xF;
        let start = (entry >> 16) as usize;
        self.take(entry & 0xFF)?;
        let table: &[u32] = if litlen { &self.litlen } else { &self.dist };
        let entry = table.get(start + (self.bitbuf & ((1_u64 << index_bits) - 1)) as usize).copied().unwrap_or(EXC);
        // A subtable never points at another one.
        Ok(if entry & SUB != 0 { EXC } else { entry })
    }

    /// Decodes symbols one at a time, checking the input and the output room for each. Used near
    /// either end, where the fast loop's margins do not hold.
    fn careful_loop(&mut self, out: &mut [u8], written: &mut usize) -> Result<Status> {
        let mut op = *written;
        loop {
            // A whole match is at most 48 bits; after this, a shortage of bits is the input's end.
            self.refill();
            let (saved_buf, saved_cnt) = (self.bitbuf, self.bitcnt);
            let entry = self.lookup(true)?;
            if entry & LIT != 0 {
                let Some(slot) = out.get_mut(op) else {
                    (self.bitbuf, self.bitcnt) = (saved_buf, saved_cnt);
                    *written = op;
                    return Ok(Status::NeedOutput(1));
                };
                self.take(entry & 0xFF)?;
                *slot = (entry >> 16) as u8;
                op += 1;
                continue;
            }
            if entry & EXC != 0 {
                *written = op;
                if entry & EOB != 0 {
                    self.take(entry & 0xFF)?;
                    return Ok(Status::Done);
                }
                // Either a codeword nothing was assigned to, or one cut off by the end of the input.
                return Err(if self.bitcnt < 15 {
                    self.truncated()
                } else {
                    self.invalid("invalid literal/length code")
                });
            }
            let total = entry & 0xFF;
            let len =
                (entry >> 16) as usize + ((self.bitbuf & ((1_u64 << total) - 1)) >> ((entry >> 8) & 0xF)) as usize;
            self.take(total)?;

            let entry = self.lookup(false)?;
            if entry & EXC != 0 {
                *written = op;
                return Err(if self.bitcnt < 15 { self.truncated() } else { self.invalid("invalid distance code") });
            }
            let total = entry & 0xFF;
            let distance =
                (entry >> 16) as usize + ((self.bitbuf & ((1_u64 << total) - 1)) >> ((entry >> 8) & 0xF)) as usize;
            self.take(total)?;
            if distance > op {
                *written = op;
                return Err(self.invalid("match distance reaches before the start of the output"));
            }
            if out.len() - op < len {
                (self.bitbuf, self.bitcnt) = (saved_buf, saved_cnt);
                *written = op;
                return Ok(Status::NeedOutput(len));
            }
            copy_match_exact(out, op, distance, len);
            op += len;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decodes `sym` codewords bit by bit against a table, the slow obvious way, to check a table.
    fn decode_all(table: &[u32], table_bits: u32, lens: &[u8]) -> Vec<(usize, u32)> {
        // Canonical codes as RFC 1951 section 3.2.2 computes them, independently of `build_table`.
        let max = usize::from(*lens.iter().max().unwrap());
        let mut count = vec![0_u32; max + 1];
        for &len in lens.iter().filter(|&&len| len != 0) {
            count[usize::from(len)] += 1;
        }
        let mut next = vec![0_u32; max + 1];
        let mut code = 0_u32;
        for len in 1..=max {
            code = (code + count[len - 1]) << 1;
            next[len] = code;
        }
        let mut found = Vec::new();
        for (sym, &len) in lens.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let code = next[usize::from(len)];
            next[usize::from(len)] += 1;
            let reversed = u64::from(code.reverse_bits() >> (32 - u32::from(len)));
            // Garbage above the codeword must not matter.
            for high in [0_u64, !0] {
                let mut bits = reversed | (high << len);
                let mut entry = table[(bits & ((1 << table_bits) - 1)) as usize];
                let mut used = 0;
                if entry & (EXC | SUB) == (EXC | SUB) {
                    used = entry & 0xFF;
                    bits >>= used;
                    entry = table[(entry >> 16) as usize + (bits & ((1 << ((entry >> 8) & 0xF)) - 1)) as usize];
                }
                assert_eq!(used + ((entry >> 8) & 0xF), u32::from(len), "symbol {sym}: codeword length");
                found.push((sym, entry));
            }
        }
        found
    }

    #[test]
    fn tables_decode_every_symbol_of_the_fixed_codes() {
        let mut lens = [0_u8; FIXED_LITLEN];
        for (sym, len) in lens.iter_mut().enumerate() {
            *len = fixed_litlen_len(sym);
        }
        let mut table = [0_u32; LITLEN_TABLE];
        build_table(&mut table, &lens, LITLEN_BITS, MAX_CODE_LEN, Kind::Litlen).unwrap();
        for (sym, entry) in decode_all(&table, LITLEN_BITS, &lens) {
            match sym {
                0..=255 => assert_eq!(entry & (LIT | 0x00FF_0000), LIT | ((sym as u32) << 16)),
                256 => assert_eq!(entry & (EXC | EOB | SUB), EXC | EOB),
                257..=285 => {
                    assert_eq!(entry >> 16, u32::from(LEN_BASE[sym - 257]));
                    assert_eq!((entry & 0xFF) - ((entry >> 8) & 0xF), u32::from(LEN_EXTRA[sym - 257]));
                }
                _ => assert_eq!(entry & (EXC | EOB | SUB | LIT), EXC, "symbols 286 and 287 are invalid"),
            }
        }
        let mut table = [0_u32; DIST_TABLE];
        build_table(&mut table, &[5; 32], DIST_BITS, MAX_CODE_LEN, Kind::Dist).unwrap();
        for (sym, entry) in decode_all(&table, DIST_BITS, &[5; 32]) {
            if sym < 30 {
                assert_eq!(entry >> 16, u32::from(DIST_BASE[sym]));
                assert_eq!((entry & 0xFF) - ((entry >> 8) & 0xF), u32::from(DIST_EXTRA[sym]));
            } else {
                assert_eq!(entry & (EXC | SUB), EXC);
            }
        }
    }

    #[test]
    fn tables_with_subtables_decode_every_symbol() {
        // A skewed complete code with every length from 1 to 15: 1, 2, ..., 14, 15, 15.
        let mut lens: Vec<u8> = (1..=15).collect();
        lens.push(15);
        let mut table = [0_u32; LITLEN_TABLE];
        build_table(&mut table, &lens, LITLEN_BITS, MAX_CODE_LEN, Kind::Litlen).unwrap();
        for (sym, entry) in decode_all(&table, LITLEN_BITS, &lens) {
            assert_eq!(entry & (LIT | 0x00FF_0000), LIT | ((sym as u32) << 16));
        }
        let mut table = [0_u32; DIST_TABLE];
        build_table(&mut table, &lens, DIST_BITS, MAX_CODE_LEN, Kind::Dist).unwrap();
        for (sym, entry) in decode_all(&table, DIST_BITS, &lens) {
            assert_eq!(entry >> 16, u32::from(DIST_BASE[sym]));
        }
        // A flat code that needs the widest subtables: 256 symbols of 8 bits and 32 of... all 286
        // symbols at lengths 8 and 9, like the fixed code, plus a deep tail.
        let mut lens = vec![9_u8; 286];
        lens[..226].fill(8); // 226/256 + 60/512 = 1 exactly
        build_table(&mut [0_u32; LITLEN_TABLE], &lens, LITLEN_BITS, MAX_CODE_LEN, Kind::Litlen).unwrap();
    }

    #[test]
    fn invalid_length_sets_are_refused_as_zlib_refuses_them() {
        let mut t = [0_u32; LITLEN_TABLE];
        let build = |t: &mut [u32], lens: &[u8], kind| {
            let (bits, max) = match kind {
                Kind::Precode => (PRECODE_BITS, MAX_PRECODE_LEN),
                Kind::Litlen => (LITLEN_BITS, MAX_CODE_LEN),
                Kind::Dist => (DIST_BITS, MAX_CODE_LEN),
            };
            build_table(t, lens, bits, max, kind)
        };
        // Over-subscribed: three one-bit codes; two one-bit codes and a longer one.
        for kind in [Kind::Precode, Kind::Litlen, Kind::Dist] {
            assert_eq!(build(&mut t, &[1, 1, 1], kind), Err("over-subscribed code length set"));
            assert_eq!(build(&mut t, &[1, 1, 5], kind), Err("over-subscribed code length set"));
            assert_eq!(build(&mut t, &[2, 2, 2, 2, 7], kind), Err("over-subscribed code length set"));
            // Incomplete: one two-bit code, or 1 + 2.
            assert_eq!(build(&mut t, &[2], kind), Err("incomplete code length set"));
            assert_eq!(build(&mut t, &[1, 2], kind), Err("incomplete code length set"));
            assert_eq!(build(&mut t, &[1, 2, 3, 0, 4], kind), Err("incomplete code length set"));
            // Complete.
            assert_eq!(build(&mut t, &[1, 1], kind), Ok(()));
            assert_eq!(build(&mut t, &[1, 2, 3, 3], kind), Ok(()));
        }
        // The single one-bit code: fine for the two data codes, not for the code-length code.
        assert_eq!(build(&mut t, &[0, 1, 0], Kind::Precode), Err("incomplete code length set"));
        assert_eq!(build(&mut t, &[0, 1, 0], Kind::Litlen), Ok(()));
        assert_eq!(t[0] & (LIT | 0x00FF_0000), LIT | (1 << 16));
        assert_eq!(t[1], EXC, "the unused codeword is an error");
        assert_eq!(build(&mut t, &[0, 1, 0], Kind::Dist), Ok(()));
        // No code at all: only a distance code may be empty.
        assert_eq!(build(&mut t, &[0, 0], Kind::Dist), Ok(()));
        assert!(t[..1 << DIST_BITS].iter().all(|&e| e == EXC));
        assert_eq!(build(&mut t, &[0, 0], Kind::Litlen), Err("incomplete code length set"));
        assert_eq!(build(&mut t, &[0, 0], Kind::Precode), Err("incomplete code length set"));
        // Out-of-range parameters are errors, not panics.
        assert!(build(&mut t, &[8], Kind::Precode).is_err());
        assert!(build(&mut t, &[16], Kind::Litlen).is_err());
        assert!(build(&mut t, &[1; 289], Kind::Litlen).is_err());
        assert!(build_table(&mut t[..100], &[1, 1], LITLEN_BITS, MAX_CODE_LEN, Kind::Litlen).is_err());
    }

    #[test]
    fn match_copies_agree_with_the_byte_by_byte_definition() {
        for distance in 1..=40_usize {
            for len in 3..=70_usize {
                let mut expected: Vec<u8> = (0..distance as u8 + 3).map(|i| i.wrapping_mul(37) ^ 0x5A).collect();
                let op = expected.len();
                let mut fast = expected.clone();
                fast.resize(op + len + 16, 0xEE);
                let mut exact = expected.clone();
                exact.resize(op + len, 0xEE);
                for i in 0..len {
                    let byte = expected[op + i - distance];
                    expected.push(byte);
                }
                copy_match_fast(&mut fast, op, distance, len);
                copy_match_exact(&mut exact, op, distance, len);
                assert_eq!(&fast[..op + len], &expected[..], "fast, distance {distance}, length {len}");
                assert_eq!(exact, expected, "exact, distance {distance}, length {len}");
            }
        }
    }

    #[test]
    fn sinks_enforce_their_bounds() {
        let mut v = vec![9_u8; 3];
        let mut sink = VecSink::new(&mut v, 10, 4);
        assert_eq!(sink.buf().len(), 4);
        assert!(sink.grow(4, 6).is_ok());
        assert_eq!(sink.buf().len(), 10, "growth is clamped to the bound");
        assert!(matches!(sink.grow(10, 1), Err(Error::Limit { limit: 10, got: 11, .. })));
        assert!(matches!(sink.grow(usize::MAX, 2), Err(Error::Limit { .. })));
        assert_eq!(v.len(), 13);
        let mut out = [0_u8; 5];
        let mut sink = SliceSink::new(&mut out);
        assert!(sink.grow(2, 3).is_ok());
        assert!(matches!(sink.grow(2, 4), Err(Error::Limit { limit: 5, got: 6, .. })));
    }
}
