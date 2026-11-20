//! Huffman coding of the entropy-coded segments (T.81 annex C, F.2.2.3, F.1.2): the decoding table
//! with its fast lookup, the bit reader that undoes byte stuffing and stops at markers, and the
//! encoder's code table and bit writer.

use super::FORMAT;
use super::tables::HuffSpec;
use crate::error::{Error, Result};

/// Codes up to this many bits are resolved by one table lookup.
const LOOKUP_BITS: u32 = 10;

/// A Huffman table ready for decoding.
#[derive(Clone, Debug)]
pub(super) struct DecodeTable {
    /// For each `LOOKUP_BITS`-bit prefix: `(code length << 8) | symbol`, or 0 when the code is longer.
    lookup: [u16; 1 << LOOKUP_BITS],
    /// For AC use: per `LOOKUP_BITS`-bit prefix holding a whole run/size code *and* its magnitude
    /// bits, `(value << 16) | (run << 8) | total length`; 0 when they do not fit.
    fast_ac: [i32; 1 << LOOKUP_BITS],
    /// The largest code of each length 1 to 16 (-1 when there is none).
    max_code: [i32; 17],
    /// For each length: the index in `symbols` of its first code, minus that code.
    val_offset: [i32; 17],
    /// The symbols, shortest code first.
    symbols: [u8; 256],
    /// How many symbols there are.
    count: usize,
}

/// The canonical codes of a table specification (T.81 C.2): `(length, code)` per symbol, in the order
/// of `spec.symbols`. Refuses more than 256 codes and code sets that do not fit their lengths (a
/// prefix-free code needs `code < 2^length` at every length).
fn canonical_codes(spec: &HuffSpec, offset: usize) -> Result<Vec<(u8, u16)>> {
    let total: usize = spec.counts.iter().map(|&c| usize::from(c)).sum();
    if total > 256 || total != spec.symbols.len() {
        return Err(Error::Invalid { format: FORMAT, offset, what: "Huffman table has a bad symbol count" });
    }
    let mut codes = Vec::with_capacity(total);
    let mut code: u32 = 0;
    for (len, &n) in (1_u8..=16).zip(&spec.counts) {
        for _ in 0..n {
            if code >= 1 << len {
                return Err(Error::Invalid { format: FORMAT, offset, what: "Huffman table is over-subscribed" });
            }
            codes.push((len, code as u16));
            code += 1;
        }
        code <<= 1;
    }
    Ok(codes)
}

impl DecodeTable {
    /// Builds the decoding table of a DHT specification; `offset` locates errors.
    pub(super) fn new(spec: &HuffSpec, offset: usize) -> Result<Self> {
        let codes = canonical_codes(spec, offset)?;
        let mut table = Self {
            lookup: [0; 1 << LOOKUP_BITS],
            fast_ac: [0; 1 << LOOKUP_BITS],
            max_code: [-1; 17],
            val_offset: [0; 17],
            symbols: [0; 256],
            count: codes.len(),
        };
        table.symbols[..codes.len()].copy_from_slice(&spec.symbols);
        let mut index = 0_usize;
        for len in 1..=16_u8 {
            let n = usize::from(spec.counts[usize::from(len - 1)]);
            if n > 0 {
                let first = codes[index].1;
                table.val_offset[usize::from(len)] = index as i32 - i32::from(first);
                table.max_code[usize::from(len)] = i32::from(codes[index + n - 1].1);
            }
            index += n;
        }
        for (i, &(len, code)) in codes.iter().enumerate() {
            let len = u32::from(len);
            if len <= LOOKUP_BITS {
                let shift = LOOKUP_BITS - len;
                let start = usize::from(code) << shift;
                let entry = (len as u16) << 8 | u16::from(spec.symbols[i]);
                for slot in &mut table.lookup[start..start + (1 << shift)] {
                    *slot = entry;
                }
            }
        }
        for (prefix, &entry) in table.lookup.iter().enumerate() {
            let (len, symbol) = (u32::from(entry >> 8), entry & 0xFF);
            let (run, size) = (i32::from(symbol >> 4), u32::from(symbol & 15));
            if entry != 0 && size != 0 && len + size <= LOOKUP_BITS {
                let bits = (prefix as i32 >> (LOOKUP_BITS - len - size)) & ((1 << size) - 1);
                let value = if bits < 1 << (size - 1) { bits - (1 << size) + 1 } else { bits };
                table.fast_ac[prefix] = value << 16 | run << 8 | (len + size) as i32;
            }
        }
        Ok(table)
    }
}

/// Reads bits from an entropy-coded segment, most significant first.
///
/// A `0xFF 0x00` pair is one `0xFF` data byte. At a marker (or the end of the input) the reader stops
/// and feeds zero bits; consuming any of them sets [`BitReader::overrun`], which the decoder turns
/// into [`Error::Truncated`]. Every operation is bounded, so corrupt data costs at most one bit per
/// call and never an endless loop.
#[derive(Clone, Debug)]
pub(super) struct BitReader<'a> {
    data: &'a [u8],
    /// The next byte to load.
    pos: usize,
    /// Buffered bits, left-aligned.
    bits: u64,
    /// How many bits of `bits` are valid.
    count: u32,
    /// How many of the buffered bits are zero padding past a marker or the end.
    padding: u32,
    /// Whether loading stopped at a marker (`pos` is then at its first `0xFF`) or the end.
    stopped: bool,
    /// Whether padding bits were consumed.
    overrun: bool,
}

impl<'a> BitReader<'a> {
    /// A reader of `data` starting at byte `pos`.
    pub(super) const fn new(data: &'a [u8], pos: usize) -> Self {
        Self { data, pos, bits: 0, count: 0, padding: 0, stopped: false, overrun: false }
    }

    /// Whether the reader has consumed bits past the end of its segment.
    pub(super) const fn overrun(&self) -> bool {
        self.overrun
    }

    /// Loads bytes until at least 57 bits are buffered.
    #[inline]
    fn fill(&mut self) {
        // Fast path: the next eight bytes hold no `0xFF` (no stuffing, no marker), so the bytes
        // needed are loaded at once.
        if !self.stopped
            && let Some(next) = self.data.get(self.pos..self.pos + 8)
            && let Ok(next) = <[u8; 8]>::try_from(next)
        {
            let word = u64::from_be_bytes(next);
            // A byte of `!word` is zero exactly where `word` has 0xFF (the classic zero-byte test).
            let inv = !word;
            if inv.wrapping_sub(0x0101_0101_0101_0101) & !inv & 0x8080_8080_8080_8080 == 0 {
                // 1 to 8 whole bytes fit (count <= 56 here when called, but stay general).
                let take = (64 - self.count) / 8;
                if take > 0 {
                    let bits = 8 * take;
                    self.bits |= (word >> (64 - bits)) << (64 - bits - self.count);
                    self.count += bits;
                    self.pos += take as usize;
                }
                return;
            }
        }
        while self.count <= 56 {
            let mut byte = 0_u8;
            if self.stopped {
                self.padding += 8;
            } else {
                match self.data.get(self.pos..) {
                    Some([0xFF, 0x00, ..]) => {
                        byte = 0xFF;
                        self.pos += 2;
                    }
                    Some([0xFF, ..] | []) | None => {
                        self.stopped = true;
                        self.padding += 8;
                    }
                    Some([b, ..]) => {
                        byte = *b;
                        self.pos += 1;
                    }
                }
            }
            self.bits |= u64::from(byte) << (56 - self.count);
            self.count += 8;
        }
    }

    /// Drops `n` buffered bits (`n <= self.count`).
    #[inline]
    fn consume(&mut self, n: u32) {
        self.bits <<= n;
        self.count -= n;
        if self.count < self.padding {
            self.overrun = true;
            self.padding = self.count;
        }
    }

    /// Reads `n` bits (0 to 16) as an unsigned number.
    #[inline]
    pub(super) fn bits(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        if self.count < n {
            self.fill();
        }
        let v = (self.bits >> (64 - n)) as u32;
        self.consume(n);
        v
    }

    /// Reads one bit.
    #[inline]
    pub(super) fn bit(&mut self) -> bool {
        self.bits(1) != 0
    }

    /// Reads an `s`-bit magnitude and sign-extends it as T.81 F.2.2.1 `EXTEND` does (`s` up to 15).
    #[inline]
    pub(super) fn receive_extend(&mut self, s: u32) -> i32 {
        if s == 0 {
            return 0;
        }
        let v = self.bits(s) as i32;
        if v < 1 << (s - 1) { v - (1 << s) + 1 } else { v }
    }

    /// Decodes an AC run/size code and its value at once when both fit in the lookup prefix:
    /// `(run, value)`. `None` (nothing consumed) when the slower [`Self::decode`] path is needed.
    #[inline]
    pub(super) fn ac_fast(&mut self, table: &DecodeTable) -> Option<(usize, i32)> {
        if self.count < 16 {
            self.fill();
        }
        let f = table.fast_ac[(self.bits >> (64 - LOOKUP_BITS)) as usize];
        if f == 0 {
            return None;
        }
        self.consume((f & 0xFF) as u32);
        Some((((f >> 8) & 0xFF) as usize, f >> 16))
    }

    /// Decodes one symbol.
    #[inline]
    pub(super) fn decode(&mut self, table: &DecodeTable) -> Result<u8> {
        if self.count < 16 {
            self.fill();
        }
        let entry = table.lookup[(self.bits >> (64 - LOOKUP_BITS)) as usize];
        if entry != 0 {
            self.consume(u32::from(entry >> 8));
            return Ok(entry as u8);
        }
        for len in LOOKUP_BITS + 1..=16 {
            let code = (self.bits >> (64 - len)) as i32;
            if code <= table.max_code[len as usize] {
                self.consume(len);
                let index = (code + table.val_offset[len as usize]) as usize;
                return match table.symbols.get(index) {
                    Some(&s) if index < table.count => Ok(s),
                    _ => Err(self.invalid("bad Huffman code")),
                };
            }
        }
        Err(self.invalid("bad Huffman code"))
    }

    /// An [`Error::Invalid`] at the reader's position.
    pub(super) const fn invalid(&self, what: &'static str) -> Error {
        Error::Invalid { format: FORMAT, offset: self.pos, what }
    }

    /// Moves to the next marker, skipping whatever entropy-coded bytes are left, and returns its
    /// code with the position of its first `0xFF` (fill bytes `0xFF` before it are skipped too).
    /// `None` when the input ends first. Buffered bits are dropped.
    pub(super) fn next_marker(&mut self) -> Option<(u8, usize)> {
        self.bits = 0;
        self.count = 0;
        self.padding = 0;
        self.stopped = false;
        let mut i = self.pos;
        while i + 1 < self.data.len() {
            if self.data[i] == 0xFF {
                let code = self.data[i + 1];
                if code != 0x00 && code != 0xFF {
                    self.pos = i;
                    return Some((code, i));
                }
                // A stuffed zero is data; a second 0xFF is fill before the marker.
                i += if code == 0x00 { 2 } else { 1 };
            } else {
                i += 1;
            }
        }
        self.pos = self.data.len();
        None
    }

    /// Handles a restart marker between intervals: the next marker must be `RST(n mod 8)`; the
    /// reader then continues after it with an empty bit buffer.
    pub(super) fn restart(&mut self, n: u32) -> Result<()> {
        if self.overrun {
            return Err(Error::Truncated { format: FORMAT, offset: self.pos });
        }
        match self.next_marker() {
            Some((code, at)) if u32::from(code) == 0xD0 + (n & 7) => {
                self.pos = at + 2;
                Ok(())
            }
            Some(_) => Err(self.invalid("missing or out-of-order restart marker")),
            None => Err(Error::Truncated { format: FORMAT, offset: self.pos }),
        }
    }
}

/// A Huffman table ready for encoding: the code and its length for every symbol.
#[derive(Clone, Debug)]
pub(super) struct EncodeTable {
    code: [u16; 256],
    len: [u8; 256],
}

impl EncodeTable {
    /// Builds the encoding table of a specification (one of the Annex K tables in practice).
    pub(super) fn new(spec: &HuffSpec) -> Result<Self> {
        let codes = canonical_codes(spec, 0)?;
        let mut table = Self { code: [0; 256], len: [0; 256] };
        for (&symbol, &(len, code)) in spec.symbols.iter().zip(&codes) {
            table.code[usize::from(symbol)] = code;
            table.len[usize::from(symbol)] = len;
        }
        Ok(table)
    }
}

/// Writes bits most significant first, stuffing a zero byte after every `0xFF`.
#[derive(Debug)]
pub(super) struct BitWriter<'o> {
    out: &'o mut Vec<u8>,
    acc: u64,
    count: u32,
}

impl<'o> BitWriter<'o> {
    /// A writer appending to `out`.
    pub(super) const fn new(out: &'o mut Vec<u8>) -> Self {
        Self { out, acc: 0, count: 0 }
    }

    /// Appends the low `n` bits of `v` (`n` up to 24).
    #[inline]
    pub(super) fn put(&mut self, v: u32, n: u32) {
        self.acc = self.acc << n | u64::from(v & ((1 << n) - 1));
        self.count += n;
        while self.count >= 8 {
            self.count -= 8;
            let byte = (self.acc >> self.count) as u8;
            self.out.push(byte);
            if byte == 0xFF {
                self.out.push(0);
            }
        }
    }

    /// Appends the code of `symbol`.
    #[inline]
    pub(super) fn symbol(&mut self, table: &EncodeTable, symbol: u8) {
        let s = usize::from(symbol);
        self.put(u32::from(table.code[s]), u32::from(table.len[s]));
    }

    /// Pads the last byte with one bits (T.81 F.1.2.3).
    pub(super) fn flush(&mut self) {
        if self.count > 0 {
            let pad = 8 - self.count;
            self.put((1 << pad) - 1, pad);
        }
        self.acc = 0;
    }

    /// The output written so far.
    pub(super) fn out(&mut self) -> &mut Vec<u8> {
        self.out
    }
}

/// The number of bits of the magnitude of `v` (T.81 F.1.2.1 `SSSS`): 0 for 0.
#[inline]
pub(super) const fn magnitude_bits(v: i32) -> u32 {
    32 - v.unsigned_abs().leading_zeros()
}

/// The `s` extra bits that code `v` after its size category: `v` itself when positive, `v - 1` (in
/// two's complement, low bits) when negative.
#[inline]
pub(super) const fn magnitude_code(v: i32) -> u32 {
    if v < 0 { (v - 1) as u32 } else { v as u32 }
}

#[cfg(test)]
mod tests {
    use super::super::tables::std_huffman;
    use super::*;

    #[test]
    fn standard_tables_build_and_round_trip_every_symbol() {
        for (ac, chroma) in [(false, false), (false, true), (true, false), (true, true)] {
            let spec = std_huffman(ac, chroma);
            let dec = DecodeTable::new(&spec, 0).unwrap();
            let enc = EncodeTable::new(&spec).unwrap();
            let mut out = Vec::new();
            let mut w = BitWriter::new(&mut out);
            for &s in &spec.symbols {
                assert!(enc.len[usize::from(s)] != 0);
                w.symbol(&enc, s);
            }
            w.flush();
            let mut r = BitReader::new(&out, 0);
            for &s in &spec.symbols {
                assert_eq!(r.decode(&dec).unwrap(), s);
            }
            assert!(!r.overrun());
        }
    }

    #[test]
    fn long_codes_take_the_slow_path() {
        // One code of each length 1..=16 except the all-ones code: symbols 1..=16 by length.
        let mut counts = [1_u8; 16];
        counts[15] = 1;
        let spec = HuffSpec { counts, symbols: (1..=16).collect() };
        let dec = DecodeTable::new(&spec, 0).unwrap();
        let enc = EncodeTable::new(&spec).unwrap();
        let mut out = Vec::new();
        let mut w = BitWriter::new(&mut out);
        let order = [16_u8, 3, 12, 1, 10, 9, 15, 2];
        for &s in &order {
            w.symbol(&enc, s);
        }
        w.flush();
        let mut r = BitReader::new(&out, 0);
        for &s in &order {
            assert_eq!(r.decode(&dec).unwrap(), s);
        }
    }

    #[test]
    fn bad_tables_and_codes_are_refused() {
        // Three codes of length 1 cannot exist.
        let spec = HuffSpec { counts: [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], symbols: vec![0, 1, 2] };
        assert!(matches!(DecodeTable::new(&spec, 7), Err(Error::Invalid { offset: 7, .. })));
        let spec = HuffSpec { counts: [0; 16], symbols: vec![1] };
        assert!(DecodeTable::new(&spec, 0).is_err());
        // A table with only "0" leaves "1..." undecodable.
        let mut counts = [0_u8; 16];
        counts[0] = 1;
        let dec = DecodeTable::new(&HuffSpec { counts, symbols: vec![5] }, 0).unwrap();
        let data = [0x7F_u8, 0xD9];
        let mut r = BitReader::new(&data, 0);
        assert_eq!(r.decode(&dec).unwrap(), 5);
        assert!(r.decode(&dec).is_err());
    }

    #[test]
    fn stuffing_markers_and_padding() {
        // 0xFF 0x00 is one data byte; reading into the marker sets the overrun flag.
        let data = [0xFF, 0x00, 0xA5, 0xFF, 0xD0, 0x12];
        let mut r = BitReader::new(&data, 0);
        assert_eq!(r.bits(8), 0xFF);
        assert_eq!(r.bits(8), 0xA5);
        assert!(!r.overrun());
        assert_eq!(r.bits(4), 0);
        assert!(r.overrun());
        assert_eq!(r.next_marker(), Some((0xD0, 3)));
        // Restart: the expected RST is found after leftover bits, the wrong one is refused.
        let mut r = BitReader::new(&data, 0);
        r.bits(3);
        r.restart(0).unwrap();
        assert_eq!(r.bits(8), 0x12);
        let mut r = BitReader::new(&data, 0);
        assert!(matches!(r.restart(1), Err(Error::Invalid { .. })));
        let mut r = BitReader::new(&[0x00, 0x01], 0);
        assert!(matches!(r.restart(0), Err(Error::Truncated { .. })));
        // Fill bytes before a marker are skipped.
        let mut r = BitReader::new(&[0x01, 0xFF, 0xFF, 0xD9], 0);
        assert_eq!(r.next_marker(), Some((0xD9, 2)));
    }

    #[test]
    fn extend_and_magnitudes_are_inverse() {
        for v in [-2047, -1024, -1023, -2, -1, 1, 2, 3, 255, 1023, 2047] {
            let s = magnitude_bits(v);
            let mut out = Vec::new();
            let mut w = BitWriter::new(&mut out);
            w.put(magnitude_code(v), s);
            w.flush();
            assert_eq!(BitReader::new(&out, 0).receive_extend(s), v, "{v}");
        }
        assert_eq!(magnitude_bits(0), 0);
        assert_eq!(magnitude_bits(-1), 1);
        assert_eq!(magnitude_bits(1024), 11);
    }

    #[test]
    fn the_writer_stuffs_and_pads_with_ones() {
        let mut out = Vec::new();
        let mut w = BitWriter::new(&mut out);
        w.put(0xFF, 8);
        w.put(0b101, 3);
        w.flush();
        assert_eq!(w.out().as_slice(), &[0xFF, 0x00, 0b1011_1111]);
    }
}
