//! Reading a 128-bit block least significant bit first.

/// A cursor over the 128 bits of one block. Reads past the end give zero bits, so a decoder that
/// trusts a mode table can never index out of range.
pub(crate) struct Bits {
    value: u128,
    pos: u32,
}

impl Bits {
    /// A cursor at bit 0 of `block` (little-endian).
    pub(crate) const fn new(block: &[u8; 16]) -> Self {
        Self { value: u128::from_le_bytes(*block), pos: 0 }
    }

    /// Skips `n` bits.
    pub(crate) fn skip(&mut self, n: u32) {
        self.pos = self.pos.saturating_add(n);
    }

    /// The current bit position.
    #[cfg(test)]
    pub(crate) const fn pos(&self) -> u32 {
        self.pos
    }

    /// Reads `n` (0 to 32) bits; the first bit read is bit 0 of the result.
    pub(crate) fn read(&mut self, n: u32) -> u32 {
        let n = n.min(32);
        let value = if n == 0 || self.pos >= 128 {
            0
        } else {
            let mask = (1_u128 << n) - 1;
            ((self.value >> self.pos) & mask) as u32
        };
        self.skip(n);
        value
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::Bits;

    /// Builds a block bit by bit, least significant bit first (test helper).
    #[derive(Default)]
    pub(crate) struct Writer {
        value: u128,
        pos: u32,
    }

    impl Writer {
        /// Appends the low `n` bits of `v`.
        pub(crate) fn put(&mut self, v: u32, n: u32) {
            assert!(self.pos + n <= 128, "block overflow");
            if n > 0 {
                let masked = u128::from(v) & ((1_u128 << n) - 1);
                self.value |= masked << self.pos;
            }
            self.pos += n;
        }

        /// The block (unwritten bits are zero).
        pub(crate) fn finish(&self) -> [u8; 16] {
            self.value.to_le_bytes()
        }

        /// Bits written so far.
        pub(crate) fn pos(&self) -> u32 {
            self.pos
        }
    }

    #[test]
    fn reads_back_what_was_written() {
        let mut w = Writer::default();
        w.put(0b101, 3);
        w.put(0xABCD, 16);
        w.put(1, 1);
        let block = w.finish();
        let mut r = Bits::new(&block);
        assert_eq!(r.read(3), 0b101);
        assert_eq!(r.read(16), 0xABCD);
        assert_eq!(r.read(1), 1);
        assert_eq!(r.pos(), 20);
        r.skip(200);
        assert_eq!(r.read(5), 0, "reads past the end are zero");
    }
}
