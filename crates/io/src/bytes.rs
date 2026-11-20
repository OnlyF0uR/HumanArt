//! Checked reading and plain writing of binary data.
//!
//! [`Reader`] never indexes past its input: every read returns [`Error::Truncated`] instead, which is
//! what lets the codecs survive hostile files `[ROB-01]`. Both byte orders are explicit in the method
//! names; nothing here depends on the host's endianness.

use crate::error::{Error, Format, Result};

/// A cursor over a byte slice whose every read is bounds-checked.
#[derive(Clone, Debug)]
pub struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    format: Format,
}

macro_rules! read_int {
    ($($le:ident, $be:ident, $ty:ty, $n:literal, $doc:literal;)*) => {$(
        #[doc = concat!("Reads a little-endian ", $doc, ".")]
        #[inline]
        pub fn $le(&mut self) -> Result<$ty> {
            let b = self.array::<$n>()?;
            Ok(<$ty>::from_le_bytes(b))
        }

        #[doc = concat!("Reads a big-endian ", $doc, ".")]
        #[inline]
        pub fn $be(&mut self) -> Result<$ty> {
            let b = self.array::<$n>()?;
            Ok(<$ty>::from_be_bytes(b))
        }
    )*};
}

impl<'a> Reader<'a> {
    /// A reader at the start of `data`; `format` labels the errors it reports.
    #[must_use]
    pub const fn new(data: &'a [u8], format: Format) -> Self {
        Self { data, pos: 0, format }
    }

    /// The current byte offset.
    #[must_use]
    pub const fn pos(&self) -> usize {
        self.pos
    }

    /// How many bytes are left.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    /// Whether every byte has been read.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.pos >= self.data.len()
    }

    /// The whole input, regardless of the position.
    #[must_use]
    pub const fn input(&self) -> &'a [u8] {
        self.data
    }

    /// The unread rest of the input, without consuming it.
    #[must_use]
    pub fn rest(&self) -> &'a [u8] {
        &self.data[self.pos..]
    }

    /// The error for "more input was needed here".
    #[must_use]
    pub const fn truncated(&self) -> Error {
        Error::Truncated { format: self.format, offset: self.pos }
    }

    /// An [`Error::Invalid`] at the current position.
    #[must_use]
    pub const fn invalid(&self, what: &'static str) -> Error {
        Error::Invalid { format: self.format, offset: self.pos, what }
    }

    /// Moves to an absolute offset, which must lie within the input (its end included).
    pub fn seek(&mut self, pos: usize) -> Result<()> {
        if pos > self.data.len() {
            return Err(Error::Truncated { format: self.format, offset: self.data.len() });
        }
        self.pos = pos;
        Ok(())
    }

    /// Skips `n` bytes.
    pub fn skip(&mut self, n: usize) -> Result<()> {
        self.bytes(n).map(|_| ())
    }

    /// Reads the next `n` bytes as a slice of the input.
    #[inline]
    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.remaining() {
            return Err(self.truncated());
        }
        let out = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    /// Reads the next `N` bytes as an array.
    #[inline]
    pub fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let slice = self.bytes(N)?;
        let mut out = [0_u8; N];
        out.copy_from_slice(slice);
        Ok(out)
    }

    /// Looks at the next byte without consuming it.
    #[inline]
    pub fn peek(&self) -> Option<u8> {
        self.data.get(self.pos).copied()
    }

    /// Reads one byte.
    #[inline]
    pub fn u8(&mut self) -> Result<u8> {
        let b = *self.data.get(self.pos).ok_or(Error::Truncated { format: self.format, offset: self.pos })?;
        self.pos += 1;
        Ok(b)
    }

    /// Reads one signed byte.
    #[inline]
    pub fn i8(&mut self) -> Result<i8> {
        self.u8().map(|b| b as i8)
    }

    read_int! {
        u16_le, u16_be, u16, 2, "`u16`";
        i16_le, i16_be, i16, 2, "`i16`";
        u32_le, u32_be, u32, 4, "`u32`";
        i32_le, i32_be, i32, 4, "`i32`";
        u64_le, u64_be, u64, 8, "`u64`";
        i64_le, i64_be, i64, 8, "`i64`";
        f32_le, f32_be, f32, 4, "`f32`";
        f64_le, f64_be, f64, 8, "`f64`";
    }

    /// Reads a little-endian `u32` that is used as a size or count, as `usize`.
    #[inline]
    pub fn len_u32_le(&mut self) -> Result<usize> {
        self.u32_le().map(|v| v as usize)
    }

    /// Reads a little-endian `u64` that is used as a size or offset; fails if it cannot index memory
    /// on this platform.
    #[inline]
    pub fn len_u64_le(&mut self) -> Result<usize> {
        let at = self.pos;
        let v = self.u64_le()?;
        usize::try_from(v).map_err(|_| Error::Invalid {
            format: self.format,
            offset: at,
            what: "size does not fit in memory",
        })
    }

    /// A reader over the next `n` bytes, which are consumed from this one. Offsets reported by the
    /// sub-reader are relative to its own start.
    pub fn sub(&mut self, n: usize) -> Result<Reader<'a>> {
        let format = self.format;
        self.bytes(n).map(|slice| Reader::new(slice, format))
    }

    /// Checks that `count` items of `item_size` bytes fit in the remaining input, before a caller
    /// allocates for them. Returns the total byte length.
    pub fn check_count(&self, count: usize, item_size: usize) -> Result<usize> {
        match count.checked_mul(item_size) {
            Some(total) if total <= self.remaining() => Ok(total),
            _ => Err(self.truncated()),
        }
    }
}

/// An append-only byte buffer with explicit byte orders.
#[derive(Clone, Debug, Default)]
pub struct Writer {
    buf: Vec<u8>,
}

macro_rules! write_int {
    ($($le:ident, $be:ident, $ty:ty, $doc:literal;)*) => {$(
        #[doc = concat!("Appends a little-endian ", $doc, ".")]
        #[inline]
        pub fn $le(&mut self, v: $ty) {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }

        #[doc = concat!("Appends a big-endian ", $doc, ".")]
        #[inline]
        pub fn $be(&mut self, v: $ty) {
            self.buf.extend_from_slice(&v.to_be_bytes());
        }
    )*};
}

impl Writer {
    /// An empty writer.
    #[must_use]
    pub const fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// An empty writer with room for `capacity` bytes.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self { buf: Vec::with_capacity(capacity) }
    }

    /// How many bytes have been written.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether nothing has been written.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// What has been written so far.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.buf
    }

    /// Hands over the buffer.
    #[must_use]
    pub fn into_vec(self) -> Vec<u8> {
        self.buf
    }

    /// Appends one byte.
    #[inline]
    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    /// Appends raw bytes.
    #[inline]
    pub fn bytes(&mut self, v: &[u8]) {
        self.buf.extend_from_slice(v);
    }

    write_int! {
        u16_le, u16_be, u16, "`u16`";
        i16_le, i16_be, i16, "`i16`";
        u32_le, u32_be, u32, "`u32`";
        i32_le, i32_be, i32, "`i32`";
        u64_le, u64_be, u64, "`u64`";
        i64_le, i64_be, i64, "`i64`";
        f32_le, f32_be, f32, "`f32`";
        f64_le, f64_be, f64, "`f64`";
    }

    /// Appends zero bytes until the length is a multiple of `align` (which must be non-zero).
    pub fn pad_to(&mut self, align: usize) {
        if align > 1 {
            let rem = self.buf.len() % align;
            if rem != 0 {
                self.buf.resize(self.buf.len() + align - rem, 0);
            }
        }
    }

    /// Overwrites four bytes at `at` with a little-endian `u32` (for sizes known only after the
    /// data is written). Does nothing if the range is not inside what was written.
    pub fn patch_u32_le(&mut self, at: usize, v: u32) {
        if let Some(slot) = self.buf.get_mut(at..at.saturating_add(4)) {
            slot.copy_from_slice(&v.to_le_bytes());
        }
    }

    /// Overwrites eight bytes at `at` with a little-endian `u64`. Does nothing if out of range.
    pub fn patch_u64_le(&mut self, at: usize, v: u64) {
        if let Some(slot) = self.buf.get_mut(at..at.saturating_add(8)) {
            slot.copy_from_slice(&v.to_le_bytes());
        }
    }

    /// Overwrites four bytes at `at` with a big-endian `u32`. Does nothing if out of range.
    pub fn patch_u32_be(&mut self, at: usize, v: u32) {
        if let Some(slot) = self.buf.get_mut(at..at.saturating_add(4)) {
            slot.copy_from_slice(&v.to_be_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_every_width_in_both_orders() {
        let mut w = Writer::new();
        w.u8(0xAB);
        w.u16_le(0x1234);
        w.u16_be(0x1234);
        w.i32_le(-5);
        w.u32_be(0xDEAD_BEEF);
        w.u64_le(u64::MAX - 1);
        w.i64_be(i64::MIN);
        w.f32_le(1.5);
        w.f64_be(-2.25);
        w.bytes(b"xyz");
        let bytes = w.into_vec();
        let mut r = Reader::new(&bytes, Format::Chunk);
        assert_eq!(r.u8().unwrap(), 0xAB);
        assert_eq!(r.u16_le().unwrap(), 0x1234);
        assert_eq!(r.u16_be().unwrap(), 0x1234);
        assert_eq!(r.i32_le().unwrap(), -5);
        assert_eq!(r.u32_be().unwrap(), 0xDEAD_BEEF);
        assert_eq!(r.u64_le().unwrap(), u64::MAX - 1);
        assert_eq!(r.i64_be().unwrap(), i64::MIN);
        assert_eq!(r.f32_le().unwrap(), 1.5);
        assert_eq!(r.f64_be().unwrap(), -2.25);
        assert_eq!(r.bytes(3).unwrap(), b"xyz");
        assert!(r.is_empty());
    }

    #[test]
    fn every_read_past_the_end_is_an_error_at_the_right_offset() {
        let data = [1_u8, 2, 3];
        let mut r = Reader::new(&data, Format::Png);
        assert_eq!(r.u16_le().unwrap(), 0x0201);
        assert_eq!(r.u32_le(), Err(Error::Truncated { format: Format::Png, offset: 2 }));
        assert_eq!(r.pos(), 2, "a failed read consumes nothing");
        assert_eq!(r.peek(), Some(3));
        assert_eq!(r.u8().unwrap(), 3);
        assert_eq!(r.u8(), Err(Error::Truncated { format: Format::Png, offset: 3 }));
        assert_eq!(r.peek(), None);
        assert!(r.bytes(usize::MAX).is_err());
        assert!(r.skip(1).is_err());
        assert!(r.seek(3).is_ok());
        assert!(r.seek(4).is_err());
    }

    #[test]
    fn counts_are_checked_before_allocation() {
        let data = [0_u8; 16];
        let r = Reader::new(&data, Format::Ktx2);
        assert_eq!(r.check_count(4, 4).unwrap(), 16);
        assert!(r.check_count(5, 4).is_err());
        assert!(r.check_count(usize::MAX, 8).is_err(), "overflow is caught");
    }

    #[test]
    fn sub_readers_and_patching() {
        let mut w = Writer::with_capacity(16);
        w.u32_le(0);
        w.bytes(b"abc");
        w.pad_to(4);
        assert_eq!(w.len(), 8);
        w.patch_u32_le(0, 7);
        w.patch_u32_le(6, 1); // out of range: ignored
        w.patch_u64_le(1, 1); // out of range: ignored
        w.u32_be(0);
        w.patch_u32_be(8, 0x0102_0304);
        let bytes = w.into_vec();
        let mut r = Reader::new(&bytes, Format::Chunk);
        assert_eq!(r.len_u32_le().unwrap(), 7);
        let mut sub = r.sub(4).unwrap();
        assert_eq!(sub.bytes(3).unwrap(), b"abc");
        assert_eq!(sub.remaining(), 1);
        assert_eq!(r.rest(), &[1, 2, 3, 4]);
        assert!(r.sub(5).is_err());
        assert_eq!(r.input().len(), 12);
        assert!(r.invalid("x").to_string().contains("at byte 8"));
    }
}
