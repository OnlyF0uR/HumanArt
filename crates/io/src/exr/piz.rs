//! OpenEXR's PIZ compression (`ImfPizCompressor.cpp` and `ImfWav.cpp` of the reference library),
//! written from the format's description `[DEP-05]`.
//!
//! A chunk is split into 16-bit words, gathered per channel (a `half` is one word, a `float` or
//! `uint` two, low word first). The words that occur are recorded in a 65 536-bit bitmap and
//! renumbered densely through a lookup table, each channel goes through a 2D Haar wavelet transform
//! (exact in 14-bit arithmetic when the renumbered range allows it, modulo 2^16 otherwise), and the
//! result is Huffman coded ([`super::huf`]).
//!
//! Layout: `min` and `max` (`u16`, the first and last non-zero bitmap bytes), the bitmap bytes
//! `min..=max` (none if `min > max`), an `i32` length and the Huffman block.

use super::huf;
use crate::bytes::Reader;
use crate::error::{Error, Format, Result};

const FORMAT: Format = Format::Exr;
/// Bytes in the bitmap of the 65 536 possible 16-bit words.
const BITMAP_SIZE: usize = 8192;
/// The number of distinct 16-bit words.
const USHORT_RANGE: usize = 1 << 16;

/// The shape of one chunk: every channel has `width` samples on each of `lines` lines.
pub(super) struct Layout<'a> {
    /// Samples per line of every channel.
    pub width: usize,
    /// Lines in the chunk.
    pub lines: usize,
    /// Each channel's sample size in 16-bit words (1 for `half`, 2 for `float` and `uint`), in the
    /// order the channels are stored.
    pub words: &'a [usize],
}

/// Buffers reused for every chunk of a file.
pub(super) struct Scratch {
    words: Vec<u16>,
    lut: Vec<u16>,
    bitmap: Vec<u8>,
    huf: huf::Decoder,
}

impl Scratch {
    /// Buffers for chunks of up to `words` 16-bit words.
    pub(super) fn new(words: usize) -> Self {
        Self {
            words: Vec::with_capacity(words),
            lut: vec![0; USHORT_RANGE],
            bitmap: vec![0; BITMAP_SIZE],
            huf: huf::Decoder::new(),
        }
    }
}

/// `(l, h)` of the 14-bit Haar step: the floor of the mean and the difference, in signed 16 bits.
#[inline]
const fn wenc14(a: u16, b: u16) -> (u16, u16) {
    let (a, b) = (a as i16 as i32, b as i16 as i32);
    (((a + b) >> 1) as u16, (a - b) as u16)
}

/// The inverse of [`wenc14`].
#[inline]
const fn wdec14(l: u16, h: u16) -> (u16, u16) {
    let (l, h) = (l as i16 as i32, h as i16 as i32);
    let a = l + (h & 1) + (h >> 1);
    (a as u16, (a - h) as u16)
}

/// Offset and mask of the 16-bit Haar step.
const A_OFFSET: i32 = 1 << 15;
const M_OFFSET: i32 = 1 << 15;
const MOD_MASK: i32 = (1 << 16) - 1;

/// The Haar step modulo 2^16, for words whose range does not fit 14 bits.
#[inline]
const fn wenc16(a: u16, b: u16) -> (u16, u16) {
    let ao = (a as i32 + A_OFFSET) & MOD_MASK;
    let b = b as i32;
    let mut m = (ao + b) >> 1;
    let d = ao - b;
    if d < 0 {
        m = (m + M_OFFSET) & MOD_MASK;
    }
    (m as u16, (d & MOD_MASK) as u16)
}

/// The inverse of [`wenc16`].
#[inline]
const fn wdec16(l: u16, h: u16) -> (u16, u16) {
    let (m, d) = (l as i32, h as i32);
    let b = (m - (d >> 1)) & MOD_MASK;
    let a = (d + b - A_OFFSET) & MOD_MASK;
    (a as u16, b as u16)
}

/// One Haar step on two words of `buf`, in place: `(buf[i], buf[j]) = step(buf[i], buf[j])`.
#[inline]
fn pair(buf: &mut [u16], i: usize, j: usize, step: fn(u16, u16) -> (u16, u16)) {
    let (l, h) = step(buf[i], buf[j]);
    buf[i] = l;
    buf[j] = h;
}

/// The forward 2D wavelet transform of an `nx × ny` array whose element `(x, y)` is
/// `buf[x · ox + y · oy]` (the reference's `wav2Encode`). `max` is the largest word.
fn wav2_encode(buf: &mut [u16], nx: usize, ox: usize, ny: usize, oy: usize, max: u16) {
    let step: fn(u16, u16) -> (u16, u16) = if max < 1 << 14 { wenc14 } else { wenc16 };
    let n = nx.min(ny);
    let (mut p, mut p2) = (1, 2);
    while p2 <= n {
        let (ox1, oy1) = (ox * p, oy * p);
        let mut y = 0;
        while y + p2 <= ny {
            let mut x = 0;
            while x + p2 <= nx {
                let p00 = y * oy + x * ox;
                let (p01, p10) = (p00 + ox1, p00 + oy1);
                let p11 = p10 + ox1;
                pair(buf, p00, p01, step);
                pair(buf, p10, p11, step);
                pair(buf, p00, p10, step);
                pair(buf, p01, p11, step);
                x += p2;
            }
            if nx & p != 0 {
                let p00 = y * oy + x * ox;
                pair(buf, p00, p00 + oy1, step);
            }
            y += p2;
        }
        if ny & p != 0 {
            let mut x = 0;
            while x + p2 <= nx {
                let p00 = y * oy + x * ox;
                pair(buf, p00, p00 + ox1, step);
                x += p2;
            }
        }
        p = p2;
        p2 <<= 1;
    }
}

/// The inverse of [`wav2_encode`] (the reference's `wav2Decode`).
fn wav2_decode(buf: &mut [u16], nx: usize, ox: usize, ny: usize, oy: usize, max: u16) {
    let step: fn(u16, u16) -> (u16, u16) = if max < 1 << 14 { wdec14 } else { wdec16 };
    let n = nx.min(ny);
    let mut p = 1;
    while p <= n {
        p <<= 1;
    }
    p >>= 1;
    let mut p2 = p;
    p >>= 1;
    while p >= 1 {
        let (ox1, oy1) = (ox * p, oy * p);
        let mut y = 0;
        while y + p2 <= ny {
            let mut x = 0;
            while x + p2 <= nx {
                let p00 = y * oy + x * ox;
                let (p01, p10) = (p00 + ox1, p00 + oy1);
                let p11 = p10 + ox1;
                pair(buf, p00, p10, step);
                pair(buf, p01, p11, step);
                pair(buf, p00, p01, step);
                pair(buf, p10, p11, step);
                x += p2;
            }
            if nx & p != 0 {
                let p00 = y * oy + x * ox;
                pair(buf, p00, p00 + oy1, step);
            }
            y += p2;
        }
        if ny & p != 0 {
            let mut x = 0;
            while x + p2 <= nx {
                let p00 = y * oy + x * ox;
                pair(buf, p00, p00 + ox1, step);
                x += p2;
            }
        }
        p2 = p;
        p >>= 1;
    }
}

/// Runs `transform` over every channel of the gathered words, once per word of a sample.
fn transform_channels(
    words: &mut [u16],
    layout: &Layout<'_>,
    max: u16,
    transform: fn(&mut [u16], usize, usize, usize, usize, u16),
) {
    let mut start = 0;
    for &size in layout.words {
        let len = layout.width * layout.lines * size;
        let channel = &mut words[start..start + len];
        for j in 0..size {
            transform(&mut channel[j..], layout.width, size, layout.lines, layout.width * size, max);
        }
        start += len;
    }
}

/// Decompresses one PIZ chunk into `out`, the chunk's uncompressed bytes (lines of channels of
/// little-endian samples). `base` is the file offset of `input`.
pub(super) fn decompress(
    input: &[u8],
    base: usize,
    layout: &Layout<'_>,
    out: &mut [u8],
    scratch: &mut Scratch,
) -> Result<()> {
    let mut r = Reader::new(input, FORMAT);
    let shift = |e: Error| match e {
        Error::Truncated { format, offset } => Error::Truncated { format, offset: offset + base },
        Error::Invalid { format, offset, what } => Error::Invalid { format, offset: offset + base, what },
        other => other,
    };
    let min = usize::from(r.u16_le().map_err(shift)?);
    let max = usize::from(r.u16_le().map_err(shift)?);
    if max >= BITMAP_SIZE {
        return Err(Error::Invalid { format: FORMAT, offset: base + 2, what: "PIZ bitmap range is out of bounds" });
    }
    scratch.bitmap.fill(0);
    if min <= max {
        let bytes = r.bytes(max - min + 1).map_err(shift)?;
        scratch.bitmap[min..=max].copy_from_slice(bytes);
    }
    // The reverse table: dense index -> word; zero is always present.
    let mut k = 0;
    for word in 0..USHORT_RANGE {
        if word == 0 || scratch.bitmap[word >> 3] & (1 << (word & 7)) != 0 {
            scratch.lut[k] = word as u16;
            k += 1;
        }
    }
    let max_value = (k - 1) as u16;
    scratch.lut[k..].fill(0);

    let at = r.pos();
    let length = r.i32_le().map_err(shift)?;
    let Ok(length) = usize::try_from(length) else {
        return Err(Error::Invalid { format: FORMAT, offset: base + at, what: "negative PIZ Huffman length" });
    };
    let block_at = base + r.pos();
    let block = r.bytes(length).map_err(shift)?;
    let n_words = out.len() / 2;
    scratch.words.clear();
    scratch.words.resize(n_words, 0);
    scratch.huf.decompress(block, block_at, &mut scratch.words)?;

    transform_channels(&mut scratch.words, layout, max_value, wav2_decode);
    for word in &mut scratch.words {
        *word = scratch.lut[usize::from(*word)];
    }

    // Channels are stored one after another; the output interleaves them line by line.
    let mut o = 0;
    for line in 0..layout.lines {
        let mut start = 0;
        for &size in layout.words {
            let n = layout.width * size;
            let row = &scratch.words[start + line * n..start + (line + 1) * n];
            for &word in row {
                out[o..o + 2].copy_from_slice(&word.to_le_bytes());
                o += 2;
            }
            start += n * layout.lines;
        }
    }
    Ok(())
}

/// Appends one PIZ-compressed chunk made from `raw`, the chunk's uncompressed bytes.
pub(super) fn compress(raw: &[u8], layout: &Layout<'_>, out: &mut Vec<u8>) -> Result<()> {
    let n_words = raw.len() / 2;
    let mut words = vec![0_u16; n_words];
    // Gather every channel's words into one block per channel.
    let mut i = 0;
    for line in 0..layout.lines {
        let mut start = 0;
        for &size in layout.words {
            let n = layout.width * size;
            for slot in &mut words[start + line * n..start + (line + 1) * n] {
                *slot = u16::from_le_bytes([raw[i], raw[i + 1]]);
                i += 2;
            }
            start += n * layout.lines;
        }
    }

    let mut bitmap = vec![0_u8; BITMAP_SIZE];
    for &word in &words {
        bitmap[usize::from(word >> 3)] |= 1 << (word & 7);
    }
    // Zero is implied, never stored.
    bitmap[0] &= !1;
    let min = bitmap.iter().position(|&b| b != 0).unwrap_or(BITMAP_SIZE - 1);
    let max = bitmap.iter().rposition(|&b| b != 0).unwrap_or(0);
    let mut lut = vec![0_u16; USHORT_RANGE];
    let mut k = 0_u16;
    for (word, slot) in lut.iter_mut().enumerate() {
        if word == 0 || bitmap[word >> 3] & (1 << (word & 7)) != 0 {
            *slot = k;
            k = k.wrapping_add(1);
        }
    }
    // k is the count of distinct words (at most 65 536, which wraps to 0): the largest index is k - 1.
    let max_value = k.wrapping_sub(1);
    for word in &mut words {
        *word = lut[usize::from(*word)];
    }

    out.extend_from_slice(&(min as u16).to_le_bytes());
    out.extend_from_slice(&(max as u16).to_le_bytes());
    if min <= max {
        out.extend_from_slice(&bitmap[min..=max]);
    }
    transform_channels(&mut words, layout, max_value, wav2_encode);
    let length_at = out.len();
    out.extend_from_slice(&[0; 4]);
    huf::compress(&words, out)?;
    let length = out.len() - length_at - 4;
    let Ok(length) = i32::try_from(length) else {
        return Err(Error::Encode { format: FORMAT, what: "a PIZ chunk is larger than 2 GiB" });
    };
    out[length_at..length_at + 4].copy_from_slice(&length.to_le_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    #[test]
    fn haar_steps_invert_exactly() {
        let mut rng = Rng::new(5);
        for _ in 0..100_000 {
            let (a, b) = (rng.next_u32() as u16, rng.next_u32() as u16);
            let (l, h) = wenc16(a, b);
            assert_eq!(wdec16(l, h), (a, b));
            // The 14-bit step is exact for words below 2^14.
            let (a, b) = (a & 0x3FFF, b & 0x3FFF);
            let (l, h) = wenc14(a, b);
            assert_eq!(wdec14(l, h), (a, b));
        }
    }

    #[test]
    fn wavelet_inverts_on_every_small_shape() {
        let mut rng = Rng::new(9);
        for nx in 1..=19 {
            for ny in 1..=19 {
                for (size, max) in [(1, 0x3FFF_u16), (2, 0xFFFF)] {
                    let len = nx * ny * size;
                    let original: Vec<u16> = (0..len).map(|_| rng.next_u32() as u16 & max).collect();
                    let mut buf = original.clone();
                    for j in 0..size {
                        wav2_encode(&mut buf[j..], nx, size, ny, nx * size, max);
                    }
                    for j in 0..size {
                        wav2_decode(&mut buf[j..], nx, size, ny, nx * size, max);
                    }
                    assert_eq!(buf, original, "{nx}x{ny}x{size}");
                }
            }
        }
    }

    #[test]
    fn chunks_round_trip_and_the_bitmap_is_bounded() {
        let mut rng = Rng::new(2);
        let words = [1_usize, 2, 1];
        for (width, lines) in [(1, 1), (7, 3), (37, 32), (64, 5)] {
            let layout = Layout { width, lines, words: &words };
            let bytes = width * lines * 8;
            // Smooth data (the wavelet's case), random data and all zeros.
            let smooth: Vec<u8> = (0..bytes).map(|i| ((i / 8) % 7) as u8).collect();
            let noise: Vec<u8> = (0..bytes).map(|_| rng.next_u32() as u8).collect();
            for raw in [smooth, noise, vec![0; bytes]] {
                let mut packed = Vec::new();
                compress(&raw, &layout, &mut packed).unwrap();
                let mut back = vec![0; bytes];
                decompress(&packed, 0, &layout, &mut back, &mut Scratch::new(bytes / 2)).unwrap();
                assert_eq!(back, raw, "{width}x{lines}");
            }
        }
        let layout = Layout { width: 1, lines: 1, words: &[1] };
        let mut out = [0; 2];
        let mut scratch = Scratch::new(1);
        let bad = [0, 0, 0x00, 0x20, 0, 0, 0, 0];
        assert!(matches!(
            decompress(&bad, 0, &layout, &mut out, &mut scratch),
            Err(Error::Invalid { what: "PIZ bitmap range is out of bounds", .. })
        ));
        let negative = [0xFF, 0x1F, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF];
        assert!(decompress(&negative, 0, &layout, &mut out, &mut scratch).is_err());
    }
}
