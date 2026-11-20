//! The two EXIF tags the decoder honours `[IO-02]`: Orientation (0x0112, in IFD0) and ColorSpace
//! (0xA001, in the Exif sub-IFD). Everything else in the APP1 segment is ignored.
//!
//! Hostile input: the TIFF structure is read through bounds-checked offsets; at most
//! [`MAX_IFD_ENTRIES`] entries are read per IFD and only IFD0 and the one Exif sub-IFD it points to
//! are visited, so there is no recursion and no loop. A malformed EXIF block is ignored (the image
//! itself is still valid), never an error.

/// The most entries read from one IFD.
pub const MAX_IFD_ENTRIES: usize = 512;

/// What the EXIF block says about the image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Exif {
    /// The Orientation tag, when present and in 1..=8.
    pub orientation: Option<u8>,
    /// The ColorSpace tag: 1 is sRGB, 0xFFFF "uncalibrated" (Adobe RGB and others).
    pub color_space: Option<u16>,
}

/// A TIFF byte order.
#[derive(Clone, Copy)]
struct Tiff<'a> {
    data: &'a [u8],
    big_endian: bool,
}

impl Tiff<'_> {
    fn u16(&self, at: usize) -> Option<u16> {
        let b: [u8; 2] = self.data.get(at..at.checked_add(2)?)?.try_into().ok()?;
        Some(if self.big_endian { u16::from_be_bytes(b) } else { u16::from_le_bytes(b) })
    }

    fn u32(&self, at: usize) -> Option<u32> {
        let b: [u8; 4] = self.data.get(at..at.checked_add(4)?)?.try_into().ok()?;
        Some(if self.big_endian { u32::from_be_bytes(b) } else { u32::from_le_bytes(b) })
    }

    /// The entries `(tag, type, count, value offset field position)` of the IFD at `at`.
    fn entries(&self, at: usize) -> impl Iterator<Item = (u16, u16, u32, usize)> + '_ {
        // A table that runs past the end yields fewer entries (`map_while` stops at the first).
        let count = self.u16(at).map_or(0, usize::from).min(MAX_IFD_ENTRIES);
        (0..count).map_while(move |i| {
            // Saturating: an offset near the top of the address space just reads nothing.
            let e = at.saturating_add(2 + i * 12);
            Some((self.u16(e)?, self.u16(e.saturating_add(2))?, self.u32(e.saturating_add(4))?, e.saturating_add(8)))
        })
    }

    /// A SHORT value stored inline in an entry.
    fn short(&self, kind: u16, count: u32, field: usize) -> Option<u16> {
        if kind == 3 && count >= 1 { self.u16(field) } else { None }
    }
}

/// Reads the payload of an APP1 segment. Returns `None` unless it starts with `Exif\0\0` and a TIFF
/// header.
pub(super) fn parse(segment: &[u8]) -> Option<Exif> {
    let data = segment.strip_prefix(b"Exif\0\0")?;
    let big_endian = match data.get(..4)? {
        [b'I', b'I', 42, 0] => false,
        [b'M', b'M', 0, 42] => true,
        _ => return None,
    };
    let tiff = Tiff { data, big_endian };
    let ifd0 = tiff.u32(4)? as usize;
    let mut exif = Exif::default();
    let mut sub_ifd = None;
    for (tag, kind, count, field) in tiff.entries(ifd0) {
        match tag {
            0x0112 => {
                exif.orientation = tiff.short(kind, count, field).filter(|o| (1..=8).contains(o)).map(|o| o as u8);
            }
            0x8769 if kind == 4 || kind == 13 => sub_ifd = tiff.u32(field).map(|o| o as usize),
            _ => {}
        }
    }
    if let Some(at) = sub_ifd {
        for (tag, kind, count, field) in tiff.entries(at) {
            if tag == 0xA001 {
                exif.color_space = tiff.short(kind, count, field);
            }
        }
    }
    Some(exif)
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    /// Builds an APP1 payload holding the given tags (the inverse of [`parse`], for tests and fixtures).
    fn build(big_endian: bool, orientation: Option<u16>, color_space: Option<u16>) -> Vec<u8> {
        let u16b = |v: u16| if big_endian { v.to_be_bytes() } else { v.to_le_bytes() };
        let u32b = |v: u32| if big_endian { v.to_be_bytes() } else { v.to_le_bytes() };
        let mut out = b"Exif\0\0".to_vec();
        out.extend_from_slice(if big_endian { b"MM\0\x2A" } else { b"II\x2A\0" });
        out.extend_from_slice(&u32b(8));
        let ifd0_entries = u16::from(orientation.is_some()) + u16::from(color_space.is_some());
        out.extend_from_slice(&u16b(ifd0_entries));
        let sub_at = 8 + 2 + 12 * u32::from(ifd0_entries) + 4;
        if let Some(o) = orientation {
            out.extend_from_slice(&u16b(0x0112));
            out.extend_from_slice(&u16b(3));
            out.extend_from_slice(&u32b(1));
            out.extend_from_slice(&u16b(o));
            out.extend_from_slice(&[0, 0]);
        }
        if color_space.is_some() {
            out.extend_from_slice(&u16b(0x8769));
            out.extend_from_slice(&u16b(4));
            out.extend_from_slice(&u32b(1));
            out.extend_from_slice(&u32b(sub_at));
        }
        out.extend_from_slice(&u32b(0));
        if let Some(c) = color_space {
            out.extend_from_slice(&u16b(1));
            out.extend_from_slice(&u16b(0xA001));
            out.extend_from_slice(&u16b(3));
            out.extend_from_slice(&u32b(1));
            out.extend_from_slice(&u16b(c));
            out.extend_from_slice(&[0, 0]);
            out.extend_from_slice(&u32b(0));
        }
        out
    }

    #[test]
    fn both_byte_orders_and_both_tags() {
        for big_endian in [false, true] {
            let e = parse(&build(big_endian, Some(6), Some(0xFFFF))).unwrap();
            assert_eq!(e, Exif { orientation: Some(6), color_space: Some(0xFFFF) });
            let e = parse(&build(big_endian, Some(3), None)).unwrap();
            assert_eq!(e, Exif { orientation: Some(3), color_space: None });
            let e = parse(&build(big_endian, None, Some(1))).unwrap();
            assert_eq!(e, Exif { orientation: None, color_space: Some(1) });
        }
        // Out-of-range orientations are ignored; foreign APP1 payloads are not EXIF.
        assert_eq!(parse(&build(false, Some(9), None)).unwrap().orientation, None);
        assert_eq!(parse(b"http://ns.adobe.com/xap/1.0/\0<x/>"), None);
        assert_eq!(parse(b"Exif\0\0XX*\0"), None);
    }

    #[test]
    fn truncated_and_mutated_blocks_never_panic() {
        let good = build(true, Some(8), Some(1));
        for n in 0..good.len() {
            let _ = parse(&good[..n]);
        }
        let mut rng = Rng::new(0xE81F);
        for _ in 0..20_000 {
            let mut m = good.clone();
            for _ in 0..=rng.index(4) {
                let i = rng.index(m.len());
                m[i] = rng.index(256) as u8;
            }
            let _ = parse(&m);
        }
    }
}
