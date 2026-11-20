//! The `.anthair` strand file `[IO-12]` `[HAIR-01]`: versioned, one chunk per per-strand
//! attribute, each chunk stored raw or LZ-compressed (`humanart_io::lz`, `docs/format-lz.md`).
//!
//! # Byte layout, version 1 (all integers little-endian)
//!
//! ```text
//! header, 24 bytes
//!   0  magic        4 bytes  "HAIR"
//!   4  version      u16      1
//!   6  flags        u16      0 (no flag is defined; a reader refuses others)
//!   8  strands      u32      strand count n, at most MAX_STRANDS (2^22)
//!  12  points       u32      control points per strand p, 2..=32
//!  16  chunks       u32      chunk count, at most MAX_CHUNKS (64)
//!  20  reserved     u32      0
//! then `chunks` chunks, each a 16-byte header followed by its stored bytes, back to back:
//!   0  tag          4 bytes  ASCII, see below
//!   4  codec        u8       0 = raw, 1 = one LZ frame (docs/format-lz.md)
//!   5  reserved     3 bytes  0
//!   8  raw_len      u32      length of the decoded content
//!  12  stored_len   u32      length of the stored bytes that follow
//! ```
//!
//! Chunk contents (decoded), arrays of `n` entries unless noted:
//!
//! | tag    | content                                                    | raw_len    |
//! |--------|------------------------------------------------------------|------------|
//! | `ROOT` | root triangle `u32` ×n, then barycentric `(u, v)` `f32`×2 ×n | 12 n       |
//! | `OFFS` | point offsets in the root frame, `f16` x, y, z, n × p points | 6 n p      |
//! | `LENG` | strand length, metres, `f32`                               | 4 n        |
//! | `WIDR` | root diameter, metres, `f32`                               | 4 n        |
//! | `WIDT` | tip diameter, metres, `f32`                                | 4 n        |
//! | `COLR` | colour variation id, `u16`                                 | 2 n        |
//! | `CURL` | curl phase, fraction of a turn in `[0, 1)`, `f32`         | 4 n        |
//! | `CLMP` | clump id (index of the clump's centre strand), `u32`       | 4 n        |
//! | `STIF` | stiffness hint, 0..1, `f32`                                | 4 n        |
//! | `GREY` | grey level, 0 pigmented .. 255 unpigmented, `u8`           | n          |
//!
//! Every one of the ten tags is required exactly once; chunks may come in any order; a chunk with
//! an unknown tag is skipped (later versions may add attributes). The file ends after the last
//! chunk. Offsets are in the root frame of [`super::RootFrame`] (`t`, `b`, `n`), so the file
//! needs the skin (triangles, positions, normals, UVs) it was made on to place strands in space.
//!
//! The reader checks the header, every chunk length against the strand count before anything is
//! allocated, the codec, the decoded length, and finally the strands themselves
//! ([`super::Strands::check`]: clump ids, finiteness) `[ROB-01]`.
//!
//! The guide blob of a document groom ([`write_guides`], [`read_guides`]) is documented on
//! [`write_guides`].

use humanart_io::{Limits, lz};
use humanart_math::{F16, Vec3};

use super::error::{HairError, HairResult};
use super::strands::{Guides, MAX_GUIDE_POINTS, MAX_GUIDES, MAX_STRAND_POINTS, MAX_STRANDS, Strands};

/// The file magic.
pub const MAGIC: [u8; 4] = *b"HAIR";
/// The format version this module writes and reads.
pub const VERSION: u16 = 1;
/// Bytes in the file header.
pub const HEADER_LEN: usize = 24;
/// Bytes in a chunk header.
pub const CHUNK_HEADER_LEN: usize = 16;
/// The most chunks a file may declare.
pub const MAX_CHUNKS: u32 = 64;

const CODEC_RAW: u8 = 0;
const CODEC_LZ: u8 = 1;

/// The required tags and their bytes per strand (`OFFS` per strand and point).
const TAGS: [([u8; 4], u64); 10] = [
    (*b"ROOT", 12),
    (*b"OFFS", 6),
    (*b"LENG", 4),
    (*b"WIDR", 4),
    (*b"WIDT", 4),
    (*b"COLR", 2),
    (*b"CURL", 4),
    (*b"CLMP", 4),
    (*b"STIF", 4),
    (*b"GREY", 1),
];

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Encodes `strands` as an `.anthair` file; `compress` stores each chunk LZ-compressed when that
/// is smaller.
pub fn write(strands: &Strands, compress: bool) -> HairResult<Vec<u8>> {
    strands.check(None)?;
    let n = strands.len();
    let mut root = Vec::with_capacity(n * 12);
    root.extend(strands.triangle.iter().flat_map(|t| t.to_le_bytes()));
    root.extend(strands.bary.iter().flat_map(|b| [b[0].to_le_bytes(), b[1].to_le_bytes()]).flatten());
    let chunks: [([u8; 4], Vec<u8>); 10] = [
        (TAGS[0].0, root),
        (TAGS[1].0, strands.offsets.iter().flat_map(|h| h.to_bits().to_le_bytes()).collect()),
        (TAGS[2].0, f32s(&strands.length)),
        (TAGS[3].0, f32s(&strands.width_root)),
        (TAGS[4].0, f32s(&strands.width_tip)),
        (TAGS[5].0, strands.color_id.iter().flat_map(|c| c.to_le_bytes()).collect()),
        (TAGS[6].0, f32s(&strands.curl_phase)),
        (TAGS[7].0, strands.clump.iter().flat_map(|c| c.to_le_bytes()).collect()),
        (TAGS[8].0, f32s(&strands.stiffness)),
        (TAGS[9].0, strands.grey.clone()),
    ];
    let mut out = Vec::with_capacity(HEADER_LEN + chunks.iter().map(|c| c.1.len() + CHUNK_HEADER_LEN).sum::<usize>());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&0_u16.to_le_bytes());
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.extend_from_slice(&strands.points.to_le_bytes());
    out.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
    out.extend_from_slice(&0_u32.to_le_bytes());
    for (tag, raw) in &chunks {
        let raw_len = u32::try_from(raw.len())
            .map_err(|_| HairError::limit("chunk bytes", u64::from(u32::MAX), raw.len() as u64))?;
        let packed = if compress && !raw.is_empty() { Some(lz::compress(raw)) } else { None };
        let (codec, stored) = match &packed {
            Some(p) if p.len() < raw.len() => (CODEC_LZ, p.as_slice()),
            _ => (CODEC_RAW, raw.as_slice()),
        };
        out.extend_from_slice(tag);
        out.extend_from_slice(&[codec, 0, 0, 0]);
        out.extend_from_slice(&raw_len.to_le_bytes());
        out.extend_from_slice(&(stored.len() as u32).to_le_bytes());
        out.extend_from_slice(stored);
    }
    Ok(out)
}

/// A cursor over the file that reports offsets in errors.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> HairResult<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.data.len());
        let Some(end) = end else { return Err(HairError::format(self.pos, "file ends early")) };
        let s = &self.data[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u32(&mut self) -> HairResult<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u16(&mut self) -> HairResult<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
}

fn le_u32(b: &[u8]) -> u32 {
    let mut a = [0; 4];
    a.copy_from_slice(&b[..4]);
    u32::from_le_bytes(a)
}

fn le_f32(b: &[u8]) -> f32 {
    f32::from_bits(le_u32(b))
}

fn read_f32s(raw: &[u8]) -> Vec<f32> {
    raw.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

/// Decodes an `.anthair` file. `limits.max_bytes` bounds the decoded size of every chunk and of
/// the whole file's content.
pub fn read(bytes: &[u8], limits: &Limits) -> HairResult<Strands> {
    let mut c = Cursor { data: bytes, pos: 0 };
    if c.take(4)? != MAGIC {
        return Err(HairError::format(0, "not an .anthair file (bad magic)"));
    }
    let version = c.u16()?;
    if version != VERSION {
        return Err(HairError::format(4, "unsupported .anthair version"));
    }
    if c.u16()? != 0 {
        return Err(HairError::format(6, "unknown flags"));
    }
    let n = c.u32()? as usize;
    if n > MAX_STRANDS {
        return Err(HairError::limit("strand count", MAX_STRANDS as u64, n as u64));
    }
    let points = c.u32()?;
    if !(2..=MAX_STRAND_POINTS).contains(&points) {
        return Err(HairError::format(12, "point count outside 2..=32"));
    }
    let chunk_count = c.u32()?;
    if chunk_count > MAX_CHUNKS {
        return Err(HairError::limit("chunk count", u64::from(MAX_CHUNKS), u64::from(chunk_count)));
    }
    if c.u32()? != 0 {
        return Err(HairError::format(20, "reserved header field is not 0"));
    }
    // Every expected length, computed in u64 and bounded before anything is allocated.
    let expected = |tag: usize| -> u64 {
        let per = TAGS[tag].1;
        if tag == 1 { per * n as u64 * u64::from(points) } else { per * n as u64 }
    };
    let total: u64 = (0..TAGS.len()).map(expected).sum();
    if total > limits.max_bytes {
        return Err(HairError::limit("decoded strand bytes", limits.max_bytes, total));
    }
    let mut found: [Option<Vec<u8>>; 10] = Default::default();
    for _ in 0..chunk_count {
        let at = c.pos;
        let header = c.take(CHUNK_HEADER_LEN)?;
        let tag = [header[0], header[1], header[2], header[3]];
        let codec = header[4];
        if header[5..8] != [0, 0, 0] {
            return Err(HairError::format(at + 5, "reserved chunk bytes are not 0"));
        }
        let raw_len = u64::from(le_u32(&header[8..12]));
        let stored_len = le_u32(&header[12..16]) as usize;
        let stored = c.take(stored_len)?;
        let Some(slot) = TAGS.iter().position(|t| t.0 == tag) else { continue };
        if found[slot].is_some() {
            return Err(HairError::format(at, "chunk appears twice"));
        }
        if raw_len != expected(slot) {
            return Err(HairError::format(at + 8, "chunk length does not match the strand count"));
        }
        let raw = match codec {
            CODEC_RAW if stored.len() as u64 == raw_len => stored.to_vec(),
            CODEC_RAW => return Err(HairError::format(at + 12, "raw chunk length differs from its decoded length")),
            CODEC_LZ => {
                let chunk_limits = Limits { max_bytes: raw_len, ..*limits };
                let raw = lz::decompress(stored, &chunk_limits).map_err(HairError::Codec)?;
                if raw.len() as u64 != raw_len {
                    return Err(HairError::format(at + 8, "decoded chunk length differs from its header"));
                }
                raw
            }
            _ => return Err(HairError::format(at + 4, "unknown chunk codec")),
        };
        found[slot] = Some(raw);
    }
    if c.pos != bytes.len() {
        return Err(HairError::format(c.pos, "bytes after the last chunk"));
    }
    let mut parts = Vec::with_capacity(TAGS.len());
    for (i, f) in found.into_iter().enumerate() {
        match f {
            Some(raw) => parts.push(raw),
            None => return Err(HairError::invalid("anthair chunk", i, "required chunk missing")),
        }
    }
    let mut s = Strands::new(points)?;
    let (tri, bary) = parts[0].split_at(4 * n);
    s.triangle = tri.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect();
    s.bary = bary.as_chunks::<8>().0.iter().map(|b| [le_f32(&b[..4]), le_f32(&b[4..])]).collect();
    s.offsets = parts[1].as_chunks::<2>().0.iter().map(|b| F16::from_bits(u16::from_le_bytes(*b))).collect();
    s.length = read_f32s(&parts[2]);
    s.width_root = read_f32s(&parts[3]);
    s.width_tip = read_f32s(&parts[4]);
    s.color_id = parts[5].as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect();
    s.curl_phase = read_f32s(&parts[6]);
    s.clump = parts[7].as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect();
    s.stiffness = read_f32s(&parts[8]);
    s.grey = std::mem::take(&mut parts[9]);
    if let Some(i) = s.bary.iter().position(|b| !(b[0].is_finite() && b[1].is_finite())) {
        return Err(HairError::invalid("strand", i, "barycentric root not finite"));
    }
    s.check(None)?;
    Ok(s)
}

/// The magic of a guide blob.
pub const GUIDES_MAGIC: [u8; 4] = *b"HAGD";
/// Bytes in a guide blob header.
pub const GUIDES_HEADER_LEN: usize = 32;

/// Encodes `guides` as the guide blob a document groom references (`Groom::guides`, the
/// `groom_guide_edit` stroke payload) `[DOC-08]`. Layout, little-endian:
///
/// ```text
///  0  magic       "HAGD"
///  4  version     u16 1
///  6  flags       u16 0
///  8  guides      u32 n, at most MAX_GUIDES (2^18)
/// 12  points      u32 p, 2..=64
/// 16  codec       u8 0 raw, 1 LZ frame; then 3 bytes 0
/// 20  raw_len     u32 = 12 n + 12 n p
/// 24  stored_len  u32, the stored content that follows; nothing comes after it
/// 28  reserved    u32 0
/// 32  content: root triangle u32 x n, barycentric (u, v) f32 x 2 x n,
///     offsets f32 x, y, z x n x p (root frame, the first of each guide is the root)
/// ```
///
/// The guides round-trip bit for bit (no resampling).
pub fn write_guides(guides: &Guides, compress: bool) -> HairResult<Vec<u8>> {
    let n = guides.len();
    let mut raw = Vec::with_capacity(12 * n + 12 * guides.all_offsets().len());
    raw.extend(guides.triangles().iter().flat_map(|t| t.to_le_bytes()));
    raw.extend(guides.barys().iter().flat_map(|b| [b[0].to_le_bytes(), b[1].to_le_bytes()]).flatten());
    raw.extend(
        guides.all_offsets().iter().flat_map(|p| [p.x.to_le_bytes(), p.y.to_le_bytes(), p.z.to_le_bytes()]).flatten(),
    );
    let raw_len =
        u32::try_from(raw.len()).map_err(|_| HairError::limit("guide bytes", u64::from(u32::MAX), raw.len() as u64))?;
    let packed = if compress && !raw.is_empty() { Some(lz::compress(&raw)) } else { None };
    let (codec, stored) = match &packed {
        Some(p) if p.len() < raw.len() => (CODEC_LZ, p.as_slice()),
        _ => (CODEC_RAW, raw.as_slice()),
    };
    let mut out = Vec::with_capacity(GUIDES_HEADER_LEN + stored.len());
    out.extend_from_slice(&GUIDES_MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&0_u16.to_le_bytes());
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.extend_from_slice(&guides.points().to_le_bytes());
    out.extend_from_slice(&[codec, 0, 0, 0]);
    out.extend_from_slice(&raw_len.to_le_bytes());
    out.extend_from_slice(&(stored.len() as u32).to_le_bytes());
    out.extend_from_slice(&0_u32.to_le_bytes());
    out.extend_from_slice(stored);
    Ok(out)
}

/// Decodes a guide blob written by [`write_guides`] (`limits.max_bytes` bounds the content).
pub fn read_guides(bytes: &[u8], limits: &Limits) -> HairResult<Guides> {
    let mut c = Cursor { data: bytes, pos: 0 };
    if c.take(4)? != GUIDES_MAGIC {
        return Err(HairError::format(0, "not a guide blob (bad magic)"));
    }
    if c.u16()? != VERSION {
        return Err(HairError::format(4, "unsupported guide blob version"));
    }
    if c.u16()? != 0 {
        return Err(HairError::format(6, "unknown flags"));
    }
    let n = c.u32()? as usize;
    if n > MAX_GUIDES {
        return Err(HairError::limit("guide count", MAX_GUIDES as u64, n as u64));
    }
    let points = c.u32()?;
    if !(2..=MAX_GUIDE_POINTS).contains(&points) {
        return Err(HairError::format(12, "point count outside 2..=64"));
    }
    let head = c.take(4)?;
    let codec = head[0];
    if head[1..] != [0, 0, 0] {
        return Err(HairError::format(17, "reserved bytes are not 0"));
    }
    let raw_len = u64::from(c.u32()?);
    let stored_len = c.u32()? as usize;
    if c.u32()? != 0 {
        return Err(HairError::format(28, "reserved header field is not 0"));
    }
    let expected = 12 * n as u64 + 12 * n as u64 * u64::from(points);
    if raw_len != expected {
        return Err(HairError::format(20, "content length does not match the guide count"));
    }
    if raw_len > limits.max_bytes {
        return Err(HairError::limit("decoded guide bytes", limits.max_bytes, raw_len));
    }
    let stored = c.take(stored_len)?;
    if c.pos != bytes.len() {
        return Err(HairError::format(c.pos, "bytes after the content"));
    }
    let raw = match codec {
        CODEC_RAW if stored.len() as u64 == raw_len => stored.to_vec(),
        CODEC_RAW => return Err(HairError::format(24, "raw content length differs from its decoded length")),
        CODEC_LZ => {
            let raw = lz::decompress(stored, &Limits { max_bytes: raw_len, ..*limits }).map_err(HairError::Codec)?;
            if raw.len() as u64 != raw_len {
                return Err(HairError::format(20, "decoded content length differs from its header"));
            }
            raw
        }
        _ => return Err(HairError::format(16, "unknown codec")),
    };
    let (tri, rest) = raw.split_at(4 * n);
    let (bary, offs) = rest.split_at(8 * n);
    let triangle = tri.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect();
    let bary = bary.as_chunks::<8>().0.iter().map(|b| [le_f32(&b[..4]), le_f32(&b[4..])]).collect();
    let offsets = offs
        .as_chunks::<12>()
        .0
        .iter()
        .map(|b| Vec3::new(le_f32(&b[..4]), le_f32(&b[4..8]), le_f32(&b[8..])))
        .collect();
    Guides::from_raw(points, triangle, bary, offsets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::placement::place;
    use crate::hair::presets::preset_groom;
    use crate::hair::surface::ProxyHead;
    use crate::hair::{GroomMaps, GroomParams, ScalpMask, generate_groom};

    fn sample(count: f32) -> Strands {
        let head = ProxyHead::new(24).unwrap();
        let mask = ScalpMask::new(&head.surface, head.scalp.clone()).unwrap();
        let p = preset_groom("wavy", &head.surface, &mask, 60, None, 3).unwrap();
        let params = GroomParams { strand_count: count, grey_fraction: 0.2, ..p.params };
        let layout = place(&head.surface, &mask, &p.guides, &params, &GroomMaps::default(), 5).unwrap();
        generate_groom(&head.surface, &layout, &p.guides, &params, None).unwrap().strands
    }

    #[test]
    fn io_12_round_trip_raw_and_compressed() {
        let s = sample(800.0);
        for compress in [false, true] {
            let bytes = write(&s, compress).unwrap();
            assert_eq!(&bytes[..4], b"HAIR");
            let back = read(&bytes, &Limits::DEFAULT).unwrap();
            assert_eq!(back, s, "compress {compress}");
        }
        let raw = write(&s, false).unwrap();
        let packed = write(&s, true).unwrap();
        assert!(packed.len() < raw.len(), "{} {}", packed.len(), raw.len());
        // An empty groom is a valid file.
        let empty = Strands::new(4).unwrap();
        assert_eq!(read(&write(&empty, true).unwrap(), &Limits::DEFAULT).unwrap(), empty);
    }

    #[test]
    fn io_12_unknown_chunks_are_skipped_and_errors_are_specific() {
        let s = sample(50.0);
        let mut bytes = write(&s, false).unwrap();
        // Append an unknown chunk and bump the count: still readable.
        let mut extra = bytes.clone();
        extra.extend_from_slice(b"XTRA\0\0\0\0");
        extra.extend_from_slice(&3_u32.to_le_bytes());
        extra.extend_from_slice(&3_u32.to_le_bytes());
        extra.extend_from_slice(&[1, 2, 3]);
        extra[16..20].copy_from_slice(&11_u32.to_le_bytes());
        assert_eq!(read(&extra, &Limits::DEFAULT).unwrap(), s);
        // Version, magic, trailing bytes, limits.
        let mut v2 = bytes.clone();
        v2[4] = 2;
        assert!(matches!(read(&v2, &Limits::DEFAULT), Err(HairError::Format { offset: 4, .. })));
        assert!(read(b"NOPE", &Limits::DEFAULT).is_err());
        bytes.push(0);
        assert!(matches!(read(&bytes, &Limits::DEFAULT), Err(HairError::Format { .. })));
        let tiny = Limits { max_bytes: 100, ..Limits::DEFAULT };
        assert!(matches!(read(&write(&s, false).unwrap(), &tiny), Err(HairError::Limit { .. })));
        // A declared huge strand count with no data is refused before allocation.
        let mut huge = write(&Strands::new(4).unwrap(), false).unwrap();
        huge[8..12].copy_from_slice(&(MAX_STRANDS as u32).to_le_bytes());
        assert!(read(&huge, &Limits::DEFAULT).is_err());
    }

    /// `[ROB-01]`: mutated `.anthair` files never panic or hang; accepted ones are consistent.
    #[test]
    fn io_12_anthair_mutation_fuzz() {
        let s = sample(40.0);
        let corpus =
            vec![write(&s, false).unwrap(), write(&s, true).unwrap(), write(&Strands::new(2).unwrap(), true).unwrap()];
        humanart_core::fuzz::run("anthair", &corpus, 4000, 0x0A17_4A12, std::time::Duration::from_secs(60), |bytes| {
            let limits = Limits { max_bytes: 1 << 20, ..Limits::DEFAULT };
            if let Ok(s) = read(bytes, &limits) {
                assert!(s.check(None).is_ok());
                let again = write(&s, false).unwrap();
                assert_eq!(read(&again, &limits).unwrap(), s);
            }
        });
    }

    #[test]
    fn guide_blob_round_trips_and_survives_mutation() {
        let head = ProxyHead::new(24).unwrap();
        let mask = ScalpMask::new(&head.surface, head.scalp.clone()).unwrap();
        let g = preset_groom("curly", &head.surface, &mask, 40, None, 3).unwrap().guides;
        for compress in [false, true] {
            let bytes = write_guides(&g, compress).unwrap();
            assert_eq!(read_guides(&bytes, &Limits::DEFAULT).unwrap(), g);
        }
        let empty = Guides::new(5).unwrap();
        assert_eq!(read_guides(&write_guides(&empty, true).unwrap(), &Limits::DEFAULT).unwrap(), empty);
        let mut bad = write_guides(&g, false).unwrap();
        bad[8..12].copy_from_slice(&(MAX_GUIDES as u32 + 1).to_le_bytes());
        assert!(matches!(read_guides(&bad, &Limits::DEFAULT), Err(HairError::Limit { .. })));
        let corpus = vec![write_guides(&g, false).unwrap(), write_guides(&g, true).unwrap()];
        humanart_core::fuzz::run(
            "guide-blob",
            &corpus,
            3000,
            0x006D_1DE5,
            std::time::Duration::from_secs(60),
            |bytes| {
                if let Ok(g) = read_guides(bytes, &Limits { max_bytes: 1 << 20, ..Limits::DEFAULT }) {
                    let again = write_guides(&g, false).unwrap();
                    assert_eq!(read_guides(&again, &Limits::DEFAULT).unwrap(), g);
                }
            },
        );
    }
}
