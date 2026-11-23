//! The `.antmpl` template package `[IO-21]` `[IO-24]` `[MOD-02]`: a versioned little-endian binary
//! container of the template's sections, each optionally LZ-compressed and checksummed.
//!
//! The byte layout is documented in `docs/format-antmpl.md`. In short: a 32-byte header (magic
//! `HATP`, version, section count, file length, checksum of header and table), a table of 48-byte
//! section entries (tag, section version, flags, offset, stored and raw length, 128-bit hash of the
//! stored bytes), then the sections, each starting at the next 16-byte boundary after the previous
//! one. The metadata section is JSON; the others are binary.
//!
//! The writer is deterministic: the same template always gives the same bytes, so writing what was
//! read reproduces the file byte for byte. The reader checks every offset, length, count and
//! checksum against the input and [`PackageLimits`] before it allocates or indexes `[ROB-01]`,
//! skips unknown optional sections (forward compatibility) and refuses unknown required ones, and
//! finally checks the decoded template with [`Template::new`].

mod sections;

use humanart_core::hash::{Hash128, Hasher128, hash128};
use humanart_io::bytes::{Reader, Writer};
use humanart_io::{Error as IoError, Format, Limits};

use crate::error::{ModelError, Result};
use crate::expression::pose_space::PoseDriver;
use crate::mesh::MAX_VERTICES;
use crate::template::{Template, TemplateParts};
use sections::Dec;

/// The four magic bytes at the start of a package.
pub const MAGIC: [u8; 4] = *b"HATP";
/// The major format version this code writes and reads; a different major version is refused.
pub const VERSION_MAJOR: u16 = 1;
/// The minor format version this code writes; any minor version of the same major is read.
pub const VERSION_MINOR: u16 = 0;
/// Header length in bytes.
pub const HEADER_LEN: usize = 32;
/// Section table entry length in bytes.
pub const ENTRY_LEN: usize = 48;
/// The most sections a package may have.
pub const MAX_SECTIONS: usize = 64;
/// Sections start at multiples of this.
pub const SECTION_ALIGN: usize = 16;
/// Section flag: the stored bytes are an LZ frame (`humanart_io::lz`) of the payload.
pub const FLAG_COMPRESSED: u16 = 1;
/// Section flag: a reader that does not know the tag must refuse the package.
pub const FLAG_REQUIRED: u16 = 2;
/// The version of every section this code writes.
pub const SECTION_VERSION: u16 = 1;

/// Section tags, in the order they are written.
pub mod tag {
    /// JSON metadata: name, version, generator, licence, provenance, extra data.
    pub const META: [u8; 4] = *b"META";
    /// LOD0 mesh.
    pub const MESH: [u8; 4] = *b"MESH";
    /// UV tile table.
    pub const TILE: [u8; 4] = *b"TILE";
    /// Regions and per-vertex region arrays.
    pub const REGN: [u8; 4] = *b"REGN";
    /// Landmarks.
    pub const LMRK: [u8; 4] = *b"LMRK";
    /// Skeleton and joint regressor.
    pub const SKEL: [u8; 4] = *b"SKEL";
    /// Skin weights.
    pub const SKIN: [u8; 4] = *b"SKIN";
    /// Delta bases.
    pub const DLTA: [u8; 4] = *b"DLTA";
    /// Expression and corrective definitions.
    pub const EXPR: [u8; 4] = *b"EXPR";
    /// LOD1 and further.
    pub const LODS: [u8; 4] = *b"LODS";
    /// Pose drivers: written after the others, and only when the template has any (a required
    /// section then, since a reader that skipped it would activate the correctives differently).
    pub const PDRV: [u8; 4] = *b"PDRV";
    /// Every tag this version always writes, in order.
    pub const ALL: [[u8; 4]; 10] = [META, MESH, TILE, REGN, LMRK, SKEL, SKIN, DLTA, EXPR, LODS];
}

/// The limits a reader enforces `[ROB-02]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackageLimits {
    /// Largest package file, bytes.
    pub max_file_bytes: u64,
    /// Largest decoded section payload, bytes.
    pub max_section_bytes: u64,
    /// Most vertices of one mesh (also capped by [`MAX_VERTICES`]).
    pub max_vertices: usize,
    /// Most elements of any one array (faces, corners, delta entries, ...), on top of the
    /// per-array limits of the data model.
    pub max_elements: usize,
}

impl PackageLimits {
    /// The limits for real templates: 4 GiB files, 2 GiB sections, 2^24 vertices, 2^28 elements.
    pub const DEFAULT: Self =
        Self { max_file_bytes: 4 << 30, max_section_bytes: 2 << 30, max_vertices: MAX_VERTICES, max_elements: 1 << 28 };
    /// Small limits for fuzz tests and untrusted input: 64 MiB files, 16 MiB sections, 2^16
    /// vertices, 2^20 elements.
    pub const FUZZ: Self =
        Self { max_file_bytes: 64 << 20, max_section_bytes: 16 << 20, max_vertices: 1 << 16, max_elements: 1 << 20 };
}

impl Default for PackageLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// One entry of the section table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SectionEntry {
    /// Four ASCII bytes naming the section.
    pub tag: [u8; 4],
    /// The section's own format version.
    pub version: u16,
    /// [`FLAG_COMPRESSED`], [`FLAG_REQUIRED`].
    pub flags: u16,
    /// Offset of the stored bytes from the start of the file.
    pub offset: u64,
    /// Length of the stored bytes.
    pub stored_len: u64,
    /// Length of the payload after decompression (equal to `stored_len` when not compressed).
    pub raw_len: u64,
    /// MurmurHash3 x64_128 of the stored bytes.
    pub hash: Hash128,
}

impl SectionEntry {
    /// Whether the stored bytes are an LZ frame.
    #[must_use]
    pub const fn compressed(&self) -> bool {
        self.flags & FLAG_COMPRESSED != 0
    }

    /// Whether a reader must understand the section.
    #[must_use]
    pub const fn required(&self) -> bool {
        self.flags & FLAG_REQUIRED != 0
    }
}

/// Writes `template` as a package.
pub fn write_package(template: &Template) -> Result<Vec<u8>> {
    let payloads = encode_sections(template.parts())?;
    let mut sections: Vec<([u8; 4], u16, &[u8])> =
        tag::ALL.iter().zip(&payloads).map(|(&t, p)| (t, FLAG_REQUIRED, p.as_slice())).collect();
    let drivers = &template.parts().expressions.pose_drivers;
    let mut w = Writer::default();
    if !drivers.is_empty() {
        sections::write_pose_drivers(&mut w, drivers);
    }
    let pdrv = w.into_vec();
    if !drivers.is_empty() {
        sections.push((tag::PDRV, FLAG_REQUIRED, pdrv.as_slice()));
    }
    Ok(assemble(&sections))
}

/// Reads a package, checks it, and returns the template.
pub fn read_package(bytes: &[u8], limits: &PackageLimits) -> Result<Template> {
    let entries = read_directory(bytes, limits)?;
    let mut payloads: Vec<Vec<u8>> = vec![Vec::new(); tag::ALL.len()];
    let mut found = [false; tag::ALL.len()];
    let mut drivers = None;
    for e in &entries {
        if e.tag == tag::PDRV {
            if e.version != SECTION_VERSION {
                return Err(ModelError::Codec(IoError::Unsupported { format: Format::Chunk, what: "section version" })
                    .in_section(e.tag));
            }
            drivers = Some(section_payload(bytes, e).map_err(|err| err.in_section(e.tag))?);
            continue;
        }
        match tag::ALL.iter().position(|&t| t == e.tag) {
            Some(i) => {
                if e.version != SECTION_VERSION {
                    return Err(ModelError::Codec(IoError::Unsupported {
                        format: Format::Chunk,
                        what: "section version",
                    })
                    .in_section(e.tag));
                }
                payloads[i] = section_payload(bytes, e).map_err(|err| err.in_section(e.tag))?;
                found[i] = true;
            }
            None if e.required() => {
                return Err(ModelError::Codec(IoError::Unsupported {
                    format: Format::Chunk,
                    what: "a required section of a newer format",
                })
                .in_section(e.tag));
            }
            None => {}
        }
    }
    if let Some(i) = found.iter().position(|&f| !f) {
        return Err(ModelError::Codec(IoError::Invalid {
            format: Format::Chunk,
            offset: 0,
            what: "section is missing",
        })
        .in_section(tag::ALL[i]));
    }
    let views: Vec<&[u8]> = payloads.iter().map(Vec::as_slice).collect();
    let drivers = match &drivers {
        Some(raw) => section(tag::PDRV, raw, limits, sections::read_pose_drivers)?,
        None => Vec::new(),
    };
    decode_template(&views, drivers, limits)
}

/// Reads and checks the header and section table: magic, version, flags, file length, table
/// checksum, canonical section placement, zero padding, and the hash of every section's stored
/// bytes. Does not decompress or decode the sections.
pub fn read_directory(bytes: &[u8], limits: &PackageLimits) -> Result<Vec<SectionEntry>> {
    let invalid =
        |offset: usize, what: &'static str| ModelError::Codec(IoError::Invalid { format: Format::Chunk, offset, what });
    if bytes.len() as u64 > limits.max_file_bytes {
        return Err(ModelError::limit("package size", limits.max_file_bytes, bytes.len() as u64));
    }
    let mut r = Reader::new(bytes, Format::Chunk);
    if r.array::<4>()? != MAGIC {
        return Err(invalid(0, "not a HumanArt template package"));
    }
    let major = r.u16_le()?;
    let _minor = r.u16_le()?;
    if major != VERSION_MAJOR {
        return Err(ModelError::Codec(IoError::Unsupported { format: Format::Chunk, what: "package major version" }));
    }
    let n = r.u32_le()? as usize;
    if n > MAX_SECTIONS {
        return Err(ModelError::limit("section count", MAX_SECTIONS as u64, n as u64));
    }
    if r.u32_le()? != 0 {
        return Err(invalid(12, "unknown header flags"));
    }
    if r.u64_le()? != bytes.len() as u64 {
        return Err(invalid(16, "file length does not match the header"));
    }
    let checksum = r.u64_le()?;
    let table = r.bytes(n * ENTRY_LEN)?;
    if table_checksum(&bytes[..24], table) != checksum {
        return Err(invalid(24, "header checksum mismatch"));
    }
    let mut t = Reader::new(table, Format::Chunk);
    let mut entries: Vec<SectionEntry> = Vec::with_capacity(n);
    let mut end = HEADER_LEN + n * ENTRY_LEN;
    for i in 0..n {
        let at = HEADER_LEN + i * ENTRY_LEN;
        let e = SectionEntry {
            tag: t.array::<4>()?,
            version: t.u16_le()?,
            flags: t.u16_le()?,
            offset: t.u64_le()?,
            stored_len: t.u64_le()?,
            raw_len: t.u64_le()?,
            hash: Hash128::from_bytes(t.array::<16>()?),
        };
        if e.flags & !(FLAG_COMPRESSED | FLAG_REQUIRED) != 0 {
            return Err(invalid(at + 6, "unknown section flags"));
        }
        if entries.iter().any(|x| x.tag == e.tag) {
            return Err(invalid(at, "section tag used twice"));
        }
        let start = align(end);
        if e.offset != start as u64 {
            return Err(invalid(at + 8, "section is not at the next 16-byte boundary"));
        }
        let stop = e
            .offset
            .checked_add(e.stored_len)
            .filter(|&s| s <= bytes.len() as u64)
            .ok_or(ModelError::Codec(IoError::Truncated { format: Format::Chunk, offset: bytes.len() }))?
            as usize;
        if bytes[end..start].iter().any(|&b| b != 0) {
            return Err(invalid(end, "padding is not zero"));
        }
        if !e.compressed() && e.stored_len != e.raw_len {
            return Err(invalid(at + 24, "stored section length differs from its raw length"));
        }
        if e.raw_len > limits.max_section_bytes {
            return Err(ModelError::limit("section size", limits.max_section_bytes, e.raw_len));
        }
        if hash128(&bytes[start..stop]) != e.hash {
            return Err(invalid(start, "section checksum mismatch"));
        }
        entries.push(e);
        end = stop;
    }
    if end != bytes.len() {
        return Err(invalid(end, "trailing bytes after the last section"));
    }
    Ok(entries)
}

/// The decoded payload of a section listed by [`read_directory`] for the same `bytes`.
pub fn section_payload(bytes: &[u8], entry: &SectionEntry) -> Result<Vec<u8>> {
    let truncated = ModelError::Codec(IoError::Truncated { format: Format::Chunk, offset: bytes.len() });
    let (Ok(start), Some(Ok(stop))) =
        (usize::try_from(entry.offset), entry.offset.checked_add(entry.stored_len).map(usize::try_from))
    else {
        return Err(truncated);
    };
    let stored = bytes.get(start..stop).ok_or(truncated)?;
    if !entry.compressed() {
        return Ok(stored.to_vec());
    }
    let limits = Limits { max_bytes: entry.raw_len, ..Limits::DEFAULT };
    let raw = humanart_io::lz::decompress(stored, &limits)?;
    if raw.len() as u64 != entry.raw_len {
        return Err(ModelError::Codec(IoError::Invalid {
            format: Format::Chunk,
            offset: start,
            what: "decompressed length differs from the table",
        }));
    }
    Ok(raw)
}

const fn align(n: usize) -> usize {
    n.div_ceil(SECTION_ALIGN) * SECTION_ALIGN
}

fn table_checksum(header: &[u8], table: &[u8]) -> u64 {
    let mut h = Hasher128::new();
    h.update(header);
    h.update(table);
    h.finish().fold64()
}

/// The payloads of the sections of `tag::ALL`, in that order.
pub(crate) fn encode_sections(t: &TemplateParts) -> Result<Vec<Vec<u8>>> {
    let mut out = vec![sections::write_meta(&t.info)?];
    let writers: [&dyn Fn(&mut Writer); 9] = [
        &|w| sections::write_mesh(w, &t.mesh),
        &|w| sections::write_tiles(w, &t.tiles),
        &|w| sections::write_regions(w, &t.regions),
        &|w| sections::write_landmarks(w, &t.landmarks),
        &|w| sections::write_skeleton(w, &t.skeleton),
        &|w| sections::write_skin(w, &t.skin),
        &|w| sections::write_deltas(w, &t.deltas),
        &|w| sections::write_expressions(w, &t.expressions),
        &|w| sections::write_lods(w, &t.lods),
    ];
    for write in writers {
        let mut w = Writer::default();
        write(&mut w);
        out.push(w.into_vec());
    }
    Ok(out)
}

/// Builds a package from `(tag, flags, payload)` triples: compresses each payload when that makes
/// it smaller (setting [`FLAG_COMPRESSED`]), lays the sections out and fills in the table.
pub(crate) fn assemble(sections: &[([u8; 4], u16, &[u8])]) -> Vec<u8> {
    let n = sections.len();
    let mut stored: Vec<(Vec<u8>, u16)> = Vec::with_capacity(n);
    for &(_, flags, payload) in sections {
        let packed = humanart_io::lz::compress(payload);
        if packed.len() < payload.len() {
            stored.push((packed, flags | FLAG_COMPRESSED));
        } else {
            stored.push((payload.to_vec(), flags & !FLAG_COMPRESSED));
        }
    }
    let mut w = Writer::default();
    w.bytes(&MAGIC);
    w.u16_le(VERSION_MAJOR);
    w.u16_le(VERSION_MINOR);
    w.u32_le(n as u32);
    w.u32_le(0);
    w.u64_le(0); // file length, patched below
    w.u64_le(0); // checksum, patched below
    let mut offset = HEADER_LEN + n * ENTRY_LEN;
    for (&(tag, _, payload), (bytes, flags)) in sections.iter().zip(&stored) {
        offset = align(offset);
        w.bytes(&tag);
        w.u16_le(SECTION_VERSION);
        w.u16_le(*flags);
        w.u64_le(offset as u64);
        w.u64_le(bytes.len() as u64);
        w.u64_le(payload.len() as u64);
        w.bytes(&hash128(bytes).to_bytes());
        offset += bytes.len();
    }
    for (bytes, _) in &stored {
        w.pad_to(SECTION_ALIGN);
        w.bytes(bytes);
    }
    let len = w.len() as u64;
    w.patch_u64_le(16, len);
    let mut out = w.into_vec();
    let sum = table_checksum(&out[..24], &out[HEADER_LEN..HEADER_LEN + n * ENTRY_LEN]);
    out[24..32].copy_from_slice(&sum.to_le_bytes());
    out
}

fn section<T>(
    tag: [u8; 4],
    raw: &[u8],
    limits: &PackageLimits,
    read: impl FnOnce(&mut Dec<'_>) -> Result<T>,
) -> Result<T> {
    let mut d = Dec::new(raw, limits);
    read(&mut d).and_then(|v| d.finish().map(|()| v)).map_err(|e| e.in_section(tag))
}

/// Decodes the payloads of `tag::ALL` (in that order) into a checked template.
pub(crate) fn decode_template(
    raw: &[&[u8]],
    pose_drivers: Vec<PoseDriver>,
    limits: &PackageLimits,
) -> Result<Template> {
    let [meta, mesh, tiles, regions, landmarks, skeleton, skin, deltas, expressions, lods] = raw else {
        return Err(ModelError::invalid("package", 0, "wrong number of sections"));
    };
    let info = sections::read_meta(meta).map_err(|e| e.in_section(tag::META))?;
    let mut parts = TemplateParts {
        info,
        mesh: section(tag::MESH, mesh, limits, sections::read_mesh)?,
        tiles: section(tag::TILE, tiles, limits, sections::read_tiles)?,
        regions: section(tag::REGN, regions, limits, sections::read_regions)?,
        landmarks: section(tag::LMRK, landmarks, limits, sections::read_landmarks)?,
        skeleton: section(tag::SKEL, skeleton, limits, sections::read_skeleton)?,
        skin: section(tag::SKIN, skin, limits, sections::read_skin)?,
        deltas: section(tag::DLTA, deltas, limits, sections::read_deltas)?,
        expressions: section(tag::EXPR, expressions, limits, sections::read_expressions)?,
        lods: section(tag::LODS, lods, limits, sections::read_lods)?,
    };
    parts.expressions.pose_drivers = pose_drivers;
    Template::new(parts)
}

#[cfg(test)]
mod tests;
