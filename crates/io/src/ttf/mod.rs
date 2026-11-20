//! TrueType font parser for font baking and text layout `[UI-03]` `[DEP-05]`.
//!
//! [`Font`] borrows the bytes of a `.ttf` file and reads what the glyph-atlas baker and the text
//! layout need: metrics, the character map, advances, kerning and quadratic outlines. Constructing one
//! validates the table directory and the required tables and is otherwise cheap; everything else is
//! read on demand with every offset checked, so a damaged or hostile file yields an [`Error`] (or
//! "no glyph", "no kerning"), never a panic `[ROB-01]`.
//!
//! # What is read
//!
//! - The sfnt directory of TrueType-outline fonts (version `0x00010000` or `true`). A TrueType
//!   collection (`ttcf`) is supported by face index ([`Font::from_collection`], [`Font::face_count`]).
//!   [`Font::new`] checks that every table lies inside the file; [`Font::verify_checksums`] (or
//!   [`Font::new_verified`], which does both) additionally checks every table checksum and the
//!   whole-file checksum. The two are separate because rasterisers traditionally accept fonts with
//!   stale checksums, while a font baker wants to know that its input is intact.
//! - `head`, `maxp`, `hhea`, `hmtx`, `OS/2`, `post`, `name` ([`Metrics`], [`MaxProfile`],
//!   [`Font::advance`], [`Font::h_metrics`], [`Font::name`]).
//! - `cmap` formats 0, 4, 6 and 12; the best Unicode subtable is chosen once ([`Font::glyph_index`],
//!   [`Font::for_each_mapping`]).
//! - `loca` + `glyf`: simple and composite glyphs ([`Font::outline`]), with every component transform,
//!   offsets by value and by point matching, `USE_MY_METRICS`, a nesting limit, cycle detection and
//!   the point, contour, component and depth limits the font declares in `maxp`.
//! - Kerning ([`Font::kerning`], [`Font::kerning_pairs`]): `GPOS` pair adjustment (lookup type 2,
//!   formats 1 and 2, also behind extension lookups) for the feature `kern` of the scripts `DFLT` and
//!   `latn`, else the `kern` table (format 0, horizontal).
//!
//! # Limits `[ROB-02]`
//!
//! An outline has at most the points, contours, components and nesting the font's own `maxp` table
//! declares (so at most 65 535 points), and never more than 1024 components and 8 nesting levels; at most 65 536
//! `GPOS` pair subtables are indexed. Nothing is allocated in proportion to a count read from the
//! file before that count has been checked against the size of its table.
//!
//! # What is not
//!
//! - CFF outlines (`OTTO`, `CFF `, `CFF2`), WOFF containers and bitmap-only fonts are
//!   [`Error::Unsupported`].
//! - Variable fonts: only the default instance is read. The variation tables (`fvar`, `gvar`, `avar`,
//!   `HVAR`, `MVAR`, GPOS feature variations and device tables) are ignored, so outlines, advances
//!   and kerning are those of the default master; [`Font::is_variable`] tells a caller that it is so.
//! - Hinting instructions, `GSUB` substitutions (ligatures), GPOS lookups other than pair adjustment,
//!   `kern` subtable formats other than 0, vertical metrics and cmap variation selectors (format 14)
//!   are ignored.

use crate::bytes::Reader;
use crate::error::{Error, Format, Result};

mod glyf;
mod kerning;
mod path;

pub use glyf::OutlineSize;
pub use kerning::KerningPair;
pub use path::{
    Orientation, OutlineSink, Path, PathCommand, flatten_quad, orientation, signed_area, signed_area_of_commands,
};

/// The format every error of this module names.
const FORMAT: Format = Format::Ttf;

/// An [`Error::Invalid`] of this module.
const fn invalid(offset: usize, what: &'static str) -> Error {
    Error::Invalid { format: FORMAT, offset, what }
}

/// An [`Error::Unsupported`] of this module.
const fn unsupported(what: &'static str) -> Error {
    Error::Unsupported { format: FORMAT, what }
}

/// The index of a glyph in a font. Glyph 0 is the "missing glyph" (`.notdef`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GlyphId(pub u16);

impl GlyphId {
    /// The missing-glyph glyph every font has at index 0.
    pub const NOTDEF: Self = Self(0);
}

/// An axis-aligned box in font units, Y up.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BBox {
    /// The left edge.
    pub x_min: i16,
    /// The bottom edge.
    pub y_min: i16,
    /// The right edge.
    pub x_max: i16,
    /// The top edge.
    pub y_max: i16,
}

impl BBox {
    /// The width (zero if the box is inverted).
    #[must_use]
    pub fn width(&self) -> i32 {
        (i32::from(self.x_max) - i32::from(self.x_min)).max(0)
    }

    /// The height (zero if the box is inverted).
    #[must_use]
    pub fn height(&self) -> i32 {
        (i32::from(self.y_max) - i32::from(self.y_min)).max(0)
    }
}

/// Font-wide measurements, in font units (divide by [`Metrics::units_per_em`] and multiply by the
/// font size to get pixels). Y is up: descenders are negative.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Metrics {
    /// Font units per em square (`head`).
    pub units_per_em: u16,
    /// The number of glyphs (`maxp`).
    pub num_glyphs: u16,
    /// The box enclosing every glyph (`head`).
    pub bbox: BBox,
    /// The typographic ascender: `OS/2` `sTypoAscender`, or the `hhea` ascender when the font has no
    /// usable `OS/2` table (absent, too short, or both typographic values zero).
    pub ascender: i16,
    /// The typographic descender (negative below the baseline), with the same fallback.
    pub descender: i16,
    /// The typographic line gap, with the same fallback.
    pub line_gap: i16,
    /// The `hhea` ascender, as stored.
    pub hhea_ascender: i16,
    /// The `hhea` descender, as stored.
    pub hhea_descender: i16,
    /// The `hhea` line gap, as stored.
    pub hhea_line_gap: i16,
    /// The height of a lower-case x (`OS/2` version 2 and later).
    pub x_height: Option<i16>,
    /// The height of a flat capital (`OS/2` version 2 and later).
    pub cap_height: Option<i16>,
    /// The weight class, 1 to 1000 (400 regular, 600 semi-bold, 700 bold); 400 without an `OS/2` table.
    pub weight_class: u16,
    /// The distance of the top of the underline from the baseline (`post`); 0 without a `post` table.
    pub underline_position: i16,
    /// The thickness of the underline (`post`); 0 without a `post` table.
    pub underline_thickness: i16,
    /// Whether the font declares itself monospaced (`post`).
    pub is_fixed_pitch: bool,
    /// The italic angle in degrees, counter-clockwise from vertical (`post`); 0 for upright fonts.
    pub italic_angle: f32,
}

impl Metrics {
    /// The distance between baselines of consecutive lines: ascender − descender + line gap.
    #[must_use]
    pub fn line_height(&self) -> i32 {
        i32::from(self.ascender) - i32::from(self.descender) + i32::from(self.line_gap)
    }
}

/// The outline complexity a font declares for itself in `maxp` (version 1.0). The glyph loader
/// enforces these as limits, so a baker can size its buffers from them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaxProfile {
    /// The most points of a simple glyph.
    pub max_points: u16,
    /// The most contours of a simple glyph.
    pub max_contours: u16,
    /// The most points of a composite glyph, all components together.
    pub max_composite_points: u16,
    /// The most contours of a composite glyph, all components together.
    pub max_composite_contours: u16,
    /// The most components one composite glyph refers to directly.
    pub max_component_elements: u16,
    /// The deepest nesting of composite glyphs: 1 when components are all simple glyphs.
    pub max_component_depth: u16,
}

/// The horizontal metrics of one glyph, in font units.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HMetrics {
    /// How far the pen moves after the glyph.
    pub advance: u16,
    /// The distance from the pen position to the left edge of the outline.
    pub left_side_bearing: i16,
}

/// The `name` table identifiers of the strings a UI shows.
pub mod name_id {
    /// The font family ("Inter"); for non-RIBBI styles this may include the style ("Inter SemiBold").
    pub const FAMILY: u16 = 1;
    /// The style within the family ("Regular", "Bold Italic").
    pub const SUBFAMILY: u16 = 2;
    /// The full name ("Inter SemiBold").
    pub const FULL_NAME: u16 = 4;
    /// The version string.
    pub const VERSION: u16 = 5;
    /// The PostScript name.
    pub const POSTSCRIPT_NAME: u16 = 6;
    /// The typographic family, when it differs from [`FAMILY`].
    pub const TYPOGRAPHIC_FAMILY: u16 = 16;
    /// The typographic style, when it differs from [`SUBFAMILY`].
    pub const TYPOGRAPHIC_SUBFAMILY: u16 = 17;
}

/// One table (or a part of one) with bounds-checked big-endian random access.
///
/// `base` is the offset of `data` in the file, so errors can point into the file.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Table<'a> {
    data: &'a [u8],
    base: usize,
}

impl Table<'_> {
    fn len(self) -> usize {
        self.data.len()
    }

    fn u8(self, at: usize) -> Option<u8> {
        self.data.get(at).copied()
    }

    fn u16(self, at: usize) -> Option<u16> {
        let b = self.data.get(at..at.checked_add(2)?)?;
        Some(u16::from_be_bytes([b[0], b[1]]))
    }

    fn i16(self, at: usize) -> Option<i16> {
        self.u16(at).map(|v| v as i16)
    }

    fn u32(self, at: usize) -> Option<u32> {
        let b = self.data.get(at..at.checked_add(4)?)?;
        Some(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// The part of the table from `at` to its end.
    fn tail(self, at: usize) -> Option<Self> {
        Some(Self { data: self.data.get(at..)?, base: self.base + at })
    }

    /// The error for a read at `at` that did not fit.
    fn truncated(self, at: usize) -> Error {
        Error::Truncated { format: FORMAT, offset: self.base + at.min(self.data.len()) }
    }

    fn need_u16(self, at: usize) -> Result<u16> {
        self.u16(at).ok_or_else(|| self.truncated(at))
    }

    fn need_i16(self, at: usize) -> Result<i16> {
        self.i16(at).ok_or_else(|| self.truncated(at))
    }

    fn need_u32(self, at: usize) -> Result<u32> {
        self.u32(at).ok_or_else(|| self.truncated(at))
    }
}

/// How the code points of the chosen cmap subtable relate to Unicode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CmapEncoding {
    /// Unicode code points.
    Unicode,
    /// A Windows symbol font: glyphs usually sit at U+F000 + character code.
    Symbol,
    /// Mac Roman character codes.
    MacRoman,
}

/// The layout of the chosen cmap subtable, validated when the font is opened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CmapKind {
    /// Byte encoding table: 256 glyph ids of one byte each.
    Format0,
    /// Segment mapping to delta values.
    Format4 { segments: usize },
    /// Trimmed table mapping.
    Format6 { first: u32, count: u32 },
    /// Segmented coverage.
    Format12 { groups: usize },
}

/// The cmap subtable used for every lookup.
#[derive(Clone, Copy, Debug)]
struct Cmap<'a> {
    /// The subtable, from its first byte to the end of the cmap table.
    table: Table<'a>,
    kind: CmapKind,
    encoding: CmapEncoding,
}

/// Mac Roman 0x80..=0xFF as Unicode (all in the basic plane). Checked against Python's `mac_roman`
/// codec in the tests' cross-check script.
const MAC_ROMAN_HIGH: [u16; 128] = [
    0x00C4, 0x00C5, 0x00C7, 0x00C9, 0x00D1, 0x00D6, 0x00DC, 0x00E1, 0x00E0, 0x00E2, 0x00E4, 0x00E3, 0x00E5, 0x00E7,
    0x00E9, 0x00E8, 0x00EA, 0x00EB, 0x00ED, 0x00EC, 0x00EE, 0x00EF, 0x00F1, 0x00F3, 0x00F2, 0x00F4, 0x00F6, 0x00F5,
    0x00FA, 0x00F9, 0x00FB, 0x00FC, 0x2020, 0x00B0, 0x00A2, 0x00A3, 0x00A7, 0x2022, 0x00B6, 0x00DF, 0x00AE, 0x00A9,
    0x2122, 0x00B4, 0x00A8, 0x2260, 0x00C6, 0x00D8, 0x221E, 0x00B1, 0x2264, 0x2265, 0x00A5, 0x00B5, 0x2202, 0x2211,
    0x220F, 0x03C0, 0x222B, 0x00AA, 0x00BA, 0x03A9, 0x00E6, 0x00F8, 0x00BF, 0x00A1, 0x00AC, 0x221A, 0x0192, 0x2248,
    0x2206, 0x00AB, 0x00BB, 0x2026, 0x00A0, 0x00C0, 0x00C3, 0x00D5, 0x0152, 0x0153, 0x2013, 0x2014, 0x201C, 0x201D,
    0x2018, 0x2019, 0x00F7, 0x25CA, 0x00FF, 0x0178, 0x2044, 0x20AC, 0x2039, 0x203A, 0xFB01, 0xFB02, 0x2021, 0x00B7,
    0x201A, 0x201E, 0x2030, 0x00C2, 0x00CA, 0x00C1, 0x00CB, 0x00C8, 0x00CD, 0x00CE, 0x00CF, 0x00CC, 0x00D3, 0x00D4,
    0xF8FF, 0x00D2, 0x00DA, 0x00DB, 0x00D9, 0x0131, 0x02C6, 0x02DC, 0x00AF, 0x02D8, 0x02D9, 0x02DA, 0x00B8, 0x02DD,
    0x02DB, 0x02C7,
];

/// The Unicode character of a Mac Roman byte.
fn mac_roman_to_char(byte: u8) -> char {
    if byte < 0x80 {
        char::from(byte)
    } else {
        char::from_u32(u32::from(MAC_ROMAN_HIGH[usize::from(byte - 0x80)])).unwrap_or(char::REPLACEMENT_CHARACTER)
    }
}

/// The Mac Roman byte of a character, if it has one.
fn char_to_mac_roman(c: char) -> Option<u8> {
    let code = u32::from(c);
    if code < 0x80 {
        return Some(code as u8);
    }
    MAC_ROMAN_HIGH.iter().position(|&u| u32::from(u) == code).map(|i| (i + 0x80) as u8)
}

/// A parsed TrueType font, borrowing the file's bytes.
///
/// All measurements are in font units; see [`Metrics::units_per_em`].
#[derive(Clone, Debug)]
pub struct Font<'a> {
    glyf: Table<'a>,
    loca: Table<'a>,
    hmtx: Table<'a>,
    cmap: Cmap<'a>,
    name: Option<Table<'a>>,
    kern: Option<Table<'a>>,
    gpos: Option<kerning::Gpos<'a>>,
    num_h_metrics: u16,
    long_loca: bool,
    variable: bool,
    metrics: Metrics,
    max: MaxProfile,
}

/// The tables picked out of the directory.
#[derive(Default)]
struct Tables<'a> {
    head: Option<Table<'a>>,
    maxp: Option<Table<'a>>,
    hhea: Option<Table<'a>>,
    hmtx: Option<Table<'a>>,
    cmap: Option<Table<'a>>,
    loca: Option<Table<'a>>,
    glyf: Option<Table<'a>>,
    name: Option<Table<'a>>,
    os2: Option<Table<'a>>,
    post: Option<Table<'a>>,
    kern: Option<Table<'a>>,
    gpos: Option<Table<'a>>,
    cff: bool,
    variable: bool,
}

const TAG_TRUETYPE: u32 = 0x0001_0000;
const TAG_TRUE: u32 = u32::from_be_bytes(*b"true");
const TAG_OTTO: u32 = u32::from_be_bytes(*b"OTTO");
const TAG_TYP1: u32 = u32::from_be_bytes(*b"typ1");
const TAG_TTCF: u32 = u32::from_be_bytes(*b"ttcf");
const TAG_WOFF: u32 = u32::from_be_bytes(*b"wOFF");
const TAG_WOF2: u32 = u32::from_be_bytes(*b"wOF2");
/// What the 32-bit words of a whole font file add up to.
const FILE_CHECKSUM: u32 = 0xB1B0_AFBA;

/// The sum of the big-endian 32-bit words of `bytes`, the last one padded with zeros: the checksum
/// the table directory stores for every table.
fn checksum(bytes: &[u8]) -> u32 {
    let (words, rest) = bytes.as_chunks::<4>();
    let mut sum = words.iter().fold(0_u32, |sum, w| sum.wrapping_add(u32::from_be_bytes(*w)));
    if !rest.is_empty() {
        let mut last = [0_u8; 4];
        last[..rest.len()].copy_from_slice(rest);
        sum = sum.wrapping_add(u32::from_be_bytes(last));
    }
    sum
}

impl<'a> Font<'a> {
    /// Opens a font file: a single TrueType font, or the first face of a TrueType collection.
    ///
    /// Validates the table directory (every table must lie inside the file) and the tables every
    /// operation relies on: `head`, `maxp`, `hhea`, `hmtx`, `loca`, `glyf` and `cmap`, plus the
    /// structure of `GPOS` when present.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        Self::from_collection(data, 0)
    }

    /// The number of faces in a file: the face count of a TrueType collection, 1 for anything else
    /// that starts like a font.
    pub fn face_count(data: &[u8]) -> Result<u32> {
        let mut r = Reader::new(data, FORMAT);
        if r.u32_be()? == TAG_TTCF {
            r.skip(4)?;
            return r.u32_be();
        }
        Ok(1)
    }

    /// Opens face `index` of a TrueType collection (`.ttc`). A single-font file has exactly face 0.
    pub fn from_collection(data: &'a [u8], index: u32) -> Result<Self> {
        let (mut r, _) = Self::face_directory(data, index)?;
        let tables = Self::read_directory(data, &mut r)?;
        Self::from_tables(&tables)
    }

    /// Opens a font file like [`Font::new`] after checking its checksums with
    /// [`Font::verify_checksums`]: what a font baker uses, so that a damaged file is reported instead
    /// of baked.
    pub fn new_verified(data: &'a [u8]) -> Result<Self> {
        Self::verify_checksums(data, 0)?;
        Self::from_collection(data, 0)
    }

    /// Checks the checksum of every table of face `index` against the table directory and, for a
    /// single-font file, that the whole file adds up to the constant the format prescribes (which
    /// involves `checkSumAdjustment` of `head`). The first mismatch is an [`Error::Invalid`] at the
    /// directory record of the table (or at `checkSumAdjustment`).
    pub fn verify_checksums(data: &[u8], index: u32) -> Result<()> {
        let (mut r, collection) = Self::face_directory(data, index)?;
        let num_tables = usize::from(r.u16_be()?);
        r.skip(6)?;
        r.check_count(num_tables, 16)?;
        let mut adjustment_at = None;
        for _ in 0..num_tables {
            let at = r.pos();
            let tag = r.array::<4>()?;
            let expected = r.u32_be()?;
            let offset = r.u32_be()? as usize;
            let length = r.u32_be()? as usize;
            let bytes = offset
                .checked_add(length)
                .and_then(|end| data.get(offset..end))
                .ok_or_else(|| invalid(at, "table lies outside the file"))?;
            let mut sum = checksum(bytes);
            if &tag == b"head" {
                // The checksum of head is defined with checkSumAdjustment taken as zero.
                let adjustment = bytes.get(8..12).ok_or_else(|| invalid(at, "head table is too short"))?;
                sum = sum.wrapping_sub(checksum(adjustment));
                adjustment_at = Some(offset + 8);
            }
            if sum != expected {
                return Err(invalid(at, "table checksum mismatch"));
            }
        }
        // A collection shares tables between faces and has no whole-file checksum.
        if !collection
            && let Some(at) = adjustment_at
            && checksum(data) != FILE_CHECKSUM
        {
            return Err(invalid(at, "file checksum mismatch"));
        }
        Ok(())
    }

    /// Finds the table directory of face `index`: a reader positioned at its `numTables` field, and
    /// whether the file is a collection. Rejects the outline flavours this parser does not read.
    fn face_directory(data: &[u8], index: u32) -> Result<(Reader<'_>, bool)> {
        let mut r = Reader::new(data, FORMAT);
        let mut version = r.u32_be()?;
        let collection = version == TAG_TTCF;
        if collection {
            r.skip(4)?;
            let count = r.u32_be()?;
            if index >= count {
                return Err(invalid(8, "face index out of range"));
            }
            r.skip((index as usize).checked_mul(4).ok_or_else(|| r.truncated())?)?;
            let offset = r.u32_be()? as usize;
            r.seek(offset)?;
            version = r.u32_be()?;
        } else if index != 0 {
            return Err(invalid(0, "face index out of range"));
        }
        match version {
            TAG_TRUETYPE | TAG_TRUE => Ok((r, collection)),
            TAG_OTTO => Err(unsupported("CFF outlines (OTTO)")),
            TAG_TYP1 => Err(unsupported("PostScript Type 1 outlines")),
            TAG_WOFF | TAG_WOF2 => Err(unsupported("the WOFF container")),
            _ => Err(invalid(r.pos() - 4, "not a TrueType font")),
        }
    }

    /// Reads the table records at the reader's position, checking every one against the file.
    fn read_directory(data: &'a [u8], r: &mut Reader<'a>) -> Result<Tables<'a>> {
        let num_tables = usize::from(r.u16_be()?);
        r.skip(6)?;
        r.check_count(num_tables, 16)?;
        let mut tables = Tables::default();
        for _ in 0..num_tables {
            let at = r.pos();
            let tag = r.array::<4>()?;
            r.skip(4)?;
            let offset = r.u32_be()? as usize;
            let length = r.u32_be()? as usize;
            let bytes = offset
                .checked_add(length)
                .and_then(|end| data.get(offset..end))
                .ok_or_else(|| invalid(at, "table lies outside the file"))?;
            let table = Some(Table { data: bytes, base: offset });
            match &tag {
                b"head" => tables.head = table,
                b"maxp" => tables.maxp = table,
                b"hhea" => tables.hhea = table,
                b"hmtx" => tables.hmtx = table,
                b"cmap" => tables.cmap = table,
                b"loca" => tables.loca = table,
                b"glyf" => tables.glyf = table,
                b"name" => tables.name = table,
                b"OS/2" => tables.os2 = table,
                b"post" => tables.post = table,
                b"kern" => tables.kern = table,
                b"GPOS" => tables.gpos = table,
                b"CFF " | b"CFF2" => tables.cff = true,
                b"fvar" => tables.variable = true,
                _ => {}
            }
        }
        Ok(tables)
    }

    /// Validates the required tables and gathers the font-wide values.
    fn from_tables(tables: &Tables<'a>) -> Result<Self> {
        let Some(glyf) = tables.glyf else {
            return Err(if tables.cff { unsupported("CFF outlines") } else { invalid(0, "missing glyf table") });
        };
        let head = tables.head.ok_or_else(|| invalid(0, "missing head table"))?;
        let maxp = tables.maxp.ok_or_else(|| invalid(0, "missing maxp table"))?;
        let hhea = tables.hhea.ok_or_else(|| invalid(0, "missing hhea table"))?;
        let hmtx = tables.hmtx.ok_or_else(|| invalid(0, "missing hmtx table"))?;
        let loca = tables.loca.ok_or_else(|| invalid(0, "missing loca table"))?;
        let cmap = tables.cmap.ok_or_else(|| invalid(0, "missing cmap table"))?;

        if head.need_u32(12)? != 0x5F0F_3CF5 {
            return Err(invalid(head.base + 12, "bad head magic number"));
        }
        let units_per_em = head.need_u16(18)?;
        if units_per_em == 0 {
            return Err(invalid(head.base + 18, "unitsPerEm is zero"));
        }
        let bbox = BBox {
            x_min: head.need_i16(36)?,
            y_min: head.need_i16(38)?,
            x_max: head.need_i16(40)?,
            y_max: head.need_i16(42)?,
        };
        let long_loca = match head.need_u16(50)? {
            0 => false,
            1 => true,
            _ => return Err(invalid(head.base + 50, "bad indexToLocFormat")),
        };

        if maxp.need_u32(0)? != 0x0001_0000 {
            return Err(invalid(maxp.base, "TrueType outlines need a version 1.0 maxp table"));
        }
        let num_glyphs = maxp.need_u16(4)?;
        if num_glyphs == 0 {
            return Err(invalid(maxp.base + 4, "font has no glyphs"));
        }
        let max = MaxProfile {
            max_points: maxp.need_u16(6)?,
            max_contours: maxp.need_u16(8)?,
            max_composite_points: maxp.need_u16(10)?,
            max_composite_contours: maxp.need_u16(12)?,
            max_component_elements: maxp.need_u16(28)?,
            max_component_depth: maxp.need_u16(30)?,
        };
        let hhea_ascender = hhea.need_i16(4)?;
        let hhea_descender = hhea.need_i16(6)?;
        let hhea_line_gap = hhea.need_i16(8)?;
        let num_h_metrics = hhea.need_u16(34)?;
        if num_h_metrics == 0 {
            return Err(invalid(hhea.base + 34, "numberOfHMetrics is zero"));
        }
        if usize::from(num_h_metrics) * 4 > hmtx.len() {
            return Err(hmtx.truncated(hmtx.len()));
        }
        let loca_entry = if long_loca { 4 } else { 2 };
        if (usize::from(num_glyphs) + 1) * loca_entry > loca.len() {
            return Err(loca.truncated(loca.len()));
        }

        let mut metrics = Metrics {
            units_per_em,
            num_glyphs,
            bbox,
            ascender: hhea_ascender,
            descender: hhea_descender,
            line_gap: hhea_line_gap,
            hhea_ascender,
            hhea_descender,
            hhea_line_gap,
            x_height: None,
            cap_height: None,
            weight_class: 400,
            underline_position: 0,
            underline_thickness: 0,
            is_fixed_pitch: false,
            italic_angle: 0.0,
        };
        if let Some(os2) = tables.os2 {
            if let Some(weight) = os2.u16(4) {
                metrics.weight_class = weight;
            }
            if let (Some(ascender), Some(descender), Some(gap)) = (os2.i16(68), os2.i16(70), os2.i16(72))
                && (ascender != 0 || descender != 0)
            {
                metrics.ascender = ascender;
                metrics.descender = descender;
                metrics.line_gap = gap;
            }
            if os2.u16(0).is_some_and(|version| version >= 2) {
                metrics.x_height = os2.i16(86);
                metrics.cap_height = os2.i16(88);
            }
        }
        if let Some(post) = tables.post
            && let (Some(angle), Some(position), Some(thickness), Some(fixed)) =
                (post.u32(4), post.i16(8), post.i16(10), post.u32(12))
        {
            metrics.italic_angle = (angle as i32) as f32 / 65536.0;
            metrics.underline_position = position;
            metrics.underline_thickness = thickness;
            metrics.is_fixed_pitch = fixed != 0;
        }

        let cmap = Self::choose_cmap(cmap)?;
        let gpos = match tables.gpos {
            Some(table) => kerning::Gpos::parse(table)?,
            None => None,
        };
        Ok(Self {
            glyf,
            loca,
            hmtx,
            cmap,
            name: tables.name,
            kern: tables.kern,
            gpos,
            num_h_metrics,
            long_loca,
            variable: tables.variable,
            metrics,
            max,
        })
    }

    /// Picks the subtable that covers the most of Unicode in a format this parser reads.
    fn choose_cmap(cmap: Table<'a>) -> Result<Cmap<'a>> {
        let count = usize::from(cmap.need_u16(2)?);
        let mut best: Option<(u32, Cmap<'a>)> = None;
        for i in 0..count {
            let record = 4 + i * 8;
            let platform = cmap.need_u16(record)?;
            let encoding = cmap.need_u16(record + 2)?;
            let offset = cmap.need_u32(record + 4)? as usize;
            let (rank, encoding) = match (platform, encoding) {
                (3, 10) | (0, 4 | 6) => (4, CmapEncoding::Unicode),
                (3, 1) | (0, 0..=3) => (3, CmapEncoding::Unicode),
                (3, 0) => (2, CmapEncoding::Symbol),
                (1, 0) => (1, CmapEncoding::MacRoman),
                _ => continue,
            };
            let Some(table) = cmap.tail(offset) else {
                return Err(invalid(cmap.base + record + 4, "cmap subtable lies outside the table"));
            };
            let Some((format_rank, kind)) = Self::cmap_kind(table) else { continue };
            let score = rank * 10 + format_rank;
            if best.as_ref().is_none_or(|(s, _)| score > *s) {
                best = Some((score, Cmap { table, kind, encoding }));
            }
        }
        best.map(|(_, cmap)| cmap).ok_or(unsupported("a cmap without a Unicode subtable of format 0, 4, 6 or 12"))
    }

    /// Checks that a cmap subtable is one of the readable formats and that its arrays fit.
    fn cmap_kind(table: Table<'a>) -> Option<(u32, CmapKind)> {
        match table.u16(0)? {
            0 => (table.len() >= 262).then_some((1, CmapKind::Format0)),
            4 => {
                let seg_x2 = usize::from(table.u16(6)?);
                let segments = seg_x2 / 2;
                // Four parallel arrays and the reserved pad follow the 14-byte header.
                (seg_x2 % 2 == 0 && segments > 0 && 16 + segments * 8 <= table.len())
                    .then_some((3, CmapKind::Format4 { segments }))
            }
            6 => {
                let first = u32::from(table.u16(6)?);
                let count = u32::from(table.u16(8)?);
                (10 + count as usize * 2 <= table.len()).then_some((2, CmapKind::Format6 { first, count }))
            }
            12 => {
                let groups = table.u32(12)? as usize;
                (groups.checked_mul(12)?.checked_add(16)? <= table.len()).then_some((4, CmapKind::Format12 { groups }))
            }
            _ => None,
        }
    }

    /// The font-wide measurements.
    #[must_use]
    pub const fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    /// The outline complexity limits of `maxp`, which the glyph loader enforces.
    #[must_use]
    pub const fn max_profile(&self) -> &MaxProfile {
        &self.max
    }

    /// Whether the file is a variable font (has an `fvar` table). Only its default instance is read.
    #[must_use]
    pub const fn is_variable(&self) -> bool {
        self.variable
    }

    /// Font units per em square.
    #[must_use]
    pub const fn units_per_em(&self) -> u16 {
        self.metrics.units_per_em
    }

    /// The number of glyphs; valid glyph ids are `0..glyph_count()`.
    #[must_use]
    pub const fn glyph_count(&self) -> u16 {
        self.metrics.num_glyphs
    }

    /// The glyph that draws `c`, or `None` when the font has no glyph for it (the caller then falls
    /// back to another font or to [`GlyphId::NOTDEF`]).
    #[must_use]
    pub fn glyph_index(&self, c: char) -> Option<GlyphId> {
        let code = match self.cmap.encoding {
            CmapEncoding::Unicode => u32::from(c),
            CmapEncoding::MacRoman => u32::from(char_to_mac_roman(c)?),
            CmapEncoding::Symbol => {
                let code = u32::from(c);
                if code < 0x100
                    && let Some(glyph) = self.lookup_code(0xF000 + code)
                {
                    return Some(glyph);
                }
                code
            }
        };
        self.lookup_code(code)
    }

    /// Maps a character code of the chosen subtable to a glyph, dropping glyph 0 and ids past the
    /// end of the font.
    fn lookup_code(&self, code: u32) -> Option<GlyphId> {
        let t = self.cmap.table;
        let glyph = match self.cmap.kind {
            CmapKind::Format0 => u32::from(t.u8(6 + usize::try_from(code).ok().filter(|&c| c < 256)?)?),
            CmapKind::Format4 { segments } => {
                if code > 0xFFFF {
                    return None;
                }
                // The first segment whose end code is not below the character.
                let (mut lo, mut hi) = (0, segments);
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    if u32::from(t.u16(14 + mid * 2)?) < code {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                if lo == segments {
                    return None;
                }
                u32::from(Self::format4_glyph(t, segments, lo, code)?)
            }
            CmapKind::Format6 { first, count } => {
                let index = code.checked_sub(first).filter(|&i| i < count)?;
                u32::from(t.u16(10 + index as usize * 2)?)
            }
            CmapKind::Format12 { groups } => {
                let (mut lo, mut hi) = (0, groups);
                let mut found = None;
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    let group = 16 + mid * 12;
                    let (start, end) = (t.u32(group)?, t.u32(group + 4)?);
                    if code < start {
                        hi = mid;
                    } else if code > end {
                        lo = mid + 1;
                    } else {
                        found = Some(t.u32(group + 8)?.checked_add(code - start)?);
                        break;
                    }
                }
                found?
            }
        };
        self.checked_glyph(glyph)
    }

    /// A glyph id from a cmap, unless it is 0 or past the end of the font.
    fn checked_glyph(&self, glyph: u32) -> Option<GlyphId> {
        (glyph != 0 && glyph < u32::from(self.metrics.num_glyphs)).then_some(GlyphId(glyph as u16))
    }

    /// The glyph of `code` in segment `segment` of a format 4 subtable, if the segment contains it.
    fn format4_glyph(t: Table<'a>, segments: usize, segment: usize, code: u32) -> Option<u16> {
        let starts = 16 + segments * 2;
        let deltas = starts + segments * 2;
        let ranges = deltas + segments * 2;
        let start = u32::from(t.u16(starts + segment * 2)?);
        if code < start {
            return None;
        }
        let delta = t.u16(deltas + segment * 2)?;
        let range_at = ranges + segment * 2;
        let range_offset = usize::from(t.u16(range_at)?);
        if range_offset == 0 {
            return Some((code as u16).wrapping_add(delta));
        }
        // The offset is relative to the position of the range-offset entry itself.
        let glyph = t.u16(range_at + range_offset + (code - start) as usize * 2)?;
        if glyph == 0 { Some(0) } else { Some(glyph.wrapping_add(delta)) }
    }

    /// Calls `visit` for every character the font maps to a glyph, in increasing code order.
    ///
    /// Segments or groups that go backwards in a damaged subtable are skipped, so at most 65 536
    /// (formats 0, 4, 6) or 1 114 112 (format 12) codes are ever examined.
    pub fn for_each_mapping(&self, mut visit: impl FnMut(char, GlyphId)) {
        let t = self.cmap.table;
        let mut emit = |code: u32, glyph: u32| {
            let c = match self.cmap.encoding {
                CmapEncoding::Unicode | CmapEncoding::Symbol => char::from_u32(code),
                CmapEncoding::MacRoman => u8::try_from(code).ok().map(mac_roman_to_char),
            };
            if let (Some(c), Some(glyph)) = (c, self.checked_glyph(glyph)) {
                visit(c, glyph);
            }
        };
        match self.cmap.kind {
            CmapKind::Format0 => {
                for code in 0..256_u32 {
                    if let Some(glyph) = t.u8(6 + code as usize) {
                        emit(code, u32::from(glyph));
                    }
                }
            }
            CmapKind::Format4 { segments } => {
                let mut next = 0_u32;
                for segment in 0..segments {
                    let (Some(end), Some(start)) = (t.u16(14 + segment * 2), t.u16(16 + segments * 2 + segment * 2))
                    else {
                        break;
                    };
                    let (start, end) = (u32::from(start).max(next), u32::from(end));
                    for code in start..=end {
                        if let Some(glyph) = Self::format4_glyph(t, segments, segment, code) {
                            emit(code, u32::from(glyph));
                        }
                    }
                    next = next.max(end + 1);
                }
            }
            CmapKind::Format6 { first, count } => {
                for index in 0..count {
                    if let Some(glyph) = t.u16(10 + index as usize * 2) {
                        emit(first + index, u32::from(glyph));
                    }
                }
            }
            CmapKind::Format12 { groups } => {
                let mut next = 0_u32;
                for group in 0..groups {
                    let at = 16 + group * 12;
                    let (Some(first), Some(end), Some(first_glyph)) = (t.u32(at), t.u32(at + 4), t.u32(at + 8)) else {
                        break;
                    };
                    let end = end.min(0x0010_FFFF);
                    for code in first.max(next)..=end {
                        if let Some(glyph) = first_glyph.checked_add(code - first) {
                            emit(code, glyph);
                        }
                    }
                    if end >= next {
                        next = end + 1;
                    }
                }
            }
        }
    }

    /// Every character the font maps to a glyph, in increasing code order.
    #[must_use]
    pub fn mappings(&self) -> Vec<(char, GlyphId)> {
        let mut out = Vec::new();
        self.for_each_mapping(|c, glyph| out.push((c, glyph)));
        out
    }

    /// The horizontal metrics of a glyph as a rasteriser uses them: those of `hmtx`, unless the glyph
    /// is a composite with a component flagged `USE_MY_METRICS`, in which case that component's
    /// metrics apply (followed through nested composites; see [`Font::metrics_source`]). In a
    /// well-built static font the two agree.
    #[must_use]
    pub fn h_metrics(&self, glyph: GlyphId) -> HMetrics {
        let source = self.metrics_source(glyph);
        HMetrics { advance: self.advance(source), left_side_bearing: self.left_side_bearing(source) }
    }

    /// The horizontal advance of a glyph as stored in `hmtx`. Glyphs past `numberOfHMetrics` share
    /// the last advance (the monospace rule of `hmtx`); a glyph id outside the font advances by 0.
    #[must_use]
    pub fn advance(&self, glyph: GlyphId) -> u16 {
        if glyph.0 >= self.metrics.num_glyphs {
            return 0;
        }
        let index = glyph.0.min(self.num_h_metrics - 1);
        self.hmtx.u16(usize::from(index) * 4).unwrap_or(0)
    }

    /// The left side bearing of a glyph as stored in `hmtx`: the distance from the pen position to the
    /// left edge of its outline. 0 for a glyph id outside the font or a bearing the table does not hold.
    #[must_use]
    pub fn left_side_bearing(&self, glyph: GlyphId) -> i16 {
        if glyph.0 >= self.metrics.num_glyphs {
            return 0;
        }
        let at = if glyph.0 < self.num_h_metrics {
            usize::from(glyph.0) * 4 + 2
        } else {
            usize::from(self.num_h_metrics) * 4 + usize::from(glyph.0 - self.num_h_metrics) * 2
        };
        self.hmtx.i16(at).unwrap_or(0)
    }

    /// The adjustment to the advance of `left` when `right` follows it, in font units (negative moves
    /// the glyphs together).
    ///
    /// When the font's `GPOS` table has pair-adjustment lookups under the feature `kern` (scripts
    /// `DFLT` and `latn`, default language; every `kern` feature if the font has neither script), the
    /// result is the sum over those lookups of the first glyph's `XAdvance`, each lookup contributing
    /// its first matching subtable, as a shaper applies them. Only when `GPOS` has no such lookups is
    /// the `kern` table consulted: the sum of its horizontal format 0 subtables. The result is
    /// clamped to the `i16` range; a font without kerning yields 0.
    #[must_use]
    pub fn kerning(&self, left: GlyphId, right: GlyphId) -> i16 {
        let value = match (&self.gpos, self.kern) {
            (Some(gpos), _) if gpos.has_pairs() => gpos.pair_advance(left.0, right.0),
            (_, Some(kern)) => kerning::kern_table_pair(kern, left.0, right.0),
            _ => 0,
        };
        value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
    }

    /// Whether [`Font::kerning`] can return anything but 0: the font has `GPOS` pair lookups for
    /// `kern`, or a `kern` table.
    #[must_use]
    pub fn has_kerning(&self) -> bool {
        self.gpos.as_ref().is_some_and(kerning::Gpos::has_pairs) || self.kern.is_some()
    }

    /// A string of the `name` table by identifier (see [`name_id`]), decoded from UTF-16BE (Windows
    /// and Unicode platforms) or Mac Roman. The Windows US-English record is preferred, then Unicode,
    /// then any Windows language, then Macintosh.
    #[must_use]
    pub fn name(&self, id: u16) -> Option<String> {
        let t = self.name?;
        let count = usize::from(t.u16(2)?);
        let storage = usize::from(t.u16(4)?);
        let mut best: Option<(u32, bool, &[u8])> = None;
        for i in 0..count {
            let record = 6 + i * 12;
            // A record past the end of the table ends the scan; what was found before it stands.
            let (Some(platform), Some(encoding), Some(language), Some(name), Some(length), Some(offset)) = (
                t.u16(record),
                t.u16(record + 2),
                t.u16(record + 4),
                t.u16(record + 6),
                t.u16(record + 8),
                t.u16(record + 10),
            ) else {
                break;
            };
            if name != id {
                continue;
            }
            let (rank, utf16) = match (platform, encoding) {
                (3, 0 | 1 | 10) if language == 0x0409 => (5, true),
                (0, _) => (4, true),
                (3, 0 | 1 | 10) => (3, true),
                (1, 0) if language == 0 => (2, false),
                (1, 0) => (1, false),
                _ => continue,
            };
            let start = storage + usize::from(offset);
            let Some(bytes) = t.data.get(start..start + usize::from(length)) else { continue };
            if best.is_none_or(|(r, _, _)| rank > r) {
                best = Some((rank, utf16, bytes));
            }
        }
        let (_, utf16, bytes) = best?;
        Some(if utf16 {
            let units = bytes.as_chunks::<2>().0.iter().map(|pair| u16::from_be_bytes(*pair));
            char::decode_utf16(units).map(|unit| unit.unwrap_or(char::REPLACEMENT_CHARACTER)).collect()
        } else {
            bytes.iter().map(|&byte| mac_roman_to_char(byte)).collect()
        })
    }

    /// The family name (name id 1), e.g. "Inter".
    #[must_use]
    pub fn family_name(&self) -> Option<String> {
        self.name(name_id::FAMILY)
    }

    /// The style name within the family (name id 2), e.g. "Regular".
    #[must_use]
    pub fn subfamily_name(&self) -> Option<String> {
        self.name(name_id::SUBFAMILY)
    }

    /// The full name (name id 4), e.g. "Inter SemiBold".
    #[must_use]
    pub fn full_name(&self) -> Option<String> {
        self.name(name_id::FULL_NAME)
    }
}

#[cfg(test)]
mod tests;
