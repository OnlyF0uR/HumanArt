//! Tests of the TrueType parser: on synthetic fonts whose every table is written by hand below, on
//! the fonts the UI ships (compared line by line with what the independent Python parser
//! `tools/ttf_crosscheck.py` read from them) and under mutation `[ROB-01]`.

use super::PathCommand::{Close, MoveTo, QuadTo};
use super::kerning::tests::{KernPairs, kern_table, sample_gpos};
use super::*;
use std::time::Duration;

const INTER_REGULAR: &[u8] = include_bytes!("../../../../data/fonts/Inter-Regular.ttf");
const INTER_SEMIBOLD: &[u8] = include_bytes!("../../../../data/fonts/Inter-SemiBold.ttf");
const INTER_ITALIC: &[u8] = include_bytes!("../../../../data/fonts/Inter-Italic.ttf");
const JETBRAINS_MONO: &[u8] = include_bytes!("../../../../data/fonts/JetBrainsMono-Regular.ttf");
const LUCIDE: &[u8] = include_bytes!("../../../../data/fonts/Lucide.ttf");

// ---------------------------------------------------------------------------------------------
// A font builder: enough of the format to write any table layout a test needs.
// ---------------------------------------------------------------------------------------------

/// Big-endian bytes of a list of 16-bit words.
fn words(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_be_bytes()).collect()
}

/// Lays tables out as an sfnt file with correct table checksums and `checkSumAdjustment`.
fn sfnt(version: u32, tables: &[([u8; 4], Vec<u8>)]) -> Vec<u8> {
    let mut sorted = tables.to_vec();
    sorted.sort_by_key(|t| t.0);
    let directory = 12 + 16 * sorted.len();
    let mut out = version.to_be_bytes().to_vec();
    out.extend(words(&[sorted.len() as u16, 0, 0, 0]));
    let mut body = Vec::new();
    let mut head_at = None;
    for (tag, data) in &sorted {
        let offset = directory + body.len();
        if tag == b"head" {
            head_at = Some(offset);
        }
        out.extend(tag);
        out.extend(checksum(data).to_be_bytes());
        out.extend((offset as u32).to_be_bytes());
        out.extend((data.len() as u32).to_be_bytes());
        body.extend(data);
        while body.len() % 4 != 0 {
            body.push(0);
        }
    }
    out.extend(body);
    if let Some(at) = head_at {
        let adjustment = FILE_CHECKSUM.wrapping_sub(checksum(&out));
        out[at + 8..at + 12].copy_from_slice(&adjustment.to_be_bytes());
    }
    out
}

/// One outline point: x, y, on-curve.
type Pt = (i16, i16, bool);

/// A simple glyph with every coordinate stored as a 16-bit delta and no instructions.
fn simple(contours: &[&[Pt]]) -> Vec<u8> {
    let points: Vec<Pt> = contours.concat();
    let min = |f: fn(&Pt) -> i16| points.iter().map(f).min().unwrap_or(0) as u16;
    let max = |f: fn(&Pt) -> i16| points.iter().map(f).max().unwrap_or(0) as u16;
    let mut out = words(&[contours.len() as u16, min(|p| p.0), min(|p| p.1), max(|p| p.0), max(|p| p.1)]);
    let mut end = 0;
    for contour in contours {
        end += contour.len();
        out.extend(((end - 1) as u16).to_be_bytes());
    }
    out.extend(words(&[0]));
    out.extend(points.iter().map(|p| u8::from(p.2)));
    for axis in [|p: &Pt| p.0, |p: &Pt| p.1] {
        let mut previous = 0;
        for p in &points {
            out.extend((axis(p) - previous).to_be_bytes());
            previous = axis(p);
        }
    }
    out
}

/// How a component is placed.
#[derive(Clone, Copy)]
enum Args {
    /// An offset in font units.
    Offset(i16, i16),
    /// Point numbers to match: of the glyph so far, of the component.
    Points(u16, u16),
}

/// The transform of a component.
#[derive(Clone, Copy)]
enum Xform {
    None,
    Scale(f32),
    XY(f32, f32),
    /// As stored: xscale, scale01, scale10, yscale.
    Matrix([f32; 4]),
}

/// One component of a composite glyph; `flags` are extra flag bits (`USE_MY_METRICS`, ...).
#[derive(Clone, Copy)]
struct Comp {
    glyph: u16,
    flags: u16,
    args: Args,
    xform: Xform,
}

/// A component at an offset, untransformed.
fn at(glyph: u16, x: i16, y: i16) -> Comp {
    Comp { glyph, flags: 0, args: Args::Offset(x, y), xform: Xform::None }
}

const F_USE_MY_METRICS: u16 = 0x0200;
const F_SCALED_OFFSET: u16 = 0x0800;

/// A composite glyph; arguments are stored as bytes when they fit.
fn composite(components: &[Comp]) -> Vec<u8> {
    let mut out = words(&[0xFFFF, 0, 0, 0, 0]);
    for (i, c) in components.iter().enumerate() {
        let mut flags = c.flags | if i + 1 < components.len() { 0x0020 } else { 0 };
        let (a, b, wide) = match c.args {
            Args::Offset(x, y) => {
                flags |= 0x0002;
                (x as u16, y as u16, i8::try_from(x).is_err() || i8::try_from(y).is_err())
            }
            Args::Points(p, q) => (p, q, p > 255 || q > 255),
        };
        flags |= u16::from(wide);
        let scales: Vec<f32> = match c.xform {
            Xform::None => vec![],
            Xform::Scale(s) => {
                flags |= 0x0008;
                vec![s]
            }
            Xform::XY(x, y) => {
                flags |= 0x0040;
                vec![x, y]
            }
            Xform::Matrix(m) => {
                flags |= 0x0080;
                m.to_vec()
            }
        };
        out.extend(words(&[flags, c.glyph]));
        if wide {
            out.extend(words(&[a, b]));
        } else {
            out.extend([a as u8, b as u8]);
        }
        for s in scales {
            out.extend(((s * 16384.0).round() as i16).to_be_bytes());
        }
    }
    out
}

/// A cmap table of the given (platform, encoding, subtable) records.
fn cmap_table(subtables: &[(u16, u16, Vec<u8>)]) -> Vec<u8> {
    let mut out = words(&[0, subtables.len() as u16]);
    let mut offset = 4 + 8 * subtables.len();
    for (platform, encoding, data) in subtables {
        out.extend(words(&[*platform, *encoding]));
        out.extend((offset as u32).to_be_bytes());
        offset += data.len();
    }
    for (_, _, data) in subtables {
        out.extend(data);
    }
    out
}

/// A format 0 subtable mapping the given bytes to glyphs.
fn cmap0(map: &[(u8, u8)]) -> Vec<u8> {
    let mut out = words(&[0, 262, 0]);
    let mut ids = [0_u8; 256];
    for &(code, glyph) in map {
        ids[usize::from(code)] = glyph;
    }
    out.extend(ids);
    out
}

/// A format 4 subtable of (start, end, delta, glyph ids) segments; a segment with glyph ids uses
/// `idRangeOffset`. The closing 0xFFFF segment is added.
fn cmap4(segments: &[(u16, u16, u16, &[u16])]) -> Vec<u8> {
    let mut segments = segments.to_vec();
    segments.push((0xFFFF, 0xFFFF, 1, &[]));
    let n = segments.len();
    let mut ends = Vec::new();
    let mut starts = Vec::new();
    let mut deltas = Vec::new();
    let mut ranges = Vec::new();
    let mut ids: Vec<u16> = Vec::new();
    for (i, &(start, end, delta, glyphs)) in segments.iter().enumerate() {
        starts.push(start);
        ends.push(end);
        deltas.push(delta);
        // Relative to the position of this very idRangeOffset entry.
        ranges.push(if glyphs.is_empty() { 0 } else { ((n - i) * 2 + ids.len() * 2) as u16 });
        ids.extend(glyphs);
    }
    let length = 16 + n * 8 + ids.len() * 2;
    let mut out = words(&[4, length as u16, 0, (n * 2) as u16, 0, 0, 0]);
    out.extend(words(&ends));
    out.extend(words(&[0]));
    out.extend(words(&starts));
    out.extend(words(&deltas));
    out.extend(words(&ranges));
    out.extend(words(&ids));
    out
}

/// A format 6 subtable: glyphs of consecutive codes from `first`.
fn cmap6(first: u16, glyphs: &[u16]) -> Vec<u8> {
    let mut out = words(&[6, (10 + glyphs.len() * 2) as u16, 0, first, glyphs.len() as u16]);
    out.extend(words(glyphs));
    out
}

/// A format 12 subtable of (first code, last code, first glyph) groups.
fn cmap12(groups: &[(u32, u32, u32)]) -> Vec<u8> {
    let mut out = words(&[12, 0]);
    out.extend((16 + groups.len() as u32 * 12).to_be_bytes());
    out.extend(0_u32.to_be_bytes());
    out.extend((groups.len() as u32).to_be_bytes());
    for &(first, last, glyph) in groups {
        out.extend([first, last, glyph].iter().flat_map(|v| v.to_be_bytes()));
    }
    out
}

/// A (3, 1) format 4 cmap mapping `'A'`, `'B'`, ... to glyphs 1, 2, ... `count`.
fn letters_cmap(count: u16) -> Vec<u8> {
    cmap_table(&[(3, 1, cmap4(&[(0x41, 0x40 + count, 1_u16.wrapping_sub(0x41), &[])]))])
}

/// A name table of (platform, encoding, language, name id, bytes) records.
fn name_table(records: &[(u16, u16, u16, u16, Vec<u8>)]) -> Vec<u8> {
    let mut out = words(&[0, records.len() as u16, (6 + 12 * records.len()) as u16]);
    let mut offset = 0;
    for (platform, encoding, language, id, bytes) in records {
        out.extend(words(&[*platform, *encoding, *language, *id, bytes.len() as u16, offset]));
        offset += bytes.len() as u16;
    }
    for record in records {
        out.extend(&record.4);
    }
    out
}

fn utf16(text: &str) -> Vec<u8> {
    text.encode_utf16().flat_map(u16::to_be_bytes).collect()
}

/// The glyphs, metrics and tables of a font under construction.
#[derive(Clone)]
struct Builder {
    glyphs: Vec<Vec<u8>>,
    metrics: Vec<(u16, i16)>,
    /// `numberOfHMetrics`; all glyphs when `None`.
    num_h_metrics: Option<u16>,
    cmap: Option<Vec<u8>>,
    /// maxPoints, maxContours, maxCompositePoints, maxCompositeContours, maxComponentElements,
    /// maxComponentDepth.
    max: [u16; 6],
    long_loca: bool,
    extra: Vec<([u8; 4], Vec<u8>)>,
}

impl Builder {
    /// A font whose glyph 0 is an empty `.notdef`.
    fn new() -> Self {
        Self {
            glyphs: vec![vec![]],
            metrics: vec![(500, 0)],
            num_h_metrics: None,
            cmap: None,
            max: [1000, 100, 1000, 100, 16, 4],
            long_loca: false,
            extra: Vec::new(),
        }
    }

    /// Adds a glyph and returns its id.
    fn glyph(&mut self, data: Vec<u8>, advance: u16, lsb: i16) -> u16 {
        self.glyphs.push(data);
        self.metrics.push((advance, lsb));
        (self.glyphs.len() - 1) as u16
    }

    #[allow(clippy::trivially_copy_pass_by_ref, reason = "call sites pass byte-string literals")]
    fn with(mut self, tag: &[u8; 4], data: Vec<u8>) -> Self {
        self.extra.push((*tag, data));
        self
    }

    fn tables(&self) -> Vec<([u8; 4], Vec<u8>)> {
        let count = self.glyphs.len() as u16;
        let mut glyf = Vec::new();
        let mut offsets = vec![0_u32];
        for glyph in &self.glyphs {
            glyf.extend(glyph);
            if glyf.len() % 2 == 1 {
                glyf.push(0);
            }
            offsets.push(glyf.len() as u32);
        }
        let loca: Vec<u8> = if self.long_loca {
            offsets.iter().flat_map(|o| o.to_be_bytes()).collect()
        } else {
            offsets.iter().flat_map(|o| ((o / 2) as u16).to_be_bytes()).collect()
        };
        let num_h_metrics = self.num_h_metrics.unwrap_or(count);
        let mut hmtx = Vec::new();
        for (i, &(advance, lsb)) in self.metrics.iter().enumerate() {
            if i < usize::from(num_h_metrics) {
                hmtx.extend(advance.to_be_bytes());
            }
            hmtx.extend(lsb.to_be_bytes());
        }
        let mut head = vec![0_u8; 54];
        head[0..4].copy_from_slice(&0x0001_0000_u32.to_be_bytes());
        head[12..16].copy_from_slice(&0x5F0F_3CF5_u32.to_be_bytes());
        head[18..20].copy_from_slice(&1000_u16.to_be_bytes());
        head[36..44].copy_from_slice(&words(&[(-10_i16) as u16, (-20_i16) as u16, 900, 800]));
        head[50..52].copy_from_slice(&u16::from(self.long_loca).to_be_bytes());
        let mut maxp = vec![0_u8; 32];
        maxp[0..4].copy_from_slice(&0x0001_0000_u32.to_be_bytes());
        maxp[4..6].copy_from_slice(&count.to_be_bytes());
        maxp[6..14].copy_from_slice(&words(&self.max[..4]));
        maxp[28..32].copy_from_slice(&words(&self.max[4..]));
        let mut hhea = vec![0_u8; 36];
        hhea[0..4].copy_from_slice(&0x0001_0000_u32.to_be_bytes());
        hhea[4..10].copy_from_slice(&words(&[800, (-200_i16) as u16, 90]));
        hhea[34..36].copy_from_slice(&num_h_metrics.to_be_bytes());
        let cmap = self.cmap.clone().unwrap_or_else(|| letters_cmap(count - 1));
        let mut tables = vec![
            (*b"head", head),
            (*b"maxp", maxp),
            (*b"hhea", hhea),
            (*b"hmtx", hmtx),
            (*b"loca", loca),
            (*b"glyf", glyf),
            (*b"cmap", cmap),
        ];
        tables.extend(self.extra.iter().cloned());
        tables
    }

    fn build(&self) -> Vec<u8> {
        sfnt(TAG_TRUETYPE, &self.tables())
    }
}

/// The triangle (0, 0) (100, 0) (0, 200) every composite test builds on.
const TRIANGLE: &[Pt] = &[(0, 0, true), (100, 0, true), (0, 200, true)];

/// A font with `.notdef` and the triangle as glyph 1 (`'A'`).
fn triangle_font() -> Builder {
    let mut b = Builder::new();
    b.glyph(simple(&[TRIANGLE]), 500, 5);
    b
}

/// The points of a path that has only straight segments.
fn corners(path: &Path) -> Vec<[f32; 2]> {
    path.commands
        .iter()
        .filter_map(|c| match *c {
            PathCommand::MoveTo(p) | PathCommand::LineTo(p) => Some(p),
            PathCommand::QuadTo(..) => panic!("unexpected curve"),
            PathCommand::Close => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Synthetic fonts.
// ---------------------------------------------------------------------------------------------

#[test]
fn directory_metrics_names_and_checksums() {
    let mut os2 = vec![0_u8; 96];
    os2[0..2].copy_from_slice(&4_u16.to_be_bytes());
    os2[4..6].copy_from_slice(&600_u16.to_be_bytes());
    os2[68..74].copy_from_slice(&words(&[750, (-250_i16) as u16, 100]));
    os2[86..90].copy_from_slice(&words(&[480, 700]));
    let mut post = vec![0_u8; 32];
    post[0..4].copy_from_slice(&0x0003_0000_u32.to_be_bytes());
    post[4..8].copy_from_slice(&(-(12_i32 << 16) - 0x8000).to_be_bytes());
    post[8..12].copy_from_slice(&words(&[(-75_i16) as u16, 50]));
    post[12..16].copy_from_slice(&1_u32.to_be_bytes());
    let name = name_table(&[
        (1, 0, 0, 1, b"Mac Family".to_vec()),
        (3, 1, 0x0407, 1, utf16("Deutsche Familie")),
        (3, 1, 0x0409, 1, utf16("Win Family")),
        (3, 1, 0x0407, 2, utf16("Fett")),
        (1, 0, 0, 4, b"Caf\x8E Mono".to_vec()),
        (2, 0, 0, 6, b"ignored".to_vec()),
    ]);
    let builder = triangle_font().with(b"OS/2", os2).with(b"post", post).with(b"name", name);
    let data = builder.build();
    let font = Font::new_verified(&data).unwrap();
    assert_eq!(Font::face_count(&data).unwrap(), 1);
    let m = *font.metrics();
    assert_eq!((m.units_per_em, m.num_glyphs), (1000, 2));
    assert_eq!((font.units_per_em(), font.glyph_count()), (1000, 2));
    assert_eq!(m.bbox, BBox { x_min: -10, y_min: -20, x_max: 900, y_max: 800 });
    assert_eq!((m.bbox.width(), m.bbox.height()), (910, 820));
    assert_eq!((m.ascender, m.descender, m.line_gap), (750, -250, 100));
    assert_eq!((m.hhea_ascender, m.hhea_descender, m.hhea_line_gap), (800, -200, 90));
    assert_eq!(m.line_height(), 1100);
    assert_eq!((m.x_height, m.cap_height, m.weight_class), (Some(480), Some(700), 600));
    assert_eq!((m.underline_position, m.underline_thickness, m.is_fixed_pitch), (-75, 50, true));
    assert_eq!(m.italic_angle, -12.5);
    assert_eq!(
        *font.max_profile(),
        MaxProfile {
            max_points: 1000,
            max_contours: 100,
            max_composite_points: 1000,
            max_composite_contours: 100,
            max_component_elements: 16,
            max_component_depth: 4,
        }
    );
    assert!(!font.is_variable() && !font.has_kerning());
    // Windows US English wins over other Windows languages, those over Macintosh.
    assert_eq!(font.family_name().as_deref(), Some("Win Family"));
    assert_eq!(font.subfamily_name().as_deref(), Some("Fett"));
    assert_eq!(font.full_name().as_deref(), Some("Café Mono"));
    assert_eq!(font.name(name_id::POSTSCRIPT_NAME), None);

    // Without OS/2 and post the hhea values are the line metrics and the rest is unknown.
    let plain = triangle_font().with(b"fvar", vec![0; 16]).build();
    let font = Font::new_verified(&plain).unwrap();
    let m = font.metrics();
    assert_eq!((m.ascender, m.descender, m.line_gap, m.weight_class), (800, -200, 90, 400));
    assert_eq!((m.x_height, m.cap_height, m.underline_thickness), (None, None, 0));
    assert_eq!(font.family_name(), None);
    assert!(font.is_variable(), "an fvar table marks a variable font");
    // An OS/2 table whose typographic values are both zero does not override hhea; version 1 has
    // no x-height.
    let font_data = triangle_font().with(b"OS/2", vec![0; 78]).build();
    let font = Font::new(&font_data).unwrap();
    assert_eq!((font.metrics().ascender, font.metrics().x_height, font.metrics().weight_class), (800, None, 0));

    // A flipped bit in a table: the font still opens, the checksum check names the table record.
    let glyf_at = data.windows(4).position(|w| w == b"glyf").unwrap();
    let glyf_offset = u32::from_be_bytes(data[glyf_at + 8..glyf_at + 12].try_into().unwrap()) as usize;
    let mut damaged = data.clone();
    damaged[glyf_offset + 11] ^= 0x10;
    assert!(Font::new(&damaged).is_ok());
    assert_eq!(Font::verify_checksums(&damaged, 0), Err(invalid(glyf_at, "table checksum mismatch")));
    assert!(Font::new_verified(&damaged).is_err());
    // A wrong checkSumAdjustment alone leaves every table checksum right.
    let head_at = data.windows(4).position(|w| w == b"head").unwrap();
    let head_offset = u32::from_be_bytes(data[head_at + 8..head_at + 12].try_into().unwrap()) as usize;
    let mut damaged = data.clone();
    damaged[head_offset + 8] ^= 1;
    assert_eq!(Font::verify_checksums(&damaged, 0), Err(invalid(head_offset + 8, "file checksum mismatch")));
}

#[test]
fn containers_flavours_and_missing_tables() {
    let builder = triangle_font();
    let data = builder.build();
    // Outline flavours this parser does not read are named.
    for (version, what) in
        [(TAG_OTTO, "CFF outlines (OTTO)"), (TAG_WOFF, "the WOFF container"), (TAG_TYP1, "PostScript Type 1 outlines")]
    {
        let mut other = data.clone();
        other[0..4].copy_from_slice(&version.to_be_bytes());
        assert_eq!(Font::new(&other).err(), Some(unsupported(what)));
    }
    // A TrueType-flavoured file whose outlines are CFF after all.
    let mut tables: Vec<_> = builder.tables().into_iter().filter(|t| &t.0 != b"glyf").collect();
    assert_eq!(Font::new(&sfnt(TAG_TRUETYPE, &tables)).err(), Some(invalid(0, "missing glyf table")));
    tables.push((*b"CFF ", vec![1, 0, 4, 1]));
    assert_eq!(Font::new(&sfnt(TAG_TRUETYPE, &tables)).err(), Some(unsupported("CFF outlines")));
    // Apple's `true` version tag is TrueType.
    assert!(Font::new(&sfnt(TAG_TRUE, &builder.tables())).is_ok());
    for tag in [b"head", b"maxp", b"hhea", b"hmtx", b"loca", b"cmap"] {
        let tables: Vec<_> = builder.tables().into_iter().filter(|t| &t.0 != tag).collect();
        assert!(matches!(Font::new(&sfnt(TAG_TRUETYPE, &tables)), Err(Error::Invalid { .. })), "{tag:?}");
    }
    assert!(matches!(Font::new(b"\0\0\0\0 not a font at all"), Err(Error::Invalid { offset: 0, .. })));
    assert_eq!(Font::new(&[]).err(), Some(Error::Truncated { format: Format::Ttf, offset: 0 }));
    assert!(matches!(Font::from_collection(&data, 1), Err(Error::Invalid { .. })));

    // A table record that points past the end of the file.
    let mut wild = data.clone();
    let record = 12 + 16 * 3;
    wild[record + 12..record + 16].copy_from_slice(&0x00FF_FFFF_u32.to_be_bytes());
    assert_eq!(Font::new(&wild).err(), Some(invalid(record, "table lies outside the file")));
    wild[record + 8..record + 16].copy_from_slice(&[0xFF; 8]);
    assert_eq!(Font::new(&wild).err(), Some(invalid(record, "table lies outside the file")));
    // Tables that are too short for what their headers promise.
    let mut b = triangle_font();
    b.num_h_metrics = Some(40);
    assert!(matches!(Font::new(&b.build()), Err(Error::Truncated { .. })));
    let mut bad = data.clone();
    let maxp_at = bad.windows(4).position(|w| w == b"maxp").unwrap();
    let maxp_offset = u32::from_be_bytes(bad[maxp_at + 8..maxp_at + 12].try_into().unwrap()) as usize;
    bad[maxp_offset + 4..maxp_offset + 6].copy_from_slice(&500_u16.to_be_bytes());
    assert!(matches!(Font::new(&bad), Err(Error::Truncated { .. })), "loca is too short for 500 glyphs");

    // A collection of two faces that share nothing but the file: the second has one more glyph.
    let mut second = triangle_font();
    second.glyph(simple(&[TRIANGLE]), 700, 0);
    let faces = [builder.tables(), second.tables()];
    let mut ttc = TAG_TTCF.to_be_bytes().to_vec();
    ttc.extend(0x0001_0000_u32.to_be_bytes());
    ttc.extend(2_u32.to_be_bytes());
    let mut face_offset = 12 + 8;
    let mut bodies = Vec::new();
    for tables in &faces {
        // Rebase the table offsets of a stand-alone file to its place in the collection.
        let mut face = sfnt(TAG_TRUETYPE, tables);
        for i in 0..tables.len() {
            let at = 12 + 16 * i + 8;
            let offset = u32::from_be_bytes(face[at..at + 4].try_into().unwrap()) + face_offset as u32;
            face[at..at + 4].copy_from_slice(&offset.to_be_bytes());
        }
        ttc.extend((face_offset as u32).to_be_bytes());
        face_offset += face.len();
        bodies.push(face);
    }
    ttc.extend(bodies.concat());
    assert_eq!(Font::face_count(&ttc).unwrap(), 2);
    assert_eq!(Font::new(&ttc).unwrap().glyph_count(), 2);
    assert_eq!(Font::from_collection(&ttc, 1).unwrap().glyph_count(), 3);
    assert_eq!(Font::from_collection(&ttc, 2).err(), Some(invalid(8, "face index out of range")));
    assert!(Font::verify_checksums(&ttc, 1).is_ok(), "table checksums hold; a collection has no file checksum");

    // Every truncation of the file is an error or a font that answers every query.
    for cut in 0..data.len() {
        exercise(&data[..cut]);
    }
}

#[test]
fn cmap_formats_0_4_6_and_12() {
    let font_with = |cmap: Vec<u8>| {
        let mut b = Builder::new();
        for _ in 0..3 {
            b.glyph(simple(&[TRIANGLE]), 500, 0);
        }
        b.cmap = Some(cmap);
        b.build()
    };
    let mapped =
        |data: &[u8]| Font::new(data).unwrap().mappings().into_iter().map(|(c, g)| (c, g.0)).collect::<Vec<_>>();

    // Format 4: a delta segment, a glyph-array segment with a hole, and a segment whose glyphs do
    // not exist in the font.
    let delta = 1_u16.wrapping_sub(0x41);
    let data = font_with(cmap_table(&[(
        3,
        1,
        cmap4(&[(0x41, 0x43, delta, &[]), (0x5A, 0x5A, 200, &[]), (0x3B1, 0x3B3, 0, &[3, 0, 2])]),
    )]));
    let font = Font::new(&data).unwrap();
    assert_eq!(font.glyph_index('A'), Some(GlyphId(1)));
    assert_eq!(font.glyph_index('C'), Some(GlyphId(3)));
    assert_eq!(font.glyph_index('D'), None);
    assert_eq!(font.glyph_index('@'), None);
    assert_eq!(font.glyph_index('Z'), None, "glyph 290 is not in the font");
    assert_eq!(font.glyph_index('α'), Some(GlyphId(3)));
    assert_eq!(font.glyph_index('β'), None, "glyph 0 in the array means unmapped");
    assert_eq!(font.glyph_index('γ'), Some(GlyphId(2)));
    assert_eq!(font.glyph_index('😀'), None);
    assert_eq!(mapped(&data), vec![('A', 1), ('B', 2), ('C', 3), ('α', 3), ('γ', 2)]);

    // Format 12 reaches past the basic plane and is preferred over a format 4 subtable.
    let data = font_with(cmap_table(&[
        (3, 1, cmap4(&[(0x41, 0x41, 3_u16.wrapping_sub(0x41), &[])])),
        (3, 10, cmap12(&[(0x41, 0x42, 1), (0x1F600, 0x1F601, 2), (0x1F700, 0x1F700, 9)])),
    ]));
    let font = Font::new(&data).unwrap();
    assert_eq!(font.glyph_index('A'), Some(GlyphId(1)));
    assert_eq!(font.glyph_index('😀'), Some(GlyphId(2)));
    assert_eq!(font.glyph_index('😁'), Some(GlyphId(3)));
    assert_eq!(font.glyph_index('\u{1F700}'), None);
    assert_eq!(mapped(&data), vec![('A', 1), ('B', 2), ('😀', 2), ('😁', 3)]);
    // Groups that overlap or run backwards are visited once: the walk is bounded by Unicode.
    let data = font_with(cmap_table(&[(0, 4, cmap12(&[(0, 0x10_FFFF, 1), (0, 0xFFFF_FFFF, 1), (5, 2, 1)]))]));
    assert_eq!(mapped(&data), vec![('\0', 1), ('\u{1}', 2), ('\u{2}', 3)]);

    // Format 6 under the Unicode platform.
    let data = font_with(cmap_table(&[(0, 3, cmap6(0x30, &[1, 2, 0, 3]))]));
    let font = Font::new(&data).unwrap();
    assert_eq!(font.glyph_index('0'), Some(GlyphId(1)));
    assert_eq!(font.glyph_index('2'), None);
    assert_eq!(font.glyph_index('3'), Some(GlyphId(3)));
    assert_eq!(font.glyph_index('4'), None);
    assert_eq!(font.glyph_index('/'), None);
    assert_eq!(mapped(&data), vec![('0', 1), ('1', 2), ('3', 3)]);

    // Format 0 in Mac Roman: byte 0x8E is é.
    let data = font_with(cmap_table(&[(1, 0, cmap0(&[(0x41, 1), (0x8E, 2), (0xFF, 3)]))]));
    let font = Font::new(&data).unwrap();
    assert_eq!(font.glyph_index('A'), Some(GlyphId(1)));
    assert_eq!(font.glyph_index('é'), Some(GlyphId(2)));
    assert_eq!(font.glyph_index('ˇ'), Some(GlyphId(3)));
    assert_eq!(font.glyph_index('€'), None);
    assert_eq!(font.glyph_index('😀'), None);
    assert_eq!(mapped(&data), vec![('A', 1), ('é', 2), ('ˇ', 3)]);
    // The whole upper half of the table round-trips (the script checks it against Python's codec).
    for byte in 0x80..=0xFF_u8 {
        assert_eq!(char_to_mac_roman(mac_roman_to_char(byte)), Some(byte));
    }
    let sum: u32 = (0x80..=0xFF_u8).map(|b| u32::from(b) * u32::from(mac_roman_to_char(b))).sum();
    assert_eq!(sum, 99_768_012, "tools/ttf_crosscheck.py: mac_roman 0x80..0xFF: sum of byte*codepoint");

    // A Windows symbol font keeps its glyphs at U+F000 + code; both spellings find them.
    let data = font_with(cmap_table(&[(3, 0, cmap4(&[(0xF041, 0xF042, 1_u16.wrapping_sub(0xF041), &[])]))]));
    let font = Font::new(&data).unwrap();
    assert_eq!(font.glyph_index('A'), Some(GlyphId(1)));
    assert_eq!(font.glyph_index('\u{F042}'), Some(GlyphId(2)));
    assert_eq!(font.glyph_index('C'), None);

    // Subtables of other formats or platforms are passed over; with nothing left the font is refused.
    let format2 = words(&[2, 6, 0]);
    let data = font_with(cmap_table(&[(3, 1, format2.clone()), (4, 0, cmap6(0x30, &[1])), (0, 3, cmap6(0x30, &[1]))]));
    assert_eq!(Font::new(&data).unwrap().glyph_index('0'), Some(GlyphId(1)));
    let data = font_with(cmap_table(&[(3, 1, format2)]));
    assert!(matches!(Font::new(&data), Err(Error::Unsupported { .. })));
    // A subtable offset outside the table is an error; a subtable cut short is passed over.
    let mut table = cmap_table(&[(3, 1, cmap4(&[(0x41, 0x43, delta, &[])]))]);
    table[8..12].copy_from_slice(&0x0001_0000_u32.to_be_bytes());
    assert!(matches!(Font::new(&font_with(table)), Err(Error::Invalid { .. })));
    let mut table = cmap_table(&[(3, 1, cmap4(&[(0x41, 0x43, delta, &[])]))]);
    table.truncate(table.len() - 6);
    assert!(matches!(Font::new(&font_with(table)), Err(Error::Unsupported { .. })));
}

#[test]
fn horizontal_metrics_follow_the_monospace_rule() {
    let mut b = Builder::new();
    b.glyph(simple(&[TRIANGLE]), 600, 7);
    b.glyph(simple(&[&[(30, 0, true), (130, 0, true), (30, 200, true)]]), 600, 30);
    b.glyph(vec![], 600, -4);
    // Two full records; glyphs 2 and 3 store only a bearing and share the last advance.
    b.metrics[0] = (450, 1);
    b.num_h_metrics = Some(2);
    let data = b.build();
    let font = Font::new_verified(&data).unwrap();
    let all: Vec<(u16, i16)> = (0..5).map(|g| (font.advance(GlyphId(g)), font.left_side_bearing(GlyphId(g)))).collect();
    assert_eq!(all, vec![(450, 1), (600, 7), (600, 30), (600, -4), (0, 0)], "glyph 4 does not exist");
    assert_eq!(font.h_metrics(GlyphId(2)), HMetrics { advance: 600, left_side_bearing: 30 });
    assert_eq!(font.glyph_bbox(GlyphId(2)), Some(BBox { x_min: 30, y_min: 0, x_max: 130, y_max: 200 }));
    assert_eq!(font.right_side_bearing(GlyphId(2)), 600 - 30 - 100);
    assert_eq!(font.glyph_bbox(GlyphId(3)), None, "no outline");
    assert_eq!(font.right_side_bearing(GlyphId(3)), 604);
    assert_eq!(font.glyph_bbox(GlyphId(9)), None);
    assert!(font.outline_to_vec(GlyphId(3)).unwrap().is_empty());
    assert!(matches!(font.outline_to_vec(GlyphId(4)), Err(Error::Invalid { what: "glyph id out of range", .. })));
    // The long loca format reads the same font.
    b.long_loca = true;
    let long = b.build();
    let font = Font::new_verified(&long).unwrap();
    assert_eq!(font.outline_size(GlyphId(2)).unwrap(), OutlineSize { points: 3, contours: 1, components: 0 });
    assert_eq!(font.glyph_bbox(GlyphId(3)), None);
}

#[test]
fn simple_outlines_resolve_implied_points() {
    let mut b = Builder::new();
    let glyph = b.glyph(
        simple(&[
            // Starts on-curve and ends off-curve: the last curve closes the contour.
            &[(0, 0, true), (50, 100, false), (100, 0, true), (50, -100, false)],
            // Starts off-curve, ends on-curve: starts at its last point.
            &[(10, 10, false), (20, 0, true), (0, 0, true)],
            // No on-curve point at all: starts midway between the last and the first.
            &[(0, 0, false), (10, 0, false), (10, 10, false), (0, 10, false)],
            // A lone point (an anchor) draws nothing.
            &[(7, 7, true)],
        ]),
        500,
        0,
    );
    let data = b.build();
    let font = Font::new_verified(&data).unwrap();
    let path = font.outline_to_vec(GlyphId(glyph)).unwrap();
    assert_eq!(
        path.commands,
        vec![
            MoveTo([0.0, 0.0]),
            QuadTo([50.0, 100.0], [100.0, 0.0]),
            QuadTo([50.0, -100.0], [0.0, 0.0]),
            Close,
            MoveTo([0.0, 0.0]),
            QuadTo([10.0, 10.0], [20.0, 0.0]),
            Close,
            MoveTo([0.0, 5.0]),
            QuadTo([0.0, 0.0], [5.0, 0.0]),
            QuadTo([10.0, 0.0], [10.0, 5.0]),
            QuadTo([10.0, 10.0], [5.0, 10.0]),
            QuadTo([0.0, 10.0], [0.0, 5.0]),
            Close,
        ]
    );
    assert_eq!(font.outline_size(GlyphId(glyph)).unwrap(), OutlineSize { points: 12, contours: 4, components: 0 });
    assert_eq!(path.contours().count(), 3);
    assert_eq!(path.control_bounds(), Some([0.0, -100.0, 100.0, 100.0]));
    assert_eq!(path.bounds(), Some([0.0, -50.0, 100.0, 50.0]), "the lens is half as tall as its control points");

    // The compact encodings: byte deltas with their sign in the flags, "same as before", a
    // repeated flag, and instructions to skip. (0,0) (10,0) (20,0) (20,50) (0,50).
    let mut compact = words(&[1, 0, 0, 20, 50, 4, 2]);
    compact.extend([0xB0, 0x00]);
    compact.extend([0x31, 0x3B, 0x01, 0x35, 0x23]);
    compact.extend([10, 10, 20]);
    compact.extend([50]);
    let mut b = Builder::new();
    b.glyph(compact.clone(), 100, 0);
    let data = b.build();
    let font = Font::new_verified(&data).unwrap();
    let path = font.outline_to_vec(GlyphId(1)).unwrap();
    assert_eq!(corners(&path), vec![[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [20.0, 50.0], [0.0, 50.0]]);
    assert_eq!(path.commands.last(), Some(&Close));
    assert_eq!(orientation(path.contour_areas()[0]), Orientation::CounterClockwise);
    // Cut anywhere, the glyph is an error (never a panic), and the error points into the file.
    for cut in 0..compact.len() {
        let mut b = Builder::new();
        b.glyph(compact[..cut].to_vec(), 100, 0);
        let data = b.build();
        let font = Font::new(&data).unwrap();
        let mut sink = Path::default();
        match font.outline(GlyphId(1), &mut sink) {
            Ok(()) => assert_eq!(cut, 0, "an empty glyph has no outline"),
            Err(e) => assert!(e.offset().is_some_and(|at| at <= data.len()), "{e}"),
        }
        assert!(sink.is_empty(), "nothing is sent on an error");
    }
    // Contour end points must increase.
    let mut bad = simple(&[TRIANGLE, TRIANGLE]);
    bad[12..14].copy_from_slice(&1_u16.to_be_bytes());
    let mut b = Builder::new();
    b.glyph(bad, 100, 0);
    let data = b.build();
    let error = Font::new(&data).unwrap().outline_to_vec(GlyphId(1)).unwrap_err();
    assert!(matches!(error, Error::Invalid { what: "contour end points do not increase", .. }), "{error}");
}

#[test]
fn composite_transforms_offsets_and_point_matching() {
    let mut b = triangle_font();
    let shifted = b.glyph(composite(&[at(1, 10, 20)]), 0, 0);
    let scaled = b.glyph(
        composite(&[Comp { glyph: 1, flags: 0, args: Args::Offset(300, -200), xform: Xform::Scale(0.5) }]),
        0,
        0,
    );
    let stretched =
        b.glyph(composite(&[Comp { glyph: 1, flags: 0, args: Args::Offset(0, 0), xform: Xform::XY(1.5, 0.5) }]), 0, 0);
    // A quarter turn counter-clockwise: x' = −y, y' = x.
    let turned = b.glyph(
        composite(&[Comp {
            glyph: 1,
            flags: 0,
            args: Args::Offset(0, 0),
            xform: Xform::Matrix([0.0, 1.0, -1.0, 0.0]),
        }]),
        0,
        0,
    );
    let scaled_offset = b.glyph(
        composite(&[Comp { glyph: 1, flags: F_SCALED_OFFSET, args: Args::Offset(100, 40), xform: Xform::Scale(0.5) }]),
        0,
        0,
    );
    // The second triangle's point 2 is put on the first one's point 1.
    let matched = b.glyph(
        composite(&[at(1, 0, 0), Comp { glyph: 1, flags: 0, args: Args::Points(1, 2), xform: Xform::None }]),
        0,
        0,
    );
    let nested = b.glyph(composite(&[at(shifted, 1000, 0), at(1, -5, -5)]), 0, 0);
    let borrowed = b.glyph(
        composite(&[
            at(1, 0, 0),
            Comp { glyph: 1, flags: F_USE_MY_METRICS, args: Args::Offset(60, 0), xform: Xform::None },
        ]),
        999,
        9,
    );
    let twice_borrowed = b.glyph(
        composite(&[Comp { glyph: borrowed, flags: F_USE_MY_METRICS, args: Args::Offset(0, 0), xform: Xform::None }]),
        123,
        4,
    );
    let data = b.build();
    let font = Font::new_verified(&data).unwrap();
    let points = |glyph: u16| corners(&font.outline_to_vec(GlyphId(glyph)).unwrap());

    assert_eq!(points(1), vec![[0.0, 0.0], [100.0, 0.0], [0.0, 200.0]]);
    assert_eq!(points(shifted), vec![[10.0, 20.0], [110.0, 20.0], [10.0, 220.0]]);
    assert_eq!(points(scaled), vec![[300.0, -200.0], [350.0, -200.0], [300.0, -100.0]]);
    assert_eq!(points(stretched), vec![[0.0, 0.0], [150.0, 0.0], [0.0, 100.0]]);
    assert_eq!(points(turned), vec![[0.0, 0.0], [0.0, 100.0], [-200.0, 0.0]]);
    assert_eq!(points(scaled_offset), vec![[50.0, 20.0], [100.0, 20.0], [50.0, 120.0]]);
    assert_eq!(
        points(matched),
        vec![[0.0, 0.0], [100.0, 0.0], [0.0, 200.0], [100.0, -200.0], [200.0, -200.0], [100.0, 0.0]]
    );
    assert_eq!(
        points(nested),
        vec![[1010.0, 20.0], [1110.0, 20.0], [1010.0, 220.0], [-5.0, -5.0], [95.0, -5.0], [-5.0, 195.0]]
    );

    assert!(!font.is_composite(GlyphId(1)) && font.is_composite(GlyphId(shifted)) && !font.is_composite(GlyphId(0)));
    assert_eq!(font.outline_size(GlyphId(matched)).unwrap(), OutlineSize { points: 6, contours: 2, components: 2 });
    assert_eq!(font.outline_size(GlyphId(nested)).unwrap(), OutlineSize { points: 6, contours: 2, components: 3 });
    // A rotation keeps the orientation, so the turned triangle still has the area of the original.
    let area = |glyph: u16| font.outline_to_vec(GlyphId(glyph)).unwrap().contour_areas();
    assert_eq!(area(1), vec![10_000.0]);
    assert_eq!(area(turned), vec![10_000.0]);
    assert_eq!(area(scaled), vec![2_500.0]);

    // USE_MY_METRICS: the rasteriser's metrics are the component's, through nested composites;
    // the hmtx entries themselves stay readable.
    assert_eq!(font.metrics_source(GlyphId(borrowed)), GlyphId(1));
    assert_eq!(font.metrics_source(GlyphId(twice_borrowed)), GlyphId(1));
    assert_eq!(font.metrics_source(GlyphId(matched)), GlyphId(matched));
    assert_eq!(font.metrics_source(GlyphId(1)), GlyphId(1));
    assert_eq!(font.h_metrics(GlyphId(twice_borrowed)), HMetrics { advance: 500, left_side_bearing: 5 });
    assert_eq!((font.advance(GlyphId(borrowed)), font.left_side_bearing(GlyphId(borrowed))), (999, 9));
    assert_eq!(font.h_metrics(GlyphId(matched)), HMetrics { advance: 0, left_side_bearing: 0 });
}

#[test]
fn composite_recursion_and_sizes_are_limited() {
    let limit = |result: Result<Path>| match result {
        Err(Error::Limit { what, limit, got, .. }) => (what, limit, got),
        other => panic!("expected a limit error, got {other:?}"),
    };
    // A glyph that contains itself, directly and through another glyph.
    let mut b = triangle_font();
    b.glyph(composite(&[at(1, 0, 0), at(2, 5, 5)]), 0, 0);
    b.glyph(composite(&[at(4, 0, 0)]), 0, 0);
    b.glyph(composite(&[at(1, 0, 0), at(3, 0, 0)]), 0, 0);
    // A component that does not exist, and point numbers that do not exist.
    b.glyph(composite(&[at(77, 0, 0)]), 0, 0);
    b.glyph(composite(&[at(1, 0, 0), Comp { glyph: 1, flags: 0, args: Args::Points(3, 0), xform: Xform::None }]), 0, 0);
    b.glyph(composite(&[Comp { glyph: 1, flags: 0, args: Args::Points(0, 0), xform: Xform::None }]), 0, 0);
    let data = b.build();
    let font = Font::new(&data).unwrap();
    for glyph in [2, 3, 4] {
        let error = font.outline_to_vec(GlyphId(glyph)).unwrap_err();
        assert!(matches!(error, Error::Invalid { what: "composite glyph contains itself", .. }), "{glyph}: {error}");
        assert_eq!(font.metrics_source(GlyphId(glyph)), GlyphId(glyph));
    }
    let error = font.outline_to_vec(GlyphId(5)).unwrap_err();
    assert!(matches!(error, Error::Invalid { what: "glyph id out of range", .. }), "{error}");
    for glyph in [6, 7] {
        let error = font.outline_to_vec(GlyphId(glyph)).unwrap_err();
        assert!(matches!(error, Error::Invalid { what: "composite point index out of range", .. }), "{error}");
    }
    let mut sink = Path::default();
    assert!(font.outline(GlyphId(2), &mut sink).is_err() && sink.is_empty(), "nothing is sent on an error");

    // A chain of composites: glyph k + 1 contains glyph k. The font's own maxComponentDepth holds...
    let chain = |depth_limit: u16| {
        let mut b = triangle_font();
        b.max[5] = depth_limit;
        for k in 1..=12 {
            b.glyph(composite(&[at(k, 1, 1)]), 0, 0);
        }
        b.build()
    };
    let data = chain(3);
    let font = Font::new(&data).unwrap();
    assert_eq!(corners(&font.outline_to_vec(GlyphId(4)).unwrap())[0], [3.0, 3.0]);
    assert_eq!(limit(font.outline_to_vec(GlyphId(5))), ("composite glyph nesting (maxp maxComponentDepth)", 3, 4));
    // ...and whatever it claims, eight levels are the most that are followed.
    let data = chain(u16::MAX);
    let font = Font::new(&data).unwrap();
    assert_eq!(font.outline_size(GlyphId(9)).unwrap().components, 8);
    assert_eq!(limit(font.outline_to_vec(GlyphId(10))), ("composite glyph nesting (maxp maxComponentDepth)", 8, 9));
    assert_eq!(limit(font.outline_to_vec(GlyphId(13))).0, "composite glyph nesting (maxp maxComponentDepth)");
    // A font whose maxp says "no composites" (depth 0) has none that load.
    let data = chain(0);
    assert_eq!(limit(Font::new(&data).unwrap().outline_to_vec(GlyphId(2))).1, 0);

    // Components per glyph (maxComponentElements) and in total (1024): 40 × 40 triangles.
    let fan = |elements: u16| {
        let mut b = triangle_font();
        b.max = [1000, 100, u16::MAX, u16::MAX, elements, 4];
        let row: Vec<Comp> = (0..40).map(|i| at(1, i, 0)).collect();
        b.glyph(composite(&row), 0, 0);
        let rows: Vec<Comp> = (0..40).map(|i| at(2, 0, i)).collect();
        b.glyph(composite(&rows), 0, 0);
        b.build()
    };
    let data = fan(39);
    let font = Font::new(&data).unwrap();
    assert_eq!(
        limit(font.outline_to_vec(GlyphId(2))),
        ("composite glyph component count (maxp maxComponentElements)", 39, 40)
    );
    let data = fan(40);
    let font = Font::new(&data).unwrap();
    assert_eq!(font.outline_size(GlyphId(2)).unwrap(), OutlineSize { points: 120, contours: 40, components: 40 });
    assert_eq!(limit(font.outline_to_vec(GlyphId(3))), ("composite glyph total component count", 1024, 1025));

    // Points and contours: of one simple glyph, and of a composite as a whole.
    let sized = |max: [u16; 4]| {
        let mut b = triangle_font();
        b.max = [max[0], max[1], max[2], max[3], 16, 4];
        b.glyph(simple(&[TRIANGLE, TRIANGLE]), 0, 0);
        b.glyph(composite(&[at(1, 0, 0), at(1, 1, 0), at(1, 2, 0)]), 0, 0);
        b.build()
    };
    let data = sized([5, 2, 9, 3]);
    let font = Font::new(&data).unwrap();
    assert_eq!(limit(font.outline_to_vec(GlyphId(2))), ("glyph point count (maxp maxPoints)", 5, 6));
    assert!(font.outline_to_vec(GlyphId(1)).is_ok() && font.outline_to_vec(GlyphId(3)).is_ok());
    let data = sized([6, 1, 9, 3]);
    let font = Font::new(&data).unwrap();
    assert_eq!(limit(font.outline_to_vec(GlyphId(2))), ("glyph contour count (maxp maxContours)", 1, 2));
    let data = sized([6, 2, 8, 3]);
    let font = Font::new(&data).unwrap();
    assert_eq!(limit(font.outline_to_vec(GlyphId(3))), ("composite glyph point count (maxp maxCompositePoints)", 8, 9));
    assert!(font.outline_to_vec(GlyphId(2)).is_ok(), "a simple glyph is not held to the composite limits");
    let data = sized([6, 2, 9, 2]);
    let font = Font::new(&data).unwrap();
    assert_eq!(
        limit(font.outline_to_vec(GlyphId(3))),
        ("composite glyph contour count (maxp maxCompositeContours)", 2, 3)
    );
}

#[test]
fn kerning_from_gpos_and_from_the_kern_table() {
    let glyphs = |count: u16| {
        let mut b = Builder::new();
        for _ in 0..count {
            b.glyph(simple(&[TRIANGLE]), 500, 0);
        }
        b
    };
    let pairs: &[(u16, u16, i16)] = &[(5, 6, -11), (5, 9, 3), (7, 6, -2)];
    let kern = kern_table(&[(0x0001, pairs)]);
    let ids: Vec<GlyphId> = (0..10).map(GlyphId).collect();

    // A kern table alone.
    let data = glyphs(9).with(b"kern", kern.clone()).build();
    let font = Font::new_verified(&data).unwrap();
    assert!(font.has_kerning());
    assert_eq!(font.gpos_pair_lookup_counts(), (0, 0));
    assert_eq!(font.kerning(GlyphId(5), GlyphId(6)), -11);
    assert_eq!(font.kerning(GlyphId(6), GlyphId(5)), 0);
    let expected: Vec<KerningPair> =
        pairs.iter().map(|&(l, r, value)| KerningPair { left: GlyphId(l), right: GlyphId(r), value }).collect();
    assert_eq!(font.kerning_pairs(&ids), expected);

    // GPOS pair adjustments take precedence over the kern table (see `sample_gpos` for the values).
    for script in [b"DFLT", b"latn"] {
        let data = glyphs(9).with(b"kern", kern.clone()).with(b"GPOS", sample_gpos(script)).build();
        let font = Font::new_verified(&data).unwrap();
        assert_eq!(font.gpos_pair_lookup_counts(), (2, 3));
        assert_eq!(font.kerning(GlyphId(5), GlyphId(6)), -50);
        assert_eq!(font.kerning(GlyphId(5), GlyphId(9)), 20);
        assert_eq!(font.kerning(GlyphId(7), GlyphId(6)), -37);
        assert_eq!(font.kerning(GlyphId(7), GlyphId(9)), 0);
        let found: Vec<(u16, u16, i16)> =
            font.kerning_pairs(&ids).iter().map(|p| (p.left.0, p.right.0, p.value)).collect();
        assert_eq!(found, vec![(5, 6, -50), (5, 9, 20), (7, 6, -37)]);
        // The order of the caller's glyph list is kept, and only its glyphs are paired.
        let subset = [GlyphId(9), GlyphId(6), GlyphId(5)];
        let found: Vec<(u16, u16, i16)> =
            font.kerning_pairs(&subset).iter().map(|p| (p.left.0, p.right.0, p.value)).collect();
        assert_eq!(found, vec![(5, 9, 20), (5, 6, -50)]);
    }
    // A GPOS table without pair adjustments for `kern` leaves the kern table in force.
    let mut no_kern = sample_gpos(b"latn");
    let tag = no_kern.windows(4).position(|w| w == b"kern").unwrap();
    no_kern[tag..tag + 4].copy_from_slice(b"mkmk");
    let data = glyphs(9).with(b"kern", kern).with(b"GPOS", no_kern.clone()).build();
    let font = Font::new_verified(&data).unwrap();
    assert_eq!(font.kerning(GlyphId(5), GlyphId(6)), -11);
    // And with neither there is none.
    let data = glyphs(9).with(b"GPOS", no_kern).build();
    let font = Font::new_verified(&data).unwrap();
    assert!(!font.has_kerning());
    assert_eq!(font.kerning(GlyphId(5), GlyphId(6)), 0);
    assert!(font.kerning_pairs(&ids).is_empty());
    // A GPOS table whose offsets leave the table makes the font unreadable rather than half-read.
    let mut wild = sample_gpos(b"latn");
    wild[6..8].copy_from_slice(&[0xFF, 0xF0]);
    assert!(matches!(Font::new(&glyphs(9).with(b"GPOS", wild).build()), Err(Error::Invalid { .. })));
}

/// `[ROB-02]` Every count in the script and feature lists is 16 bits wide, but nested they multiply:
/// the walk as a whole is bounded.
#[test]
fn gpos_layout_walk_has_a_budget() {
    let gpos = |scripts: u16, features: u16| {
        // Script list: `scripts` records that all point at one Script table, whose default language
        // system lists `features` feature indices.
        let mut script_list = words(&[scripts]);
        let script_at = 2 + 6 * scripts;
        for _ in 0..scripts {
            script_list.extend(b"DFLT");
            script_list.extend(script_at.to_be_bytes());
        }
        script_list.extend(words(&[4, 0]));
        script_list.extend(words(&[0, 0xFFFF, features]));
        script_list.extend(vec![0_u8; usize::from(features) * 2]);
        let feature_at = 10 + script_list.len() as u16;
        let mut table = words(&[1, 0, 10, feature_at, feature_at + 2]);
        table.extend(script_list);
        table.extend(words(&[0, 0]));
        table
    };
    let font_with = |table: Vec<u8>| triangle_font().with(b"GPOS", table).build();
    assert!(Font::new(&font_with(gpos(10, 10))).is_ok());
    assert!(Font::new(&font_with(gpos(1000, 1000))).is_ok(), "a million entries are still walked");
    let data = font_with(gpos(1100, 1100));
    let Err(Error::Limit { what, limit, .. }) = Font::new(&data) else { panic!("expected a limit error") };
    assert_eq!((what, limit), ("GPOS layout entries examined", u64::from(kerning::MAX_LAYOUT_STEPS)));
}

// ---------------------------------------------------------------------------------------------
// The fonts the UI ships, against the independent parser's fixtures.
// ---------------------------------------------------------------------------------------------

/// A cursor over the space-separated fields of a fixture line.
struct Fields<'a>(std::iter::Peekable<std::str::Split<'a, char>>);

impl<'a> Fields<'a> {
    fn new(line: &'a str) -> Self {
        Self(line.split(' ').peekable())
    }

    fn text(&mut self) -> &'a str {
        self.0.next().expect("fixture line is too short")
    }

    fn int(&mut self) -> i64 {
        self.text().parse().expect("fixture integer")
    }

    fn float(&mut self) -> f64 {
        self.text().parse().expect("fixture number")
    }

    /// An integer, or `None` for `-`.
    fn optional(&mut self) -> Option<i64> {
        match self.text() {
            "-" => None,
            text => Some(text.parse().expect("fixture integer")),
        }
    }

    /// Four numbers, or `None` for `-`.
    fn bounds(&mut self) -> Option<[f64; 4]> {
        if self.0.peek() == Some(&"-") {
            self.0.next();
            return None;
        }
        Some([self.float(), self.float(), self.float(), self.float()])
    }
}

/// Whether two boxes agree to a hundredth of a font unit (composites are transformed in `f32` here
/// and in double precision in the script; everything else is integral and so exactly equal).
fn boxes_agree(ours: Option<[f32; 4]>, theirs: Option<[f64; 4]>) -> bool {
    match (ours, theirs) {
        (None, None) => true,
        (Some(a), Some(b)) => a.iter().zip(b).all(|(&a, b)| (f64::from(a) - b).abs() <= 0.01),
        _ => false,
    }
}

/// Compares everything the parser reads from `data` with the fixture `tools/ttf_crosscheck.py`
/// wrote for the same file; returns (mappings, glyphs, kerning pairs) compared.
fn check_against_fixture(data: &[u8], fixture: &str) -> (usize, usize, usize) {
    let font = Font::new_verified(data).unwrap();
    let m = *font.metrics();
    let mappings = font.mappings();
    let mut next_mapping = mappings.iter();
    let (mut glyph_lines, mut kern_lines) = (0_usize, Vec::new());
    let mut kerning: Option<Vec<KerningPair>> = None;
    let mut hhea = [0_i64; 3];
    let mut seen = Vec::new();
    for line in fixture.lines() {
        let mut f = Fields::new(line);
        let kind = f.text();
        if !matches!(kind, "c" | "g" | "k" | "#") {
            seen.push(kind);
        }
        match kind {
            "#" => {}
            "font" => {
                f.text();
                assert_eq!(f.int(), data.len() as i64, "the fixture was written for another file");
            }
            "head" => {
                assert_eq!(i64::from(m.units_per_em), f.int());
                let b = m.bbox;
                assert_eq!([b.x_min, b.y_min, b.x_max, b.y_max].map(i64::from), [f.int(), f.int(), f.int(), f.int()]);
                assert_eq!(i64::from(font.long_loca), f.int());
            }
            "maxp" => {
                let p = font.max_profile();
                let ours = [
                    m.num_glyphs,
                    p.max_points,
                    p.max_contours,
                    p.max_composite_points,
                    p.max_composite_contours,
                    p.max_component_elements,
                    p.max_component_depth,
                ];
                assert_eq!(ours.map(i64::from).to_vec(), (0..7).map(|_| f.int()).collect::<Vec<_>>());
            }
            "hhea" => {
                hhea = [f.int(), f.int(), f.int()];
                assert_eq!([m.hhea_ascender, m.hhea_descender, m.hhea_line_gap].map(i64::from), hhea);
                assert_eq!(i64::from(font.num_h_metrics), f.int());
            }
            "os2" => {
                let version = f.int();
                assert_eq!(i64::from(m.weight_class), f.int());
                let typo = [f.int(), f.int(), f.int()];
                let expected = if typo[0] != 0 || typo[1] != 0 { typo } else { hhea };
                assert_eq!([m.ascender, m.descender, m.line_gap].map(i64::from), expected);
                assert_eq!(m.line_height(), (expected[0] - expected[1] + expected[2]) as i32);
                assert_eq!([m.x_height, m.cap_height].map(|v| v.map(i64::from)), [f.optional(), f.optional()]);
                assert_eq!(m.x_height.is_some(), version >= 2);
            }
            "post" => {
                assert_eq!([m.underline_position, m.underline_thickness].map(i64::from), [f.int(), f.int()]);
                assert_eq!(i64::from(m.is_fixed_pitch), f.int());
                assert_eq!(f64::from(m.italic_angle), f.float() / 65536.0);
            }
            "variable" => assert_eq!(i64::from(font.is_variable()), f.int()),
            "checksums" => {
                f.int();
                assert_eq!(f.int(), 0, "the script found wrong table checksums where verify_checksums found none");
                assert_eq!(f.text(), "b1b0afba");
            }
            "name" => {
                let id = f.int() as u16;
                let text = line.splitn(3, ' ').nth(2).unwrap();
                assert_eq!(font.name(id).as_deref(), Some(text));
            }
            "cmap" => assert_eq!(mappings.len() as i64, f.int()),
            "c" => {
                let code = u32::from_str_radix(f.text(), 16).unwrap();
                let glyph = GlyphId(f.int() as u16);
                let c = char::from_u32(code).unwrap();
                assert_eq!(next_mapping.next(), Some(&(c, glyph)), "U+{code:04X}");
                assert_eq!(font.glyph_index(c), Some(glyph), "U+{code:04X}");
            }
            "glyphs" => {
                assert!(next_mapping.next().is_none(), "the font maps more characters than the fixture");
                assert_eq!(i64::from(font.glyph_count()), f.int());
            }
            "g" => {
                glyph_lines += 1;
                let id = f.int();
                let glyph = GlyphId(id as u16);
                let (advance, lsb) = (f.int(), f.int());
                assert_eq!(
                    (i64::from(font.advance(glyph)), i64::from(font.left_side_bearing(glyph))),
                    (advance, lsb),
                    "{id}"
                );
                let source = GlyphId(f.int() as u16);
                assert_eq!(font.metrics_source(glyph), source, "glyph {id}");
                let expected =
                    HMetrics { advance: font.advance(source), left_side_bearing: font.left_side_bearing(source) };
                assert_eq!(font.h_metrics(glyph), expected);
                let kind = f.text();
                let stored = f.bounds();
                let bbox = font.glyph_bbox(glyph);
                assert_eq!(bbox.map(|b| [b.x_min, b.y_min, b.x_max, b.y_max].map(f64::from)), stored, "glyph {id}");
                assert_eq!((font.is_composite(glyph), bbox.is_none()), (kind == "c", kind == "e"), "glyph {id}");
                let width = stored.map_or(0, |b| (b[2] - b[0]) as i64);
                assert_eq!(i64::from(font.right_side_bearing(glyph)), advance - lsb - width);
                let size = font.outline_size(glyph).unwrap();
                let ours = [size.points as i64, size.contours as i64, i64::from(size.components)];
                assert_eq!(ours, [f.int(), f.int(), f.int()], "points, contours, components of glyph {id}");
                let path = font.outline_to_vec(glyph).unwrap();
                let count =
                    |wanted: fn(&PathCommand) -> bool| path.commands.iter().filter(|c| wanted(c)).count() as i64;
                let lines = count(|c| matches!(c, PathCommand::LineTo(_)));
                let quads = count(|c| matches!(c, PathCommand::QuadTo(..)));
                assert_eq!([lines, quads], [f.int(), f.int()], "lines, quads of glyph {id}");
                let control = f.bounds();
                assert!(
                    boxes_agree(path.control_bounds(), control),
                    "glyph {id}: {:?} {control:?}",
                    path.control_bounds()
                );
                let exact = if f.0.peek() == Some(&"=") {
                    f.0.next();
                    control
                } else {
                    f.bounds()
                };
                assert!(boxes_agree(path.bounds(), exact), "glyph {id}: {:?} {exact:?}", path.bounds());
                let area = f.float();
                let ours: f64 = path.contour_areas().iter().map(|&a| f64::from(a)).sum();
                assert!((ours - area).abs() <= 0.5 + area.abs() * 1e-5, "area of glyph {id}: {ours} {area}");
            }
            "kern" => {
                let source = f.text();
                let mut glyphs: Vec<GlyphId> = mappings
                    .iter()
                    .filter(|(c, _)| matches!(u32::from(*c), 0x20..=0x7E | 0xA0..=0xFF))
                    .map(|&(_, glyph)| glyph)
                    .collect();
                glyphs.sort_unstable();
                glyphs.dedup();
                assert_eq!(glyphs.len() as i64, f.int());
                let pairs = font.kerning_pairs(&glyphs);
                assert_eq!(pairs.len() as i64, f.int());
                assert_eq!(font.has_kerning(), source != "none");
                assert_eq!(font.gpos_pair_lookup_counts().0 > 0, source == "gpos");
                // The direct query agrees with the enumeration for every pair, zero ones included.
                let mut listed = pairs.iter().peekable();
                for &left in &glyphs {
                    for &right in &glyphs {
                        let value = listed.next_if(|p| (p.left, p.right) == (left, right)).map_or(0, |p| p.value);
                        assert_eq!(font.kerning(left, right), value, "{left:?} {right:?}");
                    }
                }
                kerning = Some(pairs);
            }
            "k" => kern_lines.push(KerningPair {
                left: GlyphId(f.int() as u16),
                right: GlyphId(f.int() as u16),
                value: f.int() as i16,
            }),
            other => panic!("unknown fixture line {other}"),
        }
    }
    for required in ["font", "head", "maxp", "hhea", "os2", "post", "variable", "checksums", "cmap", "glyphs", "kern"] {
        assert!(seen.contains(&required), "the fixture has no {required} line");
    }
    assert_eq!(glyph_lines, usize::from(font.glyph_count()));
    assert_eq!(kerning.as_ref(), Some(&kern_lines));
    (mappings.len(), glyph_lines, kern_lines.len())
}

#[test]
fn inter_regular_agrees_with_the_independent_parser() {
    let counts = check_against_fixture(INTER_REGULAR, include_str!("../../tests/fixtures/ttf/Inter-Regular.txt"));
    assert_eq!(counts, (2852, 2937, 4141));
}

#[test]
fn inter_semibold_agrees_with_the_independent_parser() {
    let counts = check_against_fixture(INTER_SEMIBOLD, include_str!("../../tests/fixtures/ttf/Inter-SemiBold.txt"));
    assert_eq!(counts, (2852, 2937, 4663));
}

#[test]
fn inter_italic_agrees_with_the_independent_parser() {
    let counts = check_against_fixture(INTER_ITALIC, include_str!("../../tests/fixtures/ttf/Inter-Italic.txt"));
    assert_eq!(counts, (2816, 2901, 4136));
}

#[test]
fn jetbrains_mono_agrees_with_the_independent_parser() {
    let counts =
        check_against_fixture(JETBRAINS_MONO, include_str!("../../tests/fixtures/ttf/JetBrainsMono-Regular.txt"));
    assert_eq!(counts, (1363, 1743, 0));
}

#[test]
fn lucide_agrees_with_the_independent_parser() {
    let counts = check_against_fixture(LUCIDE, include_str!("../../tests/fixtures/ttf/Lucide.txt"));
    assert_eq!(counts, (2071, 2065, 0));
}

/// The facts a reader can check against a font viewer, spelled out for the main UI font.
#[test]
fn inter_regular_spot_checks() {
    let font = Font::new_verified(INTER_REGULAR).unwrap();
    assert_eq!(font.family_name().as_deref(), Some("Inter"));
    assert_eq!(font.subfamily_name().as_deref(), Some("Regular"));
    let m = font.metrics();
    assert_eq!((m.units_per_em, m.ascender, m.descender, m.line_gap), (2048, 1984, -494, 0));
    assert_eq!((m.x_height, m.cap_height, m.weight_class), (Some(1118), Some(1490), 400));
    assert_eq!((m.underline_position, m.underline_thickness, m.is_fixed_pitch), (-348, 140, false));
    let glyph = |c: char| font.glyph_index(c).unwrap();
    assert_eq!(
        (glyph('A'), glyph('V'), glyph('T'), glyph('o')),
        (GlyphId(2), GlyphId(456), GlyphId(411), GlyphId(790))
    );
    assert_eq!(font.h_metrics(glyph('A')), HMetrics { advance: 1413, left_side_bearing: 52 });
    assert_eq!(font.glyph_bbox(glyph('A')), Some(BBox { x_min: 52, y_min: 0, x_max: 1361, y_max: 1490 }));
    assert_eq!(font.right_side_bearing(glyph('A')), 52);
    assert_eq!(font.kerning(glyph('A'), glyph('V')), -140);
    assert_eq!(font.kerning(glyph('T'), glyph('o')), -160);
    assert_eq!(font.kerning(glyph('H'), glyph('H')), 0);
    assert_eq!(font.glyph_index('\u{4E2D}'), None);
    assert_eq!(font.glyph_bbox(glyph(' ')), None);
    // An "o" is an outer contour drawn clockwise and a counter drawn the other way.
    let areas = font.outline_to_vec(glyph('o')).unwrap().contour_areas();
    assert_eq!(areas.len(), 2);
    assert_eq!(orientation(areas[0]), Orientation::Clockwise);
    assert_eq!(orientation(areas[1]), Orientation::CounterClockwise);
    assert!(areas[0] + areas[1] < 0.0);
    // "i" is a composite of the dotless i and a dot, and takes the dotless i's metrics.
    let i = glyph('i');
    assert!(font.is_composite(i));
    assert_eq!(font.outline_size(i).unwrap(), OutlineSize { points: 16, contours: 2, components: 2 });
    assert_eq!(font.metrics_source(i), glyph('ı'));
    // The monospaced font says so, and every glyph it maps advances alike.
    let mono = Font::new_verified(JETBRAINS_MONO).unwrap();
    assert!(mono.metrics().is_fixed_pitch && !mono.has_kerning());
    assert!(mono.mappings().iter().all(|&(_, g)| mono.advance(g) == 600 || mono.advance(g) == 0));
    // The icon font maps its icons in the private use area.
    let icons = Font::new_verified(LUCIDE).unwrap();
    assert_eq!(icons.glyph_index('\u{E038}'), Some(GlyphId(6)));
    assert_eq!(icons.units_per_em(), 1000);
}

/// Every outline of every shipped font is well formed, and flattening it stays inside the exact
/// bounds and encloses the exact area up to what the tolerance allows.
#[test]
fn shipped_outlines_flatten_within_their_exact_bounds() {
    const TOLERANCE: f32 = 0.05;
    let mut outlines = 0;
    for data in [INTER_REGULAR, INTER_SEMIBOLD, INTER_ITALIC, JETBRAINS_MONO, LUCIDE] {
        let font = Font::new(data).unwrap();
        for id in 0..font.glyph_count() {
            let path = font.outline_to_vec(GlyphId(id)).unwrap();
            let Some(exact) = path.bounds() else { continue };
            outlines += 1;
            let control = path.control_bounds().unwrap();
            assert!(
                control[0] <= exact[0] && control[1] <= exact[1] && control[2] >= exact[2] && control[3] >= exact[3]
            );
            for contour in path.contours() {
                assert!(matches!(contour[0], PathCommand::MoveTo(_)) && contour.last() == Some(&PathCommand::Close));
                assert!(contour[1..].iter().all(|c| !matches!(c, PathCommand::MoveTo(_))));
            }
            let polygons = path.flatten(TOLERANCE);
            let (mut area, mut perimeter) = (0.0_f64, 0.0_f64);
            for polygon in &polygons {
                area += f64::from(signed_area(polygon));
                let mut previous = *polygon.last().unwrap();
                for &p in polygon {
                    let slack = 5e-3;
                    assert!(
                        p[0] >= exact[0] - slack
                            && p[1] >= exact[1] - slack
                            && p[0] <= exact[2] + slack
                            && p[1] <= exact[3] + slack,
                        "glyph {id}: {p:?} outside {exact:?}"
                    );
                    perimeter += f64::from((p[0] - previous[0]).hypot(p[1] - previous[1]));
                    previous = p;
                }
            }
            let exact_area: f64 = path.contour_areas().iter().map(|&a| f64::from(a)).sum();
            let allowed = f64::from(TOLERANCE) * perimeter + 1.0 + exact_area.abs() * 1e-5;
            assert!((area - exact_area).abs() <= allowed, "glyph {id}: {area} vs {exact_area} (allowed {allowed})");
        }
    }
    assert!(outlines > 12_000, "{outlines}");
}

// ---------------------------------------------------------------------------------------------
// Mutation fuzzing [ROB-01].
// ---------------------------------------------------------------------------------------------

/// Opens `data` and, when that succeeds, asks the font everything the baker and the layout ask.
fn exercise(data: &[u8]) {
    let _ = Font::face_count(data);
    let _ = Font::verify_checksums(data, 0);
    let _ = Font::from_collection(data, 1);
    let Ok(font) = Font::new(data) else { return };
    let _ = font.metrics().line_height();
    let _ = (font.family_name(), font.subfamily_name(), font.full_name(), font.name(name_id::VERSION));
    let mut mapped = Vec::new();
    let mut count = 0_u32;
    font.for_each_mapping(|c, glyph| {
        count += 1;
        if mapped.len() < 64 {
            mapped.push(glyph);
        }
        assert!(glyph.0 < font.glyph_count() && glyph.0 != 0, "{c:?}");
    });
    assert!(count <= 0x11_0000);
    for c in ['A', 'é', '\u{F041}', '😀', '\u{10FFFF}'] {
        let _ = font.glyph_index(c);
    }
    // Every glyph of the font and a few ids past its end.
    let mut path = Path::default();
    for id in 0..=font.glyph_count().saturating_add(2) {
        let glyph = GlyphId(id);
        let _ =
            (font.h_metrics(glyph), font.right_side_bearing(glyph), font.glyph_bbox(glyph), font.is_composite(glyph));
        path.commands.clear();
        if font.outline(glyph, &mut path).is_ok() {
            let size = font.outline_size(glyph).expect("outline_size fails exactly when outline does");
            assert!(u16::try_from(size.points).is_ok() && size.components <= 1024);
            let _ = (path.bounds(), path.control_bounds(), path.contour_areas());
            if id < 32 {
                let _ = path.flatten(f32::from(font.units_per_em()) / 64.0);
            }
        } else {
            assert!(path.is_empty(), "nothing is sent on an error");
        }
    }
    // Kerning among mapped glyphs and among raw ids (GPOS tables are indexed by glyph id alone).
    mapped.extend((0..48).map(GlyphId));
    let pairs = font.kerning_pairs(&mapped);
    assert!(pairs.iter().all(|p| p.value != 0));
    let _ = font.kerning(GlyphId(u16::MAX), GlyphId(u16::MAX));
    let _ = font.gpos_pair_lookup_counts();
}

/// The bytes of one table of a font file.
#[allow(clippy::trivially_copy_pass_by_ref, reason = "call sites pass byte-string literals")]
fn table_of(data: &[u8], tag: &[u8; 4]) -> Vec<u8> {
    let count = usize::from(u16::from_be_bytes([data[4], data[5]]));
    let record = (0..count).map(|i| &data[12 + 16 * i..28 + 16 * i]).find(|r| &r[..4] == tag).unwrap();
    let offset = u32::from_be_bytes(record[8..12].try_into().unwrap()) as usize;
    let length = u32::from_be_bytes(record[12..16].try_into().unwrap()) as usize;
    data[offset..offset + length].to_vec()
}

/// Small fonts in which nearly every byte is structure: composites of every kind, both kerning
/// tables, several cmap formats, names. A mutation here almost always lands in a header, an offset
/// or a count.
fn dense_corpus() -> Vec<Vec<u8>> {
    let mut b = triangle_font();
    b.glyph(simple(&[&[(0, 0, true), (50, 100, false), (100, 0, true), (50, -100, false)], TRIANGLE]), 400, 0);
    b.glyph(
        composite(&[at(1, 10, 20), Comp { glyph: 2, flags: 0, args: Args::Points(1, 2), xform: Xform::Scale(0.5) }]),
        300,
        1,
    );
    b.glyph(
        composite(&[
            Comp {
                glyph: 3,
                flags: F_USE_MY_METRICS,
                args: Args::Offset(300, -200),
                xform: Xform::Matrix([0.5, 0.25, -0.25, 0.5]),
            },
            Comp { glyph: 1, flags: F_SCALED_OFFSET, args: Args::Offset(1, 2), xform: Xform::XY(1.5, 0.5) },
        ]),
        200,
        2,
    );
    for _ in 0..6 {
        b.glyph(simple(&[TRIANGLE]), 500, 0);
    }
    b.cmap = Some(cmap_table(&[
        (3, 1, cmap4(&[(0x41, 0x45, 1_u16.wrapping_sub(0x41), &[]), (0x3B1, 0x3B3, 0, &[3, 0, 2])])),
        (3, 10, cmap12(&[(0x41, 0x45, 1), (0x1F600, 0x1F603, 2)])),
        (1, 0, cmap0(&[(0x41, 1), (0x8E, 2)])),
        (0, 3, cmap6(0x30, &[1, 2, 0, 3])),
    ]));
    let name = name_table(&[(3, 1, 0x0409, 1, utf16("Dense")), (1, 0, 0, 2, b"R\x8Egulier".to_vec())]);
    let pairs: KernPairs<'_> = &[(5, 6, -11), (5, 9, 3), (7, 6, -2)];
    let replacement: KernPairs<'_> = &[(5, 6, 4)];
    let kern = kern_table(&[(0x0001, pairs), (0x0009, replacement)]);
    let with_gpos =
        b.clone().with(b"name", name.clone()).with(b"kern", kern.clone()).with(b"GPOS", sample_gpos(b"latn"));
    let with_kern = b.clone().with(b"name", name).with(b"kern", kern);
    // The real layout and character-map tables of the UI font around the small glyph set: GPOS is
    // indexed by glyph id alone, so its pair subtables are reached all the same.
    let mut real_tables = b.clone().with(b"GPOS", table_of(INTER_REGULAR, b"GPOS"));
    real_tables.cmap = Some(table_of(INTER_REGULAR, b"cmap"));
    let mut mono_tables =
        b.with(b"GPOS", table_of(JETBRAINS_MONO, b"GPOS")).with(b"name", table_of(JETBRAINS_MONO, b"name"));
    mono_tables.cmap = Some(table_of(LUCIDE, b"cmap"));
    vec![with_gpos.build(), with_kern.build(), real_tables.build(), mono_tables.build()]
}

#[test]
fn the_dense_corpus_is_valid_before_mutation() {
    for (entry, data) in dense_corpus().iter().enumerate() {
        let font = Font::new_verified(data).unwrap();
        for id in 0..font.glyph_count() {
            assert!(font.outline_to_vec(GlyphId(id)).is_ok(), "entry {entry}, glyph {id}");
        }
        assert!(font.has_kerning() || entry == 3, "entry {entry}");
    }
    let corpus = dense_corpus();
    let font = Font::new(&corpus[2]).unwrap();
    // Inter's kerning of A (glyph 2) and V (glyph 456), read through the transplanted table.
    assert_eq!(font.kerning(GlyphId(2), GlyphId(456)), -140);
    assert!(!font.kerning_pairs(&(0..48).map(GlyphId).collect::<Vec<_>>()).is_empty());
}

/// `[ROB-01]` Mutated structure-dense fonts: no panic, no hang.
#[test]
fn fuzz_dense_fonts() {
    humanart_core::fuzz::run("ttf-dense", &dense_corpus(), 2500, 0x7F_0001, Duration::from_secs(300), exercise);
}

/// `[ROB-01]` The fonts the UI ships, mutated: no panic, no hang.
#[test]
fn fuzz_shipped_fonts() {
    let corpus = [INTER_REGULAR, INTER_ITALIC, JETBRAINS_MONO, LUCIDE].map(<[u8]>::to_vec);
    humanart_core::fuzz::run("ttf-shipped", &corpus, 60, 0x7F_0002, Duration::from_secs(300), exercise);
}

/// `[ROB-01]` The directory of a real font and the first 64 bytes of every table the parser reads,
/// where a random mutation of the whole file rarely lands: each byte in turn set to edge values.
#[test]
fn shipped_font_headers_survive_edge_values() {
    let mut data = JETBRAINS_MONO.to_vec();
    let tables = ["head", "hhea", "maxp", "OS/2", "loca", "cmap", "GPOS", "kern"];
    // The directory, then the first bytes of every table the parser reads.
    let mut spots: Vec<usize> = (0..12 + 16 * 17).collect();
    for tag in tables {
        if let Some(at) = JETBRAINS_MONO.windows(4).take(12 + 16 * 17).position(|w| w == tag.as_bytes()) {
            let offset = u32::from_be_bytes(JETBRAINS_MONO[at + 8..at + 12].try_into().unwrap()) as usize;
            spots.extend(offset..offset + 64);
        }
    }
    for at in spots {
        let original = data[at];
        for value in [0x00, 0x01, 0x7F, 0x80, 0xFF] {
            data[at] = value;
            if let Ok(font) = Font::new(&data) {
                let _ = font.mappings().len();
                for id in (0..font.glyph_count()).step_by(97) {
                    let _ = font.outline_to_vec(GlyphId(id));
                    let _ = font.h_metrics(GlyphId(id));
                }
                let _ = font.kerning(GlyphId(3), GlyphId(4));
            }
        }
        data[at] = original;
    }
}
