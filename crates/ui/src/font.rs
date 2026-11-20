//! Baked fonts: the `.anfont` file and its in-memory form `[UI-03]`.
//!
//! [`crate::fontbake`] turns a TrueType font into a [`BakedFont`]: a multi-channel signed distance
//! field atlas (RGB, `page_size²` texels per page), per-glyph metrics in em units, the kerning pairs
//! between the baked glyphs and the font's vertical metrics. [`BakedFont::to_bytes`] and
//! [`BakedFont::from_bytes`] write and read it as an `.anfont` file; the text layout
//! ([`crate::text`]) reads it at run time. Nothing here touches the GPU or the file system.
//!
//! Every glyph is reachable by code point; a code point the font did not bake maps to the
//! **fallback glyph**, which is always present and stored under U+FFFD: the font's own replacement
//! character when it has one, otherwise its `.notdef` outline (glyph 0, a box in most fonts).
//!
//! # File format (`.anfont`, version 1, little-endian)
//!
//! | Offset | Size | Field |
//! |---|---|---|
//! | 0 | 4 | magic `HAFN` |
//! | 4 | 2 | version, 1 |
//! | 6 | 2 | reserved, 0 |
//! | 8 | 1 | name length `n` (at most 255), then `n` bytes of UTF-8 |
//! | | 28 | metrics: ascender, descender, line gap, x-height, cap height, underline position, underline thickness (`f32`, em, Y up) |
//! | | 8 | `em_px`, `range_px` (`f32`): texels per em, distance range in texels |
//! | | 4 + 2 + 2 | page size (`u32`), page count (`u16`), reserved |
//! | | 12 | glyph count, kerning-pair count, fallback glyph index (`u32` each) |
//! | | 34 × glyphs | code point `u32`, advance `f32`, plane left/bottom/right/top `f32` (em), page `u16`, atlas x/y/w/h `u16` |
//! | | 8 × pairs | left glyph index `u16`, right glyph index `u16`, adjustment `f32` (em) |
//! | | 4 + m | compressed atlas length `m` (`u32`), then a zlib stream of `page_size² × pages × 3` bytes (RGB, page by page, rows top-down) |
//!
//! Glyphs are sorted by strictly increasing code point; kerning pairs by strictly increasing
//! `(left, right)`. The reader checks every count, size and index against the remaining input and
//! the limits below before it allocates, and rejects trailing bytes `[ROB-01]`.

use crate::error::UiError;

/// The file signature.
pub const MAGIC: [u8; 4] = *b"HAFN";
/// The format version this crate writes and reads.
pub const VERSION: u16 = 1;
/// The code point the fallback glyph is stored under.
pub const FALLBACK: char = '\u{FFFD}';
/// Largest atlas page side in texels.
pub const MAX_PAGE_SIZE: u32 = 4096;
/// Most atlas pages in one font.
pub const MAX_PAGES: u16 = 64;
/// Largest decompressed atlas, bytes.
pub const MAX_ATLAS_BYTES: u64 = 64 << 20;
/// Most glyphs in one font (indices are `u16`).
pub const MAX_GLYPHS: u32 = 65_535;
/// Most kerning pairs in one font.
pub const MAX_KERNING_PAIRS: u32 = 1 << 20;
/// Largest magnitude of any metric, in em.
pub const MAX_EM: f32 = 64.0;

/// Bytes per glyph record.
const GLYPH_BYTES: usize = 34;
/// Bytes per kerning record.
const PAIR_BYTES: usize = 8;

/// The vertical metrics of a font, in em, Y up from the baseline.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FontMetrics {
    /// Typographic ascender (positive).
    pub ascender: f32,
    /// Typographic descender (negative).
    pub descender: f32,
    /// Extra gap between lines.
    pub line_gap: f32,
    /// Height of a lower-case x.
    pub x_height: f32,
    /// Height of a flat capital.
    pub cap_height: f32,
    /// Top of the underline (negative: below the baseline).
    pub underline_position: f32,
    /// Underline thickness.
    pub underline_thickness: f32,
}

impl FontMetrics {
    /// The distance between baselines: ascender − descender + line gap.
    #[inline]
    pub fn line_height(&self) -> f32 {
        self.ascender - self.descender + self.line_gap
    }

    fn values(&self) -> [f32; 7] {
        [
            self.ascender,
            self.descender,
            self.line_gap,
            self.x_height,
            self.cap_height,
            self.underline_position,
            self.underline_thickness,
        ]
    }
}

/// One baked glyph.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Glyph {
    /// The code point.
    pub codepoint: u32,
    /// Horizontal advance, em.
    pub advance: f32,
    /// The quad relative to the pen on the baseline, em, Y up: left, bottom, right, top. Zero for a
    /// blank glyph (space).
    pub plane: [f32; 4],
    /// The atlas page.
    pub page: u16,
    /// The quad's texels in its page: x, y (top-left), width, height. Zero size for a blank glyph.
    pub atlas: [u16; 4],
}

impl Glyph {
    /// Whether the glyph draws nothing.
    #[inline]
    pub fn is_blank(&self) -> bool {
        self.atlas[2] == 0 || self.atlas[3] == 0
    }
}

/// A kerning adjustment between two glyphs (indices into [`BakedFont::glyphs`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KerningPair {
    /// The left glyph.
    pub left: u16,
    /// The right glyph.
    pub right: u16,
    /// The change of the left glyph's advance, em (negative: closer).
    pub value: f32,
}

/// A font baked into a distance-field atlas.
#[derive(Clone, Debug, PartialEq)]
pub struct BakedFont {
    /// The font's full name (provenance and diagnostics).
    pub name: String,
    /// Vertical metrics.
    pub metrics: FontMetrics,
    /// Atlas texels per em at bake time.
    pub em_px: f32,
    /// The distance range of the field in texels (a texel value of 0 or 1 is `range_px / 2` texels
    /// outside or inside the outline).
    pub range_px: f32,
    /// The side of a square atlas page in texels.
    pub page_size: u32,
    /// The number of pages.
    pub pages: u16,
    /// The glyphs, by strictly increasing code point.
    pub glyphs: Vec<Glyph>,
    /// The index of the fallback glyph.
    pub fallback: u16,
    /// Kerning pairs by strictly increasing `(left, right)`.
    pub kerning: Vec<KerningPair>,
    /// RGB texels, `page_size² × pages × 3` bytes, page after page, rows top-down.
    pub atlas: Vec<u8>,
    /// Glyph index of every ASCII code point (`u16::MAX` when absent), for allocation-free lookups.
    ascii: [u16; 128],
}

impl BakedFont {
    /// Assembles a font from its parts and checks it as the reader would (the baker's output path).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        metrics: FontMetrics,
        em_px: f32,
        range_px: f32,
        page_size: u32,
        pages: u16,
        glyphs: Vec<Glyph>,
        fallback: u16,
        kerning: Vec<KerningPair>,
        atlas: Vec<u8>,
    ) -> Result<Self, UiError> {
        let mut font = Self {
            name,
            metrics,
            em_px,
            range_px,
            page_size,
            pages,
            glyphs,
            fallback,
            kerning,
            atlas,
            ascii: [u16::MAX; 128],
        };
        font.validate(0)?;
        font.index_ascii();
        Ok(font)
    }

    fn index_ascii(&mut self) {
        self.ascii = [u16::MAX; 128];
        for (i, g) in self.glyphs.iter().enumerate() {
            if let (Some(slot), Ok(i)) = (self.ascii.get_mut(g.codepoint as usize), u16::try_from(i)) {
                *slot = i;
            }
        }
    }

    /// The glyph index of `c`, or the fallback glyph's. No allocation.
    #[inline]
    pub fn glyph_index(&self, c: char) -> u16 {
        let code = c as u32;
        if let Some(&i) = self.ascii.get(code as usize) {
            return if i == u16::MAX { self.fallback } else { i };
        }
        match self.glyphs.binary_search_by_key(&code, |g| g.codepoint) {
            Ok(i) => u16::try_from(i).unwrap_or(self.fallback),
            Err(_) => self.fallback,
        }
    }

    /// Whether `c` has its own glyph (not the fallback).
    pub fn has_glyph(&self, c: char) -> bool {
        self.glyphs.binary_search_by_key(&(c as u32), |g| g.codepoint).is_ok()
    }

    /// The glyph at `index` (the fallback glyph for an index out of range).
    #[inline]
    pub fn glyph(&self, index: u16) -> &Glyph {
        self.glyphs.get(usize::from(index)).or_else(|| self.glyphs.get(usize::from(self.fallback))).unwrap_or(&BLANK)
    }

    /// The kerning between two glyph indices, em. No allocation.
    #[inline]
    pub fn kerning(&self, left: u16, right: u16) -> f32 {
        if self.kerning.is_empty() {
            return 0.0;
        }
        let key = (u32::from(left) << 16) | u32::from(right);
        self.kerning
            .binary_search_by_key(&key, |p| (u32::from(p.left) << 16) | u32::from(p.right))
            .ok()
            .and_then(|i| self.kerning.get(i))
            .map_or(0.0, |p| p.value)
    }

    /// The atlas pages as RGBA8 (alpha 255), page after page.
    pub fn atlas_rgba(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.atlas.len() / 3 * 4);
        for px in self.atlas.as_chunks::<3>().0 {
            out.extend_from_slice(&[px[0], px[1], px[2], 255]);
        }
        out
    }

    /// The `.anfont` bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 + self.glyphs.len() * GLYPH_BYTES + self.kerning.len() * PAIR_BYTES);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&0_u16.to_le_bytes());
        let name = truncate_utf8(&self.name, 255);
        out.push(u8::try_from(name.len()).unwrap_or(0));
        out.extend_from_slice(name.as_bytes());
        for v in self.metrics.values() {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.em_px.to_le_bytes());
        out.extend_from_slice(&self.range_px.to_le_bytes());
        out.extend_from_slice(&self.page_size.to_le_bytes());
        out.extend_from_slice(&self.pages.to_le_bytes());
        out.extend_from_slice(&0_u16.to_le_bytes());
        let count = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
        out.extend_from_slice(&count(self.glyphs.len()).to_le_bytes());
        out.extend_from_slice(&count(self.kerning.len()).to_le_bytes());
        out.extend_from_slice(&u32::from(self.fallback).to_le_bytes());
        for g in &self.glyphs {
            out.extend_from_slice(&g.codepoint.to_le_bytes());
            out.extend_from_slice(&g.advance.to_le_bytes());
            for v in g.plane {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.extend_from_slice(&g.page.to_le_bytes());
            for v in g.atlas {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        for p in &self.kerning {
            out.extend_from_slice(&p.left.to_le_bytes());
            out.extend_from_slice(&p.right.to_le_bytes());
            out.extend_from_slice(&p.value.to_le_bytes());
        }
        let packed = humanart_io::deflate::zlib_compress(&self.atlas, humanart_io::deflate::Level::Best);
        out.extend_from_slice(&count(packed.len()).to_le_bytes());
        out.extend_from_slice(&packed);
        out
    }

    /// Reads an `.anfont` file, checking every field `[ROB-01]`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, UiError> {
        let mut r = Reader { bytes, pos: 0 };
        if r.take(4)? != MAGIC {
            return Err(r.error(0, "not a baked font (bad magic)"));
        }
        let version = r.u16()?;
        if version != VERSION {
            return Err(r.error(4, "unsupported version"));
        }
        let _reserved = r.u16()?;
        let name_len = usize::from(r.u8()?);
        let name_at = r.pos;
        let name =
            std::str::from_utf8(r.take(name_len)?).map_err(|_| r.error(name_at, "name is not UTF-8"))?.to_owned();
        let mut m = [0.0_f32; 7];
        for v in &mut m {
            *v = r.f32()?;
        }
        let metrics = FontMetrics {
            ascender: m[0],
            descender: m[1],
            line_gap: m[2],
            x_height: m[3],
            cap_height: m[4],
            underline_position: m[5],
            underline_thickness: m[6],
        };
        let em_px = r.f32()?;
        let range_px = r.f32()?;
        let page_size = r.u32()?;
        let pages = r.u16()?;
        let _reserved = r.u16()?;
        let sizes_at = r.pos;
        let glyph_count = r.u32()?;
        let pair_count = r.u32()?;
        let fallback = r.u32()?;
        if glyph_count == 0 || glyph_count > MAX_GLYPHS {
            return Err(r.error(sizes_at, "glyph count out of range"));
        }
        if pair_count > MAX_KERNING_PAIRS {
            return Err(r.error(sizes_at + 4, "kerning-pair count over the limit"));
        }
        let fallback = u16::try_from(fallback).map_err(|_| r.error(sizes_at + 8, "fallback index out of range"))?;
        // Both tables must fit in what is left before anything is allocated.
        let table_bytes = glyph_count as usize * GLYPH_BYTES + pair_count as usize * PAIR_BYTES;
        if table_bytes > r.remaining() {
            return Err(r.error(r.pos, "glyph and kerning tables run past the end"));
        }
        let mut glyphs = Vec::with_capacity(glyph_count as usize);
        for _ in 0..glyph_count {
            let codepoint = r.u32()?;
            let advance = r.f32()?;
            let plane = [r.f32()?, r.f32()?, r.f32()?, r.f32()?];
            let page = r.u16()?;
            let atlas = [r.u16()?, r.u16()?, r.u16()?, r.u16()?];
            glyphs.push(Glyph { codepoint, advance, plane, page, atlas });
        }
        let mut kerning = Vec::with_capacity(pair_count as usize);
        for _ in 0..pair_count {
            kerning.push(KerningPair { left: r.u16()?, right: r.u16()?, value: r.f32()? });
        }
        let packed_at = r.pos;
        let packed_len = r.u32()? as usize;
        if packed_len != r.remaining() {
            return Err(r.error(packed_at, "atlas length does not match the rest of the file"));
        }
        let atlas_bytes =
            atlas_size(page_size, pages).ok_or_else(|| r.error(sizes_at - 8, "atlas size out of range"))?;
        let mut atlas = vec![0_u8; atlas_bytes];
        let packed = r.take(packed_len)?;
        let decoded = humanart_io::deflate::zlib_decompress_into(packed, &mut atlas)
            .map_err(|_| r.error(packed_at + 4, "atlas data is damaged"))?;
        if decoded.written != atlas_bytes || decoded.consumed != packed_len {
            return Err(r.error(packed_at + 4, "atlas data is shorter than its pages"));
        }
        let mut font = Self {
            name,
            metrics,
            em_px,
            range_px,
            page_size,
            pages,
            glyphs,
            fallback,
            kerning,
            atlas,
            ascii: [u16::MAX; 128],
        };
        font.validate(sizes_at)?;
        font.index_ascii();
        Ok(font)
    }

    /// Checks the invariants the layout relies on; `at` is the offset reported for table errors.
    fn validate(&self, at: usize) -> Result<(), UiError> {
        let bad = |what| Err(UiError::FontFile { offset: at, what });
        let finite_em = |v: f32| v.is_finite() && v.abs() <= MAX_EM;
        if !self.metrics.values().into_iter().all(finite_em) {
            return bad("metric out of range");
        }
        if !(self.em_px.is_finite() && (1.0..=1024.0).contains(&self.em_px)) {
            return bad("em size out of range");
        }
        if !(self.range_px.is_finite() && (0.5..=64.0).contains(&self.range_px)) {
            return bad("distance range out of range");
        }
        let Some(atlas_bytes) = atlas_size(self.page_size, self.pages) else {
            return bad("atlas size out of range");
        };
        if self.atlas.len() != atlas_bytes {
            return bad("atlas length does not match its pages");
        }
        if self.glyphs.is_empty() || self.glyphs.len() > MAX_GLYPHS as usize {
            return bad("glyph count out of range");
        }
        let mut previous = None;
        for g in &self.glyphs {
            if char::from_u32(g.codepoint).is_none() || previous.is_some_and(|p| p >= g.codepoint) {
                return bad("glyph code points are not valid and strictly increasing");
            }
            previous = Some(g.codepoint);
            if !finite_em(g.advance) || !g.plane.into_iter().all(finite_em) {
                return bad("glyph metric out of range");
            }
            let [x, y, w, h] = g.atlas.map(u32::from);
            if w > 0 && h > 0 && (g.page >= self.pages || x + w > self.page_size || y + h > self.page_size) {
                return bad("glyph lies outside its atlas page");
            }
        }
        if usize::from(self.fallback) >= self.glyphs.len() {
            return bad("fallback index out of range");
        }
        let count = self.glyphs.len();
        let mut previous = None;
        for p in &self.kerning {
            let key = (u32::from(p.left) << 16) | u32::from(p.right);
            if usize::from(p.left) >= count || usize::from(p.right) >= count || previous.is_some_and(|k| k >= key) {
                return bad("kerning pairs are not valid and strictly increasing");
            }
            previous = Some(key);
            if !finite_em(p.value) {
                return bad("kerning value out of range");
            }
        }
        Ok(())
    }
}

/// A blank glyph, returned only if a font somehow has no fallback (cannot happen after validation).
static BLANK: Glyph = Glyph { codepoint: 0, advance: 0.0, plane: [0.0; 4], page: 0, atlas: [0; 4] };

/// The decompressed atlas size of `pages` pages of `page_size²` RGB texels, if within the limits.
fn atlas_size(page_size: u32, pages: u16) -> Option<usize> {
    if page_size == 0 || page_size > MAX_PAGE_SIZE || pages == 0 || pages > MAX_PAGES {
        return None;
    }
    let bytes = u64::from(page_size) * u64::from(page_size) * u64::from(pages) * 3;
    if bytes > MAX_ATLAS_BYTES {
        return None;
    }
    usize::try_from(bytes).ok()
}

/// The longest prefix of `s` of at most `max` bytes that ends on a character boundary.
fn truncate_utf8(s: &str, max: usize) -> &str {
    let mut end = s.len().min(max);
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    s.get(..end).unwrap_or("")
}

/// A bounds-checked little-endian reader.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    #[allow(clippy::unused_self)] // A method so every error site reads `r.error(..)`.
    fn error(&self, offset: usize, what: &'static str) -> UiError {
        UiError::FontFile { offset, what }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], UiError> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.bytes.len());
        let slice = end
            .and_then(|e| self.bytes.get(self.pos..e))
            .ok_or(UiError::FontFile { offset: self.pos, what: "file ends early" })?;
        self.pos += n;
        Ok(slice)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], UiError> {
        let mut out = [0; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, UiError> {
        Ok(self.array::<1>()?[0])
    }

    fn u16(&mut self) -> Result<u16, UiError> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32, UiError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn f32(&mut self) -> Result<f32, UiError> {
        Ok(f32::from_le_bytes(self.array()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small hand-made font: blank space, two inked glyphs, the fallback, one kerning pair.
    pub(crate) fn sample() -> BakedFont {
        let glyph = |codepoint: u32, x: u16| Glyph {
            codepoint,
            advance: 0.6,
            plane: [0.0, 0.0, 0.5, 0.7],
            page: 0,
            atlas: if codepoint == 32 { [0; 4] } else { [x, 0, 6, 8] },
        };
        let mut atlas = vec![0_u8; 16 * 16 * 3];
        for (i, v) in atlas.iter_mut().enumerate() {
            *v = (i * 7 % 256) as u8;
        }
        BakedFont::new(
            "Sample Regular".into(),
            FontMetrics {
                ascender: 0.9,
                descender: -0.25,
                line_gap: 0.0,
                x_height: 0.5,
                cap_height: 0.7,
                underline_position: -0.1,
                underline_thickness: 0.05,
            },
            32.0,
            4.0,
            16,
            1,
            vec![glyph(32, 0), glyph('A' as u32, 0), glyph('V' as u32, 7), glyph(0xFFFD, 0)],
            3,
            vec![KerningPair { left: 1, right: 2, value: -0.08 }],
            atlas,
        )
        .unwrap()
    }

    #[test]
    fn ui_03_round_trip_and_lookups() {
        let font = sample();
        let bytes = font.to_bytes();
        let back = BakedFont::from_bytes(&bytes).unwrap();
        assert_eq!(back, font);
        assert_eq!(back.glyph_index('A'), 1);
        assert_eq!(back.glyph_index('é'), 3, "unknown characters get the fallback");
        assert_eq!(back.glyph_index('\u{1F600}'), 3);
        assert!(back.has_glyph('V') && !back.has_glyph('W'));
        assert_eq!(back.kerning(1, 2), -0.08);
        assert_eq!(back.kerning(2, 1), 0.0);
        assert!(back.glyph(0).is_blank());
        assert_eq!(back.glyph(999).codepoint, 0xFFFD);
        assert_eq!(back.atlas_rgba().len(), 16 * 16 * 4);
        assert!((back.metrics.line_height() - 1.15).abs() < 1e-6);
    }

    #[test]
    fn rob_01_rejects_damaged_files() {
        let bytes = sample().to_bytes();
        // Every truncation fails cleanly.
        for len in 0..bytes.len() {
            assert!(BakedFont::from_bytes(&bytes[..len]).is_err(), "truncated to {len}");
        }
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(BakedFont::from_bytes(&extra).is_err(), "trailing byte");
        // A huge glyph count is refused before allocating.
        let name_len = usize::from(bytes[8]);
        let counts = 9 + name_len + 28 + 8 + 8;
        let mut huge = bytes.clone();
        huge[counts..counts + 4].copy_from_slice(&60_000_u32.to_le_bytes());
        assert!(matches!(BakedFont::from_bytes(&huge), Err(UiError::FontFile { .. })));
        let mut pages = bytes.clone();
        pages[counts - 4..counts - 2].copy_from_slice(&65_000_u16.to_le_bytes());
        assert!(BakedFont::from_bytes(&pages).is_err());
        // Out-of-order glyphs.
        let mut order = sample();
        order.glyphs.swap(1, 2);
        assert!(BakedFont::from_bytes(&order.to_bytes()).is_err());
    }

    #[test]
    fn rob_01_mutation_fuzz() {
        // [ROB-01] Fixed-seed mutations of a valid file never panic or hang.
        let corpus = vec![sample().to_bytes()];
        humanart_core::fuzz::run("anfont", &corpus, 20_000, 0xA4F0_0001, std::time::Duration::from_secs(60), |b| {
            let _ = BakedFont::from_bytes(b);
        });
    }
}
