//! Run-time text: the font set, layout with kerning and wrapping, a layout cache and drawing
//! `[UI-03]`.
//!
//! [`Fonts`] holds the five baked faces ([`FontFace`]) and assembles their atlas pages into the one
//! array texture the renderer binds. [`layout_into`] turns UTF-8 text and a [`TextStyle`] into a
//! [`TextLayout`]: glyphs with their pen positions in logical pixels (kerning from the font's pair
//! tables, as baked), lines broken at `\n` and, with a maximum width, wrapped at spaces (a word wider
//! than the line breaks between characters). [`TextCache`] keeps layouts across frames keyed by
//! text, style and width, so steady-state frames lay out nothing and allocate nothing.
//! [`draw_layout`] emits one glyph quad per inked glyph.
//!
//! **Sizing and positioning.** Sizes are logical pixels; a glyph quad is scaled to physical pixels
//! at draw time (DPI-aware sizing), its horizontal position keeps its fraction (subpixel
//! positioning: the distance field is sampled at the exact offset) while each baseline is rounded
//! to a whole physical pixel so stems and the x-height sit on the pixel grid.
//!
//! **Scope.** Characters map one-to-one to glyphs: no ligatures, no contextual shaping, no
//! combining-mark placement, no bidirectional reordering (complex scripts are out of scope for the
//! UI, `[UI-03]`). Characters a face lacks draw its fallback glyph ([`crate::font::FALLBACK`]). A tab
//! advances to the next multiple of four spaces; other control characters are skipped.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::color::Color;
use crate::error::UiError;
use crate::font::BakedFont;
use crate::geom::{Point, Rect, Size};
use crate::paint::{DrawList, GlyphQuad};
use crate::render::AtlasImage;
use crate::theme::{FontFace, TextStyle};

/// The baked faces, one per [`FontFace`], and where their pages sit in the combined atlas.
#[derive(Clone, Debug)]
pub struct Fonts {
    faces: [BakedFont; 5],
    /// The first atlas layer of each face.
    layer_base: [u32; 5],
}

impl Fonts {
    /// The font set from baked fonts in [`FontFace::ALL`] order. Every face must use the same page
    /// size (the pages become layers of one array texture).
    pub fn new(faces: [BakedFont; 5]) -> Result<Self, UiError> {
        let page = faces[0].page_size;
        let mut layer_base = [0_u32; 5];
        let mut next = 0_u32;
        for (base, face) in layer_base.iter_mut().zip(&faces) {
            if face.page_size != page {
                return Err(UiError::FontFile { offset: 0, what: "faces use different atlas page sizes" });
            }
            *base = next;
            next += u32::from(face.pages);
        }
        Ok(Self { faces, layer_base })
    }

    /// Reads the five `.anfont` files (bytes in [`FontFace::ALL`] order).
    pub fn from_bytes(files: [&[u8]; 5]) -> Result<Self, UiError> {
        let [a, b, c, d, e] = files;
        Self::new([
            BakedFont::from_bytes(a)?,
            BakedFont::from_bytes(b)?,
            BakedFont::from_bytes(c)?,
            BakedFont::from_bytes(d)?,
            BakedFont::from_bytes(e)?,
        ])
    }

    /// The baked font of `face`.
    #[inline]
    pub fn face(&self, face: FontFace) -> &BakedFont {
        &self.faces[face.index()]
    }

    /// The combined atlas for [`crate::render::UiRenderer`]: every page of every face, in face order.
    pub fn atlas(&self) -> AtlasImage {
        let page = self.faces[0].page_size;
        let layers: u32 = self.faces.iter().map(|f| u32::from(f.pages)).sum();
        let mut rgba = Vec::with_capacity((page * page * layers * 4) as usize);
        for face in &self.faces {
            rgba.extend_from_slice(&face.atlas_rgba());
        }
        AtlasImage { width: page, height: page, layers, rgba }
    }

    /// The vertical metrics of `style` in logical pixels: (ascender, descender, line height).
    pub fn line_metrics(&self, style: TextStyle) -> (f32, f32, f32) {
        let m = &self.face(style.face).metrics;
        (m.ascender * style.size, m.descender * style.size, style.size * style.line_height)
    }
}

/// A glyph placed by a layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlacedGlyph {
    /// The glyph index in the face.
    pub glyph: u16,
    /// Pen position on the baseline, logical pixels from the layout's left edge.
    pub x: f32,
    /// The advance after kerning, logical pixels.
    pub advance: f32,
    /// The line.
    pub line: u32,
    /// The byte offset of the character in the text.
    pub byte: u32,
}

/// One line of a layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineBox {
    /// Its glyphs: `glyphs[start..end]`.
    pub start: u32,
    /// One past its last glyph.
    pub end: u32,
    /// The byte range of the line's text (without the breaking `\n`).
    pub bytes: (u32, u32),
    /// The top of the line box, logical pixels from the layout's top.
    pub top: f32,
    /// The baseline, logical pixels from the layout's top.
    pub baseline: f32,
    /// The width of its ink advance (trailing spaces excluded), logical pixels.
    pub width: f32,
}

/// Laid-out text: glyphs and lines in logical pixels, relative to the top-left corner of its box.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextLayout {
    /// The face and size it was laid out in.
    pub style: Option<TextStyle>,
    /// The glyphs, in text order.
    pub glyphs: Vec<PlacedGlyph>,
    /// The lines, top to bottom (at least one, possibly empty).
    pub lines: Vec<LineBox>,
    /// The size of the box: widest line × line count × line height.
    pub size: Size,
    /// The length of the text in bytes.
    pub text_len: u32,
}

impl TextLayout {
    /// The caret position for byte offset `byte` (clamped to the text; inside a character, at its
    /// start): the x on its line and the line index.
    pub fn caret(&self, byte: usize) -> (f32, usize) {
        let byte = u32::try_from(byte).unwrap_or(u32::MAX).min(self.text_len);
        for (i, line) in self.lines.iter().enumerate() {
            let last = i + 1 == self.lines.len();
            if byte < line.bytes.0 || (byte > line.bytes.1 && !last) {
                continue;
            }
            // A wrapped line's end is the next line's start: prefer the next line.
            if byte == line.bytes.1 && !last && self.lines.get(i + 1).is_some_and(|n| n.bytes.0 == byte) {
                continue;
            }
            let glyphs = self.glyphs.get(line.start as usize..line.end as usize).unwrap_or(&[]);
            let mut x = 0.0;
            for g in glyphs {
                if g.byte >= byte {
                    return (g.x, i);
                }
                x = g.x + g.advance;
            }
            return (x, i);
        }
        (0.0, 0)
    }

    /// The byte offset of the caret position nearest to `point` (logical pixels in the layout's
    /// box): the line under it, then the nearest character boundary on that line.
    pub fn hit(&self, point: Point) -> usize {
        let Some(line) = self.lines.iter().rev().find(|l| point.y >= l.top).or(self.lines.first()) else {
            return 0;
        };
        let glyphs = self.glyphs.get(line.start as usize..line.end as usize).unwrap_or(&[]);
        for g in glyphs {
            if point.x < g.x + g.advance * 0.5 {
                return g.byte as usize;
            }
        }
        line.bytes.1 as usize
    }
}

/// Lays `text` out in `style`, wrapping at `max_width` logical pixels when given, into `out`
/// (reusing its memory).
pub fn layout_into(fonts: &Fonts, text: &str, style: TextStyle, max_width: Option<f32>, out: &mut TextLayout) {
    let font = fonts.face(style.face);
    let size = if style.size.is_finite() { style.size.clamp(1.0, 1000.0) } else { 12.0 };
    let (ascender, descender, line_height) = fonts.line_metrics(TextStyle { size, ..style });
    let max_width = max_width.filter(|w| w.is_finite() && *w > 0.0);
    out.style = Some(style);
    out.glyphs.clear();
    out.lines.clear();
    out.text_len = u32::try_from(text.len()).unwrap_or(u32::MAX);
    let space = font.glyph(font.glyph_index(' ')).advance * size;
    let tab = space * 4.0;
    // Baseline within a line box: half the leading above the ascender.
    let lead = ((line_height - (ascender - descender)) * 0.5).max(0.0);

    let mut line_start_glyph = 0_u32;
    let mut line_start_byte = 0_u32;
    let mut pen = 0.0_f32;
    let mut previous: Option<u16> = None;
    // The last break opportunity on this line: (glyph index after the space, byte, pen there).
    let mut last_break: Option<(usize, u32, f32)> = None;
    let mut line_no = 0_u32;
    let mut widest = 0.0_f32;

    let finish_line = |out: &mut TextLayout, start: u32, end: u32, bytes: (u32, u32), line_no: u32| -> f32 {
        let glyphs = out.glyphs.get(start as usize..end as usize).unwrap_or(&[]);
        // Width up to the end of the last non-space glyph.
        let mut width = 0.0_f32;
        for g in glyphs {
            if g.advance > 0.0 && !is_space_at(text, g.byte) {
                width = width.max(g.x + g.advance);
            }
        }
        #[allow(clippy::cast_precision_loss)] // INVARIANT: line counts are far below 2^24.
        let top = line_no as f32 * line_height;
        out.lines.push(LineBox { start, end, bytes, top, baseline: top + lead + ascender, width });
        width
    };

    for (byte, c) in text.char_indices() {
        let byte = u32::try_from(byte).unwrap_or(u32::MAX);
        if c == '\n' {
            let end = u32::try_from(out.glyphs.len()).unwrap_or(u32::MAX);
            widest = widest.max(finish_line(out, line_start_glyph, end, (line_start_byte, byte), line_no));
            line_no += 1;
            line_start_glyph = end;
            line_start_byte = byte + 1;
            pen = 0.0;
            previous = None;
            last_break = None;
            continue;
        }
        if c.is_control() && c != '\t' {
            continue;
        }
        let glyph = font.glyph_index(c);
        let mut advance = if c == '\t' { tab - (pen % tab) } else { font.glyph(glyph).advance * size };
        if let Some(p) = previous {
            let kern = font.kerning(p, glyph) * size;
            if kern != 0.0 {
                pen += kern;
                if let Some(last) = out.glyphs.last_mut() {
                    last.advance += kern;
                }
            }
        }
        if c == '\t' {
            advance = advance.max(0.0);
        }
        // Wrap before this glyph if it does not fit (spaces may hang past the edge).
        if let Some(limit) = max_width
            && pen + advance > limit
            && !c.is_whitespace()
            && out.glyphs.len() > line_start_glyph as usize
        {
            let (cut_glyph, cut_byte) = match last_break {
                Some((g, b, _)) if g > line_start_glyph as usize => (g, b),
                _ => (out.glyphs.len(), byte),
            };
            let cut = u32::try_from(cut_glyph).unwrap_or(u32::MAX);
            widest = widest.max(finish_line(out, line_start_glyph, cut, (line_start_byte, cut_byte), line_no));
            line_no += 1;
            // Move the glyphs after the cut to the new line.
            let shift = out.glyphs.get(cut_glyph).map_or(pen, |g| g.x);
            for g in out.glyphs.iter_mut().skip(cut_glyph) {
                g.x -= shift;
                g.line = line_no;
            }
            pen -= shift;
            line_start_glyph = cut;
            line_start_byte = cut_byte;
            last_break = None;
        }
        out.glyphs.push(PlacedGlyph { glyph, x: pen, advance, line: line_no, byte });
        pen += advance;
        previous = Some(glyph);
        if c == ' ' || c == '\t' || c == '-' || c == '/' {
            last_break = Some((out.glyphs.len(), byte + u32::try_from(c.len_utf8()).unwrap_or(1), pen));
        }
    }
    let end = u32::try_from(out.glyphs.len()).unwrap_or(u32::MAX);
    widest = widest.max(finish_line(out, line_start_glyph, end, (line_start_byte, out.text_len), line_no));
    #[allow(clippy::cast_precision_loss)] // INVARIANT: line counts are far below 2^24.
    let height = out.lines.len() as f32 * line_height;
    out.size = Size::new(widest, height);
}

/// Whether the character at `byte` is a space or a tab.
fn is_space_at(text: &str, byte: u32) -> bool {
    matches!(text.as_bytes().get(byte as usize), Some(b' ' | b'\t'))
}

/// The width and height of `text` on one line (no wrapping), without allocating.
pub fn measure(fonts: &Fonts, text: &str, style: TextStyle) -> Size {
    let font = fonts.face(style.face);
    let (_, _, line_height) = fonts.line_metrics(style);
    let mut widest = 0.0_f32;
    let mut pen = 0.0_f32;
    let mut lines = 1_u32;
    let mut previous = None;
    for c in text.chars() {
        if c == '\n' {
            widest = widest.max(pen);
            pen = 0.0;
            lines += 1;
            previous = None;
            continue;
        }
        if c.is_control() {
            continue;
        }
        let glyph = font.glyph_index(c);
        if let Some(p) = previous {
            pen += font.kerning(p, glyph) * style.size;
        }
        pen += font.glyph(glyph).advance * style.size;
        previous = Some(glyph);
    }
    #[allow(clippy::cast_precision_loss)] // INVARIANT: line counts are far below 2^24.
    let height = lines as f32 * line_height;
    Size::new(widest.max(pen), height)
}

/// Horizontal alignment of lines inside a box.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Align {
    /// Flush left.
    #[default]
    Start,
    /// Centred.
    Center,
    /// Flush right.
    End,
}

/// Draws a layout with its box's top-left corner at `origin`, each line aligned within `width`
/// logical pixels.
pub fn draw_layout(
    list: &mut DrawList,
    fonts: &Fonts,
    layout: &TextLayout,
    origin: Point,
    align: Align,
    width: f32,
    color: Color,
) {
    let Some(style) = layout.style else { return };
    let font = fonts.face(style.face);
    let base = fonts.layer_base[style.face.index()];
    let scale = list.scale();
    let size_px = style.size * scale;
    #[allow(clippy::cast_precision_loss)] // INVARIANT: page sizes are at most 4096.
    let page = font.page_size as f32;
    let screen_range = (font.range_px * size_px / font.em_px).max(1.0);
    let darken = stem_darkening(size_px);
    let clip = list.clip();
    for line in &layout.lines {
        let top = origin.y + line.top;
        // Skip whole lines outside the clip (long scrolled texts).
        if top > clip.bottom() || top + style.size * style.line_height < clip.y {
            continue;
        }
        let shift = match align {
            Align::Start => 0.0,
            Align::Center => ((width - line.width) * 0.5).max(0.0),
            Align::End => (width - line.width).max(0.0),
        };
        let baseline = ((origin.y + line.baseline) * scale).round();
        let glyphs = layout.glyphs.get(line.start as usize..line.end as usize).unwrap_or(&[]);
        for g in glyphs {
            let glyph = font.glyph(g.glyph);
            if glyph.is_blank() {
                continue;
            }
            let pen = (origin.x + shift + g.x) * scale;
            let [l, b, r, t] = glyph.plane;
            let [ax, ay, aw, ah] = glyph.atlas.map(f32::from);
            let quad = GlyphQuad {
                rect: [pen + l * size_px, baseline - t * size_px, pen + r * size_px, baseline - b * size_px],
                uv: [ax / page, ay / page, (ax + aw) / page, (ay + ah) / page],
                layer: base + u32::from(glyph.page),
                screen_range,
                darken,
            };
            list.glyph(&quad, color);
        }
    }
}

/// The outward edge shift of glyphs drawn `size_px` physical pixels tall: [`MAX_DARKENING`] at
/// 9 px and below, fading to none at [`DARKENING_END`] px. Thin stems of small text otherwise fall
/// below one pixel and lose contrast; larger text is drawn at the font's true weight.
pub fn stem_darkening(size_px: f32) -> f32 {
    MAX_DARKENING * ((DARKENING_END - size_px) / (DARKENING_END - 9.0)).clamp(0.0, 1.0)
}

/// The edge shift of the smallest text, physical pixels.
pub const MAX_DARKENING: f32 = 0.12;
/// The size from which text is not darkened, physical pixels.
pub const DARKENING_END: f32 = 16.0;

/// A pass-through hasher for keys that already are hashes.
#[derive(Default)]
struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3);
        }
    }

    fn write_u64(&mut self, v: u64) {
        self.0 = v;
    }
}

/// A cached layout.
#[derive(Debug, Default)]
struct CacheEntry {
    key: u64,
    text: String,
    style: Option<TextStyle>,
    max_width: Option<u32>,
    layout: TextLayout,
    last_used: u64,
    live: bool,
}

/// Counters of the layout cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Lookups answered from the cache.
    pub hits: u64,
    /// Layouts computed.
    pub misses: u64,
    /// Entries evicted.
    pub evictions: u64,
}

/// Layouts kept across frames `[UI-03]`: the same text, style and width laid out last frame is
/// returned without work. Entries unused for [`TextCache::MAX_AGE`] frames are recycled, keeping
/// their memory, so a steady UI allocates nothing.
#[derive(Debug, Default)]
pub struct TextCache {
    entries: Vec<CacheEntry>,
    index: HashMap<u64, u32, BuildHasherDefault<KeyHasher>>,
    free: Vec<u32>,
    frame: u64,
    stats: CacheStats,
}

impl TextCache {
    /// Frames an unused layout is kept.
    pub const MAX_AGE: u64 = 120;
    /// Most cached layouts; beyond it the least recently used is replaced.
    pub const CAPACITY: usize = 8192;

    /// An empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Starts a frame: layouts unused for [`TextCache::MAX_AGE`] frames are recycled.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
        for (i, e) in self.entries.iter_mut().enumerate() {
            if e.live && self.frame.saturating_sub(e.last_used) > Self::MAX_AGE {
                e.live = false;
                self.index.remove(&e.key);
                if let Ok(i) = u32::try_from(i) {
                    self.free.push(i);
                }
                self.stats.evictions += 1;
            }
        }
    }

    /// The counters.
    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// The number of live layouts.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether no layout is cached.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// The layout of `text` in `style`, wrapped at `max_width` when given; computed on a miss.
    pub fn layout(&mut self, fonts: &Fonts, text: &str, style: TextStyle, max_width: Option<f32>) -> &TextLayout {
        let width_bits = max_width.filter(|w| w.is_finite() && *w > 0.0).map(f32::to_bits);
        let key = key_of(text, style, width_bits);
        let frame = self.frame;
        if let Some(&slot) = self.index.get(&key)
            && let Some(e) = self.entries.get(slot as usize)
            && e.text == text
            && e.style == Some(style)
            && e.max_width == width_bits
        {
            self.stats.hits += 1;
            let e = &mut self.entries[slot as usize];
            e.last_used = frame;
            return &e.layout;
        }
        self.stats.misses += 1;
        let slot = if let Some(slot) = self.free.pop() {
            slot as usize
        } else if self.entries.len() < Self::CAPACITY {
            self.entries.push(CacheEntry::default());
            self.entries.len() - 1
        } else {
            // Replace the least recently used entry.
            let lru = self.entries.iter().enumerate().min_by_key(|(_, e)| e.last_used).map_or(0, |(i, _)| i);
            if let Some(e) = self.entries.get(lru) {
                self.index.remove(&e.key);
            }
            self.stats.evictions += 1;
            lru
        };
        // A colliding key keeps the newer layout.
        if let Some(old) = self.index.insert(key, u32::try_from(slot).unwrap_or(0))
            && old as usize != slot
            && let Some(e) = self.entries.get_mut(old as usize)
        {
            e.live = false;
            self.free.push(old);
        }
        let e = &mut self.entries[slot];
        e.key = key;
        e.text.clear();
        e.text.push_str(text);
        e.style = Some(style);
        e.max_width = width_bits;
        e.last_used = frame;
        e.live = true;
        layout_into(fonts, text, style, max_width, &mut e.layout);
        &e.layout
    }
}

/// The cache key: FNV-1a over the text and the style.
fn key_of(text: &str, style: TextStyle, width_bits: Option<u32>) -> u64 {
    let mut h = KeyHasher(0xCBF2_9CE4_8422_2325);
    h.write(text.as_bytes());
    h.write(&[style.face as u8]);
    h.write(&style.size.to_bits().to_le_bytes());
    h.write(&style.line_height.to_bits().to_le_bytes());
    h.write(&width_bits.unwrap_or(u32::MAX).to_le_bytes());
    h.0
}

/// Draws `text` in `style` and `color` with its box's top-left corner at `origin` (no wrapping),
/// through the cache; returns the box.
pub fn draw_text(
    list: &mut DrawList,
    fonts: &Fonts,
    cache: &mut TextCache,
    text: &str,
    style: TextStyle,
    origin: Point,
    color: Color,
) -> Rect {
    let layout = cache.layout(fonts, text, style, None);
    draw_layout(list, fonts, layout, origin, Align::Start, layout.size.w, color);
    Rect::new(origin.x, origin.y, layout.size.w, layout.size.h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{assert_golden, fonts, gpu_lock, render, renderer};
    use crate::theme::Theme;
    use std::fmt::Write as _;

    fn style(size: f32) -> TextStyle {
        TextStyle::new(FontFace::Regular, size, 1.4)
    }

    #[test]
    fn ac_wp12_3_kerning_is_applied() {
        // [AC-WP12-3] "AV" sits closer than the advances of A and V by the font's kerning.
        let f = fonts();
        let font = f.face(FontFace::Regular);
        let (a, v) = (font.glyph_index('A'), font.glyph_index('V'));
        let kern = font.kerning(a, v);
        assert!(kern < 0.0);
        let mut layout = TextLayout::default();
        layout_into(f, "AV", style(20.0), None, &mut layout);
        let expect = (font.glyph(a).advance + kern) * 20.0;
        assert!((layout.glyphs[1].x - expect).abs() < 1e-4, "{} vs {expect}", layout.glyphs[1].x);
        let plain = measure(f, "A", style(20.0)).w + measure(f, "V", style(20.0)).w;
        assert!((measure(f, "AV", style(20.0)).w - (plain + kern * 20.0)).abs() < 1e-4);
        // The mono face has no kerning: advances add up.
        let mono = TextStyle::new(FontFace::Mono, 12.0, 1.5);
        let one = measure(f, "0", mono).w;
        assert!((measure(f, "0123456789", mono).w - 10.0 * one).abs() < 1e-3);
    }

    #[test]
    fn wraps_at_spaces_and_breaks_long_words() {
        let f = fonts();
        let mut layout = TextLayout::default();
        let text = "The quick brown fox jumps over the lazy dog";
        layout_into(f, text, style(13.0), Some(80.0), &mut layout);
        assert!(layout.lines.len() >= 4, "{}", layout.lines.len());
        for line in &layout.lines {
            assert!(line.width <= 80.0 + 1e-3, "{line:?}");
            let s = &text[line.bytes.0 as usize..line.bytes.1 as usize];
            assert!(!s.starts_with(' '), "{s:?}");
        }
        // Every byte of the text is in exactly one line (breaking spaces stay at line ends).
        assert_eq!(layout.lines[0].bytes.0, 0);
        for w in layout.lines.windows(2) {
            assert_eq!(w[0].bytes.1, w[1].bytes.0);
        }
        // A word wider than the line breaks between characters.
        layout_into(f, "Supercalifragilistic", style(13.0), Some(30.0), &mut layout);
        assert!(layout.lines.len() > 3);
        assert!(layout.lines.iter().all(|l| l.width <= 30.0 + 1e-3 || l.end - l.start == 1));
        // Hard breaks, an empty last line, unknown characters take the fallback glyph.
        layout_into(f, "a\n\nb\n", style(13.0), None, &mut layout);
        assert_eq!(layout.lines.len(), 4);
        layout_into(f, "\u{4E2D}", style(13.0), None, &mut layout);
        assert_eq!(layout.glyphs[0].glyph, f.face(FontFace::Regular).fallback);
    }

    #[test]
    fn caret_and_hit_testing_agree() {
        let f = fonts();
        let mut layout = TextLayout::default();
        let text = "Hello w\u{f6}rld, line two";
        layout_into(f, text, style(14.0), Some(70.0), &mut layout);
        for (byte, _) in text.char_indices().chain([(text.len(), ' ')]) {
            let (x, line) = layout.caret(byte);
            let l = layout.lines[line];
            let back = layout.hit(Point::new(x + 0.01, l.top + 2.0));
            assert_eq!(back, byte, "byte {byte} at ({x}, line {line})");
        }
        assert_eq!(layout.hit(Point::new(-5.0, -5.0)), 0);
        assert_eq!(layout.hit(Point::new(1e4, 1e4)), text.len());
    }

    #[test]
    fn perf_r1_cache_hits_do_not_allocate() {
        // [UI-03] [PERF-R1] A cached layout is returned without work or allocation.
        let f = fonts();
        let mut cache = TextCache::new();
        let labels: Vec<String> = (0..200).map(|i| format!("Property {i}: value")).collect();
        let frame = |cache: &mut TextCache| {
            cache.begin_frame();
            let mut total = 0.0;
            for l in &labels {
                total += cache.layout(f, l, style(12.0), None).size.w;
            }
            total
        };
        let first = frame(&mut cache);
        assert_eq!(cache.stats().misses, 200);
        let (second, stats) = humanart_core::alloc::measure(|| frame(&mut cache));
        assert_eq!(first, second);
        assert_eq!(cache.stats().hits, 200);
        assert_eq!(stats.events(), 0);
        // Unused layouts are recycled after MAX_AGE frames.
        for _ in 0..=TextCache::MAX_AGE + 1 {
            cache.begin_frame();
        }
        assert!(cache.is_empty());
        assert_eq!(cache.stats().evictions, 200);
    }

    /// The horizontal ink profile of a row, normalised 0..1 (white text on black).
    fn row(rgba: &[u8], width: u32, y: u32) -> Vec<f32> {
        (0..width).map(|x| f32::from(rgba[((y * width + x) * 4) as usize]) / 255.0).collect()
    }

    /// Every stem crossed by a row: (peak ink, total ink = stem width in pixels, pixels touched).
    fn stems(profile: &[f32]) -> Vec<(f32, f32, usize)> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < profile.len() {
            if profile[i] > 0.05 {
                let start = i;
                while i < profile.len() && profile[i] > 0.05 {
                    i += 1;
                }
                let s = &profile[start..i];
                out.push((s.iter().copied().fold(0.0, f32::max), s.iter().sum(), s.len()));
            }
            i += 1;
        }
        out
    }

    #[test]
    fn ac_wp12_3_text_is_crisp_from_9_to_48_px() {
        // [AC-WP12-3] Stems of "l" rendered white on black at fractional pen positions. Crisp means:
        // the ink of a stem w pixels wide touches at most ceil(w) + 1 pixels (no blur spreading it),
        // the strongest pixel reaches at least half ink even at 9 px, and a stem two or more pixels
        // wide has a fully inked core.
        let _lock = gpu_lock();
        let f = fonts();
        let mut r = renderer(f.atlas());
        let mut cache = TextCache::new();
        let mut report = String::new();
        for size in [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 16.0, 20.0, 24.0, 32.0, 40.0, 48.0] {
            let (w, h) = (240_u32, 80_u32);
            let mut list = DrawList::new();
            list.begin(w, h, 1.0);
            let st = TextStyle::new(FontFace::Regular, size, 1.2);
            draw_text(&mut list, f, &mut cache, "l l l l", st, Point::new(4.3, 4.0), Color::WHITE);
            list.finish();
            let rgba = render(&mut r, &list, w, h, Color::BLACK);
            let layout = cache.layout(f, "l l l l", st, None);
            let baseline = 4.0 + layout.lines[0].baseline;
            let y = (baseline - size * 0.3).round() as u32;
            let found = stems(&row(&rgba, w, y));
            assert_eq!(found.len(), 4, "{size} px: {found:?}");
            for &(peak, ink, touched) in &found {
                let _ = write!(report, "{size}px: w {ink:.2} peak {peak:.2} px {touched}; ");
                assert!(touched <= ink.ceil() as usize + 1, "{size} px blurred: {report}");
                assert!(peak >= 0.5, "{size} px faint: {report}");
                if ink >= 2.0 {
                    assert!(peak >= 0.95, "{size} px soft core: {report}");
                }
            }
        }
        eprintln!("{report}");
    }

    #[test]
    fn ac_wp12_3_text_golden_at_100_150_200_percent() {
        // [AC-WP12-3] [UI-09] A type specimen, 9 to 48 px, every face, three UI scales.
        let _lock = gpu_lock();
        let f = fonts();
        let theme = Theme::dark();
        let mut r = renderer(f.atlas());
        let mut cache = TextCache::new();
        let sample = "Hamburgefonstiv AV To 0123 \u{e9}\u{df}\u{20ac}";
        for (scale, name) in [(1.0, "text_specimen_100"), (1.5, "text_specimen_150"), (2.0, "text_specimen_200")] {
            let (w, h) = ((420.0 * scale) as u32, (330.0 * scale) as u32);
            let mut list = DrawList::new();
            list.begin(w, h, scale);
            let mut y = 6.0;
            for size in [9.0, 11.0, 13.0, 16.0, 20.0, 28.0, 48.0] {
                let st = TextStyle::new(FontFace::Regular, size, 1.25);
                y += draw_text(&mut list, f, &mut cache, sample, st, Point::new(8.0, y), theme.palette.text).h;
            }
            for (face, text) in [
                (FontFace::SemiBold, "SemiBold: Export settings"),
                (FontFace::Italic, "Italic: colour-neutral mode"),
                (FontFace::Mono, "Mono: 0.4250 -12.75 [x, y]"),
            ] {
                let st = TextStyle::new(face, 13.0, 1.4);
                y += draw_text(&mut list, f, &mut cache, text, st, Point::new(8.0, y), theme.palette.text_muted).h;
            }
            // A few icons and a wrapped paragraph.
            let icons: String = f
                .face(FontFace::Icons)
                .glyphs
                .iter()
                .skip(40)
                .step_by(37)
                .take(12)
                .filter_map(|g| char::from_u32(g.codepoint))
                .collect();
            let icon_style = TextStyle::icon(16.0);
            y +=
                draw_text(&mut list, f, &mut cache, &icons, icon_style, Point::new(8.0, y + 2.0), theme.palette.text).h;
            let words = "Wrapped text keeps kerning (AV, To) and breaks at spaces.";
            let para = cache.layout(f, words, theme.text.small, Some(200.0)).clone();
            draw_layout(&mut list, f, &para, Point::new(8.0, y + 4.0), Align::Start, 200.0, theme.palette.text_muted);
            list.finish();
            let rgba = render(&mut r, &list, w, h, theme.palette.bg);
            assert_eq!(r.stats().draw_calls, 1);
            assert_golden(name, w, h, &rgba, 2, (w * h / 1000) as usize);
        }
    }
}
