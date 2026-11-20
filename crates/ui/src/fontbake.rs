//! The font baker: TrueType outlines to a multi-channel signed distance field atlas `[UI-03]`.
//!
//! [`bake`] reads a font with `humanart-io::ttf`, and for every requested character renders its
//! outline into a small multi-channel distance field (MSDF), packs the fields into square atlas
//! pages, and collects the advances, the plane bounds, the kerning pairs between the baked glyphs
//! (GPOS pair adjustments or the `kern` table, as `humanart-io` reports them) and the vertical
//! metrics into a [`BakedFont`]. It runs offline (`tools/fontbake/`), never per frame.
//!
//! **Method** (after Chlumský, *Shape Decomposition for Multi-channel Distance Fields*, 2015):
//!
//! 1. The outline's line and quadratic edges are flattened (chord error at most
//!    [`FLATTEN_TOLERANCE`] texels) and cut into pieces at most one texel long.
//! 2. **Overlaps.** Fonts made from strokes (the icon font) have overlapping contours. A piece is
//!    kept only if the fill (non-zero winding of all contours) differs on its two sides, that is if
//!    it lies on the boundary of the union; interior pieces are dropped. Each kept piece records
//!    which side is inside, so signs never depend on contour orientation.
//! 3. **Edge colouring.** Contour edges meeting at a corner (direction change over
//!    [`CORNER_ANGLE`]) get different channel sets (cyan, magenta, yellow; a smooth contour is
//!    white; a contour with one corner is split into three coloured parts), so that the median of
//!    the three channels reproduces sharp corners.
//! 4. **Distances.** Per texel and channel, the nearest edge carrying the channel gives a signed
//!    pseudo-distance (extended along the end tangents beyond an edge's ends); ties go to the more
//!    orthogonal edge. The true signed distance (sign from the winding rule) is kept alongside.
//! 5. **Error correction.** A texel whose channel median disagrees in sign with the true distance
//!    takes the true distance in all channels; then texels whose interpolation with a neighbour
//!    would create a false edge (a "clash", Chlumský's detector) take their median.
//!
//! Values are stored as `0.5 + distance / range_px` in 8 bits; above 0.5 is inside.

use humanart_io::ttf::{Font, GlyphId, OutlineSink};

use crate::error::UiError;
use crate::font::{BakedFont, FALLBACK, FontMetrics, Glyph, KerningPair, MAX_GLYPHS, MAX_PAGES};

/// Largest chord error when flattening curves, texels.
pub const FLATTEN_TOLERANCE: f64 = 0.02;
/// The angle (radians) above which two consecutive edges meet at a corner.
pub const CORNER_ANGLE: f64 = 3.0;
/// Empty texels between glyphs in a page (stops bilinear filtering from bleeding).
const GUTTER: u32 = 1;

/// How to bake.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BakeOptions {
    /// Atlas texels per em.
    pub em_px: f32,
    /// The distance range in texels: the field saturates `range_px / 2` texels from the outline.
    pub range_px: f32,
    /// The side of a square atlas page, texels.
    pub page_size: u32,
}

impl BakeOptions {
    /// The UI text faces: 32 texels per em, range 4, 1024-texel pages.
    pub const TEXT: Self = Self { em_px: 32.0, range_px: 4.0, page_size: 1024 };
    /// The icon font: 24 texels per em (Lucide's design grid), range 3, 1024-texel pages.
    pub const ICONS: Self = Self { em_px: 24.0, range_px: 3.0, page_size: 1024 };
}

/// The character set of the text faces `[UI-03]`: printable ASCII, Latin-1, Latin Extended-A,
/// and the common punctuation and symbols of Western European text (dashes, quotes, bullet,
/// ellipsis, euro, trade mark, minus, arrows). Characters a font lacks are skipped.
pub fn latin_charset() -> Vec<char> {
    let mut out: Vec<char> = (0x20_u32..=0x7E).chain(0xA0..=0x17F).filter_map(char::from_u32).collect();
    out.extend([
        '\u{2013}', '\u{2014}', '\u{2018}', '\u{2019}', '\u{201A}', '\u{201C}', '\u{201D}', '\u{201E}', '\u{2020}',
        '\u{2021}', '\u{2022}', '\u{2026}', '\u{2030}', '\u{2039}', '\u{203A}', '\u{20AC}', '\u{2122}', '\u{2190}',
        '\u{2191}', '\u{2192}', '\u{2193}', '\u{2212}', '\u{2248}', '\u{2260}', '\u{2264}', '\u{2265}', '\u{00B7}',
    ]);
    out.sort_unstable();
    out.dedup();
    out
}

/// Every character the font maps (the icon font: every icon).
pub fn all_mapped(font: &Font<'_>) -> Vec<char> {
    let mut out = Vec::new();
    font.for_each_mapping(|c, _| out.push(c));
    out.sort_unstable();
    out.dedup();
    out
}

/// Bakes the characters `chars` of the TrueType font `ttf` (characters the font does not map are
/// skipped; the fallback glyph is always added).
pub fn bake(ttf: &[u8], chars: &[char], options: &BakeOptions) -> Result<BakedFont, UiError> {
    if !(options.em_px.is_finite() && (4.0..=256.0).contains(&options.em_px)) {
        return Err(UiError::Bake("em size must be 4 to 256 texels"));
    }
    if !(options.range_px.is_finite() && (1.0..=32.0).contains(&options.range_px)) {
        return Err(UiError::Bake("distance range must be 1 to 32 texels"));
    }
    if !(64..=crate::font::MAX_PAGE_SIZE).contains(&options.page_size) {
        return Err(UiError::Bake("page size must be 64 to 4096 texels"));
    }
    let font = Font::new(ttf)?;
    let upem = f64::from(font.units_per_em().max(1));
    let scale = f64::from(options.em_px) / upem;

    // The glyphs to bake, by code point, with the fallback under U+FFFD.
    let mut wanted: Vec<(u32, GlyphId)> = Vec::with_capacity(chars.len() + 1);
    for &c in chars {
        if c == FALLBACK {
            continue;
        }
        if let Some(g) = font.glyph_index(c) {
            wanted.push((c as u32, g));
        }
    }
    wanted.push((FALLBACK as u32, font.glyph_index(FALLBACK).unwrap_or(GlyphId::NOTDEF)));
    wanted.sort_unstable_by_key(|w| w.0);
    wanted.dedup_by_key(|w| w.0);
    if wanted.len() > MAX_GLYPHS as usize {
        return Err(UiError::Bake("more glyphs than a baked font holds"));
    }

    // Render every distinct glyph once.
    let mut fields: Vec<(GlyphId, Option<Field>)> = Vec::new();
    for &(_, gid) in &wanted {
        if fields.iter().any(|f| f.0 == gid) {
            continue;
        }
        let shape = Shape::load(&font, gid, scale)?;
        fields.push((gid, shape.render(options.range_px.into())));
    }

    // Pack the inked fields into pages, tallest first.
    let mut order: Vec<usize> = (0..fields.len()).filter(|&i| fields[i].1.is_some()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(fields[i].1.as_ref().map_or(0, |f| f.height)));
    let page = options.page_size;
    let mut places: Vec<Option<(u16, u32, u32)>> = vec![None; fields.len()];
    let (mut page_index, mut x, mut y, mut shelf) = (0_u32, GUTTER, GUTTER, 0_u32);
    for &i in &order {
        let Some(field) = &fields[i].1 else { continue };
        if field.width + 2 * GUTTER > page || field.height + 2 * GUTTER > page {
            return Err(UiError::Bake("a glyph is larger than an atlas page"));
        }
        if x + field.width + GUTTER > page {
            x = GUTTER;
            y += shelf + GUTTER;
            shelf = 0;
        }
        if y + field.height + GUTTER > page {
            page_index += 1;
            x = GUTTER;
            y = GUTTER;
            shelf = 0;
        }
        let p = u16::try_from(page_index).map_err(|_| UiError::Bake("too many atlas pages"))?;
        places[i] = Some((p, x, y));
        x += field.width + GUTTER;
        shelf = shelf.max(field.height);
    }
    let pages = u16::try_from(page_index + 1).map_err(|_| UiError::Bake("too many atlas pages"))?;
    if pages > MAX_PAGES || crate::font::MAX_ATLAS_BYTES < u64::from(page) * u64::from(page) * 3 * u64::from(pages) {
        return Err(UiError::Bake("the atlas exceeds the size limit; lower em_px or bake fewer glyphs"));
    }
    let page_bytes = (page * page * 3) as usize;
    let mut atlas = vec![0_u8; page_bytes * usize::from(pages)];
    for (i, (_, field)) in fields.iter().enumerate() {
        let (Some(field), Some((p, px, py))) = (field, places[i]) else { continue };
        let base = usize::from(p) * page_bytes;
        for row in 0..field.height {
            let src = (row * field.width * 3) as usize..((row + 1) * field.width * 3) as usize;
            let dst = base + (((py + row) * page + px) * 3) as usize;
            atlas[dst..dst + src.len()].copy_from_slice(&field.rgb[src]);
        }
    }

    // Glyph records.
    let em = f64::from(options.em_px);
    let mut glyphs = Vec::with_capacity(wanted.len());
    let mut fallback = 0_u16;
    for (index, &(codepoint, gid)) in wanted.iter().enumerate() {
        let advance = (f64::from(font.advance(gid)) / upem) as f32;
        let slot = fields.iter().position(|f| f.0 == gid).unwrap_or(0);
        let mut glyph = Glyph { codepoint, advance, plane: [0.0; 4], page: 0, atlas: [0; 4] };
        if let (Some((_, Some(field))), Some(Some((p, px, py)))) = (fields.get(slot), places.get(slot)) {
            let to_u16 = |v: u32| u16::try_from(v).map_err(|_| UiError::Bake("atlas coordinate over 65535"));
            glyph.page = *p;
            glyph.atlas = [to_u16(*px)?, to_u16(*py)?, to_u16(field.width)?, to_u16(field.height)?];
            glyph.plane = [
                (f64::from(field.origin_x) / em) as f32,
                (f64::from(field.origin_y) / em) as f32,
                (f64::from(field.origin_x + i32::try_from(field.width).unwrap_or(0)) / em) as f32,
                (f64::from(field.origin_y + i32::try_from(field.height).unwrap_or(0)) / em) as f32,
            ];
        }
        if codepoint == FALLBACK as u32 {
            fallback = u16::try_from(index).unwrap_or(0);
        }
        glyphs.push(glyph);
    }

    // Kerning between the baked glyphs.
    let mut distinct: Vec<GlyphId> = wanted.iter().map(|w| w.1).collect();
    distinct.sort_unstable_by_key(|g| g.0);
    distinct.dedup();
    let mut kerning = Vec::new();
    font.for_each_kerning_pair(&distinct, |pair| {
        let value = (f64::from(pair.value) / upem) as f32;
        for (l, wl) in wanted.iter().enumerate() {
            if wl.1 != pair.left {
                continue;
            }
            for (r, wr) in wanted.iter().enumerate() {
                if wr.1 == pair.right
                    && let (Ok(left), Ok(right)) = (u16::try_from(l), u16::try_from(r))
                {
                    kerning.push(KerningPair { left, right, value });
                }
            }
        }
    });
    kerning.sort_unstable_by_key(|p| (p.left, p.right));
    kerning.dedup_by_key(|p| (p.left, p.right));

    let m = font.metrics();
    let em_of = |v: i16| (f64::from(v) / upem) as f32;
    let metrics = FontMetrics {
        ascender: em_of(m.ascender),
        descender: em_of(m.descender),
        line_gap: em_of(m.line_gap),
        x_height: m.x_height.map_or(0.5, em_of),
        cap_height: m.cap_height.map_or(0.7, em_of),
        underline_position: em_of(m.underline_position),
        underline_thickness: em_of(m.underline_thickness),
    };
    let name = font.full_name().or_else(|| font.family_name()).unwrap_or_default();
    BakedFont::new(name, metrics, options.em_px, options.range_px, page, pages, glyphs, fallback, kerning, atlas)
}

/// A rendered glyph field.
struct Field {
    width: u32,
    height: u32,
    /// The bottom-left corner of the field in texels relative to the pen on the baseline (Y up).
    origin_x: i32,
    origin_y: i32,
    rgb: Vec<u8>,
}

/// A piece of the outline on the boundary of the fill, in texels (Y up).
#[derive(Clone, Copy, Debug)]
struct Piece {
    a: [f64; 2],
    b: [f64; 2],
    /// Whether the fill lies to the right of `a → b`.
    inside_right: bool,
}

/// An edge of a contour: a run of kept pieces with its colour and end tangents.
#[derive(Clone, Debug)]
struct Edge {
    pieces: std::ops::Range<usize>,
    color: u8,
    start_dir: [f64; 2],
    end_dir: [f64; 2],
    /// Whether the first kept piece starts at the edge's start (the pseudo-distance may extend
    /// it), and the last ends at its end.
    starts_at_start: bool,
    ends_at_end: bool,
}

/// One raw edge from the font, in texels.
#[derive(Clone, Copy, Debug)]
enum Raw {
    Line([f64; 2], [f64; 2]),
    Quad([f64; 2], [f64; 2], [f64; 2]),
}

impl Raw {
    fn start_dir(&self) -> [f64; 2] {
        match *self {
            Self::Line(a, b) => sub(b, a),
            Self::Quad(a, c, b) => {
                let d = sub(c, a);
                if dot(d, d) > 1e-18 { d } else { sub(b, a) }
            }
        }
    }

    fn end_dir(&self) -> [f64; 2] {
        match *self {
            Self::Line(a, b) => sub(b, a),
            Self::Quad(a, c, b) => {
                let d = sub(b, c);
                if dot(d, d) > 1e-18 { d } else { sub(b, a) }
            }
        }
    }

    /// The flattened points after the start, appended to `out`.
    fn flatten(&self, out: &mut Vec<[f64; 2]>) {
        match *self {
            Self::Line(_, b) => out.push(b),
            Self::Quad(a, c, b) => {
                let d = [a[0] - 2.0 * c[0] + b[0], a[1] - 2.0 * c[1] + b[1]];
                let n = ((dot(d, d).sqrt() / (4.0 * FLATTEN_TOLERANCE)).sqrt().ceil() as usize).clamp(1, 256);
                for i in 1..=n {
                    let t = i as f64 / n as f64;
                    let u = 1.0 - t;
                    out.push([
                        u * u * a[0] + 2.0 * u * t * c[0] + t * t * b[0],
                        u * u * a[1] + 2.0 * u * t * c[1] + t * t * b[1],
                    ]);
                }
            }
        }
    }
}

/// Collects a glyph outline as raw edges per contour, scaled to texels.
struct Collector {
    scale: f64,
    contours: Vec<Vec<Raw>>,
    current: Vec<Raw>,
    start: [f64; 2],
    at: [f64; 2],
}

impl Collector {
    fn point(&self, p: [f32; 2]) -> [f64; 2] {
        [f64::from(p[0]) * self.scale, f64::from(p[1]) * self.scale]
    }

    fn finish_contour(&mut self) {
        if self.at != self.start {
            self.current.push(Raw::Line(self.at, self.start));
        }
        if !self.current.is_empty() {
            self.contours.push(std::mem::take(&mut self.current));
        }
        self.at = self.start;
    }
}

impl OutlineSink for Collector {
    fn move_to(&mut self, to: [f32; 2]) {
        self.finish_contour();
        self.start = self.point(to);
        self.at = self.start;
    }

    fn line_to(&mut self, to: [f32; 2]) {
        let to = self.point(to);
        if to != self.at {
            self.current.push(Raw::Line(self.at, to));
        }
        self.at = to;
    }

    fn quad_to(&mut self, control: [f32; 2], to: [f32; 2]) {
        let (c, to) = (self.point(control), self.point(to));
        if to != self.at {
            self.current.push(Raw::Quad(self.at, c, to));
        }
        self.at = to;
    }

    fn close(&mut self) {
        self.finish_contour();
    }
}

/// A glyph prepared for distance evaluation.
struct Shape {
    /// Every flattened segment of every contour (the winding rule uses all of them).
    segments: Vec<([f64; 2], [f64; 2])>,
    /// The kept boundary pieces.
    pieces: Vec<Piece>,
    /// Edges with at least one kept piece.
    edges: Vec<Edge>,
    /// Bounds of the outline, texels.
    bounds: [f64; 4],
}

const RED: u8 = 1;
const GREEN: u8 = 2;
const BLUE: u8 = 4;
const YELLOW: u8 = RED | GREEN;
const MAGENTA: u8 = RED | BLUE;
const CYAN: u8 = GREEN | BLUE;
const WHITE: u8 = 7;

impl Shape {
    fn load(font: &Font<'_>, gid: GlyphId, scale: f64) -> Result<Self, UiError> {
        let mut c = Collector { scale, contours: Vec::new(), current: Vec::new(), start: [0.0; 2], at: [0.0; 2] };
        font.outline(gid, &mut c)?;
        c.finish_contour();
        let contours = c.contours;

        // Flatten every edge; keep per edge its point run.
        let mut flat: Vec<Vec<Vec<[f64; 2]>>> = Vec::with_capacity(contours.len());
        let mut segments = Vec::new();
        let mut bounds = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for contour in &contours {
            let mut edges = Vec::with_capacity(contour.len());
            for raw in contour {
                let start = match *raw {
                    Raw::Line(a, _) | Raw::Quad(a, _, _) => a,
                };
                let mut points = vec![start];
                raw.flatten(&mut points);
                for w in points.windows(2) {
                    segments.push((w[0], w[1]));
                }
                for p in &points {
                    bounds = [bounds[0].min(p[0]), bounds[1].min(p[1]), bounds[2].max(p[0]), bounds[3].max(p[1])];
                }
                edges.push(points);
            }
            flat.push(edges);
        }

        // Colour the edges, then cut them into pieces and keep the boundary ones.
        let mut pieces = Vec::new();
        let mut edges = Vec::new();
        for (contour, points) in contours.iter().zip(&flat) {
            let colors = color_contour(contour);
            for ((raw, pts), color) in contour.iter().zip(points).zip(colors) {
                let first = pieces.len();
                let mut first_kept = None;
                let mut last_kept = None;
                let mut total = 0_usize;
                for w in pts.windows(2) {
                    let (a, b) = (w[0], w[1]);
                    let len = dot(sub(b, a), sub(b, a)).sqrt();
                    let n = (len.ceil() as usize).clamp(1, 4096);
                    for i in 0..n {
                        let pa = lerp(a, b, i as f64 / n as f64);
                        let pb = lerp(a, b, (i + 1) as f64 / n as f64);
                        if let Some(inside_right) = classify(&segments, pa, pb) {
                            if first_kept.is_none() {
                                first_kept = Some(total);
                            }
                            last_kept = Some(total);
                            pieces.push(Piece { a: pa, b: pb, inside_right });
                        }
                        total += 1;
                    }
                }
                if pieces.len() > first {
                    edges.push(Edge {
                        pieces: first..pieces.len(),
                        color,
                        start_dir: normalize(raw.start_dir()),
                        end_dir: normalize(raw.end_dir()),
                        starts_at_start: first_kept == Some(0),
                        ends_at_end: last_kept == Some(total.saturating_sub(1)),
                    });
                }
            }
        }
        Ok(Self { segments, pieces, edges, bounds })
    }

    /// The non-zero winding number of `p` against every segment.
    fn inside(&self, p: [f64; 2]) -> bool {
        winding(&self.segments, p) != 0
    }

    /// The nearest piece of `edge` to `p`: (signed distance, orthogonality, piece index, t).
    fn edge_distance(&self, edge: &Edge, p: [f64; 2]) -> (f64, f64, usize, f64) {
        let mut best = (f64::INFINITY, f64::INFINITY, edge.pieces.start, 0.0);
        for i in edge.pieces.clone() {
            let piece = &self.pieces[i];
            let ab = sub(piece.b, piece.a);
            let ap = sub(p, piece.a);
            let len2 = dot(ab, ab).max(1e-24);
            let t = (dot(ap, ab) / len2).clamp(0.0, 1.0);
            let q = [piece.a[0] + ab[0] * t, piece.a[1] + ab[1] * t];
            let qp = sub(p, q);
            let d = dot(qp, qp).sqrt();
            let ortho = if t > 0.0 && t < 1.0 || d < 1e-12 { 0.0 } else { (dot(ab, qp) / (len2.sqrt() * d)).abs() };
            let right = cross(ap, ab) > 0.0;
            let signed = if right == piece.inside_right { d } else { -d };
            if d < best.0.abs() - 1e-12 || ((d - best.0.abs()).abs() <= 1e-12 && ortho < best.1) {
                best = (signed, ortho, i, t);
            }
        }
        best
    }

    /// The pseudo-distance: the signed distance, extended along the edge's end tangents beyond its
    /// ends.
    fn pseudo(&self, edge: &Edge, p: [f64; 2], nearest: (f64, f64, usize, f64)) -> f64 {
        let (d, _, index, t) = nearest;
        let piece = &self.pieces[index];
        let extend = |origin: [f64; 2], dir: [f64; 2], before: bool| -> Option<f64> {
            let op = sub(p, origin);
            let along = dot(op, dir);
            if (before && along < 0.0) || (!before && along > 0.0) {
                let c = cross(op, dir);
                let right = c > 0.0;
                let pd = if right == piece.inside_right { c.abs() } else { -c.abs() };
                if pd.abs() <= d.abs() {
                    return Some(pd);
                }
            }
            None
        };
        if index == edge.pieces.start
            && t <= 0.0
            && edge.starts_at_start
            && let Some(pd) = extend(piece.a, edge.start_dir, true)
        {
            return pd;
        }
        if index + 1 == edge.pieces.end
            && t >= 1.0
            && edge.ends_at_end
            && let Some(pd) = extend(piece.b, edge.end_dir, false)
        {
            return pd;
        }
        d
    }

    /// Renders the field, or `None` for a glyph without ink.
    fn render(&self, range: f64) -> Option<Field> {
        if self.edges.is_empty() || !self.bounds[0].is_finite() {
            return None;
        }
        let pad = (range * 0.5).ceil() + 1.0;
        let x0 = (self.bounds[0].floor() - pad) as i32;
        let y0 = (self.bounds[1].floor() - pad) as i32;
        let x1 = (self.bounds[2].ceil() + pad) as i32;
        let y1 = (self.bounds[3].ceil() + pad) as i32;
        let width = u32::try_from(x1 - x0).ok()?;
        let height = u32::try_from(y1 - y0).ok()?;
        let (w, h) = (width as usize, height as usize);
        // Distances per texel, rows top-down: msd (3) and the true distance.
        let mut msd = vec![[0.0_f64; 3]; w * h];
        let mut sd = vec![0.0_f64; w * h];
        let mut nearest = Vec::with_capacity(self.edges.len());
        for row in 0..h {
            let y = f64::from(y1) - row as f64 - 0.5;
            for col in 0..w {
                let x = f64::from(x0) + col as f64 + 0.5;
                let p = [x, y];
                nearest.clear();
                nearest.extend(self.edges.iter().map(|e| self.edge_distance(e, p)));
                let mut true_d = f64::INFINITY;
                for n in &nearest {
                    true_d = true_d.min(n.0.abs());
                }
                let true_d = if self.inside(p) { true_d } else { -true_d };
                let mut channels = [0.0_f64; 3];
                for (c, bit) in [RED, GREEN, BLUE].into_iter().enumerate() {
                    let mut best: Option<usize> = None;
                    for (i, (e, n)) in self.edges.iter().zip(&nearest).enumerate() {
                        if e.color & bit == 0 {
                            continue;
                        }
                        let better = match best {
                            None => true,
                            Some(b) => {
                                let m = &nearest[b];
                                n.0.abs() < m.0.abs() - 1e-12 || ((n.0.abs() - m.0.abs()).abs() <= 1e-12 && n.1 < m.1)
                            }
                        };
                        if better {
                            best = Some(i);
                        }
                    }
                    channels[c] = match best {
                        Some(i) => self.pseudo(&self.edges[i], p, nearest[i]),
                        None => true_d,
                    };
                }
                let i = row * w + col;
                msd[i] = channels;
                sd[i] = true_d;
            }
        }
        // Error correction 1: the median must agree in sign with the true distance.
        for (m, &t) in msd.iter_mut().zip(&sd) {
            let med = median(m[0], m[1], m[2]);
            if (med > 0.0) != (t > 0.0) {
                *m = [t; 3];
            }
        }
        // Normalise to 0..1 (0.5 on the outline).
        let norm: Vec<[f64; 3]> = msd.iter().map(|m| m.map(|v| (v / range + 0.5).clamp(0.0, 1.0))).collect();
        // Error correction 2: clashes between neighbours (Chlumský's detector).
        let threshold = 1.001 / range;
        let mut clash = vec![false; w * h];
        for row in 0..h {
            for col in 0..w {
                let a = norm[row * w + col];
                let mut hit = false;
                if col > 0 {
                    hit |= detect_clash(a, norm[row * w + col - 1], threshold);
                }
                if col + 1 < w {
                    hit |= detect_clash(a, norm[row * w + col + 1], threshold);
                }
                if row > 0 {
                    hit |= detect_clash(a, norm[(row - 1) * w + col], threshold);
                }
                if row + 1 < h {
                    hit |= detect_clash(a, norm[(row + 1) * w + col], threshold);
                }
                clash[row * w + col] = hit;
            }
        }
        let mut rgb = Vec::with_capacity(w * h * 3);
        for (m, &hit) in norm.iter().zip(&clash) {
            let v = if hit {
                let med = median(m[0], m[1], m[2]);
                [med; 3]
            } else {
                *m
            };
            for c in v {
                rgb.push((c * 255.0).round().clamp(0.0, 255.0) as u8);
            }
        }
        Some(Field { width, height, origin_x: x0, origin_y: y0, rgb })
    }
}

/// Keeps a piece when the fill differs on its two sides; returns whether the fill is to the right.
fn classify(segments: &[([f64; 2], [f64; 2])], a: [f64; 2], b: [f64; 2]) -> Option<bool> {
    let d = sub(b, a);
    let len = dot(d, d).sqrt();
    if len < 1e-12 {
        return None;
    }
    let mid = lerp(a, b, 0.5);
    // Right of the direction with Y up.
    let right = [d[1] / len, -d[0] / len];
    let eps = 0.01;
    let inside_right = winding(segments, [mid[0] + right[0] * eps, mid[1] + right[1] * eps]) != 0;
    let inside_left = winding(segments, [mid[0] - right[0] * eps, mid[1] - right[1] * eps]) != 0;
    (inside_right != inside_left).then_some(inside_right)
}

/// The winding number of `p` with respect to closed polylines given as segments.
fn winding(segments: &[([f64; 2], [f64; 2])], p: [f64; 2]) -> i32 {
    let mut w = 0;
    for &(a, b) in segments {
        if a[1] <= p[1] {
            if b[1] > p[1] && cross(sub(b, a), sub(p, a)) > 0.0 {
                w += 1;
            }
        } else if b[1] <= p[1] && cross(sub(b, a), sub(p, a)) < 0.0 {
            w -= 1;
        }
    }
    w
}

/// Assigns channel sets to the edges of a contour (msdfgen's simple colouring).
fn color_contour(contour: &[Raw]) -> Vec<u8> {
    let n = contour.len();
    let mut colors = vec![WHITE; n];
    if n == 0 {
        return colors;
    }
    let sin_limit = CORNER_ANGLE.sin();
    let corners: Vec<usize> = (0..n)
        .filter(|&i| {
            let prev = normalize(contour[(i + n - 1) % n].end_dir());
            let next = normalize(contour[i].start_dir());
            dot(prev, next) <= 0.0 || cross(prev, next).abs() > sin_limit
        })
        .collect();
    let mut seed = 0_u64;
    match corners.len() {
        0 => {}
        1 => {
            // A teardrop: three colours around the contour, starting at the corner.
            let mut c = WHITE;
            switch_color(&mut c, &mut seed, 0);
            let first = c;
            switch_color(&mut c, &mut seed, 0);
            let palette = [first, WHITE, c];
            let corner = corners[0];
            if n >= 3 {
                for i in 0..n {
                    let t = (3.0 + 2.875 * i as f64 / (n - 1) as f64 - 1.4375 + 0.5) as i64 - 3;
                    colors[(corner + i) % n] = palette[(1 + t).clamp(0, 2) as usize];
                }
            } else {
                // One or two edges: colour them distinctly (the corner still separates channels).
                for i in 0..n {
                    colors[(corner + i) % n] = palette[i * 2 % 3];
                }
            }
        }
        count => {
            let start = corners[0];
            let mut spline = 0;
            let mut c = WHITE;
            switch_color(&mut c, &mut seed, 0);
            let initial = c;
            for i in 0..n {
                let index = (start + i) % n;
                if spline + 1 < count && corners[spline + 1] == index {
                    spline += 1;
                    let banned = if spline == count - 1 { initial } else { 0 };
                    switch_color(&mut c, &mut seed, banned);
                }
                colors[index] = c;
            }
        }
    }
    colors
}

/// msdfgen's `switchColor`: the next of cyan, magenta, yellow, avoiding `banned`.
fn switch_color(color: &mut u8, seed: &mut u64, banned: u8) {
    let combined = *color & banned;
    if combined == RED || combined == GREEN || combined == BLUE {
        *color = combined ^ WHITE;
        return;
    }
    if *color == 0 || *color == WHITE {
        const START: [u8; 3] = [CYAN, MAGENTA, YELLOW];
        *color = START[(*seed % 3) as usize];
        *seed /= 3;
        return;
    }
    let shifted = *color << (1 + (*seed & 1));
    *color = (shifted | (shifted >> 3)) & WHITE;
    *seed >>= 1;
}

/// Chlumský's clash detector between neighbouring texels `a` and `b` (normalised values).
fn detect_clash(a: [f64; 3], b: [f64; 3], threshold: f64) -> bool {
    let (mut a0, mut a1, mut a2) = (a[0], a[1], a[2]);
    let (mut b0, mut b1, mut b2) = (b[0], b[1], b[2]);
    if (b1 - a1).abs() < (b2 - a2).abs() {
        std::mem::swap(&mut a1, &mut a2);
        std::mem::swap(&mut b1, &mut b2);
    }
    if (b0 - a0).abs() < (b1 - a1).abs() {
        std::mem::swap(&mut a0, &mut a1);
        std::mem::swap(&mut b0, &mut b1);
        if (b1 - a1).abs() < (b2 - a2).abs() {
            std::mem::swap(&mut a1, &mut a2);
            std::mem::swap(&mut b1, &mut b2);
        }
    }
    (b1 - a1).abs() >= threshold && !(b0 == b1 && b0 == b2) && (a2 - 0.5).abs() >= (b2 - 0.5).abs()
}

fn median(a: f64, b: f64, c: f64) -> f64 {
    a.min(b).max(a.max(b).min(c))
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn cross(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn lerp(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

fn normalize(v: [f64; 2]) -> [f64; 2] {
    let l = dot(v, v).sqrt();
    if l > 1e-18 { [v[0] / l, v[1] / l] } else { [0.0, 0.0] }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(name: &str) -> Vec<u8> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/fonts").join(name);
        std::fs::read(path).unwrap()
    }

    /// The share of texels of each inked glyph whose decoded median agrees with the outline's fill
    /// (non-zero winding at the texel centre), ignoring texels within `margin` texels of the outline.
    fn agreement(ttf: &[u8], baked: &BakedFont, margin: f64) -> f64 {
        let font = Font::new(ttf).unwrap();
        let scale = f64::from(baked.em_px) / f64::from(font.units_per_em());
        let (mut agree, mut total) = (0_u64, 0_u64);
        for g in &baked.glyphs {
            if g.is_blank() {
                continue;
            }
            let c = char::from_u32(g.codepoint).unwrap();
            let gid = if c == FALLBACK {
                font.glyph_index(c).unwrap_or(GlyphId::NOTDEF)
            } else {
                font.glyph_index(c).unwrap()
            };
            let shape = Shape::load(&font, gid, scale).unwrap();
            let page = baked.page_size as usize;
            let [ax, ay, w, h] = g.atlas.map(usize::from);
            let left = f64::from(g.plane[0]) * f64::from(baked.em_px);
            let top = f64::from(g.plane[3]) * f64::from(baked.em_px);
            for row in 0..h {
                for col in 0..w {
                    let p = [left + col as f64 + 0.5, top - row as f64 - 0.5];
                    let near = shape.pieces.iter().any(|pc| {
                        let ab = sub(pc.b, pc.a);
                        let t = (dot(sub(p, pc.a), ab) / dot(ab, ab).max(1e-24)).clamp(0.0, 1.0);
                        let q = lerp(pc.a, pc.b, t);
                        dot(sub(p, q), sub(p, q)).sqrt() < margin
                    });
                    if near {
                        continue;
                    }
                    let i = ((usize::from(g.page) * page + ay + row) * page + ax + col) * 3;
                    let px = &baked.atlas[i..i + 3];
                    let med = median(f64::from(px[0]), f64::from(px[1]), f64::from(px[2])) / 255.0;
                    total += 1;
                    if (med > 0.5) == shape.inside(p) {
                        agree += 1;
                    }
                }
            }
        }
        agree as f64 / total.max(1) as f64
    }

    #[test]
    fn ui_03_bakes_inter_with_metrics_kerning_and_a_faithful_field() {
        let ttf = read("Inter-Regular.ttf");
        let chars: Vec<char> = "AVTo@ie,g ".chars().collect();
        let baked = bake(&ttf, &chars, &BakeOptions::TEXT).unwrap();
        let font = Font::new(&ttf).unwrap();
        let upem = f32::from(font.units_per_em());
        assert_eq!(baked.glyphs.len(), chars.len() + 1, "the characters plus the fallback");
        assert!(baked.has_glyph(FALLBACK));
        for &c in &chars {
            let g = baked.glyph(baked.glyph_index(c));
            let gid = font.glyph_index(c).unwrap();
            assert_eq!(g.codepoint, c as u32);
            assert!((g.advance - f32::from(font.advance(gid)) / upem).abs() < 1e-6);
            assert_eq!(g.is_blank(), c == ' ');
        }
        // [AC-WP12-3] Kerning as the font defines it.
        let (a, v) = (baked.glyph_index('A'), baked.glyph_index('V'));
        let expect = f32::from(font.kerning(font.glyph_index('A').unwrap(), font.glyph_index('V').unwrap())) / upem;
        assert!(expect < 0.0, "Inter kerns AV");
        assert!((baked.kerning(a, v) - expect).abs() < 1e-6);
        assert!(baked.metrics.ascender > 0.8 && baked.metrics.descender < -0.15);
        // The field reproduces the outline away from the edge.
        let share = agreement(&ttf, &baked, 1.0);
        assert!(share > 0.999, "{share}");
        // And survives the file format.
        let back = BakedFont::from_bytes(&baked.to_bytes()).unwrap();
        assert_eq!(back, baked);
    }

    #[test]
    fn ui_03_overlapping_icon_strokes_bake_without_seams() {
        // Lucide glyphs are unions of overlapping stroke outlines: the interior pieces are dropped,
        // so the inside of every stroke reads as inside.
        let ttf = read("Lucide.ttf");
        let font = Font::new(&ttf).unwrap();
        let chars: Vec<char> = all_mapped(&font).into_iter().step_by(97).take(12).collect();
        assert!(chars.len() >= 10);
        let baked = bake(&ttf, &chars, &BakeOptions::ICONS).unwrap();
        let share = agreement(&ttf, &baked, 0.75);
        assert!(share > 0.995, "{share}");
    }

    #[test]
    fn options_and_bad_fonts_are_refused() {
        let ttf = read("Inter-Regular.ttf");
        let bad = BakeOptions { em_px: 0.5, ..BakeOptions::TEXT };
        assert!(matches!(bake(&ttf, &['a'], &bad), Err(UiError::Bake(_))));
        let bad = BakeOptions { range_px: f32::NAN, ..BakeOptions::TEXT };
        assert!(bake(&ttf, &['a'], &bad).is_err());
        let bad = BakeOptions { page_size: 8, ..BakeOptions::TEXT };
        assert!(bake(&ttf, &['a'], &bad).is_err());
        // A glyph larger than a page.
        let big = BakeOptions { em_px: 200.0, page_size: 64, ..BakeOptions::TEXT };
        assert!(bake(&ttf, &['W'], &big).is_err());
        assert!(matches!(bake(&ttf[..100], &['a'], &BakeOptions::TEXT), Err(UiError::Io(_))));
        // Characters the font lacks are skipped; the fallback is always there.
        let baked = bake(&ttf, &['\u{10FFFD}'], &BakeOptions::TEXT).unwrap();
        assert_eq!(baked.glyphs.len(), 1);
        assert_eq!(baked.glyph(baked.glyph_index('\u{10FFFD}')).codepoint, FALLBACK as u32);
    }

    #[test]
    fn colouring_separates_corners() {
        // A square: four corners, neighbouring edges never share all channels.
        let p = |x: f64, y: f64| [x, y];
        let square = [
            Raw::Line(p(0.0, 0.0), p(0.0, 10.0)),
            Raw::Line(p(0.0, 10.0), p(10.0, 10.0)),
            Raw::Line(p(10.0, 10.0), p(10.0, 0.0)),
            Raw::Line(p(10.0, 0.0), p(0.0, 0.0)),
        ];
        let colors = color_contour(&square);
        for i in 0..4 {
            let (a, b) = (colors[i], colors[(i + 1) % 4]);
            assert_ne!(a, b);
            assert_eq!((a & b).count_ones(), 1, "neighbours share exactly one channel");
        }
        // A smooth contour (no corners) is white.
        let circle =
            [Raw::Quad(p(0.0, 0.0), p(5.0, 5.0), p(10.0, 0.0)), Raw::Quad(p(10.0, 0.0), p(5.0, -5.0), p(0.0, 0.0))];
        let c = color_contour(&circle);
        assert!(c.iter().all(|&x| x != 0));
        assert!(latin_charset().contains(&'é') && latin_charset().contains(&'€'));
    }
}
