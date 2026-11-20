//! Glyph outlines from `loca` + `glyf`: simple glyphs, composite glyphs, and the conversion of
//! TrueType's point lists to move / line / quadratic commands.

use super::path::{OutlineSink, Path};
use super::{BBox, FORMAT, Font, GlyphId, invalid};
use crate::bytes::Reader;
use crate::error::{Error, Result};

/// The deepest nesting of composite glyphs that is followed, whatever `maxp` says. Real fonts use
/// one or two levels.
pub(super) const MAX_COMPOSITE_DEPTH: usize = 8;
/// The most components one outline may be assembled from, counting every nesting level.
pub(super) const MAX_COMPONENTS: u32 = 1024;

// Simple glyph flags.
const ON_CURVE: u8 = 0x01;
const X_SHORT: u8 = 0x02;
const Y_SHORT: u8 = 0x04;
const REPEAT: u8 = 0x08;
const X_SAME_OR_POSITIVE: u8 = 0x10;
const Y_SAME_OR_POSITIVE: u8 = 0x20;

// Composite glyph flags.
const ARGS_ARE_WORDS: u16 = 0x0001;
const ARGS_ARE_XY_VALUES: u16 = 0x0002;
const HAVE_SCALE: u16 = 0x0008;
const MORE_COMPONENTS: u16 = 0x0020;
const HAVE_XY_SCALE: u16 = 0x0040;
const HAVE_TWO_BY_TWO: u16 = 0x0080;
const USE_MY_METRICS: u16 = 0x0200;
const SCALED_COMPONENT_OFFSET: u16 = 0x0800;
const UNSCALED_COMPONENT_OFFSET: u16 = 0x1000;

/// How large the outline of a glyph is, as stored in the font.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OutlineSize {
    /// The number of points (on-curve and off-curve), every component of a composite included.
    pub points: usize,
    /// The number of contours, every component included.
    pub contours: usize,
    /// The number of component references followed, at every nesting level; 0 for a simple glyph.
    pub components: u32,
}

/// One outline point in font units.
#[derive(Clone, Copy, Debug)]
struct Point {
    x: f32,
    y: f32,
    on_curve: bool,
}

/// The points of a glyph being assembled, contour after contour.
#[derive(Default)]
struct Buffer {
    points: Vec<Point>,
    /// For each contour, the index one past its last point.
    contour_ends: Vec<usize>,
    /// Scratch for the flags of the simple glyph being decoded.
    flags: Vec<u8>,
    /// How many components have been loaded so far.
    components: u32,
    /// The glyphs on the path from the requested glyph to the one being loaded.
    stack: [u16; MAX_COMPOSITE_DEPTH + 1],
}

/// One component record of a composite glyph.
struct Component {
    flags: u16,
    glyph: u16,
    /// The two arguments, sign-extended when they are offsets, zero-extended when point numbers.
    args: (u16, u16),
    /// The transform as `(xx, yx, xy, yy)`: x' = xx·x + xy·y, y' = yx·x + yy·y.
    transform: (f32, f32, f32, f32),
}

impl Component {
    /// Reads the record at the reader's position.
    fn read(r: &mut Reader<'_>) -> Result<Self> {
        let flags = r.u16_be()?;
        let glyph = r.u16_be()?;
        let args = if flags & ARGS_ARE_WORDS != 0 {
            (r.u16_be()?, r.u16_be()?)
        } else if flags & ARGS_ARE_XY_VALUES != 0 {
            // Sign-extend the bytes so both widths read as signed values later.
            (r.i8()? as u16, r.i8()? as u16)
        } else {
            (u16::from(r.u8()?), u16::from(r.u8()?))
        };
        let mut f2dot14 = || r.i16_be().map(|v| f32::from(v) / 16384.0);
        let transform = if flags & HAVE_SCALE != 0 {
            let scale = f2dot14()?;
            (scale, 0.0, 0.0, scale)
        } else if flags & HAVE_XY_SCALE != 0 {
            (f2dot14()?, 0.0, 0.0, f2dot14()?)
        } else if flags & HAVE_TWO_BY_TWO != 0 {
            // Stored as xscale, scale01, scale10, yscale.
            (f2dot14()?, f2dot14()?, f2dot14()?, f2dot14()?)
        } else {
            (1.0, 0.0, 0.0, 1.0)
        };
        Ok(Self { flags, glyph, args, transform })
    }
}

/// The error for an outline that is larger than the font's `maxp` table (or this parser) allows.
fn over_limit(what: &'static str, limit: usize, got: usize) -> Error {
    Error::Limit { format: FORMAT, what, limit: limit as u64, got: got as u64 }
}

/// Moves the offset of an error found in glyph data at file offset `at` from glyph-relative to
/// file-relative.
fn located(at: usize) -> impl Fn(Error) -> Error {
    move |e| match e {
        Error::Truncated { format, offset } => Error::Truncated { format, offset: at + offset },
        Error::Invalid { format, offset, what } => Error::Invalid { format, offset: at + offset, what },
        other => other,
    }
}

impl<'a> Font<'a> {
    /// The bytes of a glyph in `glyf` and their offset in the file; `None` for a glyph without an
    /// outline (a space).
    fn glyph_data(&self, glyph: u16) -> Result<Option<(&'a [u8], usize)>> {
        if glyph >= self.metrics.num_glyphs {
            return Err(invalid(self.loca.base, "glyph id out of range"));
        }
        let index = usize::from(glyph);
        let (at, start, end) = if self.long_loca {
            (index * 4, self.loca.need_u32(index * 4)? as usize, self.loca.need_u32(index * 4 + 4)? as usize)
        } else {
            let (start, end) = (self.loca.need_u16(index * 2)?, self.loca.need_u16(index * 2 + 2)?);
            (index * 2, usize::from(start) * 2, usize::from(end) * 2)
        };
        if start == end {
            return Ok(None);
        }
        match self.glyf.data.get(start..end) {
            Some(data) => Ok(Some((data, self.glyf.base + start))),
            None => Err(invalid(self.loca.base + at, "glyph lies outside the glyf table")),
        }
    }

    /// The bounding box stored with a glyph's outline, or `None` for a glyph without one (a space, a
    /// glyph id outside the font, or damaged glyph data).
    #[must_use]
    pub fn glyph_bbox(&self, glyph: GlyphId) -> Option<BBox> {
        let (data, _) = self.glyph_data(glyph.0).ok()??;
        let mut r = Reader::new(data, FORMAT);
        r.skip(2).ok()?;
        Some(BBox {
            x_min: r.i16_be().ok()?,
            y_min: r.i16_be().ok()?,
            x_max: r.i16_be().ok()?,
            y_max: r.i16_be().ok()?,
        })
    }

    /// The right side bearing of a glyph: what is left of its `hmtx` advance to the right of its
    /// outline, `advance − left side bearing − (x_max − x_min)` (negative when the outline overhangs
    /// the next pen position). For a glyph without an outline this is the advance minus the bearing.
    #[must_use]
    pub fn right_side_bearing(&self, glyph: GlyphId) -> i32 {
        let width = self.glyph_bbox(glyph).map_or(0, |b| i32::from(b.x_max) - i32::from(b.x_min));
        i32::from(self.advance(glyph)) - i32::from(self.left_side_bearing(glyph)) - width
    }

    /// Whether the glyph is assembled from other glyphs. `false` for a simple glyph, a glyph without
    /// an outline and a glyph id outside the font.
    #[must_use]
    pub fn is_composite(&self, glyph: GlyphId) -> bool {
        matches!(self.glyph_data(glyph.0), Ok(Some((data, _))) if data.first().is_some_and(|&b| b & 0x80 != 0))
    }

    /// Sends the outline of a glyph to `sink`, in font units with Y up.
    ///
    /// Each contour arrives as one `move_to`, then `line_to` / `quad_to` segments, then `close`. The
    /// points TrueType leaves implicit are resolved: two consecutive off-curve points have an
    /// on-curve point midway between them, and a contour that starts off-curve starts at its last
    /// point if that is on-curve, else at the midpoint of its last and first points. The closing
    /// segment back to the start is a `quad_to` when the contour ends off-curve and is otherwise left
    /// to `close`. Contours of a single point (anchors) are dropped. As the format prescribes, outer
    /// contours run clockwise (negative [`signed_area`](super::signed_area)) and holes
    /// counter-clockwise, unless a composite mirrors a component.
    ///
    /// Composite glyphs are assembled from their components with every transform form (uniform scale,
    /// separate x/y scale, 2×2 matrix), offsets given as values (scaled by the component transform
    /// only when the font asks for that with `SCALED_COMPONENT_OFFSET`) or by matching a point of the
    /// component to a point of the glyph so far. Coordinates are not rounded (`ROUND_XY_TO_GRID`
    /// concerns hinted rendering). `USE_MY_METRICS` does not change the outline; it is honoured by
    /// [`Font::h_metrics`].
    ///
    /// A glyph without an outline (a space) sends nothing and succeeds. Errors: a glyph id outside the
    /// font, damaged glyph data, a composite that contains itself, an outline over the limits the
    /// font declares in `maxp` (see [`MaxProfile`](super::MaxProfile): points and contours of a
    /// simple glyph, total points and contours of a composite, components per composite, nesting
    /// depth) and one over this parser's own limits of 8 nesting levels and 1024 components in
    /// total. On an error nothing is sent.
    pub fn outline(&self, glyph: GlyphId, sink: &mut impl OutlineSink) -> Result<()> {
        let mut buffer = Buffer::default();
        self.load_glyph(glyph.0, 0, &mut buffer)?;
        emit(&buffer, sink);
        Ok(())
    }

    /// The outline of a glyph as an owned [`Path`]; see [`Font::outline`].
    pub fn outline_to_vec(&self, glyph: GlyphId) -> Result<Path> {
        let mut path = Path::default();
        self.outline(glyph, &mut path)?;
        Ok(path)
    }

    /// The number of points, contours and components of a glyph's outline as the font stores it
    /// (before implied points are resolved), with the same errors as [`Font::outline`].
    pub fn outline_size(&self, glyph: GlyphId) -> Result<OutlineSize> {
        let mut buffer = Buffer::default();
        self.load_glyph(glyph.0, 0, &mut buffer)?;
        Ok(OutlineSize {
            points: buffer.points.len(),
            contours: buffer.contour_ends.len(),
            components: buffer.components,
        })
    }

    /// The glyph whose `hmtx` metrics apply to `glyph`: the glyph itself, or, for a composite with a
    /// component flagged `USE_MY_METRICS`, that component (the last one flagged, as rasterisers do),
    /// followed through at most 8 levels of nested composites. Damaged glyph data ends the search at
    /// the glyph reached so far.
    #[must_use]
    pub fn metrics_source(&self, glyph: GlyphId) -> GlyphId {
        let mut current = glyph.0;
        for _ in 0..MAX_COMPOSITE_DEPTH {
            match self.metrics_component(current) {
                Some(next) if next != current && next < self.metrics.num_glyphs => current = next,
                _ => break,
            }
        }
        GlyphId(current)
    }

    /// The last component of a composite glyph that carries `USE_MY_METRICS`.
    fn metrics_component(&self, glyph: u16) -> Option<u16> {
        let (data, _) = self.glyph_data(glyph).ok()??;
        let mut r = Reader::new(data, FORMAT);
        if r.i16_be().ok()? >= 0 {
            return None;
        }
        r.skip(8).ok()?;
        let mut found = None;
        // Every record consumes at least six bytes, so the loop ends with the glyph data.
        while let Ok(component) = Component::read(&mut r) {
            if component.flags & USE_MY_METRICS != 0 {
                found = Some(component.glyph);
            }
            if component.flags & MORE_COMPONENTS == 0 {
                break;
            }
        }
        found
    }

    /// Appends the points of `glyph`, in its own coordinates, to the buffer.
    fn load_glyph(&self, glyph: u16, depth: usize, buffer: &mut Buffer) -> Result<()> {
        let Some((data, at)) = self.glyph_data(glyph)? else {
            return Ok(());
        };
        buffer.stack[depth] = glyph;
        let mut r = Reader::new(data, FORMAT);
        let contours = r.i16_be().map_err(located(at))?;
        r.skip(8).map_err(located(at))?;
        if contours >= 0 {
            self.load_simple(&mut r, contours as usize, depth > 0, buffer).map_err(located(at))
        } else {
            self.load_composite(&mut r, at, depth, buffer)
        }
    }

    /// Decodes a simple glyph: contour end points, flags with repeats, then x and y deltas.
    fn load_simple(&self, r: &mut Reader<'_>, contours: usize, component: bool, buffer: &mut Buffer) -> Result<()> {
        if contours == 0 {
            return Ok(());
        }
        let max = &self.max;
        if contours > usize::from(max.max_contours) {
            return Err(over_limit("glyph contour count (maxp maxContours)", usize::from(max.max_contours), contours));
        }
        let total_contours = buffer.contour_ends.len() + contours;
        if component && total_contours > usize::from(max.max_composite_contours) {
            return Err(over_limit(
                "composite glyph contour count (maxp maxCompositeContours)",
                usize::from(max.max_composite_contours),
                total_contours,
            ));
        }
        let base = buffer.points.len();
        let mut count = 0_usize;
        for _ in 0..contours {
            let end = usize::from(r.u16_be()?) + 1;
            if end <= count {
                return Err(Error::Invalid {
                    format: FORMAT,
                    offset: r.pos() - 2,
                    what: "contour end points do not increase",
                });
            }
            count = end;
            buffer.contour_ends.push(base + end);
        }
        if count > usize::from(max.max_points) {
            return Err(over_limit("glyph point count (maxp maxPoints)", usize::from(max.max_points), count));
        }
        if component && base + count > usize::from(max.max_composite_points) {
            return Err(over_limit(
                "composite glyph point count (maxp maxCompositePoints)",
                usize::from(max.max_composite_points),
                base + count,
            ));
        }
        let instructions = usize::from(r.u16_be()?);
        r.skip(instructions)?;

        buffer.flags.clear();
        while buffer.flags.len() < count {
            let flag = r.u8()?;
            let repeats = if flag & REPEAT != 0 { usize::from(r.u8()?) } else { 0 };
            // A repeat count that overshoots the point count is cut off, as rasterisers do.
            let run = (1 + repeats).min(count - buffer.flags.len());
            buffer.flags.extend(std::iter::repeat_n(flag, run));
        }

        let mut x = 0_i32;
        for &flag in &buffer.flags {
            x += Self::delta(r, flag, X_SHORT, X_SAME_OR_POSITIVE)?;
            buffer.points.push(Point { x: x as f32, y: 0.0, on_curve: flag & ON_CURVE != 0 });
        }
        let mut y = 0_i32;
        for (point, &flag) in buffer.points[base..].iter_mut().zip(&buffer.flags) {
            y += Self::delta(r, flag, Y_SHORT, Y_SAME_OR_POSITIVE)?;
            point.y = y as f32;
        }
        Ok(())
    }

    /// Reads one coordinate delta: a byte with its sign in the flags, nothing ("same as before"), or a
    /// signed word.
    fn delta(r: &mut Reader<'_>, flag: u8, short: u8, same_or_positive: u8) -> Result<i32> {
        Ok(if flag & short != 0 {
            let magnitude = i32::from(r.u8()?);
            if flag & same_or_positive != 0 { magnitude } else { -magnitude }
        } else if flag & same_or_positive != 0 {
            0
        } else {
            i32::from(r.i16_be()?)
        })
    }

    /// Assembles a composite glyph from its components.
    fn load_composite(&self, r: &mut Reader<'_>, at: usize, depth: usize, buffer: &mut Buffer) -> Result<()> {
        // A composite at nesting level `depth` has components at level `depth + 1`; maxp counts a
        // composite of simple glyphs as depth 1.
        let depth_limit = usize::from(self.max.max_component_depth).min(MAX_COMPOSITE_DEPTH);
        if depth + 1 > depth_limit {
            return Err(over_limit("composite glyph nesting (maxp maxComponentDepth)", depth_limit, depth + 1));
        }
        let base = buffer.points.len();
        let mut elements = 0_usize;
        loop {
            let record = at + r.pos();
            let component = Component::read(r).map_err(located(at))?;
            let flags = component.flags;
            let (xx, yx, xy, yy) = component.transform;

            elements += 1;
            if elements > usize::from(self.max.max_component_elements) {
                return Err(over_limit(
                    "composite glyph component count (maxp maxComponentElements)",
                    usize::from(self.max.max_component_elements),
                    elements,
                ));
            }
            buffer.components += 1;
            if buffer.components > MAX_COMPONENTS {
                return Err(over_limit(
                    "composite glyph total component count",
                    MAX_COMPONENTS as usize,
                    buffer.components as usize,
                ));
            }
            if buffer.stack[..=depth].contains(&component.glyph) {
                return Err(invalid(record, "composite glyph contains itself"));
            }
            let start = buffer.points.len();
            self.load_glyph(component.glyph, depth + 1, buffer)?;

            if (xx, yx, xy, yy) != (1.0, 0.0, 0.0, 1.0) {
                for p in &mut buffer.points[start..] {
                    (p.x, p.y) = (xx * p.x + xy * p.y, yx * p.x + yy * p.y);
                }
            }
            let (dx, dy) = if flags & ARGS_ARE_XY_VALUES != 0 {
                let (mut dx, mut dy) = (f32::from(component.args.0 as i16), f32::from(component.args.1 as i16));
                if flags & SCALED_COMPONENT_OFFSET != 0 && flags & UNSCALED_COMPONENT_OFFSET == 0 {
                    dx *= xx.hypot(xy);
                    dy *= yy.hypot(yx);
                }
                (dx, dy)
            } else {
                // Point matching: a point of the glyph assembled so far and a point of this
                // component (after its transform) are brought together.
                let parent = buffer.points[base..start].get(usize::from(component.args.0));
                let child = buffer.points[start..].get(usize::from(component.args.1));
                let (Some(parent), Some(child)) = (parent, child) else {
                    return Err(invalid(record + 4, "composite point index out of range"));
                };
                (parent.x - child.x, parent.y - child.y)
            };
            if dx != 0.0 || dy != 0.0 {
                for p in &mut buffer.points[start..] {
                    p.x += dx;
                    p.y += dy;
                }
            }
            if flags & MORE_COMPONENTS == 0 {
                break;
            }
        }
        Ok(())
    }
}

/// Converts the assembled points to path commands.
fn emit(buffer: &Buffer, sink: &mut impl OutlineSink) {
    let mut first = 0;
    for &end in &buffer.contour_ends {
        let contour = buffer.points.get(first..end).unwrap_or(&[]);
        first = end;
        let (Some(head), Some(tail)) = (contour.first(), contour.last()) else { continue };
        if contour.len() < 2 {
            continue;
        }
        // Where the contour starts, and which of its points remain to be visited.
        let (start, rest) = if head.on_curve {
            ([head.x, head.y], &contour[1..])
        } else if tail.on_curve {
            ([tail.x, tail.y], &contour[..contour.len() - 1])
        } else {
            ([f32::midpoint(head.x, tail.x), f32::midpoint(head.y, tail.y)], contour)
        };
        sink.move_to(start);
        let mut control: Option<[f32; 2]> = None;
        for p in rest {
            let here = [p.x, p.y];
            match (control, p.on_curve) {
                (Some(c), true) => {
                    sink.quad_to(c, here);
                    control = None;
                }
                (None, true) => sink.line_to(here),
                (Some(c), false) => {
                    sink.quad_to(c, [f32::midpoint(c[0], here[0]), f32::midpoint(c[1], here[1])]);
                    control = Some(here);
                }
                (None, false) => control = Some(here),
            }
        }
        if let Some(c) = control {
            sink.quad_to(c, start);
        }
        sink.close();
    }
}
