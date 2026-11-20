//! Small painting and formatting helpers shared by the widget modules: one call per common
//! primitive so widget code reads as what it draws. Colours, radii and styles always come from the
//! caller (theme tokens); nothing here allocates.

use crate::color::Color;
use crate::layout::NodeId;
use crate::paint::Corners;
use crate::text::Align;
use crate::theme::{Shadow, TextStyle};
use crate::ui::{Anchor, Area, Paint, Ui};

/// A filled rounded rectangle.
pub(crate) fn fill(ui: &mut Ui, node: NodeId, area: Area, color: Color, radius: f32) {
    if color.a <= 0.0 {
        return;
    }
    ui.paint(node, Paint::Rect { area, fill: color, radius: Corners::all(radius), border: 0.0, border_color: color });
}

/// A filled rounded rectangle with a border of the theme's border width.
pub(crate) fn frame(ui: &mut Ui, node: NodeId, area: Area, fill: Color, radius: f32, border: Color) {
    let width = ui.theme().metrics.border;
    ui.paint(node, Paint::Rect { area, fill, radius: Corners::all(radius), border: width, border_color: border });
}

/// One line of text in `area`, centred vertically.
pub(crate) fn text(ui: &mut Ui, node: NodeId, area: Area, s: &str, style: TextStyle, color: Color, align: Align) {
    if s.is_empty() {
        return;
    }
    let t = ui.intern(s);
    ui.paint(node, Paint::Text { area, text: t, style, color, align, wrap: false });
}

/// Text wrapped to the width of `area`, from its top.
pub(crate) fn wrapped(ui: &mut Ui, node: NodeId, area: Area, s: &str, style: TextStyle, color: Color) {
    if s.is_empty() {
        return;
    }
    let t = ui.intern(s);
    ui.paint(node, Paint::Text { area, text: t, style, color, align: Align::Start, wrap: true });
}

/// An icon of the icon face centred on (`cx`, `cy`) of the box, `size` logical pixels.
pub(crate) fn icon(ui: &mut Ui, node: NodeId, cx: Anchor, cy: Anchor, glyph: char, size: f32, color: Color) {
    let mut buf = [0_u8; 4];
    let t = ui.intern(glyph.encode_utf8(&mut buf));
    let h = size * 0.5 + 2.0;
    let area = Area {
        x0: Anchor { frac: cx.frac, px: cx.px - h },
        y0: Anchor { frac: cy.frac, px: cy.px - h },
        x1: Anchor { frac: cx.frac, px: cx.px + h },
        y1: Anchor { frac: cy.frac, px: cy.px + h },
    };
    ui.paint(
        node,
        Paint::Text { area, text: t, style: TextStyle::icon(size), color, align: Align::Center, wrap: false },
    );
}

/// An icon centred in its box.
pub(crate) fn icon_centered(ui: &mut Ui, node: NodeId, glyph: char, size: f32, color: Color) {
    icon(ui, node, Anchor::at(0.5, 0.0), Anchor::at(0.5, 0.0), glyph, size, color);
}

/// A straight line between two anchored points.
pub(crate) fn line(ui: &mut Ui, node: NodeId, a: (Anchor, Anchor), b: (Anchor, Anchor), width: f32, color: Color) {
    ui.paint(node, Paint::Line { ax: a.0, ay: a.1, bx: b.0, by: b.1, width, color });
}

/// The theme's focus ring around `area`.
pub(crate) fn ring(ui: &mut Ui, node: NodeId, area: Area, radius: f32) {
    ui.paint(node, Paint::FocusRing { area, radius });
}

/// A floating surface: its shadow (offset down, `strength × appear`) and the bordered card, in
/// the reference's recipe (section 16.4): shadow first, then `fill` with a `border_strong` border.
pub(crate) fn card(ui: &mut Ui, node: NodeId, shadow: Shadow, fill: Color, radius: f32, appear: f32) {
    let p = ui.theme().palette;
    if shadow.strength > 0.0 && p.shadow.a > 0.0 {
        let area = Area {
            x0: Anchor::start(shadow.inset_x),
            y0: Anchor::start(shadow.offset_y),
            x1: Anchor::end(-shadow.inset_x),
            y1: Anchor::end(shadow.offset_y),
        };
        ui.paint(
            node,
            Paint::Shadow {
                area,
                color: p.shadow.fade(shadow.strength * appear),
                radius: shadow.radius,
                blur: shadow.blur,
            },
        );
    }
    frame(ui, node, Area::FULL, fill.fade(appear), radius, p.border_strong.fade(appear));
}

/// A rectangle `w × h` at (`x`, `y`) from the box's top-left corner.
pub(crate) const fn at(x: f32, y: f32, w: f32, h: f32) -> Area {
    Area::offset(x, y, w, h)
}

/// The grow factor for a control that should fill the width of its parent: 1 in a row (where
/// growing is horizontal), 0 in a column (where it would grow the height; columns stretch their
/// children's width already).
pub(crate) fn grow_x(ui: &Ui) -> f32 {
    let parent = ui.parent();
    match ui.layout().style(parent).map(|s| s.direction) {
        Some(crate::layout::Direction::Row) => 1.0,
        _ => 0.0,
    }
}

/// The box's area inset by `x` pixels left and right, full height.
pub(crate) const fn inset_x(left: f32, right: f32) -> Area {
    Area { x0: Anchor::start(left), y0: Anchor::start(0.0), x1: Anchor::end(-right), y1: Anchor::end(0.0) }
}

/// A short text buffer for numbers and labels formatted without allocating.
#[derive(Clone, Copy)]
pub(crate) struct Buf<const N: usize> {
    bytes: [u8; N],
    len: usize,
}

impl<const N: usize> Buf<N> {
    /// An empty buffer.
    pub(crate) const fn new() -> Self {
        Self { bytes: [0; N], len: 0 }
    }

    /// The text written so far.
    pub(crate) fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..self.len]).unwrap_or("")
    }
}

impl<const N: usize> std::fmt::Write for Buf<N> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let room = N - self.len;
        // Cut at a character boundary when the text does not fit.
        let mut take = s.len().min(room);
        while !s.is_char_boundary(take) {
            take -= 1;
        }
        self.bytes[self.len..self.len + take].copy_from_slice(&s.as_bytes()[..take]);
        self.len += take;
        if take < s.len() { Err(std::fmt::Error) } else { Ok(()) }
    }
}

/// `args` formatted into a 64-byte buffer (longer text is cut).
pub(crate) fn fmt64(args: std::fmt::Arguments<'_>) -> Buf<64> {
    let mut b = Buf::new();
    let _ = std::fmt::write(&mut b, args);
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffers_format_without_allocating_and_cut_at_char_boundaries() {
        assert_eq!(fmt64(format_args!("{:.2}", 1.234_f32)).as_str(), "1.23");
        let mut b = Buf::<5>::new();
        let _ = std::fmt::Write::write_str(&mut b, "ab\u{e9}\u{e9}");
        assert_eq!(b.as_str(), "ab\u{e9}");
    }
}
