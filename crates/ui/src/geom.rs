//! Geometry in logical pixels: points, sizes, rectangles and insets.
//!
//! The UI works in logical pixels with the origin at the top-left corner of the window and Y down.
//! The painter multiplies by the DPI scale (display scale × UI scale) when it records instances, and
//! [`Rect::snap`] rounds edges to whole physical pixels so 1 px borders stay crisp `[UI-04]`.

/// A point (or a 2D offset) in logical pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    /// Distance from the left edge.
    pub x: f32,
    /// Distance from the top edge.
    pub y: f32,
}

impl Point {
    /// The origin.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// A point.
    #[inline]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// The point a fraction `t` of the way to `other`.
    #[inline]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self::new(self.x + (other.x - self.x) * t, self.y + (other.y - self.y) * t)
    }

    /// The Euclidean distance to `other`.
    #[inline]
    pub fn distance(self, other: Self) -> f32 {
        let d = self - other;
        (d.x * d.x + d.y * d.y).sqrt()
    }
}

impl std::ops::Add for Point {
    type Output = Self;

    /// The component-wise sum.
    #[inline]
    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }
}

impl std::ops::Sub for Point {
    type Output = Self;

    /// The component-wise difference.
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }
}

/// A width and a height in logical pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Size {
    /// Width.
    pub w: f32,
    /// Height.
    pub h: f32,
}

impl Size {
    /// No extent.
    pub const ZERO: Self = Self { w: 0.0, h: 0.0 };

    /// A size.
    #[inline]
    pub const fn new(w: f32, h: f32) -> Self {
        Self { w, h }
    }
}

/// Distances inward from the four edges of a rectangle (padding, margins).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Insets {
    /// From the left edge.
    pub left: f32,
    /// From the top edge.
    pub top: f32,
    /// From the right edge.
    pub right: f32,
    /// From the bottom edge.
    pub bottom: f32,
}

impl Insets {
    /// No inset.
    pub const ZERO: Self = Self { left: 0.0, top: 0.0, right: 0.0, bottom: 0.0 };

    /// The same inset on every edge.
    #[inline]
    pub const fn all(v: f32) -> Self {
        Self { left: v, top: v, right: v, bottom: v }
    }

    /// `x` on the left and right, `y` on the top and bottom.
    #[inline]
    pub const fn xy(x: f32, y: f32) -> Self {
        Self { left: x, top: y, right: x, bottom: y }
    }

    /// Left plus right.
    #[inline]
    pub fn horizontal(self) -> f32 {
        self.left + self.right
    }

    /// Top plus bottom.
    #[inline]
    pub fn vertical(self) -> f32 {
        self.top + self.bottom
    }
}

/// An axis-aligned rectangle in logical pixels: its top-left corner and its size.
///
/// A rectangle with a zero or negative width or height is empty; operations never produce a
/// negative size.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rect {
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width.
    pub w: f32,
    /// Height.
    pub h: f32,
}

impl Rect {
    /// The empty rectangle at the origin.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, w: 0.0, h: 0.0 };

    /// A rectangle that contains every finite point (the "no clip" clip rectangle).
    pub const EVERYTHING: Self = Self { x: -1.0e7, y: -1.0e7, w: 2.0e7, h: 2.0e7 };

    /// A rectangle from its corner and size.
    #[inline]
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    /// The rectangle between two corners (in any order).
    #[inline]
    pub fn from_corners(a: Point, b: Point) -> Self {
        let (x0, x1) = if a.x <= b.x { (a.x, b.x) } else { (b.x, a.x) };
        let (y0, y1) = if a.y <= b.y { (a.y, b.y) } else { (b.y, a.y) };
        Self::new(x0, y0, x1 - x0, y1 - y0)
    }

    /// A rectangle of `size` centred on `center`.
    #[inline]
    pub fn centered(center: Point, size: Size) -> Self {
        Self::new(center.x - size.w * 0.5, center.y - size.h * 0.5, size.w, size.h)
    }

    /// The right edge.
    #[inline]
    pub fn right(self) -> f32 {
        self.x + self.w
    }

    /// The bottom edge.
    #[inline]
    pub fn bottom(self) -> f32 {
        self.y + self.h
    }

    /// The top-left corner.
    #[inline]
    pub fn origin(self) -> Point {
        Point::new(self.x, self.y)
    }

    /// The size.
    #[inline]
    pub fn size(self) -> Size {
        Size::new(self.w, self.h)
    }

    /// The centre.
    #[inline]
    pub fn center(self) -> Point {
        Point::new(self.x + self.w * 0.5, self.y + self.h * 0.5)
    }

    /// Whether the width or the height is zero or negative (or NaN).
    #[inline]
    pub fn is_empty(self) -> bool {
        !(self.w > 0.0 && self.h > 0.0)
    }

    /// Whether `p` lies inside (the left and top edges are inside, the right and bottom are not).
    #[inline]
    pub fn contains(self, p: Point) -> bool {
        p.x >= self.x && p.x < self.right() && p.y >= self.y && p.y < self.bottom()
    }

    /// Whether the two rectangles share an area larger than zero.
    #[inline]
    pub fn intersects(self, other: Self) -> bool {
        self.x < other.right() && other.x < self.right() && self.y < other.bottom() && other.y < self.bottom()
    }

    /// The overlap of two rectangles; empty (zero size, never negative) when they are disjoint.
    #[inline]
    #[must_use]
    pub fn intersect(self, other: Self) -> Self {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        Self::new(
            x,
            y,
            (self.right().min(other.right()) - x).max(0.0),
            (self.bottom().min(other.bottom()) - y).max(0.0),
        )
    }

    /// The smallest rectangle containing both.
    #[inline]
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        Self::new(x, y, self.right().max(other.right()) - x, self.bottom().max(other.bottom()) - y)
    }

    /// The rectangle moved by `offset`.
    #[inline]
    #[must_use]
    pub fn translate(self, offset: Point) -> Self {
        Self::new(self.x + offset.x, self.y + offset.y, self.w, self.h)
    }

    /// The rectangle with `insets` taken off its edges; the size never goes below zero.
    #[inline]
    #[must_use]
    pub fn inset(self, insets: Insets) -> Self {
        Self::new(
            self.x + insets.left,
            self.y + insets.top,
            (self.w - insets.horizontal()).max(0.0),
            (self.h - insets.vertical()).max(0.0),
        )
    }

    /// The rectangle shrunk by `d` on every edge (grown when `d` is negative), never below zero size.
    #[inline]
    #[must_use]
    pub fn shrink(self, d: f32) -> Self {
        self.inset(Insets::all(d))
    }

    /// The rectangle grown by `d` on every edge.
    #[inline]
    #[must_use]
    pub fn expand(self, d: f32) -> Self {
        self.shrink(-d)
    }

    /// The rectangle with its edges rounded to whole physical pixels at `scale` physical pixels per
    /// logical pixel `[UI-04]`. Edges are rounded independently, so two rectangles that share an edge
    /// still share it after snapping.
    #[inline]
    #[must_use]
    pub fn snap(self, scale: f32) -> Self {
        if scale <= 0.0 || !scale.is_finite() {
            return self;
        }
        let x0 = (self.x * scale).round() / scale;
        let y0 = (self.y * scale).round() / scale;
        let x1 = (self.right() * scale).round() / scale;
        let y1 = (self.bottom() * scale).round() / scale;
        Self::new(x0, y0, (x1 - x0).max(0.0), (y1 - y0).max(0.0))
    }

    /// Splits off a strip of `amount` from the left; returns `(strip, rest)`.
    #[inline]
    pub fn split_left(self, amount: f32) -> (Self, Self) {
        let a = amount.clamp(0.0, self.w.max(0.0));
        (Self::new(self.x, self.y, a, self.h), Self::new(self.x + a, self.y, (self.w - a).max(0.0), self.h))
    }

    /// Splits off a strip of `amount` from the top; returns `(strip, rest)`.
    #[inline]
    pub fn split_top(self, amount: f32) -> (Self, Self) {
        let a = amount.clamp(0.0, self.h.max(0.0));
        (Self::new(self.x, self.y, self.w, a), Self::new(self.x, self.y + a, self.w, (self.h - a).max(0.0)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersect_never_negative() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        let b = Rect::new(20.0, 20.0, 5.0, 5.0);
        let c = a.intersect(b);
        assert!(c.w >= 0.0 && c.h >= 0.0 && c.is_empty());
        assert!(!a.intersects(b));
        assert_eq!(a.intersect(Rect::new(5.0, 5.0, 10.0, 10.0)), Rect::new(5.0, 5.0, 5.0, 5.0));
    }

    #[test]
    fn contains_is_half_open() {
        let r = Rect::new(1.0, 1.0, 2.0, 2.0);
        assert!(r.contains(Point::new(1.0, 1.0)));
        assert!(!r.contains(Point::new(3.0, 2.0)));
        assert!(!r.contains(Point::new(f32::NAN, 2.0)));
    }

    #[test]
    fn snap_keeps_shared_edges() {
        // [UI-04] Two neighbours meeting at a fractional edge still meet after snapping.
        let a = Rect::new(0.3, 0.0, 10.4, 5.0);
        let b = Rect::new(10.7, 0.0, 3.0, 5.0);
        for scale in [1.0, 1.25, 1.5, 2.0] {
            let (sa, sb) = (a.snap(scale), b.snap(scale));
            assert!((sa.right() - sb.x).abs() < 1e-5, "scale {scale}");
            assert!(((sa.x * scale) - (sa.x * scale).round()).abs() < 1e-4);
        }
    }

    #[test]
    fn inset_and_split_clamp() {
        let r = Rect::new(0.0, 0.0, 10.0, 4.0);
        assert_eq!(r.shrink(3.0).h, 0.0);
        let (left, rest) = r.split_left(14.0);
        assert_eq!((left.w, rest.w), (10.0, 0.0));
        let (top, rest) = r.split_top(1.0);
        assert_eq!((top.h, rest.y, rest.h), (1.0, 1.0, 3.0));
        assert_eq!(Rect::from_corners(Point::new(3.0, 4.0), Point::new(1.0, 1.0)), Rect::new(1.0, 1.0, 2.0, 3.0));
        assert_eq!(r.union(Rect::new(-1.0, 2.0, 1.0, 5.0)), Rect::new(-1.0, 0.0, 11.0, 7.0));
    }
}
