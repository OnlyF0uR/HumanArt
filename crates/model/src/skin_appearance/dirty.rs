//! Dirty-rectangle tracking per UV tile `[SKN-02]`: what the compositor must re-resolve.
//!
//! Every tile keeps at most [`MAX_RECTS_PER_TILE`] rectangles. A new rectangle is merged with an
//! existing one when they overlap or touch and the union wastes little area; when the tile's list
//! is full, the pair whose union grows least is merged. Rectangles therefore always cover every
//! texel marked, and a brush stroke stays a few small rectangles instead of growing into the
//! whole tile. All storage is allocated at creation; marking and draining allocate nothing
//! `[PERF-R1]`.

use super::error::SkinError;

/// Most rectangles kept per tile.
pub const MAX_RECTS_PER_TILE: usize = 16;
/// Most tiles of one texture set.
pub const MAX_TILES: u32 = 64;
/// Largest tile side in texels.
pub const MAX_TILE_SIZE: u32 = 16384;

/// A texel rectangle inside one tile: `x..x + w`, `y..y + h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Rect {
    /// Left texel.
    pub x: u32,
    /// Top texel.
    pub y: u32,
    /// Width in texels.
    pub w: u32,
    /// Height in texels.
    pub h: u32,
}

impl Rect {
    /// A rectangle.
    pub const fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }

    /// The texel count.
    pub const fn area(self) -> u64 {
        self.w as u64 * self.h as u64
    }

    /// Whether the rectangle covers no texel.
    pub const fn is_empty(self) -> bool {
        self.w == 0 || self.h == 0
    }

    /// One past the right edge.
    pub const fn right(self) -> u32 {
        self.x + self.w
    }

    /// One past the bottom edge.
    pub const fn bottom(self) -> u32 {
        self.y + self.h
    }

    /// The smallest rectangle covering both.
    pub fn union(self, other: Self) -> Self {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        Self { x, y, w: self.right().max(other.right()) - x, h: self.bottom().max(other.bottom()) - y }
    }

    /// The overlap, empty when there is none.
    pub fn intersect(self, other: Self) -> Self {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let r = self.right().min(other.right());
        let b = self.bottom().min(other.bottom());
        if r <= x || b <= y { Self::default() } else { Self { x, y, w: r - x, h: b - y } }
    }

    /// Whether the rectangles overlap or share an edge.
    pub const fn touches(self, other: Self) -> bool {
        self.x <= other.right() && other.x <= self.right() && self.y <= other.bottom() && other.y <= self.bottom()
    }

    /// Whether `other` lies inside this rectangle.
    pub const fn contains(self, other: Self) -> bool {
        other.x >= self.x && other.y >= self.y && other.right() <= self.right() && other.bottom() <= self.bottom()
    }

    /// The rectangle clipped to a `size × size` tile.
    pub fn clip(self, size: u32) -> Self {
        self.intersect(Self::new(0, 0, size, size))
    }

    /// The rectangle covering the texels a disc of `radius` texels around `(cx, cy)` touches,
    /// clipped to a `size × size` tile.
    pub fn around(cx: f32, cy: f32, radius: f32, size: u32) -> Self {
        if !(cx.is_finite() && cy.is_finite() && radius.is_finite()) || radius < 0.0 {
            return Self::default();
        }
        let s = size as f32;
        let x0 = (cx - radius).floor().clamp(0.0, s);
        let y0 = (cy - radius).floor().clamp(0.0, s);
        let x1 = (cx + radius).ceil().clamp(0.0, s);
        let y1 = (cy + radius).ceil().clamp(0.0, s);
        Self::new(x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32)
    }
}

/// The dirty rectangles of every tile of one texture set.
#[derive(Clone, Debug)]
pub struct DirtyRects {
    tile_size: u32,
    /// `tiles * MAX_RECTS_PER_TILE` slots.
    rects: Vec<Rect>,
    /// Rectangles in use per tile.
    counts: Vec<u8>,
}

impl DirtyRects {
    /// Tracking for `tiles` tiles of `tile_size²` texels, everything clean.
    pub fn new(tiles: u32, tile_size: u32) -> Result<Self, SkinError> {
        if tiles == 0 || tiles > MAX_TILES {
            return Err(SkinError::limit("tile count", u64::from(MAX_TILES), u64::from(tiles)));
        }
        if tile_size == 0 || tile_size > MAX_TILE_SIZE {
            return Err(SkinError::limit("tile size", u64::from(MAX_TILE_SIZE), u64::from(tile_size)));
        }
        Ok(Self {
            tile_size,
            rects: vec![Rect::default(); tiles as usize * MAX_RECTS_PER_TILE],
            counts: vec![0; tiles as usize],
        })
    }

    /// The tile count.
    pub fn tiles(&self) -> u32 {
        self.counts.len() as u32
    }

    /// The tile side in texels.
    pub const fn tile_size(&self) -> u32 {
        self.tile_size
    }

    /// The dirty rectangles of `tile` (empty for an unknown tile).
    pub fn rects(&self, tile: u32) -> &[Rect] {
        let Some(&count) = self.counts.get(tile as usize) else { return &[] };
        let start = tile as usize * MAX_RECTS_PER_TILE;
        &self.rects[start..start + usize::from(count)]
    }

    /// Whether nothing is dirty.
    pub fn is_clean(&self) -> bool {
        self.counts.iter().all(|&c| c == 0)
    }

    /// The summed area of the dirty rectangles: an upper bound on the texels to resolve (merged
    /// rectangles may still overlap).
    pub fn dirty_area(&self) -> u64 {
        (0..self.tiles()).flat_map(|t| self.rects(t).iter()).map(|r| r.area()).sum()
    }

    /// Marks a rectangle of `tile` dirty. The rectangle is clipped to the tile; an empty one or an
    /// unknown tile is ignored (false).
    pub fn mark(&mut self, tile: u32, rect: Rect) -> bool {
        let rect = rect.clip(self.tile_size);
        if rect.is_empty() || tile >= self.tiles() {
            return false;
        }
        let start = tile as usize * MAX_RECTS_PER_TILE;
        let mut count = usize::from(self.counts[tile as usize]);
        let list = &mut self.rects[start..start + MAX_RECTS_PER_TILE];
        let mut pending = rect;
        // Absorb every rectangle the new one swallows or should merge with, repeatedly, since a
        // merge can make the result touch further rectangles.
        loop {
            let mut merged = false;
            let mut i = 0;
            while i < count {
                let r = list[i];
                let u = r.union(pending);
                let cheap = r.touches(pending) && u.area() <= (r.area() + pending.area()) * 5 / 4;
                if r.contains(pending) {
                    return true;
                }
                if pending.contains(r) || cheap {
                    pending = u;
                    count -= 1;
                    list[i] = list[count];
                    merged = true;
                } else {
                    i += 1;
                }
            }
            if !merged {
                break;
            }
        }
        if count == MAX_RECTS_PER_TILE {
            // Full: merge the pending rectangle with the one whose union grows least.
            let mut best = 0;
            let mut best_growth = u64::MAX;
            for (i, r) in list.iter().enumerate() {
                let growth = r.union(pending).area() - r.area();
                if growth < best_growth {
                    best = i;
                    best_growth = growth;
                }
            }
            list[best] = list[best].union(pending);
        } else {
            list[count] = pending;
            count += 1;
        }
        self.counts[tile as usize] = count as u8;
        true
    }

    /// Marks a whole tile dirty.
    pub fn mark_tile(&mut self, tile: u32) -> bool {
        if tile >= self.tiles() {
            return false;
        }
        let start = tile as usize * MAX_RECTS_PER_TILE;
        self.rects[start] = Rect::new(0, 0, self.tile_size, self.tile_size);
        self.counts[tile as usize] = 1;
        true
    }

    /// Marks every tile dirty (a global parameter changed).
    pub fn mark_all(&mut self) {
        for tile in 0..self.tiles() {
            self.mark_tile(tile);
        }
    }

    /// Calls `visit(tile, rect)` for every dirty rectangle, then clears everything.
    pub fn drain(&mut self, mut visit: impl FnMut(u32, Rect)) {
        for tile in 0..self.tiles() {
            for &r in self.rects(tile) {
                visit(tile, r);
            }
        }
        self.counts.fill(0);
    }

    /// Clears everything.
    pub fn clear(&mut self) {
        self.counts.fill(0);
    }
}

#[cfg(test)]
mod tests {
    use humanart_core::rng::Rng;

    use super::*;

    #[test]
    fn rect_algebra() {
        let a = Rect::new(0, 0, 10, 10);
        let b = Rect::new(5, 5, 10, 10);
        assert_eq!(a.union(b), Rect::new(0, 0, 15, 15));
        assert_eq!(a.intersect(b), Rect::new(5, 5, 5, 5));
        assert!(a.intersect(Rect::new(20, 20, 1, 1)).is_empty());
        assert!(a.touches(Rect::new(10, 0, 5, 5)), "shared edge");
        assert!(!a.touches(Rect::new(11, 0, 5, 5)));
        assert_eq!(Rect::new(250, 250, 20, 20).clip(256), Rect::new(250, 250, 6, 6));
        assert_eq!(Rect::around(10.5, 10.5, 2.0, 256), Rect::new(8, 8, 5, 5));
        assert!(Rect::around(f32::NAN, 0.0, 1.0, 256).is_empty());
        assert_eq!(Rect::around(-5.0, -5.0, 2.0, 256), Rect::default());
    }

    #[test]
    fn skn02_marks_cover_every_texel_and_stay_bounded() {
        // Property: after any sequence of marks, every marked texel is inside some rectangle, at
        // most MAX_RECTS_PER_TILE rectangles exist per tile, and all lie inside the tile.
        let mut rng = Rng::new(5);
        for _ in 0..50 {
            let size = 64;
            let mut dirty = DirtyRects::new(2, size).unwrap_or_else(|e| panic!("{e}"));
            let mut truth = vec![false; 2 * 64 * 64];
            for _ in 0..rng.index(60) {
                let tile = rng.below(2) as u32;
                let r =
                    Rect::new(rng.below(70) as u32, rng.below(70) as u32, rng.below(12) as u32, rng.below(12) as u32);
                dirty.mark(tile, r);
                let c = r.clip(size);
                for y in c.y..c.bottom() {
                    for x in c.x..c.right() {
                        truth[(tile * 64 * 64 + y * 64 + x) as usize] = true;
                    }
                }
            }
            for tile in 0..2 {
                let rects = dirty.rects(tile);
                assert!(rects.len() <= MAX_RECTS_PER_TILE);
                assert!(rects.iter().all(|r| Rect::new(0, 0, size, size).contains(*r) && !r.is_empty()));
                for y in 0..64 {
                    for x in 0..64 {
                        if truth[(tile * 64 * 64 + y * 64 + x) as usize] {
                            assert!(rects.iter().any(|r| r.contains(Rect::new(x, y, 1, 1))), "texel {x},{y} lost");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn skn02_a_stroke_stays_small() {
        // A diagonal brush stroke of 40 dabs (radius 6) across a 2048 tile stays far below the
        // tile area: re-resolve cost follows the stroke, not the tile.
        let mut dirty = DirtyRects::new(1, 2048).unwrap_or_else(|e| panic!("{e}"));
        for i in 0..40 {
            let t = i as f32 * 10.0;
            dirty.mark(0, Rect::around(500.0 + t, 700.0 + t * 0.5, 6.0, 2048));
        }
        let area = dirty.dirty_area();
        assert!(area < 2048 * 2048 / 50, "dirty area {area}");
        assert!(area >= 40 * 100, "covers the dabs");
        let mut seen = 0;
        dirty.drain(|tile, _| {
            assert_eq!(tile, 0);
            seen += 1;
        });
        assert!(seen > 0);
        assert!(dirty.is_clean());
    }

    #[test]
    fn whole_tiles_and_limits() {
        assert!(DirtyRects::new(0, 64).is_err());
        assert!(DirtyRects::new(65, 64).is_err());
        assert!(DirtyRects::new(1, 0).is_err());
        let mut d = DirtyRects::new(3, 128).unwrap_or_else(|e| panic!("{e}"));
        assert!(!d.mark(3, Rect::new(0, 0, 4, 4)));
        assert!(!d.mark(0, Rect::new(200, 0, 4, 4)), "outside the tile");
        d.mark_all();
        assert_eq!(d.dirty_area(), 3 * 128 * 128);
        assert!(d.mark(1, Rect::new(3, 3, 4, 4)));
        assert_eq!(d.rects(1), &[Rect::new(0, 0, 128, 128)], "contained marks add nothing");
        d.clear();
        assert!(d.is_clean());
        assert!(d.rects(9).is_empty());
    }
}
