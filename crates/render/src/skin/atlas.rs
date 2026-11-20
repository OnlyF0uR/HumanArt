//! A shelf allocator for the compositor's content atlas: rectangles of layer content packed into
//! one texture. Editor-time (layer changes), never per frame.

/// A placed rectangle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    /// Left texel in the atlas.
    pub x: u32,
    /// Top texel in the atlas.
    pub y: u32,
}

/// One shelf: a row of rectangles of at most `height` texels.
#[derive(Clone, Copy, Debug)]
struct Shelf {
    y: u32,
    height: u32,
    cursor: u32,
}

/// The allocator over a `size × size` atlas.
#[derive(Clone, Debug)]
pub struct ShelfAtlas {
    size: u32,
    shelves: Vec<Shelf>,
    used: u64,
}

impl ShelfAtlas {
    /// An empty atlas.
    pub fn new(size: u32) -> Self {
        Self { size, shelves: Vec::new(), used: 0 }
    }

    /// The side in texels.
    pub const fn size(&self) -> u32 {
        self.size
    }

    /// Texels handed out.
    pub const fn used(&self) -> u64 {
        self.used
    }

    /// Forgets every placement.
    pub fn reset(&mut self) {
        self.shelves.clear();
        self.used = 0;
    }

    /// Places a `w × h` rectangle: on the first shelf tall enough (wasting at most half its
    /// height) with room left, else on a new shelf. `None` when the atlas is full.
    pub fn allocate(&mut self, w: u32, h: u32) -> Option<Placement> {
        if w == 0 || h == 0 || w > self.size || h > self.size {
            return None;
        }
        for shelf in &mut self.shelves {
            if h <= shelf.height && h * 2 >= shelf.height && self.size - shelf.cursor >= w {
                let at = Placement { x: shelf.cursor, y: shelf.y };
                shelf.cursor += w;
                self.used += u64::from(w) * u64::from(h);
                return Some(at);
            }
        }
        let top = self.shelves.last().map_or(0, |s| s.y + s.height);
        if self.size - top < h {
            return None;
        }
        self.shelves.push(Shelf { y: top, height: h, cursor: w });
        self.used += u64::from(w) * u64::from(h);
        Some(Placement { x: 0, y: top })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placements_never_overlap_and_fill_up() {
        let mut atlas = ShelfAtlas::new(64);
        let mut placed: Vec<(Placement, u32, u32)> = Vec::new();
        let sizes = [(10, 10), (20, 8), (5, 12), (30, 30), (64, 4), (9, 9), (12, 6)];
        for (w, h) in sizes.iter().cycle().take(40) {
            if let Some(p) = atlas.allocate(*w, *h) {
                assert!(p.x + w <= 64 && p.y + h <= 64);
                for (q, qw, qh) in &placed {
                    let apart = p.x + w <= q.x || q.x + qw <= p.x || p.y + h <= q.y || q.y + qh <= p.y;
                    assert!(apart, "{p:?} overlaps {q:?}");
                }
                placed.push((p, *w, *h));
            }
        }
        assert!(atlas.allocate(65, 1).is_none());
        assert!(atlas.allocate(0, 1).is_none());
        assert!(atlas.used() <= 64 * 64);
        atlas.reset();
        assert_eq!(atlas.allocate(64, 64), Some(Placement { x: 0, y: 0 }));
        assert!(atlas.allocate(1, 1).is_none());
    }
}
