//! A flexbox-like layout engine `[UI-04]`.
//!
//! A [`LayoutTree`] is rebuilt every frame by the immediate-mode code: [`LayoutTree::clear`], then
//! [`LayoutTree::add`] for each box, parent before child, with a [`Style`]: direction (row or
//! column), wrapping, gaps, padding, margins, fixed / percentage / automatic sizes with minimum and
//! maximum, grow and shrink factors with a basis, main-axis justification and cross-axis
//! alignment, scrolling, and absolute positioning for overlays. [`LayoutTree::compute`] places every
//! box in logical pixels and snaps the edges to the physical pixel grid of the given scale.
//!
//! The algorithm is the core of CSS flexbox, simplified: no `order`, no baseline alignment, no
//! `align-content` other than packing lines at the start, margins are never negative, and a leaf's
//! content size is given (height-for-width text is measured by the caller at a known width).
//!
//! **Guarantees** (tested on random trees, `[AC-WP12-4]`): every size is finite and non-negative,
//! and in-flow siblings never overlap: when their minimum sizes do not fit they overflow their
//! parent in order instead of being squeezed below their minimum or stacked on each other.
//!
//! **Cost.** Two linear passes over the nodes, no recursion (nodes are stored parent before child,
//! so a reverse sweep sees children first), and no allocation once the tree's vectors have grown to
//! the frame's size `[PERF-R1]`.

use std::ops::Range;

use crate::geom::{Insets, Point, Rect, Size};

/// A box in a [`LayoutTree`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

/// The main axis of a container.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Direction {
    /// Children left to right.
    #[default]
    Row,
    /// Children top to bottom.
    Column,
}

/// A length.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Dim {
    /// From the content (or stretched by the parent's alignment).
    #[default]
    Auto,
    /// Logical pixels.
    Px(f32),
    /// A fraction (0 to 1) of the parent's inner size on the same axis.
    Percent(f32),
}

/// Distribution of free space along the main axis.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Justify {
    /// Packed at the start.
    #[default]
    Start,
    /// Centred.
    Center,
    /// Packed at the end.
    End,
    /// First and last at the edges, equal gaps between.
    SpaceBetween,
    /// Equal space around each item.
    SpaceAround,
    /// Equal space between items and edges.
    SpaceEvenly,
}

/// Alignment on the cross axis.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Align {
    /// At the line's start.
    Start,
    /// Centred in the line.
    Center,
    /// At the line's end.
    End,
    /// Filling the line (when the cross size is automatic).
    #[default]
    Stretch,
}

/// How a box takes part in its parent's layout.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Position {
    /// In the flow of its parent's children.
    #[default]
    Flow,
    /// Out of the flow, placed by distances from the parent's edges (overlays, popups, badges).
    Absolute {
        /// From the parent's left edge.
        left: Option<f32>,
        /// From the parent's top edge.
        top: Option<f32>,
        /// From the parent's right edge.
        right: Option<f32>,
        /// From the parent's bottom edge.
        bottom: Option<f32>,
    },
}

/// The layout properties of a box. Sizes are logical pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Style {
    /// The main axis of its children.
    pub direction: Direction,
    /// Whether children wrap onto new lines when the main axis is full.
    pub wrap: bool,
    /// Space between children on the main axis.
    pub gap: f32,
    /// Space between wrapped lines.
    pub line_gap: f32,
    /// Space inside the box around its children.
    pub padding: Insets,
    /// Space outside the box (never negative).
    pub margin: Insets,
    /// Width.
    pub width: Dim,
    /// Height.
    pub height: Dim,
    /// Smallest width.
    pub min_width: f32,
    /// Smallest height.
    pub min_height: f32,
    /// Largest width.
    pub max_width: f32,
    /// Largest height.
    pub max_height: f32,
    /// Share of the free main-axis space it takes.
    pub grow: f32,
    /// Share of the missing main-axis space it gives up (weighted by its basis).
    pub shrink: f32,
    /// The main size before growing and shrinking; `Auto` uses the width or height, then the content.
    pub basis: Dim,
    /// Distribution of free space among its children.
    pub justify: Justify,
    /// Cross-axis alignment of its children.
    pub align_items: Align,
    /// Its own cross-axis alignment, overriding the parent's `align_items`.
    pub align_self: Option<Align>,
    /// Flow or absolute.
    pub position: Position,
    /// Whether its children may extend beyond it along its main axis (a scroll area): they are laid
    /// out at their natural size and moved by the scroll offset.
    pub scroll: bool,
    /// The size of a leaf's own content (text, an icon, an image).
    pub content: Size,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            direction: Direction::Row,
            wrap: false,
            gap: 0.0,
            line_gap: 0.0,
            padding: Insets::ZERO,
            margin: Insets::ZERO,
            width: Dim::Auto,
            height: Dim::Auto,
            min_width: 0.0,
            min_height: 0.0,
            max_width: f32::INFINITY,
            max_height: f32::INFINITY,
            grow: 0.0,
            shrink: 1.0,
            basis: Dim::Auto,
            justify: Justify::Start,
            align_items: Align::Stretch,
            align_self: None,
            position: Position::Flow,
            scroll: false,
            content: Size::ZERO,
        }
    }
}

impl Style {
    /// A row container.
    pub fn row() -> Self {
        Self { direction: Direction::Row, ..Self::default() }
    }

    /// A column container.
    pub fn column() -> Self {
        Self { direction: Direction::Column, ..Self::default() }
    }

    /// A leaf of fixed content size.
    pub fn leaf(content: Size) -> Self {
        Self { content, ..Self::default() }
    }

    /// The style with every number made safe: finite, margins and gaps not negative, minimum not
    /// above maximum.
    fn sanitized(mut self) -> Self {
        let fin = |v: f32| if v.is_finite() { v.max(0.0) } else { 0.0 };
        let ins = |i: Insets| Insets { left: fin(i.left), top: fin(i.top), right: fin(i.right), bottom: fin(i.bottom) };
        let dim = |d: Dim| match d {
            Dim::Px(v) if v.is_finite() => Dim::Px(v.max(0.0)),
            Dim::Percent(v) if v.is_finite() => Dim::Percent(v.clamp(0.0, 1.0)),
            _ => Dim::Auto,
        };
        self.gap = fin(self.gap);
        self.line_gap = fin(self.line_gap);
        self.padding = ins(self.padding);
        self.margin = ins(self.margin);
        self.width = dim(self.width);
        self.height = dim(self.height);
        self.basis = dim(self.basis);
        self.min_width = fin(self.min_width);
        self.min_height = fin(self.min_height);
        self.max_width = if self.max_width.is_nan() { f32::INFINITY } else { self.max_width.max(self.min_width) };
        self.max_height = if self.max_height.is_nan() { f32::INFINITY } else { self.max_height.max(self.min_height) };
        self.grow = fin(self.grow);
        self.shrink = fin(self.shrink);
        self.content = Size::new(fin(self.content.w), fin(self.content.h));
        if let Position::Absolute { left, top, right, bottom } = self.position {
            let o = |v: Option<f32>| v.filter(|v| v.is_finite());
            self.position = Position::Absolute { left: o(left), top: o(top), right: o(right), bottom: o(bottom) };
        }
        self
    }
}

/// "No node".
const NONE: u32 = u32::MAX;

/// One box.
#[derive(Clone, Copy, Debug)]
struct Node {
    style: Style,
    parent: u32,
    first_child: u32,
    last_child: u32,
    next: u32,
    /// Intrinsic border-box size (content-based, clamped).
    intrinsic: Size,
    /// Scratch: the item's main and cross size while its parent is arranged.
    main: f32,
    cross: f32,
    frozen: bool,
    /// The final rectangle (border box), logical pixels.
    rect: Rect,
    /// The extent of the children (scroll areas).
    content_size: Size,
    /// The scroll offset applied to the children.
    scroll: Point,
}

/// One line of a container while it is arranged.
#[derive(Clone, Copy, Debug)]
struct Line {
    /// The first child of the line.
    first: u32,
    /// The number of children.
    count: u32,
    cross: f32,
}

/// The boxes of one frame and their computed rectangles.
#[derive(Debug, Default)]
pub struct LayoutTree {
    nodes: Vec<Node>,
    lines: Vec<Line>,
}

impl LayoutTree {
    /// An empty tree.
    pub fn new() -> Self {
        Self::default()
    }

    /// Forgets every box, keeping the memory.
    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    /// The number of boxes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Adds a box under `parent` (a root when `None` or when `parent` is not a node of this tree).
    pub fn add(&mut self, parent: Option<NodeId>, style: Style) -> NodeId {
        let id = u32::try_from(self.nodes.len()).unwrap_or(NONE - 1);
        let parent = parent.map_or(NONE, |p| p.0);
        let parent = if (parent as usize) < self.nodes.len() { parent } else { NONE };
        self.nodes.push(Node {
            style: style.sanitized(),
            parent,
            first_child: NONE,
            last_child: NONE,
            next: NONE,
            intrinsic: Size::ZERO,
            main: 0.0,
            cross: 0.0,
            frozen: false,
            rect: Rect::ZERO,
            content_size: Size::ZERO,
            scroll: Point::ZERO,
        });
        if parent != NONE {
            let last = self.nodes[parent as usize].last_child;
            if last == NONE {
                self.nodes[parent as usize].first_child = id;
            } else {
                self.nodes[last as usize].next = id;
            }
            self.nodes[parent as usize].last_child = id;
        }
        NodeId(id)
    }

    /// Sets the scroll offset of a scroll area (applied by the next [`LayoutTree::compute`]).
    pub fn set_scroll(&mut self, node: NodeId, offset: Point) {
        if let Some(n) = self.nodes.get_mut(node.0 as usize) {
            n.scroll = Point::new(
                if offset.x.is_finite() { offset.x } else { 0.0 },
                if offset.y.is_finite() { offset.y } else { 0.0 },
            );
        }
    }

    /// The computed rectangle of a box (empty for an unknown id).
    pub fn rect(&self, node: NodeId) -> Rect {
        self.nodes.get(node.0 as usize).map_or(Rect::ZERO, |n| n.rect)
    }

    /// The extent of a box's children relative to its inner top-left corner (for scroll bars).
    pub fn content_size(&self, node: NodeId) -> Size {
        self.nodes.get(node.0 as usize).map_or(Size::ZERO, |n| n.content_size)
    }

    /// The children of a box, in order.
    pub fn children(&self, node: NodeId) -> Children<'_> {
        Children { tree: self, next: self.nodes.get(node.0 as usize).map_or(NONE, |n| n.first_child) }
    }

    /// The parent of a box.
    pub fn parent(&self, node: NodeId) -> Option<NodeId> {
        self.nodes.get(node.0 as usize).map(|n| n.parent).filter(|&p| p != NONE).map(NodeId)
    }

    /// The style of a box (sanitised).
    pub fn style(&self, node: NodeId) -> Option<&Style> {
        self.nodes.get(node.0 as usize).map(|n| &n.style)
    }

    /// Lays every root out in `viewport` (logical pixels) and snaps all edges to the pixel grid of
    /// `scale` physical pixels per logical pixel.
    pub fn compute(&mut self, viewport: Rect, scale: f32) {
        self.measure();
        for i in 0..self.nodes.len() {
            if self.nodes[i].parent == NONE {
                let n = &self.nodes[i];
                let s = &n.style;
                let w = resolve(s.width, viewport.w).unwrap_or(viewport.w).clamp(s.min_width, s.max_width);
                let h = resolve(s.height, viewport.h).unwrap_or(viewport.h).clamp(s.min_height, s.max_height);
                self.nodes[i].rect = Rect::new(viewport.x + s.margin.left, viewport.y + s.margin.top, fin0(w), fin0(h));
            }
            self.arrange(i);
        }
        for n in &mut self.nodes {
            n.rect = n.rect.snap(scale);
        }
    }

    /// Intrinsic sizes, children before parents.
    fn measure(&mut self) {
        for i in (0..self.nodes.len()).rev() {
            let n = self.nodes[i];
            let s = n.style;
            let row = s.direction == Direction::Row;
            let (mut main, mut cross, mut count) = (0.0_f32, 0.0_f32, 0_u32);
            let mut c = n.first_child;
            while c != NONE {
                let child = &self.nodes[c as usize];
                if child.style.position == Position::Flow {
                    let outer = outer_size(child.intrinsic, child.style.margin);
                    let (m, x) = if row { (outer.w, outer.h) } else { (outer.h, outer.w) };
                    main += m;
                    cross = cross.max(x);
                    count += 1;
                }
                c = child.next;
            }
            #[allow(clippy::cast_precision_loss)] // INVARIANT: child counts are far below 2^24.
            let gaps = s.gap * count.saturating_sub(1) as f32;
            let (cw, ch) = if row { (main + gaps, cross) } else { (cross, main + gaps) };
            let content_w = cw.max(s.content.w) + s.padding.horizontal();
            let content_h = ch.max(s.content.h) + s.padding.vertical();
            let w = match s.width {
                Dim::Px(v) => v,
                _ => content_w,
            };
            let h = match s.height {
                Dim::Px(v) => v,
                _ => content_h,
            };
            self.nodes[i].intrinsic =
                Size::new(fin0(w.clamp(s.min_width, s.max_width)), fin0(h.clamp(s.min_height, s.max_height)));
        }
    }

    /// Places the children of node `i`, whose rectangle is final.
    #[allow(clippy::too_many_lines)]
    fn arrange(&mut self, i: usize) {
        let node = self.nodes[i];
        if node.first_child == NONE {
            node_content(&mut self.nodes[i], Size::ZERO);
            return;
        }
        let s = node.style;
        let row = s.direction == Direction::Row;
        let inner = node.rect.inset(s.padding);
        let (inner_main, inner_cross) = if row { (inner.w, inner.h) } else { (inner.h, inner.w) };
        // A scroll area lays its children out at their natural main size.
        let avail_main = if s.scroll { f32::INFINITY } else { inner_main };

        // Hypothetical sizes of the in-flow children.
        let mut c = node.first_child;
        while c != NONE {
            let child = &mut self.nodes[c as usize];
            let cs = child.style;
            if cs.position == Position::Flow {
                let (main_dim, cross_dim) = if row { (cs.width, cs.height) } else { (cs.height, cs.width) };
                let (min_m, max_m) = if row { (cs.min_width, cs.max_width) } else { (cs.min_height, cs.max_height) };
                let (min_c, max_c) = if row { (cs.min_height, cs.max_height) } else { (cs.min_width, cs.max_width) };
                let (intr_m, intr_c) =
                    if row { (child.intrinsic.w, child.intrinsic.h) } else { (child.intrinsic.h, child.intrinsic.w) };
                let basis = resolve(cs.basis, inner_main).or_else(|| resolve(main_dim, inner_main)).unwrap_or(intr_m);
                child.main = basis.clamp(min_m, max_m);
                child.cross = resolve(cross_dim, inner_cross).unwrap_or(intr_c).clamp(min_c, max_c);
                child.frozen = false;
            }
            c = child.next;
        }

        // Break into lines.
        self.lines.clear();
        let mut c = node.first_child;
        let mut line = Line { first: NONE, count: 0, cross: 0.0 };
        let mut used = 0.0_f32;
        while c != NONE {
            let child = self.nodes[c as usize];
            if child.style.position == Position::Flow {
                let outer_main = child.main + main_margins(&child.style, row);
                let extra = if line.count > 0 { s.gap } else { 0.0 };
                if s.wrap && line.count > 0 && used + extra + outer_main > avail_main {
                    self.lines.push(line);
                    line = Line { first: NONE, count: 0, cross: 0.0 };
                    used = 0.0;
                }
                if line.count == 0 {
                    line.first = c;
                    used = outer_main;
                } else {
                    used += s.gap + outer_main;
                }
                line.count += 1;
            }
            c = child.next;
        }
        if line.count > 0 {
            self.lines.push(line);
        }

        // Resolve flexible lengths and place each line.
        let single = self.lines.len() == 1 && !s.wrap;
        let mut cross_pos = 0.0_f32;
        let mut extent = Size::ZERO;
        for li in 0..self.lines.len() {
            let line = self.lines[li];
            self.flex_line(line, row, avail_main, s.gap);
            // The line's cross size.
            let mut line_cross = 0.0_f32;
            self.for_line(line, |n| {
                line_cross = line_cross.max(n.cross + cross_margins(&n.style, row));
            });
            if single && inner_cross.is_finite() {
                line_cross = inner_cross;
            }
            self.lines[li].cross = line_cross;
            // Main-axis positions.
            let mut total = 0.0_f32;
            let mut count = 0_u32;
            self.for_line(line, |n| {
                total += n.main + main_margins(&n.style, row);
                count += 1;
            });
            #[allow(clippy::cast_precision_loss)] // INVARIANT: child counts are far below 2^24.
            let n_items = count as f32;
            total += s.gap * (n_items - 1.0).max(0.0);
            let free = if avail_main.is_finite() { avail_main - total } else { 0.0 };
            let (mut pos, between) = if free > 0.0 {
                match s.justify {
                    Justify::Center => (free * 0.5, 0.0),
                    Justify::End => (free, 0.0),
                    Justify::SpaceBetween if count > 1 => (0.0, free / (n_items - 1.0)),
                    Justify::Start | Justify::SpaceBetween => (0.0, 0.0),
                    Justify::SpaceAround => (free / n_items * 0.5, free / n_items),
                    Justify::SpaceEvenly => (free / (n_items + 1.0), free / (n_items + 1.0)),
                }
            } else {
                (0.0, 0.0)
            };
            let mut c = line.first;
            let mut placed = 0;
            while c != NONE && placed < line.count {
                let n = &mut self.nodes[c as usize];
                if n.style.position == Position::Flow {
                    let cs = n.style;
                    let (m0, m1) =
                        if row { (cs.margin.left, cs.margin.right) } else { (cs.margin.top, cs.margin.bottom) };
                    let (c0, c1) =
                        if row { (cs.margin.top, cs.margin.bottom) } else { (cs.margin.left, cs.margin.right) };
                    let cross_dim = if row { cs.height } else { cs.width };
                    let (min_c, max_c) =
                        if row { (cs.min_height, cs.max_height) } else { (cs.min_width, cs.max_width) };
                    let align = cs.align_self.unwrap_or(s.align_items);
                    let room = (line_cross - c0 - c1).max(0.0);
                    if align == Align::Stretch && cross_dim == Dim::Auto {
                        n.cross = room.clamp(min_c, max_c);
                    }
                    let offset = match align {
                        Align::Start | Align::Stretch => 0.0,
                        Align::Center => ((room - n.cross) * 0.5).max(0.0),
                        Align::End => (room - n.cross).max(0.0),
                    };
                    let main_at = pos + m0;
                    let cross_at = cross_pos + c0 + offset;
                    n.rect = if row {
                        Rect::new(
                            inner.x + main_at - node.scroll.x,
                            inner.y + cross_at - node.scroll.y,
                            n.main,
                            n.cross,
                        )
                    } else {
                        Rect::new(
                            inner.x + cross_at - node.scroll.x,
                            inner.y + main_at - node.scroll.y,
                            n.cross,
                            n.main,
                        )
                    };
                    let main_end = main_at + n.main + m1;
                    let cross_end = cross_at + n.cross + c1;
                    extent = if row {
                        Size::new(extent.w.max(main_end), extent.h.max(cross_end))
                    } else {
                        Size::new(extent.w.max(cross_end), extent.h.max(main_end))
                    };
                    pos = main_end + s.gap + between;
                    placed += 1;
                }
                c = n.next;
            }
            cross_pos += line_cross + s.line_gap;
        }

        // Absolute children, relative to the padding box.
        let mut c = node.first_child;
        while c != NONE {
            let n = &mut self.nodes[c as usize];
            if let Position::Absolute { left, top, right, bottom } = n.style.position {
                let cs = n.style;
                let r = node.rect;
                let w = resolve(cs.width, r.w)
                    .or_else(|| left.zip(right).map(|(l, rr)| r.w - l - rr))
                    .unwrap_or(n.intrinsic.w)
                    .clamp(cs.min_width, cs.max_width);
                let h = resolve(cs.height, r.h)
                    .or_else(|| top.zip(bottom).map(|(t, b)| r.h - t - b))
                    .unwrap_or(n.intrinsic.h)
                    .clamp(cs.min_height, cs.max_height);
                let (w, h) = (fin0(w), fin0(h));
                let x = left.map_or_else(|| right.map_or(r.x, |rr| r.right() - rr - w), |l| r.x + l);
                let y = top.map_or_else(|| bottom.map_or(r.y, |b| r.bottom() - b - h), |t| r.y + t);
                n.rect = Rect::new(x, y, w, h);
            }
            c = n.next;
        }
        node_content(&mut self.nodes[i], extent);
    }

    /// Calls `f` on each in-flow child of `line`.
    fn for_line(&self, line: Line, mut f: impl FnMut(&Node)) {
        let mut c = line.first;
        let mut seen = 0;
        while c != NONE && seen < line.count {
            let n = &self.nodes[c as usize];
            if n.style.position == Position::Flow {
                f(n);
                seen += 1;
            }
            c = n.next;
        }
    }

    /// Grows or shrinks the items of a line to fill `avail` along the main axis, honouring their
    /// minimum and maximum sizes (items that hit a limit are frozen and the rest redistributed).
    fn flex_line(&mut self, line: Line, row: bool, avail: f32, gap: f32) {
        if !avail.is_finite() {
            return;
        }
        for _ in 0..=line.count {
            let mut used = 0.0_f32;
            let mut grow = 0.0_f32;
            let mut shrink = 0.0_f32;
            let mut count = 0_u32;
            self.for_line(line, |n| {
                used += n.main + main_margins(&n.style, row);
                count += 1;
                if !n.frozen {
                    grow += n.style.grow;
                    shrink += n.style.shrink * n.main;
                }
            });
            #[allow(clippy::cast_precision_loss)] // INVARIANT: child counts are far below 2^24.
            let gaps = gap * count.saturating_sub(1) as f32;
            let free = avail - used - gaps;
            if free.abs() < 1e-4 || (free > 0.0 && grow <= 0.0) || (free < 0.0 && shrink <= 0.0) {
                return;
            }
            let mut changed = false;
            let mut c = line.first;
            let mut seen = 0;
            while c != NONE && seen < line.count {
                let n = &mut self.nodes[c as usize];
                c = n.next;
                if n.style.position != Position::Flow {
                    continue;
                }
                seen += 1;
                if n.frozen {
                    continue;
                }
                let (min_m, max_m) =
                    if row { (n.style.min_width, n.style.max_width) } else { (n.style.min_height, n.style.max_height) };
                let target = if free > 0.0 {
                    n.main + free * n.style.grow / grow
                } else {
                    n.main + free * (n.style.shrink * n.main) / shrink
                };
                let clamped = target.clamp(min_m, max_m).max(0.0);
                if (clamped - target).abs() > 1e-4
                    || (free > 0.0 && n.style.grow <= 0.0)
                    || (free < 0.0 && n.style.shrink <= 0.0)
                {
                    n.frozen = true;
                }
                if (clamped - n.main).abs() > 1e-6 {
                    changed = true;
                }
                n.main = clamped;
            }
            if !changed {
                return;
            }
        }
    }
}

/// The children of a box.
#[derive(Debug)]
pub struct Children<'a> {
    tree: &'a LayoutTree,
    next: u32,
}

impl Iterator for Children<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<NodeId> {
        let n = self.tree.nodes.get(self.next as usize)?;
        let id = NodeId(self.next);
        self.next = n.next;
        Some(id)
    }
}

fn node_content(node: &mut Node, extent: Size) {
    node.content_size = extent;
}

/// A length resolved against the parent's inner size; `None` for `Auto` (or a percentage of an
/// unbounded size).
fn resolve(dim: Dim, parent: f32) -> Option<f32> {
    match dim {
        Dim::Auto => None,
        Dim::Px(v) => Some(v),
        Dim::Percent(p) => parent.is_finite().then(|| (p * parent).max(0.0)),
    }
}

fn outer_size(size: Size, margin: Insets) -> Size {
    Size::new(size.w + margin.horizontal(), size.h + margin.vertical())
}

fn main_margins(s: &Style, row: bool) -> f32 {
    if row { s.margin.horizontal() } else { s.margin.vertical() }
}

fn cross_margins(s: &Style, row: bool) -> f32 {
    if row { s.margin.vertical() } else { s.margin.horizontal() }
}

/// A finite, non-negative value (0 otherwise).
fn fin0(v: f32) -> f32 {
    if v.is_finite() { v.max(0.0) } else { 0.0 }
}

/// The rows of a virtualised list that intersect the viewport `[UI-04]`: `count` rows of `row`
/// logical pixels scrolled by `offset`, seen through `viewport` pixels, plus `overscan` rows on each
/// side. Only these rows are built, so a list of any length costs what is visible.
pub fn visible_rows(offset: f32, viewport: f32, row: f32, count: usize, overscan: usize) -> Range<usize> {
    let usable = row.is_finite() && row > 0.0 && offset.is_finite() && viewport.is_finite() && count > 0;
    if !usable {
        return 0..0;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Clamped to 0..count first.
    let first = ((offset.max(0.0) / row).floor() as usize).min(count);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let last = (((offset.max(0.0) + viewport.max(0.0)) / row).ceil() as usize).min(count);
    first.saturating_sub(overscan)..last.saturating_add(overscan).min(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    fn px(v: f32) -> Dim {
        Dim::Px(v)
    }

    #[test]
    fn row_grows_and_shrinks_with_limits() {
        let mut t = LayoutTree::new();
        let root = t.add(None, Style { gap: 10.0, padding: Insets::all(5.0), ..Style::row() });
        let a = t.add(Some(root), Style { width: px(50.0), ..Style::default() });
        let b = t.add(Some(root), Style { grow: 1.0, ..Style::default() });
        let c = t.add(Some(root), Style { grow: 3.0, max_width: 60.0, ..Style::default() });
        t.compute(Rect::new(0.0, 0.0, 300.0, 40.0), 1.0);
        assert_eq!(t.rect(a), Rect::new(5.0, 5.0, 50.0, 30.0));
        // 290 inner - 20 gaps - 50 = 220 free: c capped at 60, b takes the rest (160).
        assert_eq!(t.rect(c).w, 60.0);
        assert_eq!(t.rect(b), Rect::new(65.0, 5.0, 160.0, 30.0));
        assert_eq!(t.rect(c).x, 235.0);
        // Shrinking stops at the minimum; the row then overflows instead of overlapping.
        t.clear();
        let root = t.add(None, Style::row());
        let a = t.add(Some(root), Style { width: px(100.0), min_width: 80.0, ..Style::default() });
        let b = t.add(Some(root), Style { width: px(100.0), ..Style::default() });
        t.compute(Rect::new(0.0, 0.0, 120.0, 10.0), 1.0);
        assert_eq!(t.rect(a).w, 80.0);
        assert_eq!(t.rect(b).w, 40.0);
        t.clear();
        let root = t.add(None, Style::row());
        let a = t.add(Some(root), Style { width: px(100.0), min_width: 100.0, ..Style::default() });
        let b = t.add(Some(root), Style { width: px(100.0), min_width: 100.0, ..Style::default() });
        t.compute(Rect::new(0.0, 0.0, 120.0, 10.0), 1.0);
        assert_eq!((t.rect(a).x, t.rect(b).x), (0.0, 100.0));
    }

    #[test]
    fn column_alignment_justify_and_percent() {
        let mut t = LayoutTree::new();
        let root = t.add(None, Style { justify: Justify::Center, align_items: Align::Center, ..Style::column() });
        let a = t.add(Some(root), Style::leaf(Size::new(40.0, 20.0)));
        let b = t.add(Some(root), Style { width: Dim::Percent(0.5), height: px(10.0), ..Style::default() });
        t.compute(Rect::new(0.0, 0.0, 200.0, 100.0), 1.0);
        assert_eq!(t.rect(a), Rect::new(80.0, 35.0, 40.0, 20.0));
        assert_eq!(t.rect(b), Rect::new(50.0, 55.0, 100.0, 10.0));
        for (justify, xs) in [
            (Justify::End, [60.0, 80.0]),
            (Justify::SpaceBetween, [0.0, 80.0]),
            (Justify::SpaceEvenly, [20.0, 60.0]),
            (Justify::SpaceAround, [15.0, 65.0]),
        ] {
            t.clear();
            let root = t.add(None, Style { justify, ..Style::row() });
            let a = t.add(Some(root), Style::leaf(Size::new(20.0, 5.0)));
            let b = t.add(Some(root), Style::leaf(Size::new(20.0, 5.0)));
            t.compute(Rect::new(0.0, 0.0, 100.0, 5.0), 1.0);
            assert_eq!([t.rect(a).x, t.rect(b).x], xs, "{justify:?}");
        }
    }

    #[test]
    fn wrapping_scrolling_and_absolute() {
        let mut t = LayoutTree::new();
        let root =
            t.add(None, Style { wrap: true, gap: 4.0, line_gap: 6.0, align_items: Align::Start, ..Style::row() });
        let items: Vec<NodeId> = (0..5).map(|_| t.add(Some(root), Style::leaf(Size::new(30.0, 10.0)))).collect();
        t.compute(Rect::new(0.0, 0.0, 100.0, 100.0), 1.0);
        // Three per line (30 + 4 + 30 + 4 + 30 = 98), then two.
        assert_eq!(t.rect(items[2]).y, 0.0);
        assert_eq!(t.rect(items[3]), Rect::new(0.0, 16.0, 30.0, 10.0));
        // A scroll area keeps natural sizes and moves its children by the offset.
        t.clear();
        let list = t.add(None, Style { scroll: true, ..Style::column() });
        let rows: Vec<NodeId> =
            (0..50).map(|_| t.add(Some(list), Style { height: px(20.0), ..Style::default() })).collect();
        t.set_scroll(list, Point::new(0.0, 45.0));
        t.compute(Rect::new(0.0, 0.0, 120.0, 100.0), 1.0);
        assert_eq!(t.rect(rows[0]).y, -45.0);
        assert_eq!(t.rect(rows[49]).h, 20.0);
        assert_eq!(t.content_size(list), Size::new(120.0, 1000.0));
        assert_eq!(visible_rows(45.0, 100.0, 20.0, 50, 0), 2..8);
        assert_eq!(visible_rows(45.0, 100.0, 20.0, 50, 2), 0..10);
        assert_eq!(visible_rows(990.0, 100.0, 20.0, 50, 1), 48..50);
        assert_eq!(visible_rows(0.0, 100.0, 0.0, 50, 1), 0..0);
        // Absolute: pinned to the bottom-right corner, and stretched between left and right.
        t.clear();
        let root = t.add(None, Style::row());
        let badge = t.add(
            Some(root),
            Style {
                position: Position::Absolute { left: None, top: None, right: Some(4.0), bottom: Some(4.0) },
                width: px(10.0),
                height: px(8.0),
                ..Style::default()
            },
        );
        let bar = t.add(
            Some(root),
            Style {
                position: Position::Absolute { left: Some(10.0), top: Some(0.0), right: Some(10.0), bottom: None },
                height: px(3.0),
                ..Style::default()
            },
        );
        t.compute(Rect::new(0.0, 0.0, 100.0, 50.0), 1.0);
        assert_eq!(t.rect(badge), Rect::new(86.0, 38.0, 10.0, 8.0));
        assert_eq!(t.rect(bar), Rect::new(10.0, 0.0, 80.0, 3.0));
        assert_eq!(t.children(root).count(), 2);
        assert_eq!(t.parent(bar), Some(root));
    }

    #[test]
    fn ui_04_pixel_snapping_at_fractional_scales() {
        let mut t = LayoutTree::new();
        let root = t.add(None, Style::row());
        let a = t.add(Some(root), Style { grow: 1.0, ..Style::default() });
        let b = t.add(Some(root), Style { grow: 1.0, ..Style::default() });
        let c = t.add(Some(root), Style { grow: 1.0, ..Style::default() });
        for scale in [0.75, 1.0, 1.25, 1.5, 1.75, 2.0] {
            t.compute(Rect::new(0.0, 0.0, 100.0, 20.0), scale);
            for id in [a, b, c] {
                let r = t.rect(id);
                for v in [r.x, r.right()] {
                    assert!((v * scale - (v * scale).round()).abs() < 1e-3, "scale {scale}: {r:?}");
                }
            }
            assert!((t.rect(a).right() - t.rect(b).x).abs() < 1e-4, "shared edges stay shared");
            assert!((t.rect(b).right() - t.rect(c).x).abs() < 1e-4);
        }
    }

    /// A random style for the property test.
    fn random_style(rng: &mut Rng) -> Style {
        let dim = |rng: &mut Rng| match rng.below(4) {
            0 => Dim::Px(rng.range_f32(0.0, 120.0)),
            1 => Dim::Percent(rng.range_f32(0.0, 1.0)),
            _ => Dim::Auto,
        };
        let ins = |rng: &mut Rng| {
            if rng.chance(0.5) {
                Insets::ZERO
            } else {
                Insets {
                    left: rng.range_f32(0.0, 8.0),
                    top: rng.range_f32(0.0, 8.0),
                    right: rng.range_f32(0.0, 8.0),
                    bottom: rng.range_f32(0.0, 8.0),
                }
            }
        };
        let min_width = if rng.chance(0.3) { rng.range_f32(0.0, 60.0) } else { 0.0 };
        let min_height = if rng.chance(0.3) { rng.range_f32(0.0, 60.0) } else { 0.0 };
        Style {
            direction: if rng.chance(0.5) { Direction::Row } else { Direction::Column },
            wrap: rng.chance(0.3),
            gap: if rng.chance(0.5) { rng.range_f32(0.0, 10.0) } else { 0.0 },
            line_gap: rng.range_f32(0.0, 6.0),
            padding: ins(rng),
            margin: ins(rng),
            width: dim(rng),
            height: dim(rng),
            min_width,
            min_height,
            max_width: if rng.chance(0.2) { rng.range_f32(0.0, 200.0) } else { f32::INFINITY },
            max_height: if rng.chance(0.2) { rng.range_f32(0.0, 200.0) } else { f32::INFINITY },
            grow: if rng.chance(0.5) { rng.range_f32(0.0, 3.0) } else { 0.0 },
            shrink: if rng.chance(0.8) { rng.range_f32(0.0, 2.0) } else { 0.0 },
            basis: dim(rng),
            justify: [
                Justify::Start,
                Justify::Center,
                Justify::End,
                Justify::SpaceBetween,
                Justify::SpaceAround,
                Justify::SpaceEvenly,
            ][rng.index(6)],
            align_items: [Align::Start, Align::Center, Align::End, Align::Stretch][rng.index(4)],
            align_self: if rng.chance(0.2) {
                Some([Align::Start, Align::Center, Align::End, Align::Stretch][rng.index(4)])
            } else {
                None
            },
            position: if rng.chance(0.1) {
                let side = |rng: &mut Rng| rng.chance(0.5).then(|| rng.range_f32(-20.0, 60.0));
                Position::Absolute { left: side(rng), top: side(rng), right: side(rng), bottom: side(rng) }
            } else {
                Position::Flow
            },
            scroll: rng.chance(0.1),
            content: Size::new(rng.range_f32(0.0, 80.0), rng.range_f32(0.0, 30.0)),
        }
    }

    #[test]
    fn ac_wp12_4_random_trees_have_no_overlaps_or_negative_sizes() {
        // [AC-WP12-4] Property test over random trees (fixed seeds): every rectangle is finite and
        // non-negative, and no two in-flow siblings overlap, at several UI scales.
        let mut rng = Rng::new(0x1A70_0001);
        let mut tree = LayoutTree::new();
        let mut checked_pairs = 0_u64;
        for case in 0..3000 {
            tree.clear();
            let count = 1 + rng.index(60);
            let root = tree.add(None, random_style(&mut rng));
            let mut ids = vec![root];
            for _ in 1..count {
                let parent = ids[rng.index(ids.len())];
                ids.push(tree.add(Some(parent), random_style(&mut rng)));
            }
            let viewport = Rect::new(0.0, 0.0, rng.range_f32(0.0, 600.0), rng.range_f32(0.0, 400.0));
            let scale = [0.75, 1.0, 1.25, 1.5, 2.0][rng.index(5)];
            tree.compute(viewport, scale);
            for &id in &ids {
                let r = tree.rect(id);
                assert!(r.x.is_finite() && r.y.is_finite(), "case {case}: {r:?}");
                assert!(r.w >= 0.0 && r.h >= 0.0 && r.w.is_finite() && r.h.is_finite(), "case {case}: {r:?}");
                let kids: Vec<NodeId> = tree
                    .children(id)
                    .filter(|&k| tree.style(k).is_some_and(|s| s.position == Position::Flow))
                    .collect();
                for (i, &a) in kids.iter().enumerate() {
                    for &b in &kids[i + 1..] {
                        let (ra, rb) = (tree.rect(a), tree.rect(b));
                        // Compare in whole physical pixels, where the snapped edges live.
                        let p = |r: Rect| [r.x, r.y, r.right(), r.bottom()].map(|v| (v * scale).round());
                        let (pa, pb) = (p(ra), p(rb));
                        let overlap = pa[0] < pb[2] && pb[0] < pa[2] && pa[1] < pb[3] && pb[1] < pa[3];
                        assert!(!overlap, "case {case}: siblings {ra:?} and {rb:?} overlap");
                        checked_pairs += 1;
                    }
                }
            }
        }
        assert!(checked_pairs > 10_000, "{checked_pairs}");
    }

    #[test]
    fn perf_r1_relayout_does_not_allocate() {
        let mut t = LayoutTree::new();
        let build = |t: &mut LayoutTree| {
            t.clear();
            let root = t.add(None, Style { wrap: true, gap: 2.0, ..Style::row() });
            for i in 0..600 {
                let leaf = Style::leaf(Size::new(20.0 + (i % 7) as f32, 12.0));
                t.add(Some(root), Style { grow: (i % 3) as f32, ..leaf });
            }
            t.compute(Rect::new(0.0, 0.0, 800.0, 600.0), 1.25);
        };
        build(&mut t);
        let stats = humanart_core::alloc::count_in(|| build(&mut t));
        assert_eq!(stats.events(), 0);
    }

    #[test]
    fn hostile_numbers_are_sanitised() {
        let mut t = LayoutTree::new();
        let root = t.add(
            None,
            Style { gap: f32::NAN, padding: Insets::all(-5.0), width: Dim::Px(f32::INFINITY), ..Style::row() },
        );
        let a = t.add(Some(root), Style { min_width: 50.0, max_width: 10.0, grow: f32::NAN, ..Style::default() });
        let b = t.add(Some(NodeId(999)), Style::default());
        t.compute(Rect::new(0.0, 0.0, 100.0, 100.0), f32::NAN);
        assert_eq!(t.rect(a).w, 50.0, "max below min becomes min");
        assert_eq!(t.parent(b), None, "an unknown parent makes a root");
        assert_eq!(t.rect(NodeId(12345)), Rect::ZERO);
    }
}
