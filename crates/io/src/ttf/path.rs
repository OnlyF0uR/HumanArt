//! Outline paths: the sink a font draws into, an owned path, flattening of quadratic curves and the
//! area / orientation helpers the distance-field generator needs.

/// Receives the outline of a glyph, contour by contour. Coordinates are font units, Y up.
///
/// A contour is one `move_to`, any number of `line_to` / `quad_to`, then `close` (which joins the
/// current point to the contour's start with a straight line when they differ).
pub trait OutlineSink {
    /// Starts a contour at `to`.
    fn move_to(&mut self, to: [f32; 2]);
    /// A straight segment from the current point to `to`.
    fn line_to(&mut self, to: [f32; 2]);
    /// A quadratic Bézier segment from the current point to `to` with the control point `control`.
    fn quad_to(&mut self, control: [f32; 2], to: [f32; 2]);
    /// Ends the contour, closing it back to its start.
    fn close(&mut self);
}

/// One drawing command of a [`Path`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathCommand {
    /// Start a contour at the point.
    MoveTo([f32; 2]),
    /// A straight segment to the point.
    LineTo([f32; 2]),
    /// A quadratic Bézier segment: control point, end point.
    QuadTo([f32; 2], [f32; 2]),
    /// Close the contour back to its start.
    Close,
}

/// Which way a closed contour runs, seen with Y up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    /// Negative signed area. TrueType draws outer contours this way.
    Clockwise,
    /// Positive signed area. TrueType draws holes this way.
    CounterClockwise,
    /// No area (or not a finite one).
    Degenerate,
}

/// The orientation that goes with a signed area.
#[must_use]
pub fn orientation(signed_area: f32) -> Orientation {
    if signed_area > 0.0 {
        Orientation::CounterClockwise
    } else if signed_area < 0.0 {
        Orientation::Clockwise
    } else {
        Orientation::Degenerate
    }
}

/// The signed area of a closed polygon (the last point joins the first): positive when the points
/// run counter-clockwise with Y up, negative when clockwise.
#[must_use]
pub fn signed_area(points: &[[f32; 2]]) -> f32 {
    let Some(&last) = points.last() else { return 0.0 };
    let mut previous = last;
    let mut twice = 0.0_f64;
    for &p in points {
        twice += cross(previous, p);
        previous = p;
    }
    (twice * 0.5) as f32
}

/// `a × b` in double precision, twice the signed area of the triangle (origin, a, b).
fn cross(a: [f32; 2], b: [f32; 2]) -> f64 {
    f64::from(a[0]) * f64::from(b[1]) - f64::from(a[1]) * f64::from(b[0])
}

/// The exact signed area enclosed by the commands of one contour, curves included (no flattening):
/// positive when it runs counter-clockwise with Y up. An unclosed contour is treated as closed.
#[must_use]
pub fn signed_area_of_commands(commands: &[PathCommand]) -> f32 {
    let mut start = [0.0_f32; 2];
    let mut current = start;
    let mut twice = 0.0_f64;
    for command in commands {
        match *command {
            PathCommand::MoveTo(to) => {
                twice += cross(current, start);
                start = to;
                current = to;
            }
            PathCommand::LineTo(to) => {
                twice += cross(current, to);
                current = to;
            }
            PathCommand::QuadTo(control, to) => {
                // The region between a quadratic and its chord is two thirds of the control triangle.
                twice += (2.0 * cross(current, control) + 2.0 * cross(control, to) + cross(current, to)) / 3.0;
                current = to;
            }
            PathCommand::Close => {
                twice += cross(current, start);
                current = start;
            }
        }
    }
    twice += cross(current, start);
    (twice * 0.5) as f32
}

/// The most segments one curve is split into, whatever the tolerance.
const MAX_SEGMENTS: f32 = 1024.0;

/// Appends a polyline approximating the quadratic Bézier `from` → `to` (control point `control`) to
/// `out`, without the start point and ending exactly at `to`.
///
/// The curve is split into equal parameter steps, as few as keep the polyline within `tolerance`
/// (same units as the points) of the curve, and at most 1024; a tolerance that is not positive asks
/// for the maximum.
pub fn flatten_quad(from: [f32; 2], control: [f32; 2], to: [f32; 2], tolerance: f32, out: &mut Vec<[f32; 2]>) {
    // A quadratic strays from its chord by |from − 2·control + to| / 4; n steps divide that by n².
    let ddx = from[0] - 2.0 * control[0] + to[0];
    let ddy = from[1] - 2.0 * control[1] + to[1];
    let deviation = ddx.hypot(ddy) * 0.25;
    let wanted = if tolerance > 0.0 { (deviation / tolerance).sqrt().ceil() } else { MAX_SEGMENTS };
    // NaN (from non-finite input) falls through the comparison chain to a single segment.
    let segments = if wanted >= MAX_SEGMENTS {
        MAX_SEGMENTS as usize
    } else if wanted > 1.0 {
        wanted as usize
    } else {
        1
    };
    for i in 1..segments {
        let t = i as f32 / segments as f32;
        let u = 1.0 - t;
        out.push([
            u * u * from[0] + 2.0 * u * t * control[0] + t * t * to[0],
            u * u * from[1] + 2.0 * u * t * control[1] + t * t * to[1],
        ]);
    }
    out.push(to);
}

/// An owned outline: a flat list of commands, contour after contour.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Path {
    /// The commands, in drawing order.
    pub commands: Vec<PathCommand>,
}

impl OutlineSink for Path {
    fn move_to(&mut self, to: [f32; 2]) {
        self.commands.push(PathCommand::MoveTo(to));
    }

    fn line_to(&mut self, to: [f32; 2]) {
        self.commands.push(PathCommand::LineTo(to));
    }

    fn quad_to(&mut self, control: [f32; 2], to: [f32; 2]) {
        self.commands.push(PathCommand::QuadTo(control, to));
    }

    fn close(&mut self) {
        self.commands.push(PathCommand::Close);
    }
}

impl Path {
    /// Whether the path has no commands (the glyph has no outline).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// The commands of each contour: every slice starts at a `MoveTo` (commands before the first
    /// `MoveTo`, which a font never produces, form a contour of their own).
    pub fn contours(&self) -> impl Iterator<Item = &[PathCommand]> {
        let commands = self.commands.as_slice();
        let mut start = 0;
        std::iter::from_fn(move || {
            if start >= commands.len() {
                return None;
            }
            let rest = &commands[start + 1..];
            let length = 1 + rest.iter().position(|c| matches!(c, PathCommand::MoveTo(_))).unwrap_or(rest.len());
            let contour = &commands[start..start + length];
            start += length;
            Some(contour)
        })
    }

    /// The exact signed area of each contour, curves included; see [`signed_area_of_commands`].
    #[must_use]
    pub fn contour_areas(&self) -> Vec<f32> {
        self.contours().map(signed_area_of_commands).collect()
    }

    /// The path as closed polygons, one per contour, with curves flattened to within `tolerance`
    /// (see [`flatten_quad`]). The first point of a polygon is not repeated at its end.
    #[must_use]
    pub fn flatten(&self, tolerance: f32) -> Vec<Vec<[f32; 2]>> {
        let mut polygons = Vec::new();
        for contour in self.contours() {
            let mut points: Vec<[f32; 2]> = Vec::new();
            let mut current = [0.0_f32; 2];
            for command in contour {
                match *command {
                    PathCommand::MoveTo(to) | PathCommand::LineTo(to) => {
                        points.push(to);
                        current = to;
                    }
                    PathCommand::QuadTo(control, to) => {
                        flatten_quad(current, control, to, tolerance, &mut points);
                        current = to;
                    }
                    PathCommand::Close => {}
                }
            }
            if points.len() > 1 && points.first() == points.last() {
                points.pop();
            }
            polygons.push(points);
        }
        polygons
    }

    /// The smallest box around every point of the path, control points included, as
    /// `[x_min, y_min, x_max, y_max]`; `None` for an empty path. Curves lie inside the hull of their
    /// control points, so this box contains the outline but may be larger than the tight one.
    #[must_use]
    pub fn control_bounds(&self) -> Option<[f32; 4]> {
        let mut bounds: Option<[f32; 4]> = None;
        let mut add = |p: [f32; 2]| {
            let b = bounds.get_or_insert([p[0], p[1], p[0], p[1]]);
            *b = [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])];
        };
        for command in &self.commands {
            match *command {
                PathCommand::MoveTo(to) | PathCommand::LineTo(to) => add(to),
                PathCommand::QuadTo(control, to) => {
                    add(control);
                    add(to);
                }
                PathCommand::Close => {}
            }
        }
        bounds
    }

    /// The exact bounding box of the outline as `[x_min, y_min, x_max, y_max]`: the end points of
    /// every segment and, for a curve that bulges past its end points, the extremum of the curve
    /// itself (not its control point). `None` for an empty path. This is the box a rasteriser or a
    /// distance-field generator has to cover; it lies inside [`Path::control_bounds`].
    #[must_use]
    pub fn bounds(&self) -> Option<[f32; 4]> {
        let mut bounds: Option<[f32; 4]> = None;
        let mut add = |p: [f32; 2]| {
            let b = bounds.get_or_insert([p[0], p[1], p[0], p[1]]);
            *b = [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])];
        };
        let mut current = [0.0_f32; 2];
        for command in &self.commands {
            match *command {
                PathCommand::MoveTo(to) | PathCommand::LineTo(to) => {
                    add(to);
                    current = to;
                }
                PathCommand::QuadTo(control, to) => {
                    add(to);
                    // Per axis the curve is monotonic unless the control value lies outside the
                    // span of the end values; the other axis of the point added is an end value,
                    // which is in the box already.
                    add([
                        quad_extremum(current[0], control[0], to[0]).unwrap_or(to[0]),
                        quad_extremum(current[1], control[1], to[1]).unwrap_or(to[1]),
                    ]);
                    current = to;
                }
                PathCommand::Close => {}
            }
        }
        bounds
    }
}

/// The extreme value strictly inside a one-dimensional quadratic Bézier from `a` to `b` with the
/// control value `c`, if it has one: the value at `t = (a − c) / (a − 2c + b)`, which simplifies to
/// `(a·b − c²) / (a − 2c + b)`. `None` when the curve is monotonic (its extremes are `a` and `b`).
fn quad_extremum(a: f32, c: f32, b: f32) -> Option<f32> {
    if c >= a.min(b) && c <= a.max(b) {
        return None;
    }
    let (a, c, b) = (f64::from(a), f64::from(c), f64::from(b));
    let denominator = a - 2.0 * c + b;
    let value = (a * b - c * c) / denominator;
    value.is_finite().then_some(value as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(clockwise: bool) -> Vec<[f32; 2]> {
        let mut points = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        if clockwise {
            points.reverse();
        }
        points
    }

    #[test]
    fn signed_area_gives_orientation() {
        assert_eq!(signed_area(&square(false)), 100.0);
        assert_eq!(signed_area(&square(true)), -100.0);
        assert_eq!(orientation(signed_area(&square(false))), Orientation::CounterClockwise);
        assert_eq!(orientation(signed_area(&square(true))), Orientation::Clockwise);
        assert_eq!(orientation(signed_area(&[])), Orientation::Degenerate);
        assert_eq!(orientation(signed_area(&[[1.0, 1.0], [2.0, 2.0]])), Orientation::Degenerate);
        assert_eq!(orientation(f32::NAN), Orientation::Degenerate);
    }

    #[test]
    fn flatten_quad_stays_within_tolerance_and_ends_exactly() {
        let (from, control, to) = ([0.0_f32, 0.0], [50.0, 100.0], [100.0, 0.0]);
        for tolerance in [10.0_f32, 1.0, 0.1, 0.01] {
            let mut line = vec![from];
            flatten_quad(from, control, to, tolerance, &mut line);
            assert_eq!(*line.last().unwrap(), to);
            // The curve is y = 2·x·(100 − x)/100: measure the vertical distance of every chord midpoint.
            let mut worst = 0.0_f32;
            for pair in line.windows(2) {
                let (mx, my) = (f32::midpoint(pair[0][0], pair[1][0]), f32::midpoint(pair[0][1], pair[1][1]));
                worst = worst.max((2.0 * mx * (100.0 - mx) / 100.0 - my).abs());
            }
            assert!(worst <= tolerance * 1.001, "tolerance {tolerance}: off by {worst}");
            // And not wastefully fine: half as many segments would break the tolerance.
            let segments = line.len() - 1;
            let deviation = 50.0_f32;
            assert!(segments == 1 || deviation / ((segments - 1) as f32).powi(2) > tolerance, "{segments} segments");
        }
        // A straight "curve" is one segment; a useless tolerance is bounded.
        let mut out = Vec::new();
        flatten_quad([0.0, 0.0], [5.0, 5.0], [10.0, 10.0], 0.1, &mut out);
        assert_eq!(out, vec![[10.0, 10.0]]);
        out.clear();
        flatten_quad(from, control, to, 0.0, &mut out);
        assert_eq!(out.len(), 1024);
        out.clear();
        flatten_quad(from, [f32::NAN, 0.0], to, 1.0, &mut out);
        assert_eq!(out.len(), 1);
        out.clear();
        flatten_quad(from, [f32::INFINITY, 0.0], to, 1.0, &mut out);
        assert_eq!(out.len(), 1024);
    }

    /// A path of two contours: a clockwise quarter-disc-like shape with one curve, and a triangle.
    fn sample() -> Path {
        let mut path = Path::default();
        path.move_to([0.0, 0.0]);
        path.line_to([0.0, 30.0]);
        path.quad_to([30.0, 30.0], [30.0, 0.0]);
        path.close();
        path.move_to([100.0, 0.0]);
        path.line_to([110.0, 0.0]);
        path.line_to([100.0, 10.0]);
        path.close();
        path
    }

    #[test]
    fn contours_areas_and_flattening() {
        let path = sample();
        assert!(!path.is_empty() && Path::default().is_empty());
        let contours: Vec<&[PathCommand]> = path.contours().collect();
        assert_eq!(contours.len(), 2);
        assert_eq!(contours[0].len(), 4);
        assert_eq!(contours[1][0], PathCommand::MoveTo([100.0, 0.0]));
        // Triangle (0,0) (0,30) (30,0) is 450 clockwise; the curve adds two thirds of the control
        // triangle (0,30) (30,30) (30,0), which is 450: 450 + 300 = 750.
        let areas = path.contour_areas();
        assert_eq!(areas, vec![-750.0, 50.0]);
        assert_eq!(orientation(areas[0]), Orientation::Clockwise);
        assert_eq!(orientation(areas[1]), Orientation::CounterClockwise);

        let polygons = path.flatten(0.01);
        assert_eq!(polygons.len(), 2);
        assert_eq!(polygons[1], vec![[100.0, 0.0], [110.0, 0.0], [100.0, 10.0]]);
        assert_eq!(polygons[0][0], [0.0, 0.0]);
        assert!(polygons[0].len() > 10);
        // The flattened polygon's area converges on the exact one from inside.
        let flat = signed_area(&polygons[0]);
        assert!((flat - -750.0).abs() < 0.5 && flat > -750.0, "{flat}");
        assert_eq!(path.control_bounds(), Some([0.0, 0.0, 110.0, 30.0]));
        assert_eq!(path.bounds(), Some([0.0, 0.0, 110.0, 30.0]));
        assert_eq!(Path::default().bounds(), None);
        assert_eq!(Path::default().control_bounds(), None);
        assert!(Path::default().flatten(1.0).is_empty());
    }

    #[test]
    fn exact_bounds_follow_the_curve_not_its_control_points() {
        // An arch from (0, 0) to (100, 0) with the control point at (50, 100) peaks at y = 50; a
        // second curve leans out to the right of both of its end points, a third to the left.
        let mut path = Path::default();
        path.move_to([0.0, 0.0]);
        path.quad_to([50.0, 100.0], [100.0, 0.0]);
        path.quad_to([120.0, -30.0], [100.0, -60.0]);
        path.quad_to([-40.0, -30.0], [0.0, 0.0]);
        path.close();
        assert_eq!(path.control_bounds(), Some([-40.0, -60.0, 120.0, 100.0]));
        // x extremes: (100·100 − 120²)/(200 − 240) = 110 and (100·0 − 40²)/(100 + 80) = −8.888…
        let exact = path.bounds().unwrap();
        assert_eq!([exact[1], exact[2], exact[3]], [-60.0, 110.0, 50.0]);
        assert!((exact[0] - -1600.0 / 180.0).abs() < 1e-5, "{}", exact[0]);
        // The flattened outline never leaves the exact box and comes within the tolerance of it.
        let flat = path.flatten(0.01);
        let mut reached = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
        for p in &flat[0] {
            assert!(p[0] >= exact[0] && p[1] >= exact[1] && p[0] <= exact[2] && p[1] <= exact[3], "{p:?}");
            reached = [reached[0].min(p[0]), reached[1].min(p[1]), reached[2].max(p[0]), reached[3].max(p[1])];
        }
        for (r, e) in reached.iter().zip(exact) {
            assert!((r - e).abs() <= 0.02, "{reached:?} vs {exact:?}");
        }
        // Monotonic curves and non-finite input fall back to the end points.
        assert_eq!(quad_extremum(0.0, 5.0, 10.0), None);
        assert_eq!(quad_extremum(0.0, 0.0, 10.0), None);
        assert_eq!(quad_extremum(0.0, 20.0, 0.0), Some(10.0));
        assert_eq!(quad_extremum(0.0, f32::INFINITY, 0.0), None);
        assert_eq!(quad_extremum(0.0, f32::NAN, 0.0), None);
    }

    #[test]
    fn a_contour_that_returns_to_its_start_does_not_repeat_the_point() {
        let mut path = Path::default();
        path.move_to([0.0, 0.0]);
        path.line_to([4.0, 0.0]);
        path.quad_to([4.0, 4.0], [0.0, 0.0]);
        path.close();
        let polygon = &path.flatten(100.0)[0];
        assert_eq!(polygon, &vec![[0.0, 0.0], [4.0, 0.0]]);
        // Unclosed input and commands before any MoveTo are tolerated.
        let odd = Path { commands: vec![PathCommand::LineTo([1.0, 0.0]), PathCommand::LineTo([1.0, 1.0])] };
        assert_eq!(odd.contours().count(), 1);
        assert_eq!(odd.contour_areas(), vec![0.5]);
    }
}
