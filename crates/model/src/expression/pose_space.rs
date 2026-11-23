//! Pose-space drivers `[SKIN-02]` (todo 1.3): radial basis interpolation over several example poses
//! of one joint, with the rotation split into swing and twist about the bone.
//!
//! A [`crate::expression::PoseCorrective`] on its own fades in around one target rotation with a
//! compact kernel. That is weakest where a joint reaches the same region of skin through very
//! different rotations (the shoulder and the armpit): two correctives whose kernels overlap both
//! switch on half way between their targets and add up. A [`PoseDriver`] groups the correctives of
//! one joint and solves them together, so that
//!
//! - at each example pose (a corrective's `target`) exactly that corrective is fully on and every
//!   other one is off;
//! - at the rest pose (an implicit example) every one is off;
//! - between examples the activations blend smoothly (Gaussian kernel of width
//!   [`PoseDriver::radius`]), and far from every example they fade to zero.
//!
//! **Metrics.** The distance between two rotations compares the part of the rotation the driver is
//! about ([`PoseMetric`]): the **swing** (where the bone points; the twist about the bone, the joint
//! frame's Y axis, removed), the **twist** (the unwound angle about the bone, so that +120° and
//! −120° are 240° apart, not 120°), or the **full** rotation. A shoulder corrective driven by swing
//! stays on when the arm is also rolled; a forearm corrective driven by twist ignores how far the
//! elbow bends.
//!
//! **Solve.** With the examples `p_0 = rest, p_1..p_k` (the drivers' correctives), the kernel matrix
//! `Φ_ij = φ(d(p_i, p_j))`, `φ(d) = exp(−(d / radius)²)`, and the wanted outputs `Y` (row 0 zero, row
//! `i` the unit vector of corrective `i`), the coefficients `W = Φ⁻¹ Y` are computed once
//! ([`PoseSolver::new`], in `f64`); an evaluation is then one kernel row and a `k × (k + 1)` product,
//! with no allocation `[PERF-R1]`. Each activation is clamped to `[0, 1]` and the set is scaled down
//! when it sums above 1.
//!
//! **Reference.** The solver follows the interpolative RBF solver and the swing / twist metrics of
//! Epic Games' OpenRigLogic (`src/riglogic/rbf/cpu/InterpolativeRBFSolver.cpp`, `RBFSolver.cpp`;
//! <https://github.com/EpicGames/OpenRigLogic>), used under its MIT licence:
//!
//! > MIT License. Copyright (c) 2026 Epic Games, Inc.
//! >
//! > Permission is hereby granted, free of charge, to any person obtaining a copy of this software
//! > and associated documentation files (the "Software"), to deal in the Software without
//! > restriction, including without limitation the rights to use, copy, modify, merge, publish,
//! > distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
//! > Software is furnished to do so, subject to the following conditions:
//! >
//! > The above copyright notice and this permission notice shall be included in all copies or
//! > substantial portions of the Software.
//! >
//! > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
//! > BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
//! > NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//! > DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! > OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//!
//! Differences: the kernel is `exp(−(d/r)²)` (OpenRigLogic's Gaussian is `exp(−d/r²)`), the rest pose
//! is always an example with zero output, and the inverse is a pivoted LU in `f64`
//! ([`humanart_math::solve::Lu`]) with a small ridge so coinciding examples are refused, not
//! silently blended.

use humanart_math::Quat;
use humanart_math::solve::{DenseMatrix, Lu};

use super::{ExpressionSet, PoseCorrective};
use crate::error::{ModelError, Result};

/// The most example poses (correctives) one driver solves together. The kernel matrix is
/// `(n + 1)²`; real joints use a handful.
pub const MAX_DRIVER_POSES: usize = 32;

/// Relative ridge added to the kernel matrix's diagonal: keeps the solve well posed for examples
/// that are close but distinct; examples closer than this resolves are refused.
const RIDGE: f64 = 1e-9;

/// Smallest distance, in radians, two examples of one driver (the rest pose included) may be apart.
pub const MIN_EXAMPLE_SEPARATION: f32 = 1e-3;

/// Which part of a joint rotation a driver compares.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PoseMetric {
    /// The swing: the direction of the bone (the joint frame's Y axis), twist removed.
    #[default]
    Swing,
    /// The unwound twist angle about the bone.
    Twist,
    /// The whole rotation.
    Full,
}

impl PoseMetric {
    /// Every metric, in code order.
    pub const ALL: [Self; 3] = [Self::Swing, Self::Twist, Self::Full];

    /// The stored code.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::Swing => 0,
            Self::Twist => 1,
            Self::Full => 2,
        }
    }

    /// The metric of a stored code.
    #[must_use]
    pub const fn from_u8(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Swing),
            1 => Some(Self::Twist),
            2 => Some(Self::Full),
            _ => None,
        }
    }

    /// The stable name used in data files.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Swing => "swing",
            Self::Twist => "twist",
            Self::Full => "full",
        }
    }

    /// The metric with this name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.name() == name)
    }
}

/// The form of a rotation that a metric compares, kept as four numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Coord([f32; 4]);

/// The twist about the bone (the joint frame's Y axis) and the swing of `q`: `q = swing · twist`.
#[must_use]
pub fn swing_twist_y(q: Quat) -> (Quat, Quat) {
    let twist = Quat::from_xyzw(0.0, q.y, 0.0, q.w);
    let len = twist.length();
    let twist = if len > 1e-12 { twist * len.recip() } else { Quat::IDENTITY };
    (q * twist.conjugate(), twist)
}

/// The unwound twist angle of `q` about the bone, in `[−π, π]`.
#[must_use]
pub fn twist_angle_y(q: Quat) -> f32 {
    let (_, t) = swing_twist_y(q);
    let mut a = 2.0 * t.y.atan2(t.w);
    if a > std::f32::consts::PI {
        a -= std::f32::consts::TAU;
    } else if a < -std::f32::consts::PI {
        a += std::f32::consts::TAU;
    }
    a
}

impl PoseMetric {
    fn coord(self, q: Quat) -> Coord {
        let q = q.try_normalize().unwrap_or(Quat::IDENTITY);
        match self {
            Self::Swing => Coord(swing_twist_y(q).0.to_array()),
            Self::Twist => Coord([twist_angle_y(q), 0.0, 0.0, 0.0]),
            Self::Full => Coord(q.to_array()),
        }
    }

    fn distance(self, a: Coord, b: Coord) -> f32 {
        match self {
            Self::Twist => (a.0[0] - b.0[0]).abs(),
            Self::Swing | Self::Full => Quat::from_array(a.0).angle_between(Quat::from_array(b.0)),
        }
    }
}

/// A group of pose correctives on one joint solved together by radial basis interpolation.
#[derive(Clone, Debug, PartialEq)]
pub struct PoseDriver {
    /// Stable name.
    pub name: String,
    /// The driving joint; every corrective of the driver must name it.
    pub joint: u16,
    /// Which part of the rotation is compared.
    pub metric: PoseMetric,
    /// Kernel width, radians, in `(0, π]`.
    pub radius: f32,
    /// Indices into [`ExpressionSet::pose_correctives`], 1 to [`MAX_DRIVER_POSES`], distinct. Each
    /// corrective's `target` is one example pose. A corrective belongs to at most one driver.
    pub correctives: Vec<u32>,
}

impl PoseDriver {
    /// Checks the driver against the correctives of its set (index `i` for messages).
    pub fn check(&self, i: usize, correctives: &[PoseCorrective]) -> Result<()> {
        let bad = |what: &'static str| Err(ModelError::invalid("pose driver", i, what));
        if self.correctives.is_empty() || self.correctives.len() > MAX_DRIVER_POSES {
            return bad("needs 1 to 32 correctives");
        }
        if !(self.radius > 0.0 && self.radius <= std::f32::consts::PI) {
            return bad("radius outside (0, pi]");
        }
        let mut sorted = self.correctives.clone();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return bad("corrective used twice");
        }
        for &c in &self.correctives {
            let Some(pc) = correctives.get(c as usize) else { return bad("corrective index out of range") };
            if pc.joint != self.joint {
                return bad("corrective drives another joint");
            }
        }
        PoseSolver::new(self, correctives).map(|_| ())
    }
}

/// The precomputed solve of one [`PoseDriver`].
#[derive(Clone, Debug, PartialEq)]
pub struct PoseSolver {
    joint: u16,
    metric: PoseMetric,
    radius: f32,
    /// The examples in metric form: the rest pose first, then one per output.
    examples: Vec<Coord>,
    /// Row-major `outputs × examples`.
    coefficients: Vec<f32>,
}

impl PoseSolver {
    /// Solves `driver` once for its correctives (`correctives` is the whole set's list).
    pub fn new(driver: &PoseDriver, correctives: &[PoseCorrective]) -> Result<Self> {
        let invalid = |what: &'static str| ModelError::invalid("pose driver", 0, what);
        let metric = driver.metric;
        let mut examples = Vec::with_capacity(driver.correctives.len() + 1);
        examples.push(metric.coord(Quat::IDENTITY));
        for &c in &driver.correctives {
            let pc = correctives.get(c as usize).ok_or_else(|| invalid("corrective index out of range"))?;
            examples.push(metric.coord(pc.target));
        }
        let n = examples.len();
        let r = driver.radius;
        if !(r > 0.0 && r.is_finite()) {
            return Err(invalid("radius must be positive"));
        }
        for i in 0..n {
            for j in i + 1..n {
                if metric.distance(examples[i], examples[j]) < MIN_EXAMPLE_SEPARATION {
                    return Err(invalid("two example poses coincide (the rest pose counts as one)"));
                }
            }
        }
        let phi = DenseMatrix::from_fn(n, n, |i, j| {
            f64::from(kernel(metric.distance(examples[i], examples[j]), r)) + if i == j { RIDGE } else { 0.0 }
        });
        let lu = Lu::new(&phi).map_err(|_| invalid("kernel matrix cannot be factored"))?;
        let outputs = n - 1;
        let mut coefficients = vec![0.0_f32; outputs * n];
        let mut rhs = vec![0.0_f64; n];
        let mut col = vec![0.0_f64; n];
        for o in 0..outputs {
            rhs.fill(0.0);
            rhs[o + 1] = 1.0;
            lu.solve_into(&rhs, &mut col).map_err(|_| invalid("kernel matrix is singular"))?;
            for (dst, &c) in coefficients[o * n..(o + 1) * n].iter_mut().zip(&col) {
                if !c.is_finite() {
                    return Err(invalid("kernel matrix is singular"));
                }
                *dst = c as f32;
            }
        }
        Ok(Self { joint: driver.joint, metric, radius: r, examples, coefficients })
    }

    /// The driving joint.
    #[must_use]
    pub const fn joint(&self) -> u16 {
        self.joint
    }

    /// The number of outputs (the driver's correctives).
    #[must_use]
    pub fn outputs(&self) -> usize {
        self.examples.len() - 1
    }

    /// The number of examples, the rest pose included (the length of the kernel scratch row).
    #[must_use]
    pub fn examples(&self) -> usize {
        self.examples.len()
    }

    /// Writes the activation of each output for the joint rotation `rotation` (relative to rest)
    /// into `out[..outputs]`, using `phi[..examples]` as scratch. Shorter slices leave the missing
    /// outputs unwritten. No allocation.
    pub fn solve(&self, rotation: Quat, phi: &mut [f32], out: &mut [f32]) {
        let x = self.metric.coord(rotation);
        let n = self.examples.len();
        for (p, e) in phi.iter_mut().zip(&self.examples) {
            *p = kernel(self.metric.distance(x, *e), self.radius);
        }
        let Some(phi) = phi.get(..n) else { return };
        let mut sum = 0.0;
        for (o, slot) in out.iter_mut().enumerate().take(self.outputs()) {
            let row = &self.coefficients[o * n..(o + 1) * n];
            let w: f32 = row.iter().zip(phi).map(|(c, p)| c * p).sum();
            let w = if w.is_finite() { w.clamp(0.0, 1.0) } else { 0.0 };
            *slot = w;
            sum += w;
        }
        if sum > 1.0 {
            let s = sum.recip();
            for slot in out.iter_mut().take(self.outputs()) {
                *slot *= s;
            }
        }
    }
}

fn kernel(d: f32, r: f32) -> f32 {
    let t = d / r;
    (-t * t).exp()
}

/// Every pose corrective's activation for a pose: the drivers' solves for the grouped
/// correctives, each corrective's own kernel ([`PoseCorrective::activation`]) for the others.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PoseSpace {
    solvers: Vec<PoseSolver>,
    /// Per solver, the corrective index of each output.
    outputs: Vec<Vec<u32>>,
    /// Per corrective: whether a driver owns it.
    driven: Vec<bool>,
    phi: Vec<f32>,
    scratch: Vec<f32>,
}

impl PoseSpace {
    /// Solves every driver of `set`.
    pub fn new(set: &ExpressionSet) -> Result<Self> {
        let mut driven = vec![false; set.pose_correctives.len()];
        let mut solvers = Vec::with_capacity(set.pose_drivers.len());
        let mut outputs = Vec::with_capacity(set.pose_drivers.len());
        let mut widest = 0;
        for (i, d) in set.pose_drivers.iter().enumerate() {
            d.check(i, &set.pose_correctives)?;
            for &c in &d.correctives {
                let slot = driven.get_mut(c as usize).ok_or(ModelError::invalid("pose driver", i, "bad corrective"))?;
                if *slot {
                    return Err(ModelError::invalid("pose driver", i, "corrective belongs to two drivers"));
                }
                *slot = true;
            }
            let s = PoseSolver::new(d, &set.pose_correctives)?;
            widest = widest.max(s.examples());
            solvers.push(s);
            outputs.push(d.correctives.clone());
        }
        Ok(Self { solvers, outputs, driven, phi: vec![0.0; widest], scratch: vec![0.0; widest] })
    }

    /// Whether no driver exists (every corrective uses its own kernel).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.solvers.is_empty()
    }

    /// Writes the activation of every corrective of `correctives` (the same list the space was
    /// built from) for the joint rotations `pose` (missing joints at rest) into `out`. No allocation.
    pub fn activations(&mut self, correctives: &[PoseCorrective], pose: &[Quat], out: &mut [f32]) {
        let rotation = |j: u16| pose.get(j as usize).and_then(|q| q.try_normalize()).unwrap_or(Quat::IDENTITY);
        for (i, (pc, slot)) in correctives.iter().zip(out.iter_mut()).enumerate() {
            if !self.driven.get(i).copied().unwrap_or(false) {
                *slot = pc.activation(rotation(pc.joint));
            }
        }
        for (s, outs) in self.solvers.iter().zip(&self.outputs) {
            s.solve(rotation(s.joint()), &mut self.phi, &mut self.scratch);
            for (&c, &a) in outs.iter().zip(&self.scratch) {
                if let Some(slot) = out.get_mut(c as usize) {
                    *slot = a;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::{FRAC_PI_2, PI};

    use humanart_core::rng::Rng;
    use humanart_math::Vec3;

    use super::*;

    fn corrective(name: &str, joint: u16, target: Quat) -> PoseCorrective {
        PoseCorrective { name: name.into(), joint, target, radius: 1.0, component: 0 }
    }

    fn driver(metric: PoseMetric, radius: f32, n: usize) -> PoseDriver {
        PoseDriver { name: "d".into(), joint: 0, metric, radius, correctives: (0..n as u32).collect() }
    }

    fn solve(s: &PoseSolver, q: Quat) -> Vec<f32> {
        let mut phi = vec![0.0; s.examples()];
        let mut out = vec![0.0; s.outputs()];
        s.solve(q, &mut phi, &mut out);
        out
    }

    #[test]
    fn todo_1_3_examples_interpolate_exactly_and_rest_is_off() {
        let targets = [
            Quat::from_rotation_z(FRAC_PI_2),
            Quat::from_rotation_x(FRAC_PI_2),
            Quat::from_rotation_z(-FRAC_PI_2),
            Quat::from_rotation_x(-1.0),
        ];
        let cs: Vec<_> = targets.iter().enumerate().map(|(i, &t)| corrective(&format!("c{i}"), 0, t)).collect();
        let s = PoseSolver::new(&driver(PoseMetric::Swing, 1.2, 4), &cs).unwrap();
        for (i, &t) in targets.iter().enumerate() {
            let a = solve(&s, t);
            for (j, &x) in a.iter().enumerate() {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((x - want).abs() < 1e-3, "example {i} output {j}: {x}");
            }
        }
        assert!(solve(&s, Quat::IDENTITY).iter().all(|&x| x.abs() < 1e-4));
        // Far from every example (a half turn): everything fades.
        let far = solve(&s, Quat::from_rotation_z(PI));
        assert!(far.iter().all(|&x| x < 0.2), "{far:?}");
    }

    #[test]
    fn todo_1_3_swing_ignores_twist_and_twist_ignores_swing() {
        let bend = Quat::from_rotation_z(1.2);
        let cs = [corrective("bend", 0, bend)];
        let s = PoseSolver::new(&driver(PoseMetric::Swing, 1.0, 1), &cs).unwrap();
        // The same swing with the bone rolled 60 degrees: still fully on.
        let rolled = bend * Quat::from_rotation_y(1.05);
        assert!((solve(&s, rolled)[0] - 1.0).abs() < 1e-3);
        let roll = Quat::from_rotation_y(1.5);
        let cs = [corrective("roll", 0, roll)];
        let t = PoseSolver::new(&driver(PoseMetric::Twist, 1.0, 1), &cs).unwrap();
        assert!((solve(&t, Quat::from_rotation_x(0.9) * roll)[0] - 1.0).abs() < 1e-3);
        // A full metric sees both.
        let f = PoseSolver::new(&driver(PoseMetric::Full, 1.0, 1), &cs).unwrap();
        assert!(solve(&f, Quat::from_rotation_x(0.9) * roll)[0] < 0.7);
    }

    #[test]
    fn todo_1_3_twist_is_unwound_past_a_half_turn() {
        // +120 and -120 degrees are 240 degrees apart, so each example stays its own.
        let a = Quat::from_rotation_y(2.0 * PI / 3.0);
        let b = Quat::from_rotation_y(-2.0 * PI / 3.0);
        assert!((twist_angle_y(a) - 2.0 * PI / 3.0).abs() < 1e-4);
        assert!((twist_angle_y(b) + 2.0 * PI / 3.0).abs() < 1e-4);
        assert!((twist_angle_y(-a) - 2.0 * PI / 3.0).abs() < 1e-4, "q and -q have one twist");
        let cs = [corrective("p", 0, a), corrective("n", 0, b)];
        let s = PoseSolver::new(&driver(PoseMetric::Twist, 1.0, 2), &cs).unwrap();
        let at_a = solve(&s, a);
        assert!(at_a[0] > 0.99 && at_a[1] < 0.01, "{at_a:?}");
    }

    #[test]
    fn todo_1_3_activations_are_bounded_and_continuous() {
        let cs = [
            corrective("a", 0, Quat::from_rotation_z(0.8)),
            corrective("b", 0, Quat::from_rotation_z(1.6)),
            corrective("c", 0, Quat::from_rotation_x(1.0)),
        ];
        let s = PoseSolver::new(&driver(PoseMetric::Swing, 0.9, 3), &cs).unwrap();
        let mut prev = solve(&s, Quat::IDENTITY);
        let steps = 400;
        for k in 1..=steps {
            let q = Quat::from_rotation_z(2.4 * k as f32 / steps as f32);
            let a = solve(&s, q);
            assert!(a.iter().all(|&x| (0.0..=1.0).contains(&x)));
            assert!(a.iter().sum::<f32>() <= 1.0 + 1e-5);
            for (x, y) in a.iter().zip(&prev) {
                assert!((x - y).abs() < 0.05, "jump at step {k}: {a:?} after {prev:?}");
            }
            prev = a;
        }
    }

    #[test]
    fn todo_1_3_bad_drivers_are_refused() {
        let cs = [corrective("a", 0, Quat::from_rotation_z(0.8)), corrective("b", 1, Quat::from_rotation_z(1.0))];
        let ok = driver(PoseMetric::Swing, 1.0, 1);
        assert!(ok.check(0, &cs).is_ok());
        let cases: [(&str, PoseDriver); 6] = [
            ("empty", PoseDriver { correctives: vec![], ..ok.clone() }),
            ("radius", PoseDriver { radius: 0.0, ..ok.clone() }),
            ("radius nan", PoseDriver { radius: f32::NAN, ..ok.clone() }),
            ("twice", PoseDriver { correctives: vec![0, 0], ..ok.clone() }),
            ("range", PoseDriver { correctives: vec![5], ..ok.clone() }),
            ("joint", PoseDriver { correctives: vec![0, 1], ..ok.clone() }),
        ];
        for (what, d) in cases {
            assert!(d.check(0, &cs).is_err(), "{what} accepted");
        }
        // An example at the rest pose coincides with the implicit rest example.
        let at_rest = [corrective("r", 0, Quat::IDENTITY)];
        assert!(ok.check(0, &at_rest).is_err());
        // Two examples with the same swing coincide under the swing metric.
        let same = [corrective("a", 0, Quat::from_rotation_z(0.8)), {
            corrective("b", 0, Quat::from_rotation_z(0.8) * Quat::from_rotation_y(1.0))
        }];
        assert!(driver(PoseMetric::Swing, 1.0, 2).check(0, &same).is_err());
        assert!(driver(PoseMetric::Full, 1.0, 2).check(0, &same).is_ok());
    }

    #[test]
    fn todo_1_3_pose_space_mixes_driven_and_free_correctives() {
        let set = ExpressionSet {
            pose_correctives: vec![
                corrective("free", 1, Quat::from_rotation_x(1.0)),
                corrective("a", 0, Quat::from_rotation_z(1.0)),
                corrective("b", 0, Quat::from_rotation_z(-1.0)),
            ],
            pose_drivers: vec![PoseDriver {
                name: "d".into(),
                joint: 0,
                metric: PoseMetric::Swing,
                radius: 1.0,
                correctives: vec![1, 2],
            }],
            ..ExpressionSet::default()
        };
        let mut space = PoseSpace::new(&set).unwrap();
        let mut out = [9.0; 3];
        space.activations(&set.pose_correctives, &[Quat::from_rotation_z(-1.0), Quat::from_rotation_x(1.0)], &mut out);
        assert!((out[0] - 1.0).abs() < 1e-5, "free corrective at its target: {out:?}");
        assert!(out[1] < 1e-3 && (out[2] - 1.0).abs() < 1e-3, "{out:?}");
        // A corrective in two drivers is refused.
        let mut twice = set.clone();
        twice.pose_drivers.push(PoseDriver { name: "e".into(), correctives: vec![2], ..twice.pose_drivers[0].clone() });
        assert!(PoseSpace::new(&twice).is_err());
    }

    #[test]
    fn rob_01_random_drivers_never_give_non_finite_or_out_of_range_activations() {
        let mut rng = Rng::new(0x5eed_0130);
        for case in 0..200 {
            let n = 1 + (rng.next_u32() % 6) as usize;
            let mut cs = Vec::new();
            for i in 0..n {
                let axis = Vec3::new(rng.range_f32(-1.0, 1.0), rng.range_f32(-1.0, 1.0), rng.range_f32(-1.0, 1.0));
                let q = Quat::from_axis_angle(axis.normalize_or(Vec3::X), rng.range_f32(0.05, 3.0));
                cs.push(corrective(&format!("c{i}"), 0, q));
            }
            let metric = PoseMetric::ALL[(rng.next_u32() % 3) as usize];
            let d = driver(metric, rng.range_f32(0.05, 3.1), n);
            let Ok(s) = PoseSolver::new(&d, &cs) else { continue };
            for _ in 0..20 {
                let q = Quat::from_xyzw(
                    rng.range_f32(-1.0, 1.0),
                    rng.range_f32(-1.0, 1.0),
                    rng.range_f32(-1.0, 1.0),
                    rng.range_f32(-1.0, 1.0),
                );
                let a = solve(&s, q);
                assert!(a.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x)), "case {case}: {a:?}");
            }
        }
    }
}
