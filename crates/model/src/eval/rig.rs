//! The rig logic `[EXP-03]`: a data-driven graph from named controls to expression weights.
//!
//! **Model.** A rig has `C` named controls and `N` nodes. Their values live in one register file:
//! registers `0..C` hold the controls, register `C + k` holds node `k`. A node reads only registers
//! below its own (controls and earlier nodes), so evaluation is one pass in node order, with no
//! recursion and no cycles by construction. Node operations:
//!
//! | op | inputs | value |
//! |---|---|---|
//! | `linear` | 1..=64 `(register, weight)` | `bias + Σ weight·r` |
//! | `clamp` | 1 | `clamp(r, min, max)` |
//! | `product` | 1..=64 | `Π r` |
//! | `min` / `max` | 1..=64 | smallest / largest `r` |
//! | `curve` | 1 | piecewise-linear through 2..=64 points `(x, y)` with strictly increasing `x`, constant beyond the ends |
//!
//! Every intermediate value is saturated to `±`[`MAX_WEIGHT`] (products step by step), so finite
//! controls always give finite weights. **Outputs** map nodes or controls to expressions (at most
//! one output per expression); [`RigLogic::evaluate`] writes those expressions' weights and leaves
//! the others as the caller set them, so directly driven expressions can be mixed in.
//!
//! The arrays are flat (SoA: op codes, parameters, argument ranges into shared register/weight
//! arrays, curve ranges into shared `x`/`y` arrays) so the same tables upload to a GPU pass.
//!
//! **Data file** (JSON, e.g. the template's `rig_logic` metadata or a `rig.json`):
//!
//! ```json
//! {
//!   "controls": ["brow_down_L", "smile_L"],
//!   "nodes": [
//!     {"name": "smile_L_lin", "op": "linear", "inputs": [["smile_L", 1.2]], "bias": 0.0},
//!     {"name": "smile_L_w", "op": "clamp", "input": "smile_L_lin", "min": 0, "max": 1},
//!     {"name": "smile_L_ease", "op": "curve", "input": "smile_L_w", "points": [[0, 0], [0.5, 0.35], [1, 1]]},
//!     {"name": "frown", "op": "min", "inputs": ["brow_down_L", "smile_L_w"]}
//!   ],
//!   "outputs": {"smile-L": "smile_L_ease", "brow-lower-L": "brow_down_L"}
//! }
//! ```
//!
//! Names follow the stable-name rules of [`crate::names`] and are unique across controls and nodes;
//! output keys are expression names of the template. Unknown members are refused.

use std::collections::BTreeMap;

use humanart_io::json::{Node, Value};

use super::error::{EvalError, EvalResult};
use super::evaluator::MAX_WEIGHT;
use crate::expression::ExpressionSet;
use crate::names::check_name;

/// The most controls a rig may have.
pub const MAX_CONTROLS: usize = 4096;
/// The most nodes a rig may have.
pub const MAX_NODES: usize = 1 << 16;
/// The most inputs of one node.
pub const MAX_NODE_INPUTS: usize = 64;
/// The most points of one curve.
pub const MAX_CURVE_POINTS: usize = 64;
/// The largest magnitude of a linear weight, bias, clamp bound or curve coordinate.
pub const MAX_PARAMETER: f32 = 1.0e3;

/// A node operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RigOp {
    /// `bias + Σ weight·input`.
    Linear,
    /// `clamp(input, min, max)`.
    Clamp,
    /// The product of the inputs.
    Product,
    /// The smallest input.
    Min,
    /// The largest input.
    Max,
    /// A piecewise-linear curve of the input.
    Curve,
}

impl RigOp {
    /// The data-file name of the operation.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Clamp => "clamp",
            Self::Product => "product",
            Self::Min => "min",
            Self::Max => "max",
            Self::Curve => "curve",
        }
    }

    const ALL: [Self; 6] = [Self::Linear, Self::Clamp, Self::Product, Self::Min, Self::Max, Self::Curve];

    const fn single_input(self) -> bool {
        matches!(self, Self::Clamp | Self::Curve)
    }
}

/// The description of one node for [`RigLogic::add_node`].
#[derive(Clone, Debug, PartialEq)]
pub struct NodeSpec<'a> {
    /// Stable name of the node's register.
    pub name: &'a str,
    /// The operation.
    pub op: RigOp,
    /// Input registers with their weights (weights are used by `linear` only).
    pub inputs: &'a [(u32, f32)],
    /// `linear`: the bias; `clamp`: the minimum; otherwise unused.
    pub a: f32,
    /// `clamp`: the maximum; otherwise unused.
    pub b: f32,
    /// `curve`: the points.
    pub points: &'a [(f32, f32)],
}

/// A checked rig-logic graph, flat arrays.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RigLogic {
    names: Vec<String>,
    controls: usize,
    ops: Vec<RigOp>,
    params: Vec<[f32; 2]>,
    arg_offsets: Vec<u32>,
    arg_registers: Vec<u32>,
    arg_weights: Vec<f32>,
    curve_offsets: Vec<u32>,
    curve_x: Vec<f32>,
    curve_y: Vec<f32>,
    output_expressions: Vec<u32>,
    output_registers: Vec<u32>,
    lookup: BTreeMap<String, u32>,
}

impl RigLogic {
    /// A rig with the given controls and no nodes or outputs.
    pub fn new<'a>(controls: impl IntoIterator<Item = &'a str>) -> EvalResult<Self> {
        let mut rig = Self { arg_offsets: vec![0], curve_offsets: vec![0], ..Self::default() };
        for name in controls {
            if rig.names.len() >= MAX_CONTROLS {
                return Err(EvalError::limit("rig control count", MAX_CONTROLS, rig.names.len() + 1));
            }
            rig.add_name("rig control", name)?;
        }
        rig.controls = rig.names.len();
        Ok(rig)
    }

    fn add_name(&mut self, table: &'static str, name: &str) -> EvalResult<u32> {
        check_name(table, name)?;
        let reg = self.names.len() as u32;
        if self.lookup.insert(name.to_owned(), reg).is_some() {
            return Err(crate::error::ModelError::Name { table, name: name.to_owned(), what: "used twice" }.into());
        }
        self.names.push(name.to_owned());
        Ok(reg)
    }

    /// Appends a node and returns its register.
    pub fn add_node(&mut self, spec: &NodeSpec<'_>) -> EvalResult<u32> {
        let index = self.ops.len();
        if index >= MAX_NODES {
            return Err(EvalError::limit("rig node count", MAX_NODES, index + 1));
        }
        let reg_limit = self.names.len() as u32;
        let n = spec.inputs.len();
        if n == 0 || n > MAX_NODE_INPUTS || (spec.op.single_input() && n != 1) {
            return Err(EvalError::Range { what: "rig node", index, rule: "wrong number of inputs for its op" });
        }
        let bounded = |x: f32| x.is_finite() && x.abs() <= MAX_PARAMETER;
        if spec.inputs.iter().any(|&(r, _)| r >= reg_limit) {
            return Err(EvalError::Missing { what: "earlier rig register", index });
        }
        if !spec.inputs.iter().all(|&(_, w)| bounded(w)) || !bounded(spec.a) || !bounded(spec.b) {
            return Err(EvalError::Range {
                what: "rig node",
                index,
                rule: "parameter not finite or above MAX_PARAMETER",
            });
        }
        if spec.op == RigOp::Clamp && spec.a > spec.b {
            return Err(EvalError::Range { what: "rig node", index, rule: "clamp minimum above maximum" });
        }
        if spec.op == RigOp::Curve {
            let p = spec.points;
            if !(2..=MAX_CURVE_POINTS).contains(&p.len())
                || !p.iter().all(|&(x, y)| bounded(x) && bounded(y))
                || p.windows(2).any(|w| w[0].0 >= w[1].0)
            {
                return Err(EvalError::Range {
                    what: "rig node",
                    index,
                    rule: "curve needs 2..=64 bounded points with increasing x",
                });
            }
        }
        let reg = self.add_name("rig node", spec.name)?;
        self.ops.push(spec.op);
        self.params.push([spec.a, spec.b]);
        for &(r, w) in spec.inputs {
            self.arg_registers.push(r);
            self.arg_weights.push(w);
        }
        self.arg_offsets.push(self.arg_registers.len() as u32);
        if spec.op == RigOp::Curve {
            for &(x, y) in spec.points {
                self.curve_x.push(x);
                self.curve_y.push(y);
            }
        }
        self.curve_offsets.push(self.curve_x.len() as u32);
        Ok(reg)
    }

    /// Drives expression `expression` (an index below `expression_count`) from `register`.
    pub fn add_output(&mut self, expression: u32, register: u32, expression_count: usize) -> EvalResult<()> {
        let index = self.output_expressions.len();
        if expression as usize >= expression_count {
            return Err(EvalError::Missing { what: "expression", index });
        }
        if register as usize >= self.names.len() {
            return Err(EvalError::Missing { what: "rig register", index });
        }
        if self.output_expressions.contains(&expression) {
            return Err(EvalError::Range { what: "rig output", index, rule: "expression driven twice" });
        }
        self.output_expressions.push(expression);
        self.output_registers.push(register);
        Ok(())
    }

    /// Reads a rig from its JSON form (see the module comment), resolving output names against
    /// `expressions`.
    pub fn from_json(value: &Value, expressions: &ExpressionSet) -> EvalResult<Self> {
        let root = value.node();
        root.deny_unknown(&["controls", "nodes", "outputs"])?;
        let controls = root.field("controls")?;
        if controls.array_len()? > MAX_CONTROLS {
            return Err(EvalError::limit("rig control count", MAX_CONTROLS, controls.array_len()?));
        }
        let mut names: Vec<&str> = Vec::with_capacity(controls.array_len()?);
        for c in controls.elements()? {
            names.push(c.str()?);
        }
        let mut rig = Self::new(names)?;
        let mut inputs: Vec<(u32, f32)> = Vec::new();
        let mut points: Vec<(f32, f32)> = Vec::new();
        if let Some(nodes) = root.opt_field("nodes")? {
            if nodes.array_len()? > MAX_NODES {
                return Err(EvalError::limit("rig node count", MAX_NODES, nodes.array_len()?));
            }
            for node in nodes.elements()? {
                node.deny_unknown(&["name", "op", "inputs", "input", "bias", "min", "max", "points"])?;
                let op = node.field("op")?.choice(&RigOp::ALL.map(|o| (o.name(), o)))?;
                inputs.clear();
                points.clear();
                rig.read_inputs(&node, op, &mut inputs)?;
                let (a, b) = match op {
                    RigOp::Linear => (node.get_or("bias", 0.0_f32)?, 0.0),
                    RigOp::Clamp => (node.get_or("min", 0.0_f32)?, node.get_or("max", 1.0_f32)?),
                    _ => (0.0, 0.0),
                };
                if op == RigOp::Curve {
                    let list = node.field("points")?;
                    if list.array_len()? > MAX_CURVE_POINTS {
                        return Err(list.invalid("too many curve points").into());
                    }
                    for p in list.elements()? {
                        if p.array_len()? != 2 {
                            return Err(p.invalid("a curve point is [x, y]").into());
                        }
                        points.push((p.at(0)?.f32()?, p.at(1)?.f32()?));
                    }
                }
                let name = node.field("name")?.str()?;
                rig.add_node(&NodeSpec { name, op, inputs: &inputs, a, b, points: &points })?;
            }
        }
        if let Some(outputs) = root.opt_field("outputs")? {
            for (expression, from) in outputs.members()? {
                let Some(e) = expressions.expressions.iter().position(|x| x.name == expression) else {
                    return Err(from.invalid("no expression of this name").into());
                };
                let reg = rig.register(from.str()?).ok_or_else(|| from.invalid("no control or node of this name"))?;
                rig.add_output(e as u32, reg, expressions.expressions.len())?;
            }
        }
        Ok(rig)
    }

    fn read_inputs(&self, node: &Node<'_, '_>, op: RigOp, out: &mut Vec<(u32, f32)>) -> EvalResult<()> {
        let resolve = |n: &Node<'_, '_>| -> EvalResult<u32> {
            let name = n.str()?;
            Ok(self.register(name).ok_or_else(|| n.invalid("no earlier control or node of this name"))?)
        };
        if op.single_input() {
            out.push((resolve(&node.field("input")?)?, 1.0));
            return Ok(());
        }
        let list = node.field("inputs")?;
        if list.array_len()? > MAX_NODE_INPUTS {
            return Err(list.invalid("too many inputs").into());
        }
        for item in list.elements()? {
            if op == RigOp::Linear {
                if item.array_len()? != 2 {
                    return Err(item.invalid("a linear input is [name, weight]").into());
                }
                out.push((resolve(&item.at(0)?)?, item.at(1)?.f32()?));
            } else {
                out.push((resolve(&item)?, 1.0));
            }
        }
        Ok(())
    }

    /// The register of the control or node called `name`.
    #[must_use]
    pub fn register(&self, name: &str) -> Option<u32> {
        self.lookup.get(name).copied()
    }

    /// How many controls there are.
    #[must_use]
    pub fn control_count(&self) -> usize {
        self.controls
    }

    /// The control names, in register order.
    #[must_use]
    pub fn control_names(&self) -> &[String] {
        &self.names[..self.controls]
    }

    /// How many registers an evaluation needs (controls + nodes).
    #[must_use]
    pub fn register_count(&self) -> usize {
        self.names.len()
    }

    /// The node operations, in order.
    #[must_use]
    pub fn ops(&self) -> &[RigOp] {
        &self.ops
    }

    /// The `(expression, register)` outputs.
    pub fn outputs(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.output_expressions.iter().copied().zip(self.output_registers.iter().copied())
    }

    /// Evaluates the rig: `controls` (at most [`control_count`](Self::control_count); missing
    /// controls are 0) through the nodes into `registers` (at least
    /// [`register_count`](Self::register_count) entries, scratch the caller keeps), then writes
    /// every output expression's weight into `expressions`. Allocation-free.
    pub fn evaluate(&self, controls: &[f32], registers: &mut [f32], expressions: &mut [f32]) -> EvalResult<()> {
        if controls.len() > self.controls {
            return Err(EvalError::Length { what: "rig controls", limit: self.controls, got: controls.len() });
        }
        if registers.len() < self.names.len() {
            return Err(EvalError::Length {
                what: "rig registers (too few)",
                limit: self.names.len(),
                got: registers.len(),
            });
        }
        if let Some(i) = controls.iter().position(|c| !c.is_finite()) {
            return Err(EvalError::NonFinite { what: "rig control", index: i });
        }
        if let Some(i) = self.output_expressions.iter().position(|&e| e as usize >= expressions.len()) {
            return Err(EvalError::Missing { what: "expression weight slot", index: i });
        }
        let sat = |x: f32| x.clamp(-MAX_WEIGHT, MAX_WEIGHT);
        for (r, slot) in registers.iter_mut().take(self.controls).enumerate() {
            *slot = sat(controls.get(r).copied().unwrap_or(0.0));
        }
        for (k, &op) in self.ops.iter().enumerate() {
            let args = self.arg_offsets[k] as usize..self.arg_offsets[k + 1] as usize;
            let regs = &self.arg_registers[args.clone()];
            let weights = &self.arg_weights[args];
            let (done, rest) = registers.split_at_mut(self.controls + k);
            let x = |i: usize| done.get(regs[i] as usize).copied().unwrap_or(0.0);
            let [a, b] = self.params[k];
            let value = match op {
                RigOp::Linear => (0..regs.len()).fold(a, |s, i| s + weights[i] * x(i)),
                RigOp::Clamp => x(0).clamp(a, b),
                RigOp::Product => (0..regs.len()).fold(1.0, |p, i| sat(p * x(i))),
                RigOp::Min => (0..regs.len()).map(x).fold(f32::INFINITY, f32::min),
                RigOp::Max => (0..regs.len()).map(x).fold(f32::NEG_INFINITY, f32::max),
                RigOp::Curve => {
                    let c = self.curve_offsets[k] as usize..self.curve_offsets[k + 1] as usize;
                    piecewise_linear(&self.curve_x[c.clone()], &self.curve_y[c], x(0))
                }
            };
            if let Some(slot) = rest.first_mut() {
                *slot = sat(value);
            }
        }
        for (&e, &r) in self.output_expressions.iter().zip(&self.output_registers) {
            expressions[e as usize] = registers.get(r as usize).copied().unwrap_or(0.0);
        }
        Ok(())
    }
}

/// The piecewise-linear function through `(xs[i], ys[i])` (`xs` strictly increasing) at `x`,
/// constant beyond the ends; 0 for no points.
#[must_use]
pub fn piecewise_linear(xs: &[f32], ys: &[f32], x: f32) -> f32 {
    let n = xs.len().min(ys.len());
    if n == 0 {
        return 0.0;
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    // The first point with xs[i] > x; 1 <= i < n here.
    let i = xs[..n].partition_point(|&p| p <= x).clamp(1, n - 1);
    let (x0, x1, y0, y1) = (xs[i - 1], xs[i], ys[i - 1], ys[i]);
    let t = (x - x0) / (x1 - x0);
    y0 + (y1 - y0) * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::tests::face_patch;
    use humanart_io::json::parse_str;

    const RIG: &str = r#"{
        "controls": ["smile_L", "brow_up"],
        "nodes": [
            {"name": "smile_lin", "op": "linear", "inputs": [["smile_L", 1.5]], "bias": -0.25},
            {"name": "smile_w", "op": "clamp", "input": "smile_lin", "min": 0, "max": 1},
            {"name": "brow_ease", "op": "curve", "input": "brow_up", "points": [[0, 0], [0.5, 0.25], [1, 1]]},
            {"name": "both", "op": "product", "inputs": ["smile_w", "brow_ease"]},
            {"name": "least", "op": "min", "inputs": ["smile_w", "brow_ease"]},
            {"name": "most", "op": "max", "inputs": ["smile_w", "brow_ease"]}
        ],
        "outputs": {"smile-L": "smile_w", "brow-raise": "brow_ease", "smile-R": "most"}
    }"#;

    #[test]
    fn exp_03_rig_from_json_evaluates_every_op() {
        let t = face_patch(4, 0);
        let rig = RigLogic::from_json(&parse_str(RIG).unwrap(), t.expressions()).unwrap();
        assert_eq!(rig.control_count(), 2);
        assert_eq!(rig.register_count(), 8);
        assert_eq!(rig.control_names(), &["smile_L".to_string(), "brow_up".to_string()]);
        let mut regs = vec![0.0; rig.register_count()];
        let mut ex = vec![-1.0; 4];
        rig.evaluate(&[0.5, 0.75], &mut regs, &mut ex).unwrap();
        // smile: 1.5 * 0.5 - 0.25 = 0.5; brow: halfway between (0.5, 0.25) and (1, 1) = 0.625.
        assert!((ex[0] - 0.5).abs() < 1e-6);
        assert!((ex[2] - 0.625).abs() < 1e-6);
        assert!((ex[1] - 0.625).abs() < 1e-6, "max");
        assert_eq!(ex[3], -1.0, "undriven expression untouched");
        let both = rig.register("both").unwrap() as usize;
        let least = rig.register("least").unwrap() as usize;
        assert!((regs[both] - 0.3125).abs() < 1e-6);
        assert!((regs[least] - 0.5).abs() < 1e-6);
        // Clamping and curve ends.
        rig.evaluate(&[3.0, -2.0], &mut regs, &mut ex).unwrap();
        assert_eq!(ex[0], 1.0);
        assert_eq!(ex[2], 0.0);
        rig.evaluate(&[], &mut regs, &mut ex).unwrap();
        assert_eq!(ex[0], 0.0);
        assert_eq!(rig.outputs().count(), 3);
        assert_eq!(rig.ops()[0], RigOp::Linear);
    }

    #[test]
    fn exp_03_extreme_controls_stay_finite() {
        let mut rig = RigLogic::new(["a"]).unwrap();
        let mut reg = 0;
        for k in 0..20 {
            let name = format!("p{k}");
            reg = rig
                .add_node(&NodeSpec {
                    name: &name,
                    op: RigOp::Product,
                    inputs: &[(reg, 1.0), (reg, 1.0)],
                    a: 0.0,
                    b: 0.0,
                    points: &[],
                })
                .unwrap();
        }
        rig.add_output(0, reg, 1).unwrap();
        let mut regs = vec![0.0; rig.register_count()];
        let mut ex = [0.0];
        for c in [f32::MAX, -f32::MAX, 1e30, -3.0] {
            rig.evaluate(&[c], &mut regs, &mut ex).unwrap();
            assert!(ex[0].is_finite() && ex[0].abs() <= MAX_WEIGHT);
        }
        assert!(rig.evaluate(&[f32::NAN], &mut regs, &mut ex).is_err());
        assert!(rig.evaluate(&[1.0, 2.0], &mut regs, &mut ex).is_err());
        assert!(rig.evaluate(&[1.0], &mut regs[..1], &mut ex).is_err());
        assert!(rig.evaluate(&[1.0], &mut regs, &mut []).is_err());
    }

    #[test]
    fn bad_graphs_are_refused() {
        let t = face_patch(4, 0);
        let bad = [
            r#"{"controls": ["a", "a"]}"#,
            r#"{"controls": ["a"], "nodes": [{"name": "n", "op": "linear", "inputs": [["b", 1]]}]}"#,
            r#"{"controls": ["a"], "nodes": [{"name": "n", "op": "linear", "inputs": [["n", 1]]}]}"#,
            r#"{"controls": ["a"], "nodes": [{"name": "n", "op": "clamp", "input": "a", "min": 2, "max": 1}]}"#,
            r#"{"controls": ["a"], "nodes": [{"name": "n", "op": "curve", "input": "a", "points": [[1, 0], [0, 1]]}]}"#,
            r#"{"controls": ["a"], "nodes": [{"name": "n", "op": "curve", "input": "a", "points": [[0, 0]]}]}"#,
            r#"{"controls": ["a"], "nodes": [{"name": "n", "op": "product", "inputs": []}]}"#,
            r#"{"controls": ["a"], "nodes": [{"name": "n", "op": "linear", "inputs": [["a", 1e9]]}]}"#,
            r#"{"controls": ["a"], "nodes": [{"name": "n", "op": "spline", "input": "a"}]}"#,
            r#"{"controls": ["a"], "outputs": {"no-such-expression": "a"}}"#,
            r#"{"controls": ["a"], "outputs": {"smile-L": "b"}}"#,
            r#"{"controls": ["a"], "extra": 1}"#,
            r#"{"controls": ["bad name"]}"#,
        ];
        for (i, text) in bad.iter().enumerate() {
            let v = parse_str(text).unwrap();
            assert!(RigLogic::from_json(&v, t.expressions()).is_err(), "case {i} accepted");
        }
        let mut rig = RigLogic::new(["a"]).unwrap();
        rig.add_output(0, 0, 2).unwrap();
        assert!(rig.add_output(0, 0, 2).is_err(), "expression driven twice");
        assert!(rig.add_output(1, 9, 2).is_err());
        assert!(rig.add_output(5, 0, 2).is_err());
    }

    #[test]
    fn piecewise_linear_edges() {
        assert_eq!(piecewise_linear(&[], &[], 1.0), 0.0);
        assert_eq!(piecewise_linear(&[0.0], &[2.0], 1.0), 2.0);
        let (xs, ys) = ([0.0, 1.0, 2.0], [0.0, 10.0, 0.0]);
        assert_eq!(piecewise_linear(&xs, &ys, 1.0), 10.0);
        assert_eq!(piecewise_linear(&xs, &ys, 1.5), 5.0);
        assert_eq!(piecewise_linear(&xs, &ys, 9.0), 0.0);
        assert_eq!(piecewise_linear(&xs, &ys, -9.0), 0.0);
    }

    /// `[ROB-01]`: mutated rig files never panic or hang; accepted ones evaluate finitely.
    #[test]
    fn rig_json_mutation_fuzz() {
        let expressions = face_patch(2, 0).expressions().clone();
        let corpus = vec![RIG.as_bytes().to_vec(), br#"{"controls": ["a"], "nodes": [], "outputs": {}}"#.to_vec()];
        humanart_core::fuzz::run(
            "rig-logic",
            &corpus,
            3000,
            0x0E03_F022,
            std::time::Duration::from_secs(60),
            move |bytes| {
                let Ok(v) = humanart_io::json::parse(bytes) else { return };
                if let Ok(rig) = RigLogic::from_json(&v, &expressions) {
                    let mut regs = vec![0.0; rig.register_count()];
                    let mut ex = vec![0.0; expressions.expressions.len()];
                    let controls = vec![0.7; rig.control_count()];
                    rig.evaluate(&controls, &mut regs, &mut ex).unwrap();
                    assert!(ex.iter().all(|w| w.is_finite()));
                }
            },
        );
    }
}
