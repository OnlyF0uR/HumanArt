//! The expression control rig and the control-surface export (WP-19, `[EXP-03]`, `[EXP-08]`).
//!
//! A [`ControlRig`] maps the named controls (the registry's expression parameters, e.g.
//! `expression.au12_lip_corner_puller.l`, plus the viseme controls `viseme.*`) to
//! - expression weights, through the rig-logic graph of [`crate::eval::rig`] read from
//!   `data/template/expressions/rig.json` ([`RIG_JSON`]), and
//! - joint rotations, through the joint drivers of `data/template/expressions/drivers.json`
//!   ([`DRIVERS_JSON`]): jaw, tongue (skeleton-driven, `[EXP-01]`).
//!
//! [`ControlRig::evaluate`] does no heap allocation `[PERF-R1]`; the caller owns the register,
//! weight and pose arrays.
//!
//! **Driver data file:**
//!
//! ```json
//! {"format": "humanart-joint-drivers", "version": 1, "description": "...",
//!  "drivers": [{"input": "jaw.open", "joint": "jaw", "axis": [1, 0, 0], "angle": 1.0}]}
//! ```
//!
//! `input` names a register of the rig (a control or node), `joint` a joint of the template,
//! `axis` a non-zero world-space direction at rest, `angle` radians per unit of input with
//! `|angle| <= ` [`MAX_DRIVER_ANGLE`]. At most [`MAX_DRIVERS`] entries; unknown members are
//! refused. Entries on one joint add as rotation vectors in the joint's frame.
//!
//! [`controls_json`] writes the control-surface export `controls.json` `[EXP-08]` `[IO-13]`: every
//! control with its range and default and what it drives, the expressions, the correctives, and the
//! full mapping (rig graph and drivers) verbatim.

use humanart_io::json::{JsonWriter, Value, parse};
use humanart_math::{Quat, Vec3};

use super::error::{ExprError, ExprResult};
use crate::eval::rig::RigLogic;
use crate::eval::wrinkle::WrinkleDrivers;
use crate::names::Side;
use crate::template::Template;

/// The checked-in rig-logic graph of the template's controls.
pub const RIG_JSON: &str = include_str!("../../../../data/template/expressions/rig.json");
/// The checked-in joint drivers of the template's controls.
pub const DRIVERS_JSON: &str = include_str!("../../../../data/template/expressions/drivers.json");
/// The checked-in wrinkle drivers of the template `[EXP-04]` (format of [`crate::eval::wrinkle`]):
/// forehead, glabella, crow's feet and nasolabial folds, driven by their action units and the
/// measured compression of their region; gains are authored model choices.
pub const WRINKLES_JSON: &str = include_str!("../../../../data/template/expressions/wrinkles.json");
/// The most driver entries.
pub const MAX_DRIVERS: usize = 1024;
/// The largest driver angle per unit of input, radians.
pub const MAX_DRIVER_ANGLE: f32 = std::f32::consts::PI;
/// The most bytes of a rig or driver data file.
pub const MAX_DATA_BYTES: usize = 4 << 20;

/// One joint driver, resolved against a rig and a template.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointDriver {
    /// The rig register read.
    pub register: u32,
    /// The joint rotated.
    pub joint: u16,
    /// The unit axis in world space at rest.
    pub axis: Vec3,
    /// Radians per unit of the register.
    pub angle: f32,
    /// The unit axis in the joint's rest frame.
    pub local_axis: Vec3,
}

/// The rig logic plus the joint drivers of a template.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlRig {
    logic: RigLogic,
    drivers: Vec<JointDriver>,
    rig_json: Value,
    drivers_json: Value,
}

fn parse_limited(bytes: &[u8]) -> ExprResult<Value> {
    if bytes.len() > MAX_DATA_BYTES {
        return Err(ExprError::invalid("data file", 0, "larger than 4 MiB"));
    }
    Ok(parse(bytes)?)
}

impl ControlRig {
    /// Reads a rig graph and a driver file and resolves them against `template` (its expression
    /// names and joints).
    pub fn from_json_bytes(template: &Template, rig: &[u8], drivers: &[u8]) -> ExprResult<Self> {
        let rig_json = parse_limited(rig)?;
        let drivers_json = parse_limited(drivers)?;
        let logic = RigLogic::from_json(&rig_json, template.expressions())?;
        let root = drivers_json.node();
        root.deny_unknown(&["format", "version", "description", "drivers"])?;
        if root.field("format")?.str()? != "humanart-joint-drivers" {
            return Err(ExprError::invalid("driver file", 0, "format is not humanart-joint-drivers"));
        }
        if root.field("version")?.u64()? != 1 {
            return Err(ExprError::invalid("driver file", 0, "unsupported version"));
        }
        let list = root.field("drivers")?;
        if list.array_len()? > MAX_DRIVERS {
            return Err(ExprError::invalid("driver file", 0, "too many drivers"));
        }
        let rest = template.skeleton().rest_world();
        let mut out = Vec::with_capacity(list.array_len()?);
        for (i, d) in list.elements()?.enumerate() {
            d.deny_unknown(&["input", "joint", "axis", "angle"])?;
            let input = d.field("input")?.str()?;
            let register = logic.register(input).ok_or_else(|| ExprError::missing("rig register", input))?;
            let jname = d.field("joint")?.str()?;
            let joint = template.skeleton().find(jname).ok_or_else(|| ExprError::missing("joint", jname))?;
            let a = d.field("axis")?;
            if a.array_len()? != 3 {
                return Err(ExprError::invalid("driver", i, "axis is [x, y, z]"));
            }
            let axis = Vec3::new(a.at(0)?.f32()?, a.at(1)?.f32()?, a.at(2)?.f32()?);
            let len = axis.length();
            if !len.is_finite() || len < 1e-6 || len > 1e6 {
                return Err(ExprError::invalid("driver", i, "axis is zero or not finite"));
            }
            let angle = d.field("angle")?.f32()?;
            if !angle.is_finite() || angle.abs() > MAX_DRIVER_ANGLE {
                return Err(ExprError::invalid("driver", i, "angle outside [-pi, pi]"));
            }
            let axis = axis / len;
            let rb = rest.get(joint as usize).map_or(Quat::IDENTITY, |t| t.rotation);
            let local_axis = (rb.conjugate() * axis).normalize_or_zero();
            out.push(JointDriver { register, joint, axis, angle, local_axis });
        }
        out.sort_by_key(|d| d.joint);
        Ok(Self { logic, drivers: out, rig_json, drivers_json })
    }

    /// The template's own rig ([`RIG_JSON`], [`DRIVERS_JSON`]) resolved against `template`.
    pub fn builtin(template: &Template) -> ExprResult<Self> {
        Self::from_json_bytes(template, RIG_JSON.as_bytes(), DRIVERS_JSON.as_bytes())
    }

    /// The rig-logic graph.
    #[must_use]
    pub const fn logic(&self) -> &RigLogic {
        &self.logic
    }

    /// The joint drivers, sorted by joint.
    #[must_use]
    pub fn drivers(&self) -> &[JointDriver] {
        &self.drivers
    }

    /// The control names.
    #[must_use]
    pub fn controls(&self) -> &[String] {
        self.logic.control_names()
    }

    /// How many registers the caller's register array needs.
    #[must_use]
    pub fn register_count(&self) -> usize {
        self.logic.register_count()
    }

    /// Evaluates `controls` (indexed like [`ControlRig::controls`]) into the rig's expression
    /// weights (other entries of `expressions` are left as they are) and the driven joints' rest-
    /// relative rotations in `pose` (other joints are left as they are). No heap allocation.
    pub fn evaluate(
        &self,
        controls: &[f32],
        registers: &mut [f32],
        expressions: &mut [f32],
        pose: &mut [Quat],
    ) -> ExprResult<()> {
        self.logic.evaluate(controls, registers, expressions)?;
        let mut i = 0;
        while i < self.drivers.len() {
            let joint = self.drivers[i].joint;
            let mut v = Vec3::ZERO;
            while i < self.drivers.len() && self.drivers[i].joint == joint {
                let d = &self.drivers[i];
                let x = registers.get(d.register as usize).copied().unwrap_or(0.0);
                v += d.local_axis * (d.angle * x);
                i += 1;
            }
            if let Some(p) = pose.get_mut(joint as usize) {
                *p = Quat::from_scaled_axis(v);
            }
        }
        Ok(())
    }

    /// For control `c`: the expressions whose weight depends on it and the joints it rotates
    /// (names), found by following the graph's node inputs in the rig file.
    #[must_use]
    pub fn drives(&self, template: &Template, c: usize) -> (Vec<String>, Vec<String>) {
        let Some(control) = self.controls().get(c) else { return (Vec::new(), Vec::new()) };
        let mut reach: Vec<&str> = vec![control.as_str()];
        if let Some(nodes) = self.rig_json.get("nodes").and_then(Value::as_array) {
            for n in nodes {
                let Some(name) = n.get("name").and_then(Value::as_str) else { continue };
                let mut inputs: Vec<&str> = Vec::new();
                if let Some(x) = n.get("input").and_then(Value::as_str) {
                    inputs.push(x);
                }
                for x in n.get("inputs").and_then(Value::as_array).unwrap_or(&[]) {
                    let v = x.as_array().and_then(|a| a.first()).unwrap_or(x);
                    if let Some(s) = v.as_str() {
                        inputs.push(s);
                    }
                }
                if inputs.iter().any(|i| reach.contains(i)) {
                    reach.push(name);
                }
            }
        }
        let mut expressions = Vec::new();
        if let Some(outputs) = self.rig_json.get("outputs").and_then(Value::as_object) {
            for (e, from) in outputs {
                if from.as_str().is_some_and(|f| reach.contains(&f)) {
                    expressions.push(e.to_owned());
                }
            }
        }
        let mut joints: Vec<String> = Vec::new();
        for d in &self.drivers {
            let reg = self.logic.control_names().len();
            let name = if (d.register as usize) < reg {
                self.controls().get(d.register as usize).map(String::as_str)
            } else {
                reach.iter().copied().find(|r| self.logic.register(r) == Some(d.register))
            };
            if name.is_some_and(|n| reach.contains(&n))
                && let Some(j) = template.skeleton().joints.get(d.joint as usize)
                && !joints.contains(&j.name)
            {
                joints.push(j.name.clone());
            }
        }
        (expressions, joints)
    }
}

/// The template's wrinkle drivers ([`WRINKLES_JSON`]) resolved against its regions and expressions.
pub fn wrinkle_drivers(template: &Template) -> ExprResult<WrinkleDrivers> {
    let value = parse_limited(WRINKLES_JSON.as_bytes())?;
    Ok(WrinkleDrivers::from_json(&value, template.regions(), template.expressions())?)
}

const fn side_name(s: Side) -> &'static str {
    match s {
        Side::Center => "center",
        Side::Left => "left",
        Side::Right => "right",
    }
}

/// The control-surface export `controls.json` `[EXP-08]`: format, template, controls (name, range,
/// default, driven expressions and joints), expressions, combination correctives and the mapping
/// (rig graph and joint drivers, verbatim). Every control has range `[0, 1]` and default 0.
pub fn controls_json(template: &Template, rig: &ControlRig) -> ExprResult<String> {
    let mut text = String::new();
    let mut w = JsonWriter::pretty(&mut text);
    let io = |_: humanart_io::Error| ExprError::invalid("controls.json", 0, "write failed");
    (|| -> Result<(), humanart_io::Error> {
        w.begin_object()?;
        w.key("format")?.string("humanart-controls")?;
        w.key("version")?.u64(1)?;
        w.key("template")?.string(&template.info().name)?;
        w.key("template_version")?.u64(u64::from(template.info().version))?;
        w.key("controls")?.begin_array()?;
        for (c, name) in rig.controls().iter().enumerate() {
            let (exprs, joints) = rig.drives(template, c);
            w.begin_object()?;
            w.key("name")?.string(name)?;
            w.key("min")?.f32(0.0)?.key("max")?.f32(1.0)?.key("default")?.f32(0.0)?;
            w.key("expressions")?.begin_array_inline()?;
            for e in &exprs {
                w.string(e)?;
            }
            w.end_array()?;
            w.key("joints")?.begin_array_inline()?;
            for j in &joints {
                w.string(j)?;
            }
            w.end_array()?;
            w.end_object()?;
        }
        w.end_array()?;
        let set = template.expressions();
        w.key("expressions")?.begin_array()?;
        for e in &set.expressions {
            w.begin_object()?;
            w.key("name")?.string(&e.name)?;
            w.key("kind")?.string(match e.kind {
                super::ExpressionKind::ActionUnit => "action-unit",
                super::ExpressionKind::Viseme => "viseme",
                super::ExpressionKind::Tongue => "tongue",
                super::ExpressionKind::Support => "support",
            })?;
            w.key("side")?.string(side_name(e.side))?;
            w.key("facs")?.u64(u64::from(e.facs))?;
            w.end_object()?;
        }
        w.end_array()?;
        w.key("correctives")?.begin_array()?;
        for c in &set.combinations {
            w.begin_object()?;
            w.key("name")?.string(&c.name)?;
            w.key("op")?.string(match c.op {
                super::CombineOp::Product => "product",
                super::CombineOp::Min => "min",
            })?;
            w.key("inputs")?.begin_array_inline()?;
            for &i in &c.inputs {
                w.string(set.expressions.get(i as usize).map_or("", |e| e.name.as_str()))?;
            }
            w.end_array()?;
            w.end_object()?;
        }
        w.end_array()?;
        w.key("rig")?.value(&rig.rig_json)?;
        w.key("drivers")?.value(&rig.drivers_json)?;
        w.end_object()?;
        Ok(())
    })()
    .map_err(io)?;
    w.finish().map_err(io)?;
    text.push('\n');
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::template::fixtures;
    use humanart_core::fuzz;

    fn cube() -> Template {
        Template::new(fixtures::cube_template()).unwrap()
    }

    const RIG: &str =
        r#"{"controls": ["a", "b"], "nodes": [{"name": "n", "op": "linear", "inputs": [["a", 0.5]]}], "outputs": {}}"#;
    const DRV: &str = r#"{"format": "humanart-joint-drivers", "version": 1, "drivers": [{"input": "n", "joint": "JOINT", "axis": [0, 0, 2], "angle": 1.0}, {"input": "b", "joint": "JOINT", "axis": [0, 0, 1], "angle": -0.5}]}"#;

    #[test]
    fn exp_03_drivers_rotate_joints() {
        let t = cube();
        let joint = t.skeleton().joints[0].name.clone();
        let drv = DRV.replace("JOINT", &joint);
        let rig = ControlRig::from_json_bytes(&t, RIG.as_bytes(), drv.as_bytes()).unwrap();
        let mut regs = vec![0.0; rig.register_count()];
        let mut exprs = vec![0.0; t.expressions().expressions.len()];
        let mut pose = vec![Quat::IDENTITY; t.skeleton().joints.len()];
        rig.evaluate(&[1.0, 0.4], &mut regs, &mut exprs, &mut pose).unwrap();
        // 0.5 * 1.0 rad - 0.4 * 0.5 rad about world Z, expressed in the joint frame.
        let rb = t.skeleton().rest_world()[0].rotation;
        let world = rb * pose[0] * rb.conjugate();
        let expected = Quat::from_axis_angle(Vec3::Z, 0.3);
        assert!(world.angle_between(expected) < 1e-5, "{world:?}");
        let (_, joints) = rig.drives(&t, 1);
        assert_eq!(joints, vec![joint.clone()]);
        let (_, joints) = rig.drives(&t, 0);
        assert_eq!(joints, vec![joint]);
    }

    #[test]
    fn driver_files_are_checked() {
        let t = cube();
        let joint = t.skeleton().joints[0].name.clone();
        let ok = DRV.replace("JOINT", &joint);
        assert!(ControlRig::from_json_bytes(&t, RIG.as_bytes(), ok.as_bytes()).is_ok());
        for bad in [
            ok.replace("humanart-joint-drivers", "x"),
            ok.replace("\"version\": 1", "\"version\": 3"),
            ok.replace("\"input\": \"n\"", "\"input\": \"zz\""),
            ok.replace(&format!("\"joint\": \"{joint}\""), "\"joint\": \"zz\""),
            ok.replace("[0, 0, 2]", "[0, 0, 0]"),
            ok.replace("[0, 0, 2]", "[0, 2]"),
            ok.replace("\"angle\": 1.0", "\"angle\": 9.0"),
            ok.replace("\"angle\": 1.0", "\"angle\": 1.0, \"x\": 1"),
        ] {
            assert!(ControlRig::from_json_bytes(&t, RIG.as_bytes(), bad.as_bytes()).is_err(), "{bad}");
        }
        assert!(ControlRig::from_json_bytes(&t, &vec![b' '; MAX_DATA_BYTES + 1], ok.as_bytes()).is_err());
    }

    #[test]
    fn rob_01_driver_file_mutation_fuzz() {
        let corpus = vec![DRIVERS_JSON.as_bytes().to_vec()];
        fuzz::run("drivers.json", &corpus, 3000, 0x19_0003, std::time::Duration::from_secs(20), |b| {
            let t = Template::new(fixtures::cube_template()).unwrap();
            let rig = r#"{"controls": ["w.ctl.tongue-up", "jaw.open", "jaw.yaw"], "outputs": {}}"#;
            let _ = ControlRig::from_json_bytes(&t, rig.as_bytes(), b);
        });
    }
}
