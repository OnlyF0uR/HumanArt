//! Building the expression model into a template (WP-19, `[EXP-01]`, `[EXP-02]`).
//!
//! [`with_expressions`] takes a template (normally the generated `humanart-neutral`), evaluates
//! every authored target of [`super::recipes`] on it, derives the combination correctives of a
//! [`CorrectiveSpecs`] data file (`data/template/expressions/correctives.json`, [`CORRECTIVES_JSON`])
//! and returns a new template whose delta bases, expression set, joint regressor and provenance
//! include them. Existing delta components (identity, pose correctives) are kept and re-indexed by
//! name.
//!
//! Each component's bounding region is the lowest common ancestor of the regions of the vertices
//! it moves, so the validator's "inside its region" rule holds by construction; every vertex of a
//! welded node gets the node's displacement, so no seam splits.
//!
//! **Corrective data file** (`[EXP-02]`, extensible without code changes):
//!
//! ```json
//! {"format": "humanart-correctives", "version": 1, "description": "...",
//!  "correctives": [{"name": "AU06-L.AU12-L", "op": "product", "inputs": ["AU06-L", "AU12-L"],
//!                   "rule": "saturate", "strength": 0.6}]}
//! ```
//!
//! `op` is `product` or `min`; 2 to 4 distinct inputs name expressions of the template; `rule` is
//! `saturate` (the only rule): where the inputs' summed displacement is longer than the longest
//! input's, the excess is reduced by `strength` in `[0, 1]`; correctives over a subset of the
//! inputs are subtracted first. At most [`super::MAX_DEFINITIONS`] entries, names follow the
//! stable-name rules, unknown members are refused.

use humanart_io::json::{Value, parse};
use humanart_math::Vec3;

use super::error::{ExprError, ExprResult};
use super::field::FaceField;
use super::recipes::{Recipes, Target};
use super::{CombinationCorrective, CombineOp, Expression, ExpressionSet, MAX_COMBINATION_INPUTS, MAX_DEFINITIONS};
use crate::delta::{DeltaBases, DeltaBasesBuilder, DeltaKind};
use crate::names::check_name;
use crate::skeleton::JointRegressor;
use crate::template::{ProvenanceEntry, Template};

/// The checked-in corrective definitions of the template.
pub const CORRECTIVES_JSON: &str = include_str!("../../../../data/template/expressions/correctives.json");

/// The most bytes a corrective data file may have.
pub const MAX_CORRECTIVES_BYTES: usize = 1 << 20;

/// Displacements shorter than this (metres) are not stored.
pub const EPSILON: f32 = 1.0e-6;

/// How a corrective's shape follows from its inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorrectiveRule {
    /// Reduce the excess of the summed displacement over the longest input's by `strength`.
    Saturate,
}

/// One corrective definition of the data file.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrectiveSpec {
    /// Stable name.
    pub name: String,
    /// How the activation follows the input weights.
    pub op: CombineOp,
    /// Input expression names, 2 to 4, distinct.
    pub inputs: Vec<String>,
    /// How the shape is derived.
    pub rule: CorrectiveRule,
    /// The rule's strength in `[0, 1]`.
    pub strength: f32,
}

/// The corrective definitions of a data file.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CorrectiveSpecs {
    /// The definitions in file order.
    pub correctives: Vec<CorrectiveSpec>,
}

impl CorrectiveSpecs {
    /// Parses and checks a corrective data file (see the module comment).
    pub fn from_json_bytes(bytes: &[u8]) -> ExprResult<Self> {
        if bytes.len() > MAX_CORRECTIVES_BYTES {
            return Err(ExprError::invalid("corrective file", 0, "larger than 1 MiB"));
        }
        let value = parse(bytes)?;
        Self::from_json(&value)
    }

    /// Reads the definitions from a parsed data file.
    pub fn from_json(value: &Value) -> ExprResult<Self> {
        let root = value.node();
        root.deny_unknown(&["format", "version", "description", "correctives"])?;
        if root.field("format")?.str()? != "humanart-correctives" {
            return Err(ExprError::invalid("corrective file", 0, "format is not humanart-correctives"));
        }
        if root.field("version")?.u64()? != 1 {
            return Err(ExprError::invalid("corrective file", 0, "unsupported version"));
        }
        let list = root.field("correctives")?;
        if list.array_len()? > MAX_DEFINITIONS {
            return Err(ExprError::invalid("corrective file", 0, "too many correctives"));
        }
        let mut correctives = Vec::with_capacity(list.array_len()?);
        for (i, item) in list.elements()?.enumerate() {
            item.deny_unknown(&["name", "op", "inputs", "rule", "strength"])?;
            let name = item.field("name")?.str()?;
            check_name("combination corrective", name)?;
            let op = item.field("op")?.choice(&[("product", CombineOp::Product), ("min", CombineOp::Min)])?;
            let rule = item.field("rule")?.choice(&[("saturate", CorrectiveRule::Saturate)])?;
            let strength = item.field("strength")?.f32()?;
            if !(0.0..=1.0).contains(&strength) {
                return Err(ExprError::invalid("corrective", i, "strength outside [0, 1]"));
            }
            let ins = item.field("inputs")?;
            if !(2..=MAX_COMBINATION_INPUTS).contains(&ins.array_len()?) {
                return Err(ExprError::invalid("corrective", i, "needs 2 to 4 inputs"));
            }
            let mut inputs: Vec<String> = Vec::with_capacity(ins.array_len()?);
            for x in ins.elements()? {
                let s = x.str()?;
                check_name("expression", s)?;
                if inputs.iter().any(|y| y == s) {
                    return Err(ExprError::invalid("corrective", i, "input used twice"));
                }
                inputs.push(s.to_owned());
            }
            if correctives.iter().any(|c: &CorrectiveSpec| c.name == name) {
                return Err(ExprError::invalid("corrective", i, "name used twice"));
            }
            correctives.push(CorrectiveSpec { name: name.to_owned(), op, inputs, rule, strength });
        }
        Ok(Self { correctives })
    }

    /// The template's own definitions ([`CORRECTIVES_JSON`]).
    pub fn builtin() -> ExprResult<Self> {
        Self::from_json_bytes(CORRECTIVES_JSON.as_bytes())
    }
}

/// The per-node corrective shapes of `specs` from the target shapes (indexed like `targets`):
/// smaller corrective sets first, each one's subsets subtracted.
pub fn corrective_shapes(targets: &[Target], specs: &CorrectiveSpecs) -> ExprResult<Vec<Vec<Vec3>>> {
    let n = targets.first().map_or(0, |t| t.shape.displacements().len());
    let mut resolved: Vec<Vec<usize>> = Vec::with_capacity(specs.correctives.len());
    for c in &specs.correctives {
        let mut ins = Vec::with_capacity(c.inputs.len());
        for name in &c.inputs {
            ins.push(
                targets.iter().position(|t| &t.name == name).ok_or_else(|| ExprError::missing("expression", name))?,
            );
        }
        resolved.push(ins);
    }
    let mut order: Vec<usize> = (0..specs.correctives.len()).collect();
    order.sort_by_key(|&i| resolved[i].len());
    let mut out: Vec<Vec<Vec3>> = vec![Vec::new(); specs.correctives.len()];
    for &i in &order {
        let ins = &resolved[i];
        let k = specs.correctives[i].strength;
        let mut shape = vec![Vec3::ZERO; n];
        for (node, c) in shape.iter_mut().enumerate() {
            let mut sum = Vec3::ZERO;
            let mut longest = 0.0_f32;
            for &t in ins {
                let d = targets[t].shape.displacements().get(node).copied().unwrap_or(Vec3::ZERO);
                sum += d;
                longest = longest.max(d.length());
            }
            let len = sum.length();
            if len > longest && len > 0.0 {
                let wanted = longest + (1.0 - k) * (len - longest);
                *c = sum * (wanted / len) - sum;
            }
        }
        for &j in &order {
            let sub = &resolved[j];
            if j != i && sub.len() < ins.len() && sub.iter().all(|x| ins.contains(x)) && !out[j].is_empty() {
                for (c, d) in shape.iter_mut().zip(&out[j]) {
                    *c -= *d;
                }
            }
        }
        out[i] = shape;
    }
    Ok(out)
}

/// The lowest common ancestor of the regions of `vertices` (region 0 when there are none).
fn bounding_region(template: &Template, vertices: impl Iterator<Item = u32>) -> u16 {
    let regions = template.regions();
    let chain = |r: u16| {
        let mut out = vec![r];
        let mut cur = r;
        while let Some(p) = regions.regions.get(cur as usize).and_then(|x| x.parent) {
            if out.len() > regions.regions.len() {
                break;
            }
            out.push(p);
            cur = p;
        }
        out
    };
    let mut lca: Option<Vec<u16>> = None;
    for v in vertices {
        let r = regions.vertex_regions.get(v as usize).copied().unwrap_or(0);
        lca = Some(match lca {
            None => chain(r),
            Some(cur) => {
                if regions.is_within(r, cur[0]) {
                    cur
                } else {
                    let mine = chain(r);
                    cur.into_iter().skip_while(|a| !mine.contains(a)).collect()
                }
            }
        });
    }
    lca.and_then(|c| c.first().copied()).unwrap_or(0)
}

/// What [`with_expressions`] produced, for reports.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BuildReport {
    /// Per expression: name, stored entries, largest displacement (metres).
    pub expressions: Vec<(String, usize, f32)>,
    /// Per combination corrective: name, stored entries, largest displacement (metres).
    pub correctives: Vec<(String, usize, f32)>,
}

/// The template with the authored expression targets and the correctives of `specs` added.
/// Refuses a template that already has expressions or combination correctives.
pub fn with_expressions(template: &Template, specs: &CorrectiveSpecs) -> ExprResult<(Template, BuildReport)> {
    if !template.expressions().expressions.is_empty() || !template.expressions().combinations.is_empty() {
        return Err(ExprError::invalid("template", 0, "already has expressions"));
    }
    let field = FaceField::new(template)?;
    let recipes = Recipes::new(&field)?;
    let targets = recipes.all()?;
    let corrective = corrective_shapes(&targets, specs)?;
    let old = template.deltas();
    let mut builder = DeltaBasesBuilder::new();
    for (c, comp) in old.components().iter().enumerate() {
        let entries: Vec<(u32, Vec3)> =
            old.indices(c).iter().zip(old.deltas(c)).map(|(&v, &d)| (v, DeltaBases::delta_f32(d))).collect();
        builder.add(&comp.name, comp.kind, comp.region, &entries)?;
    }
    let mut report = BuildReport::default();
    for t in &targets {
        let entries = field.vertex_entries(t.shape.displacements(), EPSILON);
        let region = bounding_region(template, entries.iter().map(|e| e.0));
        report.expressions.push((t.name.clone(), entries.len(), t.shape.max_displacement()));
        builder.add(&t.name, DeltaKind::Expression, region, &entries)?;
    }
    for (spec, shape) in specs.correctives.iter().zip(&corrective) {
        let entries = field.vertex_entries(shape, EPSILON);
        let region = bounding_region(template, entries.iter().map(|e| e.0));
        report.correctives.push((
            spec.name.clone(),
            entries.len(),
            shape.iter().map(|d| d.length()).fold(0.0, f32::max),
        ));
        builder.add(&spec.name, DeltaKind::CombinationCorrective, region, &entries)?;
    }
    let deltas = builder.finish()?;
    let find = |name: &str| deltas.find(name).ok_or_else(|| ExprError::missing("delta component", name));
    let mut set = ExpressionSet::default();
    for t in &targets {
        set.expressions.push(Expression {
            name: t.name.clone(),
            kind: t.kind,
            side: t.side,
            facs: t.facs,
            component: find(&t.name)?,
        });
    }
    for spec in &specs.correctives {
        let mut inputs = Vec::with_capacity(spec.inputs.len());
        for i in &spec.inputs {
            let e =
                set.expressions.iter().position(|e| &e.name == i).ok_or_else(|| ExprError::missing("expression", i))?;
            inputs.push(e as u32);
        }
        set.combinations.push(CombinationCorrective {
            name: spec.name.clone(),
            op: spec.op,
            inputs,
            component: find(&spec.name)?,
        });
    }
    for p in &template.expressions().pose_correctives {
        let name = old.components().get(p.component as usize).map_or("", |c| c.name.as_str());
        let mut p = p.clone();
        p.component = find(name)?;
        set.pose_correctives.push(p);
    }
    let mut parts = template.clone().into_parts();
    // The regressor is indexed by component: move each old component's rows to its new index.
    let mut regressor = JointRegressor::empty(deltas.len());
    let mut rows: Vec<(Vec<u16>, Vec<Vec3>)> = vec![(Vec::new(), Vec::new()); deltas.len()];
    for (c, comp) in old.components().iter().enumerate() {
        let (j, d) = parts.skeleton.regressor.component(c);
        rows[find(&comp.name)? as usize] = (j.to_vec(), d.to_vec());
    }
    regressor.offsets = vec![0];
    regressor.joints.clear();
    regressor.deltas.clear();
    for (j, d) in rows {
        regressor.joints.extend(j);
        regressor.deltas.extend(d);
        regressor.offsets.push(regressor.joints.len() as u32);
    }
    parts.skeleton.regressor = regressor;
    parts.deltas = deltas;
    parts.expressions = set;
    parts.info.provenance.push(ProvenanceEntry {
        item: "expression targets and combination correctives (WP-19)".into(),
        source: "authored procedurally by humanart-model expression::recipes and data/template/expressions/correctives.json; amounts are model choices, not measurements".into(),
        licence: parts.info.licence.clone(),
    });
    Ok((Template::new(parts)?, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::fuzz;

    #[test]
    fn exp_02_builtin_correctives_parse() {
        let s = CorrectiveSpecs::builtin().unwrap();
        assert!(s.correctives.len() >= 20);
        assert!(s.correctives.iter().any(|c| c.inputs.len() == 3));
        assert!(s.correctives.iter().any(|c| c.name == "AU06-L.AU12-L"));
    }

    #[test]
    fn corrective_files_are_checked() {
        let ok = r#"{"format":"humanart-correctives","version":1,"correctives":[{"name":"a.b","op":"min","inputs":["a","b"],"rule":"saturate","strength":0.5}]}"#;
        assert_eq!(CorrectiveSpecs::from_json_bytes(ok.as_bytes()).unwrap().correctives[0].op, CombineOp::Min);
        for bad in [
            ok.replace("humanart-correctives", "x"),
            ok.replace("\"version\":1", "\"version\":2"),
            ok.replace("0.5", "1.5"),
            ok.replace("[\"a\",\"b\"]", "[\"a\"]"),
            ok.replace("[\"a\",\"b\"]", "[\"a\",\"a\"]"),
            ok.replace("\"min\"", "\"max\""),
            ok.replace("\"saturate\"", "\"other\""),
            ok.replace("\"name\":\"a.b\"", "\"name\":\"a b\""),
            ok.replace("\"strength\"", "\"extra\":1,\"strength\""),
            ok.replace("0.5}]}", "0.5},{\"name\":\"a.b\",\"op\":\"min\",\"inputs\":[\"a\",\"b\"],\"rule\":\"saturate\",\"strength\":0.5}]}"),
        ] {
            assert!(CorrectiveSpecs::from_json_bytes(bad.as_bytes()).is_err(), "{bad}");
        }
        assert!(CorrectiveSpecs::from_json_bytes(&vec![b' '; MAX_CORRECTIVES_BYTES + 1]).is_err());
    }

    #[test]
    fn rob_01_corrective_file_mutation_fuzz() {
        let corpus = vec![CORRECTIVES_JSON.as_bytes().to_vec()];
        fuzz::run("correctives.json", &corpus, 3000, 0x19_0002, std::time::Duration::from_secs(20), |b| {
            let _ = CorrectiveSpecs::from_json_bytes(b);
        });
    }

    #[test]
    fn saturate_rule_limits_double_motion() {
        use super::super::ExpressionKind;
        use super::super::action::Shape;
        use crate::names::Side;
        let mk = |name: &str, d: Vec3| {
            let mut s = Shape::new(2);
            s.displacements_mut()[0] = d;
            Target { name: name.into(), kind: ExpressionKind::ActionUnit, side: Side::Center, facs: 0, shape: s }
        };
        let t = vec![mk("a", Vec3::X), mk("b", Vec3::X), mk("c", Vec3::Y), mk("d", Vec3::X)];
        let spec = |name: &str, inputs: &[&str], strength: f32| CorrectiveSpec {
            name: name.into(),
            op: CombineOp::Product,
            inputs: inputs.iter().map(|s| (*s).to_owned()).collect(),
            rule: CorrectiveRule::Saturate,
            strength,
        };
        let specs = CorrectiveSpecs {
            correctives: vec![
                spec("abd", &["a", "b", "d"], 1.0),
                spec("ab", &["a", "b"], 1.0),
                spec("ac", &["a", "c"], 1.0),
            ],
        };
        let c = corrective_shapes(&t, &specs).unwrap();
        // Parallel inputs: full strength keeps the longest input only.
        assert!((c[1][0] - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-6);
        // Perpendicular inputs are reduced toward the longest length along their sum.
        let sum = Vec3::X + Vec3::Y;
        assert!(((sum + c[2][0]).length() - 1.0).abs() < 1e-6);
        // The triple has its pair subtracted: a + b + d + C_ab + C_abd has length 1.
        assert!(((Vec3::X * 3.0 + c[1][0] + c[0][0]).length() - 1.0).abs() < 1e-6);
        assert!(c.iter().all(|s| s[1] == Vec3::ZERO));
        let missing = CorrectiveSpecs { correctives: vec![spec("x", &["a", "zz"], 0.5)] };
        assert!(corrective_shapes(&t, &missing).is_err());
    }
}
