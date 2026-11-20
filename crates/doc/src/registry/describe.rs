//! The JSON forms of registry entries and parameter values `[REG-04]`, `[AC-WP07-2]`.
//!
//! [`describe_param`] writes every field of an entry, with `null` where a field does not apply, in
//! the format of a registry file entry with nothing left to defaults: [`param_from_description`]
//! reads it back to an equal [`Param`], which is how the tests show the description is complete.

use humanart_core::hash::Hash128;
use humanart_io::json::{Node, Number, Object, Value};

use super::{
    DESCRIBE_FORMAT_NAME, Evaluation, FORMAT_VERSION, GROUPS, Param, ParamKind, ParamValue, Registry, Symmetry, parse,
};
use crate::error::{DocError, Result};

fn number(v: f64) -> Value {
    Value::from(Number::Float(v))
}

fn opt_string(v: Option<&str>) -> Value {
    v.map_or(Value::Null, Value::from)
}

/// The JSON form of a value, as documents and agent tools write it: a number, an array of numbers,
/// a string (enum choice, blob hash), a bool, an array of `[x, y]` pairs, or `null` for "no asset".
pub fn value_to_json(value: &ParamValue) -> Value {
    match value {
        ParamValue::Scalar(v) => number(*v),
        ParamValue::Vector(vs) => Value::Array(vs.iter().map(|v| number(*v)).collect()),
        ParamValue::Color(c) => Value::Array(c.iter().map(|v| number(*v)).collect()),
        ParamValue::Enum(choice) => Value::from(choice.as_str()),
        ParamValue::Bool(b) => Value::from(*b),
        ParamValue::Curve(points) => {
            Value::Array(points.iter().map(|[x, y]| Value::Array(vec![number(*x), number(*y)])).collect())
        }
        ParamValue::AssetRef(hash) => {
            if hash.is_zero() {
                Value::Null
            } else {
                Value::from(hash.to_hex().as_str())
            }
        }
    }
}

fn numbers(value: &Value, limit: usize) -> Option<Vec<f64>> {
    let items = value.as_array()?;
    if items.len() > limit {
        return None;
    }
    items.iter().map(Value::as_f64).collect()
}

/// Reads the JSON form of a value of `kind` (with `dimension` components for a vector), checking
/// only its shape. Integers are accepted where numbers are expected.
pub fn decode_value(kind: ParamKind, dimension: u8, value: &Value) -> Option<ParamValue> {
    Some(match kind {
        ParamKind::Scalar => ParamValue::Scalar(value.as_f64()?),
        ParamKind::Vector => {
            let vs = numbers(value, usize::from(dimension))?;
            if vs.len() != usize::from(dimension) {
                return None;
            }
            ParamValue::Vector(vs)
        }
        ParamKind::Color => {
            let vs = numbers(value, 3)?;
            ParamValue::Color([*vs.first()?, *vs.get(1)?, *vs.get(2)?])
        }
        ParamKind::Enum => ParamValue::Enum(value.as_str()?.to_owned()),
        ParamKind::Bool => ParamValue::Bool(value.as_bool()?),
        ParamKind::Curve => {
            let items = value.as_array()?;
            if items.len() > super::MAX_CURVE_POINTS {
                return None;
            }
            let mut points = Vec::with_capacity(items.len());
            for item in items {
                let xy = numbers(item, 2)?;
                points.push([*xy.first()?, *xy.get(1)?]);
            }
            ParamValue::Curve(points)
        }
        ParamKind::AssetRef => {
            if value.is_null() {
                ParamValue::AssetRef(Hash128::ZERO)
            } else {
                ParamValue::AssetRef(Hash128::parse(value.as_str()?).ok()?)
            }
        }
    })
}

/// Reads the JSON form of a value of `param` and checks it ([`Param::check_value`]).
pub fn value_from_json(param: &Param, value: &Value) -> Result<ParamValue> {
    let decoded = decode_value(param.kind, param.dimension, value)
        .ok_or_else(|| DocError::param_value(&param.id, format!("not a {} value", param.kind.name())))?;
    param.check_value(&decoded)?;
    Ok(decoded)
}

/// The full JSON description of one entry `[REG-04]`. Members, in this order: `id`,
/// `display_name`, `description`, `group`, `region`, `kind`, `dimension`, `choices`, `unit`, `min`,
/// `max`, `soft_min`, `soft_max`, `step`, `default`, `symmetry` (`kind`, `partner`, `side`),
/// `affects`, `evaluation` (`kind`, `target`, `landmarks`, `axis`), `read_only`,
/// `locked_by_default`, `plausibility_weight`, `since_version`, `tags`, `provisional`,
/// `editor_only`. Fields that do not apply are `null`.
pub fn describe_param(param: &Param) -> Value {
    let mut o = Object::with_capacity(25);
    o.insert("id", param.id.as_str());
    o.insert("display_name", param.display_name.as_str());
    o.insert("description", param.description.as_str());
    o.insert("group", param.group.as_str());
    o.insert("region", opt_string(param.region.as_deref()));
    o.insert("kind", param.kind.name());
    o.insert("dimension", Value::from(Number::from(param.dimension)));
    o.insert(
        "choices",
        if param.kind == ParamKind::Enum {
            Value::Array(param.choices.iter().map(Value::from).collect())
        } else {
            Value::Null
        },
    );
    o.insert("unit", param.unit.as_str());
    let range = |f: fn(&super::Range) -> f64| param.range.as_ref().map_or(Value::Null, |r| number(f(r)));
    o.insert("min", range(|r| r.min));
    o.insert("max", range(|r| r.max));
    o.insert("soft_min", range(|r| r.soft_min));
    o.insert("soft_max", range(|r| r.soft_max));
    o.insert("step", range(|r| r.step));
    o.insert("default", value_to_json(&param.default));
    let mut symmetry = Object::with_capacity(3);
    let (kind, partner, side) = match &param.symmetry {
        Symmetry::None => ("none", None, None),
        Symmetry::Mirror { partner, side } => ("mirror", Some(partner.as_str()), Some(side.name())),
        Symmetry::Sided { side } => ("sided", None, Some(side.name())),
    };
    symmetry.insert("kind", kind);
    symmetry.insert("partner", opt_string(partner));
    symmetry.insert("side", opt_string(side));
    o.insert("symmetry", symmetry);
    o.insert("affects", Value::Array(param.affects.iter().map(|c| Value::from(c.name())).collect()));
    let mut evaluation = Object::with_capacity(4);
    evaluation.insert("kind", param.evaluation.kind_name());
    evaluation.insert("target", opt_string(param.evaluation.target()));
    if let Evaluation::Measurement { landmarks, axis } = &param.evaluation {
        evaluation.insert("landmarks", Value::Array(landmarks.iter().map(Value::from).collect()));
        evaluation.insert("axis", opt_string(axis.map(super::Axis::name)));
    } else {
        evaluation.insert("landmarks", Value::Null);
        evaluation.insert("axis", Value::Null);
    }
    o.insert("evaluation", evaluation);
    o.insert("read_only", param.read_only);
    o.insert("locked_by_default", param.locked_by_default);
    o.insert("plausibility_weight", number(param.plausibility_weight));
    o.insert("since_version", Value::from(Number::from(param.since_version)));
    o.insert("tags", Value::Array(param.tags.iter().map(Value::from).collect()));
    o.insert("provisional", opt_string(param.provisional.as_deref()));
    o.insert("editor_only", param.editor_only());
    Value::Object(o)
}

/// Reads one entry in the form [`describe_param`] writes (or any complete registry file entry
/// without `sides`).
pub fn param_from_description(value: &Value) -> Result<Param> {
    let mut params = parse::parse_entry(&Node::root(value), "description")?;
    match (params.pop(), params.is_empty()) {
        (Some(param), true) => Ok(param),
        _ => Err(DocError::registry("description", "a description holds exactly one parameter, without `sides`")),
    }
}

/// The description of the whole registry: `format`, `version`, `count`, `groups` (the `[REG-03]`
/// top-level groups) and `parameters` (one [`describe_param`] object per entry, in id order).
pub(super) fn describe_registry(registry: &Registry) -> Value {
    let mut o = Object::with_capacity(5);
    o.insert("format", DESCRIBE_FORMAT_NAME);
    o.insert("version", Value::from(Number::from(FORMAT_VERSION)));
    o.insert("count", Value::from(Number::from(registry.len())));
    o.insert("groups", Value::Array(GROUPS.iter().map(|g| Value::from(*g)).collect()));
    o.insert("parameters", Value::Array(registry.params().iter().map(describe_param).collect()));
    Value::Object(o)
}
