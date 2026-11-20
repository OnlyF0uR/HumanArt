//! Tool argument schemas `[AG-02]`: one declaration per tool gives both the machine-readable JSON
//! Schema the server publishes and the validation of every call, so there is no hand-maintained
//! duplicate.
//!
//! A tool lists its arguments as [`Field`]s of a [`Ty`]. Types that name document vocabulary are
//! derived from the data, not written out: [`Ty::Param`] is the registry's id list,
//! [`Ty::ParamValues`] has one property per registry parameter with its kind, dimension, choices,
//! unit and range (in the description and the `x-humanart-range` member: ranges are applied by the
//! command's range policy `[CMD-05]`, so a value outside them is not a schema error), and
//! [`Ty::Region`] / [`Ty::Landmark`] / [`Ty::Joint`] are the template's names.
//!
//! [`validate`] checks a call's `arguments` against the declaration: required members, no unknown
//! members, JSON types, integer and number bounds, string lengths, list and map sizes, enum
//! membership and registry vocabulary. Handlers then read arguments that are known to be well
//! formed.

use std::fmt::Write as _;

use humanart_doc::registry::{ParamKind, Registry, decode_value};
use humanart_io::json::{Number, Object, Value};
use humanart_model::Template;

/// Which registry parameters a [`Ty::ParamValues`] accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamFilter {
    /// Every parameter.
    All,
    /// Parameters whose group is one of these or below one of them (`eyes` takes `eyes/iris`).
    Groups(&'static [&'static str]),
}

impl ParamFilter {
    /// Whether a parameter of `group` passes.
    pub fn accepts(self, group: &str) -> bool {
        match self {
            Self::All => true,
            Self::Groups(gs) => {
                gs.iter().any(|g| group == *g || group.strip_prefix(g).is_some_and(|r| r.starts_with('/')))
            }
        }
    }
}

/// The type of an argument.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ty {
    /// `true` or `false`.
    Bool,
    /// A whole number in `min..=max`.
    Int {
        /// Smallest.
        min: i64,
        /// Largest.
        max: i64,
    },
    /// A finite number in `min..=max`.
    Num {
        /// Smallest.
        min: f64,
        /// Largest.
        max: f64,
    },
    /// A string of at most `max` bytes.
    Str {
        /// Longest.
        max: usize,
    },
    /// One of these strings.
    Choice(&'static [&'static str]),
    /// A 128-bit id as 32 lowercase hex digits.
    Id,
    /// Three finite numbers.
    Vec3,
    /// A quaternion `(x, y, z, w)`.
    Quat,
    /// A list of at most `max` items.
    List(&'static Ty, usize),
    /// An object with these members.
    Record(&'static [Field]),
    /// An object with arbitrary keys (at most 128 bytes) and at most `max` members.
    Map(&'static Ty, usize),
    /// A registry parameter id.
    Param,
    /// Registry parameter id to value, for the parameters the filter accepts.
    ParamValues(ParamFilter),
    /// A template region name.
    Region,
    /// A template landmark name.
    Landmark,
    /// A skeleton joint name.
    Joint,
    /// Any JSON object (its contents are checked by the document's codec).
    Fields,
}

/// One argument.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Field {
    /// The member name.
    pub name: &'static str,
    /// Its type.
    pub ty: Ty,
    /// Whether it must be given.
    pub required: bool,
    /// One sentence for the agent.
    pub doc: &'static str,
}

impl Field {
    /// A required field.
    pub const fn req(name: &'static str, ty: Ty, doc: &'static str) -> Self {
        Self { name, ty, required: true, doc }
    }

    /// An optional field.
    pub const fn opt(name: &'static str, ty: Ty, doc: &'static str) -> Self {
        Self { name, ty, required: false, doc }
    }
}

/// The vocabulary the derived types draw on.
#[derive(Clone, Copy, Debug)]
pub struct Vocab<'a> {
    /// The parameter registry.
    pub registry: &'a Registry,
    /// The template (regions, landmarks, joints).
    pub template: &'a Template,
}

impl Vocab<'_> {
    fn has_region(&self, name: &str) -> bool {
        self.template.regions().find(name).is_some()
    }

    fn has_landmark(&self, name: &str) -> bool {
        self.template.landmarks().iter().any(|l| l.name == name)
    }

    fn has_joint(&self, name: &str) -> bool {
        self.template.skeleton().find(name).is_some()
    }
}

fn num(v: f64) -> Value {
    Value::from(Number::from(v))
}

fn int(v: i64) -> Value {
    Value::from(Number::from(v as isize))
}

fn strings<'s>(items: impl Iterator<Item = &'s str>) -> Value {
    Value::Array(items.map(Value::from).collect())
}

fn obj(members: &[(&str, Value)]) -> Value {
    let mut o = Object::with_capacity(members.len());
    for (k, v) in members {
        o.insert(*k, v.clone());
    }
    Value::Object(o)
}

fn numbers_array(n: usize) -> Value {
    let n = int(n as i64);
    obj(&[
        ("type", "array".into()),
        ("items", obj(&[("type", "number".into())])),
        ("minItems", n.clone()),
        ("maxItems", n),
    ])
}

/// The schema of one parameter's value (kind, dimension, choices; range in the description).
pub fn param_value_schema(p: &humanart_doc::registry::Param) -> Value {
    let mut o = match p.kind {
        ParamKind::Scalar => obj(&[("type", "number".into())]),
        ParamKind::Vector => numbers_array(usize::from(p.dimension)),
        ParamKind::Color => numbers_array(3),
        ParamKind::Enum => obj(&[("type", "string".into()), ("enum", strings(p.choices.iter().map(String::as_str)))]),
        ParamKind::Bool => obj(&[("type", "boolean".into())]),
        ParamKind::Curve => obj(&[("type", "array".into()), ("items", numbers_array(2))]),
        ParamKind::AssetRef => obj(&[("type", ["string", "null"].into())]),
    };
    let mut description = format!("{}: {}", p.display_name, p.description);
    if let Some(r) = p.range {
        let _ =
            write!(description, " Range {}..{} (plausible {}..{}) {}.", r.min, r.max, r.soft_min, r.soft_max, p.unit);
        if let Value::Object(m) = &mut o {
            m.insert("x-humanart-range", Value::Array(vec![num(r.min), num(r.max)]));
        }
    }
    if let Value::Object(m) = &mut o {
        m.insert("description", description);
    }
    o
}

/// The JSON Schema of `ty`.
pub fn ty_schema(ty: &Ty, vocab: &Vocab<'_>) -> Value {
    match ty {
        Ty::Bool => obj(&[("type", "boolean".into())]),
        Ty::Int { min, max } => obj(&[("type", "integer".into()), ("minimum", int(*min)), ("maximum", int(*max))]),
        Ty::Num { min, max } => obj(&[("type", "number".into()), ("minimum", num(*min)), ("maximum", num(*max))]),
        Ty::Str { max } => obj(&[("type", "string".into()), ("maxLength", int(*max as i64))]),
        Ty::Choice(c) => obj(&[("type", "string".into()), ("enum", strings(c.iter().copied()))]),
        Ty::Id => obj(&[("type", "string".into()), ("pattern", "^[0-9a-f]{32}$".into())]),
        Ty::Vec3 => numbers_array(3),
        Ty::Quat => numbers_array(4),
        Ty::List(item, max) => {
            obj(&[("type", "array".into()), ("items", ty_schema(item, vocab)), ("maxItems", int(*max as i64))])
        }
        Ty::Record(fields) => record_schema(fields, vocab),
        Ty::Map(item, max) => obj(&[
            ("type", "object".into()),
            ("additionalProperties", ty_schema(item, vocab)),
            ("maxProperties", int(*max as i64)),
        ]),
        Ty::Param => {
            obj(&[("type", "string".into()), ("enum", strings(vocab.registry.params().iter().map(|p| p.id.as_str())))])
        }
        Ty::ParamValues(filter) => {
            let mut props = Object::new();
            for p in vocab.registry.params().iter().filter(|p| filter.accepts(&p.group) && !p.read_only) {
                props.insert(p.id.as_str(), param_value_schema(p));
            }
            let count = props.len();
            obj(&[
                ("type", "object".into()),
                ("properties", Value::Object(props)),
                ("additionalProperties", false.into()),
                ("minProperties", int(1)),
                ("maxProperties", int(count as i64)),
            ])
        }
        Ty::Region => obj(&[
            ("type", "string".into()),
            ("enum", strings(vocab.template.regions().regions.iter().map(|r| r.name.as_str()))),
        ]),
        Ty::Landmark => obj(&[
            ("type", "string".into()),
            ("enum", strings(vocab.template.landmarks().iter().map(|l| l.name.as_str()))),
        ]),
        Ty::Joint => obj(&[
            ("type", "string".into()),
            ("enum", strings(vocab.template.skeleton().joints.iter().map(|j| j.name.as_str()))),
        ]),
        Ty::Fields => obj(&[("type", "object".into())]),
    }
}

/// The JSON Schema of an object with `fields`.
pub fn record_schema(fields: &[Field], vocab: &Vocab<'_>) -> Value {
    let mut props = Object::with_capacity(fields.len());
    for f in fields {
        let mut s = ty_schema(&f.ty, vocab);
        if let Value::Object(m) = &mut s
            && !m.contains_key("description")
        {
            m.insert("description", f.doc);
        }
        props.insert(f.name, s);
    }
    obj(&[
        ("type", "object".into()),
        ("properties", Value::Object(props)),
        ("required", strings(fields.iter().filter(|f| f.required).map(|f| f.name))),
        ("additionalProperties", false.into()),
    ])
}

/// Why arguments do not fit their schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchemaError {
    /// Where (a JSON pointer, e.g. `/values/face.nose.width`).
    pub path: String,
    /// What is wrong.
    pub message: String,
}

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid argument at '{}': {}", if self.path.is_empty() { "/" } else { &self.path }, self.message)
    }
}

impl std::error::Error for SchemaError {}

fn fail(path: &str, message: impl Into<String>) -> SchemaError {
    SchemaError { path: path.to_owned(), message: message.into() }
}

fn finite_numbers(value: &Value, n: usize, path: &str) -> Result<(), SchemaError> {
    let items = value.as_array().ok_or_else(|| fail(path, format!("expected an array of {n} numbers")))?;
    if items.len() != n || !items.iter().all(|v| v.as_f64().is_some_and(f64::is_finite)) {
        return Err(fail(path, format!("expected exactly {n} finite numbers")));
    }
    Ok(())
}

/// Checks `value` against `ty`.
///
/// # Errors
/// The first mismatch, with its path.
pub fn validate_ty(ty: &Ty, value: &Value, vocab: &Vocab<'_>, path: &str) -> Result<(), SchemaError> {
    match ty {
        Ty::Bool => value.as_bool().map(|_| ()).ok_or_else(|| fail(path, "expected true or false")),
        Ty::Int { min, max } => match value.as_i64() {
            Some(v) if (*min..=*max).contains(&v) => Ok(()),
            Some(v) => Err(fail(path, format!("{v} is outside {min}..={max}"))),
            None => Err(fail(path, "expected a whole number")),
        },
        Ty::Num { min, max } => match value.as_f64() {
            Some(v) if v.is_finite() && v >= *min && v <= *max => Ok(()),
            Some(v) => Err(fail(path, format!("{v} is outside {min}..={max}"))),
            None => Err(fail(path, "expected a number")),
        },
        Ty::Str { max } => match value.as_str() {
            Some(s) if s.len() <= *max => Ok(()),
            Some(_) => Err(fail(path, format!("longer than {max} bytes"))),
            None => Err(fail(path, "expected a string")),
        },
        Ty::Choice(choices) => match value.as_str() {
            Some(s) if choices.contains(&s) => Ok(()),
            _ => Err(fail(path, format!("expected one of {}", choices.join(", ")))),
        },
        Ty::Id => match value.as_str() {
            Some(s) if s.len() == 32 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) => Ok(()),
            _ => Err(fail(path, "expected an id of 32 lowercase hex digits")),
        },
        Ty::Vec3 => finite_numbers(value, 3, path),
        Ty::Quat => finite_numbers(value, 4, path),
        Ty::List(item, max) => {
            let items = value.as_array().ok_or_else(|| fail(path, "expected an array"))?;
            if items.len() > *max {
                return Err(fail(path, format!("more than {max} items")));
            }
            for (i, v) in items.iter().enumerate() {
                validate_ty(item, v, vocab, &format!("{path}/{i}"))?;
            }
            Ok(())
        }
        Ty::Record(fields) => validate_record(fields, value, vocab, path),
        Ty::Map(item, max) => {
            let o = value.as_object().ok_or_else(|| fail(path, "expected an object"))?;
            if o.len() > *max {
                return Err(fail(path, format!("more than {max} members")));
            }
            for (k, v) in o.entries() {
                if k.is_empty() || k.len() > 128 {
                    return Err(fail(path, "keys are 1 to 128 bytes"));
                }
                validate_ty(item, v, vocab, &format!("{path}/{k}"))?;
            }
            Ok(())
        }
        Ty::Param => match value.as_str() {
            Some(s) if vocab.registry.get(s).is_some() => Ok(()),
            Some(s) => Err(fail(path, format!("'{s}' is not a registry parameter"))),
            None => Err(fail(path, "expected a parameter id")),
        },
        Ty::ParamValues(filter) => {
            let o = value.as_object().ok_or_else(|| fail(path, "expected an object of parameter values"))?;
            if o.is_empty() {
                return Err(fail(path, "give at least one parameter"));
            }
            for (id, v) in o.entries() {
                let p =
                    vocab.registry.get(id).ok_or_else(|| fail(path, format!("'{id}' is not a registry parameter")))?;
                let here = format!("{path}/{id}");
                if !filter.accepts(&p.group) {
                    return Err(fail(&here, format!("'{id}' (group {}) is not accepted by this tool", p.group)));
                }
                if p.read_only {
                    return Err(fail(&here, "the parameter is read-only (measured)"));
                }
                match decode_value(p.kind, p.dimension, v) {
                    Some(humanart_doc::registry::ParamValue::Enum(c)) if !p.choices.contains(&c) => {
                        return Err(fail(&here, format!("expected one of {}", p.choices.join(", "))));
                    }
                    Some(_) => {}
                    None => return Err(fail(&here, format!("expected a {} value", p.kind.name()))),
                }
                if let Some(x) = v.as_f64()
                    && !x.is_finite()
                {
                    return Err(fail(&here, "not finite"));
                }
            }
            Ok(())
        }
        Ty::Region => match value.as_str() {
            Some(s) if vocab.has_region(s) => Ok(()),
            _ => Err(fail(path, "expected a template region name")),
        },
        Ty::Landmark => match value.as_str() {
            Some(s) if vocab.has_landmark(s) => Ok(()),
            _ => Err(fail(path, "expected a template landmark name")),
        },
        Ty::Joint => match value.as_str() {
            Some(s) if vocab.has_joint(s) => Ok(()),
            _ => Err(fail(path, "expected a skeleton joint name")),
        },
        Ty::Fields => value.as_object().map(|_| ()).ok_or_else(|| fail(path, "expected an object")),
    }
}

/// Checks an object against `fields`: required members present, no unknown members, each member's
/// type.
///
/// # Errors
/// The first mismatch, with its path.
pub fn validate_record(fields: &[Field], value: &Value, vocab: &Vocab<'_>, path: &str) -> Result<(), SchemaError> {
    let empty = Object::new();
    let o = match value {
        Value::Null => &empty,
        Value::Object(o) => o,
        _ => return Err(fail(path, "expected an object")),
    };
    for (k, _) in o.entries() {
        if !fields.iter().any(|f| f.name == k) {
            return Err(fail(path, format!("unknown member '{k}'")));
        }
    }
    for f in fields {
        let here = format!("{path}/{}", f.name);
        match o.get(f.name) {
            None if f.required => return Err(fail(&here, "required")),
            None => {}
            Some(v) => validate_ty(&f.ty, v, vocab, &here)?,
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::tests::{registry, template};
    use humanart_io::json::parse_str;

    const POINT: [Field; 2] =
        [Field::req("x", Ty::Num { min: -1.0, max: 1.0 }, "X."), Field::opt("tag", Ty::Choice(&["a", "b"]), "Tag.")];
    const FIELDS: [Field; 6] = [
        Field::req("values", Ty::ParamValues(ParamFilter::Groups(&["eyes"])), "Values."),
        Field::opt("count", Ty::Int { min: 1, max: 8 }, "Count."),
        Field::opt("id", Ty::Id, "Id."),
        Field::opt("points", Ty::List(&Ty::Record(&POINT), 2), "Points."),
        Field::opt("where", Ty::Region, "Region."),
        Field::opt("rot", Ty::Quat, "Rotation."),
    ];

    fn vocab() -> Vocab<'static> {
        Vocab { registry: registry(), template: template() }
    }

    fn eye_param() -> &'static str {
        registry()
            .params()
            .iter()
            .find(|p| p.group.starts_with("eyes/") && p.kind == ParamKind::Scalar)
            .unwrap()
            .id
            .as_str()
    }

    #[test]
    fn ag_02_schemas_are_derived_from_the_registry_and_template() {
        let v = vocab();
        let s = record_schema(&FIELDS, &v);
        let props = s.get("properties").unwrap();
        let values = props.get("values").unwrap().get("properties").unwrap().as_object().unwrap();
        let eyes = registry()
            .params()
            .iter()
            .filter(|p| ParamFilter::Groups(&["eyes"]).accepts(&p.group) && !p.read_only)
            .count();
        assert_eq!(values.len(), eyes);
        assert!(values.keys().all(|k| registry().get(k).unwrap().group.starts_with("eyes")));
        assert!(values.get(eye_param()).unwrap().get("x-humanart-range").is_some());
        let regions = props.get("where").unwrap().get("enum").unwrap().as_array().unwrap();
        assert_eq!(regions.len(), template().regions().regions.len());
        assert_eq!(s.get("required").unwrap().as_array().unwrap().len(), 1);
        assert_eq!(s.get("additionalProperties"), Some(&Value::Bool(false)));
    }

    #[test]
    fn ag_02_arguments_are_validated_against_the_declaration() {
        let v = vocab();
        let region = template().regions().regions[0].name.clone();
        let p = eye_param();
        let good = parse_str(&format!(
            r#"{{"values":{{"{p}":0.5}},"count":3,"id":"0123456789abcdef0123456789abcdef","points":[{{"x":0.5,"tag":"a"}}],"where":"{region}","rot":[0,0,0,1]}}"#
        ))
        .unwrap();
        validate_record(&FIELDS, &good, &v, "").unwrap();
        let face = registry().params().iter().find(|q| q.group.starts_with("face/")).unwrap().id.clone();
        let bad = [
            ("{}".to_owned(), "/values"),
            (format!(r#"{{"values":{{"{p}":0.5}},"extra":1}}"#), ""),
            (format!(r#"{{"values":{{"{p}":"x"}}}}"#), "/values/"),
            (format!(r#"{{"values":{{"{face}":0.5}}}}"#), "/values/"),
            (r#"{"values":{"no.such.param":1}}"#.to_owned(), "/values"),
            (r#"{"values":{}}"#.to_owned(), "/values"),
            (format!(r#"{{"values":{{"{p}":1}},"count":9}}"#), "/count"),
            (format!(r#"{{"values":{{"{p}":1}},"count":1.5}}"#), "/count"),
            (format!(r#"{{"values":{{"{p}":1}},"id":"XYZ"}}"#), "/id"),
            (format!(r#"{{"values":{{"{p}":1}},"points":[{{"x":2}}]}}"#), "/points/0/x"),
            (format!(r#"{{"values":{{"{p}":1}},"points":[{{}},{{}},{{}}]}}"#), "/points"),
            (format!(r#"{{"values":{{"{p}":1}},"points":[{{"x":0,"tag":"c"}}]}}"#), "/points/0/tag"),
            (format!(r#"{{"values":{{"{p}":1}},"where":"nowhere"}}"#), "/where"),
            (format!(r#"{{"values":{{"{p}":1}},"rot":[0,0,1]}}"#), "/rot"),
        ];
        for (text, path) in bad {
            let e = validate_record(&FIELDS, &parse_str(&text).unwrap(), &v, "").unwrap_err();
            assert!(e.path.starts_with(path), "{text}: {e}");
        }
        assert!(validate_record(&FIELDS, &Value::from(3u64), &v, "").is_err());
    }
}
