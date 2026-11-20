//! The structured result of a command `[CMD-06]`: what changed, warnings, plausibility.

use humanart_core::id::Id128;
use humanart_io::json::{Number, Object, Value};

use crate::registry::{ParamValue, Registry, decode_value, value_to_json};
use crate::text::Edit;

/// What happened to an object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ObjectOp {
    /// Created.
    Added,
    /// Deleted.
    Removed,
    /// Moved to another place in its list.
    Moved,
    /// Fields changed.
    Modified {
        /// The field names, sorted.
        fields: Vec<String>,
    },
}

impl ObjectOp {
    fn name(&self) -> &'static str {
        match self {
            Self::Added => "added",
            Self::Removed => "removed",
            Self::Moved => "moved",
            Self::Modified { .. } => "modified",
        }
    }
}

/// One change a command made.
#[derive(Clone, Debug, PartialEq)]
pub enum Change {
    /// A parameter value; defaults are spelled out.
    Param {
        /// The parameter id.
        id: String,
        /// The value before.
        old: ParamValue,
        /// The value after.
        new: ParamValue,
    },
    /// A joint of the preview pose (`None`: at rest).
    Pose {
        /// The joint.
        joint: String,
        /// Before.
        old: Option<[f64; 4]>,
        /// After.
        new: Option<[f64; 4]>,
    },
    /// An object of a list (layer, groom, preset, lock, pin, snapshot, attachment, reference,
    /// author).
    Object {
        /// Its section kind.
        kind: String,
        /// Its id (the section key).
        id: String,
        /// What happened.
        op: ObjectOp,
    },
    /// Entries of `[meta]` or `[settings]`.
    Section {
        /// The section kind.
        kind: String,
        /// The entry keys that changed.
        keys: Vec<String>,
    },
}

/// The kinds of warnings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WarningCode {
    /// A value was clamped into the parameter's range (the requested value is in the message).
    Clamped,
    /// A value lies outside the parameter's plausible span (allowed; never altered) `[ID-04]`.
    Implausible,
    /// An agent change in a proposal touches something a human changed last; accepting it is the
    /// human's decision `[LOCK-02]`.
    NeedsReview,
    /// The plausibility score dropped.
    PlausibilityDrop,
}

impl WarningCode {
    /// The stable name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Clamped => "clamped",
            Self::Implausible => "implausible",
            Self::NeedsReview => "needs_review",
            Self::PlausibilityDrop => "plausibility_drop",
        }
    }
}

/// A warning attached to a command result `[CMD-05]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Warning {
    /// The kind.
    pub code: WarningCode,
    /// What it is about (a parameter id, an object).
    pub target: String,
    /// A sentence for people and agents.
    pub message: String,
}

/// The structured result of a command `[CMD-06]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    /// The command's id.
    pub command: Id128,
    /// Who made it.
    pub author: String,
    /// The command kind ([`super::Action::kind_name`], or `undo` / `redo`).
    pub kind: String,
    /// The label.
    pub label: String,
    /// When, Unix milliseconds.
    pub time_ms: u64,
    /// The proposal it went to, or `None` for the main document.
    pub proposal: Option<Id128>,
    /// What changed.
    pub changes: Vec<Change>,
    /// Warnings (clamped values, implausible values ...).
    pub warnings: Vec<Warning>,
    /// Vertices, texels or guide points a stroke touched (0 for other commands).
    pub elements: u64,
    /// The plausibility score before and after, when a scorer is installed.
    pub plausibility: Option<[f64; 2]>,
    /// `true` when the command was merged into the previous history entry `[CMD-03]`.
    pub coalesced: bool,
}

impl Outcome {
    /// The JSON form the agent reads.
    pub fn to_json(&self) -> Value {
        let mut o = Object::with_capacity(11);
        o.insert("command", self.command.to_hex().as_str());
        o.insert("author", self.author.as_str());
        o.insert("kind", self.kind.as_str());
        o.insert("label", self.label.as_str());
        o.insert("time", Value::from(Number::from(self.time_ms)));
        o.insert("proposal", self.proposal.map_or(Value::Null, |p| Value::from(p.to_hex().as_str())));
        o.insert("changes", Value::Array(self.changes.iter().map(change_json).collect()));
        o.insert("warnings", Value::Array(self.warnings.iter().map(warning_json).collect()));
        o.insert("elements", Value::from(Number::from(self.elements)));
        o.insert(
            "plausibility",
            self.plausibility.map_or(Value::Null, |[a, b]| {
                Value::Array(vec![Value::from(Number::Float(a)), Value::from(Number::Float(b))])
            }),
        );
        o.insert("coalesced", self.coalesced);
        Value::Object(o)
    }

    /// One line for the history panel and the agent: "human.artist set face.nose.bridge_height
    /// 0.35 -> 0.47 (+0.12)".
    pub fn summary(&self) -> String {
        let mut parts = vec![format!("{} {}", self.author, self.kind)];
        for change in self.changes.iter().take(4) {
            parts.push(match change {
                Change::Param { id, old, new } => {
                    let delta = match (old.as_scalar(), new.as_scalar()) {
                        (Some(a), Some(b)) => format!(" ({:+.3})", b - a),
                        _ => String::new(),
                    };
                    format!("{id} {} -> {}{delta}", short(old), short(new))
                }
                Change::Pose { joint, .. } => format!("pose {joint}"),
                Change::Object { kind, id, op } => format!("{kind} {id} {}", op.name()),
                Change::Section { kind, keys } => format!("{kind} {}", keys.join(",")),
            });
        }
        if self.changes.len() > 4 {
            parts.push(format!("and {} more", self.changes.len() - 4));
        }
        parts.join("; ")
    }
}

fn short(value: &ParamValue) -> String {
    let mut s = String::new();
    // Writing a value the document accepted cannot fail (it is finite); an empty string is shown
    // otherwise.
    if humanart_io::json::write(&value_to_json(value), humanart_io::json::Style::COMPACT, &mut s).is_err() {
        s.clear();
    }
    s
}

fn quat_json(q: Option<[f64; 4]>) -> Value {
    q.map_or(Value::Null, |q| Value::Array(q.iter().map(|v| Value::from(Number::Float(*v))).collect()))
}

fn change_json(change: &Change) -> Value {
    let mut o = Object::with_capacity(4);
    match change {
        Change::Param { id, old, new } => {
            o.insert("type", "param");
            o.insert("id", id.as_str());
            o.insert("old", value_to_json(old));
            o.insert("new", value_to_json(new));
        }
        Change::Pose { joint, old, new } => {
            o.insert("type", "pose");
            o.insert("joint", joint.as_str());
            o.insert("old", quat_json(*old));
            o.insert("new", quat_json(*new));
        }
        Change::Object { kind, id, op } => {
            o.insert("type", "object");
            o.insert("kind", kind.as_str());
            o.insert("id", id.as_str());
            o.insert("op", op.name());
            if let ObjectOp::Modified { fields } = op {
                o.insert("fields", Value::Array(fields.iter().map(Value::from).collect()));
            }
        }
        Change::Section { kind, keys } => {
            o.insert("type", "section");
            o.insert("kind", kind.as_str());
            o.insert("keys", Value::Array(keys.iter().map(Value::from).collect()));
        }
    }
    Value::Object(o)
}

fn warning_json(w: &Warning) -> Value {
    let mut o = Object::with_capacity(3);
    o.insert("code", w.code.name());
    o.insert("target", w.target.as_str());
    o.insert("message", w.message.as_str());
    Value::Object(o)
}

fn param_value(registry: &Registry, id: &str, json: Option<&Value>) -> Option<ParamValue> {
    let param = registry.get(id)?;
    match json {
        None => Some(param.default.clone()),
        Some(v) => decode_value(param.kind, param.dimension, v),
    }
}

fn quat(json: Option<&Value>) -> Option<[f64; 4]> {
    let items = json?.as_array()?;
    let mut q = [0.0; 4];
    if items.len() != 4 {
        return None;
    }
    for (out, item) in q.iter_mut().zip(items) {
        *out = item.as_f64()?;
    }
    Some(q)
}

/// The changes `edits` make, in a stable order: parameters and pose joints in edit order, then
/// objects and sections in edit order, one entry per object.
pub fn describe(edits: &[Edit], registry: &Registry) -> Vec<Change> {
    let mut out: Vec<Change> = Vec::new();
    for edit in edits {
        match edit {
            Edit::Set { section, key, before, after } => match section.kind.as_str() {
                "params" => {
                    if let (Some(old), Some(new)) =
                        (param_value(registry, key, before.as_ref()), param_value(registry, key, after.as_ref()))
                    {
                        out.push(Change::Param { id: key.clone(), old, new });
                    }
                }
                "param_authors" => {}
                "pose" => {
                    out.push(Change::Pose {
                        joint: key.clone(),
                        old: quat(before.as_ref()),
                        new: quat(after.as_ref()),
                    });
                }
                kind => {
                    if let Some(id) = &section.key {
                        let existing = out.iter_mut().find_map(|c| match c {
                            Change::Object { kind: k, id: i, op } if k == kind && i == id => Some(op),
                            _ => None,
                        });
                        match existing {
                            Some(ObjectOp::Modified { fields }) => {
                                if let Err(at) = fields.binary_search(key) {
                                    fields.insert(at, key.clone());
                                }
                            }
                            Some(_) => {}
                            None => out.push(Change::Object {
                                kind: kind.to_owned(),
                                id: id.clone(),
                                op: ObjectOp::Modified { fields: vec![key.clone()] },
                            }),
                        }
                    } else {
                        let existing = out.iter_mut().find_map(|c| match c {
                            Change::Section { kind: k, keys } if k == kind => Some(keys),
                            _ => None,
                        });
                        match existing {
                            Some(keys) => {
                                if let Err(at) = keys.binary_search(key) {
                                    keys.insert(at, key.clone());
                                }
                            }
                            None => out.push(Change::Section { kind: kind.to_owned(), keys: vec![key.clone()] }),
                        }
                    }
                }
            },
            Edit::Insert { section, .. } | Edit::Remove { section, .. } => {
                let inserted = matches!(edit, Edit::Insert { .. });
                let kind = section.kind().to_owned();
                let id = section.key().unwrap_or_default().to_owned();
                let existing = out.iter_mut().find_map(|c| match c {
                    Change::Object { kind: k, id: i, op } if *k == kind && *i == id => Some(op),
                    _ => None,
                });
                match existing {
                    // Removed then inserted again: a move (`text::diff` moves sections that way).
                    Some(op @ ObjectOp::Removed) if inserted => *op = ObjectOp::Moved,
                    Some(op) => *op = if inserted { ObjectOp::Added } else { ObjectOp::Removed },
                    None => {
                        let op = if inserted { ObjectOp::Added } else { ObjectOp::Removed };
                        out.push(Change::Object { kind, id, op });
                    }
                }
            }
        }
    }
    out
}
