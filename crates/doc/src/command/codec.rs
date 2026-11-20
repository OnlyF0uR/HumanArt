//! The JSON form of commands and edits `[CMD-01]`: what the agent server sends, what the journal
//! and proposal files store.
//!
//! The input comes from agents and from files, so it is treated as hostile `[ROB-01]`: the caller
//! parses the JSON with its own size and depth limits, and decoding here checks every count against
//! [`CommandLimits`], nests batches at most [`super::resolve::MAX_BATCH_DEPTH`] deep, refuses
//! unknown members and never indexes without checking.
//!
//! ```text
//! command = {"id": hex?, "author": str, "time": u64?, "label": str?, "gesture": bool?, "action": action}
//! action  = {"type": "set_params", "values": {param: value}, "relative": bool?, "policy": "reject"|"clamp"?}
//!         | {"type": "reset_params", "ids": [param]}
//!         | {"type": "set_pose", "joints": {joint: [x, y, z, w] | null}}
//!         | {"type": "apply_expression_preset" | "apply_pose_preset", "preset": hex}
//!         | {"type": "add_object", "kind": kind, "id": hex?, "index": u64?, "fields": {...}}
//!         | {"type": "modify_object", "kind": kind, "id": hex, "fields": {...}}
//!         | {"type": "delete_object", "kind": kind, "id": hex}
//!         | {"type": "reorder_objects", "kind": kind, "order": [hex]}
//!         | {"type": "stroke", "stroke": "sculpt"|"paint"|"guides", "target": hex, "blob": hex|null,
//!            "regions": [str]?, "elements": u64?}
//!         | {"type": "set_lock", "lock": "region"|"parameter", "target": str, "locked": bool}
//!         | {"type": "set_object_lock", "kind": kind, "id": hex, "locked": bool}
//!         | {"type": "pin_landmark", "landmark": str, "position": [x, y, z]}
//!         | {"type": "unpin_landmark", "landmark": str}
//!         | {"type": "create_snapshot", "id": hex?, "name": str, "state": hex|null?}
//!         | {"type": "restore_snapshot", "id": hex}
//!         | {"type": "add_author", "id": str, "kind": "human"|"agent", "name": str}
//!         | {"type": "set_settings", "protect_human_edits": bool}
//!         | {"type": "batch", "actions": [action]}
//!         | {"type": "edits", "edits": [edit]}
//! edit    = {"op": "set", "section": kind, "section_key": str?, "key": str, "before": json?, "after": json?}
//!         | {"op": "insert"|"remove", "index": u64, "section": kind, "section_key": str?, "entries": {...}}
//! ```
//!
//! A member that is absent and one that is `null` differ for `before` / `after` (absent: the entry
//! does not exist).

use humanart_core::hash::Hash128;
use humanart_core::id::Id128;
use humanart_io::json::{Number, Object, Value};

use super::resolve::MAX_BATCH_DEPTH;
use super::{Action, Command, ObjectKind, RangePolicy, StrokeKind};
use crate::document::{Author, AuthorKind, LockTarget};
use crate::error::{DocError, Result};
use crate::registry::{Registry, decode_value, value_to_json};
use crate::text::{Edit, Section, SectionRef};

/// Size limits of decoding commands and edits `[ROB-01]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommandLimits {
    /// Most edits in one list.
    pub max_edits: usize,
    /// Most actions in one batch (all levels together).
    pub max_actions: usize,
    /// Most values in one parameter or pose action, ids in a reset, objects in an order, regions.
    pub max_values: usize,
    /// Largest section index an edit may name.
    pub max_index: usize,
}

impl CommandLimits {
    /// 65 536 edits, 1024 actions, 4096 values, indices below 2^20.
    pub const DEFAULT: Self = Self { max_edits: 1 << 16, max_actions: 1024, max_values: 4096, max_index: 1 << 20 };
}

impl Default for CommandLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

fn bad(key: &str, what: impl Into<String>) -> DocError {
    DocError::field("command", key, what)
}

fn object<'a>(v: &'a Value, what: &str) -> Result<&'a Object> {
    v.as_object().ok_or_else(|| bad(what, "expected an object"))
}

fn member<'a>(o: &'a Object, key: &str) -> Result<&'a Value> {
    o.get(key).ok_or_else(|| bad(key, "missing"))
}

fn known(o: &Object, keys: &[&str]) -> Result<()> {
    match o.entries().iter().find(|(k, _)| !keys.contains(&k.as_str())) {
        Some((k, _)) => Err(bad(k, "unknown member")),
        None => Ok(()),
    }
}

fn string<'a>(o: &'a Object, key: &str) -> Result<&'a str> {
    member(o, key)?.as_str().ok_or_else(|| bad(key, "expected a string"))
}

fn opt_u64(o: &Object, key: &str) -> Result<Option<u64>> {
    match o.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v.as_u64().map(Some).ok_or_else(|| bad(key, "expected an unsigned integer")),
    }
}

fn boolean(o: &Object, key: &str, default: Option<bool>) -> Result<bool> {
    match (o.get(key), default) {
        (None, Some(d)) => Ok(d),
        (Some(v), _) => v.as_bool().ok_or_else(|| bad(key, "expected true or false")),
        (None, None) => Err(bad(key, "missing")),
    }
}

fn id_of(v: &Value, key: &str) -> Result<Id128> {
    v.as_str().and_then(|s| Id128::parse(s).ok()).ok_or_else(|| bad(key, "expected a 32-digit hex id"))
}

fn opt_id(o: &Object, key: &str) -> Result<Id128> {
    match o.get(key) {
        None | Some(Value::Null) => Ok(Id128::NIL),
        Some(v) => id_of(v, key),
    }
}

fn hash_of(o: &Object, key: &str) -> Result<Hash128> {
    match o.get(key) {
        None | Some(Value::Null) => Ok(Hash128::ZERO),
        Some(v) => v.as_str().and_then(|s| Hash128::parse(s).ok()).ok_or_else(|| bad(key, "expected a hash or null")),
    }
}

fn hash_json(h: Hash128) -> Value {
    if h.is_zero() { Value::Null } else { Value::from(h.to_hex().as_str()) }
}

fn float_array<const N: usize>(v: &Value, key: &str) -> Result<[f64; N]> {
    let items = v.as_array().ok_or_else(|| bad(key, "expected an array of numbers"))?;
    if items.len() != N {
        return Err(bad(key, format!("expected {N} numbers")));
    }
    let mut out = [0.0; N];
    for (o, item) in out.iter_mut().zip(items) {
        *o = item.as_f64().filter(|x| x.is_finite()).ok_or_else(|| bad(key, "expected a finite number"))?;
    }
    Ok(out)
}

fn floats(vs: &[f64]) -> Value {
    Value::Array(vs.iter().map(|v| Value::from(Number::Float(*v))).collect())
}

fn kind_of(o: &Object) -> Result<ObjectKind> {
    let name = string(o, "kind")?;
    ObjectKind::from_section(name).ok_or_else(|| bad("kind", format!("unknown object kind {name}")))
}

fn limited<'a>(v: &'a Value, key: &str, limit: usize) -> Result<&'a [Value]> {
    let items = v.as_array().ok_or_else(|| bad(key, "expected an array"))?;
    if items.len() > limit {
        return Err(DocError::Limit { what: "items in one command list", limit: limit as u64 });
    }
    Ok(items)
}

fn limited_object<'a>(v: &'a Value, key: &str, limit: usize) -> Result<&'a Object> {
    let o = object(v, key)?;
    if o.len() > limit {
        return Err(DocError::Limit { what: "members in one command object", limit: limit as u64 });
    }
    Ok(o)
}

// --- Edits -------------------------------------------------------------------------------------

/// The JSON form of an edit.
pub fn edit_to_json(edit: &Edit) -> Value {
    let mut o = Object::with_capacity(6);
    let section_fields = |o: &mut Object, kind: &str, key: Option<&str>| {
        o.insert("section", kind);
        if let Some(key) = key {
            o.insert("section_key", key);
        }
    };
    match edit {
        Edit::Set { section, key, before, after } => {
            o.insert("op", "set");
            section_fields(&mut o, &section.kind, section.key.as_deref());
            o.insert("key", key.as_str());
            if let Some(b) = before {
                o.insert("before", b.clone());
            }
            if let Some(a) = after {
                o.insert("after", a.clone());
            }
        }
        Edit::Insert { index, section } | Edit::Remove { index, section } => {
            o.insert("op", if matches!(edit, Edit::Insert { .. }) { "insert" } else { "remove" });
            o.insert("index", Value::from(Number::from(*index)));
            section_fields(&mut o, section.kind(), section.key());
            let mut entries = Object::with_capacity(section.len());
            for (k, v) in section.entries() {
                entries.insert(k.as_str(), v.clone());
            }
            o.insert("entries", Value::Object(entries));
        }
    }
    Value::Object(o)
}

/// Reads an edit.
pub fn edit_from_json(value: &Value, limits: &CommandLimits) -> Result<Edit> {
    let o = object(value, "edit")?;
    let kind = string(o, "section")?;
    let section_key = match o.get("section_key") {
        None => None,
        Some(v) => Some(v.as_str().ok_or_else(|| bad("section_key", "expected a string"))?),
    };
    // Section::new checks the grammar of kind and key.
    let empty = Section::new(kind, section_key)?;
    match string(o, "op")? {
        "set" => {
            known(o, &["op", "section", "section_key", "key", "before", "after"])?;
            let key = string(o, "key")?;
            if !crate::text::is_key(key) {
                return Err(bad("key", "invalid entry key"));
            }
            Ok(Edit::Set {
                section: SectionRef::new(kind, section_key),
                key: key.to_owned(),
                before: o.get("before").cloned(),
                after: o.get("after").cloned(),
            })
        }
        op @ ("insert" | "remove") => {
            known(o, &["op", "index", "section", "section_key", "entries"])?;
            let index = opt_u64(o, "index")?.ok_or_else(|| bad("index", "missing"))?;
            if index > limits.max_index as u64 {
                return Err(DocError::Limit { what: "section index in an edit", limit: limits.max_index as u64 });
            }
            let entries = limited_object(member(o, "entries")?, "entries", limits.max_values)?;
            let mut section = empty;
            for (k, v) in entries.entries() {
                section.set(k, v.clone())?;
            }
            let index = index as usize;
            Ok(if op == "insert" { Edit::Insert { index, section } } else { Edit::Remove { index, section } })
        }
        other => Err(bad("op", format!("unknown edit op {other}"))),
    }
}

/// The JSON array of edits.
pub fn edits_to_json(edits: &[Edit]) -> Value {
    Value::Array(edits.iter().map(edit_to_json).collect())
}

/// Reads an array of edits.
pub fn edits_from_json(value: &Value, limits: &CommandLimits) -> Result<Vec<Edit>> {
    let items = limited(value, "edits", limits.max_edits)?;
    items.iter().map(|v| edit_from_json(v, limits)).collect()
}

// --- Actions -----------------------------------------------------------------------------------

/// The JSON form of an author.
pub fn author_to_json(author: &Author) -> Value {
    let mut o = Object::with_capacity(3);
    o.insert("id", author.id.as_str());
    o.insert("kind", if author.kind == AuthorKind::Human { "human" } else { "agent" });
    o.insert("name", author.name.as_str());
    Value::Object(o)
}

fn author_from(o: &Object) -> Result<Author> {
    let kind = match string(o, "kind")? {
        "human" => AuthorKind::Human,
        "agent" => AuthorKind::Agent,
        other => return Err(bad("kind", format!("unknown author kind {other}"))),
    };
    Ok(Author { id: string(o, "id")?.to_owned(), kind, name: string(o, "name")?.to_owned() })
}

/// The JSON form of an action.
pub fn action_to_json(action: &Action) -> Value {
    let mut o = Object::with_capacity(6);
    let kind_name = match action {
        Action::AddObject { .. } => "add_object",
        Action::ModifyObject { .. } => "modify_object",
        Action::DeleteObject { .. } => "delete_object",
        Action::ReorderObjects { .. } => "reorder_objects",
        Action::Stroke { .. } => "stroke",
        other => other.kind_name(),
    };
    o.insert("type", kind_name);
    match action {
        Action::SetParams { values, relative, policy } => {
            let mut v = Object::with_capacity(values.len());
            for (id, value) in values {
                v.insert(id.as_str(), value_to_json(value));
            }
            o.insert("values", Value::Object(v));
            o.insert("relative", *relative);
            o.insert("policy", if *policy == RangePolicy::Clamp { "clamp" } else { "reject" });
        }
        Action::ResetParams { ids } => {
            o.insert("ids", Value::Array(ids.iter().map(Value::from).collect()));
        }
        Action::SetPose { joints } => {
            let mut j = Object::with_capacity(joints.len());
            for (name, q) in joints {
                j.insert(name.as_str(), q.map_or(Value::Null, |q| floats(&q)));
            }
            o.insert("joints", Value::Object(j));
        }
        Action::ApplyExpressionPreset { preset } | Action::ApplyPosePreset { preset } => {
            o.insert("preset", preset.to_hex().as_str());
        }
        Action::AddObject { kind, id, index, fields } => {
            o.insert("kind", kind.section());
            if !id.is_nil() {
                o.insert("id", id.to_hex().as_str());
            }
            if let Some(i) = index {
                o.insert("index", Value::from(Number::from(*i)));
            }
            o.insert("fields", Value::Object(fields.clone()));
        }
        Action::ModifyObject { kind, id, fields } => {
            o.insert("kind", kind.section());
            o.insert("id", id.to_hex().as_str());
            o.insert("fields", Value::Object(fields.clone()));
        }
        Action::DeleteObject { kind, id } => {
            o.insert("kind", kind.section());
            o.insert("id", id.to_hex().as_str());
        }
        Action::ReorderObjects { kind, order } => {
            o.insert("kind", kind.section());
            o.insert("order", Value::Array(order.iter().map(|i| Value::from(i.to_hex().as_str())).collect()));
        }
        Action::Stroke { stroke, target, blob, regions, elements } => {
            o.insert("stroke", stroke.name());
            o.insert("target", target.to_hex().as_str());
            o.insert("blob", hash_json(*blob));
            o.insert("regions", Value::Array(regions.iter().map(Value::from).collect()));
            o.insert("elements", Value::from(Number::from(*elements)));
        }
        Action::SetLock { target, locked } => {
            let (kind, name) = match target {
                LockTarget::Region(r) => ("region", r),
                LockTarget::Parameter(p) => ("parameter", p),
            };
            o.insert("lock", kind);
            o.insert("target", name.as_str());
            o.insert("locked", *locked);
        }
        Action::SetObjectLock { kind, id, locked } => {
            o.insert("kind", kind.section());
            o.insert("id", id.to_hex().as_str());
            o.insert("locked", *locked);
        }
        Action::PinLandmark { landmark, position } => {
            o.insert("landmark", landmark.as_str());
            o.insert("position", floats(position));
        }
        Action::UnpinLandmark { landmark } => {
            o.insert("landmark", landmark.as_str());
        }
        Action::CreateSnapshot { id, name, state } => {
            if !id.is_nil() {
                o.insert("id", id.to_hex().as_str());
            }
            o.insert("name", name.as_str());
            o.insert("state", hash_json(*state));
        }
        Action::RestoreSnapshot { id } => {
            o.insert("id", id.to_hex().as_str());
        }
        Action::AddAuthor { author } => {
            if let Value::Object(a) = author_to_json(author) {
                for (k, v) in a {
                    o.insert(k, v);
                }
            }
        }
        Action::SetSettings { protect_human_edits } => {
            o.insert("protect_human_edits", *protect_human_edits);
        }
        Action::Batch { actions } => {
            o.insert("actions", Value::Array(actions.iter().map(action_to_json).collect()));
        }
        Action::Edits { edits } => {
            o.insert("edits", edits_to_json(edits));
        }
    }
    Value::Object(o)
}

/// Reads an action. Parameter values are read in the shape of their registry entry; ranges are
/// checked when the command is applied (by its [`RangePolicy`]).
pub fn action_from_json(value: &Value, registry: &Registry, limits: &CommandLimits) -> Result<Action> {
    let mut budget = limits.max_actions;
    read_action(value, registry, limits, 0, &mut budget)
}

fn read_action(
    value: &Value,
    registry: &Registry,
    limits: &CommandLimits,
    depth: u32,
    budget: &mut usize,
) -> Result<Action> {
    if *budget == 0 {
        return Err(DocError::Limit { what: "actions in one command", limit: limits.max_actions as u64 });
    }
    *budget -= 1;
    let o = object(value, "action")?;
    let ty = string(o, "type")?;
    Ok(match ty {
        "set_params" => {
            known(o, &["type", "values", "relative", "policy"])?;
            let values = limited_object(member(o, "values")?, "values", limits.max_values)?;
            let mut out = Vec::with_capacity(values.len());
            for (id, v) in values.entries() {
                let param = registry.require(id)?;
                let decoded = decode_value(param.kind, param.dimension, v)
                    .ok_or_else(|| DocError::param_value(id.as_str(), format!("not a {} value", param.kind.name())))?;
                out.push((id.clone(), decoded));
            }
            let policy = match o.get("policy").map(|p| p.as_str()) {
                None | Some(Some("reject")) => RangePolicy::Reject,
                Some(Some("clamp")) => RangePolicy::Clamp,
                Some(_) => return Err(bad("policy", "expected \"reject\" or \"clamp\"")),
            };
            Action::SetParams { values: out, relative: boolean(o, "relative", Some(false))?, policy }
        }
        "reset_params" => {
            known(o, &["type", "ids"])?;
            let ids = limited(member(o, "ids")?, "ids", limits.max_values)?;
            let ids = ids
                .iter()
                .map(|v| v.as_str().map(str::to_owned).ok_or_else(|| bad("ids", "expected strings")))
                .collect::<Result<Vec<_>>>()?;
            Action::ResetParams { ids }
        }
        "set_pose" => {
            known(o, &["type", "joints"])?;
            let joints = limited_object(member(o, "joints")?, "joints", limits.max_values)?;
            let mut out = Vec::with_capacity(joints.len());
            for (name, q) in joints.entries() {
                let q = if q.is_null() { None } else { Some(float_array::<4>(q, "joints")?) };
                out.push((name.clone(), q));
            }
            Action::SetPose { joints: out }
        }
        "apply_expression_preset" | "apply_pose_preset" => {
            known(o, &["type", "preset"])?;
            let preset = id_of(member(o, "preset")?, "preset")?;
            if ty == "apply_pose_preset" {
                Action::ApplyPosePreset { preset }
            } else {
                Action::ApplyExpressionPreset { preset }
            }
        }
        "add_object" => {
            known(o, &["type", "kind", "id", "index", "fields"])?;
            let fields = limited_object(member(o, "fields")?, "fields", limits.max_values)?.clone();
            let index = opt_u64(o, "index")?.map(|i| i.min(limits.max_index as u64) as usize);
            Action::AddObject { kind: kind_of(o)?, id: opt_id(o, "id")?, index, fields }
        }
        "modify_object" => {
            known(o, &["type", "kind", "id", "fields"])?;
            let fields = limited_object(member(o, "fields")?, "fields", limits.max_values)?.clone();
            Action::ModifyObject { kind: kind_of(o)?, id: id_of(member(o, "id")?, "id")?, fields }
        }
        "delete_object" => {
            known(o, &["type", "kind", "id"])?;
            Action::DeleteObject { kind: kind_of(o)?, id: id_of(member(o, "id")?, "id")? }
        }
        "reorder_objects" => {
            known(o, &["type", "kind", "order"])?;
            let order = limited(member(o, "order")?, "order", limits.max_values)?;
            let order = order.iter().map(|v| id_of(v, "order")).collect::<Result<Vec<_>>>()?;
            Action::ReorderObjects { kind: kind_of(o)?, order }
        }
        "stroke" => {
            known(o, &["type", "stroke", "target", "blob", "regions", "elements"])?;
            let name = string(o, "stroke")?;
            let stroke = StrokeKind::ALL
                .iter()
                .find(|(_, n)| *n == name)
                .map(|(k, _)| *k)
                .ok_or_else(|| bad("stroke", format!("unknown stroke kind {name}")))?;
            let regions = match o.get("regions") {
                None => Vec::new(),
                Some(v) => limited(v, "regions", limits.max_values)?
                    .iter()
                    .map(|r| r.as_str().map(str::to_owned).ok_or_else(|| bad("regions", "expected strings")))
                    .collect::<Result<Vec<_>>>()?,
            };
            Action::Stroke {
                stroke,
                target: id_of(member(o, "target")?, "target")?,
                blob: hash_of(o, "blob")?,
                regions,
                elements: opt_u64(o, "elements")?.unwrap_or(0),
            }
        }
        "set_lock" => {
            known(o, &["type", "lock", "target", "locked"])?;
            let name = string(o, "target")?.to_owned();
            let target = match string(o, "lock")? {
                "region" => LockTarget::Region(name),
                "parameter" => LockTarget::Parameter(name),
                other => return Err(bad("lock", format!("unknown lock kind {other}"))),
            };
            Action::SetLock { target, locked: boolean(o, "locked", None)? }
        }
        "set_object_lock" => {
            known(o, &["type", "kind", "id", "locked"])?;
            Action::SetObjectLock {
                kind: kind_of(o)?,
                id: id_of(member(o, "id")?, "id")?,
                locked: boolean(o, "locked", None)?,
            }
        }
        "pin_landmark" => {
            known(o, &["type", "landmark", "position"])?;
            Action::PinLandmark {
                landmark: string(o, "landmark")?.to_owned(),
                position: float_array::<3>(member(o, "position")?, "position")?,
            }
        }
        "unpin_landmark" => {
            known(o, &["type", "landmark"])?;
            Action::UnpinLandmark { landmark: string(o, "landmark")?.to_owned() }
        }
        "create_snapshot" => {
            known(o, &["type", "id", "name", "state"])?;
            Action::CreateSnapshot {
                id: opt_id(o, "id")?,
                name: string(o, "name")?.to_owned(),
                state: hash_of(o, "state")?,
            }
        }
        "restore_snapshot" => {
            known(o, &["type", "id"])?;
            Action::RestoreSnapshot { id: id_of(member(o, "id")?, "id")? }
        }
        "add_author" => {
            known(o, &["type", "id", "kind", "name"])?;
            Action::AddAuthor { author: author_from(o)? }
        }
        "set_settings" => {
            known(o, &["type", "protect_human_edits"])?;
            Action::SetSettings { protect_human_edits: boolean(o, "protect_human_edits", None)? }
        }
        "batch" => {
            known(o, &["type", "actions"])?;
            if depth >= MAX_BATCH_DEPTH {
                return Err(DocError::Limit { what: "nested batches", limit: u64::from(MAX_BATCH_DEPTH) });
            }
            let items = limited(member(o, "actions")?, "actions", limits.max_actions)?;
            let mut actions = Vec::with_capacity(items.len());
            for item in items {
                actions.push(read_action(item, registry, limits, depth + 1, budget)?);
            }
            Action::Batch { actions }
        }
        "edits" => {
            known(o, &["type", "edits"])?;
            Action::Edits { edits: edits_from_json(member(o, "edits")?, limits)? }
        }
        other => return Err(bad("type", format!("unknown action type {other}"))),
    })
}

/// The JSON form of a command.
pub fn command_to_json(command: &Command) -> Value {
    let mut o = Object::with_capacity(6);
    if !command.id.is_nil() {
        o.insert("id", command.id.to_hex().as_str());
    }
    o.insert("author", command.author.as_str());
    o.insert("time", Value::from(Number::from(command.time_ms)));
    o.insert("label", command.label.as_str());
    o.insert("gesture", command.gesture);
    o.insert("action", action_to_json(&command.action));
    Value::Object(o)
}

/// Reads a command.
pub fn command_from_json(value: &Value, registry: &Registry, limits: &CommandLimits) -> Result<Command> {
    let o = object(value, "command")?;
    known(o, &["id", "author", "time", "label", "gesture", "action"])?;
    Ok(Command {
        id: opt_id(o, "id")?,
        author: string(o, "author")?.to_owned(),
        time_ms: opt_u64(o, "time")?.unwrap_or(0),
        label: match o.get("label") {
            None => String::new(),
            Some(v) => v.as_str().ok_or_else(|| bad("label", "expected a string"))?.to_owned(),
        },
        gesture: boolean(o, "gesture", Some(false))?,
        action: action_from_json(member(o, "action")?, registry, limits)?,
    })
}
