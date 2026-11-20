//! Turning an [`Action`] into text edits against the current document `[CMD-01]`, `[CMD-05]`.
//!
//! Parameter and pose actions build their `Set` edits directly from the typed document (cheap: a
//! slider move costs a few small allocations). Object actions change a copy of the document text
//! and read it back, so that every field is checked exactly as a file is and the edits come out in
//! canonical form.

use humanart_core::hash::Hash128;
use humanart_core::id::Id128;
use humanart_io::json::{Number, Object, Value};

use super::outcome::{Warning, WarningCode};
use super::{Action, Command, CommandError, ObjectKind, RangePolicy, Rejection, StrokeKind, derive_id};
use crate::blob::{BlobStore, Compression};
use crate::document::{Document, LockTarget};
use crate::error::DocError;
use crate::registry::{Param, ParamValue, Registry, value_to_json};
use crate::text::{self, Edit, Section, SectionRef, TextFile, TextLimits};

/// Deepest nesting of [`Action::Batch`].
pub const MAX_BATCH_DEPTH: u32 = 4;

/// What resolving needs besides the document.
#[derive(Clone, Copy)]
pub struct Context<'a> {
    /// The parameter registry.
    pub registry: &'a Registry,
    /// The project's blob store: snapshots are written to it and stroke blobs must be in it
    /// `[SAVE-02]`. `None` for documents without a project (snapshots then need a given state).
    pub blobs: Option<&'a BlobStore<'a>>,
}

/// The resolved form of a command.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Resolved {
    /// The forward edits.
    pub edits: Vec<Edit>,
    /// Warnings found while resolving (clamped and implausible values).
    pub warnings: Vec<Warning>,
    /// Regions strokes declare they touched (for region locks).
    pub regions: Vec<String>,
    /// Elements strokes touched.
    pub elements: u64,
}

/// Resolves `command` against `doc`. Nothing is changed (except that a new snapshot's text is put
/// into the blob store, which is harmless if the command is then refused).
pub fn resolve(doc: &Document, command: &Command, ctx: &Context<'_>) -> Result<Resolved, CommandError> {
    let mut out = Resolved::default();
    let mut ids = 0;
    resolve_action(doc, command, &command.action, ctx, 0, &mut ids, &mut out)?;
    Ok(out)
}

fn invalid(section: &str, key: &str, what: impl Into<String>) -> CommandError {
    CommandError::Invalid(DocError::field(section, key, what))
}

fn hash_json(h: Hash128) -> Value {
    if h.is_zero() { Value::Null } else { Value::from(h.to_hex().as_str()) }
}

fn float(v: f64) -> Value {
    Value::from(Number::Float(v))
}

fn floats(vs: &[f64]) -> Value {
    Value::Array(vs.iter().map(|v| float(*v)).collect())
}

fn resolve_action(
    doc: &Document,
    cmd: &Command,
    action: &Action,
    ctx: &Context<'_>,
    depth: u32,
    ids: &mut u64,
    out: &mut Resolved,
) -> Result<(), CommandError> {
    let mut next_id = || {
        *ids += 1;
        derive_id(cmd.id, *ids)
    };
    match action {
        Action::SetParams { values, relative, policy } => {
            set_params(doc, ctx.registry, &cmd.author, values, *relative, *policy, out)?;
        }
        Action::ResetParams { ids } => {
            let mut values = Vec::with_capacity(ids.len());
            for id in ids {
                values.push((id.clone(), ctx.registry.require(id)?.default.clone()));
            }
            set_params(doc, ctx.registry, &cmd.author, &values, false, RangePolicy::Reject, out)?;
        }
        Action::SetPose { joints } => set_pose(doc, joints, out)?,
        Action::ApplyExpressionPreset { preset } => {
            let p = doc
                .expression_presets
                .iter()
                .find(|p| p.id == *preset)
                .ok_or_else(|| invalid("expression_preset", &preset.to_hex(), "no such preset"))?;
            let mut values = Vec::new();
            for id in p.weights.keys() {
                let param = ctx.registry.require(id)?;
                if !ctx.registry.in_group("expression").any(|q| q.id == param.id) {
                    return Err(invalid("expression_preset", id, "not an expression parameter"));
                }
            }
            for param in ctx.registry.in_group("expression") {
                if param.read_only || param.default.as_scalar().is_none() {
                    continue;
                }
                let value = p.weights.get(&param.id).map_or_else(|| param.default.clone(), |w| ParamValue::Scalar(*w));
                values.push((param.id.clone(), value));
            }
            set_params(doc, ctx.registry, &cmd.author, &values, false, RangePolicy::Reject, out)?;
        }
        Action::ApplyPosePreset { preset } => {
            let p = doc
                .pose_presets
                .iter()
                .find(|p| p.id == *preset)
                .ok_or_else(|| invalid("pose_preset", &preset.to_hex(), "no such preset"))?;
            let mut joints: Vec<(String, Option<[f64; 4]>)> =
                doc.pose.keys().filter(|j| !p.joints.contains_key(*j)).map(|j| (j.clone(), None)).collect();
            joints.extend(p.joints.iter().map(|(j, q)| (j.clone(), Some(*q))));
            set_pose(doc, &joints, out)?;
        }
        Action::AddObject { kind, id, index, fields } => {
            let id = if id.is_nil() { next_id() } else { *id };
            add_object(doc, cmd, ctx, *kind, id, *index, fields, out)?;
        }
        Action::ModifyObject { kind, id, fields } => {
            let file = doc.to_text()?;
            let key = id.to_hex();
            let mut next = file.clone();
            let section = next
                .section_mut(kind.section(), Some(&key))
                .ok_or_else(|| invalid(kind.section(), &key, "no such object"))?;
            for (field, value) in fields.entries() {
                if matches!(field.as_str(), "author" | "created_by_command" | "locked") {
                    return Err(invalid(kind.section(), field, "this field cannot be changed by modify_object"));
                }
                if section.get(field).is_none() {
                    return Err(invalid(kind.section(), field, "no such field"));
                }
                section.set(field, value.clone())?;
            }
            out.edits.extend(via_text(ctx.registry, &file, &next)?);
        }
        Action::DeleteObject { kind, id } => {
            let file = doc.to_text()?;
            let key = id.to_hex();
            let index =
                file.find(kind.section(), Some(&key)).ok_or_else(|| invalid(kind.section(), &key, "no such object"))?;
            let section = file.sections()[index].clone();
            out.edits.push(Edit::Remove { index, section });
        }
        Action::ReorderObjects { kind, order } => {
            let file = doc.to_text()?;
            let slots: Vec<usize> =
                (0..file.sections().len()).filter(|i| file.sections()[*i].kind() == kind.section()).collect();
            let mut wanted: Vec<String> = order.iter().map(|id| id.to_hex().as_str().to_owned()).collect();
            let mut have: Vec<String> =
                slots.iter().filter_map(|i| file.sections()[*i].key().map(str::to_owned)).collect();
            let in_order = wanted.clone();
            wanted.sort_unstable();
            have.sort_unstable();
            if wanted != have {
                return Err(invalid(kind.section(), "", "the order must name every object of the list once"));
            }
            let mut next = TextFile::new(file.version);
            let mut placed = in_order.iter();
            for (i, section) in file.sections().iter().enumerate() {
                if slots.contains(&i) {
                    let key = placed.next().ok_or_else(|| invalid(kind.section(), "", "order too short"))?;
                    let moved = file
                        .section(kind.section(), Some(key))
                        .ok_or_else(|| invalid(kind.section(), key, "no such object"))?;
                    next.push(moved.clone())?;
                } else {
                    next.push(section.clone())?;
                }
            }
            out.edits.extend(text::diff(&file, &next));
        }
        Action::Stroke { stroke, target, blob, regions, elements } => {
            let (kind, field) = match stroke {
                StrokeKind::Sculpt => ("sculpt_layer", "payload"),
                StrokeKind::Paint => ("skin_layer", "source"),
                StrokeKind::Guides => ("groom", "guides"),
            };
            if let Some(store) = ctx.blobs
                && !blob.is_zero()
                && !store.contains(*blob)
            {
                return Err(CommandError::Invalid(DocError::Blob {
                    hash: *blob,
                    what: "a stroke's blob must be stored before the command",
                }));
            }
            let file = doc.to_text()?;
            let key = target.to_hex();
            let section = file.section(kind, Some(&key)).ok_or_else(|| invalid(kind, &key, "no such object"))?;
            let before = section.get(field).cloned();
            let after = if *stroke == StrokeKind::Paint {
                let painted = before.as_ref().and_then(|s| s.get("kind")).and_then(Value::as_str) == Some("painted");
                if !painted {
                    return Err(invalid(kind, &key, "paint strokes go to painted skin layers"));
                }
                let mut source = Object::with_capacity(2);
                source.insert("kind", "painted");
                source.insert("texture", hash_json(*blob));
                Value::Object(source)
            } else {
                hash_json(*blob)
            };
            if before.as_ref() != Some(&after) {
                out.edits.push(Edit::Set {
                    section: SectionRef::new(kind, Some(&key)),
                    key: field.to_owned(),
                    before,
                    after: Some(after),
                });
            }
            out.regions.extend(regions.iter().cloned());
            out.elements = out.elements.saturating_add(*elements);
        }
        Action::SetLock { target, locked } => {
            let file = doc.to_text()?;
            let existing: Vec<usize> = doc
                .locks
                .iter()
                .filter(|l| l.target == *target)
                .filter_map(|l| file.find("lock", Some(&l.id.to_hex())))
                .collect();
            if *locked {
                if existing.is_empty() {
                    let (kind, name) = match target {
                        LockTarget::Region(r) => ("region", r),
                        LockTarget::Parameter(p) => {
                            ctx.registry.require(p)?;
                            ("parameter", p)
                        }
                    };
                    let mut section = Section::new("lock", Some(&next_id().to_hex()))?;
                    section.set("kind", Value::from(kind))?;
                    section.set("owner", Value::from(cmd.author.as_str()))?;
                    section.set("target", Value::from(name.as_str()))?;
                    let mut next = file.clone();
                    let at = after_last_of(&next, "lock");
                    next.insert(at, section)?;
                    out.edits.extend(via_text(ctx.registry, &file, &next)?);
                }
            } else {
                let mut indices = existing;
                indices.sort_unstable_by(|a, b| b.cmp(a));
                for index in indices {
                    out.edits.push(Edit::Remove { index, section: file.sections()[index].clone() });
                }
            }
        }
        Action::SetObjectLock { kind, id, locked } => {
            if !kind.lockable() {
                return Err(invalid(kind.section(), "", "only layers and grooms have locks"));
            }
            let file = doc.to_text()?;
            let key = id.to_hex();
            let section = file
                .section(kind.section(), Some(&key))
                .ok_or_else(|| invalid(kind.section(), &key, "no such object"))?;
            let before = section.get("locked").cloned();
            let after = Value::from(*locked);
            if before.as_ref() != Some(&after) {
                out.edits.push(Edit::Set {
                    section: SectionRef::new(kind.section(), Some(&key)),
                    key: "locked".to_owned(),
                    before,
                    after: Some(after),
                });
            }
        }
        Action::PinLandmark { landmark, position } => {
            if let Some(pin) = doc.pins.iter().find(|p| p.landmark == *landmark) {
                if pin.position.iter().zip(position).any(|(a, b)| a.to_bits() != b.to_bits()) {
                    out.edits.push(Edit::Set {
                        section: SectionRef::new("pin", Some(&pin.id.to_hex())),
                        key: "position".to_owned(),
                        before: Some(floats(&pin.position)),
                        after: Some(floats(position)),
                    });
                }
            } else {
                let file = doc.to_text()?;
                let mut section = Section::new("pin", Some(&next_id().to_hex()))?;
                section.set("landmark", Value::from(landmark.as_str()))?;
                section.set("owner", Value::from(cmd.author.as_str()))?;
                section.set("position", floats(position))?;
                let mut next = file.clone();
                let at = after_last_of(&next, "pin");
                next.insert(at, section)?;
                out.edits.extend(via_text(ctx.registry, &file, &next)?);
            }
        }
        Action::UnpinLandmark { landmark } => {
            let file = doc.to_text()?;
            let pin = doc
                .pins
                .iter()
                .find(|p| p.landmark == *landmark)
                .ok_or_else(|| invalid("pin", landmark, "the landmark is not pinned"))?;
            let index = file.find("pin", Some(&pin.id.to_hex())).ok_or_else(|| invalid("pin", landmark, "missing"))?;
            out.edits.push(Edit::Remove { index, section: file.sections()[index].clone() });
        }
        Action::CreateSnapshot { id, name, state } => {
            let id = if id.is_nil() { next_id() } else { *id };
            let state = if state.is_zero() {
                let store = ctx.blobs.ok_or_else(|| invalid("snapshot", "", "taking a snapshot needs a blob store"))?;
                store.put(doc.snapshot_text()?.as_bytes(), Compression::Auto).map_err(CommandError::Storage)?
            } else {
                if let Some(store) = ctx.blobs
                    && !store.contains(*state)
                {
                    return Err(CommandError::Invalid(DocError::Blob {
                        hash: *state,
                        what: "a snapshot's state must be stored before the command",
                    }));
                }
                *state
            };
            let file = doc.to_text()?;
            let mut section = Section::new("snapshot", Some(&id.to_hex()))?;
            section.set("author", Value::from(cmd.author.as_str()))?;
            section.set("created", Value::from(Number::from(cmd.time_ms)))?;
            section.set("name", Value::from(name.as_str()))?;
            section.set("state", hash_json(state))?;
            let mut next = file.clone();
            let at = after_last_of(&next, "snapshot");
            next.insert(at, section)?;
            out.edits.extend(via_text(ctx.registry, &file, &next)?);
        }
        Action::RestoreSnapshot { id } => {
            let snapshot = doc
                .snapshots
                .iter()
                .find(|s| s.id == *id)
                .ok_or_else(|| invalid("snapshot", &id.to_hex(), "no such snapshot"))?;
            let store = ctx.blobs.ok_or_else(|| invalid("snapshot", "", "restoring a snapshot needs a blob store"))?;
            let bytes = store.get(snapshot.state).map_err(CommandError::Storage)?;
            let mut target = Document::parse(&bytes, ctx.registry, &TextLimits::DEFAULT)?;
            target.meta = doc.meta.clone();
            target.settings = doc.settings;
            for author in &doc.authors {
                target.add_author(author.clone());
            }
            target.snapshots.clone_from(&doc.snapshots);
            target.check(ctx.registry)?;
            out.edits.extend(doc.diff(&target)?);
        }
        Action::AddAuthor { author } => {
            if doc.author(&author.id) != Some(author) {
                let mut next = doc.clone();
                next.add_author(author.clone());
                next.check(ctx.registry)?;
                out.edits.extend(doc.diff(&next)?);
            }
        }
        Action::SetSettings { protect_human_edits } => {
            if doc.settings.protect_human_edits != *protect_human_edits {
                out.edits.push(Edit::Set {
                    section: SectionRef::new("settings", None),
                    key: "protect_human_edits".to_owned(),
                    before: Some(Value::from(doc.settings.protect_human_edits)),
                    after: Some(Value::from(*protect_human_edits)),
                });
            }
        }
        Action::Batch { actions } => {
            if depth >= MAX_BATCH_DEPTH {
                return Err(invalid("batch", "", "batches nested too deeply"));
            }
            let mut scratch = doc.clone();
            for action in actions {
                let mut part = Resolved::default();
                resolve_action(&scratch, cmd, action, ctx, depth + 1, ids, &mut part)?;
                super::apply_edits(&mut scratch, ctx.registry, &part.edits)?;
                out.edits.extend(part.edits);
                out.warnings.extend(part.warnings);
                out.regions.extend(part.regions);
                out.elements = out.elements.saturating_add(part.elements);
            }
        }
        Action::Edits { edits } => out.edits.extend(edits.iter().cloned()),
    }
    Ok(())
}

/// The index after the last section of `kind`, or the end of the file.
fn after_last_of(file: &TextFile, kind: &str) -> usize {
    file.sections().iter().rposition(|s| s.kind() == kind).map_or(file.sections().len(), |i| i + 1)
}

/// The canonical edits from `file` to `next`: `next` is read as a document (every check a file
/// gets) and written back.
fn via_text(registry: &Registry, file: &TextFile, next: &TextFile) -> Result<Vec<Edit>, CommandError> {
    let parsed = Document::from_text(next, registry)?;
    Ok(text::diff(file, &parsed.to_text()?))
}

fn defaults(kind: ObjectKind, cmd: &Command) -> Vec<(&'static str, Value)> {
    match kind {
        ObjectKind::SculptLayer => vec![
            ("blend", Value::from("additive")),
            ("created_by_command", Value::from(cmd.id.to_hex().as_str())),
            ("locked", Value::from(false)),
            ("opacity", float(1.0)),
            ("payload", Value::Null),
            ("region", Value::Null),
            ("visible", Value::from(true)),
        ],
        ObjectKind::SkinLayer => vec![
            ("blend", Value::from("mix")),
            ("locked", Value::from(false)),
            ("opacity", float(1.0)),
            ("region", Value::Null),
            ("visible", Value::from(true)),
        ],
        ObjectKind::Groom => vec![
            ("guides", Value::Null),
            ("locked", Value::from(false)),
            ("maps", Value::Object(Object::new())),
            ("params", Value::Object(Object::new())),
            ("seed", Value::from(Number::from(0_u64))),
            ("visible", Value::from(true)),
        ],
        ObjectKind::ExpressionPreset => vec![("weights", Value::Object(Object::new()))],
        ObjectKind::PosePreset => vec![("joints", Value::Object(Object::new()))],
        ObjectKind::Attachment => vec![
            ("binding", Value::from("rigid")),
            ("materials", Value::Null),
            ("rotation", floats(&[0.0, 0.0, 0.0, 1.0])),
            ("scale", floats(&[1.0, 1.0, 1.0])),
            ("translation", floats(&[0.0, 0.0, 0.0])),
            ("visible", Value::from(true)),
        ],
        ObjectKind::Reference => vec![("hash", Value::Null)],
    }
}

fn add_object(
    doc: &Document,
    cmd: &Command,
    ctx: &Context<'_>,
    kind: ObjectKind,
    id: Id128,
    index: Option<usize>,
    fields: &Object,
    out: &mut Resolved,
) -> Result<(), CommandError> {
    let file = doc.to_text()?;
    let key = id.to_hex();
    if file.find(kind.section(), Some(&key)).is_some() {
        return Err(invalid(kind.section(), &key, "an object with this id exists"));
    }
    let mut section = Section::new(kind.section(), Some(&key))?;
    for (field, value) in defaults(kind, cmd) {
        section.set(field, value)?;
    }
    for (field, value) in fields.entries() {
        if matches!(field.as_str(), "author" | "created_by_command") {
            return Err(invalid(kind.section(), field, "set from the command"));
        }
        section.set(field, value.clone())?;
    }
    if matches!(kind, ObjectKind::SculptLayer | ObjectKind::SkinLayer | ObjectKind::Groom | ObjectKind::Attachment) {
        section.set("author", Value::from(cmd.author.as_str()))?;
    }
    let same: Vec<usize> =
        (0..file.sections().len()).filter(|i| file.sections()[*i].kind() == kind.section()).collect();
    let at = match index {
        Some(i) if i < same.len() => same[i],
        _ => same.last().map_or(file.sections().len(), |last| last + 1),
    };
    let mut next = file.clone();
    next.insert(at, section)?;
    out.edits.extend(via_text(ctx.registry, &file, &next)?);
    Ok(())
}

fn set_params(
    doc: &Document,
    registry: &Registry,
    author: &str,
    values: &[(String, ParamValue)],
    relative: bool,
    policy: RangePolicy,
    out: &mut Resolved,
) -> Result<(), CommandError> {
    if values.len() > 1 {
        let mut ids: Vec<&str> = values.iter().map(|(id, _)| id.as_str()).collect();
        ids.sort_unstable();
        if let Some(pair) = ids.windows(2).find(|w| w[0] == w[1]) {
            return Err(CommandError::Invalid(DocError::param_value(pair[0], "named twice in one command")));
        }
    }
    for (id, value) in values {
        let param = registry.require(id)?;
        if param.read_only {
            return Err(CommandError::Invalid(DocError::param_value(
                id.as_str(),
                "read-only measurement: set it through the landmark solver",
            )));
        }
        let current = doc.params.get(id).unwrap_or(&param.default);
        let mut v = if relative { add(param, current, value)? } else { value.clone() };
        if let Err(error) = param.check_value(&v) {
            match policy {
                RangePolicy::Reject => {
                    return Err(Rejection::OutOfRange { param: id.clone(), what: error.to_string() }.into());
                }
                RangePolicy::Clamp => {
                    let clamped = param.clamp_value(&v);
                    out.warnings.push(Warning {
                        code: WarningCode::Clamped,
                        target: id.clone(),
                        message: format!("{error}; clamped to {}", compact(&clamped)),
                    });
                    v = clamped;
                }
            }
        }
        implausible(param, &v, out);
        param_edits(doc, id, &v, param.is_default(&v), author, &mut out.edits);
    }
    Ok(())
}

fn compact(value: &ParamValue) -> String {
    let mut s = String::new();
    if humanart_io::json::write(&value_to_json(value), humanart_io::json::Style::COMPACT, &mut s).is_err() {
        s.clear();
    }
    s
}

fn add(param: &Param, current: &ParamValue, delta: &ParamValue) -> Result<ParamValue, CommandError> {
    Ok(match (current, delta) {
        (ParamValue::Scalar(a), ParamValue::Scalar(b)) => ParamValue::Scalar(a + b),
        (ParamValue::Vector(a), ParamValue::Vector(b)) if a.len() == b.len() => {
            ParamValue::Vector(a.iter().zip(b).map(|(x, y)| x + y).collect())
        }
        (ParamValue::Color(a), ParamValue::Color(b)) => ParamValue::Color([a[0] + b[0], a[1] + b[1], a[2] + b[2]]),
        _ => {
            return Err(CommandError::Invalid(DocError::param_value(
                param.id.as_str(),
                "relative values need a scalar, vector or colour of the parameter's shape",
            )));
        }
    })
}

fn implausible(param: &Param, value: &ParamValue, out: &mut Resolved) {
    let Some(r) = param.range else { return };
    let outside = |x: f64| x < r.soft_min || x > r.soft_max;
    let any = match value {
        ParamValue::Scalar(x) => outside(*x),
        ParamValue::Vector(vs) => vs.iter().any(|x| outside(*x)),
        ParamValue::Color(c) => c.iter().any(|x| outside(*x)),
        ParamValue::Curve(points) => points.iter().any(|[_, y]| outside(*y)),
        _ => false,
    };
    if any {
        out.warnings.push(Warning {
            code: WarningCode::Implausible,
            target: param.id.clone(),
            message: format!("outside the plausible span {}..{} (allowed)", r.soft_min, r.soft_max),
        });
    }
}

/// The edits that set parameter `id` to `value` by `author` (nothing when the stored value does
/// not change).
fn param_edits(doc: &Document, id: &str, value: &ParamValue, is_default: bool, author: &str, edits: &mut Vec<Edit>) {
    let before = doc.params.get(id).map(value_to_json);
    let after = (!is_default).then(|| value_to_json(value));
    if before == after {
        return;
    }
    let author_before = doc.param_authors.get(id).map(|a| Value::from(a.as_str()));
    let author_after = after.is_some().then(|| Value::from(author));
    edits.push(Edit::Set { section: SectionRef::new("params", None), key: id.to_owned(), before, after });
    if author_before != author_after {
        edits.push(Edit::Set {
            section: SectionRef::new("param_authors", None),
            key: id.to_owned(),
            before: author_before,
            after: author_after,
        });
    }
}

fn set_pose(doc: &Document, joints: &[(String, Option<[f64; 4]>)], out: &mut Resolved) -> Result<(), CommandError> {
    if joints.len() > 1 {
        let mut names: Vec<&str> = joints.iter().map(|(j, _)| j.as_str()).collect();
        names.sort_unstable();
        if let Some(pair) = names.windows(2).find(|w| w[0] == w[1]) {
            return Err(invalid("pose", pair[0], "named twice in one command"));
        }
    }
    for (joint, q) in joints {
        let before = doc.pose.get(joint).map(|q| floats(q));
        let after = q.map(|q| floats(&q));
        if before != after {
            out.edits.push(Edit::Set { section: SectionRef::new("pose", None), key: joint.clone(), before, after });
        }
    }
    Ok(())
}
