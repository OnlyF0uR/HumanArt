//! Applying edits to a document, with a fast path for slider drags.
//!
//! [`Document::apply_edits`] re-encodes and re-reads the whole document so that every edit is
//! checked exactly as a file would be. Edits that only set entries of `[params]`,
//! `[param_authors]` and `[pose]` (what slider drags and the agent's parameter tools produce) are
//! applied to the typed maps directly instead, with the same checks: the expected `before` value,
//! the parameter's kind and range, canonical form (no stored defaults, numbers written as the
//! writer writes them), known authors, unit quaternions and joint names. A property test checks
//! that both paths accept and refuse the same edits and give the same document.

use humanart_io::json::{Number, Value};

use crate::document::Document;
use crate::error::{DocError, Result};
use crate::registry::{Registry, value_from_json, value_to_json};
use crate::text::Edit;

/// Applies `edits` to `doc` atomically: on error the document is unchanged.
pub fn apply_edits(doc: &mut Document, registry: &Registry, edits: &[Edit]) -> Result<()> {
    if edits.iter().all(is_fast) { apply_fast(doc, registry, edits) } else { doc.apply_edits(edits, registry) }
}

fn is_fast(edit: &Edit) -> bool {
    matches!(edit, Edit::Set { section, .. }
        if section.key.is_none() && matches!(section.kind.as_str(), "params" | "param_authors" | "pose"))
}

fn apply_fast(doc: &mut Document, registry: &Registry, edits: &[Edit]) -> Result<()> {
    for (done, edit) in edits.iter().enumerate() {
        if let Err(error) = apply_one(doc, registry, edit) {
            roll_back(doc, registry, &edits[..done]);
            return Err(error);
        }
    }
    // A stored author needs a stored value (the codec refuses orphan author entries).
    for edit in edits {
        if let Edit::Set { key, .. } = edit
            && doc.param_authors.contains_key(key)
            && !doc.params.contains_key(key)
        {
            roll_back(doc, registry, edits);
            return Err(DocError::field("param_authors", key.as_str(), "no stored value for this parameter"));
        }
    }
    Ok(())
}

fn roll_back(doc: &mut Document, registry: &Registry, applied: &[Edit]) {
    for edit in applied.iter().rev() {
        // The inverse of an edit that has just been applied finds exactly the value the edit left
        // and restores a value the valid document held, so it cannot fail; if it did, the
        // remaining edits are still undone as far as possible.
        let _ = apply_one(doc, registry, &edit.inverse());
    }
}

fn conflict(section: &str, key: &str, what: &'static str) -> DocError {
    DocError::Conflict { section: section.to_owned(), key: key.to_owned(), what }
}

const STALE: &str = "the entry does not hold the expected value";
const NO_OP: &str = "an edit that neither sets nor removes";

fn apply_one(doc: &mut Document, registry: &Registry, edit: &Edit) -> Result<()> {
    let Edit::Set { section, key, before, after } = edit else {
        return Err(conflict("", "", "not a fast edit"));
    };
    match section.kind.as_str() {
        "params" => {
            let param = registry.require(key)?;
            let matches = match (doc.params.get(key), before) {
                (None, None) => true,
                (Some(current), Some(expected)) => value_to_json(current) == *expected,
                _ => false,
            };
            if !matches {
                return Err(conflict("params", key, STALE));
            }
            if let Some(json) = after {
                if param.read_only {
                    return Err(DocError::param_value(key.as_str(), "read-only measurements are not stored"));
                }
                let value = value_from_json(param, json)?;
                if param.is_default(&value) || value_to_json(&value) != *json {
                    return Err(DocError::field(
                        "params",
                        key.as_str(),
                        "the edit leaves a non-canonical value (a default or a differently written number)",
                    ));
                }
                if let Some(slot) = doc.params.get_mut(key) {
                    *slot = value;
                } else {
                    doc.params.insert(key.clone(), value);
                }
            } else {
                if before.is_none() {
                    return Err(conflict("params", key, NO_OP));
                }
                doc.params.remove(key);
            }
        }
        "param_authors" => {
            let matches = match (doc.param_authors.get(key), before) {
                (None, None) => true,
                (Some(current), Some(Value::String(expected))) => current == expected,
                _ => false,
            };
            if !matches {
                return Err(conflict("param_authors", key, STALE));
            }
            match after {
                Some(Value::String(author)) => {
                    if doc.author(author).is_none() {
                        return Err(DocError::field("param_authors", key.as_str(), format!("unknown author {author}")));
                    }
                    if let Some(slot) = doc.param_authors.get_mut(key) {
                        slot.clone_from(author);
                    } else {
                        doc.param_authors.insert(key.clone(), author.clone());
                    }
                }
                Some(_) => return Err(DocError::field("param_authors", key.as_str(), "expected a string")),
                None => {
                    if before.is_none() {
                        return Err(conflict("param_authors", key, NO_OP));
                    }
                    doc.param_authors.remove(key);
                }
            }
        }
        "pose" => {
            let matches = match (doc.pose.get(key), before) {
                (None, None) => true,
                (Some(current), Some(expected)) => quat_from(expected).is_some_and(|q| same_bits(&q, current)),
                _ => false,
            };
            if !matches {
                return Err(conflict("pose", key, STALE));
            }
            if let Some(json) = after {
                let q = quat_from(json)
                    .filter(is_unit_quat)
                    .ok_or_else(|| DocError::field("pose", key.as_str(), "needs a unit quaternion"))?;
                if humanart_model::names::check_name("document", key).is_err() {
                    return Err(DocError::field("pose", key.as_str(), "needs a joint name"));
                }
                doc.pose.insert(key.clone(), q);
            } else {
                if before.is_none() {
                    return Err(conflict("pose", key, NO_OP));
                }
                doc.pose.remove(key);
            }
        }
        _ => return Err(conflict(&section.kind, key, "not a fast edit")),
    }
    Ok(())
}

/// Four floats written as floats (the canonical form of a quaternion in the text).
fn quat_from(value: &Value) -> Option<[f64; 4]> {
    let items = value.as_array()?;
    if items.len() != 4 {
        return None;
    }
    let mut q = [0.0; 4];
    for (out, item) in q.iter_mut().zip(items) {
        match item {
            Value::Number(Number::Float(v)) => *out = *v,
            _ => return None,
        }
    }
    Some(q)
}

fn same_bits(a: &[f64; 4], b: &[f64; 4]) -> bool {
    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn is_unit_quat(q: &[f64; 4]) -> bool {
    let n = q.iter().map(|v| v * v).sum::<f64>();
    q.iter().all(|v| v.is_finite()) && (n - 1.0).abs() <= 1e-6
}
