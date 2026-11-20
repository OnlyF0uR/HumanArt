//! Locks, pins and human-edit protection `[LOCK-01]`..`[LOCK-03]`, as the validation step of every
//! command `[CMD-05]`.
//!
//! [`check`] looks at the edits a command resolved to, so every command kind (and raw edits) gets
//! the same rules:
//!
//! - **Parameter locks** refuse any change of the locked parameter.
//! - **Region locks** refuse changes of parameters acting on the region, of sculpt and skin layers
//!   whose region mask is the region (before or after the change), and strokes that declare they
//!   touched it. Whole-body layers have no region mask: the stroke tool declares the regions it
//!   touched ([`crate::command::Action::Stroke`]).
//! - **Layer locks** refuse any change of a locked layer or groom except unlocking it.
//! - **Human-edit protection** (`[LOCK-02]`, on unless the project's `protect_human_edits` setting
//!   is off): an agent may not change a parameter value or a layer or groom a human changed last,
//!   may not remove a human's lock or pin, and may not change project settings or human authors.
//!   The refusal names the protection and the way out (ask the human, request an unlock, or make a
//!   proposal). Inside a proposal, protected targets only raise a [`WarningCode::NeedsReview`]
//!   warning, since nothing reaches the document until the human accepts; locks still apply.
//! - **Landmark pins** `[LOCK-03]` are stored with an owner; keeping the pinned positions is the
//!   landmark solver's job (section 6.3.4), which reads them from [`Document::pins`]. Here pins
//!   are protected like locks.

use humanart_core::id::Id128;
use humanart_io::json::Value;

use crate::command::{Rejection, Warning, WarningCode};
use crate::document::{Author, AuthorKind, Document, LockTarget};
use crate::registry::Registry;
use crate::text::Edit;

/// Where the edits go, which decides how human-edit protection applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The main document: protection refuses agent changes.
    Main,
    /// A proposal: protection only warns.
    Proposal,
    /// A human accepting a proposal: protection does not apply (the human decides), locks do.
    Accept,
}

/// Who changed an object last, for protection: the session knows it from the history. Called
/// with the section kind and key; `None` means "unknown", and the object's creator is used.
pub trait LastModifier {
    /// The author id of the last change of the object, if known.
    fn last_modifier(&self, kind: &str, key: &str) -> Option<String>;
}

/// No history: the creator counts as the last modifier.
pub struct CreatorOnly;

impl LastModifier for CreatorOnly {
    fn last_modifier(&self, _kind: &str, _key: &str) -> Option<String> {
        None
    }
}

struct ObjectState<'a> {
    locked: bool,
    region: Option<&'a str>,
    creator: &'a str,
}

fn object_state<'a>(doc: &'a Document, kind: &str, key: &str) -> Option<ObjectState<'a>> {
    let id = Id128::parse(key).ok()?;
    match kind {
        "sculpt_layer" => doc.sculpt_layers.iter().find(|l| l.id == id).map(|l| ObjectState {
            locked: l.locked,
            region: l.region.as_deref(),
            creator: &l.author,
        }),
        "skin_layer" => doc.skin_layers.iter().find(|l| l.id == id).map(|l| ObjectState {
            locked: l.locked,
            region: l.region.as_deref(),
            creator: &l.author,
        }),
        "groom" => doc.grooms.iter().find(|g| g.id == id).map(|g| ObjectState {
            locked: g.locked,
            region: None,
            creator: &g.author,
        }),
        _ => None,
    }
}

fn region_lock<'a>(doc: &'a Document, region: &str) -> Option<&'a str> {
    doc.locks.iter().find(|l| matches!(&l.target, LockTarget::Region(r) if r == region)).map(|l| l.owner.as_str())
}

/// The locks of parameter `id` (its own lock, its region's lock), then, when `protect` is on, the
/// human who changed it last (`Some`: the change is protected). Allocates only to refuse.
fn param_rule<'d>(
    doc: &'d Document,
    registry: &Registry,
    id: &str,
    protect: bool,
) -> Result<Option<&'d str>, Rejection> {
    if let Some(lock) = doc.locks.iter().find(|l| matches!(&l.target, LockTarget::Parameter(p) if p == id)) {
        return Err(Rejection::ParameterLocked { param: id.to_owned(), owner: lock.owner.clone() });
    }
    if let Some(region) = registry.get(id).and_then(|p| p.region.as_deref())
        && let Some(owner) = region_lock(doc, region)
    {
        return Err(Rejection::RegionLocked { region: region.to_owned(), owner: owner.to_owned() });
    }
    if protect
        && let Some(by) = doc.param_authors.get(id)
        && doc.author(by).is_some_and(|a| a.kind == AuthorKind::Human)
    {
        return Ok(Some(by.as_str()));
    }
    Ok(None)
}

/// [`check`] for a change of the value of parameter `id` by `author` on the main document
/// ([`Mode::Main`]): the same rules as for a `params` edit, without allocating unless it refuses
/// (the frames of a slider drag `[PERF-R1]`).
pub fn check_param(doc: &Document, registry: &Registry, id: &str, author: &Author) -> Result<(), Rejection> {
    let protect = author.kind == AuthorKind::Agent && doc.settings.protect_human_edits;
    match param_rule(doc, registry, id, protect)? {
        Some(by) => Err(Rejection::HumanProtected { target: id.to_owned(), human: by.to_owned() }),
        None => Ok(()),
    }
}

/// Checks `edits` by `author` against the document's locks and protection. Returns the warnings
/// to attach to the result, or why the command is refused.
pub fn check(
    doc: &Document,
    registry: &Registry,
    edits: &[Edit],
    author: &Author,
    regions: &[String],
    history: &dyn LastModifier,
    mode: Mode,
) -> Result<Vec<Warning>, Rejection> {
    let agent = author.kind == AuthorKind::Agent;
    let protect = agent && doc.settings.protect_human_edits && mode != Mode::Accept;
    let human = |id: &str| doc.author(id).is_some_and(|a| a.kind == AuthorKind::Human);
    let mut warnings: Vec<Warning> = Vec::new();
    let mut protected = |target: String, by: &str| -> Result<(), Rejection> {
        if mode == Mode::Proposal {
            if !warnings.iter().any(|w| w.target == target) {
                warnings.push(Warning {
                    code: WarningCode::NeedsReview,
                    message: format!(
                        "{target} was last changed by the human {by}; accepting this change is their decision"
                    ),
                    target,
                });
            }
            Ok(())
        } else {
            Err(Rejection::HumanProtected { target, human: by.to_owned() })
        }
    };
    for region in regions {
        if let Some(owner) = region_lock(doc, region) {
            return Err(Rejection::RegionLocked { region: region.clone(), owner: owner.to_owned() });
        }
    }
    for edit in edits {
        let (kind, section_key, entry) = match edit {
            Edit::Set { section, key, .. } => (section.kind.as_str(), section.key.as_deref(), Some(key.as_str())),
            Edit::Insert { section, .. } | Edit::Remove { section, .. } => (section.kind(), section.key(), None),
        };
        match kind {
            "settings" if agent => return Err(Rejection::HumanOnly { what: "change project policies" }),
            "params" | "param_authors" => {
                let Some(id) = entry else { continue };
                if let Some(by) = param_rule(doc, registry, id, protect)? {
                    protected(id.to_owned(), by)?;
                }
            }
            "sculpt_layer" | "skin_layer" | "groom" => {
                let key = section_key.unwrap_or_default();
                let state = object_state(doc, kind, key);
                if let Some(state) = &state {
                    let unlocking_only = entry == Some("locked");
                    if state.locked && !unlocking_only {
                        return Err(Rejection::ObjectLocked { kind: kind.to_owned(), id: key.to_owned() });
                    }
                }
                let mut touched_regions: Vec<&str> = Vec::new();
                if let Some(region) = state.as_ref().and_then(|s| s.region) {
                    touched_regions.push(region);
                }
                match edit {
                    Edit::Set { key, after: Some(Value::String(region)), .. } if key == "region" => {
                        touched_regions.push(region);
                    }
                    Edit::Insert { section, .. } => {
                        if let Some(Value::String(region)) = section.get("region") {
                            touched_regions.push(region);
                        }
                    }
                    _ => {}
                }
                if kind != "groom" {
                    for region in touched_regions {
                        if let Some(owner) = region_lock(doc, region) {
                            return Err(Rejection::RegionLocked { region: region.to_owned(), owner: owner.to_owned() });
                        }
                    }
                }
                if protect && let Some(state) = &state {
                    let last = history.last_modifier(kind, key).unwrap_or_else(|| state.creator.to_owned());
                    if human(&last) {
                        protected(format!("{kind} {key}"), &last)?;
                    }
                }
            }
            "lock" if agent && !matches!(edit, Edit::Insert { .. }) => {
                let key = section_key.unwrap_or_default();
                if let Some(lock) = doc.locks.iter().find(|l| l.id.to_hex().as_str() == key)
                    && human(&lock.owner)
                {
                    return Err(Rejection::HumanProtected { target: format!("lock {key}"), human: lock.owner.clone() });
                }
            }
            "pin" if agent && !matches!(edit, Edit::Insert { .. }) => {
                let key = section_key.unwrap_or_default();
                if let Some(pin) = doc.pins.iter().find(|p| p.id.to_hex().as_str() == key)
                    && human(&pin.owner)
                {
                    return Err(Rejection::HumanProtected {
                        target: format!("pin {}", pin.landmark),
                        human: pin.owner.clone(),
                    });
                }
            }
            "author" if agent => {
                let changes_existing = !matches!(edit, Edit::Insert { .. });
                let adds_human =
                    matches!(edit, Edit::Insert { section, .. } if section.get("kind") == Some(&Value::from("human")));
                if changes_existing || adds_human {
                    return Err(Rejection::HumanOnly {
                        what: "add or change human authors, or change existing authors",
                    });
                }
            }
            _ => {}
        }
    }
    Ok(warnings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::{Action, Command, Context, ObjectKind, RangePolicy, apply_edits, resolve};
    use crate::document::tests::sample;
    use crate::registry::ParamValue;
    use crate::registry::tests::real_registry;
    use humanart_io::json::Object;

    fn agent(doc: &Document) -> Author {
        doc.author("agent.assistant").cloned().unwrap()
    }

    fn human(doc: &Document) -> Author {
        doc.author("human.artist").cloned().unwrap()
    }

    fn edits(doc: &Document, reg: &Registry, author: &str, action: Action) -> Vec<Edit> {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(77);
        let mut cmd = Command::new(author, action);
        cmd.id = Id128::from_u128(u128::from(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
        resolve(doc, &cmd, &Context { registry: reg, blobs: None }).unwrap().edits
    }

    fn set(id: &str, v: f64) -> Action {
        Action::SetParams {
            values: vec![(id.into(), ParamValue::Scalar(v))],
            relative: false,
            policy: RangePolicy::Reject,
        }
    }

    #[test]
    fn lock02_agent_cannot_change_human_values_but_can_propose_them() {
        let reg = real_registry();
        let doc = sample(&reg);
        // face.nose.bridge_height was set by the human in the sample.
        let e = edits(&doc, &reg, "agent.assistant", set("face.nose.bridge_height", 0.1));
        let err = check(&doc, &reg, &e, &agent(&doc), &[], &CreatorOnly, Mode::Main).unwrap_err();
        assert_eq!(err.code(), "human_protected");
        assert!(err.to_string().contains("ask the human"), "{err}");
        let warnings = check(&doc, &reg, &e, &agent(&doc), &[], &CreatorOnly, Mode::Proposal).unwrap();
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].code, WarningCode::NeedsReview);
        assert!(check(&doc, &reg, &e, &agent(&doc), &[], &CreatorOnly, Mode::Accept).unwrap().is_empty());
        // The human may; values the agent set last are not protected.
        assert!(check(&doc, &reg, &e, &human(&doc), &[], &CreatorOnly, Mode::Main).is_ok());
        let e2 = edits(&doc, &reg, "agent.assistant", set("expression.au12_lip_corner_puller.l", 0.2));
        assert!(check(&doc, &reg, &e2, &agent(&doc), &[], &CreatorOnly, Mode::Main).is_ok());
        // The policy can be switched off by the human only.
        let mut off = doc.clone();
        off.settings.protect_human_edits = false;
        assert!(check(&off, &reg, &e, &agent(&doc), &[], &CreatorOnly, Mode::Main).is_ok());
        let s = edits(&doc, &reg, "agent.assistant", Action::SetSettings { protect_human_edits: false });
        assert_eq!(
            check(&doc, &reg, &s, &agent(&doc), &[], &CreatorOnly, Mode::Main).unwrap_err().code(),
            "human_only"
        );
        assert!(check(&doc, &reg, &s, &human(&doc), &[], &CreatorOnly, Mode::Main).is_ok());
    }

    #[test]
    fn lock01_parameter_region_and_layer_locks_refuse_everyone() {
        let reg = real_registry();
        let mut doc = sample(&reg);
        let lock = |doc: &mut Document, target: LockTarget, author: &str| {
            let e = edits(doc, &reg, author, Action::SetLock { target, locked: true });
            apply_edits(doc, &reg, &e).unwrap();
        };
        let id = "face.nose.bridge_height";
        lock(&mut doc, LockTarget::Parameter(id.into()), "human.artist");
        let e = edits(&doc, &reg, "human.artist", set(id, 0.2));
        assert_eq!(
            check(&doc, &reg, &e, &human(&doc), &[], &CreatorOnly, Mode::Main).unwrap_err().code(),
            "parameter_locked"
        );
        // Region lock on the region of some parameter.
        let p = reg
            .params()
            .iter()
            .find(|p| {
                p.region.as_deref().is_some_and(|r| r != "nasal-tip")
                    && p.kind == crate::registry::ParamKind::Scalar
                    && p.range.is_some()
                    && !p.read_only
            })
            .unwrap();
        let region = p.region.clone().unwrap();
        lock(&mut doc, LockTarget::Region(region.clone()), "agent.assistant");
        let range = p.range.unwrap();
        let value = if p.default == ParamValue::Scalar(range.soft_max) { range.soft_min } else { range.soft_max };
        let e = edits(&doc, &reg, "human.artist", set(&p.id, value));
        let err = check(&doc, &reg, &e, &human(&doc), &[], &CreatorOnly, Mode::Main).unwrap_err();
        assert_eq!(err.code(), "region_locked", "{err}");
        // Strokes that declare the region are refused too.
        assert_eq!(
            check(&doc, &reg, &[], &human(&doc), &[region], &CreatorOnly, Mode::Accept).unwrap_err().code(),
            "region_locked"
        );
        // Layer 11 is locked in the sample: changing it fails, unlocking it works.
        let layer = Id128::from_u128(0x0123_4567_89ab_cdef_0000_0000_0000_0000 + 11);
        let mut fields = Object::new();
        fields.insert("opacity", humanart_io::json::Value::from(humanart_io::json::Number::Float(0.5)));
        let e = edits(
            &doc,
            &reg,
            "human.artist",
            Action::ModifyObject { kind: ObjectKind::SculptLayer, id: layer, fields },
        );
        assert_eq!(
            check(&doc, &reg, &e, &human(&doc), &[], &CreatorOnly, Mode::Main).unwrap_err().code(),
            "object_locked"
        );
        let e = edits(
            &doc,
            &reg,
            "human.artist",
            Action::SetObjectLock { kind: ObjectKind::SculptLayer, id: layer, locked: false },
        );
        assert!(check(&doc, &reg, &e, &human(&doc), &[], &CreatorOnly, Mode::Main).is_ok());
        // An agent may not remove the human's parameter lock, but may remove its own region lock.
        let e = edits(
            &doc,
            &reg,
            "agent.assistant",
            Action::SetLock { target: LockTarget::Parameter(id.into()), locked: false },
        );
        assert_eq!(
            check(&doc, &reg, &e, &agent(&doc), &[], &CreatorOnly, Mode::Main).unwrap_err().code(),
            "human_protected"
        );
        let e = edits(
            &doc,
            &reg,
            "agent.assistant",
            Action::SetLock { target: LockTarget::Region(p.region.clone().unwrap()), locked: false },
        );
        assert!(check(&doc, &reg, &e, &agent(&doc), &[], &CreatorOnly, Mode::Main).is_ok());
    }

    #[test]
    fn lock02_layers_follow_the_last_modifier() {
        struct LastBy(&'static str);
        impl LastModifier for LastBy {
            fn last_modifier(&self, _: &str, _: &str) -> Option<String> {
                Some(self.0.to_owned())
            }
        }
        let reg = real_registry();
        let doc = sample(&reg);
        // Layer 20 (freckles) was created by the agent.
        let layer = Id128::from_u128(0x0123_4567_89ab_cdef_0000_0000_0000_0000 + 20);
        let mut fields = Object::new();
        fields.insert("opacity", humanart_io::json::Value::from(humanart_io::json::Number::Float(0.3)));
        let e = edits(
            &doc,
            &reg,
            "agent.assistant",
            Action::ModifyObject { kind: ObjectKind::SkinLayer, id: layer, fields },
        );
        assert!(check(&doc, &reg, &e, &agent(&doc), &[], &CreatorOnly, Mode::Main).is_ok());
        assert_eq!(
            check(&doc, &reg, &e, &agent(&doc), &[], &LastBy("human.artist"), Mode::Main).unwrap_err().code(),
            "human_protected"
        );
        // Agents cannot make themselves human.
        let a = Author { id: "agent.x".into(), kind: AuthorKind::Human, name: "X".into() };
        let e = edits(&doc, &reg, "agent.assistant", Action::AddAuthor { author: a });
        assert_eq!(
            check(&doc, &reg, &e, &agent(&doc), &[], &CreatorOnly, Mode::Main).unwrap_err().code(),
            "human_only"
        );
    }
}
