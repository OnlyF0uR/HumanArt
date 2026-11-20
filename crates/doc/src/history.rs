//! The shared linear history with per-author awareness `[CMD-04]` and gesture coalescing
//! `[CMD-03]`.
//!
//! The history is a list of [`Entry`]s and a cursor: entries before the cursor are applied, the
//! ones after it can be redone, and a new entry drops them. Plain undo steps the cursor back over
//! the most recent entry whoever made it. "Undo my last" / "undo the agent's last" picks that
//! author's most recent applied entry; when it is not the top one, it is reverted only if no later
//! applied entry touches the same targets ([`crate::command::Target`]), and the revert is recorded
//! as a new entry (`kind` `"undo"`, [`Entry::reverts`]) so the history stays linear and the revert
//! itself can be undone.
//!
//! A continuous gesture (slider drag, brush stroke) keeps the top entry *open* with its
//! coalescing key (author, command kind, target): further commands with the same key fold their
//! edits into it ([`compose`]), so the gesture is one undo step. The session seals the entry on
//! release, after a pause longer than its threshold, and before any other operation.

use humanart_core::id::Id128;
use humanart_io::json::{Number, Object, Value};

use crate::command::codec::{CommandLimits, edits_from_json, edits_to_json};
use crate::command::{Rejection, Target, any_overlap, targets};
use crate::error::{DocError, Result};
use crate::lock::LastModifier;
use crate::text::{Edit, invert};

/// One history entry: an applied command.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    /// The command's id.
    pub id: Id128,
    /// Its author.
    pub author: String,
    /// The command kind.
    pub kind: String,
    /// The label.
    pub label: String,
    /// When it was made, Unix milliseconds.
    pub time_ms: u64,
    /// When it was last extended by its gesture.
    pub last_ms: u64,
    /// The forward edits; the inverse is [`Entry::inverse`] `[CMD-01]`.
    pub edits: Vec<Edit>,
    /// For a selective undo: the entry it reverts.
    pub reverts: Option<Id128>,
    /// The coalescing key while the entry is open for its gesture.
    pub open: Option<String>,
    /// Elements strokes touched.
    pub elements: u64,
}

impl Entry {
    /// The edits that undo this entry.
    pub fn inverse(&self) -> Vec<Edit> {
        invert(&self.edits)
    }

    /// What the entry touches.
    pub fn targets(&self) -> Vec<Target> {
        targets(&self.edits)
    }

    /// The JSON form (journal and proposal files).
    pub fn to_json(&self) -> Value {
        let mut o = Object::with_capacity(10);
        o.insert("id", self.id.to_hex().as_str());
        o.insert("author", self.author.as_str());
        o.insert("kind", self.kind.as_str());
        o.insert("label", self.label.as_str());
        o.insert("time", Value::from(Number::from(self.time_ms)));
        o.insert("last", Value::from(Number::from(self.last_ms)));
        o.insert("edits", edits_to_json(&self.edits));
        o.insert("reverts", self.reverts.map_or(Value::Null, |r| Value::from(r.to_hex().as_str())));
        o.insert("open", self.open.as_deref().map_or(Value::Null, Value::from));
        o.insert("elements", Value::from(Number::from(self.elements)));
        Value::Object(o)
    }

    /// Reads the JSON form.
    pub fn from_json(value: &Value, limits: &CommandLimits) -> Result<Self> {
        let bad = |key: &str, what: &str| DocError::field("history entry", key, what);
        let o = value.as_object().ok_or_else(|| bad("", "expected an object"))?;
        if let Some((k, _)) = o.entries().iter().find(|(k, _)| {
            !matches!(
                k.as_str(),
                "id" | "author" | "kind" | "label" | "time" | "last" | "edits" | "reverts" | "open" | "elements"
            )
        }) {
            return Err(bad(k, "unknown member"));
        }
        let text = |key: &str| -> Result<String> {
            o.get(key).and_then(Value::as_str).map(str::to_owned).ok_or_else(|| bad(key, "expected a string"))
        };
        let number = |key: &str| -> Result<u64> {
            o.get(key).and_then(Value::as_u64).ok_or_else(|| bad(key, "expected an unsigned integer"))
        };
        let id = |key: &str| -> Result<Option<Id128>> {
            match o.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(v) => {
                    v.as_str().and_then(|s| Id128::parse(s).ok()).map(Some).ok_or_else(|| bad(key, "expected an id"))
                }
            }
        };
        let open = match o.get("open") {
            None | Some(Value::Null) => None,
            Some(v) => Some(v.as_str().ok_or_else(|| bad("open", "expected a string"))?.to_owned()),
        };
        Ok(Self {
            id: id("id")?.ok_or_else(|| bad("id", "missing"))?,
            author: text("author")?,
            kind: text("kind")?,
            label: text("label")?,
            time_ms: number("time")?,
            last_ms: number("last")?,
            edits: edits_from_json(o.get("edits").ok_or_else(|| bad("edits", "missing"))?, limits)?,
            reverts: id("reverts")?,
            open,
            elements: number("elements")?,
        })
    }
}

/// Folds `more` into `edits` for coalescing: a `Set` of an entry already set keeps the first
/// `before` and takes the new `after`; sets that end where they started are dropped. Returns
/// `false` (and changes nothing) unless every edit of both lists is a `Set`.
pub fn compose(edits: &mut Vec<Edit>, more: &[Edit]) -> bool {
    let all_sets = |list: &[Edit]| list.iter().all(|e| matches!(e, Edit::Set { .. }));
    if !all_sets(edits) || !all_sets(more) {
        return false;
    }
    for edit in more {
        let Edit::Set { section, key, after, .. } = edit else { continue };
        let existing = edits.iter_mut().find_map(|e| match e {
            Edit::Set { section: s, key: k, after: a, .. } if s == section && k == key => Some(a),
            _ => None,
        });
        match existing {
            Some(slot) => slot.clone_from(after),
            None => edits.push(edit.clone()),
        }
    }
    edits.retain(|e| !matches!(e, Edit::Set { before, after, .. } if before == after));
    true
}

/// The in-place form of [`compose`] for one more frame of a drag `[PERF-R1]`: moves the `after`
/// of the `Set` of entry `key` of the unkeyed section `kind` to `after`, without allocating.
/// Returns `false` (and changes nothing) when `edits` holds no such `Set` or the set would end
/// where it started (then [`compose`] must drop it).
pub fn retarget(edits: &mut [Edit], kind: &str, key: &str, after: &Value) -> bool {
    let found = edits.iter_mut().find_map(|e| match e {
        Edit::Set { section, key: k, before, after: a }
            if section.kind == kind && section.key.is_none() && k == key =>
        {
            Some((before, a))
        }
        _ => None,
    });
    match found {
        Some((before, slot)) if before.as_ref() != Some(after) => {
            match slot {
                Some(value) => value.clone_from(after),
                None => *slot = Some(after.clone()),
            }
            true
        }
        _ => false,
    }
}

/// Which entries an undo may pick `[CMD-04]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Scope {
    /// The most recent entry, whoever made it.
    Any,
    /// The most recent entry of this author ("undo my last", "undo the agent's last").
    Author(String),
}

/// How an undo is carried out.
#[derive(Clone, Debug, PartialEq)]
pub enum UndoPlan {
    /// Step the cursor back over the top entry.
    Top,
    /// Revert the entry at this index with a new entry (selective undo).
    Selective {
        /// The index of the entry to revert.
        index: usize,
    },
}

/// The history.
#[derive(Clone, Debug, PartialEq)]
pub struct History {
    entries: Vec<Entry>,
    cursor: usize,
    limit: usize,
}

impl History {
    /// An empty history keeping at most `limit` entries (at least 1).
    pub fn new(limit: usize) -> Self {
        Self { entries: Vec::new(), cursor: 0, limit: limit.max(1) }
    }

    /// Every entry, oldest first.
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// The number of applied entries; `entries()[cursor..]` can be redone.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// The applied entries, oldest first.
    pub fn applied(&self) -> &[Entry] {
        &self.entries[..self.cursor]
    }

    /// The entry with this id.
    pub fn find(&self, id: Id128) -> Option<&Entry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// The top entry if it is open for a gesture with coalescing key `key`.
    pub fn open_with(&self, key: &str) -> Option<&Entry> {
        if self.cursor != self.entries.len() {
            return None;
        }
        self.entries.last().filter(|e| e.open.as_deref() == Some(key))
    }

    /// The open top entry, if any.
    pub fn open(&self) -> Option<&Entry> {
        if self.cursor != self.entries.len() {
            return None;
        }
        self.entries.last().filter(|e| e.open.is_some())
    }

    /// The open top entry, to fold one more frame of its gesture into in place.
    pub(crate) fn open_mut(&mut self) -> Option<&mut Entry> {
        self.open()?;
        self.entries.last_mut()
    }

    /// Records an applied entry, dropping the redo tail and the oldest entries beyond the limit.
    pub fn push(&mut self, entry: Entry) {
        self.entries.truncate(self.cursor);
        self.entries.push(entry);
        if self.entries.len() > self.limit {
            let excess = self.entries.len() - self.limit;
            self.entries.drain(..excess);
        }
        self.cursor = self.entries.len();
    }

    /// Folds `edits` into the open top entry ([`compose`]). Returns `false` when there is no open
    /// entry or the edits cannot be folded.
    pub fn extend_open(&mut self, edits: &[Edit], now_ms: u64, elements: u64) -> bool {
        if self.open().is_none() {
            return false;
        }
        let Some(top) = self.entries.last_mut() else { return false };
        if !compose(&mut top.edits, edits) {
            return false;
        }
        top.last_ms = now_ms;
        top.elements = top.elements.saturating_add(elements);
        true
    }

    /// Ends the open gesture. Returns the sealed entry, or `None` when nothing was open or the
    /// gesture ended where it started (the empty entry is dropped).
    pub fn seal(&mut self) -> Option<&Entry> {
        self.open()?;
        let top = self.entries.last_mut()?;
        top.open = None;
        if top.edits.is_empty() {
            self.entries.pop();
            self.cursor = self.entries.len();
            return None;
        }
        self.entries.last()
    }

    fn reverted(&self, id: Id128) -> bool {
        self.applied().iter().any(|e| e.reverts == Some(id))
    }

    /// Decides how to undo for `scope`, or why it cannot be done `[CMD-04]`.
    pub fn plan_undo(&self, scope: &Scope) -> std::result::Result<UndoPlan, Rejection> {
        if self.cursor == 0 {
            return Err(Rejection::NothingToDo { what: "undo" });
        }
        let author = match scope {
            Scope::Any => return Ok(UndoPlan::Top),
            Scope::Author(a) => a,
        };
        let index = (0..self.cursor)
            .rev()
            .find(|i| self.entries[*i].author == *author && !self.reverted(self.entries[*i].id))
            .ok_or(Rejection::NothingToDo { what: "undo" })?;
        if index + 1 == self.cursor {
            return Ok(UndoPlan::Top);
        }
        let entry = &self.entries[index];
        let mine = entry.targets();
        let mut later = Vec::new();
        let mut shared: Vec<Target> = Vec::new();
        for e in &self.entries[index + 1..self.cursor] {
            let overlap = any_overlap(&mine, &e.targets());
            if !overlap.is_empty() {
                later.push(e.id);
                shared.extend(overlap);
            }
        }
        if later.is_empty() {
            Ok(UndoPlan::Selective { index })
        } else {
            shared.sort_unstable();
            shared.dedup();
            Err(Rejection::UndoConflict {
                command: entry.id,
                later,
                targets: shared.iter().map(ToString::to_string).collect(),
            })
        }
    }

    /// The entry a redo would apply, checked against `scope`.
    pub fn plan_redo(&self, scope: &Scope) -> std::result::Result<&Entry, Rejection> {
        let entry = self.entries.get(self.cursor).ok_or(Rejection::NothingToDo { what: "redo" })?;
        match scope {
            Scope::Author(a) if *a != entry.author => {
                Err(Rejection::NotYours { command: entry.id, author: entry.author.clone() })
            }
            _ => Ok(entry),
        }
    }

    /// Steps the cursor back over the top entry (after its inverse was applied).
    pub fn step_back(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    /// Steps the cursor forward over the next entry (after it was applied again).
    pub fn step_forward(&mut self) {
        self.cursor = (self.cursor + 1).min(self.entries.len());
    }

    /// The author of the most recent applied entry that touched section `kind` `key` (an object of
    /// a list), for human-edit protection `[LOCK-02]`.
    pub fn last_modifier(&self, kind: &str, key: &str) -> Option<&str> {
        self.applied()
            .iter()
            .rev()
            .find(|e| {
                e.edits.iter().any(|edit| {
                    let t = Target::of(edit);
                    t.section.kind == kind && t.section.key.as_deref() == Some(key)
                })
            })
            .map(|e| e.author.as_str())
    }
}

impl LastModifier for History {
    fn last_modifier(&self, kind: &str, key: &str) -> Option<String> {
        History::last_modifier(self, kind, key).map(str::to_owned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::SectionRef;

    fn set(key: &str, before: Option<i64>, after: Option<i64>) -> Edit {
        Edit::Set {
            section: SectionRef::new("params", None),
            key: key.into(),
            before: before.map(|v| Value::from(Number::Int(v))),
            after: after.map(|v| Value::from(Number::Int(v))),
        }
    }

    fn entry(n: u128, author: &str, edits: Vec<Edit>) -> Entry {
        Entry {
            id: Id128::from_u128(n),
            author: author.into(),
            kind: "set_params".into(),
            label: String::new(),
            time_ms: n as u64,
            last_ms: n as u64,
            edits,
            reverts: None,
            open: None,
            elements: 0,
        }
    }

    #[test]
    fn cmd03_compose_keeps_first_before_and_last_after() {
        let mut edits = vec![set("a", Some(1), Some(2))];
        assert!(compose(&mut edits, &[set("a", Some(2), Some(3)), set("b", None, Some(1))]));
        assert_eq!(edits, vec![set("a", Some(1), Some(3)), set("b", None, Some(1))]);
        // Back where it started: dropped.
        assert!(compose(&mut edits, &[set("a", Some(3), Some(1))]));
        assert_eq!(edits, vec![set("b", None, Some(1))]);
        let insert = Edit::Remove { index: 0, section: crate::text::Section::new("pin", Some("x")).unwrap() };
        assert!(!compose(&mut edits, &[insert]));
    }

    #[test]
    fn cmd04_selective_undo_checks_later_commands_on_the_same_targets() {
        let mut h = History::new(100);
        h.push(entry(1, "agent", vec![set("a", None, Some(1))]));
        h.push(entry(2, "human", vec![set("b", None, Some(1))]));
        assert_eq!(h.plan_undo(&Scope::Any), Ok(UndoPlan::Top));
        assert_eq!(h.plan_undo(&Scope::Author("agent".into())), Ok(UndoPlan::Selective { index: 0 }));
        h.push(entry(3, "human", vec![set("a", Some(1), Some(2))]));
        match h.plan_undo(&Scope::Author("agent".into())) {
            Err(Rejection::UndoConflict { command, later, targets }) => {
                assert_eq!(command, Id128::from_u128(1));
                assert_eq!(later, vec![Id128::from_u128(3)]);
                assert_eq!(targets, vec!["params a".to_owned()]);
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(h.plan_undo(&Scope::Author("nobody".into())), Err(Rejection::NothingToDo { what: "undo" }));
        h.step_back();
        assert_eq!(h.plan_redo(&Scope::Author("agent".into())).unwrap_err().code(), "not_yours");
        assert_eq!(h.plan_redo(&Scope::Any).unwrap().id, Id128::from_u128(3));
        // A new entry drops the redo tail; the limit drops the oldest.
        let mut small = History::new(2);
        for n in 1..=3 {
            small.push(entry(n, "x", vec![set("a", None, Some(1))]));
        }
        assert_eq!(small.entries().len(), 2);
        assert_eq!(small.entries()[0].id, Id128::from_u128(2));
    }

    #[test]
    fn entries_round_trip_as_json() {
        let mut e = entry(7, "human.artist", vec![set("a", None, Some(3))]);
        e.reverts = Some(Id128::from_u128(5));
        e.open = Some("k".into());
        let back = Entry::from_json(&e.to_json(), &CommandLimits::DEFAULT).unwrap();
        assert_eq!(back, e);
        let mut bad = e.to_json();
        if let Some(o) = bad.as_object_mut() {
            o.insert("extra", 1_u64);
        }
        assert!(Entry::from_json(&bad, &CommandLimits::DEFAULT).is_err());
    }
}
