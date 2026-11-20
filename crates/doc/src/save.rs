//! Crash safety `[SAVE-01]`..`[SAVE-03]`: what the command journal records, how it is replayed,
//! and rotated autosaves.
//!
//! The session appends one [`Record`] per change to the project's journal ([`crate::journal`])
//! before it reports the change as done, so the journal always continues the saved text with
//! every committed change `[SAVE-01]`. Each record carries the exact edits it made, so replay does
//! not depend on anything but the saved text: [`replay`] applies the records in order and rebuilds
//! the history as far as the journal reaches (an undo or redo of a command from before the last
//! save changes the document but has no history entry to move over).
//!
//! The record JSON is `{"op": <op>, ...}` with:
//!
//! - `commit`: `"entry"` (a history [`Entry`]); a gesture's first command is committed with its
//!   entry open;
//! - `extend`: `"id"`, `"edits"`, `"last"`, `"elements"`: the edits a gesture added since the last
//!   record, folded into its open entry (written at most every journal interval, and when the
//!   gesture ends: the batching of slider drags);
//! - `seal`: `"id"`: the gesture ended;
//! - `undo`: `"id"`, `"edits"` (the inverse that was applied): plain undo of the top entry;
//! - `redo`: `"entry"`: the entry applied again;
//! - `accept`: `"proposal"`, `"entries"`: proposal commands merged into the main history `[PROP-02]`.
//!
//! Full saves are the project's ([`crate::project::Project::save`]: blobs first, atomic text
//! write, fresh journal `[SAVE-02]`). [`Autosaves`] writes additional full copies into
//! `autosave/` on idle and every N commands and keeps only the newest few `[SAVE-03]`; they are
//! recovery points that never replace the user's own save.

use humanart_core::id::Id128;
use humanart_core::platform::{FileKind, FileSystem, join_path};
use humanart_io::json::{Number, Object, Value};

use crate::command::apply_edits;
use crate::command::codec::{CommandLimits, edits_from_json, edits_to_json};
use crate::document::Document;
use crate::error::{DocError, Result};
use crate::history::{Entry, History};
use crate::journal::JournalRecord;
use crate::project::Project;
use crate::registry::Registry;
use crate::text::Edit;

/// The autosave directory's name inside a project.
pub const AUTOSAVE_DIR: &str = "autosave";

/// The prefix of autosave file names.
pub const AUTOSAVE_PREFIX: &str = "autosave-";

/// The suffix of autosave file names.
pub const AUTOSAVE_SUFFIX: &str = ".anthropos";

/// Most entries one `accept` record may hold.
pub const MAX_ACCEPT_ENTRIES: usize = 1 << 16;

/// One journal record.
#[derive(Clone, Debug, PartialEq)]
pub enum Record {
    /// A command entered the history.
    Commit(Entry),
    /// A gesture's open entry grew.
    Extend {
        /// The open entry.
        id: Id128,
        /// The edits added (already composed).
        edits: Vec<Edit>,
        /// The entry's new `last_ms`.
        last_ms: u64,
        /// Elements the strokes added touched.
        elements: u64,
    },
    /// A gesture ended.
    Seal {
        /// The entry.
        id: Id128,
    },
    /// The top entry was undone.
    Undo {
        /// The entry.
        id: Id128,
        /// The edits applied (the entry's inverse).
        edits: Vec<Edit>,
    },
    /// An entry was redone.
    Redo(Entry),
    /// Proposal commands were merged.
    Accept {
        /// The proposal.
        proposal: Id128,
        /// The merged entries, in order.
        entries: Vec<Entry>,
    },
}

fn bad(key: &str, what: &str) -> DocError {
    DocError::field("journal record", key, what)
}

fn id_json(id: Id128) -> Value {
    Value::from(id.to_hex().as_str())
}

impl Record {
    /// The `op` member.
    pub fn op(&self) -> &'static str {
        match self {
            Self::Commit(_) => "commit",
            Self::Extend { .. } => "extend",
            Self::Seal { .. } => "seal",
            Self::Undo { .. } => "undo",
            Self::Redo(_) => "redo",
            Self::Accept { .. } => "accept",
        }
    }

    /// The JSON form (see the module documentation).
    pub fn to_json(&self) -> Value {
        let mut o = Object::with_capacity(5);
        o.insert("op", self.op());
        match self {
            Self::Commit(entry) | Self::Redo(entry) => {
                o.insert("entry", entry.to_json());
            }
            Self::Extend { id, edits, last_ms, elements } => {
                o.insert("id", id_json(*id));
                o.insert("edits", edits_to_json(edits));
                o.insert("last", Value::from(Number::from(*last_ms)));
                o.insert("elements", Value::from(Number::from(*elements)));
            }
            Self::Seal { id } => {
                o.insert("id", id_json(*id));
            }
            Self::Undo { id, edits } => {
                o.insert("id", id_json(*id));
                o.insert("edits", edits_to_json(edits));
            }
            Self::Accept { proposal, entries } => {
                o.insert("proposal", id_json(*proposal));
                o.insert("entries", Value::Array(entries.iter().map(Entry::to_json).collect()));
            }
        }
        Value::Object(o)
    }

    /// Reads the JSON form.
    pub fn from_json(value: &Value, limits: &CommandLimits) -> Result<Self> {
        let o = value.as_object().ok_or_else(|| bad("", "expected an object"))?;
        let op = o.get("op").and_then(Value::as_str).ok_or_else(|| bad("op", "expected a string"))?;
        let known: &[&str] = match op {
            "commit" | "redo" => &["op", "entry"],
            "extend" => &["op", "id", "edits", "last", "elements"],
            "seal" => &["op", "id"],
            "undo" => &["op", "id", "edits"],
            "accept" => &["op", "proposal", "entries"],
            _ => return Err(bad("op", "unknown record kind")),
        };
        if let Some((k, _)) = o.entries().iter().find(|(k, _)| !known.contains(&k.as_str())) {
            return Err(bad(k, "unknown member"));
        }
        let get = |key: &str| o.get(key).ok_or_else(|| bad(key, "missing"));
        let id = |key: &str| -> Result<Id128> {
            get(key)?.as_str().and_then(|s| Id128::parse(s).ok()).ok_or_else(|| bad(key, "expected an id"))
        };
        let number =
            |key: &str| -> Result<u64> { get(key)?.as_u64().ok_or_else(|| bad(key, "expected an unsigned integer")) };
        Ok(match op {
            "commit" => Self::Commit(Entry::from_json(get("entry")?, limits)?),
            "redo" => Self::Redo(Entry::from_json(get("entry")?, limits)?),
            "extend" => Self::Extend {
                id: id("id")?,
                edits: edits_from_json(get("edits")?, limits)?,
                last_ms: number("last")?,
                elements: number("elements")?,
            },
            "seal" => Self::Seal { id: id("id")? },
            "undo" => Self::Undo { id: id("id")?, edits: edits_from_json(get("edits")?, limits)? },
            _ => {
                let list = get("entries")?.as_array().ok_or_else(|| bad("entries", "expected an array"))?;
                if list.len() > MAX_ACCEPT_ENTRIES {
                    return Err(DocError::Limit {
                        what: "entries in an accept record",
                        limit: MAX_ACCEPT_ENTRIES as u64,
                    });
                }
                let entries = list.iter().map(|v| Entry::from_json(v, limits)).collect::<Result<Vec<_>>>()?;
                Self::Accept { proposal: id("proposal")?, entries }
            }
        })
    }
}

/// What [`replay`] did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Replayed {
    /// Records replayed.
    pub records: usize,
    /// The proposals whose merge was replayed (their files are obsolete).
    pub accepted: Vec<Id128>,
    /// The sequence number of the last record (0 when there were none).
    pub last_seq: u64,
}

/// Applies the journal `records` to `doc` and rebuilds `history` `[SAVE-01]`. All or nothing: on
/// error (a record that does not read or does not apply, which means the journal does not belong
/// to this text) `doc` and `history` are unchanged and the error names the record's line.
pub fn replay(
    doc: &mut Document,
    history: &mut History,
    registry: &Registry,
    records: &[JournalRecord],
    limits: &CommandLimits,
) -> Result<Replayed> {
    let mut work = doc.clone();
    let mut hist = history.clone();
    let mut out = Replayed::default();
    for (n, record) in records.iter().enumerate() {
        let failed =
            |_: DocError| DocError::Journal { line: n + 2, what: "record does not apply to the saved document" };
        let parsed = Record::from_json(&record.payload, limits).map_err(failed)?;
        replay_one(&mut work, &mut hist, registry, parsed, &mut out.accepted).map_err(failed)?;
        out.records += 1;
        out.last_seq = record.seq;
    }
    *doc = work;
    *history = hist;
    Ok(out)
}

fn replay_one(
    doc: &mut Document,
    history: &mut History,
    registry: &Registry,
    record: Record,
    accepted: &mut Vec<Id128>,
) -> Result<()> {
    match record {
        Record::Commit(entry) => {
            history.seal();
            apply_edits(doc, registry, &entry.edits)?;
            history.push(entry);
        }
        Record::Extend { id, edits, last_ms, elements } => {
            apply_edits(doc, registry, &edits)?;
            if history.open().is_some_and(|e| e.id == id) {
                history.extend_open(&edits, last_ms, elements);
            }
        }
        Record::Seal { id } => {
            if history.open().is_some_and(|e| e.id == id) {
                history.seal();
            }
        }
        Record::Undo { id, edits } => {
            history.seal();
            apply_edits(doc, registry, &edits)?;
            if history.applied().last().is_some_and(|e| e.id == id) {
                history.step_back();
            }
        }
        Record::Redo(entry) => {
            history.seal();
            apply_edits(doc, registry, &entry.edits)?;
            if history.entries().get(history.cursor()).is_some_and(|e| e.id == entry.id) {
                history.step_forward();
            } else {
                history.push(entry);
            }
        }
        Record::Accept { proposal, entries } => {
            history.seal();
            for entry in &entries {
                apply_edits(doc, registry, &entry.edits)?;
            }
            for entry in entries {
                history.push(entry);
            }
            accepted.push(proposal);
        }
    }
    Ok(())
}

/// The rotated autosaves of a project `[SAVE-03]`.
pub struct Autosaves<'fs> {
    fs: &'fs dyn FileSystem,
    dir: String,
}

fn autosave_number(name: &str) -> Option<u64> {
    let digits = name.strip_prefix(AUTOSAVE_PREFIX)?.strip_suffix(AUTOSAVE_SUFFIX)?;
    if digits.len() != 20 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

impl<'fs> Autosaves<'fs> {
    /// The autosaves of `project`.
    pub fn new(fs: &'fs dyn FileSystem, project: &Project) -> Self {
        Self { fs, dir: join_path(project.root(), AUTOSAVE_DIR) }
    }

    /// The directory.
    pub fn dir(&self) -> &str {
        &self.dir
    }

    /// The autosave file names with their numbers, oldest first. A missing directory means none.
    pub fn list(&self) -> Result<Vec<(u64, String)>> {
        let entries = match self.fs.list(&self.dir) {
            Ok(entries) => entries,
            Err(humanart_core::error::PlatformError::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(DocError::platform(&self.dir, e)),
        };
        let mut out: Vec<(u64, String)> = entries
            .into_iter()
            .filter(|e| e.kind == FileKind::File)
            .filter_map(|e| autosave_number(&e.name).map(|n| (n, e.name)))
            .collect();
        out.sort_unstable();
        Ok(out)
    }

    /// The path of the newest autosave, if any.
    pub fn latest(&self) -> Result<Option<String>> {
        Ok(self.list()?.pop().map(|(_, name)| join_path(&self.dir, &name)))
    }

    /// Writes `doc` as a new autosave (checked like a full save: its blobs must be in the
    /// project's store), atomically, then deletes all but the newest `keep` (at least 1). Returns
    /// the new file's path.
    pub fn write(&self, project: &Project, doc: &Document, registry: &Registry, keep: usize) -> Result<String> {
        doc.check(registry)?;
        let store = project.blobs(self.fs);
        if let Some(missing) = doc.blob_refs().into_iter().find(|h| !store.contains(*h)) {
            return Err(DocError::Blob { hash: missing, what: "referenced by the document but not in the blob store" });
        }
        let text = doc.write()?;
        self.fs.create_dir_all(&self.dir).map_err(|e| DocError::platform(&self.dir, e))?;
        let existing = self.list()?;
        let next = existing.last().map_or(1, |(n, _)| n.saturating_add(1));
        let path = join_path(&self.dir, &format!("{AUTOSAVE_PREFIX}{next:020}{AUTOSAVE_SUFFIX}"));
        self.fs.write_atomic(&path, text.as_bytes()).map_err(|e| DocError::platform(&path, e))?;
        let total = existing.len() + 1;
        for (_, name) in existing.iter().take(total.saturating_sub(keep.max(1))) {
            let old = join_path(&self.dir, name);
            match self.fs.remove(&old) {
                Ok(()) | Err(humanart_core::error::PlatformError::NotFound) => {}
                Err(e) => return Err(DocError::platform(&old, e)),
            }
        }
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::tests::sample;
    use crate::document::{Author, AuthorKind};
    use crate::registry::tests::real_registry;
    use crate::text::SectionRef;
    use humanart_core::platform::MemoryFs;
    use humanart_io::json;

    fn set(key: &str, before: Option<f64>, after: Option<f64>) -> Edit {
        Edit::Set {
            section: SectionRef::new("params", None),
            key: key.into(),
            before: before.map(|v| Value::from(Number::Float(v))),
            after: after.map(|v| Value::from(Number::Float(v))),
        }
    }

    fn entry(n: u128, edits: Vec<Edit>) -> Entry {
        Entry {
            id: Id128::from_u128(n),
            author: "human.artist".into(),
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

    fn records(list: &[Record]) -> Vec<JournalRecord> {
        list.iter().enumerate().map(|(n, r)| JournalRecord { seq: n as u64 + 1, payload: r.to_json() }).collect()
    }

    #[test]
    fn save01_records_round_trip_and_replay_rebuilds_document_and_history() {
        let reg = real_registry();
        let base = sample(&reg);
        let jaw = "face.jaw.width";
        let mut open = entry(1, vec![set(jaw, None, Some(0.1))]);
        open.open = Some("human.artist set_params face.jaw.width".into());
        let list = vec![
            Record::Commit(open),
            Record::Extend {
                id: Id128::from_u128(1),
                edits: vec![set(jaw, Some(0.1), Some(0.2))],
                last_ms: 5,
                elements: 0,
            },
            Record::Seal { id: Id128::from_u128(1) },
            Record::Commit(entry(2, vec![set(jaw, Some(0.2), Some(0.3))])),
            Record::Undo { id: Id128::from_u128(2), edits: vec![set(jaw, Some(0.3), Some(0.2))] },
            Record::Redo(entry(2, vec![set(jaw, Some(0.2), Some(0.3))])),
            Record::Accept {
                proposal: Id128::from_u128(9),
                entries: vec![entry(3, vec![set(jaw, Some(0.3), Some(0.4))])],
            },
        ];
        for r in &list {
            let text = json::to_string(&r.to_json()).unwrap();
            let back = Record::from_json(&json::parse_str(&text).unwrap(), &CommandLimits::DEFAULT).unwrap();
            assert_eq!(&back, r);
        }
        let mut doc = base.clone();
        let mut history = History::new(100);
        let done = replay(&mut doc, &mut history, &reg, &records(&list), &CommandLimits::DEFAULT).unwrap();
        assert_eq!(done, Replayed { records: 7, accepted: vec![Id128::from_u128(9)], last_seq: 7 });
        assert_eq!(doc.param(&reg, jaw).unwrap(), &crate::registry::ParamValue::Scalar(0.4));
        let ids: Vec<Id128> = history.entries().iter().map(|e| e.id).collect();
        assert_eq!(ids, [1, 2, 3].map(Id128::from_u128).to_vec());
        assert_eq!(history.cursor(), 3);
        assert_eq!(history.entries()[0].edits, vec![set(jaw, None, Some(0.2))]);
        assert!(history.open().is_none());
        // A record that does not apply leaves everything unchanged and names its line.
        let mut wrong = list.clone();
        wrong.insert(3, Record::Commit(entry(4, vec![set(jaw, Some(0.9), Some(0.1))])));
        let mut doc2 = base.clone();
        let mut h2 = History::new(100);
        let err = replay(&mut doc2, &mut h2, &reg, &records(&wrong), &CommandLimits::DEFAULT).unwrap_err();
        assert!(matches!(err, DocError::Journal { line: 5, .. }), "{err}");
        assert_eq!(doc2, base);
        assert!(h2.entries().is_empty());
        let mut unknown = Object::new();
        unknown.insert("op", "explode");
        assert!(Record::from_json(&Value::Object(unknown), &CommandLimits::DEFAULT).is_err());
    }

    #[test]
    fn save03_autosaves_rotate() {
        let reg = real_registry();
        let fs = MemoryFs::new();
        let artist = Author { id: "human.artist".into(), kind: AuthorKind::Human, name: "Artist".into() };
        let mut doc = Document::new(Id128::from_u128(1), "A", "humanart-template", 3, 0, artist);
        doc.set_param(&reg, "face.jaw.width", crate::registry::ParamValue::Scalar(0.2), "human.artist").unwrap();
        let project = Project::create(&fs, "p", &doc, &reg).unwrap();
        let autosaves = Autosaves::new(&fs, &project);
        assert_eq!(autosaves.latest().unwrap(), None);
        for _ in 0..5 {
            autosaves.write(&project, &doc, &reg, 3).unwrap();
        }
        let names: Vec<u64> = autosaves.list().unwrap().into_iter().map(|(n, _)| n).collect();
        assert_eq!(names, vec![3, 4, 5]);
        let latest = autosaves.latest().unwrap().unwrap();
        assert!(latest.ends_with("autosave-00000000000000000005.anthropos"));
        assert_eq!(Document::parse(&fs.read(&latest).unwrap(), &reg, &crate::text::TextLimits::DEFAULT).unwrap(), doc);
        fs.write("p/autosave/autosave-1.anthropos", b"x").unwrap();
        assert_eq!(autosaves.list().unwrap().len(), 3);
    }
}
