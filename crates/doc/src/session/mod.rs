//! The editing session: one document, its shared history, proposals, events and crash safety.
//!
//! A [`Session`] is what the editor and the agent server both drive. Every change goes through it,
//! so the human and the agent get the same validation, history, notifications and journal:
//!
//! - [`Session::execute`] resolves a [`Command`], validates it against locks and human-edit
//!   protection ([`crate::lock::check`], `[CMD-05]`, `[LOCK-01]`, `[LOCK-02]`), applies it, journals
//!   it `[SAVE-01]`, records it in the [`History`] and emits an event `[NOTE-01]`. The result is the
//!   structured [`Outcome`] `[CMD-06]`. A command marked as part of a gesture coalesces with the
//!   open history entry of the same (author, kind, target) into one undo step `[CMD-03]`; the
//!   gesture ends on [`Session::end_gesture`] (release), after a pause longer than
//!   [`SessionConfig::coalesce_pause_ms`] ([`Session::tick`]) and before any other operation.
//!   Every frame of a gesture is applied at once (the evaluation snapshot streams it); its edits
//!   reach the journal at most every [`SessionConfig::journal_interval_ms`] and when it ends.
//! - [`Session::undo`] / [`Session::redo`] work on the shared linear history with per-author scope
//!   `[CMD-04]`.
//! - Proposals `[PROP-01]`..`[PROP-04]`: [`Session::open_proposal`], [`Session::execute_in`],
//!   [`Session::undo_in`], [`Session::accept`] (whole or a selection, with dependency and conflict
//!   checks), [`Session::discard`]. Proposals are files in the project and reload with it.
//! - [`Session::save`] is the full atomic save `[SAVE-02]`; [`Session::tick`] seals paused
//!   gestures, flushes batched journal records and writes rotated autosaves on idle and every N
//!   changes `[SAVE-03]`; [`Session::open`] detects an unclean shutdown and replays the journal.
//!
//! Failure rules: a refused or invalid command changes nothing. When the journal cannot be
//! written, the change is rolled back and reported as [`CommandError::Storage`], so the document
//! never runs ahead of the journal (except for the batched frames of an open gesture, which are
//! retried until they are written). After a failed write the journal is re-read before the next
//! record and cut back to the records the session knows it wrote.
//!
//! Times: command timestamps come from the wall clock; pauses and intervals are measured on the
//! monotonic clock. Command and proposal ids the caller leaves [`Id128::NIL`] are derived from the
//! session's seed (the host draws a random seed per session) and a counter, so a scripted session
//! is deterministic.

use humanart_core::hash::{Hash128, hash128};
use humanart_core::id::Id128;
use humanart_core::platform::{Clock, FileSystem, join_path};

use crate::command::{
    Command, CommandError, Context, Outcome, Rejection, Warning, WarningCode, apply_edits, derive_id, describe, resolve,
};
use crate::document::{Author, AuthorKind, Document};
use crate::error::DocError;
use crate::history::{Entry, History, Scope, UndoPlan, compose};
use crate::journal::{Journal, JournalHeader, JournalLimits, JournalRecord, encode_header, encode_record};
use crate::lock::{self, Mode};
use crate::notify::{EventKind, EventLog};
use crate::project::{JournalState, LoadLimits, Project};
use crate::proposal::{LoadedProposals, PROPOSALS_DIR, Proposal, ProposalLimits, ProposalStore, SelectError};
use crate::registry::Registry;
use crate::save::{Autosaves, Record, replay};
use crate::text::{Edit, invert};

/// The session's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SessionConfig {
    /// Most history entries kept.
    pub history_limit: usize,
    /// Most events kept for "events since N" `[NOTE-02]`.
    pub event_capacity: usize,
    /// A gesture pausing longer than this ends its undo step `[CMD-03]`.
    pub coalesce_pause_ms: u64,
    /// Longest time a gesture's frames wait before they are journaled `[SAVE-01]`.
    pub journal_interval_ms: u64,
    /// Autosave after this many changes (0: never) `[SAVE-03]`.
    pub autosave_every: u32,
    /// Autosave after this long without changes (0: never) `[SAVE-03]`.
    pub autosave_idle_ms: u64,
    /// Autosaves kept.
    pub autosave_keep: usize,
    /// A plausibility drop larger than this attaches a warning.
    pub plausibility_drop: f64,
    /// Limits of loading the project.
    pub load: LoadLimits,
    /// Limits of reading proposals and journal records.
    pub proposals: ProposalLimits,
}

impl SessionConfig {
    /// 10 000 history entries, 4096 events, 1 s gesture pause, 100 ms journal batching, autosave
    /// every 50 changes or after 30 s idle keeping 5, warning on a plausibility drop over 0.05.
    pub const DEFAULT: Self = Self {
        history_limit: 10_000,
        event_capacity: 4096,
        coalesce_pause_ms: 1000,
        journal_interval_ms: 100,
        autosave_every: 50,
        autosave_idle_ms: 30_000,
        autosave_keep: 5,
        plausibility_drop: 0.05,
        load: LoadLimits { text: crate::text::TextLimits::DEFAULT, journal: JournalLimits::DEFAULT },
        proposals: ProposalLimits::DEFAULT,
    };
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A plausibility scorer (WP-25 provides the real one); with one installed, outcomes carry the
/// score before and after `[CMD-06]` and a drop raises a warning.
pub trait Plausibility {
    /// The score of `doc`, higher is more plausible.
    fn score(&self, doc: &Document, registry: &Registry) -> f64;
}

/// What happened to the journal when a project was opened.
#[derive(Debug)]
pub enum JournalOutcome {
    /// There was none; a fresh one was started.
    Missing,
    /// The last session ended cleanly.
    Clean,
    /// The last session did not end cleanly; this many records were replayed `[SAVE-01]`.
    Recovered {
        /// Records replayed.
        records: usize,
    },
    /// The journal belonged to another save or document and was ignored.
    Stale,
    /// The journal continues the saved text but does not replay; it was moved aside and the saved
    /// text opened.
    Unreplayable {
        /// Why.
        error: DocError,
        /// Where the journal was moved.
        kept_at: String,
    },
}

/// What [`Session::open`] found.
#[derive(Debug)]
pub struct OpenReport {
    /// The format version of the file when it was migrated.
    pub migrated_from: Option<u32>,
    /// Blobs the document references that are not in the store.
    pub missing_blobs: Vec<Hash128>,
    /// The journal.
    pub journal: JournalOutcome,
    /// Proposal files that could not be read (left on disk).
    pub damaged_proposals: Vec<(String, DocError)>,
}

/// The result of accepting a proposal.
#[derive(Clone, Debug, PartialEq)]
pub struct Accepted {
    /// What the merge changed (`kind` `accept_proposal`, `command` the proposal id).
    pub outcome: Outcome,
    /// The merged commands, in order.
    pub accepted: Vec<Id128>,
    /// The commands left out.
    pub dropped: Vec<Id128>,
}

#[derive(Clone, Copy, Debug, Default)]
struct JournalLink {
    /// The sequence number of the last record written.
    seq: u64,
    /// A write failed: re-read the journal before the next one.
    suspect: bool,
    /// Start a fresh journal on this base before the next record.
    restart: Option<Hash128>,
}

#[derive(Clone, Debug)]
struct Pending {
    id: Id128,
    edits: Vec<Edit>,
    since_ms: u64,
    last_ms: u64,
    elements: u64,
}

/// An editing session (see the module documentation).
pub struct Session<'a> {
    registry: &'a Registry,
    fs: &'a dyn FileSystem,
    clock: &'a dyn Clock,
    scorer: Option<&'a dyn Plausibility>,
    config: SessionConfig,
    project: Option<Project>,
    doc: Document,
    history: History,
    events: EventLog,
    proposals: Vec<Proposal>,
    closed: Vec<Id128>,
    seed: Id128,
    ids: u64,
    link: JournalLink,
    pending: Option<Pending>,
    /// The project's journal path (empty without a project), kept so that a batched write does
    /// not build it `[PERF-R1]`.
    journal_path: String,
    /// The record line being written, reused between batches `[PERF-R1]`.
    line: Vec<u8>,
    /// The edits of the last batch written, reused by the next drag batch `[PERF-R1]`.
    spare_edits: Vec<Edit>,
    gesture_ms: u64,
    since_autosave: u32,
    last_change_ms: u64,
    revision: u64,
}

fn storage_or_invalid(e: DocError) -> CommandError {
    match e {
        DocError::Platform { .. } => CommandError::Storage(e),
        other => CommandError::Invalid(other),
    }
}

fn no_project() -> CommandError {
    CommandError::Invalid(DocError::field("session", "project", "the session has no project directory"))
}

impl<'a> Session<'a> {
    /// A session on `doc` without a project directory: nothing is journaled or saved and
    /// proposals live in memory. Strokes and snapshots need a project (its blob store).
    pub fn new(
        doc: Document,
        registry: &'a Registry,
        fs: &'a dyn FileSystem,
        clock: &'a dyn Clock,
        config: SessionConfig,
        seed: Id128,
    ) -> Self {
        let now = clock.now_nanos() / 1_000_000;
        Self {
            registry,
            fs,
            clock,
            scorer: None,
            config,
            project: None,
            doc,
            history: History::new(config.history_limit),
            events: EventLog::new(config.event_capacity),
            proposals: Vec::new(),
            closed: Vec::new(),
            seed,
            ids: 0,
            link: JournalLink::default(),
            pending: None,
            journal_path: String::new(),
            line: Vec::new(),
            spare_edits: Vec::new(),
            gesture_ms: now,
            since_autosave: 0,
            last_change_ms: now,
            revision: 0,
        }
    }

    /// Creates the project directory `root` with `doc` as its first save and a session on it.
    pub fn create(
        root: &str,
        doc: Document,
        registry: &'a Registry,
        fs: &'a dyn FileSystem,
        clock: &'a dyn Clock,
        config: SessionConfig,
        seed: Id128,
    ) -> Result<Self, CommandError> {
        let project = Project::create(fs, root, &doc, registry).map_err(storage_or_invalid)?;
        let mut session = Self::new(doc, registry, fs, clock, config, seed);
        session.journal_path = project.journal_path();
        session.project = Some(project);
        Ok(session)
    }

    /// Opens the project at `root`: loads (migrating) the document, its proposals, and replays the
    /// journal when the last session did not end cleanly `[SAVE-01]`, `[PROP-04]`.
    pub fn open(
        root: &str,
        registry: &'a Registry,
        fs: &'a dyn FileSystem,
        clock: &'a dyn Clock,
        config: SessionConfig,
        seed: Id128,
    ) -> Result<(Self, OpenReport), CommandError> {
        let project = Project::at(root);
        let loaded = project.load(fs, registry, &config.load).map_err(storage_or_invalid)?;
        let LoadedProposals { proposals, damaged } = ProposalStore::new(fs, &join_path(root, PROPOSALS_DIR))
            .load_all(registry, &config.proposals)
            .map_err(storage_or_invalid)?;
        let mut session = Self::new(loaded.document, registry, fs, clock, config, seed);
        session.proposals = proposals;
        let journal = match loaded.journal {
            JournalState::Missing => {
                session.link.restart = Some(loaded.text_hash);
                JournalOutcome::Missing
            }
            JournalState::Stale(_) => {
                session.link.restart = Some(loaded.text_hash);
                JournalOutcome::Stale
            }
            JournalState::Clean => JournalOutcome::Clean,
            JournalState::Recover(records) => {
                match replay(&mut session.doc, &mut session.history, registry, &records, &config.proposals.commands) {
                    Ok(done) => {
                        session.link.seq = done.last_seq;
                        session.proposals.retain(|p| !done.accepted.contains(&p.id));
                        session.closed = done.accepted;
                        session.since_autosave = u32::try_from(done.records).unwrap_or(u32::MAX);
                        session.events.push(EventKind::Recovered { records: done.records });
                        JournalOutcome::Recovered { records: done.records }
                    }
                    Err(error) => {
                        let path = project.journal_path();
                        let kept_at = format!("{path}.damaged");
                        fs.rename(&path, &kept_at).map_err(|e| CommandError::Storage(DocError::platform(&path, e)))?;
                        session.link.restart = Some(loaded.text_hash);
                        JournalOutcome::Unreplayable { error, kept_at }
                    }
                }
            }
        };
        session.journal_path = project.journal_path();
        session.project = Some(project);
        session.seal_open()?;
        // Obsolete proposal files are retried by every save and tick if this fails.
        let _ = session.remove_closed();
        Ok((
            session,
            OpenReport {
                migrated_from: loaded.migrated_from,
                missing_blobs: loaded.missing_blobs,
                journal,
                damaged_proposals: damaged,
            },
        ))
    }

    /// The same session with a plausibility scorer.
    #[must_use]
    pub fn with_scorer(mut self, scorer: &'a dyn Plausibility) -> Self {
        self.scorer = Some(scorer);
        self
    }

    /// The main document.
    pub fn document(&self) -> &Document {
        &self.doc
    }

    /// Counts up with every change of the main document made by a command, undo, redo or
    /// accept, including each frame of an open gesture (which emits no event until it ends). A
    /// mirror of the document (the evaluated model) re-reads it when this or the event sequence
    /// moved.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// The history.
    pub fn history(&self) -> &History {
        &self.history
    }

    /// The event log `[NOTE-02]`.
    pub fn events(&self) -> &EventLog {
        &self.events
    }

    /// The open proposals, oldest first.
    pub fn proposals(&self) -> &[Proposal] {
        &self.proposals
    }

    /// The open proposal with this id.
    pub fn proposal(&self, id: Id128) -> Option<&Proposal> {
        self.proposals.iter().find(|p| p.id == id)
    }

    /// The project, if the session has one.
    pub fn project(&self) -> Option<&Project> {
        self.project.as_ref()
    }

    fn mono_ms(&self) -> u64 {
        self.clock.now_nanos() / 1_000_000
    }

    fn next_id(&mut self) -> Id128 {
        self.ids += 1;
        derive_id(self.seed, self.ids)
    }

    fn author(&self, id: &str) -> Result<Author, CommandError> {
        self.doc.author(id).cloned().ok_or_else(|| Rejection::UnknownAuthor { author: id.to_owned() }.into())
    }

    fn prepare(&mut self, cmd: &mut Command) {
        if cmd.id.is_nil() {
            cmd.id = self.next_id();
        }
        if cmd.time_ms == 0 {
            cmd.time_ms = self.clock.unix_millis();
        }
        if cmd.label.is_empty() {
            cmd.action.kind_name().clone_into(&mut cmd.label);
        }
    }

    fn touch(&mut self) {
        self.revision = self.revision.wrapping_add(1);
        self.last_change_ms = self.mono_ms();
        self.since_autosave = self.since_autosave.saturating_add(1);
    }

    fn score(&self, doc: &Document) -> Option<f64> {
        self.scorer.map(|s| s.score(doc, self.registry))
    }

    fn scored(&self, before: Option<f64>, warnings: &mut Vec<Warning>) -> Option<[f64; 2]> {
        let before = before?;
        let after = self.score(&self.doc)?;
        if after < before - self.config.plausibility_drop {
            warnings.push(Warning {
                code: WarningCode::PlausibilityDrop,
                target: "document".to_owned(),
                message: format!("plausibility dropped from {before:.3} to {after:.3}"),
            });
        }
        Some([before, after])
    }

    fn outcome(&self, entry: &Entry, edits: &[Edit], proposal: Option<Id128>) -> Outcome {
        Outcome {
            command: entry.id,
            author: entry.author.clone(),
            kind: entry.kind.clone(),
            label: entry.label.clone(),
            time_ms: entry.time_ms,
            proposal,
            changes: describe(edits, self.registry),
            warnings: Vec::new(),
            elements: entry.elements,
            plausibility: None,
            coalesced: false,
        }
    }

    /// Undoes edits just applied to the main document after a later step failed.
    fn revert(&mut self, edits: &[Edit]) {
        // The inverse of edits that were just applied finds exactly the values they left, so it
        // cannot fail; if it did, the error that made us revert is still the one reported.
        let _ = apply_edits(&mut self.doc, self.registry, &invert(edits));
    }

    // ---- the journal -------------------------------------------------------------------------

    fn journal(&self) -> Option<Journal<'a>> {
        self.project.as_ref().map(|p| p.journal(self.fs))
    }

    /// Re-reads the journal after a failed write and cuts it back to the records this session
    /// knows it wrote.
    fn resync(&self, journal: &Journal<'_>) -> Result<(), DocError> {
        let contents = journal
            .repair(&self.config.load.journal)?
            .ok_or(DocError::Journal { line: 1, what: "the journal disappeared" })?;
        if contents.records.last().map_or(0, |r| r.seq) != self.link.seq {
            let mut text = encode_header(&contents.header);
            for record in contents.records.iter().take_while(|r| r.seq <= self.link.seq) {
                text.push_str(&encode_record(record)?);
            }
            self.fs.write_atomic(journal.path(), text.as_bytes()).map_err(|e| DocError::platform(journal.path(), e))?;
        }
        Ok(())
    }

    fn append(&mut self, record: &Record) -> Result<(), DocError> {
        let Some(journal) = self.journal() else { return Ok(()) };
        if let Some(base) = self.link.restart {
            journal.start(&JournalHeader { document: self.doc.meta.id, base })?;
            self.link = JournalLink::default();
        }
        if self.link.suspect {
            self.resync(&journal)?;
            self.link.suspect = false;
        }
        let line = JournalRecord { seq: self.link.seq + 1, payload: record.to_json() };
        match journal.append(&line) {
            Ok(()) => {
                self.link.seq += 1;
                Ok(())
            }
            Err(e) => {
                self.link.suspect = true;
                Err(e)
            }
        }
    }

    /// Journals the batched gesture frames. A failed write keeps them pending.
    fn flush_pending(&mut self) -> Result<(), DocError> {
        let Some(p) = self.pending.take() else { return Ok(()) };
        match self.append_extend(&p) {
            Ok(()) => {
                // Their strings are reused by the next batch of a drag.
                self.spare_edits = p.edits;
                Ok(())
            }
            Err(e) => {
                self.pending = Some(p);
                Err(e)
            }
        }
    }

    /// Appends the `extend` record of `p`. In the steady state (journal started and in step, all
    /// edits sets) the line is written into the reused buffer and appended directly, so nothing
    /// is allocated here `[PERF-R1]`; the bytes are those of [`Session::append`].
    fn append_extend(&mut self, p: &Pending) -> Result<(), DocError> {
        /// A buffer grown past this by a large record is not kept.
        const KEEP_LINE_BYTES: usize = 64 << 10;
        if self.project.is_some() && self.link.restart.is_none() && !self.link.suspect {
            let seq = self.link.seq + 1;
            let encoded = crate::journal::encode_record_into(&mut self.line, seq, |out| {
                crate::save::write_extend(out, p.id, &p.edits, p.last_ms, p.elements)
            });
            let result = match encoded {
                Ok(true) => match self.fs.append(&self.journal_path, &self.line) {
                    Ok(()) => {
                        self.link.seq = seq;
                        Ok(true)
                    }
                    Err(e) => {
                        self.link.suspect = true;
                        Err(DocError::platform(&self.journal_path, e))
                    }
                },
                other => other,
            };
            if self.line.capacity() > KEEP_LINE_BYTES {
                self.line = Vec::new();
            }
            if result? {
                return Ok(());
            }
        }
        self.append(&Record::Extend { id: p.id, edits: p.edits.clone(), last_ms: p.last_ms, elements: p.elements })
    }

    /// Ends an open gesture: journals its last frames, seals the entry and emits its event.
    fn seal_open(&mut self) -> Result<Option<Outcome>, CommandError> {
        let Some(id) = self.history.open().map(|e| e.id) else { return Ok(None) };
        self.flush_pending().map_err(CommandError::Storage)?;
        self.append(&Record::Seal { id }).map_err(CommandError::Storage)?;
        let Some(entry) = self.history.seal().cloned() else { return Ok(None) };
        let outcome = self.outcome(&entry, &entry.edits, None);
        self.events.push(EventKind::Committed(outcome.clone()));
        Ok(Some(outcome))
    }

    fn remove_closed(&mut self) -> Result<(), DocError> {
        let Some(project) = &self.project else {
            self.closed.clear();
            return Ok(());
        };
        let store = ProposalStore::new(self.fs, &join_path(project.root(), PROPOSALS_DIR));
        while let Some(id) = self.closed.last().copied() {
            store.remove(id)?;
            self.closed.pop();
        }
        Ok(())
    }

    // ---- commands ----------------------------------------------------------------------------

    /// Applies `cmd` to the main document `[CMD-01]`..`[CMD-06]`. Ids, times and labels the
    /// command leaves empty are filled in. On error nothing changed.
    pub fn execute(&mut self, mut cmd: Command) -> Result<Outcome, CommandError> {
        self.prepare(&mut cmd);
        let author = self.author(&cmd.author)?;
        let kind = cmd.action.kind_name();
        let key =
            if cmd.gesture { cmd.action.coalesce_target().map(|t| format!("{} {kind} {t}", cmd.author)) } else { None };
        let extending = key.as_deref().is_some_and(|k| self.history.open_with(k).is_some());
        if !extending {
            self.seal_open()?;
        }
        let resolved = {
            let store = self.project.as_ref().map(|p| p.blobs(self.fs));
            resolve(&self.doc, &cmd, &Context { registry: self.registry, blobs: store.as_ref() })?
        };
        let mut warnings = resolved.warnings;
        let edits = resolved.edits;
        if edits.is_empty() && key.is_none() {
            // Nothing to change: no history entry, journal record or event.
            let entry = Self::entry_for(&cmd, Vec::new(), None, resolved.elements);
            let mut outcome = self.outcome(&entry, &[], None);
            outcome.warnings = warnings;
            return Ok(outcome);
        }
        warnings.extend(lock::check(
            &self.doc,
            self.registry,
            &edits,
            &author,
            &resolved.regions,
            &self.history,
            Mode::Main,
        )?);
        let before = self.score(&self.doc);
        apply_edits(&mut self.doc, self.registry, &edits)?;
        let plausibility = self.scored(before, &mut warnings);
        let now = self.mono_ms();
        if extending {
            if self.history.extend_open(&edits, cmd.time_ms, resolved.elements) {
                self.gesture_ms = now;
                let open_id = self.history.open().map_or(cmd.id, |e| e.id);
                match &mut self.pending {
                    Some(p) => {
                        compose(&mut p.edits, &edits);
                        p.last_ms = cmd.time_ms;
                        p.elements = p.elements.saturating_add(resolved.elements);
                    }
                    None => {
                        self.pending = Some(Pending {
                            id: open_id,
                            edits: edits.clone(),
                            since_ms: now,
                            last_ms: cmd.time_ms,
                            elements: resolved.elements,
                        });
                    }
                }
                if self
                    .pending
                    .as_ref()
                    .is_some_and(|p| now.saturating_sub(p.since_ms) >= self.config.journal_interval_ms)
                {
                    // A failed flush keeps the frames pending; the gesture's end retries it and
                    // reports the error.
                    let _ = self.flush_pending();
                }
                self.touch();
                let entry = Self::entry_for(&cmd, Vec::new(), None, resolved.elements);
                let mut outcome = self.outcome(&entry, &edits, None);
                outcome.warnings = warnings;
                outcome.plausibility = plausibility;
                outcome.coalesced = true;
                return Ok(outcome);
            }
            // The frame cannot be folded into the open entry: the gesture ends here.
            if let Err(e) = self.seal_open() {
                self.revert(&edits);
                return Err(e);
            }
        }
        let entry = Self::entry_for(&cmd, edits, key, resolved.elements);
        if let Err(e) = self.append(&Record::Commit(entry.clone())) {
            self.revert(&entry.edits);
            return Err(CommandError::Storage(e));
        }
        let mut outcome = self.outcome(&entry, &entry.edits, None);
        outcome.warnings = warnings;
        outcome.plausibility = plausibility;
        let gesture = entry.open.is_some();
        self.history.push(entry);
        if gesture {
            self.gesture_ms = now;
        } else {
            self.events.push(EventKind::Committed(outcome.clone()));
        }
        self.touch();
        Ok(outcome)
    }

    fn entry_for(cmd: &Command, edits: Vec<Edit>, open: Option<String>, elements: u64) -> Entry {
        Entry {
            id: cmd.id,
            author: cmd.author.clone(),
            kind: cmd.action.kind_name().to_owned(),
            label: cmd.label.clone(),
            time_ms: cmd.time_ms,
            last_ms: cmd.time_ms,
            edits,
            reverts: None,
            open,
            elements,
        }
    }

    /// Ends the open gesture (the slider or brush was released) `[CMD-03]`. Returns the outcome of
    /// the whole gesture, or `None` when none was open or it ended where it started.
    pub fn end_gesture(&mut self) -> Result<Option<Outcome>, CommandError> {
        self.seal_open()
    }

    /// Undoes for `author` `[CMD-04]`: [`Scope::Any`] steps back over the most recent entry
    /// whoever made it; [`Scope::Author`] reverts that author's most recent entry, with a new
    /// entry when later commands exist and none of them touches the same targets, and otherwise
    /// reports the conflict. The undo is validated like a command by `author`.
    pub fn undo(&mut self, author: &str, scope: &Scope) -> Result<Outcome, CommandError> {
        let who = self.author(author)?;
        self.seal_open()?;
        let plan = self.history.plan_undo(scope)?;
        let now = self.clock.unix_millis();
        let index = match plan {
            UndoPlan::Top => self.history.cursor().checked_sub(1),
            UndoPlan::Selective { index } => Some(index),
        };
        let target = index
            .and_then(|i| self.history.entries().get(i))
            .cloned()
            .ok_or(Rejection::NothingToDo { what: "undo" })?;
        let edits = target.inverse();
        let mut warnings = lock::check(&self.doc, self.registry, &edits, &who, &[], &self.history, Mode::Main)?;
        let before = self.score(&self.doc);
        apply_edits(&mut self.doc, self.registry, &edits)?;
        let plausibility = self.scored(before, &mut warnings);
        let label = format!("undo {}", target.label);
        let mut outcome = match plan {
            UndoPlan::Top => {
                if let Err(e) = self.append(&Record::Undo { id: target.id, edits: edits.clone() }) {
                    self.revert(&edits);
                    return Err(CommandError::Storage(e));
                }
                self.history.step_back();
                let mut outcome = self.outcome(&target, &edits, None);
                outcome.author = who.id;
                "undo".clone_into(&mut outcome.kind);
                outcome.label = label;
                outcome.time_ms = now;
                outcome.elements = 0;
                outcome
            }
            UndoPlan::Selective { .. } => {
                let entry = Entry {
                    id: self.next_id(),
                    author: who.id,
                    kind: "undo".to_owned(),
                    label,
                    time_ms: now,
                    last_ms: now,
                    edits,
                    reverts: Some(target.id),
                    open: None,
                    elements: 0,
                };
                if let Err(e) = self.append(&Record::Commit(entry.clone())) {
                    self.revert(&entry.edits);
                    return Err(CommandError::Storage(e));
                }
                let outcome = self.outcome(&entry, &entry.edits, None);
                self.history.push(entry);
                outcome
            }
        };
        outcome.warnings = warnings;
        outcome.plausibility = plausibility;
        self.events.push(match plan {
            UndoPlan::Top => EventKind::Undone(outcome.clone()),
            UndoPlan::Selective { .. } => EventKind::Committed(outcome.clone()),
        });
        self.touch();
        Ok(outcome)
    }

    /// Redoes the next undone entry `[CMD-04]`; with [`Scope::Author`] only when it is that
    /// author's. Validated like a command by `author`.
    pub fn redo(&mut self, author: &str, scope: &Scope) -> Result<Outcome, CommandError> {
        let who = self.author(author)?;
        self.seal_open()?;
        let entry = self.history.plan_redo(scope)?.clone();
        let mut warnings = lock::check(&self.doc, self.registry, &entry.edits, &who, &[], &self.history, Mode::Main)?;
        let before = self.score(&self.doc);
        apply_edits(&mut self.doc, self.registry, &entry.edits)?;
        let plausibility = self.scored(before, &mut warnings);
        if let Err(e) = self.append(&Record::Redo(entry.clone())) {
            self.revert(&entry.edits);
            return Err(CommandError::Storage(e));
        }
        self.history.step_forward();
        let mut outcome = self.outcome(&entry, &entry.edits, None);
        outcome.author = who.id;
        "redo".clone_into(&mut outcome.kind);
        outcome.label = format!("redo {}", entry.label);
        outcome.time_ms = self.clock.unix_millis();
        outcome.warnings = warnings;
        outcome.plausibility = plausibility;
        self.events.push(EventKind::Redone(outcome.clone()));
        self.touch();
        Ok(outcome)
    }

    /// Reports that `author` now works on `region` / `landmark` `[NOTE-01]`. Returns the event's
    /// sequence number.
    pub fn select(&mut self, author: &str, region: Option<&str>, landmark: Option<&str>) -> Result<u64, CommandError> {
        self.author(author)?;
        self.seal_open()?;
        Ok(self.events.push(EventKind::SelectionChanged {
            author: author.to_owned(),
            region: region.map(str::to_owned),
            landmark: landmark.map(str::to_owned),
        }))
    }

    // ---- proposals ---------------------------------------------------------------------------

    fn proposal_index(&self, id: Id128) -> Result<usize, CommandError> {
        self.proposals.iter().position(|p| p.id == id).ok_or_else(|| Rejection::UnknownProposal { proposal: id }.into())
    }

    fn proposal_store(&self) -> Option<ProposalStore<'a>> {
        self.project.as_ref().map(|p| ProposalStore::new(self.fs, &join_path(p.root(), PROPOSALS_DIR)))
    }

    /// Opens a proposal by `author` on the current main document `[PROP-01]` and stores it
    /// `[PROP-04]`. Returns its id.
    pub fn open_proposal(&mut self, author: &str, name: &str) -> Result<Id128, CommandError> {
        self.author(author)?;
        self.seal_open()?;
        let id = self.next_id();
        let proposal = Proposal::new(id, author, name, self.clock.unix_millis(), self.doc.clone());
        if let Some(store) = self.proposal_store() {
            store.save(&proposal).map_err(storage_or_invalid)?;
        }
        self.proposals.push(proposal);
        self.events.push(EventKind::ProposalOpened { proposal: id, author: author.to_owned(), name: name.to_owned() });
        Ok(id)
    }

    /// Applies `cmd` inside proposal `proposal` `[PROP-01]`. Only the proposal's author adds
    /// commands; locks apply, human-edit protection only warns (the human reviews everything).
    /// Commands in a proposal do not coalesce.
    pub fn execute_in(&mut self, proposal: Id128, mut cmd: Command) -> Result<Outcome, CommandError> {
        self.prepare(&mut cmd);
        let author = self.author(&cmd.author)?;
        self.seal_open()?;
        let index = self.proposal_index(proposal)?;
        let store = self.proposal_store();
        let blobs = self.project.as_ref().map(|p| p.blobs(self.fs));
        let ctx = Context { registry: self.registry, blobs: blobs.as_ref() };
        let Some(p) = self.proposals.get_mut(index) else { return Err(Rejection::UnknownProposal { proposal }.into()) };
        if p.author != cmd.author {
            return Err(Rejection::NotProposalAuthor { proposal, author: p.author.clone() }.into());
        }
        let resolved = resolve(&p.doc, &cmd, &ctx)?;
        let mut warnings = resolved.warnings;
        warnings.extend(lock::check(
            &p.doc,
            self.registry,
            &resolved.edits,
            &author,
            &resolved.regions,
            &self.history,
            Mode::Proposal,
        )?);
        // Locks set on the main document after the proposal was opened apply too, and so does
        // protection of what a human changed there since: the branch document does not see that
        // edit, so its `needs_review` warning comes from this check (once per target).
        for w in lock::check(
            &self.doc,
            self.registry,
            &resolved.edits,
            &author,
            &resolved.regions,
            &self.history,
            Mode::Proposal,
        )? {
            if !warnings.iter().any(|x| x.code == w.code && x.target == w.target) {
                warnings.push(w);
            }
        }
        let before = self.scorer.map(|s| s.score(&p.doc, self.registry));
        apply_edits(&mut p.doc, self.registry, &resolved.edits)?;
        let after = self.scorer.map(|s| s.score(&p.doc, self.registry));
        let entry = Entry {
            id: cmd.id,
            author: cmd.author.clone(),
            kind: cmd.action.kind_name().to_owned(),
            label: cmd.label.clone(),
            time_ms: cmd.time_ms,
            last_ms: cmd.time_ms,
            edits: resolved.edits,
            reverts: None,
            open: None,
            elements: resolved.elements,
        };
        p.entries.push(entry);
        if let Some(store) = store
            && let Err(e) = store.save(p)
        {
            if let Some(entry) = p.entries.pop() {
                let _ = apply_edits(&mut p.doc, self.registry, &entry.inverse());
            }
            return Err(storage_or_invalid(e));
        }
        let Some(entry) = self.proposals.get(index).and_then(|p| p.entries.last()) else {
            return Err(Rejection::UnknownProposal { proposal }.into());
        };
        let mut outcome = self.outcome(entry, &entry.edits, Some(proposal));
        if let (Some(b), Some(a)) = (before, after) {
            if a < b - self.config.plausibility_drop {
                warnings.push(Warning {
                    code: WarningCode::PlausibilityDrop,
                    target: "document".to_owned(),
                    message: format!("plausibility dropped from {b:.3} to {a:.3}"),
                });
            }
            outcome.plausibility = Some([b, a]);
        }
        outcome.warnings = warnings;
        self.events.push(EventKind::Committed(outcome.clone()));
        Ok(outcome)
    }

    /// Takes back the last command of a proposal (its author only).
    pub fn undo_in(&mut self, proposal: Id128, author: &str) -> Result<Outcome, CommandError> {
        self.author(author)?;
        self.seal_open()?;
        let index = self.proposal_index(proposal)?;
        let store = self.proposal_store();
        let Some(p) = self.proposals.get_mut(index) else { return Err(Rejection::UnknownProposal { proposal }.into()) };
        if p.author != author {
            return Err(Rejection::NotProposalAuthor { proposal, author: p.author.clone() }.into());
        }
        let entry = p.entries.pop().ok_or(Rejection::NothingToDo { what: "undo" })?;
        let edits = entry.inverse();
        if let Err(e) = apply_edits(&mut p.doc, self.registry, &edits) {
            p.entries.push(entry);
            return Err(e.into());
        }
        if let Some(store) = store
            && let Err(e) = store.save(p)
        {
            let _ = apply_edits(&mut p.doc, self.registry, &entry.edits);
            p.entries.push(entry);
            return Err(storage_or_invalid(e));
        }
        let mut outcome = self.outcome(&entry, &edits, Some(proposal));
        "undo".clone_into(&mut outcome.kind);
        outcome.label = format!("undo {}", entry.label);
        outcome.time_ms = self.clock.unix_millis();
        self.events.push(EventKind::Undone(outcome.clone()));
        Ok(outcome)
    }

    /// Accepts proposal `proposal`, wholly (`selection` `None`) or the selected commands
    /// `[PROP-02]`, `[PROP-03]`. Only a human accepts. The selection must include every earlier
    /// command a selected one builds on, and none of the selected commands may change what the
    /// main document changed since the proposal was opened: either is refused with the commands
    /// and targets concerned. The accepted commands join the main history as they are, with the
    /// agent as author; locks apply, human-edit protection does not (the human decides). The
    /// proposal is closed; commands left out are dropped.
    pub fn accept(
        &mut self,
        proposal: Id128,
        author: &str,
        selection: Option<&[Id128]>,
    ) -> Result<Accepted, CommandError> {
        let who = self.author(author)?;
        if who.kind != AuthorKind::Human {
            return Err(Rejection::HumanOnly { what: "accept a proposal" }.into());
        }
        self.seal_open()?;
        let index = self.proposal_index(proposal)?;
        let Some(p) = self.proposals.get(index) else { return Err(Rejection::UnknownProposal { proposal }.into()) };
        let selected = p.select(selection).map_err(|e| match e {
            SelectError::NotInProposal(id) => CommandError::Invalid(DocError::field(
                "proposal",
                id.to_hex().as_str(),
                "not a command of this proposal",
            )),
            SelectError::Rejected(r) => CommandError::Rejected(r),
        })?;
        let conflicts = p.conflicts(&self.doc, &selected)?;
        if !conflicts.is_empty() {
            return Err(Rejection::ProposalConflict { conflicts }.into());
        }
        let entries: Vec<Entry> = selected.iter().filter_map(|&i| p.entries.get(i).cloned()).collect();
        let mut work = self.doc.clone();
        let mut warnings = Vec::new();
        let mut all_edits = Vec::new();
        for entry in &entries {
            let agent = work
                .author(&entry.author)
                .cloned()
                .ok_or_else(|| Rejection::UnknownAuthor { author: entry.author.clone() })?;
            warnings.extend(lock::check(&work, self.registry, &entry.edits, &agent, &[], &self.history, Mode::Accept)?);
            apply_edits(&mut work, self.registry, &entry.edits)?;
            all_edits.extend(entry.edits.iter().cloned());
        }
        let accepted: Vec<Id128> = entries.iter().map(|e| e.id).collect();
        let dropped: Vec<Id128> = p.entries.iter().map(|e| e.id).filter(|id| !accepted.contains(id)).collect();
        let label = format!("accept {}", p.name);
        let before = self.score(&self.doc);
        self.append(&Record::Accept { proposal, entries: entries.clone() }).map_err(CommandError::Storage)?;
        self.doc = work;
        for entry in entries {
            self.history.push(entry);
        }
        self.proposals.remove(index);
        self.closed.push(proposal);
        // A file that cannot be removed now is retried by every save and tick.
        let _ = self.remove_closed();
        let plausibility = self.scored(before, &mut warnings);
        let outcome = Outcome {
            command: proposal,
            author: who.id,
            kind: "accept_proposal".to_owned(),
            label,
            time_ms: self.clock.unix_millis(),
            proposal: Some(proposal),
            changes: describe(&all_edits, self.registry),
            warnings,
            elements: 0,
            plausibility,
            coalesced: false,
        };
        self.events.push(EventKind::ProposalAccepted {
            proposal,
            accepted: accepted.clone(),
            dropped: dropped.clone(),
        });
        self.touch();
        Ok(Accepted { outcome, accepted, dropped })
    }

    /// Discards proposal `proposal` `[PROP-02]`: a human, or the proposal's own author, may.
    pub fn discard(&mut self, proposal: Id128, author: &str) -> Result<(), CommandError> {
        let who = self.author(author)?;
        let index = self.proposal_index(proposal)?;
        let owner = self.proposals.get(index).is_some_and(|p| p.author == author);
        if who.kind != AuthorKind::Human && !owner {
            return Err(Rejection::HumanOnly { what: "discard another author's proposal" }.into());
        }
        self.seal_open()?;
        if let Some(store) = self.proposal_store() {
            store.remove(proposal).map_err(storage_or_invalid)?;
        }
        self.proposals.remove(index);
        self.events.push(EventKind::ProposalDiscarded { proposal });
        Ok(())
    }

    // ---- saving ------------------------------------------------------------------------------

    /// Saves the document atomically and starts a fresh journal `[SAVE-02]`. Returns the hash of
    /// the written text.
    pub fn save(&mut self) -> Result<Hash128, CommandError> {
        let project = self.project.clone().ok_or_else(no_project)?;
        self.seal_open()?;
        self.remove_closed().map_err(CommandError::Storage)?;
        match project.save(self.fs, &self.doc, self.registry) {
            Ok(hash) => {
                self.link = JournalLink::default();
                self.pending = None;
                self.since_autosave = 0;
                self.events.push(EventKind::Saved { hash });
                Ok(hash)
            }
            Err(e) => {
                // When the text did reach the disk, the old journal no longer continues it: start
                // a fresh one before the next record.
                if let (Ok(text), Ok(disk)) = (self.doc.write(), self.fs.read(&project.document_path()))
                    && disk == text.as_bytes()
                {
                    self.link.restart = Some(hash128(&disk));
                }
                Err(storage_or_invalid(e))
            }
        }
    }

    /// Writes an autosave now and rotates the old ones `[SAVE-03]`. Returns its path.
    pub fn autosave(&mut self) -> Result<String, CommandError> {
        let project = self.project.clone().ok_or_else(no_project)?;
        self.seal_open()?;
        let path = Autosaves::new(self.fs, &project)
            .write(&project, &self.doc, self.registry, self.config.autosave_keep)
            .map_err(storage_or_invalid)?;
        self.since_autosave = 0;
        Ok(path)
    }

    /// The host calls this regularly (every frame or on a timer): it ends gestures that paused
    /// longer than the coalescing threshold `[CMD-03]`, journals batched gesture frames
    /// `[SAVE-01]`, retries removing closed proposal files, and autosaves when enough changes
    /// piled up or the document was idle long enough `[SAVE-03]`. Returns the autosave's path
    /// when one was written.
    pub fn tick(&mut self) -> Result<Option<String>, CommandError> {
        let now = self.mono_ms();
        if self.history.open().is_some() && now.saturating_sub(self.gesture_ms) > self.config.coalesce_pause_ms {
            self.seal_open()?;
        }
        if self.pending.as_ref().is_some_and(|p| now.saturating_sub(p.since_ms) >= self.config.journal_interval_ms) {
            self.flush_pending().map_err(CommandError::Storage)?;
        }
        self.remove_closed().map_err(CommandError::Storage)?;
        let every = self.config.autosave_every > 0 && self.since_autosave >= self.config.autosave_every;
        let idle =
            self.config.autosave_idle_ms > 0 && now.saturating_sub(self.last_change_ms) >= self.config.autosave_idle_ms;
        if self.project.is_some() && self.history.open().is_none() && self.since_autosave > 0 && (every || idle) {
            return self.autosave().map(Some);
        }
        Ok(None)
    }
}

mod drag;

#[cfg(test)]
mod tests;
