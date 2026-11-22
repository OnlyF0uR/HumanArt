//! The frames of a scalar slider drag without allocation `[CMD-03]` `[PERF-R1]`.
//!
//! [`Session::execute`] is a full transaction: it resolves the command to text edits, checks the
//! locks, describes the outcome and records the history entry, and every step allocates. A drag
//! sends one command per frame, and once its gesture is open every further frame only moves one
//! stored number. [`Session::drag_scalar`] recognises that case and does exactly what `execute`
//! would do to the document, the history, the batched journal frames, the id counter and the
//! clocks, in place; any other case goes through `execute`. A property test in the session tests
//! checks that both give the same document, history and journal.

use humanart_core::id::Id128;
use humanart_io::json::Value;

use crate::command::{Action, Command, CommandError, Outcome, RangePolicy};
use crate::history::{compose, retarget};
use crate::lock;
use crate::registry::{ParamKind, ParamValue, value_to_json};
use crate::text::{Edit, SectionRef};

use super::{Pending, Session};

/// The section of parameter values.
const PARAMS: &str = "params";

impl Session<'_> {
    /// One frame of a slider drag by `author` on scalar parameter `id`: the same as executing
    /// `set_params {id: value}` with [`RangePolicy::Clamp`] as part of a gesture
    /// ([`Command::in_gesture`]). A frame that only moves the open gesture of this author on this
    /// parameter (the steady state of a drag) is folded in place without allocating and returns
    /// `None`: its warnings (clamped, outside the plausible span, the
    /// [`crate::plausibility::RegistryPrior`] score) are not reported, and a session with an
    /// installed plausibility scorer ([`Session::with_scorer`]) always takes the full path. Every other frame (the
    /// first, one that lands on the default, a refused one) goes through [`Session::execute`]
    /// and returns its outcome or error. End the gesture with [`Session::end_gesture`].
    pub fn drag_scalar(&mut self, author: &str, id: &str, value: f64) -> Result<Option<Outcome>, CommandError> {
        if self.fold_scalar(author, id, value) {
            return Ok(None);
        }
        let cmd = Command::new(
            author,
            Action::SetParams {
                values: vec![(id.to_owned(), ParamValue::Scalar(value))],
                relative: false,
                policy: RangePolicy::Clamp,
            },
        )
        .in_gesture();
        self.execute(cmd).map(Some)
    }

    /// Folds the frame into the open gesture when that is all `execute` would do; `false` (and
    /// nothing changed) otherwise.
    fn fold_scalar(&mut self, author: &str, id: &str, value: f64) -> bool {
        let Some(v) = self.foldable(author, id, value) else { return false };
        // What `execute` does from here on for a frame that extends the open gesture.
        self.next_id();
        let time_ms = self.clock.unix_millis();
        let now = self.mono_ms();
        let after = value_to_json(&ParamValue::Scalar(v));
        let changed = match self.doc.params.get_mut(id) {
            Some(slot) if *slot != ParamValue::Scalar(v) => {
                let before = value_to_json(slot);
                *slot = ParamValue::Scalar(v);
                Some(before)
            }
            _ => None,
        };
        if let Some(top) = self.history.open_mut() {
            if let Some(before) = &changed {
                fold(&mut top.edits, id, before, &after);
            }
            top.last_ms = time_ms;
        }
        self.gesture_ms = now;
        match &mut self.pending {
            Some(p) => {
                if let Some(before) = &changed {
                    fold(&mut p.edits, id, before, &after);
                }
                p.last_ms = time_ms;
            }
            // Without a project nothing is journaled: no batch to start.
            None if self.project.is_none() => {}
            None => {
                let open_id = self.history.open().map_or(Id128::NIL, |e| e.id);
                let mut edits = std::mem::take(&mut self.spare_edits);
                match &changed {
                    Some(before) => reuse_set(&mut edits, id, before, &after),
                    None => edits.clear(),
                }
                self.pending = Some(Pending { id: open_id, edits, since_ms: now, last_ms: time_ms, elements: 0 });
            }
        }
        if self.pending.as_ref().is_some_and(|p| now.saturating_sub(p.since_ms) >= self.config.journal_interval_ms) {
            // As in `execute`: a failed flush keeps the frames pending; the gesture's end retries
            // it and reports the error.
            let _ = self.flush_pending();
        }
        self.touch();
        true
    }

    /// The value the frame stores, when the frame can be folded in place: the open top entry is
    /// this author's `set_params` gesture on `id` alone and holds only sets, no plausibility
    /// scorer is installed, the parameter is an editable scalar already stored with this author, the
    /// (clamped) value is not its default, and the locks allow the change.
    fn foldable(&self, author: &str, id: &str, value: f64) -> Option<f64> {
        if self.scorer.is_some() || !value.is_finite() {
            return None;
        }
        let open = self.history.open()?;
        let key = open.open.as_deref()?;
        // The key `execute` gives the gesture: "{author} {kind} {targets}".
        if key.strip_prefix(author)?.strip_prefix(" set_params ")? != id
            || !open.edits.iter().all(|e| matches!(e, Edit::Set { .. }))
        {
            return None;
        }
        let param = self.registry.get(id)?;
        if param.kind != ParamKind::Scalar || param.read_only {
            return None;
        }
        let v = param.range.map_or(value, |r| r.clamp(value));
        if param.range.is_some_and(|r| !r.contains(v)) || param.is_default(&ParamValue::Scalar(v)) {
            return None;
        }
        let who = self.doc.author(author)?;
        if !matches!(self.doc.params.get(id), Some(ParamValue::Scalar(_)))
            || self.doc.param_authors.get(id).map(String::as_str) != Some(author)
            || lock::check_param(&self.doc, self.registry, id, who).is_err()
        {
            return None;
        }
        Some(v)
    }
}

/// The set of `params.id` from `before` to `after` (the edit `execute` resolves the frame to).
fn set_edit(id: &str, before: &Value, after: &Value) -> Edit {
    Edit::Set {
        section: SectionRef::new(PARAMS, None),
        key: id.to_owned(),
        before: Some(before.clone()),
        after: Some(after.clone()),
    }
}

/// Makes `edits` (the last batch's, already written) the single set of `params.id` from `before`
/// to `after`, rewriting its values in place when it already is a set of that entry (the next
/// batch of the same drag), so the strings it holds are not built again.
fn reuse_set(edits: &mut Vec<Edit>, id: &str, before: &Value, after: &Value) {
    if let [Edit::Set { section, key, before: b, after: a }] = edits.as_mut_slice()
        && section.kind == PARAMS
        && section.key.is_none()
        && key == id
    {
        *b = Some(before.clone());
        *a = Some(after.clone());
        return;
    }
    edits.clear();
    edits.push(set_edit(id, before, after));
}

/// [`compose`]s the frame's set into `edits`, in place when the list already sets the entry and
/// keeps doing so.
fn fold(edits: &mut Vec<Edit>, id: &str, before: &Value, after: &Value) {
    if !retarget(edits, PARAMS, id, after) {
        // `edits` holds only sets (checked for the history entry; the batched journal frames are
        // composed from the same frames), so composing cannot be refused.
        let _ = compose(edits, &[set_edit(id, before, after)]);
    }
}
