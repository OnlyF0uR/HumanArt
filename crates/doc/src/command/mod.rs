//! Commands `[CMD-01]`..`[CMD-06]`: every change of a document, whoever makes it.
//!
//! A [`Command`] is plain data: an id, the author's id, a timestamp, a label and an [`Action`]
//! (what to do, in the terms of the editor and the agent tools). The UI, scripts and the agent
//! server build the same commands; [`codec`] reads and writes them as JSON, which is how the agent
//! server (WP-23) sends them and how the journal stores what they did.
//!
//! Applying a command has three steps, all in this module and [`crate::lock`]:
//!
//! 1. [`resolve`] turns the action into text [`Edit`]s against the current document (the forward
//!    payload). Every edit records the value before and after, so the inverse payload is
//!    [`crate::text::invert`] of the forward one and is exact `[CMD-01]`. Range policy is applied
//!    here: values outside a parameter's range are rejected or clamped, and either way the result
//!    says so `[CMD-05]`.
//! 2. [`crate::lock::check`] validates the edits against locks, pins and the human-protection
//!    policy `[LOCK-01]`, `[LOCK-02]`, `[CMD-05]`.
//! 3. [`apply_edits`] applies them, all or nothing. Edits that only touch parameter values, their
//!    authors and the preview pose (slider drags) take a fast path that changes the typed document
//!    in place; everything else goes through [`Document::apply_edits`].
//!
//! [`describe`] turns applied edits into the structured [`Outcome`] the history panel shows and the
//! agent reads `[CMD-06]`.
//!
//! The command kinds of `[CMD-02]` map to actions as follows: set parameter(s), eye and mouth
//! settings and expression weights are [`Action::SetParams`] (eyes, mouth and expressions are
//! registry parameters, `[DOC-05]`..`[DOC-09]`) or [`Action::ApplyExpressionPreset`]; the pose is
//! [`Action::SetPose`] / [`Action::ApplyPosePreset`]; create / modify / delete / reorder of sculpt
//! layers, skin layers (procedural settings are the layer's `source`), grooms, attachments
//! (attach / detach) and references (import / replace) are [`Action::AddObject`],
//! [`Action::ModifyObject`], [`Action::DeleteObject`] and [`Action::ReorderObjects`]; sculpt
//! strokes, paint strokes and groom guide edits are [`Action::Stroke`] (the tool stores the touched
//! data as a blob first); locks and pins are [`Action::SetLock`], [`Action::SetObjectLock`],
//! [`Action::PinLandmark`], [`Action::UnpinLandmark`]; snapshots are [`Action::CreateSnapshot`] and
//! [`Action::RestoreSnapshot`]; [`Action::Batch`] is the atomic group.

mod apply;
pub mod codec;
mod outcome;
mod resolve;
mod target;

use std::fmt;

use humanart_core::hash::{Hash128, hash128};
use humanart_core::id::Id128;
use humanart_io::json::Object;

use crate::document::{Author, LockTarget};
use crate::error::DocError;
use crate::registry::ParamValue;
use crate::text::Edit;

pub use apply::apply_edits;
pub use outcome::{Change, ObjectOp, Outcome, Warning, WarningCode, describe};
pub use resolve::{Context, Resolved, resolve};
pub use target::{Target, any_overlap, targets};

/// The kinds of document objects the generic object actions work on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ObjectKind {
    /// `[DOC-03]` sculpt layers.
    SculptLayer,
    /// `[DOC-04]` skin layers.
    SkinLayer,
    /// `[DOC-08]` grooms.
    Groom,
    /// `[DOC-09]` expression presets.
    ExpressionPreset,
    /// `[DOC-09]` pose presets.
    PosePreset,
    /// `[DOC-12]` attachments.
    Attachment,
    /// `[DOC-13]` references.
    Reference,
}

impl ObjectKind {
    /// Every kind with its section kind in the document text.
    pub const ALL: [(Self, &'static str); 7] = [
        (Self::SculptLayer, "sculpt_layer"),
        (Self::SkinLayer, "skin_layer"),
        (Self::Groom, "groom"),
        (Self::ExpressionPreset, "expression_preset"),
        (Self::PosePreset, "pose_preset"),
        (Self::Attachment, "attachment"),
        (Self::Reference, "reference"),
    ];

    /// The section kind in the document text.
    pub fn section(self) -> &'static str {
        Self::ALL.iter().find(|(k, _)| *k == self).map_or("", |(_, s)| s)
    }

    /// The kind whose section kind is `section`.
    pub fn from_section(section: &str) -> Option<Self> {
        Self::ALL.iter().find(|(_, s)| *s == section).map(|(k, _)| *k)
    }

    /// `true` for the kinds with a `locked` flag (layer locks, `[LOCK-01]`).
    pub const fn lockable(self) -> bool {
        matches!(self, Self::SculptLayer | Self::SkinLayer | Self::Groom)
    }

    fn index(self) -> usize {
        Self::ALL.iter().position(|(k, _)| *k == self).unwrap_or(0)
    }
}

/// What a stroke changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StrokeKind {
    /// A sculpt stroke: the sculpt layer's delta payload.
    Sculpt,
    /// A paint stroke: the painted skin layer's texture.
    Paint,
    /// A groom guide edit: the groom's guide curves.
    Guides,
}

impl StrokeKind {
    /// Every kind with its name in JSON.
    pub const ALL: [(Self, &'static str); 3] =
        [(Self::Sculpt, "sculpt"), (Self::Paint, "paint"), (Self::Guides, "guides")];

    /// The name in JSON.
    pub fn name(self) -> &'static str {
        Self::ALL.iter().find(|(k, _)| *k == self).map_or("", |(_, s)| s)
    }
}

/// What to do when a parameter value is outside its hard range `[CMD-05]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum RangePolicy {
    /// Refuse the command.
    #[default]
    Reject,
    /// Clamp the value and attach a [`WarningCode::Clamped`] warning.
    Clamp,
}

/// What a command does. Every variant is data that [`codec`] reads and writes.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// Sets parameters. With `relative`, numeric values are added to the current ones.
    SetParams {
        /// Parameter id and value, each id at most once.
        values: Vec<(String, ParamValue)>,
        /// Add to the current values instead of replacing them.
        relative: bool,
        /// What to do with values outside the range.
        policy: RangePolicy,
    },
    /// Puts parameters back to their registry defaults.
    ResetParams {
        /// The parameter ids.
        ids: Vec<String>,
    },
    /// Sets joints of the preview pose; `None` puts a joint back to rest.
    SetPose {
        /// Joint name and unit quaternion `(x, y, z, w)`.
        joints: Vec<(String, Option<[f64; 4]>)>,
    },
    /// Sets every expression parameter to the preset's weight (absent: default).
    ApplyExpressionPreset {
        /// The preset's id.
        preset: Id128,
    },
    /// Replaces the preview pose with a preset's.
    ApplyPosePreset {
        /// The preset's id.
        preset: Id128,
    },
    /// Creates an object from its text fields (`docs/format-anthropos.md`). Fields with a usual
    /// value may be left out; `author` (and a sculpt layer's `created_by_command`) are set from
    /// the command.
    AddObject {
        /// Which list.
        kind: ObjectKind,
        /// The new object's id; [`Id128::NIL`] draws one from the command id.
        id: Id128,
        /// Its position in the list (bottom first); `None` puts it on top.
        index: Option<usize>,
        /// The fields.
        fields: Object,
    },
    /// Changes fields of an object. `author`, `created_by_command` and `locked` cannot be changed
    /// this way (locks have [`Action::SetObjectLock`]).
    ModifyObject {
        /// Which list.
        kind: ObjectKind,
        /// The object's id.
        id: Id128,
        /// The fields to change and their new values.
        fields: Object,
    },
    /// Deletes an object.
    DeleteObject {
        /// Which list.
        kind: ObjectKind,
        /// The object's id.
        id: Id128,
    },
    /// Reorders a list: `order` names every object of the list once, bottom first.
    ReorderObjects {
        /// Which list.
        kind: ObjectKind,
        /// The ids in their new order.
        order: Vec<Id128>,
    },
    /// A sculpt stroke, paint stroke or guide edit. The tool has already stored the layer's new
    /// payload (or texture, or guides) as a blob holding only what changed in its delta encoding.
    Stroke {
        /// What the stroke changes.
        stroke: StrokeKind,
        /// The layer or groom.
        target: Id128,
        /// The new blob ([`Hash128::ZERO`]: empty).
        blob: Hash128,
        /// The template regions the stroke touched (checked against region locks).
        regions: Vec<String>,
        /// How many vertices, texels or guide points it touched (reported in the outcome).
        elements: u64,
    },
    /// Sets or removes a region or parameter lock `[LOCK-01]`.
    SetLock {
        /// What the lock protects.
        target: LockTarget,
        /// Lock (`true`) or unlock.
        locked: bool,
    },
    /// Locks or unlocks a layer or groom `[LOCK-01]`.
    SetObjectLock {
        /// Which list (a lockable kind).
        kind: ObjectKind,
        /// The object's id.
        id: Id128,
        /// Lock or unlock.
        locked: bool,
    },
    /// Pins a landmark at a rest-pose position, or moves an existing pin `[LOCK-03]`.
    PinLandmark {
        /// The landmark name.
        landmark: String,
        /// The position it keeps, metres.
        position: [f64; 3],
    },
    /// Removes a landmark pin.
    UnpinLandmark {
        /// The landmark name.
        landmark: String,
    },
    /// Takes a snapshot `[DOC-11]`. With `state` [`Hash128::ZERO`] the snapshot text is computed
    /// and stored in the blob store of the [`Context`].
    CreateSnapshot {
        /// The snapshot's id; [`Id128::NIL`] draws one from the command id.
        id: Id128,
        /// Its name.
        name: String,
        /// The blob holding the snapshot text, or [`Hash128::ZERO`].
        state: Hash128,
    },
    /// Restores the character state of a snapshot (metadata, settings, authors and the snapshot
    /// list stay as they are).
    RestoreSnapshot {
        /// The snapshot's id.
        id: Id128,
    },
    /// Adds an author to the document. A command by an author the document does not know yet may
    /// only add that author.
    AddAuthor {
        /// The author.
        author: Author,
    },
    /// Changes project policies (humans only).
    SetSettings {
        /// `[LOCK-02]` protect values and layers last changed by a human from agents.
        protect_human_edits: bool,
    },
    /// Several actions applied together, all or nothing.
    Batch {
        /// The actions, in order.
        actions: Vec<Action>,
    },
    /// Raw text edits (scripts, merges). Validated like every other command.
    Edits {
        /// The edits.
        edits: Vec<Edit>,
    },
}

const ADD_NAMES: [&str; 7] = [
    "add_sculpt_layer",
    "add_skin_layer",
    "add_groom",
    "add_expression_preset",
    "add_pose_preset",
    "attach",
    "import_reference",
];
const MODIFY_NAMES: [&str; 7] = [
    "modify_sculpt_layer",
    "modify_skin_layer",
    "modify_groom",
    "modify_expression_preset",
    "modify_pose_preset",
    "modify_attachment",
    "replace_reference",
];
const DELETE_NAMES: [&str; 7] = [
    "delete_sculpt_layer",
    "delete_skin_layer",
    "delete_groom",
    "delete_expression_preset",
    "delete_pose_preset",
    "detach",
    "remove_reference",
];
const REORDER_NAMES: [&str; 7] = [
    "reorder_sculpt_layers",
    "reorder_skin_layers",
    "reorder_grooms",
    "reorder_expression_presets",
    "reorder_pose_presets",
    "reorder_attachments",
    "reorder_references",
];

impl Action {
    /// The command kind's stable name, used in the history, events and coalescing keys.
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::SetParams { .. } => "set_params",
            Self::ResetParams { .. } => "reset_params",
            Self::SetPose { .. } => "set_pose",
            Self::ApplyExpressionPreset { .. } => "apply_expression_preset",
            Self::ApplyPosePreset { .. } => "apply_pose_preset",
            Self::AddObject { kind, .. } => ADD_NAMES[kind.index()],
            Self::ModifyObject { kind, .. } => MODIFY_NAMES[kind.index()],
            Self::DeleteObject { kind, .. } => DELETE_NAMES[kind.index()],
            Self::ReorderObjects { kind, .. } => REORDER_NAMES[kind.index()],
            Self::Stroke { stroke: StrokeKind::Sculpt, .. } => "sculpt_stroke",
            Self::Stroke { stroke: StrokeKind::Paint, .. } => "paint_stroke",
            Self::Stroke { stroke: StrokeKind::Guides, .. } => "groom_guide_edit",
            Self::SetLock { .. } => "set_lock",
            Self::SetObjectLock { .. } => "set_object_lock",
            Self::PinLandmark { .. } => "pin_landmark",
            Self::UnpinLandmark { .. } => "unpin_landmark",
            Self::CreateSnapshot { .. } => "create_snapshot",
            Self::RestoreSnapshot { .. } => "restore_snapshot",
            Self::AddAuthor { .. } => "add_author",
            Self::SetSettings { .. } => "set_settings",
            Self::Batch { .. } => "batch",
            Self::Edits { .. } => "edits",
        }
    }

    /// The target part of the coalescing key `[CMD-03]`, for the actions a continuous gesture
    /// produces (slider drags, brush strokes, dragging a layer property); `None` for the others.
    pub fn coalesce_target(&self) -> Option<String> {
        match self {
            Self::SetParams { values, .. } => Some(join(values.iter().map(|(id, _)| id.as_str()))),
            Self::ResetParams { ids } => Some(join(ids.iter().map(String::as_str))),
            Self::SetPose { joints } => Some(join(joints.iter().map(|(j, _)| j.as_str()))),
            Self::ModifyObject { kind, id, fields } => {
                Some(format!("{} {id} {}", kind.section(), join(fields.entries().iter().map(|(k, _)| k.as_str()))))
            }
            Self::Stroke { target, .. } => Some(target.to_string()),
            _ => None,
        }
    }
}

fn join<'a>(items: impl Iterator<Item = &'a str>) -> String {
    let mut sorted: Vec<&str> = items.collect();
    sorted.sort_unstable();
    sorted.join(",")
}

/// One command `[CMD-01]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Command {
    /// Its id ([`Id128::NIL`]: the session draws one).
    pub id: Id128,
    /// The author's id (an [`Author`] of the document).
    pub author: String,
    /// When it was made, Unix milliseconds (0: the session sets it from its clock).
    pub time_ms: u64,
    /// The text the history shows (empty: the session uses the kind name).
    pub label: String,
    /// Part of a continuous gesture (slider drag, brush stroke): merged with the previous command
    /// of the same author, kind and target into one undo step `[CMD-03]`.
    pub gesture: bool,
    /// What to do.
    pub action: Action,
}

impl Command {
    /// A command by `author` with no id, time or label yet (the session fills them in).
    pub fn new(author: &str, action: Action) -> Self {
        Self { id: Id128::NIL, author: author.to_owned(), time_ms: 0, label: String::new(), gesture: false, action }
    }

    /// The same command with a label.
    pub fn labelled(mut self, label: &str) -> Self {
        label.clone_into(&mut self.label);
        self
    }

    /// The same command marked as part of a continuous gesture.
    pub fn in_gesture(mut self) -> Self {
        self.gesture = true;
        self
    }
}

/// An id derived from a command id and a counter: the ids of objects a command creates when the
/// caller gave none. Never [`Id128::NIL`].
pub fn derive_id(command: Id128, n: u64) -> Id128 {
    let mut bytes = [0_u8; 24];
    bytes[..16].copy_from_slice(&command.to_le_bytes());
    bytes[16..].copy_from_slice(&n.to_le_bytes());
    let id = Id128::from_u128(hash128(&bytes).to_u128());
    if id.is_nil() { Id128::from_u128(1) } else { id }
}

/// Why a command was refused `[CMD-05]`, `[LOCK-01]`, `[LOCK-02]`, `[CMD-04]`.
#[derive(Clone, Debug, PartialEq)]
pub enum Rejection {
    /// The author is not in the document.
    UnknownAuthor {
        /// The author id.
        author: String,
    },
    /// A parameter lock pins the parameter.
    ParameterLocked {
        /// The parameter id.
        param: String,
        /// Who set the lock.
        owner: String,
    },
    /// A region lock protects the region.
    RegionLocked {
        /// The region.
        region: String,
        /// Who set the lock.
        owner: String,
    },
    /// The layer or groom is locked.
    ObjectLocked {
        /// Its section kind.
        kind: String,
        /// Its id.
        id: String,
    },
    /// `[LOCK-02]` the target was last changed by a human and agents may not change it.
    HumanProtected {
        /// What is protected (a parameter id, `sculpt_layer <id>`, a lock).
        target: String,
        /// The human who changed it last.
        human: String,
    },
    /// Only a human may do this.
    HumanOnly {
        /// What.
        what: &'static str,
    },
    /// A value outside the parameter's range under [`RangePolicy::Reject`].
    OutOfRange {
        /// The parameter id.
        param: String,
        /// The problem.
        what: String,
    },
    /// `[CMD-04]` a selective undo would revert a command that later commands built on.
    UndoConflict {
        /// The command to undo.
        command: Id128,
        /// The later commands that touch the same targets.
        later: Vec<Id128>,
        /// The targets they share.
        targets: Vec<String>,
    },
    /// Nothing to undo or redo (for this author).
    NothingToDo {
        /// `"undo"` or `"redo"`.
        what: &'static str,
    },
    /// The next command to redo is another author's.
    NotYours {
        /// The command.
        command: Id128,
        /// Its author.
        author: String,
    },
    /// There is no open proposal with this id.
    UnknownProposal {
        /// The proposal id.
        proposal: Id128,
    },
    /// Only the proposal's author may add commands to it or undo them.
    NotProposalAuthor {
        /// The proposal id.
        proposal: Id128,
        /// Its author.
        author: String,
    },
    /// `[PROP-02]` a selected command builds on a proposal command that was not selected.
    ProposalDependency {
        /// The selected command.
        command: Id128,
        /// The earlier, unselected commands on the same targets it needs.
        needs: Vec<Id128>,
    },
    /// `[PROP-03]` the main document changed targets the proposal also changes since it was opened.
    ProposalConflict {
        /// Every conflicting proposal command with the targets it shares with the main changes.
        conflicts: Vec<(Id128, Vec<String>)>,
    },
}

impl Rejection {
    /// A stable code for the agent and the UI.
    pub fn code(&self) -> &'static str {
        match self {
            Self::UnknownAuthor { .. } => "unknown_author",
            Self::ParameterLocked { .. } => "parameter_locked",
            Self::RegionLocked { .. } => "region_locked",
            Self::ObjectLocked { .. } => "object_locked",
            Self::HumanProtected { .. } => "human_protected",
            Self::HumanOnly { .. } => "human_only",
            Self::OutOfRange { .. } => "out_of_range",
            Self::UndoConflict { .. } => "undo_conflict",
            Self::NothingToDo { .. } => "nothing_to_do",
            Self::NotYours { .. } => "not_yours",
            Self::UnknownProposal { .. } => "unknown_proposal",
            Self::NotProposalAuthor { .. } => "not_proposal_author",
            Self::ProposalDependency { .. } => "proposal_dependency",
            Self::ProposalConflict { .. } => "proposal_conflict",
        }
    }
}

impl fmt::Display for Rejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownAuthor { author } => write!(f, "unknown author {author}: add the author first"),
            Self::ParameterLocked { param, owner } => {
                write!(f, "parameter {param} is locked by {owner}: unlock it first")
            }
            Self::RegionLocked { region, owner } => write!(f, "region {region} is locked by {owner}: unlock it first"),
            Self::ObjectLocked { kind, id } => write!(f, "{kind} {id} is locked: unlock it first"),
            Self::HumanProtected { target, human } => write!(
                f,
                "{target} was last changed by the human {human} and is protected from agent changes \
                 (human-edit protection): ask the human, request an unlock, or put the change in a proposal"
            ),
            Self::HumanOnly { what } => write!(f, "only a human may {what}"),
            Self::OutOfRange { param, what } => write!(f, "parameter {param}: {what}"),
            Self::UndoConflict { command, later, targets } => write!(
                f,
                "command {command} cannot be undone on its own: {} later command(s) changed the same targets ({})",
                later.len(),
                targets.join(", ")
            ),
            Self::NothingToDo { what } => write!(f, "nothing to {what}"),
            Self::NotYours { command, author } => write!(f, "the next command to redo ({command}) is by {author}"),
            Self::UnknownProposal { proposal } => write!(f, "there is no open proposal {proposal}"),
            Self::NotProposalAuthor { proposal, author } => {
                write!(f, "proposal {proposal} belongs to {author}: only its author may change it")
            }
            Self::ProposalDependency { command, needs } => write!(
                f,
                "command {command} builds on {} unselected command(s) of the proposal on the same targets; select \
                 them too",
                needs.len()
            ),
            Self::ProposalConflict { conflicts } => {
                write!(f, "the main document changed what {} proposal command(s) change:", conflicts.len())?;
                for (command, targets) in conflicts {
                    write!(f, " {command} ({})", targets.join(", "))?;
                }
                write!(f, "; leave them out of the selection or discard the proposal")
            }
        }
    }
}

/// The error of applying a command.
#[derive(Debug)]
pub enum CommandError {
    /// Validation refused the command; nothing changed.
    Rejected(Rejection),
    /// The command does not fit the document (unknown parameter, bad field, stale edit); nothing
    /// changed.
    Invalid(DocError),
    /// Storing failed (journal, blob store, proposal file). The document is unchanged unless the
    /// operation says otherwise.
    Storage(DocError),
}

impl CommandError {
    /// A stable code for the agent and the UI.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Rejected(r) => r.code(),
            Self::Invalid(_) => "invalid",
            Self::Storage(_) => "storage",
        }
    }
}

impl fmt::Display for CommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rejected(r) => write!(f, "rejected: {r}"),
            Self::Invalid(e) => write!(f, "invalid command: {e}"),
            Self::Storage(e) => write!(f, "could not store the change: {e}"),
        }
    }
}

impl std::error::Error for CommandError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Rejected(_) => None,
            Self::Invalid(e) | Self::Storage(e) => Some(e),
        }
    }
}

impl From<Rejection> for CommandError {
    fn from(r: Rejection) -> Self {
        Self::Rejected(r)
    }
}

impl From<DocError> for CommandError {
    fn from(e: DocError) -> Self {
        Self::Invalid(e)
    }
}

#[cfg(test)]
mod tests;
