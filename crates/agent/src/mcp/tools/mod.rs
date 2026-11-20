//! The agent's tools (section 10.3 observation, 10.4 mutation) and their table `[AG-02]`.
//!
//! Each tool is a [`ToolSpec`]: its name, a description, its arguments as schema [`Field`]s (the
//! published `inputSchema` and the validation of every call come from them, [`super::schema`]),
//! whether it reads or changes the document, and its handler. Handlers get a [`ToolCtx`] (the
//! engine, the sandbox, the render service and the server's own state) and validated arguments,
//! and return a [`ToolOutput`] (structured JSON plus PNG images) or a [`ToolError`].
//!
//! Every mutation builds a document [`Command`] by the agent author and runs it through the
//! session (validation, locks, human protection, history, journal, events). A refused command is
//! a [`ToolError`] whose `code` is the rejection's stable code (`human_protected`,
//! `region_locked`, `out_of_range` ...): a mutation never reports success when the command was
//! rejected `[CMD-06]`, `[LOCK-02]`. Without `proposal_id` a mutation edits the main document only
//! when the project's policy allows it `[SEC-03]`.
//!
//! Not implemented (their systems do not exist yet; nothing is stubbed): `compare_to_reference`
//! (WP-22), `quality_report` (WP-24 metrics), `paint_stroke` and the described sculpt operations
//! of `add_sculpt_layer` (WP-20 tools), `generate_skin_features` (OTH-02 generators),
//! `groom_apply_modifier` and `groom_preset` (the editor's groom commands and the template scalp
//! of WP-17), `sample_identity` (the ID-06 prior has no sampler and no data), `blend_snapshots`,
//! `set_reference_calibration` (WP-22) and `export_character` (WP-29).

pub mod mutate;
pub mod observe;

use humanart_core::id::Id128;
use humanart_core::platform::FileSystem;
use humanart_doc::command::{Command, CommandError};
use humanart_doc::document::Author;
use humanart_io::json::{Number, Object, Value};
use humanart_render::camera::Camera;
use humanart_render::light::BuiltinRig;
use humanart_render::renderer::Quality;

use super::render::{Image, RenderLimits, RenderService};
use super::schema::{Field, Ty, Vocab, record_schema};
use super::security::Sandbox;
use crate::engine::Engine;

/// What a tool does to the document.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    /// Reads only.
    Read,
    /// Produces commands (queued while the agent is paused, `[APP-04]`).
    Mutate,
    /// Editor-only state (view defaults): no document change.
    Editor,
}

/// A tool's result.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolOutput {
    /// The structured result (`structuredContent`).
    pub data: Value,
    /// PNG images returned as MCP image content items.
    pub images: Vec<Vec<u8>>,
}

impl ToolOutput {
    /// A result with `data` and no images.
    pub fn data(data: Value) -> Self {
        Self { data, images: Vec::new() }
    }
}

/// A tool failure the agent can act on.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolError {
    /// A stable code (`invalid_arguments`, `policy`, `human_protected`, `region_locked` ...).
    pub code: String,
    /// One sentence for the agent, naming the unlock path where there is one.
    pub message: String,
}

impl ToolError {
    /// An error with `code` and `message`.
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        Self { code: code.to_owned(), message: message.into() }
    }

    /// The JSON form (`{"error": {"code", "message"}}`).
    pub fn to_json(&self) -> Value {
        let mut e = Object::with_capacity(2);
        e.insert("code", self.code.as_str());
        e.insert("message", self.message.as_str());
        let mut o = Object::with_capacity(1);
        o.insert("error", Value::Object(e));
        Value::Object(o)
    }
}

impl From<CommandError> for ToolError {
    fn from(e: CommandError) -> Self {
        let message = match &e {
            CommandError::Rejected(r) => format!("{r} (ask the human, or request an unlock)"),
            other => other.to_string(),
        };
        Self { code: e.code().to_owned(), message }
    }
}

/// What the agent may do without a proposal `[SEC-03]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AgentPolicy {
    /// Mutations may edit the main document directly (default: off, only inside proposals).
    pub direct_edits: bool,
    /// `commit_proposal` merges the proposal itself instead of handing it to the human for
    /// review (default: off).
    pub commit_without_review: bool,
}

impl AgentPolicy {
    /// The default: proposals only, the human reviews.
    pub const DEFAULT: Self = Self { direct_edits: false, commit_without_review: false };
}

impl Default for AgentPolicy {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// The view defaults `set_lighting_rig` and `set_camera` change (editor-only, not provenance).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewDefaults {
    /// The rig renders use when they name none.
    pub rig: BuiltinRig,
    /// The tier renders use when they name none.
    pub quality: Quality,
    /// The camera preset renders use when they name none (`None`: an explicit camera).
    pub preset: Option<&'static str>,
    /// The explicit camera, when one was set.
    pub camera: Option<Camera>,
}

impl Default for ViewDefaults {
    fn default() -> Self {
        Self { rig: BuiltinRig::NeutralStudio, quality: Quality::Review, preset: Some("head"), camera: None }
    }
}

/// A render the server remembers (`humanart://renders/<id>`).
#[derive(Clone, Debug, PartialEq)]
pub struct RenderRecord {
    /// Its id.
    pub id: u64,
    /// The file in the sandbox (display form).
    pub file: String,
    /// The PNG bytes.
    pub png: Vec<u8>,
    /// The image (for region colour statistics).
    pub image: Image,
    /// The metadata returned with it.
    pub meta: Value,
}

/// A render requested with `async: true`, done after the current requests.
#[derive(Clone, Debug, PartialEq)]
pub struct PendingRender {
    /// The ids reserved for its views.
    pub ids: Vec<u64>,
    /// The tool arguments.
    pub args: Value,
}

/// The server's own state that tools read and change.
#[derive(Debug, Default)]
pub struct ToolState {
    /// The render service.
    pub renders: RenderService,
    /// Remembered renders, oldest first.
    pub records: Vec<RenderRecord>,
    /// The next render id.
    pub next_render: u64,
    /// Async renders waiting.
    pub pending: Vec<PendingRender>,
    /// View defaults.
    pub view: ViewDefaults,
    /// Proposals the agent handed to the human for review.
    pub in_review: Vec<Id128>,
}

/// Everything a handler works with.
pub struct ToolCtx<'c, 'a> {
    /// The engine (session and evaluated character).
    pub engine: &'c mut Engine<'a>,
    /// The project's file system (blob store).
    pub fs: &'c dyn FileSystem,
    /// The agent's author record.
    pub agent: &'c Author,
    /// The policy.
    pub policy: AgentPolicy,
    /// Paths the agent may use.
    pub sandbox: &'c Sandbox,
    /// Render limits.
    pub render_limits: RenderLimits,
    /// The server's state.
    pub state: &'c mut ToolState,
    /// Notifications the tool raised (`(method, params)`), sent after its response.
    pub notes: &'c mut Vec<(&'static str, Value)>,
}

/// A handler.
pub type Handler = fn(&mut ToolCtx<'_, '_>, &Value) -> Result<ToolOutput, ToolError>;

/// One tool.
#[derive(Clone, Copy)]
pub struct ToolSpec {
    /// The name.
    pub name: &'static str,
    /// A short human title.
    pub title: &'static str,
    /// What it does.
    pub description: &'static str,
    /// Its arguments.
    pub fields: &'static [Field],
    /// What it does to the document.
    pub access: Access,
    /// The handler.
    pub handler: Handler,
}

impl std::fmt::Debug for ToolSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolSpec").field("name", &self.name).field("access", &self.access).finish_non_exhaustive()
    }
}

impl ToolSpec {
    /// The MCP tool definition (`name`, `title`, `description`, `inputSchema`, `annotations`).
    pub fn definition(&self, vocab: &Vocab<'_>) -> Value {
        let mut ann = Object::with_capacity(3);
        ann.insert("readOnlyHint", self.access == Access::Read);
        ann.insert("destructiveHint", self.access == Access::Mutate);
        ann.insert("openWorldHint", false);
        let mut o = Object::with_capacity(5);
        o.insert("name", self.name);
        o.insert("title", self.title);
        o.insert("description", self.description);
        o.insert("inputSchema", record_schema(self.fields, vocab));
        o.insert("annotations", Value::Object(ann));
        Value::Object(o)
    }
}

/// Every tool, observation tools first.
pub fn all() -> impl Iterator<Item = &'static ToolSpec> {
    observe::TOOLS.iter().chain(mutate::TOOLS.iter())
}

/// The tool named `name`.
pub fn find(name: &str) -> Option<&'static ToolSpec> {
    all().find(|t| t.name == name)
}

/// The optional `proposal_id` every mutation takes.
pub const PROPOSAL: Field =
    Field::opt("proposal_id", Ty::Id, "Apply inside this open proposal instead of the main document.");
/// The optional `reason` every mutation takes (the history label, recorded in provenance).
pub const REASON: Field = Field::opt("reason", Ty::Str { max: 512 }, "Why: recorded as the command's label.");

/// Argument readers for validated arguments.
pub(crate) mod args {
    use humanart_core::hash::Hash128;
    use humanart_core::id::Id128;
    use humanart_io::json::Value;

    use super::ToolError;

    pub fn str<'v>(a: &'v Value, name: &str) -> Option<&'v str> {
        a.get(name).and_then(Value::as_str)
    }

    pub fn f64(a: &Value, name: &str) -> Option<f64> {
        a.get(name).and_then(Value::as_f64)
    }

    pub fn u64(a: &Value, name: &str, default: u64) -> u64 {
        a.get(name).and_then(Value::as_u64).unwrap_or(default)
    }

    pub fn bool(a: &Value, name: &str, default: bool) -> bool {
        a.get(name).and_then(Value::as_bool).unwrap_or(default)
    }

    pub fn id(a: &Value, name: &str) -> Result<Option<Id128>, ToolError> {
        match str(a, name) {
            None => Ok(None),
            Some(s) => {
                Id128::parse(s).map(Some).map_err(|_| ToolError::new("invalid_arguments", format!("{name}: not an id")))
            }
        }
    }

    pub fn req_id(a: &Value, name: &str) -> Result<Id128, ToolError> {
        id(a, name)?.ok_or_else(|| ToolError::new("invalid_arguments", format!("{name} is required")))
    }

    pub fn hash(a: &Value, name: &str) -> Result<Option<Hash128>, ToolError> {
        match str(a, name) {
            None => Ok(None),
            Some(s) => Hash128::parse(s)
                .map(Some)
                .map_err(|_| ToolError::new("invalid_arguments", format!("{name}: not a blob hash"))),
        }
    }

    pub fn vec3(a: &Value, name: &str) -> Option<[f64; 3]> {
        let v = a.get(name)?.as_array()?;
        Some([v.first()?.as_f64()?, v.get(1)?.as_f64()?, v.get(2)?.as_f64()?])
    }
}

/// `n` as a JSON number.
pub(crate) fn num(n: f64) -> Value {
    Value::from(Number::from(n))
}

/// `n` as a JSON integer.
pub(crate) fn uint(n: u64) -> Value {
    Value::from(Number::from(n))
}

/// An object from members.
pub(crate) fn object(members: Vec<(&str, Value)>) -> Value {
    let mut o = Object::with_capacity(members.len());
    for (k, v) in members {
        o.insert(k, v);
    }
    Value::Object(o)
}

/// Runs `command` by the agent: inside `proposal` when given, else on the main document when
/// the policy allows it. Makes sure the document knows the agent first.
pub(crate) fn submit(ctx: &mut ToolCtx<'_, '_>, proposal: Option<Id128>, command: Command) -> Result<Value, ToolError> {
    ensure_agent(ctx)?;
    let outcome = match proposal {
        Some(p) => ctx.engine.session_mut().execute_in(p, command)?,
        None if ctx.policy.direct_edits => ctx.engine.execute(command)?,
        None => {
            return Err(ToolError::new(
                "policy",
                "this project lets the agent edit only inside proposals: call open_proposal and pass its proposal_id",
            ));
        }
    };
    ctx.engine.update().map_err(|e| ToolError::new("engine", e.to_string()))?;
    Ok(outcome.to_json())
}

/// Adds the agent to the document's authors when it is not there yet (a command by an unknown
/// author may only add itself).
pub(crate) fn ensure_agent(ctx: &mut ToolCtx<'_, '_>) -> Result<(), ToolError> {
    if ctx.engine.session().document().author(&ctx.agent.id).is_some() {
        return Ok(());
    }
    let cmd = Command::new(&ctx.agent.id, humanart_doc::command::Action::AddAuthor { author: ctx.agent.clone() });
    ctx.engine.execute(cmd)?;
    Ok(())
}
