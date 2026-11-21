//! What `serve`, `batch`, `turntable` and `validate` share: the data folder (registry and
//! template), the project (opened, created or none), the engine on it, the agent server admitted
//! into the document by the human, and the read-only reading of a saved project.
//!
//! **Files.** The session's file system is rooted at the project's parent folder
//! (`humanart_platform::StdFileSystem::rooted`), so nothing the document names can reach outside
//! it; without a project it is rooted at the exchange folder. The agent's own file paths go
//! through the server's sandbox (the exchange folder and the project).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use humanart_agent::engine::{Engine, Model};
use humanart_agent::mcp::security::Sandbox;
use humanart_agent::mcp::server::{Server, ServerConfig};
use humanart_agent::mcp::tools::AgentPolicy;
use humanart_core::id::Id128;
use humanart_core::platform::{Clock, Entropy, FileSystem};
use humanart_doc::document::{Author, AuthorKind, Document};
use humanart_doc::project::Project;
use humanart_doc::registry::Registry;
use humanart_doc::session::{JournalOutcome, Session, SessionConfig};
use humanart_doc::{DocError, migrate, text};
use humanart_model::Template;
use humanart_platform::{OsEntropy, StdFileSystem, SystemClock};

use crate::PROGRAM;
use crate::args::{OptSpec, Parsed, UsageError};
use crate::error::CliError;

/// `--data`.
pub const DATA: OptSpec = OptSpec {
    name: "data",
    value: Some("DIR"),
    default: Some("data"),
    help: "Data folder (template/params registry, template/template.antmpl).",
};
/// `--project`.
pub const PROJECT: OptSpec = OptSpec {
    name: "project",
    value: Some("DIR"),
    default: None,
    help: "Project folder to open (without it: a new character in memory, nothing is saved).",
};
/// `--create`.
pub const CREATE: OptSpec = OptSpec {
    name: "create",
    value: None,
    default: None,
    help: "Create the --project folder's project when it holds none yet.",
};
/// `--exchange`.
pub const EXCHANGE: OptSpec = OptSpec {
    name: "exchange",
    value: Some("DIR"),
    default: Some("exchange"),
    help: "Folder the agent may read and write files in (created when missing).",
};
/// `--agent-id`.
pub const AGENT_ID: OptSpec = OptSpec {
    name: "agent-id",
    value: Some("ID"),
    default: Some("agent.cli"),
    help: "Author id the agent's commands are made under.",
};
/// `--agent-name`.
pub const AGENT_NAME: OptSpec =
    OptSpec { name: "agent-name", value: Some("NAME"), default: Some("Agent"), help: "The agent's display name." };
/// `--human`.
pub const HUMAN: OptSpec = OptSpec {
    name: "human",
    value: Some("ID"),
    default: None,
    help: "Human author who admits the agent (default: the document's first human author).",
};
/// `--direct-edits`.
pub const DIRECT_EDITS: OptSpec = OptSpec {
    name: "direct-edits",
    value: None,
    default: None,
    help: "Let the agent edit the document directly (default: only inside proposals).",
};
/// `--commit-without-review`.
pub const COMMIT_WITHOUT_REVIEW: OptSpec = OptSpec {
    name: "commit-without-review",
    value: None,
    default: None,
    help: "Let commit_proposal merge a proposal itself instead of leaving it for review.",
};
/// `--save`.
pub const SAVE: OptSpec = OptSpec {
    name: "save",
    value: None,
    default: None,
    help: "Save the project at the end (otherwise its journal keeps the edits for the editor).",
};
/// `--json-errors`.
pub const JSON_ERRORS: OptSpec =
    OptSpec { name: "json-errors", value: None, default: None, help: "Errors as one JSON line on stderr." };

/// The author id of the human when a new document is made without `--human`.
pub const DEFAULT_HUMAN: &str = "human.cli";
/// Longest author id or name taken from the command line.
pub const MAX_ID_BYTES: usize = 128;

/// The host options of a parsed command line (see the option constants).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostOptions {
    /// The data folder.
    pub data: PathBuf,
    /// The project folder.
    pub project: Option<PathBuf>,
    /// Create the project when the folder holds none.
    pub create: bool,
    /// The agent's exchange folder.
    pub exchange: PathBuf,
    /// The agent's author id.
    pub agent_id: String,
    /// The agent's display name.
    pub agent_name: String,
    /// The human who admits the agent.
    pub human: Option<String>,
    /// What the agent may do without the human's review.
    pub policy: AgentPolicy,
    /// Save the project at the end.
    pub save: bool,
}

/// Checks an author id or name from the command line.
fn author_text(p: &Parsed, name: &str) -> Result<Option<String>, CliError> {
    let Some(text) = p.get(name) else { return Ok(None) };
    if text.is_empty() || text.len() > MAX_ID_BYTES || text.chars().any(char::is_control) {
        return Err(CliError::Usage(UsageError(format!(
            "--{name}: 1 to {MAX_ID_BYTES} bytes without control characters"
        ))));
    }
    Ok(Some(text.to_owned()))
}

impl HostOptions {
    /// The host options of a parsed command line whose spec lists the host options it uses
    /// (absent ones take their defaults).
    ///
    /// # Errors
    /// [`CliError::Usage`] for an author id or name that is empty, too long or has control
    /// characters, and for `--create` or `--save` without `--project`.
    pub fn from_args(p: &Parsed) -> Result<Self, CliError> {
        let flag = |name: &str| p.get(name).is_some();
        let o = Self {
            data: PathBuf::from(p.get("data").unwrap_or("data")),
            project: p.get("project").map(PathBuf::from),
            create: flag("create"),
            exchange: PathBuf::from(p.get("exchange").unwrap_or("exchange")),
            agent_id: author_text(p, "agent-id")?.unwrap_or_else(|| "agent.cli".to_owned()),
            agent_name: author_text(p, "agent-name")?.unwrap_or_else(|| "Agent".to_owned()),
            human: author_text(p, "human")?,
            policy: AgentPolicy {
                direct_edits: flag("direct-edits"),
                commit_without_review: flag("commit-without-review"),
            },
            save: flag("save"),
        };
        if o.project.is_none() && (o.create || o.save) {
            return Err(CliError::Usage(UsageError("--create and --save need --project".to_owned())));
        }
        Ok(o)
    }
}

/// Splits a project folder into its parent (the root of the session's file system) and its name
/// (the project's path inside that file system).
///
/// # Errors
/// [`CliError::Project`] when the folder has no usable name.
pub fn split_project(dir: &Path) -> Result<(PathBuf, String), CliError> {
    let name = dir
        .file_name()
        .and_then(|n| n.to_str())
        .filter(|n| !n.is_empty() && *n != "." && *n != "..")
        .ok_or_else(|| CliError::Project(format!("{}: not a folder name the project can use", dir.display())))?;
    let parent =
        dir.parent().filter(|p| !p.as_os_str().is_empty()).map_or_else(|| PathBuf::from("."), Path::to_path_buf);
    Ok((parent, name.to_owned()))
}

/// The loaded data and the file system a session works in.
pub struct Base {
    /// The parameter registry.
    pub registry: Registry,
    /// The template.
    pub template: Arc<Template>,
    /// Where the template came from (`package` or `generated`).
    pub source: &'static str,
    /// The session's file system (see the module documentation).
    pub fs: StdFileSystem,
    /// The clock.
    pub clock: SystemClock,
    /// The project's path inside [`Base::fs`].
    pub project: Option<String>,
    /// The project folder on disk.
    pub project_dir: Option<PathBuf>,
}

impl std::fmt::Debug for Base {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Base").field("source", &self.source).field("project", &self.project).finish_non_exhaustive()
    }
}

/// A platform error as a [`CliError::Project`] about `what`.
fn project_error(what: &Path, e: impl std::fmt::Display) -> CliError {
    CliError::Project(format!("{}: {e}", what.display()))
}

impl Base {
    /// Loads the registry and template of `data` and roots the file system for `project` (its
    /// parent folder; created with `create`), or for `exchange` without a project.
    ///
    /// # Errors
    /// [`CliError::Data`] / [`CliError::Io`] for the data folder, [`CliError::Project`] when the
    /// folder cannot be used.
    pub fn load(data: &Path, project: Option<&Path>, create: bool, exchange: &Path) -> Result<Self, CliError> {
        let registry = crate::render::load_registry(data)?;
        let (template, source) = crate::render::load_template(data)?;
        let (fs, project_name, project_dir) = match project {
            Some(dir) => {
                let (parent, name) = split_project(dir)?;
                if create {
                    std::fs::create_dir_all(&parent).map_err(|e| CliError::io(&parent, e))?;
                }
                let fs = StdFileSystem::rooted(&parent).map_err(|e| project_error(&parent, e))?;
                (fs, Some(name), Some(dir.to_path_buf()))
            }
            None => (StdFileSystem::create_rooted(exchange).map_err(|e| project_error(exchange, e))?, None, None),
        };
        Ok(Self {
            registry,
            template: Arc::new(template),
            source,
            fs,
            clock: SystemClock::new(),
            project: project_name,
            project_dir,
        })
    }

    /// The engine on the project (opened, replaying its journal; or created with `create`), or
    /// on a new document without a project; and the id of the human who admits the agent.
    ///
    /// # Errors
    /// [`CliError::Project`] when the project cannot be opened or created, or the human is not a
    /// human author of it; [`CliError::Render`] when the model cannot follow the document.
    pub fn engine(&self, human: Option<&str>, create: bool) -> Result<(Engine<'_>, String), CliError> {
        let seed = Id128::random(&OsEntropy).map_err(|e| CliError::Project(format!("no entropy: {e}")))?;
        let config = SessionConfig::default();
        let new_doc = |human: &str| {
            let me = Author { id: human.to_owned(), kind: AuthorKind::Human, name: PROGRAM.to_owned() };
            Document::new(seed, "Untitled character", self.source, 1, self.clock.unix_millis(), me)
        };
        let session = match &self.project {
            Some(root) => {
                let shown = self.project_dir.as_deref().unwrap_or(Path::new(root));
                if self.fs.exists(&Project::at(root).document_path()) {
                    let (session, report) = Session::open(root, &self.registry, &self.fs, &self.clock, config, seed)
                        .map_err(|e| project_error(shown, e))?;
                    log_open(&report.journal);
                    for (name, e) in &report.damaged_proposals {
                        eprintln!("{PROGRAM}: proposal {name} is damaged and was left alone: {e}");
                    }
                    session
                } else if create {
                    let doc = new_doc(human.unwrap_or(DEFAULT_HUMAN));
                    Session::create(root, doc, &self.registry, &self.fs, &self.clock, config, seed)
                        .map_err(|e| project_error(shown, e))?
                } else {
                    return Err(CliError::Project(format!(
                        "{}: no project there (--create makes one)",
                        shown.display()
                    )));
                }
            }
            None => Session::new(
                new_doc(human.unwrap_or(DEFAULT_HUMAN)),
                &self.registry,
                &self.fs,
                &self.clock,
                config,
                seed,
            ),
        };
        let human = pick_human(session.document(), human)?;
        let model = Model::new(Arc::clone(&self.template), &self.registry);
        let mut engine = Engine::new(session, &self.registry, model);
        engine.update()?;
        Ok((engine, human))
    }

    /// The agent server of `options`, admitted into the engine's document by `human`.
    ///
    /// # Errors
    /// [`CliError::Server`] when the sandbox or the entropy fails, [`CliError::Project`] when the
    /// session refuses the agent.
    pub fn server(&self, options: &HostOptions, engine: &mut Engine<'_>, human: &str) -> Result<Server, CliError> {
        let mut config = ServerConfig::new(&options.agent_id, &options.agent_name);
        config.policy = options.policy;
        let provenance = options.data.join("..").join("docs").join("data-provenance.md");
        config.provenance = provenance.is_file().then_some(provenance);
        let sandbox = Sandbox::new(&options.exchange, self.project_dir.as_deref())
            .map_err(|e| CliError::Server(format!("the sandbox could not be set up: {e}")))?;
        let mut seed = [0_u8; 8];
        OsEntropy.fill(&mut seed).map_err(|e| CliError::Server(format!("no entropy: {e}")))?;
        let server = Server::new(config, sandbox, u64::from_le_bytes(seed));
        server
            .admit(engine, human)
            .map_err(|e| CliError::Project(format!("the agent could not be admitted by {human}: {e}")))?;
        engine.update()?;
        Ok(server)
    }
}

/// The human who admits the agent: `wanted` when it is a human author of `doc`, else the first
/// human author.
fn pick_human(doc: &Document, wanted: Option<&str>) -> Result<String, CliError> {
    match wanted {
        Some(id) => match doc.author(id) {
            Some(a) if a.kind == AuthorKind::Human => Ok(id.to_owned()),
            _ => Err(CliError::Project(format!("`{id}` is not a human author of the document"))),
        },
        None => doc
            .authors
            .iter()
            .find(|a| a.kind == AuthorKind::Human)
            .map(|a| a.id.clone())
            .ok_or_else(|| CliError::Project("the document has no human author to admit the agent".to_owned())),
    }
}

/// Reports on stderr what opening the project did with its journal.
fn log_open(journal: &JournalOutcome) {
    match journal {
        JournalOutcome::Recovered { records } => {
            eprintln!("{PROGRAM}: recovered {records} unsaved changes from the journal");
        }
        JournalOutcome::Unreplayable { error, kept_at } => {
            eprintln!("{PROGRAM}: the journal could not be replayed ({error}); it was kept as {kept_at}");
        }
        JournalOutcome::Missing | JournalOutcome::Clean | JournalOutcome::Stale => {}
    }
}

/// After a pump: re-syncs the model and lets the session journal and autosave; problems go to
/// stderr (the serve loop goes on).
pub fn settle(engine: &mut Engine<'_>) {
    if let Err(e) = engine.update() {
        eprintln!("{PROGRAM}: {e}");
    }
    match engine.session_mut().tick() {
        Ok(Some(path)) => eprintln!("{PROGRAM}: autosaved {path}"),
        Ok(None) => {}
        Err(e) => eprintln!("{PROGRAM}: autosave failed: {e}"),
    }
}

/// Saves the project when `save` asks for it and one is open; whether it saved.
///
/// # Errors
/// [`CliError::Project`] when the save fails.
pub fn finish(engine: &mut Engine<'_>, save: bool) -> Result<bool, CliError> {
    if save && engine.session().project().is_some() {
        engine.session_mut().save().map_err(|e| CliError::Project(format!("the project could not be saved: {e}")))?;
        return Ok(true);
    }
    Ok(false)
}

/// Reads the saved document of the project at `root` in `fs` without writing anything (no
/// journal repair, no replay): the document and the format version it was migrated from.
///
/// # Errors
/// [`DocError`] when the file is missing, too large, or does not parse or check.
pub fn read_document(
    fs: &dyn FileSystem,
    root: &str,
    registry: &Registry,
) -> Result<(Document, Option<u32>), DocError> {
    let limits = SessionConfig::default().load;
    let path = Project::at(root).document_path();
    let meta = fs.metadata(&path).map_err(|e| DocError::platform(&path, e))?;
    if meta.len > limits.text.max_bytes {
        return Err(DocError::Limit { what: "bytes of document text", limit: limits.text.max_bytes });
    }
    let bytes = fs.read(&path).map_err(|e| DocError::platform(&path, e))?;
    let file = text::parse(&bytes, &limits.text)?;
    let version = file.version;
    let doc = Document::from_text(&migrate::migrate(file)?, registry)?;
    Ok((doc, (version != humanart_doc::document::FORMAT_VERSION).then_some(version)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::CommandSpec;

    const OPTIONS: [OptSpec; 11] = [
        DATA,
        PROJECT,
        CREATE,
        EXCHANGE,
        AGENT_ID,
        AGENT_NAME,
        HUMAN,
        DIRECT_EDITS,
        COMMIT_WITHOUT_REVIEW,
        SAVE,
        JSON_ERRORS,
    ];
    static SPEC: CommandSpec = CommandSpec { name: "host", summary: "Host options.", options: &OPTIONS };

    fn options(list: &[&str]) -> Result<HostOptions, CliError> {
        let args: Vec<String> = list.iter().map(|s| (*s).to_owned()).collect();
        HostOptions::from_args(&Parsed::parse(&SPEC, &args)?)
    }

    #[test]
    fn cli_02_host_options_map_and_refuse_bad_values() {
        let d = options(&[]).unwrap();
        assert_eq!((d.agent_id.as_str(), d.agent_name.as_str(), d.human.as_deref()), ("agent.cli", "Agent", None));
        assert_eq!(d.policy, AgentPolicy::DEFAULT);
        assert!(d.project.is_none() && !d.create && !d.save);
        let o = options(&["--project", "p", "--create", "--save", "--direct-edits", "--human", "human.me"]).unwrap();
        assert!(o.policy.direct_edits && !o.policy.commit_without_review && o.create && o.save);
        assert_eq!(o.human.as_deref(), Some("human.me"));
        for bad in [&["--create"][..], &["--save"], &["--agent-id", ""], &["--agent-name", "a\u{7}b"]] {
            assert!(matches!(options(bad), Err(CliError::Usage(_))), "{bad:?}");
        }
        let long = "x".repeat(MAX_ID_BYTES + 1);
        assert!(options(&["--human", &long]).is_err());
    }

    #[test]
    fn project_folders_split_into_root_and_name() {
        let (parent, name) = split_project(Path::new("work/chars/alice")).unwrap();
        assert_eq!((parent, name.as_str()), (PathBuf::from("work/chars"), "alice"));
        assert_eq!(split_project(Path::new("alice")).unwrap().0, PathBuf::from("."));
        assert!(matches!(split_project(Path::new("..")), Err(CliError::Project(_))));
        assert!(matches!(split_project(Path::new("/")), Err(CliError::Project(_))));
    }

    #[test]
    fn the_admitting_human_must_be_a_human_author() {
        let me = Author { id: "human.a".into(), kind: AuthorKind::Human, name: "A".into() };
        let mut doc = Document::new(Id128::from_u128(3), "t", "generated", 1, 0, me);
        doc.authors.push(Author { id: "agent.x".into(), kind: AuthorKind::Agent, name: "X".into() });
        assert_eq!(pick_human(&doc, None).unwrap(), "human.a");
        assert_eq!(pick_human(&doc, Some("human.a")).unwrap(), "human.a");
        assert!(matches!(pick_human(&doc, Some("agent.x")), Err(CliError::Project(_))));
        assert!(matches!(pick_human(&doc, Some("nobody")), Err(CliError::Project(_))));
    }
}
