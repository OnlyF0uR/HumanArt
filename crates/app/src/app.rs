//! The editor's state and its frame: the engine, the shell's UI state, preferences, provenance,
//! the agent link, and the intents the UI produced.
//!
//! [`App::frame`] runs one UI frame: input and shortcuts, the shell's panels (which only read the
//! document and push [`Intent`]s), layout, then the intents are applied in order through the
//! engine's session (so a click is the same command an agent would send), the model is re-synced
//! and the provenance state reads the new events. The GPU side ([`crate::studio::Studio`]) then
//! draws [`App::ui`] and [`App::viewport_frames`].

use std::sync::Arc;

use humanart_agent::engine::{Engine, Framing, Highlight, Model, ViewSettings, view::frame_camera};
use humanart_core::id::Id128;
use humanart_core::platform::{Clock, Entropy, FileSystem};
use humanart_doc::command::{Command, CommandError};
use humanart_doc::document::{Author, AuthorKind, Document};
use humanart_doc::history::Scope;
use humanart_doc::registry::Registry;
use humanart_doc::session::{JournalOutcome, Session, SessionConfig};
use humanart_doc::starter::{DEFAULT as STARTER, Starters};
use humanart_model::Template;
use humanart_render::renderer::Quality;
use humanart_ui::geom::{Rect, Size};
use humanart_ui::input::{InputQueue, Key, Keymap, Modifiers, Shortcut};
use humanart_ui::popup::ToastKind;
use humanart_ui::text::Fonts;
use humanart_ui::theme::Theme;
use humanart_ui::ui::{FrameOutput, Ui};
use humanart_ui::views::LogLevel;

use crate::agent_host::{AgentHost, AgentServerOptions, policy_of};
use crate::agent_link::AgentLink;
use crate::error::AppError;
use crate::modes::Mode;
use crate::prefs::{Prefs, QualityChoice, ThemeChoice};
use crate::provenance::Provenance;
use crate::shell::Shell;
use crate::studio::{MAX_VIEWPORTS, ViewportFrame};

/// The keymap actions (stable names, `[UI-06]`).
pub const ACTIONS: &[&str] = &[
    "undo",
    "redo",
    "save",
    "search",
    "help",
    "preferences",
    "frame.body",
    "frame.head",
    "agent.pause",
    "mode.shape",
    "mode.skin",
    "mode.eyes_mouth",
    "mode.hair",
    "mode.expression",
    "mode.pose",
    "mode.light_camera",
    "mode.compare",
    "mode.review",
    "mode.export",
];

/// The default keymap.
pub fn default_keymap() -> Keymap {
    let mut k = Keymap::new();
    let ctrl = Modifiers::CTRL;
    let ctrl_shift = Modifiers { shift: true, ..Modifiers::CTRL };
    k.bind("undo", Shortcut::new(Key::Letter('Z'), ctrl));
    k.bind("redo", Shortcut::new(Key::Letter('Y'), ctrl));
    k.bind("redo", Shortcut::new(Key::Letter('Z'), ctrl_shift));
    k.bind("save", Shortcut::new(Key::Letter('S'), ctrl));
    k.bind("search", Shortcut::new(Key::Letter('F'), ctrl));
    k.bind("help", Shortcut::new(Key::F(1), Modifiers::NONE));
    k.bind("preferences", Shortcut::new(Key::Letter('P'), ctrl));
    k.bind("frame.body", Shortcut::new(Key::Home, Modifiers::NONE));
    k.bind("frame.head", Shortcut::new(Key::Letter('F'), Modifiers::NONE));
    k.bind("agent.pause", Shortcut::new(Key::Letter('P'), ctrl_shift));
    for (m, d) in Mode::ALL.iter().zip(['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']) {
        k.bind(m.id(), Shortcut::new(Key::Digit(d), ctrl));
    }
    k
}

/// What the UI asked for this frame, applied after layout in order.
#[derive(Debug)]
pub enum Intent {
    /// A command of the human on the main document.
    Execute(Command),
    /// One frame of the human's drag of the scalar parameter with this registry index to this
    /// value (allocation-free once the gesture is open, [`Session::drag_scalar`]).
    Drag(usize, f64),
    /// The slider gesture ended (release): one undo step.
    EndGesture,
    /// Undo, in a scope (the agent's last: selective undo `[CMD-04]`).
    Undo(Scope),
    /// Redo.
    Redo,
    /// The region the human works on (`None`: clear).
    Select(Option<String>),
    /// Switch mode.
    Mode(Mode),
    /// Save the project.
    Save,
    /// Open the project directory.
    Open(String),
    /// Create a project directory from the current document.
    SaveAs(String),
    /// Accept a proposal whole.
    Accept(Id128),
    /// Discard a proposal.
    Discard(Id128),
    /// Pause or resume the agent.
    AgentPause(bool),
    /// Revoke (`true`) or grant the agent's access.
    AgentRevoke(bool),
    /// Write the preferences file.
    SavePrefs,
    /// Apply the theme and scale of the preferences.
    ApplyPrefs,
    /// Frame a viewport's camera.
    Frame(usize, Framing),
    /// The human looked at the changes: clear "changed since I last looked".
    MarkLooked,
}

/// One viewport's state in the shell `[VP-04]`.
#[derive(Clone, Copy, Debug)]
pub struct ViewportState {
    /// What it shows.
    pub settings: ViewSettings,
    /// Its image rectangle last frame, logical pixels (zero when not shown).
    pub rect: Rect,
    /// Whether its panel is in the layout.
    pub open: bool,
    /// Whether its camera was framed on the model yet.
    pub framed: bool,
}

/// What the host gives the editor.
#[derive(Clone)]
pub struct Env<'a> {
    /// The parameter registry.
    pub registry: &'a Registry,
    /// The file system (projects, preferences, data).
    pub fs: &'a dyn FileSystem,
    /// The clock.
    pub clock: &'a dyn Clock,
    /// The template.
    pub template: Arc<Template>,
    /// The person at the editor.
    pub human: Author,
    /// Where `prefs.json`, `keymap.json`, `layout.json` and `last_project.txt` live.
    pub settings_dir: String,
    /// Seed of command ids (random per session).
    pub seed: Id128,
    /// Randomness for the agent server's token and session ids.
    pub entropy: &'a dyn Entropy,
    /// Where and as whom the agent is served when the human grants the link.
    pub agent: AgentServerOptions,
}

impl std::fmt::Debug for Env<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Env")
            .field("human", &self.human.id)
            .field("settings_dir", &self.settings_dir)
            .finish_non_exhaustive()
    }
}

/// The editor (see the module documentation).
pub struct App<'a> {
    env: Env<'a>,
    engine: Engine<'a>,
    ui: Ui,
    keymap: Keymap,
    /// The preferences.
    pub prefs: Prefs,
    /// The provenance state `[APP-03]`.
    pub provenance: Provenance,
    /// The agent link `[APP-04]`.
    pub agent: AgentLink,
    agent_server: AgentHost,
    /// The shell's UI state.
    pub shell: Shell,
    /// The viewports.
    pub viewports: [ViewportState; MAX_VIEWPORTS],
    intents: Vec<Intent>,
    highlights: Vec<Vec<Highlight>>,
    project: Option<String>,
}

impl std::fmt::Debug for App<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("App").field("project", &self.project).field("mode", &self.shell.mode).finish_non_exhaustive()
    }
}

/// The theme of the preferences.
pub fn theme_of(prefs: &Prefs) -> Theme {
    let t = match prefs.theme {
        ThemeChoice::Dark => Theme::dark(),
        ThemeChoice::Light => Theme::light(),
        ThemeChoice::HighContrast => Theme::high_contrast(),
    };
    let t = if prefs.colour_neutral { t.colour_neutral() } else { t };
    t.with_reduced_motion(prefs.reduced_motion)
}

fn quality_of(q: QualityChoice) -> Quality {
    match q {
        QualityChoice::Interactive => Quality::Interactive,
        QualityChoice::Review => Quality::Review,
        QualityChoice::Still => Quality::Still,
    }
}

/// The sample document (`[APP-08]`): a new character with the human as its author, starting from
/// the default starter character (todo 1.3: `data/template/characters.json`) where `registry` takes
/// its values.
pub fn sample_document(human: &Author, template: &str, registry: &Registry) -> Document {
    let mut doc =
        Document::new(Id128::from_u128(0x4855_4d41_4e41_5254), "Sample character", template, 1, 0, human.clone());
    if let Ok(starters) = Starters::builtin()
        && let Some(starter) = starters.find(STARTER)
    {
        // A starter the registry refuses leaves the registry defaults (the file's test checks it).
        let _ = starter.apply(&mut doc, registry);
    }
    doc
}

impl<'a> App<'a> {
    /// The editor on the sample document, with `prefs` and the baked `fonts`.
    pub fn new(env: Env<'a>, prefs: Prefs, fonts: Fonts) -> Self {
        let doc = sample_document(&env.human, "generated", env.registry);
        let session = Session::new(doc, env.registry, env.fs, env.clock, SessionConfig::default(), env.seed);
        let model = Model::new(Arc::clone(&env.template), env.registry);
        let engine = Engine::new(session, env.registry, model);
        let mut ui = Ui::new(fonts, theme_of(&prefs));
        ui.set_background(None);
        let mut keymap = default_keymap();
        let keymap_path = humanart_core::platform::join_path(&env.settings_dir, "keymap.json");
        let keymap_note = match env.fs.read(&keymap_path) {
            Ok(bytes) => match std::str::from_utf8(&bytes).map(|t| keymap.apply_json(t, ACTIONS, &[])) {
                Ok(Ok(())) => None,
                _ => Some("keymap.json is not valid; the default keymap is used"),
            },
            Err(_) => None,
        };
        let quality = quality_of(prefs.quality);
        let viewport = ViewportState {
            settings: ViewSettings { quality, ..ViewSettings::default() },
            rect: Rect::default(),
            open: false,
            framed: false,
        };
        let mut shell = Shell::new(env.registry, &env.template);
        if let Ok(bytes) = env.fs.read(&humanart_core::platform::join_path(&env.settings_dir, "layout.json")) {
            shell.restore_layout(&bytes);
        }
        if let Some(note) = keymap_note {
            shell.log(LogLevel::Warn, 0.0, note);
        }
        let mut app = Self {
            engine,
            ui,
            keymap,
            prefs,
            provenance: Provenance::new(),
            agent: AgentLink::new(),
            agent_server: AgentHost::new(),
            shell,
            viewports: [viewport; MAX_VIEWPORTS],
            intents: Vec::with_capacity(16),
            highlights: (0..MAX_VIEWPORTS).map(|_| Vec::with_capacity(16)).collect(),
            project: None,
            env,
        };
        app.shell.onboarding = !app.env.fs.exists(&app.settings_path("prefs.json"));
        app
    }

    fn settings_path(&self, name: &str) -> String {
        humanart_core::platform::join_path(&self.env.settings_dir, name)
    }

    /// The engine.
    pub fn engine(&self) -> &Engine<'a> {
        &self.engine
    }

    /// The engine, for the host (an agent transport submits through [`App::agent`]).
    pub fn engine_mut(&mut self) -> &mut Engine<'a> {
        &mut self.engine
    }

    /// The UI (its draw list after [`App::frame`]).
    pub fn ui(&self) -> &Ui {
        &self.ui
    }

    /// The open project directory.
    pub fn project(&self) -> Option<&str> {
        self.project.as_deref()
    }

    /// The person at the editor.
    pub fn human(&self) -> &str {
        &self.env.human.id
    }

    /// Opens the project at `root`, replaying its journal after an unclean shutdown (crash
    /// recovery from the UI, `[SAVE-01]`, `[AC-WP21-3]`), and remembers it for the next launch.
    pub fn open_project(&mut self, root: &str) -> Result<(), AppError> {
        let (session, report) = Session::open(
            root,
            self.env.registry,
            self.env.fs,
            self.env.clock,
            SessionConfig::default(),
            self.env.seed,
        )?;
        self.engine.replace_session(session);
        self.provenance = Provenance::new();
        self.project = Some(root.to_owned());
        let now = self.ui.time();
        match report.journal {
            JournalOutcome::Recovered { records } => {
                let text = format!("Recovered {records} unsaved changes from the last session");
                self.shell.toast(ToastKind::Success, &text);
                self.shell.log(LogLevel::Info, now, &text);
            }
            JournalOutcome::Unreplayable { error, kept_at } => {
                let text = format!("The journal could not be replayed ({error}); it was kept as {kept_at}");
                self.shell.toast(ToastKind::Error, &text);
                self.shell.log(LogLevel::Error, now, &text);
            }
            JournalOutcome::Missing | JournalOutcome::Clean | JournalOutcome::Stale => {
                self.shell.log(LogLevel::Info, now, &format!("Opened {root}"));
            }
        }
        for (name, e) in &report.damaged_proposals {
            self.shell.log(LogLevel::Warn, now, &format!("Proposal {name} is damaged: {e}"));
        }
        self.remember_project(root);
        self.engine.update()?;
        self.rebind_agent()
    }

    /// Creates the project directory `root` from the current document and continues in it.
    pub fn save_as(&mut self, root: &str) -> Result<(), AppError> {
        let doc = self.engine.session().document().clone();
        let session = Session::create(
            root,
            doc,
            self.env.registry,
            self.env.fs,
            self.env.clock,
            SessionConfig::default(),
            self.env.seed,
        )?;
        self.engine.replace_session(session);
        self.project = Some(root.to_owned());
        self.remember_project(root);
        self.shell.log(LogLevel::Info, self.ui.time(), &format!("Created {root}"));
        self.rebind_agent()
    }

    /// The agent server (running while the human grants the link).
    pub fn agent_server(&self) -> &AgentHost {
        &self.agent_server
    }

    /// The human grants the agent access: the link opens and the agent server starts on
    /// loopback HTTP, with the agent admitted into the document `[APP-04]`, `[AG-01]`.
    ///
    /// # Errors
    /// [`AppError::AgentServer`] / [`AppError::Command`] when the server cannot start.
    pub fn grant_agent(&mut self) -> Result<(), AppError> {
        self.agent.grant();
        if self.agent_server.is_running() {
            return Ok(());
        }
        let project = self.project.as_deref().map(std::path::Path::new);
        self.agent_server.start(
            &mut self.engine,
            &self.env.human.id,
            &self.env.agent,
            policy_of(&self.prefs.agent),
            self.env.entropy,
            project,
        )?;
        self.agent_server.sync_gate(&self.agent);
        self.agent.connect(&self.env.agent.agent_name);
        let now = self.ui.time();
        let text = format!("Agent server on {}", self.agent_server.url().unwrap_or_default());
        self.shell.log(LogLevel::Info, now, &text);
        Ok(())
    }

    /// The human revokes the agent's access: the link refuses it, the server answers what is
    /// queued with "access revoked", ends its subscriptions and stops listening. Returns how many
    /// commands the link dropped.
    pub fn revoke_agent(&mut self) -> usize {
        let dropped = self.agent.revoke();
        self.agent_server.sync_gate(&self.agent);
        if self.agent_server.is_running() {
            self.agent_server.stop();
            self.agent.disconnect();
        }
        dropped
    }

    /// Handles what the agent sent (the gate follows the link first) and re-syncs the model
    /// when something was handled; returns how many messages. The frame calls it; the host may
    /// also call it between frames and redraw when it returns more than zero. Allocation free
    /// when nothing is waiting.
    pub fn pump_agent(&mut self) -> usize {
        if !self.agent_server.is_running() {
            return 0;
        }
        self.agent_server.sync_gate(&self.agent);
        let handled = self.agent_server.pump(&mut self.engine, self.env.fs);
        if handled > 0
            && let Err(e) = self.engine.update()
        {
            self.shell.log(LogLevel::Error, self.ui.time(), &e.to_string());
        }
        handled
    }

    /// Re-admits the agent after the session was replaced.
    fn rebind_agent(&mut self) -> Result<(), AppError> {
        let project = self.project.as_deref().map(std::path::Path::new);
        self.agent_server.rebind(&mut self.engine, &self.env.human.id, &self.env.agent, project)
    }

    fn remember_project(&mut self, root: &str) {
        let path = self.settings_path("last_project.txt");
        let _ = self.env.fs.create_dir_all(&self.env.settings_dir);
        if let Err(e) = self.env.fs.write_atomic(&path, root.as_bytes()) {
            self.shell.log(LogLevel::Warn, self.ui.time(), &format!("{path}: {e}"));
        }
    }

    /// The project opened last time (relaunch restores it).
    pub fn last_project(&self) -> Option<String> {
        let bytes = self.env.fs.read(&self.settings_path("last_project.txt")).ok()?;
        let text = String::from_utf8(bytes).ok()?;
        let t = text.trim();
        (!t.is_empty() && t.len() <= crate::prefs::MAX_PATH_CHARS).then(|| t.to_owned())
    }

    /// Runs one UI frame of `size` logical pixels at `scale` (see the module documentation).
    pub fn frame(&mut self, input: &mut InputQueue, size: Size, scale: f32) -> FrameOutput {
        let scale = humanart_ui::ui::clamp_ui_scale(scale * self.prefs.ui_scale);
        let size = Size::new(size.w / self.prefs.ui_scale, size.h / self.prefs.ui_scale);
        self.ui.begin_frame(input, size, scale);
        while let Some(action) = self.ui.shortcut(&self.keymap) {
            self.shortcut(action);
        }
        let now = self.ui.time();
        self.shell.build(
            &mut self.ui,
            &crate::shell::Ctx {
                session: self.engine.session(),
                registry: self.env.registry,
                model: self.engine.model(),
                provenance: &self.provenance,
                agent: &self.agent,
                agent_server: self.agent_server.url().zip(self.agent_server.token()),
                prefs: &self.prefs,
                human: &self.env.human.id,
                project: self.project.as_deref(),
                fs: self.env.fs,
                now,
            },
            &mut self.viewports,
            &mut self.intents,
        );
        let out = self.ui.end_frame().clone();
        self.shell.read_layout(&self.ui, &mut self.viewports);
        self.apply_intents();
        self.pump_agent();
        if let Err(e) = self.engine.update() {
            self.shell.log(LogLevel::Error, now, &e.to_string());
        }
        self.provenance.update(self.engine.session(), self.env.registry, now);
        match self.engine.session_mut().tick() {
            Ok(Some(path)) => self.shell.log(LogLevel::Info, now, &format!("Autosaved {path}")),
            Ok(None) => {}
            Err(e) => self.shell.log(LogLevel::Warn, now, &format!("Autosave failed: {e}")),
        }
        self.frame_new_viewports();
        self.update_highlights(now);
        let mut out = out;
        out.animating |= self.provenance.animating(now, self.prefs.flash_seconds);
        out
    }

    fn shortcut(&mut self, action: &str) {
        match action {
            "undo" => self.intents.push(Intent::Undo(Scope::Any)),
            "redo" => self.intents.push(Intent::Redo),
            "save" => self.intents.push(Intent::Save),
            "search" => self.shell.focus_search = true,
            "help" => self.shell.onboarding = !self.shell.onboarding,
            "preferences" => self.shell.open_prefs(&self.prefs),
            "frame.body" => self.intents.push(Intent::Frame(self.shell.active_viewport, Framing::Body)),
            "frame.head" => self.intents.push(Intent::Frame(self.shell.active_viewport, Framing::Head)),
            "agent.pause" => self.intents.push(Intent::AgentPause(!self.agent.is_paused())),
            other => {
                if let Some(m) = Mode::from_id(other) {
                    self.intents.push(Intent::Mode(m));
                }
            }
        }
    }

    fn apply_intents(&mut self) {
        let now = self.ui.time();
        if self.intents.is_empty() {
            return;
        }
        let mut intents = std::mem::take(&mut self.intents);
        for intent in intents.drain(..) {
            let result: Result<(), AppError> = match intent {
                Intent::Execute(cmd) => self.engine.execute(cmd).map(|_| ()).map_err(AppError::from),
                Intent::Drag(index, value) => match self.env.registry.params().get(index) {
                    Some(p) => self
                        .engine
                        .session_mut()
                        .drag_scalar(&self.env.human.id, &p.id, value)
                        .map(|_| ())
                        .map_err(AppError::from),
                    None => Ok(()),
                },
                Intent::EndGesture => self.engine.session_mut().end_gesture().map(|_| ()).map_err(AppError::from),
                Intent::Undo(scope) => {
                    self.engine.session_mut().undo(&self.env.human.id, &scope).map(|_| ()).map_err(AppError::from)
                }
                Intent::Redo => {
                    self.engine.session_mut().redo(&self.env.human.id, &Scope::Any).map(|_| ()).map_err(AppError::from)
                }
                Intent::Select(region) => {
                    self.shell.selected_region.clone_from(&region);
                    self.engine
                        .session_mut()
                        .select(&self.env.human.id, region.as_deref(), None)
                        .map(|_| ())
                        .map_err(AppError::from)
                }
                Intent::Mode(m) => {
                    self.shell.mode = m;
                    Ok(())
                }
                Intent::Save => self.save(),
                Intent::Open(root) => self.open_project(&root),
                Intent::SaveAs(root) => self.save_as(&root),
                Intent::Accept(id) => {
                    self.engine.session_mut().accept(id, &self.env.human.id, None).map(|_| ()).map_err(AppError::from)
                }
                Intent::Discard(id) => {
                    self.engine.session_mut().discard(id, &self.env.human.id).map_err(AppError::from)
                }
                Intent::AgentPause(pause) => {
                    if pause {
                        self.agent.pause();
                    } else {
                        for r in self.agent.resume(self.engine.session_mut()) {
                            if let Err(e) = r {
                                self.shell.log(LogLevel::Warn, now, &format!("Queued agent command refused: {e}"));
                            }
                        }
                    }
                    Ok(())
                }
                Intent::AgentRevoke(revoke) => {
                    if revoke {
                        let dropped = self.revoke_agent();
                        self.shell.log(
                            LogLevel::Warn,
                            now,
                            &format!("Agent access revoked ({dropped} queued commands dropped)"),
                        );
                        Ok(())
                    } else {
                        self.grant_agent()
                    }
                }
                Intent::SavePrefs => self.save_prefs(),
                Intent::ApplyPrefs => {
                    if let Some(p) = self.shell.prefs_draft.take() {
                        self.prefs = p;
                    }
                    self.ui.set_theme(theme_of(&self.prefs));
                    self.agent_server.set_policy(policy_of(&self.prefs.agent));
                    Ok(())
                }
                Intent::MarkLooked => {
                    self.provenance.mark_looked();
                    Ok(())
                }
                Intent::Frame(slot, framing) => {
                    if let Some(v) = self.viewports.get_mut(slot) {
                        frame_viewport(v, self.engine.model(), framing);
                    }
                    Ok(())
                }
            };
            if let Err(e) = result {
                self.report(&e);
            }
        }
        // Keep the vector's memory; intents pushed meanwhile (none today) go after.
        intents.append(&mut self.intents);
        self.intents = intents;
    }

    fn report(&mut self, e: &AppError) {
        let text = e.to_string();
        self.shell.log(LogLevel::Error, self.ui.time(), &text);
        let refused = matches!(e, AppError::Command(CommandError::Rejected(_)));
        self.shell.toast(if refused { ToastKind::Warning } else { ToastKind::Error }, &text);
    }

    /// Saves the project (or asks for a directory when there is none).
    pub fn save(&mut self) -> Result<(), AppError> {
        if self.project.is_none() {
            self.shell.open_browser(crate::shell::BrowseFor::SaveAs, &self.prefs.projects_dir);
            return Ok(());
        }
        self.engine.session_mut().save()?;
        self.shell.log(LogLevel::Info, self.ui.time(), "Saved");
        self.shell.toast(ToastKind::Success, "Saved");
        Ok(())
    }

    /// Writes the preferences, the keymap and the dock layout to the settings directory.
    pub fn save_prefs(&mut self) -> Result<(), AppError> {
        let fs = self.env.fs;
        let dir = &self.env.settings_dir;
        fs.create_dir_all(dir).map_err(|source| AppError::Platform { path: dir.clone(), source })?;
        let write = |name: &str, text: &str| {
            let path = humanart_core::platform::join_path(dir, name);
            fs.write_atomic(&path, text.as_bytes()).map_err(|source| AppError::Platform { path, source })
        };
        write("prefs.json", &self.prefs.to_json())?;
        write("keymap.json", &self.keymap.to_json()?)?;
        if let Some(layout) = self.shell.layout_json() {
            write("layout.json", &layout)?;
        }
        Ok(())
    }

    fn frame_new_viewports(&mut self) {
        for v in &mut self.viewports {
            if v.open && !v.framed && v.rect.w > 0.0 && v.rect.h > 0.0 {
                frame_viewport(v, self.engine.model(), Framing::Body);
                v.framed = true;
            }
        }
    }

    fn update_highlights(&mut self, now: f64) {
        let p = self.ui.theme().palette;
        let model = self.engine.model();
        let selected = self.shell.selected_region.as_deref().and_then(|r| model.region_id(r));
        for list in &mut self.highlights {
            list.clear();
            if let Some(region) = selected {
                let c = p.accent.to_srgb8();
                list.push(Highlight { region, color: srgb(c, 1.0) });
            }
            for (name, f) in self.provenance.region_flashes(now, self.prefs.flash_seconds) {
                if let Some(region) = model.region_id(name) {
                    list.push(Highlight { region, color: srgb(p.warning.to_srgb8(), f) });
                }
            }
        }
    }

    /// The viewports to draw this frame, in physical pixels of a target scaled by `scale` (the
    /// window's scale factor). They are drawn under modal overlays too: the viewport images are
    /// part of the UI pass, so whatever the UI draws later covers them.
    pub fn viewport_frames(&self, scale: f32) -> impl Iterator<Item = ViewportFrame<'_>> + '_ {
        let s = scale * self.prefs.ui_scale;
        self.viewports.iter().enumerate().filter(move |(_, v)| v.open && v.rect.w >= 1.0 && v.rect.h >= 1.0).map(
            move |(slot, v)| {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Non-negative pixel sizes.
                let px = |x: f32| (x * s).round().max(0.0) as u32;
                ViewportFrame {
                    slot,
                    rect: [px(v.rect.x), px(v.rect.y), px(v.rect.w), px(v.rect.h)],
                    settings: v.settings,
                    highlights: self.highlights.get(slot).map_or(&[], Vec::as_slice),
                }
            },
        )
    }

    /// Queues an intent for the next frame (a native dialog's answer, a host command).
    pub fn push_intent(&mut self, intent: Intent) {
        self.intents.push(intent);
    }

    /// Records the frame drawn by the last [`App::frame`] into `target` (format
    /// [`crate::studio::FRAME_FORMAT`], physical pixels of a window at `scale`): the viewports and
    /// the UI. No allocation once the viewers exist.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        studio: &mut crate::studio::Studio,
        gpu: &humanart_gpu::Gpu,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::Texture,
        view: &wgpu::TextureView,
        scale: f32,
    ) -> Result<(), AppError> {
        let s = scale * self.prefs.ui_scale;
        let none: &[Highlight] = &[];
        let mut frames = [ViewportFrame { slot: 0, rect: [0; 4], settings: ViewSettings::default(), highlights: none };
            MAX_VIEWPORTS];
        let mut n = 0;
        for (slot, v) in self.viewports.iter().enumerate() {
            if !v.open || v.rect.w < 1.0 || v.rect.h < 1.0 {
                continue;
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Non-negative pixel sizes.
            let px = |x: f32| (x * s).round().max(0.0) as u32;
            if let Some(f) = frames.get_mut(n) {
                *f = ViewportFrame {
                    slot,
                    rect: [px(v.rect.x), px(v.rect.y), px(v.rect.w), px(v.rect.h)],
                    settings: v.settings,
                    highlights: self.highlights.get(slot).map_or(none, Vec::as_slice),
                };
                n += 1;
            }
        }
        let clear = self.ui.theme().palette.panel;
        let keep = [0, 1, 2, 3].map(|slot| self.viewports.get(slot).is_some_and(|v| v.open));
        studio.retain(&keep);
        studio.record(
            gpu,
            encoder,
            target,
            view,
            &self.ui,
            clear,
            self.engine.model_mut(),
            frames.get(..n).unwrap_or(&[]),
        )
    }

    /// The UI (to draw) and the model (to render), borrowed together for the GPU side.
    pub fn ui_and_model(&mut self) -> (&Ui, &mut Model) {
        (&self.ui, self.engine.model_mut())
    }

    /// The model, for the GPU side.
    pub fn model_mut(&mut self) -> &mut Model {
        self.engine.model_mut()
    }

    /// The clear colour of a frame (the theme's panel colour).
    pub fn clear_color(&self) -> humanart_ui::Color {
        self.ui.theme().palette.panel
    }

    /// Runs an agent command through the agent link (pause, revoke) `[APP-04]`.
    pub fn agent_submit(&mut self, cmd: Command) -> Result<crate::agent_link::Submitted, CommandError> {
        self.agent.submit(self.engine.session_mut(), cmd)
    }

    /// Registers an agent author on the document so its commands are accepted.
    pub fn add_agent_author(&mut self, id: &str, name: &str) -> Result<(), CommandError> {
        if self.engine.session().document().author(id).is_some() {
            return Ok(());
        }
        let human = self.env.human.id.clone();
        let author = Author { id: id.to_owned(), kind: AuthorKind::Agent, name: name.to_owned() };
        self.engine.execute(Command::new(&human, humanart_doc::command::Action::AddAuthor { author })).map(|_| ())
    }
}

fn srgb(c: [u8; 4], alpha: f32) -> [f32; 4] {
    [f32::from(c[0]) / 255.0, f32::from(c[1]) / 255.0, f32::from(c[2]) / 255.0, alpha]
}

/// Points a viewport's camera at the model with `framing`, keeping its exposure.
pub fn frame_viewport(v: &mut ViewportState, model: &Model, framing: Framing) {
    let aspect = if v.rect.h > 0.0 { v.rect.w / v.rect.h } else { 1.0 };
    let exposure = v.settings.camera.exposure;
    v.settings.camera = frame_camera(model, framing, aspect);
    v.settings.camera.exposure = exposure;
}
