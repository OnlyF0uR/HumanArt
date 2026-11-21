//! The editor's window content `[APP-01]`: toolbar with the mode selector, the docking area with
//! its panels, the status bar, and the overlays (quick start, review tour, preferences, file
//! browser, toasts).
//!
//! Panels (stable ids, saved with the layout): `viewport.1` ... `viewport.4` (`[VP-04]`),
//! `outliner` (regions), `properties` (the registry-driven property grid with search, region
//! filter, author badges and change flashes, `[REG-04]`, `[APP-03]`), `history` (authorship,
//! before/after preview, undo, redo, undo the agent's last, `[APP-06]`), `console`, `agent`
//! (`[APP-04]`), `proposals`, `compare`, `profiler`, `data`, `export`, `modes` (the tools of the
//! current mode). The shell only reads the document: every change is an [`Intent`] applied by
//! [`crate::app::App`] after layout.

use std::fmt::Write as _;

use humanart_agent::engine::{Framing, Model};
use humanart_core::collections::ArrayString;
use humanart_core::platform::{DirEntry, FileKind, FileSystem};
use humanart_doc::command::{Action, Command, RangePolicy};
use humanart_doc::document::LockTarget;
use humanart_doc::history::Scope;
use humanart_doc::registry::{Param, ParamKind, ParamValue, Registry};
use humanart_doc::session::Session;
use humanart_model::Template;
use humanart_render::camera::ReviewCamera;
use humanart_render::light::BuiltinRig;
use humanart_render::renderer::{DebugView, Quality};
use humanart_ui::Color;
use humanart_ui::controls::{
    Collapsible, IconButton, begin_status_bar, begin_toolbar, checkbox, search_box, status_item, tabs, toggle,
    toolbar_separator,
};
use humanart_ui::dock::{DockLayout, DockNode, SplitAxis, dock_area};
use humanart_ui::editors::color_edit;
use humanart_ui::geom::{Insets, Rect};
use humanart_ui::icons;
use humanart_ui::layout::{Align as BoxAlign, Dim, Style};
use humanart_ui::paint::Corners;
use humanart_ui::popup::{Dropdown, Modal, ToastKind, Toasts};
use humanart_ui::text::Align;
use humanart_ui::ui::{Anchor, Area, Paint, Ui};
use humanart_ui::views::{ConsoleLog, LogLevel, list_view};
use humanart_ui::widgets::{Button, ButtonKind, EditPhase, Slider, TextField, label, label_styled};

use crate::agent_link::AgentLink;
use crate::app::{Intent, ViewportState};
use crate::modes::Mode;
use crate::prefs::{Prefs, QualityChoice, ThemeChoice};
use crate::provenance::{Badge, Provenance, badge};
use crate::studio::MAX_VIEWPORTS;

/// Every panel id the shell knows.
pub const PANELS: &[&str] = &[
    "viewport.1",
    "viewport.2",
    "viewport.3",
    "viewport.4",
    "outliner",
    "properties",
    "modes",
    "history",
    "console",
    "agent",
    "proposals",
    "compare",
    "profiler",
    "data",
    "export",
];

/// The review tour `[APP-08]`: the rubric scenes, a review camera under a built-in rig each.
pub const TOUR: &[(ReviewCamera, BuiltinRig, &str)] = &[
    (ReviewCamera::FullBody, BuiltinRig::NeutralStudio, "Proportions: the whole body, front, neutral studio light."),
    (ReviewCamera::Front, BuiltinRig::FlatBeauty, "Face front under flat beauty light: symmetry and features."),
    (ReviewCamera::ThreeQuarterLeft, BuiltinRig::NeutralStudio, "Three-quarter view: cheek and jaw volume."),
    (ReviewCamera::ProfileLeft, BuiltinRig::HarshSidelight, "Profile under a hard side light: nose, lips, chin."),
    (ReviewCamera::Front, BuiltinRig::Backlight, "Backlight: ear and silhouette translucency."),
    (ReviewCamera::Front, BuiltinRig::OvercastOutdoor, "Overcast daylight: skin tone without highlights."),
];

/// The rig names of the Light & Camera tools, in [`BuiltinRig::ALL`] order.
const RIG_NAMES: [&str; 7] =
    ["Neutral studio", "Flat beauty", "Harsh sidelight", "Backlight", "Overcast outdoor", "Golden hour", "Reference"];
const QUALITY_NAMES: [&str; 3] = ["Interactive", "Review", "Still"];
const QUALITIES: [Quality; 3] = [Quality::Interactive, Quality::Review, Quality::Still];
/// The mode selector's labels, in [`Mode::ALL`] order.
const LABELS: [&str; 10] =
    ["Shape", "Skin", "Eyes & Mouth", "Hair", "Expression", "Pose", "Light & Camera", "Compare", "Review", "Export"];
/// The view modes, in [`DebugView::ALL`] order.
const VIEWS: [&str; 15] = [
    "Beauty",
    "Albedo",
    "Diffuse",
    "Specular",
    "Irradiance",
    "Normals",
    "Roughness",
    "Metalness",
    "Thickness",
    "Occlusion",
    "Material class",
    "Depth",
    "Motion vectors",
    "Shadow map",
    "False colour",
];
const THEME_NAMES: [&str; 3] = ["Dark", "Light", "High contrast"];

/// What the file browser picks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrowseFor {
    /// A project directory to open.
    Open,
    /// A new project directory to save into.
    SaveAs,
}

/// The in-app file browser `[APP-07]` (the fallback when no native dialog is available).
#[derive(Debug)]
pub struct Browser {
    /// What it picks.
    pub purpose: BrowseFor,
    /// The directory shown.
    pub dir: String,
    /// Its entries (directories first), read when the directory changes.
    pub entries: Vec<DirEntry>,
    /// The selected row.
    pub selected: Option<usize>,
    /// The new directory's name (save as).
    pub name: String,
    /// The last listing error.
    pub error: Option<String>,
}

impl Browser {
    fn read(&mut self, fs: &dyn FileSystem) {
        self.selected = None;
        match fs.list(&self.dir) {
            Ok(mut list) => {
                list.sort_by(|a, b| {
                    (b.kind == FileKind::Directory).cmp(&(a.kind == FileKind::Directory)).then(a.name.cmp(&b.name))
                });
                self.entries = list;
                self.error = None;
            }
            Err(e) => {
                self.entries.clear();
                self.error = Some(e.to_string());
            }
        }
    }
}

/// What the panels read (the shell never changes the document itself).
pub struct Ctx<'c, 's> {
    /// The session.
    pub session: &'c Session<'s>,
    /// The registry.
    pub registry: &'c Registry,
    /// The evaluated character.
    pub model: &'c Model,
    /// The provenance state.
    pub provenance: &'c Provenance,
    /// The agent link.
    pub agent: &'c AgentLink,
    /// The agent server's address and bearer token while it runs.
    pub agent_server: Option<(&'c str, &'c str)>,
    /// The preferences.
    pub prefs: &'c Prefs,
    /// The person at the editor.
    pub human: &'c str,
    /// The open project.
    pub project: Option<&'c str>,
    /// The file system.
    pub fs: &'c dyn FileSystem,
    /// The UI clock, seconds.
    pub now: f64,
}

/// Pass timings of a viewport, filled by the host from the renderer `[PERF-R6]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Profile {
    /// Frame CPU time of the whole editor frame, milliseconds.
    pub frame_ms: f32,
    /// Pass names and GPU milliseconds (when timed).
    pub passes: [(&'static str, Option<f32>); 24],
    /// Entries used.
    pub count: usize,
}

/// The shell's UI state (see the module documentation).
pub struct Shell {
    /// The dock layout.
    pub layout: DockLayout,
    /// The current mode.
    pub mode: Mode,
    /// The parameter search text.
    pub search: String,
    /// Put the keyboard focus into the search box next frame.
    pub focus_search: bool,
    /// Show only values changed since the last look.
    pub changed_only: bool,
    /// The region the human selected (name).
    pub selected_region: Option<String>,
    /// Limit the property grid to the selected region.
    pub region_filter: bool,
    /// The viewport keyboard framing applies to.
    pub active_viewport: usize,
    /// The quick-start overlay is open `[APP-08]`.
    pub onboarding: bool,
    /// The review tour's step, when it runs.
    pub tour: Option<usize>,
    /// The preferences being edited.
    pub prefs_draft: Option<Prefs>,
    /// The file browser, when open.
    pub browser: Option<Browser>,
    /// Pass timings for the profiler panel.
    pub profile: Profile,
    console: ConsoleLog,
    toasts: Toasts,
    regions: Vec<(String, u16)>,
    region_row: Option<usize>,
    choices: Vec<(usize, Vec<String>)>,
    history_row: Option<usize>,
    preview: Option<(usize, String)>,
    pose_joint: usize,
    joints: Vec<String>,
    data_dir: Option<(String, Vec<DirEntry>)>,
    data_row: Option<usize>,
    proposal_row: Option<usize>,
    spare: Option<DockLayout>,
    open_slot: Option<usize>,
}

impl std::fmt::Debug for Shell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shell").field("mode", &self.mode).field("search", &self.search).finish_non_exhaustive()
    }
}

/// The default workspace `[APP-01]`: outliner left, the viewport in the middle over history,
/// console, agent and proposals, properties and the mode tools right.
pub fn default_layout() -> DockLayout {
    DockLayout::from_root(DockNode::split(
        SplitAxis::Row,
        0.17,
        DockNode::tabs(&["outliner", "data"]),
        DockNode::split(
            SplitAxis::Row,
            0.70,
            DockNode::split(
                SplitAxis::Column,
                0.72,
                DockNode::tabs(&["viewport.1"]),
                DockNode::tabs(&["history", "console", "agent", "proposals", "profiler"]),
            ),
            DockNode::split(
                SplitAxis::Column,
                0.62,
                DockNode::tabs(&["properties", "compare", "export"]),
                DockNode::tabs(&["modes"]),
            ),
        ),
    ))
}

fn title(panel: &str) -> &str {
    match panel {
        "viewport.1" => "Viewport 1",
        "viewport.2" => "Viewport 2",
        "viewport.3" => "Viewport 3",
        "viewport.4" => "Viewport 4",
        "outliner" => "Regions",
        "properties" => "Properties",
        "modes" => "Tools",
        "history" => "History",
        "console" => "Console",
        "agent" => "Agent",
        "proposals" => "Proposals",
        "compare" => "Compare",
        "profiler" => "Profiler",
        "data" => "Data",
        "export" => "Export",
        other => other,
    }
}

/// The viewport slot of a panel id.
fn viewport_slot(panel: &str) -> Option<usize> {
    panel
        .strip_prefix("viewport.")
        .and_then(|n| n.parse::<usize>().ok())
        .and_then(|n| n.checked_sub(1))
        .filter(|&s| s < MAX_VIEWPORTS)
}

fn param_cmd(human: &str, id: &str, value: ParamValue) -> Command {
    Command::new(
        human,
        Action::SetParams { values: vec![(id.to_owned(), value)], relative: false, policy: RangePolicy::Clamp },
    )
}

#[allow(clippy::cast_possible_truncation)] // UI values are f32.
fn f(v: f64) -> f32 {
    v as f32
}

impl Shell {
    /// The shell for `registry` and `template`, in the default workspace.
    pub fn new(registry: &Registry, template: &Template) -> Self {
        let r = template.regions();
        let regions = r
            .regions
            .iter()
            .map(|reg| {
                let mut depth = 0_u16;
                let mut parent = reg.parent;
                while let Some(p) = parent {
                    depth += 1;
                    parent = r.regions.get(usize::from(p)).and_then(|x| x.parent);
                    if depth > 64 {
                        break;
                    }
                }
                (reg.name.clone(), depth)
            })
            .collect();
        let choices = registry
            .params()
            .iter()
            .enumerate()
            .filter(|(_, p)| p.kind == ParamKind::Enum)
            .map(|(i, p)| (i, p.choices.clone()))
            .collect();
        Self {
            layout: default_layout(),
            mode: Mode::Shape,
            search: String::with_capacity(64),
            focus_search: false,
            changed_only: false,
            selected_region: None,
            region_filter: false,
            active_viewport: 0,
            onboarding: false,
            tour: None,
            prefs_draft: None,
            browser: None,
            profile: Profile::default(),
            console: ConsoleLog::new(),
            toasts: Toasts::new(),
            regions,
            region_row: None,
            choices,
            history_row: None,
            preview: None,
            pose_joint: 0,
            joints: template.skeleton().joints.iter().map(|j| j.name.clone()).collect(),
            data_dir: None,
            data_row: None,
            proposal_row: None,
            spare: None,
            open_slot: None,
        }
    }

    /// Replaces the layout with a saved one (unknown panels or a damaged file keep the default).
    pub fn restore_layout(&mut self, bytes: &[u8]) {
        if let Ok(text) = std::str::from_utf8(bytes)
            && let Ok(layout) = DockLayout::from_json(text, PANELS)
        {
            self.layout = layout;
        }
    }

    /// The layout as JSON.
    pub fn layout_json(&self) -> Option<String> {
        self.layout.to_json().ok()
    }

    /// Adds a console line.
    pub fn log(&mut self, level: LogLevel, time: f64, text: &str) {
        self.console.push(level, time, text);
    }

    /// The console.
    pub fn console(&self) -> &ConsoleLog {
        &self.console
    }

    /// Shows a toast.
    pub fn toast(&mut self, kind: ToastKind, text: &str) {
        self.toasts.push(kind, text, 4.0);
    }

    /// Opens the preferences dialog on a copy of `prefs`.
    pub fn open_prefs(&mut self, prefs: &Prefs) {
        self.prefs_draft = Some(prefs.clone());
    }

    /// Opens the in-app file browser at `dir`.
    pub fn open_browser(&mut self, purpose: BrowseFor, dir: &str) {
        self.browser = Some(Browser {
            purpose,
            dir: dir.to_owned(),
            entries: Vec::new(),
            selected: None,
            name: String::new(),
            error: Some(String::new()),
        });
    }

    /// Whether a modal overlay is open (quick start, tour, preferences, file browser).
    pub fn overlay_open(&self) -> bool {
        self.onboarding || self.tour.is_some() || self.prefs_draft.is_some() || self.browser.is_some()
    }

    /// Opens a viewport panel (1-4 viewports `[APP-01]`).
    pub fn open_viewport(&mut self, slot: usize) {
        let mut id = ArrayString::<16>::new();
        let _ = write!(id, "viewport.{}", slot + 1);
        if slot < MAX_VIEWPORTS && !self.layout.contains(id.as_str()) {
            self.layout.ensure(id.as_str());
        }
    }

    /// Reads the viewports' image rectangles after layout.
    pub fn read_layout(&self, ui: &Ui, viewports: &mut [ViewportState; MAX_VIEWPORTS]) {
        for (slot, v) in viewports.iter_mut().enumerate() {
            let mut id = ArrayString::<16>::new();
            let _ = write!(id, "viewport.{}", slot + 1);
            v.open = self.layout.contains(id.as_str()) && self.layout.active_of(id.as_str()) == Some(id.as_str());
            let image = image_id(id.as_str());
            v.rect = if v.open { ui.state(image).map_or(Rect::default(), |s| s.rect) } else { Rect::default() };
        }
    }

    /// Builds the frame's UI (between `begin_frame` and `end_frame`).
    pub fn build(
        &mut self,
        ui: &mut Ui,
        ctx: &Ctx<'_, '_>,
        viewports: &mut [ViewportState; MAX_VIEWPORTS],
        intents: &mut Vec<Intent>,
    ) {
        let theme = *ui.theme();
        let p = theme.palette;
        ui.begin(Style { width: Dim::Percent(100.0), height: Dim::Percent(100.0), ..Style::column() });
        self.toolbar(ui, ctx, intents);
        ui.begin(Style { grow: 1.0, basis: Dim::Px(0.0), ..Style::column() });
        // The layout leaves `self` while the panels are built (they borrow the rest of the shell);
        // a placeholder kept from the last frame takes its place, so nothing is allocated.
        let placeholder = self.spare.take().unwrap_or_else(|| DockLayout::new(&[]));
        let mut layout = std::mem::replace(&mut self.layout, placeholder);
        dock_area(ui, "Workspace", &mut layout, &title, &mut |ui: &mut Ui, panel: &str| {
            self.panel(ui, panel, ctx, viewports, intents);
        });
        self.spare = Some(std::mem::replace(&mut self.layout, layout));
        if let Some(slot) = self.open_slot.take() {
            self.open_viewport(slot);
        }
        ui.end();
        begin_status_bar(ui);
        let (icon, text, color) = if ctx.agent.is_revoked() {
            (icons::LOCK, "Agent revoked", p.danger)
        } else if ctx.agent.is_paused() {
            (icons::PAUSE, "Agent paused", p.warning)
        } else {
            (icons::CHECK_CIRCLE, ctx.agent.status().label(), p.positive)
        };
        status_item(ui, Some(icon), text, Some(color));
        let mut s = ArrayString::<96>::new();
        let _ = write!(s, "{} changed since last look", ctx.provenance.changed_count());
        status_item(ui, None, s.as_str(), None);
        ui.node(Style { grow: 1.0, ..Style::default() });
        status_item(ui, None, ctx.project.unwrap_or("Unsaved sample"), None);
        status_item(ui, None, self.mode.label(), None);
        ui.end();
        ui.end();
        self.overlays(ui, ctx, viewports, intents);
        let top = theme.metrics.control_height * 2.0;
        self.toasts.show(ui, top);
    }

    fn toolbar(&mut self, ui: &mut Ui, ctx: &Ctx<'_, '_>, intents: &mut Vec<Intent>) {
        begin_toolbar(ui);
        label_styled(ui, "HumanArt", ui.theme().text.label, ui.theme().palette.text);
        toolbar_separator(ui);
        let mut index = Mode::ALL.iter().position(|m| *m == self.mode).unwrap_or(0);
        if tabs(ui, "Mode", &mut index, &LABELS).changed
            && let Some(m) = Mode::ALL.get(index)
        {
            intents.push(Intent::Mode(*m));
        }
        ui.node(Style { grow: 1.0, ..Style::default() });
        let history = ctx.session.history();
        if IconButton::new(icons::CHEVRON_LEFT, "Undo (Ctrl+Z)").enabled(history.cursor() > 0).show(ui).clicked {
            intents.push(Intent::Undo(Scope::Any));
        }
        let redo = history.cursor() < history.entries().len();
        if IconButton::new(icons::CHEVRON_RIGHT, "Redo (Ctrl+Y)").enabled(redo).show(ui).clicked {
            intents.push(Intent::Redo);
        }
        if IconButton::new(icons::FILE, "Save (Ctrl+S)").show(ui).clicked {
            intents.push(Intent::Save);
        }
        if IconButton::new(icons::FOLDER, "Open project").show(ui).clicked {
            self.open_browser(BrowseFor::Open, &ctx.prefs.projects_dir);
        }
        toolbar_separator(ui);
        if IconButton::new(icons::SETTINGS, "Preferences (Ctrl+P)").show(ui).clicked {
            self.open_prefs(ctx.prefs);
        }
        if IconButton::new(icons::INFO, "Quick start (F1)").selected(self.onboarding).show(ui).clicked {
            self.onboarding = !self.onboarding;
        }
        ui.end();
    }

    fn panel(
        &mut self,
        ui: &mut Ui,
        panel: &str,
        ctx: &Ctx<'_, '_>,
        viewports: &mut [ViewportState; MAX_VIEWPORTS],
        intents: &mut Vec<Intent>,
    ) {
        if let Some(slot) = viewport_slot(panel) {
            if let Some(v) = viewports.get_mut(slot) {
                self.viewport(ui, panel, slot, v, ctx, intents);
            }
            return;
        }
        let m = ui.theme().metrics;
        let id = ui.id(panel).with("scroll");
        ui.begin_scroll(id, Style { grow: 1.0, padding: Insets::all(m.space_4), gap: m.space_3, ..Style::column() });
        match panel {
            "outliner" => self.outliner(ui, intents),
            "properties" => self.properties(ui, ctx, intents),
            "modes" => self.mode_tools(ui, ctx, viewports, intents),
            "history" => self.history(ui, ctx, intents),
            "console" => self.console.show(ui, "Console", 220.0),
            "agent" => agent_panel(ui, ctx, intents),
            "proposals" => self.proposals(ui, ctx, intents),
            "compare" => compare_panel(ui, ctx),
            "profiler" => self.profiler(ui),
            "data" => self.data(ui, ctx),
            "export" => self.export(ui, ctx, intents),
            _ => {
                label(ui, "Unknown panel");
            }
        }
        ui.end_scroll(id);
    }

    fn viewport(
        &mut self,
        ui: &mut Ui,
        panel: &str,
        slot: usize,
        v: &mut ViewportState,
        ctx: &Ctx<'_, '_>,
        intents: &mut Vec<Intent>,
    ) {
        let theme = *ui.theme();
        let (m, p) = (theme.metrics, theme.palette);
        ui.begin(Style { grow: 1.0, ..Style::column() });
        ui.begin(Style {
            gap: m.space_2,
            padding: Insets::xy(m.space_3, m.space_2),
            align_items: BoxAlign::Center,
            ..Style::row()
        });
        ui.push_id(ui.id(panel));
        for (icon, tip, framing) in [
            (icons::MAXIMIZE, "Frame the body (Home)", Framing::Body),
            (icons::ZOOM_IN, "Frame the head (F)", Framing::Head),
            (icons::ROWS, "Three-quarter view", Framing::ThreeQuarter),
            (icons::COLUMNS, "Profile view", Framing::Profile),
        ] {
            if IconButton::new(icon, tip).show(ui).clicked {
                intents.push(Intent::Frame(slot, framing));
            }
        }
        // The view and rig are chosen in the Light & Camera tools; the label shows them.
        let view = DebugView::ALL.iter().position(|d| *d == v.settings.debug).and_then(|i| VIEWS.get(i)).unwrap_or(&"");
        let rig =
            BuiltinRig::ALL.iter().position(|r| *r == v.settings.rig).and_then(|i| RIG_NAMES.get(i)).unwrap_or(&"");
        let mut s = ArrayString::<64>::new();
        let _ = write!(s, "{view} / {rig}");
        label_styled(ui, s.as_str(), theme.text.small, p.text_muted);
        if IconButton::new(icons::BOX, "Wireframe").selected(v.settings.wireframe).show(ui).clicked {
            v.settings.wireframe = !v.settings.wireframe;
        }
        ui.pop_id();
        ui.end();
        // The image: the GPU side copies the viewer's frame into the slot's UI image, drawn here
        // in the UI's z-order (popups and tooltips cover it). The surround shows until the first
        // frame of the viewer.
        let image = ui.node(Style { grow: 1.0, basis: Dim::Px(0.0), ..Style::default() });
        ui.paint(
            image,
            Paint::Rect {
                area: Area::FULL,
                fill: p.viewport_surround,
                radius: Corners::all(0.0),
                border: 0.0,
                border_color: Color::TRANSPARENT,
            },
        );
        ui.paint(
            image,
            Paint::Image {
                area: Area::FULL,
                texture: crate::studio::viewport_image(slot),
                uv: [0.0, 0.0, 1.0, 1.0],
                tint: Color::WHITE,
                radius: 0.0,
            },
        );
        let iid = image_id(panel);
        let r = ui.interact(iid, image, humanart_ui::ui::Sense::DRAG);
        if r.hovered || r.dragged {
            self.active_viewport = slot;
        }
        if r.dragged {
            let d = r.drag_delta;
            if ui.modifiers().shift {
                let dist = (v.settings.camera.position - v.settings.camera.target).length();
                let k = dist * 0.0015;
                v.settings.camera.pan(-d.x * k, d.y * k);
            } else {
                v.settings.camera.orbit(-d.x * 0.008, d.y * 0.008);
            }
        }
        let wheel = ui.wheel();
        if r.hovered && wheel.y != 0.0 {
            v.settings.camera.dolly(0.9_f32.powf(wheel.y.clamp(-10.0, 10.0)));
        }
        if r.clicked
            && let Some(pt) = ui.pointer()
        {
            // CPU fallback picking `[VP-01]`: the ray through the clicked pixel.
            let (w, h) = (r.rect.w.max(1.0), r.rect.h.max(1.0));
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Logical sizes.
            let (origin, dir) = humanart_agent::engine::view::pixel_ray(
                &v.settings.camera,
                pt.x - r.rect.x,
                pt.y - r.rect.y,
                w.round() as u32,
                h.round() as u32,
            );
            let region = ctx
                .model
                .pick(origin, dir)
                .and_then(|hit| ctx.model.template().regions().regions.get(usize::from(hit.region)))
                .map(|reg| reg.name.clone());
            intents.push(Intent::Select(region));
        }
        ui.semantic(iid, humanart_ui::ui::Role::Image, title(panel), humanart_ui::ui::SemanticValue::None);
        ui.end();
    }

    fn outliner(&mut self, ui: &mut Ui, intents: &mut Vec<Intent>) {
        let regions = &self.regions;
        let selected = self.selected_region.as_deref();
        self.region_row = selected.and_then(|s| regions.iter().position(|(n, _)| n == s));
        let mut row = self.region_row;
        let r = list_view(ui, "Regions", regions.len(), 420.0, &mut row, &|i, out| {
            if let Some((name, depth)) = regions.get(i) {
                for _ in 0..*depth {
                    let _ = out.write_str("   ");
                }
                let _ = out.write_str(name);
            }
        });
        if r.selection_changed {
            intents.push(Intent::Select(row.and_then(|i| regions.get(i)).map(|(n, _)| n.clone())));
        }
        if self.selected_region.is_some() && Button::new("Clear selection").kind(ButtonKind::Ghost).show(ui).clicked {
            intents.push(Intent::Select(None));
        }
    }

    fn properties(&mut self, ui: &mut Ui, ctx: &Ctx<'_, '_>, intents: &mut Vec<Intent>) {
        let theme = *ui.theme();
        let m = theme.metrics;
        let search_id = ui.id("Search parameters");
        if self.focus_search {
            ui.request_focus(search_id);
            self.focus_search = false;
        }
        search_box(ui, "Search parameters", &mut self.search, "Search all parameters (name, id, tag)");
        ui.begin(Style { gap: m.space_4, align_items: BoxAlign::Center, ..Style::row() });
        checkbox(ui, "Changed since I last looked", &mut self.changed_only);
        if let Some(region) = self.selected_region.as_deref() {
            let mut s = ArrayString::<96>::new();
            let _ = write!(s, "Only {region}");
            let mut on = self.region_filter;
            if checkbox(ui, s.as_str(), &mut on).changed {
                self.region_filter = on;
            }
        }
        ui.end();
        let searching = !self.search.trim().is_empty();
        let mut shown = 0_usize;
        let params = ctx.registry.params();
        let mut start = 0;
        while start < params.len() {
            let group = params[start].group.as_str();
            let end = params[start..].iter().position(|p| p.group != group).map_or(params.len(), |k| start + k);
            let visible = |p: &Param| self.visible(p, ctx, searching);
            if params[start..end].iter().any(visible) {
                Collapsible::new(group).default_open(true).show(ui, |ui| {
                    for (i, p) in params.iter().enumerate().take(end).skip(start) {
                        if self.visible(p, ctx, searching) {
                            shown += 1;
                            self.param_row(ui, ctx, i, p, intents);
                        }
                    }
                });
            }
            start = end;
        }
        if shown == 0 {
            label_styled(ui, "No parameters match", theme.text.small, theme.palette.text_faint);
        }
        if self.changed_only && Button::new("Mark all as seen").kind(ButtonKind::Secondary).show(ui).clicked {
            self.changed_only = false;
            intents.push(Intent::MarkLooked);
        }
    }

    fn visible(&self, p: &Param, ctx: &Ctx<'_, '_>, searching: bool) -> bool {
        let f = self.search.trim();
        let matches = if searching {
            use humanart_ui::popup::contains_ignore_case as has;
            has(&p.display_name, f) || has(&p.id, f) || has(&p.group, f) || p.tags.iter().any(|t| has(t, f))
        } else {
            self.mode.shows_group(&p.group)
        };
        let region_ok =
            !self.region_filter || self.selected_region.as_deref().is_none_or(|r| p.region.as_deref() == Some(r));
        let changed_ok = !self.changed_only || ctx.provenance.changed_since_look(&p.id);
        matches && region_ok && changed_ok
    }

    fn param_row(&mut self, ui: &mut Ui, ctx: &Ctx<'_, '_>, index: usize, p: &Param, intents: &mut Vec<Intent>) {
        let theme = *ui.theme();
        let (m, pal) = (theme.metrics, theme.palette);
        let doc = ctx.session.document();
        let row = ui.begin(Style {
            gap: m.space_3,
            align_items: BoxAlign::Center,
            min_height: m.control_height,
            padding: Insets::xy(m.space_2, 0.0),
            ..Style::row()
        });
        let flash = ctx.provenance.flash(&p.id, ctx.now, ctx.prefs.flash_seconds);
        if flash > 0.0 {
            ui.paint(
                row,
                Paint::Rect {
                    area: Area::FULL,
                    fill: pal.warning.with_alpha(0.35 * flash),
                    radius: Corners::all(m.radius_sm),
                    border: 0.0,
                    border_color: Color::TRANSPARENT,
                },
            );
        }
        ui.push_id(ui.id(&p.id));
        badge_node(ui, badge(doc, &p.id));
        let name = ui.node(Style {
            width: Dim::Px(130.0),
            height: Dim::Px(m.control_height),
            shrink: 0.0,
            ..Style::default()
        });
        let nid = ui.id("name");
        let nr = ui.interact(nid, name, humanart_ui::ui::Sense::HOVER);
        ui.semantic(nid, humanart_ui::ui::Role::Label, &p.display_name, humanart_ui::ui::SemanticValue::None);
        let t = ui.intern(&p.display_name);
        let color = if ctx.provenance.changed_since_look(&p.id) { pal.text } else { pal.text_muted };
        ui.paint(
            name,
            Paint::Text { area: Area::FULL, text: t, style: theme.text.small, color, align: Align::Start, wrap: false },
        );
        humanart_ui::popup::tooltip(ui, &nr, &humanart_ui::popup::Tip::new(&p.display_name).text(&p.description));
        let value = doc.params.get(&p.id).unwrap_or(&p.default);
        let locked = doc.locks.iter().any(|l| matches!(&l.target, LockTarget::Parameter(id) if *id == p.id));
        let human = ctx.human;
        match (p.kind, value) {
            _ if p.read_only => {
                let mut s = ArrayString::<64>::new();
                write_value(&mut s, value);
                let _ = s.write_str("  (measured)");
                label_styled(ui, s.as_str(), theme.text.small, pal.text_faint);
            }
            (ParamKind::Scalar, ParamValue::Scalar(v)) => {
                let (min, max, soft, step) = p.range.as_ref().map_or((-1.0, 1.0, None, 0.01), |r| {
                    (f(r.min), f(r.max), Some((f(r.soft_min), f(r.soft_max))), f(r.step))
                });
                let mut x = f(*v);
                let mut lock = locked;
                let default = match &p.default {
                    ParamValue::Scalar(d) => f(*d),
                    _ => 0.0,
                };
                let mut s = Slider::new(&p.display_name, &mut x, min, max)
                    .label_width(0.0)
                    .default_value(default)
                    .lock(&mut lock);
                if step > 0.0 {
                    s = s.step(step);
                }
                if let Some((a, b)) = soft {
                    s = s.soft_range(a, b);
                }
                let r = s.show(ui);
                match r.phase {
                    // [PERF-R1] drag frames go through the session's allocation-free path.
                    EditPhase::Begin | EditPhase::Update if r.response.changed => {
                        intents.push(Intent::Drag(index, f64::from(x)));
                    }
                    EditPhase::Commit => {
                        if r.response.changed {
                            intents.push(Intent::Drag(index, f64::from(x)));
                        }
                        intents.push(Intent::EndGesture);
                    }
                    EditPhase::Step => {
                        intents.push(Intent::Execute(param_cmd(human, &p.id, ParamValue::Scalar(f64::from(x)))));
                    }
                    _ => {}
                }
                if lock != locked {
                    intents.push(Intent::Execute(Command::new(
                        human,
                        Action::SetLock { target: LockTarget::Parameter(p.id.clone()), locked: lock },
                    )));
                }
            }
            (ParamKind::Bool, ParamValue::Bool(b)) => {
                let mut on = *b;
                if toggle(ui, &p.display_name, &mut on).changed {
                    intents.push(Intent::Execute(param_cmd(human, &p.id, ParamValue::Bool(on))));
                }
            }
            (ParamKind::Enum, ParamValue::Enum(current)) => {
                if let Some((_, list)) = self.choices.iter().find(|(i, _)| *i == index) {
                    let mut sel = list.iter().position(|c| c == current).unwrap_or(0);
                    let mut names = [""; 32];
                    for (slot, c) in names.iter_mut().zip(list) {
                        *slot = c.as_str();
                    }
                    let n = list.len().min(names.len());
                    if Dropdown::new(&p.display_name, &mut sel, &names[..n]).show(ui).changed
                        && let Some(c) = list.get(sel)
                    {
                        intents.push(Intent::Execute(param_cmd(human, &p.id, ParamValue::Enum(c.clone()))));
                    }
                }
            }
            (ParamKind::Color, ParamValue::Color(c)) => {
                let mut col = Color::linear(f(c[0]), f(c[1]), f(c[2]), 1.0);
                let r = color_edit(ui, &p.display_name, &mut col, None, &[]);
                if r.changed {
                    let v = ParamValue::Color([f64::from(col.r), f64::from(col.g), f64::from(col.b)]);
                    let cmd = param_cmd(human, &p.id, v);
                    intents.push(Intent::Execute(if r.phase == EditPhase::Step { cmd } else { cmd.in_gesture() }));
                }
                if r.phase == EditPhase::Commit {
                    intents.push(Intent::EndGesture);
                }
            }
            _ => {
                let mut s = ArrayString::<64>::new();
                write_value(&mut s, value);
                label_styled(ui, s.as_str(), theme.text.small, pal.text_faint);
            }
        }
        ui.pop_id();
        ui.end();
    }

    fn mode_tools(
        &mut self,
        ui: &mut Ui,
        ctx: &Ctx<'_, '_>,
        viewports: &mut [ViewportState; MAX_VIEWPORTS],
        intents: &mut Vec<Intent>,
    ) {
        let theme = *ui.theme();
        label_styled(ui, self.mode.label(), theme.text.heading, theme.palette.text);
        let slot = self.active_viewport.min(MAX_VIEWPORTS - 1);
        match self.mode {
            Mode::Shape | Mode::Skin | Mode::EyesMouth | Mode::Hair | Mode::Expression => {
                let c = ctx.model.coverage();
                let mut s = ArrayString::<160>::new();
                let _ = write!(s, "Bound to the template: {} shape, {} expression", c.shape, c.expression);
                label_styled(ui, s.as_str(), theme.text.small, theme.palette.text_muted);
                let mut s = ArrayString::<160>::new();
                let _ = write!(s, "Not in the template yet: {}", c.missing);
                label_styled(ui, s.as_str(), theme.text.small, theme.palette.text_muted);
                label_styled(ui, "Ctrl+F searches every parameter.", theme.text.small, theme.palette.text_faint);
            }
            Mode::Pose => self.pose_tools(ui, ctx, intents),
            Mode::LightCamera => {
                if let Some(v) = viewports.get_mut(slot) {
                    light_camera_tools(ui, v);
                }
                ui.begin(Style { gap: theme.metrics.space_2, ..Style::row() });
                for s in 0..MAX_VIEWPORTS {
                    let mut name = ArrayString::<16>::new();
                    let _ = write!(name, "Viewport {}", s + 1);
                    if Button::new(name.as_str()).kind(ButtonKind::Secondary).show(ui).clicked {
                        self.open_slot = Some(s);
                    }
                }
                ui.end();
            }
            Mode::Review => {
                for (i, (camera, rig, text)) in TOUR.iter().enumerate() {
                    ui.push_id(ui.id("scene").with_index(i as u64));
                    if Button::new(camera.name()).kind(ButtonKind::Secondary).show(ui).clicked
                        && let Some(v) = viewports.get_mut(slot)
                    {
                        review_scene(v, ctx.model, *camera, *rig);
                    }
                    label_styled(ui, text, theme.text.small, theme.palette.text_muted);
                    ui.pop_id();
                }
                if Button::new("Start the review tour").kind(ButtonKind::Primary).show(ui).clicked {
                    self.tour = Some(0);
                }
            }
            Mode::Compare => compare_panel(ui, ctx),
            Mode::Export => self.export(ui, ctx, intents),
        }
    }

    fn pose_tools(&mut self, ui: &mut Ui, ctx: &Ctx<'_, '_>, intents: &mut Vec<Intent>) {
        let theme = *ui.theme();
        if self.joints.is_empty() {
            label(ui, "The template has no joints");
            return;
        }
        let mut names = [""; 128];
        for (slot, j) in names.iter_mut().zip(&self.joints) {
            *slot = j.as_str();
        }
        let n = self.joints.len().min(names.len());
        Dropdown::new("Joint", &mut self.pose_joint, &names[..n]).show(ui);
        let Some(joint) = self.joints.get(self.pose_joint) else { return };
        let q = ctx.session.document().pose.get(joint).copied().unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let mut euler = quat_to_euler_deg(q);
        let before = euler;
        let mut phase = EditPhase::Idle;
        for (i, axis) in ["X (degrees)", "Y (degrees)", "Z (degrees)"].iter().enumerate() {
            if let Some(e) = euler.get_mut(i) {
                let r = Slider::new(axis, e, -180.0, 180.0).default_value(0.0).decimals(1).show(ui);
                if r.phase != EditPhase::Idle {
                    phase = r.phase;
                }
            }
        }
        let human = ctx.human;
        let set = |e: [f32; 3]| {
            let q = euler_deg_to_quat(e);
            Command::new(human, Action::SetPose { joints: vec![(joint.clone(), Some(q))] })
        };
        match phase {
            EditPhase::Begin | EditPhase::Update if euler != before => {
                intents.push(Intent::Execute(set(euler).in_gesture()));
            }
            EditPhase::Commit => intents.push(Intent::EndGesture),
            EditPhase::Step if euler != before => intents.push(Intent::Execute(set(euler))),
            _ => {}
        }
        if Button::new("Rest pose").kind(ButtonKind::Secondary).show(ui).clicked {
            let joints = ctx.session.document().pose.keys().map(|k| (k.clone(), None)).collect();
            intents.push(Intent::Execute(Command::new(human, Action::SetPose { joints })));
        }
        label_styled(
            ui,
            "The diagnostic pose is a preview; it is saved with the document.",
            theme.text.small,
            theme.palette.text_faint,
        );
    }

    fn history(&mut self, ui: &mut Ui, ctx: &Ctx<'_, '_>, intents: &mut Vec<Intent>) {
        let theme = *ui.theme();
        let m = theme.metrics;
        let h = ctx.session.history();
        let doc = ctx.session.document();
        let entries = h.entries();
        let cursor = h.cursor();
        ui.begin(Style { gap: m.space_2, ..Style::row() });
        if Button::new("Undo").kind(ButtonKind::Secondary).enabled(cursor > 0).show(ui).clicked {
            intents.push(Intent::Undo(Scope::Any));
        }
        if Button::new("Redo").kind(ButtonKind::Secondary).enabled(cursor < entries.len()).show(ui).clicked {
            intents.push(Intent::Redo);
        }
        let agent = doc.authors.iter().find(|a| a.kind == humanart_doc::document::AuthorKind::Agent);
        if let Some(a) = agent
            && Button::new("Undo the agent's last").kind(ButtonKind::Secondary).show(ui).clicked
        {
            intents.push(Intent::Undo(Scope::Author(a.id.clone())));
        }
        ui.end();
        let r = list_view(ui, "History", entries.len(), 200.0, &mut self.history_row, &|i, out| {
            // Newest first.
            let Some(e) = entries.len().checked_sub(i + 1).and_then(|k| entries.get(k)) else { return };
            let k = entries.len() - i - 1;
            let who = match doc.author(&e.author).map(|a| a.kind) {
                Some(humanart_doc::document::AuthorKind::Agent) => "agent",
                Some(_) => "human",
                None => "?",
            };
            let undone = if k >= cursor { "  (undone)" } else { "" };
            let label = if e.label.is_empty() { e.kind.as_str() } else { e.label.as_str() };
            let _ = write!(out, "[{who}] {label}{undone}");
        });
        let _ = r;
        // The before/after preview of the selected entry `[APP-06]`.
        if let Some(row) = self.history_row
            && let Some(k) = entries.len().checked_sub(row + 1)
            && let Some(e) = entries.get(k)
        {
            if self.preview.as_ref().is_none_or(|(i, _)| *i != k) {
                let mut text = String::new();
                for edit in e.edits.iter().take(12) {
                    if let humanart_doc::text::Edit::Set { key, before, after, .. } = edit {
                        let show = |v: &Option<humanart_io::json::Value>| {
                            v.as_ref()
                                .and_then(|v| humanart_io::json::to_string(v).ok())
                                .unwrap_or_else(|| "default".into())
                        };
                        let _ = writeln!(text, "{key}: {} -> {}", show(before), show(after));
                    }
                }
                if e.edits.len() > 12 {
                    let _ = writeln!(text, "... {} more", e.edits.len() - 12);
                }
                self.preview = Some((k, text));
            }
            if let Some((_, text)) = &self.preview {
                label_styled(ui, "Before -> after", theme.text.label, theme.palette.text);
                let n = ui.node(Style {
                    width: Dim::Percent(100.0),
                    height: Dim::Auto,
                    min_height: 60.0,
                    ..Style::default()
                });
                let t = ui.intern(text);
                ui.paint(
                    n,
                    Paint::Text {
                        area: Area::FULL,
                        text: t,
                        style: theme.text.mono,
                        color: theme.palette.text_muted,
                        align: Align::Start,
                        wrap: true,
                    },
                );
            }
        }
    }

    fn proposals(&mut self, ui: &mut Ui, ctx: &Ctx<'_, '_>, intents: &mut Vec<Intent>) {
        let theme = *ui.theme();
        let list = ctx.session.proposals();
        if list.is_empty() {
            label_styled(
                ui,
                "No open proposals. Agent proposals appear here for review.",
                theme.text.small,
                theme.palette.text_muted,
            );
            return;
        }
        list_view(ui, "Proposals", list.len(), 160.0, &mut self.proposal_row, &|i, out| {
            if let Some(p) = list.get(i) {
                let _ = write!(out, "{}  by {}  ({} commands)", p.name, p.author, p.entries.len());
            }
        });
        if let Some(p) = self.proposal_row.and_then(|i| list.get(i)) {
            ui.begin(Style { gap: theme.metrics.space_2, ..Style::row() });
            if Button::new("Accept").kind(ButtonKind::Primary).show(ui).clicked {
                intents.push(Intent::Accept(p.id));
                self.proposal_row = None;
            }
            if Button::new("Discard").kind(ButtonKind::Danger).show(ui).clicked {
                intents.push(Intent::Discard(p.id));
                self.proposal_row = None;
            }
            ui.end();
        }
    }

    fn profiler(&self, ui: &mut Ui) {
        let theme = *ui.theme();
        let mut s = ArrayString::<64>::new();
        let _ = write!(s, "Frame (CPU): {:.2} ms", self.profile.frame_ms);
        label(ui, s.as_str());
        for (name, ms) in self.profile.passes.iter().take(self.profile.count) {
            let mut s = ArrayString::<64>::new();
            match ms {
                Some(ms) => {
                    let _ = write!(s, "{name}: {ms:.3} ms");
                }
                None => {
                    let _ = write!(s, "{name}: not timed");
                }
            }
            label_styled(ui, s.as_str(), theme.text.small, theme.palette.text_muted);
        }
        if self.profile.count == 0 {
            label_styled(
                ui,
                "GPU pass timings appear when the device has timestamp queries.",
                theme.text.small,
                theme.palette.text_faint,
            );
        }
    }

    fn data(&mut self, ui: &mut Ui, ctx: &Ctx<'_, '_>) {
        let theme = *ui.theme();
        let dir = ctx.prefs.data_dir.as_str();
        if self.data_dir.as_ref().is_none_or(|(d, _)| d != dir)
            || Button::new("Refresh").kind(ButtonKind::Ghost).show(ui).clicked
        {
            let mut list = ctx.fs.list(dir).unwrap_or_default();
            list.sort_by(|a, b| a.name.cmp(&b.name));
            self.data_dir = Some((dir.to_owned(), list));
        }
        if let Some((d, list)) = &self.data_dir {
            label_styled(ui, d, theme.text.label, theme.palette.text);
            list_view(ui, "Data files", list.len(), 300.0, &mut self.data_row, &|i, out| {
                if let Some(e) = list.get(i) {
                    let kind = if e.kind == FileKind::Directory { "dir " } else { "file" };
                    let _ = write!(out, "{kind}  {}  {} KiB", e.name, e.len / 1024);
                }
            });
        }
    }

    fn export(&mut self, ui: &mut Ui, ctx: &Ctx<'_, '_>, intents: &mut Vec<Intent>) {
        let theme = *ui.theme();
        let where_ = ctx.project.unwrap_or("not saved yet");
        let mut s = ArrayString::<256>::new();
        let _ = write!(s, "Project: {where_}");
        label_styled(ui, s.as_str(), theme.text.small, theme.palette.text_muted);
        ui.begin(Style { gap: theme.metrics.space_2, ..Style::row() });
        if Button::new("Save").kind(ButtonKind::Primary).show(ui).clicked {
            intents.push(Intent::Save);
        }
        if Button::new("Save as...").kind(ButtonKind::Secondary).show(ui).clicked {
            self.open_browser(BrowseFor::SaveAs, &ctx.prefs.projects_dir);
        }
        ui.end();
    }

    fn overlays(
        &mut self,
        ui: &mut Ui,
        ctx: &Ctx<'_, '_>,
        viewports: &mut [ViewportState; MAX_VIEWPORTS],
        intents: &mut Vec<Intent>,
    ) {
        let theme = *ui.theme();
        if self.onboarding {
            let (choice, out) = Modal::new("Welcome to HumanArt", 520.0, 360.0).show(ui, |ui| {
                for line in [
                    "1. The sample character is open. Drag in the viewport to orbit, Shift+drag to pan, scroll to zoom.",
                    "2. Pick a mode at the top (Ctrl+1 ... Ctrl+0); Properties lists its parameters.",
                    "3. Ctrl+F searches every parameter. H and A badges show who set a value last.",
                    "4. Changes made by the agent flash; the Agent panel pauses or revokes it.",
                    "5. Ctrl+Z / Ctrl+Y undo and redo; History shows each change before and after.",
                ] {
                    let n = ui.node(Style { width: Dim::Percent(100.0), min_height: 34.0, ..Style::default() });
                    let t = ui.intern(line);
                    ui.paint(n, Paint::Text { area: Area::FULL, text: t, style: theme.text.body, color: theme.palette.text, align: Align::Start, wrap: true });
                }
                let mut pick = 0;
                ui.begin(Style { gap: theme.metrics.space_3, ..Style::row() });
                if Button::new("Start with the sample").kind(ButtonKind::Primary).show(ui).clicked {
                    pick = 1;
                }
                if Button::new("Open a project...").kind(ButtonKind::Secondary).show(ui).clicked {
                    pick = 2;
                }
                if Button::new("Review tour").kind(ButtonKind::Secondary).show(ui).clicked {
                    pick = 3;
                }
                ui.end();
                pick
            });
            if out.close || choice == 1 {
                self.onboarding = false;
            }
            if choice == 2 {
                self.onboarding = false;
                self.open_browser(BrowseFor::Open, &ctx.prefs.projects_dir);
            }
            if choice == 3 {
                self.onboarding = false;
                self.tour = Some(0);
            }
        }
        if let Some(step) = self.tour {
            let slot = self.active_viewport.min(MAX_VIEWPORTS - 1);
            if let Some((camera, rig, text)) = TOUR.get(step) {
                if let Some(v) = viewports.get_mut(slot)
                    && (v.settings.rig != *rig)
                {
                    review_scene(v, ctx.model, *camera, *rig);
                }
                let mut heading = ArrayString::<48>::new();
                let _ = write!(heading, "Review tour: scene {} of {}", step + 1, TOUR.len());
                let (next, out) = Modal::new(heading.as_str(), 460.0, 200.0).show(ui, |ui| {
                    label(ui, text);
                    let mut next = false;
                    if Button::new(if step + 1 < TOUR.len() { "Next scene" } else { "Finish" })
                        .kind(ButtonKind::Primary)
                        .show(ui)
                        .clicked
                    {
                        next = true;
                    }
                    next
                });
                if next {
                    self.tour = (step + 1 < TOUR.len()).then_some(step + 1);
                    if let (Some(v), Some((c, r, _))) = (viewports.get_mut(slot), self.tour.and_then(|s| TOUR.get(s))) {
                        review_scene(v, ctx.model, *c, *r);
                    }
                }
                if out.close {
                    self.tour = None;
                }
            } else {
                self.tour = None;
            }
        }
        if self.prefs_draft.is_some() {
            self.prefs_dialog(ui, intents);
        }
        if self.browser.is_some() {
            self.browser_dialog(ui, ctx.fs, intents);
        }
    }

    fn prefs_dialog(&mut self, ui: &mut Ui, intents: &mut Vec<Intent>) {
        let Some(mut d) = self.prefs_draft.take() else { return };
        let (action, out) = Modal::new("Preferences", 520.0, 560.0).show(ui, |ui| {
            let mut theme_i = ThemeChoice::ALL.iter().position(|t| *t == d.theme).unwrap_or(0);
            if Dropdown::new("Theme", &mut theme_i, &THEME_NAMES).show(ui).changed {
                d.theme = ThemeChoice::ALL.get(theme_i).copied().unwrap_or(ThemeChoice::Dark);
            }
            Slider::new("UI scale", &mut d.ui_scale, 0.75, 2.0).step(0.05).default_value(1.0).show(ui);
            toggle(ui, "Colour-neutral UI", &mut d.colour_neutral);
            toggle(ui, "Reduce motion", &mut d.reduced_motion);
            let mut q = QualityChoice::ALL.iter().position(|x| *x == d.quality).unwrap_or(0);
            if Dropdown::new("Default quality", &mut q, &QUALITY_NAMES).show(ui).changed {
                d.quality = QualityChoice::ALL.get(q).copied().unwrap_or(QualityChoice::Interactive);
            }
            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                let mut mem = d.memory_budget_mb as f32;
                Slider::new("GPU memory budget (MiB)", &mut mem, 256.0, 16384.0).step(256.0).decimals(0).show(ui);
                d.memory_budget_mb = mem.round().clamp(256.0, 65536.0) as u32;
                let mut auto = d.autosave_seconds as f32;
                Slider::new("Autosave after idle (s, 0 = off)", &mut auto, 0.0, 600.0).step(10.0).decimals(0).show(ui);
                d.autosave_seconds = auto.round().clamp(0.0, 3600.0) as u32;
            }
            Slider::new("Agent change flash (s)", &mut d.flash_seconds, 0.0, 10.0).step(0.5).decimals(1).show(ui);
            toggle(ui, "Protect human edits from the agent", &mut d.agent.protect_human_edits);
            toggle(ui, "Auto-accept agent proposals", &mut d.agent.auto_accept);
            TextField::new("Projects folder", &mut d.projects_dir).show(ui);
            TextField::new("Data folder", &mut d.data_dir).show(ui);
            let mut action = 0;
            ui.begin(Style { gap: ui.theme().metrics.space_3, ..Style::row() });
            if Button::new("Apply").kind(ButtonKind::Secondary).show(ui).clicked {
                action = 1;
            }
            if Button::new("Save").kind(ButtonKind::Primary).show(ui).clicked {
                action = 2;
            }
            ui.end();
            action
        });
        let valid = d.validate().is_ok();
        if !out.close {
            self.prefs_draft = Some(d);
        }
        if action > 0 && valid {
            intents.push(Intent::ApplyPrefs);
            if action == 2 {
                intents.push(Intent::SavePrefs);
            }
        }
    }

    fn browser_dialog(&mut self, ui: &mut Ui, fs: &dyn FileSystem, intents: &mut Vec<Intent>) {
        let Some(mut b) = self.browser.take() else { return };
        if b.error.as_deref() == Some("") {
            b.read(fs);
        }
        let title = if b.purpose == BrowseFor::Open { "Open project" } else { "Save project as" };
        let mut go: Option<String> = None;
        let mut pick: Option<String> = None;
        let ((), out) = Modal::new(title, 560.0, 480.0).show(ui, |ui| {
            let theme = *ui.theme();
            label_styled(ui, &b.dir, theme.text.mono, theme.palette.text_muted);
            if let Some(e) = &b.error {
                label_styled(ui, e, theme.text.small, theme.palette.danger);
            }
            let entries = &b.entries;
            let r = list_view(ui, "Entries", entries.len(), 240.0, &mut b.selected, &|i, out| {
                if let Some(e) = entries.get(i) {
                    let _ = write!(out, "{}{}", e.name, if e.kind == FileKind::Directory { "/" } else { "" });
                }
            });
            if let Some(i) = r.activated
                && let Some(e) = b.entries.get(i).filter(|e| e.kind == FileKind::Directory)
            {
                go = Some(humanart_core::platform::join_path(&b.dir, &e.name));
            }
            ui.begin(Style { gap: theme.metrics.space_2, ..Style::row() });
            if Button::new("Up").kind(ButtonKind::Ghost).show(ui).clicked {
                go = Some(humanart_core::platform::parent_path(&b.dir).to_owned());
            }
            if b.purpose == BrowseFor::Open {
                let target = b.selected.and_then(|i| b.entries.get(i)).filter(|e| e.kind == FileKind::Directory);
                if Button::new("Open").kind(ButtonKind::Primary).enabled(target.is_some()).show(ui).clicked
                    && let Some(e) = target
                {
                    pick = Some(humanart_core::platform::join_path(&b.dir, &e.name));
                }
            } else {
                TextField::new("Name", &mut b.name).placeholder("New project folder").show(ui);
                let ok = !b.name.trim().is_empty() && !b.name.contains(['/', '\\']);
                if Button::new("Save here").kind(ButtonKind::Primary).enabled(ok).show(ui).clicked {
                    pick = Some(humanart_core::platform::join_path(&b.dir, b.name.trim()));
                }
            }
            ui.end();
        });
        if let Some(dir) = go {
            b.dir = dir;
            b.read(fs);
        }
        if let Some(path) = pick {
            intents.push(if b.purpose == BrowseFor::Open { Intent::Open(path) } else { Intent::SaveAs(path) });
            return;
        }
        if !out.close {
            self.browser = Some(b);
        }
    }
}

/// The id of a viewport panel's image (independent of the id scope it is built in).
fn image_id(panel: &str) -> humanart_ui::ui::Id {
    humanart_ui::ui::Id::ROOT.with("viewport image").with(panel)
}

/// Writes a short text form of a value.
fn write_value(out: &mut impl std::fmt::Write, v: &ParamValue) {
    let _ = match v {
        ParamValue::Scalar(x) => write!(out, "{x:.3}"),
        ParamValue::Bool(b) => write!(out, "{}", if *b { "on" } else { "off" }),
        ParamValue::Enum(e) => write!(out, "{e}"),
        ParamValue::Color(c) => write!(out, "rgb {:.2} {:.2} {:.2}", c[0], c[1], c[2]),
        ParamValue::Vector(list) => write!(out, "{} values", list.len()),
        ParamValue::Curve(points) => write!(out, "curve of {} points", points.len()),
        ParamValue::AssetRef(h) => write!(out, "asset {}", h.to_hex().as_str().get(..12).unwrap_or("")),
    };
}

/// The author badge `[APP-03]`: H (human) or A (agent) in a small disc; nothing for defaults.
fn badge_node(ui: &mut Ui, b: Badge) {
    let theme = *ui.theme();
    let n = ui.node(Style { width: Dim::Px(18.0), height: Dim::Px(18.0), shrink: 0.0, ..Style::default() });
    let (letter, color, name) = match b {
        Badge::Default => return,
        Badge::Human => ("H", theme.palette.positive, "set by a human"),
        Badge::Agent => ("A", theme.palette.link, "set by the agent"),
    };
    ui.paint(n, Paint::Circle { cx: Anchor::at(0.5, 0.0), cy: Anchor::at(0.5, 0.0), radius: 8.0, color });
    let t = ui.intern(letter);
    ui.paint(
        n,
        Paint::Text {
            area: Area::FULL,
            text: t,
            style: theme.text.tiny,
            color: theme.palette.on_accent,
            align: Align::Center,
            wrap: false,
        },
    );
    let id = ui.id("badge");
    ui.interact(id, n, humanart_ui::ui::Sense::HOVER);
    ui.semantic(id, humanart_ui::ui::Role::Custom("badge"), name, humanart_ui::ui::SemanticValue::None);
}

fn agent_panel(ui: &mut Ui, ctx: &Ctx<'_, '_>, intents: &mut Vec<Intent>) {
    let theme = *ui.theme();
    let a = ctx.agent;
    let mut s = ArrayString::<128>::new();
    let _ = write!(s, "{}: {}", a.name().unwrap_or("No agent"), a.status().label());
    label_styled(ui, s.as_str(), theme.text.label, theme.palette.text);
    if a.is_paused() {
        let mut s = ArrayString::<64>::new();
        let _ = write!(s, "Paused: {} commands queued", a.queued());
        label_styled(ui, s.as_str(), theme.text.small, theme.palette.warning);
    }
    if a.is_revoked() {
        label_styled(ui, "Access revoked: agent commands are refused.", theme.text.small, theme.palette.danger);
    }
    if let Some((url, token)) = ctx.agent_server {
        let mut s = ArrayString::<96>::new();
        let _ = write!(s, "Agent server: {url}");
        label_styled(ui, s.as_str(), theme.text.small, theme.palette.text_muted);
        ui.begin(Style { gap: theme.metrics.space_2, ..Style::row() });
        if Button::new("Copy address").kind(ButtonKind::Ghost).show(ui).clicked {
            ui.copy(url);
        }
        if Button::new("Copy token").kind(ButtonKind::Ghost).show(ui).clicked {
            ui.copy(token);
        }
        ui.end();
    }
    ui.begin(Style { gap: theme.metrics.space_2, ..Style::row() });
    let pause_label = if a.is_paused() { "Resume agent" } else { "Pause agent" };
    if Button::new(pause_label).kind(ButtonKind::Secondary).enabled(!a.is_revoked()).show(ui).clicked {
        intents.push(Intent::AgentPause(!a.is_paused()));
    }
    // Granting starts the agent server; revoking stops it [APP-04].
    let serving = ctx.agent_server.is_some();
    if a.is_revoked() || !serving {
        let label = if a.is_revoked() { "Grant access" } else { "Start agent server" };
        if Button::new(label).kind(ButtonKind::Secondary).show(ui).clicked {
            intents.push(Intent::AgentRevoke(false));
        }
    }
    if !a.is_revoked()
        && (serving || a.status() != crate::agent_link::AgentStatus::Disconnected)
        && Button::new("Revoke agent").kind(ButtonKind::Danger).show(ui).clicked
    {
        intents.push(Intent::AgentRevoke(true));
    }
    ui.end();
    let mut s = ArrayString::<96>::new();
    let _ = write!(
        s,
        "Policy: human edits {}, auto-accept {}",
        if ctx.prefs.agent.protect_human_edits { "protected" } else { "not protected" },
        if ctx.prefs.agent.auto_accept { "on" } else { "off" }
    );
    label_styled(ui, s.as_str(), theme.text.small, theme.palette.text_muted);
    let mut s = ArrayString::<64>::new();
    let _ = write!(s, "Open proposals: {}", ctx.session.proposals().len());
    label_styled(ui, s.as_str(), theme.text.small, theme.palette.text_muted);
    label_styled(ui, "Recent agent commands", theme.text.label, theme.palette.text);
    let recent = ctx.provenance.agent_commands();
    if recent.is_empty() {
        label_styled(ui, "None yet", theme.text.small, theme.palette.text_faint);
    }
    for (who, what) in recent.iter().rev().take(8) {
        let mut s = ArrayString::<160>::new();
        let _ = write!(s, "{who}: {what}");
        label_styled(ui, s.as_str(), theme.text.small, theme.palette.text_muted);
    }
}

fn compare_panel(ui: &mut Ui, ctx: &Ctx<'_, '_>) {
    let theme = *ui.theme();
    let refs = &ctx.session.document().references;
    if refs.is_empty() {
        label_styled(
            ui,
            "This document has no reference images. Comparison views come with WP-22.",
            theme.text.small,
            theme.palette.text_muted,
        );
    }
    for r in refs {
        let mut s = ArrayString::<256>::new();
        let _ = write!(s, "{} ({})", r.name, r.path);
        label_styled(ui, s.as_str(), theme.text.small, theme.palette.text);
    }
}

fn light_camera_tools(ui: &mut Ui, v: &mut ViewportState) {
    let mut view = DebugView::ALL.iter().position(|d| *d == v.settings.debug).unwrap_or(0);
    if Dropdown::new("View mode", &mut view, &VIEWS).show(ui).changed
        && let Some(d) = DebugView::ALL.get(view)
    {
        v.settings.debug = *d;
    }
    let mut rig = BuiltinRig::ALL.iter().position(|r| *r == v.settings.rig).unwrap_or(0);
    if Dropdown::new("Light rig", &mut rig, &RIG_NAMES).show(ui).changed
        && let Some(r) = BuiltinRig::ALL.get(rig)
    {
        v.settings.rig = *r;
    }
    let mut q = QUALITIES.iter().position(|x| *x == v.settings.quality).unwrap_or(0);
    if Dropdown::new("Quality", &mut q, &QUALITY_NAMES).show(ui).changed
        && let Some(x) = QUALITIES.get(q)
    {
        v.settings.quality = *x;
    }
    let c = &mut v.settings.camera;
    Slider::new("Focal length (mm)", &mut c.focal_mm, 18.0, 200.0).default_value(85.0).decimals(0).show(ui);
    Slider::new("Exposure compensation (EV)", &mut c.exposure.compensation, -4.0, 4.0)
        .default_value(0.0)
        .step(0.1)
        .show(ui);
    let mut wire = v.settings.wireframe;
    if toggle(ui, "Wireframe", &mut wire).changed {
        v.settings.wireframe = wire;
    }
}

/// Points a viewport at a rubric scene.
fn review_scene(v: &mut ViewportState, model: &Model, camera: ReviewCamera, rig: BuiltinRig) {
    let aspect = if v.rect.h > 0.0 { v.rect.w / v.rect.h } else { 1.0 };
    let (center, radius) = if camera == ReviewCamera::FullBody { model.bounds() } else { model.head_bounds() };
    let exposure = v.settings.camera.exposure;
    v.settings.camera = humanart_render::camera::Camera::review(camera, center, radius, aspect);
    v.settings.camera.exposure = exposure;
    if camera == ReviewCamera::FullBody {
        crate::app::frame_viewport(v, model, Framing::Body);
    }
    v.settings.rig = rig;
}

/// XYZ Euler angles (degrees) of a unit quaternion `(x, y, z, w)`.
#[allow(clippy::cast_possible_truncation)] // Angles in degrees.
pub fn quat_to_euler_deg(q: [f64; 4]) -> [f32; 3] {
    let [x, y, z, w] = q;
    let sinr = 2.0 * (w * x + y * z);
    let cosr = 1.0 - 2.0 * (x * x + y * y);
    let sinp = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0);
    let siny = 2.0 * (w * z + x * y);
    let cosy = 1.0 - 2.0 * (y * y + z * z);
    [sinr.atan2(cosr).to_degrees() as f32, sinp.asin().to_degrees() as f32, siny.atan2(cosy).to_degrees() as f32]
}

/// The unit quaternion `(x, y, z, w)` of XYZ Euler angles in degrees (the inverse of
/// [`quat_to_euler_deg`]).
pub fn euler_deg_to_quat(e: [f32; 3]) -> [f64; 4] {
    let [r, p, y] = e.map(|a| f64::from(a).to_radians() * 0.5);
    let (sr, cr, sp, cp, sy, cy) = (r.sin(), r.cos(), p.sin(), p.cos(), y.sin(), y.cos());
    [sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euler_round_trip() {
        for e in [[0.0, 0.0, 0.0], [30.0, -20.0, 45.0], [-90.0, 10.0, 170.0]] {
            let back = quat_to_euler_deg(euler_deg_to_quat(e));
            for (a, b) in e.iter().zip(back) {
                assert!((a - b).abs() < 1e-3, "{e:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn viewport_slots() {
        assert_eq!(viewport_slot("viewport.1"), Some(0));
        assert_eq!(viewport_slot("viewport.4"), Some(3));
        assert_eq!(viewport_slot("viewport.5"), None);
        assert_eq!(viewport_slot("viewport.0"), None);
        assert_eq!(viewport_slot("properties"), None);
    }
}
