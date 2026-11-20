//! The HumanArt desktop editor: the window, its surface and event loop around the editor of the
//! `humanart` library (which also runs offscreen in the tests).
//!
//! Start-up: the data directory is `HUMANART_DATA` or `./data`; preferences, keymap, layout and the
//! last project live in `HUMANART_SETTINGS` or `<APPDATA or HOME>/HumanArt`. The template is the
//! package `data/template/template.antmpl` or, until it exists, generated (ADR-006). The project
//! opened last is reopened, replaying its journal after a crash `[AC-WP21-3]`.
//!
//! The window follows the reference application (`docs/reference-conventions.md` sections 1-3):
//! created hidden and shown after the first frame, resize applied at the next frame, a zero size
//! means hidden, a surface that is outdated or lost is reconfigured with one immediate retry,
//! redraws only on input, animation or a pending autosave. The surface must offer the frame format
//! with copy access (the viewports are copied into the frame); the backends are tried in order
//! (the environment's choice, DX12, Vulkan, GL) until one does.
#![forbid(unsafe_code)]

use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use humanart::app::{App, Env, Intent};
use humanart::prefs::Prefs;
use humanart::shell::BrowseFor;
use humanart::studio::{Studio, load_fonts};
use humanart::window::{cursor_of, surface_config, translate};
use humanart_core::id::Id128;
use humanart_core::platform::{Clock, FileSystem, join_path};
use humanart_doc::document::{Author, AuthorKind};
use humanart_doc::registry::{Registry, RegistryLimits};
use humanart_gpu::device::block_on;
use humanart_gpu::{DeviceRequest, Gpu};
use humanart_platform::{
    Clipboard, DialogKind, DialogOutcome, DialogRequest, FileDialogs, OsEntropy, PendingDialog, StdFileSystem,
    SystemClock,
};
use humanart_ui::geom::{Point, Size};
use humanart_ui::input::{Event, InputQueue};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// How often an open project is ticked for autosaves while idle.
const TICK: Duration = Duration::from_millis(1000);

/// Where the editor's files are.
struct Paths {
    data: String,
    settings: String,
}

fn paths() -> Paths {
    let cwd = std::env::current_dir().map_or_else(|_| ".".into(), |p| p.to_string_lossy().into_owned());
    let data = std::env::var("HUMANART_DATA").unwrap_or_else(|_| join_path(&cwd, "data"));
    let settings = std::env::var("HUMANART_SETTINGS").unwrap_or_else(|_| {
        let base = std::env::var("APPDATA").or_else(|_| std::env::var("HOME")).unwrap_or(cwd);
        join_path(&base, "HumanArt")
    });
    Paths { data, settings }
}

/// Everything the window needs, alive from the first `resumed` to exit.
struct Running {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    gpu: Gpu,
    config: wgpu::SurfaceConfiguration,
    resized: bool,
    retried: bool,
    studio: Studio,
    app: App<'static>,
    clock: &'static SystemClock,
    events: Vec<Event>,
    pointer: Point,
    clipboard: Clipboard,
    dialogs: FileDialogs,
    dialog: Option<(BrowseFor, PendingDialog)>,
    shown: bool,
    redraw: bool,
}

#[derive(Default)]
struct Host {
    running: Option<Running>,
    error: Option<String>,
}

fn start(event_loop: &ActiveEventLoop) -> Result<Running, String> {
    let paths = paths();
    let fs: &'static StdFileSystem = Box::leak(Box::new(StdFileSystem::host()));
    let clock: &'static SystemClock = Box::leak(Box::new(SystemClock::new()));
    let registry: &'static Registry = Box::leak(Box::new(
        Registry::load_dir(fs, &join_path(&paths.data, "template/params"), &RegistryLimits::DEFAULT)
            .map_err(|e| format!("The parameter registry could not be read: {e}"))?,
    ));
    let (template, source) =
        humanart_agent::engine::load_template(fs, &join_path(&paths.data, "template/template.antmpl"))
            .map_err(|e| format!("The template could not be loaded: {e}"))?;
    let fonts =
        load_fonts(fs, &join_path(&paths.data, "fonts")).map_err(|e| format!("The UI fonts could not be read: {e}"))?;
    let prefs = match fs.read(&join_path(&paths.settings, "prefs.json")) {
        Ok(bytes) => Prefs::from_json(&bytes).unwrap_or_default(),
        Err(_) => Prefs { data_dir: paths.data.clone(), ..Prefs::default() },
    };
    let seed = Id128::random(&OsEntropy).unwrap_or(Id128::from_u128(u128::from(clock.now_nanos())));
    let user = std::env::var("USERNAME").or_else(|_| std::env::var("USER")).unwrap_or_else(|_| "owner".into());
    let human = Author { id: "human.owner".into(), kind: AuthorKind::Human, name: user };
    let env = Env { registry, fs, clock, template, human, settings_dir: paths.settings.clone(), seed };
    let budget = prefs.memory_budget_mb;
    let mut app = App::new(env, prefs, fonts.clone());
    if source == humanart_agent::engine::TemplateSource::Generated {
        app.shell.log(humanart_ui::views::LogLevel::Info, 0.0, "The template was generated (no package yet)");
    }
    if let Some(last) = app.last_project()
        && let Err(e) = app.open_project(&last)
    {
        app.shell.log(humanart_ui::views::LogLevel::Warn, 0.0, &format!("{last} could not be reopened: {e}"));
    }

    let attributes = Window::default_attributes()
        .with_title("HumanArt")
        .with_inner_size(LogicalSize::new(1600.0, 960.0))
        .with_min_inner_size(LogicalSize::new(960.0, 600.0))
        .with_visible(false);
    let window = Arc::new(event_loop.create_window(attributes).map_err(|e| format!("No window: {e}"))?);
    window.set_ime_allowed(true);
    let size = window.inner_size();
    let display = event_loop.owned_display_handle();
    let mut last_error = String::from("no backend offers a window surface the editor can draw into");
    for backends in [None, Some(wgpu::Backends::DX12), Some(wgpu::Backends::VULKAN), Some(wgpu::Backends::GL)] {
        let mut desc = wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(display.clone()));
        if let Some(b) = backends {
            desc.backends = b;
        }
        let instance = wgpu::Instance::new(desc);
        let surface = match instance.create_surface(Arc::clone(&window)) {
            Ok(s) => s,
            Err(e) => {
                last_error = format!("surface: {e}");
                continue;
            }
        };
        let gpu = match block_on(Gpu::new(instance, Some(&surface), &DeviceRequest::default())) {
            Ok(g) => g,
            Err(e) => {
                last_error = format!("device: {e}");
                continue;
            }
        };
        let caps = surface.get_capabilities(gpu.adapter());
        let Some(config) = surface_config(&caps, size.width, size.height) else {
            last_error = format!("{}: the surface has no RGBA8 format with copy access", gpu.adapter_info().backend);
            continue;
        };
        if size.width > 0 && size.height > 0 {
            surface.configure(&gpu.device, &config);
        }
        let studio = Studio::new(&gpu, &fonts, budget).map_err(|e| format!("GPU set-up failed: {e}"))?;
        return Ok(Running {
            window,
            surface,
            gpu,
            config,
            resized: false,
            retried: false,
            studio,
            app,
            clock,
            events: Vec::with_capacity(64),
            pointer: Point::new(0.0, 0.0),
            clipboard: Clipboard::native(),
            dialogs: FileDialogs::native(),
            dialog: None,
            shown: false,
            redraw: true,
        });
    }
    Err(format!("The GPU could not be set up: {last_error}"))
}

impl Running {
    fn now(&self) -> f64 {
        #[allow(clippy::cast_precision_loss)] // Seconds since start, far below 2^53 ns.
        let t = self.clock.now_nanos() as f64 / 1e9;
        t
    }

    /// Hands the in-app browser over to the native dialog when one is available `[APP-07]`.
    fn native_dialogs(&mut self) {
        if self.dialog.is_none()
            && self.dialogs.is_available()
            && let Some(b) = self.app.shell.browser.take()
        {
            let title = if b.purpose == BrowseFor::Open { "Open project" } else { "Choose a folder for the project" };
            let request = DialogRequest::new(DialogKind::Folder, title).in_dir(&b.dir);
            self.dialog = Some((b.purpose, self.dialogs.spawn(&request)));
        }
        if let Some((purpose, pending)) = &mut self.dialog
            && let Some(result) = pending.try_take()
        {
            let purpose = *purpose;
            self.dialog = None;
            match result {
                Ok(DialogOutcome::Chosen(paths)) => {
                    if let Some(path) = paths.into_iter().next() {
                        self.app.push_intent(if purpose == BrowseFor::Open {
                            Intent::Open(path)
                        } else {
                            Intent::SaveAs(join_path(&path, "Untitled character"))
                        });
                    }
                }
                Ok(DialogOutcome::Cancelled) => {}
                Err(_) => {
                    // The native dialog failed: the in-app browser is the fallback.
                    let dir = self.app.prefs.projects_dir.clone();
                    self.app.shell.open_browser(purpose, &dir);
                    self.dialogs = FileDialogs::disabled();
                }
            }
            self.redraw = true;
        }
    }

    fn draw(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_lost() {
            eprintln!("The GPU device was lost; the editor closes (unsaved edits are in the journal).");
            event_loop.exit();
            return;
        }
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }
        if self.resized {
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.gpu.device, &self.config);
            self.resized = false;
        }
        let scale = self.window.scale_factor();
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Window sizes.
        let logical = Size::new((f64::from(size.width) / scale) as f32, (f64::from(size.height) / scale) as f32);
        let mut input = InputQueue { events: std::mem::take(&mut self.events), time: self.now() };
        let t0 = self.clock.now_nanos();
        #[allow(clippy::cast_possible_truncation)] // Scale factors are small.
        let out = self.app.frame(&mut input, logical, scale as f32);
        self.events = input.events;
        self.events.clear();
        if let Some(text) = &out.copied {
            self.clipboard.set_text(text);
        }
        self.window.set_cursor(cursor_of(out.cursor));
        self.window.set_ime_allowed(out.text_input);
        self.native_dialogs();
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) => f,
            wgpu::CurrentSurfaceTexture::Suboptimal(f) => {
                self.resized = true;
                f
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.gpu.device, &self.config);
                if !self.retried {
                    self.retried = true;
                    self.window.request_redraw();
                }
                return;
            }
            _ => return,
        };
        self.retried = false;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        #[allow(clippy::cast_possible_truncation)] // Scale factors are small.
        if let Err(e) = self.app.record(&mut self.studio, &self.gpu, &mut encoder, &frame.texture, &view, scale as f32)
        {
            self.app.shell.log(humanart_ui::views::LogLevel::Error, input_time(self), &e.to_string());
        }
        self.gpu.queue.submit([encoder.finish()]);
        self.gpu.queue.present(frame);
        self.studio.after_submit();
        self.studio.fill_profile(&self.gpu.device, &mut self.app.shell.profile);
        #[allow(clippy::cast_precision_loss)] // Nanoseconds of one frame.
        {
            self.app.shell.profile.frame_ms = self.clock.now_nanos().saturating_sub(t0) as f32 / 1e6;
        }
        if !self.shown {
            self.window.set_visible(true);
            self.shown = true;
        }
        if out.animating || self.dialog.is_some() {
            self.window.request_redraw();
        }
    }
}

fn input_time(r: &Running) -> f64 {
    r.now()
}

impl ApplicationHandler for Host {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.running.is_some() {
            return;
        }
        match start(event_loop) {
            Ok(mut r) => {
                // The first frame is drawn at once: a hidden window gets no redraw request.
                r.draw(event_loop);
                self.running = Some(r);
            }
            Err(e) => {
                self.error = Some(e);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(r) = &mut self.running else { return };
        match &event {
            WindowEvent::CloseRequested => {
                if let Err(e) = r.app.save_prefs() {
                    eprintln!("The preferences could not be saved: {e}");
                }
                event_loop.exit();
                return;
            }
            WindowEvent::RedrawRequested => {
                r.draw(event_loop);
                return;
            }
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => r.resized = true,
            _ => {}
        }
        let scale = r.window.scale_factor();
        translate(&event, scale, &mut r.pointer, &mut r.events);
        if let WindowEvent::KeyboardInput { event: k, .. } = &event
            && k.state == winit::event::ElementState::Pressed
            && r.app.ui().modifiers().ctrl
            && k.physical_key == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyV)
            && let Some(text) = r.clipboard.get_text()
        {
            r.events.push(Event::Paste(text));
        }
        r.window.request_redraw();
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(r) = &mut self.running else { return };
        if r.redraw {
            r.redraw = false;
            r.window.request_redraw();
        }
        // An open project is ticked for autosaves; otherwise the editor sleeps until input.
        let flow = if r.app.project().is_some() || r.dialog.is_some() {
            ControlFlow::WaitUntil(Instant::now() + TICK)
        } else {
            ControlFlow::Wait
        };
        event_loop.set_control_flow(flow);
    }

    fn new_events(&mut self, _: &ActiveEventLoop, cause: winit::event::StartCause) {
        if let (winit::event::StartCause::ResumeTimeReached { .. }, Some(r)) = (cause, &mut self.running) {
            r.redraw = true;
        }
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        if let Some(r) = self.running.take() {
            r.studio.destroy();
        }
    }
}

fn main() -> ExitCode {
    let event_loop = match EventLoop::new() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("The window system could not be started: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut host = Host::default();
    if let Err(e) = event_loop.run_app(&mut host) {
        eprintln!("The event loop failed: {e}");
        return ExitCode::FAILURE;
    }
    match host.error {
        Some(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
        None => ExitCode::SUCCESS,
    }
}
