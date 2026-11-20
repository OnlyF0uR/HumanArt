//! Shared test fixtures of the application crate: the project registry and template, sessions with
//! a human and an agent author, the fonts.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use humanart_core::id::Id128;
use humanart_core::platform::{FileSystem, ManualClock, MemoryFs};
use humanart_doc::command::{Action, Command, RangePolicy};
use humanart_doc::document::{Author, AuthorKind, Document};
use humanart_doc::registry::{ParamKind, ParamValue, Registry, RegistryLimits};
use humanart_doc::session::{Session, SessionConfig};
use humanart_model::Template;
use humanart_ui::text::Fonts;

// [PERF-R1] The test binary counts allocations so steady-state tests can assert zero.
#[global_allocator]
static ALLOCATOR: humanart_core::alloc::CountingAllocator = humanart_core::alloc::CountingAllocator;

/// The human author of the fixtures.
pub const HUMAN: &str = "human.owner";
/// The agent author of the fixtures.
pub const AGENT: &str = "agent.claude";

/// The repository root.
pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Where the shell's offscreen shots go: `HUMANART_SHOTS_DIR` when set, else
/// `target/agents/app-shell/shots`.
pub fn shots_dir() -> PathBuf {
    let dir = std::env::var_os("HUMANART_SHOTS_DIR")
        .map_or_else(|| root().join("target/agents/app-shell/shots"), PathBuf::from);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Copies directory `from` (on disk) into `fs` at `to`.
pub fn load_dir(fs: &MemoryFs, from: &Path, to: &str) {
    fs.create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            fs.write(&format!("{to}/{name}"), &std::fs::read(&path).unwrap()).unwrap();
        }
    }
}

/// The project's registry (`data/template/params`).
pub fn registry() -> &'static Registry {
    static R: OnceLock<Registry> = OnceLock::new();
    R.get_or_init(|| {
        let fs = MemoryFs::new();
        load_dir(&fs, &root().join("data/template/params"), "params");
        Registry::load_dir(&fs, "params", &RegistryLimits::DEFAULT).unwrap()
    })
}

/// The generated template without LODs, built once.
pub fn template() -> Arc<Template> {
    static T: OnceLock<Arc<Template>> = OnceLock::new();
    Arc::clone(T.get_or_init(|| {
        let fs = MemoryFs::new();
        humanart_agent::engine::load_template(&fs, "none").unwrap().0
    }))
}

/// The baked UI fonts (`data/fonts`).
pub fn fonts() -> &'static Fonts {
    static F: OnceLock<Fonts> = OnceLock::new();
    F.get_or_init(|| {
        let fs = MemoryFs::new();
        load_dir(&fs, &root().join("data/fonts"), "fonts");
        crate::studio::load_fonts(&fs, "fonts").unwrap()
    })
}

/// A memory file system, a manual clock and the project registry.
pub struct Fixture {
    /// The file system.
    pub fs: MemoryFs,
    /// The clock.
    pub clock: ManualClock,
    /// The registry.
    pub registry: &'static Registry,
}

impl Fixture {
    /// A fresh fixture.
    pub fn new() -> Self {
        Self { fs: MemoryFs::new(), clock: ManualClock::new(1_700_000_000_000), registry: registry() }
    }

    /// A document with a human and an agent author and human-edit protection off.
    pub fn document() -> Document {
        let me = Author { id: HUMAN.into(), kind: AuthorKind::Human, name: "Owner".into() };
        let mut doc = Document::new(Id128::from_u128(0x5eed), "Sample", "generated", 1, 0, me);
        doc.authors.push(Author { id: AGENT.into(), kind: AuthorKind::Agent, name: "Claude".into() });
        doc.settings.protect_human_edits = false;
        doc
    }

    /// A session on [`Fixture::document`] without a project.
    pub fn session(&self) -> Session<'_> {
        Session::new(
            Self::document(),
            self.registry,
            &self.fs,
            &self.clock,
            SessionConfig::default(),
            Id128::from_u128(1),
        )
    }

    /// Two editable scalar parameters that name a region.
    pub fn two_scalar_params_with_regions(&self) -> (String, String) {
        let mut it = self.registry.params().iter().filter(|p| {
            p.kind == ParamKind::Scalar
                && p.region.is_some()
                && !p.read_only
                && !p.locked_by_default
                && p.range.is_some()
        });
        let a = it.next().unwrap().id.clone();
        let b = it.next().unwrap().id.clone();
        (a, b)
    }

    /// A command setting scalar `id` to `v` (clamped into its range).
    pub fn set(author: &str, id: &str, v: f64) -> Command {
        Command::new(
            author,
            Action::SetParams {
                values: vec![(id.into(), ParamValue::Scalar(v))],
                relative: false,
                policy: RangePolicy::Clamp,
            },
        )
    }
}

/// A settings directory and an [`crate::app::App`] on the sample document.
pub struct AppFixture {
    /// The file system, clock and registry.
    pub f: Fixture,
}

impl AppFixture {
    /// A fresh fixture.
    pub fn new() -> Self {
        Self { f: Fixture::new() }
    }

    /// The environment of an app.
    pub fn env(&self) -> crate::app::Env<'_> {
        crate::app::Env {
            registry: self.f.registry,
            fs: &self.f.fs,
            clock: &self.f.clock,
            template: template(),
            human: Author { id: HUMAN.into(), kind: AuthorKind::Human, name: "Owner".into() },
            settings_dir: "settings".into(),
            seed: Id128::from_u128(77),
        }
    }

    /// An app with the quick start closed.
    pub fn app(&self) -> crate::app::App<'_> {
        let mut app = crate::app::App::new(self.env(), crate::prefs::Prefs::default(), fonts().clone());
        app.shell.onboarding = false;
        app
    }
}

/// Runs one frame of `app` with `events` at 1600 x 900.
pub fn frame(app: &mut crate::app::App<'_>, events: Vec<humanart_ui::input::Event>, time: &mut f64) {
    *time += 1.0 / 60.0;
    let mut input = humanart_ui::input::InputQueue { events, time: *time };
    app.frame(&mut input, humanart_ui::geom::Size::new(1600.0, 900.0), 1.0);
}

/// Whether the last frame described a widget labelled `label`.
pub fn has_semantic(app: &crate::app::App<'_>, label: &str) -> bool {
    let ui = app.ui();
    ui.semantics().iter().any(|s| ui.text(s.label) == label)
}

/// Reads an RGBA8 texture back (blocking; test code).
pub fn read_rgba(gpu: &humanart_gpu::Gpu, texture: &wgpu::Texture) -> Vec<u8> {
    let (w, h) = (texture.width(), texture.height());
    let row = (w * 4).div_ceil(256) * 256;
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test readback"),
        size: u64::from(row * h),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
    enc.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    gpu.queue.submit([enc.finish()]);
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data = buffer.slice(..).get_mapped_range().unwrap();
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        let start = (y * row) as usize;
        out.extend_from_slice(&data[start..start + (w * 4) as usize]);
    }
    out
}

/// Renders the app's last frame offscreen (1600 x 900) and writes `shots/<name>`; returns the
/// pixels.
pub fn shot(
    app: &mut crate::app::App<'_>,
    studio: &mut crate::studio::Studio,
    gpu: &humanart_gpu::Gpu,
    name: &str,
) -> Vec<u8> {
    shot_with(app, studio, gpu, name, true)
}

/// [`shot`], with the viewers rendered or not (`viewports`).
pub fn shot_with(
    app: &mut crate::app::App<'_>,
    studio: &mut crate::studio::Studio,
    gpu: &humanart_gpu::Gpu,
    name: &str,
    viewports: bool,
) -> Vec<u8> {
    let target = crate::studio::offscreen_target(gpu, 1600, 900);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("shot") });
    let clear = app.clear_color();
    let viewports: Vec<_> = app
        .viewport_frames(1.0)
        .filter(|_| viewports)
        .map(|v| (v.slot, v.rect, v.settings, v.highlights.to_vec()))
        .collect();
    let frames: Vec<crate::studio::ViewportFrame<'_>> = viewports
        .iter()
        .map(|(slot, rect, settings, h)| crate::studio::ViewportFrame {
            slot: *slot,
            rect: *rect,
            settings: *settings,
            highlights: h,
        })
        .collect();
    let (ui, model) = app.ui_and_model();
    studio.record(gpu, &mut enc, &target, &view, ui, clear, model, &frames).unwrap();
    gpu.queue.submit([enc.finish()]);
    studio.after_submit();
    let rgba = read_rgba(gpu, &target);
    let png = humanart_render::capture::encode_png_rgba8(1600, 900, &rgba).unwrap();
    std::fs::write(shots_dir().join(name), png).unwrap();
    rgba
}

/// The headless GPU of the tests: the process's shared device.
pub fn gpu() -> &'static humanart_gpu::Gpu {
    humanart_gpu::shared_headless().expect("a GPU")
}

mod integration {
    use super::*;
    use humanart_ui::input::{Event, Key, Modifiers};

    use crate::modes::Mode;
    use crate::provenance::{Badge, badge};

    fn key(k: Key, mods: Modifiers) -> Vec<Event> {
        vec![
            Event::Modifiers(mods),
            Event::Key { key: k, pressed: true, repeat: false },
            Event::Key { key: k, pressed: false, repeat: false },
            Event::Modifiers(Modifiers::NONE),
        ]
    }

    #[test]
    fn ac_wp21_1_every_parameter_is_reachable_and_searchable() {
        // [AC-WP21-1] [REG-04] every registry parameter is found by the property search and gets
        // its row (a widget labelled with its display name) in the Properties panel.
        let fx = AppFixture::new();
        let mut app = fx.app();
        let mut t = 0.0;
        frame(&mut app, vec![], &mut t);
        let mut missing = Vec::new();
        for p in fx.f.registry.params() {
            app.shell.search.clear();
            app.shell.search.push_str(&p.id);
            frame(&mut app, vec![], &mut t);
            if !has_semantic(&app, &p.display_name) {
                missing.push(p.id.clone());
            }
        }
        assert!(missing.is_empty(), "parameters without a row when searched: {missing:?}");
        // Without a search, each parameter is listed by the modes that own its group.
        app.shell.search.clear();
        for m in Mode::ALL {
            app.shell.mode = m;
            frame(&mut app, vec![], &mut t);
            for p in fx.f.registry.params().iter().filter(|p| m.shows_group(&p.group)) {
                assert!(has_semantic(&app, &p.display_name), "{} is not listed in {}", p.id, m.label());
            }
        }
    }

    #[test]
    fn ac_wp21_1_modes_and_undo_from_the_keyboard() {
        // [AC-WP21-1] [UI-06] Ctrl+1 ... Ctrl+0 select the modes; Ctrl+Z / Ctrl+Y undo and redo.
        let fx = AppFixture::new();
        let mut app = fx.app();
        let mut t = 0.0;
        for (m, d) in Mode::ALL.iter().zip(['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']) {
            frame(&mut app, key(Key::Digit(d), Modifiers::CTRL), &mut t);
            assert_eq!(app.shell.mode, *m, "Ctrl+{d}");
        }
        let (id, _) = fx.f.two_scalar_params_with_regions();
        app.engine_mut().execute(Fixture::set(HUMAN, &id, 0.5)).unwrap();
        frame(&mut app, vec![], &mut t);
        assert!(app.engine().session().document().params.contains_key(&id));
        frame(&mut app, key(Key::Letter('Z'), Modifiers::CTRL), &mut t);
        assert!(!app.engine().session().document().params.contains_key(&id), "Ctrl+Z undoes");
        frame(&mut app, key(Key::Letter('Y'), Modifiers::CTRL), &mut t);
        assert!(app.engine().session().document().params.contains_key(&id), "Ctrl+Y redoes");
        // F1 opens the quick start.
        frame(&mut app, key(Key::F(1), Modifiers::NONE), &mut t);
        assert!(app.shell.onboarding);
        frame(&mut app, vec![], &mut t);
        assert!(has_semantic(&app, "Start with the sample"));
    }

    #[test]
    fn ac_wp21_2_badges_and_flashes_in_the_properties_panel() {
        // [AC-WP21-2] [APP-03] a human change shows the human badge without a flash; an agent
        // change (through the agent link) shows the agent badge, flashes its row and outlines its
        // region in the viewport; the flash ends after the configured time.
        let fx = AppFixture::new();
        let mut app = fx.app();
        app.add_agent_author(AGENT, "Claude").unwrap();
        let mut t = 0.0;
        frame(&mut app, vec![], &mut t);
        let (human_id, agent_id) = fx.f.two_scalar_params_with_regions();
        app.engine_mut().execute(Fixture::set(HUMAN, &human_id, 0.4)).unwrap();
        app.agent.connect("Claude");
        let applied = app.agent_submit(Fixture::set(AGENT, &agent_id, 0.6)).unwrap();
        assert!(matches!(applied, crate::agent_link::Submitted::Applied(_)));
        app.shell.search = human_id.split('.').next().unwrap_or_default().to_owned();
        frame(&mut app, vec![], &mut t);
        let doc = app.engine().session().document();
        assert_eq!(badge(doc, &human_id), Badge::Human);
        assert_eq!(badge(doc, &agent_id), Badge::Agent);
        assert!(
            has_semantic(&app, "set by a human") && has_semantic(&app, "set by the agent"),
            "both badges are drawn"
        );
        let flash = app.provenance.flash(&agent_id, t, app.prefs.flash_seconds);
        assert!(flash > 0.9, "the agent's change flashes: {flash}");
        assert!(app.provenance.flash(&human_id, t, app.prefs.flash_seconds).abs() < f32::EPSILON);
        let region = fx.f.registry.get(&agent_id).and_then(|p| p.region.clone()).unwrap();
        assert!(app.provenance.region_flashes(t, app.prefs.flash_seconds).any(|(r, _)| r == region));
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let frames = (60.0 * f64::from(app.prefs.flash_seconds)) as usize + 2;
        for _ in 0..frames {
            frame(&mut app, vec![], &mut t);
        }
        assert!(app.provenance.flash(&agent_id, t, app.prefs.flash_seconds).abs() < f32::EPSILON, "the flash ends");
        // Pausing queues the agent's command; the panel offers to resume.
        app.agent.pause();
        let queued = app.agent_submit(Fixture::set(AGENT, &agent_id, 0.2)).unwrap();
        assert!(matches!(queued, crate::agent_link::Submitted::Queued(1)));
        frame(&mut app, vec![], &mut t);
        assert!(app.agent.is_paused());
    }

    #[test]
    fn perf_r1_steady_ui_frames_do_not_allocate() {
        // [PERF-R1] [UI-10] once the panels were built, a frame without input allocates nothing.
        let fx = AppFixture::new();
        let mut app = fx.app();
        let mut t = 0.0;
        for _ in 0..5 {
            frame(&mut app, vec![], &mut t);
        }
        let mut events = Vec::with_capacity(8);
        let counts = humanart_core::alloc::count_in(|| {
            t += 1.0 / 60.0;
            let mut input = humanart_ui::input::InputQueue { events: std::mem::take(&mut events), time: t };
            app.frame(&mut input, humanart_ui::geom::Size::new(1600.0, 900.0), 1.0);
            events = input.events;
        });
        println!("steady frame: {} allocations, {} reallocations", counts.allocations, counts.reallocations);
        assert_eq!(counts.allocations + counts.reallocations, 0, "[PERF-R1] a steady frame allocates nothing");
    }

    #[test]
    fn ac_wp21_3_crash_recovery_from_the_ui() {
        // [AC-WP21-3] [SAVE-01] the editor saves a project, more edits reach the journal, the
        // process dies without saving; the relaunched editor finds the last project and opening it
        // replays the journal.
        let fx = AppFixture::new();
        let (a, b) = fx.f.two_scalar_params_with_regions();
        {
            let mut app = fx.app();
            app.save_as("projects/crash").unwrap();
            app.engine_mut().execute(Fixture::set(HUMAN, &a, 0.3)).unwrap();
            app.engine_mut().execute(Fixture::set(HUMAN, &b, -0.2)).unwrap();
            // No save: the app is dropped as a crash would leave it.
        }
        let mut app = fx.app();
        let last = app.last_project().expect("the relaunch knows the last project");
        assert_eq!(last, "projects/crash");
        app.open_project(&last).unwrap();
        let doc = app.engine().session().document();
        let value = |id: &str| match doc.params.get(id) {
            Some(ParamValue::Scalar(v)) => Some(*v),
            _ => None,
        };
        assert!(value(&a).is_some_and(|v| (v - 0.3).abs() < 1e-9), "{:?}", value(&a));
        assert!(value(&b).is_some_and(|v| (v + 0.2).abs() < 1e-9), "{:?}", value(&b));
        assert!(app.shell.console().entries().any(|e| e.text.contains("Recovered 2")), "the console reports it");
    }

    #[test]
    fn app_01_shell_renders_offscreen_with_its_viewport() {
        // [APP-01] [VP-04] the default workspace drawn offscreen: toolbar, panels, the viewport's
        // image composed into its panel. The shots are looked at by eye.
        let fx = AppFixture::new();
        let mut app = fx.app();
        app.add_agent_author(AGENT, "Claude").unwrap();
        let gpu = gpu();
        let mut studio = crate::studio::Studio::new(gpu, fonts(), 2048).unwrap();
        let mut t = 0.0;
        for _ in 0..3 {
            frame(&mut app, vec![], &mut t);
        }
        let v = app.viewports[0];
        assert!(v.open && v.rect.w > 200.0 && v.rect.h > 200.0, "the viewport has a rectangle: {:?}", v.rect);
        for _ in 0..3 {
            shot(&mut app, &mut studio, gpu, "shell_default.png");
        }
        let rgba = shot(&mut app, &mut studio, gpu, "shell_default.png");
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let (cx, cy) = ((v.rect.x + v.rect.w * 0.5) as usize, (v.rect.y + v.rect.h * 0.45) as usize);
        let px = &rgba[(cy * 1600 + cx) * 4..][..3];
        let surround = app.ui().theme().palette.viewport_surround.to_srgb8();
        assert_ne!(px, &surround[..3], "the viewport image is composed into the frame");

        let (human_id, agent_id) = fx.f.two_scalar_params_with_regions();
        app.engine_mut().execute(Fixture::set(HUMAN, &human_id, 0.4)).unwrap();
        app.agent.connect("Claude");
        app.agent_submit(Fixture::set(AGENT, &agent_id, 0.6)).unwrap();
        app.shell.search = human_id.split('.').next().unwrap_or_default().to_owned();
        frame(&mut app, vec![], &mut t);
        frame(&mut app, vec![], &mut t);
        shot(&mut app, &mut studio, gpu, "shell_agent_flash.png");
        app.shell.search.clear();
        app.shell.onboarding = true;
        for _ in 0..30 {
            frame(&mut app, vec![], &mut t);
        }
        shot(&mut app, &mut studio, gpu, "shell_quick_start.png");
        studio.destroy();
    }

    #[test]
    fn app_01_viewports_are_drawn_in_the_ui_z_order_under_a_modal() {
        // [APP-01] [VP-04] The viewport image is part of the UI pass, so a modal overlay drawn
        // after it covers it: with the quick start open the viewport still renders (under the
        // scrim), and where the dialog lies over the viewport the frame is the same whether the
        // viewer drew or not.
        let fx = AppFixture::new();
        let mut app = fx.app();
        let gpu = gpu();
        let mut t = 0.0;
        app.shell.onboarding = true;
        for _ in 0..30 {
            frame(&mut app, vec![], &mut t);
        }
        assert!(app.viewport_frames(1.0).next().is_some(), "viewports render under a modal");
        let v = app.viewports[0];
        let mut drawn = crate::studio::Studio::new(gpu, fonts(), 2048).unwrap();
        let with = shot_with(&mut app, &mut drawn, gpu, "zorder_modal.png", true);
        drawn.destroy();
        let mut blank = crate::studio::Studio::new(gpu, fonts(), 2048).unwrap();
        let without = shot_with(&mut app, &mut blank, gpu, "zorder_modal_no_viewer.png", false);
        blank.destroy();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let [x0, y0, x1, y1] = [v.rect.x, v.rect.y, v.rect.x + v.rect.w, v.rect.y + v.rect.h].map(|c| c as usize);
        let (mut same, mut differ, mut outside) = (0_usize, 0_usize, 0_usize);
        for y in 0..900 {
            for x in 0..1600 {
                let i = (y * 1600 + x) * 4;
                let equal = with[i..i + 4] == without[i..i + 4];
                let inside = (x0..x1).contains(&x) && (y0..y1).contains(&y);
                match (inside, equal) {
                    (true, true) => same += 1,
                    (true, false) => differ += 1,
                    (false, false) => outside += 1,
                    (false, true) => {}
                }
            }
        }
        // The dialog covers part of the viewport (same pixels), the scrim shows the image
        // through elsewhere (different pixels), nothing changes outside the viewport.
        assert_eq!(outside, 0, "the viewer changed pixels outside its rectangle");
        assert!(differ > 10_000, "the viewport shows through the scrim: {differ} pixels");
        assert!(same > 10_000, "the dialog covers the viewport: {same} pixels");
        // The dialog's centre is the window's centre, inside the viewport.
        assert!((x0..x1).contains(&800) && (y0..y1).contains(&450));
        let c = (450 * 1600 + 800) * 4;
        assert_eq!(with[c..c + 4], without[c..c + 4], "the dialog is drawn over the viewport");
    }

    #[test]
    fn perf_r1_slider_drag_frames_do_not_allocate() {
        // [PERF-R1] [CMD-03] dragging a parameter slider in the Properties panel: once the gesture
        // is open, every further frame of the drag (pointer move, slider, the command into the
        // session, the model update, the frame) allocates nothing.
        use humanart_ui::geom::Point;
        use humanart_ui::input::PointerButton;
        use humanart_ui::ui::Role;
        let fx = AppFixture::new();
        let mut app = fx.app();
        let mut t = 0.0;
        let (id, _) = fx.f.two_scalar_params_with_regions();
        let display = fx.f.registry.get(&id).unwrap().display_name.clone();
        app.shell.search.clone_from(&id);
        for _ in 0..3 {
            frame(&mut app, vec![], &mut t);
        }
        let ui = app.ui();
        let rect = ui
            .semantics()
            .iter()
            .find(|s| s.role == Role::Slider && ui.text(s.label) == display)
            .map(|s| s.rect)
            .unwrap();
        let y = rect.y + rect.h * 0.5;
        // A triangle sweep over 60 % of the track, 240 frames per period: the value (and its
        // text) keeps changing, as in a real drag, so the UI's layout cache misses and recycles.
        let x0 = rect.x + rect.w * 0.2;
        let at = |i: usize| {
            let phase = i % 240;
            #[allow(clippy::cast_precision_loss)]
            let u = if phase < 120 { phase } else { 240 - phase } as f32 / 120.0;
            Point::new(x0 + u * rect.w * 0.6, y)
        };
        frame(&mut app, vec![Event::PointerMoved(at(0))], &mut t);
        frame(&mut app, vec![Event::Button { button: PointerButton::Primary, pressed: true, pos: at(0) }], &mut t);
        // Warm-up: the gesture opens, buffers and caches reach their working size (the UI's text
        // layouts are recycled after `TextCache::MAX_AGE` frames).
        for i in 1..400 {
            frame(&mut app, vec![Event::PointerMoved(at(i))], &mut t);
        }
        let before = app.engine().session().document().params.get(&id).cloned();
        let revision = app.engine().model().revision();
        let mut events = Vec::with_capacity(8);
        let mut total = humanart_core::alloc::AllocStats::default();
        let mut worst = 0;
        let frames = 200;
        for i in 400..400 + frames {
            events.push(Event::PointerMoved(at(i)));
            let counts = humanart_core::alloc::count_in(|| {
                t += 1.0 / 60.0;
                let mut input = humanart_ui::input::InputQueue { events: std::mem::take(&mut events), time: t };
                app.frame(&mut input, humanart_ui::geom::Size::new(1600.0, 900.0), 1.0);
                events = input.events;
            });
            events.clear();
            worst = worst.max(counts.allocations + counts.reallocations);
            total.allocations += counts.allocations;
            total.reallocations += counts.reallocations;
        }
        println!(
            "slider drag: {frames} frames, {} allocations, {} reallocations, worst frame {worst}",
            total.allocations, total.reallocations
        );
        let after = app.engine().session().document().params.get(&id).cloned();
        assert_ne!(before, after, "the drag changed the parameter");
        let evaluated = app.engine().model().revision() - revision;
        println!("the model re-evaluated {evaluated} times during the {frames} frames");
        assert!(evaluated >= 100, "the model follows the drag");
        frame(&mut app, vec![Event::Button { button: PointerButton::Primary, pressed: false, pos: at(0) }], &mut t);
        assert_eq!(total.allocations + total.reallocations, 0, "[PERF-R1] drag frames allocate nothing");
    }
}
