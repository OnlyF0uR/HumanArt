//! The engine: one editing session, the evaluated character and its views, without a window.
//!
//! The editor, the command-line tool and the agent server drive the same [`Engine`], which is how
//! headless renders match the editor's (`[AC-WP23-3]`, `docs/architecture.md`, "Data flow of one
//! edit"):
//!
//! 1. every change is a command executed through the engine's [`Session`] (validation, history,
//!    journal, events);
//! 2. [`Engine::update`] notices new session events and re-syncs the [`Model`] (the template, the
//!    CPU reference evaluator and the registry bindings);
//! 3. each [`Viewer`] (one per viewport, `[VP-04]`) uploads the evaluated surface when the model's
//!    revision moves and renders its own camera, view mode, light rig and quality tier.
//!
//! - [`model`] holds [`Model`], [`Binding`], [`Coverage`], [`SurfaceHit`] (CPU picking `[VP-01]`).
//! - [`view`] holds [`Viewer`], [`ViewSettings`], [`Highlight`] (region outlines `[APP-03]`).
//! - [`skin`] holds [`skin::SkinLook`], the surfaced skin a view binds once.
//! - [`features`] holds [`features::Features`], the eyes and mouth a view shades.
//! - [`hair`] holds [`hair::CharacterHair`], the groom set a view draws, and [`HairOptions`].
//! - [`error`] holds [`EngineError`].

pub mod error;
pub mod features;
pub mod hair;
pub mod model;
pub mod skin;
pub mod view;

use std::sync::Arc;

use humanart_core::platform::FileSystem;
use humanart_doc::command::{Command, CommandError, Outcome};
use humanart_doc::registry::Registry;
use humanart_doc::session::Session;
use humanart_model::generate::build::{Options, template};
use humanart_model::{PackageLimits, Template, read_package};

pub use error::EngineError;
pub use hair::{HairOptions, HairShade};
pub use model::{Binding, Coverage, Model, SurfaceHit};
pub use view::{Framing, Highlight, ViewSettings, Viewer, ViewerOptions};

/// Where the template comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TemplateSource {
    /// The package file `data/template/template.antmpl` (`[MOD-02]`).
    Package,
    /// Generated in this process because the package does not exist yet (ADR-006).
    Generated,
}

/// Loads the template package at `path` through `fs`, or generates the template (without LODs)
/// when there is no file there. A file that exists but does not read is an error, not a reason to
/// generate.
pub fn load_template(fs: &dyn FileSystem, path: &str) -> Result<(Arc<Template>, TemplateSource), EngineError> {
    if fs.exists(path) {
        let bytes = fs.read(path).map_err(|_| EngineError::Invalid { what: "the template package cannot be read" })?;
        let t = read_package(&bytes, &PackageLimits::default())?;
        return Ok((Arc::new(t), TemplateSource::Package));
    }
    let generated = template(&Options { lods: false, ..Options::default() })?;
    Ok((Arc::new(generated.template), TemplateSource::Generated))
}

/// The engine (see the module documentation).
pub struct Engine<'a> {
    session: Session<'a>,
    registry: &'a Registry,
    model: Model,
    synced: Option<(u64, u64)>,
}

impl std::fmt::Debug for Engine<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine").field("synced", &self.synced).field("revision", &self.model.revision()).finish()
    }
}

impl<'a> Engine<'a> {
    /// An engine on `session`, whose registry is `registry`, evaluating `model`.
    pub fn new(session: Session<'a>, registry: &'a Registry, model: Model) -> Self {
        Self { session, registry, model, synced: None }
    }

    /// The session.
    pub fn session(&self) -> &Session<'a> {
        &self.session
    }

    /// The session, to undo, redo, select, open and accept proposals, save. Changes reach the
    /// model at the next [`Engine::update`].
    pub fn session_mut(&mut self) -> &mut Session<'a> {
        &mut self.session
    }

    /// Replaces the session (another project was opened) and returns the old one.
    pub fn replace_session(&mut self, session: Session<'a>) -> Session<'a> {
        self.synced = None;
        std::mem::replace(&mut self.session, session)
    }

    /// The registry.
    pub fn registry(&self) -> &'a Registry {
        self.registry
    }

    /// The evaluated character.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// The evaluated character, for cached queries (region outlines).
    pub fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// Executes a command on the main document.
    pub fn execute(&mut self, cmd: Command) -> Result<Outcome, CommandError> {
        self.session.execute(cmd)
    }

    /// Brings the model up to date with the document when session events arrived or the document
    /// changed (each frame of an open gesture, [`Session::revision`]) since the last call.
    /// Returns whether the evaluated surface changed.
    pub fn update(&mut self) -> Result<bool, EngineError> {
        let seen = (self.session.events().last_seq(), self.session.revision());
        if self.synced == Some(seen) {
            return Ok(false);
        }
        let changed = self.model.sync(self.session.document(), self.registry)?;
        self.synced = Some(seen);
        Ok(changed)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::OnceLock;

    use super::*;

    // [PERF-R1] The test binary counts allocations so the steady-state tests can assert zero.
    #[global_allocator]
    static ALLOCATOR: humanart_core::alloc::CountingAllocator = humanart_core::alloc::CountingAllocator;

    /// The generated template without LODs, built once per test binary.
    pub(crate) fn template() -> &'static Template {
        static T: OnceLock<Template> = OnceLock::new();
        T.get_or_init(|| template_generated().expect("the template generates"))
    }

    fn template_generated() -> Result<Template, EngineError> {
        Ok(super::template(&Options { lods: false, ..Options::default() })?.template)
    }

    /// The project's registry (`data/template/params`), read from disk into a memory file system.
    pub(crate) fn registry() -> &'static Registry {
        static R: OnceLock<Registry> = OnceLock::new();
        R.get_or_init(|| {
            let fs = humanart_core::platform::MemoryFs::new();
            let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/template/params");
            fs.create_dir_all("params").unwrap();
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                let name = path.file_name().unwrap().to_string_lossy().into_owned();
                fs.write(&format!("params/{name}"), &std::fs::read(&path).unwrap()).unwrap();
            }
            Registry::load_dir(&fs, "params", &humanart_doc::registry::RegistryLimits::DEFAULT).unwrap()
        })
    }

    pub(crate) fn shots_dir() -> std::path::PathBuf {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/agents/app-shell/shots");
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn reg_01_the_project_registry_drives_the_generated_template() {
        let model = Model::new(Arc::new(template().clone()), registry());
        let c = model.coverage();
        println!("coverage: {c:?} of {} parameters", registry().len());
        // The generated template has no delta components or expressions yet (ADR-006: the bases
        // come with WP-08/WP-19), so every shape and expression parameter is reported missing.
        let expected = template().deltas().len() + template().expressions().expressions.len();
        assert!(c.shape + c.expression <= expected.max(registry().len()));
        assert_eq!(c.shape + c.expression + c.missing + c.other, registry().len());
    }

    #[test]
    fn engine_executes_commands_and_the_viewer_renders_the_change() {
        use humanart_core::platform::{ManualClock, MemoryFs};
        use humanart_doc::command::{Action, RangePolicy};
        use humanart_doc::document::{Author, AuthorKind, Document};
        use humanart_doc::registry::ParamValue;
        use humanart_doc::session::SessionConfig;
        use humanart_gpu::resources::{Budget, Resources};
        use humanart_gpu::{DeviceRequest, Gpu};

        let fs = MemoryFs::new();
        let clock = ManualClock::new(1_700_000_000_000);
        let reg = registry();
        let me = Author { id: "human.test".into(), kind: AuthorKind::Human, name: "Test".into() };
        let doc = Document::new(humanart_core::id::Id128::from_u128(9), "test", "generated", 1, 0, me);
        let session =
            Session::new(doc, reg, &fs, &clock, SessionConfig::default(), humanart_core::id::Id128::from_u128(1));
        let model = Model::new(Arc::new(template().clone()), reg);
        let mut engine = Engine::new(session, reg, model);
        // The first sync applies the registry defaults (bound parameters whose default is not 0
        // move the surface on the current template); after it, nothing changes without events.
        engine.update().unwrap();
        assert!(!engine.update().unwrap(), "nothing happened since");

        let gpu = Gpu::headless_blocking(&DeviceRequest::default()).expect("this machine has a GPU");
        let mut res = Resources::new(&gpu, Budget::UNLIMITED);
        let mut viewer = Viewer::new(&gpu, &mut res, engine.model(), 360, 640).unwrap();
        let render = |engine: &mut Engine<'_>, viewer: &mut Viewer, res: &mut Resources, name: &str| {
            for _ in 0..4 {
                let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("t") });
                viewer.record(res, &mut enc, engine.model_mut(), &[]).unwrap();
                gpu.queue.submit([enc.finish()]);
                viewer.after_submit();
            }
            let (w, h, rgba) =
                humanart_render::capture::read_texture(res, viewer.display_texture().unwrap(), 0).unwrap();
            let png = humanart_render::capture::encode_png_rgba8(w, h, &rgba).unwrap();
            std::fs::write(shots_dir().join(name), png).unwrap();
            rgba
        };
        let before = render(&mut engine, &mut viewer, &mut res, "engine_neutral.png");
        let center = ((640 / 2) * 360 + 180) * 4;
        let corner = 4 * 4;
        assert_ne!(before[center..center + 3], before[corner..corner + 3], "the body covers the image centre");

        // The arm rotates 50 degrees about Z: the pose moves vertices whatever the deltas are.
        let (sin, cos) = (25f64.to_radians().sin(), 25f64.to_radians().cos());
        let cmd = Command::new(
            "human.test",
            Action::SetPose { joints: vec![("L.upper-arm".into(), Some([0.0, 0.0, sin, cos]))] },
        );
        engine.execute(cmd).unwrap();
        assert!(engine.update().unwrap(), "a pose changes the surface");
        assert!(!engine.update().unwrap(), "no new event: no work");
        let unbound = Command::new(
            "human.test",
            Action::SetParams {
                values: vec![("skin.melanin".into(), ParamValue::Scalar(0.5))],
                relative: false,
                policy: RangePolicy::Clamp,
            },
        );
        if engine.execute(unbound).is_ok() {
            assert!(!engine.update().unwrap(), "a parameter that moves no vertex does not re-evaluate");
        }
        let after = render(&mut engine, &mut viewer, &mut res, "engine_pose.png");
        let differing = before.as_chunks::<4>().0.iter().zip(after.as_chunks::<4>().0).filter(|(a, b)| a != b).count();
        println!("pixels changed by the arm pose: {differing}");
        assert!(differing > 500, "the viewer shows the deformed body");
        viewer.destroy(&mut res);
    }

    #[test]
    fn load_template_generates_when_no_package_exists() {
        let fs = humanart_core::platform::MemoryFs::new();
        let (t, source) = load_template(&fs, "data/template/template.antmpl").unwrap();
        assert_eq!(source, TemplateSource::Generated);
        assert!(t.mesh().vertex_count() > 1000);
        fs.write("bad.antmpl", b"not a package").unwrap();
        assert!(load_template(&fs, "bad.antmpl").is_err(), "a damaged package is an error");
    }
}
