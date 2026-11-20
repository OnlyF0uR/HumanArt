//! Shared helpers of the unit tests: one headless GPU per test binary (on this PC's real adapter,
//! never a window), offscreen rendering with readback, and golden screenshots stored as PNG files
//! written and read by `humanart-io::png` under `crates/ui/tests/golden/`.
//!
//! A golden image that does not exist yet is written by the first run (and reported on stderr); set
//! `HUMANART_BLESS=1` to rewrite all of them after an intended visual change. Every new or
//! rewritten golden is looked at before it is committed.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use humanart_gpu::{DeviceRequest, Gpu};
use humanart_io::image::{ColorSpace, Image, Limits, Samples};

use crate::color::Color;
use crate::paint::DrawList;
use crate::render::{AtlasImage, UiRenderer};
use crate::text::Fonts;
use crate::theme::FontFace;

/// The device every GPU test shares.
pub(crate) fn gpu() -> &'static Gpu {
    static GPU: OnceLock<Gpu> = OnceLock::new();
    GPU.get_or_init(|| Gpu::headless_blocking(&DeviceRequest::default()).expect("this machine has a GPU"))
}

/// Serialises GPU tests that share the device's queue-level state (uncaptured-error scopes).
pub(crate) fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The format of the offscreen targets: UNORM, so the shader encodes sRGB as for a typical surface.
pub(crate) const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// A renderer for [`FORMAT`] with `atlas`.
pub(crate) fn renderer(atlas: AtlasImage) -> UiRenderer {
    let gpu = gpu();
    UiRenderer::new(&gpu.device, &gpu.queue, FORMAT, atlas).expect("the UI renderer builds")
}

/// An offscreen `width × height` target of [`FORMAT`] that can be rendered to and read back.
pub(crate) fn target(width: u32, height: u32) -> wgpu::Texture {
    gpu().device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ui test target"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Records `list` into `target` with `record` (full or region), submits, and reads the whole
/// target back as tightly packed sRGB RGBA8 rows.
pub(crate) fn draw_and_read(
    renderer: &mut UiRenderer,
    list: &DrawList,
    target: &wgpu::Texture,
    record: &mut dyn FnMut(&mut UiRenderer, &mut wgpu::CommandEncoder, &wgpu::TextureView),
) -> Vec<u8> {
    let gpu = gpu();
    let (device, queue) = (&gpu.device, &gpu.queue);
    let (width, height) = (target.width(), target.height());
    renderer.prepare(device, queue, list, width, height).expect("prepare");
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ui test") });
    record(renderer, &mut encoder, &view);
    let row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ui test readback"),
        size: u64::from(row * height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None },
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    queue.submit([encoder.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("the readback maps"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("the device answers");
    let mapped = staging.slice(..).get_mapped_range().expect("mapped").to_vec();
    staging.unmap();
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height as usize {
        let start = y * row as usize;
        out.extend_from_slice(&mapped[start..start + (width * 4) as usize]);
    }
    out
}

/// Renders a finished draw list into a new `width × height` target cleared to `clear` and reads
/// it back as tightly packed sRGB RGBA8 rows.
pub(crate) fn render(renderer: &mut UiRenderer, list: &DrawList, width: u32, height: u32, clear: Color) -> Vec<u8> {
    let target = target(width, height);
    draw_and_read(renderer, list, &target, &mut |r, encoder, view| {
        r.record(encoder, view, Some(clear), list);
    })
}

/// The RGBA8 pixel at `(x, y)`.
pub(crate) fn pixel(rgba: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * width + x) * 4) as usize;
    [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
}

/// Where golden images live.
fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("golden").join(format!("{name}.png"))
}

/// Compares a rendering with its golden PNG: every channel within `tolerance` (8-bit steps) except
/// at most `max_outliers` pixels. Writes the golden when it does not exist or when
/// `HUMANART_BLESS=1`. On a mismatch the rendering is written next to it as `<name>.actual.png`.
pub(crate) fn assert_golden(name: &str, width: u32, height: u32, rgba: &[u8], tolerance: u8, max_outliers: usize) {
    let path = golden_path(name);
    let image = Image::new(humanart_io::Format::Png, width, height, 4, ColorSpace::Srgb, Samples::U8(rgba.to_vec()))
        .expect("image");
    let bless = std::env::var("HUMANART_BLESS").is_ok_and(|v| v == "1");
    if bless || !path.exists() {
        std::fs::create_dir_all(path.parent().expect("golden dir")).expect("create golden dir");
        std::fs::write(&path, humanart_io::png::encode(&image).expect("encode")).expect("write golden");
        eprintln!("wrote golden {}", path.display());
        return;
    }
    let bytes = std::fs::read(&path).expect("read golden");
    let golden = humanart_io::png::decode(&bytes, &Limits::DEFAULT).expect("decode golden");
    assert_eq!((golden.width, golden.height, golden.channels), (width, height, 4), "{name}: size");
    let expected = golden.as_u8().expect("8-bit golden");
    let mut outliers = 0;
    let mut worst = 0_u8;
    for (a, b) in rgba.as_chunks::<4>().0.iter().zip(expected.as_chunks::<4>().0) {
        let diff = a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0);
        worst = worst.max(diff);
        if diff > tolerance {
            outliers += 1;
        }
    }
    if outliers > max_outliers {
        let actual = path.with_extension("actual.png");
        std::fs::write(&actual, humanart_io::png::encode(&image).expect("encode")).expect("write actual");
        panic!("{name}: {outliers} pixels differ by more than {tolerance} (worst {worst}); see {}", actual.display());
    }
}

/// The shipped baked fonts (`data/fonts/*.anfont`, made by `tools/fontbake/bake.sh`).
pub(crate) fn fonts() -> &'static Fonts {
    static FONTS: OnceLock<Fonts> = OnceLock::new();
    FONTS.get_or_init(|| {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/fonts");
        let files =
            FontFace::ALL.map(|face| std::fs::read(dir.join(format!("{}.anfont", face.name()))).expect("baked font"));
        let [a, b, c, d, e] = &files;
        Fonts::from_bytes([a, b, c, d, e]).expect("the baked fonts load")
    })
}

/// Drives a [`crate::ui::Ui`] frame by frame with synthetic input, as the shell would.
pub(crate) struct Driver {
    /// The UI.
    pub ui: crate::ui::Ui,
    /// The clock.
    pub time: f64,
    /// Logical size of the window.
    pub size: crate::geom::Size,
    /// The scale passed to `begin_frame`.
    pub scale: f32,
    /// What the last frame asked of the shell.
    pub output: crate::ui::FrameOutput,
}

impl Driver {
    /// A driver for a `w × h` window with `theme`.
    pub fn new(theme: &crate::theme::Theme, w: f32, h: f32) -> Self {
        Self {
            ui: crate::ui::Ui::new(fonts().clone(), *theme),
            time: 0.0,
            size: crate::geom::Size::new(w, h),
            scale: 1.0,
            output: crate::ui::FrameOutput::default(),
        }
    }

    /// One frame with `events`, `dt` seconds after the last.
    pub fn frame_dt(&mut self, events: Vec<crate::input::Event>, dt: f64, build: &mut dyn FnMut(&mut crate::ui::Ui)) {
        self.time += dt;
        let mut q = crate::input::InputQueue { events, time: self.time };
        self.ui.begin_frame(&mut q, self.size, self.scale);
        build(&mut self.ui);
        self.output.clone_from(self.ui.end_frame());
    }

    /// One frame 16 ms after the last.
    pub fn frame(&mut self, events: Vec<crate::input::Event>, build: &mut dyn FnMut(&mut crate::ui::Ui)) {
        self.frame_dt(events, 0.016, build);
    }

    /// A key press with modifiers (the modifiers stay held until the next call).
    pub fn key(
        &mut self,
        key: crate::input::Key,
        mods: crate::input::Modifiers,
        build: &mut dyn FnMut(&mut crate::ui::Ui),
    ) {
        use crate::input::Event;
        self.frame(vec![Event::Modifiers(mods), Event::Key { key, pressed: true, repeat: false }], build);
    }

    /// A click at `p`: the pointer moves there in one frame (as a real pointer does before it
    /// clicks; hit testing uses last frame's rectangles), then presses and releases.
    pub fn click(&mut self, p: crate::geom::Point, build: &mut dyn FnMut(&mut crate::ui::Ui)) {
        use crate::input::{Event, PointerButton};
        self.frame(vec![Event::PointerMoved(p)], build);
        self.frame(
            vec![
                Event::Button { button: PointerButton::Primary, pressed: true, pos: p },
                Event::Button { button: PointerButton::Primary, pressed: false, pos: p },
            ],
            build,
        );
    }

    /// The rectangle of the first semantic entry labelled `label` (any role).
    pub fn rect_of(&self, label: &str) -> crate::geom::Rect {
        self.ui.semantics().iter().find(|s| self.ui.text(s.label) == label).map_or(crate::geom::Rect::ZERO, |s| s.rect)
    }

    /// The semantic entry labelled `label`.
    pub fn semantic(&self, label: &str) -> Option<crate::ui::Semantic> {
        self.ui.semantics().iter().find(|s| self.ui.text(s.label) == label).copied()
    }

    /// The label of the focused widget.
    pub fn focused_label(&self) -> String {
        let f = self.ui.focused();
        self.ui
            .semantics()
            .iter()
            .find(|s| Some(s.id) == f)
            .map(|s| self.ui.text(s.label).to_owned())
            .unwrap_or_default()
    }

    /// Renders the last frame and compares it with the golden `name`.
    pub fn golden(&self, name: &str) {
        let _lock = gpu_lock();
        let mut r = renderer(fonts().atlas());
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let (w, h) = ((self.size.w * self.scale).round() as u32, (self.size.h * self.scale).round() as u32);
        let rgba = render(&mut r, self.ui.draw_list(), w, h, self.ui.theme().palette.bg);
        assert!(r.stats().draw_calls <= 8, "[UI-02] {:?}", r.stats());
        assert_golden(name, w, h, &rgba, 2, (w * h / 1000) as usize);
    }
}
