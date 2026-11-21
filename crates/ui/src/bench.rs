//! The UI benchmark scene and its runner `[UI-10]` (scenario 10 of `[BENCH-01]`).
//!
//! [`BenchScene`] is the reference scene: a toolbar, a docking area with an outliner holding a
//! 2 000-row virtual table, three docked viewports, a properties panel of 600 widgets (150 rows of
//! label, slider, checkbox and button, all built every frame inside a scroll area) and a status
//! bar. [`run_benchmark`] renders it offscreen for a number of steady frames and records what the
//! contract asks for: UI CPU time per frame (build + layout + draw list, and the renderer's upload
//! and recording separately), draw calls, the GPU time of the UI pass (timestamp queries through
//! `humanart_gpu::profiler`, when the device has them) and the heap allocations of the CPU part
//! (when the counting allocator of `humanart_core::alloc` is installed).
//!
//! CPU times come from the clock the application installs (`humanart_core::time`); without one
//! they are 0 and [`UiBenchReport::cpu_timed`] is false. `humanart-cli bench --scenario 10` installs
//! its clock, runs this and writes [`UiBenchReport::to_value`] into its report; the shipped binary
//! keeps the system allocator, so its `allocations` is `null` (the tests count them).

use humanart_gpu::Gpu;
use humanart_gpu::profiler::GpuProfiler;
use humanart_gpu::resources::{Budget, Resources};
use humanart_io::json::{self, Object, Value};

use crate::color::Color;
use crate::controls::{IconButton, begin_status_bar, begin_toolbar, checkbox, status_item, toolbar_separator};
use crate::dock::{DockLayout, DockNode, SplitAxis, dock_area};
use crate::error::UiError;
use crate::geom::{Insets, Size};
use crate::input::InputQueue;
use crate::kit;
use crate::layout::{Align as BoxAlign, Dim, Style};
use crate::render::UiRenderer;
use crate::text::Fonts;
use crate::theme::Theme;
use crate::ui::{Area, Ui};
use crate::views::{Column, Table};
use crate::widgets::{Button, Slider, label};

/// Rows of the properties panel (each a label, a slider, a checkbox and a button: 600 widgets).
pub const BENCH_PROPERTY_ROWS: usize = 150;
/// Rows of the outliner's virtual table.
pub const BENCH_TABLE_ROWS: usize = 2000;

/// The state of the benchmark scene.
#[derive(Debug)]
pub struct BenchScene {
    layout: DockLayout,
    values: Vec<f32>,
    flags: Vec<bool>,
    names: Vec<String>,
    selected: Option<usize>,
}

impl Default for BenchScene {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchScene {
    /// The scene in its starting state.
    pub fn new() -> Self {
        let layout = DockLayout::from_root(DockNode::split(
            SplitAxis::Row,
            0.18,
            DockNode::tabs(&["outliner"]),
            DockNode::split(
                SplitAxis::Row,
                0.72,
                DockNode::split(
                    SplitAxis::Column,
                    0.6,
                    DockNode::tabs(&["viewport.1"]),
                    DockNode::split(
                        SplitAxis::Row,
                        0.5,
                        DockNode::tabs(&["viewport.2"]),
                        DockNode::tabs(&["viewport.3"]),
                    ),
                ),
                DockNode::tabs(&["properties"]),
            ),
        ));
        #[allow(clippy::cast_precision_loss)] // Small indices.
        let values = (0..BENCH_PROPERTY_ROWS).map(|i| (i % 10) as f32 / 10.0).collect();
        Self {
            layout,
            values,
            flags: (0..BENCH_PROPERTY_ROWS).map(|i| i % 3 == 0).collect(),
            names: (0..BENCH_PROPERTY_ROWS).map(|i| format!("Parameter {i:03}")).collect(),
            selected: Some(3),
        }
    }

    /// The number of interactive and labelled widgets the properties panel builds per frame.
    pub const fn property_widgets() -> usize {
        BENCH_PROPERTY_ROWS * 4
    }

    /// Builds one frame of the scene into `ui` (between `begin_frame` and `end_frame`).
    pub fn build(&mut self, ui: &mut Ui) {
        let theme = *ui.theme();
        let p = theme.palette;
        ui.begin(Style { width: Dim::Percent(100.0), height: Dim::Percent(100.0), ..Style::column() });
        begin_toolbar(ui);
        for (icon, tip) in
            [(crate::icons::BOX, "Select"), (crate::icons::SLIDERS, "Sculpt"), (crate::icons::PALETTE, "Paint")]
        {
            IconButton::new(icon, tip).show(ui);
        }
        toolbar_separator(ui);
        IconButton::new(crate::icons::PLAY, "Play").show(ui);
        ui.end();
        let Self { layout, values, flags, names, selected } = self;
        let mut content = |ui: &mut Ui, panel: &str| match panel {
            "outliner" => {
                const COLUMNS: [Column<'static>; 2] =
                    [Column::new("Region", 140.0), Column::new("Weight", 60.0).right()];
                Table::new("Regions", &COLUMNS, BENCH_TABLE_ROWS, 1000.0).show(ui, selected, &|r, c, out| {
                    #[allow(clippy::cast_precision_loss)] // Below 100.
                    let weight = (r % 100) as f32 / 100.0;
                    let _ = if c == 0 { write!(out, "Region {r}") } else { write!(out, "{weight:.2}") };
                });
            }
            "properties" => {
                let id = ui.id("properties scroll");
                ui.begin_scroll(id, Style { grow: 1.0, padding: Insets::all(8.0), gap: 4.0, ..Style::column() });
                for i in 0..BENCH_PROPERTY_ROWS {
                    ui.push_id(id.with_index(i as u64));
                    ui.begin(Style { gap: 6.0, align_items: BoxAlign::Center, ..Style::row() });
                    label(ui, &names[i]);
                    ui.end();
                    Slider::new("Value", &mut values[i], 0.0, 1.0).label_width(0.0).show(ui);
                    ui.begin(Style { gap: 6.0, align_items: BoxAlign::Center, ..Style::row() });
                    checkbox(ui, "Enabled", &mut flags[i]);
                    Button::new("Reset").show(ui);
                    ui.end();
                    ui.pop_id();
                }
                ui.end_scroll(id);
            }
            _ => {
                // A viewport: the application renders the 3D view into this rectangle.
                let n = ui.node(Style { grow: 1.0, ..Style::default() });
                kit::fill(ui, n, Area::FULL, Color::MID_GREY, 0.0);
            }
        };
        ui.begin(Style { grow: 1.0, basis: Dim::Px(0.0), ..Style::column() });
        dock_area(ui, "Workspace", layout, &bench_title, &mut content);
        ui.end();
        begin_status_bar(ui);
        status_item(ui, Some(crate::icons::CHECK_CIRCLE), "Ready", Some(p.positive));
        ui.node(Style { grow: 1.0, ..Style::default() });
        status_item(ui, None, "Benchmark scene", None);
        ui.end();
        ui.end();
    }
}

/// The tab titles of the benchmark scene's panels.
fn bench_title(panel: &str) -> &str {
    match panel {
        "outliner" => "Outliner",
        "viewport.1" => "Perspective",
        "viewport.2" => "Front",
        "viewport.3" => "Side",
        "properties" => "Properties",
        _ => panel,
    }
}

/// How to run the benchmark.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BenchOptions {
    /// Frames before measuring (caches warm, buffers grown).
    pub warmup: usize,
    /// Frames measured.
    pub frames: usize,
    /// Window size, logical pixels.
    pub size: Size,
    /// UI scale.
    pub scale: f32,
}

impl Default for BenchOptions {
    fn default() -> Self {
        Self { warmup: 30, frames: 240, size: Size::new(1920.0, 1080.0), scale: 1.0 }
    }
}

/// What the benchmark measured `[UI-10]`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UiBenchReport {
    /// Frames measured.
    pub frames: usize,
    /// Physical size of the target.
    pub target: [u32; 2],
    /// Whether a clock was installed (CPU times are 0 otherwise).
    pub cpu_timed: bool,
    /// UI build + layout + draw list per frame: median, 95th percentile, maximum (ms).
    pub build_ms: [f64; 3],
    /// Renderer upload and pass recording per frame: median, 95th percentile, maximum (ms).
    pub record_ms: [f64; 3],
    /// GPU time of the UI pass: median, 95th percentile, maximum (ms); `None` without timestamp
    /// queries.
    pub gpu_ms: Option<[f64; 3]>,
    /// Draw calls of the last frame.
    pub draw_calls: u32,
    /// Instances (primitives) of the last frame.
    pub instances: u32,
    /// Widgets with a semantic description in the last frame.
    pub semantic_widgets: usize,
    /// Heap allocations (plus reallocations) during the measured CPU parts, all frames; `None`
    /// when the counting allocator is not installed.
    pub allocations: Option<u64>,
}

impl UiBenchReport {
    /// The report as a JSON value (`humanart-cli bench` puts it into its own report).
    pub fn to_value(&self) -> Value {
        let triple = |v: [f64; 3]| {
            let mut o = Object::new();
            o.insert("median", v[0]);
            o.insert("p95", v[1]);
            o.insert("max", v[2]);
            Value::from(o)
        };
        let mut o = Object::new();
        o.insert("scenario", "ui");
        o.insert("frames", self.frames);
        o.insert("target", Value::Array(vec![self.target[0].into(), self.target[1].into()]));
        o.insert("cpu_timed", self.cpu_timed);
        o.insert("build_ms", triple(self.build_ms));
        o.insert("record_ms", triple(self.record_ms));
        o.insert("gpu_ms", self.gpu_ms.map_or(Value::Null, triple));
        o.insert("draw_calls", self.draw_calls);
        o.insert("instances", self.instances);
        o.insert("semantic_widgets", self.semantic_widgets);
        o.insert("allocations", self.allocations.map_or(Value::Null, Value::from));
        Value::from(o)
    }

    /// The report as JSON text.
    pub fn to_json(&self) -> Result<String, UiError> {
        json::to_string_pretty(&self.to_value()).map_err(|e| UiError::DockLayout(format!("bench report: {e}")))
    }
}

/// Median, 95th percentile and maximum of `samples` (nanoseconds) in milliseconds.
#[allow(clippy::cast_precision_loss)] // Nanosecond counts far below 2^52.
fn stats_ms(samples: &mut [u64]) -> [f64; 3] {
    if samples.is_empty() {
        return [0.0; 3];
    }
    samples.sort_unstable();
    let at = |q: f64| {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let i = ((samples.len() - 1) as f64 * q).round() as usize;
        samples[i.min(samples.len() - 1)] as f64 / 1e6
    };
    [at(0.5), at(0.95), at(1.0)]
}

/// Renders the benchmark scene offscreen on `gpu` and measures it `[UI-10]`.
#[allow(clippy::too_many_lines)]
pub fn run_benchmark(gpu: &Gpu, fonts: &Fonts, options: &BenchOptions) -> Result<UiBenchReport, UiError> {
    let (device, queue) = (&gpu.device, &gpu.queue);
    let scale = crate::ui::clamp_ui_scale(options.scale);
    let size = Size::new(options.size.w.clamp(64.0, 8192.0), options.size.h.clamp(64.0, 8192.0));
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Clamped above.
    let (w, h) = ((size.w * scale).round() as u32, (size.h * scale).round() as u32);
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let mut renderer = UiRenderer::new(device, queue, format, fonts.atlas())?;
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ui bench target"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut resources = Resources::new(gpu, Budget::UNLIMITED);
    let mut profiler = GpuProfiler::new(gpu, &mut resources, 3, 4)?;
    profiler.set_enabled(true);
    let theme = Theme::dark();
    let mut ui = Ui::new(fonts.clone(), theme);
    ui.set_background(Some(theme.palette.bg));
    let mut scene = BenchScene::new();
    let total = options.warmup + options.frames;
    let mut build = Vec::with_capacity(options.frames);
    let mut record = Vec::with_capacity(options.frames);
    let mut gpu_ns = Vec::with_capacity(options.frames);
    let counting = humanart_core::alloc::is_installed();
    let mut allocations = 0_u64;
    let mut time = 0.0_f64;
    let mut events = Vec::new();
    let mut report = UiBenchReport {
        frames: options.frames,
        target: [w, h],
        cpu_timed: humanart_core::time::clock_installed(),
        ..UiBenchReport::default()
    };
    for frame in 0..total {
        let measured = frame >= options.warmup;
        time += 1.0 / 60.0;
        let mut input = InputQueue { events: std::mem::take(&mut events), time };
        let t0 = humanart_core::time::now_nanos();
        let mut cpu = || {
            ui.begin_frame(&mut input, size, scale);
            scene.build(&mut ui);
            ui.end_frame();
        };
        if measured && counting {
            let s = humanart_core::alloc::count_in(&mut cpu);
            allocations += s.allocations + s.reallocations;
        } else {
            cpu();
        }
        let t1 = humanart_core::time::now_nanos();
        events = input.events;
        events.clear();
        profiler.begin_frame();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ui bench") });
        renderer.prepare(device, queue, ui.draw_list(), w, h)?;
        let writes = profiler.render_pass_writes("ui");
        renderer.record_timed(&mut encoder, &view, Some(theme.palette.bg), ui.draw_list(), writes);
        profiler.end_frame(&mut encoder);
        let t2 = humanart_core::time::now_nanos();
        queue.submit([encoder.finish()]);
        profiler.after_submit();
        humanart_gpu::device::wait_for(device, None, humanart_gpu::device::GPU_WAIT_LIMIT, "the queued work")
            .map_err(UiError::Gpu)?;
        if profiler.collect()
            && measured
            && let Some(ns) = profiler.latest().gpu_total_nanos()
        {
            gpu_ns.push(ns);
        }
        if measured {
            build.push(t1.saturating_sub(t0));
            record.push(t2.saturating_sub(t1));
        }
    }
    // Late timestamp results of the last frames.
    for _ in 0..3 {
        humanart_gpu::device::wait_for(device, None, humanart_gpu::device::GPU_WAIT_LIMIT, "the queued work")
            .map_err(UiError::Gpu)?;
        if profiler.collect()
            && let Some(ns) = profiler.latest().gpu_total_nanos()
        {
            gpu_ns.push(ns);
        }
    }
    profiler.destroy(&mut resources);
    report.build_ms = stats_ms(&mut build);
    report.record_ms = stats_ms(&mut record);
    report.gpu_ms = (!gpu_ns.is_empty()).then(|| stats_ms(&mut gpu_ns));
    report.draw_calls = renderer.stats().draw_calls;
    report.instances = renderer.stats().instances;
    report.semantic_widgets = ui.semantics().len();
    report.allocations = counting.then_some(allocations);
    Ok(report)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::tests::{fonts, gpu, gpu_lock};

    /// A test clock on `Instant` (test code only; the application installs the platform clock).
    struct TestClock(std::time::Instant);

    impl humanart_core::platform::Clock for TestClock {
        fn now_nanos(&self) -> u64 {
            u64::try_from(self.0.elapsed().as_nanos()).unwrap_or(u64::MAX)
        }

        fn unix_millis(&self) -> u64 {
            0
        }
    }

    #[test]
    fn ac_wp12_1_ui_benchmark_scene_records_its_contract() {
        // [UI-10] [AC-WP12-1] [PERF-R1] 600 widgets, a 2000-row virtual table, three docked
        // viewports: CPU time, draw calls, GPU time of the UI pass, allocations.
        let _lock = gpu_lock();
        humanart_core::time::install_clock(Arc::new(TestClock(std::time::Instant::now())));
        let options = BenchOptions { warmup: 30, frames: 120, ..BenchOptions::default() };
        let report = run_benchmark(gpu(), fonts(), &options).expect("the benchmark runs");
        eprintln!("{}", report.to_json().expect("json"));
        assert!(report.cpu_timed);
        assert_eq!(report.allocations, Some(0), "[PERF-R1] steady frames do not allocate");
        assert!(report.draw_calls <= 8, "[UI-02] {} draw calls", report.draw_calls);
        assert!(report.semantic_widgets >= BenchScene::property_widgets(), "{}", report.semantic_widgets);
        assert!(report.build_ms[0] > 0.0 && report.record_ms[0] > 0.0);
        assert_eq!(report.target, [1920, 1080]);
    }

    #[test]
    fn ui_10_benchmark_scene_golden() {
        // The benchmark scene as the benchmark draws it, looked at by eye.
        let mut d = crate::tests::Driver::new(&Theme::dark().with_reduced_motion(true), 1280.0, 720.0);
        let mut scene = BenchScene::new();
        let mut build = |ui: &mut Ui| scene.build(ui);
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        d.golden("bench_scene");
    }
}
