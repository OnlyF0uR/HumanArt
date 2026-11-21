//! `humanart-cli bench`: the benchmark scenarios, run headless on the machine's GPU.
//!
//! One scenario exists so far: **scenario 10, the UI** `[UI-10]`. It renders the reference scene
//! of `humanart_ui::bench` (a toolbar, a docked outliner with a 2 000-row virtual table, three
//! viewports, a properties panel of 600 widgets, a status bar) offscreen for `--frames` steady
//! frames after `--warmup` and reports, per frame, the UI's CPU time (build + layout + draw list,
//! and the renderer's upload and pass recording), the GPU time of the UI pass (when the device has
//! timestamp queries), the draw calls and the instances. The other scenarios of the benchmark
//! suite (todo 5) are not written yet: asking for one is a usage error, not an empty report.
//!
//! The report is JSON, on stdout or in `--out`. Timings are those of the machine as it is: run it
//! on an idle machine, a GPU that is busy with something else gives numbers that mean nothing.
//!
//! **Allocations.** The UI report has an `allocations` field (`[PERF-R1]`, heap calls of the
//! measured CPU parts). It is counted by `humanart_core::alloc::CountingAllocator`, which only
//! test binaries install: the shipped command keeps the system allocator and reports `null`. This
//! crate's test binary installs it, so the test of this module shows the count through the
//! command's own path.

use std::path::PathBuf;
use std::sync::Arc;

use humanart_gpu::{DeviceRequest, Gpu};
use humanart_io::json::{Object, Value, to_string_pretty};
use humanart_platform::SystemClock;
use humanart_ui::bench::{BenchOptions, UiBenchReport, run_benchmark};
use humanart_ui::geom::Size;
use humanart_ui::text::Fonts;
use humanart_ui::theme::FontFace;
use humanart_ui::ui::UI_SCALE_RANGE;

use crate::args::{CommandSpec, OptSpec, Parsed, UsageError};
use crate::error::CliError;
use crate::host;

/// The number of the UI scenario in the benchmark suite.
pub const UI_SCENARIO: u32 = 10;

/// The names `--scenario` takes (all of them the UI scenario).
const SCENARIOS: [&str; 2] = ["10", "ui"];

const OPTIONS: [OptSpec; 9] = [
    OptSpec {
        name: "scenario",
        value: Some("N"),
        default: None,
        help: "The scenario to run: 10 (or ui), the UI benchmark scene. The others are not written yet.",
    },
    OptSpec { name: "frames", value: Some("N"), default: Some("240"), help: "Frames measured." },
    OptSpec { name: "warmup", value: Some("N"), default: Some("30"), help: "Frames run before measuring." },
    OptSpec { name: "width", value: Some("PX"), default: Some("1920"), help: "Window width, logical pixels." },
    OptSpec { name: "height", value: Some("PX"), default: Some("1080"), help: "Window height, logical pixels." },
    OptSpec { name: "scale", value: Some("X"), default: Some("1"), help: "UI scale, 0.75 to 2." },
    OptSpec {
        name: "data",
        value: Some("DIR"),
        default: Some("data"),
        help: "Data folder (the baked UI fonts are read from its fonts folder).",
    },
    OptSpec { name: "out", value: Some("FILE"), default: None, help: "Write the JSON report here instead of stdout." },
    host::JSON_ERRORS,
];

/// The `bench` command.
pub static BENCH: CommandSpec = CommandSpec {
    name: "bench",
    summary: "Runs a benchmark scenario headless and reports its timings as JSON.",
    options: &OPTIONS,
};

/// What `bench` was asked to do (scenario 10).
#[derive(Clone, Debug, PartialEq)]
pub struct BenchJob {
    /// Frames measured.
    pub frames: u32,
    /// Frames run before measuring.
    pub warmup: u32,
    /// Window width, logical pixels.
    pub width: u32,
    /// Window height, logical pixels.
    pub height: u32,
    /// UI scale.
    pub scale: f32,
    /// The data folder.
    pub data: PathBuf,
    /// Where the JSON report goes (`None`: stdout).
    pub out: Option<PathBuf>,
}

impl BenchJob {
    /// The job of a parsed `bench` command line.
    ///
    /// # Errors
    /// [`CliError::Usage`] when `--scenario` is missing or names a scenario that does not exist,
    /// or a number is out of range.
    pub fn from_args(p: &Parsed) -> Result<Self, CliError> {
        let Some(scenario) = p.get("scenario") else {
            return Err(CliError::Usage(UsageError("bench: --scenario is required (10, the UI scene)".into())));
        };
        if !SCENARIOS.contains(&scenario) {
            return Err(CliError::Usage(UsageError(format!(
                "bench: scenario '{scenario}' is not written; the one scenario is 10 (ui)"
            ))));
        }
        let text = p.get("scale").unwrap_or("1");
        let scale: f32 = text.parse().map_err(|_| UsageError(format!("--scale: '{text}' is not a number")))?;
        if !(scale.is_finite() && (UI_SCALE_RANGE.0..=UI_SCALE_RANGE.1).contains(&scale)) {
            return Err(CliError::Usage(UsageError(format!(
                "--scale: {text} is outside {}..={}",
                UI_SCALE_RANGE.0, UI_SCALE_RANGE.1
            ))));
        }
        Ok(Self {
            frames: p.number("frames", 1, 100_000)?,
            warmup: p.number("warmup", 0, 10_000)?,
            width: p.number("width", 64, 8192)?,
            height: p.number("height", 64, 8192)?,
            scale,
            data: PathBuf::from(p.get("data").unwrap_or("data")),
            out: p.get("out").map(PathBuf::from),
        })
    }
}

/// What a `bench` run measured.
#[derive(Clone, Debug, PartialEq)]
pub struct BenchSummary {
    /// The GPU adapter.
    pub adapter: String,
    /// The graphics backend.
    pub backend: String,
    /// Frames run before measuring.
    pub warmup: u32,
    /// UI scale.
    pub scale: f32,
    /// The UI scenario's measurements.
    pub ui: UiBenchReport,
}

impl BenchSummary {
    /// The report as JSON: the scenario, the device, the run's options and the measurements.
    ///
    /// # Errors
    /// [`CliError::Data`] when the report cannot be written as JSON (a non-finite timing).
    pub fn to_json(&self) -> Result<String, CliError> {
        let mut o = Object::new();
        o.insert("scenario", UI_SCENARIO);
        o.insert("name", "ui");
        o.insert("adapter", self.adapter.as_str());
        o.insert("backend", self.backend.as_str());
        o.insert("warmup", self.warmup);
        o.insert("scale", f64::from(self.scale));
        o.insert("ui", self.ui.to_value());
        to_string_pretty(&Value::from(o)).map_err(|e| CliError::Data(format!("bench report: {e}")))
    }

    /// One line for the terminal when the report went to a file.
    pub fn line(&self) -> String {
        let gpu = self.ui.gpu_ms.map_or_else(|| "not timed".to_owned(), |ms| format!("{:.3} ms", ms[0]));
        format!(
            "scenario {UI_SCENARIO} (ui) on {} ({}): {} frames at {}x{}, build {:.3} ms, record {:.3} ms, GPU {gpu} (medians), {} draw calls",
            self.adapter,
            self.backend,
            self.ui.frames,
            self.ui.target[0],
            self.ui.target[1],
            self.ui.build_ms[0],
            self.ui.record_ms[0],
            self.ui.draw_calls
        )
    }
}

/// Reads the five baked UI fonts of `<data>/fonts`.
fn load_fonts(job: &BenchJob) -> Result<Fonts, CliError> {
    let dir = job.data.join("fonts");
    let mut files: [Vec<u8>; 5] = Default::default();
    for (slot, face) in files.iter_mut().zip(FontFace::ALL) {
        let path = dir.join(format!("{}.anfont", face.name()));
        *slot = std::fs::read(&path).map_err(|e| CliError::io(&path, e))?;
    }
    let [a, b, c, d, e] = &files;
    Fonts::from_bytes([a, b, c, d, e]).map_err(|e| CliError::Data(format!("{}: {e}", dir.display())))
}

/// Measures `job` on `gpu` with `fonts`. Installs the process clock when none is installed yet,
/// so the CPU times are measured.
///
/// # Errors
/// [`CliError::Gpu`] when the scene cannot be drawn or the GPU does not answer in time.
pub fn measure(gpu: &Gpu, fonts: &Fonts, job: &BenchJob) -> Result<BenchSummary, CliError> {
    // Refused when a clock is already installed: that one keeps measuring.
    let _ = humanart_core::time::install_clock(Arc::new(SystemClock::new()));
    #[allow(clippy::cast_precision_loss)] // INVARIANT: sizes are at most 8192 (from_args).
    let size = Size::new(job.width as f32, job.height as f32);
    let options = BenchOptions { warmup: job.warmup as usize, frames: job.frames as usize, size, scale: job.scale };
    let ui = run_benchmark(gpu, fonts, &options).map_err(|e| CliError::Gpu(e.to_string()))?;
    let info = gpu.adapter_info();
    Ok(BenchSummary {
        adapter: info.name.clone(),
        backend: format!("{:?}", info.backend),
        warmup: job.warmup,
        scale: job.scale,
        ui,
    })
}

/// Runs `job`: reads the fonts, measures on a new headless device and writes the report to
/// `job.out` when it is set.
///
/// # Errors
/// [`CliError::Io`] / [`CliError::Data`] for the fonts (before any GPU work), [`CliError::Gpu`]
/// when there is no adapter or device or [`measure`] fails, and [`CliError::Io`] when the report
/// cannot be written.
pub fn run(job: &BenchJob) -> Result<BenchSummary, CliError> {
    let fonts = load_fonts(job)?;
    let gpu = Gpu::headless_blocking(&DeviceRequest::default()).map_err(|e| CliError::Gpu(e.to_string()))?;
    let summary = measure(&gpu, &fonts, job)?;
    if let Some(out) = &job.out {
        std::fs::write(out, summary.to_json()?).map_err(|e| CliError::io(out, e))?;
    }
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use humanart_io::json::parse_str;

    use super::*;

    fn job(args: &[&str]) -> Result<BenchJob, CliError> {
        let args: Vec<String> = args.iter().map(|s| (*s).to_owned()).collect();
        BenchJob::from_args(&Parsed::parse(&BENCH, &args)?)
    }

    #[test]
    fn ui_10_bench_command_lines() {
        let d = job(&["--scenario", "10"]).unwrap();
        assert_eq!((d.frames, d.warmup, d.width, d.height), (240, 30, 1920, 1080));
        assert!((d.scale - 1.0).abs() < f32::EPSILON && d.out.is_none() && d.data == std::path::Path::new("data"));
        let j = job(&["--scenario=ui", "--frames", "60", "--warmup=0", "--width", "1280", "--height", "720"]).unwrap();
        assert_eq!((j.frames, j.warmup, j.width, j.height), (60, 0, 1280, 720));
        let j = job(&["--scenario", "10", "--scale", "1.5", "--out", "ui.json"]).unwrap();
        assert!((j.scale - 1.5).abs() < f32::EPSILON);
        assert_eq!(j.out, Some(PathBuf::from("ui.json")));
        for bad in [
            &[][..],
            &["--scenario", "3"],
            &["--scenario", "hair"],
            &["--scenario", "10", "--frames", "0"],
            &["--scenario", "10", "--width", "32"],
            &["--scenario", "10", "--scale", "4"],
            &["--scenario", "10", "--scale", "NaN"],
            &["--scenario", "10", "--scale", "big"],
            &["10"],
        ] {
            assert!(matches!(job(bad), Err(CliError::Usage(_))), "{bad:?}");
        }
    }

    #[test]
    fn ui_10_bench_scenario_10_measures_the_ui_scene() {
        // [UI-10] (todo 6) `bench --scenario 10` runs `humanart_ui::bench::run_benchmark` on the
        // repository's fonts and reports its contract as JSON. No timing bound: the numbers are
        // the machine's. The process's shared device (ADR-040) stands for the command's own.
        let gpu = humanart_gpu::shared_headless().unwrap_or_else(|e| panic!("this machine has a GPU: {e}"));
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let j = BenchJob {
            frames: 30,
            warmup: 30,
            width: 1280,
            height: 720,
            data: root.join("data"),
            ..job(&["--scenario", "10"]).unwrap()
        };
        let summary = measure(gpu, &load_fonts(&j).unwrap(), &j).unwrap();
        let ui = &summary.ui;
        assert_eq!((ui.frames, ui.target), (30, [1280, 720]));
        assert!(ui.cpu_timed && ui.build_ms[0] > 0.0 && ui.record_ms[0] > 0.0, "{ui:?}");
        assert!(ui.build_ms[0] <= ui.build_ms[1] && ui.build_ms[1] <= ui.build_ms[2], "{ui:?}");
        assert!((1..=8).contains(&ui.draw_calls) && ui.instances >= 100, "{ui:?}");
        assert!(ui.semantic_widgets >= humanart_ui::bench::BenchScene::property_widgets(), "{ui:?}");
        // This test binary counts allocations (see `main.rs`): the steady frames make none.
        assert_eq!(ui.allocations, Some(0), "[PERF-R1] steady UI frames do not allocate");
        assert!(!summary.adapter.is_empty() && !summary.backend.is_empty());
        let text = summary.to_json().unwrap();
        let report = parse_str(&text).unwrap();
        assert_eq!(report.get("scenario").and_then(Value::as_u64), Some(10));
        assert_eq!(report.get("name").and_then(Value::as_str), Some("ui"));
        assert_eq!(report.get("warmup").and_then(Value::as_u64), Some(30));
        let inner = report.get("ui").unwrap();
        assert_eq!(inner.get("frames").and_then(Value::as_u64), Some(30));
        for key in ["build_ms", "record_ms", "gpu_ms", "draw_calls", "instances", "semantic_widgets", "allocations"] {
            assert!(inner.get(key).is_some(), "{key} in {text}");
        }
        assert!(summary.line().contains("scenario 10 (ui)") && summary.line().contains("30 frames at 1280x720"));
        eprintln!("{}", summary.line());
        // Fonts that are not there are a file error, before anything is drawn.
        let missing = BenchJob { data: root.join("no-such-data-folder"), ..j };
        assert!(matches!(run(&missing), Err(CliError::Io { .. })));
    }
}
