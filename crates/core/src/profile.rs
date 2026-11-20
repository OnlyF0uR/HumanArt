//! The CPU scope profiler `[PERF-R6]`: always compiled in, near free when off, with per-scope
//! statistics for the on-screen overlay and a machine-readable dump.
//!
//! ```
//! use humanart_core::profile_scope;
//! fn evaluate_layers() {
//!     profile_scope!("doc.evaluate");
//!     // ... measured until the end of the block
//! }
//! humanart_core::profile::global().set_enabled(true);
//! evaluate_layers();
//! humanart_core::profile::global().end_frame();
//! ```
//!
//! - A scope ([`Profiler::scope`], [`scope`], [`profile_scope!`](crate::profile_scope)) measures
//!   from its creation to the drop of its guard on the [installed clock](crate::time), and records
//!   name, thread lane, nesting depth, start and duration. Disabled, a scope is one atomic load.
//! - [`Profiler::record`] adds a measurement taken elsewhere: GPU timestamp queries go on
//!   [`GPU_LANE`] when their asynchronous readback arrives `[PERF-R3]`.
//! - [`Profiler::end_frame`] closes the frame: its events become the "last frame" (for the overlay's
//!   timeline and [`Profiler::write_trace_json`]), per-name statistics ([`ScopeStats`]) are updated,
//!   and the frame time joins the history graph.
//!
//! **Cost.** Enabling reserves room for [`MAX_EVENTS`] events per frame and [`MAX_SCOPES`] names;
//! after that nothing allocates `[PERF-R1]`: events past the limit are counted as dropped, names past
//! the limit get no statistics. Recording takes one uncontended lock per scope.
//!
//! The machine-readable dump is the Chrome trace-event JSON format (`chrome://tracing`, Perfetto),
//! with one track per thread lane.

use std::cell::Cell;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use crate::collections::{ArrayVec, RingBuffer};
use crate::platform::Clock;
use crate::time::{self, DurationText};

/// Events kept per frame; more are counted in [`FrameInfo::dropped`].
pub const MAX_EVENTS: usize = 8192;
/// Distinct scope names that get statistics.
pub const MAX_SCOPES: usize = 256;
/// Frame times kept for the history graph.
pub const FRAME_HISTORY: usize = 240;
/// Thread lanes that can be given a name.
pub const MAX_LANE_NAMES: usize = 64;
/// The lane of measurements taken on the GPU.
pub const GPU_LANE: u32 = u32::MAX;
/// Weight of the newest frame in [`ScopeStats::average_nanos`] (an exponential moving average).
pub const AVERAGE_WEIGHT: f64 = 0.1;

/// One measured scope.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScopeEvent {
    /// The scope's name.
    pub name: &'static str,
    /// The thread lane ([`current_lane`]), or [`GPU_LANE`].
    pub lane: u32,
    /// How many scopes of the same thread were open around this one.
    pub depth: u16,
    /// Monotonic start time in nanoseconds.
    pub start_nanos: u64,
    /// Duration in nanoseconds.
    pub duration_nanos: u64,
}

/// Statistics of one scope name over the frames, for the overlay.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScopeStats {
    /// The scope's name.
    pub name: &'static str,
    /// Calls in the last frame.
    pub calls: u32,
    /// Total time of those calls (a scope nested in itself counts twice).
    pub last_nanos: u64,
    /// Exponential moving average of the per-frame total ([`AVERAGE_WEIGHT`]).
    pub average_nanos: f64,
    /// The largest per-frame total since the statistics were reset.
    pub max_nanos: u64,
    /// Frames since the name first appeared.
    pub frames: u64,
}

/// What [`Profiler::last_frame`] reports about the last closed frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameInfo {
    /// The frame's number, counting from 0.
    pub index: u64,
    /// Monotonic time at which the frame started.
    pub start_nanos: u64,
    /// The frame's duration.
    pub duration_nanos: u64,
    /// Events recorded.
    pub events: usize,
    /// Events that did not fit in [`MAX_EVENTS`].
    pub dropped: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct LaneName {
    lane: u32,
    name: &'static str,
}

struct FrameData {
    index: u64,
    start: u64,
    current: Vec<ScopeEvent>,
    dropped: u64,
    previous: Vec<ScopeEvent>,
    last: FrameInfo,
    stats: Vec<ScopeStats>,
    history: RingBuffer<u64, FRAME_HISTORY>,
}

static NEXT_LANE: AtomicU32 = AtomicU32::new(1);

thread_local! {
    /// This thread's lane, 0 until first asked.
    static LANE: Cell<u32> = const { Cell::new(0) };
    /// Scopes open on this thread.
    static DEPTH: Cell<u16> = const { Cell::new(0) };
}

/// The lane number of the calling thread: a small number given out on first use, stable for the
/// life of the thread.
pub fn current_lane() -> u32 {
    LANE.with(|lane| {
        if lane.get() == 0 {
            lane.set(NEXT_LANE.fetch_add(1, Ordering::Relaxed));
        }
        lane.get()
    })
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    // Nothing that can panic runs under these locks except the caller's visitors, and the data stays
    // consistent at every point, so a poisoned lock is used as it is.
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A profiler. The program uses the [`global`] one; separate ones are for tests and tools.
pub struct Profiler {
    enabled: AtomicBool,
    clock: Option<Arc<dyn Clock>>,
    data: Mutex<FrameData>,
    lanes: Mutex<ArrayVec<LaneName, MAX_LANE_NAMES>>,
}

impl fmt::Debug for Profiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Profiler")
            .field("enabled", &self.is_enabled())
            .field("last_frame", &self.last_frame())
            .finish_non_exhaustive()
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Profiler {
    /// A disabled profiler on the [installed clock](crate::time::install_clock).
    pub fn new() -> Self {
        Self::build(None)
    }

    /// A disabled profiler on its own clock.
    pub fn with_clock(clock: Arc<dyn Clock>) -> Self {
        Self::build(Some(clock))
    }

    fn build(clock: Option<Arc<dyn Clock>>) -> Self {
        let data = FrameData {
            index: 0,
            start: 0,
            current: Vec::new(),
            dropped: 0,
            previous: Vec::new(),
            last: FrameInfo::default(),
            stats: Vec::new(),
            history: RingBuffer::new(),
        };
        Self { enabled: AtomicBool::new(false), clock, data: Mutex::new(data), lanes: Mutex::new(ArrayVec::new()) }
    }

    fn now(&self) -> u64 {
        self.clock.as_ref().map_or_else(time::now_nanos, |clock| clock.now_nanos())
    }

    /// `true` while scopes are recorded.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Turns recording on or off. Turning it on reserves the event and statistics buffers (the only
    /// allocation the profiler makes) and starts a new frame.
    pub fn set_enabled(&self, enabled: bool) {
        if enabled {
            let now = self.now();
            let mut guard = lock(&self.data);
            let data = &mut *guard;
            for events in [&mut data.current, &mut data.previous] {
                events.reserve_exact(MAX_EVENTS.saturating_sub(events.len()));
            }
            let named = data.stats.len();
            data.stats.reserve_exact(MAX_SCOPES.saturating_sub(named));
            data.start = now;
        }
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Clears the statistics, the frame history and the events of the current frame.
    pub fn reset(&self) {
        let mut data = lock(&self.data);
        data.stats.clear();
        data.history.clear();
        data.current.clear();
        data.dropped = 0;
    }

    /// Starts measuring a scope; the measurement ends when the guard is dropped.
    #[inline]
    pub fn scope(&self, name: &'static str) -> Scope<'_> {
        if !self.is_enabled() {
            return Scope { profiler: None, name, depth: 0, start: 0 };
        }
        let depth = DEPTH.with(|depth| {
            let outer = depth.get();
            depth.set(outer.saturating_add(1));
            outer
        });
        Scope { profiler: Some(self), name, depth, start: self.now() }
    }

    /// Adds a measurement taken elsewhere (GPU timestamps on [`GPU_LANE`]) to the current frame.
    pub fn record(&self, name: &'static str, lane: u32, start_nanos: u64, duration_nanos: u64) {
        if self.is_enabled() {
            self.push(ScopeEvent { name, lane, depth: 0, start_nanos, duration_nanos });
        }
    }

    fn push(&self, event: ScopeEvent) {
        let mut data = lock(&self.data);
        if data.current.len() < data.current.capacity() {
            data.current.push(event);
        } else {
            data.dropped += 1;
        }
    }

    /// Gives the calling thread's lane a name for the dump (`"main"`, `"job-0"`). Returns `false`
    /// when [`MAX_LANE_NAMES`] other lanes already have one.
    pub fn name_thread(&self, name: &'static str) -> bool {
        self.name_lane(current_lane(), name)
    }

    /// Gives a lane a name for the dump. Returns `false` when the table is full.
    pub fn name_lane(&self, lane: u32, name: &'static str) -> bool {
        let mut lanes = lock(&self.lanes);
        if let Some(existing) = lanes.iter_mut().find(|entry| entry.lane == lane) {
            existing.name = name;
            return true;
        }
        lanes.push(LaneName { lane, name }).is_ok()
    }

    /// Closes the current frame and starts the next (call once per frame, after presenting).
    /// Does nothing while the profiler is disabled.
    pub fn end_frame(&self) {
        if !self.is_enabled() {
            return;
        }
        let now = self.now();
        let mut guard = lock(&self.data);
        let data = &mut *guard;
        std::mem::swap(&mut data.current, &mut data.previous);
        data.current.clear();
        let duration = now.saturating_sub(data.start);
        data.last = FrameInfo {
            index: data.index,
            start_nanos: data.start,
            duration_nanos: duration,
            events: data.previous.len(),
            dropped: data.dropped,
        };
        data.dropped = 0;
        data.index += 1;
        data.start = now;
        data.history.push(duration);

        for stats in &mut data.stats {
            stats.calls = 0;
            stats.last_nanos = 0;
        }
        for event in &data.previous {
            let index = match data.stats.iter().position(|stats| stats.name == event.name) {
                Some(index) => index,
                // Within the reserved capacity, so this never allocates.
                None if data.stats.len() < MAX_SCOPES => {
                    data.stats.push(ScopeStats { name: event.name, ..ScopeStats::default() });
                    data.stats.len() - 1
                }
                None => continue,
            };
            if let Some(stats) = data.stats.get_mut(index) {
                stats.calls = stats.calls.saturating_add(1);
                stats.last_nanos = stats.last_nanos.saturating_add(event.duration_nanos);
            }
        }
        for stats in &mut data.stats {
            let last = stats.last_nanos as f64;
            stats.average_nanos = if stats.frames == 0 {
                last
            } else {
                stats.average_nanos + AVERAGE_WEIGHT * (last - stats.average_nanos)
            };
            stats.max_nanos = stats.max_nanos.max(stats.last_nanos);
            stats.frames += 1;
        }
    }

    /// The last closed frame.
    pub fn last_frame(&self) -> FrameInfo {
        lock(&self.data).last
    }

    /// Visits the events of the last closed frame in the order they ended. `visit` runs under the
    /// profiler's lock: it must not use this profiler.
    pub fn for_each_event(&self, mut visit: impl FnMut(&ScopeEvent)) {
        lock(&self.data).previous.iter().for_each(&mut visit);
    }

    /// Visits the statistics of every scope name, in order of first appearance. `visit` runs under
    /// the profiler's lock: it must not use this profiler.
    pub fn for_each_stat(&self, mut visit: impl FnMut(&ScopeStats)) {
        lock(&self.data).stats.iter().for_each(&mut visit);
    }

    /// Visits the durations of the last [`FRAME_HISTORY`] frames, oldest first.
    pub fn for_each_frame_time(&self, mut visit: impl FnMut(u64)) {
        lock(&self.data).history.iter().for_each(|&nanos| visit(nanos));
    }

    /// Writes the last closed frame as Chrome trace-event JSON: one complete event (`"ph":"X"`) per
    /// scope with times in microseconds, a `thread_name` record per named lane, and the frame's
    /// number, duration and dropped count under `"otherData"`.
    pub fn write_trace_json(&self, out: &mut dyn fmt::Write) -> fmt::Result {
        let lanes = *lock(&self.lanes);
        let data = lock(&self.data);
        out.write_str("{\"traceEvents\":[")?;
        let mut first = true;
        for lane in &lanes {
            if !first {
                out.write_char(',')?;
            }
            first = false;
            write!(
                out,
                "\n{{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":{},\"args\":{{\"name\":",
                lane.lane
            )?;
            write_json_string(out, lane.name)?;
            out.write_str("}}")?;
        }
        for event in &data.previous {
            if !first {
                out.write_char(',')?;
            }
            first = false;
            out.write_str("\n{\"name\":")?;
            write_json_string(out, event.name)?;
            let category = if event.lane == GPU_LANE { "gpu" } else { "cpu" };
            write!(out, ",\"cat\":\"{category}\",\"ph\":\"X\",\"ts\":")?;
            write_micros(out, event.start_nanos)?;
            out.write_str(",\"dur\":")?;
            write_micros(out, event.duration_nanos)?;
            write!(out, ",\"pid\":1,\"tid\":{},\"args\":{{\"depth\":{}}}}}", event.lane, event.depth)?;
        }
        let last = data.last;
        write!(
            out,
            "\n],\"displayTimeUnit\":\"ns\",\"otherData\":{{\"frame\":{},\"frame_nanos\":{},\"dropped\":{}}}}}\n",
            last.index, last.duration_nanos, last.dropped
        )
    }

    /// Writes the statistics as a text table, one scope per line: name, calls, last, average, max.
    pub fn write_summary(&self, out: &mut dyn fmt::Write) -> fmt::Result {
        let data = lock(&self.data);
        writeln!(
            out,
            "frame {} took {} ({} events, {} dropped)",
            data.last.index,
            DurationText(data.last.duration_nanos),
            data.last.events,
            data.last.dropped
        )?;
        for stats in &data.stats {
            writeln!(
                out,
                "{:<32} {:>5}x {:>12} avg {:>12} max {:>12}",
                stats.name,
                stats.calls,
                // Formatted to a string first so the column width applies.
                time::format_duration(stats.last_nanos),
                time::format_duration(stats.average_nanos as u64),
                time::format_duration(stats.max_nanos),
            )?;
        }
        Ok(())
    }
}

/// Nanoseconds as microseconds with three decimals, without floating point.
fn write_micros(out: &mut dyn fmt::Write, nanos: u64) -> fmt::Result {
    write!(out, "{}.{:03}", nanos / 1_000, nanos % 1_000)
}

/// A JSON string literal.
fn write_json_string(out: &mut dyn fmt::Write, text: &str) -> fmt::Result {
    out.write_char('"')?;
    for c in text.chars() {
        match c {
            '"' => out.write_str("\\\"")?,
            '\\' => out.write_str("\\\\")?,
            '\n' => out.write_str("\\n")?,
            '\r' => out.write_str("\\r")?,
            '\t' => out.write_str("\\t")?,
            c if (c as u32) < 0x20 => write!(out, "\\u{:04x}", c as u32)?,
            c => out.write_char(c)?,
        }
    }
    out.write_char('"')
}

/// The guard of a measured scope; the measurement ends when it is dropped.
#[must_use = "a scope is measured until its guard is dropped"]
pub struct Scope<'a> {
    profiler: Option<&'a Profiler>,
    name: &'static str,
    depth: u16,
    start: u64,
}

impl fmt::Debug for Scope<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scope").field("name", &self.name).field("recording", &self.profiler.is_some()).finish()
    }
}

impl Drop for Scope<'_> {
    #[inline]
    fn drop(&mut self) {
        if let Some(profiler) = self.profiler {
            let end = profiler.now();
            DEPTH.with(|depth| depth.set(self.depth));
            profiler.push(ScopeEvent {
                name: self.name,
                lane: current_lane(),
                depth: self.depth,
                start_nanos: self.start,
                duration_nanos: end.saturating_sub(self.start),
            });
        }
    }
}

static GLOBAL: OnceLock<Profiler> = OnceLock::new();

/// The global profiler (disabled until [`Profiler::set_enabled`]), created on first use.
pub fn global() -> &'static Profiler {
    GLOBAL.get_or_init(Profiler::new)
}

/// Starts a scope on the global profiler; while it does not exist yet this costs one atomic load.
#[inline]
pub fn scope(name: &'static str) -> Scope<'static> {
    match GLOBAL.get() {
        Some(profiler) => profiler.scope(name),
        None => Scope { profiler: None, name, depth: 0, start: 0 },
    }
}

/// Measures the rest of the enclosing block on the global profiler: `profile_scope!("name");`.
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        let _profile_scope = $crate::profile::scope($name);
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::{count_in, is_installed};
    use crate::platform::ManualClock;

    fn profiler() -> (Profiler, Arc<ManualClock>) {
        let clock = Arc::new(ManualClock::new(0));
        let profiler = Profiler::with_clock(clock.clone());
        (profiler, clock)
    }

    fn stats(profiler: &Profiler) -> Vec<ScopeStats> {
        let mut out = Vec::new();
        profiler.for_each_stat(|s| out.push(*s));
        out
    }

    // [PERF-R6] Disabled, a scope records nothing and allocates nothing.
    #[test]
    fn a_disabled_profiler_records_nothing() {
        let (profiler, _clock) = profiler();
        let stats = count_in(|| {
            for _ in 0..1_000 {
                let _scope = profiler.scope("off");
                profiler.record("gpu.off", GPU_LANE, 0, 10);
            }
            profiler.end_frame();
        });
        assert!(is_installed());
        assert_eq!(stats.events(), 0, "{stats:?}");
        assert_eq!(profiler.last_frame(), FrameInfo::default());
        assert!(format!("{profiler:?}").contains("enabled: false"));
        let scope = profiler.scope("x");
        assert!(format!("{scope:?}").contains("recording: false"));
    }

    // [PERF-R6] Nested scopes record depth, duration and lane; the frame closes into statistics.
    #[test]
    fn nested_scopes_and_frame_statistics() {
        let (profiler, clock) = profiler();
        profiler.set_enabled(true);
        clock.advance_nanos(100);
        {
            let _outer = profiler.scope("frame.update");
            clock.advance_nanos(1_000);
            for _ in 0..2 {
                let _inner = profiler.scope("doc.evaluate");
                clock.advance_nanos(300);
            }
            clock.advance_nanos(50);
        }
        profiler.record("gpu.shadow", GPU_LANE, 1_000, 4_000);
        clock.advance_nanos(10_000);
        profiler.end_frame();

        let frame = profiler.last_frame();
        assert_eq!(frame, FrameInfo { index: 0, start_nanos: 0, duration_nanos: 11_750, events: 4, dropped: 0 });
        let mut events = Vec::new();
        profiler.for_each_event(|e| events.push(*e));
        let lane = current_lane();
        assert_eq!(
            events[0],
            ScopeEvent { name: "doc.evaluate", lane, depth: 1, start_nanos: 1_100, duration_nanos: 300 }
        );
        assert_eq!(events[1].start_nanos, 1_400);
        assert_eq!(
            events[2],
            ScopeEvent { name: "frame.update", lane, depth: 0, start_nanos: 100, duration_nanos: 1_650 }
        );
        assert_eq!(events[3].lane, GPU_LANE);

        let all = stats(&profiler);
        assert_eq!(all.len(), 3);
        assert_eq!(
            (all[0].name, all[0].calls, all[0].last_nanos, all[0].average_nanos),
            ("doc.evaluate", 2, 600, 600.0)
        );
        assert_eq!((all[1].name, all[1].last_nanos, all[1].max_nanos), ("frame.update", 1_650, 1_650));

        // A frame without those scopes: their totals drop to 0 and the averages decay.
        {
            let _inner = profiler.scope("doc.evaluate");
            clock.advance_nanos(1_600);
        }
        profiler.end_frame();
        let all = stats(&profiler);
        assert_eq!((all[0].calls, all[0].last_nanos, all[0].max_nanos), (1, 1_600, 1_600));
        assert!((all[0].average_nanos - 700.0).abs() < 1e-9);
        assert_eq!((all[1].calls, all[1].last_nanos, all[1].max_nanos, all[1].frames), (0, 0, 1_650, 2));
        let mut history = Vec::new();
        profiler.for_each_frame_time(|t| history.push(t));
        assert_eq!(history, [11_750, 1_600]);
        assert_eq!(profiler.last_frame().index, 1);

        profiler.reset();
        assert!(stats(&profiler).is_empty());
        profiler.set_enabled(false);
        let _ignored = profiler.scope("off");
    }

    // [PERF-R1] [PERF-R6] Enabled, 600 frames of scopes allocate nothing, and a frame that overflows
    // drops events instead of growing.
    #[test]
    fn steady_state_profiling_does_not_allocate() {
        let (profiler, clock) = profiler();
        clock.set_auto_advance(10);
        profiler.set_enabled(true);
        let names = ["a", "b", "c", "d"];
        let frame = |profiler: &Profiler| {
            for name in names {
                let _outer = profiler.scope(name);
                let _inner = profiler.scope("inner");
            }
            profiler.end_frame();
        };
        frame(&profiler);
        let stats = count_in(|| {
            for _ in 0..600 {
                frame(&profiler);
            }
            for _ in 0..MAX_EVENTS + 10 {
                let _scope = profiler.scope("flood");
            }
            profiler.end_frame();
        });
        assert!(is_installed());
        assert_eq!(stats.events(), 0, "{stats:?}");
        let last = profiler.last_frame();
        assert_eq!((last.events, last.dropped), (MAX_EVENTS, 10));
        assert_eq!(last.index, 601);
    }

    #[test]
    fn names_past_the_limit_get_no_statistics() {
        let (profiler, _clock) = profiler();
        profiler.set_enabled(true);
        let names: Vec<&'static str> =
            (0..MAX_SCOPES + 5).map(|i| &*Box::leak(format!("s{i}").into_boxed_str())).collect();
        for &name in &names {
            let _scope = profiler.scope(name);
        }
        profiler.end_frame();
        assert_eq!(stats(&profiler).len(), MAX_SCOPES);
        assert_eq!(profiler.last_frame().events, MAX_SCOPES + 5);
    }

    // [PERF-R6] The machine-readable dump: Chrome trace events with escaped names and lane names.
    #[test]
    fn trace_json_dump() {
        let (profiler, clock) = profiler();
        profiler.set_enabled(true);
        assert!(profiler.name_thread("main \"ui\""));
        assert!(profiler.name_lane(GPU_LANE, "gpu"));
        assert!(profiler.name_lane(GPU_LANE, "gpu queue"), "renaming a lane");
        clock.advance_nanos(1_234_567);
        {
            let _scope = profiler.scope("io\\png\n");
            clock.advance_nanos(2_500);
        }
        profiler.record("gpu.main", GPU_LANE, 1_000_000, 999);
        profiler.end_frame();
        let mut json = String::new();
        profiler.write_trace_json(&mut json).expect("write");
        let lane = current_lane();
        assert!(json.starts_with("{\"traceEvents\":["), "{json}");
        assert!(json.contains(&format!("\"tid\":{lane},\"args\":{{\"name\":\"main \\\"ui\\\"\"}}")), "{json}");
        assert!(json.contains("\"args\":{\"name\":\"gpu queue\"}"), "{json}");
        assert!(
            json.contains(&format!(
                "{{\"name\":\"io\\\\png\\n\",\"cat\":\"cpu\",\"ph\":\"X\",\"ts\":1234.567,\"dur\":2.500,\"pid\":1,\"tid\":{lane},\"args\":{{\"depth\":0}}}}"
            )),
            "{json}"
        );
        assert!(json.contains("\"cat\":\"gpu\",\"ph\":\"X\",\"ts\":1000.000,\"dur\":0.999"), "{json}");
        assert!(json.ends_with("\"otherData\":{\"frame\":0,\"frame_nanos\":1237067,\"dropped\":0}}\n"), "{json}");
        let opens = json.matches(['{', '[']).count();
        let closes = json.matches(['}', ']']).count();
        assert_eq!(opens, closes, "balanced brackets");

        let mut summary = String::new();
        profiler.write_summary(&mut summary).expect("write");
        assert!(summary.starts_with("frame 0 took 1.24 ms (2 events, 0 dropped)\n"), "{summary}");
        assert!(summary.contains("gpu.main") && summary.contains("999 ns"), "{summary}");

        let mut control = String::new();
        write_json_string(&mut control, "\u{1}\t\r").expect("write");
        assert_eq!(control, "\"\\u0001\\t\\r\"");
    }

    #[test]
    fn lane_table_is_bounded() {
        let (profiler, _clock) = profiler();
        for lane in 0..MAX_LANE_NAMES as u32 {
            assert!(profiler.name_lane(1_000 + lane, "worker"));
        }
        assert!(!profiler.name_lane(5_000, "one too many"));
    }

    #[test]
    fn threads_get_their_own_lanes() {
        let (profiler, _clock) = profiler();
        profiler.set_enabled(true);
        let profiler = Arc::new(profiler);
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let profiler = Arc::clone(&profiler);
                std::thread::spawn(move || {
                    let _scope = profiler.scope("work");
                    current_lane()
                })
            })
            .collect();
        let mut lanes: Vec<u32> = threads.into_iter().map(|t| t.join().expect("thread")).collect();
        profiler.end_frame();
        let mut recorded = Vec::new();
        profiler.for_each_event(|e| recorded.push(e.lane));
        lanes.sort_unstable();
        recorded.sort_unstable();
        assert_eq!(lanes, recorded);
        lanes.dedup();
        assert_eq!(lanes.len(), 4);
        assert_eq!(current_lane(), current_lane());
    }

    #[test]
    fn global_profiler_and_macro() {
        // The global profiler stays disabled in this test binary, so this only shows the macro and
        // the free function compile and do nothing; enabled behaviour is covered on instances above.
        crate::profile_scope!("global.test");
        let scope = scope("global.free");
        drop(scope);
        assert!(!global().is_enabled());
        global().end_frame();
        assert_eq!(global().last_frame(), FrameInfo::default());
    }
}
