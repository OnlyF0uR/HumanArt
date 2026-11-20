//! The logger `[ERR-04]`: levels, target tags, a ring buffer that the UI console reads, and sinks
//! (a file sink and a sink for any `std::io::Write`).
//!
//! A log statement names a level and a target, a short `&'static str` tag for the subsystem
//! (`"gpu"`, `"io.png"`, `"doc"`):
//!
//! ```
//! use humanart_core::{log_info, log_warn};
//! log_info!("doc", "opened {} with {} layers", "face.anthropos", 12);
//! log_warn!("gpu", "falling back to {}", "Dx12");
//! ```
//!
//! **Cost.** The macros first ask [`enabled`], which reads two atomics: a disabled statement does not
//! evaluate its arguments, takes no lock and never allocates. An enabled statement formats into an
//! [`ArrayString`] on the stack (messages longer than [`MESSAGE_CAPACITY`] bytes are cut and marked),
//! copies it into the ring and hands it to the sinks; with no sinks attached that also allocates
//! nothing, so logging is allowed in per-frame code `[PERF-R1]`.
//!
//! **Ring.** Every entry gets a sequence number from one atomic counter; the number picks its slot
//! in a fixed array of slots, each behind its own mutex. Writers never wait for each other unless
//! two of them land on the same slot, which needs a full lap of the ring between them, and readers
//! copy one slot at a time. That is as lock-free as it gets without `unsafe`, which this crate
//! allows only in `alloc` `[UNS-01]`. The UI console polls [`Logger::read_since`] with the cursor it
//! got last time; that is the "UI console sink" of `[ERR-04]`.
//!
//! **Sinks** ([`Sink`]) receive every entry that passes the level filter, on the thread that logged
//! it. [`FileSink`] writes through a [`FileSystem`] (so it also works on wasm with a host-provided
//! file system), buffers up to [`FILE_BUFFER_BYTES`], flushes at once on errors and rotates the file
//! to `<path>.old` when it grows past its limit. A sink that logs while it writes does not recurse:
//! its entry goes to the ring only. A sink that fails is counted ([`Logger::sink_failures`]) and its
//! last error kept for [`Logger::take_last_failure`]; the logger itself never fails.
//!
//! **Filter.** One global level ([`Logger::set_level`], default [`Level::Info`]) plus up to
//! [`MAX_TARGET_OVERRIDES`] per-target levels ([`Logger::set_target_level`]). An override on `io`
//! also covers `io.png` and `io.png.inflate`; the longest matching override wins.

use std::cell::Cell;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, RwLock, RwLockReadGuard};

use crate::collections::{ArrayString, ArrayVec};
use crate::error::{LogError, PlatformError};
use crate::platform::{FileSystem, normalize_path, parent_path};
use crate::time::{self, UtcTimestamp};

/// Bytes of message text an entry holds; longer messages are cut at a character boundary.
pub const MESSAGE_CAPACITY: usize = 240;
/// Entries the global logger's ring keeps.
pub const DEFAULT_CAPACITY: usize = 1024;
/// The largest ring [`Logger::new`] builds (larger requests are clamped).
pub const MAX_CAPACITY: usize = 1 << 16;
/// How many per-target levels a logger holds.
pub const MAX_TARGET_OVERRIDES: usize = 32;
/// The level a logger starts with.
pub const DEFAULT_LEVEL: Level = Level::Info;
/// The size of a [`FileSink`]'s write buffer.
pub const FILE_BUFFER_BYTES: usize = 16 * 1024;
/// The longest line a [`FileSink`] writes, newline included; longer lines are cut.
pub const LINE_CAPACITY: usize = 512;

/// How important an entry is. The order is `Error < Warn < Info < Debug < Trace`: a filter set to a
/// level lets that level and every more important one through.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Level {
    /// Something failed and the user will notice.
    Error = 1,
    /// Something is off but the program carries on (a fallback was taken, input was repaired).
    Warn = 2,
    /// What happened, at the grain of user actions and subsystem start-up.
    #[default]
    Info = 3,
    /// Details for whoever debugs a subsystem.
    Debug = 4,
    /// Everything, including per-frame events.
    Trace = 5,
}

impl Level {
    /// Every level, most important first.
    pub const ALL: [Self; 5] = [Self::Error, Self::Warn, Self::Info, Self::Debug, Self::Trace];

    /// The upper-case name used in log lines: `ERROR`, `WARN`, `INFO`, `DEBUG`, `TRACE`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Error => "ERROR",
            Self::Warn => "WARN",
            Self::Info => "INFO",
            Self::Debug => "DEBUG",
            Self::Trace => "TRACE",
        }
    }

    /// The level with the numeric value `value` (1 = error to 5 = trace).
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Error),
            2 => Some(Self::Warn),
            3 => Some(Self::Info),
            4 => Some(Self::Debug),
            5 => Some(Self::Trace),
            _ => None,
        }
    }

    /// Parses a level name in any case (`warning` is accepted for `warn`), as given on a command
    /// line or in a settings file.
    pub fn parse(text: &str) -> Option<Self> {
        let text = text.trim();
        Self::ALL
            .into_iter()
            .find(|level| text.eq_ignore_ascii_case(level.as_str()))
            .or_else(|| text.eq_ignore_ascii_case("warning").then_some(Self::Warn))
    }

    const fn index(self) -> usize {
        self as usize - 1
    }
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `pad` so that `{:<5}` lines the levels up.
        f.pad(self.as_str())
    }
}

/// A filter value: 0 lets nothing through, otherwise the numeric value of the least important
/// level that passes.
const fn filter_of(level: Option<Level>) -> u8 {
    match level {
        Some(level) => level as u8,
        None => 0,
    }
}

/// The sequence number of a slot that was never written.
const EMPTY_SEQ: u64 = u64::MAX;

/// One log entry, as the ring stores it and the sinks receive it.
#[derive(Clone, Copy, Debug)]
pub struct Entry {
    /// The entry's position in the logger's history, counting from 0.
    pub seq: u64,
    /// The level.
    pub level: Level,
    /// The subsystem tag.
    pub target: &'static str,
    /// Wall-clock time of the entry ([`time::unix_millis`]).
    pub unix_millis: u64,
    /// Monotonic time of the entry ([`time::now_nanos`]).
    pub nanos: u64,
    /// The message; see [`Entry::is_truncated`].
    pub message: ArrayString<MESSAGE_CAPACITY>,
}

impl Entry {
    const EMPTY: Self =
        Self { seq: EMPTY_SEQ, level: Level::Info, target: "", unix_millis: 0, nanos: 0, message: ArrayString::new() };

    /// `true` when the message did not fit in [`MESSAGE_CAPACITY`] bytes and was cut.
    pub const fn is_truncated(&self) -> bool {
        self.message.is_truncated()
    }
}

/// The line form: `2026-11-19T10:23:45.678Z WARN  gpu: message`, with `…` after a cut message.
impl fmt::Display for Entry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {:<5} {}: {}", UtcTimestamp(self.unix_millis), self.level, self.target, self.message)?;
        if self.is_truncated() {
            f.write_str("…")?;
        }
        Ok(())
    }
}

/// What [`Logger::read_since`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReadSummary {
    /// The cursor to pass next time.
    pub next: u64,
    /// Entries handed to the visitor.
    pub read: u64,
    /// Entries between the old cursor and `next` that were overwritten before they could be read.
    pub missed: u64,
}

/// Somewhere entries go besides the ring.
///
/// `write` runs on the thread that logged, while the logger holds its sink list for reading: it
/// should be quick and must not add or remove sinks of the same logger (that would deadlock).
pub trait Sink: Send + Sync {
    /// Takes one entry.
    fn write(&self, entry: &Entry) -> Result<(), LogError>;

    /// Pushes buffered entries to their destination.
    fn flush(&self) -> Result<(), LogError> {
        Ok(())
    }
}

/// A per-target level.
#[derive(Clone, Copy, Debug, Default)]
struct TargetLevel {
    target: &'static str,
    filter: u8,
}

/// `true` when the override for `prefix` covers `target`: equal, or `target` continues with `.`.
fn covers(prefix: &str, target: &str) -> bool {
    target.strip_prefix(prefix).is_some_and(|rest| rest.is_empty() || rest.starts_with('.'))
}

thread_local! {
    /// Set while this thread runs the sinks, so a sink that logs does not recurse into them.
    static DISPATCHING: Cell<bool> = const { Cell::new(false) };
}

/// Clears [`DISPATCHING`] when the dispatch ends, even by a panicking sink.
struct DispatchGuard;

impl Drop for DispatchGuard {
    fn drop(&mut self) {
        DISPATCHING.with(|flag| flag.set(false));
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    // Nothing panics while a logger lock is held except a sink, and the data behind every lock stays
    // consistent in that case, so a poisoned lock is used as it is.
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn read<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(PoisonError::into_inner)
}

/// A logger: filter, ring and sinks. The program has one global logger ([`global`]); separate ones
/// are for tests and tools.
pub struct Logger {
    /// The global filter.
    filter: AtomicU8,
    /// The largest filter of the global level and all overrides: anything above it is rejected
    /// without looking at the overrides.
    widest: AtomicU8,
    has_overrides: AtomicBool,
    overrides: RwLock<ArrayVec<TargetLevel, MAX_TARGET_OVERRIDES>>,
    slots: Box<[Mutex<Entry>]>,
    next_seq: AtomicU64,
    counts: [AtomicU64; 5],
    sinks: RwLock<Vec<Arc<dyn Sink>>>,
    sink_failures: AtomicU64,
    last_failure: Mutex<Option<LogError>>,
}

impl fmt::Debug for Logger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Logger")
            .field("level", &self.level())
            .field("capacity", &self.capacity())
            .field("next_seq", &self.next_seq())
            .field("sinks", &self.sink_count())
            .finish_non_exhaustive()
    }
}

impl Default for Logger {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }
}

impl Logger {
    /// A logger whose ring keeps the last `capacity` entries (clamped to `1..=MAX_CAPACITY`), at
    /// level [`DEFAULT_LEVEL`], with no sinks. The ring is allocated here, once.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.clamp(1, MAX_CAPACITY);
        let filter = filter_of(Some(DEFAULT_LEVEL));
        Self {
            filter: AtomicU8::new(filter),
            widest: AtomicU8::new(filter),
            has_overrides: AtomicBool::new(false),
            overrides: RwLock::new(ArrayVec::new()),
            slots: (0..capacity).map(|_| Mutex::new(Entry::EMPTY)).collect(),
            next_seq: AtomicU64::new(0),
            counts: [const { AtomicU64::new(0) }; 5],
            sinks: RwLock::new(Vec::new()),
            sink_failures: AtomicU64::new(0),
            last_failure: Mutex::new(None),
        }
    }

    /// Entries the ring keeps.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// The global level; `None` when logging is off.
    pub fn level(&self) -> Option<Level> {
        Level::from_u8(self.filter.load(Ordering::Relaxed))
    }

    /// Sets the global level; `None` turns logging off (per-target levels still apply).
    pub fn set_level(&self, level: Option<Level>) {
        self.filter.store(filter_of(level), Ordering::Relaxed);
        self.update_widest(&read(&self.overrides));
    }

    /// Gives `target` (and its `target.sub` children) its own level; `None` silences it. Returns
    /// `false`, changing nothing, when [`MAX_TARGET_OVERRIDES`] other targets already have one.
    pub fn set_target_level(&self, target: &'static str, level: Option<Level>) -> bool {
        let mut overrides = self.overrides.write().unwrap_or_else(PoisonError::into_inner);
        let filter = filter_of(level);
        if let Some(existing) = overrides.iter_mut().find(|o| o.target == target) {
            existing.filter = filter;
        } else if overrides.push(TargetLevel { target, filter }).is_err() {
            return false;
        }
        self.has_overrides.store(true, Ordering::Release);
        self.update_widest(&overrides);
        true
    }

    /// Removes the level of `target`, which follows the global level again.
    pub fn clear_target_level(&self, target: &str) {
        let mut overrides = self.overrides.write().unwrap_or_else(PoisonError::into_inner);
        overrides.retain(|o| o.target != target);
        self.has_overrides.store(!overrides.is_empty(), Ordering::Release);
        self.update_widest(&overrides);
    }

    fn update_widest(&self, overrides: &ArrayVec<TargetLevel, MAX_TARGET_OVERRIDES>) {
        let global = self.filter.load(Ordering::Relaxed);
        let widest = overrides.iter().map(|o| o.filter).fold(global, u8::max);
        self.widest.store(widest, Ordering::Relaxed);
    }

    /// `true` when an entry of `level` for `target` would be recorded.
    #[inline]
    pub fn enabled(&self, level: Level, target: &str) -> bool {
        let level = level as u8;
        if level > self.widest.load(Ordering::Relaxed) {
            return false;
        }
        if !self.has_overrides.load(Ordering::Acquire) {
            return level <= self.filter.load(Ordering::Relaxed);
        }
        self.enabled_with_overrides(level, target)
    }

    #[cold]
    fn enabled_with_overrides(&self, level: u8, target: &str) -> bool {
        let overrides = read(&self.overrides);
        let best = overrides
            .iter()
            .filter(|o| covers(o.target, target))
            .max_by_key(|o| o.target.len())
            .map_or_else(|| self.filter.load(Ordering::Relaxed), |o| o.filter);
        level <= best
    }

    /// Records a message built from format arguments, if `level` is enabled for `target`.
    pub fn log(&self, level: Level, target: &'static str, args: fmt::Arguments<'_>) {
        if !self.enabled(level, target) {
            return;
        }
        let mut message = ArrayString::new();
        // Writing into an `ArrayString` cannot fail; a `Display` impl that reports an error leaves
        // the text written so far, which is what the entry then holds.
        let _ = fmt::write(&mut message, args);
        self.push(level, target, message);
    }

    /// Records a message that is already text (from a callback, for instance), if `level` is
    /// enabled for `target`.
    pub fn record(&self, level: Level, target: &'static str, message: &str) {
        if self.enabled(level, target) {
            self.push(level, target, ArrayString::from_str_truncated(message));
        }
    }

    fn push(&self, level: Level, target: &'static str, message: ArrayString<MESSAGE_CAPACITY>) {
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        let entry = Entry { seq, level, target, unix_millis: time::unix_millis(), nanos: time::now_nanos(), message };
        let slot_index = (seq % self.slots.len() as u64) as usize;
        if let Some(slot) = self.slots.get(slot_index) {
            let mut slot = lock(slot);
            // A writer that was overtaken by a full lap of the ring must not overwrite the newer entry.
            if slot.seq == EMPTY_SEQ || slot.seq < seq {
                *slot = entry;
            }
        }
        if let Some(count) = self.counts.get(level.index()) {
            count.fetch_add(1, Ordering::Relaxed);
        }
        self.dispatch(&entry);
    }

    fn dispatch(&self, entry: &Entry) {
        if DISPATCHING.with(|flag| flag.replace(true)) {
            return;
        }
        let _guard = DispatchGuard;
        let sinks = read(&self.sinks);
        for sink in sinks.iter() {
            if let Err(error) = sink.write(entry) {
                self.note_failure(error);
            }
        }
    }

    fn note_failure(&self, error: LogError) {
        self.sink_failures.fetch_add(1, Ordering::Relaxed);
        *lock(&self.last_failure) = Some(error);
    }

    /// Hands every entry from `cursor` on that is still in the ring to `visit`, oldest first, and
    /// says where to continue. Start with cursor 0. Entries are copied out of the ring before
    /// `visit` sees them, so `visit` may log.
    ///
    /// Reading stops before an entry whose number was taken but whose text is not written yet, so
    /// the next call returns it; entries that were overwritten before they could be read are
    /// counted in [`ReadSummary::missed`].
    pub fn read_since(&self, cursor: u64, mut visit: impl FnMut(&Entry)) -> ReadSummary {
        let end = self.next_seq.load(Ordering::Acquire);
        if cursor >= end {
            return ReadSummary { next: cursor.min(end), read: 0, missed: 0 };
        }
        let oldest = end.saturating_sub(self.slots.len() as u64);
        let mut seq = cursor.max(oldest);
        let mut summary = ReadSummary { next: seq, read: 0, missed: seq - cursor };
        while seq < end {
            let Some(slot) = self.slots.get((seq % self.slots.len() as u64) as usize) else { break };
            let entry = *lock(slot);
            if entry.seq == seq {
                visit(&entry);
                summary.read += 1;
            } else if entry.seq != EMPTY_SEQ && entry.seq > seq {
                summary.missed += 1;
            } else {
                break;
            }
            seq += 1;
        }
        summary.next = seq;
        summary
    }

    /// The number the next entry will get, which is also how many entries were ever recorded.
    pub fn next_seq(&self) -> u64 {
        self.next_seq.load(Ordering::Relaxed)
    }

    /// How many entries of `level` were recorded (for the console's error and warning badges).
    pub fn count(&self, level: Level) -> u64 {
        self.counts.get(level.index()).map_or(0, |count| count.load(Ordering::Relaxed))
    }

    /// Adds a sink. Every later entry that passes the filter reaches it.
    pub fn add_sink(&self, sink: Arc<dyn Sink>) {
        self.sinks.write().unwrap_or_else(PoisonError::into_inner).push(sink);
    }

    /// Flushes and removes every sink. Returns the first flush error.
    pub fn clear_sinks(&self) -> Result<(), LogError> {
        let result = self.flush();
        self.sinks.write().unwrap_or_else(PoisonError::into_inner).clear();
        result
    }

    /// How many sinks are attached.
    pub fn sink_count(&self) -> usize {
        read(&self.sinks).len()
    }

    /// Flushes every sink (the application calls this from time to time and at exit). Every sink
    /// is flushed even if one fails; the first error is returned.
    pub fn flush(&self) -> Result<(), LogError> {
        let sinks = read(&self.sinks);
        let mut first = Ok(());
        for sink in sinks.iter() {
            if let Err(error) = sink.flush()
                && first.is_ok()
            {
                first = Err(error);
            }
        }
        first
    }

    /// How many times a sink reported an error while taking an entry.
    pub fn sink_failures(&self) -> u64 {
        self.sink_failures.load(Ordering::Relaxed)
    }

    /// The most recent error a sink reported while taking an entry, if it was not taken yet.
    pub fn take_last_failure(&self) -> Option<LogError> {
        lock(&self.last_failure).take()
    }
}

static GLOBAL: OnceLock<Logger> = OnceLock::new();

/// Makes `logger` the global logger. Returns `false`, dropping `logger`, when the global logger
/// already exists (it is created on first use with [`DEFAULT_CAPACITY`]); call this first thing in
/// `main` to choose another capacity.
pub fn install(logger: Logger) -> bool {
    let mut logger = Some(logger);
    GLOBAL.get_or_init(|| logger.take().unwrap_or_default());
    logger.is_none()
}

/// The global logger, created on first use.
pub fn global() -> &'static Logger {
    GLOBAL.get_or_init(Logger::default)
}

/// `true` when the global logger records `level` for `target`. Before the global logger exists this
/// answers for [`DEFAULT_LEVEL`] without creating it.
#[inline]
pub fn enabled(level: Level, target: &str) -> bool {
    match GLOBAL.get() {
        Some(logger) => logger.enabled(level, target),
        None => level <= DEFAULT_LEVEL,
    }
}

/// Records an entry in the global logger. The macros ([`log_info!`](crate::log_info) and friends)
/// call this after [`enabled`].
pub fn log(level: Level, target: &'static str, args: fmt::Arguments<'_>) {
    global().log(level, target, args);
}

/// Logs at a level given as a value: `log_at!(level, "target", "format {}", args)`.
#[macro_export]
macro_rules! log_at {
    ($level:expr, $target:expr, $($arg:tt)+) => {{
        let level: $crate::log::Level = $level;
        let target: &'static str = $target;
        if $crate::log::enabled(level, target) {
            $crate::log::log(level, target, ::core::format_args!($($arg)+));
        }
    }};
}

/// Logs an error: `log_error!("target", "format {}", args)`.
#[macro_export]
macro_rules! log_error {
    ($target:expr, $($arg:tt)+) => { $crate::log_at!($crate::log::Level::Error, $target, $($arg)+) };
}

/// Logs a warning: `log_warn!("target", "format {}", args)`.
#[macro_export]
macro_rules! log_warn {
    ($target:expr, $($arg:tt)+) => { $crate::log_at!($crate::log::Level::Warn, $target, $($arg)+) };
}

/// Logs information: `log_info!("target", "format {}", args)`.
#[macro_export]
macro_rules! log_info {
    ($target:expr, $($arg:tt)+) => { $crate::log_at!($crate::log::Level::Info, $target, $($arg)+) };
}

/// Logs a debugging detail: `log_debug!("target", "format {}", args)`.
#[macro_export]
macro_rules! log_debug {
    ($target:expr, $($arg:tt)+) => { $crate::log_at!($crate::log::Level::Debug, $target, $($arg)+) };
}

/// Logs a trace event: `log_trace!("target", "format {}", args)`.
#[macro_export]
macro_rules! log_trace {
    ($target:expr, $($arg:tt)+) => { $crate::log_at!($crate::log::Level::Trace, $target, $($arg)+) };
}

/// A sink that writes each entry as one line to a `std::io::Write` (standard error on native, a
/// `Vec<u8>` in tests). It flushes after error entries.
pub struct WriterSink<W: std::io::Write + Send> {
    writer: Mutex<W>,
}

impl<W: std::io::Write + Send> WriterSink<W> {
    /// A sink around `writer`.
    pub fn new(writer: W) -> Self {
        Self { writer: Mutex::new(writer) }
    }

    /// Takes the writer back.
    pub fn into_inner(self) -> W {
        self.writer.into_inner().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<W: std::io::Write + Send> fmt::Debug for WriterSink<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WriterSink").finish_non_exhaustive()
    }
}

impl<W: std::io::Write + Send> Sink for WriterSink<W> {
    fn write(&self, entry: &Entry) -> Result<(), LogError> {
        let mut writer = lock(&self.writer);
        writeln!(writer, "{entry}")?;
        if entry.level == Level::Error {
            writer.flush()?;
        }
        Ok(())
    }

    fn flush(&self) -> Result<(), LogError> {
        lock(&self.writer).flush().map_err(LogError::Io)
    }
}

/// The buffered part of a [`FileSink`].
struct FileState {
    buffer: Vec<u8>,
    file_len: u64,
}

/// A sink that appends lines to a log file through a [`FileSystem`].
///
/// Lines collect in a buffer of [`FILE_BUFFER_BYTES`] that is written when it is full, on every
/// error entry, on [`Sink::flush`] and when the sink is dropped. When the file would grow past its
/// limit it is first renamed to `<path>.old` (replacing an older one), so a log never takes more
/// than about twice the limit on disk; the same happens on opening a file that is already too big.
pub struct FileSink {
    fs: Arc<dyn FileSystem>,
    path: String,
    old_path: String,
    max_bytes: u64,
    state: Mutex<FileState>,
}

impl fmt::Debug for FileSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileSink").field("path", &self.path).field("max_bytes", &self.max_bytes).finish_non_exhaustive()
    }
}

impl FileSink {
    /// Opens (or creates) the log file at `path`, creating its directory, rotating it first when it
    /// is already larger than `max_bytes`.
    pub fn open(fs: Arc<dyn FileSystem>, path: &str, max_bytes: u64) -> Result<Self, LogError> {
        let path = normalize_path(path)?;
        if path.is_empty() {
            return Err(LogError::Platform(PlatformError::InvalidPath));
        }
        let parent = parent_path(&path);
        if !parent.is_empty() {
            fs.create_dir_all(parent)?;
        }
        let old_path = format!("{path}.old");
        let mut file_len = match fs.metadata(&path) {
            Ok(metadata) if metadata.is_file() => metadata.len,
            Ok(_) => return Err(LogError::Platform(PlatformError::IsADirectory)),
            Err(PlatformError::NotFound) => 0,
            Err(error) => return Err(error.into()),
        };
        if file_len > max_bytes {
            fs.rename(&path, &old_path)?;
            file_len = 0;
        }
        let state = FileState { buffer: Vec::with_capacity(FILE_BUFFER_BYTES), file_len };
        Ok(Self { fs, path, old_path, max_bytes, state: Mutex::new(state) })
    }

    /// The normalised path of the log file.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The path the file is renamed to when it is rotated.
    pub fn old_path(&self) -> &str {
        &self.old_path
    }

    /// Writes the buffer out, rotating first if the file would pass its limit. The buffer is
    /// emptied even when the write fails, so a broken disk cannot make it grow.
    fn write_out(&self, state: &mut FileState) -> Result<(), LogError> {
        if state.buffer.is_empty() {
            return Ok(());
        }
        let result = self.append_buffer(state);
        state.buffer.clear();
        result.map_err(LogError::Platform)
    }

    fn append_buffer(&self, state: &mut FileState) -> Result<(), PlatformError> {
        let pending = state.buffer.len() as u64;
        if state.file_len > 0 && state.file_len.saturating_add(pending) > self.max_bytes {
            match self.fs.rename(&self.path, &self.old_path) {
                Ok(()) | Err(PlatformError::NotFound) => state.file_len = 0,
                Err(error) => return Err(error),
            }
        }
        self.fs.append(&self.path, &state.buffer)?;
        state.file_len = state.file_len.saturating_add(pending);
        Ok(())
    }
}

impl Sink for FileSink {
    fn write(&self, entry: &Entry) -> Result<(), LogError> {
        use fmt::Write as _;
        let mut line = ArrayString::<LINE_CAPACITY>::new();
        let _ = write!(line, "{entry}");
        if line.is_truncated() {
            // Keep room for the newline.
            line.truncate(LINE_CAPACITY - 1);
        }
        line.push('\n');
        let mut state = lock(&self.state);
        let mut result = Ok(());
        let pending = (state.buffer.len() + line.len()) as u64;
        // Write out before the buffer overflows, and before the file would pass its limit, so the
        // rotation happens between lines and neither file grows past the limit.
        if !state.buffer.is_empty()
            && (pending > FILE_BUFFER_BYTES as u64 || state.file_len.saturating_add(pending) > self.max_bytes)
        {
            result = self.write_out(&mut state);
        }
        // Never grows the buffer: it has room for `FILE_BUFFER_BYTES` and a line is far shorter.
        state.buffer.extend_from_slice(line.as_bytes());
        if entry.level == Level::Error {
            let flushed = self.write_out(&mut state);
            result = result.and(flushed);
        }
        result
    }

    fn flush(&self) -> Result<(), LogError> {
        let mut state = lock(&self.state);
        self.write_out(&mut state)
    }
}

impl Drop for FileSink {
    fn drop(&mut self) {
        // Nobody is left to report a failure to.
        let _ = Sink::flush(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::{count_in, is_installed};
    use crate::platform::MemoryFs;

    /// A sink that keeps the lines it gets.
    #[derive(Default)]
    struct Collect(Mutex<Vec<String>>);

    impl Sink for Collect {
        fn write(&self, entry: &Entry) -> Result<(), LogError> {
            lock(&self.0).push(format!("{} {}: {}", entry.level, entry.target, entry.message));
            Ok(())
        }
    }

    fn messages(logger: &Logger) -> Vec<String> {
        let mut out = Vec::new();
        logger.read_since(0, |entry| out.push(entry.message.to_string()));
        out
    }

    #[test]
    fn levels_order_parse_and_print() {
        assert!(Level::Error < Level::Warn && Level::Debug < Level::Trace);
        for level in Level::ALL {
            assert_eq!(Level::from_u8(level as u8), Some(level));
            assert_eq!(Level::parse(&level.as_str().to_ascii_lowercase()), Some(level));
        }
        assert_eq!(Level::parse(" Warning "), Some(Level::Warn));
        assert_eq!(Level::parse("verbose"), None);
        assert_eq!((Level::from_u8(0), Level::from_u8(6)), (None, None));
        assert_eq!(format!("[{:<5}]", Level::Warn), "[WARN ]");
        assert_eq!(Level::default(), Level::Info);
    }

    // [ERR-04] Levels filter globally and per target; an override covers its dotted children.
    #[test]
    fn filter_by_level_and_target() {
        let logger = Logger::new(16);
        assert_eq!(logger.level(), Some(Level::Info));
        assert!(logger.enabled(Level::Info, "doc") && !logger.enabled(Level::Debug, "doc"));
        logger.set_level(Some(Level::Warn));
        assert!(!logger.enabled(Level::Info, "doc"));
        assert!(logger.set_target_level("gpu", Some(Level::Debug)));
        assert!(logger.enabled(Level::Debug, "gpu") && logger.enabled(Level::Debug, "gpu.pass"));
        assert!(!logger.enabled(Level::Trace, "gpu") && !logger.enabled(Level::Debug, "gpux"));
        assert!(!logger.enabled(Level::Info, "io"));
        assert!(logger.set_target_level("gpu.pass", Some(Level::Trace)));
        assert!(logger.enabled(Level::Trace, "gpu.pass.shadow") && !logger.enabled(Level::Trace, "gpu"));
        assert!(logger.set_target_level("io", None));
        assert!(!logger.enabled(Level::Error, "io") && logger.enabled(Level::Error, "doc"));
        logger.clear_target_level("io");
        assert!(logger.enabled(Level::Error, "io"));
        logger.set_level(None);
        assert!(!logger.enabled(Level::Error, "doc") && logger.enabled(Level::Debug, "gpu"));
        logger.clear_target_level("gpu");
        logger.clear_target_level("gpu.pass");
        assert!(!logger.enabled(Level::Error, "gpu"));

        let names: Vec<&'static str> =
            (0..MAX_TARGET_OVERRIDES).map(|i| &*Box::leak(format!("t{i}").into_boxed_str())).collect();
        for name in &names {
            assert!(logger.set_target_level(name, Some(Level::Info)));
        }
        assert!(!logger.set_target_level("one-too-many", Some(Level::Info)), "the table is full");
        assert!(logger.set_target_level(names[0], Some(Level::Trace)), "an existing target can change");
        assert!(logger.enabled(Level::Trace, names[0]));
    }

    #[test]
    fn entries_reach_ring_and_sinks_in_order() {
        let logger = Logger::new(8);
        let sink = Arc::new(Collect::default());
        logger.add_sink(sink.clone());
        assert_eq!(logger.sink_count(), 1);
        logger.log(Level::Warn, "gpu", format_args!("adapter {} lost", 2));
        logger.record(Level::Error, "io", "disk full");
        logger.log(Level::Debug, "gpu", format_args!("filtered"));
        assert_eq!(messages(&logger), ["adapter 2 lost", "disk full"]);
        assert_eq!(*lock(&sink.0), ["WARN gpu: adapter 2 lost", "ERROR io: disk full"]);
        assert_eq!((logger.count(Level::Warn), logger.count(Level::Error), logger.count(Level::Debug)), (1, 1, 0));
        assert_eq!(logger.next_seq(), 2);
        assert!(logger.clear_sinks().is_ok());
        logger.record(Level::Info, "doc", "after");
        assert_eq!(lock(&sink.0).len(), 2);
        assert!(format!("{logger:?}").contains("capacity: 8"));
    }

    #[test]
    fn entry_line_form_and_truncation() {
        let logger = Logger::new(4);
        let long = "é".repeat(200);
        logger.record(Level::Warn, "io.png", &long);
        let mut entries = Vec::new();
        logger.read_since(0, |entry| entries.push(*entry));
        let entry = entries[0];
        assert!(entry.is_truncated());
        assert_eq!(entry.message.len(), MESSAGE_CAPACITY);
        let line = entry.to_string();
        assert!(line.ends_with("é…"), "{line}");
        let fixed = Entry { unix_millis: 1_795_083_825_678, message: ArrayString::from_str_truncated("hi"), ..entry };
        assert_eq!(fixed.to_string(), "2026-11-19T10:23:45.678Z WARN  io.png: hi");
    }

    // [ERR-04] The ring keeps the newest entries; a reader learns how many it missed.
    #[test]
    fn ring_overwrites_the_oldest_and_counts_what_was_missed() {
        let logger = Logger::new(4);
        assert_eq!(logger.read_since(0, |_| {}), ReadSummary { next: 0, read: 0, missed: 0 });
        for i in 0..3 {
            logger.log(Level::Info, "t", format_args!("{i}"));
        }
        let first = logger.read_since(0, |_| {});
        assert_eq!(first, ReadSummary { next: 3, read: 3, missed: 0 });
        for i in 3..10 {
            logger.log(Level::Info, "t", format_args!("{i}"));
        }
        let mut seen = Vec::new();
        let second = logger.read_since(first.next, |entry| seen.push(entry.message.to_string()));
        assert_eq!(second, ReadSummary { next: 10, read: 4, missed: 3 });
        assert_eq!(seen, ["6", "7", "8", "9"]);
        assert_eq!(logger.read_since(10, |_| {}).read, 0);
        assert_eq!(logger.read_since(99, |_| {}).next, 10, "a cursor past the end is clamped");
        assert_eq!(Logger::new(0).capacity(), 1);
        assert_eq!(Logger::new(usize::MAX).capacity(), MAX_CAPACITY);
    }

    // [ERR-04] [PERF-R1] A disabled statement allocates nothing and does not evaluate its arguments;
    // an enabled one without sinks allocates nothing either.
    #[test]
    fn logging_does_not_allocate() {
        let logger = Logger::new(64);
        logger.set_target_level("hot", Some(Level::Info));
        let mut evaluated = 0;
        let mut expensive = || {
            evaluated += 1;
            evaluated
        };
        let stats = count_in(|| {
            for i in 0..1_000 {
                if logger.enabled(Level::Trace, "hot") {
                    logger.log(Level::Trace, "hot", format_args!("{} {}", i, expensive()));
                }
                logger.log(Level::Info, "hot", format_args!("frame {i} took {:.2} ms", 16.6));
                logger.record(Level::Warn, "other", "a warning");
            }
            let mut total = 0;
            logger.read_since(0, |entry| total += entry.message.len());
            std::hint::black_box(total);
        });
        assert!(is_installed());
        assert_eq!(stats.events(), 0, "{stats:?}");
        assert_eq!(evaluated, 0);
        assert_eq!(logger.next_seq(), 2_000);
    }

    #[test]
    fn global_logger_and_macros() {
        let logger = global();
        assert!(!install(Logger::new(8)), "the global logger already exists");
        assert!(logger.set_target_level("core.log.test", Some(Level::Trace)));
        let before = logger.next_seq();
        crate::log_error!("core.log.test", "e {}", 1);
        crate::log_warn!("core.log.test", "w");
        crate::log_info!("core.log.test", "i");
        crate::log_debug!("core.log.test", "d");
        crate::log_trace!("core.log.test", "t {x}", x = 5);
        crate::log_at!(Level::Info, "core.log.test", "at");
        let mut seen = Vec::new();
        logger.read_since(before, |entry| {
            if entry.target == "core.log.test" {
                seen.push(format!("{} {}", entry.level, entry.message));
            }
        });
        assert_eq!(seen, ["ERROR e 1", "WARN w", "INFO i", "DEBUG d", "TRACE t 5", "INFO at"]);
        assert!(logger.set_target_level("core.log.test", None));
        let mut evaluated = false;
        crate::log_error!("core.log.test", "{}", {
            evaluated = true;
            1
        });
        assert!(!evaluated, "a disabled statement does not evaluate its arguments");
        assert!(enabled(Level::Error, "core.log.other"));
    }

    // [ERR-04] A sink that logs while writing does not recurse or deadlock.
    #[test]
    fn a_sink_that_logs_does_not_recurse() {
        struct Echo(Arc<Logger>);
        impl Sink for Echo {
            fn write(&self, entry: &Entry) -> Result<(), LogError> {
                self.0.record(Level::Info, "echo", entry.message.as_str());
                Ok(())
            }
        }
        let logger = Arc::new(Logger::new(8));
        logger.add_sink(Arc::new(Echo(logger.clone())));
        logger.record(Level::Info, "t", "hello");
        assert_eq!(messages(&logger), ["hello", "hello"]);
        logger.record(Level::Info, "t", "again");
        assert_eq!(logger.next_seq(), 4, "the dispatch flag was cleared after the first entry");
        // Break the reference cycle.
        assert!(logger.clear_sinks().is_ok());
    }

    #[test]
    fn sink_failures_are_counted_not_raised() {
        struct Broken;
        impl Sink for Broken {
            fn write(&self, _entry: &Entry) -> Result<(), LogError> {
                Err(LogError::Sink("broken".to_owned()))
            }
            fn flush(&self) -> Result<(), LogError> {
                Err(LogError::Sink("cannot flush".to_owned()))
            }
        }
        let logger = Logger::new(4);
        let good = Arc::new(Collect::default());
        logger.add_sink(Arc::new(Broken));
        logger.add_sink(good.clone());
        logger.record(Level::Info, "t", "one");
        logger.record(Level::Info, "t", "two");
        assert_eq!(logger.sink_failures(), 2);
        assert_eq!(lock(&good.0).len(), 2, "a broken sink does not stop the others");
        assert!(matches!(logger.take_last_failure(), Some(LogError::Sink(m)) if m == "broken"));
        assert!(logger.take_last_failure().is_none());
        assert!(matches!(logger.flush(), Err(LogError::Sink(m)) if m == "cannot flush"));
    }

    #[test]
    fn writer_sink_writes_lines() {
        let logger = Logger::new(4);
        let sink = Arc::new(WriterSink::new(Vec::new()));
        logger.add_sink(sink.clone());
        logger.record(Level::Info, "doc", "saved");
        logger.record(Level::Error, "io", "failed");
        assert!(logger.flush().is_ok());
        assert!(logger.clear_sinks().is_ok());
        let sink = Arc::into_inner(sink).expect("only reference");
        let text = String::from_utf8(sink.into_inner()).expect("utf-8");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].ends_with(" INFO  doc: saved") && lines[1].ends_with(" ERROR io: failed"));
        assert!(format!("{:?}", WriterSink::new(Vec::<u8>::new())).contains("WriterSink"));
    }

    // [ERR-04] The file sink buffers, flushes on errors and rotates past its limit.
    #[test]
    fn file_sink_buffers_flushes_and_rotates() {
        let fs = Arc::new(MemoryFs::new());
        let logger = Logger::new(64);
        let sink = Arc::new(FileSink::open(fs.clone(), "logs//humanart.log", 4_000).expect("open"));
        assert_eq!((sink.path(), sink.old_path()), ("logs/humanart.log", "logs/humanart.log.old"));
        assert!(format!("{sink:?}").contains("humanart.log"));
        logger.add_sink(sink.clone());
        logger.record(Level::Info, "doc", "buffered");
        assert!(!fs.exists("logs/humanart.log"), "info entries wait in the buffer");
        logger.record(Level::Error, "io", "written at once");
        let text = String::from_utf8(fs.read("logs/humanart.log").expect("log")).expect("utf-8");
        assert!(text.contains(" INFO  doc: buffered\n") && text.ends_with(" ERROR io: written at once\n"));

        for i in 0..100 {
            logger.log(Level::Info, "fill", format_args!("line {i:03} {}", "x".repeat(40)));
        }
        assert!(logger.flush().is_ok());
        let current = fs.read("logs/humanart.log").expect("log");
        let old = fs.read("logs/humanart.log.old").expect("rotated log");
        assert!(current.len() <= 4_000 && old.len() <= 4_000, "{} {}", current.len(), old.len());
        let current = String::from_utf8(current).expect("utf-8");
        assert!(current.ends_with(&format!("line 099 {}\n", "x".repeat(40))));
        assert_eq!(logger.sink_failures(), 0);

        // Reopening a file that is already too big rotates it at once.
        fs.write("big.log", &[b'a'; 100]).expect("write");
        let small = FileSink::open(fs.clone(), "big.log", 50).expect("open");
        assert!(!fs.exists("big.log") && fs.exists("big.log.old"));
        drop(small);
    }

    #[test]
    fn file_sink_drops_flush_and_survive_a_failing_disk() {
        let fs = Arc::new(MemoryFs::new());
        {
            let sink = FileSink::open(fs.clone(), "a.log", 1 << 20).expect("open");
            let entry = Entry { seq: 0, message: ArrayString::from_str_truncated("kept"), ..Entry::EMPTY };
            sink.write(&entry).expect("buffered");
        }
        assert!(String::from_utf8(fs.read("a.log").expect("log")).expect("utf-8").ends_with("kept\n"));

        let logger = Logger::new(4);
        logger.add_sink(Arc::new(FileSink::open(fs.clone(), "b.log", 1 << 20).expect("open")));
        fs.fail_after(Some(0));
        logger.record(Level::Error, "io", "lost");
        assert_eq!(logger.sink_failures(), 1);
        assert!(matches!(logger.take_last_failure(), Some(LogError::Platform(_))));
        fs.fail_after(None);
        logger.record(Level::Error, "io", "after recovery");
        let text = String::from_utf8(fs.read("b.log").expect("log")).expect("utf-8");
        assert!(!text.contains("lost") && text.contains("after recovery"), "the failed buffer is dropped");

        assert!(matches!(FileSink::open(fs.clone(), "", 10), Err(LogError::Platform(PlatformError::InvalidPath))));
        assert!(matches!(
            FileSink::open(fs.clone(), "../x.log", 10),
            Err(LogError::Platform(PlatformError::InvalidPath))
        ));
        fs.create_dir_all("dir.log").expect("dir");
        assert!(matches!(FileSink::open(fs, "dir.log", 10), Err(LogError::Platform(PlatformError::IsADirectory))));
    }

    // [ERR-04] Many writers and a concurrent reader: every entry gets a unique number, the reader
    // sees each thread's entries in order, and read + missed accounts for every entry.
    #[test]
    fn concurrent_writers_and_reader() {
        const THREADS: u64 = 8;
        const PER_THREAD: u64 = 5_000;
        let logger = Arc::new(Logger::new(256));
        let done = Arc::new(AtomicBool::new(false));
        let reader = {
            let (logger, done) = (logger.clone(), done.clone());
            std::thread::spawn(move || {
                let mut cursor = 0;
                let mut last = [None::<u64>; THREADS as usize];
                let (mut read, mut missed) = (0, 0);
                loop {
                    let finished = done.load(Ordering::Acquire);
                    let summary = logger.read_since(cursor, |entry| {
                        let (thread, index) = entry.message.split_once(' ').expect("two numbers");
                        let thread: usize = thread.parse().expect("thread");
                        let index: u64 = index.parse().expect("index");
                        assert!(last[thread].is_none_or(|previous| previous < index), "order within a thread");
                        last[thread] = Some(index);
                    });
                    assert!(summary.next >= cursor);
                    cursor = summary.next;
                    read += summary.read;
                    missed += summary.missed;
                    if finished && cursor == logger.next_seq() {
                        return (read, missed);
                    }
                }
            })
        };
        let writers: Vec<_> = (0..THREADS)
            .map(|thread| {
                let logger = logger.clone();
                std::thread::spawn(move || {
                    for i in 0..PER_THREAD {
                        logger.log(Level::Info, "stress", format_args!("{thread} {i}"));
                    }
                })
            })
            .collect();
        for writer in writers {
            writer.join().expect("writer");
        }
        done.store(true, Ordering::Release);
        let (read, missed) = reader.join().expect("reader");
        assert_eq!(read + missed, THREADS * PER_THREAD);
        assert!(read > 0);
        assert_eq!(logger.count(Level::Info), THREADS * PER_THREAD);
        let tail = logger.read_since(0, |_| {});
        assert_eq!((tail.read, tail.missed), (256, THREADS * PER_THREAD - 256));
    }
}
