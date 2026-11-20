//! The services the library crates may not reach directly `[PLAT-03]`: files, clocks, threads and
//! randomness, as traits.
//!
//! The crates that compile for `wasm32-unknown-unknown` never name `std::fs`, `std::thread::spawn`,
//! `Instant` or `SystemTime`; they take a [`FileSystem`], [`Clock`], [`Spawner`] or [`Entropy`] from
//! whoever hosts them. The native implementations live in `humanart-platform`. This module holds the
//! traits and the implementations that need no operating system: [`MemoryFs`], [`ManualClock`],
//! [`InlineSpawner`] and [`SeededEntropy`], which tests and the single-threaded wasm build use.
//!
//! Paths are UTF-8 strings. `/` separates components everywhere (the native file system also accepts
//! the platform's own separator and absolute paths); [`normalize_path`], [`join_path`],
//! [`parent_path`], [`file_name`] and [`extension`] do the string work without `std::path`, whose
//! rules differ per target.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::error::PlatformError;
use crate::rng::Rng;

// ---------------------------------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------------------------------

/// Brings a path to its canonical relative form: components joined by single `/`, no leading or
/// trailing separator, no `.` components, `..` resolved.
///
/// Both `/` and `\` are accepted as separators. The empty string is the root. A path that climbs
/// above the root (`a/../..`) or contains a NUL character is [`PlatformError::InvalidPath`], which is
/// what keeps a path taken from a document inside the directory it is resolved against.
pub fn normalize_path(path: &str) -> Result<String, PlatformError> {
    let mut out = String::with_capacity(path.len());
    for part in path.split(['/', '\\']) {
        match part {
            "" | "." => {}
            ".." => {
                if out.is_empty() {
                    return Err(PlatformError::InvalidPath);
                }
                let cut = out.rfind('/').unwrap_or(0);
                out.truncate(cut);
            }
            _ => {
                if part.contains('\0') {
                    return Err(PlatformError::InvalidPath);
                }
                if !out.is_empty() {
                    out.push('/');
                }
                out.push_str(part);
            }
        }
    }
    Ok(out)
}

/// `base` and `relative` joined by exactly one `/` (either may be empty).
pub fn join_path(base: &str, relative: &str) -> String {
    let base = base.trim_end_matches(['/', '\\']);
    let relative = relative.trim_start_matches(['/', '\\']);
    if base.is_empty() {
        relative.to_owned()
    } else if relative.is_empty() {
        base.to_owned()
    } else {
        format!("{base}/{relative}")
    }
}

/// The path without its last component (`""` for a single component).
pub fn parent_path(path: &str) -> &str {
    let trimmed = path.trim_end_matches(['/', '\\']);
    trimmed.rfind(['/', '\\']).map_or("", |cut| &trimmed[..cut])
}

/// The last component of the path.
pub fn file_name(path: &str) -> &str {
    let trimmed = path.trim_end_matches(['/', '\\']);
    trimmed.rfind(['/', '\\']).map_or(trimmed, |cut| &trimmed[cut + 1..])
}

/// The text after the last `.` of the file name, without the dot; `None` when there is no dot or the
/// name only starts with one (`.gitignore`).
pub fn extension(path: &str) -> Option<&str> {
    let name = file_name(path);
    match name.rfind('.') {
        Some(dot) if dot > 0 => Some(&name[dot + 1..]),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------------------------------

/// What a directory entry is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FileKind {
    /// A regular file.
    File,
    /// A directory.
    Directory,
}

/// What [`FileSystem::metadata`] reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Metadata {
    /// File or directory.
    pub kind: FileKind,
    /// Size in bytes (0 for a directory).
    pub len: u64,
    /// Last modification as Unix time in milliseconds, when the file system records it.
    pub modified_unix_millis: Option<u64>,
}

impl Metadata {
    /// `true` for a regular file.
    pub fn is_file(&self) -> bool {
        self.kind == FileKind::File
    }

    /// `true` for a directory.
    pub fn is_dir(&self) -> bool {
        self.kind == FileKind::Directory
    }
}

/// One entry of [`FileSystem::list`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirEntry {
    /// The entry's name inside its directory (no separators).
    pub name: String,
    /// File or directory.
    pub kind: FileKind,
    /// Size in bytes (0 for a directory).
    pub len: u64,
}

/// A file system. Every method takes a path as described in the module documentation.
pub trait FileSystem: Send + Sync {
    /// The whole content of a file.
    fn read(&self, path: &str) -> Result<Vec<u8>, PlatformError>;

    /// Creates the file or replaces its content. The parent directory must exist.
    ///
    /// A crash in the middle can leave a partial file: use [`FileSystem::write_atomic`] for anything
    /// that must not be lost.
    fn write(&self, path: &str, bytes: &[u8]) -> Result<(), PlatformError>;

    /// Adds `bytes` to the end of the file, creating it if needed (journals, log files).
    fn append(&self, path: &str, bytes: &[u8]) -> Result<(), PlatformError>;

    /// Moves a file or directory. An existing file at `to` is replaced in one step; a directory can
    /// only move to a name that is not taken.
    fn rename(&self, from: &str, to: &str) -> Result<(), PlatformError>;

    /// The entries of a directory, sorted by name.
    fn list(&self, dir: &str) -> Result<Vec<DirEntry>, PlatformError>;

    /// Kind, size and modification time of a file or directory.
    fn metadata(&self, path: &str) -> Result<Metadata, PlatformError>;

    /// Deletes a file or an empty directory.
    fn remove(&self, path: &str) -> Result<(), PlatformError>;

    /// Creates the directory and every missing parent. Succeeds if it already exists.
    fn create_dir_all(&self, path: &str) -> Result<(), PlatformError>;

    /// Replaces the file's content so that a reader, or a crash at any moment, sees either all of
    /// the old content or all of the new `[SAVE-02]`.
    ///
    /// The default writes a temporary file next to the target ([`temp_path`]) and renames it over
    /// the target; if either step fails the temporary file is removed and the target is untouched.
    /// A native implementation overrides this to flush the file to the device before the rename.
    /// One writer per path at a time.
    fn write_atomic(&self, path: &str, bytes: &[u8]) -> Result<(), PlatformError> {
        let temp = temp_path(path);
        let result = self.write(&temp, bytes).and_then(|()| self.rename(&temp, path));
        if result.is_err() {
            // The original error is the one to report; a failed clean-up changes nothing about it.
            let _ = self.remove(&temp);
        }
        result
    }

    /// `true` when a file or directory exists at `path`.
    fn exists(&self, path: &str) -> bool {
        self.metadata(path).is_ok()
    }
}

/// The name of the temporary file [`FileSystem::write_atomic`] writes before renaming it to `path`.
/// Anything left under such a name after a crash is garbage and may be deleted.
pub fn temp_path(path: &str) -> String {
    format!("{path}.tmp~")
}

/// A source of time.
pub trait Clock: Send + Sync {
    /// Nanoseconds since an arbitrary fixed moment; never goes backwards. For measuring intervals.
    fn now_nanos(&self) -> u64;

    /// Wall-clock time: milliseconds since 1970-01-01T00:00:00Z. For timestamps shown to people and
    /// stored in files; it can jump when the system clock is adjusted.
    fn unix_millis(&self) -> u64;
}

/// The work a [`Spawner`] runs on a new thread.
pub type ThreadWork = Box<dyn FnOnce() + Send + 'static>;

/// Creates threads.
pub trait Spawner: Send + Sync {
    /// How many threads can usefully run at once (the number of hardware threads). 0 means this
    /// platform has no threads at all: callers then do their work inline `[THR-04]`.
    fn available_threads(&self) -> usize;

    /// Starts a thread called `name` (shown in debuggers and panic messages) that runs `work` and
    /// then ends. The thread is detached: whoever needs to wait for it arranges that inside `work`.
    fn spawn(&self, name: &str, work: ThreadWork) -> Result<(), PlatformError>;
}

/// A source of unpredictable bytes (document ids, the agent server's bearer token).
pub trait Entropy: Send + Sync {
    /// Fills `out` with random bytes.
    fn fill(&self, out: &mut [u8]) -> Result<(), PlatformError>;
}

// ---------------------------------------------------------------------------------------------------
// MemoryFs
// ---------------------------------------------------------------------------------------------------

enum Node {
    File { bytes: Vec<u8>, modified: u64 },
    Directory,
}

struct MemoryState {
    /// Every file and directory by normalised path. The root (`""`) is implicit.
    nodes: BTreeMap<String, Node>,
    read_only: bool,
    /// Mutations that may still succeed before injected failures start.
    failure_budget: Option<usize>,
    /// The modification time given to files written from now on.
    now_millis: u64,
}

impl MemoryState {
    fn is_dir(&self, path: &str) -> bool {
        path.is_empty() || matches!(self.nodes.get(path), Some(Node::Directory))
    }

    /// Checks that `path` can hold a file: it is not the root or a directory, and its parent is a
    /// directory.
    fn check_file_slot(&self, path: &str) -> Result<(), PlatformError> {
        if self.is_dir(path) {
            return Err(PlatformError::IsADirectory);
        }
        let parent = parent_path(path);
        match self.nodes.get(parent) {
            _ if parent.is_empty() => Ok(()),
            Some(Node::Directory) => Ok(()),
            Some(Node::File { .. }) => Err(PlatformError::NotADirectory),
            None => Err(PlatformError::NotFound),
        }
    }

    /// The paths strictly below directory `dir`, in order.
    fn below<'a>(&'a self, dir: &str) -> impl Iterator<Item = (&'a String, &'a Node)> + use<'a> {
        let prefix = if dir.is_empty() { String::new() } else { format!("{dir}/") };
        let start = prefix.clone();
        self.nodes.range(start..).take_while(move |(path, _)| path.starts_with(&prefix))
    }
}

/// A file system held in memory, for tests and for hosts without one.
///
/// It behaves like the native one in everything the [`FileSystem`] contract states: the same errors
/// for missing parents, directories in the place of files and non-empty directories. Names are
/// case-sensitive. Tests can make it fail on purpose ([`MemoryFs::set_read_only`],
/// [`MemoryFs::fail_after`]) to prove that a save interrupted at any step loses nothing.
pub struct MemoryFs {
    state: Mutex<MemoryState>,
}

impl MemoryFs {
    /// An empty file system: only the root directory exists.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Mutex::new(MemoryState {
                nodes: BTreeMap::new(),
                read_only: false,
                failure_budget: None,
                now_millis: 0,
            }),
        }
    }

    /// While `true`, every operation that would change something fails with
    /// [`PlatformError::PermissionDenied`].
    pub fn set_read_only(&self, read_only: bool) {
        self.lock().read_only = read_only;
    }

    /// Lets `operations` more changes (write, append, rename, remove, create directory) succeed and
    /// makes every later one fail with `PlatformError::Other("injected failure")`, like a disk that
    /// fills up or a process that dies at that point. `None` turns the injection off.
    pub fn fail_after(&self, operations: Option<usize>) {
        self.lock().failure_budget = operations;
    }

    /// Sets the modification time (Unix milliseconds) recorded for files written from now on.
    pub fn set_time_millis(&self, unix_millis: u64) {
        self.lock().now_millis = unix_millis;
    }

    /// Every file path, sorted. Handy for asserting that nothing unexpected was left behind.
    pub fn file_paths(&self) -> Vec<String> {
        let state = self.lock();
        state.nodes.iter().filter(|(_, node)| matches!(node, Node::File { .. })).map(|(path, _)| path.clone()).collect()
    }

    /// The total size of all files in bytes.
    pub fn total_bytes(&self) -> u64 {
        let state = self.lock();
        state
            .nodes
            .values()
            .map(|node| match node {
                Node::File { bytes, .. } => bytes.len() as u64,
                Node::Directory => 0,
            })
            .sum()
    }

    fn lock(&self) -> MutexGuard<'_, MemoryState> {
        // A panic while the lock was held cannot leave the map half-updated in a way that matters
        // to a test double, so a poisoned lock is simply taken over.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Runs a changing operation under the lock, applying read-only mode and injected failures.
    fn mutate<R>(&self, change: impl FnOnce(&mut MemoryState) -> Result<R, PlatformError>) -> Result<R, PlatformError> {
        let mut state = self.lock();
        if state.read_only {
            return Err(PlatformError::PermissionDenied);
        }
        if state.failure_budget == Some(0) {
            return Err(PlatformError::Other("injected failure".to_owned()));
        }
        let result = change(&mut state)?;
        if let Some(budget) = &mut state.failure_budget {
            *budget -= 1;
        }
        Ok(result)
    }
}

impl Default for MemoryFs {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MemoryFs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.lock();
        f.debug_struct("MemoryFs").field("entries", &state.nodes.len()).field("read_only", &state.read_only).finish()
    }
}

impl FileSystem for MemoryFs {
    fn read(&self, path: &str) -> Result<Vec<u8>, PlatformError> {
        let path = normalize_path(path)?;
        let state = self.lock();
        match state.nodes.get(&path) {
            Some(Node::File { bytes, .. }) => Ok(bytes.clone()),
            Some(Node::Directory) => Err(PlatformError::IsADirectory),
            None if path.is_empty() => Err(PlatformError::IsADirectory),
            None => Err(PlatformError::NotFound),
        }
    }

    fn write(&self, path: &str, bytes: &[u8]) -> Result<(), PlatformError> {
        let path = normalize_path(path)?;
        self.mutate(|state| {
            state.check_file_slot(&path)?;
            let modified = state.now_millis;
            state.nodes.insert(path, Node::File { bytes: bytes.to_vec(), modified });
            Ok(())
        })
    }

    fn append(&self, path: &str, bytes: &[u8]) -> Result<(), PlatformError> {
        let path = normalize_path(path)?;
        self.mutate(|state| {
            state.check_file_slot(&path)?;
            let now = state.now_millis;
            match state.nodes.entry(path).or_insert(Node::File { bytes: Vec::new(), modified: now }) {
                Node::File { bytes: existing, modified } => {
                    existing.extend_from_slice(bytes);
                    *modified = now;
                    Ok(())
                }
                Node::Directory => Err(PlatformError::IsADirectory),
            }
        })
    }

    fn rename(&self, from: &str, to: &str) -> Result<(), PlatformError> {
        let from = normalize_path(from)?;
        let to = normalize_path(to)?;
        self.mutate(|state| {
            if from.is_empty() || to.is_empty() {
                return Err(PlatformError::InvalidPath);
            }
            match state.nodes.get(&from) {
                None => Err(PlatformError::NotFound),
                Some(Node::File { .. }) => {
                    if from == to {
                        return Ok(());
                    }
                    state.check_file_slot(&to)?;
                    if let Some(node) = state.nodes.remove(&from) {
                        state.nodes.insert(to, node);
                    }
                    Ok(())
                }
                Some(Node::Directory) => {
                    if from == to {
                        return Ok(());
                    }
                    if state.nodes.contains_key(&to) {
                        return Err(PlatformError::AlreadyExists);
                    }
                    if !state.is_dir(parent_path(&to)) {
                        return Err(PlatformError::NotFound);
                    }
                    if to.starts_with(&format!("{from}/")) {
                        // A directory cannot move into itself.
                        return Err(PlatformError::InvalidPath);
                    }
                    let moved: Vec<String> = state.below(&from).map(|(path, _)| path.clone()).collect();
                    for old in moved {
                        if let Some(node) = state.nodes.remove(&old) {
                            state.nodes.insert(format!("{to}{}", &old[from.len()..]), node);
                        }
                    }
                    state.nodes.remove(&from);
                    state.nodes.insert(to, Node::Directory);
                    Ok(())
                }
            }
        })
    }

    fn list(&self, dir: &str) -> Result<Vec<DirEntry>, PlatformError> {
        let dir = normalize_path(dir)?;
        let state = self.lock();
        match state.nodes.get(&dir) {
            _ if dir.is_empty() => {}
            Some(Node::Directory) => {}
            Some(Node::File { .. }) => return Err(PlatformError::NotADirectory),
            None => return Err(PlatformError::NotFound),
        }
        let skip = if dir.is_empty() { 0 } else { dir.len() + 1 };
        let mut entries: Vec<DirEntry> = state
            .below(&dir)
            .filter_map(|(path, node)| {
                let name = path.get(skip..)?;
                if name.contains('/') {
                    return None;
                }
                let (kind, len) = match node {
                    Node::File { bytes, .. } => (FileKind::File, bytes.len() as u64),
                    Node::Directory => (FileKind::Directory, 0),
                };
                Some(DirEntry { name: name.to_owned(), kind, len })
            })
            .collect();
        // Map order compares whole paths ("a.b" sorts before "a/b"); the contract is order by name.
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(entries)
    }

    fn metadata(&self, path: &str) -> Result<Metadata, PlatformError> {
        let path = normalize_path(path)?;
        let state = self.lock();
        match state.nodes.get(&path) {
            Some(Node::File { bytes, modified }) => {
                Ok(Metadata { kind: FileKind::File, len: bytes.len() as u64, modified_unix_millis: Some(*modified) })
            }
            Some(Node::Directory) => Ok(Metadata { kind: FileKind::Directory, len: 0, modified_unix_millis: None }),
            None if path.is_empty() => Ok(Metadata { kind: FileKind::Directory, len: 0, modified_unix_millis: None }),
            None => Err(PlatformError::NotFound),
        }
    }

    fn remove(&self, path: &str) -> Result<(), PlatformError> {
        let path = normalize_path(path)?;
        self.mutate(|state| {
            if path.is_empty() {
                return Err(PlatformError::InvalidPath);
            }
            match state.nodes.get(&path) {
                None => return Err(PlatformError::NotFound),
                Some(Node::Directory) if state.below(&path).next().is_some() => return Err(PlatformError::NotEmpty),
                Some(_) => {}
            }
            state.nodes.remove(&path);
            Ok(())
        })
    }

    fn create_dir_all(&self, path: &str) -> Result<(), PlatformError> {
        let path = normalize_path(path)?;
        self.mutate(|state| {
            if path.is_empty() {
                return Ok(());
            }
            // Every prefix that ends just before a separator, then the whole path. The offsets come
            // from `match_indices`, so they are character boundaries whatever the names contain.
            let ends = path.match_indices('/').map(|(at, _)| at).chain(std::iter::once(path.len()));
            for end in ends {
                let Some(prefix) = path.get(..end) else {
                    return Err(PlatformError::InvalidPath);
                };
                match state.nodes.get(prefix) {
                    Some(Node::Directory) => {}
                    Some(Node::File { .. }) => return Err(PlatformError::NotADirectory),
                    None => {
                        state.nodes.insert(prefix.to_owned(), Node::Directory);
                    }
                }
            }
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------------------------------
// ManualClock, InlineSpawner, SeededEntropy
// ---------------------------------------------------------------------------------------------------

/// A clock that moves only when told to, for deterministic tests.
///
/// With [`ManualClock::set_auto_advance`] every reading of the monotonic time also moves the clock
/// forward by a fixed step, which gives code that measures intervals something to measure.
#[derive(Debug, Default)]
pub struct ManualClock {
    nanos: AtomicU64,
    unix_millis: AtomicU64,
    step_nanos: AtomicU64,
}

impl ManualClock {
    /// A clock at monotonic time 0 and wall-clock time `unix_millis`.
    #[must_use]
    pub const fn new(unix_millis: u64) -> Self {
        Self { nanos: AtomicU64::new(0), unix_millis: AtomicU64::new(unix_millis), step_nanos: AtomicU64::new(0) }
    }

    /// Moves both the monotonic and the wall clock forward by `nanos` (the wall clock by the whole
    /// milliseconds crossed).
    pub fn advance_nanos(&self, nanos: u64) {
        self.step(nanos);
    }

    /// Moves the monotonic clock by `nanos` and the wall clock by the whole milliseconds crossed;
    /// returns the monotonic time before the move. Both saturate at `u64::MAX` instead of wrapping,
    /// so the monotonic time never goes backwards.
    fn step(&self, nanos: u64) -> u64 {
        let mut before = self.nanos.load(Ordering::Relaxed);
        loop {
            let after = before.saturating_add(nanos);
            match self.nanos.compare_exchange_weak(before, after, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => {
                    let millis = after / 1_000_000 - before / 1_000_000;
                    let mut wall = self.unix_millis.load(Ordering::Relaxed);
                    while let Err(seen) = self.unix_millis.compare_exchange_weak(
                        wall,
                        wall.saturating_add(millis),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        wall = seen;
                    }
                    return before;
                }
                Err(seen) => before = seen,
            }
        }
    }

    /// Moves both clocks forward by `millis` milliseconds.
    pub fn advance_millis(&self, millis: u64) {
        self.advance_nanos(millis.saturating_mul(1_000_000));
    }

    /// Sets the wall-clock time without touching the monotonic clock (a user changing the date).
    pub fn set_unix_millis(&self, unix_millis: u64) {
        self.unix_millis.store(unix_millis, Ordering::Relaxed);
    }

    /// Makes every call of [`Clock::now_nanos`] advance the clock by `step_nanos` after reading it
    /// (0 turns this off).
    pub fn set_auto_advance(&self, step_nanos: u64) {
        self.step_nanos.store(step_nanos, Ordering::Relaxed);
    }
}

impl Clock for ManualClock {
    fn now_nanos(&self) -> u64 {
        let step = self.step_nanos.load(Ordering::Relaxed);
        if step == 0 { self.nanos.load(Ordering::Relaxed) } else { self.step(step) }
    }

    fn unix_millis(&self) -> u64 {
        self.unix_millis.load(Ordering::Relaxed)
    }
}

/// The spawner of a platform without threads: it reports zero available threads, and `spawn` runs the
/// work on the calling thread before it returns.
///
/// That is right for one-shot background work in tests (it has simply finished when `spawn`
/// returns) and wrong for a loop that waits for something: code with such a loop checks
/// [`Spawner::available_threads`] first, as the job system does `[THR-04]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct InlineSpawner;

impl Spawner for InlineSpawner {
    fn available_threads(&self) -> usize {
        0
    }

    fn spawn(&self, _name: &str, work: ThreadWork) -> Result<(), PlatformError> {
        work();
        Ok(())
    }
}

/// Reproducible "entropy" from a seed, for tests. Never use it where unpredictability matters.
#[derive(Debug)]
pub struct SeededEntropy {
    rng: Mutex<Rng>,
}

impl SeededEntropy {
    /// A source whose whole output is decided by `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { rng: Mutex::new(Rng::new(seed)) }
    }
}

impl Entropy for SeededEntropy {
    fn fill(&self, out: &mut [u8]) -> Result<(), PlatformError> {
        self.rng.lock().unwrap_or_else(PoisonError::into_inner).fill(out);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn paths_normalise_and_refuse_to_leave_the_root() {
        for (input, expect) in [
            ("", ""),
            ("/", ""),
            ("a", "a"),
            ("/a/b/", "a/b"),
            ("a//b", "a/b"),
            ("a\\b\\c.txt", "a/b/c.txt"),
            ("./a/./b", "a/b"),
            ("a/b/../c", "a/c"),
            ("a/b/../../c", "c"),
            ("a/..", ""),
            ("C:/Users/x", "C:/Users/x"),
            ("a/.../b", "a/.../b"),
            ("ä/ö", "ä/ö"),
        ] {
            assert_eq!(normalize_path(input).as_deref(), Ok(expect), "{input:?}");
        }
        for bad in ["..", "../a", "a/../..", "a/../../b", "/..", "a\0b", "x/\0"] {
            assert_eq!(normalize_path(bad), Err(PlatformError::InvalidPath), "{bad:?}");
        }
    }

    #[test]
    fn path_helpers_split_and_join() {
        assert_eq!(join_path("a/b", "c.txt"), "a/b/c.txt");
        assert_eq!(join_path("a/b/", "/c.txt"), "a/b/c.txt");
        assert_eq!(join_path("", "c"), "c");
        assert_eq!(join_path("a", ""), "a");
        assert_eq!(join_path("C:\\x\\", "y"), "C:\\x/y");
        assert_eq!((parent_path("a/b/c.txt"), file_name("a/b/c.txt")), ("a/b", "c.txt"));
        assert_eq!((parent_path("c.txt"), file_name("c.txt")), ("", "c.txt"));
        assert_eq!((parent_path("a/b/"), file_name("a/b/")), ("a", "b"));
        assert_eq!((parent_path("a\\b"), file_name("a\\b")), ("a", "b"));
        assert_eq!((parent_path(""), file_name("")), ("", ""));
        assert_eq!(extension("a/b/c.tar.gz"), Some("gz"));
        assert_eq!(extension("a.b/c"), None);
        assert_eq!(extension(".gitignore"), None);
        assert_eq!(extension("x."), Some(""));
        assert_eq!(temp_path("doc/a.json"), "doc/a.json.tmp~");
    }

    #[test]
    fn memory_fs_reads_writes_appends_and_reports_metadata() {
        let fs = MemoryFs::new();
        assert_eq!(fs.read("a.txt"), Err(PlatformError::NotFound));
        assert!(!fs.exists("a.txt") && fs.exists("") && fs.exists("/"));
        fs.set_time_millis(1000);
        fs.write("a.txt", b"hello").expect("write");
        assert_eq!(fs.read("/a.txt").as_deref(), Ok(&b"hello"[..]));
        assert_eq!(fs.read("./x/../a.txt").as_deref(), Ok(&b"hello"[..]));
        fs.set_time_millis(2000);
        fs.append("a.txt", b", world").expect("append");
        fs.append("log.txt", b"line\n").expect("append creates");
        assert_eq!(fs.read("a.txt").as_deref(), Ok(&b"hello, world"[..]));
        let meta = fs.metadata("a.txt").expect("metadata");
        assert!(meta.is_file() && !meta.is_dir());
        assert_eq!((meta.len, meta.modified_unix_millis), (12, Some(2000)));
        fs.write("a.txt", b"x").expect("replace");
        assert_eq!(fs.metadata("a.txt").map(|m| m.len), Ok(1));
        let root = fs.metadata("").expect("root");
        assert!(root.is_dir() && root.modified_unix_millis.is_none());
        assert_eq!(fs.metadata("nope"), Err(PlatformError::NotFound));
        assert_eq!((fs.file_paths(), fs.total_bytes()), (vec!["a.txt".to_owned(), "log.txt".to_owned()], 6));
        assert!(format!("{fs:?}").contains("entries: 2"));
    }

    #[test]
    fn memory_fs_directories_behave_like_real_ones() {
        let fs = MemoryFs::default();
        assert_eq!(fs.write("dir/a.txt", b"1"), Err(PlatformError::NotFound), "missing parent");
        fs.create_dir_all("dir/sub/deep").expect("mkdir -p");
        fs.create_dir_all("dir/sub").expect("existing directory is fine");
        fs.create_dir_all("").expect("the root exists");
        fs.write("dir/a.txt", b"1").expect("write");
        fs.write("dir/sub/b.bin", b"22").expect("write");
        fs.write("dir.txt", b"333").expect("write");

        assert_eq!(fs.write("dir", b"x"), Err(PlatformError::IsADirectory));
        assert_eq!(fs.write("", b"x"), Err(PlatformError::IsADirectory));
        assert_eq!(fs.append("dir/sub", b"x"), Err(PlatformError::IsADirectory));
        assert_eq!(fs.read("dir"), Err(PlatformError::IsADirectory));
        assert_eq!(fs.read(""), Err(PlatformError::IsADirectory));
        assert_eq!(fs.write("dir/a.txt/c", b"x"), Err(PlatformError::NotADirectory));
        assert_eq!(fs.create_dir_all("dir/a.txt/c"), Err(PlatformError::NotADirectory));
        assert_eq!(fs.list("dir/a.txt"), Err(PlatformError::NotADirectory));
        assert_eq!(fs.list("nope"), Err(PlatformError::NotFound));
        assert_eq!(fs.write("../escape", b"x"), Err(PlatformError::InvalidPath));

        let names = |dir: &str| -> Vec<(String, FileKind, u64)> {
            fs.list(dir).expect("list").into_iter().map(|e| (e.name, e.kind, e.len)).collect()
        };
        assert_eq!(names(""), [("dir".to_owned(), FileKind::Directory, 0), ("dir.txt".to_owned(), FileKind::File, 3)]);
        assert_eq!(names("dir"), [("a.txt".to_owned(), FileKind::File, 1), ("sub".to_owned(), FileKind::Directory, 0)]);
        assert_eq!(
            names("dir/sub"),
            [("b.bin".to_owned(), FileKind::File, 2), ("deep".to_owned(), FileKind::Directory, 0)]
        );
        assert!(names("dir/sub/deep").is_empty());

        assert_eq!(fs.remove("dir"), Err(PlatformError::NotEmpty));
        assert_eq!(fs.remove("dir/sub"), Err(PlatformError::NotEmpty));
        assert_eq!(fs.remove("nope"), Err(PlatformError::NotFound));
        assert_eq!(fs.remove(""), Err(PlatformError::InvalidPath));
        fs.remove("dir/sub/deep").expect("empty directory");
        fs.remove("dir/sub/b.bin").expect("file");
        fs.remove("dir/sub").expect("now empty");
        assert_eq!(names("dir"), [("a.txt".to_owned(), FileKind::File, 1)]);
    }

    #[test]
    fn memory_fs_rename_moves_files_and_directory_trees() {
        let fs = MemoryFs::new();
        fs.create_dir_all("a/b").expect("dirs");
        fs.write("a/one", b"1").expect("write");
        fs.write("a/b/two", b"2").expect("write");
        fs.write("a.sibling", b"s").expect("write");
        fs.write("target", b"old").expect("write");

        fs.rename("a/one", "target").expect("replace a file");
        assert_eq!(fs.read("target").as_deref(), Ok(&b"1"[..]));
        assert_eq!(fs.read("a/one"), Err(PlatformError::NotFound));
        fs.rename("target", "target").expect("rename onto itself");
        assert_eq!(fs.rename("missing", "x"), Err(PlatformError::NotFound));
        assert_eq!(fs.rename("target", "a"), Err(PlatformError::IsADirectory));
        assert_eq!(fs.rename("target", "no/dir/x"), Err(PlatformError::NotFound));
        assert_eq!(fs.rename("", "x"), Err(PlatformError::InvalidPath));

        assert_eq!(fs.rename("a", "target"), Err(PlatformError::AlreadyExists));
        assert_eq!(fs.rename("a", "a/b/inside"), Err(PlatformError::InvalidPath));
        assert_eq!(fs.rename("a", "no/where"), Err(PlatformError::NotFound));
        fs.rename("a", "moved").expect("move a tree");
        assert_eq!(fs.read("moved/b/two").as_deref(), Ok(&b"2"[..]));
        assert!(fs.metadata("moved/b").expect("dir").is_dir());
        assert!(!fs.exists("a") && !fs.exists("a/b/two"));
        assert_eq!(fs.read("a.sibling").as_deref(), Ok(&b"s"[..]), "a name that merely starts alike stays");
        assert_eq!(fs.file_paths(), ["a.sibling", "moved/b/two", "target"]);
    }

    // [SAVE-02] An atomic write interrupted at any step leaves the old content or the new, and no
    // temporary file.
    #[test]
    fn write_atomic_is_all_or_nothing_under_injected_failures() {
        let fs = MemoryFs::new();
        fs.create_dir_all("doc").expect("dir");
        fs.write_atomic("doc/state.json", b"old").expect("first save");
        assert_eq!(fs.file_paths(), ["doc/state.json"]);

        for budget in 0..4 {
            fs.fail_after(Some(budget));
            let result = fs.write_atomic("doc/state.json", b"new content");
            fs.fail_after(None);
            let content = fs.read("doc/state.json").expect("the file always exists");
            if result.is_ok() {
                assert_eq!(content, b"new content");
                assert!(budget >= 2, "write and rename are two operations");
            } else {
                assert_eq!(result, Err(PlatformError::Other("injected failure".to_owned())));
                assert_eq!(content, b"old", "failure after {budget} operations lost data");
            }
            // The clean-up itself may have been refused; one more attempt always clears it.
            if fs.exists(&temp_path("doc/state.json")) {
                fs.remove(&temp_path("doc/state.json")).expect("remove temp");
            }
            fs.write_atomic("doc/state.json", b"old").expect("restore");
            assert_eq!(fs.file_paths(), ["doc/state.json"]);
        }

        // With clean-up possible nothing is left behind: the write succeeds, the rename is refused.
        assert_eq!(fs.write_atomic("missing-dir/x", b"1"), Err(PlatformError::NotFound));
        fs.create_dir_all("taken").expect("dir");
        assert_eq!(fs.write_atomic("taken", b"1"), Err(PlatformError::IsADirectory));
        assert_eq!(fs.file_paths(), ["doc/state.json"]);
    }

    #[test]
    fn memory_fs_read_only_mode_refuses_every_change() {
        let fs = MemoryFs::new();
        fs.write("a", b"1").expect("write");
        fs.set_read_only(true);
        let denied = Err(PlatformError::PermissionDenied);
        assert_eq!(fs.write("a", b"2"), denied);
        assert_eq!(fs.write_atomic("a", b"2"), denied);
        assert_eq!(fs.append("a", b"2"), denied);
        assert_eq!(fs.rename("a", "b"), denied);
        assert_eq!(fs.remove("a"), denied);
        assert_eq!(fs.create_dir_all("d"), denied);
        assert_eq!(fs.read("a").as_deref(), Ok(&b"1"[..]));
        assert_eq!(fs.list("").map(|l| l.len()), Ok(1));
        fs.set_read_only(false);
        fs.write("a", b"2").expect("writable again");
    }

    #[test]
    fn memory_fs_is_usable_through_the_trait_from_several_threads() {
        let fs: Arc<dyn FileSystem> = Arc::new(MemoryFs::new());
        let workers: Vec<_> = (0..4)
            .map(|t| {
                let fs = Arc::clone(&fs);
                std::thread::spawn(move || {
                    for i in 0..100 {
                        fs.append("journal", &[t as u8]).expect("append");
                        fs.write_atomic(&format!("file-{t}"), &[i as u8]).expect("atomic");
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().expect("join");
        }
        let journal = fs.read("journal").expect("journal");
        assert_eq!(journal.len(), 400);
        for t in 0..4_u8 {
            assert_eq!(journal.iter().map(|&b| usize::from(b == t)).sum::<usize>(), 100);
            assert_eq!(fs.read(&format!("file-{t}")).as_deref(), Ok(&[99_u8][..]));
        }
        assert_eq!(fs.list("").map(|l| l.len()), Ok(5));
    }

    #[test]
    fn manual_clock_moves_only_when_told() {
        let clock = ManualClock::new(1_700_000_000_000);
        assert_eq!((clock.now_nanos(), clock.now_nanos(), clock.unix_millis()), (0, 0, 1_700_000_000_000));
        clock.advance_nanos(1_500_000);
        assert_eq!((clock.now_nanos(), clock.unix_millis()), (1_500_000, 1_700_000_000_001));
        clock.advance_nanos(500_000);
        assert_eq!((clock.now_nanos(), clock.unix_millis()), (2_000_000, 1_700_000_000_002));
        clock.advance_millis(8);
        assert_eq!((clock.now_nanos(), clock.unix_millis()), (10_000_000, 1_700_000_000_010));
        clock.set_unix_millis(5);
        assert_eq!((clock.now_nanos(), clock.unix_millis()), (10_000_000, 5));

        clock.set_auto_advance(250);
        assert_eq!((clock.now_nanos(), clock.now_nanos(), clock.now_nanos()), (10_000_000, 10_000_250, 10_000_500));
        clock.set_auto_advance(0);
        assert_eq!(clock.now_nanos(), 10_000_750);
        let as_trait: &dyn Clock = &ManualClock::default();
        assert_eq!((as_trait.now_nanos(), as_trait.unix_millis()), (0, 0));

        // The end of the range saturates: the monotonic time never wraps backwards.
        let late = ManualClock::new(u64::MAX - 1);
        late.advance_millis(u64::MAX);
        late.advance_nanos(u64::MAX);
        assert_eq!((late.now_nanos(), late.unix_millis()), (u64::MAX, u64::MAX));
    }

    // Names are UTF-8: no operation may cut a path in the middle of a character.
    #[test]
    fn memory_fs_handles_names_outside_ascii() {
        let fs = MemoryFs::new();
        fs.create_dir_all("ä/ö€/名前").expect("mkdir -p");
        fs.write("ä/ö€/名前/ß.txt", b"1").expect("write");
        fs.create_dir_all("ä/ö€").expect("again");
        assert_eq!(fs.list("ä").map(|l| l.into_iter().map(|e| e.name).collect::<Vec<_>>()), Ok(vec!["ö€".to_owned()]));
        fs.rename("ä", "é").expect("move the tree");
        assert_eq!(fs.read("é/ö€/名前/ß.txt").as_deref(), Ok(&b"1"[..]));
        assert_eq!(fs.file_paths(), ["é/ö€/名前/ß.txt"]);
        assert_eq!((parent_path("é/ö€"), file_name("é/ö€"), extension("名.前")), ("é", "ö€", Some("前")));
    }

    #[test]
    fn inline_spawner_has_no_threads_and_runs_work_in_place() {
        let spawner = InlineSpawner;
        assert_eq!(spawner.available_threads(), 0);
        let ran = Arc::new(AtomicUsize::new(0));
        let here = std::thread::current().id();
        let counter = Arc::clone(&ran);
        spawner
            .spawn(
                "inline",
                Box::new(move || {
                    assert_eq!(std::thread::current().id(), here);
                    counter.fetch_add(1, Ordering::SeqCst);
                }),
            )
            .expect("spawn");
        assert_eq!(ran.load(Ordering::SeqCst), 1, "the work has finished when spawn returns");
    }

    #[test]
    fn seeded_entropy_is_reproducible_and_fills_everything() {
        let (a, b, c) = (SeededEntropy::new(9), SeededEntropy::new(9), SeededEntropy::new(10));
        let (mut x, mut y, mut z) = ([0_u8; 37], [0_u8; 37], [0_u8; 37]);
        a.fill(&mut x).expect("fill");
        b.fill(&mut y).expect("fill");
        c.fill(&mut z).expect("fill");
        assert_eq!(x, y);
        assert_ne!(x, z);
        assert!(x.iter().map(|&byte| usize::from(byte == 0)).sum::<usize>() < 5);
        a.fill(&mut y).expect("fill");
        assert_ne!(x, y, "the stream continues");
        a.fill(&mut []).expect("empty");
    }
}
