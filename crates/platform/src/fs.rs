//! [`StdFileSystem`]: the native [`FileSystem`] `[PLAT-03]`, over `std::fs`.
//!
//! Two scopes:
//!
//! - **Rooted** ([`StdFileSystem::rooted`]): every path is relative to a root directory and cannot
//!   leave it. `..` above the root, drive prefixes and (on Windows) names the system treats as
//!   devices or that it cannot store are [`PlatformError::InvalidPath`]; a symbolic link inside the
//!   root that leads outside it is refused the same way. This is what a document, a plug-in or the
//!   agent server gets: a path written in a file can never reach anything outside its folder. The
//!   check guards against paths in data, not against another process rearranging the directory
//!   between the check and the use.
//! - **Host** ([`StdFileSystem::host`]): absolute paths in the platform's own form (`C:\x` or
//!   `C:/x` on Windows, `/x` elsewhere), as a file dialog returns them, for the editor's own
//!   open and save. Relative paths resolve against the process's working directory.
//!
//! [`FileSystem::write_atomic`] is overridden to flush the temporary file to the device before it is
//! renamed over the target `[SAVE-02]`, and (on Unix) to flush the directory after the rename.

use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use humanart_core::error::PlatformError;
use humanart_core::platform::{DirEntry, FileKind, FileSystem, Metadata, normalize_path, temp_path};

#[derive(Clone, Debug)]
enum Scope {
    /// Paths are relative to this canonical directory and cannot leave it.
    Rooted(PathBuf),
    /// Paths are the platform's own.
    Host,
}

/// The native file system. See the module documentation for the two scopes.
#[derive(Clone, Debug)]
pub struct StdFileSystem {
    scope: Scope,
}

impl StdFileSystem {
    /// A file system confined to `root`, which must be an existing directory.
    pub fn rooted(root: impl AsRef<Path>) -> Result<Self, PlatformError> {
        let canonical = fs::canonicalize(root.as_ref())?;
        if !canonical.is_dir() {
            return Err(PlatformError::NotADirectory);
        }
        Ok(Self { scope: Scope::Rooted(canonical) })
    }

    /// Creates `root` (and its parents) if needed, then [`StdFileSystem::rooted`].
    pub fn create_rooted(root: impl AsRef<Path>) -> Result<Self, PlatformError> {
        fs::create_dir_all(root.as_ref())?;
        Self::rooted(root)
    }

    /// A file system that takes the platform's own absolute paths.
    #[must_use]
    pub const fn host() -> Self {
        Self { scope: Scope::Host }
    }

    /// The canonical root directory, `None` for the host scope.
    pub fn root(&self) -> Option<&Path> {
        match &self.scope {
            Scope::Rooted(root) => Some(root),
            Scope::Host => None,
        }
    }

    /// The operating-system path `path` refers to, after the scope's checks.
    pub fn os_path(&self, path: &str) -> Result<PathBuf, PlatformError> {
        match &self.scope {
            Scope::Host => {
                if path.is_empty() || path.contains('\0') {
                    return Err(PlatformError::InvalidPath);
                }
                Ok(PathBuf::from(path))
            }
            Scope::Rooted(root) => {
                let relative = normalize_path(path)?;
                let mut out = root.clone();
                for component in relative.split('/').filter(|c| !c.is_empty()) {
                    check_component(component)?;
                    out.push(component);
                }
                confine(root, &out)?;
                Ok(out)
            }
        }
    }

    /// `true` when `path` names the root of a rooted file system.
    fn is_root(&self, path: &str) -> bool {
        matches!(self.scope, Scope::Rooted(_)) && normalize_path(path).is_ok_and(|p| p.is_empty())
    }
}

/// Refuses a component the operating system would not store under that name, or would read as
/// something other than a file name.
fn check_component(component: &str) -> Result<(), PlatformError> {
    if cfg!(windows) {
        let bad_char = component.chars().any(|c| c < ' ' || matches!(c, ':' | '<' | '>' | '"' | '|' | '?' | '*'));
        // Windows drops trailing dots and spaces, so `a.` and `a` would be the same file.
        let bad_end = component.ends_with(['.', ' ']);
        let stem = component.split('.').next().unwrap_or(component).trim_end();
        let device = matches!(stem.to_ascii_uppercase().as_str(), "CON" | "PRN" | "AUX" | "NUL")
            || (stem.len() == 4
                && (stem[..3].eq_ignore_ascii_case("COM") || stem[..3].eq_ignore_ascii_case("LPT"))
                && stem.as_bytes()[3].is_ascii_digit());
        if bad_char || bad_end || device {
            return Err(PlatformError::InvalidPath);
        }
    }
    Ok(())
}

/// Checks that `path`, with every symbolic link resolved, stays below `root`. The deepest existing
/// ancestor is resolved; a dangling link is refused because writing through it would create its
/// target wherever it points.
fn confine(root: &Path, path: &Path) -> Result<(), PlatformError> {
    let mut probe = path;
    loop {
        match fs::canonicalize(probe) {
            Ok(real) => return if real.starts_with(root) { Ok(()) } else { Err(PlatformError::InvalidPath) },
            Err(error) if error.kind() == ErrorKind::NotFound => {
                if fs::symlink_metadata(probe).is_ok() {
                    return Err(PlatformError::InvalidPath);
                }
                probe = probe.parent().ok_or(PlatformError::NotFound)?;
            }
            Err(error) => return Err(error.into()),
        }
    }
}

/// Checks that `path` can hold a file: it is not a directory, and its parent is one.
fn check_file_slot(path: &Path) -> Result<(), PlatformError> {
    if path.is_dir() {
        return Err(PlatformError::IsADirectory);
    }
    match path.parent() {
        None => Err(PlatformError::InvalidPath),
        Some(parent) if parent.as_os_str().is_empty() => Ok(()),
        Some(parent) => match fs::metadata(parent) {
            Ok(meta) if meta.is_dir() => Ok(()),
            Ok(_) => Err(PlatformError::NotADirectory),
            Err(error) => Err(error.into()),
        },
    }
}

/// `fs::rename`, retried briefly on Windows, where a virus scanner or the search indexer can hold
/// the target open for a moment and make a replace fail with "access denied".
fn rename_with_retry(from: &Path, to: &Path) -> std::io::Result<()> {
    let attempts = if cfg!(windows) { 5 } else { 1 };
    let mut delay_ms = 2;
    let mut result = fs::rename(from, to);
    for _ in 1..attempts {
        match &result {
            Err(error) if error.kind() == ErrorKind::PermissionDenied => {
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
                delay_ms *= 2;
                result = fs::rename(from, to);
            }
            _ => break,
        }
    }
    result
}

/// Flushes a directory's entries to the device after a rename (Unix; NTFS journals its metadata and
/// Windows offers no portable way to flush a directory). Best effort: some file systems refuse it.
fn sync_directory(dir: Option<&Path>) {
    #[cfg(unix)]
    if let Some(dir) = dir.filter(|d| !d.as_os_str().is_empty())
        && let Ok(handle) = File::open(dir)
    {
        let _ = handle.sync_all();
    }
    #[cfg(not(unix))]
    let _ = dir;
}

fn modified_millis(meta: &fs::Metadata) -> Option<u64> {
    let since = meta.modified().ok()?.duration_since(UNIX_EPOCH).ok()?;
    u64::try_from(since.as_millis()).ok()
}

impl FileSystem for StdFileSystem {
    fn read(&self, path: &str) -> Result<Vec<u8>, PlatformError> {
        let os = self.os_path(path)?;
        if os.is_dir() {
            return Err(PlatformError::IsADirectory);
        }
        Ok(fs::read(os)?)
    }

    fn write(&self, path: &str, bytes: &[u8]) -> Result<(), PlatformError> {
        let os = self.os_path(path)?;
        check_file_slot(&os)?;
        Ok(fs::write(os, bytes)?)
    }

    fn append(&self, path: &str, bytes: &[u8]) -> Result<(), PlatformError> {
        let os = self.os_path(path)?;
        check_file_slot(&os)?;
        let mut file = OpenOptions::new().append(true).create(true).open(os)?;
        Ok(file.write_all(bytes)?)
    }

    fn rename(&self, from: &str, to: &str) -> Result<(), PlatformError> {
        if self.is_root(from) || self.is_root(to) {
            return Err(PlatformError::InvalidPath);
        }
        let from_os = self.os_path(from)?;
        let to_os = self.os_path(to)?;
        let meta = fs::symlink_metadata(&from_os)?;
        if from_os == to_os {
            return Ok(());
        }
        if meta.is_dir() {
            if fs::symlink_metadata(&to_os).is_ok() {
                return Err(PlatformError::AlreadyExists);
            }
            if to_os.starts_with(&from_os) {
                // A directory cannot move into itself.
                return Err(PlatformError::InvalidPath);
            }
            match to_os.parent().map(fs::metadata) {
                Some(Ok(parent)) if parent.is_dir() => {}
                Some(Ok(_)) => return Err(PlatformError::NotADirectory),
                Some(Err(error)) => return Err(error.into()),
                None => return Err(PlatformError::InvalidPath),
            }
        } else {
            check_file_slot(&to_os)?;
        }
        Ok(rename_with_retry(&from_os, &to_os)?)
    }

    fn list(&self, dir: &str) -> Result<Vec<DirEntry>, PlatformError> {
        let os = self.os_path(dir)?;
        if !fs::metadata(&os)?.is_dir() {
            return Err(PlatformError::NotADirectory);
        }
        let mut entries = Vec::new();
        for entry in fs::read_dir(&os)? {
            let entry = entry?;
            // A name that is not UTF-8 cannot be expressed as a path of this API: it is left out.
            let Ok(name) = entry.file_name().into_string() else { continue };
            let mut file_type = entry.file_type()?;
            let mut len = 0;
            if file_type.is_symlink() {
                // Report what the link leads to; a dangling link is left out.
                let Ok(target) = fs::metadata(entry.path()) else { continue };
                file_type = target.file_type();
                len = target.len();
            } else if file_type.is_file() {
                len = entry.metadata()?.len();
            }
            let kind = if file_type.is_dir() { FileKind::Directory } else { FileKind::File };
            entries.push(DirEntry { name, kind, len: if kind == FileKind::Directory { 0 } else { len } });
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(entries)
    }

    fn metadata(&self, path: &str) -> Result<Metadata, PlatformError> {
        let meta = fs::metadata(self.os_path(path)?)?;
        let modified_unix_millis = modified_millis(&meta);
        Ok(if meta.is_dir() {
            Metadata { kind: FileKind::Directory, len: 0, modified_unix_millis }
        } else {
            Metadata { kind: FileKind::File, len: meta.len(), modified_unix_millis }
        })
    }

    fn remove(&self, path: &str) -> Result<(), PlatformError> {
        if self.is_root(path) {
            return Err(PlatformError::InvalidPath);
        }
        let os = self.os_path(path)?;
        let meta = fs::symlink_metadata(&os)?;
        if meta.is_dir() {
            fs::remove_dir(&os).map_err(|error| {
                if fs::read_dir(&os).is_ok_and(|mut entries| entries.next().is_some()) {
                    PlatformError::NotEmpty
                } else {
                    error.into()
                }
            })
        } else {
            Ok(fs::remove_file(&os)?)
        }
    }

    fn create_dir_all(&self, path: &str) -> Result<(), PlatformError> {
        let os = self.os_path(path)?;
        // A file anywhere on the way makes the path impossible; say so rather than whatever the
        // operating system calls it.
        if os.ancestors().any(Path::is_file) {
            return Err(PlatformError::NotADirectory);
        }
        Ok(fs::create_dir_all(os)?)
    }

    fn write_atomic(&self, path: &str, bytes: &[u8]) -> Result<(), PlatformError> {
        let os = self.os_path(path)?;
        check_file_slot(&os)?;
        let temp = self.os_path(&temp_path(path))?;
        let result = (|| -> std::io::Result<()> {
            let mut file = File::create(&temp)?;
            file.write_all(bytes)?;
            file.sync_all()?;
            drop(file);
            rename_with_retry(&temp, &os)
        })();
        match result {
            Ok(()) => {
                sync_directory(os.parent());
                Ok(())
            }
            Err(error) => {
                // The write's error is the one to report; a failed clean-up changes nothing about it.
                let _ = fs::remove_file(&temp);
                Err(error.into())
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use humanart_core::platform::MemoryFs;

    use super::*;

    /// A fresh directory under the system's temporary directory, deleted when dropped.
    pub(crate) struct TempDir(PathBuf);

    impl TempDir {
        pub(crate) fn new(label: &str) -> Self {
            static NEXT: AtomicU32 = AtomicU32::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let nanos = std::time::SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos();
            let dir =
                std::env::temp_dir().join(format!("humanart-platform-{label}-{}-{n}-{nanos}", std::process::id()));
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        pub(crate) fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn outcome<T: std::fmt::Debug>(result: Result<T, PlatformError>) -> String {
        match result {
            Ok(value) => format!("ok {value:?}"),
            Err(error) => format!("err {error:?}"),
        }
    }

    /// One scripted session through the trait; the log of outcomes is compared between file systems.
    fn exercise(fs: &dyn FileSystem) -> Vec<String> {
        let mut log = Vec::new();
        log.push(outcome(fs.create_dir_all("docs/sub")));
        log.push(outcome(fs.write("docs/a.txt", b"alpha")));
        log.push(outcome(fs.append("docs/a.txt", b"-beta")));
        log.push(outcome(fs.read("docs/a.txt").map(String::from_utf8)));
        log.push(outcome(fs.append("docs/new.log", b"line\n")));
        log.push(outcome(fs.write("docs/\u{e9}t\u{e9} \u{1f600}.txt", "h\u{e9}".as_bytes())));
        log.push(outcome(fs.list("docs")));
        log.push(outcome(fs.metadata("docs/a.txt").map(|m| (m.kind, m.len))));
        log.push(outcome(fs.metadata("docs").map(|m| (m.kind, m.len))));
        log.push(outcome(fs.read("docs")));
        log.push(outcome(fs.read("docs/missing")));
        log.push(outcome(fs.write("nowhere/x", b"x")));
        log.push(outcome(fs.write("docs/a.txt/x", b"x")));
        log.push(outcome(fs.write("docs", b"x")));
        log.push(outcome(fs.create_dir_all("docs/a.txt/deeper")));
        log.push(outcome(fs.list("docs/a.txt")));
        log.push(outcome(fs.list("missing")));
        log.push(outcome(fs.remove("docs")));
        log.push(outcome(fs.remove("docs/missing")));
        log.push(outcome(fs.remove("")));
        log.push(outcome(fs.read("../outside")));
        log.push(outcome(fs.write("docs/b.txt", b"bravo")));
        log.push(outcome(fs.rename("docs/b.txt", "docs/a.txt")));
        log.push(outcome(fs.read("docs/a.txt").map(String::from_utf8)));
        log.push(format!("exists {}", fs.exists("docs/b.txt")));
        log.push(outcome(fs.rename("docs/missing", "docs/c.txt")));
        log.push(outcome(fs.rename("docs/sub", "docs/a.txt")));
        log.push(outcome(fs.rename("docs", "docs/sub/inside")));
        log.push(outcome(fs.write("docs/sub/deep.bin", &[0, 1, 2, 255])));
        log.push(outcome(fs.rename("docs/sub", "moved")));
        log.push(outcome(fs.read("moved/deep.bin")));
        log.push(format!("exists {}", fs.exists("docs/sub")));
        log.push(outcome(fs.write_atomic("docs/a.txt", b"charlie")));
        log.push(outcome(fs.read("docs/a.txt").map(String::from_utf8)));
        log.push(outcome(fs.write_atomic("fresh.txt", b"new")));
        log.push(outcome(fs.write_atomic("docs", b"x")));
        log.push(outcome(fs.write_atomic("nowhere/x", b"x")));
        log.push(outcome(fs.remove("moved/deep.bin")));
        log.push(outcome(fs.remove("moved")));
        log.push(outcome(fs.list("")));
        log
    }

    #[test]
    fn std_file_system_behaves_like_memory_fs_on_the_whole_contract() {
        let dir = TempDir::new("contract");
        let native = StdFileSystem::rooted(dir.path()).unwrap();
        let memory = MemoryFs::new();
        let expected = exercise(&memory);
        let actual = exercise(&native);
        for (step, (want, got)) in expected.iter().zip(&actual).enumerate() {
            assert_eq!(want, got, "step {step}");
        }
        assert_eq!(expected.len(), actual.len());
        // Nothing of the atomic writes is left behind.
        let names: Vec<String> = native.list("docs").unwrap().into_iter().map(|e| e.name).collect();
        assert!(names.iter().all(|n| !n.ends_with(".tmp~")), "{names:?}");
    }

    #[test]
    fn rooted_paths_cannot_leave_the_root() {
        let dir = TempDir::new("escape");
        fs::create_dir_all(dir.path().join("inner")).unwrap();
        fs::write(dir.path().join("secret.txt"), b"outside").unwrap();
        let fs = StdFileSystem::rooted(dir.path().join("inner")).unwrap();
        for path in ["..", "../secret.txt", "a/../../secret.txt", "..\\secret.txt", "x\0y"] {
            assert_eq!(fs.read(path), Err(PlatformError::InvalidPath), "{path:?}");
            assert_eq!(fs.write(path, b"x"), Err(PlatformError::InvalidPath), "{path:?}");
        }
        // A leading separator is the root, not the drive's root.
        fs.write("/top.txt", b"inside").unwrap();
        assert_eq!(fs::read(dir.path().join("inner/top.txt")).unwrap(), b"inside");
        assert_eq!(fs::read(dir.path().join("secret.txt")).unwrap(), b"outside");
        assert!(fs.os_path("a/b").unwrap().starts_with(fs.root().unwrap()));
    }

    #[cfg(windows)]
    #[test]
    fn rooted_paths_refuse_drives_devices_and_unstorable_names_on_windows() {
        let dir = TempDir::new("names");
        let fs = StdFileSystem::rooted(dir.path()).unwrap();
        for path in [
            "C:/Windows/win.ini",
            "C:secret",
            "file.txt:stream",
            "CON",
            "nul.txt",
            "Com1",
            "lpt9.log",
            "a?",
            "trail.",
            "trail ",
        ] {
            assert_eq!(fs.os_path(path), Err(PlatformError::InvalidPath), "{path:?}");
        }
        for path in ["CONTENT", "com", "console.txt", "a.b.c"] {
            assert!(fs.os_path(path).is_ok(), "{path:?}");
        }
    }

    /// A directory link from `link` to `target`. Windows: a junction (`mklink /J`), which needs no
    /// privilege, unlike a symbolic link; it may point at a directory that does not exist.
    fn link_dir(target: &Path, link: &Path) {
        #[cfg(unix)]
        std::os::unix::fs::symlink(target, link).unwrap();
        #[cfg(windows)]
        {
            let status = std::process::Command::new("cmd")
                .args(["/C", "mklink", "/J"])
                .arg(link)
                .arg(target)
                .stdout(std::process::Stdio::null())
                .status()
                .unwrap();
            assert!(status.success());
        }
    }

    #[test]
    fn rooted_paths_refuse_links_that_lead_outside() {
        let dir = TempDir::new("links");
        let inner = dir.path().join("inner");
        let outside = dir.path().join("outside");
        fs::create_dir_all(inner.join("own")).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("secret.txt"), b"outside").unwrap();
        link_dir(&outside, &inner.join("door"));
        link_dir(&outside.join("gone"), &inner.join("dangling"));
        link_dir(&inner.join("own"), &inner.join("alias"));
        let fs = StdFileSystem::rooted(&inner).unwrap();
        // A link that stays inside is followed.
        fs.write("alias/through.txt", b"inside").unwrap();
        assert_eq!(fs.read("own/through.txt").unwrap(), b"inside");
        assert_eq!(fs.read("door/secret.txt"), Err(PlatformError::InvalidPath));
        assert_eq!(fs.write("door/new.txt", b"x"), Err(PlatformError::InvalidPath));
        assert_eq!(fs.write("dangling", b"x"), Err(PlatformError::InvalidPath));
        assert_eq!(fs.create_dir_all("dangling/x"), Err(PlatformError::InvalidPath));
        assert_eq!(fs.write("dangling/x", b"x"), Err(PlatformError::InvalidPath));
        assert!(!outside.join("new.txt").exists());
        assert!(!outside.join("gone").exists());
    }

    #[test]
    fn host_scope_takes_absolute_platform_paths() {
        let dir = TempDir::new("host");
        let fs = StdFileSystem::host();
        assert!(fs.root().is_none());
        let file = dir.path().join("picked.txt");
        let text = file.to_str().unwrap();
        fs.write_atomic(text, b"from a dialog").unwrap();
        assert_eq!(fs.read(text).unwrap(), b"from a dialog");
        let forward = text.replace('\\', "/");
        assert_eq!(fs.metadata(&forward).unwrap().len, 13);
        assert!(fs.metadata(&forward).unwrap().modified_unix_millis.unwrap() > 1_577_836_800_000);
        assert_eq!(fs.read(""), Err(PlatformError::InvalidPath));
        assert_eq!(fs.read("a\0b"), Err(PlatformError::InvalidPath));
    }

    #[test]
    fn rooted_needs_an_existing_directory() {
        let dir = TempDir::new("root");
        assert_eq!(StdFileSystem::rooted(dir.path().join("missing")).unwrap_err(), PlatformError::NotFound);
        fs::write(dir.path().join("file"), b"x").unwrap();
        assert_eq!(StdFileSystem::rooted(dir.path().join("file")).unwrap_err(), PlatformError::NotADirectory);
        let created = StdFileSystem::create_rooted(dir.path().join("a/b")).unwrap();
        assert!(created.root().unwrap().is_dir());
    }

    #[test]
    fn write_atomic_keeps_the_old_content_when_the_rename_cannot_happen() {
        let dir = TempDir::new("atomic");
        let fs = StdFileSystem::rooted(dir.path()).unwrap();
        fs.write("doc.bin", b"old").unwrap();
        // The temporary name is occupied by a directory: the write fails before touching the target.
        fs.create_dir_all(&temp_path("doc.bin")).unwrap();
        assert!(fs.write_atomic("doc.bin", b"new").is_err());
        assert_eq!(fs.read("doc.bin").unwrap(), b"old");
        fs.remove(&temp_path("doc.bin")).unwrap();
        // Repeated replacement, as autosave does it.
        for round in 0..20u8 {
            fs.write_atomic("doc.bin", &[round; 3000]).unwrap();
            assert_eq!(fs.read("doc.bin").unwrap(), vec![round; 3000]);
        }
        assert_eq!(fs.list("").unwrap().len(), 1);
    }

    #[test]
    fn the_file_sink_logs_through_the_native_file_system() {
        use std::sync::Arc;
        let dir = TempDir::new("log");
        let native: Arc<dyn FileSystem> = Arc::new(StdFileSystem::rooted(dir.path()).unwrap());
        let sink = humanart_core::log::FileSink::open(Arc::clone(&native), "app.log", 1 << 20).unwrap();
        let logger = humanart_core::log::Logger::new(16);
        logger.add_sink(Arc::new(sink));
        logger.record(humanart_core::log::Level::Error, "platform", "written to disk");
        logger.flush().unwrap();
        let text = String::from_utf8(native.read("app.log").unwrap()).unwrap();
        assert!(text.contains("written to disk"), "{text}");
    }
}
