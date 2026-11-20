//! The server's security measures `[SEC-01]`, `[SEC-02]`.
//!
//! - [`Token`]: the random bearer token generated at startup (256 bits from the platform's
//!   entropy source), compared in constant time.
//! - [`check_host`] / [`check_origin`]: the `Host` header must name a loopback address with the
//!   server's port, and an `Origin` header, when present, must be a loopback `http` origin with that
//!   port. This defeats DNS rebinding (a page on `evil.example` resolving to 127.0.0.1 still sends
//!   `Host: evil.example`) and cross-origin requests from browsers.
//! - [`RateLimiter`]: a token bucket over all requests.
//! - [`Sandbox`]: the only directories agent-supplied paths may name (the project and the exchange
//!   directory). Paths are checked lexically first (no `..`, no drive or stream colons, no device
//!   names, no control characters), then canonicalised (symbolic links resolved) and required to
//!   stay below a root, for reads and for writes.

use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use humanart_core::error::PlatformError;
use humanart_core::platform::Entropy;

/// The bearer token `[SEC-01]`.
#[derive(Clone, PartialEq, Eq)]
pub struct Token(String);

impl std::fmt::Debug for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Token(..)")
    }
}

impl Token {
    /// A fresh token of 32 random bytes (64 hex digits).
    ///
    /// # Errors
    /// The entropy source's error.
    pub fn generate(entropy: &dyn Entropy) -> Result<Self, PlatformError> {
        let mut bytes = [0u8; 32];
        entropy.fill(&mut bytes)?;
        Ok(Self(humanart_core::hex::encode_string(&bytes)))
    }

    /// A token with the given text (tests, or a launcher that chose it). Must be at least 16
    /// visible ASCII characters.
    pub fn from_text(text: &str) -> Option<Self> {
        (text.len() >= 16 && text.len() <= 256 && text.bytes().all(|b| b.is_ascii_graphic()))
            .then(|| Self(text.to_owned()))
    }

    /// The token text (shown in the Agent panel, passed to the launcher).
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Whether an `Authorization` header value carries this token (`Bearer <token>`), compared in
    /// constant time.
    pub fn matches_header(&self, header: Option<&str>) -> bool {
        let Some(value) = header else { return false };
        let Some((scheme, given)) = value.split_once(' ') else { return false };
        if !scheme.eq_ignore_ascii_case("bearer") {
            return false;
        }
        let (a, b) = (self.0.as_bytes(), given.trim().as_bytes());
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
    }
}

const LOOPBACK_NAMES: [&str; 3] = ["127.0.0.1", "localhost", "[::1]"];

/// Whether a `Host` header names a loopback host with `port` `[SEC-01]`.
pub fn check_host(host: &str, port: u16) -> bool {
    let Some((name, p)) = host.rsplit_once(':') else { return false };
    p.parse::<u16>().ok() == Some(port) && LOOPBACK_NAMES.iter().any(|n| n.eq_ignore_ascii_case(name))
}

/// Whether an `Origin` header (absent: `None`) is acceptable: absent, or `http://` + a loopback
/// host + `:port` `[SEC-01]`.
pub fn check_origin(origin: Option<&str>, port: u16) -> bool {
    match origin {
        None => true,
        Some(o) => o.strip_prefix("http://").is_some_and(|rest| check_host(rest, port)),
    }
}

/// A token bucket over all requests `[SEC-01]`.
#[derive(Debug)]
pub struct RateLimiter {
    state: Mutex<(f64, Instant)>,
    capacity: f64,
    per_second: f64,
}

impl RateLimiter {
    /// A bucket of `capacity` requests refilled at `per_second`.
    pub fn new(capacity: u32, per_second: u32) -> Self {
        Self {
            state: Mutex::new((f64::from(capacity), Instant::now())),
            capacity: f64::from(capacity),
            per_second: f64::from(per_second),
        }
    }

    /// Takes one request from the bucket; `false` when it is empty.
    pub fn allow(&self) -> bool {
        let Ok(mut guard) = self.state.lock() else { return false };
        let now = Instant::now();
        let (tokens, last) = *guard;
        let refilled = (tokens + now.duration_since(last).as_secs_f64() * self.per_second).min(self.capacity);
        if refilled >= 1.0 {
            *guard = (refilled - 1.0, now);
            true
        } else {
            *guard = (refilled, now);
            false
        }
    }
}

/// Why a path was refused `[SEC-02]`.
#[derive(Debug)]
pub enum SandboxError {
    /// The path text is not acceptable (empty, too long, `..`, device names ...).
    Invalid(&'static str),
    /// The path resolves outside the project and exchange directories.
    Outside,
    /// The file system failed (missing file, permission).
    Io(std::io::Error),
}

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(why) => write!(f, "the path is not allowed: {why}"),
            Self::Outside => f.write_str("the path is outside the project and exchange directories"),
            Self::Io(e) => write!(f, "file system error: {e}"),
        }
    }
}

impl std::error::Error for SandboxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

/// The longest path the agent may give.
pub const MAX_PATH_BYTES: usize = 1024;

const DEVICE_NAMES: [&str; 22] = [
    "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9", "lpt1", "lpt2",
    "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
];

/// The relative components of an agent path, checked lexically.
fn components(text: &str) -> Result<Vec<&str>, SandboxError> {
    let mut out = Vec::new();
    for part in text.split(['/', '\\']) {
        if part.is_empty() || part == "." {
            continue;
        }
        if part == ".." {
            return Err(SandboxError::Invalid("'..' is not allowed"));
        }
        if part.contains(':') {
            return Err(SandboxError::Invalid("':' (drive or stream) is not allowed in a relative path"));
        }
        if part.ends_with('.') || part.ends_with(' ') {
            return Err(SandboxError::Invalid("a name may not end in a dot or space"));
        }
        if part.bytes().any(|b| b < 0x20 || matches!(b, b'<' | b'>' | b'"' | b'|' | b'?' | b'*')) {
            return Err(SandboxError::Invalid("a name contains a reserved character"));
        }
        let stem = part.split('.').next().unwrap_or(part).to_ascii_lowercase();
        if DEVICE_NAMES.contains(&stem.as_str()) {
            return Err(SandboxError::Invalid("device names are not allowed"));
        }
        out.push(part);
    }
    if out.is_empty() {
        return Err(SandboxError::Invalid("the path names no file"));
    }
    Ok(out)
}

/// The directories agent paths may name `[SEC-02]` (see the module documentation).
#[derive(Clone, Debug)]
pub struct Sandbox {
    exchange: PathBuf,
    project: Option<PathBuf>,
}

impl Sandbox {
    /// A sandbox over `exchange` (created when missing) and the project directory, if any. Both
    /// are canonicalised here.
    ///
    /// # Errors
    /// [`SandboxError::Io`] when a directory cannot be created or resolved.
    pub fn new(exchange: &Path, project: Option<&Path>) -> Result<Self, SandboxError> {
        std::fs::create_dir_all(exchange).map_err(SandboxError::Io)?;
        let exchange = std::fs::canonicalize(exchange).map_err(SandboxError::Io)?;
        let project = project.map(std::fs::canonicalize).transpose().map_err(SandboxError::Io)?;
        Ok(Self { exchange, project })
    }

    /// The exchange directory (canonical).
    pub fn exchange(&self) -> &Path {
        &self.exchange
    }

    /// The project directory (canonical), if any.
    pub fn project(&self) -> Option<&Path> {
        self.project.as_deref()
    }

    fn roots(&self) -> impl Iterator<Item = &Path> {
        std::iter::once(self.exchange.as_path()).chain(self.project.as_deref())
    }

    fn inside(&self, canonical: &Path) -> bool {
        self.roots().any(|r| canonical.starts_with(r))
    }

    /// The root and relative components of `text`: `project:rel` names the project, an absolute
    /// path is taken as is, anything else is relative to the exchange directory.
    fn split<'t>(&self, text: &'t str) -> Result<(PathBuf, Vec<&'t str>), SandboxError> {
        if text.is_empty() || text.len() > MAX_PATH_BYTES {
            return Err(SandboxError::Invalid("empty or longer than 1024 bytes"));
        }
        if text.bytes().any(|b| b < 0x20) {
            return Err(SandboxError::Invalid("control characters"));
        }
        if text.starts_with(r"\\") || text.starts_with("//") {
            return Err(SandboxError::Invalid("UNC and device paths are not allowed"));
        }
        if let Some(rest) = text.strip_prefix("project:") {
            let root = self.project.clone().ok_or(SandboxError::Outside)?;
            return Ok((root, components(rest)?));
        }
        if let Some(rest) = text.strip_prefix("exchange:") {
            return Ok((self.exchange.clone(), components(rest)?));
        }
        let path = Path::new(text);
        let rooted = path.components().any(|c| matches!(c, Component::Prefix(_) | Component::RootDir));
        if rooted && !path.is_absolute() {
            return Err(SandboxError::Invalid("drive-relative or root-relative paths are not allowed"));
        }
        if path.is_absolute() {
            // An absolute path must canonicalise into a root; `..` is refused lexically anyway.
            let lexical: Vec<&str> = text.split(['/', '\\']).collect();
            if lexical.contains(&"..") {
                return Err(SandboxError::Invalid("'..' is not allowed"));
            }
            return Ok((PathBuf::from(text), Vec::new()));
        }
        Ok((self.exchange.clone(), components(text)?))
    }

    /// An existing file the agent may read.
    ///
    /// # Errors
    /// [`SandboxError`] when the path is invalid, missing or outside the roots.
    pub fn resolve_read(&self, text: &str) -> Result<PathBuf, SandboxError> {
        let (root, parts) = self.split(text)?;
        let mut path = root;
        path.extend(parts);
        let canonical = std::fs::canonicalize(&path).map_err(SandboxError::Io)?;
        if !self.inside(&canonical) {
            return Err(SandboxError::Outside);
        }
        if !canonical.is_file() {
            return Err(SandboxError::Invalid("not a file"));
        }
        Ok(canonical)
    }

    /// A file the agent may write (its directory is created inside the root when missing).
    ///
    /// # Errors
    /// [`SandboxError`] when the path is invalid or resolves outside the roots.
    pub fn resolve_write(&self, text: &str) -> Result<PathBuf, SandboxError> {
        let (root, parts) = self.split(text)?;
        let mut path = root;
        path.extend(&parts);
        let name = path.file_name().ok_or(SandboxError::Invalid("the path names no file"))?.to_owned();
        let parent = path.parent().ok_or(SandboxError::Invalid("the path has no directory"))?.to_path_buf();
        if !parts.is_empty() {
            // Relative: the lexical path is inside its root; create the directories, then check
            // the real location (a symbolic link on the way could point elsewhere).
            std::fs::create_dir_all(&parent).map_err(SandboxError::Io)?;
        }
        let parent = std::fs::canonicalize(&parent).map_err(SandboxError::Io)?;
        if !self.inside(&parent) {
            return Err(SandboxError::Outside);
        }
        let target = parent.join(name);
        if target.exists() {
            let canonical = std::fs::canonicalize(&target).map_err(SandboxError::Io)?;
            if !self.inside(&canonical) || canonical.is_dir() {
                return Err(SandboxError::Outside);
            }
        }
        Ok(target)
    }

    /// How a path inside the roots is shown to the agent: `exchange:<rel>` or `project:<rel>`
    /// with forward slashes, or the absolute path when it is outside (never for paths this
    /// sandbox resolved).
    pub fn display(&self, path: &Path) -> String {
        let rel = |root: &Path, prefix: &str| {
            path.strip_prefix(root).ok().map(|r| format!("{prefix}{}", r.to_string_lossy().replace('\\', "/")))
        };
        rel(&self.exchange, "exchange:")
            .or_else(|| self.project.as_deref().and_then(|p| rel(p, "project:")))
            .unwrap_or_else(|| path.display().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sec_01_token_is_random_and_checked_exactly() {
        struct Fixed(u8);
        impl Entropy for Fixed {
            fn fill(&self, out: &mut [u8]) -> Result<(), PlatformError> {
                out.iter_mut().enumerate().for_each(|(i, b)| *b = self.0.wrapping_add(i as u8));
                Ok(())
            }
        }
        let t = Token::generate(&Fixed(1)).unwrap();
        assert_eq!(t.as_str().len(), 64);
        assert_ne!(t, Token::generate(&Fixed(2)).unwrap());
        let good = format!("Bearer {}", t.as_str());
        assert!(t.matches_header(Some(&good)));
        assert!(t.matches_header(Some(&format!("bearer {}", t.as_str()))));
        for bad in [None, Some(""), Some("Bearer"), Some("Basic abc"), Some(&good[..good.len() - 1])] {
            assert!(!t.matches_header(bad), "{bad:?}");
        }
        let mut flipped = good.clone().into_bytes();
        let last = flipped.len() - 1;
        flipped[last] ^= 1;
        assert!(!t.matches_header(Some(std::str::from_utf8(&flipped).unwrap())));
        assert!(format!("{t:?}").find(t.as_str()).is_none(), "the token never appears in debug output");
        assert!(Token::from_text("short").is_none());
    }

    #[test]
    fn sec_01_host_and_origin_must_be_loopback_with_our_port() {
        for ok in ["127.0.0.1:4000", "localhost:4000", "LOCALHOST:4000", "[::1]:4000"] {
            assert!(check_host(ok, 4000), "{ok}");
        }
        for bad in
            ["127.0.0.1:4001", "127.0.0.1", "evil.example:4000", "127.0.0.2:4000", "localhost.evil:4000", "", ":4000"]
        {
            assert!(!check_host(bad, 4000), "{bad}");
        }
        assert!(check_origin(None, 4000));
        assert!(check_origin(Some("http://localhost:4000"), 4000));
        for bad in ["null", "http://evil.example:4000", "https://localhost:4000", "http://localhost:80", "file://"] {
            assert!(!check_origin(Some(bad), 4000), "{bad}");
        }
    }

    #[test]
    fn sec_01_rate_limit_empties_and_refills() {
        let r = RateLimiter::new(3, 1000);
        assert!(r.allow() && r.allow() && r.allow());
        assert!(!r.allow());
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(r.allow());
    }

    #[test]
    fn sec_02_paths_stay_inside_the_project_and_exchange() {
        let base = std::env::temp_dir().join(format!("humanart-sandbox-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let (ex, proj, outside) = (base.join("exchange"), base.join("project"), base.join("secret"));
        std::fs::create_dir_all(&proj).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("key.txt"), b"secret").unwrap();
        std::fs::write(proj.join("character.anthropos"), b"doc").unwrap();
        let sb = Sandbox::new(&ex, Some(&proj)).unwrap();
        let w = sb.resolve_write("renders/a.png").unwrap();
        assert!(w.starts_with(sb.exchange()));
        std::fs::write(&w, b"png").unwrap();
        assert_eq!(sb.resolve_read("renders/a.png").unwrap(), w);
        assert_eq!(sb.display(&w), "exchange:renders/a.png");
        assert!(sb.resolve_read("project:character.anthropos").is_ok());
        let abs = proj.join("character.anthropos");
        assert!(sb.resolve_read(&abs.to_string_lossy()).is_ok());
        for bad in [
            "../secret/key.txt",
            "renders/../../secret/key.txt",
            "project:../secret/key.txt",
            "",
            "C:relative.txt",
            "a.txt:stream",
            "NUL",
            "con.txt",
            "dir./x",
            "a\u{1}b",
            r"\\server\share\x",
            "//server/share/x",
        ] {
            assert!(sb.resolve_read(bad).is_err(), "read {bad:?}");
            assert!(sb.resolve_write(bad).is_err(), "write {bad:?}");
        }
        let secret = outside.join("key.txt");
        assert!(matches!(sb.resolve_read(&secret.to_string_lossy()), Err(SandboxError::Outside)));
        assert!(matches!(sb.resolve_write(&outside.join("new.txt").to_string_lossy()), Err(SandboxError::Outside)));
        assert!(sb.resolve_read(&"a/".repeat(600)).is_err());
        let no_project = Sandbox::new(&ex, None).unwrap();
        assert!(matches!(no_project.resolve_read("project:character.anthropos"), Err(SandboxError::Outside)));
        std::fs::remove_dir_all(&base).unwrap();
    }
}
