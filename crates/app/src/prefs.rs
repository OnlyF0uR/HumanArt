//! Preferences `[APP-05]`: theme, UI scale, default quality tier, memory budget, autosave interval,
//! agent policy, data paths, flash duration; the keymap lives in its own file
//! ([`humanart_ui::input::Keymap::to_json`]).
//!
//! The file is JSON (`prefs.json` in the user's settings directory):
//!
//! ```json
//! {"version": 1, "theme": "dark", "ui_scale": 1.0, "colour_neutral": false, "reduced_motion": false,
//!  "quality": "interactive", "memory_budget_mb": 2048, "autosave_seconds": 120, "flash_seconds": 2.0,
//!  "agent": {"protect_human_edits": true, "auto_accept": false},
//!  "paths": {"projects": "projects", "data": "data"}}
//! ```
//!
//! Every member is optional (a missing one keeps its default); unknown members, values outside
//! their documented range and files over [`MAX_PREFS_BYTES`] are errors `[ROB-01]`, and on error the
//! caller keeps the defaults. Auto-accept of agent proposals is off by default (`[APP-05]`).

use std::fmt;

use humanart_io::json::{self, AccessError, Node, Object, Value};

/// The largest preferences file read.
pub const MAX_PREFS_BYTES: usize = 64 * 1024;
/// The longest path accepted.
pub const MAX_PATH_CHARS: usize = 1024;
/// The UI scale range `[UI-09]`.
pub const UI_SCALE_RANGE: (f32, f32) = (0.75, 2.0);
/// The memory budget range, MiB.
pub const MEMORY_RANGE_MB: (u32, u32) = (256, 65_536);
/// The longest autosave interval, seconds (0 turns idle autosaves off).
pub const MAX_AUTOSAVE_SECONDS: u32 = 3600;
/// The longest change flash, seconds.
pub const MAX_FLASH_SECONDS: f32 = 10.0;

/// The UI theme.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThemeChoice {
    /// The dark theme (the reference application's Black scheme).
    Dark,
    /// The light theme.
    Light,
    /// The high-contrast theme `[UI-09]`.
    HighContrast,
}

impl ThemeChoice {
    /// Every theme, in menu order.
    pub const ALL: [Self; 3] = [Self::Dark, Self::Light, Self::HighContrast];

    /// The stable name in the file.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Dark => "dark",
            Self::Light => "light",
            Self::HighContrast => "high_contrast",
        }
    }

    /// The theme of a name.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.name() == name)
    }
}

/// The default quality tier of new viewports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualityChoice {
    /// Fewer shadow taps, for editing.
    Interactive,
    /// The review tier.
    Review,
    /// The still tier.
    Still,
}

impl QualityChoice {
    /// Every tier.
    pub const ALL: [Self; 3] = [Self::Interactive, Self::Review, Self::Still];

    /// The stable name in the file.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Interactive => "interactive",
            Self::Review => "review",
            Self::Still => "still",
        }
    }

    /// The tier of a name.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.name() == name)
    }
}

/// The agent policy `[APP-05]`, `[LOCK-02]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AgentPolicy {
    /// The agent may not overwrite a value a human set (human-edit protection).
    pub protect_human_edits: bool,
    /// Accept agent proposals without review (off by default).
    pub auto_accept: bool,
}

/// The preferences (see the module documentation).
#[derive(Clone, Debug, PartialEq)]
pub struct Prefs {
    /// The theme.
    pub theme: ThemeChoice,
    /// The UI scale, [`UI_SCALE_RANGE`].
    pub ui_scale: f32,
    /// The colour-neutral UI mode `[UI-08]`.
    pub colour_neutral: bool,
    /// Reduced motion `[UI-09]`.
    pub reduced_motion: bool,
    /// The default quality tier of viewports.
    pub quality: QualityChoice,
    /// The GPU memory budget, MiB.
    pub memory_budget_mb: u32,
    /// Idle autosave interval, seconds (0: off).
    pub autosave_seconds: u32,
    /// How long an agent change flashes `[APP-03]`, seconds.
    pub flash_seconds: f32,
    /// The agent policy.
    pub agent: AgentPolicy,
    /// Where projects are created and opened.
    pub projects_dir: String,
    /// The data directory (template, registry, fonts).
    pub data_dir: String,
}

impl Default for Prefs {
    fn default() -> Self {
        Self {
            theme: ThemeChoice::Dark,
            ui_scale: 1.0,
            colour_neutral: false,
            reduced_motion: false,
            quality: QualityChoice::Interactive,
            memory_budget_mb: 2048,
            autosave_seconds: 120,
            flash_seconds: 2.0,
            agent: AgentPolicy { protect_human_edits: true, auto_accept: false },
            projects_dir: "projects".to_owned(),
            data_dir: "data".to_owned(),
        }
    }
}

/// Why a preferences file was refused.
#[derive(Debug)]
pub enum PrefsError {
    /// The file is larger than [`MAX_PREFS_BYTES`].
    TooLarge {
        /// Its size.
        bytes: usize,
    },
    /// Not UTF-8 or not JSON.
    Parse(json::ParseError),
    /// A member is missing its type, unknown, or out of range.
    Access(AccessError),
    /// A value outside its documented range.
    Range {
        /// The member.
        what: &'static str,
    },
}

impl fmt::Display for PrefsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge { bytes } => write!(f, "preferences file of {bytes} bytes (limit {MAX_PREFS_BYTES})"),
            Self::Parse(e) => write!(f, "preferences: {e}"),
            Self::Access(e) => write!(f, "preferences: {e}"),
            Self::Range { what } => write!(f, "preferences: {what} is outside its range"),
        }
    }
}

impl std::error::Error for PrefsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse(e) => Some(e),
            Self::Access(e) => Some(e),
            Self::TooLarge { .. } | Self::Range { .. } => None,
        }
    }
}

impl From<AccessError> for PrefsError {
    fn from(e: AccessError) -> Self {
        Self::Access(e)
    }
}

fn path_member(node: &Node<'_, '_>, key: &str, default: &str, what: &'static str) -> Result<String, PrefsError> {
    let s: String = node.get_or(key, default.to_owned())?;
    if s.is_empty() || s.chars().count() > MAX_PATH_CHARS || s.contains('\0') {
        return Err(PrefsError::Range { what });
    }
    Ok(s)
}

impl Prefs {
    /// Reads a preferences file (see the module documentation).
    pub fn from_json(bytes: &[u8]) -> Result<Self, PrefsError> {
        if bytes.len() > MAX_PREFS_BYTES {
            return Err(PrefsError::TooLarge { bytes: bytes.len() });
        }
        let text = std::str::from_utf8(bytes).map_err(|_| PrefsError::Range { what: "the encoding (not UTF-8)" })?;
        let value = json::parse_str(text).map_err(PrefsError::Parse)?;
        let root = value.node();
        root.deny_unknown(&[
            "version",
            "theme",
            "ui_scale",
            "colour_neutral",
            "reduced_motion",
            "quality",
            "memory_budget_mb",
            "autosave_seconds",
            "flash_seconds",
            "agent",
            "paths",
        ])?;
        let d = Self::default();
        let version: u32 = root.get_or("version", 1)?;
        if version != 1 {
            return Err(PrefsError::Range { what: "version" });
        }
        let theme: String = root.get_or("theme", d.theme.name().to_owned())?;
        let quality: String = root.get_or("quality", d.quality.name().to_owned())?;
        let mut p = Self {
            theme: ThemeChoice::from_name(&theme).ok_or(PrefsError::Range { what: "theme" })?,
            ui_scale: root.get_or("ui_scale", d.ui_scale)?,
            colour_neutral: root.get_or("colour_neutral", d.colour_neutral)?,
            reduced_motion: root.get_or("reduced_motion", d.reduced_motion)?,
            quality: QualityChoice::from_name(&quality).ok_or(PrefsError::Range { what: "quality" })?,
            memory_budget_mb: root.get_or("memory_budget_mb", d.memory_budget_mb)?,
            autosave_seconds: root.get_or("autosave_seconds", d.autosave_seconds)?,
            flash_seconds: root.get_or("flash_seconds", d.flash_seconds)?,
            ..d
        };
        if let Some(agent) = root.opt_field("agent")? {
            agent.deny_unknown(&["protect_human_edits", "auto_accept"])?;
            p.agent.protect_human_edits = agent.get_or("protect_human_edits", p.agent.protect_human_edits)?;
            p.agent.auto_accept = agent.get_or("auto_accept", p.agent.auto_accept)?;
        }
        if let Some(paths) = root.opt_field("paths")? {
            paths.deny_unknown(&["projects", "data"])?;
            p.projects_dir = path_member(&paths, "projects", &p.projects_dir, "paths.projects")?;
            p.data_dir = path_member(&paths, "data", &p.data_dir, "paths.data")?;
        }
        p.validate()?;
        Ok(p)
    }

    /// Checks every range.
    pub fn validate(&self) -> Result<(), PrefsError> {
        if !(UI_SCALE_RANGE.0..=UI_SCALE_RANGE.1).contains(&self.ui_scale) {
            return Err(PrefsError::Range { what: "ui_scale" });
        }
        if !(MEMORY_RANGE_MB.0..=MEMORY_RANGE_MB.1).contains(&self.memory_budget_mb) {
            return Err(PrefsError::Range { what: "memory_budget_mb" });
        }
        if self.autosave_seconds > MAX_AUTOSAVE_SECONDS {
            return Err(PrefsError::Range { what: "autosave_seconds" });
        }
        if !(0.0..=MAX_FLASH_SECONDS).contains(&self.flash_seconds) {
            return Err(PrefsError::Range { what: "flash_seconds" });
        }
        Ok(())
    }

    /// The JSON form (see the module documentation).
    pub fn to_json(&self) -> String {
        let mut agent = Object::with_capacity(2);
        agent.insert("protect_human_edits", self.agent.protect_human_edits);
        agent.insert("auto_accept", self.agent.auto_accept);
        let mut paths = Object::with_capacity(2);
        paths.insert("projects", self.projects_dir.as_str());
        paths.insert("data", self.data_dir.as_str());
        let mut o = Object::with_capacity(11);
        o.insert("version", 1_u32);
        o.insert("theme", self.theme.name());
        o.insert("ui_scale", f64::from(self.ui_scale));
        o.insert("colour_neutral", self.colour_neutral);
        o.insert("reduced_motion", self.reduced_motion);
        o.insert("quality", self.quality.name());
        o.insert("memory_budget_mb", self.memory_budget_mb);
        o.insert("autosave_seconds", self.autosave_seconds);
        o.insert("flash_seconds", f64::from(self.flash_seconds));
        o.insert("agent", agent);
        o.insert("paths", paths);
        // A tree of finite numbers, strings and booleans always writes.
        json::to_string_pretty(&Value::from(o)).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_05_prefs_round_trip_and_defaults() {
        let p = Prefs {
            theme: ThemeChoice::Light,
            ui_scale: 1.5,
            agent: AgentPolicy { protect_human_edits: false, auto_accept: true },
            projects_dir: "C:/work/heads".into(),
            ..Prefs::default()
        };
        let text = p.to_json();
        assert_eq!(Prefs::from_json(text.as_bytes()).unwrap(), p);
        assert_eq!(Prefs::from_json(b"{}").unwrap(), Prefs::default());
        assert!(!Prefs::default().agent.auto_accept, "[APP-05] auto-accept is off by default");
    }

    #[test]
    fn app_05_prefs_refuse_bad_values() {
        for bad in [
            &br#"{"ui_scale": 3.0}"#[..],
            br#"{"theme": "pink"}"#,
            br#"{"unknown": 1}"#,
            br#"{"memory_budget_mb": 1}"#,
            br#"{"flash_seconds": -1}"#,
            br#"{"paths": {"data": ""}}"#,
            br#"{"version": 2}"#,
            b"\xff\xfe",
            b"[1, 2",
        ] {
            assert!(Prefs::from_json(bad).is_err(), "{}", String::from_utf8_lossy(bad));
        }
        assert!(matches!(Prefs::from_json(&vec![b' '; MAX_PREFS_BYTES + 1]), Err(PrefsError::TooLarge { .. })));
    }

    #[test]
    fn rob_01_prefs_parser_survives_mutations() {
        let corpus = vec![Prefs::default().to_json().into_bytes(), br#"{"agent": {"auto_accept": true}}"#.to_vec()];
        humanart_core::fuzz::run("prefs", &corpus, 3000, 0x0A05, std::time::Duration::from_secs(60), |bytes| {
            let _ = Prefs::from_json(bytes);
        });
    }
}
