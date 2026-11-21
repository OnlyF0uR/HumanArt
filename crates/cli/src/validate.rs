//! `humanart-cli validate`: checks the data folder and, with `--project`, a project, without
//! writing anything.
//!
//! - **registry**: every `template/params/*.json` file reads and the registry builds;
//! - **template**: the package (or the generated template) builds, and how much of the registry
//!   drives it (`Model::coverage`; parameters naming a component the template does not have are
//!   a warning);
//! - **project**: the document reads, migrates and checks against the registry (an older format
//!   version is a warning); every blob it references is in the store; the journal is read (not
//!   repaired): unsaved records, a stale or a torn journal are warnings, an unreadable one an
//!   error; every proposal file reads (a damaged one is an error).
//!
//! Each finding is one line `ok|warning|error <area>: <text>` (or, with `--json`, one JSON
//! report). The exit code is 9 when there is an error, 0 otherwise.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use humanart_agent::engine::Model;
use humanart_core::platform::FileSystem as _;
use humanart_doc::journal::JournalContents;
use humanart_doc::project::Project;
use humanart_doc::proposal::{PROPOSALS_DIR, ProposalStore};
use humanart_doc::session::SessionConfig;
use humanart_io::json::{Object, Value, to_string_pretty};
use humanart_platform::StdFileSystem;

use crate::args::{CommandSpec, OptSpec, Parsed};
use crate::error::CliError;
use crate::host;

const OPTIONS: [OptSpec; 4] = [
    host::DATA,
    OptSpec { name: "project", value: Some("DIR"), default: None, help: "A project folder to check as well." },
    OptSpec { name: "json", value: None, default: None, help: "Print the report as JSON." },
    host::JSON_ERRORS,
];

/// The `validate` command.
pub static VALIDATE: CommandSpec = CommandSpec {
    name: "validate",
    summary: "Checks the data folder and a project without changing them.",
    options: &OPTIONS,
};

/// How bad a finding is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// All is well.
    Ok,
    /// Worth knowing; the data works.
    Warning,
    /// Broken.
    Error,
}

impl Level {
    /// The level's word in the report.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }
}

/// One finding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Finding {
    /// How bad.
    pub level: Level,
    /// `registry`, `template`, `document`, `blobs`, `journal` or `proposals`.
    pub area: &'static str,
    /// What was found.
    pub text: String,
}

/// What `validate` was asked to do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidateJob {
    /// The data folder.
    pub data: PathBuf,
    /// The project folder.
    pub project: Option<PathBuf>,
    /// JSON output.
    pub json: bool,
}

impl ValidateJob {
    /// The job of a parsed `validate` command line.
    pub fn from_args(p: &Parsed) -> Self {
        Self {
            data: PathBuf::from(p.get("data").unwrap_or("data")),
            project: p.get("project").map(PathBuf::from),
            json: p.get("json").is_some(),
        }
    }
}

fn found(out: &mut Vec<Finding>, level: Level, area: &'static str, text: String) {
    out.push(Finding { level, area, text });
}

/// Runs `job` and returns its findings. Never fails: what cannot be read is an error finding.
pub fn run(job: &ValidateJob) -> Vec<Finding> {
    let mut out = Vec::new();
    let registry = match crate::render::load_registry(&job.data) {
        Ok(r) => {
            found(&mut out, Level::Ok, "registry", format!("{} parameters", r.params().len()));
            r
        }
        Err(e) => {
            found(&mut out, Level::Error, "registry", e.to_string());
            return out;
        }
    };
    match crate::render::load_template(&job.data) {
        Ok((template, source)) => {
            let c = Model::new(Arc::new(template), &registry).coverage();
            let text =
                format!("{source}: {} shape, {} expression, {} other parameters bound", c.shape, c.expression, c.other);
            found(&mut out, Level::Ok, "template", text);
            if c.missing > 0 {
                let text = format!("{} parameters name a component or expression the template lacks", c.missing);
                found(&mut out, Level::Warning, "template", text);
            }
        }
        Err(e) => found(&mut out, Level::Error, "template", e.to_string()),
    }
    if let Some(dir) = &job.project {
        check_project(dir, &registry, &mut out);
    }
    out
}

/// The project checks (see the module documentation).
fn check_project(dir: &Path, registry: &humanart_doc::registry::Registry, out: &mut Vec<Finding>) {
    let (parent, root) = match host::split_project(dir) {
        Ok(split) => split,
        Err(e) => return found(out, Level::Error, "document", e.to_string()),
    };
    let fs = match StdFileSystem::rooted(&parent) {
        Ok(fs) => fs,
        Err(e) => return found(out, Level::Error, "document", format!("{}: {e}", parent.display())),
    };
    let doc = match host::read_document(&fs, &root, registry) {
        Ok((doc, migrated)) => {
            found(out, Level::Ok, "document", format!("`{}`, {} authors", doc.meta.name, doc.authors.len()));
            if let Some(v) = migrated {
                found(out, Level::Warning, "document", format!("format version {v}: it is migrated when opened"));
            }
            doc
        }
        Err(e) => return found(out, Level::Error, "document", format!("{}: {e}", dir.display())),
    };
    let project = Project::at(&root);
    let store = project.blobs(&fs);
    let refs = doc.blob_refs();
    let missing: Vec<String> =
        refs.iter().filter(|h| !store.contains(**h)).map(|h| h.to_hex().as_str().to_owned()).collect();
    if missing.is_empty() {
        found(out, Level::Ok, "blobs", format!("{} referenced, all present", refs.len()));
    } else {
        found(
            out,
            Level::Error,
            "blobs",
            format!("{} of {} missing: {}", missing.len(), refs.len(), missing.join(", ")),
        );
    }
    let config = SessionConfig::default();
    match project.journal(&fs).read(&config.load.journal) {
        Ok(None) => found(out, Level::Warning, "journal", "no journal (the next save starts one)".to_owned()),
        Ok(Some(JournalContents { header, records, torn_tail, .. })) => {
            let saved = fs.read(&project.document_path()).map(|b| humanart_core::hash::hash128(&b)).ok();
            if header.document != doc.meta.id || Some(header.base) != saved {
                found(out, Level::Warning, "journal", "stale: it belongs to another save and is ignored".to_owned());
            } else if records.is_empty() {
                found(out, Level::Ok, "journal", "clean".to_owned());
            } else {
                let text = format!("{} unsaved changes: opening the project recovers them", records.len());
                found(out, Level::Warning, "journal", text);
            }
            if let Some(at) = torn_tail {
                let text = format!("torn last record at byte {at}: opening the project cuts it off");
                found(out, Level::Warning, "journal", text);
            }
        }
        Err(e) => found(out, Level::Error, "journal", e.to_string()),
    }
    let dir = humanart_core::platform::join_path(&root, PROPOSALS_DIR);
    match ProposalStore::new(&fs, &dir).load_all(registry, &config.proposals) {
        Ok(loaded) => {
            found(out, Level::Ok, "proposals", format!("{} read", loaded.proposals.len()));
            for (name, e) in loaded.damaged {
                found(out, Level::Error, "proposals", format!("{name}: {e}"));
            }
        }
        Err(e) => found(out, Level::Error, "proposals", e.to_string()),
    }
}

/// The report's text: one line per finding and a summary, or the JSON report.
///
/// # Errors
/// [`CliError::Data`] when the JSON cannot be written.
pub fn report(findings: &[Finding], json: bool) -> Result<String, CliError> {
    let errors = findings.iter().filter(|f| f.level == Level::Error).count();
    let warnings = findings.iter().filter(|f| f.level == Level::Warning).count();
    if json {
        let list = findings
            .iter()
            .map(|f| {
                let mut o = Object::with_capacity(3);
                o.insert("level", f.level.name());
                o.insert("area", f.area);
                o.insert("text", f.text.as_str());
                Value::Object(o)
            })
            .collect();
        let mut o = Object::with_capacity(3);
        o.insert("findings", Value::Array(list));
        o.insert("errors", errors);
        o.insert("warnings", warnings);
        return to_string_pretty(&Value::Object(o)).map_err(|e| CliError::Data(format!("the report: {e}")));
    }
    let mut text = String::new();
    for f in findings {
        let _ = writeln!(text, "{:<7} {}: {}", f.level.name(), f.area, f.text);
    }
    let _ = writeln!(text, "{errors} errors, {warnings} warnings");
    Ok(text)
}

/// The command's outcome: [`CliError::Failed`] when a finding is an error.
///
/// # Errors
/// See above.
pub fn verdict(findings: &[Finding]) -> Result<(), CliError> {
    let errors = findings.iter().filter(|f| f.level == Level::Error).count();
    if errors == 0 { Ok(()) } else { Err(CliError::Failed(format!("validation found {errors} errors"))) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finding(level: Level) -> Finding {
        Finding { level, area: "journal", text: "t \"q\"".into() }
    }

    #[test]
    fn validate_reports_and_verdicts() {
        let fine = [finding(Level::Ok), finding(Level::Warning)];
        assert!(verdict(&fine).is_ok());
        assert!(matches!(verdict(&[finding(Level::Error)]), Err(CliError::Failed(_))));
        let text = report(&fine, false).unwrap();
        assert!(text.starts_with("ok      journal: t") && text.ends_with("0 errors, 1 warnings\n"), "{text}");
        let json = humanart_io::json::parse_str(&report(&[finding(Level::Error)], true).unwrap()).unwrap();
        assert_eq!(json.get("errors").and_then(Value::as_u64), Some(1));
        assert_eq!(
            json.get("findings")
                .and_then(Value::as_array)
                .and_then(|l| l.first())
                .and_then(|f| f.get("text"))
                .and_then(Value::as_str),
            Some("t \"q\"")
        );
    }

    #[test]
    fn validate_reports_a_missing_data_folder_and_project_as_errors() {
        let job = ValidateJob { data: PathBuf::from("no-such-data-folder"), project: None, json: false };
        let f = run(&job);
        assert_eq!(f.len(), 1);
        assert_eq!((f[0].level, f[0].area), (Level::Error, "registry"));
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut out = Vec::new();
        let registry = crate::render::load_registry(&root.join("data")).unwrap();
        check_project(&std::env::temp_dir().join("humanart-no-such-project-folder"), &registry, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].level, out[0].area), (Level::Error, "document"));
    }
}
