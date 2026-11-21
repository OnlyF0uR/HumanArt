//! `humanart-cli batch` `[AG-02]`: runs a job file of tool calls through the agent server, as an
//! agent connected over stdio would, and prints a JSON report.
//!
//! The job file is JSON:
//!
//! ```json
//! {"version": 1, "steps": [
//!   {"tool": "open_proposal", "arguments": {"name": "batch"}},
//!   {"tool": "describe_character"}
//! ]}
//! ```
//!
//! Each step is one `tools/call` (with the 2026-07-28 request metadata) under the agent's author
//! id and policy; after it the server runs pending asynchronous renders and the model is
//! re-synced. The run stops at the first failed step unless `--keep-going`. The report on
//! standard output lists every step with its `structuredContent` (or its error); the exit code is
//! 9 when a step failed. `--save` saves the project only when every step succeeded.
//!
//! The job file is untrusted input `[ROB-01]`: its size, depth, step count, tool names and
//! members are checked against the documented limits before anything runs.

use std::path::PathBuf;
use std::sync::mpsc::channel;
use std::time::{Duration, Instant};

use humanart_agent::mcp::jsonrpc::{Id, Incoming};
use humanart_agent::mcp::server::{Envelope, Inbound, Reply, Transport};
use humanart_io::json::{JsonLimits, Object, Value, parse, parse_with, to_string_pretty};

use crate::args::{CommandSpec, OptSpec, Parsed};
use crate::error::CliError;
use crate::host::{self, Base, HostOptions};

/// Largest job file, bytes.
pub const MAX_JOB_BYTES: u64 = 8 << 20;
/// Most steps in one job.
pub const MAX_STEPS: usize = 4096;
/// Longest tool name, bytes.
pub const MAX_TOOL_NAME: usize = 64;
/// The JSON limits of a job file: 64 levels, [`MAX_JOB_BYTES`], 1 MiB per string, 2^16 members
/// per container, 2^20 values.
pub const JOB_LIMITS: JsonLimits = JsonLimits {
    max_depth: 64,
    max_input_bytes: MAX_JOB_BYTES,
    max_string_bytes: 1 << 20,
    max_members: 1 << 16,
    max_values: 1 << 20,
};
/// How long one step may take before the server would drop it (it never waits: the gate is open).
const STEP_DEADLINE: Duration = Duration::from_secs(3600);

const OPTIONS: [OptSpec; 14] = [
    OptSpec { name: "job", value: Some("FILE"), default: None, help: "The job file (JSON, see the documentation)." },
    OptSpec { name: "keep-going", value: None, default: None, help: "Run the remaining steps after a failed one." },
    host::PROJECT,
    host::CREATE,
    host::DATA,
    host::EXCHANGE,
    host::AGENT_ID,
    host::AGENT_NAME,
    host::HUMAN,
    host::DIRECT_EDITS,
    host::COMMIT_WITHOUT_REVIEW,
    OptSpec {
        name: "save",
        value: None,
        default: None,
        help: "Save the project after the job when every step succeeded.",
    },
    OptSpec { name: "quiet", value: None, default: None, help: "No log lines on standard error." },
    host::JSON_ERRORS,
];

/// The `batch` command.
pub static BATCH: CommandSpec = CommandSpec {
    name: "batch",
    summary: "Runs a job file of agent tool calls and prints a JSON report.",
    options: &OPTIONS,
};

/// One step of a job.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    /// The tool's name.
    pub tool: String,
    /// Its arguments (an object).
    pub arguments: Value,
}

/// What `batch` was asked to do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchJob {
    /// The job file.
    pub file: PathBuf,
    /// Run on after a failed step.
    pub keep_going: bool,
    /// The host options.
    pub host: HostOptions,
    /// No log lines.
    pub quiet: bool,
}

impl BatchJob {
    /// The job of a parsed `batch` command line.
    ///
    /// # Errors
    /// [`CliError::Usage`] without `--job` and for bad host options.
    pub fn from_args(p: &Parsed) -> Result<Self, CliError> {
        let file = p
            .get("job")
            .ok_or_else(|| CliError::Usage(crate::args::UsageError("batch: --job is required".to_owned())))?;
        Ok(Self {
            file: PathBuf::from(file),
            keep_going: p.get("keep-going").is_some(),
            host: HostOptions::from_args(p)?,
            quiet: p.get("quiet").is_some(),
        })
    }
}

/// Fails with `what` unless `o` has only the members in `known`.
fn only(o: &Object, known: &[&str], what: &str) -> Result<(), String> {
    match o.iter().find(|(k, _)| !known.contains(k)) {
        Some((k, _)) => Err(format!("{what}: unknown member `{k}`")),
        None => Ok(()),
    }
}

/// Parses a job file (see the module documentation).
///
/// # Errors
/// A message naming what is wrong: not JSON, beyond a limit, a wrong version, an unknown member,
/// a step without a valid tool name, arguments that are not an object.
pub fn parse_job(bytes: &[u8]) -> Result<Vec<Step>, String> {
    if bytes.len() as u64 > MAX_JOB_BYTES {
        return Err(format!("the job file is larger than {MAX_JOB_BYTES} bytes"));
    }
    let root = parse_with(bytes, &JOB_LIMITS).map_err(|e| format!("not valid JSON: {e}"))?;
    let root = root.as_object().ok_or("the job is a JSON object")?;
    only(root, &["version", "steps"], "the job")?;
    match root.get("version").and_then(Value::as_u64) {
        Some(1) => {}
        _ => return Err("the job needs \"version\": 1".to_owned()),
    }
    let list = root.get("steps").and_then(Value::as_array).ok_or("the job needs a \"steps\" array")?;
    if list.len() > MAX_STEPS {
        return Err(format!("more than {MAX_STEPS} steps"));
    }
    let mut steps = Vec::with_capacity(list.len());
    for (i, step) in list.iter().enumerate() {
        let n = i + 1;
        let o = step.as_object().ok_or_else(|| format!("step {n} is not an object"))?;
        only(o, &["tool", "arguments"], &format!("step {n}"))?;
        let tool = o.get("tool").and_then(Value::as_str).ok_or_else(|| format!("step {n} has no \"tool\" name"))?;
        let valid = !tool.is_empty()
            && tool.len() <= MAX_TOOL_NAME
            && tool.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_');
        if !valid {
            return Err(format!("step {n}: `{tool}` is not a tool name (a-z, 0-9, _; at most {MAX_TOOL_NAME} bytes)"));
        }
        let arguments = match o.get("arguments") {
            None => Value::Object(Object::new()),
            Some(Value::Object(a)) => Value::Object(a.clone()),
            Some(_) => return Err(format!("step {n}: \"arguments\" is not an object")),
        };
        steps.push(Step { tool: tool.to_owned(), arguments });
    }
    Ok(steps)
}

/// The `tools/call` parameters of `step`, with the request metadata.
fn call_params(step: &Step) -> Value {
    let mut info = Object::with_capacity(2);
    info.insert("name", "humanart-cli batch");
    info.insert("version", env!("CARGO_PKG_VERSION"));
    let mut meta = Object::with_capacity(3);
    meta.insert("io.modelcontextprotocol/protocolVersion", humanart_agent::mcp::server::MODERN_VERSIONS[0]);
    meta.insert("io.modelcontextprotocol/clientInfo", Value::Object(info));
    meta.insert("io.modelcontextprotocol/clientCapabilities", Value::Object(Object::new()));
    let mut p = Object::with_capacity(3);
    p.insert("name", step.tool.as_str());
    p.insert("arguments", step.arguments.clone());
    p.insert("_meta", Value::Object(meta));
    Value::Object(p)
}

/// The report entry of one answered step: whether it succeeded, and the entry.
fn entry(index: usize, step: &Step, reply: Option<Reply>) -> (bool, Value) {
    let mut e = Object::with_capacity(4);
    e.insert("step", index + 1);
    e.insert("tool", step.tool.as_str());
    let message = match reply {
        Some(Reply::Message { body, .. }) => parse(&body).ok(),
        _ => None,
    };
    let ok = if let Some(result) = message.as_ref().and_then(|m| m.get("result")) {
        let failed = result.get("isError").and_then(Value::as_bool).unwrap_or(false);
        let content = result.get("structuredContent").unwrap_or(result).clone();
        e.insert(if failed { "error" } else { "result" }, content);
        !failed
    } else {
        let error = message.as_ref().and_then(|m| m.get("error")).cloned().unwrap_or_else(|| {
            let mut o = Object::with_capacity(1);
            o.insert("message", "the server gave no answer");
            Value::Object(o)
        });
        e.insert("error", error);
        false
    };
    e.insert("ok", ok);
    (ok, Value::Object(e))
}

/// What a finished batch reports.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    /// The JSON report.
    pub json: Value,
    /// Steps that ran.
    pub ran: usize,
    /// Steps that failed.
    pub failed: usize,
}

/// Runs `job`: reads and checks the job file, opens the host, runs the steps.
///
/// # Errors
/// [`CliError::Io`] / [`CliError::Job`] for the job file, the host's errors. A failed step is not
/// an error here (see [`Report::failed`]).
pub fn run(job: &BatchJob) -> Result<Report, CliError> {
    let meta = std::fs::metadata(&job.file).map_err(|e| CliError::io(&job.file, e))?;
    if meta.len() > MAX_JOB_BYTES {
        return Err(CliError::Job(format!("{}: larger than {MAX_JOB_BYTES} bytes", job.file.display())));
    }
    let bytes = std::fs::read(&job.file).map_err(|e| CliError::io(&job.file, e))?;
    let steps = parse_job(&bytes).map_err(|e| CliError::Job(format!("{}: {e}", job.file.display())))?;
    let o = &job.host;
    let base = Base::load(&o.data, o.project.as_deref(), o.create, &o.exchange)?;
    let (mut engine, human) = base.engine(o.human.as_deref(), o.create)?;
    let mut server = base.server(o, &mut engine, &human)?;
    let (_idle_tx, idle) = channel::<Inbound>();
    let mut entries = Vec::with_capacity(steps.len());
    let mut failed = 0;
    for (i, step) in steps.iter().enumerate() {
        let (tx, rx) = channel();
        let id = Id::Number(i64::try_from(i + 1).unwrap_or(i64::MAX));
        let incoming = Incoming::Request { id, method: "tools/call".to_owned(), params: call_params(step) };
        let envelope =
            Envelope { incoming, transport: Transport::Stdio, reply: tx, deadline: Instant::now() + STEP_DEADLINE };
        server.handle(&mut engine, &base.fs, Inbound::Message(envelope));
        server.pump(&mut engine, &base.fs, &idle);
        host::settle(&mut engine);
        let (ok, e) = entry(i, step, rx.try_recv().ok());
        if !job.quiet {
            eprintln!("{} batch: step {} {} {}", crate::PROGRAM, i + 1, step.tool, if ok { "ok" } else { "FAILED" });
        }
        entries.push(e);
        if !ok {
            failed += 1;
            if !job.keep_going {
                break;
            }
        }
    }
    let ran = entries.len();
    let mut report = Object::with_capacity(5);
    report.insert("job", job.file.display().to_string());
    report.insert("steps", Value::Array(entries));
    report.insert("ran", ran);
    report.insert("failed", failed);
    report.insert("skipped", steps.len() - ran);
    if failed == 0 && host::finish(&mut engine, o.save)? && !job.quiet {
        eprintln!("{} batch: saved the project", crate::PROGRAM);
    }
    Ok(Report { json: Value::Object(report), ran, failed })
}

/// The report's text (pretty JSON).
///
/// # Errors
/// [`CliError::Job`] when a value cannot be written (a non-finite number).
pub fn report_text(report: &Report) -> Result<String, CliError> {
    to_string_pretty(&report.json).map_err(|e| CliError::Job(format!("the report cannot be written: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rob_01_job_files_parse_strictly() {
        let steps = parse_job(
            br#"{"version":1,"steps":[{"tool":"describe_character"},{"tool":"open_proposal","arguments":{"name":"x"}}]}"#,
        )
        .unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].arguments, Value::Object(Object::new()));
        assert_eq!(steps[1].arguments.get("name").and_then(Value::as_str), Some("x"));
        assert!(parse_job(br#"{"version":1,"steps":[]}"#).unwrap().is_empty());
        for bad in [
            &b"not json"[..],
            b"[]",
            br#"{"steps":[]}"#,
            br#"{"version":2,"steps":[]}"#,
            br#"{"version":1}"#,
            br#"{"version":1,"steps":[],"extra":0}"#,
            br#"{"version":1,"steps":[3]}"#,
            br#"{"version":1,"steps":[{}]}"#,
            br#"{"version":1,"steps":[{"tool":"Bad-Name"}]}"#,
            br#"{"version":1,"steps":[{"tool":""}]}"#,
            br#"{"version":1,"steps":[{"tool":"a","arguments":[]}]}"#,
            br#"{"version":1,"steps":[{"tool":"a","why":1}]}"#,
        ] {
            assert!(parse_job(bad).is_err(), "{}", String::from_utf8_lossy(bad));
        }
        let long = format!(r#"{{"version":1,"steps":[{{"tool":"{}"}}]}}"#, "a".repeat(MAX_TOOL_NAME + 1));
        assert!(parse_job(long.as_bytes()).is_err());
        let many = format!(r#"{{"version":1,"steps":[{}]}}"#, vec![r#"{"tool":"a"}"#; MAX_STEPS + 1].join(","));
        assert!(parse_job(many.as_bytes()).unwrap_err().contains("steps"));
        let deep = format!(
            r#"{{"version":1,"steps":[{{"tool":"a","arguments":{{"x":{}1{}}}}}]}}"#,
            "[".repeat(100),
            "]".repeat(100)
        );
        assert!(parse_job(deep.as_bytes()).is_err(), "depth limit");
    }

    #[test]
    fn rob_01_job_parser_survives_mutation_fuzz() {
        let corpus = vec![
            br#"{"version":1,"steps":[{"tool":"describe_character"},{"tool":"set_parameters","arguments":{"values":{"a.b":0.5},"reason":"r"}}]}"#.to_vec(),
            br#"{"version":1,"steps":[{"tool":"render_view","arguments":{"width":32,"height":32,"camera":"front","async":true}}]}"#.to_vec(),
        ];
        humanart_core::fuzz::run("batch-job", &corpus, 2000, 0x0ba7_c400, Duration::from_secs(60), |bytes| {
            if let Ok(steps) = parse_job(bytes) {
                assert!(steps.len() <= MAX_STEPS);
                for s in &steps {
                    assert!(!s.tool.is_empty() && s.tool.len() <= MAX_TOOL_NAME);
                    assert!(s.arguments.as_object().is_some());
                    let _ = call_params(s);
                }
            }
        });
    }

    #[test]
    fn ag_02_report_entries_read_results_and_errors() {
        let step = Step { tool: "t".into(), arguments: Value::Object(Object::new()) };
        let reply = |body: &str| Some(Reply::Message { body: body.as_bytes().to_vec(), status: 200, session: None });
        let (ok, e) = entry(
            0,
            &step,
            reply(r#"{"jsonrpc":"2.0","id":1,"result":{"isError":false,"structuredContent":{"a":1}}}"#),
        );
        assert!(ok);
        assert_eq!(e.get("result").and_then(|r| r.get("a")).and_then(Value::as_u64), Some(1));
        let (ok, e) = entry(
            0,
            &step,
            reply(
                r#"{"jsonrpc":"2.0","id":1,"result":{"isError":true,"structuredContent":{"error":{"code":"policy"}}}}"#,
            ),
        );
        assert!(!ok);
        assert!(e.get("error").is_some());
        let (ok, e) = entry(0, &step, reply(r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32602,"message":"bad"}}"#));
        assert!(!ok);
        assert_eq!(e.get("error").and_then(|r| r.get("code")).and_then(Value::as_i64), Some(-32602));
        assert!(!entry(0, &step, None).0);
    }
}
