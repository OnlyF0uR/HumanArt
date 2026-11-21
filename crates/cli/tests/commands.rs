//! The built `humanart-cli` binary, driven as a user or an agent drives it: help and argument
//! errors with their exit codes `[CLI-02]`, `[CLI-03]`; `serve` over stdio and over loopback HTTP
//! with its token `[AG-01]`, `[SEC-01]`; `batch` and `validate` on a project.
#![forbid(unsafe_code)]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{BufRead, BufReader};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

use humanart_agent::mcp::client::{HttpClient, StdioClient};
use humanart_core::platform::join_path;
use humanart_doc::registry::{ParamKind, ParamValue, Registry, RegistryLimits};
use humanart_io::json::{Object, Value, parse_str};
use humanart_platform::StdFileSystem;

const BIN: &str = env!("CARGO_BIN_EXE_humanart-cli");
const LONG: Duration = Duration::from_secs(300);

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

/// A fresh scratch folder for one test.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("humanart-cli-it-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn s(p: &Path) -> String {
    p.display().to_string()
}

/// Runs the binary to its end with no standard input.
fn cli(args: &[&str]) -> Output {
    Command::new(BIN).args(args).stdin(Stdio::null()).output().unwrap()
}

fn code(out: &Output) -> i32 {
    out.status.code().unwrap_or(-1)
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Waits for `child` to exit (killing it after [`LONG`]) and returns its exit code.
fn wait(child: &mut Child) -> i32 {
    let end = Instant::now() + LONG;
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status.code().unwrap_or(-1);
        }
        if Instant::now() > end {
            let _ = child.kill();
            panic!("humanart-cli did not stop");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// An editable scalar parameter of the registry and a value inside its range.
fn scalar_param() -> (String, f64) {
    let fs = StdFileSystem::host();
    let registry =
        Registry::load_dir(&fs, &join_path(&s(&data()), "template/params"), &RegistryLimits::DEFAULT).unwrap();
    let p = registry
        .params()
        .iter()
        .find(|p| p.kind == ParamKind::Scalar && !p.read_only && !p.locked_by_default && p.range.is_some())
        .unwrap();
    let r = p.range.unwrap();
    let v = r.soft_min + (r.soft_max - r.soft_min) * 0.375;
    let v = if p.is_default(&ParamValue::Scalar(v)) { r.soft_min + (r.soft_max - r.soft_min) * 0.625 } else { v };
    (p.id.clone(), v)
}

fn args_of(pairs: &[(&str, Value)]) -> Value {
    let mut o = Object::new();
    for (k, v) in pairs {
        o.insert(*k, v.clone());
    }
    Value::Object(o)
}

fn is_ok(result: &Value) -> bool {
    result.get("isError") == Some(&Value::Bool(false))
}

#[test]
fn cli_03_help_usage_errors_and_exit_codes() {
    let help = cli(&["--help"]);
    assert_eq!(code(&help), 0);
    let usage = text(&help.stdout);
    for command in ["render", "turntable", "serve", "batch", "validate", "bench", "fontbake"] {
        assert!(usage.contains(&format!("  {command}  ")), "{usage}");
    }
    let serve_help = cli(&["serve", "--help"]);
    assert_eq!(code(&serve_help), 0);
    for option in ["--transport", "--port", "--project", "--direct-edits", "--save"] {
        assert!(text(&serve_help.stdout).contains(option), "{option}");
    }
    for command in ["batch", "turntable", "validate", "bench", "fontbake"] {
        assert_eq!(code(&cli(&[command, "--help"])), 0, "{command} --help");
    }
    // Usage errors exit 2, before any data is read.
    assert_eq!(code(&cli(&["no-such-command"])), 2);
    assert_eq!(code(&cli(&["batch"])), 2, "--job is required");
    assert_eq!(code(&cli(&["serve", "--port", "8080"])), 2, "--port belongs to http");
    assert_eq!(code(&cli(&["turntable", "--views", "0"])), 2);
    assert_eq!(code(&cli(&["bench"])), 2, "--scenario is required");
    let unwritten = cli(&["bench", "--scenario", "3"]);
    assert_eq!(code(&unwritten), 2, "only scenario 10 exists");
    assert!(text(&unwritten.stderr).contains("scenario '3' is not written"), "{}", text(&unwritten.stderr));
    assert_eq!(code(&cli(&["fontbake", "--font", "a.ttf"])), 2, "--font needs --out");
    let json = cli(&["serve", "--transport", "tcp", "--json-errors"]);
    assert_eq!(code(&json), 2);
    let line = parse_str(text(&json.stderr).trim()).unwrap();
    let error = line.get("error").unwrap();
    assert_eq!(error.get("class").and_then(Value::as_str), Some("usage"));
    assert_eq!(error.get("code").and_then(Value::as_u64), Some(2));
    // A job file that is missing (6) or malformed (8) is refused before the data is loaded.
    let dir = scratch("usage");
    assert_eq!(code(&cli(&["batch", "--job", &s(&dir.join("missing.json"))])), 6);
    let bad = dir.join("bad.json");
    std::fs::write(&bad, br#"{"version":1,"steps":[{"tool":"Not A Tool"}]}"#).unwrap();
    assert_eq!(code(&cli(&["batch", "--job", &s(&bad)])), 8);
    // A data folder that is not there: validate reports it (9), serve fails on it (6).
    assert_eq!(code(&cli(&["validate", "--data", &s(&dir.join("nowhere"))])), 9);
    assert_eq!(code(&cli(&["serve", "--data", &s(&dir.join("nowhere"))])), 6);
    // Fonts that are not there: both font commands fail on the file (6) before any GPU work, and
    // fontbake writes nothing.
    assert_eq!(code(&cli(&["fontbake", "--dir", &s(&dir.join("nowhere"))])), 6);
    assert_eq!(code(&cli(&["bench", "--scenario", "10", "--data", &s(&dir.join("nowhere"))])), 6);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn ag_01_serve_over_stdio_edits_and_saves_the_project() {
    let dir = scratch("stdio");
    let project = dir.join("alice");
    let (id, value) = scalar_param();
    let mut child = Command::new(BIN)
        .args([
            "serve",
            "--data",
            &s(&data()),
            "--project",
            &s(&project),
            "--create",
            "--exchange",
            &s(&dir.join("exchange")),
            "--direct-edits",
            "--save",
            "--quiet",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let mut client = StdioClient::new(child.stdout.take().unwrap(), child.stdin.take().unwrap()).unwrap();
    let discover = client.request_within("server/discover", Value::Null, LONG).unwrap();
    assert!(discover.get("supportedVersions").is_some(), "{discover:?}");
    let tools = client.request_within("tools/list", Value::Null, LONG).unwrap();
    assert!(tools.get("tools").and_then(Value::as_array).is_some_and(|t| t.len() > 10));
    assert!(is_ok(&client.call_tool("describe_character", args_of(&[])).unwrap()));
    let mut values = Object::new();
    values.insert(id.as_str(), value);
    let set = client
        .call_tool("set_parameters", args_of(&[("values", Value::Object(values)), ("reason", "stdio test".into())]))
        .unwrap();
    assert!(is_ok(&set), "{set:?}");
    let unknown = client.call_tool("no_such_tool", args_of(&[]));
    assert!(unknown.is_err() || unknown.is_ok_and(|r| !is_ok(&r)), "an unknown tool is refused");
    // Closing standard input ends the server; --save writes the project.
    drop(client);
    assert_eq!(wait(&mut child), 0);
    let saved = std::fs::read_to_string(project.join("character.anthropos")).unwrap();
    assert!(saved.contains(&id), "the agent's edit is in the saved document");
    let check = cli(&["validate", "--data", &s(&data()), "--project", &s(&project)]);
    assert_eq!(code(&check), 0, "{}", text(&check.stdout));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn ag_01_sec_01_serve_over_http_with_its_token() {
    let dir = scratch("http");
    let mut child = Command::new(BIN)
        .args([
            "serve",
            "--transport",
            "http",
            "--data",
            &s(&data()),
            "--exchange",
            &s(&dir.join("exchange")),
            "--quiet",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let stdin = child.stdin.take().unwrap();
    let mut first = String::new();
    BufReader::new(child.stdout.take().unwrap()).read_line(&mut first).unwrap();
    let hello = parse_str(first.trim()).unwrap();
    let url = hello.get("url").and_then(Value::as_str).unwrap().to_owned();
    let token = hello.get("token").and_then(Value::as_str).unwrap().to_owned();
    assert_eq!(token.len(), 64, "a 256-bit token in hex");
    let port: u16 =
        url.strip_prefix("http://127.0.0.1:").and_then(|r| r.strip_suffix("/mcp")).unwrap().parse().unwrap();
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let mut client = HttpClient::new(addr, &token);
    let discover = client.request("server/discover", Value::Null).unwrap();
    assert!(discover.get("supportedVersions").is_some());
    assert!(is_ok(&client.call_tool("describe_character", args_of(&[])).unwrap()));
    // Without the token, or with another one: 401.
    let body = br#"{"jsonrpc":"2.0","id":1,"method":"server/discover","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}}}}"#;
    let host = format!("127.0.0.1:{port}");
    for bearer in ["", "Bearer wrong-token-0123456789abcdef"] {
        let mut headers =
            vec![("Host", host.as_str()), ("Content-Type", "application/json"), ("Accept", "application/json")];
        if !bearer.is_empty() {
            headers.push(("Authorization", bearer));
        }
        assert_eq!(client.raw("POST", &headers, body).unwrap().status, 401, "{bearer}");
    }
    drop(stdin);
    assert_eq!(wait(&mut child), 0);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn ag_02_batch_runs_jobs_and_validate_checks_the_project() {
    let dir = scratch("batch");
    let project = dir.join("bob");
    let exchange = dir.join("exchange");
    let (id, value) = scalar_param();
    let job = dir.join("job.json");
    let steps = format!(
        r#"{{"version":1,"steps":[{{"tool":"describe_character"}},{{"tool":"set_parameters","arguments":{{"values":{{"{id}":{value}}},"reason":"batch"}}}},{{"tool":"get_events","arguments":{{"since":0}}}}]}}"#
    );
    std::fs::write(&job, steps).unwrap();
    let host = |extra: &[&str]| -> Vec<String> {
        let mut a: Vec<String> =
            ["--data", &s(&data()), "--project", &s(&project), "--exchange", &s(&exchange), "--quiet"]
                .iter()
                .map(|x| (*x).to_owned())
                .collect();
        a.extend(extra.iter().map(|x| (*x).to_owned()));
        a
    };
    let run = |command: &str, extra: &[&str]| {
        let mut a = vec![command.to_owned()];
        a.extend(host(extra));
        Command::new(BIN).args(&a).stdin(Stdio::null()).output().unwrap()
    };
    // No project there and no --create: exit 7.
    assert_eq!(code(&run("batch", &["--job", &s(&job)])), 7);
    let out = run("batch", &["--job", &s(&job), "--create", "--direct-edits", "--save"]);
    assert_eq!(code(&out), 0, "{}", text(&out.stderr));
    let report = parse_str(&text(&out.stdout)).unwrap();
    assert_eq!(report.get("ran").and_then(Value::as_u64), Some(3));
    assert_eq!(report.get("failed").and_then(Value::as_u64), Some(0));
    assert!(std::fs::read_to_string(project.join("character.anthropos")).unwrap().contains(&id));
    let check = cli(&["validate", "--data", &s(&data()), "--project", &s(&project), "--json"]);
    assert_eq!(code(&check), 0, "{}", text(&check.stdout));
    assert_eq!(parse_str(&text(&check.stdout)).unwrap().get("errors").and_then(Value::as_u64), Some(0));
    // A failing step stops the job (exit 9); the report says what ran.
    let failing = dir.join("failing.json");
    std::fs::write(
        &failing,
        br#"{"version":1,"steps":[{"tool":"describe_character"},{"tool":"no_such_tool"},{"tool":"describe_character"}]}"#,
    )
    .unwrap();
    let out = run("batch", &["--job", &s(&failing)]);
    assert_eq!(code(&out), 9);
    let report = parse_str(&text(&out.stdout)).unwrap();
    let counts = ["ran", "failed", "skipped"].map(|k| report.get(k).and_then(Value::as_u64));
    assert_eq!(counts, [Some(2), Some(1), Some(1)]);
    // Without --direct-edits the default policy refuses a direct edit: the step fails.
    let out = run("batch", &["--job", &s(&job)]);
    assert_eq!(code(&out), 9);
    // A damaged document: validate finds it (9), serve refuses the project (7).
    std::fs::write(project.join("character.anthropos"), b"not a document").unwrap();
    let check = cli(&["validate", "--data", &s(&data()), "--project", &s(&project)]);
    assert_eq!(code(&check), 9);
    assert!(text(&check.stdout).contains("error   document"), "{}", text(&check.stdout));
    assert_eq!(code(&run("serve", &[])), 7);
    std::fs::remove_dir_all(&dir).unwrap();
}
