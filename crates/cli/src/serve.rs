//! `humanart-cli serve` `[AG-01]`: the agent server of `humanart-agent` on the character, without
//! the editor.
//!
//! - `--transport stdio` (default): newline-delimited JSON-RPC on standard input and output;
//!   standard output carries nothing but protocol messages (logs go to standard error). The
//!   server runs until standard input closes.
//! - `--transport http`: Streamable HTTP on `127.0.0.1` only (`--port`, 0: any free port) behind a
//!   fresh 256-bit bearer token `[SEC-01]`. The first line on standard output is
//!   `{"url":"http://127.0.0.1:PORT/mcp","token":"..."}`; the server runs until standard input
//!   closes, or for `--seconds` when given (standard input is then not read).
//!
//! In both cases the human (`--human`, or the document's first human author) admits the agent
//! into the document first, as the editor does when the human grants the link. With `--save` the
//! project is saved at the end; otherwise its journal holds the edits and the editor recovers them.

use std::io::Write as _;
use std::sync::mpsc::{TryRecvError, channel};
use std::time::{Duration, Instant};

use humanart_agent::mcp::jsonrpc::RpcLimits;
use humanart_agent::mcp::security::Token;
use humanart_agent::mcp::transport::{HttpConfig, HttpServer, spawn_stdio};
use humanart_platform::OsEntropy;

use crate::PROGRAM;
use crate::args::{CommandSpec, OptSpec, Parsed};
use crate::error::CliError;
use crate::host::{self, Base, HostOptions};

/// How long one wait for a message lasts before the loop checks its stop conditions.
const WAIT: Duration = Duration::from_millis(50);
/// Most `--seconds`.
pub const MAX_SECONDS: u32 = 7 * 24 * 3600;

const OPTIONS: [OptSpec; 15] = [
    OptSpec { name: "transport", value: Some("KIND"), default: Some("stdio"), help: "stdio or http." },
    OptSpec { name: "port", value: Some("N"), default: Some("0"), help: "HTTP port on 127.0.0.1 (0: any free port)." },
    OptSpec {
        name: "seconds",
        value: Some("N"),
        default: Some("0"),
        help: "HTTP: stop after N seconds (0: when standard input closes).",
    },
    host::PROJECT,
    host::CREATE,
    host::DATA,
    host::EXCHANGE,
    host::AGENT_ID,
    host::AGENT_NAME,
    host::HUMAN,
    host::DIRECT_EDITS,
    host::COMMIT_WITHOUT_REVIEW,
    host::SAVE,
    OptSpec { name: "quiet", value: None, default: None, help: "No log lines on standard error." },
    host::JSON_ERRORS,
];

/// The `serve` command.
pub static SERVE: CommandSpec =
    CommandSpec { name: "serve", summary: "Runs the agent server (MCP) on stdio or loopback HTTP.", options: &OPTIONS };

/// The transport `serve` was asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Standard input and output.
    Stdio,
    /// Loopback HTTP on this port (0: any).
    Http {
        /// The port.
        port: u16,
        /// Stop after this many seconds (0: when standard input closes).
        seconds: u32,
    },
}

/// What `serve` was asked to do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServeJob {
    /// The transport.
    pub kind: Kind,
    /// The host options.
    pub host: HostOptions,
    /// No log lines.
    pub quiet: bool,
}

impl ServeJob {
    /// The job of a parsed `serve` command line.
    ///
    /// # Errors
    /// [`CliError::Usage`] for values out of range or not among the choices.
    pub fn from_args(p: &Parsed) -> Result<Self, CliError> {
        let kind = match p.choice("transport", &["stdio", "http"])? {
            0 => {
                if p.get("port") != Some("0") || p.get("seconds") != Some("0") {
                    return Err(CliError::Usage(crate::args::UsageError(
                        "--port and --seconds belong to --transport http".to_owned(),
                    )));
                }
                Kind::Stdio
            }
            _ => Kind::Http {
                port: u16::try_from(p.number("port", 0, u32::from(u16::MAX))?).unwrap_or(0),
                seconds: p.number("seconds", 0, MAX_SECONDS)?,
            },
        };
        Ok(Self { kind, host: HostOptions::from_args(p)?, quiet: p.get("quiet").is_some() })
    }
}

/// Runs `job` until its stop condition (see the module documentation).
///
/// # Errors
/// The host's errors ([`CliError::Data`], [`CliError::Project`], ...), [`CliError::Server`] when
/// the transport cannot start.
pub fn run(job: &ServeJob) -> Result<(), CliError> {
    let o = &job.host;
    let base = Base::load(&o.data, o.project.as_deref(), o.create, &o.exchange)?;
    let (mut engine, human) = base.engine(o.human.as_deref(), o.create)?;
    let mut server = base.server(o, &mut engine, &human)?;
    let log = |text: &str| {
        if !job.quiet {
            eprintln!("{PROGRAM} serve: {text}");
        }
    };
    let what =
        o.project.as_ref().map_or_else(|| "a new character (no project)".to_owned(), |p| p.display().to_string());
    match job.kind {
        Kind::Stdio => {
            let (inbox, threads) = spawn_stdio(std::io::stdin(), std::io::stdout(), RpcLimits::DEFAULT)
                .map_err(|e| CliError::Server(format!("the stdio transport could not start: {e}")))?;
            log(&format!("stdio, {what}, admitted by {human}, template {}", base.source));
            while server.pump_blocking(&mut engine, &base.fs, &inbox, WAIT).is_some() {
                host::settle(&mut engine);
            }
            host::settle(&mut engine);
            server.shutdown();
            drop(server);
            drop(inbox);
            threads.join();
        }
        Kind::Http { port, seconds } => {
            let token =
                Token::generate(&OsEntropy).map_err(|e| CliError::Server(format!("no token without entropy: {e}")))?;
            let secret = token.as_str().to_owned();
            let (tx, inbox) = channel();
            let mut http = HttpServer::start(port, token, HttpConfig::DEFAULT, tx)
                .map_err(|e| CliError::Server(format!("127.0.0.1:{port} could not be bound: {e}")))?;
            let mut out = std::io::stdout().lock();
            writeln!(out, "{{\"url\":\"{}\",\"token\":\"{secret}\"}}", http.url())
                .and_then(|()| out.flush())
                .map_err(|e| CliError::Server(format!("standard output: {e}")))?;
            drop(out);
            log(&format!("{} for {what}, admitted by {human}, template {}", http.url(), base.source));
            let stop = stop_signal(seconds)?;
            loop {
                server.pump_blocking(&mut engine, &base.fs, &inbox, WAIT);
                host::settle(&mut engine);
                if stop() {
                    break;
                }
            }
            server.shutdown();
            http.stop();
            server.pump(&mut engine, &base.fs, &inbox);
            host::settle(&mut engine);
        }
    }
    let saved = host::finish(&mut engine, o.save)?;
    log(if saved { "stopped; the project was saved" } else { "stopped" });
    Ok(())
}

/// The HTTP server's stop condition: `seconds` passed, or (with 0) standard input closed.
fn stop_signal(seconds: u32) -> Result<Box<dyn Fn() -> bool>, CliError> {
    if seconds > 0 {
        let end = Instant::now() + Duration::from_secs(u64::from(seconds));
        return Ok(Box::new(move || Instant::now() >= end));
    }
    let (tx, rx) = channel::<()>();
    std::thread::Builder::new()
        .name("serve-stdin".into())
        .spawn(move || {
            let _ = std::io::copy(&mut std::io::stdin().lock(), &mut std::io::sink());
            let _ = tx.send(());
        })
        .map_err(|e| CliError::Server(format!("no thread to watch standard input: {e}")))?;
    Ok(Box::new(move || !matches!(rx.try_recv(), Err(TryRecvError::Empty))))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(list: &[&str]) -> Result<ServeJob, CliError> {
        let args: Vec<String> = list.iter().map(|s| (*s).to_owned()).collect();
        ServeJob::from_args(&Parsed::parse(&SERVE, &args)?)
    }

    #[test]
    fn ag_01_serve_arguments_map_to_the_job() {
        assert_eq!(job(&[]).unwrap().kind, Kind::Stdio);
        let h = job(&["--transport", "http", "--port", "8123", "--seconds", "5", "--quiet"]).unwrap();
        assert_eq!(h.kind, Kind::Http { port: 8123, seconds: 5 });
        assert!(h.quiet);
        for bad in [
            &["--transport", "tcp"][..],
            &["--port", "80"],
            &["--transport", "http", "--port", "70000"],
            &["--transport", "http", "--seconds", "-1"],
            &["--save"],
            &["stray"],
        ] {
            assert!(matches!(job(bad), Err(CliError::Usage(_))), "{bad:?}");
        }
    }
}
