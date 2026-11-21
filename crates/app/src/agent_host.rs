//! The editor's agent server `[APP-04]`, `[AG-01]`, `[SEC-01]`, `[SEC-03]`: the MCP server of
//! `humanart-agent` on loopback HTTP, run while the human grants the agent link.
//!
//! - **Start** ([`AgentHost::start`]): when the human grants access (the Agent panel), a fresh
//!   256-bit bearer token is drawn, the HTTP transport binds `127.0.0.1` (the configured port, 0:
//!   any free one), and the human admits the agent into the document ([`Server::admit`]: the
//!   session refuses commands from unknown authors). The Agent panel shows the address and copies
//!   the token.
//! - **Pump** ([`AgentHost::pump`]): every editor frame, on the thread that owns the engine. With
//!   nothing waiting it allocates nothing (`[PERF-R1]`): the queue, the channel and the
//!   notification pass are all empty.
//! - **Gate** ([`gate_of`], [`AgentHost::sync_gate`]): the link's pause queues the agent's
//!   mutations in the server (reads go on), resume runs them, revoke refuses everything.
//! - **Stop** ([`AgentHost::stop`]): revoking first closes the gate (queued calls are answered
//!   "access revoked"), then ends every subscription and stops the listener.
//! - **Project change** ([`AgentHost::rebind`]): opening or creating a project replaces the
//!   session, so the agent is admitted again and its sandbox follows the new project folder; the
//!   transport and the token stay, open subscriptions end.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};

use humanart_agent::engine::Engine;
use humanart_agent::mcp::security::{Sandbox, Token};
use humanart_agent::mcp::server::{Gate, Inbound, Server, ServerConfig};
use humanart_agent::mcp::tools::AgentPolicy;
use humanart_agent::mcp::transport::{HttpConfig, HttpServer};
use humanart_core::platform::{Entropy, FileSystem};

use crate::agent_link::AgentLink;
use crate::error::AppError;

/// Where and as whom the editor serves the agent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentServerOptions {
    /// The loopback port (0: any free port).
    pub port: u16,
    /// The folder the agent may read and write files in (on disk; created when missing).
    pub exchange: PathBuf,
    /// The agent's author id.
    pub agent_id: String,
    /// The agent's display name.
    pub agent_name: String,
}

impl AgentServerOptions {
    /// Options with any free port, the agent `agent.mcp` and its files in `exchange`.
    pub fn new(exchange: PathBuf) -> Self {
        Self { port: 0, exchange, agent_id: "agent.mcp".to_owned(), agent_name: "MCP agent".to_owned() }
    }
}

/// The server's gate for the link's state.
pub fn gate_of(link: &AgentLink) -> Gate {
    if link.is_revoked() {
        Gate::Revoked
    } else if link.is_paused() {
        Gate::Paused
    } else {
        Gate::Open
    }
}

/// The agent policy of the preferences: auto-accept lets `commit_proposal` merge without review;
/// direct edits stay off (the agent works in proposals).
pub const fn policy_of(prefs: &crate::prefs::AgentPolicy) -> AgentPolicy {
    AgentPolicy { direct_edits: false, commit_without_review: prefs.auto_accept }
}

struct Running {
    server: Server,
    http: HttpServer,
    inbox: Receiver<Inbound>,
    url: String,
    token: String,
    seed: u64,
}

/// The agent server while the link is granted (see the module documentation).
#[derive(Default)]
pub struct AgentHost {
    running: Option<Running>,
}

impl std::fmt::Debug for AgentHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentHost").field("url", &self.url()).finish_non_exhaustive()
    }
}

/// The sandbox of `options` with `project` when it is a folder on disk.
fn sandbox(options: &AgentServerOptions, project: Option<&Path>) -> Result<Sandbox, AppError> {
    let project = project.filter(|p| p.is_dir());
    Sandbox::new(&options.exchange, project).map_err(|e| AppError::AgentServer(format!("the sandbox: {e}")))
}

impl AgentHost {
    /// A host with no server running.
    pub fn new() -> Self {
        Self { running: None }
    }

    /// Whether the server runs.
    pub fn is_running(&self) -> bool {
        self.running.is_some()
    }

    /// The server's address (`http://127.0.0.1:PORT/mcp`) while it runs.
    pub fn url(&self) -> Option<&str> {
        self.running.as_ref().map(|r| r.url.as_str())
    }

    /// The bearer token while the server runs.
    pub fn token(&self) -> Option<&str> {
        self.running.as_ref().map(|r| r.token.as_str())
    }

    /// The server while it runs.
    pub fn server(&self) -> Option<&Server> {
        self.running.as_ref().map(|r| &r.server)
    }

    /// Starts the server and admits the agent into the engine's document as `human` (nothing
    /// happens when it already runs). `project` is the open project's folder on disk, if any.
    ///
    /// # Errors
    /// [`AppError::AgentServer`] when the token, the sandbox or the socket fails;
    /// [`AppError::Command`] when the session refuses the agent.
    pub fn start(
        &mut self,
        engine: &mut Engine<'_>,
        human: &str,
        options: &AgentServerOptions,
        policy: AgentPolicy,
        entropy: &dyn Entropy,
        project: Option<&Path>,
    ) -> Result<(), AppError> {
        if self.running.is_some() {
            return Ok(());
        }
        let token = Token::generate(entropy).map_err(|e| AppError::AgentServer(format!("no token: {e}")))?;
        let mut seed = [0_u8; 8];
        entropy.fill(&mut seed).map_err(|e| AppError::AgentServer(format!("no entropy: {e}")))?;
        let seed = u64::from_le_bytes(seed);
        let server = Self::server_for(engine, human, options, policy, project, seed)?;
        let secret = token.as_str().to_owned();
        let (tx, inbox) = channel();
        let http = HttpServer::start(options.port, token, HttpConfig::DEFAULT, tx)
            .map_err(|e| AppError::AgentServer(format!("127.0.0.1:{} could not be bound: {e}", options.port)))?;
        let url = http.url();
        self.running = Some(Running { server, http, inbox, url, token: secret, seed });
        Ok(())
    }

    fn server_for(
        engine: &mut Engine<'_>,
        human: &str,
        options: &AgentServerOptions,
        policy: AgentPolicy,
        project: Option<&Path>,
        seed: u64,
    ) -> Result<Server, AppError> {
        let mut config = ServerConfig::new(&options.agent_id, &options.agent_name);
        config.policy = policy;
        let server = Server::new(config, sandbox(options, project)?, seed);
        server.admit(engine, human)?;
        engine.update()?;
        Ok(server)
    }

    /// The session was replaced (a project opened or created): admits the agent into the new
    /// document and points the sandbox at `project`; the transport and token stay, open
    /// subscriptions end. Nothing happens when the server does not run.
    ///
    /// # Errors
    /// As [`AgentHost::start`]; the server keeps running with its old sandbox when the new one
    /// cannot be made.
    pub fn rebind(
        &mut self,
        engine: &mut Engine<'_>,
        human: &str,
        options: &AgentServerOptions,
        project: Option<&Path>,
    ) -> Result<(), AppError> {
        let Some(r) = &mut self.running else { return Ok(()) };
        let gate = r.server.gate();
        let policy = r.server.config().policy;
        let mut server = Self::server_for(engine, human, options, policy, project, r.seed.wrapping_add(1))?;
        server.set_gate(gate);
        r.seed = r.seed.wrapping_add(1);
        r.server.shutdown();
        r.server = server;
        Ok(())
    }

    /// Changes the agent's policy (the preferences changed).
    pub fn set_policy(&mut self, policy: AgentPolicy) {
        if let Some(r) = &mut self.running {
            r.server.set_policy(policy);
        }
    }

    /// Applies the link's pause / revoke to the server's gate (when it differs).
    pub fn sync_gate(&mut self, link: &AgentLink) {
        if let Some(r) = &mut self.running {
            let gate = gate_of(link);
            if r.server.gate() != gate {
                r.server.set_gate(gate);
            }
        }
    }

    /// Handles what the agent sent since the last frame; returns how many messages. Allocation
    /// free when nothing is waiting.
    pub fn pump(&mut self, engine: &mut Engine<'_>, fs: &dyn FileSystem) -> usize {
        match &mut self.running {
            Some(r) => r.server.pump(engine, fs, &r.inbox),
            None => 0,
        }
    }

    /// Stops the server: the gate closes (queued calls are refused), subscriptions end, the
    /// listener stops.
    pub fn stop(&mut self) {
        if let Some(mut r) = self.running.take() {
            r.server.set_gate(Gate::Revoked);
            r.server.shutdown();
            r.http.stop();
        }
    }
}

impl Drop for AgentHost {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    use humanart_agent::mcp::client::HttpClient;
    use humanart_io::json::{Object, Value};

    use crate::app::{App, Intent};
    use crate::tests::{AppFixture, frame};

    const LONG: Duration = Duration::from_secs(120);

    fn named(name: &str) -> Value {
        let mut o = Object::new();
        o.insert("name", name);
        Value::Object(o)
    }

    /// Runs frames until `done` holds (or fails after [`LONG`]).
    fn frames_until(app: &mut App<'_>, t: &mut f64, mut done: impl FnMut(&App<'_>) -> bool) {
        let end = Instant::now() + LONG;
        while !done(app) {
            assert!(Instant::now() < end, "timed out");
            frame(app, vec![], t);
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Runs `script` against the server on its own thread while the editor runs frames.
    fn with_agent<T: Send + 'static>(
        app: &mut App<'_>,
        t: &mut f64,
        script: impl FnOnce(HttpClient) -> T + Send + 'static,
    ) -> T {
        let url = app.agent_server().url().unwrap().to_owned();
        let token = app.agent_server().token().unwrap().to_owned();
        let port: u16 =
            url.strip_prefix("http://127.0.0.1:").and_then(|r| r.strip_suffix("/mcp")).unwrap().parse().unwrap();
        let handle = std::thread::spawn(move || script(HttpClient::new(([127, 0, 0, 1], port).into(), &token)));
        frames_until(app, t, |_| handle.is_finished());
        handle.join().unwrap()
    }

    #[test]
    fn app_04_ag_01_granting_serves_the_agent_and_revoking_stops_it() {
        // [APP-04] [AG-01] [SEC-03] the human grants the link: the server listens on loopback
        // with a token and the agent is admitted; the editor's frames pump it (allocation free
        // when idle); pause queues its mutations until resume; revoke stops the server; a new
        // project re-admits the agent.
        let fx = AppFixture::new();
        let mut app = fx.app();
        let mut t = 0.0;
        frame(&mut app, vec![], &mut t);
        assert!(!app.agent_server().is_running(), "nothing listens before the human grants access");
        app.push_intent(Intent::AgentRevoke(false));
        frame(&mut app, vec![], &mut t);
        assert!(app.agent_server().is_running());
        assert!(app.agent_server().url().unwrap().starts_with("http://127.0.0.1:"));
        assert_eq!(app.agent_server().token().map(str::len), Some(64));
        assert_eq!(app.agent.name(), Some("MCP agent"));
        let agent = app.agent_server().server().unwrap().config().agent.id.clone();
        assert!(app.engine().session().document().author(&agent).is_some(), "the agent was admitted");
        let opened = with_agent(&mut app, &mut t, |mut c| {
            let d = c.call_tool("describe_character", Value::Object(Object::new())).unwrap();
            assert_eq!(d.get("isError"), Some(&Value::Bool(false)));
            c.call_tool("open_proposal", named("from the agent")).unwrap()
        });
        assert_eq!(opened.get("isError"), Some(&Value::Bool(false)), "{opened:?}");
        assert_eq!(app.engine().session().proposals().len(), 1);
        // Idle frames with the server running allocate nothing [PERF-R1].
        for _ in 0..3 {
            frame(&mut app, vec![], &mut t);
        }
        let counts = humanart_core::alloc::count_in(|| frame(&mut app, vec![], &mut t));
        assert_eq!(counts.allocations + counts.reallocations, 0, "[PERF-R1] an idle frame with the server allocates");
        // Pause: the agent's mutation waits in the server; resume runs it.
        app.push_intent(Intent::AgentPause(true));
        frame(&mut app, vec![], &mut t);
        let url = app.agent_server().url().unwrap().to_owned();
        let token = app.agent_server().token().unwrap().to_owned();
        let port: u16 =
            url.strip_prefix("http://127.0.0.1:").and_then(|r| r.strip_suffix("/mcp")).unwrap().parse().unwrap();
        let waiting = std::thread::spawn(move || {
            HttpClient::new(([127, 0, 0, 1], port).into(), &token).call_tool("open_proposal", named("while paused"))
        });
        frames_until(&mut app, &mut t, |a| a.agent_server().server().is_some_and(|s| s.queued() == 1));
        assert_eq!(app.engine().session().proposals().len(), 1, "paused: not run");
        app.push_intent(Intent::AgentPause(false));
        frames_until(&mut app, &mut t, |_| waiting.is_finished());
        assert_eq!(waiting.join().unwrap().unwrap().get("isError"), Some(&Value::Bool(false)));
        frame(&mut app, vec![], &mut t);
        assert_eq!(app.engine().session().proposals().len(), 2, "resumed: run");
        // A new project: the agent is admitted into it and keeps working.
        app.save_as("projects/agent").unwrap();
        assert!(app.engine().session().document().author(&agent).is_some(), "re-admitted");
        let r = with_agent(&mut app, &mut t, |mut c| {
            c.call_tool("describe_character", Value::Object(Object::new())).unwrap()
        });
        assert_eq!(r.get("isError"), Some(&Value::Bool(false)));
        // Revoke: the server stops listening.
        app.push_intent(Intent::AgentRevoke(true));
        frame(&mut app, vec![], &mut t);
        assert!(!app.agent_server().is_running() && app.agent.is_revoked());
        let refused = HttpClient::new(([127, 0, 0, 1], port).into(), "x".repeat(64).as_str())
            .request("server/discover", Value::Null);
        assert!(refused.is_err(), "nothing answers after the revoke");
        // Granting again starts a new server with a new token.
        app.push_intent(Intent::AgentRevoke(false));
        frame(&mut app, vec![], &mut t);
        assert!(app.agent_server().is_running() && !app.agent.is_revoked());
    }

    #[test]
    fn app_04_the_link_maps_to_the_gate_and_prefs_to_the_policy() {
        let mut link = AgentLink::new();
        assert_eq!(gate_of(&link), Gate::Open);
        link.pause();
        assert_eq!(gate_of(&link), Gate::Paused);
        link.revoke();
        assert_eq!(gate_of(&link), Gate::Revoked);
        link.grant();
        assert_eq!(gate_of(&link), Gate::Open);
        let p = crate::prefs::AgentPolicy { protect_human_edits: true, auto_accept: true };
        assert_eq!(policy_of(&p), AgentPolicy { direct_edits: false, commit_without_review: true });
        let host = AgentHost::new();
        assert!(!host.is_running() && host.url().is_none() && host.token().is_none());
    }
}
