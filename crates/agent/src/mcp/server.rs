//! The MCP server core `[AG-01]`, `[AG-02]`: protocol methods, tool dispatch, the pause / revoke
//! gate `[SEC-03]`, subscriptions and notifications `[NOTE-01]`.
//!
//! **Protocol.** The server implements MCP revision **2026-07-28** (stateless: every request
//! carries `_meta["io.modelcontextprotocol/protocolVersion"]` and `clientCapabilities`;
//! `server/discover`; `subscriptions/listen` for long-lived notification streams; results carry
//! `resultType`). It is *dual-era*: a client that opens with `initialize` is served with the
//! legacy revision it asks for among [`LEGACY_VERSIONS`] (sessions over HTTP through
//! `Mcp-Session-Id`, a GET event stream, server notifications on stdio after the handshake).
//!
//! **Threads.** Transports ([`super::transport`]) read and check bytes on their own threads and
//! hand each message to the server as an [`Inbound`] over a channel; the server lives on the
//! engine's thread (the editor's main loop or the CLI's serve loop), which calls
//! [`Server::pump`]. Replies go back through each message's own channel.
//!
//! **Gate.** While the human has paused the agent, `tools/call` of mutating tools are queued
//! (in order, up to [`ServerConfig::max_queued`], each until its deadline) and run on resume;
//! read-only tools keep working. Revoking answers the queue and every later request (except
//! `server/discover`, `initialize` and `ping`) with [`ACCESS_REVOKED`].
//!
//! **Notifications.** After each pump the server reads the session's event log and sends, on
//! every subscription that asked for them (filter member `com.humanart/events`):
//! `notifications/humanart/event` (every NOTE-01 event with its sequence number, the human's
//! edits included), `notifications/humanart/proposal_state_changed` (opened, accepted, discarded,
//! and `awaiting_review` from `commit_proposal`) and `notifications/humanart/render_ready` (async
//! renders); and `notifications/resources/updated` for `humanart://document` to subscriptions
//! that list it.

use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

use humanart_core::platform::FileSystem;
use humanart_core::rng::Rng;
use humanart_doc::document::Author;
use humanart_doc::notify::{EventKind, Since};
use humanart_io::json::{Object, Value};

use super::jsonrpc::{
    ACCESS_REVOKED, HEADER_MISMATCH, INVALID_PARAMS, Id, Incoming, METHOD_NOT_FOUND, MISSING_CAPABILITY, RpcError,
    TIMED_OUT, UNSUPPORTED_VERSION, encode, error_response, notification, response,
};
use super::render::{CAMERA_PRESETS, RIG_NAMES, RenderLimits};
use super::schema::{SchemaError, Vocab, validate_record};
use super::security::Sandbox;
use super::tools::{self, Access, AgentPolicy, ToolCtx, ToolState, object, uint};
use crate::engine::Engine;

/// The modern revisions served.
pub const MODERN_VERSIONS: [&str; 1] = ["2026-07-28"];
/// The legacy (handshake) revisions served after `initialize`.
pub const LEGACY_VERSIONS: [&str; 3] = ["2025-11-25", "2025-06-18", "2025-03-26"];
/// The filter member that asks for HumanArt's events on a subscription.
pub const EVENTS_FILTER: &str = "com.humanart/events";
/// The URI of the document resource.
pub const DOCUMENT_URI: &str = "humanart://document";

const META_VERSION: &str = "io.modelcontextprotocol/protocolVersion";
const META_CAPS: &str = "io.modelcontextprotocol/clientCapabilities";
const META_SUB: &str = "io.modelcontextprotocol/subscriptionId";
const META_SERVER: &str = "io.modelcontextprotocol/serverInfo";

/// The server's settings.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// `serverInfo.name`.
    pub name: String,
    /// `serverInfo.version`.
    pub version: String,
    /// The agent's author record (commands are made under its id).
    pub agent: Author,
    /// What the agent may do without a proposal.
    pub policy: AgentPolicy,
    /// Render limits.
    pub render_limits: RenderLimits,
    /// Most tool calls queued while paused.
    pub max_queued: usize,
    /// The data-provenance table served as `humanart://provenance`, when known.
    pub provenance: Option<std::path::PathBuf>,
}

impl ServerConfig {
    /// Settings for an agent with author id `agent_id`.
    pub fn new(agent_id: &str, agent_name: &str) -> Self {
        Self {
            name: "humanart".to_owned(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
            agent: Author {
                id: agent_id.to_owned(),
                kind: humanart_doc::document::AuthorKind::Agent,
                name: agent_name.to_owned(),
            },
            policy: AgentPolicy::DEFAULT,
            render_limits: RenderLimits::DEFAULT,
            max_queued: 1024,
            provenance: None,
        }
    }
}

/// The human's control over the agent `[SEC-03]`, `[APP-04]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate {
    /// Requests run.
    Open,
    /// Mutations queue.
    Paused,
    /// Everything is refused.
    Revoked,
}

/// What the HTTP layer mirrored from the body into headers (2026-07-28 request metadata).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Mirror {
    /// `MCP-Protocol-Version`.
    pub protocol_version: Option<String>,
    /// `Mcp-Method`.
    pub method: Option<String>,
    /// `Mcp-Name`, already decoded from the base64 sentinel form.
    pub name: Option<String>,
    /// `Mcp-Session-Id` (legacy sessions).
    pub session: Option<String>,
}

/// Where a message came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Transport {
    /// Standard input (one client per process).
    Stdio,
    /// One HTTP POST.
    Http(Mirror),
}

/// What goes back to a transport.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reply {
    /// A JSON-RPC message, the HTTP status for it, and a legacy session id to announce.
    Message {
        /// The message bytes.
        body: Vec<u8>,
        /// The HTTP status (stdio ignores it).
        status: u16,
        /// `Mcp-Session-Id` to set (legacy `initialize` over HTTP).
        session: Option<String>,
    },
    /// HTTP 202: a notification was accepted.
    Accepted,
    /// The HTTP response becomes an event stream; later messages on this channel are events.
    StreamOpen,
    /// A bare HTTP status (unknown legacy session, legacy session ended).
    Status(u16),
}

/// One message for the server.
#[derive(Debug)]
pub struct Envelope {
    /// The message.
    pub incoming: Incoming,
    /// Its transport.
    pub transport: Transport,
    /// Where the replies go.
    pub reply: Sender<Reply>,
    /// After this the transport stopped waiting; a queued call is not run any more.
    pub deadline: Instant,
}

/// What transports send the server.
#[derive(Debug)]
pub enum Inbound {
    /// A JSON-RPC message.
    Message(Envelope),
    /// Legacy HTTP: GET opens the session's event stream.
    LegacyStream {
        /// The session id.
        session: String,
        /// The stream's channel.
        reply: Sender<Reply>,
    },
    /// Legacy HTTP: DELETE ends the session.
    LegacyEnd {
        /// The session id.
        session: String,
        /// Where the status goes.
        reply: Sender<Reply>,
    },
}

#[derive(Debug)]
struct Subscription {
    /// The `subscriptions/listen` request id (modern), `None` for a legacy stream.
    id: Option<Id>,
    sink: Sender<Reply>,
    events: bool,
    resources: Vec<String>,
    /// Events up to this sequence number were handled.
    seen: u64,
    /// The legacy HTTP session it belongs to.
    session: Option<String>,
    /// It lives on stdio.
    stdio: bool,
}

/// The server (see the module documentation).
pub struct Server {
    config: ServerConfig,
    sandbox: Sandbox,
    state: ToolState,
    gate: Gate,
    queued: VecDeque<Envelope>,
    subs: Vec<Subscription>,
    sessions: Vec<String>,
    stdio_legacy: Option<String>,
    rng: Rng,
    notes: Vec<(&'static str, Value)>,
    tools_list: Option<Value>,
}

impl std::fmt::Debug for Server {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Server")
            .field("gate", &self.gate)
            .field("queued", &self.queued.len())
            .field("subscriptions", &self.subs.len())
            .finish_non_exhaustive()
    }
}

fn status_of(code: i64) -> u16 {
    match code {
        METHOD_NOT_FOUND => 404,
        HEADER_MISMATCH | UNSUPPORTED_VERSION | MISSING_CAPABILITY | INVALID_PARAMS => 400,
        ACCESS_REVOKED => 403,
        TIMED_OUT => 504,
        _ => 200,
    }
}

fn meta_server(config: &ServerConfig) -> Value {
    let mut m = Object::with_capacity(1);
    m.insert(
        META_SERVER,
        object(vec![("name", config.name.as_str().into()), ("version", config.version.as_str().into())]),
    );
    Value::Object(m)
}

/// A result object with `resultType` and the server info in `_meta`.
fn complete(config: &ServerConfig, members: Vec<(&str, Value)>) -> Value {
    let mut o = Object::with_capacity(members.len() + 2);
    o.insert("resultType", "complete");
    for (k, v) in members {
        o.insert(k, v);
    }
    o.insert("_meta", meta_server(config));
    Value::Object(o)
}

impl Server {
    /// A server for `config` whose agent paths live in `sandbox`; `seed` seeds the legacy session
    /// ids (draw it from the platform's entropy).
    pub fn new(config: ServerConfig, sandbox: Sandbox, seed: u64) -> Self {
        Self {
            config,
            sandbox,
            state: ToolState::default(),
            gate: Gate::Open,
            queued: VecDeque::new(),
            subs: Vec::new(),
            sessions: Vec::new(),
            stdio_legacy: None,
            rng: Rng::new(seed),
            notes: Vec::new(),
            tools_list: None,
        }
    }

    /// The configuration.
    pub fn config(&self) -> &ServerConfig {
        &self.config
    }

    /// Registers the agent as an author of the document, as a command made by the human
    /// `human_id` (the host calls this when the human grants the link): the session refuses
    /// commands from unknown authors. Does nothing when the agent is already an author.
    ///
    /// # Errors
    /// The session's refusal (for example when `human_id` is not a human author).
    pub fn admit(&self, engine: &mut Engine<'_>, human_id: &str) -> Result<(), humanart_doc::command::CommandError> {
        if engine.session().document().author(&self.config.agent.id) == Some(&self.config.agent) {
            return Ok(());
        }
        let action = humanart_doc::command::Action::AddAuthor { author: self.config.agent.clone() };
        engine.execute(humanart_doc::command::Command::new(human_id, action)).map(|_| ())
    }

    /// Changes the policy (the human changed the project's settings).
    pub fn set_policy(&mut self, policy: AgentPolicy) {
        self.config.policy = policy;
    }

    /// The gate.
    pub fn gate(&self) -> Gate {
        self.gate
    }

    /// Pauses, resumes or revokes the agent. Resuming runs the queue at the next pump; revoking
    /// refuses it.
    pub fn set_gate(&mut self, gate: Gate) {
        self.gate = gate;
        if gate == Gate::Revoked {
            for env in self.queued.drain(..) {
                if let Incoming::Request { id, .. } = &env.incoming {
                    let e = RpcError::new(ACCESS_REVOKED, "the human revoked the agent's access");
                    let _ = env.reply.send(Reply::Message {
                        body: encode(&error_response(Some(id), &e)),
                        status: 403,
                        session: None,
                    });
                }
            }
        }
    }

    /// Tool calls waiting for the human to resume the agent.
    pub fn queued(&self) -> usize {
        self.queued.len()
    }

    /// Open subscriptions (event streams).
    pub fn subscriptions(&self) -> usize {
        self.subs.len()
    }

    /// The renders the server remembers (newest last).
    pub fn renders(&self) -> &[tools::RenderRecord] {
        &self.state.records
    }

    /// Handles everything waiting in `inbox`, the queue (when open) and async renders, then sends
    /// the notifications of new events. Returns how many messages were handled.
    pub fn pump(&mut self, engine: &mut Engine<'_>, fs: &dyn FileSystem, inbox: &Receiver<Inbound>) -> usize {
        let mut handled = 0;
        if self.gate == Gate::Open {
            while let Some(env) = self.queued.pop_front() {
                if Instant::now() > env.deadline {
                    if let Incoming::Request { id, .. } = &env.incoming {
                        let e = RpcError::new(TIMED_OUT, "the request waited too long while the agent was paused");
                        let _ = env.reply.send(Reply::Message {
                            body: encode(&error_response(Some(id), &e)),
                            status: 504,
                            session: None,
                        });
                    }
                    continue;
                }
                self.handle_envelope(engine, fs, env);
                handled += 1;
            }
        }
        while let Ok(inbound) = inbox.try_recv() {
            self.handle(engine, fs, inbound);
            handled += 1;
        }
        self.run_pending(engine, fs);
        self.flush(engine);
        handled
    }

    /// Like [`Server::pump`], but waits up to `wait` for the first message (the CLI's loop).
    /// `None` when every transport has gone (stdin closed, listener stopped).
    pub fn pump_blocking(
        &mut self,
        engine: &mut Engine<'_>,
        fs: &dyn FileSystem,
        inbox: &Receiver<Inbound>,
        wait: Duration,
    ) -> Option<usize> {
        match inbox.recv_timeout(wait) {
            Ok(first) => {
                self.handle(engine, fs, first);
                Some(1 + self.pump(engine, fs, inbox))
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Some(self.pump(engine, fs, inbox)),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                self.pump(engine, fs, inbox);
                None
            }
        }
    }

    /// Ends every subscription gracefully (the final response of each `subscriptions/listen`).
    pub fn shutdown(&mut self) {
        for sub in self.subs.drain(..) {
            if let Some(id) = &sub.id {
                let mut meta = Object::with_capacity(1);
                meta.insert(META_SUB, id.to_json());
                let mut r = Object::with_capacity(2);
                r.insert("resultType", "complete");
                r.insert("_meta", Value::Object(meta));
                let _ = sub.sink.send(Reply::Message {
                    body: encode(&response(id, Value::Object(r))),
                    status: 200,
                    session: None,
                });
            }
        }
    }

    /// Handles one inbound message.
    pub fn handle(&mut self, engine: &mut Engine<'_>, fs: &dyn FileSystem, inbound: Inbound) {
        match inbound {
            Inbound::Message(env) => self.handle_envelope(engine, fs, env),
            Inbound::LegacyStream { session, reply } => {
                if self.sessions.contains(&session) {
                    let _ = reply.send(Reply::StreamOpen);
                    let seen = engine.session().events().last_seq();
                    self.subs.push(Subscription {
                        id: None,
                        sink: reply,
                        events: true,
                        resources: Vec::new(),
                        seen,
                        session: Some(session),
                        stdio: false,
                    });
                } else {
                    let _ = reply.send(Reply::Status(404));
                }
            }
            Inbound::LegacyEnd { session, reply } => {
                let known = self.sessions.contains(&session);
                self.sessions.retain(|s| *s != session);
                self.subs.retain(|s| s.session.as_deref() != Some(session.as_str()));
                let _ = reply.send(Reply::Status(if known { 200 } else { 404 }));
            }
        }
    }

    fn reply_error(env: &Envelope, id: Option<&Id>, error: &RpcError) {
        let status = status_of(error.code);
        let _ = env.reply.send(Reply::Message { body: encode(&error_response(id, error)), status, session: None });
    }

    fn handle_envelope(&mut self, engine: &mut Engine<'_>, fs: &dyn FileSystem, env: Envelope) {
        let (id, method, params) = match &env.incoming {
            Incoming::Response { .. } => {
                if matches!(env.transport, Transport::Http(_)) {
                    let _ = env.reply.send(Reply::Status(400));
                }
                return;
            }
            Incoming::Notification { method, params } => {
                if method == "notifications/cancelled" {
                    let target = params.get("requestId").and_then(|v| match v {
                        Value::String(s) => Some(Id::String(s.clone())),
                        other => other.as_i64().map(Id::Number),
                    });
                    if let Some(t) = target {
                        self.subs.retain(|s| !(s.stdio && s.id.as_ref() == Some(&t)));
                    }
                }
                if matches!(env.transport, Transport::Http(_)) {
                    let _ = env.reply.send(Reply::Accepted);
                }
                return;
            }
            Incoming::Request { id, method, params } => (id.clone(), method.clone(), params.clone()),
        };
        // Era and request metadata.
        let meta = params.get("_meta");
        let version = meta.and_then(|m| m.get(META_VERSION)).and_then(Value::as_str);
        let legacy_ok = match &env.transport {
            Transport::Stdio => self.stdio_legacy.is_some(),
            Transport::Http(m) => m.session.as_ref().is_some_and(|s| self.sessions.contains(s)),
        };
        if let Some(v) = version {
            if !MODERN_VERSIONS.contains(&v) {
                let mut all: Vec<Value> = MODERN_VERSIONS.iter().map(|s| Value::from(*s)).collect();
                all.extend(LEGACY_VERSIONS.iter().map(|s| Value::from(*s)));
                let data = object(vec![("supported", Value::Array(all)), ("requested", v.into())]);
                Self::reply_error(
                    &env,
                    Some(&id),
                    &RpcError::new(UNSUPPORTED_VERSION, "Unsupported protocol version").with_data(data),
                );
                return;
            }
            if !meta.and_then(|m| m.get(META_CAPS)).is_some_and(|c| matches!(c, Value::Object(_))) {
                Self::reply_error(
                    &env,
                    Some(&id),
                    &RpcError::new(INVALID_PARAMS, "_meta lacks io.modelcontextprotocol/clientCapabilities"),
                );
                return;
            }
            if let Transport::Http(m) = &env.transport
                && let Err(e) = check_mirror(m, v, &method, &params)
            {
                Self::reply_error(&env, Some(&id), &e);
                return;
            }
        } else if method == "initialize" {
            self.initialize(engine, &env, &id, &params);
            return;
        } else if !legacy_ok {
            if let Transport::Http(m) = &env.transport
                && m.session.is_some()
            {
                let _ = env.reply.send(Reply::Status(404));
                return;
            }
            Self::reply_error(
                &env,
                Some(&id),
                &RpcError::new(INVALID_PARAMS, "the request's _meta lacks io.modelcontextprotocol/protocolVersion"),
            );
            return;
        }
        if self.gate == Gate::Revoked && !matches!(method.as_str(), "server/discover" | "ping") {
            Self::reply_error(&env, Some(&id), &RpcError::new(ACCESS_REVOKED, "the human revoked the agent's access"));
            return;
        }
        let result = match method.as_str() {
            "server/discover" => Ok(self.discover()),
            "ping" | "logging/setLevel" => Ok(complete(&self.config, Vec::new())),
            "tools/list" => Ok(self.tools_list(engine)),
            "tools/call" => {
                if self.gate == Gate::Paused && is_mutation(&params) {
                    if self.queued.len() >= self.config.max_queued {
                        Err(RpcError::new(TIMED_OUT, "too many calls are queued while the agent is paused"))
                    } else {
                        self.queued.push_back(env);
                        return;
                    }
                } else {
                    self.call_tool(engine, fs, &params)
                }
            }
            "resources/list" => Ok(self.resources_list()),
            "resources/templates/list" => Ok(complete(
                &self.config,
                vec![(
                    "resourceTemplates",
                    Value::Array(vec![object(vec![
                        ("uriTemplate", "humanart://renders/{id}".into()),
                        ("name", "render".into()),
                        ("mimeType", "image/png".into()),
                    ])]),
                )],
            )),
            "resources/read" => self.resources_read(engine, fs, &params),
            "subscriptions/listen" => {
                self.listen(engine, &env, &id, &params);
                return;
            }
            "resources/subscribe" | "resources/unsubscribe" if version.is_none() => {
                let uri = params.get("uri").and_then(Value::as_str).unwrap_or("").to_owned();
                let subscribe = method == "resources/subscribe";
                for s in self.subs.iter_mut().filter(|s| s.id.is_none()) {
                    s.resources.retain(|u| *u != uri);
                    if subscribe {
                        s.resources.push(uri.clone());
                    }
                }
                Ok(complete(&self.config, Vec::new()))
            }
            _ => Err(RpcError::new(METHOD_NOT_FOUND, format!("method not found: {method}"))),
        };
        match result {
            Ok(r) => {
                let _ = env.reply.send(Reply::Message { body: encode(&response(&id, r)), status: 200, session: None });
            }
            Err(e) => Self::reply_error(&env, Some(&id), &e),
        }
        self.flush(engine);
    }

    fn capabilities() -> Value {
        object(vec![
            ("tools", object(vec![("listChanged", false.into())])),
            ("resources", object(vec![("subscribe", true.into()), ("listChanged", true.into())])),
        ])
    }

    fn instructions() -> &'static str {
        "HumanArt edits a realistic human character. Read with describe_character, describe_parameters and render_view; \
         edit inside a proposal (open_proposal, then pass proposal_id to mutations, then commit_proposal for the human's \
         review). Values and layers the human set are protected: a refused command names the protection. Subscribe with \
         subscriptions/listen and {\"com.humanart/events\": true} to see what the human does."
    }

    fn discover(&self) -> Value {
        let mut versions: Vec<Value> = MODERN_VERSIONS.iter().map(|s| Value::from(*s)).collect();
        versions.extend(LEGACY_VERSIONS.iter().map(|s| Value::from(*s)));
        complete(
            &self.config,
            vec![
                ("supportedVersions", Value::Array(versions)),
                ("capabilities", Self::capabilities()),
                ("instructions", Self::instructions().into()),
            ],
        )
    }

    fn initialize(&mut self, engine: &Engine<'_>, env: &Envelope, id: &Id, params: &Value) {
        let asked = params.get("protocolVersion").and_then(Value::as_str).unwrap_or("");
        let version = LEGACY_VERSIONS.iter().find(|v| **v == asked).copied().unwrap_or(LEGACY_VERSIONS[0]);
        let result = object(vec![
            ("protocolVersion", version.into()),
            ("capabilities", Self::capabilities()),
            (
                "serverInfo",
                object(vec![
                    ("name", self.config.name.as_str().into()),
                    ("version", self.config.version.as_str().into()),
                ]),
            ),
            ("instructions", Self::instructions().into()),
        ]);
        let session = match &env.transport {
            Transport::Stdio => {
                self.stdio_legacy = Some(version.to_owned());
                // Legacy stdio: the server sends notifications whenever it likes after the handshake.
                self.subs.retain(|s| !(s.stdio && s.id.is_none()));
                let seen = engine.session().events().last_seq();
                self.subs.push(Subscription {
                    id: None,
                    sink: env.reply.clone(),
                    events: true,
                    resources: Vec::new(),
                    seen,
                    session: None,
                    stdio: true,
                });
                None
            }
            Transport::Http(_) => {
                let sid = format!("{:016x}{:016x}", self.rng.next_u64(), self.rng.next_u64());
                self.sessions.push(sid.clone());
                if self.sessions.len() > 64 {
                    self.sessions.remove(0);
                }
                Some(sid)
            }
        };
        let _ = env.reply.send(Reply::Message { body: encode(&response(id, result)), status: 200, session });
    }

    fn listen(&mut self, engine: &Engine<'_>, env: &Envelope, id: &Id, params: &Value) {
        let filter = params.get("notifications");
        let events = filter.and_then(|f| f.get(EVENTS_FILTER)).and_then(Value::as_bool).unwrap_or(false);
        let resources: Vec<String> = filter
            .and_then(|f| f.get("resourceSubscriptions"))
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .filter(|u| u.starts_with("humanart://"))
                    .take(64)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        let mut ack = Object::with_capacity(3);
        if events {
            ack.insert(EVENTS_FILTER, true);
        }
        if !resources.is_empty() {
            ack.insert("resourceSubscriptions", Value::Array(resources.iter().map(Value::from).collect()));
        }
        let mut meta = Object::with_capacity(1);
        meta.insert(META_SUB, id.to_json());
        let mut p = Object::with_capacity(2);
        p.insert("_meta", Value::Object(meta));
        p.insert("notifications", Value::Object(ack));
        let stdio = env.transport == Transport::Stdio;
        if !stdio {
            let _ = env.reply.send(Reply::StreamOpen);
        }
        let body = encode(&notification("notifications/subscriptions/acknowledged", Value::Object(p)));
        let _ = env.reply.send(Reply::Message { body, status: 200, session: None });
        let seen = engine.session().events().last_seq();
        self.subs.push(Subscription {
            id: Some(id.clone()),
            sink: env.reply.clone(),
            events,
            resources,
            seen,
            session: None,
            stdio,
        });
    }

    fn tools_list(&mut self, engine: &Engine<'_>) -> Value {
        let tools = self
            .tools_list
            .get_or_insert_with(|| {
                let vocab = Vocab { registry: engine.registry(), template: engine.model().template() };
                Value::Array(tools::all().map(|t| t.definition(&vocab)).collect())
            })
            .clone();
        complete(&self.config, vec![("tools", tools)])
    }

    fn call_tool(&mut self, engine: &mut Engine<'_>, fs: &dyn FileSystem, params: &Value) -> Result<Value, RpcError> {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| RpcError::new(INVALID_PARAMS, "tools/call needs a name"))?;
        let tool = tools::find(name).ok_or_else(|| RpcError::new(INVALID_PARAMS, format!("unknown tool: {name}")))?;
        let empty = Value::Object(Object::new());
        let arguments = params.get("arguments").unwrap_or(&empty);
        let vocab = Vocab { registry: engine.registry(), template: engine.model().template() };
        let checked: Result<(), SchemaError> = validate_record(tool.fields, arguments, &vocab, "");
        let outcome = if let Err(e) = checked {
            Err(tools::ToolError::new("invalid_arguments", e.to_string()))
        } else {
            {
                let mut ctx = ToolCtx {
                    engine,
                    fs,
                    agent: &self.config.agent,
                    policy: self.config.policy,
                    sandbox: &self.sandbox,
                    render_limits: self.config.render_limits,
                    state: &mut self.state,
                    notes: &mut self.notes,
                };
                (tool.handler)(&mut ctx, arguments)
            }
        };
        Ok(self.tool_result(outcome))
    }

    fn tool_result(&self, outcome: Result<tools::ToolOutput, tools::ToolError>) -> Value {
        let (data, images, is_error) = match outcome {
            Ok(o) => (o.data, o.images, false),
            Err(e) => (e.to_json(), Vec::new(), true),
        };
        let text = humanart_io::json::to_string(&data).unwrap_or_else(|_| "{}".to_owned());
        let mut content = vec![object(vec![("type", "text".into()), ("text", text.into())])];
        for png in images {
            content.push(object(vec![
                ("type", "image".into()),
                ("data", humanart_io::gltf::base64::encode(&png).into()),
                ("mimeType", "image/png".into()),
            ]));
        }
        complete(
            &self.config,
            vec![("content", Value::Array(content)), ("structuredContent", data), ("isError", is_error.into())],
        )
    }

    fn resources_list(&self) -> Value {
        let mut list = vec![
            object(vec![
                ("uri", DOCUMENT_URI.into()),
                ("name", "document".into()),
                ("mimeType", "application/json".into()),
                ("description", "The character document (read-only JSON view).".into()),
            ]),
            object(vec![
                ("uri", "humanart://registry".into()),
                ("name", "registry".into()),
                ("mimeType", "application/json".into()),
                ("description", "The parameter registry.".into()),
            ]),
            object(vec![
                ("uri", "humanart://cameras".into()),
                ("name", "cameras".into()),
                ("mimeType", "application/json".into()),
                ("description", "Standard review cameras and framings.".into()),
            ]),
            object(vec![
                ("uri", "humanart://rigs".into()),
                ("name", "rigs".into()),
                ("mimeType", "application/json".into()),
                ("description", "Lighting rigs.".into()),
            ]),
        ];
        if self.config.provenance.is_some() {
            list.push(object(vec![
                ("uri", "humanart://provenance".into()),
                ("name", "provenance".into()),
                ("mimeType", "text/markdown".into()),
                ("description", "The data-provenance table.".into()),
            ]));
        }
        for r in &self.state.records {
            list.push(object(vec![
                ("uri", format!("humanart://renders/{}", r.id).into()),
                ("name", format!("render {}", r.id).into()),
                ("mimeType", "image/png".into()),
            ]));
        }
        complete(&self.config, vec![("resources", Value::Array(list))])
    }

    fn resources_read(
        &mut self,
        engine: &mut Engine<'_>,
        fs: &dyn FileSystem,
        params: &Value,
    ) -> Result<Value, RpcError> {
        let uri = params
            .get("uri")
            .and_then(Value::as_str)
            .ok_or_else(|| RpcError::new(INVALID_PARAMS, "resources/read needs a uri"))?
            .to_owned();
        let text = |mime: &str, v: &Value| {
            object(vec![
                ("uri", uri.as_str().into()),
                ("mimeType", mime.into()),
                ("text", humanart_io::json::to_string(v).unwrap_or_default().into()),
            ])
        };
        let item = match uri.as_str() {
            DOCUMENT_URI => {
                let empty = Value::Object(Object::new());
                let summary = {
                    let mut ctx = ToolCtx {
                        engine,
                        fs,
                        agent: &self.config.agent,
                        policy: self.config.policy,
                        sandbox: &self.sandbox,
                        render_limits: self.config.render_limits,
                        state: &mut self.state,
                        notes: &mut self.notes,
                    };
                    tools::find("describe_character").map(|t| (t.handler)(&mut ctx, &empty))
                };
                let mut v = summary.and_then(Result::ok).map_or(Value::Null, |o| o.data);
                let doc = engine.session().document();
                let reg = engine.registry();
                let params: Object = doc
                    .params
                    .iter()
                    .map(|(k, val)| (k.as_str(), humanart_doc::registry::value_to_json(val)))
                    .collect();
                let pose: Object = doc
                    .pose
                    .iter()
                    .map(|(k, q)| (k.as_str(), Value::Array(q.iter().map(|x| tools::num(*x)).collect())))
                    .collect();
                if let Value::Object(o) = &mut v {
                    o.insert("parameters", Value::Object(params));
                    o.insert("pose", Value::Object(pose));
                    o.insert("registry_size", uint(reg.len() as u64));
                }
                text("application/json", &v)
            }
            "humanart://registry" => text("application/json", &engine.registry().describe()),
            "humanart://cameras" => {
                text("application/json", &Value::Array(CAMERA_PRESETS.iter().map(|c| Value::from(*c)).collect()))
            }
            "humanart://rigs" => {
                text("application/json", &Value::Array(RIG_NAMES.iter().map(|c| Value::from(*c)).collect()))
            }
            "humanart://provenance" => {
                let path = self
                    .config
                    .provenance
                    .as_ref()
                    .ok_or_else(|| RpcError::new(INVALID_PARAMS, "resource not found"))?;
                let body =
                    std::fs::read_to_string(path).map_err(|_| RpcError::new(INVALID_PARAMS, "resource not found"))?;
                object(vec![("uri", uri.as_str().into()), ("mimeType", "text/markdown".into()), ("text", body.into())])
            }
            other => {
                let id = other.strip_prefix("humanart://renders/").and_then(|s| s.parse::<u64>().ok());
                let rec = id
                    .and_then(|id| self.state.records.iter().find(|r| r.id == id))
                    .ok_or_else(|| RpcError::new(INVALID_PARAMS, format!("resource not found: {other}")))?;
                object(vec![
                    ("uri", uri.as_str().into()),
                    ("mimeType", "image/png".into()),
                    ("blob", humanart_io::gltf::base64::encode(&rec.png).into()),
                ])
            }
        };
        Ok(complete(&self.config, vec![("contents", Value::Array(vec![item]))]))
    }

    fn run_pending(&mut self, engine: &mut Engine<'_>, fs: &dyn FileSystem) {
        while !self.state.pending.is_empty() {
            let job = self.state.pending.remove(0);
            let mut ctx = ToolCtx {
                engine,
                fs,
                agent: &self.config.agent,
                policy: self.config.policy,
                sandbox: &self.sandbox,
                render_limits: self.config.render_limits,
                state: &mut self.state,
                notes: &mut self.notes,
            };
            let out = tools::observe::run_render(&mut ctx, &job.args, &job.ids);
            let params = match out {
                Ok(o) => object(vec![
                    ("render_ids", Value::Array(job.ids.iter().map(|i| uint(*i)).collect())),
                    ("result", o.data),
                ]),
                Err(e) => object(vec![
                    ("render_ids", Value::Array(job.ids.iter().map(|i| uint(*i)).collect())),
                    ("result", e.to_json()),
                ]),
            };
            self.notes.push(("notifications/humanart/render_ready", params));
        }
    }

    /// Sends the notifications of new events and of tool notes to the subscriptions.
    pub fn flush(&mut self, engine: &Engine<'_>) {
        let log = engine.session().events();
        let last = log.last_seq();
        let notes = std::mem::take(&mut self.notes);
        let legacy_stdio = self.stdio_legacy.is_some();
        self.subs.retain_mut(|sub| {
            let mut out: Vec<Value> = Vec::new();
            if sub.seen < last {
                let events: Vec<&humanart_doc::notify::Event> = match log.since(sub.seen) {
                    Since::Events(list) => list,
                    Since::Gap { .. } => Vec::new(),
                };
                for e in events {
                    if sub.events {
                        out.push(notification("notifications/humanart/event", e.to_json()));
                        let state = match &e.kind {
                            EventKind::ProposalOpened { proposal, .. } => Some((proposal, "open")),
                            EventKind::ProposalAccepted { proposal, .. } => Some((proposal, "accepted")),
                            EventKind::ProposalDiscarded { proposal } => Some((proposal, "discarded")),
                            _ => None,
                        };
                        if let Some((p, s)) = state {
                            out.push(notification(
                                "notifications/humanart/proposal_state_changed",
                                object(vec![("proposal", p.to_hex().as_str().into()), ("state", s.into()), ("seq", uint(e.seq))]),
                            ));
                        }
                    }
                    let changes_doc = matches!(
                        e.kind,
                        EventKind::Committed(ref o) | EventKind::Undone(ref o) | EventKind::Redone(ref o) if o.proposal.is_none()
                    ) || matches!(e.kind, EventKind::ProposalAccepted { .. } | EventKind::Recovered { .. });
                    if changes_doc && sub.resources.iter().any(|u| u == DOCUMENT_URI) {
                        out.push(notification("notifications/resources/updated", object(vec![("uri", DOCUMENT_URI.into())])));
                    }
                }
                sub.seen = last;
            }
            if sub.events {
                for (method, params) in &notes {
                    out.push(notification(method, params.clone()));
                }
            }
            for mut n in out {
                if let Some(id) = &sub.id
                    && let Some(Value::Object(p)) = n.get_mut("params")
                {
                    let mut meta = Object::with_capacity(1);
                    meta.insert(META_SUB, id.to_json());
                    p.insert("_meta", Value::Object(meta));
                }
                if sub.stdio && sub.id.is_none() && !legacy_stdio {
                    continue;
                }
                if sub.sink.send(Reply::Message { body: encode(&n), status: 200, session: None }).is_err() {
                    return false;
                }
            }
            true
        });
    }
}

fn is_mutation(params: &Value) -> bool {
    params.get("name").and_then(Value::as_str).and_then(tools::find).is_some_and(|t| t.access != Access::Read)
}

/// Checks the HTTP headers that mirror the body (2026-07-28 "Server Validation").
fn check_mirror(m: &Mirror, version: &str, method: &str, params: &Value) -> Result<(), RpcError> {
    let mismatch = |what: &str| Err(RpcError::new(HEADER_MISMATCH, format!("Header mismatch: {what}")));
    match m.protocol_version.as_deref() {
        Some(v) if v == version => {}
        Some(_) => return mismatch("MCP-Protocol-Version differs from the body's protocol version"),
        None => return mismatch("MCP-Protocol-Version is missing"),
    }
    match m.method.as_deref() {
        Some(x) if x == method => {}
        Some(_) => return mismatch("Mcp-Method differs from the body's method"),
        None => return mismatch("Mcp-Method is missing"),
    }
    let body_name = match method {
        "tools/call" | "prompts/get" => params.get("name").and_then(Value::as_str),
        "resources/read" => params.get("uri").and_then(Value::as_str),
        _ => return Ok(()),
    };
    match (m.name.as_deref(), body_name) {
        (Some(h), Some(b)) if h == b => Ok(()),
        (None, _) => mismatch("Mcp-Name is missing"),
        _ => mismatch("Mcp-Name differs from the body"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirror_headers_must_match_the_body() {
        let params = humanart_io::json::parse_str(r#"{"name":"render_view"}"#).unwrap();
        let ok = Mirror {
            protocol_version: Some("2026-07-28".into()),
            method: Some("tools/call".into()),
            name: Some("render_view".into()),
            session: None,
        };
        assert!(check_mirror(&ok, "2026-07-28", "tools/call", &params).is_ok());
        for bad in [
            Mirror { protocol_version: None, ..ok.clone() },
            Mirror { protocol_version: Some("2025-11-25".into()), ..ok.clone() },
            Mirror { method: Some("tools/list".into()), ..ok.clone() },
            Mirror { name: Some("measure".into()), ..ok.clone() },
            Mirror { name: None, ..ok.clone() },
        ] {
            assert_eq!(check_mirror(&bad, "2026-07-28", "tools/call", &params).unwrap_err().code, HEADER_MISMATCH);
        }
        let list = Mirror { name: None, method: Some("tools/list".into()), ..ok };
        assert!(check_mirror(&list, "2026-07-28", "tools/list", &Value::Null).is_ok());
        assert_eq!(status_of(METHOD_NOT_FOUND), 404);
        assert_eq!(status_of(UNSUPPORTED_VERSION), 400);
    }
}
