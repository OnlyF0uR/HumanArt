//! The transports `[AG-01]`: stdio and HTTP on loopback, each on its own threads, handing checked
//! messages to the [`super::server::Server`] as [`Inbound`]s.
//!
//! **stdio** ([`spawn_stdio`]): newline-delimited JSON-RPC on any reader / writer pair (the
//! process's standard streams in `humanart-cli serve`, pipes in tests). Lines are bounded by
//! [`RpcLimits::max_bytes`]; a longer line is skipped to its end and answered with a parse error.
//!
//! **HTTP** ([`HttpServer::start`]) `[SEC-01]`, `[ROB-03]`: binds 127.0.0.1 only (port 0: an
//! ephemeral port). Every request is read with the strict parser of [`super::http`] (size limits,
//! a total deadline against slow clients), then checked in this order: `Host` must be a loopback
//! name with our port and `Origin`, when present, a loopback origin (403: DNS rebinding and browser
//! cross-origin requests); the rate limit (429); the bearer token (401); the path `/mcp` (404); the
//! method (405); `Content-Type: application/json` (415) and an `Accept` that takes JSON or SSE
//! (406). At most [`HttpConfig::max_connections`] connections are served at once (503 beyond).
//! The headers that mirror the body (`MCP-Protocol-Version`, `Mcp-Method`, `Mcp-Name`) are passed
//! on for the server to compare with the body.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Ipv4Addr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::http::{
    HttpError, HttpLimits, Request, read_request, write_response, write_sse_comment, write_sse_event, write_sse_head,
};
use super::jsonrpc::{
    INVALID_REQUEST, Malformed, PARSE_ERROR, RpcError, RpcLimits, TIMED_OUT, encode, error_response, parse_message,
};
use super::security::{RateLimiter, Token, check_host, check_origin};
use super::server::{Envelope, Inbound, Mirror, Reply, Transport};

/// How long a stdio request may wait in the server (paused agent) before it is answered with a
/// timeout.
pub const STDIO_DEADLINE: Duration = Duration::from_hours(24);

/// The threads of a stdio transport.
#[derive(Debug)]
pub struct StdioThreads {
    reader: Option<JoinHandle<()>>,
    writer: Option<JoinHandle<()>>,
}

impl StdioThreads {
    /// Waits for the reader (end of input) and the writer (every reply channel dropped).
    pub fn join(mut self) {
        if let Some(r) = self.reader.take() {
            let _ = r.join();
        }
        if let Some(w) = self.writer.take() {
            let _ = w.join();
        }
    }
}

/// Reads one line of at most `max` bytes (without the newline). `Ok(None)` at the end of input;
/// `Err(())` for a line that was too long (it was consumed).
fn read_line<R: BufRead>(input: &mut R, max: usize, line: &mut Vec<u8>) -> std::io::Result<Option<Result<(), ()>>> {
    line.clear();
    let mut too_long = false;
    loop {
        let buf = match input.fill_buf() {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        if buf.is_empty() {
            return Ok(if line.is_empty() && !too_long { None } else { Some(if too_long { Err(()) } else { Ok(()) }) });
        }
        let (chunk, done) = match buf.iter().position(|&b| b == b'\n') {
            Some(i) => (&buf[..i], i + 1),
            None => (buf, buf.len()),
        };
        let found = done != buf.len() || buf.last() == Some(&b'\n');
        if !too_long {
            if line.len() + chunk.len() > max {
                too_long = true;
                line.clear();
            } else {
                line.extend_from_slice(chunk);
            }
        }
        input.consume(done);
        if found {
            return Ok(Some(if too_long { Err(()) } else { Ok(()) }));
        }
    }
}

/// Starts a stdio transport on `input` / `output`. Messages arrive on the returned receiver.
///
/// # Errors
/// When a thread cannot be started.
pub fn spawn_stdio<R, W>(input: R, output: W, limits: RpcLimits) -> std::io::Result<(Receiver<Inbound>, StdioThreads)>
where
    R: Read + Send + 'static,
    W: Write + Send + 'static,
{
    let (inbox_tx, inbox_rx) = channel::<Inbound>();
    let (out_tx, out_rx) = channel::<Reply>();
    let writer = std::thread::Builder::new().name("mcp-stdio-out".into()).spawn(move || {
        let mut out = output;
        while let Ok(reply) = out_rx.recv() {
            if let Reply::Message { body, .. } = reply
                && out.write_all(&body).and_then(|()| out.write_all(b"\n")).and_then(|()| out.flush()).is_err()
            {
                return;
            }
        }
    })?;
    let reader = std::thread::Builder::new().name("mcp-stdio-in".into()).spawn(move || {
        let mut input = BufReader::with_capacity(64 << 10, input);
        let mut line = Vec::new();
        let max = usize::try_from(limits.max_bytes).unwrap_or(usize::MAX);
        loop {
            match read_line(&mut input, max, &mut line) {
                Ok(None) | Err(_) => return,
                Ok(Some(Err(()))) => {
                    let e = RpcError::new(PARSE_ERROR, "the message is larger than the limit");
                    let _ = out_tx.send(Reply::Message {
                        body: encode(&error_response(None, &e)),
                        status: 400,
                        session: None,
                    });
                }
                Ok(Some(Ok(()))) => {
                    let text = line.strip_suffix(b"\r").unwrap_or(&line);
                    if text.iter().all(u8::is_ascii_whitespace) {
                        continue;
                    }
                    match parse_message(text, &limits) {
                        Ok(incoming) => {
                            let env = Envelope {
                                incoming,
                                transport: Transport::Stdio,
                                reply: out_tx.clone(),
                                deadline: Instant::now() + STDIO_DEADLINE,
                            };
                            if inbox_tx.send(Inbound::Message(env)).is_err() {
                                return;
                            }
                        }
                        Err(Malformed { id, error }) => {
                            let body = encode(&error_response(id.as_ref(), &error));
                            let _ = out_tx.send(Reply::Message { body, status: 400, session: None });
                        }
                    }
                }
            }
        }
    })?;
    Ok((inbox_rx, StdioThreads { reader: Some(reader), writer: Some(writer) }))
}

/// Settings of the HTTP transport `[SEC-01]`, `[ROB-02]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HttpConfig {
    /// Request limits.
    pub limits: HttpLimits,
    /// Message limits.
    pub rpc: RpcLimits,
    /// Most connections served at once.
    pub max_connections: usize,
    /// Requests the rate limiter lets through at once.
    pub burst: u32,
    /// Requests per second it refills.
    pub per_second: u32,
    /// Longest wait for the server's answer (a paused agent's calls wait this long at most).
    pub reply_timeout: Duration,
    /// Keep-alive comment interval on event streams.
    pub keep_alive: Duration,
}

impl HttpConfig {
    /// 16 connections, a burst of 200 and 100 requests per second, 5 minutes for an answer,
    /// keep-alive every 15 s.
    pub const DEFAULT: Self = Self {
        limits: HttpLimits::DEFAULT,
        rpc: RpcLimits::DEFAULT,
        max_connections: 16,
        burst: 200,
        per_second: 100,
        reply_timeout: Duration::from_secs(300),
        keep_alive: Duration::from_secs(15),
    };
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// The MCP endpoint path.
pub const ENDPOINT: &str = "/mcp";

struct Shared {
    token: Token,
    port: u16,
    config: HttpConfig,
    limiter: RateLimiter,
    active: AtomicUsize,
    stop: AtomicBool,
    inbox: std::sync::Mutex<Sender<Inbound>>,
}

/// The running HTTP transport.
pub struct HttpServer {
    addr: SocketAddr,
    shared: Arc<Shared>,
    accept: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for HttpServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpServer").field("addr", &self.addr).finish_non_exhaustive()
    }
}

impl HttpServer {
    /// Binds `127.0.0.1:port` (0: ephemeral) and serves until [`HttpServer::stop`]. Messages go to
    /// `inbox`.
    ///
    /// # Errors
    /// When the port cannot be bound or the thread not started.
    pub fn start(port: u16, token: Token, config: HttpConfig, inbox: Sender<Inbound>) -> std::io::Result<Self> {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, port))?;
        let addr = listener.local_addr()?;
        let shared = Arc::new(Shared {
            token,
            port: addr.port(),
            config,
            limiter: RateLimiter::new(config.burst, config.per_second),
            active: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            inbox: std::sync::Mutex::new(inbox),
        });
        let s = Arc::clone(&shared);
        let accept =
            std::thread::Builder::new().name("mcp-http-accept".into()).spawn(move || accept_loop(&listener, &s))?;
        Ok(Self { addr, shared, accept: Some(accept) })
    }

    /// The bound address (127.0.0.1 and the port).
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// The endpoint URL.
    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}{ENDPOINT}", self.addr.port())
    }

    /// Stops accepting; open event streams end within a second.
    pub fn stop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect_timeout(&self.addr, Duration::from_millis(200));
        if let Some(a) = self.accept.take() {
            let _ = a.join();
        }
    }
}

impl Drop for HttpServer {
    fn drop(&mut self) {
        self.stop();
    }
}

fn accept_loop(listener: &TcpListener, shared: &Arc<Shared>) {
    for stream in listener.incoming() {
        if shared.stop.load(Ordering::SeqCst) {
            return;
        }
        let Ok(mut stream) = stream else { continue };
        if shared.active.load(Ordering::SeqCst) >= shared.config.max_connections {
            let _ = write_response(&mut stream, 503, "Service Unavailable", &[("Retry-After", "1")], b"");
            continue;
        }
        shared.active.fetch_add(1, Ordering::SeqCst);
        let s = Arc::clone(shared);
        let spawned = std::thread::Builder::new().name("mcp-http-conn".into()).spawn(move || {
            serve_connection(&mut stream, &s);
            linger_close(&mut stream);
            s.active.fetch_sub(1, Ordering::SeqCst);
        });
        if spawned.is_err() {
            shared.active.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

fn json_error(out: &mut TcpStream, status: u16, reason: &str, code: i64, message: &str, extra: &[(&str, &str)]) {
    let body = encode(&error_response(None, &RpcError::new(code, message)));
    let mut headers = vec![("Content-Type", "application/json")];
    headers.extend_from_slice(extra);
    let _ = write_response(out, status, reason, &headers, &body);
}

/// Decodes the `=?base64?...?=` sentinel form of `Mcp-Name`.
fn decode_name(value: &str) -> Option<String> {
    match value.strip_prefix("=?base64?").and_then(|r| r.strip_suffix("?=")) {
        Some(b64) => {
            humanart_io::gltf::base64::decode(b64.as_bytes(), 4096).ok().and_then(|b| String::from_utf8(b).ok())
        }
        None => Some(value.to_owned()),
    }
}

fn header(req: &Request, name: &str) -> Result<Option<String>, HttpError> {
    req.header(name).map(|v| v.map(str::to_owned))
}

/// Most bytes [`linger_close`] drains.
const LINGER_BYTES: usize = 64 * 1024;
/// Longest time [`linger_close`] waits for the client to finish sending.
const LINGER_TIME: Duration = Duration::from_millis(250);

/// Closes a connection without losing the response: a request refused before its body was read
/// (408, 413, ...) leaves bytes unread, and closing then makes the OS send a reset that can
/// destroy the response before the client reads it. Shutting down the write side first and
/// draining what is left (bounded in bytes and time) lets the client read the answer.
fn linger_close(stream: &mut TcpStream) {
    let _ = stream.shutdown(Shutdown::Write);
    let _ = stream.set_read_timeout(Some(Duration::from_millis(50)));
    let end = Instant::now() + LINGER_TIME;
    let mut drained = 0usize;
    let mut buf = [0u8; 4096];
    while drained < LINGER_BYTES && Instant::now() < end {
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => drained += n,
        }
    }
}

fn serve_connection(stream: &mut TcpStream, shared: &Shared) {
    let config = shared.config;
    let _ = stream.set_read_timeout(Some(Duration::from_millis(100)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(10)));
    let _ = stream.set_nodelay(true);
    let req = match read_request(stream, &config.limits) {
        Ok(r) => r,
        Err(e) => {
            if let Some((status, reason)) = e.status() {
                json_error(stream, status, reason, INVALID_REQUEST, &e.to_string(), &[]);
            }
            return;
        }
    };
    // [SEC-01] DNS rebinding and cross-origin requests.
    let (host, origin) = match (req.header("host"), req.header("origin")) {
        (Ok(h), Ok(o)) => (h.unwrap_or(""), o),
        _ => return json_error(stream, 400, "Bad Request", INVALID_REQUEST, "repeated Host or Origin", &[]),
    };
    if !check_host(host, shared.port) {
        return json_error(
            stream,
            403,
            "Forbidden",
            INVALID_REQUEST,
            "the Host header is not this loopback server",
            &[],
        );
    }
    if !check_origin(origin, shared.port) {
        return json_error(stream, 403, "Forbidden", INVALID_REQUEST, "cross-origin requests are refused", &[]);
    }
    if !shared.limiter.allow() {
        return json_error(stream, 429, "Too Many Requests", INVALID_REQUEST, "rate limit", &[("Retry-After", "1")]);
    }
    let auth = req.header("authorization").ok().flatten();
    if !shared.token.matches_header(auth) {
        return json_error(
            stream,
            401,
            "Unauthorized",
            INVALID_REQUEST,
            "a valid bearer token is required",
            &[("WWW-Authenticate", "Bearer")],
        );
    }
    if req.path != ENDPOINT {
        return json_error(stream, 404, "Not Found", INVALID_REQUEST, "the MCP endpoint is /mcp", &[]);
    }
    let Ok(session) = header(&req, "mcp-session-id") else {
        return json_error(stream, 400, "Bad Request", INVALID_REQUEST, "repeated Mcp-Session-Id", &[]);
    };
    let Ok(inbox) = shared.inbox.lock().map(|g| g.clone()) else { return };
    match req.method.as_str() {
        "POST" => {}
        "GET" | "DELETE" if session.is_some() => {
            let (tx, rx) = channel();
            let s = session.unwrap_or_default();
            let inbound = if req.method == "GET" {
                Inbound::LegacyStream { session: s, reply: tx }
            } else {
                Inbound::LegacyEnd { session: s, reply: tx }
            };
            if inbox.send(inbound).is_err() {
                return json_error(stream, 503, "Service Unavailable", INVALID_REQUEST, "the server is stopping", &[]);
            }
            return answer(stream, &rx, shared);
        }
        _ => return json_error(stream, 405, "Method Not Allowed", INVALID_REQUEST, "use POST", &[("Allow", "POST")]),
    }
    let content_type = req.header("content-type").ok().flatten().unwrap_or("");
    let mime = content_type.split(';').next().unwrap_or("").trim();
    if !mime.eq_ignore_ascii_case("application/json") {
        return json_error(
            stream,
            415,
            "Unsupported Media Type",
            INVALID_REQUEST,
            "the body must be application/json",
            &[],
        );
    }
    if let Ok(Some(accept)) = req.header("accept") {
        let a = accept.to_ascii_lowercase();
        if !(a.contains("application/json") || a.contains("text/event-stream") || a.contains("*/*")) {
            return json_error(
                stream,
                406,
                "Not Acceptable",
                INVALID_REQUEST,
                "accept application/json or text/event-stream",
                &[],
            );
        }
    }
    let incoming = match parse_message(&req.body, &config.rpc) {
        Ok(m) => m,
        Err(Malformed { id, error }) => {
            let body = encode(&error_response(id.as_ref(), &error));
            let _ = write_response(stream, 400, "Bad Request", &[("Content-Type", "application/json")], &body);
            return;
        }
    };
    let mirror = Mirror {
        protocol_version: header(&req, "mcp-protocol-version").ok().flatten(),
        method: header(&req, "mcp-method").ok().flatten(),
        name: header(&req, "mcp-name").ok().flatten().and_then(|n| decode_name(&n)),
        session,
    };
    let (tx, rx) = channel();
    let env = Envelope {
        incoming,
        transport: Transport::Http(mirror),
        reply: tx,
        deadline: Instant::now() + config.reply_timeout,
    };
    if inbox.send(Inbound::Message(env)).is_err() {
        return json_error(stream, 503, "Service Unavailable", INVALID_REQUEST, "the server is stopping", &[]);
    }
    answer(stream, &rx, shared);
}

/// Writes the server's answer: one response, or an event stream until it ends.
fn answer(stream: &mut TcpStream, rx: &Receiver<Reply>, shared: &Shared) {
    let config = shared.config;
    let first = match rx.recv_timeout(config.reply_timeout) {
        Ok(r) => r,
        Err(RecvTimeoutError::Timeout) => {
            return json_error(stream, 504, "Gateway Timeout", TIMED_OUT, "no answer in time", &[]);
        }
        Err(RecvTimeoutError::Disconnected) => {
            return json_error(stream, 500, "Internal Server Error", TIMED_OUT, "the server dropped the request", &[]);
        }
    };
    match first {
        Reply::Message { body, status, session } => {
            let reason = match status {
                200 => "OK",
                400 => "Bad Request",
                403 => "Forbidden",
                404 => "Not Found",
                504 => "Gateway Timeout",
                _ => "Status",
            };
            let mut headers = vec![("Content-Type", "application/json")];
            if let Some(s) = session.as_deref() {
                headers.push(("Mcp-Session-Id", s));
            }
            let _ = write_response(stream, status, reason, &headers, &body);
        }
        Reply::Accepted => {
            let _ = write_response(stream, 202, "Accepted", &[], b"");
        }
        Reply::Status(code) => {
            let _ = write_response(stream, code, if code == 200 { "OK" } else { "Status" }, &[], b"");
        }
        Reply::StreamOpen => {
            if write_sse_head(stream).is_err() {
                return;
            }
            let tick = Duration::from_millis(250);
            let mut quiet = Duration::ZERO;
            loop {
                if shared.stop.load(Ordering::SeqCst) {
                    return;
                }
                match rx.recv_timeout(tick) {
                    Ok(Reply::Message { body, .. }) => {
                        quiet = Duration::ZERO;
                        if write_sse_event(stream, &body).is_err() {
                            return;
                        }
                    }
                    Ok(_) => {}
                    Err(RecvTimeoutError::Timeout) => {
                        quiet += tick;
                        if quiet >= config.keep_alive {
                            quiet = Duration::ZERO;
                            if write_sse_comment(stream).is_err() {
                                return;
                            }
                        }
                    }
                    Err(RecvTimeoutError::Disconnected) => return,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stdio_lines_are_bounded() {
        let data = b"short\nthis line is far too long\nok\r\nlast".to_vec();
        let mut r = BufReader::with_capacity(4, std::io::Cursor::new(data));
        let mut line = Vec::new();
        assert_eq!(read_line(&mut r, 10, &mut line).unwrap(), Some(Ok(())));
        assert_eq!(line, b"short");
        assert_eq!(read_line(&mut r, 10, &mut line).unwrap(), Some(Err(())));
        assert_eq!(read_line(&mut r, 10, &mut line).unwrap(), Some(Ok(())));
        assert_eq!(line, b"ok\r");
        assert_eq!(read_line(&mut r, 10, &mut line).unwrap(), Some(Ok(())));
        assert_eq!(line, b"last");
        assert_eq!(read_line(&mut r, 10, &mut line).unwrap(), None);
    }

    #[test]
    fn mcp_name_sentinel_decodes() {
        assert_eq!(decode_name("render_view").as_deref(), Some("render_view"));
        let enc = format!("=?base64?{}?=", humanart_io::gltf::base64::encode("Hello, 世界".as_bytes()));
        assert_eq!(decode_name(&enc).as_deref(), Some("Hello, 世界"));
        assert_eq!(decode_name("=?base64?!!!?="), None);
    }
}
