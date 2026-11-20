//! The in-house MCP client `[AC-WP23-1]`: speaks revision 2026-07-28 to the server over HTTP and
//! stdio, for the scripted session tests, the CLI's smoke checks and anyone driving the server
//! from Rust. Every request carries the per-request `_meta` and, over HTTP, the mirrored headers.

use std::fmt::Write as _;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::time::Duration;

use humanart_io::json::{Object, Value, parse_str};

use super::http::{HttpLimits, Response, read_response_head};
use super::jsonrpc::{Id, encode, request};
use super::server::MODERN_VERSIONS;

/// What can go wrong on the client side.
#[derive(Debug)]
pub enum ClientError {
    /// The connection failed.
    Io(std::io::Error),
    /// The answer is not what the protocol says.
    Protocol(String),
    /// The server answered with a JSON-RPC error (`code`, `message`) and this HTTP status.
    Rpc {
        /// The HTTP status (200 on stdio).
        status: u16,
        /// The error code.
        code: i64,
        /// The message.
        message: String,
    },
    /// No answer in time.
    Timeout,
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "connection: {e}"),
            Self::Protocol(m) => write!(f, "protocol: {m}"),
            Self::Rpc { status, code, message } => write!(f, "error {code} (HTTP {status}): {message}"),
            Self::Timeout => f.write_str("no answer in time"),
        }
    }
}

impl std::error::Error for ClientError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ClientError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// `params` with the 2026-07-28 request `_meta` added.
pub fn with_meta(params: Value) -> Value {
    let mut p = match params {
        Value::Object(o) => o,
        _ => Object::new(),
    };
    let mut meta = Object::with_capacity(3);
    meta.insert("io.modelcontextprotocol/protocolVersion", MODERN_VERSIONS[0]);
    let mut info = Object::with_capacity(2);
    info.insert("name", "humanart-test-client");
    info.insert("version", env!("CARGO_PKG_VERSION"));
    meta.insert("io.modelcontextprotocol/clientInfo", Value::Object(info));
    meta.insert("io.modelcontextprotocol/clientCapabilities", Value::Object(Object::new()));
    p.insert("_meta", Value::Object(meta));
    Value::Object(p)
}

/// The `result` of a response message, or its error.
fn result_of(message: Value, status: u16) -> Result<Value, ClientError> {
    if let Some(e) = message.get("error") {
        return Err(ClientError::Rpc {
            status,
            code: e.get("code").and_then(Value::as_i64).unwrap_or(0),
            message: e.get("message").and_then(Value::as_str).unwrap_or("").to_owned(),
        });
    }
    match message {
        Value::Object(mut o) => {
            o.remove("result").ok_or_else(|| ClientError::Protocol("a response without result".into()))
        }
        _ => Err(ClientError::Protocol("a response is an object".into())),
    }
}

/// An MCP client over HTTP.
#[derive(Debug)]
pub struct HttpClient {
    addr: SocketAddr,
    token: String,
    next: i64,
    /// Extra headers sent with every request (tests: spoofed Host / Origin).
    pub extra: Vec<(String, String)>,
}

impl HttpClient {
    /// A client of the server at `addr` with bearer `token`.
    pub fn new(addr: SocketAddr, token: &str) -> Self {
        Self { addr, token: token.to_owned(), next: 0, extra: Vec::new() }
    }

    /// Sends one raw request (head lines without the request line, body) and reads the whole
    /// response.
    ///
    /// # Errors
    /// [`ClientError::Io`] or [`ClientError::Protocol`].
    pub fn raw(&self, method: &str, headers: &[(&str, &str)], body: &[u8]) -> Result<Response, ClientError> {
        let mut s = TcpStream::connect(self.addr)?;
        s.set_read_timeout(Some(Duration::from_secs(120)))?;
        let mut head = format!("{method} /mcp HTTP/1.1\r\n");
        for (k, v) in headers {
            let _ = write!(head, "{k}: {v}\r\n");
        }
        let _ = write!(head, "Content-Length: {}\r\n\r\n", body.len());
        s.write_all(head.as_bytes())?;
        s.write_all(body)?;
        let limits = HttpLimits { deadline: Duration::from_secs(120), ..HttpLimits::DEFAULT };
        let (mut resp, mut rest) =
            read_response_head(&mut s, &limits).map_err(|e| ClientError::Protocol(e.to_string()))?;
        let len: Option<usize> = resp.header("content-length").and_then(|l| l.parse().ok());
        let mut chunk = [0u8; 8192];
        while len.is_none_or(|l| rest.len() < l) {
            match s.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => rest.extend_from_slice(&chunk[..n]),
                Err(e) => return Err(e.into()),
            }
        }
        resp.body = rest;
        Ok(resp)
    }

    fn headers(&self, method: &str, name: Option<&str>) -> Vec<(String, String)> {
        let mut h = vec![
            ("Host".to_owned(), format!("127.0.0.1:{}", self.addr.port())),
            ("Authorization".to_owned(), format!("Bearer {}", self.token)),
            ("Content-Type".to_owned(), "application/json".to_owned()),
            ("Accept".to_owned(), "application/json, text/event-stream".to_owned()),
            ("MCP-Protocol-Version".to_owned(), MODERN_VERSIONS[0].to_owned()),
            ("Mcp-Method".to_owned(), method.to_owned()),
        ];
        if let Some(n) = name {
            h.push(("Mcp-Name".to_owned(), n.to_owned()));
        }
        for (k, v) in &self.extra {
            h.retain(|(x, _)| !x.eq_ignore_ascii_case(k));
            h.push((k.clone(), v.clone()));
        }
        h
    }

    /// Sends a request and returns its `result`.
    ///
    /// # Errors
    /// See [`ClientError`].
    pub fn request(&mut self, method: &str, params: Value) -> Result<Value, ClientError> {
        self.next += 1;
        let name = params.get("name").or_else(|| params.get("uri")).and_then(Value::as_str).map(str::to_owned);
        let body = encode(&request(&Id::Number(self.next), method, with_meta(params)));
        let headers = self.headers(method, name.as_deref());
        let refs: Vec<(&str, &str)> = headers.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        let resp = self.raw("POST", &refs, &body)?;
        let text = String::from_utf8(resp.body).map_err(|_| ClientError::Protocol("the body is not UTF-8".into()))?;
        let message =
            parse_str(&text).map_err(|_| ClientError::Protocol(format!("HTTP {}: not JSON: {text}", resp.status)))?;
        result_of(message, resp.status)
    }

    /// Calls a tool and returns its result (`content`, `structuredContent`, `isError`).
    ///
    /// # Errors
    /// See [`ClientError`].
    pub fn call_tool(&mut self, name: &str, arguments: Value) -> Result<Value, ClientError> {
        let mut p = Object::with_capacity(2);
        p.insert("name", name);
        p.insert("arguments", arguments);
        self.request("tools/call", Value::Object(p))
    }

    /// Opens a `subscriptions/listen` stream with `filter` and returns it once acknowledged.
    ///
    /// # Errors
    /// See [`ClientError`].
    pub fn listen(&mut self, filter: Value) -> Result<EventStream, ClientError> {
        self.next += 1;
        let mut p = Object::with_capacity(1);
        p.insert("notifications", filter);
        let body = encode(&request(&Id::Number(self.next), "subscriptions/listen", with_meta(Value::Object(p))));
        let headers = self.headers("subscriptions/listen", None);
        let mut s = TcpStream::connect(self.addr)?;
        let mut head = "POST /mcp HTTP/1.1\r\n".to_owned();
        for (k, v) in &headers {
            let _ = write!(head, "{k}: {v}\r\n");
        }
        let _ = write!(head, "Content-Length: {}\r\n\r\n", body.len());
        s.write_all(head.as_bytes())?;
        s.write_all(&body)?;
        s.set_read_timeout(Some(Duration::from_secs(30)))?;
        let (resp, rest) =
            read_response_head(&mut s, &HttpLimits::DEFAULT).map_err(|e| ClientError::Protocol(e.to_string()))?;
        if resp.status != 200 || resp.header("content-type") != Some("text/event-stream") {
            return Err(ClientError::Protocol(format!("listen answered HTTP {}", resp.status)));
        }
        let (tx, rx) = channel();
        std::thread::Builder::new().name("mcp-client-sse".into()).spawn(move || {
            let mut reader = BufReader::new(std::io::Cursor::new(rest).chain(s));
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => return,
                    Ok(_) => {
                        if let Some(data) = line.trim_end().strip_prefix("data: ")
                            && let Ok(v) = parse_str(data)
                            && tx.send(v).is_err()
                        {
                            return;
                        }
                    }
                }
            }
        })?;
        let mut stream = EventStream { rx };
        let ack = stream.next(Duration::from_secs(30)).ok_or(ClientError::Timeout)?;
        if ack.get("method").and_then(Value::as_str) != Some("notifications/subscriptions/acknowledged") {
            return Err(ClientError::Protocol("the first event is not the acknowledgement".into()));
        }
        Ok(stream)
    }
}

/// Messages arriving on a subscription (HTTP event stream) or on stdio.
#[derive(Debug)]
pub struct EventStream {
    rx: Receiver<Value>,
}

impl EventStream {
    /// The next message, waiting up to `timeout`.
    pub fn next(&mut self, timeout: Duration) -> Option<Value> {
        self.rx.recv_timeout(timeout).ok()
    }

    /// Waits up to `timeout` for a message `pick` accepts, dropping the others.
    pub fn wait_for(&mut self, timeout: Duration, pick: impl Fn(&Value) -> bool) -> Option<Value> {
        let end = std::time::Instant::now() + timeout;
        loop {
            let left = end.checked_duration_since(std::time::Instant::now())?;
            let v = self.next(left)?;
            if pick(&v) {
                return Some(v);
            }
        }
    }
}

/// An MCP client over a pair of byte streams (a child process's stdin / stdout, or pipes).
pub struct StdioClient<W: Write> {
    out: W,
    rx: Receiver<Value>,
    next: i64,
    /// Notifications that arrived while waiting for responses.
    pub notifications: Vec<Value>,
}

impl<W: Write> std::fmt::Debug for StdioClient<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StdioClient").field("next", &self.next).finish_non_exhaustive()
    }
}

impl<W: Write> StdioClient<W> {
    /// A client writing requests to `out` and reading messages from `input` (on a thread).
    ///
    /// # Errors
    /// When the reader thread cannot start.
    pub fn new<R: Read + Send + 'static>(input: R, out: W) -> Result<Self, ClientError> {
        let (tx, rx) = channel();
        std::thread::Builder::new().name("mcp-client-stdio".into()).spawn(move || {
            for line in BufReader::new(input).lines() {
                let Ok(line) = line else { return };
                if let Ok(v) = parse_str(&line)
                    && tx.send(v).is_err()
                {
                    return;
                }
            }
        })?;
        Ok(Self { out, rx, next: 0, notifications: Vec::new() })
    }

    /// Writes one raw line.
    ///
    /// # Errors
    /// The writer's error.
    pub fn send_line(&mut self, line: &[u8]) -> Result<(), ClientError> {
        self.out.write_all(line)?;
        self.out.write_all(b"\n")?;
        self.out.flush()?;
        Ok(())
    }

    /// The next message of any kind, waiting up to `timeout`.
    pub fn next_message(&mut self, timeout: Duration) -> Option<Value> {
        self.rx.recv_timeout(timeout).ok()
    }

    /// Sends a request and waits (up to `timeout`) for its response; notifications in between
    /// are kept in [`StdioClient::notifications`].
    ///
    /// # Errors
    /// See [`ClientError`].
    pub fn request_within(&mut self, method: &str, params: Value, timeout: Duration) -> Result<Value, ClientError> {
        self.next += 1;
        let id = self.next;
        self.send_line(&encode(&request(&Id::Number(id), method, with_meta(params))))?;
        let end = std::time::Instant::now() + timeout;
        loop {
            let left = end.checked_duration_since(std::time::Instant::now()).ok_or(ClientError::Timeout)?;
            let v = match self.rx.recv_timeout(left) {
                Ok(v) => v,
                Err(RecvTimeoutError::Timeout) => return Err(ClientError::Timeout),
                Err(RecvTimeoutError::Disconnected) => return Err(ClientError::Protocol("the server closed".into())),
            };
            if v.get("id").and_then(Value::as_i64) == Some(id) {
                return result_of(v, 200);
            }
            if v.get("method").is_some() {
                self.notifications.push(v);
            }
        }
    }

    /// [`StdioClient::request_within`] with 120 s.
    ///
    /// # Errors
    /// See [`ClientError`].
    pub fn request(&mut self, method: &str, params: Value) -> Result<Value, ClientError> {
        self.request_within(method, params, Duration::from_secs(120))
    }

    /// Calls a tool.
    ///
    /// # Errors
    /// See [`ClientError`].
    pub fn call_tool(&mut self, name: &str, arguments: Value) -> Result<Value, ClientError> {
        let mut p = Object::with_capacity(2);
        p.insert("name", name);
        p.insert("arguments", arguments);
        self.request("tools/call", Value::Object(p))
    }

    /// Waits up to `timeout` for a notification `pick` accepts (looking at the kept ones first).
    pub fn wait_notification(&mut self, timeout: Duration, pick: impl Fn(&Value) -> bool) -> Option<Value> {
        if let Some(i) = self.notifications.iter().position(&pick) {
            return Some(self.notifications.remove(i));
        }
        let end = std::time::Instant::now() + timeout;
        loop {
            let left = end.checked_duration_since(std::time::Instant::now())?;
            let v = self.rx.recv_timeout(left).ok()?;
            if pick(&v) {
                return Some(v);
            }
            if v.get("method").is_some() {
                self.notifications.push(v);
            }
        }
    }
}
