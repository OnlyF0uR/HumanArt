//! A strict HTTP/1.1 request reader and response writer on `std::io` `[AG-01]`, `[ROB-03]`.
//!
//! Only what the MCP endpoint needs: origin-form targets, `Content-Length` bodies (no chunked
//! request bodies: `Transfer-Encoding` is refused with 501), one request per connection
//! (`Connection: close` on every response). Everything is checked before it is used:
//!
//! - the head (request line and headers) is at most [`HttpLimits::max_head_bytes`] and
//!   [`HttpLimits::max_headers`] headers; lines end in CRLF (a bare LF is refused), header names
//!   are RFC 9110 tokens with no space before the colon, values are visible ASCII, space or tab
//!   (no obs-fold, no control bytes, no bytes above 0x7E);
//! - `Host` must appear exactly once, `Content-Length` at most once and only as digits, and the
//!   body is at most [`HttpLimits::max_body_bytes`] (413 before anything is allocated);
//! - reading stops at [`HttpLimits::deadline`] in total, whatever the socket's per-read timeout:
//!   a client that trickles bytes (slow loris) gets 408.

use std::io::{Read, Write};
use std::time::{Duration, Instant};

/// Size and time limits of one request `[ROB-02]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HttpLimits {
    /// Largest request line plus headers.
    pub max_head_bytes: usize,
    /// Most header lines.
    pub max_headers: usize,
    /// Largest body.
    pub max_body_bytes: usize,
    /// Longest request target.
    pub max_target_bytes: usize,
    /// Longest time to read one whole request.
    pub deadline: Duration,
}

impl HttpLimits {
    /// 16 KiB head, 64 headers, 4 MiB body, 2 KiB target, 10 s per request.
    pub const DEFAULT: Self = Self {
        max_head_bytes: 16 << 10,
        max_headers: 64,
        max_body_bytes: 4 << 20,
        max_target_bytes: 2048,
        deadline: Duration::from_secs(10),
    };
}

impl Default for HttpLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Why a request could not be read; each maps to a status code ([`HttpError::status`]).
#[derive(Debug)]
pub enum HttpError {
    /// The request is malformed (400).
    BadRequest(&'static str),
    /// The head is larger than the limit (431).
    HeadTooLarge,
    /// The body is larger than the limit (413).
    BodyTooLarge,
    /// A feature this server does not implement, e.g. `Transfer-Encoding` (501).
    NotImplemented(&'static str),
    /// The request did not arrive in time (408).
    Timeout,
    /// The client closed the connection before a whole request arrived (no response).
    Closed,
    /// The socket failed (no response).
    Io(std::io::Error),
}

impl HttpError {
    /// The status code and reason phrase to answer with, or `None` when nothing can be sent.
    pub const fn status(&self) -> Option<(u16, &'static str)> {
        match self {
            Self::BadRequest(_) => Some((400, "Bad Request")),
            Self::HeadTooLarge => Some((431, "Request Header Fields Too Large")),
            Self::BodyTooLarge => Some((413, "Content Too Large")),
            Self::NotImplemented(_) => Some((501, "Not Implemented")),
            Self::Timeout => Some((408, "Request Timeout")),
            Self::Closed | Self::Io(_) => None,
        }
    }
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadRequest(why) => write!(f, "bad request: {why}"),
            Self::HeadTooLarge => f.write_str("the request head is too large"),
            Self::BodyTooLarge => f.write_str("the request body is too large"),
            Self::NotImplemented(what) => write!(f, "not implemented: {what}"),
            Self::Timeout => f.write_str("the request did not arrive in time"),
            Self::Closed => f.write_str("the client closed the connection"),
            Self::Io(e) => write!(f, "socket error: {e}"),
        }
    }
}

impl std::error::Error for HttpError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

/// One request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    /// The method (`GET`, `POST` ...).
    pub method: String,
    /// The origin-form target (`/mcp`), without the query.
    pub path: String,
    /// The query after `?`, if any.
    pub query: Option<String>,
    /// The headers in order, names as sent.
    pub headers: Vec<(String, String)>,
    /// The body.
    pub body: Vec<u8>,
}

impl Request {
    /// The value of header `name` (case-insensitive), when it appears once; `Err` when it appears
    /// more than once.
    pub fn header(&self, name: &str) -> Result<Option<&str>, HttpError> {
        let mut found = None;
        for (n, v) in &self.headers {
            if n.eq_ignore_ascii_case(name) {
                if found.is_some() {
                    return Err(HttpError::BadRequest("a header that must be single is repeated"));
                }
                found = Some(v.as_str());
            }
        }
        Ok(found)
    }

    /// Whether any header `name` (case-insensitive) exists.
    pub fn has_header(&self, name: &str) -> bool {
        self.headers.iter().any(|(n, _)| n.eq_ignore_ascii_case(name))
    }
}

const fn is_tchar(b: u8) -> bool {
    b.is_ascii_alphanumeric()
        || matches!(
            b,
            b'!' | b'#' | b'$' | b'%' | b'&' | b'\'' | b'*' | b'+' | b'-' | b'.' | b'^' | b'_' | b'`' | b'|' | b'~'
        )
}

fn is_value_byte(b: u8) -> bool {
    b == b'\t' || (0x20..0x7F).contains(&b)
}

/// Parses a complete head (everything before the blank line, without the final CRLF CRLF) into a
/// request without body, and the declared body length.
///
/// # Errors
/// [`HttpError::BadRequest`], [`HttpError::HeadTooLarge`], [`HttpError::BodyTooLarge`],
/// [`HttpError::NotImplemented`].
pub fn parse_head(head: &[u8], limits: &HttpLimits) -> Result<(Request, usize), HttpError> {
    if head.len() > limits.max_head_bytes {
        return Err(HttpError::HeadTooLarge);
    }
    let mut lines = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < head.len() {
        match head[i] {
            b'\r' => {
                if head.get(i + 1) != Some(&b'\n') {
                    return Err(HttpError::BadRequest("a CR is not followed by LF"));
                }
                lines.push(&head[start..i]);
                i += 2;
                start = i;
                if lines.len() > limits.max_headers + 1 {
                    return Err(HttpError::HeadTooLarge);
                }
            }
            b'\n' => return Err(HttpError::BadRequest("a line ends in a bare LF")),
            _ => i += 1,
        }
    }
    lines.push(&head[start..]);
    if lines.len() > limits.max_headers + 1 {
        return Err(HttpError::HeadTooLarge);
    }
    let (request_line, header_lines) = lines.split_first().ok_or(HttpError::BadRequest("empty request"))?;
    let mut parts = request_line.split(|&b| b == b' ');
    let (Some(method), Some(target), Some(version), None) = (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(HttpError::BadRequest("the request line is not 'METHOD target HTTP/1.1'"));
    };
    if method.is_empty() || method.len() > 16 || !method.iter().all(|&b| is_tchar(b)) {
        return Err(HttpError::BadRequest("invalid method"));
    }
    if version != b"HTTP/1.1" {
        return Err(HttpError::BadRequest("only HTTP/1.1 is served"));
    }
    if target.len() > limits.max_target_bytes {
        return Err(HttpError::BadRequest("the target is too long"));
    }
    if target.first() != Some(&b'/') || !target.iter().all(|b| (0x21..0x7F).contains(b)) {
        return Err(HttpError::BadRequest("the target must be an origin-form path"));
    }
    let target = String::from_utf8(target.to_vec()).map_err(|_| HttpError::BadRequest("invalid target"))?;
    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p.to_owned(), Some(q.to_owned())),
        None => (target, None),
    };
    let mut headers = Vec::with_capacity(header_lines.len());
    for line in header_lines {
        let colon = line.iter().position(|&b| b == b':').ok_or(HttpError::BadRequest("a header line has no colon"))?;
        let (name, rest) = line.split_at(colon);
        if name.is_empty() || !name.iter().all(|&b| is_tchar(b)) {
            return Err(HttpError::BadRequest("invalid header name (or obs-fold, or space before the colon)"));
        }
        let value = &rest[1..];
        if !value.iter().all(|&b| is_value_byte(b)) {
            return Err(HttpError::BadRequest("invalid byte in a header value"));
        }
        let value = value.trim_ascii();
        // Both are ASCII by the checks above.
        headers.push((String::from_utf8_lossy(name).into_owned(), String::from_utf8_lossy(value).into_owned()));
    }
    let request =
        Request { method: String::from_utf8_lossy(method).into_owned(), path, query, headers, body: Vec::new() };
    if request.has_header("transfer-encoding") {
        return Err(HttpError::NotImplemented("Transfer-Encoding request bodies"));
    }
    match request.header("host")? {
        Some(h) if !h.is_empty() => {}
        _ => return Err(HttpError::BadRequest("exactly one non-empty Host header is required")),
    }
    let length = match request.header("content-length")? {
        None => 0,
        Some(text) => {
            if text.is_empty() || text.len() > 12 || !text.bytes().all(|b| b.is_ascii_digit()) {
                return Err(HttpError::BadRequest("invalid Content-Length"));
            }
            let n: u64 = text.parse().map_err(|_| HttpError::BadRequest("invalid Content-Length"))?;
            if n > limits.max_body_bytes as u64 {
                return Err(HttpError::BodyTooLarge);
            }
            n as usize
        }
    };
    Ok((request, length))
}

fn find_head_end(buf: &[u8], from: usize) -> Option<usize> {
    let start = from.saturating_sub(3);
    buf.get(start..)?.windows(4).position(|w| w == b"\r\n\r\n").map(|p| start + p)
}

fn timed_out(e: &std::io::Error) -> bool {
    matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut)
}

/// Reads one request from `reader` within the limits. Set a short per-read timeout on a socket so
/// the total deadline is checked often.
///
/// # Errors
/// See [`HttpError`].
pub fn read_request<R: Read>(reader: &mut R, limits: &HttpLimits) -> Result<Request, HttpError> {
    let started = Instant::now();
    let mut buf: Vec<u8> = Vec::with_capacity(1024);
    let mut chunk = [0u8; 4096];
    let head_end = loop {
        if let Some(end) = find_head_end(&buf, 0) {
            break end;
        }
        if buf.len() > limits.max_head_bytes + 4 {
            return Err(HttpError::HeadTooLarge);
        }
        if started.elapsed() > limits.deadline {
            return Err(HttpError::Timeout);
        }
        let want = (limits.max_head_bytes + 4 - buf.len()).clamp(1, chunk.len());
        match reader.read(&mut chunk[..want]) {
            Ok(0) => {
                return Err(if buf.is_empty() {
                    HttpError::Closed
                } else {
                    HttpError::BadRequest("the head ends early")
                });
            }
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(e) if timed_out(&e) => {}
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(HttpError::Io(e)),
        }
    };
    let (mut request, length) = parse_head(&buf[..head_end], limits)?;
    let mut body = Vec::with_capacity(length);
    body.extend_from_slice(buf.get(head_end + 4..).unwrap_or(&[]));
    if body.len() > length {
        return Err(HttpError::BadRequest("more bytes than Content-Length (pipelining is not supported)"));
    }
    while body.len() < length {
        if started.elapsed() > limits.deadline {
            return Err(HttpError::Timeout);
        }
        let want = (length - body.len()).min(chunk.len());
        match reader.read(&mut chunk[..want]) {
            Ok(0) => return Err(HttpError::BadRequest("the body ends early")),
            Ok(n) => body.extend_from_slice(&chunk[..n]),
            Err(e) if timed_out(&e) => {}
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(HttpError::Io(e)),
        }
    }
    request.body = body;
    Ok(request)
}

/// Headers every response carries.
const COMMON: &str = "Cache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n";

/// Writes a whole response with a body and `Connection: close`.
///
/// # Errors
/// The writer's error.
pub fn write_response<W: Write>(
    out: &mut W,
    status: u16,
    reason: &str,
    headers: &[(&str, &str)],
    body: &[u8],
) -> std::io::Result<()> {
    let mut head =
        format!("HTTP/1.1 {status} {reason}\r\n{COMMON}Connection: close\r\nContent-Length: {}\r\n", body.len());
    for (name, value) in headers {
        head.push_str(name);
        head.push_str(": ");
        head.push_str(value);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    out.write_all(head.as_bytes())?;
    out.write_all(body)?;
    out.flush()
}

/// Writes the head of a Server-Sent Events response (the body follows as events).
///
/// # Errors
/// The writer's error.
pub fn write_sse_head<W: Write>(out: &mut W) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 200 OK\r\n{COMMON}Connection: close\r\nContent-Type: text/event-stream\r\nX-Accel-Buffering: no\r\n\r\n"
    );
    out.write_all(head.as_bytes())?;
    out.flush()
}

/// Writes one SSE `message` event whose data is `data` (one line: compact JSON).
///
/// # Errors
/// The writer's error.
pub fn write_sse_event<W: Write>(out: &mut W, data: &[u8]) -> std::io::Result<()> {
    out.write_all(b"event: message\ndata: ")?;
    out.write_all(data)?;
    out.write_all(b"\n\n")?;
    out.flush()
}

/// Writes an SSE comment (keep-alive).
///
/// # Errors
/// The writer's error.
pub fn write_sse_comment<W: Write>(out: &mut W) -> std::io::Result<()> {
    out.write_all(b":\n\n")?;
    out.flush()
}

/// A parsed response (the in-house client's side).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Response {
    /// The status code.
    pub status: u16,
    /// The headers.
    pub headers: Vec<(String, String)>,
    /// The body (everything until the connection closed, or `Content-Length` bytes).
    pub body: Vec<u8>,
}

impl Response {
    /// The first header `name` (case-insensitive).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(n, _)| n.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

/// Reads a response head from `reader` and returns it with the bytes read past the head.
///
/// # Errors
/// [`HttpError`] for a malformed or oversized head.
pub fn read_response_head<R: Read>(reader: &mut R, limits: &HttpLimits) -> Result<(Response, Vec<u8>), HttpError> {
    let started = Instant::now();
    let mut buf = Vec::with_capacity(1024);
    let mut chunk = [0u8; 4096];
    let end = loop {
        if let Some(end) = find_head_end(&buf, 0) {
            break end;
        }
        if buf.len() > limits.max_head_bytes || started.elapsed() > limits.deadline {
            return Err(HttpError::HeadTooLarge);
        }
        match reader.read(&mut chunk) {
            Ok(0) => return Err(HttpError::Closed),
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(e) if timed_out(&e) => {}
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(HttpError::Io(e)),
        }
    };
    let head = String::from_utf8_lossy(&buf[..end]).into_owned();
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap_or("");
    let status = status_line
        .strip_prefix("HTTP/1.1 ")
        .and_then(|r| r.get(..3))
        .and_then(|s| s.parse::<u16>().ok())
        .ok_or(HttpError::BadRequest("invalid status line"))?;
    let headers =
        lines.filter_map(|l| l.split_once(':')).map(|(n, v)| (n.trim().to_owned(), v.trim().to_owned())).collect();
    let rest = buf.get(end + 4..).unwrap_or(&[]).to_vec();
    Ok((Response { status, headers, body: Vec::new() }, rest))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(bytes: &[u8]) -> Result<Request, HttpError> {
        read_request(&mut std::io::Cursor::new(bytes.to_vec()), &HttpLimits::DEFAULT)
    }

    #[test]
    fn rob_03_a_well_formed_post_reads() {
        let r =
            read(b"POST /mcp?x=1 HTTP/1.1\r\nHost: 127.0.0.1:9\r\nContent-Length: 4\r\nX-A:  v  \r\n\r\nabcd").unwrap();
        assert_eq!((r.method.as_str(), r.path.as_str(), r.query.as_deref()), ("POST", "/mcp", Some("x=1")));
        assert_eq!(r.body, b"abcd");
        assert_eq!(r.header("x-a").unwrap(), Some("v"));
        assert_eq!(r.header("HOST").unwrap(), Some("127.0.0.1:9"));
        let g = read(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
        assert!(g.body.is_empty());
    }

    #[test]
    fn rob_03_malformed_requests_are_refused_with_their_status() {
        let status = |b: &[u8]| read(b).unwrap_err().status().map(|s| s.0);
        let cases: [(&[u8], u16); 15] = [
            (b"POST /mcp HTTP/1.0\r\nHost: a\r\n\r\n", 400),
            (b"POST mcp HTTP/1.1\r\nHost: a\r\n\r\n", 400),
            (b"POST  /mcp HTTP/1.1\r\nHost: a\r\n\r\n", 400),
            (b"POST /mcp HTTP/1.1\nHost: a\r\n\r\n", 400),
            (b"POST /mcp HTTP/1.1\r\nHost : a\r\n\r\n", 400),
            (b"POST /mcp HTTP/1.1\r\nHost: a\r\n folded\r\n\r\n", 400),
            (b"POST /mcp HTTP/1.1\r\n\r\n", 400),
            (b"POST /mcp HTTP/1.1\r\nHost: a\r\nHost: b\r\n\r\n", 400),
            (b"POST /mcp HTTP/1.1\r\nHost: a\r\nContent-Length: -1\r\n\r\n", 400),
            (b"POST /mcp HTTP/1.1\r\nHost: a\r\nContent-Length: 1\r\nContent-Length: 1\r\n\r\nx", 400),
            (b"POST /mcp HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n", 501),
            (b"POST /mcp HTTP/1.1\r\nHost: a\r\nContent-Length: 99999999999\r\n\r\n", 413),
            (b"POST /mcp HTTP/1.1\r\nHost: a\x01\r\n\r\n", 400),
            (b"POST /mcp HTTP/1.1\r\nHost: a\r\nContent-Length: 10\r\n\r\nshort", 400),
            (b"POST /mcp HTTP/1.1\r\nHost: a\r\nContent-Length: 1\r\n\r\nxy", 400),
        ];
        for (bytes, want) in cases {
            assert_eq!(status(bytes), Some(want), "{}", String::from_utf8_lossy(bytes));
        }
        let mut big = b"GET / HTTP/1.1\r\nHost: a\r\nX: ".to_vec();
        big.extend(std::iter::repeat_n(b'a', 20_000));
        assert_eq!(status(&big), Some(431));
        let mut many = b"GET / HTTP/1.1\r\nHost: a\r\n".to_vec();
        for i in 0..100 {
            many.extend_from_slice(format!("X{i}: 1\r\n").as_bytes());
        }
        many.extend_from_slice(b"\r\n");
        assert_eq!(status(&many), Some(431));
        assert!(matches!(read(b""), Err(HttpError::Closed)));
    }

    /// A reader that yields one byte per call, forever, slowly: a slow-loris client.
    struct Trickle;
    impl Read for Trickle {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            std::thread::sleep(Duration::from_millis(5));
            buf[0] = b'a';
            Ok(1)
        }
    }

    #[test]
    fn rob_03_slow_clients_hit_the_deadline() {
        let limits =
            HttpLimits { deadline: Duration::from_millis(100), max_head_bytes: 1 << 20, ..HttpLimits::DEFAULT };
        let t = Instant::now();
        assert!(matches!(read_request(&mut Trickle, &limits), Err(HttpError::Timeout)));
        assert!(t.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn responses_and_events_are_framed() {
        let mut out = Vec::new();
        write_response(&mut out, 202, "Accepted", &[("X-Y", "z")], b"").unwrap();
        let text = String::from_utf8(out.clone()).unwrap();
        assert!(text.starts_with("HTTP/1.1 202 Accepted\r\n") && text.contains("Content-Length: 0\r\n"));
        let (resp, rest) = read_response_head(&mut std::io::Cursor::new(out), &HttpLimits::DEFAULT).unwrap();
        assert_eq!(resp.status, 202);
        assert_eq!(resp.header("x-y"), Some("z"));
        assert!(rest.is_empty());
        let mut sse = Vec::new();
        write_sse_head(&mut sse).unwrap();
        write_sse_event(&mut sse, b"{}").unwrap();
        assert!(String::from_utf8(sse).unwrap().ends_with("event: message\ndata: {}\n\n"));
    }

    #[test]
    fn rob_01_mutation_fuzz_http_requests() {
        let corpus: Vec<Vec<u8>> = vec![
            b"POST /mcp HTTP/1.1\r\nHost: 127.0.0.1:4000\r\nAuthorization: Bearer abc\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}".to_vec(),
            b"GET /mcp HTTP/1.1\r\nHost: localhost:1\r\nAccept: text/event-stream\r\nOrigin: http://localhost:1\r\n\r\n".to_vec(),
        ];
        humanart_core::fuzz::run("http", &corpus, 3000, 0x5EC0_0003, Duration::from_secs(60), |bytes| {
            let limits = HttpLimits { deadline: Duration::from_millis(50), ..HttpLimits::DEFAULT };
            let _ = read_request(&mut std::io::Cursor::new(bytes.to_vec()), &limits);
            if let Some(end) = find_head_end(bytes, 0) {
                let _ = parse_head(&bytes[..end], &limits);
            }
        });
    }
}
