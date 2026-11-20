//! JSON-RPC 2.0 messages `[AG-01]`: strict reading of what a client sends and writing of what the
//! server answers.
//!
//! The bytes come from an agent, so they are hostile input `[SEC-04]`, `[ROB-01]`: the JSON parser
//! runs with [`RpcLimits::json`] (input size, depth, members, values) before anything is read, and
//! the message shape is checked member by member. MCP narrows JSON-RPC: request ids are strings or
//! integers (never `null`), and batches (arrays) are not part of the protocol, so they are refused
//! with [`INVALID_REQUEST`].

use humanart_io::json::{JsonLimits, Number, Object, Value, parse_with, to_vec};

/// Invalid JSON.
pub const PARSE_ERROR: i64 = -32700;
/// Not a valid request object.
pub const INVALID_REQUEST: i64 = -32600;
/// The method does not exist.
pub const METHOD_NOT_FOUND: i64 = -32601;
/// Invalid method parameters (also: missing required `_meta` fields).
pub const INVALID_PARAMS: i64 = -32602;
/// An internal error.
pub const INTERNAL_ERROR: i64 = -32603;
/// MCP: the HTTP headers do not match the body (`HeaderMismatch`).
pub const HEADER_MISMATCH: i64 = -32020;
/// MCP: a client capability the request needs was not declared.
pub const MISSING_CAPABILITY: i64 = -32021;
/// MCP: the requested protocol version is not supported.
pub const UNSUPPORTED_VERSION: i64 = -32022;
/// HumanArt (outside the JSON-RPC reserved range): the human revoked the agent's access `[SEC-03]`.
pub const ACCESS_REVOKED: i64 = 4001;
/// HumanArt: the request waited too long (paused agent, busy engine).
pub const TIMED_OUT: i64 = 4002;
/// HumanArt: the server is shutting down.
pub const SHUTTING_DOWN: i64 = 4003;

/// Size limits of one message `[ROB-02]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RpcLimits {
    /// Largest message in bytes.
    pub max_bytes: u64,
    /// Deepest nesting.
    pub max_depth: u32,
    /// Longest string.
    pub max_string_bytes: u64,
    /// Most members of one array or object.
    pub max_members: u64,
    /// Most values in the message.
    pub max_values: u64,
    /// Longest method name.
    pub max_method_bytes: usize,
    /// Longest string id.
    pub max_id_bytes: usize,
}

impl RpcLimits {
    /// 4 MiB messages, depth 32, 4 MiB strings, 65 536 members, 2^20 values, 128-byte methods and
    /// 256-byte ids.
    pub const DEFAULT: Self = Self {
        max_bytes: 4 << 20,
        max_depth: 32,
        max_string_bytes: 4 << 20,
        max_members: 1 << 16,
        max_values: 1 << 20,
        max_method_bytes: 128,
        max_id_bytes: 256,
    };

    /// The JSON parser limits of these message limits.
    pub const fn json(&self) -> JsonLimits {
        JsonLimits {
            max_depth: self.max_depth,
            max_input_bytes: self.max_bytes,
            max_string_bytes: self.max_string_bytes,
            max_members: self.max_members,
            max_values: self.max_values,
        }
    }
}

impl Default for RpcLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A request id (MCP: a string or an integer, never `null`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Id {
    /// An integer id.
    Number(i64),
    /// A string id.
    String(String),
}

impl Id {
    /// The JSON form.
    pub fn to_json(&self) -> Value {
        match self {
            Self::Number(n) => Value::from(Number::from(*n as isize)),
            Self::String(s) => Value::from(s.as_str()),
        }
    }
}

impl std::fmt::Display for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Number(n) => write!(f, "{n}"),
            Self::String(s) => write!(f, "\"{s}\""),
        }
    }
}

/// A JSON-RPC error object.
#[derive(Clone, Debug, PartialEq)]
pub struct RpcError {
    /// The code.
    pub code: i64,
    /// One sentence.
    pub message: String,
    /// Structured details.
    pub data: Option<Value>,
}

impl RpcError {
    /// An error with `code` and `message` and no data.
    pub fn new(code: i64, message: impl Into<String>) -> Self {
        Self { code, message: message.into(), data: None }
    }

    /// The same error with `data`.
    pub fn with_data(mut self, data: Value) -> Self {
        self.data = Some(data);
        self
    }

    /// The JSON form (`{"code", "message", "data"?}`).
    pub fn to_json(&self) -> Value {
        let mut o = Object::with_capacity(3);
        o.insert("code", Value::from(Number::from(self.code as isize)));
        o.insert("message", self.message.as_str());
        if let Some(d) = &self.data {
            o.insert("data", d.clone());
        }
        Value::Object(o)
    }
}

impl std::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JSON-RPC error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for RpcError {}

/// A message a client sent.
#[derive(Clone, Debug, PartialEq)]
pub enum Incoming {
    /// A request: it gets exactly one response.
    Request {
        /// Its id.
        id: Id,
        /// The method.
        method: String,
        /// The parameters (`Value::Object`, or `Value::Null` when absent).
        params: Value,
    },
    /// A notification: no response.
    Notification {
        /// The method.
        method: String,
        /// The parameters (`Value::Object`, or `Value::Null` when absent).
        params: Value,
    },
    /// A response (clients of this protocol do not send any; the server ignores it).
    Response {
        /// Its id, when readable.
        id: Option<Id>,
    },
}

impl Incoming {
    /// The method of a request or notification.
    pub fn method(&self) -> Option<&str> {
        match self {
            Self::Request { method, .. } | Self::Notification { method, .. } => Some(method),
            Self::Response { .. } => None,
        }
    }
}

/// Why a message could not be read: the error to answer with and the id, when the id was readable.
#[derive(Clone, Debug, PartialEq)]
pub struct Malformed {
    /// The request id, when it could be read.
    pub id: Option<Id>,
    /// The error.
    pub error: RpcError,
}

fn malformed(id: Option<Id>, code: i64, message: &str) -> Malformed {
    Malformed { id, error: RpcError::new(code, message) }
}

fn read_id(value: &Value, limits: &RpcLimits) -> Result<Id, &'static str> {
    match value {
        Value::Number(n) => n.as_i64().map(Id::Number).ok_or("the id must be a string or an integer"),
        Value::String(s) if s.len() <= limits.max_id_bytes => Ok(Id::String(s.clone())),
        Value::String(_) => Err("the id is too long"),
        _ => Err("the id must be a string or an integer"),
    }
}

/// Reads one message from `bytes` `[ROB-01]`.
///
/// # Errors
/// [`Malformed`] with [`PARSE_ERROR`] for bytes that are not JSON within the limits, and
/// [`INVALID_REQUEST`] for JSON that is not a JSON-RPC 2.0 request, notification or response
/// (batches included).
pub fn parse_message(bytes: &[u8], limits: &RpcLimits) -> Result<Incoming, Malformed> {
    let value = parse_with(bytes, &limits.json())
        .map_err(|_| malformed(None, PARSE_ERROR, "the message is not JSON within the limits"))?;
    let Value::Object(o) = value else {
        let why =
            if matches!(value, Value::Array(_)) { "batches are not supported" } else { "a message is a JSON object" };
        return Err(malformed(None, INVALID_REQUEST, why));
    };
    let id = match o.get("id") {
        None => None,
        Some(v) => Some(read_id(v, limits).map_err(|why| malformed(None, INVALID_REQUEST, why))?),
    };
    if o.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Err(malformed(id, INVALID_REQUEST, "\"jsonrpc\" must be \"2.0\""));
    }
    if o.contains_key("result") || o.contains_key("error") {
        if o.contains_key("method") {
            return Err(malformed(id, INVALID_REQUEST, "a message cannot be a request and a response"));
        }
        return Ok(Incoming::Response { id });
    }
    for (key, _) in o.entries() {
        if !matches!(key.as_str(), "jsonrpc" | "id" | "method" | "params") {
            return Err(malformed(id, INVALID_REQUEST, "unknown member in the message"));
        }
    }
    let method = match o.get("method") {
        Some(Value::String(m)) if !m.is_empty() && m.len() <= limits.max_method_bytes => m.clone(),
        Some(Value::String(_)) => return Err(malformed(id, INVALID_REQUEST, "the method name is empty or too long")),
        _ => return Err(malformed(id, INVALID_REQUEST, "\"method\" must be a string")),
    };
    let params = match o.get("params") {
        None => Value::Null,
        Some(v @ Value::Object(_)) => v.clone(),
        Some(_) => return Err(malformed(id, INVALID_REQUEST, "\"params\" must be an object")),
    };
    Ok(match id {
        Some(id) => Incoming::Request { id, method, params },
        None => Incoming::Notification { method, params },
    })
}

/// A success response.
pub fn response(id: &Id, result: Value) -> Value {
    let mut o = Object::with_capacity(3);
    o.insert("jsonrpc", "2.0");
    o.insert("id", id.to_json());
    o.insert("result", result);
    Value::Object(o)
}

/// An error response; without an id when the request's id could not be read.
pub fn error_response(id: Option<&Id>, error: &RpcError) -> Value {
    let mut o = Object::with_capacity(3);
    o.insert("jsonrpc", "2.0");
    if let Some(id) = id {
        o.insert("id", id.to_json());
    } else {
        o.insert("id", Value::Null);
    }
    o.insert("error", error.to_json());
    Value::Object(o)
}

/// A notification.
pub fn notification(method: &str, params: Value) -> Value {
    let mut o = Object::with_capacity(3);
    o.insert("jsonrpc", "2.0");
    o.insert("method", method);
    o.insert("params", params);
    Value::Object(o)
}

/// A request (what the in-house client sends).
pub fn request(id: &Id, method: &str, params: Value) -> Value {
    let mut o = Object::with_capacity(4);
    o.insert("jsonrpc", "2.0");
    o.insert("id", id.to_json());
    o.insert("method", method);
    o.insert("params", params);
    Value::Object(o)
}

/// The compact bytes of `message` (no newline inside: JSON strings escape control characters, so
/// one message is one line on stdio and one SSE `data:` line). A value that cannot be written
/// (a non-finite float) becomes an internal-error response.
pub fn encode(message: &Value) -> Vec<u8> {
    to_vec(message).unwrap_or_else(|_| {
        b"{\"jsonrpc\":\"2.0\",\"id\":null,\"error\":{\"code\":-32603,\"message\":\"the reply could not be encoded\"}}"
            .to_vec()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<Incoming, Malformed> {
        parse_message(text.as_bytes(), &RpcLimits::DEFAULT)
    }

    #[test]
    fn ag_01_requests_notifications_and_responses_are_told_apart() {
        let r = parse(r#"{"jsonrpc":"2.0","id":7,"method":"tools/list","params":{}}"#).unwrap();
        assert!(matches!(r, Incoming::Request { id: Id::Number(7), ref method, .. } if method == "tools/list"));
        let r = parse(r#"{"jsonrpc":"2.0","id":"a","method":"ping"}"#).unwrap();
        assert!(
            matches!(r, Incoming::Request { id: Id::String(ref s), ref params, .. } if s == "a" && params.is_null())
        );
        let n = parse(r#"{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":1}}"#).unwrap();
        assert!(matches!(n, Incoming::Notification { .. }));
        let resp = parse(r#"{"jsonrpc":"2.0","id":3,"result":{}}"#).unwrap();
        assert_eq!(resp, Incoming::Response { id: Some(Id::Number(3)) });
    }

    #[test]
    fn rob_01_invalid_messages_get_the_right_codes() {
        let code = |t: &str| parse(t).unwrap_err().error.code;
        assert_eq!(code("{"), PARSE_ERROR);
        assert_eq!(code("[]"), INVALID_REQUEST);
        assert_eq!(code(r#"[{"jsonrpc":"2.0","id":1,"method":"ping"}]"#), INVALID_REQUEST);
        assert_eq!(code("3"), INVALID_REQUEST);
        assert_eq!(code(r#"{"id":1,"method":"ping"}"#), INVALID_REQUEST);
        assert_eq!(code(r#"{"jsonrpc":"1.0","id":1,"method":"ping"}"#), INVALID_REQUEST);
        assert_eq!(code(r#"{"jsonrpc":"2.0","id":null,"method":"ping"}"#), INVALID_REQUEST);
        assert_eq!(code(r#"{"jsonrpc":"2.0","id":1.5,"method":"ping"}"#), INVALID_REQUEST);
        assert_eq!(code(r#"{"jsonrpc":"2.0","id":1,"method":3}"#), INVALID_REQUEST);
        assert_eq!(code(r#"{"jsonrpc":"2.0","id":1,"method":"ping","params":[1]}"#), INVALID_REQUEST);
        assert_eq!(code(r#"{"jsonrpc":"2.0","id":1,"method":"ping","x":1}"#), INVALID_REQUEST);
        assert_eq!(code(r#"{"jsonrpc":"2.0","id":1,"method":""}"#), INVALID_REQUEST);
        // The id is reported when it was readable.
        assert_eq!(parse(r#"{"jsonrpc":"2.0","id":9,"method":7}"#).unwrap_err().id, Some(Id::Number(9)));
        let deep = format!("{}{}", "[".repeat(100), "]".repeat(100));
        assert_eq!(code(&format!(r#"{{"jsonrpc":"2.0","id":1,"method":"m","params":{{"a":{deep}}}}}"#)), PARSE_ERROR);
        let small = RpcLimits { max_bytes: 16, ..RpcLimits::DEFAULT };
        assert!(parse_message(br#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#, &small).is_err());
    }

    #[test]
    fn written_messages_read_back() {
        let id = Id::String("x".into());
        let bytes = encode(&request(&id, "tools/call", Value::Object(Object::new())));
        assert!(!bytes.contains(&b'\n'));
        assert!(matches!(parse_message(&bytes, &RpcLimits::DEFAULT).unwrap(), Incoming::Request { .. }));
        let e = encode(&error_response(None, &RpcError::new(INVALID_REQUEST, "line\nbreak")));
        assert!(!e.contains(&b'\n'));
        let text = String::from_utf8(e).unwrap();
        assert!(text.contains("\"id\":null") && text.contains("-32600"), "{text}");
    }

    #[test]
    fn rob_01_mutation_fuzz_json_rpc() {
        let corpus: Vec<Vec<u8>> = [
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"set_parameters","arguments":{"values":{"a.b":0.5}},"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}}}}"#,
            r#"{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":"abc"}}"#,
            r#"{"jsonrpc":"2.0","id":"r","result":{"x":[1,2,3]}}"#,
        ]
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect();
        humanart_core::fuzz::run("json-rpc", &corpus, 3000, 0x5EC0_0004, std::time::Duration::from_secs(60), |bytes| {
            let _ = parse_message(bytes, &RpcLimits::DEFAULT);
        });
    }
}
