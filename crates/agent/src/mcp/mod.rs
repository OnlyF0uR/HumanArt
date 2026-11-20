//! The agent interface (section 10): the in-house MCP server over JSON-RPC 2.0 on stdio and on
//! loopback HTTP, its tools, resources and notifications `[AG-01]`, `[AG-02]`, and its security
//! measures `[SEC-01]`..`[SEC-04]`.
//!
//! - [`jsonrpc`]: strict JSON-RPC 2.0 messages with size limits.
//! - [`http`]: a strict HTTP/1.1 request reader and response / event-stream writer.
//! - [`security`]: bearer token, Host / Origin checks, rate limiter, path sandbox.
//! - [`schema`]: tool argument declarations: published JSON Schemas derived from the registry and
//!   the template, and the validation of every call.
//! - [`tools`]: the observation and mutation tools.
//! - [`render`]: offscreen renders through the editor's viewer.
//! - [`server`]: protocol methods, dispatch, pause / revoke, subscriptions and notifications.
//! - [`transport`]: the stdio and HTTP transports (threads).
//! - [`client`]: the in-house client the tests (and scripts) drive the server with.
//!
//! Protocol revision: MCP **2026-07-28** (stateless, per-request `_meta`), with the legacy
//! `initialize` revisions 2025-11-25, 2025-06-18 and 2025-03-26 served to clients that open with a
//! handshake (see [`server`]).
//!
//! Hosting: create a [`server::Server`], start transports that feed one channel of
//! [`server::Inbound`]s, and call [`server::Server::pump`] from the thread that owns the
//! [`crate::engine::Engine`] (each editor frame, or the CLI's serve loop).

pub mod client;
pub mod http;
pub mod jsonrpc;
pub mod render;
pub mod schema;
pub mod security;
pub mod server;
pub mod tools;
pub mod transport;

#[cfg(test)]
mod tests;
