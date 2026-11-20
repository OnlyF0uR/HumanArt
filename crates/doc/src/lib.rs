//! The parameter registry, the character document, layers, commands, history, proposals and locks.
//!
//! What is here (WP-07): the parameter [`registry`] read from `data/template/params/` `[REG-01]`..
//! `[REG-05]`; the typed [`document`] `[DOC-01]`..`[DOC-13]`; its line-oriented [`text`] form with
//! value-diff [`text::Edit`]s `[DOC-21]`; format [`migrate`]ions `[DOC-23]`; the content-addressed
//! [`blob`] store `[DOC-22]`, `[IO-30]`; the command [`journal`] `[DOC-14]`; and the [`project`]
//! directory `[DOC-20]`. The command system, history, locks enforcement and proposals (WP-08) build on
//! [`document::Document::diff`] and [`document::Document::apply_edits`].
//!
//! Files are reached only through `humanart_core::platform::FileSystem`, so the crate builds for
//! `wasm32-unknown-unknown`. The formats are specified in `docs/format-anthropos.md`.
#![forbid(unsafe_code)]

pub mod blob;
pub mod command;
pub mod document;
pub mod error;
pub mod expression_drive;
pub mod history;
pub mod journal;
pub mod lock;
pub mod migrate;
pub mod notify;
pub mod project;
pub mod proposal;
pub mod registry;
pub mod save;
pub mod session;
pub mod text;

pub use error::{DocError, Result};
