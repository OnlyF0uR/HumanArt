//! JSON (RFC 8259): values, a strict parser, deterministic writers and typed access `[DEP-05]`.

pub mod access;
pub mod number;
pub mod parse;
pub mod value;
pub mod write;

pub use access::{AccessError, Elements, FromJson, Members, Node, What};
pub use number::Number;
pub use parse::{JsonLimits, ParseError, parse, parse_str, parse_with};
pub use value::{Kind, Object, Value};
pub use write::{
    Indent, JsonWriter, Sink, Style, ToJson, stringify, to_string, to_string_pretty, to_vec, write, write_number,
    write_string,
};

#[cfg(test)]
mod tests;
