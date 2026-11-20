//! glTF 2.0 and GLB: document model, reader with validation, accessor reading, writer and an export
//! builder `[IO-10]`.
//!
//! * [`read`] takes a `.gltf` (JSON text) or `.glb` file, decodes `data:` URI buffers, asks a
//!   caller-supplied resolver for external `.bin` files (this crate never touches the file system,
//!   `[PLAT-03]`), and [`validate`]s the result.
//! * [`Gltf::read_f32`], [`Gltf::read_vec`], [`Gltf::read_u32`] and [`Gltf::read_indices`] decode
//!   accessors: byte strides, matrix column padding, normalized integers, sparse substitution.
//! * [`Builder`] packs data into one buffer with `min` / `max`, sparse morph targets and embedded
//!   images; [`Gltf::to_glb`] and [`Gltf::to_gltf`] write it.
//! * Images are not decoded here: [`Gltf::image_source`] hands back their bytes (PNG, JPEG or KTX2
//!   for [`crate::png`], [`crate::jpeg`] and [`crate::ktx2`]) or their relative path.
//!
//! **Limits `[ROB-01]`:** the JSON is parsed with [`JsonLimits::from_limits`]; each buffer's
//! `byteLength` and the total of all buffers are checked against [`Limits::max_bytes`] before any
//! buffer is decoded; every accessor's count against [`Limits::max_elements`] and its decoded size
//! against [`Limits::max_bytes`]; the node hierarchy depth against [`Limits::max_depth`]. Every
//! index, range and count is checked before it is used (see [`validate`]).
//!
//! **Errors:** problems found in the JSON tree are [`Error::Invalid`] with offset 0 (the tree keeps no
//! byte positions) and a `what` naming the member; byte-level problems in the GLB container or in
//! base64 text carry their offset.

mod accessor;
pub mod base64;
pub mod doc;
pub mod export;
pub mod glb;
mod parse;
pub mod validate;
mod write;

pub use doc::*;
pub use export::Builder;
pub use validate::validate;
pub use write::{to_json, used_extensions};

use crate::error::{Error, Format, Result};
use crate::image::Limits;
use crate::json::{self, JsonLimits, Style};
use parse::invalid;

/// A glTF document with its buffer data.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Gltf {
    /// The document.
    pub document: Document,
    /// The data of each buffer of [`Document::buffers`], in order, exactly `byteLength` bytes each.
    pub buffers: Vec<Vec<u8>>,
}

/// Where the bytes of an image are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageSource<'a> {
    /// In a buffer view of the document.
    View(&'a [u8]),
    /// In a `data:` URI, decoded.
    Data(Vec<u8>),
    /// In a file: the percent-decoded relative path, for the caller to load.
    External(String),
}

/// The MIME type of buffers embedded as data URIs.
pub const BUFFER_MIME: &str = "application/octet-stream";

/// Reads a `.gltf` or `.glb` file (told apart by the GLB magic) and validates it.
///
/// `resolve` is called with the percent-decoded relative path of every external buffer and returns
/// its bytes, or `None` when it cannot be loaded (an error). Buffers in the GLB BIN chunk and in
/// `data:` URIs need no resolver.
pub fn read(data: &[u8], limits: &Limits, mut resolve: impl FnMut(&str) -> Option<Vec<u8>>) -> Result<Gltf> {
    limits.check_bytes(Format::Gltf, "glTF file size", data.len() as u64)?;
    let (json_text, bin) = if glb::is_glb(data) {
        let parts = glb::split(data)?;
        (parts.json, parts.bin)
    } else {
        (data, None)
    };
    // A byte order mark is not allowed, but readers may ignore one.
    let json_text = json_text.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(json_text);
    let value = json::parse_with(json_text, &JsonLimits::from_limits(limits))?;
    let document = parse::document(&value)?;
    let mut total = 0_u64;
    for buffer in &document.buffers {
        total = total.saturating_add(buffer.byte_length as u64);
        limits.check_bytes(Format::Gltf, "total glTF buffer size", total)?;
    }
    let mut buffers = Vec::with_capacity(document.buffers.len());
    for (i, buffer) in document.buffers.iter().enumerate() {
        let length = buffer.byte_length;
        if length == 0 {
            return Err(invalid("buffer.byteLength is 0"));
        }
        let mut bytes = match &buffer.uri {
            None => {
                let bin = bin.filter(|_| i == 0).ok_or(invalid("buffer without a uri is not the GLB BIN chunk"))?;
                if bin.len() < length || bin.len() - length > 3 {
                    return Err(invalid("GLB BIN chunk length does not match buffer.byteLength"));
                }
                bin.to_vec()
            }
            Some(uri) => match base64::parse_data_uri(uri)? {
                Some(data_uri) => base64::decode(data_uri.payload.as_bytes(), length as u64 + 3)?,
                None => resolve(&base64::percent_decode(uri)?).ok_or(invalid("external buffer could not be loaded"))?,
            },
        };
        if bytes.len() < length {
            return Err(invalid("buffer data is shorter than buffer.byteLength"));
        }
        bytes.truncate(length);
        buffers.push(bytes);
    }
    validate(&document, &buffers, limits)?;
    Ok(Gltf { document, buffers })
}

/// Reads a self-contained file (GLB, or `.gltf` with only `data:` URIs): [`read`] with a resolver
/// that loads nothing.
pub fn read_self_contained(data: &[u8], limits: &Limits) -> Result<Gltf> {
    read(data, limits, |_| None)
}

impl Gltf {
    /// A document with its buffer data, validated against [`Limits::DEFAULT`].
    pub fn new(document: Document, buffers: Vec<Vec<u8>>) -> Result<Self> {
        validate(&document, &buffers, &Limits::DEFAULT)?;
        Ok(Self { document, buffers })
    }

    /// Writes a GLB file. A first buffer without a URI, or with a `data:` URI, goes to the BIN
    /// chunk (the GLB form of an embedded buffer: glTF 2.0 section "Binary buffer", the BIN chunk
    /// is the first buffer with an undefined uri); every other buffer needs a URI (the caller writes
    /// external ones from [`Gltf::buffers`]). External URIs and the data URIs of later buffers are
    /// kept as they are.
    pub fn to_glb(&self) -> Result<Vec<u8>> {
        validate(&self.document, &self.buffers, &Limits::DEFAULT)?;
        if self.document.buffers.iter().skip(1).any(|b| b.uri.is_none()) {
            return Err(Error::Encode {
                format: Format::Gltf,
                what: "only the first buffer can live in the GLB BIN chunk",
            });
        }
        let first_is_data_uri = match self.document.buffers.first().and_then(|b| b.uri.as_deref()) {
            Some(uri) => base64::parse_data_uri(uri)?.is_some(),
            None => false,
        };
        let json = if first_is_data_uri {
            // The decoded bytes are already in `self.buffers[0]`: drop the URI from a copy of the
            // document so the buffer refers to the BIN chunk.
            let mut document = self.document.clone();
            if let Some(first) = document.buffers.first_mut() {
                first.uri = None;
            }
            to_json(&document, Style::COMPACT)?
        } else {
            to_json(&self.document, Style::COMPACT)?
        };
        let bin = match (self.document.buffers.first(), self.buffers.first()) {
            (Some(b), Some(data)) if b.uri.is_none() || first_is_data_uri => data.get(..b.byte_length).unwrap_or(data),
            _ => &[],
        };
        glb::join(&json, bin)
    }

    /// Writes `.gltf` JSON text in `style`. Every buffer needs a URI: call [`Gltf::embed_buffers`]
    /// for a self-contained file, or set relative URIs and write [`Gltf::buffers`] beside it.
    pub fn to_gltf(&self, style: Style) -> Result<Vec<u8>> {
        validate(&self.document, &self.buffers, &Limits::DEFAULT)?;
        if self.document.buffers.iter().any(|b| b.uri.is_none()) {
            return Err(Error::Encode { format: Format::Gltf, what: "a buffer without a uri in a .gltf file" });
        }
        to_json(&self.document, style)
    }

    /// Gives every buffer without a URI a base64 `data:` URI of its data.
    pub fn embed_buffers(&mut self) {
        for (buffer, data) in self.document.buffers.iter_mut().zip(&self.buffers) {
            if buffer.uri.is_none() {
                let mut uri = format!("data:{BUFFER_MIME};base64,");
                base64::encode_into(data.get(..buffer.byte_length).unwrap_or(data), &mut uri);
                buffer.uri = Some(uri);
            }
        }
    }

    /// Where the bytes of image `index` are. A data URI is decoded, with its size checked against
    /// `limits`.
    pub fn image_source(&self, index: usize, limits: &Limits) -> Result<ImageSource<'_>> {
        let image = self.document.images.get(index).ok_or(invalid("image index is out of range"))?;
        if let Some(view) = image.buffer_view {
            return Ok(ImageSource::View(validate::view_bytes(&self.document, &self.buffers, view)?));
        }
        let uri = image.uri.as_deref().ok_or(invalid("image has neither uri nor bufferView"))?;
        match base64::parse_data_uri(uri)? {
            Some(data) => Ok(ImageSource::Data(base64::decode(data.payload.as_bytes(), limits.max_bytes)?)),
            None => Ok(ImageSource::External(base64::percent_decode(uri)?)),
        }
    }
}

#[cfg(test)]
mod tests;
