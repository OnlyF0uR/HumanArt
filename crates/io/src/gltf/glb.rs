//! The GLB binary container: a 12-byte header, a JSON chunk and an optional BIN chunk `[IO-10]`.

use crate::bytes::{Reader, Writer};
use crate::error::{Error, Format, Result};

/// `glTF` as a little-endian `u32`.
pub const MAGIC: u32 = 0x4654_6C67;
/// The container version this crate reads and writes.
pub const VERSION: u32 = 2;
/// `JSON` as a little-endian `u32`.
pub const CHUNK_JSON: u32 = 0x4E4F_534A;
/// `BIN\0` as a little-endian `u32`.
pub const CHUNK_BIN: u32 = 0x004E_4942;
/// The header length.
pub const HEADER_LEN: usize = 12;

/// The chunks of a GLB file, borrowed from its bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Glb<'a> {
    /// The JSON chunk, padding included (trailing spaces are valid JSON whitespace).
    pub json: &'a [u8],
    /// The BIN chunk, padding included, if there is one.
    pub bin: Option<&'a [u8]>,
}

/// Whether `data` starts with the GLB magic.
#[must_use]
pub fn is_glb(data: &[u8]) -> bool {
    data.get(..4) == Some(&MAGIC.to_le_bytes()[..])
}

/// Splits a GLB file into its chunks.
///
/// Checks: magic, version 2, the header length equal to the input length, chunk lengths inside the
/// input and multiples of 4, the JSON chunk first, at most one BIN chunk and only as the second
/// chunk. Chunks of unknown types are skipped, as the specification asks.
pub fn split(data: &[u8]) -> Result<Glb<'_>> {
    let mut r = Reader::new(data, Format::Gltf);
    if r.u32_le()? != MAGIC {
        return Err(r.invalid("not a GLB file"));
    }
    if r.u32_le()? != VERSION {
        return Err(Error::Unsupported { format: Format::Gltf, what: "GLB container version other than 2" });
    }
    let length = r.len_u32_le()?;
    if length != data.len() {
        return Err(Error::Invalid {
            format: Format::Gltf,
            offset: 8,
            what: "GLB length does not match the file size",
        });
    }
    let mut json = None;
    let mut bin = None;
    let mut index = 0;
    while !r.is_empty() {
        let at = r.pos();
        let len = r.len_u32_le()?;
        let kind = r.u32_le()?;
        if !len.is_multiple_of(4) {
            return Err(Error::Invalid {
                format: Format::Gltf,
                offset: at,
                what: "GLB chunk length is not a multiple of 4",
            });
        }
        let body = r.bytes(len)?;
        match (kind, index) {
            (CHUNK_JSON, 0) => json = Some(body),
            (CHUNK_JSON, _) => {
                return Err(Error::Invalid { format: Format::Gltf, offset: at, what: "second JSON chunk" });
            }
            (_, 0) => {
                return Err(Error::Invalid { format: Format::Gltf, offset: at, what: "first GLB chunk is not JSON" });
            }
            (CHUNK_BIN, 1) => bin = Some(body),
            (CHUNK_BIN, _) => {
                return Err(Error::Invalid {
                    format: Format::Gltf,
                    offset: at,
                    what: "BIN chunk is not the second chunk",
                });
            }
            _ => {}
        }
        index += 1;
    }
    let json =
        json.ok_or(Error::Invalid { format: Format::Gltf, offset: HEADER_LEN, what: "GLB has no JSON chunk" })?;
    Ok(Glb { json, bin })
}

/// Writes a GLB file: the JSON padded with spaces and the BIN data padded with zeros to 4 bytes. An
/// empty `bin` writes no BIN chunk.
pub fn join(json: &[u8], bin: &[u8]) -> Result<Vec<u8>> {
    let pad = |n: usize| n.next_multiple_of(4);
    let total =
        HEADER_LEN as u64 + 8 + pad(json.len()) as u64 + if bin.is_empty() { 0 } else { 8 + pad(bin.len()) as u64 };
    let total =
        u32::try_from(total).map_err(|_| Error::Encode { format: Format::Gltf, what: "GLB larger than 4 GiB" })?;
    let mut w = Writer::with_capacity(total as usize);
    w.u32_le(MAGIC);
    w.u32_le(VERSION);
    w.u32_le(total);
    w.u32_le(pad(json.len()) as u32);
    w.u32_le(CHUNK_JSON);
    w.bytes(json);
    while !w.len().is_multiple_of(4) {
        w.u8(b' ');
    }
    if !bin.is_empty() {
        w.u32_le(pad(bin.len()) as u32);
        w.u32_le(CHUNK_BIN);
        w.bytes(bin);
        w.pad_to(4);
    }
    Ok(w.into_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_and_split_round_trip() {
        let file = join(b"{\"a\":1}", &[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(file.len(), 12 + 8 + 8 + 8 + 8);
        assert_eq!(&file[..4], b"glTF");
        assert!(is_glb(&file));
        let glb = split(&file).unwrap();
        assert_eq!(glb.json, b"{\"a\":1} ");
        assert_eq!(glb.bin, Some(&[1, 2, 3, 4, 5, 0, 0, 0][..]));
        let no_bin = join(b"{}", &[]).unwrap();
        assert_eq!(split(&no_bin).unwrap().bin, None);
    }

    #[test]
    fn container_rules_are_enforced() {
        let good = join(b"{}", &[9; 4]).unwrap();
        let mut bad = good.clone();
        bad[4] = 1;
        assert!(matches!(split(&bad), Err(Error::Unsupported { .. })));
        let mut bad = good.clone();
        bad[8] = 0;
        assert!(split(&bad).is_err(), "length mismatch");
        let mut bad = good.clone();
        bad[12] = 3;
        assert!(split(&bad).is_err(), "chunk length not a multiple of 4");
        let mut bad = good.clone();
        bad[16..20].copy_from_slice(&CHUNK_BIN.to_le_bytes());
        assert!(split(&bad).is_err(), "BIN first");
        // An unknown third chunk is skipped; a second BIN chunk is not.
        let mut extra = good.clone();
        extra.extend_from_slice(&4_u32.to_le_bytes());
        extra.extend_from_slice(b"XTRA");
        extra.extend_from_slice(&[0; 4]);
        let len = extra.len() as u32;
        extra[8..12].copy_from_slice(&len.to_le_bytes());
        assert!(split(&extra).is_ok());
        extra[good.len() + 4..good.len() + 8].copy_from_slice(&CHUNK_BIN.to_le_bytes());
        assert!(split(&extra).is_err());
        for len in 0..good.len() {
            assert!(split(&good[..len]).is_err(), "{len}");
        }
    }
}
