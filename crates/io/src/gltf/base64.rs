//! Base64 (RFC 4648, standard alphabet), `data:` URIs and percent-decoding of glTF URIs `[IO-10]`.

use crate::error::{Error, Format, Result};

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
/// Marks a byte that is not in the alphabet.
const BAD: u8 = 0xFF;

/// The 6-bit value of every byte, or [`BAD`].
const DECODE: [u8; 256] = {
    let mut table = [BAD; 256];
    let mut i = 0;
    while i < 64 {
        table[ALPHABET[i] as usize] = i as u8;
        i += 1;
    }
    table
};

/// The length of the base64 text of `n` bytes, with padding.
#[must_use]
pub const fn encoded_len(n: usize) -> usize {
    n.div_ceil(3) * 4
}

/// Appends the padded base64 text of `data` to `out`.
pub fn encode_into(data: &[u8], out: &mut String) {
    out.reserve(encoded_len(data.len()));
    let (chunks, rest) = data.as_chunks::<3>();
    for &[a, b, c] in chunks {
        let v = (u32::from(a) << 16) | (u32::from(b) << 8) | u32::from(c);
        for shift in [18, 12, 6, 0] {
            out.push(char::from(ALPHABET[(v >> shift) as usize & 63]));
        }
    }
    match *rest {
        [a] => {
            let v = u32::from(a) << 16;
            out.push(char::from(ALPHABET[(v >> 18) as usize & 63]));
            out.push(char::from(ALPHABET[(v >> 12) as usize & 63]));
            out.push_str("==");
        }
        [a, b] => {
            let v = (u32::from(a) << 16) | (u32::from(b) << 8);
            out.push(char::from(ALPHABET[(v >> 18) as usize & 63]));
            out.push(char::from(ALPHABET[(v >> 12) as usize & 63]));
            out.push(char::from(ALPHABET[(v >> 6) as usize & 63]));
            out.push('=');
        }
        _ => {}
    }
}

/// The padded base64 text of `data`.
#[must_use]
pub fn encode(data: &[u8]) -> String {
    let mut out = String::new();
    encode_into(data, &mut out);
    out
}

/// Decodes base64 text. Padding is optional, but when present it must complete the last quad;
/// whitespace and characters outside the standard alphabet are refused, as are non-zero bits left
/// over in the last character (non-canonical text). `max_bytes` bounds the output and is checked
/// before allocating.
pub fn decode(text: &[u8], max_bytes: u64) -> Result<Vec<u8>> {
    let invalid = |offset, what| Error::Invalid { format: Format::Gltf, offset, what };
    let body = match text {
        [rest @ .., b'=', b'='] if text.len().is_multiple_of(4) => rest,
        [rest @ .., b'='] if text.len().is_multiple_of(4) => rest,
        _ => text,
    };
    if body.len() % 4 == 1 {
        return Err(invalid(body.len(), "base64 text has a dangling character"));
    }
    let out_len = body.len() / 4 * 3 + (body.len() % 4).saturating_sub(1);
    if out_len as u64 > max_bytes {
        return Err(Error::Limit {
            format: Format::Gltf,
            what: "decoded base64 size",
            limit: max_bytes,
            got: out_len as u64,
        });
    }
    let mut out = Vec::with_capacity(out_len);
    let mut acc = 0_u32;
    let mut bits = 0_u32;
    for (i, &c) in body.iter().enumerate() {
        let v = DECODE[c as usize];
        if v == BAD {
            return Err(invalid(i, "character outside the base64 alphabet"));
        }
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    if acc != 0 {
        return Err(invalid(body.len(), "base64 text has non-zero trailing bits"));
    }
    Ok(out)
}

/// A parsed `data:` URI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DataUri<'a> {
    /// The media type, empty when the URI gives none.
    pub mime_type: &'a str,
    /// The base64 payload.
    pub payload: &'a str,
}

/// Splits a `data:[<media type>][;params];base64,<payload>` URI. Returns `None` when `uri` is not a
/// data URI; a data URI that is not base64 is an error (glTF requires base64).
pub fn parse_data_uri(uri: &str) -> Result<Option<DataUri<'_>>> {
    let Some(rest) = uri.strip_prefix("data:") else { return Ok(None) };
    let comma =
        rest.find(',').ok_or(Error::Invalid { format: Format::Gltf, offset: 5, what: "data URI has no comma" })?;
    let (header, payload) = (&rest[..comma], &rest[comma + 1..]);
    let Some(meta) = header.strip_suffix(";base64") else {
        return Err(Error::Invalid { format: Format::Gltf, offset: 5, what: "data URI is not base64" });
    };
    let mime_type = meta.split(';').next().unwrap_or("");
    Ok(Some(DataUri { mime_type, payload }))
}

/// Decodes `%XX` escapes of a URI reference. Fails on a malformed escape or when the result is not
/// UTF-8.
pub fn percent_decode(uri: &str) -> Result<String> {
    let bytes = uri.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = |b: u8| (b as char).to_digit(16);
            let (Some(hi), Some(lo)) = (bytes.get(i + 1).and_then(|&b| hex(b)), bytes.get(i + 2).and_then(|&b| hex(b)))
            else {
                return Err(Error::Invalid {
                    format: Format::Gltf,
                    offset: i,
                    what: "malformed percent escape in URI",
                });
            };
            out.push((hi * 16 + lo) as u8);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).map_err(|_| Error::Invalid { format: Format::Gltf, offset: 0, what: "URI is not UTF-8" })
}

/// Escapes the characters a relative URI reference cannot hold as they are (space, `%`, `#`, `?`,
/// controls and non-ASCII), the inverse of [`percent_decode`] for file names.
#[must_use]
pub fn percent_encode(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for &b in path.as_bytes() {
        if b.is_ascii_alphanumeric() || b"-._~/!$&'()*+,;=:@".contains(&b) {
            out.push(char::from(b));
        } else {
            out.push('%');
            out.push(char::from(b"0123456789ABCDEF"[usize::from(b >> 4)]));
            out.push(char::from(b"0123456789ABCDEF"[usize::from(b & 15)]));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    #[test]
    fn rfc_4648_vectors() {
        // RFC 4648 section 10.
        for (plain, coded) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(encode(plain.as_bytes()), coded);
            assert_eq!(decode(coded.as_bytes(), 100).unwrap(), plain.as_bytes());
            assert_eq!(decode(coded.trim_end_matches('=').as_bytes(), 100).unwrap(), plain.as_bytes());
        }
    }

    #[test]
    fn random_round_trips() {
        let mut rng = Rng::new(64);
        for len in 0..200 {
            let mut data = vec![0; len];
            rng.fill(&mut data);
            let text = encode(&data);
            assert_eq!(text.len(), encoded_len(len));
            assert_eq!(decode(text.as_bytes(), len as u64).unwrap(), data);
        }
    }

    #[test]
    fn bad_text_is_refused() {
        for bad in ["Zg=", "Z", "Zm9v Yg==", "Zh==", "Zm9=", "Zm*v", "Zg===", "=Zg="] {
            assert!(decode(bad.as_bytes(), 100).is_err(), "{bad}");
        }
        assert!(matches!(decode(b"Zm9vYmFy", 5), Err(Error::Limit { .. })));
    }

    #[test]
    fn data_uris_and_percent_escapes() {
        let uri = parse_data_uri("data:application/octet-stream;base64,AAEC").unwrap().unwrap();
        assert_eq!(uri, DataUri { mime_type: "application/octet-stream", payload: "AAEC" });
        assert_eq!(parse_data_uri("data:;base64,").unwrap().unwrap().mime_type, "");
        assert_eq!(parse_data_uri("mesh.bin").unwrap(), None);
        assert!(parse_data_uri("data:text/plain,hello").is_err());
        assert!(parse_data_uri("data:base64").is_err());
        assert_eq!(percent_decode("my%20mesh%2Ebin").unwrap(), "my mesh.bin");
        assert!(percent_decode("bad%2").is_err());
        assert!(percent_decode("bad%zz").is_err());
        assert!(percent_decode("%FF").is_err());
        let name = "tex tures/skin #1 é.png";
        assert_eq!(percent_decode(&percent_encode(name)).unwrap(), name);
        assert_eq!(percent_encode("a b"), "a%20b");
    }
}
