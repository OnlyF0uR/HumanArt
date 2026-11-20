//! Strict lower-case hexadecimal encoding and decoding without allocation.
//!
//! Ids ([`crate::id::Id128`]) and content hashes ([`crate::hash::Hash128`]) are written as hexadecimal
//! text in documents, file names and the agent protocol. Every value has exactly one spelling here:
//! lower case, fixed width, no prefix, no separators. Decoding rejects everything else with a
//! [`HexError`] that says where the text went wrong, so a hand-edited or hostile file cannot smuggle in
//! a second name for the same blob.

use crate::collections::ArrayString;
use crate::error::HexError;
use std::fmt;

/// The sixteen digits, in value order.
const DIGITS: &[u8; 16] = b"0123456789abcdef";

/// The value of one hexadecimal digit, or `None` for any other byte (upper case included).
#[inline]
const fn digit_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

/// The two digits of `byte`, high nibble first.
#[inline]
const fn digits_of(byte: u8) -> [u8; 2] {
    [DIGITS[(byte >> 4) as usize], DIGITS[(byte & 0x0F) as usize]]
}

/// Displays a byte slice as lower-case hexadecimal, two digits per byte, without allocating.
///
/// ```
/// assert_eq!(format!("{}", humanart_core::hex::Hex(&[0x00, 0xAB, 0x7F])), "00ab7f");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hex<'a>(pub &'a [u8]);

impl fmt::Display for Hex<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write(f, self.0)
    }
}

/// Writes `bytes` as lower-case hexadecimal into `out`.
pub fn write(out: &mut dyn fmt::Write, bytes: &[u8]) -> fmt::Result {
    for &byte in bytes {
        let [high, low] = digits_of(byte);
        out.write_char(char::from(high))?;
        out.write_char(char::from(low))?;
    }
    Ok(())
}

/// The hexadecimal text of `bytes` in an inline string of capacity `N`.
///
/// `N` should be twice the number of bytes; when it is smaller the text is cut and the result's
/// `is_truncated()` flag is set.
pub fn encode<const N: usize>(bytes: &[u8]) -> ArrayString<N> {
    let mut out = ArrayString::new();
    for &byte in bytes {
        let [high, low] = digits_of(byte);
        out.push(char::from(high));
        out.push(char::from(low));
    }
    out
}

/// The hexadecimal text of `bytes` as a `String` (the one function here that allocates).
pub fn encode_string(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().saturating_mul(2));
    for &byte in bytes {
        let [high, low] = digits_of(byte);
        out.push(char::from(high));
        out.push(char::from(low));
    }
    out
}

/// Decodes `text` into `out`. The text must have exactly two digits per output byte.
pub fn decode_into(text: &str, out: &mut [u8]) -> Result<(), HexError> {
    let digits = text.as_bytes();
    let expected = out.len().saturating_mul(2);
    if digits.len() != expected {
        return Err(HexError::Length { expected, found: digits.len() });
    }
    for (index, (slot, pair)) in out.iter_mut().zip(digits.as_chunks::<2>().0).enumerate() {
        let high = digit_value(pair[0]).ok_or(HexError::Digit { index: index * 2 })?;
        let low = digit_value(pair[1]).ok_or(HexError::Digit { index: index * 2 + 1 })?;
        *slot = (high << 4) | low;
    }
    Ok(())
}

/// Decodes exactly `2 * N` digits into an array of `N` bytes.
pub fn decode_array<const N: usize>(text: &str) -> Result<[u8; N], HexError> {
    let mut out = [0_u8; N];
    decode_into(text, &mut out)?;
    Ok(out)
}

/// Decodes text of any even length into a vector of half that many bytes.
///
/// For an odd length the error reports the next even length as the expected one.
pub fn decode(text: &str) -> Result<Vec<u8>, HexError> {
    if !text.len().is_multiple_of(2) {
        return Err(HexError::Length { expected: text.len().saturating_add(1), found: text.len() });
    }
    let mut out = vec![0_u8; text.len() / 2];
    decode_into(text, &mut out)?;
    Ok(out)
}

/// `true` when `text` consists only of the digits `0-9a-f` (the empty string included).
pub fn is_hex(text: &str) -> bool {
    text.bytes().all(|byte| digit_value(byte).is_some())
}

/// `value` as 16 digits, most significant first.
pub fn encode_u64(value: u64) -> ArrayString<16> {
    encode(&value.to_be_bytes())
}

/// Parses exactly 16 digits, most significant first.
pub fn decode_u64(text: &str) -> Result<u64, HexError> {
    decode_array::<8>(text).map(u64::from_be_bytes)
}

/// `value` as 32 digits, most significant first.
pub fn encode_u128(value: u128) -> ArrayString<32> {
    encode(&value.to_be_bytes())
}

/// Parses exactly 32 digits, most significant first.
pub fn decode_u128(text: &str) -> Result<u128, HexError> {
    decode_array::<16>(text).map(u128::from_be_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn encodes_every_byte_value_like_the_standard_formatter() {
        for byte in 0..=255_u8 {
            assert_eq!(encode::<2>(&[byte]).as_str(), format!("{byte:02x}"));
        }
        assert_eq!(encode_string(&[]), "");
        assert_eq!(encode_string(&[0xDE, 0xAD, 0xBE, 0xEF]), "deadbeef");
        assert_eq!(Hex(&[0, 1, 0xFF]).to_string(), "0001ff");
        let mut text = String::new();
        write(&mut text, &[0x12, 0xAB]).expect("write to a String");
        assert_eq!(text, "12ab");
    }

    #[test]
    fn encode_reports_a_too_small_capacity() {
        let short = encode::<3>(&[0xAB, 0xCD]);
        assert_eq!(short.as_str(), "abc");
        assert!(short.is_truncated());
        assert!(!encode::<4>(&[0xAB, 0xCD]).is_truncated());
    }

    #[test]
    fn round_trips_random_bytes() {
        let mut rng = Rng::new(31);
        for _ in 0..500 {
            let mut bytes = vec![0_u8; rng.index(40)];
            rng.fill(&mut bytes);
            let text = encode_string(&bytes);
            assert!(is_hex(&text));
            assert_eq!(decode(&text).expect("valid hex"), bytes);
            let mut out = vec![0_u8; bytes.len()];
            decode_into(&text, &mut out).expect("valid hex");
            assert_eq!(out, bytes);
        }
        let value = rng.next_u64();
        assert_eq!(encode_u64(value).as_str(), format!("{value:016x}"));
        assert_eq!(decode_u64(&encode_u64(value)), Ok(value));
        let wide = (u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64());
        assert_eq!(encode_u128(wide).as_str(), format!("{wide:032x}"));
        assert_eq!(decode_u128(&encode_u128(wide)), Ok(wide));
        assert_eq!(decode_u64("0000000000000001"), Ok(1));
    }

    #[test]
    fn rejects_wrong_lengths_with_the_expected_count() {
        assert_eq!(decode_array::<2>("abc"), Err(HexError::Length { expected: 4, found: 3 }));
        assert_eq!(decode_array::<2>("abcdef"), Err(HexError::Length { expected: 4, found: 6 }));
        assert_eq!(decode_array::<2>(""), Err(HexError::Length { expected: 4, found: 0 }));
        assert_eq!(decode("abc"), Err(HexError::Length { expected: 4, found: 3 }));
        assert_eq!(decode(""), Ok(Vec::new()));
        assert_eq!(decode_array::<0>(""), Ok([]));
        assert_eq!(decode_u64("1"), Err(HexError::Length { expected: 16, found: 1 }));
        assert_eq!(decode_u128("00"), Err(HexError::Length { expected: 32, found: 2 }));
    }

    #[test]
    fn rejects_everything_that_is_not_a_lower_case_digit() {
        assert_eq!(decode_array::<2>("abCd"), Err(HexError::Digit { index: 2 }));
        assert_eq!(decode_array::<2>("0x12"), Err(HexError::Digit { index: 1 }));
        assert_eq!(decode_array::<2>("12 4"), Err(HexError::Digit { index: 2 }));
        assert_eq!(decode_array::<2>("123g"), Err(HexError::Digit { index: 3 }));
        assert_eq!(decode_array::<1>("+1"), Err(HexError::Digit { index: 0 }));
        // Multi-byte characters are counted in bytes and never split a panic out of the decoder.
        assert_eq!(decode_array::<1>("é"), Err(HexError::Digit { index: 0 }));
        assert_eq!(decode("€1"), Err(HexError::Digit { index: 0 }));
        assert!(!is_hex("AB"));
        assert!(!is_hex("12-34"));
        assert!(is_hex(""));
        // Every byte value outside the sixteen digits is refused in both positions.
        for byte in 0..=127_u8 {
            let valid = byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte);
            let high = [byte, b'0'];
            let low = [b'0', byte];
            let high = std::str::from_utf8(&high).expect("ascii");
            let low = std::str::from_utf8(&low).expect("ascii");
            assert_eq!(decode_array::<1>(high).is_ok(), valid, "{byte}");
            assert_eq!(decode_array::<1>(low).is_ok(), valid, "{byte}");
        }
    }

    // [ROB-01]
    #[test]
    fn decoding_survives_mutated_input() {
        let corpus = vec![b"00112233445566778899aabbccddeeff".to_vec(), b"deadbeef".to_vec(), Vec::new()];
        crate::fuzz::run("hex", &corpus, 3000, 0x4E58, std::time::Duration::from_secs(60), |bytes| {
            if let Ok(text) = std::str::from_utf8(bytes) {
                let decoded = decode(text);
                if let Ok(decoded) = &decoded {
                    assert_eq!(encode_string(decoded), text);
                }
                let _ = decode_array::<16>(text);
                let _ = decode_u64(text);
                let _ = decode_u128(text);
            }
        });
    }
}
